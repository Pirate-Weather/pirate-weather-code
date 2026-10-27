"""GFS forecast and short-term history ingest.

Downloads the surface, 2 m and 10 m fields of the newest 6-hourly GFS cycle that
reaches 240 h (0.25 degree global grid; hourly leads 1-120 h, then 3-hourly to
240 h) with Herbie, plus DUVB from the pgrb2b product. The GRIB subsets are read
directly with xarray/cfgrib (no wgrib2 conversion) and de-accumulated in
xarray/dask. Storm distance/direction are derived from precipitation.

The first 6 hours of each earlier cycle in the last ``HISTORY_PERIODS["GFS"]``
hours are archived as ``GFS_Hist_v4<run>.zarr`` stores (tarred on S3). The
history and the forecast are stacked, interpolated to an hourly axis and
published as ``GFS.zarr`` and ``GFS.time.pickle``. The forecast store starts ``HISTORY_PERIODS["GFS"]`` hours
before the first forecast lead (run + 1 h).

Set ``force_update`` to reprocess an already published cycle, and ``NO_UPLOAD``
to skip S3 uploads (and keep the processing directory) when testing.

Alexander Rey, September 2023
"""
# ruff: noqa: E402  # dotenv must be loaded before downstream imports

import logging
import os
import shutil
import time
import warnings

import dask
import dask.array as da

# Env setup
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

import numpy as np
import pandas as pd
import xarray as xr
from herbie import HerbieLatest

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_to_hourly_rate,
    history_store_exists,
    interp_time_take_blend,
    make_herbie_save_dir,
    map_history_stores,
    mask_invalid_data,
    mask_invalid_refc,
    open_history_arrays,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_history_store,
    publish_ingest_run,
    read_published_run_time,
    shard_whole_variables,
    validate_stacked_time_alignment,
    write_final_zarr,
)
from API.utils.storm_proc import compute_storm_fields_from_apcp_dataarray

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# 0.25 degree global latitude/longitude grid
GFS_GRID_SHAPE = (721, 1440)
GFS_CYCLE_HOURS = 6
# Hourly leads to 120 h, then 3-hourly to 240 h
GFS_FORECAST_HOURS = [*FORECAST_LEAD_RANGES["GFS_1"], *FORECAST_LEAD_RANGES["GFS_2"]]
# The first 6 hourly leads of each earlier cycle fill the gap to the next cycle
GFS_HISTORIC_FORECAST_HOURS = list(range(1, GFS_CYCLE_HOURS + 1))

# The run check only probes AWS and NOMADS; downloads also try Google Cloud.
GFS_LATEST_RUN_PRIORITY = ["aws", "nomads"]
GFS_DOWNLOAD_PRIORITY = ["aws", "google", "nomads"]

# pgrb2.0p25 fields. APCP is matched as the run total (0-N h) so it only needs
# differencing; PRMSL/DSWRF/TOZNE are matched on any level. The thunderstorm
# indices are the best (4-layer) lifted index (4LFTX), surface-based CIN and
# 500 hPa vertical velocity (VVEL, Pa/s), matching the GDPS/HRDPS 500 hPa level.
GFS_MATCH_STRINGS = (
    ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    "|:((CRAIN|CICEP|CSNOW|CFRZR|PRATE|PRES|VIS|GUST|CAPE|CIN|4LFTX):surface:.*hour fcst)"
    "|(:VVEL:500 mb:)"
    "|(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    "|(:TOZNE:)"
    "|(:(TCDC|REFC):entire atmosphere:.*hour fcst)"
    "|(:APCP:surface:0-[1-9]*)"
    "|(:(PRMSL|DSWRF):)"
)
# UV (DUVB) is only in the secondary pgrb2b product
GFS_DUVB_MATCH_STRING = ":DUVB:surface:"

# GRIB variables outside the generic wgrib2 stats range for valid reasons: CIN
# is negative by convention and routinely below the range minimum (-100 J/kg).
GFS_EXCLUDED_STATS_VARIABLES = ["CIN"]

# Archived history stores. The version changes whenever the stored variables
# change, so history written with the old variable set is rebuilt, not merged.
GFS_HISTORIC_STORE_PREFIX = "GFS_Hist_v4"

# Precipitation rate (mm/h) that counts as a storm cell for storm distance/direction
GFS_STORM_APCP_THRESHOLD_MM_H = 0.2

# GFS DSWRF and DUVB are averages since the most recent 6-hourly bucket reset
# (6-7 h, 6-12 h, ..., 114-120 h, then 120-123 h, 120-126 h on the 3-hourly leads).
# APCP is downloaded as the run total (0-N h), so it only needs differencing.
GFS_AVERAGE_BUCKET_HOURS = 6

# cfgrib can only open GRIB messages that share a level type (and, for the surface
# fields, a step type) as one dataset, so each group is opened across all
# forecast hours and the groups are merged.
GFS_CFGRIB_GROUPS = (
    {"typeOfLevel": "heightAboveGround", "level": 2},
    {"typeOfLevel": "heightAboveGround", "level": 10},
    {"typeOfLevel": "surface", "stepType": "instant"},
    {"typeOfLevel": "surface", "stepType": "accum"},
    {"typeOfLevel": "surface", "stepType": "avg"},
    {"typeOfLevel": "meanSea"},
    {"typeOfLevel": "atmosphere"},
    {"typeOfLevel": "atmosphereSingleLayer"},
    {"typeOfLevel": "isobaricInhPa", "level": 500},
)

# cfgrib short names -> the wgrib2-style names the GFS store has always used.
CFGRIB_TO_GFS_NAMES = {
    "t2m": "TMP_2maboveground",
    "d2m": "DPT_2maboveground",
    "r2": "RH_2maboveground",
    "aptmp": "APTMP_2maboveground",
    "u10": "UGRD_10maboveground",
    "v10": "VGRD_10maboveground",
    "vis": "VIS_surface",
    "gust": "GUST_surface",
    "sp": "PRES_surface",
    "prate": "PRATE_surface",
    "csnow": "CSNOW_surface",
    "cicep": "CICEP_surface",
    "cfrzr": "CFRZR_surface",
    "crain": "CRAIN_surface",
    "cape": "CAPE_surface",
    "tp": "APCP_surface",
    "sdswrf": "DSWRF_surface",
    "prmsl": "PRMSL_meansealevel",
    "tcc": "TCDC_entireatmosphere",
    "refc": "REFC_entireatmosphere",
    "tozne": "TOZNE_entireatmosphere_consideredasasinglelayer_",
    "duvb": "DUVB_surface",
    "lftx4": "N4LFTX_surface",
    "w": "VVEL_500mb",
    "cin": "CIN_surface",
}

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so zarr_vars[i] is column i + 1 of the
# API's GFS index map. PRES_surface is renamed PRES_station once the fields are
# corrected.
zarr_vars = (
    "VIS_surface",
    "GUST_surface",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "APTMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "TOZNE_entireatmosphere_consideredasasinglelayer_",
    "TCDC_entireatmosphere",
    "DUVB_surface",
    "Storm_Distance",
    "Storm_Direction",
    "REFC_entireatmosphere",
    "DSWRF_surface",
    "CAPE_surface",
    "PRES_station",
    "N4LFTX_surface",
    "VVEL_500mb",
    "CIN_surface",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GFS")
forecast_process_path = os.path.join(forecast_process_dir, "GFS_Process")
hist_process_path = os.path.join(forecast_process_dir, "GFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GFS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/GFS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


def _parse_bool_env(value: str) -> bool:
    """Parse environment variable as boolean."""
    return value.lower() in ("1", "true", "yes", "on")


no_upload = _parse_bool_env(os.getenv("NO_UPLOAD", os.getenv("no_upload", "")))
force_update = _parse_bool_env(os.getenv("force_update", ""))

GFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
GFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)
# GFS writes many large variables; these defaults were tuned for the GFS host.
DEFAULT_ZARR_WORKERS = 12
DEFAULT_ZARR_CONCURRENCY = 12


def download_gfs_subset(
    product: str,
    search: str,
    dataset_name: str,
    run_time: pd.Timestamp,
    forecast_hours: list[int],
    herbie_save_dir: str,
) -> list[str]:
    """Download and validate one GFS product subset with bounded retries."""
    return download_and_validate_subset(
        model="gfs",
        product=product,
        search=search,
        dataset_name=dataset_name,
        base_time=run_time,
        forecast_hours=forecast_hours,
        priority=GFS_DOWNLOAD_PRIORITY,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=GFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=GFS_RETRY_SLEEP_SECONDS,
        wgrib2_exe=wgrib2_path.strip(),
        excluded_stats_variables=GFS_EXCLUDED_STATS_VARIABLES,
    )


def historic_run_stamp(base_time: pd.Timestamp, hours_offset: int) -> str:
    """ISO 8601 stamp of an earlier cycle, as used in the history store names."""
    return (base_time - pd.Timedelta(hours=hours_offset)).strftime("%Y%m%dT%H%M%SZ")


def _open_cfgrib_group(grib_files: list[str], filter_by_keys: dict) -> xr.Dataset:
    """Open one cfgrib message group across all forecast hours, indexed by valid time."""
    ds = xr.open_mfdataset(
        grib_files,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        # ecCodes' definition parser is not thread-safe, so files are opened
        # serially; dask still reads the data in parallel later.
        parallel=False,
        decode_timedelta=True,
        coords="minimal",
        compat="override",
        backend_kwargs={"filter_by_keys": filter_by_keys},
    )
    # Only "step" is concatenated across files; valid time is run time + step.
    run_time = ds["time"].values
    ds = ds.reset_coords(drop=True)
    return ds.assign_coords(step=run_time + ds["step"].values).rename({"step": "time"})


def open_gfs_gribs(pgrb2_files: list[str], duvb_files: list[str]) -> xr.Dataset:
    """Open the Herbie pgrb2 and pgrb2b subsets as one lazy dataset.

    cfgrib returns latitude north to south; the GFS store has always been written
    south to north (wgrib2's NetCDF order), so flip it to keep the API's grid lookup.
    """
    datasets = [_open_cfgrib_group(pgrb2_files, keys) for keys in GFS_CFGRIB_GROUPS]
    datasets.append(_open_cfgrib_group(duvb_files, {"typeOfLevel": "surface"}))
    ds = xr.merge(datasets, compat="override", join="exact")
    missing = set(CFGRIB_TO_GFS_NAMES) - set(ds.data_vars)
    if missing:
        raise ValueError(f"GFS GRIB files are missing fields: {sorted(missing)}")
    ds = ds[list(CFGRIB_TO_GFS_NAMES)].rename(CFGRIB_TO_GFS_NAMES)
    # The merge step reads the raw "time" array back from zarr as unix seconds
    # (wgrib2's NetCDF encoding), so store it the same way.
    ds["time"].encoding = {
        "units": "seconds since 1970-01-01 00:00:00",
        "dtype": "float64",
    }
    return ds.isel(latitude=slice(None, None, -1))


def bucket_average_to_interval_mean(
    average: xr.DataArray, forecast_hours: np.ndarray
) -> xr.DataArray:
    """Convert GFS bucket averages to the mean over each output interval.

    The value at lead h is the mean since the bucket reset s, so the mean over
    (prev, h] is (ave_h * (h - s) - ave_prev * (prev - s)) / (h - prev). At a reset
    prev == s and the value is used as-is.
    """
    hours = np.asarray(forecast_hours, dtype="float64")
    previous_hours = np.insert(hours[:-1], 0, 0.0)
    bucket_start = (
        np.floor((hours - 1) / GFS_AVERAGE_BUCKET_HOURS) * GFS_AVERAGE_BUCKET_HOURS
    )
    # Each step must start inside its own bucket (or exactly at its reset);
    # otherwise part of the interval was averaged into an earlier bucket.
    if np.any(previous_hours < bucket_start):
        raise ValueError("GFS forecast hours skip a bucket reset.")

    weight_now = xr.DataArray(hours - bucket_start, dims="time")
    weight_previous = xr.DataArray(previous_hours - bucket_start, dims="time")
    interval_hours = xr.DataArray(hours - previous_hours, dims="time")
    previous = average.shift(time=1, fill_value=0.0)
    interval_mean = (average * weight_now - previous * weight_previous) / interval_hours
    return interval_mean.clip(min=0)


def deaccumulate_gfs_fields(
    ds: xr.Dataset, run_time: pd.Timestamp, expected_forecast_hours
) -> None:
    """Turn GFS run-total APCP into mm/h and bucket averages into interval means."""
    forecast_hours = (
        ds["time"].values - pd.Timestamp(run_time).to_datetime64()
    ) / np.timedelta64(1, "h")
    if not np.array_equal(forecast_hours, np.asarray(expected_forecast_hours)):
        raise ValueError(
            f"GFS NetCDF lead hours {forecast_hours.tolist()} do not match the "
            f"requested hours {list(expected_forecast_hours)}."
        )

    ds["APCP_surface"] = (
        ds["APCP_surface"].dims,
        deaccumulate_to_hourly_rate(ds["APCP_surface"].data, forecast_hours),
    )
    for var_name in ("DSWRF_surface", "DUVB_surface"):
        ds[var_name] = bucket_average_to_interval_mean(ds[var_name], forecast_hours)


def main() -> None:
    """Run the GFS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes(
        DEFAULT_ZARR_WORKERS, DEFAULT_ZARR_CONCURRENCY
    )

    process_chunk = CHUNK_SIZES["GFS"]
    final_chunk = FINAL_CHUNK_SIZES["GFS"]
    his_period = HISTORY_PERIODS["GFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )
    history_remote = open_remote(
        historic_path, save_type, aws_access_key_id, aws_secret_access_key
    )

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="gfs",
        n=3,
        freq="6h",
        fxx=GFS_FORECAST_HOURS[-1],
        product="pgrb2.0p25",
        verbose=False,
        priority=GFS_LATEST_RUN_PRIORITY,
        save_dir=herbie_save_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "GFS.time.pickle")
    if (
        not force_update
        and previous_base_time is not None
        and previous_base_time >= base_time
    ):
        logger.info("No Update to GFS, ending")
        return
    if force_update:
        logger.info("force_update enabled, bypassing No Update check")
    logger.info("Processing GFS run %s", base_time)

    # %% Download the forecast GRIB subsets
    pgrb2_grib_files = download_gfs_subset(
        "pgrb2.0p25",
        GFS_MATCH_STRINGS,
        "GFS forecast pgrb2.0p25",
        base_time,
        GFS_FORECAST_HOURS,
        herbie_save_dir,
    )
    duvb_grib_files = download_gfs_subset(
        "pgrb2b.0p25",
        GFS_DUVB_MATCH_STRING,
        "GFS forecast pgrb2b.0p25 DUVB",
        base_time,
        GFS_FORECAST_HOURS,
        herbie_save_dir,
    )

    # %% Read the GRIB subsets with xarray/cfgrib, then de-accumulate in xarray.
    # deaccumulate_gfs_fields raises if the valid times are not exactly the
    # requested leads.
    xarray_forecast_merged = open_gfs_gribs(pgrb2_grib_files, duvb_grib_files)
    deaccumulate_gfs_fields(xarray_forecast_merged, base_time, GFS_FORECAST_HOURS)

    # %% Valid times: hourly history before the first lead, then the forecast leads
    start = xarray_forecast_merged.time.min().values
    end = xarray_forecast_merged.time.max().values
    new_hourly_time = pd.date_range(
        start=start - pd.Timedelta(hours=his_period), end=end, freq="h"
    )
    stacked_times = np.concatenate(
        (
            pd.date_range(
                start=start - pd.Timedelta(hours=his_period),
                end=start - pd.Timedelta(hours=1),
                freq="h",
            ),
            xarray_forecast_merged.time.values,
        )
    )
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # %% Correct the forecast fields
    # Reflectivity below the detection threshold is noise
    xarray_forecast_merged["REFC_entireatmosphere"] = mask_invalid_refc(
        xarray_forecast_merged["REFC_entireatmosphere"]
    )

    # Differencing accumulations/averages can leave small negative rates
    for var_name in ("APCP_surface", "DSWRF_surface", "DUVB_surface"):
        xarray_forecast_merged[var_name] = xarray_forecast_merged[var_name].clip(min=0)

    # Nearest storm distance and direction from the precipitation field
    distanced_stacked, directions_stacked = compute_storm_fields_from_apcp_dataarray(
        apcp_dataarray=xarray_forecast_merged["APCP_surface"],
        threshold=GFS_STORM_APCP_THRESHOLD_MM_H,
        max_distance_m=None,
    )
    forecast_chunks = (len(GFS_FORECAST_HOURS), process_chunk, process_chunk)
    distanced_chunked = distanced_stacked.rechunk(forecast_chunks)
    directions_chunked = directions_stacked.rechunk(forecast_chunks)

    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        distanced_chunked.to_zarr(
            forecast_process_path + "_stormDist.zarr",
            overwrite=True,
            compute=True,
        )
        directions_chunked.to_zarr(
            forecast_process_path + "_stormDir.zarr",
            overwrite=True,
            compute=True,
        )

    # Surface pressure is stored (and read by the API) as station pressure
    xarray_forecast_merged = xarray_forecast_merged.rename(
        {"PRES_surface": "PRES_station"}
    )

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(GFS_FORECAST_HOURS),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_.zarr",
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    del (
        directions_chunked,
        distanced_chunked,
        distanced_stacked,
        directions_stacked,
        xarray_forecast_merged,
    )
    logger.info("GFS forecast processed in %.1f s", time.time() - T0)

    # %% Archive the first hours of each earlier cycle that is not yet archived
    historic_offsets = list(range(his_period, 0, -GFS_CYCLE_HOURS))
    for hours_offset in historic_offsets:
        timestamp = historic_run_stamp(base_time, hours_offset)
        hist_store_name = f"{GFS_HISTORIC_STORE_PREFIX}{timestamp}.zarr"
        if history_store_exists(history_remote, hist_store_name):
            logger.info("History store exists, skipping: %s", hist_store_name)
            continue

        hist_run_date = base_time - pd.Timedelta(hours=hours_offset)
        logger.info("Downloading historic data: %s", timestamp)

        grib_list = download_gfs_subset(
            "pgrb2.0p25",
            GFS_MATCH_STRINGS,
            "GFS historic pgrb2.0p25",
            hist_run_date,
            GFS_HISTORIC_FORECAST_HOURS,
            herbie_save_dir,
        )
        grib_list_uv = download_gfs_subset(
            "pgrb2b.0p25",
            GFS_DUVB_MATCH_STRING,
            "GFS historic pgrb2b.0p25 DUVB",
            hist_run_date,
            GFS_HISTORIC_FORECAST_HOURS,
            herbie_save_dir,
        )

        xarray_hist_merged = open_gfs_gribs(grib_list, grib_list_uv)
        deaccumulate_gfs_fields(
            xarray_hist_merged, hist_run_date, GFS_HISTORIC_FORECAST_HOURS
        )

        storm_distance_hist, storm_direction_hist = (
            compute_storm_fields_from_apcp_dataarray(
                apcp_dataarray=xarray_hist_merged["APCP_surface"],
                threshold=GFS_STORM_APCP_THRESHOLD_MM_H,
                max_distance_m=None,
            )
        )

        xarray_hist_merged["REFC_entireatmosphere"] = mask_invalid_refc(
            xarray_hist_merged["REFC_entireatmosphere"]
        )

        hist_chunks = (len(GFS_HISTORIC_FORECAST_HOURS), process_chunk, process_chunk)
        xarray_hist_merged["Storm_Distance"] = (
            ("time", "latitude", "longitude"),
            storm_distance_hist.rechunk(hist_chunks).compute(),
        )
        xarray_hist_merged["Storm_Direction"] = (
            ("time", "latitude", "longitude"),
            storm_direction_hist.rechunk(hist_chunks).compute(),
        )

        xarray_hist_merged = xarray_hist_merged.chunk(
            {
                "time": len(GFS_HISTORIC_FORECAST_HOURS),
                "latitude": process_chunk,
                "longitude": process_chunk,
            }
        )
        xarray_hist_merged = xarray_hist_merged.rename({"PRES_surface": "PRES_station"})

        # One shard per variable, so the merge downloads ~30 objects per store
        xarray_hist_merged, encoding = shard_whole_variables(
            xarray_hist_merged, {vname: {"chunks": hist_chunks} for vname in zarr_vars}
        )
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            xarray_hist_merged.to_zarr(
                hist_process_path + "_GFS_Hist_TMP.zarr",
                mode="w",
                write_empty_chunks=True,
                consolidated=False,
                encoding=encoding,
                compute=True,
                chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
            )
        del xarray_hist_merged

        # Publish the store; its root metadata goes last, so later runs skip
        # this cycle only once it is complete
        if save_type == "S3" and no_upload:
            logger.info(
                "NO_UPLOAD enabled, skipping history upload: %s", hist_store_name
            )
            shutil.rmtree(hist_process_path + "_GFS_Hist_TMP.zarr", ignore_errors=True)
        else:
            publish_history_store(
                hist_process_path + "_GFS_Hist_TMP.zarr",
                history_remote,
                hist_store_name,
            )

        logger.info("Completed historic data processing: %s", timestamp)

    # %% Open the history stores to merge ahead of the forecast
    # The history stores keep their own time array, read back below to check the
    # merged times, so it is expected alongside the published variables.
    historic_stores = open_history_arrays(
        history_remote,
        [
            f"{GFS_HISTORIC_STORE_PREFIX}"
            f"{historic_run_stamp(base_time, hours_offset)}.zarr"
            for hours_offset in historic_offsets
        ],
        ("time", *zarr_vars),
        os.path.join(forecast_process_dir, "History"),
    )

    # Check that the history and forecast times are aligned with the expected times
    stored_timesUnix = np.concatenate(
        map_history_stores(lambda store: store["time"][:], historic_stores)
        + [da.from_zarr(forecast_process_path + "_.zarr", component="time").compute()]
    )
    validate_stacked_time_alignment(stacked_timesUnix, stored_timesUnix)

    # %% Stack history and forecast into the (variable, time, y, x) layout
    dask_var_arrays_list = []
    dask_interp_arrays = []

    for dask_var in zarr_vars:
        for historic_store in historic_stores:
            dask_var_arrays_list.append(
                da.from_zarr(historic_store[dask_var], inline_array=True)
            )

        dask_var_arrays_stacked = da.stack(
            dask_var_arrays_list, allow_unknown_chunksizes=True
        )

        if dask_var == "Storm_Distance":
            dask_forecast_array = da.from_zarr(
                forecast_process_path + "_stormDist.zarr"
            )
        elif dask_var == "Storm_Direction":
            dask_forecast_array = da.from_zarr(forecast_process_path + "_stormDir.zarr")
        else:
            dask_forecast_array = da.from_zarr(
                forecast_process_path + "_.zarr", component=dask_var, inline_array=True
            )

        dask_var_arrays_reshaped = da.reshape(
            dask_var_arrays_stacked,
            (
                dask_var_arrays_stacked.shape[0] * dask_var_arrays_stacked.shape[1],
                *GFS_GRID_SHAPE,
            ),
            merge_chunks=False,
        )
        output_array = da.concatenate(
            (dask_var_arrays_reshaped, dask_forecast_array), axis=0
        )
        dask_interp_arrays.append(
            output_array.rechunk(
                (len(stacked_timesUnix), process_chunk, process_chunk)
            ).astype("float32")
        )

        dask_var_arrays_list = []
        logger.info("Processed variable: %s", dask_var)

    # Storm distance can reach very high values that are still correct. CIN is
    # negative by convention and routinely below VALID_DATA_MIN (-100 J/kg);
    # masking it would drop the strong caps the thunderstorm suppressor relies
    # on, so the API bounds it with CLIP_CIN instead.
    merged_arrays_masked = mask_invalid_data(
        da.stack(dask_interp_arrays, axis=0),
        ignoreAxis=[zarr_vars.index("Storm_Distance"), zarr_vars.index("CIN_surface")],
    )

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        merged_arrays_masked.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )
    stacked_array_disk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Interpolate to hourly and write the final store
    stacked_array_interp = interp_time_take_blend(
        stacked_array_disk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
    )
    stacked_array_padded = pad_to_chunk_size(stacked_array_interp, final_chunk)

    gfs_store_name = "GFS.zarr"
    write_final_zarr(
        stacked_array_padded,
        forecast_process_dir + "/" + gfs_store_name,
        final_chunk,
        FINAL_SHARD_SIZES["GFS"],
        zarr_store_workers,
        zarr_vars,
        times=new_hourly_time.values.astype("datetime64[s]").astype("int64"),
    )

    # %% Publish the stores, then the run time the API uses to detect them
    if save_type == "S3" and no_upload:
        logger.info("NO_UPLOAD enabled, skipping forecast S3 publish")
    else:
        publish_ingest_run(
            [gfs_store_name],
            base_time,
            "GFS.time.pickle",
            process_dir=forecast_process_dir,
            remote=forecast_remote,
        )

    # %% Clean up
    if no_upload:
        logger.info("NO_UPLOAD enabled, skipping cleanup of %s", forecast_process_dir)
    else:
        shutil.rmtree(forecast_process_dir)
    logger.info("GFS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
