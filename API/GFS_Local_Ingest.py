"""GFS Local Data Ingestion Script

Downloads GFS forecast and historic GRIB subsets with Herbie and reads them
directly with xarray/cfgrib (no wgrib2 conversion); de-accumulation happens in
xarray/dask. Generates hourly interpolated datasets with multiple variables.

Author: Alexander Rey
Date: September 2023
"""
# ruff: noqa: E402  # dotenv must be loaded before downstream imports

import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da

# Env setup
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest
from tqdm import tqdm

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR, MISSING_DATA
from API.ingest_grib_utils import (
    download_and_validate_subset,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    archive_tmp_zarr_and_upload,
    close_store,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_to_hourly_rate,
    download_extract_historic_archive,
    interp_time_take_blend,
    make_herbie_save_dir,
    mask_invalid_data,
    mask_invalid_refc,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)
from API.utils.storm_proc import compute_storm_fields_from_apcp_dataarray

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
# Grid dimensions for GFS 0.25° global grid
GRID_LAT = 721
GRID_LON = 1440
# Time dimensions
FORECAST_TIME_STEPS = 160
HISTORIC_TIME_STEPS = 6
MAP_TIME_STEPS = 36
MAP_CHUNK_SIZE = 100
# Zarr store settings
DEFAULT_ZARR_WORKERS = 12
DEFAULT_ZARR_CONCURRENCY = 12
DEFAULT_HERBIE_RETRIES = 5
DEFAULT_HERBIE_RETRY_SLEEP = 20

# ============================================================================
# Environment Configuration
# ============================================================================

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

# Paths
forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/reya/Weather/GFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "GFS_Process")
hist_process_path = os.path.join(forecast_process_dir, "GFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod/GFS")
historic_path = os.getenv("historic_path", default="/home/reya/Weather/GFS")

ingest_version = INGEST_VERSION_STR

# Save and upload settings
save_type = os.getenv("save_type", default="Download")


def _parse_bool_env(value: str) -> bool:
    """Parse environment variable as boolean."""
    return value.lower() in ("1", "true", "yes", "on")


no_upload = _parse_bool_env(os.getenv("NO_UPLOAD", os.getenv("no_upload", "")))
force_update = _parse_bool_env(os.getenv("force_update", ""))

# AWS/S3 settings
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


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
}


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
    """Run the ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    herbie_download_retries = positive_int_env(
        "herbie_download_retries", DEFAULT_HERBIE_RETRIES
    )
    herbie_retry_sleep_seconds = positive_int_env(
        "herbie_retry_sleep_seconds", DEFAULT_HERBIE_RETRY_SLEEP
    )

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes(
        DEFAULT_ZARR_WORKERS, DEFAULT_ZARR_CONCURRENCY
    )

    # Define the processing and history chunk size
    process_chunk = CHUNK_SIZES["GFS"]

    # Define the final x/y chunksize
    final_chunk = FINAL_CHUNK_SIZES["GFS"]

    his_period = HISTORY_PERIODS["GFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)
        if not os.path.exists(historic_path):
            os.makedirs(historic_path)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    ################################################################################################
    # Forecast Data Processing
    ################################################################################################

    T0 = time.time()

    latest_run = HerbieLatest(
        model="gfs",
        n=3,
        freq="6h",
        fxx=240,
        product="pgrb2.0p25",
        verbose=False,
        priority=["aws", "nomads"],
        save_dir=herbie_save_dir,
    )

    base_time = latest_run.date

    logger.info(base_time)

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/GFS.time.pickle", save_type, s3
    )
    if (
        not force_update
        and previous_base_time is not None
        and previous_base_time >= base_time
    ):
        logger.info("No Update to GFS, ending")
        return

    if force_update:
        logger.info("force_update enabled, bypassing No Update check")

    zarr_vars = (
        "time",
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
    )

    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 240 hours

    # Define the subset of variables to download as a list of strings
    matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    matchstring_su = (
        ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|PRES|VIS|GUST|CAPE|PRES):surface:.*hour fcst)"
    )
    matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    matchstring_oz = "(:TOZNE:)"
    matchstring_cl = "(:(TCDC|REFC):entire atmosphere:.*hour fcst)"
    matchstring_ap = "(:APCP:surface:0-[1-9]*)"
    matchstring_sl = "(:(PRMSL|DSWRF):)"

    # Merge matchstrings for download
    match_strings = (
        matchstring_2m
        + "|"
        + matchstring_su
        + "|"
        + matchstring_10m
        + "|"
        + matchstring_oz
        + "|"
        + matchstring_cl
        + "|"
        + matchstring_ap
        + "|"
        + matchstring_sl
    )

    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach

    # %% GFS pgrb2.0p25 download and validation

    gfs_range_1 = FORECAST_LEAD_RANGES["GFS_1"]
    gfs_range_2 = FORECAST_LEAD_RANGES["GFS_2"]
    gfs_forecast_hours = [*gfs_range_1, *gfs_range_2]

    wgrib2_exe = wgrib2_path.strip()

    pgrb2_grib_files = download_and_validate_subset(
        product="pgrb2.0p25",
        search=match_strings,
        dataset_name="GFS forecast pgrb2.0p25",
        base_time=base_time,
        wgrib2_exe=wgrib2_exe,
        forecast_hours=gfs_forecast_hours,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
    )

    # %% Download UV data from pgrb2b.0p25

    duvb_match_string = ":DUVB:surface:"

    duvb_grib_files = download_and_validate_subset(
        product="pgrb2b.0p25",
        search=duvb_match_string,
        dataset_name="GFS forecast pgrb2b.0p25 DUVB",
        base_time=base_time,
        wgrib2_exe=wgrib2_exe,
        forecast_hours=gfs_forecast_hours,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
    )

    # %% Read the GRIB subsets with xarray/cfgrib, then de-accumulate in xarray

    xarray_forecast_merged = open_gfs_gribs(pgrb2_grib_files, duvb_grib_files)

    assert len(xarray_forecast_merged.time) == len(gfs_forecast_hours), (
        "Incorrect number of timesteps! Exiting"
    )

    deaccumulate_gfs_fields(xarray_forecast_merged, base_time, gfs_forecast_hours)

    # Create a new time series
    start = xarray_forecast_merged.time.min().values  # Adjust as necessary
    end = xarray_forecast_merged.time.max().values  # Adjust as necessary
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

    # %% FIX THINGS
    # Set REFC values < 5 to 0
    xarray_forecast_merged["REFC_entireatmosphere"] = mask_invalid_refc(
        xarray_forecast_merged["REFC_entireatmosphere"]
    )

    # Clean up APCP/DSWF/DUVB rates that are below 0
    xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged[
        "APCP_surface"
    ].clip(min=0)
    xarray_forecast_merged["DSWRF_surface"] = xarray_forecast_merged[
        "DSWRF_surface"
    ].clip(min=0)
    xarray_forecast_merged["DUVB_surface"] = xarray_forecast_merged[
        "DUVB_surface"
    ].clip(min=0)

    # Compute nearest storm distance and direction from shared scipy/dask utilities.
    distanced_stacked, directions_stacked = compute_storm_fields_from_apcp_dataarray(
        apcp_dataarray=xarray_forecast_merged["APCP_surface"],
        threshold=0.2,
        max_distance_m=None,
    )

    distanced_chunked = distanced_stacked.rechunk((160, process_chunk, process_chunk))
    directions_chunked = directions_stacked.rechunk((160, process_chunk, process_chunk))

    with (
        ProgressBar(),
        dask.config.set(scheduler="threads", num_workers=zarr_store_workers),
    ):
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

    # %% Save merged and processed xarray dataset to disk using zarr with compression

    # Rename PRES_surface to PRES_station for clarity
    # From here on out, it'll be referred to as PRES_station
    xarray_forecast_merged = xarray_forecast_merged.rename(
        {"PRES_surface": "PRES_station"}
    )

    # Save the dataset with compression and filters for all variables
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": 160, "latitude": process_chunk, "longitude": process_chunk}
    )

    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_.zarr",
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # %% Delete to free memory
    del (
        directions_chunked,
        distanced_chunked,
        distanced_stacked,
        directions_stacked,
        xarray_forecast_merged,
    )
    T1 = time.time()

    logger.info(T1 - T0)

    ################################################################################################
    # Historic Data Processing
    # Loop through historical runs (6-hour intervals) and process if not already done
    ################################################################################################

    HISTORICAL_FORECAST_HOURS = range(1, 7)

    for hours_offset in range(his_period, 0, -6):
        if save_type == "S3":
            timestamp = (base_time - pd.Timedelta(hours=hours_offset)).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            s3_path = f"{historic_path}/GFS_Hist_v3{timestamp}.zarr.tar.gz"

            # Check for a done file in S3
            if s3.exists(s3_path.replace(".tar.gz", ".done")):
                logger.info("S3 file exists, skipping: %s", s3_path)
                continue
        else:
            # Local Path Setup
            timestamp = (base_time - pd.Timedelta(hours=hours_offset)).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            local_path = f"{historic_path}/GFS_Hist_v3{timestamp}.zarr"

            # Check for a local done file
            if os.path.exists(local_path.replace(".zarr", ".done")):
                logger.info("Local file exists, skipping: %s", local_path)
                continue

        hist_run_date = base_time - pd.Timedelta(hours=hours_offset)
        hist_forecast_hours = HISTORICAL_FORECAST_HOURS

        logger.info("Downloading historic data: %s", timestamp)

        grib_list = download_and_validate_subset(
            product="pgrb2.0p25",
            search=match_strings,
            dataset_name="GFS historic pgrb2.0p25",
            run_date=hist_run_date,
            forecast_hours=hist_forecast_hours,
            base_time=base_time,
            wgrib2_exe=wgrib2_exe,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
        )

        # Download and add UV data from the pgrb2b product
        duvb_match_string = ":DUVB:surface:"
        grib_list_uv = download_and_validate_subset(
            product="pgrb2b.0p25",
            search=duvb_match_string,
            dataset_name="GFS historic pgrb2b.0p25 DUVB",
            run_date=hist_run_date,
            forecast_hours=hist_forecast_hours,
            base_time=base_time,
            wgrib2_exe=wgrib2_exe,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
        )

        xarray_hist_merged = open_gfs_gribs(grib_list, grib_list_uv)
        deaccumulate_gfs_fields(
            xarray_hist_merged, hist_run_date, list(hist_forecast_hours)
        )

        # Storm distance and direction
        storm_distance_hist, storm_direction_hist = (
            compute_storm_fields_from_apcp_dataarray(
                apcp_dataarray=xarray_hist_merged["APCP_surface"],
                threshold=0.2,
                max_distance_m=None,
            )
        )

        # Set REFC values < 5 to 0
        xarray_hist_merged["REFC_entireatmosphere"] = mask_invalid_refc(
            xarray_hist_merged["REFC_entireatmosphere"]
        )

        # Copy back to main array
        # with ProgressBar():
        xarray_hist_merged["Storm_Distance"] = (
            ("time", "latitude", "longitude"),
            storm_distance_hist.rechunk((6, process_chunk, process_chunk)).compute(),
        )
        xarray_hist_merged["Storm_Direction"] = (
            ("time", "latitude", "longitude"),
            storm_direction_hist.rechunk((6, process_chunk, process_chunk)).compute(),
        )

        # Rechunk the rest of the variables in the merged dataset to the processing chunk size
        xarray_hist_merged = xarray_hist_merged.chunk(
            {
                "time": 6,
                "latitude": process_chunk,
                "longitude": process_chunk,
            }
        )

        # Rename PRES_surface to PRES_station for clarity
        # From here on out, it'll be referred to as PRES_station
        xarray_hist_merged = xarray_hist_merged.rename({"PRES_surface": "PRES_station"})

        # Save merged and processed xarray dataset to disk using zarr with compression
        # Define the path to save the zarr dataset with the run time in the filename
        # format the time following iso8601

        # Save the dataset with compression and filters for all variables
        # Use the same encoding as last time but with larger chunks to speed up read times
        encoding = {
            vname: {"chunks": (6, process_chunk, process_chunk)}
            for vname in zarr_vars[1:]
        }

        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            xarray_hist_merged.to_zarr(
                hist_process_path + "_GFS_Hist_TMP.zarr",
                mode="w",
                consolidated=False,
                encoding=encoding,
                compute=True,
                chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
            )

        # Clear the xarray dataset from memory
        del xarray_hist_merged

        # Save a done file to s3 to indicate that the historic data has been processed
        if save_type == "S3":
            if no_upload:
                logger.info(
                    "NO_UPLOAD enabled, skipping historic S3 archive upload: %s",
                    s3_path,
                )
                shutil.rmtree(
                    hist_process_path + "_GFS_Hist_TMP.zarr", ignore_errors=True
                )
            else:
                archive_tmp_zarr_and_upload(
                    tmp_zarr_path=hist_process_path + "_GFS_Hist_TMP.zarr",
                    s3_path=s3_path,
                    archive_member_name="GFS_Hist.zarr",
                    s3=s3,
                )
        else:
            # Move to Local Path
            os.rename(hist_process_path + "_GFS_Hist_TMP.zarr", local_path)

            done_file = local_path.replace(".zarr", ".done")
            with open(done_file, "w") as f:
                f.write("Done")

        logger.info("Completed historic data processing: %s", timestamp)

    ################################################################################################
    # Merge Historic and Forecast Datasets
    ################################################################################################
    if save_type == "S3":
        local_temp_dir = forecast_process_path + "_s3_temp_downloads"
        os.makedirs(local_temp_dir, exist_ok=True)

        # The function that downloads and extracts a single timestamp
        def download_and_extract(timestamp):
            # Names expected locally
            final_zarr_name = f"GFS_Hist_v3{timestamp}.zarr"
            extracted_path = download_extract_historic_archive(
                s3=s3,
                historic_path=historic_path,
                final_zarr_name=final_zarr_name,
                extracted_store_name="GFS_Hist.zarr",
                local_temp_dir=local_temp_dir,
                expected_vars=zarr_vars,
            )
            if extracted_path is None:
                tqdm.write(
                    f"Error: GFS_Hist.zarr not found inside archive for {timestamp}"
                )
            return extracted_path

        # Generate target timestamps
        timestamps = [
            (base_time - pd.Timedelta(hours=hours_offset)).strftime("%Y%m%dT%H%M%SZ")
            for hours_offset in range(his_period, 1, -6)
        ]

        logger.info(
            "Downloading and extracting %d archives from S3...", len(timestamps)
        )

        # Execute downloads in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(
                tqdm(
                    executor.map(download_and_extract, timestamps),
                    total=len(timestamps),
                    desc="S3 Archive Sync",
                )
            )

        # Filter out the missing files (None values) and keep the valid paths
        historic_zarr_paths = [path for path in results if path is not None]
    else:
        historic_zarr_paths = [
            f"{historic_path}/GFS_Hist_v3{(base_time - pd.Timedelta(hours=hours_offset)).strftime('%Y%m%dT%H%M%SZ')}.zarr"
            for hours_offset in range(his_period, 1, -6)
        ]

    # Dask Setup
    dask_var_arrays_list = []
    dask_interp_arrays = []

    for var_idx, dask_var in enumerate(zarr_vars[:]):
        for historic_zarr_path in historic_zarr_paths:
            # If not found in array, use MISSING_DATA to show missing
            try:
                dask_var_arrays_list.append(
                    da.from_zarr(
                        historic_zarr_path, component=dask_var, inline_array=True
                    )
                )
            # Add a fallback in case of a FileNotFoundError
            except FileNotFoundError:
                logger.info(
                    "File not found, adding NaN array for: %s", historic_zarr_path
                )
                dask_var_arrays_list.append(
                    da.full(
                        (HISTORIC_TIME_STEPS, GRID_LAT, GRID_LON), MISSING_DATA
                    ).rechunk((HISTORIC_TIME_STEPS, process_chunk, process_chunk))
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

        if dask_var == "time":
            # Create a time array with the same shape
            # This is because multiple steps are stored in each file
            dask_var_arrays_reshaped = da.reshape(
                dask_var_arrays_stacked,
                (
                    dask_var_arrays_stacked.shape[0] * dask_var_arrays_stacked.shape[1],
                    1,
                ),
                merge_chunks=False,
            )
            dask_times_concatenated = da.concatenate(
                (da.squeeze(dask_var_arrays_reshaped), dask_forecast_array), axis=0
            ).astype("float32")

            validate_stacked_time_alignment(
                stacked_timesUnix, dask_times_concatenated.compute()
            )

            output_array = da.broadcast_to(
                dask_times_concatenated[:, None, None],
                (dask_times_concatenated.shape[0], GRID_LAT, GRID_LON),
            ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

            dask_interp_arrays.append(output_array)

        else:
            dask_var_arrays_reshaped = da.reshape(
                dask_var_arrays_stacked,
                (
                    dask_var_arrays_stacked.shape[0] * dask_var_arrays_stacked.shape[1],
                    GRID_LAT,
                    GRID_LON,
                ),
                merge_chunks=False,
            )
            output_array = da.concatenate(
                (dask_var_arrays_reshaped, dask_forecast_array), axis=0
            )

            dask_interp_arrays.append(
                output_array[:, :, :]
                .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
                .astype("float32")
            )

        dask_var_arrays_list = []
        logger.info("Processed variable: %s", dask_var)

    # Merge the arrays into a single 4D array
    merged_arrays = da.stack(dask_interp_arrays, axis=0)

    # Mask out invalid data
    # Ignore storm distance, since it can reach very high values that are still correct
    merged_arrays_masked = mask_invalid_data(
        merged_arrays, ignoreAxis=[zarr_vars.index("Storm_Distance")]
    )

    # Write out to disk
    # This intermediate step is necessary to avoid memory overflow
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        merged_arrays_masked.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

    # Read in stacked 4D array back in
    stacked_array_disk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # 1. Interpolate the stacked array to be hourly along the time axis
    stacked_array_interp = interp_time_take_blend(
        stacked_array_disk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
    )

    # 2. Pad to chunk size
    stacked_array_padded = pad_to_chunk_size(stacked_array_interp, final_chunk)

    # 3. Write the final store
    gfs_store_name = "GFS.zarr.zip" if save_type == "S3" else "GFS.zarr"
    gfs_maps_store_name = "GFS_Maps.zarr.zip" if save_type == "S3" else "GFS_Maps.zarr"
    with ProgressBar():
        write_final_zarr(
            stacked_array_padded,
            forecast_process_dir + "/" + gfs_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # Rechunk map data for faster web access
    # Variables included: time(0), TMP(4), UGRD(8), VGRD(9), PRATE(10), PACCUM(11),
    #                     CRAIN(12), CICEP(13), CSNOW(14), CFRZR(15), REFC(21)
    # Map extent: -12 to +24 hours (36 hours total)
    # Map chunk size: 100x100 pixels for fast tiling

    MAP_VAR_INDICES = [0, 4, 8, 9, 10, 11, 12, 13, 14, 15, 21]

    # Add padding for map chunking (100x100)
    stacked_array_maps = pad_to_chunk_size(stacked_array_disk, MAP_CHUNK_SIZE)

    if save_type == "S3":
        zarr_store_maps = zarr.storage.ZipStore(
            forecast_process_dir + "/" + gfs_maps_store_name, mode="a"
        )
    else:
        zarr_store_maps = zarr.storage.LocalStore(
            forecast_process_dir + "/" + gfs_maps_store_name
        )

    for var_idx in MAP_VAR_INDICES:
        # Create a zarr backed dask array
        zarr_array = zarr.create_array(
            store=zarr_store_maps,
            name=zarr_vars[var_idx],
            shape=(
                MAP_TIME_STEPS,
                stacked_array_maps.shape[2],
                stacked_array_maps.shape[3],
            ),
            chunks=(MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )

        with (
            ProgressBar(),
            dask.config.set(scheduler="threads", num_workers=zarr_store_workers),
        ):
            da.rechunk(
                stacked_array_maps[var_idx, his_period - 12 : his_period + 24, :, :],
                (MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
            ).to_zarr(zarr_array, overwrite=True, compute=True)

        logger.info("Created map data for %s", zarr_vars[var_idx])

    close_store(zarr_store_maps)

    # %% Publish the stores, then the run time the API uses to detect them
    if save_type == "S3" and no_upload:
        logger.info("NO_UPLOAD enabled, skipping forecast S3 publish")
    else:
        publish_ingest_run(
            [gfs_store_name, gfs_maps_store_name],
            base_time,
            "GFS.time.pickle",
            process_dir=forecast_process_dir,
            publish_dir=publish_dir,
            save_type=save_type,
            s3=s3,
        )

    # Clean up
    if no_upload:
        logger.info("NO_UPLOAD enabled, skipping cleanup of %s", forecast_process_dir)
    else:
        shutil.rmtree(forecast_process_dir)

    # Timing
    T1 = time.time()
    logger.info(T1 - T0)

    # Test Read
    # G = zarr.open(forecast_path + "/" + ingest_version + "/GFS.zarr", read_only=True)
    # G.info


if __name__ == "__main__":
    main()
