"""ECMWF AIFS forecast ingest.

Downloads the newest 6-hourly ECMWF AIFS open-data cycle (0.25 degree global
grid, 6-hourly leads to 240 h): the deterministic ``oper`` surface fields plus
the ``enfo`` ensemble total precipitation and snowfall, which are reduced to
precipitation probability, mean, standard deviation and a rain/snow type. The
forecast is interpolated to an hourly axis and published as ``ECMWF_AIFS.zarr``
plus ``ECMWF_AIFS.time.pickle``. The store starts at the first lead (run + 6 h);
the API falls back to other sources for earlier hours.

Alexander Rey, March 2025
"""

import logging
import os
import shutil
import time
import warnings
from contextlib import nullcontext

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest

from API.constants.shared_const import (
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    INGEST_VERSION_STR,
)
from API.constants.unit_const import MM_PER_M
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    configure_zarr_writes,
    interp_time_take_blend,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
AIFS_SOURCE_PRIORITY = ["google", "ecmwf", "aws"]
AIFS_RUN_FREQ = "6h"
AIFS_FORECAST_HOURS = FORECAST_LEAD_RANGES["ECMWF_AIFS"]
# Every AIFS lead is 6 hours after the previous one
AIFS_LEAD_STEP_HOURS = 6

# Deterministic fields: 2 m temperature/dew point, 10 m wind, mean sea level
# pressure and total cloud cover
AIFS_OPER_MATCH_STRINGS = "(:(2d|2t):)|(:(10u|10v):)|(:(msl):)|(:(tcc):)"
# Ensemble total precipitation and snowfall (accumulated from the run start)
AIFS_ENFO_MATCH_STRING = r":((tp|sf):sfc:\d+):"

# cfgrib can only open one hypercube per call, so the oper files are opened once
# per level type and merged (in this order).
AIFS_OPER_CFGRIB_GROUPS = (
    {"typeOfLevel": "heightAboveGround", "level": [2]},
    {"typeOfLevel": "heightAboveGround", "level": [10]},
    {"typeOfLevel": "meanSea"},
    {"typeOfLevel": "entireAtmosphere"},
)

# AIFS has no precipitation type output. Classify the ensemble mean as snow when
# snowfall is more than this fraction of total precipitation, otherwise rain.
AIFS_SNOW_FRACTION_THRESHOLD = 0.5
# ECMWF GRIB precipitation type codes (code table 4.201)
PTYPE_NONE = 0
PTYPE_RAIN = 1
PTYPE_SNOW = 5

# Variables in the published store; the order is the API's ECMWF_AIFS index map
zarr_vars = (
    "time",
    "msl",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "ptype",
    "tcc",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
)
# Categorical precipitation type is interpolated with nearest neighbour
NEAREST_INTERP_VARS = ("ptype",)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/ECMWF_AIFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "ECMWF_AIFS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/ECMWF_AIFS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

AIFS_MAX_THREADS = positive_int_env("aifs_download_threads", 4)
AIFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
AIFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)
AIFS_VERBOSE_LOGS = os.getenv("aifs_verbose_logs", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def download_aifs_subset(
    product: str,
    search: str,
    dataset_name: str,
    run_date: pd.Timestamp,
    forecast_hours: list[int],
) -> list[str]:
    """Download and validate an AIFS GRIB subset with bounded concurrency."""
    return download_and_validate_subset(
        model="aifs",
        product=product,
        search=search,
        dataset_name=dataset_name,
        base_time=run_date,
        forecast_hours=forecast_hours,
        wgrib2_exe=wgrib2_path.rstrip(),
        herbie_save_dir=tmp_dir,
        herbie_download_retries=AIFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=AIFS_RETRY_SLEEP_SECONDS,
        priority=AIFS_SOURCE_PRIORITY,
        herbie_kwargs={"max_threads": AIFS_MAX_THREADS},
        download_max_threads=AIFS_MAX_THREADS,
        download_verbose=AIFS_VERBOSE_LOGS,
    )


def aifs_progress_bar():
    """Show Dask progress only when verbose AIFS logging is enabled."""
    return ProgressBar() if AIFS_VERBOSE_LOGS else nullcontext()


def open_aifs_cfgrib(grib_paths: list[str], filter_by_keys=None) -> xr.Dataset:
    """Open one cfgrib message group across all lead files, sorted by step."""
    backend_kwargs = {"filter_by_keys": filter_by_keys} if filter_by_keys else {}
    return xr.open_mfdataset(
        grib_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs=backend_kwargs,
    ).sortby("step")


def main() -> None:
    """Run the ECMWF AIFS forecast ingest."""
    logging.basicConfig(level=logging.INFO)

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    # The AIFS final write has always used 4 workers (env can still override).
    zarr_store_workers = configure_zarr_writes(default_workers=4)

    process_chunk = CHUNK_SIZES["ECMWF"]
    final_chunk = FINAL_CHUNK_SIZES["ECMWF"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="aifs",
        n=3,
        freq=AIFS_RUN_FREQ,
        fxx=AIFS_FORECAST_HOURS[-1],
        product="oper",
        verbose=AIFS_VERBOSE_LOGS,
        priority=AIFS_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/ECMWF_AIFS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to ECMWF AIFS, ending")
        return
    logger.info("Processing ECMWF AIFS run %s", base_time)

    # %% Download and convert the ensemble precipitation and snowfall
    ens_paths = download_aifs_subset(
        product="enfo",
        search=AIFS_ENFO_MATCH_STRING,
        dataset_name="AIFS ensemble forecast",
        run_date=base_time,
        forecast_hours=AIFS_FORECAST_HOURS,
    )
    ens_mf = open_aifs_cfgrib(ens_paths)

    # tp and sf accumulate from the run start: difference consecutive leads,
    # keeping the first lead's 0-6 h total as is
    for accum_var in ("tp", "sf"):
        ens_mf[accum_var + "d"] = ens_mf[accum_var].diff(dim="step")
        ens_mf[accum_var + "d"] = xr.where(
            ens_mf.step == ens_mf.step.isel(step=0),
            ens_mf[accum_var].isel(step=0),
            ens_mf[accum_var + "d"],
        )

    # AIFS ensemble tp and sf are in kg m**-2 (= mm), not metres like IFS.
    # Convert mm/6h to m/h so the API's m/h -> mm/h conversion applies to both.
    ens_mf["tpd"] = ens_mf["tpd"] / (AIFS_LEAD_STEP_HOURS * MM_PER_M)
    ens_mf["sfd"] = ens_mf["sfd"] / (AIFS_LEAD_STEP_HOURS * MM_PER_M)

    # Fraction of members above the shared ensemble threshold (mm/h -> m/h)
    X3_Precipitation_Prob = (
        ens_mf["tpd"] > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H / MM_PER_M
    ).sum(dim="number") / ens_mf.sizes["number"]
    X3_Precipitation_StdDev = ens_mf["tpd"].std(dim="number")
    X3_Precipitation_Mean = ens_mf["tpd"].mean(dim="number")
    X3_Snowfall_Mean = ens_mf["sfd"].mean(dim="number")

    # Precipitation type from the ensemble-mean snow fraction (no type in AIFS)
    X3_pType = xr.where(
        X3_Precipitation_Mean > 0,
        xr.where(
            X3_Snowfall_Mean > X3_Precipitation_Mean * AIFS_SNOW_FRACTION_THRESHOLD,
            PTYPE_SNOW,
            PTYPE_RAIN,
        ),
        PTYPE_NONE,
    )

    xr_ensoOut = xr.Dataset(
        {
            "Precipitation_Prob": X3_Precipitation_Prob,
            "APCP_StdDev": X3_Precipitation_StdDev,
            "APCP_Mean": X3_Precipitation_Mean,
            "ptype": X3_pType,
        },
        coords={
            "time": ens_mf["step"],
            "latitude": ens_mf["latitude"],
            "longitude": ens_mf["longitude"],
        },
    )

    # %% Download and convert the deterministic fields
    aifs_paths = download_aifs_subset(
        product="oper",
        search=AIFS_OPER_MATCH_STRINGS,
        dataset_name="AIFS deterministic forecast",
        run_date=base_time,
        forecast_hours=AIFS_FORECAST_HOURS,
    )
    aifs_mf = xr.merge(
        [open_aifs_cfgrib(aifs_paths, group) for group in AIFS_OPER_CFGRIB_GROUPS],
        compat="override",
    )

    xarray_forecast_merged = xr.merge(
        [aifs_mf, xr_ensoOut], compat="override", join="outer"
    )

    # Index by valid time (run time + step) and check every lead is present
    forecast_valid_times = pd.to_datetime(
        xarray_forecast_merged.time.values
    ) + pd.to_timedelta(xarray_forecast_merged.step.values, unit="h")
    expected_valid_times = pd.Timestamp(base_time) + pd.to_timedelta(
        AIFS_FORECAST_HOURS, unit="h"
    )
    if not np.array_equal(forecast_valid_times.values, expected_valid_times.values):
        raise RuntimeError(
            f"ECMWF AIFS valid times {forecast_valid_times.values} do not match "
            f"the expected leads {AIFS_FORECAST_HOURS} from {base_time}"
        )
    xarray_forecast_merged = (
        xarray_forecast_merged.assign_coords(time=("step", forecast_valid_times))
        .swap_dims({"step": "time"})
        .drop_vars("step")
    )

    # Hourly target axis from the first lead (run + 6 h) to the last; the API
    # falls back to other sources for run + 1 h to + 5 h
    new_hourly_time = pd.date_range(
        start=xarray_forecast_merged.time[0].values,
        end=xarray_forecast_merged.time[-1].values,
        freq="h",
    )
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(xarray_forecast_merged.time),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )
    with aifs_progress_bar():
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_merged.zarr",
            mode="w",
            consolidated=False,
            compute=True,
        )
    del xarray_forecast_merged, aifs_mf, ens_mf, xr_ensoOut

    logger.info("ECMWF AIFS forecast converted in %.1f s", time.time() - T0)

    # %% Read the forecast back and stack into the (variable, time, y, x) layout
    ds = xr.open_zarr(f"{forecast_process_path}_merged.zarr", consolidated=False)

    # The API reads variables by position, so a missing one must not shift the rest
    missing_vars = [v for v in zarr_vars[1:] if v not in ds.data_vars]
    if missing_vars:
        raise RuntimeError(f"ECMWF AIFS run {base_time} is missing {missing_vars}")

    # Out-of-range values become NaN
    for var in zarr_vars[1:]:
        ds_clip = ds[var]
        if np.issubdtype(ds_clip.dtype, np.number):
            ds[var] = ds_clip.where(
                (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
            )

    stacked_timesUnix = (ds.time.values - unix_epoch) / one_second

    ds_rename = ds.rename({"time": "stacked_time"})
    # The API reads time from the store like any other variable
    ds_rename["time"] = (
        ((ds_rename["stacked_time"] - unix_epoch) / one_second)
        .astype("float32")
        .expand_dims(latitude=ds_rename.latitude, longitude=ds_rename.longitude)
        .transpose("stacked_time", "latitude", "longitude")
    )
    ds_stack = ds_rename[list(zarr_vars)].to_array(dim="var", name="var")
    ds_chunk = ds_stack.chunk(
        {
            "var": 1,
            "stacked_time": len(stacked_timesUnix),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    # Interim zarr save of the stacked array. Not necessary for local, but speeds
    # things up on S3
    with aifs_progress_bar():
        ds_chunk.to_zarr(forecast_process_path + "_stack.zarr", mode="w")
    daskVarArrayStackDisk = da.from_zarr(
        forecast_process_path + "_stack.zarr", component="__xarray_dataarray_variable__"
    )

    # %% Interpolate to hourly and write the final store
    daskVarArrayStackDiskInterp = interp_time_take_blend(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        nearest_vars=[zarr_vars.index(v) for v in NEAREST_INTERP_VARS],
        dtype="float32",
        fill_value=np.nan,
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    aifs_store_name = "ECMWF_AIFS.zarr.zip" if save_type == "S3" else "ECMWF_AIFS.zarr"
    with aifs_progress_bar():
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + aifs_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [aifs_store_name],
        base_time,
        "ECMWF_AIFS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("ECMWF AIFS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
