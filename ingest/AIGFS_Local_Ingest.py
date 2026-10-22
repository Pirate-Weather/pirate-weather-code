"""AIGFS forecast ingest.

Downloads mean sea level pressure, 2 m temperature, 10 m wind and precipitation
from NOAA's AI Global Forecast System (AIGFS, an EAGLE GraphCast-based model on a
0.25 degree global grid, 6-hourly leads to 240 h) for the newest 6-hourly cycle.
The subsets are converted with a single wgrib2 NetCDF pass, precipitation is
turned from run totals into an hourly rate, and the fields are interpolated to an
hourly axis and published as ``AIGFS.zarr`` plus ``AIGFS.time.pickle``. The store
starts at the first forecast lead (run + 6 h); the API falls back to other
sources for earlier hours.

Alexander Rey, September 2023
"""

import logging
import os
import shutil
import time
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from herbie import HerbieLatest

from API.constants.shared_const import INGEST_VERSION_STR
from API.herbie_custom_templates import register_aws_aigfs_aigefs_templates
from API.ingest_grib_utils import download_and_validate_subset, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_to_hourly_rate,
    interp_time_take_blend,
    mask_invalid_data,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# 0.25 degree global latitude/longitude grid
AIGFS_GRID_SHAPE = (721, 1440)
AIGFS_FORECAST_HOURS = FORECAST_LEAD_RANGES["AIGFS"]
# Unlike the experimental AWS EAGLE ensemble (see AIGEFS), the AWS AIGFS bucket
# mirrors the operational NOMADS product, so it is tried first.
AIGFS_SOURCE_PRIORITY = ["aws", "nomads"]
# APCP is matched as the run total (0-N h) so it only needs differencing
AIGFS_MATCH_STRINGS = (
    ":((TMP):2 m above ground:)"
    "|(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    r"|:APCP:surface:0-\d+\s+hour acc fcst:"
    "|(:(PRMSL):)"
)

# Variables in the published store; the order is the API's AIGFS index map
zarr_vars = (
    "time",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "APCP_surface",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# When running the docker container, this should be "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/AIGFS")
forecast_process_path = os.path.join(forecast_process_dir, "AIGFS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/AIGFS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

AIGFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
AIGFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_aigfs_forecast(base_time: pd.Timestamp) -> list[str]:
    """Download and validate the AIGFS forecast GRIB subsets with bounded retries."""
    return download_and_validate_subset(
        model="aigfs",
        product="sfc",
        search=AIGFS_MATCH_STRINGS,
        dataset_name="AIGFS forecast",
        base_time=base_time,
        forecast_hours=list(AIGFS_FORECAST_HOURS),
        priority=AIGFS_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=AIGFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=AIGFS_RETRY_SLEEP_SECONDS,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


def main() -> None:
    """Run the AIGFS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    # Use the repository Herbie templates for the current AWS/NOMADS paths
    register_aws_aigfs_aigefs_templates()
    # The final write used dask's default thread pool (one thread per CPU).
    zarr_store_workers = configure_zarr_writes(default_workers=os.cpu_count() or 2)

    process_chunk = CHUNK_SIZES["GFS"]
    final_chunk = FINAL_CHUNK_SIZES["GFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="aigfs",
        n=3,
        freq="6h",
        fxx=AIGFS_FORECAST_HOURS[-1],
        product="sfc",
        verbose=False,
        priority=AIGFS_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "AIGFS.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to AIGFS, ending")
        return
    logger.info("Processing AIGFS run %s", base_time)

    # %% Forecast hours: 6-hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(AIGFS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download and convert the forecast in one wgrib2 pass
    grib_list = download_aigfs_forecast(base_time)

    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        "cat " + " ".join(grib_list) + f" | {wgrib2_path} - -netcdf {merged_nc_path}",
        "AIGFS wgrib2 NetCDF conversion",
    )
    xarray_forecast_merged = xr.open_mfdataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"AIGFS times {xarray_forecast_merged['time'].values} do not match "
            f"the expected forecast leads {AIGFS_FORECAST_HOURS}"
        )

    # Hourly target axis from the first lead (run + 6 h) to the last
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0], end=forecast_valid_times[-1], freq="h"
    )
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # %% Correct precipitation: APCP is a run total (0-N h), so difference
    # consecutive leads into 6-hour totals and divide into an hourly rate. GRIB
    # packing can make a difference slightly negative, which the helper clamps.
    xarray_forecast_merged["APCP_surface"].data = deaccumulate_to_hourly_rate(
        xarray_forecast_merged["APCP_surface"].data,
        AIGFS_FORECAST_HOURS,
        time_axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
    )

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": -1, "latitude": process_chunk, "longitude": process_chunk}
    )
    xarray_forecast_merged.to_zarr(
        forecast_process_path + "_forecast.zarr",
        mode="w",
        consolidated=False,
        compute=True,
    )
    xarray_forecast_merged.close()
    del xarray_forecast_merged

    # The merged NetCDF is large and no longer needed
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_forecast.zarr",
            component=dask_var,
            inline_array=True,
        )

        if dask_var == "time":
            npCatTimes = daskForecastArray.astype("float32").compute()
            # Check that the stored times are aligned with the expected forecast times
            validate_stacked_time_alignment(stacked_timesUnix, npCatTimes)
            # The API reads time from the store like any other variable
            daskVarArrayList.append(
                da.from_array(
                    np.tile(
                        npCatTimes[:, np.newaxis, np.newaxis], (1, *AIGFS_GRID_SHAPE)
                    )
                ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
            )
        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
                    (len(stacked_timesUnix), process_chunk, process_chunk)
                ).astype("float32")
            )

    daskVarArrayListMergeNaN = mask_invalid_data(da.stack(daskVarArrayList, axis=0))

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
    daskVarArrayListMergeNaN.to_zarr(
        forecast_process_path + "_stack.zarr", overwrite=True, compute=True
    )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Interpolate to hourly and write the final store
    daskVarArrayStackDiskInterp = interp_time_take_blend(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    aigfs_store_name = "AIGFS.zarr"
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/" + aigfs_store_name,
        final_chunk,
        zarr_store_workers,
        zarr_vars,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [aigfs_store_name],
        base_time,
        "AIGFS.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("AIGFS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
