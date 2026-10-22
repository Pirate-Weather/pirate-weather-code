"""HRRR sub-hourly (15-minute) ingest.

Downloads the 15-minute fields of the newest hourly HRRR cycle's sub-hourly
(subh) product (3 km CONUS Lambert conformal grid, 1799 x 1059, hourly cycles,
leads 1-5 h with four 15-minute steps per file), rotates the winds to
earth-relative and publishes ``SubH.zarr`` and ``SubH.time.pickle`` for the
currently and minutely blocks. The store starts at the first 15-minute step
(run + 15 min).

Alexander Rey, November 2023
"""

import logging
import os
import shutil
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from herbie.fast import Herbie_latest

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_subset, run_checked
from API.ingest_utils import (
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    mask_invalid_data,
    mask_invalid_refc,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# Sub-hourly files to download; each lead file holds the four 15-minute steps
# ending at that lead hour. Only the first hours are needed for currently/minutely.
SUBH_FORECAST_HOURS = list(range(1, 6))
SUBH_STEPS_PER_HOUR = 4
SUBH_STEP_MINUTES = 15
# Lead used to detect that a cycle's subh product is available
SUBH_LATEST_CHECK_HOUR = 6
HRRR_SOURCE_PRIORITY = ["aws", "google", "nomads"]
# (y, x) shape of the HRRR CONUS Lambert conformal grid
HRRR_GRID_SHAPE = (1059, 1799)
# The native HRRR grid, passed to wgrib2 -new_grid so that -new_grid_winds earth
# rotates the grid-relative U/V to earth-relative without moving any grid point.
# From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
HRRR_LAMBERT_GRID = (
    "lambert:262.500000:38.500000:38.500000:38.500000 "
    "237.280472:1799:3000.000000 21.138123:1059:3000.000000"
)

# Herbie match strings for the subh product. Its APCP records are already
# 15-minute buckets (e.g. 60-75 min acc), so no de-accumulation is needed.
SUBH_MATCH_STRINGS = (
    ":((DPT|TMP|SPFH):2 m above ground:)"
    "|:((CRAIN|CICEP|CSNOW|CFRZR|PRES|PRATE|VIS|GUST|DSWRF):surface:.*min fcst)"
    "|(:(UGRD|VGRD):10 m above ground:.*min fcst)"
    "|(:REFD:1000 m above ground:)"
    "|(:APCP:surface:)"
)

# Variables in the published store; the order is the API's HRRR_SUBH index map
zarr_vars = (
    "time",
    "GUST_surface",
    "PRES_surface",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "REFD_1000maboveground",
    "APCP_surface",
    "VIS_surface",
    "SPFH_2maboveground",
    "DSWRF_surface",
)

# SubH has no entry in the shared CHUNK_SIZES / FINAL_CHUNK_SIZES tables
SUBH_PROCESS_CHUNK = 100
SUBH_FINAL_CHUNK = 5

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/SubH")
forecast_process_path = forecast_process_dir + "/SubH_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/SubH")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

SUBH_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
SUBH_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_subh_subset(base_time: pd.Timestamp) -> list[str]:
    """Download and validate the HRRR subh subsets with bounded retries."""
    return download_and_validate_subset(
        model="hrrr",
        product="subh",
        search=SUBH_MATCH_STRINGS,
        dataset_name="HRRR sub-hourly",
        base_time=base_time,
        forecast_hours=SUBH_FORECAST_HOURS,
        priority=HRRR_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=SUBH_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=SUBH_RETRY_SLEEP_SECONDS,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


def main() -> None:
    """Run the HRRR sub-hourly ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()

    process_chunk = SUBH_PROCESS_CHUNK
    final_chunk = SUBH_FINAL_CHUNK

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

    # %% Determine the model run: the newest hourly cycle with subh output
    latest_run = Herbie_latest(
        model="hrrr",
        n=3,
        freq="1h",
        fxx=[SUBH_LATEST_CHECK_HOUR],
        product="subh",
        verbose=False,
        priority=HRRR_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "SubH.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to SubH, ending")
        return
    logger.info("Processing SubH run %s", base_time)

    # %% Forecast steps: 15-minute steps from run + 15 min to the last lead hour
    subh_step_count = len(SUBH_FORECAST_HOURS) * SUBH_STEPS_PER_HOUR
    forecast_valid_times = (
        pd.Timestamp(base_time)
        + pd.to_timedelta(
            np.arange(1, subh_step_count + 1) * SUBH_STEP_MINUTES, unit="min"
        )
    ).values

    # %% Download, merge, rotate winds and convert to NetCDF
    grib_list = download_subh_subset(base_time)

    merged_grib_path = forecast_process_path + "_wgrib2_merged.grib2"
    regrid_path = forecast_process_path + "_wgrib2_merged.regrid"
    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        "cat " + " ".join(grib_list) + f" | {wgrib2_path} - -grib {merged_grib_path}",
        "HRRR sub-hourly wgrib2 GRIB merge",
    )
    run_checked(
        f"{wgrib2_path} {merged_grib_path} -new_grid_winds earth "
        f"-new_grid {HRRR_LAMBERT_GRID} {regrid_path}",
        "HRRR sub-hourly wgrib2 wind rotation",
    )
    run_checked(
        f"{wgrib2_path} {regrid_path} -netcdf {merged_nc_path}",
        "HRRR sub-hourly wgrib2 NetCDF conversion",
    )

    xarray_forecast_merged = xr.open_mfdataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"HRRR sub-hourly run {base_time} times "
            f"{xarray_forecast_merged['time'].values} do not match the expected "
            f"{subh_step_count} 15-minute steps from {forecast_valid_times[0]}"
        )

    # %% Correct HRRR-specific quirks
    # Set REFD values below the reflectivity threshold to 0
    xarray_forecast_merged["REFD_1000maboveground"] = mask_invalid_refc(
        xarray_forecast_merged["REFD_1000maboveground"]
    )

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": subh_step_count, "x": process_chunk, "y": process_chunk}
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_xr_merged.zarr",
            mode="w",
            consolidated=False,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    xarray_forecast_merged.close()
    del xarray_forecast_merged

    # The wgrib2 intermediates are large and no longer needed
    os.remove(merged_grib_path)
    os.remove(regrid_path)
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_xr_merged.zarr",
            component=dask_var,
            inline_array=True,
        )

        if dask_var == "time":
            npCatTimes = daskForecastArray.astype("float32").compute()
            # The API reads time from the store like any other variable
            daskVarArrayList.append(
                da.from_array(
                    np.tile(
                        npCatTimes[:, np.newaxis, np.newaxis], (1, *HRRR_GRID_SHAPE)
                    )
                ).rechunk((len(npCatTimes), process_chunk, process_chunk))
            )
        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
                    (len(npCatTimes), process_chunk, process_chunk)
                ).astype("float32")
            )

    daskVarArrayListMergeNaN = mask_invalid_data(da.stack(daskVarArrayList, axis=0))

    # Write the stacked array to disk before the final rechunk; doing both in one
    # graph overflows memory
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr", overwrite=True, compute=True
        )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Write the final store (no time interpolation for 15-minute data)
    daskVarArrayStackDisk_main = pad_to_chunk_size(daskVarArrayStackDisk, final_chunk)

    subh_store_name = "SubH.zarr"
    write_final_zarr(
        daskVarArrayStackDisk_main,
        forecast_process_dir + "/" + subh_store_name,
        final_chunk,
        zarr_store_workers,
        zarr_vars,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [subh_store_name],
        base_time,
        "SubH.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("SubH run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
