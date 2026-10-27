"""RRFS CONUS 6-hourly extended (18-84 h) ingest.

Downloads the 2D fields (2dfld product) of the newest 6-hourly RRFS cycle on the
CONUS domain (3 km Lambert conformal grid, 1799 x 1059; only the 00/06/12/18 UTC
cycles run to 84 h), hourly leads 18-83 h, rotates the winds to earth-relative
and publishes ``RRFS_6H.zarr`` and ``RRFS_6H.time.pickle``. The store starts at
lead 18 h (run + 18 h); leads 0-17 come from the hourly RRFS ingest.

Alexander Rey, September 2026
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
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    FORECAST_LEAD_RANGES,
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
RRFS_FORECAST_HOURS = FORECAST_LEAD_RANGES["RRFS_6H"]
RRFS_SOURCE_PRIORITY = ["aws", "nomads"]
# The native RRFS CONUS grid (identical to HRRR's), passed to wgrib2 -new_grid so
# that -new_grid_winds earth rotates the grid-relative U/V to earth-relative
# without moving any grid point.
# From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
RRFS_LAMBERT_GRID = (
    "lambert:262.500000:38.500000:38.500000:38.500000 "
    "237.280472:1799:3000.000000 21.138123:1059:3000.000000"
)

# Herbie match strings for the 2dfld product
RRFS_MATCH_STRINGS = (
    ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    "|:((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST|DSWRF|CAPE|CIN):surface:.*hour fcst)"
    "|(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    "|(:TCDC:entire atmosphere:.*hour fcst)"
    # Like HRRR, each file holds a 0-N h run total and an (N-1)-N h bucket. This
    # selects the one-hour bucket (17-18 ... 82-83 h), which is already an hourly
    # total; differencing run totals would leave the first lead (18 h) as a
    # 0-18 h total.
    "|(:APCP:surface:[1-9][0-9]*-[0-9]+ hour acc)"
    # RRFS carries several aerosol MASSDEN fields at 8 m; smoke is the
    # particulate organic matter one.
    "|:(MASSDEN:8 m above ground:*hour fcst:aerosol=Particulate organic matter dry:)"
    "|(:(MSLMA):)"
    "|(:REFD:1000 m above ground:)"
    "|(:LFTX:500-1000 mb:)"
)

# Smoke (MASSDEN) is in kg/m^3; scale to ug/m^3 so float32 storage keeps precision
MASSDEN_KG_TO_UG = 1e9

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so zarr_vars[i] is column i + 1 of the
# RRFS index map, shared with the hourly RRFS store so the API can concatenate
# the two.
zarr_vars = (
    "VIS_surface",
    "GUST_surface",
    "MSLMA_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "TCDC_entireatmosphere",
    "MASSDEN_8maboveground",
    "REFD_1000maboveground",
    "DSWRF_surface",
    "CAPE_surface",
    "CIN_surface",
    # wgrib2 -netcdf writes the "500-1000 mb" layer as 500M1000mb
    "LFTX_500M1000mb",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/RRFS_6H"
)
forecast_process_path = forecast_process_dir + "/RRFS_6H_Process"
tmp_dir = forecast_process_dir + "/Downloads"

# Must not sit inside forecast_process_dir, which is deleted at the start of a run
forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RRFS_6H")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

RRFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RRFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_rrfs_6h_subset(base_time: pd.Timestamp) -> list[str]:
    """Download and validate the 18-83 h RRFS 2dfld subsets with bounded retries."""
    return download_and_validate_subset(
        model="rrfs",
        product="2dfld",
        search=RRFS_MATCH_STRINGS,
        dataset_name="RRFS 6-hourly",
        base_time=base_time,
        forecast_hours=list(RRFS_FORECAST_HOURS),
        priority=RRFS_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=RRFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=RRFS_RETRY_SLEEP_SECONDS,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


def main() -> None:
    """Run the 6-hourly RRFS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["RRFS_6H"]
    final_chunk = FINAL_CHUNK_SIZES["RRFS_6H"]

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

    # %% Determine the model run: the newest 6-hourly cycle whose last lead exists
    latest_run = Herbie_latest(
        model="rrfs",
        n=3,
        freq="6h",
        fxx=[RRFS_FORECAST_HOURS[-1]],
        product="2dfld",
        verbose=False,
        priority=RRFS_SOURCE_PRIORITY,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "RRFS_6H.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to RRFS_6H, ending")
        return
    logger.info("Processing RRFS_6H run %s", base_time)

    # %% Forecast hours: hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(RRFS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download, merge, rotate winds and convert to NetCDF
    grib_list = download_rrfs_6h_subset(base_time)

    merged_grib_path = forecast_process_path + "_wgrib2_merged.grib2"
    regrid_path = forecast_process_path + "_wgrib2_merged.regrid"
    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        "cat " + " ".join(grib_list) + f" | {wgrib2_path} - -grib {merged_grib_path}",
        "RRFS 6-hourly wgrib2 GRIB merge",
    )
    run_checked(
        f"{wgrib2_path} {merged_grib_path} -new_grid_winds earth "
        f"-new_grid {RRFS_LAMBERT_GRID} {regrid_path}",
        "RRFS 6-hourly wgrib2 wind rotation",
    )
    run_checked(
        f"{wgrib2_path} {regrid_path} -netcdf {merged_nc_path}",
        "RRFS 6-hourly wgrib2 NetCDF conversion",
    )

    xarray_forecast_merged = xr.open_mfdataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"RRFS 6-hourly run {base_time} times "
            f"{xarray_forecast_merged['time'].values} do not match the expected "
            f"forecast leads {RRFS_FORECAST_HOURS}"
        )

    # %% Correct RRFS-specific quirks
    # APCP is downloaded as the one-hour bucket, so it needs no differencing.
    xarray_forecast_merged["MASSDEN_8maboveground"] = (
        xarray_forecast_merged["MASSDEN_8maboveground"] * MASSDEN_KG_TO_UG
    )

    # Set REFD values below the reflectivity threshold to 0
    xarray_forecast_merged["REFD_1000maboveground"] = mask_invalid_refc(
        xarray_forecast_merged["REFD_1000maboveground"]
    )

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(RRFS_FORECAST_HOURS),
            "x": process_chunk,
            "y": process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_xr_merged.zarr",
            mode="w",
            consolidated=False,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
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
        daskVarArrayList.append(
            daskForecastArray.rechunk(
                (len(forecast_valid_times), process_chunk, process_chunk)
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

    # %% Write the final store (hourly already, so no time interpolation)
    daskVarArrayStackDisk_main = pad_to_chunk_size(daskVarArrayStackDisk, final_chunk)

    rrfs_6h_store_name = "RRFS_6H.zarr"
    write_final_zarr(
        daskVarArrayStackDisk_main,
        forecast_process_dir + "/" + rrfs_6h_store_name,
        final_chunk,
        FINAL_SHARD_SIZES["RRFS_6H"],
        zarr_store_workers,
        zarr_vars,
        times=forecast_valid_times.astype("datetime64[s]").astype("int64"),
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [rrfs_6h_store_name],
        base_time,
        "RRFS_6H.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("RRFS_6H run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
