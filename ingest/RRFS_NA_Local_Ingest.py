"""RRFS North American hourly (0-18 h) ingest.

Downloads the 2D fields (2dfld product) of the newest hourly RRFS cycle on the
North American domain (3 km rotated latitude-longitude grid; hourly cycles,
hourly leads 1-18 h), de-accumulates precipitation and publishes
``RRFS_NA.zarr`` and ``RRFS_NA.time.pickle``. The store starts at the first
forecast lead (run + 1 h); hours 18-84 come from the 6-hourly RRFS_NA ingest.

The winds are not yet rotated to earth-relative (see the wind rotation step),
so this ingest stops before publishing.

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
RRFS_NA_FORECAST_HOURS = FORECAST_LEAD_RANGES["RRFS_NA_1H"]
RRFS_SOURCE_PRIORITY = ["aws", "nomads"]
# Herbie's RRFS template selects the North American domain with domain="na"
RRFS_NA_DOMAIN = "na"

# Herbie match strings for the 2dfld product
RRFS_MATCH_STRINGS = (
    ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    "|:((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST|DSWRF|CAPE|CIN):surface:.*hour fcst)"
    "|(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    "|(:TCDC:entire atmosphere:.*hour fcst)"
    # Each file holds a 0-N h run total; it is differenced to hourly below.
    "|(:APCP:surface:0-[1-9]*)"
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
# index map; the order matches the RRFS CONUS store.
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
    "forecast_process_dir", default="/mnt/nvme/data/RRFS_NA"
)
forecast_process_path = forecast_process_dir + "/RRFS_NA_Process"
tmp_dir = forecast_process_dir + "/Downloads"

# Must not sit inside forecast_process_dir, which is deleted at the start of a run
forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RRFS_NA")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

RRFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RRFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_rrfs_na_subset(base_time: pd.Timestamp) -> list[str]:
    """Download and validate the hourly RRFS NA 2dfld subsets with bounded retries."""
    return download_and_validate_subset(
        model="rrfs",
        product="2dfld",
        search=RRFS_MATCH_STRINGS,
        dataset_name="RRFS_NA hourly",
        base_time=base_time,
        forecast_hours=list(RRFS_NA_FORECAST_HOURS),
        priority=RRFS_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=RRFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=RRFS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"domain": RRFS_NA_DOMAIN},
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


def main() -> None:
    """Run the hourly RRFS NA ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["RRFS_NA"]
    final_chunk = FINAL_CHUNK_SIZES["RRFS_NA"]

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

    # %% Determine the model run: the newest cycle whose 18 h lead exists
    latest_run = Herbie_latest(
        model="rrfs",
        n=6,
        freq="1h",
        fxx=[RRFS_NA_FORECAST_HOURS[-1]],
        product="2dfld",
        domain=RRFS_NA_DOMAIN,
        verbose=False,
        priority=RRFS_SOURCE_PRIORITY,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "RRFS_NA.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to RRFS_NA, ending")
        return
    logger.info("Processing RRFS_NA run %s", base_time)

    # %% Forecast hours: hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(RRFS_NA_FORECAST_HOURS, unit="h")
    ).values

    # %% Download, merge, rotate winds and convert to NetCDF
    grib_list = download_rrfs_na_subset(base_time)

    merged_grib_path = forecast_process_path + "_wgrib2_merged.grib2"
    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        "cat " + " ".join(grib_list) + f" | {wgrib2_path} - -grib {merged_grib_path}",
        "RRFS_NA hourly wgrib2 GRIB merge",
    )

    # The NA domain is a rotated latitude-longitude grid, so its U/V are relative
    # to the rotated grid. The CONUS approach (-new_grid_winds earth with the
    # native grid passed to -new_grid) needs a wgrib2 grid definition for this
    # rotated grid, which has not been worked out and verified yet. Publishing
    # grid-relative winds would serve wrong wind directions, so stop here.
    raise NotImplementedError(
        f"RRFS_NA run {base_time}: earth-relative wind rotation for the rotated "
        "lat-lon NA grid is not implemented; not publishing grid-relative winds"
    )

    run_checked(
        f"{wgrib2_path} {merged_grib_path} -netcdf {merged_nc_path}",
        "RRFS_NA hourly wgrib2 NetCDF conversion",
    )

    xarray_forecast_merged = xr.open_mfdataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"RRFS_NA hourly run {base_time} times "
            f"{xarray_forecast_merged['time'].values} do not match the expected "
            f"forecast leads {RRFS_NA_FORECAST_HOURS}"
        )

    # %% Correct RRFS-specific quirks
    # APCP is a 0-N h run total; difference it to hourly totals. The first lead
    # is the 0-1 h total, so prepending zero leaves it unchanged.
    xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged[
        "APCP_surface"
    ].copy(
        data=np.diff(
            xarray_forecast_merged["APCP_surface"],
            axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
            prepend=0,
        )
    )

    xarray_forecast_merged["MASSDEN_8maboveground"] = (
        xarray_forecast_merged["MASSDEN_8maboveground"] * MASSDEN_KG_TO_UG
    )

    # Set REFD values below the reflectivity threshold to 0
    xarray_forecast_merged["REFD_1000maboveground"] = mask_invalid_refc(
        xarray_forecast_merged["REFD_1000maboveground"]
    )

    # wgrib2 names the rotated grid's dimensions rlat/rlon rather than y/x
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(RRFS_NA_FORECAST_HOURS),
            "rlon": process_chunk,
            "rlat": process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "merged_zarr.zarr",
            mode="w",
            consolidated=False,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    del xarray_forecast_merged

    # The wgrib2 intermediates are large and no longer needed
    os.remove(merged_grib_path)
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "merged_zarr.zarr",
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

    rrfs_na_store_name = "RRFS_NA.zarr"
    write_final_zarr(
        daskVarArrayStackDisk_main,
        forecast_process_dir + "/" + rrfs_na_store_name,
        final_chunk,
        FINAL_SHARD_SIZES["RRFS_NA"],
        zarr_store_workers,
        zarr_vars,
        times=forecast_valid_times.astype("datetime64[s]").astype("int64"),
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [rrfs_na_store_name],
        base_time,
        "RRFS_NA.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("RRFS_NA run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
