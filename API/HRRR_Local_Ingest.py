"""HRRR hourly (0-18 h) ingest.

Downloads the surface fields of the newest hourly HRRR cycle (3 km CONUS Lambert
conformal grid, 1799 x 1059, hourly cycles, hourly leads 1-18 h), rotates the
winds to earth-relative, de-accumulates precipitation and publishes ``HRRR.zarr``,
the ``HRRR_maps.zarr`` map tiles and ``HRRR.time.pickle``. The store starts at the
first forecast lead (run + 1 h); hours 18-48 come from the 6-hourly HRRR ingest.

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
import s3fs
import xarray as xr
import zarr
from herbie.fast import Herbie_latest

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_subset, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    close_store,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    mask_invalid_data,
    mask_invalid_refc,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
HRRR_FORECAST_HOURS = FORECAST_LEAD_RANGES["HRRR_1H"]
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

# Herbie match strings for the surface (sfc) product
HRRR_MATCH_STRINGS = (
    ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    "|:((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST|DSWRF|CAPE):surface:.*hour fcst)"
    "|(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    "|(:TCDC:entire atmosphere:.*hour fcst)"
    # Each sfc file holds a 0-N h run total and an (N-1)-N h bucket; this
    # selects the run total, which is differenced to hourly below.
    "|(:APCP:surface:0-[1-9]*)"
    "|:(MASSDEN:8 m above ground:)"
    "|(:(MSLMA):)"
    "|(:REFD:1000 m above ground:)"
)

# Smoke (MASSDEN) is in kg/m^3; scale to ug/m^3 so float32 storage keeps precision
MASSDEN_KG_TO_UG = 1e9

# Map tiles: x/y chunk size and the variables served to the map layer
MAP_CHUNK = 100
HRRR_MAP_VARS = (
    "time",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "MASSDEN_8maboveground",
    "REFD_1000maboveground",
)

# Variables in the published store; the order is the API's HRRR index map
zarr_vars = (
    "time",
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
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/HRRR")
forecast_process_path = forecast_process_dir + "/HRRR_Process"
tmp_dir = forecast_process_dir + "/Downloads"

# Must not sit inside forecast_process_dir, which is deleted at the start of a run
forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/HRRR")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

HRRR_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
HRRR_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_hrrr_subset(base_time: pd.Timestamp) -> list[str]:
    """Download and validate the hourly HRRR sfc subsets with bounded retries."""
    return download_and_validate_subset(
        model="hrrr",
        product="sfc",
        search=HRRR_MATCH_STRINGS,
        dataset_name="HRRR hourly",
        base_time=base_time,
        forecast_hours=list(HRRR_FORECAST_HOURS),
        priority=HRRR_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=HRRR_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=HRRR_RETRY_SLEEP_SECONDS,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


def main() -> None:
    """Run the hourly HRRR ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["HRRR"]
    final_chunk = FINAL_CHUNK_SIZES["HRRR"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 18 h lead exists
    latest_run = Herbie_latest(
        model="hrrr",
        n=6,
        freq="1h",
        fxx=[HRRR_FORECAST_HOURS[-1]],
        product="sfc",
        verbose=False,
        priority=HRRR_SOURCE_PRIORITY,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/HRRR.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to HRRR, ending")
        return
    logger.info("Processing HRRR run %s", base_time)

    # %% Forecast hours: hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(HRRR_FORECAST_HOURS, unit="h")
    ).values

    # %% Download, merge, rotate winds and convert to NetCDF
    grib_list = download_hrrr_subset(base_time)

    merged_grib_path = forecast_process_path + "_wgrib2_merged.grib2"
    regrid_path = forecast_process_path + "_wgrib2_merged.regrid"
    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        "cat " + " ".join(grib_list) + f" | {wgrib2_path} - -grib {merged_grib_path}",
        "HRRR hourly wgrib2 GRIB merge",
    )
    run_checked(
        f"{wgrib2_path} {merged_grib_path} -new_grid_winds earth "
        f"-new_grid {HRRR_LAMBERT_GRID} {regrid_path}",
        "HRRR hourly wgrib2 wind rotation",
    )
    run_checked(
        f"{wgrib2_path} {regrid_path} -netcdf {merged_nc_path}",
        "HRRR hourly wgrib2 NetCDF conversion",
    )

    xarray_forecast_merged = xr.open_mfdataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"HRRR hourly run {base_time} times "
            f"{xarray_forecast_merged['time'].values} do not match the expected "
            f"forecast leads {HRRR_FORECAST_HOURS}"
        )

    # %% Correct HRRR-specific quirks
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

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(HRRR_FORECAST_HOURS),
            "x": process_chunk,
            "y": process_chunk,
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
    os.remove(regrid_path)
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "merged_zarr.zarr",
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

    # %% Write the final store (hourly already, so no time interpolation)
    daskVarArrayStackDisk_main = pad_to_chunk_size(daskVarArrayStackDisk, final_chunk)

    hrrr_store_name = "HRRR.zarr.zip" if save_type == "S3" else "HRRR.zarr"
    write_final_zarr(
        daskVarArrayStackDisk_main,
        forecast_process_dir + "/" + hrrr_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Maps: every forecast step of the map variables in MAP_CHUNK x/y tiles
    daskVarArrayStackDisk_maps = pad_to_chunk_size(daskVarArrayStackDisk, MAP_CHUNK)
    map_steps = daskVarArrayStackDisk_maps.shape[1]

    hrrr_maps_store_name = (
        "HRRR_maps.zarr.zip" if save_type == "S3" else "HRRR_maps.zarr"
    )
    if save_type == "S3":
        zarr_store_maps = zarr.storage.ZipStore(
            forecast_process_dir + "/" + hrrr_maps_store_name, mode="a"
        )
    else:
        zarr_store_maps = zarr.storage.LocalStore(
            forecast_process_dir + "/" + hrrr_maps_store_name
        )

    for map_var in HRRR_MAP_VARS:
        zarr_array = zarr.create_array(
            store=zarr_store_maps,
            name=map_var,
            shape=(
                map_steps,
                daskVarArrayStackDisk_maps.shape[2],
                daskVarArrayStackDisk_maps.shape[3],
            ),
            chunks=(map_steps, MAP_CHUNK, MAP_CHUNK),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            da.rechunk(
                daskVarArrayStackDisk_maps[zarr_vars.index(map_var)],
                (map_steps, MAP_CHUNK, MAP_CHUNK),
            ).to_zarr(zarr_array, overwrite=True, compute=True)

    close_store(zarr_store_maps)

    # %% Publish the stores, then the run time the API uses to detect them
    publish_ingest_run(
        [hrrr_store_name, hrrr_maps_store_name],
        base_time,
        "HRRR.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("HRRR run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
