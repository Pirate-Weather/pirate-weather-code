# %% Script to test FastHerbie.py to download HRDPS data
# Alexander Rey, April 2026

# %% Import modules
import logging
import os
import sys
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from dotenv import find_dotenv, load_dotenv

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import (
    cat_gribs,
    download_and_validate_subset,
    quote_path,
    run_checked,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    clear_directory,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_energy_to_flux,
    interp_time_map_blocks_nan,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

from herbie import HerbieLatest  # noqa: E402

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/HRDPS")
forecast_process_path = forecast_process_dir + "/HRDPS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/HRDPS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
herbie_download_retries = positive_int_env("herbie_download_retries", 5)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)
skip_hrdps_wgrib2_validation = os.getenv(
    "skip_hrdps_wgrib2_validation", "true"
).lower() in {"1", "true", "yes", "on"}


# Define the processing chunk size
process_chunk = CHUNK_SIZES["HRDPS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["HRDPS"]


def main() -> None:
    """Run the HRDPS forecast ingest."""
    configure_herbie_request_timeouts()
    logging.basicConfig(level=logging.INFO)
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    latest_run = HerbieLatest(
        model="hrdps",
        priority=["msc"],
        periods=7,
        fxx=48,
        product="continental",
        verbose=False,
        variable="TMP",
        level="AGL-2m",
        save_dir=herbie_save_dir,
    )

    base_time = latest_run.date

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/HRDPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to HRDPS, ending")
        return

    zarr_vars = (
        "time",
        "GUST_10maboveground",
        "PRMSL_meansealevel",
        "TMP_2maboveground",
        "DPT_2maboveground",
        "RH_2maboveground",
        "WIND_10maboveground",
        "WDIR_10maboveground",
        "PRATE_surface",
        "APCP_surface",
        "PTYPE_surface",
        "TCDC_surface",
        "UVI_surface",
        "DSWRF_surface",
        "CAPE_surface",
        "PRES_surface",
        "LFTX_500mb",
        "VVEL_500mb",
    )

    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 48 hours

    # Define the variables to download as a dictionary of variable and level pairs to match in the grib files
    match_strings = [
        {"variable": "TMP", "level": "AGL-2m"},
        {"variable": "DPT", "level": "AGL-2m"},
        {"variable": "RH", "level": "AGL-2m"},
        {"variable": "WIND", "level": "AGL-10m"},
        {"variable": "WDIR", "level": "AGL-10m"},
        {"variable": "GUST", "level": "AGL-10m"},
        {"variable": "APCP", "level": "Sfc"},
        {"variable": "PTYPE", "level": "Sfc"},
        {"variable": "PRATE", "level": "Sfc"},
        {"variable": "UVIndex", "level": "Sfc"},
        {"variable": "DSWRF", "level": "Sfc"},
        {"variable": "CAPE", "level": "Sfc"},
        {"variable": "PRES", "level": "Sfc"},
        {"variable": "TCDC", "level": "Sfc"},
        {"variable": "PRMSL", "level": "MSL"},
        {"variable": "LFTX", "level": "ISBL_0500"},
        {"variable": "VVEL", "level": "ISBL_0500"},
    ]

    hrdps_file_range = FORECAST_LEAD_RANGES["HRDPS"]

    # MSC models have each variable in a separate file, so we loop through the variables and levels to download each one and then merge them later
    all_files = []
    expected_total = 0
    for g in match_strings:
        grib_files = download_and_validate_subset(
            model="hrdps",
            product="continental",
            search=None,
            dataset_name=f"HRDPS forecast {g['variable']}:{g['level']}",
            base_time=base_time,
            wgrib2_exe=wgrib2_path.strip(),
            forecast_hours=hrdps_file_range,
            expected_forecast_hours=hrdps_file_range,
            skip_wgrib2_validation=skip_hrdps_wgrib2_validation,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
            herbie_kwargs={
                "variable": g["variable"],
                "level": g["level"],
                "verbose": True,
            },
        )

        all_files += grib_files
        expected_total += len(hrdps_file_range)

        logger.info(
            "Download completed for HRDPS forecast %s:%s, %s files downloaded.",
            g["variable"],
            g["level"],
            len(grib_files),
        )

    # Deduplicate and sanity-check total files
    all_files = sorted(set(all_files))
    if len(all_files) < expected_total:
        logger.error(
            f"Download incomplete, expected at least {expected_total} files but got {len(all_files)}"
        )
        sys.exit(1)

    # Create ordered list of downloaded GRIB files from collected paths.
    grib_list = all_files

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-netcdf {quote_path(forecast_process_path + '_wgrib2_merged.nc')}"
    )

    # Run wgrib2
    run_checked(cmd, "HRDPS wgrib2 NetCDF conversion")

    # Read the merged netcdf file using xarray (single combined file)
    xarray_forecast_merged = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged.nc"
    )

    if len(xarray_forecast_merged.time) != len(hrdps_file_range):
        raise ValueError("Incorrect number of timesteps! Exiting")

    # HRDPS latitude and longitude are 2-D coordinates over the y/x grid.
    # Reading either coordinate's size would return the product of both axes.
    spatial_dims = xarray_forecast_merged["TMP_2maboveground"].dims[-2:]
    NY, NX = (xarray_forecast_merged.sizes[dim] for dim in spatial_dims)

    # Create a new hourly time series from the first to the last forecast valid time
    start = xarray_forecast_merged.time.min().values
    end = xarray_forecast_merged.time.max().values
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    stacked_times = xarray_forecast_merged.time.values
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
    APCP_surface_tmp = da.diff(
        xarray_forecast_merged["APCP_surface"].data,
        axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
        prepend=0,
    )

    xarray_forecast_merged["APCP_surface"].data = APCP_surface_tmp

    # DSWRF is total accumulated energy in J/m^2. Convert each interval to its
    # average flux in W/m^2 before the global validity mask is applied.
    xarray_forecast_merged["DSWRF_surface"].data = deaccumulate_energy_to_flux(
        xarray_forecast_merged["DSWRF_surface"].data,
        xarray_forecast_merged.time.values,
        time_axis=xarray_forecast_merged["DSWRF_surface"].get_axis_num("time"),
    )

    # Save the dataset with compression and filters for all variables
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": 48, "x": process_chunk, "y": process_chunk}
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
    del (APCP_surface_tmp, xarray_forecast_merged)
    T1 = time.time()

    logger.info(T1 - T0)
    os.remove(forecast_process_path + "_wgrib2_merged.nc")

    # %% Stack the forecast variables into a single 4D array using dask
    daskVarArrayList = []

    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            # Broadcast the 1D time axis over the grid so it is stored like a variable
            daskArrayOut = da.broadcast_to(
                daskForecastArray.astype("float32")[:, None, None],
                (len(stacked_timesUnix), NY, NX),
            ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

            daskVarArrayList.append(daskArrayOut)

        else:
            daskVarArrayList.append(
                daskForecastArray[:, :, :]
                .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
                .astype("float32")
            )

        logger.info(dask_var)

    # Merge the arrays into a single 4D array
    daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

    # Mask out invalid data
    # Ignore storm distance, since it can reach very high values that are still correct
    daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

    # Write out to disk
    # This intermediate step is necessary to avoid memory overflow
    # with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    hrdps_store_name = "HRDPS.zarr.zip" if save_type == "S3" else "HRDPS.zarr"

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            # 1. Interpolate the stacked array to be hourly along the time axis
            daskVarArrayStackDiskInterp = interp_time_map_blocks_nan(
                daskVarArrayStackDisk,
                stacked_timesUnix=stacked_timesUnix,
                hourly_timesUnix=hourly_timesUnix,
                dtype="float32",
                fill_value=np.nan,
                nearest_vars=[zarr_vars.index("PTYPE_surface")],
            )

        # 2. Pad to chunk size
        daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
            daskVarArrayStackDiskInterp, final_chunk
        )

        # 3. Write the final store in source-aligned tiles
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + hrdps_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [hrdps_store_name],
        base_time,
        "HRDPS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Timing
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
