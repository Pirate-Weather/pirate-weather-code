# %% Script to test FastHerbie.py to download GFS data
# Alexander Rey, September 2023

# %% Import modules
import logging
import os
import shutil
import sys
import time
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from herbie import FastHerbie, HerbieLatest, Path

from API.constants.shared_const import INGEST_VERSION_STR
from API.herbie_custom_templates import register_aws_aigfs_aigefs_templates
from API.ingest_grib_utils import check_grib_stats, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
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


def main() -> None:
    """Run the ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    # The final write used dask's default thread pool (one thread per CPU).
    zarr_store_workers = configure_zarr_writes(default_workers=os.cpu_count() or 2)

    # Define the processing chunk size
    process_chunk = CHUNK_SIZES["GFS"]

    # Define the final x/y chunksize
    final_chunk = FINAL_CHUNK_SIZES["GFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    register_aws_aigfs_aigefs_templates()

    latest_run = HerbieLatest(
        model="aigfs",
        n=3,
        freq="6h",
        fxx=240,
        product="sfc",
        verbose=False,
        priority=["aws", "nomads"],
        save_dir=tmp_dir,
    )

    base_time = latest_run.date

    logger.info(base_time)

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/AIGFS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to AIGFS, ending")
        return

    zarr_vars = (
        "time",
        "PRMSL_meansealevel",
        "TMP_2maboveground",
        "UGRD_10maboveground",
        "VGRD_10maboveground",
        "APCP_surface",
    )

    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 240 hours

    # Define the subset of variables to download as a list of strings
    matchstring_2m = ":((TMP):2 m above ground:)"
    matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    matchstring_ap = r":APCP:surface:0-\d+\s+hour acc fcst:"
    matchstring_sl = "(:(PRMSL):)"

    # Merge matchstrings for download
    match_strings = (
        matchstring_2m
        + "|"
        + matchstring_10m
        + "|"
        + matchstring_ap
        + "|"
        + matchstring_sl
    )

    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach

    aigfs_range = FORECAST_LEAD_RANGES["AIGFS"]

    # Create FastHerbie object
    FH_forecastsub = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="6h"),
        model="aigfs",
        fxx=aigfs_range,
        product="sfc",
        verbose=False,
        priority=["aws", "nomads"],
        save_dir=tmp_dir,
    )

    # Download the subsets
    FH_forecastsub.download(match_strings, verbose=False)

    # Check for download length
    if len(FH_forecastsub.file_exists) != len(aigfs_range):
        logger.error(
            "Download failed, expected %d files but got %d",
            len(aigfs_range),
            len(FH_forecastsub.file_exists),
        )
        sys.exit(1)

    # Create list of downloaded grib files
    grib_list = [
        str(Path(x.get_localFilePath(match_strings)).expand())
        for x in FH_forecastsub.file_exists
    ]

    # Perform a check if any data seems to be invalid
    check_grib_stats(grib_list, wgrib2_path, "AIGFS forecast")

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + forecast_process_path
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2
    run_checked(cmd, "AIGFS wgrib2 NetCDF conversion")

    # Read the netcdf file using xarray (no UV product for Graphcast)
    xarray_wgrib_merged = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")

    # Use the main merged dataset as the forecast merged dataset
    xarray_forecast_merged = xarray_wgrib_merged

    assert len(xarray_forecast_merged.time) == len(aigfs_range), (
        "Incorrect number of timesteps! Exiting"
    )

    # Hourly target axis runs from the first stored forecast valid time
    # (base_time + 6 h) to the last one. AIGFS output starts at hour 6, so
    # hours base_time+1 to +5 are not in this store; the API falls back to
    # other sources for them.
    start = xarray_forecast_merged.time.min().values
    end = xarray_forecast_merged.time.max().values
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    stacked_times = xarray_forecast_merged.time.values
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # %% FIX THINGS

    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
    APCP_surface_tmp = da.diff(
        xarray_forecast_merged["APCP_surface"],
        axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
        prepend=0,
    )

    # Convert 6-hourly to 1-hourly accumulation for precipitation only
    # AIGFS provides all data at 6-hourly intervals (0, 6, 12, 18, ... 240 hours)
    # After diff, we have 6-hour accumulations, so divide by 6 to get hourly rates
    # This only applies to APCP_surface, not other variables
    APCP_surface_tmp = APCP_surface_tmp / 6

    xarray_forecast_merged["APCP_surface"].data = APCP_surface_tmp

    # %% Save merged and processed xarray dataset to disk using zarr with compression

    # Save the dataset with compression and filters for all variables
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": -1, "latitude": process_chunk, "longitude": process_chunk}
    )
    xarray_forecast_merged.to_zarr(
        forecast_process_path + "_forecast.zarr",
        mode="w",
        consolidated=False,
        compute=True,
    )

    # %% Delete to free memory (keep only existing variables)
    del (
        xarray_wgrib_merged,
        xarray_forecast_merged,
        APCP_surface_tmp,
    )
    T1 = time.time()

    logger.info(T1 - T0)
    os.remove(forecast_process_path + "_wgrib2_merged.nc")

    # %% Stack the forecast variables and then squash using dask
    daskVarArrayList = []

    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_forecast.zarr",
            component=dask_var,
            inline_array=True,
        )

        if dask_var == "time":
            # Get times as numpy
            npCatTimes = daskForecastArray.astype("float32").compute()
            validate_stacked_time_alignment(stacked_timesUnix, npCatTimes)

            daskArrayOut = da.from_array(
                np.tile(
                    np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                    (1, 721, 1440),
                )
            ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

            daskVarArrayList.append(daskArrayOut)

        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
                    (len(stacked_timesUnix), process_chunk, process_chunk)
                ).astype("float32")
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
    daskVarArrayListMergeNaN.to_zarr(
        forecast_process_path + "_stack.zarr", overwrite=True, compute=True
    )

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    aigfs_store_name = "AIGFS.zarr.zip" if save_type == "S3" else "AIGFS.zarr"

    with ProgressBar():
        # 1. Interpolate the stacked array to be hourly along the time axis
        daskVarArrayStackDiskInterp = interp_time_take_blend(
            daskVarArrayStackDisk,
            stacked_timesUnix=stacked_timesUnix,
            hourly_timesUnix=hourly_timesUnix,
            dtype="float32",
            fill_value=np.nan,
        )

        # 2. Pad to chunk size
        daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
            daskVarArrayStackDiskInterp, final_chunk
        )

        # 3. Write the final store in source-aligned tiles
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + aigfs_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # Maps generation removed for Graphcast ingest (not required)

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [aigfs_store_name],
        base_time,
        "AIGFS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)

    # Timing
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
