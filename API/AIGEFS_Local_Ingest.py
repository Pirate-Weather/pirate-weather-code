# %% Script to test FastHerbie.py to download GFS data
# Alexander Rey, September 2023

# %% Import modules
import logging
import os
import shutil
import subprocess
import time
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest

from API.constants.shared_const import INGEST_VERSION_STR
from API.herbie_custom_templates import register_aws_aigfs_aigefs_templates
from API.ingest_grib_utils import download_and_validate_subset, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# The AWS EAGLE ensemble is experimental; operational AIGEFS is on NOMADS.
AIGEFS_SOURCE_PRIORITY = ["nomads"]
AIGEFS_MAX_THREADS = positive_int_env("aigefs_download_threads", 4)
AIGEFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
AIGEFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_aigefs_member(
    run_date: pd.Timestamp,
    forecast_hours: list[int],
    member: int,
    dataset_name: str,
) -> list[str]:
    """Download and validate a complete member with bounded retries and threads."""
    return download_and_validate_subset(
        model="aigefs",
        product="sfc",
        search=None,
        dataset_name=dataset_name,
        base_time=run_date,
        forecast_hours=forecast_hours,
        priority=AIGEFS_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=AIGEFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=AIGEFS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"member": f"mem{member:03d}", "max_threads": AIGEFS_MAX_THREADS},
        download_max_threads=AIGEFS_MAX_THREADS,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/AIGEFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "AIGEFS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/AIGEFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


def main() -> None:
    """Run the ingest."""
    logging.basicConfig(level=logging.INFO)
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    # Define the processing chunk size
    process_chunk = CHUNK_SIZES["GEFS"]

    # Define the final x/y chunksize
    final_chunk = FINAL_CHUNK_SIZES["GEFS"]

    AIGEFS_MEMBER_COUNT = 30

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    register_aws_aigfs_aigefs_templates()
    configure_herbie_request_timeouts()

    latest_run = HerbieLatest(
        model="aigefs",
        n=3,
        freq="6h",
        fxx=240,
        product="sfc",
        verbose=False,
        member="mem000",
        priority=AIGEFS_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )

    base_time = latest_run.date

    logger.info(base_time)

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/AIGEFS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to AIGEFS, ending")
        return

    zarr_vars = (
        "time",
        "APCP_surface",
    )

    probVars = (
        "time",
        "Precipitation_Prob",
        "APCP_Mean",
        "APCP_StdDev",
    )

    #####################################################################################################
    # %% Download forecast data for mean and spread products

    # Create a range of forecast lead times
    aigefs_range = FORECAST_LEAD_RANGES["AIGEFS"]

    for mem in range(1, AIGEFS_MEMBER_COUNT + 1):
        grib_list = download_aigefs_member(
            base_time, list(aigefs_range), mem, f"AIGEFS member {mem} forecast"
        )

        # Create a string to pass to wgrib2 to merge all gribs into one grib
        cmd = (
            "cat "
            + " ".join(grib_list)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + '-match ":APCP:surface:" '
            + "-netcdf "
            + forecast_process_path
            + "_wgrib2_merged_m"
            + str(mem)
            + ".nc"
        )

        run_checked(cmd, f"AIGEFS member {mem} forecast GRIB conversion")

        # Fix precip and chunk each member
        xarray_wgrib = xr.open_dataset(
            forecast_process_path + "_wgrib2_merged_m" + str(mem) + ".nc"
        )

        # AIGEFS does not use accumulated precipitation, just 6 hour steps
        # Use the difference between 3 and 6 for every other timestep
        # Divide by 6 to get hourly accum
        xarray_wgrib["APCP_surface"] = xarray_wgrib["APCP_surface"] / 6

        # Keep only APCP_surface, dropping all other variables to save space and speed up processing
        # xarray_wgrib = xarray_wgrib[["time", "APCP_surface"]]

        if xarray_wgrib.sizes["time"] != len(aigefs_range):
            raise RuntimeError(
                f"AIGEFS member {mem} has {xarray_wgrib.sizes['time']} forecast timesteps; "
                f"expected {len(aigefs_range)}"
            )

        xarray_wgrib = xarray_wgrib.chunk(
            chunks={
                "time": len(aigefs_range),
                "latitude": process_chunk,
                "longitude": process_chunk,
            }
        )

        xarray_wgrib.to_zarr(
            forecast_process_path + "_xr_m" + str(mem) + ".zarr",
            consolidated=False,
            mode="w",
        )

        # Delete the wgrib netcdf to save space
        subprocess.run(
            "rm " + forecast_process_path + "_wgrib2_merged_m" + str(mem) + ".nc",
            shell=True,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )

    # Hourly target axis runs from the first stored forecast valid time
    # (base_time + 6 h) to the last one. AIGEFS output starts at hour 6, so
    # hours base_time+1 to +5 are not in this store; the API falls back to
    # other sources for them.
    start = xarray_wgrib.time.min().values
    end = xarray_wgrib.time.max().values
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    stacked_times = xarray_wgrib.time.values

    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    ncLocalWorking_paths = [
        forecast_process_path + "_xr_m" + str(i) + ".zarr"
        for i in range(1, AIGEFS_MEMBER_COUNT + 1)
    ]

    # Dask
    daskArrays = {}

    # Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
    # Note that the chunks
    for dask_var in zarr_vars:
        daskVarArrays = []
        for local_ncpath in ncLocalWorking_paths:
            daskVarArrays.append(da.from_zarr(local_ncpath, dask_var))

        # Stack times together, keeping variables separate
        daskArrays[dask_var] = da.stack(daskVarArrays, axis=0)

        daskVarArrays = []

    # Dict to hold output dask arrays
    daskOutput = {}

    # Find the probability of precipitation greater than 0.1 mm/h  across all members
    daskOutput["Precipitation_Prob"] = ((daskArrays["APCP_surface"]) > 0.1).sum(
        axis=0
    ) / AIGEFS_MEMBER_COUNT

    # Find the standard deviation of precipitation accumulation across all members
    daskOutput["APCP_StdDev"] = daskArrays["APCP_surface"].std(axis=0)

    # Find the average precipitation accumulation across all members
    daskOutput["APCP_Mean"] = daskArrays["APCP_surface"].mean(axis=0)

    # Copy time over
    daskOutput["time"] = daskArrays["time"][1, :]

    for dask_var in probVars:
        # with ProgressBar():
        if dask_var == "time":
            daskOutput[dask_var].to_zarr(
                forecast_process_path + "_" + dask_var + ".zarr",
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                overwrite=True,
                compute=True,
            )
        else:
            daskOutput[dask_var].rechunk(
                (len(aigefs_range), process_chunk, process_chunk)
            ).to_zarr(
                forecast_process_path + "_" + dask_var + ".zarr",
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                dtype="float32",
                overwrite=True,
                compute=True,
            )

    # %% Delete to free memory
    del xarray_wgrib, daskOutput, daskArrays

    T1 = time.time()

    logger.info(T1 - T0)
    if os.path.exists(forecast_process_path + "_wgrib2_merged.nc"):
        os.remove(forecast_process_path + "_wgrib2_merged.nc")

    # %% Stack the forecast variables and then squash using dask
    daskVarArrayList = []

    for dask_var in probVars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_" + dask_var + ".zarr", inline_array=True
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

    # Create a zarr backed dask array
    aigefs_store_name = "AIGEFS.zarr.zip" if save_type == "S3" else "AIGEFS.zarr"
    if save_type == "S3":
        zarr_store = zarr.storage.ZipStore(
            forecast_process_dir + "/" + aigefs_store_name, mode="a", compression=0
        )
    else:
        zarr_store = zarr.storage.LocalStore(
            forecast_process_dir + "/" + aigefs_store_name
        )

    #
    # 1. Interpolate the stacked array to be hourly along the time axis
    # 2. Pad to chunk size
    # 3. Create the zarr array
    # 4. Rechunk it to match the final array
    # 5. Write it out to the zarr array

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

        # 3. Create the zarr array
        zarr_array = zarr.create_array(
            store=zarr_store,
            shape=(
                len(probVars),
                len(hourly_timesUnix),
                daskVarArrayStackDiskInterpPad.shape[2],
                daskVarArrayStackDiskInterpPad.shape[3],
            ),
            chunks=(len(probVars), len(hourly_timesUnix), final_chunk, final_chunk),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )

        # 4. Rechunk it to match the final array
        # 5. Write it out to the zarr array
        daskVarArrayStackDiskInterpPad.round(5).rechunk(
            (len(probVars), len(hourly_timesUnix), final_chunk, final_chunk)
        ).to_zarr(zarr_array, overwrite=True, compute=True)

    if save_type == "S3":
        zarr_store.close()

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [aigefs_store_name],
        base_time,
        "AIGEFS.time.pickle",
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
