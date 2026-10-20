# %% GEFS Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, September 2023

# %% Import modules
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
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import check_grib_stats, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    download_herbie_with_retry,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

DEFAULT_HERBIE_RETRIES = 5
DEFAULT_HERBIE_RETRY_SLEEP = 20


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GEFS")
forecast_process_path = forecast_process_dir + "/GEFS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GEFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


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
    zarr_store_workers = configure_zarr_writes()

    # Define the processing chunk size
    process_chunk = CHUNK_SIZES["GEFS"]

    # Define the final x/y chunksize
    final_chunk = FINAL_CHUNK_SIZES["GEFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    # %% Find the most recent run of the GEFS model
    T0 = time.time()

    latest_run = Herbie_latest(
        model="gefs",
        n=3,
        freq="6h",
        fxx=[240],
        product="atmos.25",
        verbose=True,
        member="avg",
        priority=["aws", "nomads"],
    )

    base_time = latest_run.date
    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/GEFS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to GEFS, ending")
        return

    logger.info(base_time)

    zarr_vars = (
        "time",
        "APCP_surface",
        "CSNOW_surface",
        "CICEP_surface",
        "CFRZR_surface",
        "CRAIN_surface",
    )

    probVars = (
        "time",
        "Precipitation_Prob",
        "APCP_Mean",
        "APCP_StdDev",
        "CSNOW_Prob",
        "CICEP_Prob",
        "CFRZR_Prob",
        "CRAIN_Prob",
    )

    #####################################################################################################
    # %% Download forecast data for all 30 members to find percentages

    # Define the subset of variables to download as a list of strings
    matchstring_su = "(:(CRAIN|CICEP|CSNOW|CFRZR):)"
    matchstring_ap = "(:APCP:)"

    # Merge matchstrings for download
    match_strings = matchstring_su + "|" + matchstring_ap

    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach

    gefs_range = FORECAST_LEAD_RANGES["GEFS"]

    # Create FastHerbie object for all 30 members
    FH_forecastsubMembers = []
    mem = 0
    failCount = 0
    forecast_time_values = None
    while mem < 30:
        FH_IN = FastHerbie(
            pd.date_range(start=base_time, periods=1, freq="6h"),
            model="gefs",
            fxx=gefs_range,
            member=mem + 1,
            product="atmos.25",
            verbose=False,
            priority=["aws", "nomads"],
            save_dir=tmp_dir,
        )

        # Check for download length
        if len(FH_IN.file_exists) != 80:
            logger.warning(
                "Member %d has not downloaded all files, trying again", mem + 1
            )
            failCount += 1

            # Break after 10 failed attempts
            if failCount > 10:
                break

            continue

        FH_forecastsubMembers.append(FH_IN)

        # Download and process the subsets with retry
        download_herbie_with_retry(
            herbie_obj=FH_forecastsubMembers[mem],
            search=match_strings,
            expected_count=len(gefs_range),
            dataset_name=f"GEFS member {mem + 1} forecast",
            retries=herbie_download_retries,
            retry_sleep_s=herbie_retry_sleep_seconds,
        )

        # Create list of downloaded grib files
        grib_list = [
            str(Path(x.get_localFilePath(match_strings)).expand())
            for x in FH_forecastsubMembers[mem].file_exists
        ]

        # Perform a check if any data seems to be invalid
        check_grib_stats(grib_list, wgrib2_path, f"GEFS member {mem + 1}")

        # Create a string to pass to wgrib2 to merge all gribs into one grib
        cmd = (
            "cat "
            + " ".join(grib_list)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + "-netcdf "
            + forecast_process_path
            + "_wgrib2_merged_m"
            + str(mem + 1)
            + ".nc"
        )

        # Run wgrib2 to megre all the grib files
        run_checked(cmd, f"GEFS member {mem + 1} wgrib2 NetCDF conversion")

        # Fix precip and chunk each member
        xarray_wgrib = xr.open_dataset(
            forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc"
        )

        # Change from 3 and 6 hour accumulation to 3 hour accumulation
        # Use the difference between 3 and 6 for every other timestep
        apcp_diff_xr = xarray_wgrib["APCP_surface"].diff(dim="time")
        xarray_wgrib["APCP_surface"][slice(1, None, 2), :, :] = apcp_diff_xr[
            slice(0, None, 2), :, :
        ]

        # Sometimes there will be weird tiny negative values, set them to zero
        xarray_wgrib["APCP_surface"] = np.maximum(xarray_wgrib["APCP_surface"], 0)

        # Divide by 3 to get hourly accum
        xarray_wgrib["APCP_surface"] = xarray_wgrib["APCP_surface"] / 3

        # NOTE: Because the cateogical vars are mixed (0-3 and 0-6) intervals, there can be values even when there's no precip
        for var in ["CRAIN", "CSNOW", "CFRZR", "CICEP"]:
            xarray_wgrib[var + "_surface"] = xarray_wgrib[var + "_surface"].where(
                xarray_wgrib["APCP_surface"] != 0, 0
            )

        # Define compression and chunking for each variable
        # Compress to save space
        # Save the dataset as a nc file with compression
        # encoding = {
        #     vname: {"zlib": True, "complevel": 1, "chunksizes": (80, 60, 60)}
        #     for vname in wgribVars
        # }
        # xarray_wgrib.to_netcdf(
        #     forecast_process_path + "_xr_m" + str(mem + 1) + ".nc", encoding=encoding
        # )

        xarray_wgrib = xarray_wgrib.chunk(
            chunks={"time": 80, "latitude": process_chunk, "longitude": process_chunk}
        )

        forecast_time_values = xarray_wgrib["time"].values
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            xarray_wgrib.to_zarr(
                forecast_process_path + "_xr_m" + str(mem + 1) + ".zarr",
                consolidated=False,
                mode="w",
                chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
            )
        xarray_wgrib.close()

        # Delete the wgrib netcdf to save space
        forecast_merged_path = (
            forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc"
        )
        if os.path.exists(forecast_merged_path):
            os.remove(forecast_merged_path)

        mem += 1

    # Hourly target axis runs from the first stored forecast valid time
    # (base_time + 3 h) to the last one. GEFS starts at hour 3, so hours
    # base_time+1 and +2 are not in this store; the API falls back to other
    # sources for them.
    if forecast_time_values is None:
        raise RuntimeError("GEFS forecast processing did not produce any time values.")
    start = np.min(forecast_time_values)
    end = np.max(forecast_time_values)
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    stacked_times = forecast_time_values

    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    ncLocalWorking_paths = [
        forecast_process_path + "_xr_m" + str(i) + ".zarr" for i in range(1, 31, 1)
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
    ) / 30

    # Find the average precipitation accumulation and categorical parameters across all members
    for var in ["CRAIN", "CSNOW", "CFRZR", "CICEP"]:
        daskOutput[var + "_Prob"] = daskArrays[var + "_surface"].sum(axis=0) / 30

    # Find the standard deviation of precipitation accumulation across all members
    daskOutput["APCP_StdDev"] = daskArrays["APCP_surface"].std(axis=0)

    # Find the average precipitation accumulation across all members
    daskOutput["APCP_Mean"] = daskArrays["APCP_surface"].mean(axis=0)

    # Copy time over
    daskOutput["time"] = daskArrays["time"][1, :]

    # filters = [BitRound(keepbits=12)]  # Only keep ~ 3 significant digits
    # compressor = Blosc(cname="zstd", clevel=1)  # Use zstd compression

    for dask_var in probVars:
        # with ProgressBar():
        if dask_var == "time":
            with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
                daskOutput[dask_var].to_zarr(
                    forecast_process_path + "_" + dask_var + ".zarr",
                    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                    overwrite=True,
                    compute=True,
                )
        else:
            with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
                daskOutput[dask_var].rechunk(
                    (80, process_chunk, process_chunk)
                ).to_zarr(
                    forecast_process_path + "_" + dask_var + ".zarr",
                    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                    dtype="float32",
                    overwrite=True,
                    compute=True,
                )

    # %% Delete to free memory
    del xarray_wgrib, FH_forecastsubMembers, daskOutput, daskArrays, apcp_diff_xr

    #####################################################################################################
    # %% Stack the forecast variables and then squash using dask
    daskVarArrayList = []

    for dask_var in probVars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_" + dask_var + ".zarr", inline_array=True
        )

        if dask_var == "time":
            # Get times as numpy
            npCatTimes = daskForecastArray.astype("float32").compute()
            # Check that the stored times are aligned with the expected forecast times
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
    daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

    # Write out to disk
    # This intermediate step is necessary to avoid memory overflow
    # with ProgressBar():
    # Read in stacked 4D array back in
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr", overwrite=True, compute=True
        )

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

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

    # 3. Write the final store
    gefs_store_name = "GEFS.zarr.zip" if save_type == "S3" else "GEFS.zarr"
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/" + gefs_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [gefs_store_name],
        base_time,
        "GEFS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)

    # Test Read
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
