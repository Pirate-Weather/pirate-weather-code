# %% Script to test FastHerbie.py to download GFS data
# Alexander Rey, September 2023

# %% Import modules
import logging
import os
import pickle
import shutil
import subprocess
import sys
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
from herbie import FastHerbie, HerbieLatest, Path

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR, MISSING_DATA
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    validate_grib_stats,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
hist_process_path = os.path.join(forecast_process_dir, "AIGEFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/AIGEFS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/AIGEFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["GEFS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GEFS"]

his_period = HISTORY_PERIODS["AIGEFS"]

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download":
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)


T0 = time.time()

latest_run = HerbieLatest(
    model="aigefs",
    n=3,
    freq="6h",
    fxx=240,
    product="sfc",
    verbose=False,
    member="avg",
    priority=["nomads"],
    save_dir=tmp_dir,
)

base_time = latest_run.date
base_time = pd.Timestamp("2026-03-02 06:00:00")

logger.info(base_time)


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to AIGEFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to AIGEFS, ending")
            sys.exit()

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

# Merge matchstrings for download
match_strings = ":APCP:surface:"

# Create a range of forecast lead times
aigefs_range = FORECAST_LEAD_RANGES["AIGEFS"]


# Create FastHerbie object for all 30 members
mem = 1
failCount = 0
while mem < 31:
    FH_IN = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="6h"),
        model="aigefs",
        fxx=aigefs_range,
        member="mem" + str(mem).zfill(3),
        product="sfc",
        verbose=True,
        priority=["aws", "nomads"],
        save_dir=tmp_dir,
        max_threads=20,
    )

    # Check for download length
    if len(FH_IN.file_exists) != 40:
        logger.warning("Member %d has not downloaded all files, trying again", mem)
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            break

        continue

    # Download and process the subsets
    FH_IN.download(verbose=True, max_threads=20, overwrite=True)

    # Create list of downloaded grib files
    grib_list = [str(Path(x.get_localFilePath()).expand()) for x in FH_IN.file_exists]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

    val_check = validate_grib_stats(grib_check)

    if not val_check:
        logger.warning("Member %d has not downloaded all files, trying again", mem)
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            break

        continue
    else:
        logger.info("Grib files passed validation, proceeding with processing")

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

    # Run wgrib2 to megre all the grib files
    sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if sp_out.returncode != 0:
        logger.error(sp_out.stderr)
        sys.exit()

    # Fix precip and chunk each member
    xarray_wgrib = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged_m" + str(mem) + ".nc"
    )

    # AIGEFS does not use accumulated precipitation, just 6 hour steps
    # Use the difference between 3 and 6 for every other timestep
    # Divide by 6 to get hourly accum
    xarray_wgrib["APCP_surface"] = xarray_wgrib["APCP_surface"] / 6

    # Get a list of all variables in the dataset
    wgribVars = list(xarray_wgrib.data_vars)

    # Keep only APCP_surface, dropping all other variables to save space and speed up processing
    # xarray_wgrib = xarray_wgrib[["time", "APCP_surface"]]

    # Check that there are 40 timesteps in the array
    if xarray_wgrib.dims["time"] != 40:
        logger.warning("Member %d does not have 40 timesteps, trying again", mem)
        # Print the number of timesteps for debugging
        logger.warning("Member %d has %d timesteps", mem, xarray_wgrib.dims["time"])
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            break

        continue

    xarray_wgrib = xarray_wgrib.chunk(
        chunks={"time": 40, "latitude": process_chunk, "longitude": process_chunk}
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
    )

    mem += 1

    # Pause for 10 seconds to avoid overwhelming NOMADS
    time.sleep(2)

# Create a new time series
start = xarray_wgrib.time.min().values  # Adjust as necessary
end = xarray_wgrib.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hours=his_period), end=end, freq="h"
)

# Plus 2 since we start at Hour 3
stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hours=his_period),
            end=start - pd.Timedelta(hours=1),
            freq="6h",
        ),
        xarray_wgrib.time.values,
    )
)

unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

ncLocalWorking_paths = [
    forecast_process_path + "_xr_m" + str(i) + ".zarr" for i in range(1, 31, 1)
]

# Dask
daskArrays = dict()


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
daskOutput = dict()

# Find the probability of precipitation greater than 0.1 mm/h  across all members
daskOutput["Precipitation_Prob"] = ((daskArrays["APCP_surface"]) > 0.1).sum(axis=0) / 30

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
        daskOutput[dask_var].rechunk((40, process_chunk, process_chunk)).to_zarr(
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

################################################################################################
################################################################################################
# %% Historic data
# Create a range of dates for historic data going back 48 hours
# Loop through the runs and check if they have already been processed to s3


# Preprocess function for xarray to add a member dimension
def preprocess(ds):
    return ds.expand_dims("member", axis=0)


# Note: since these files are only 6 hour segments (aka 2 timesteps instead of 80), all the fancy dask stuff isn't necessary
# 6 hour runs
for i in range(his_period, 0, -6):
    # s3_path_NC = s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ') + '.nc'

    # Try to open the zarr file to check if it has already been saved
    if save_type == "S3":
        # Create the S3 filesystem
        s3_path = (
            historic_path
            + "/AIGEFS_HistProb_"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a done file in S3
        if s3.exists(s3_path.replace(".zarr", ".done")):
            logger.info("File already exists in S3, skipping download for: %s", s3_path)

            # If the file exists, check that it works
            try:
                hisCheckStore = zarr.storage.FsspecStore.from_url(
                    s3_path,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                )
                zarr.open(hisCheckStore)[probVars[-1]][-1, -1, -1]
                continue  # If it exists, skip to the next iteration
            except Exception:
                logger.error("### Historic Data Failure!")
                logger.exception("Exception processing historic data", exc_info=True)

                # Delete the file if it exists
                if s3.exists(s3_path):
                    s3.rm(s3_path)
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/AIGEFS_HistProb_"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a loca done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info(
                "File already exists in S3, skipping download for: %s", local_path
            )
            continue
    logger.info(
        "Downloading: %s",
        (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"),
    )

    # Create a range of dates for historic data
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(hours=i),
        periods=1,
        freq="6h",
    )

    ### Find all the model runs
    FH_forecastsubMembers = []
    for mem in range(1, 31):
        FH_forecastsubMembers.append(
            FastHerbie(
                DATES,
                model="aigefs",
                fxx=[6],
                member="mem" + str(mem).zfill(3),
                product="sfc",
                verbose=False,
                priority=["aws", "nomads"],
                save_dir=tmp_dir,
            )
        )

    ### Download the members and merge
    mem = 1
    failCount = 0
    while mem < 31:
        # Download the subsets
        FH_forecastsubMembers[mem - 1].download(verbose=False, overwrite=True)
        # Create list of downloaded grib files
        grib_list = [
            str(Path(x.get_localFilePath()).expand())
            for x in FH_forecastsubMembers[mem - 1].file_exists
        ]

        # Perform a check if any data seems to be invalid
        cmd = (
            "cat "
            + " ".join(grib_list)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + " -s -stats"
        )

        grib_check = subprocess.run(
            cmd, shell=True, capture_output=True, encoding="utf-8"
        )

        val_check = validate_grib_stats(grib_check)
        if not val_check:
            logger.warning("Member %d has not downloaded all files, trying again", mem)
            failCount += 1

            # Break after 10 failed attempts
            if failCount > 10:
                break

            continue
        else:
            logger.info("Grib files passed validation, proceeding with processing")

        # Create a string to pass to wgrib2 to merge all gribs into one grib
        cmd = (
            "cat "
            + " ".join(grib_list)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + '-match ":APCP:surface:" '
            + "-netcdf "
            + hist_process_path
            + "_wgrib2_merged_m"
            + str(mem)
            + ".nc"
        )

        # Run wgrib2 to merge all the grib files
        sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        if sp_out.returncode != 0:
            logger.error(sp_out.stderr)
            failCount += 1

            # Break after 10 failed attempts
            if failCount > 10:
                break

            continue

        # Open the NetCDF file with xarray to process and compress
        xarray_hist_wgrib = xr.open_dataset(
            hist_process_path + "_wgrib2_merged_m" + str(mem) + ".nc"
        )

        # Sometimes there will be weird negative values, set them to zero
        xarray_hist_wgrib["APCP_surface"] = np.maximum(
            xarray_hist_wgrib["APCP_surface"], 0
        )

        # Divide by 6 to get hourly accumulations
        xarray_hist_wgrib["APCP_surface"] = xarray_hist_wgrib["APCP_surface"] / 6

        # Get a list of all variables in the dataset
        wgribVars = list(xarray_hist_wgrib.data_vars)

        xarray_wgrib = xarray_hist_wgrib.chunk(
            chunks={"time": 1, "latitude": process_chunk, "longitude": process_chunk}
        )

        xarray_wgrib.to_zarr(
            hist_process_path + "_xr_merged_m" + str(mem) + ".zarr",
            consolidated=False,
            mode="w",
        )

        # Delete the netcdf to save space
        subprocess.run(
            "rm " + hist_process_path + "_wgrib2_merged_m" + str(mem) + ".nc",
            shell=True,
            capture_output=True,
            encoding="utf-8",
        )

        mem += 1

    ### Merge all the members together and calculate probabilities and standard deviation
    # Read the merged netcdf files into xarray using the preprocess function, concatenating along the member dimension
    xarray_hist_wgrib_merged = xr.open_mfdataset(
        [
            hist_process_path + "_xr_merged_m" + str(mem) + ".zarr"
            for mem in range(1, 31)
        ],
        engine="zarr",
        preprocess=preprocess,
        combine="nested",
        concat_dim="member",
        chunks={
            "member": 30,
            "time": 1,
            "latitude": process_chunk,
            "longitude": process_chunk,
        },
        consolidated=False,
    )

    # Create an empty xarray dataset to store the probability of precipitation greater than 1 mm
    xarray_hist_wgrib_prob = xr.Dataset()

    # Find the probably of precipitation greater than 0.1 mm/h across all members
    xarray_hist_wgrib_prob["Precipitation_Prob"] = (
        (xarray_hist_wgrib_merged["APCP_surface"]) > 0.1
    ).sum(dim="member") / 30

    # Find the standard deviation of precipitation accumulation across all members
    xarray_hist_wgrib_prob["APCP_StdDev"] = xarray_hist_wgrib_merged[
        "APCP_surface"
    ].std(dim="member")

    # Find the average precipitation accumulation across all members
    xarray_hist_wgrib_prob["APCP_Mean"] = xarray_hist_wgrib_merged["APCP_surface"].mean(
        dim="member"
    )

    # Chunk the xarray dataset to speed up processing
    xarray_hist_wgrib_prob = xarray_hist_wgrib_prob.chunk(
        {"time": 1, "latitude": process_chunk, "longitude": process_chunk}
    )

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times

    # Save as zarr for timemachine
    # with ProgressBar():
    # Save as Zarr to s3 for Time Machine
    if save_type == "S3":
        zarrStore = zarr.storage.FsspecStore.from_url(
            s3_path,
            storage_options={
                "key": aws_access_key_id,
                "secret": aws_secret_access_key,
            },
        )
    else:
        # Create local Zarr store
        zarrStore = zarr.storage.LocalStore(local_path)

    xarray_hist_wgrib_prob.to_zarr(store=zarrStore, mode="w", consolidated=False)

    # Clear memory
    del xarray_hist_wgrib_prob, xarray_hist_wgrib, xarray_hist_wgrib_merged

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        done_file = s3_path.replace(".zarr", ".done")
        s3.touch(done_file)
    else:
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")


# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    historic_path
    + "/AIGEFS_Hist_v2"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(his_period, 1, -6)
]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(zarr_vars[:]):
    for local_ncpath in ncLocalWorking_paths:
        # If not found in array, use MISSING_DATA to show missing
        try:
            if save_type == "S3":
                daskVarArrays.append(
                    da.from_zarr(
                        local_ncpath,
                        component=dask_var,
                        inline_array=True,
                        storage_options={
                            "key": aws_access_key_id,
                            "secret": aws_secret_access_key,
                        },
                    )
                )

            else:
                daskVarArrays.append(
                    da.from_zarr(local_ncpath, component=dask_var, inline_array=True)
                )
        # Add a fallback in case of a FileNotFoundError
        except FileNotFoundError:
            logger.warning("File not found, adding NaN array for: %s", local_ncpath)
            daskVarArrays.append(
                da.full((6, 721, 1440), MISSING_DATA).rechunk(
                    (6, process_chunk, process_chunk)
                )
            )

    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

    daskForecastArray = da.from_zarr(
        forecast_process_path + "_.zarr", component=dask_var, inline_array=True
    )

    if dask_var == "time":
        # Create a time array with the same shape
        # This is because multiple steps are stored in each file
        daskVarArraysShape = da.reshape(
            daskVarArraysStack,
            (daskVarArraysStack.shape[0] * daskVarArraysStack.shape[1], 1),
            merge_chunks=False,
        )
        daskCatTimes = da.concatenate(
            (da.squeeze(daskVarArraysShape), daskForecastArray), axis=0
        ).astype("float32")

        # Get times as numpy
        npCatTimes = daskCatTimes.compute()

        daskArrayOut = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                (1, 721, 1440),
            )
        ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack,
            (daskVarArraysStack.shape[0] * daskVarArraysStack.shape[1], 721, 1440),
            merge_chunks=False,
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
            .astype("float32")
        )

    daskVarArrays = []

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
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/AIGEFS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/AIGEFS.zarr")


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
            len(zarr_vars),
            len(hourly_timesUnix),
            daskVarArrayStackDiskInterpPad.shape[2],
            daskVarArrayStackDiskInterpPad.shape[3],
        ),
        chunks=(len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
    )

    # 4. Rechunk it to match the final array
    # 5. Write it out to the zarr array
    daskVarArrayStackDiskInterpPad.round(5).rechunk(
        (len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk)
    ).to_zarr(zarr_array, overwrite=True, compute=True)


if save_type == "S3":
    zarr_store.close()

# Maps generation removed for Graphcast ingest (not required)

# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/AIGEFS.zarr.zip",
        forecast_path + "/" + ingest_version + "/AIGEFS.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/AIGEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/AIGEFS.time.pickle",
        forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/AIGEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/AIGEFS.time.pickle",
        forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/AIGEFS.zarr",
        forecast_path + "/" + ingest_version + "/AIGEFS.zarr",
        dirs_exist_ok=True,
    )

    # Maps not generated for Graphcast ingest
# Clean up
shutil.rmtree(forecast_process_dir)

# Timing
T1 = time.time()
logger.info(T1 - T0)
