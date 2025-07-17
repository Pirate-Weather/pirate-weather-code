# %% Script to ingest NBM Fire Index

# Alexander Rey, April 2024

# %% Import modules
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta

import dask
import dask.array as da
import netCDF4 as nc
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from herbie import FastHerbie, Herbie, Path

from ingest_utils import mask_invalid_data, interp_time_block


def rounder(t):
    if t.minute >= 30:
        if t.hour == 23:
            return t.replace(second=0, microsecond=0, minute=0, hour=0, day=t.day + 1)
        else:
            return t.replace(second=0, microsecond=0, minute=0, hour=t.hour + 1)
    else:
        return t.replace(second=0, microsecond=0, minute=0)


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
ingestVersion = "v27"

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/NBM_Fire"
)
forecast_process_path = forecast_process_dir + "/NBM_Fire_Process"
hist_process_path = forecast_process_dir + "/NBM_Fire_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/NBM_Fire")
historic_path = os.getenv(
    "historic_path", default="/home/ubuntu/Weather/History/NBM_Fire"
)


saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunksize
finalChunk = 5

hisPeriod = 48

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)

if saveType == "Download":
    if not os.path.exists(forecast_path + "/" + ingestVersion):
        os.makedirs(forecast_path + "/" + ingestVersion)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

# %% Define base time from the most recent run
T0 = time.time()

# Find latest 6-hourly run

# Start from now and work backwards in 6 hour increments
current_time = datetime.now()
hour = current_time.hour
# Calculate the most recent hour from 0, 6, 12, or 18 hours ago
if hour < 6:
    recent_hour = 0
elif hour < 12:
    recent_hour = 6
elif hour < 18:
    recent_hour = 12
else:
    recent_hour = 18

# Create a new datetime object with the most recent hour
most_recent_time = datetime(
    current_time.year, current_time.month, current_time.day, recent_hour, 0, 0
)

# Select the most recent 0,6,12,18 run
base_time = False
failCount = 0
while base_time is False:
    latestRuns = Herbie(
        most_recent_time,
        model="nbm",
        fxx=192,
        product="co",
        verbose=False,
        priority=["aws", "nomdas"],
        save_dir=tmpDIR,
    )
    if latestRuns.grib:
        base_time = most_recent_time
    else:
        most_recent_time = most_recent_time - timedelta(hours=6)
        failCount = failCount + 1
        print(failCount)

        if failCount == 2:
            print("No recent runs")
            exit(1)


# base_time = pd.Timestamp("2024-03-05 16:00")
# base_time = base_time - pd.Timedelta(1,'h')
print(base_time)

# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingestVersion + "/NBM_Fire.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingestVersion + "/NBM_Fire.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM_Fire, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingestVersion + "/NBM_Fire.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingestVersion + "/NBM_Fire.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM_Fire, ending")
            sys.exit()

zarrVars = ("time", "FOSINDX_surface")

#####################################################################################################
# %% Download forecast data using Herbie Latest
# Set download rannges
nbm_range = range(6, 192, 6)

# Define the subset of variables to download as a list of strings
matchstring_su = ":FOSINDX:"

# Merge matchstrings for download
matchStrings = matchstring_su


# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="nbm",
    fxx=nbm_range,
    product="co",
    verbose=False,
    priority=["aws", "nomads"],
    save_dir=tmpDIR,
)

FH_forecastsub.download(matchStrings, verbose=False)

# Create list of downloaded grib files
try:
    gribList = [
        str(Path(x.get_localFilePath(matchStrings)).expand())
        for x in FH_forecastsub.file_exists
    ]
except Exception:
    print("Download Failure 1, wait 20 seconds and retry")
    time.sleep(20)
    FH_forecastsub.download(matchStrings, verbose=False)
    try:
        gribList = [
            str(Path(x.get_localFilePath(matchStrings)).expand())
            for x in FH_forecastsub.file_exists
        ]
    except Exception:
        print("Download Failure 2, wait 20 seconds and retry")
        time.sleep(20)
        FH_forecastsub.download(matchStrings, verbose=False)
        try:
            gribList = [
                str(Path(x.get_localFilePath(matchStrings)).expand())
                for x in FH_forecastsub.file_exists
            ]
        except Exception:
            print("Download Failure 3, wait 20 seconds and retry")
            time.sleep(20)
            FH_forecastsub.download(matchStrings, verbose=False)
            try:
                gribList = [
                    str(Path(x.get_localFilePath(matchStrings)).expand())
                    for x in FH_forecastsub.file_exists
                ]
            except Exception:
                print("Download Failure 4, wait 20 seconds and retry")
                time.sleep(20)
                FH_forecastsub.download(matchStrings, verbose=False)
                try:
                    gribList = [
                        str(Path(x.get_localFilePath(matchStrings)).expand())
                        for x in FH_forecastsub.file_exists
                    ]
                except Exception:
                    print("Download Failure 5, wait 20 seconds and retry")
                    time.sleep(20)
                    FH_forecastsub.download(matchStrings, verbose=False)
                    try:
                        gribList = [
                            str(Path(x.get_localFilePath(matchStrings)).expand())
                            for x in FH_forecastsub.file_exists
                        ]
                    except Exception:
                        print("Download Failure 6, Fail")
                        exit(1)

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(gribList)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -grib "
    + forecast_process_path
    + "_wgrib2_merged.grib2"
)
# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

# Check output from wgrib2
# print(spOUT.stdout)

# Use wgrib2 to change the order
cmd2 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_wgrib2_merged.grib2 "
    + " -ijsmall_grib "
    + " 1:2345 1:1597 "
    + forecast_process_path
    + "_wgrib2_merged_order.grib"
)
spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")
if spOUT2.returncode != 0:
    print(spOUT2.stderr)
    sys.exit()
os.remove(forecast_process_path + "_wgrib2_merged.grib2")

# Convert to NetCDF
cmd4 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_wgrib2_merged_order.grib "
    + " -set_ext_name 1 -netcdf "
    + forecast_process_path
    + "_wgrib2_merged.nc"
)

# Run wgrib2 to rotate winds and save as NetCDF
spOUT4 = subprocess.run(cmd4, shell=True, capture_output=True, encoding="utf-8")
if spOUT4.returncode != 0:
    print(spOUT4.stderr)
    sys.exit()

os.remove(forecast_process_path + "_wgrib2_merged_order.grib")

#######
# Use Dask to create a merged array (too large for xarray)

# Create base xarray for time interpolation
xarray_forecast_base = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")

# Check length for errors
assert len(xarray_forecast_base.time) == len(nbm_range), (
    "Incorrect number of timesteps! Exiting"
)

# Create a new time series
start = xarray_forecast_base.time.min().values
end = xarray_forecast_base.time.max().values
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hisPeriod, "h"), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hisPeriod, "h"),
            end=start - pd.Timedelta(1, "h"),
            freq="6h",
        ),
        xarray_forecast_base.time.values,
    )
)

unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second


# Drop all variables
xarray_forecast_base = xarray_forecast_base.drop_vars(
    [i for i in xarray_forecast_base.data_vars]
)

# Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
# Note: don't chunk on loading since we don't know how wgrib2 chunked the files. Intead, read the variable into memory and chunk later
with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    for dask_var in zarrVars:
        daskArray = da.from_array(
            nc.Dataset(forecast_process_path + "_wgrib2_merged.nc")[dask_var], lock=True
        )

        # Rechunk
        daskArray = daskArray.rechunk(
            chunks=(len(nbm_range), processChunk, processChunk)
        )

        # Save merged and processed xarray dataset to disk using zarr with compression
        # Define the path to save the zarr dataset
        # Save the dataset with compression and filters for all variables
        if dask_var == "time":
            # Save the dataset without compression and filters for all variable
            daskArray.to_zarr(
                forecast_process_path + "_zarrs/" + dask_var + ".zarr", overwrite=True
            )
        else:
            # Save the dataset with compression and filters for all variable
            daskArray.to_zarr(
                forecast_process_path + "_zarrs/" + dask_var + ".zarr",
                overwrite=True,
            )


# Del to free memory
del daskArray, xarray_forecast_base

# Remove wgrib2 temp files
os.remove(forecast_process_path + "_wgrib2_merged.nc")

T1 = time.time()
print(T0 - T1)

################################################################################################
# %%Historic data
# Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
# Loop through the runs and check if they have already been processed to s3

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
for i in range(hisPeriod, 1, -6):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    if saveType == "S3":
        s3_path = (
            historic_path
            + "/NBM_Fire_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        if s3.exists(s3_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + s3_path)
            # If the file exists, check that it works
            try:
                hisCheckStore = zarr.storage.FsspecStore.from_url(
                    s3_path,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                )
                zarr.open(hisCheckStore)[zarrVars[-1]][-1, -1, -1]
                continue  # If it exists, skip to the next iteration
            except Exception:
                print("### Historic Data Failure!")
                print(traceback.print_exc())

                # Delete the file if it exists
                if s3.exists(s3_path):
                    s3.rm(s3_path)
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/NBM_Fire_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a loca done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + local_path)
            continue

    print(
        "Downloading: " + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    )

    # Create a range of dates for historic data going back 48 hours
    # Forward looking, which makes sense since the data at 06Z is the max from 00Z to 06Z
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="1h",
    )

    # Create a range of forecast lead times
    # Only want forecast at hour 1- SLightly less accurate than initializing at hour 0 but much avoids precipitation accumulation issues
    fxx = [6]

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="nbm",
        fxx=fxx,
        product="co",
        verbose=False,
        priority="aws",
        save_dir=tmpDIR,
    )

    # Main Vars + Accum
    # Download the subsets
    FH_histsub.download(matchStrings, verbose=False)

    # Use wgrib2 to change the order
    cmd1 = (
        f"{wgrib2_path}"
        + "  "
        + str(FH_histsub.file_exists[0].get_localFilePath(matchStrings))
        + " -ijsmall_grib "
        + " 1:2345 1:1597 "
        + hist_process_path
        + "_wgrib2_merged_order.grib"
    )
    spOUT1 = subprocess.run(cmd1, shell=True, capture_output=True, encoding="utf-8")
    if spOUT1.returncode != 0:
        print(spOUT1.stderr)
        sys.exit()

    # Convert to NetCDF
    cmd3 = (
        f"{wgrib2_path}"
        + " "
        + hist_process_path
        + "_wgrib2_merged_order.grib "
        + " -set_ext_name 1 -netcdf "
        + hist_process_path
        + "_wgrib_merge.nc"
    )
    spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")
    if spOUT3.returncode != 0:
        print(spOUT3.stderr)
        sys.exit()

    # Merge the  xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib = xr.open_dataset(hist_process_path + "_wgrib_merge.nc")

    # Save merged and processed xarray dataset to disk using zarr
    # No chunking since only one time step
    # Save as Zarr to s3 for Time Machine
    if saveType == "S3":
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

    xarray_his_wgrib.to_zarr(store=zarrStore, mode="w", consolidated=False)

    # Clear the xarray dataset from memory
    del xarray_his_wgrib

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged_order.grib")
    os.remove(hist_process_path + "_wgrib_merge.nc")
    # os.remove(hist_process_path + '_ncTemp.nc')

    # Save a done file to s3 to indicate that the historic data has been processed
    if saveType == "S3":
        done_file = s3_path.replace(".zarr", ".done")
        s3.touch(done_file)
    else:
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    print((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))


# %% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################
ncLocalWorking_paths = [
    historic_path
    + "/NBM_Fire_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, 1, -6)
]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(zarrVars[:]):
    for local_ncpath in ncLocalWorking_paths:
        if saveType == "S3":
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

    daskVarArraysStack = da.stack(
        daskVarArrays, allow_unknown_chunksizes=True
    ).squeeze()

    daskForecastArray = da.from_zarr(
        forecast_process_path + "_zarrs" + "/" + dask_var + ".zarr", inline_array=True
    )

    if dask_var == "time":
        # Create a time array with the same shape
        daskCatTimes = da.concatenate(
            (da.squeeze(daskVarArraysStack), daskForecastArray), axis=0
        ).astype("float32")

        # Get times as numpy
        npCatTimes = daskCatTimes.compute()

        daskArrayOut = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                (1, 1597, 2345),
            )
        ).rechunk((len(stacked_timesUnix), processChunk, processChunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskArrayOut = da.concatenate((daskVarArraysStack, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), processChunk, processChunk))
            .astype("float32")
        )

    daskVarArrays = []

    print(dask_var)


# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Mask out invalid data
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
if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/NBM_Fire.zarr.zip", mode="a", compression=0
    )

else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/NBM_Fire.zarr")

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarrVars),
        len(hourly_timesUnix),
        daskVarArrayStackDisk.shape[2],
        daskVarArrayStackDisk.shape[3],
    ),
    chunks=(len(zarrVars), len(hourly_timesUnix), finalChunk, finalChunk),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)


# Precompute the two neighbor‐indices and the weights
x_a = np.array(stacked_timesUnix)
x_b = np.array(hourly_timesUnix)

idx = np.searchsorted(x_a, x_b) - 1
idx0 = np.clip(idx, 0, len(x_a) - 2)
idx1 = idx0 + 1
w = (x_b - x_a[idx0]) / (x_a[idx1] - x_a[idx0])  # float array, shape (T_new,)

# boolean mask of “in‐range” points
valid = (x_b >= x_a[0]) & (x_b <= x_a[-1])  # shape (T_new,)

# with ProgressBar():
da.map_blocks(
    interp_time_block,
    daskVarArrayStackDisk,
    idx0,
    idx1,
    w,
    valid,
    dtype="float32",
    chunks=(1, len(hourly_timesUnix), processChunk, processChunk),
).round(3).rechunk(
    (len(zarrVars), len(hourly_timesUnix), finalChunk, finalChunk)
).to_zarr(zarr_array, overwrite=True, compute=True)


if saveType == "S3":
    zarr_store.close()

# Test Read
# zarr_store_test = zarr.storage.ZipStore(merge_process_dir + '/NBM_Fire.zarr.zip', mode='r')
# testZarr = zarr.open_array(zarr_store_test)
# print(testZarr[:,:,100,100])

if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/NBM_Fire.zarr.zip",
        forecast_path + "/" + ingestVersion + "/NBM_Fire.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/NBM_Fire.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/NBM_Fire.time.pickle",
        forecast_path + "/" + ingestVersion + "/NBM_Fire.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/NBM_Fire.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/NBM_Fire.time.pickle",
        forecast_path + "/" + ingestVersion + "/NBM_Fire.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/NBM_Fire.zarr",
        forecast_path + "/" + ingestVersion + "/NBM_Fire.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)
