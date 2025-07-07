# %% Script to test FastHerbie.py to download ECMWF data
# Alexander Rey, March 2025

# %% Import modules
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from herbie import FastHerbie, HerbieLatest, Path

from dask.diagnostics import ProgressBar


# Scipy Interp Function
def interp_time_block(y_block, idx0, idx1, w, valid):
    # y_block is a NumPy array of shape (Vb, T_old, Yb, Xb)
    # 1) fancy-index in NumPy only:
    y0 = y_block[:, idx0, :, :]
    y1 = y_block[:, idx1, :, :]
    # 2) add back your time‐axis weights in NumPy:
    w_r = w[None, :, None, None]
    omw_r = (1 - w)[None, :, None, None]
    # 3) linear blend
    y_interp = omw_r * y0 + w_r * y1

    # 4) zero‐out (or NaN‐out) anything outside the original time range
    #    here we choose NaN so it’s clear these were out-of-range
    if not np.all(valid):
        # valid==False where x_b is outside [x_a[0], x_a[-1]]
        inv = ~valid
        y_interp[:, inv, :, :] = np.nan

    return y_interp


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
ingestVersion = "v27"

wgrib2_path = os.getenv("wgrib2_path", default="/home/ubuntu/wgrib2_build/bin/wgrib2 ")

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/ECMWF"
)
forecast_process_path = forecast_process_dir + "/ECMWF_Process"
hist_process_path = forecast_process_dir + "/ECMWF_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/ECMWF")
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/History/ECMWF")


saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunksize
finalChunk = 3

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
    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

# %% Define base time from the most recent run
T0 = time.time()

latestRun = HerbieLatest(
    model="ifs",
    n=3,
    freq="12h",
    fxx=240,
    product="oper",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

base_time = latestRun.date
# base_time = pd.Timestamp("2024-03-24 06:00:00Z")

print(base_time)


# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/ECMWF.time.pickle"):
        with s3.open(forecast_path + "/ECMWF.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to ECMWF, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/ECMWF.time.pickle"):
        # Open the file in binary mode
        with open(forecast_path + "/ECMWF.time.pickle", "rb") as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to IFS, ending")
            sys.exit()

zarrVars = (
    "time",
    "GUST_10maboveground",
    "PRES_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "TPRATE_surface",
    "var0_1_193_surface",
    "PTYPE_surface",
    "TCDC_atmoscol",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
)
#####################################################################################################
# %% Download AIFS data using Herbie Latest
# Needed for tcc
# Find the latest run with 240 hours

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
aifs_range1 = range(0, 241, 6)

# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="12h"),
    model="aifs",
    fxx=aifs_range1,
    product="oper",
    verbose=False,
    save_dir=tmpDIR,
)

# Download the subsets
FH_forecastsub.download("tcc", verbose=False)

# Create list of downloaded grib files
gribList = [
    str(Path(x.get_localFilePath("tcc")).expand()) for x in FH_forecastsub.file_exists
]

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(gribList)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_aifs_wgrib2_merged.nc"
)


# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()


#####################################################################################################
# %% Download ENS data using Herbie Latest
# Needed for tcc
# Find the latest run with 240 hours

# Create a range of forecast lead times
ifs_range1 = range(3, 144, 3)
ifs_range2 = range(144, 241, 6)
ifsFileRange = [*ifs_range1, *ifs_range2]

# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="12h"),
    model="ifs",
    fxx=ifsFileRange,
    product="enfo",
    verbose=False,
    save_dir=tmpDIR,
)
gribList = []
failCount = 0
ensNum = 1
while ensNum < 51:
    # Download the subsets
    matchString = ":(tp:sfc:" + str(ensNum) + "):"
    FH_forecastsub.download(matchString, verbose=False)

    # Create list of downloaded grib files
    gribList.append(
        [
            str(Path(x.get_localFilePath(matchString)).expand())
            for x in FH_forecastsub.file_exists
        ]
    )

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(gribList[-1])
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + forecast_process_path
        + "_enfo_"
        + str(ensNum)
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2
    spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if spOUT.returncode != 0:
        print(spOUT.stderr)
        sys.exit()

    # Open the time merged file and fix the precipitation accumulation
    # Find the difference between steps to calculate the hourly rainfall
    xarray_forecast_merged = xr.open_mfdataset(
        [forecast_process_path + "_enfo_" + str(ensNum) + "_wgrib2_merged.nc"],
        engine="netcdf4",
    )

    # Check for download length
    if len(xarray_forecast_merged["time"]) != 64:
        print(
            "Member " + str(ensNum + 1) + " has not downloaded all files, trying again"
        )
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            break

        continue

    xarray_forecast_merged["var0_1_193_surface"] = xarray_forecast_merged[
        "var0_1_193_surface"
    ].copy(
        data=np.diff(
            xarray_forecast_merged["var0_1_193_surface"],
            axis=xarray_forecast_merged["var0_1_193_surface"].get_axis_num("time"),
            prepend=0,
        )
    )

    # Change the 3 and 6 hour accumulations to hourly
    # STeps 3 to 144 are 3-hourly, the rest are 6 hourly
    xarray_forecast_merged["var0_1_193_surface"][0:48] = (
        xarray_forecast_merged["var0_1_193_surface"][0:48] / 3
    )
    xarray_forecast_merged["var0_1_193_surface"][48:] = (
        xarray_forecast_merged["var0_1_193_surface"][48:] / 6
    )

    # Save as zarr
    xarray_forecast_merged.chunk(
        chunks={"time": 64, "latitude": 100, "longitude": 100}
    ).to_zarr(
        forecast_process_path + "_enfo_" + str(ensNum) + "_wgrib2_merged.zarr",
        mode="w",
        consolidated=False,
        compute=True,
    )

    print(ensNum)
    ensNum += 1


unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
#

# Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
ncLocalWorking_paths = [
    forecast_process_path + "_enfo_" + str(i) + "_wgrib2_merged.zarr"
    for i in range(1, 31, 1)
]

daskVarArrays = []
for local_ncpath in ncLocalWorking_paths:
    daskVarArrays.append(da.from_zarr(local_ncpath, "var0_1_193_surface"))

# Stack times together, keeping variables separate
daskArrays = da.stack(daskVarArrays, axis=0)


# Dict to hold output dask arrays
daskOutput = dict()

# Create new xr dataset for enso by dropping the data variable from xarray_forecast_merged
xr_ensoOut = xarray_forecast_merged.drop_vars("var0_1_193_surface")

# Find the probability of precipitation greater than 0.1 mm/h  across all members
xr_ensoOut["Precipitation_Prob"] = (
    ("time", "latitude", "longitude"),
    ((daskArrays) > 0.1).sum(axis=0) / 50,
)

# Find the standard deviation of precipitation accumulation across all members
xr_ensoOut["APCP_StdDev"] = (("time", "latitude", "longitude"), daskArrays.std(axis=0))

# Find the average precipitation accumulation across all members
xr_ensoOut["APCP_Mean"] = (("time", "latitude", "longitude"), daskArrays.mean(axis=0))

# Save as zarr
# with ProgressBar():
xr_ensoOut.to_zarr(
    forecast_process_path + "_ensoOut.zarr", mode="w", consolidated=False, compute=True
)


#####################################################################################################
# %% Download Base IFS data using Herbie Latest
# Find the latest run with 240 hours

# Define the subset of variables to download as a list of strings
# 2 m level – use ECMWF’s 2t (temperature) and 2d (dew point)
matchstring_2m = "(:(2d|2t):)"

matchstring_10m = "(:(10u|10v|10fg3|10fg):)"
matchstring_ap = "(:(ptype|tprate|tp):)"
matchstring_sl = "(:(msl):)"


# Merge matchstrings for download
matchStrings = (
    matchstring_2m + "|" + matchstring_10m + "|" + matchstring_ap + "|" + matchstring_sl
)


# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="12h"),
    model="ifs",
    fxx=ifsFileRange,
    product="oper",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

# Download the subsets
FH_forecastsub.download(matchStrings, verbose=False)

# Create list of downloaded grib files
gribList = [
    str(Path(x.get_localFilePath(matchStrings)).expand())
    for x in FH_forecastsub.file_exists
]

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(gribList)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_wgrib2_merged.nc"
)

# TP is var0_1_193_surface

# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()


# %% Merge the IFS, ENSO, and AIFS data

# Read the files using xarray
xarray_wgrib_merged = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")
xarray_wgrib_AIFS_merged = xr.open_mfdataset(
    forecast_process_path + "_aifs_wgrib2_merged.nc"
)
xarray_wgrib_enso_merged = xr.open_mfdataset(
    forecast_process_path + "_ensoOut.zarr", consolidated=False
)

# Reinterpolate the AIFS array to the same times as the IFS arrays
xarray_wgrib_AIFS_merged = xarray_wgrib_AIFS_merged.interp(
    time=xarray_wgrib_merged.time, method="linear"
)

# Merge the xarray objects
xarray_forecast_merged = xr.merge(
    [xarray_wgrib_merged, xarray_wgrib_AIFS_merged, xarray_wgrib_enso_merged],
    compat="override",
)

assert len(xarray_forecast_merged.time) == len(ifsFileRange), (
    "Incorrect number of timesteps! Exiting"
)

# Create a new time series
start = xarray_forecast_merged.time.min().values  # Adjust as necessary
end = xarray_forecast_merged.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hisPeriod, "h"), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hisPeriod, "h"),
            end=start - pd.Timedelta(1, "h"),
            freq="3h",
        ),
        xarray_forecast_merged.time.values,
    )
)
unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

# Chunk and save as zarr
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 64, "latitude": processChunk, "longitude": processChunk}
)

with ProgressBar():
    xarray_forecast_merged.to_zarr(
        forecast_process_path + "_merged.zarr",
        mode="w",
        consolidated=False,
        compute=True,
    )


# %% Delete to free memory
del (
    xarray_wgrib_merged,
    xarray_wgrib_AIFS_merged,
    xarray_wgrib_enso_merged,
    xarray_forecast_merged,
)

T1 = time.time()
print(T1 - T0)

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(hisPeriod, 1, -12):
    if saveType == "S3":
        # S3 Path Setup
        s3_path = (
            historic_path
            + "/ECMWF_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )
        if s3.exists(s3_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + s3_path)
            # If the file exists, check that it works

            # Try to open and read data from the last variable of the zarr file to check if it has already been saved
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
            + "/ECMWF_Hist"
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
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="12h",
    )
    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach
    fxx = range(3, 13, 3)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES, model="ifs", fxx=fxx, product="oper", verbose=False, save_dir=tmpDIR
    )

    # Download the subsets
    # Start with oper
    FH_histsub.download(matchStrings, verbose=False)

    # Create list of downloaded grib files
    gribList = [
        str(Path(x.get_localFilePath(matchStrings)).expand())
        for x in FH_histsub.file_exists
    ]

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(gribList)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + hist_process_path
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2
    spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if spOUT.returncode != 0:
        print(spOUT.stderr)
        sys.exit()

    ########################################################################
    ### Download the enfo data
    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES, model="ifs", fxx=fxx, product="enfo", verbose=False, save_dir=tmpDIR
    )

    gribList = []
    for ensNum in range(1, 51):
        # Download the subsets
        matchString = ":(tp:sfc:" + str(ensNum) + "):"
        FH_histsub.download(matchString, verbose=False)

        # Create list of downloaded grib files
        gribList.append(
            [
                str(Path(x.get_localFilePath(matchString)).expand())
                for x in FH_histsub.file_exists
            ]
        )

        # Create a string to pass to wgrib2 to merge all gribs into one netcdf
        cmd = (
            "cat "
            + " ".join(gribList[-1])
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + " -netcdf "
            + hist_process_path
            + "_enfo_"
            + str(ensNum)
            + "_wgrib2_merged.nc"
        )

        # Run wgrib2
        spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        if spOUT.returncode != 0:
            print(spOUT.stderr)
            sys.exit()

        # Open the time merged file and fix the precipitation accumulation
        # Find the difference between steps to calculate the hourly rainfall
        xarray_forecast_merged = xr.open_mfdataset(
            [hist_process_path + "_enfo_" + str(ensNum) + "_wgrib2_merged.nc"],
            engine="netcdf4",
        )

        xarray_forecast_merged["var0_1_193_surface"] = xarray_forecast_merged[
            "var0_1_193_surface"
        ].copy(
            data=np.diff(
                xarray_forecast_merged["var0_1_193_surface"],
                axis=xarray_forecast_merged["var0_1_193_surface"].get_axis_num("time"),
                prepend=0,
            )
        )

        # Change the 3 hour accumulations to hourly
        xarray_forecast_merged["var0_1_193_surface"] = (
            xarray_forecast_merged["var0_1_193_surface"] / 3
        )

        # Save as zarr
        xarray_forecast_merged.chunk(
            chunks={"time": 64, "latitude": 100, "longitude": 100}
        ).to_zarr(
            hist_process_path + "_enfo_" + str(ensNum) + "_wgrib2_merged.zarr",
            mode="w",
            consolidated=False,
            compute=True,
        )

        print(ensNum)

    # Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
    ncLocalWorking_paths = [
        hist_process_path + "_enfo_" + str(i) + "_wgrib2_merged.zarr"
        for i in range(1, 31, 1)
    ]

    # Note that the chunks
    daskVarArrays = []
    for local_ncpath in ncLocalWorking_paths:
        daskVarArrays.append(da.from_zarr(local_ncpath, "var0_1_193_surface"))

    # Stack times together, keeping variables separate
    daskArrays = da.stack(daskVarArrays, axis=0)

    # Dict to hold output dask arrays
    daskOutput = dict()

    # Create new xr dataset for enso by dropping the data variable from xarray_forecast_merged
    xr_ensoOut = xarray_forecast_merged.drop_vars("var0_1_193_surface")

    # Find the probability of precipitation greater than 0.1 mm/h  across all members
    xr_ensoOut["Precipitation_Prob"] = (
        ("time", "latitude", "longitude"),
        ((daskArrays) > 0.1).sum(axis=0) / 50,
    )

    # Find the standard deviation of precipitation accumulation across all members
    xr_ensoOut["APCP_StdDev"] = (
        ("time", "latitude", "longitude"),
        daskArrays.std(axis=0),
    )

    # Find the average precipitation accumulation across all members
    xr_ensoOut["APCP_Mean"] = (
        ("time", "latitude", "longitude"),
        daskArrays.mean(axis=0),
    )

    # Save as netcdf
    with ProgressBar():
        xr_ensoOut.to_zarr(
            hist_process_path + "_ensoOut.zarr",
            mode="w",
            consolidated=False,
            compute=True,
        )

    ########################################################################
    # Save the aifs data
    aifs_range = range(6, 13, 6)
    # Create FastHerbie object
    FH_histsub = FastHerbie(
        DATES,
        model="aifs",
        fxx=aifs_range,
        product="oper",
        verbose=False,
        save_dir=tmpDIR,
    )

    # Download the subsets
    FH_histsub.download("tcc", verbose=False)

    # Create list of downloaded grib files
    gribList = [
        str(Path(x.get_localFilePath("tcc")).expand()) for x in FH_histsub.file_exists
    ]

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(gribList)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + hist_process_path
        + "_aifs_wgrib2_merged.nc"
    )

    # Run wgrib2
    spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if spOUT.returncode != 0:
        print(spOUT.stderr)
        sys.exit()

    # Read the files using xarray
    xarray_hist_IFS_merged = xr.open_mfdataset(hist_process_path + "_wgrib2_merged.nc")
    xarray_hist_AIFS_merged = xr.open_mfdataset(
        hist_process_path + "_aifs_wgrib2_merged.nc"
    )
    xarray_hist_ENSO_merged = xr.open_mfdataset(
        hist_process_path + "_ensoOut.zarr", consolidated=False
    )

    # Reinterpolate the AIFS array to the same times as the IFS arrays
    xarray_hist_ENSO_merged = xarray_hist_ENSO_merged.interp(
        time=xarray_hist_IFS_merged.time, method="linear"
    )

    # Merge the xarray objects
    xarray_hist_merged = xr.merge(
        [xarray_hist_IFS_merged, xarray_hist_AIFS_merged, xarray_hist_ENSO_merged],
        compat="override",
    )

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

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

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times

    # Chunk and save as zarr
    xarray_hist_merged = xarray_hist_merged.chunk(
        chunks={"time": 64, "latitude": 100, "longitude": 100}
    )

    # with ProgressBar():
    xarray_hist_merged.to_zarr(
        store=zarrStore,
        mode="w",
        consolidated=False,
    )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    # os.remove(hist_process_path + "_wgrib2_merged.nc")
    # os.remove(hist_process_path + "_wgrib2_merged_UV.nc")

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
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    historic_path
    + "/ECMWF_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, 1, -12)
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

    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

    daskForecastArray = da.from_zarr(
        forecast_process_path + "_merged.zarr", component=dask_var, inline_array=True
    )

    if dask_var == "time":
        # Create a time array with the same shape
        daskVarArraysShape = da.reshape(daskVarArraysStack, (12, 1), merge_chunks=False)
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
        ).rechunk((len(stacked_timesUnix), processChunk, processChunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack, (12, 721, 1440), merge_chunks=False
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), processChunk, processChunk))
            .astype("float32")
        )

    daskVarArrays = []

    print(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
# Read in stacked 4D array back in
daskVarArrayListMerge.to_zarr(
    forecast_process_path + "_stack.zarr", overwrite=True, compute=True
)

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/ECMWF.zarr.zip", mode="a"
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/ECMWF.zarr")

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

#
# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Rechunk it to match the final array
# 3. Write it out to the zarr array

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

# Rechunk subset of data for maps!
# Want variables:
# 0 (time)
# 3 (TMP)
# 5 (UGRD)
# 6 (VGRD)
# 7 (PRATE)
# 8 (PACCUM)
# 9 (PTYPE)

# Loop through variables, creating a new one with a name and 36 x 100 x 100 chunks
# Save -12:24 hours, aka steps 24:60
# Create a Zarr array in the store with zstd compression
if saveType == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/ECMWF_Maps.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(forecast_process_dir + "/ECMWF_Maps.zarr")

for z in [0, 3, 5, 6, 7, 8, 9]:
    # Create a zarr backed dask array
    zarr_array = zarr.create_array(
        store=zarr_store_maps,
        name=zarrVars[z],
        shape=(36, daskVarArrayStackDisk.shape[2], daskVarArrayStackDisk.shape[3]),
        chunks=(36, 100, 100),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
    )

    da.rechunk(daskVarArrayStackDisk[z, 24:60, :, :], (36, 100, 100)).to_zarr(
        zarr_array, overwrite=True, compute=True
    )

    print(zarrVars[z])

if saveType == "S3":
    zarr_store_maps.close()

# %% Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/ECMWF.zarr.zip",
        forecast_path + "/" + ingestVersion + "/ECMWF.zarr.zip",
    )
    s3.put_file(
        forecast_process_dir + "/ECMWF_Maps.zarr.zip",
        forecast_path + "/" + ingestVersion + "/ECMWF_Maps.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/ECMWF.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/ECMWF.time.pickle",
        forecast_path + "/" + ingestVersion + "/ECMWF.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/ECMWF.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/ECMWF.time.pickle",
        forecast_path + "/" + ingestVersion + "/ECMWF.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/ECMWF.zarr",
        forecast_path + "/" + ingestVersion + "/ECMWF.zarr",
        dirs_exist_ok=True,
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/ECMWF_Maps.zarr",
        forecast_path + "/" + ingestVersion + "/ECMWF_Maps.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

T2 = time.time()
print(T2 - T0)
