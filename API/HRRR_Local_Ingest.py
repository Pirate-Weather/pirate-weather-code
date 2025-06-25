# %% HRRR Hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023

# %% Import modules
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
import zarr.storage
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/HRRR"
)
forecast_process_path = forecast_process_dir + "/HRRR_Process"
hist_process_path = forecast_process_dir + "/HRRR_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/HRRR")
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/History/HRRR")


saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunksize
finalChunk = 5

hisPeriod = 36

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
# base_time = pd.Timestamp("2023-07-01 00:00")
T0 = time.time()

latestRun = Herbie_latest(
    model="hrrr", n=6, freq="1h", fxx=[18], product="sfc", verbose=False, priority="aws"
)

base_time = latestRun.date

print(base_time)
# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/HRRR.time.pickle"):
        with s3.open(forecast_path + "/HRRR.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to HRRR, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/HRRR.time.pickle"):
        # Open the file in binary mode
        with open(forecast_path + "/HRRR.time.pickle", "rb") as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to HRRR, ending")
            sys.exit()


zarrVars = (
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
    "REFC_entireatmosphere",
    "DSWRF_surface",
    "CAPE_surface",
)

#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours


# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
matchstring_8m = ":(MASSDEN:8 m above ground:)"
matchstring_su = (
    ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST|DSWRF|CAPE):surface:.*hour fcst)"
)
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_cl = "(:TCDC:entire atmosphere:.*hour fcst)"
matchstring_ap = "(:APCP:surface:0-[1-9]*)"
matchstring_sl = "(:(MSLMA|REFC):)"

# Merge matchstrings for download
matchStrings = (
    matchstring_2m
    + "|"
    + matchstring_su
    + "|"
    + matchstring_10m
    + "|"
    + matchstring_cl
    + "|"
    + matchstring_ap
    + "|"
    + matchstring_8m
    + "|"
    + matchstring_sl
)

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
hrrr_range1 = range(1, 19)
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="hrrr",
    fxx=hrrr_range1,
    product="sfc",
    verbose=False,
    priority="aws",
    save_dir=tmpDIR,
)

# Download the subsets
FH_forecastsub.download(matchStrings, verbose=True)

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

# Use wgrib2 to rotate the wind vectors
# From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
lambertRotation = "lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000"

cmd2 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_wgrib2_merged.grib2 "
    + "-new_grid_winds earth -new_grid "
    + lambertRotation
    + " "
    + forecast_process_path
    + "_wgrib2_merged.regrid"
)

# Run wgrib2 to rotate winds and save as NetCDF
spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")
if spOUT2.returncode != 0:
    print(spOUT2.stderr)
    sys.exit()

# Check output from wgrib2
# print(spOUT2.stdout)

# Convert to NetCDF
cmd3 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_wgrib2_merged.regrid "
    + " -netcdf "
    + forecast_process_path
    + "_wgrib2_merged.nc"
)

# Run wgrib2 to rotate winds and save as NetCDF
spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")
if spOUT3.returncode != 0:
    print(spOUT3.stderr)
    sys.exit()

# Check output from wgrib2
# print(spOUT3.stdout)

# %% Create XArray
# Read the netcdf file using xarray
xarray_forecast_merged = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")

# %% Fix things
# Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged["APCP_surface"].copy(
    data=np.diff(
        xarray_forecast_merged["APCP_surface"],
        axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
        prepend=0,
    )
)

# %% Save merged and processed xarray dataset to disk using zarr with compression
# Define the path to save the zarr dataset

assert len(xarray_forecast_merged.time) == len(hrrr_range1), (
    "Incorrect number of timesteps! Exiting"
)

# with ProgressBar():
# xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 18, "x": processChunk, "y": processChunk}
)
xarray_forecast_merged.to_zarr(
    forecast_process_path + "merged_zarr.zarr", mode="w", consolidated=False
)


# Clear the xaarray dataset from memory
del xarray_forecast_merged

# Remove wgrib2 temp files
os.remove(forecast_process_path + "_wgrib2_merged.grib2")
os.remove(forecast_process_path + "_wgrib2_merged.regrid")
os.remove(forecast_process_path + "_wgrib2_merged.nc")

print("FORECAST COMPLETE")
################################################################################################
# %%Historic data
# Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
# Create the S3 filesystem

# Saving hourly forecasts means that time machine can grab 24 of them to make a daily forecast
# SubH and 48H forecasts will not be required for time machine then!

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
for i in range(hisPeriod, -1, -1):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601
    s3_path = (
        historic_path
        + "/HRRR_Hist_v2"
        + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        + ".zarr"
    )

    # Try to open the zarr file to check if it has already been saved
    if saveType == "S3":
        # Create the S3 filesystem
        s3_path = (
            historic_path
            + "/HRRR_Hist_v2"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        if s3.exists(s3_path):
            # Check that all the data is there and that the data is the right shape
            zarrCheckStore = zarr.storage.FsspecStore.from_url(
                s3_path,
                storage_options={
                    "key": aws_access_key_id,
                    "secret": aws_secret_access_key,
                },
            )

            zarrCheck = zarr.open(zarrCheckStore, "r")

            # # Try to open the zarr file to check if it has already been saved
            if (len(zarrCheck) - 4) == len(
                zarrVars
            ):  # Subtract 4 for lat, lon, x, and y
                if zarrCheck[zarrVars[-1]].shape[1] == 1059:
                    if zarrCheck[zarrVars[-1]].shape[2] == 1799:
                        # print('Data is there and the right shape')
                        continue

    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/HRRR_Hist_v2"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check if local file exists
        if os.path.exists(local_path):
            # Check that all the data is there and that the data is the right shape
            zarrCheck = zarr.open(local_path, "r")

            # # Try to open the zarr file to check if it has already been saved
            if (len(zarrCheck) - 4) == len(
                zarrVars
            ):  # Subtract 4 for lat, lon, x, and y
                if zarrCheck[zarrVars[-1]].shape[1] == 1059:
                    if zarrCheck[zarrVars[-1]].shape[2] == 1799:
                        # print('Data is there and the right shape')
                        continue

    print(
        "Downloading: " + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    )

    # Create a range of dates for historic data going back 48 hours
    # Since the first hour forecast is used, then the time is an hour behind
    # So data for 18:00 would be the 1st hour of the 17:00 forecast.
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i + 1) + "h"),
        periods=1,
        freq="1h",
    )

    # Create a range of forecast lead times
    # Only want forecast at hour 1- SLightly less accurate than initializing at hour 0 but much avoids precipitation accumulation issues
    fxx = range(1, 2)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="hrrr",
        fxx=fxx,
        product="sfc",
        verbose=False,
        priority="aws",
        save_dir=tmpDIR,
    )

    # Download the subsets
    FH_histsub.download(matchStrings, verbose=False)

    # Use wgrib2 to rotate the wind vectors
    # From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
    lambertRotation = "lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000"

    cmd2 = (
        f"{wgrib2_path}"
        + " "
        + str(FH_histsub.file_exists[0].get_localFilePath(matchStrings))
        + " "
        + "-new_grid_winds earth -new_grid "
        + lambertRotation
        + " "
        + hist_process_path
        + "_wgrib_merge.regrid"
    )

    # Run wgrib2 to rotate winds and save as NetCDF
    spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")
    if spOUT2.returncode != 0:
        print(spOUT2.stderr)
        sys.exit()

    # Convert to NetCDF
    cmd3 = (
        f"{wgrib2_path}"
        + " "
        + hist_process_path
        + "_wgrib_merge.regrid "
        + " -netcdf "
        + hist_process_path
        + "_wgrib_merge.nc"
    )

    # Run wgrib2 to rotate winds and save as NetCDF
    spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")
    if spOUT3.returncode != 0:
        print(spOUT3.stderr)
        sys.exit()

    # Merge the  xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib = xr.open_dataset(hist_process_path + "_wgrib_merge.nc")

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Save the dataset with compression and filters for all variables
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
    os.remove(hist_process_path + "_wgrib_merge.regrid")
    os.remove(hist_process_path + "_wgrib_merge.nc")

    print((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))

# %% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################
# Get the s3 paths to the historic data
ncHistWorking_paths = [
    historic_path
    + "/HRRR_Hist_v2"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, -1, -1)
]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for dask_var in zarrVars:
    for local_ncpath in ncHistWorking_paths:
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
    # Stack historic
    daskVarArraysStack = da.stack(daskVarArrays)

    # Add zarr Forecast
    daskForecastArray = da.from_zarr(
        forecast_process_path + "merged_zarr.zarr",
        component=dask_var,
        inline_array=True,
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
                (1, 1059, 1799),
            )
        ).rechunk((len(npCatTimes), processChunk, processChunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskArrayOut = da.concatenate(
            (daskVarArraysStack.squeeze(), daskForecastArray), axis=0
        )

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(npCatTimes), processChunk, processChunk))
            .astype("float32")
        )

    daskVarArrays = []

    print(dask_var)


# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
# with ProgressBar():
daskVarArrayListMerge.to_zarr(
    forecast_process_path + "_stack.zarr", overwrite=True, compute=True
)

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/HRRR.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/HRRR.zarr")

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=daskVarArrayStackDisk.shape,
    chunks=(
        len(zarrVars),
        daskVarArrayStackDisk.shape[1],
        finalChunk,
        finalChunk,
    ),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)


# with ProgressBar():
da.rechunk(
    daskVarArrayStackDisk.round(3),
    (len(zarrVars), daskVarArrayStackDisk.shape[1], finalChunk, finalChunk),
).to_zarr(zarr_array, compute=True)


if saveType == "S3":
    zarr_store.close()


# Rechunk subset of data for maps!
# Want variables:
# 0 (time)
# 4 (TMP)
# 7 (UGRD)
# 8 (VGRD)
# 9 (PRATE)
# 11:14 (PTYPE)
# 16 (MASSDEN)
# 17 (REFC)

# Loop through variables, creating a new one with a name and 36 x 100 x 100 chunks
# Save -12:24 hours, aka steps 24:60
# Create a Zarr array in the store with zstd compression
if saveType == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/HRRR_maps.zarr.zip", mode="a"
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(forecast_process_dir + "/HRRR_maps.zarr")

for z in (0, 4, 7, 8, 9, 11, 12, 13, 14, 16, 17):
    # Create a zarr backed dask array
    zarr_array = zarr.create_array(
        store=zarr_store_maps,
        name=zarrVars[z],
        shape=(
            36,
            daskVarArrayStackDisk.shape[2],
            daskVarArrayStackDisk.shape[3],
        ),
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
        forecast_process_dir + "/HRRR.zarr.zip", forecast_path + "/HRRR.zarr.zip"
    )
    s3.put_file(
        forecast_process_dir + "/HRRR_maps.zarr.zip",
        forecast_path + "/HRRR_maps.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/HRRR.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/HRRR.time.pickle",
        forecast_path + "/HRRR.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/HRRR.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/HRRR.time.pickle",
        forecast_path + "/HRRR.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/HRRR.zarr",
        forecast_path + "/HRRR.zarr",
        dirs_exist_ok=True,
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/HRRR_maps.zarr",
        forecast_path + "/HRRR_maps.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
