# %% HRRR subhourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023
# Note that because the hourly script saves the 1-h forecast to S3, this script doesn't have to do this

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
import zarr
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/SubH"
)
forecast_process_path = forecast_process_dir + "/SubH_Process"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/SubH")
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/History/SubH")


# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunksize
finalChunk = 5

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

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

# %% Define base time from the most recent run
# base_time = pd.Timestamp("2023-07-01 00:00")
T0 = time.time()

latestRun = Herbie_latest(
    model="hrrr",
    n=3,
    freq="1h",
    fxx=[6],
    product="subh",
    verbose=False,
    priority="aws",
    save_dir=tmpDIR,
)

base_time = latestRun.date

print(base_time)

# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/SubH_v2.time.pickle"):
        with s3.open(forecast_path + "/SubH_v2.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to SubH, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/SubH_v2.time.pickle"):
        # Open the file in binary mode
        with open(forecast_path + "/SubH_v2.time.pickle", "rb") as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to SubH, ending")
            sys.exit()


zarrVars = (
    "time",
    "GUST_surface",
    "PRES_surface",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "REFC_entireatmosphere",
    "APCP_surface",
    "VIS_surface",
    "SPFH_2maboveground",
    "DSWRF_surface",
    "CAPE_surface",
)


#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours

# Do not include accum since this will only be used for currently = minutely
# Also no humidity, cloud cover, or vis data for some reason

# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|SPFH):2 m above ground:)"
matchstring_su = (
    ":((CRAIN|CICEP|CSNOW|CFRZR|PRES|PRATE|VIS|GUST|DSWRF|CAPE):surface:.*min fcst)"
)
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*min fcst)"
matchstring_sl = "(:(REFC):)"
matchstring_ap = "(:APCP:surface:)"

# Merge matchstrings for download
matchStrings = (
    matchstring_2m
    + "|"
    + matchstring_su
    + "|"
    + matchstring_10m
    + "|"
    + matchstring_sl
    + "|"
    + matchstring_ap
)

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
hrrr_range1 = range(1, 6)
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="hrrr",
    fxx=hrrr_range1,
    product="subh",
    verbose=False,
    priority="aws",
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
    + " -grib "
    + forecast_process_path
    + "_wgrib2_merged.grib2"
)

# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

# Use wgrib2 to rotate the wind vectors
# From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
lambertRotation = "lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000"

cmd2 = (
    f"{wgrib2_path}"
    + " "
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

# Convert to NetCDF
cmd3 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_wgrib2_merged.regrid"
    + " -netcdf "
    + forecast_process_path
    + "_wgrib2_merged.nc"
)

# Run wgrib2 to rotate winds and save as NetCDF
spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")
if spOUT3.returncode != 0:
    print(spOUT3.stderr)
    sys.exit()


# %% Create XArray
# Read the netcdf file using xarray
xarray_forecast_merged = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")

if len(xarray_forecast_merged.time) != len(hrrr_range1) * 4:
    print(len(xarray_forecast_merged.time))
    print(len(hrrr_range1) * 4)

    assert len(xarray_forecast_merged.time) == len(hrrr_range1) * 4, (
        "Incorrect number of timesteps! Exiting"
    )

# Save the dataset with compression
# with ProgressBar():
# xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 20, "x": processChunk, "y": processChunk}
)
xarray_forecast_merged.to_zarr(
    forecast_process_path + "_xr_merged.zarr", mode="w", consolidated=False
)

del xarray_forecast_merged

# Remove wgrib2 temp files
os.remove(forecast_process_path + "_wgrib2_merged.grib2")
os.remove(forecast_process_path + "_wgrib2_merged.regrid")
os.remove(forecast_process_path + "_wgrib2_merged.nc")


# %% Format as dask and save as zarr
#####################################################################################################\
# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

# This is quite a bit simplier since there's no historic data being ingested

for daskVarIDX, dask_var in enumerate(zarrVars[:]):
    daskForecastArray = da.from_zarr(
        forecast_process_path + "_xr_merged.zarr", component=dask_var, inline_array=True
    )

    if dask_var == "time":
        # Create a time array with the same shape
        daskCatTimes = daskForecastArray.astype("float32")

        # Get times as numpy
        npCatTimes = daskCatTimes.compute()

        daskArrayOut = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                (1, 1059, 1799),
            )
        ).rechunk((len(npCatTimes), 20, 20))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArrayList.append(
            daskForecastArray.rechunk(
                (len(npCatTimes), processChunk, processChunk)
            ).astype("float32")
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
        forecast_process_dir + "/SubH.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/SubH.zarr")


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

# %% Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/SubH.zarr.zip", forecast_path + "/v27/SubH.zarr.zip"
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/SubH.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/SubH.time.pickle",
        forecast_path + "/v27/SubH.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/SubH.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/SubH.time.pickle",
        forecast_path + "/v27/SubH.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/SubH.zarr",
        forecast_path + "/v27/SubH.zarr",
        dirs_exist_ok=True,
    )


# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
