# %% HRRR 6-hourly Processing script using Dask, FastHerbie, and wgrib2
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
from numcodecs import BitRound, Blosc

from ingest_utils import mask_invalid_data, validate_grib_stats, mask_invalid_refc


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
ingestVersion = "v27"

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/HRRR_6H"
)
forecast_process_path = forecast_process_dir + "/HRRR_6H_Process"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/HRRR_6H")

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


# %% Define base time from the most recent run
# base_time = pd.Timestamp("2025-07-16 18:00")
T0 = time.time()

latestRun = Herbie_latest(
    model="hrrr", n=3, freq="6h", fxx=[48], product="sfc", verbose=False, priority="aws"
)

base_time = latestRun.date

print(base_time)
# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingestVersion + "/HRRR_6H.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingestVersion + "/HRRR_6H.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to HRRR_6H, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingestVersion + "/HRRR_6H.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingestVersion + "/HRRR_6H.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to HRRR_6H, ending")
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
hrrr_range1 = range(18, 49)
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
FH_forecastsub.download(matchStrings, verbose=False)


# Check for download length
if len(FH_forecastsub.file_exists) != len(hrrr_range1):
    print(
        "Download failed, expected "
        + str(len(hrrr_range1))
        + " files but got "
        + str(len(FH_forecastsub.file_exists))
    )
    sys.exit(1)


# Create list of downloaded grib files
gribList = [
    str(Path(x.get_localFilePath(matchStrings)).expand())
    for x in FH_forecastsub.file_exists
]

# Perform a check if any data seems to be invalid
cmd = "cat " + " ".join(gribList) + " | " + f"{wgrib2_path}" + "- -s -stats"

gribCheck = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

validate_grib_stats(gribCheck)
print("Grib files passed validation, proceeding with processing")

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
    + " "
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

# print(xarray_forecast_merged.variables)

assert len(xarray_forecast_merged.time) == len(hrrr_range1), (
    "Incorrect number of timesteps! Exiting"
)

# %% Fix things
# Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged["APCP_surface"].copy(
    data=np.diff(
        xarray_forecast_merged["APCP_surface"],
        axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
        prepend=0,
    )
)

# Adjust smoke units to avoid rounding issues
xarray_forecast_merged["MASSDEN_8maboveground"] = (
    xarray_forecast_merged["MASSDEN_8maboveground"] * 1e9
)

# Set REFC values < 5 to 0
xarray_forecast_merged["REFC_entireatmosphere"] = mask_invalid_refc(
    xarray_forecast_merged["REFC_entireatmosphere"]
)


# Save the dataset with compression and filters for all variables
compressor = Blosc(cname="lz4", clevel=1)
filters = [BitRound(keepbits=12)]

# No chunking since only one time step

# with ProgressBar():
# xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 31, "x": processChunk, "y": processChunk}
)
xarray_forecast_merged.to_zarr(
    forecast_process_path + "_xr_merged.zarr", mode="w", consolidated=False
)

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

# This is quite a bit simpler since there's no historic data being ingested

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
        ).rechunk((len(npCatTimes), processChunk, processChunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArrayList.append(
            daskForecastArray.rechunk(
                (len(npCatTimes), processChunk, processChunk)
            ).astype("float32")
        )

    daskVarArrays = []

    print(dask_var)


# Merge the arrays
# into a single 4D array
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
        forecast_process_dir + "/HRRR_6H.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/HRRR_6H.zarr")

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarrVars),
        len(npCatTimes),
        daskVarArrayStackDisk.shape[2],
        daskVarArrayStackDisk.shape[3],
    ),
    chunks=(len(zarrVars), len(npCatTimes), finalChunk, finalChunk),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)


# with ProgressBar():
da.rechunk(
    daskVarArrayStackDisk.round(3),
    (len(zarrVars), len(npCatTimes), finalChunk, finalChunk),
).to_zarr(zarr_array, compute=True)


if saveType == "S3":
    zarr_store.close()

# %% Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/HRRR_6H.zarr.zip",
        forecast_path + "/" + ingestVersion + "/HRRR_6H.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/HRRR_6H.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/HRRR_6H.time.pickle",
        forecast_path + "/" + ingestVersion + "/HRRR_6H.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/HRRR_6H.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/HRRR_6H.time.pickle",
        forecast_path + "/" + ingestVersion + "/HRRR_6H.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/HRRR_6H.zarr",
        forecast_path + "/" + ingestVersion + "/HRRR_6H.zarr",
        dirs_exist_ok=True,
    )


# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
