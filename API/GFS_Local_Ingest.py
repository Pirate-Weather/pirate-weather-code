# %% Script to test FastHerbie.py to download GFS data
# Alexander Rey, September 2023

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
from xrspatial import direction, proximity

from dask.diagnostics import ProgressBar

from ingest_utils import mask_invalid_data, interp_time_block, validate_grib_stats


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
ingestVersion = "v27"

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/GFS"
)
forecast_process_path = forecast_process_dir + "/GFS_Process"
hist_process_path = forecast_process_dir + "/GFS_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/GFS")
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/History/GFS")


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
    if not os.path.exists(forecast_path + "/" + ingestVersion):
        os.makedirs(forecast_path + "/" + ingestVersion)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)


T0 = time.time()

latestRun = HerbieLatest(
    model="gfs",
    n=3,
    freq="6h",
    fxx=240,
    product="pgrb2.0p25",
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
    if s3.exists(forecast_path + "/" + ingestVersion + "/GFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingestVersion + "/GFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingestVersion + "/GFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingestVersion + "/GFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GFS, ending")
            sys.exit()

zarrVars = (
    "time",
    "VIS_surface",
    "GUST_surface",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "APTMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "TOZNE_entireatmosphere_consideredasasinglelayer_",
    "TCDC_entireatmosphere",
    "DUVB_surface",
    "Storm_Distance",
    "Storm_Direction",
    "REFC_entireatmosphere",
    "DSWRF_surface",
    "CAPE_surface",
)

hisPeriod = 48

#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours


# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
matchstring_su = (
    ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|PRES|VIS|GUST|CAPE):surface:.*hour fcst)"
)
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_oz = "(:TOZNE:)"
matchstring_cl = "(:(TCDC|REFC):entire atmosphere:.*hour fcst)"
matchstring_ap = "(:APCP:surface:0-[1-9]*)"
matchstring_sl = "(:(PRMSL|DSWRF):)"


# Merge matchstrings for download
matchStrings = (
    matchstring_2m
    + "|"
    + matchstring_su
    + "|"
    + matchstring_10m
    + "|"
    + matchstring_oz
    + "|"
    + matchstring_cl
    + "|"
    + matchstring_ap
    + "|"
    + matchstring_sl
)

# INV TESTING
# import datetime as datetime
# FastHerbie(pd.date_range(start=pd.to_datetime(datetime.datetime(2023,12,1,0,0,0)), periods=1, freq='1H'),
#                                  model="gfs", fxx=[237],
#                                  product="pgrb2.0p25", verbose=True, priority='aws').inventory().search_this.values

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
gfs_range1 = range(1, 121)
gfs_range2 = range(123, 241, 3)
gfsFileRange = [*gfs_range1, *gfs_range2]

# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="gfs",
    fxx=gfsFileRange,
    product="pgrb2.0p25",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

# Download the subsets
FH_forecastsub.download(matchStrings, verbose=False)

# Check for download length
if len(FH_forecastsub.file_exists) != len(gfsFileRange):
    print(
        "Download failed, expected "
        + str(len(gfsFileRange))
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

# Validate the grib files
validate_grib_stats(gribCheck)
print("Grib validation complete, no errors found.")


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


# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

# %% Download and add UV data from the pgrib2b product
FH_forecastUV = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="gfs",
    fxx=gfsFileRange,
    product="pgrb2b.0p25",
    verbose=False,
    priority="aws",
    save_dir=tmpDIR,
)

# Download UV subsets
UVmatchString = ":DUVB:surface:"
FH_forecastUV.download(UVmatchString, verbose=False)

# Check for download length
if len(FH_forecastUV.file_exists) != len(gfsFileRange):
    print(
        "Download failed, expected 160 files but got "
        + str(len(FH_forecastUV.file_exists))
    )
    sys.exit(1)


# Create list of downloaded grib files
gribListUV = [
    str(Path(x.get_localFilePath(UVmatchString)).expand())
    for x in FH_forecastUV.file_exists
]

# Perform a check if any data seems to be invalid
cmd = "cat " + " ".join(gribListUV) + " | " + f"{wgrib2_path}" + " - " + " -s -stats"

gribCheck = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

validate_grib_stats(gribCheck)
print("Grib files passed validation, proceeding with processing")

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(gribListUV)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_wgrib_merged_UV.nc"
)

# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

# %% Merge the UV data and xarrays
# Read the netcdf file using xarray
xarray_wgrib_merged = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")
xarray_wgribUV_merged = xr.open_mfdataset(forecast_process_path + "_wgrib_merged_UV.nc")

# Merge the xarray objects
xarray_forecast_merged = xr.merge(
    [xarray_wgrib_merged, xarray_wgribUV_merged], compat="override"
)

assert len(xarray_forecast_merged.time) == len(gfsFileRange), (
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
            freq="h",
        ),
        xarray_forecast_merged.time.values,
    )
)
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

# Convert 3-hourly to 1-hourly
APCP_surface_tmp[120:, :, :] = APCP_surface_tmp[120:, :, :] / 3

xarray_forecast_merged["APCP_surface"].data = APCP_surface_tmp


# Create a new xarray for storm distance processing using dask
xarray_forecast_distance = xr.Dataset()
xarray_forecast_distance["APCP_surface"] = xarray_forecast_merged["APCP_surface"].copy()

xarray_forecast_distance = xarray_forecast_distance.assign_coords(
    {
        "time": xarray_forecast_merged.time.data,
        "latitude": xarray_forecast_merged.latitude.data,
        "longitude": ((xarray_forecast_merged.longitude + 180) % 360) - 180,
    }
)

# Set threshold precp at 2 mm/h
xarray_forecast_distance["APCP_surface"] = xarray_forecast_distance[
    "APCP_surface"
].where(xarray_forecast_distance["APCP_surface"] > 0.2, 0)

distances = []
directions = []


# Find nearest storm distance and direction for first 12 hours
for t in range(0, 160):
    distances.append(
        proximity(
            xarray_forecast_distance["APCP_surface"].isel(time=t),
            distance_metric="GREAT_CIRCLE",
            x="longitude",
            y="latitude",
            max_distance=None,
        )
    )

    directions.append(
        direction(
            xarray_forecast_distance["APCP_surface"].isel(time=t),
            distance_metric="GREAT_CIRCLE",
            x="longitude",
            y="latitude",
            max_distance=None,
        )
    )


distanced_stacked = da.stack(distances)
directions_stacked = da.stack(directions)

distanced_chunked = distanced_stacked.rechunk(160, processChunk, processChunk)
directions_chunked = directions_stacked.rechunk(160, processChunk, processChunk)


with ProgressBar():
    distanced_chunked.to_zarr(
        forecast_process_path + "_stormDist.zarr", overwrite=True, compute=True
    )
    directions_chunked.to_zarr(
        forecast_process_path + "_stormDir.zarr", overwrite=True, compute=True
    )


# UV is an average from zero to 6, repeating throughout the time series.
# Correct this to 1-hour average
# Solar rad follows the same pattern, so we can use the same appraoch.
accumVars = ["DUVB_surface", "DSWRF_surface"]
for accumVar in accumVars:
    # Read out hours 1-120, reshape to 6 hour steps
    uvProc = (
        xarray_forecast_merged[accumVar]
        .isel(time=slice(0, 120))
        .values.reshape(20, 6, 721, 1440, order="C")
    )

    n = np.arange(1, 7)
    n = n[np.newaxis, :, np.newaxis, np.newaxis]

    # Save first step to concatonate later
    first_step = uvProc[:, 0, :, :]
    first_step = first_step[:, np.newaxis, :, :]

    # Create numpy array of processed UV
    uvProcHour = np.concatenate(
        (first_step, np.diff(uvProc, axis=1) * n[:, 1:, :, :] + uvProc[:, 0:5, :, :]),
        axis=1,
    )

    # Reshape back to 3D
    uvProcHour3D = uvProcHour.reshape(120, 721, 1440, order="C")

    # Read out hours 123, reshape to 6 hour steps
    uvProc = (
        xarray_forecast_merged[accumVar]
        .isel(time=slice(120, 160))
        .values.reshape(20, 2, 721, 1440, order="C")
    )

    n = np.arange(1, 3)
    n = n[np.newaxis, :, np.newaxis, np.newaxis]

    # Save first step to concatonate later
    first_step = uvProc[:, 0, :, :]
    first_step = first_step[:, np.newaxis, :, :]

    # Create numpy array of processed UV
    uvProcHour = np.concatenate(
        (first_step, np.diff(uvProc, axis=1) * n[:, 1:, :, :] + uvProc[:, 0:1, :, :]),
        axis=1,
    )

    # Reshape back to 3D
    uvProcHour3DB = uvProcHour.reshape(40, 721, 1440, order="C")

    ### Note- to get index, do this:
    #             // UVB to etyhemally UV factor 18.9 https://link.springer.com/article/10.1039/b312985c
    #             // 0.025 m2/W to get the uv index
    # ['DUVB_surface'] * 0.025 * 18.9

    # Combine and merge back into xarray dataset
    xarray_forecast_merged[accumVar] = xarray_forecast_merged[accumVar].copy(
        data=np.concatenate((uvProcHour3D, uvProcHour3DB), axis=0)
    )


# %% Save merged and processed xarray dataset to disk using zarr with compression
# Save the dataset with compression and filters for all variables
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 240, "latitude": processChunk, "longitude": processChunk}
)
xarray_forecast_merged.to_zarr(
    forecast_process_path + "_.zarr", mode="w", consolidated=False, compute=True
)

# %% Delete to free memory
del (
    uvProc,
    uvProcHour,
    uvProcHour3D,
    uvProcHour3DB,
    n,
    first_step,
    xarray_wgrib_merged,
    xarray_wgribUV_merged,
    directions,
    distances,
    directions_chunked,
    distanced_chunked,
    distanced_stacked,
    directions_stacked,
    xarray_forecast_distance,
    APCP_surface_tmp,
    xarray_forecast_merged,
)
T1 = time.time()

print(T1 - T0)
os.remove(forecast_process_path + "_wgrib_merged_UV.nc")
os.remove(forecast_process_path + "_wgrib2_merged.nc")

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(hisPeriod, 0, -6):
    if saveType == "S3":
        s3_path = (
            historic_path
            + "/GFS_Hist_v2"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a done file in S3
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
            + "/GFS_Hist_v2"
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
        freq="6h",
    )
    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach
    fxx = range(1, 7)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="gfs",
        fxx=fxx,
        product="pgrb2.0p25",
        verbose=False,
        save_dir=tmpDIR,
    )

    # Download the subsets
    FH_histsub.download(matchStrings, verbose=False)

    # Check for download length
    if len(FH_histsub.file_exists) != 6:
        print(
            "Download failed, expected 6 files but got "
            + str(len(FH_histsub.file_exists))
        )
        sys.exit(1)

    # Create list of downloaded grib files
    gribList = [
        str(Path(x.get_localFilePath(matchStrings)).expand())
        for x in FH_histsub.file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(gribList) + " | " + f"{wgrib2_path}" + " - " + " -s -stats"

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
        + " -netcdf "
        + hist_process_path
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2
    spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if spOUT.returncode != 0:
        print(spOUT.stderr)
        sys.exit()

    # Download and add UV data from the pgrib2b product
    FH_histsubUV = FastHerbie(
        DATES,
        model="gfs",
        fxx=fxx,
        product="pgrb2b.0p25",
        verbose=False,
        save_dir=tmpDIR,
    )

    # Download the subsets
    FH_histsubUV.download(UVmatchString, verbose=False)

    # Check for download length
    if len(FH_histsubUV.file_exists) != 6:
        print(
            "Download failed, expected 6 files but got "
            + str(len(FH_histsubUV.file_exists))
        )
        sys.exit(1)

    # Create list of downloaded grib files
    gribListUV = [
        str(Path(x.get_localFilePath(UVmatchString)).expand())
        for x in FH_histsubUV.file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = (
        "cat " + " ".join(gribListUV) + " | " + f"{wgrib2_path}" + " - " + " -s -stats"
    )

    gribCheck = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

    validate_grib_stats(gribCheck)
    print("Grib files passed validation, proceeding with processing")

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(gribListUV)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + hist_process_path
        + "_wgrib2_merged_UV.nc"
    )

    # Run wgrib2
    spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if spOUT.returncode != 0:
        print(spOUT.stderr)
        sys.exit()

    # Merge the UV data and xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib_merged = xr.open_dataset(hist_process_path + "_wgrib2_merged.nc")
    xarray_his_wgribUV_merged = xr.open_dataset(
        hist_process_path + "_wgrib2_merged_UV.nc"
    )

    xarray_hist_merged = xr.merge(
        [xarray_his_wgrib_merged, xarray_his_wgribUV_merged], compat="override"
    )

    # Fix things
    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time, every 6 hours
    apcpProc = xarray_hist_merged["APCP_surface"].values

    apcpProcHour = np.diff(apcpProc, axis=0, prepend=0)

    xarray_hist_merged["APCP_surface"] = xarray_hist_merged["APCP_surface"].copy(
        data=apcpProcHour
    )

    # Storm distance and direction
    xarray_hist_distance = xr.Dataset()
    xarray_hist_distance["APCP_surface"] = xarray_hist_merged["APCP_surface"].copy()

    xarray_hist_distance = xarray_hist_distance.assign_coords(
        {
            "time": xarray_hist_merged.time.data,
            "latitude": xarray_hist_merged.latitude.data,
            "longitude": ((xarray_hist_merged.longitude + 180) % 360) - 180,
        }
    )

    # Set threshold precp at 2 mm/h
    xarray_hist_distance["APCP_surface"] = xarray_hist_distance["APCP_surface"].where(
        xarray_hist_distance["APCP_surface"] > 0.2, 0
    )

    distances = []
    directions = []

    # Find nearest storm distance and direction for first 12 hours
    for t in range(0, 6):
        distances.append(
            proximity(
                xarray_hist_distance["APCP_surface"].isel(time=t),
                distance_metric="GREAT_CIRCLE",
                x="longitude",
                y="latitude",
                max_distance=None,
            )
        )

        directions.append(
            direction(
                xarray_hist_distance["APCP_surface"].isel(time=t),
                distance_metric="GREAT_CIRCLE",
                x="longitude",
                y="latitude",
                max_distance=None,
            )
        )

    # Copy back to main array
    # with ProgressBar():
    xarray_hist_merged["Storm_Distance"] = (
        ("time", "latitude", "longitude"),
        da.stack(distances).rechunk((6, 100, 100)).compute(),
    )
    xarray_hist_merged["Storm_Direction"] = (
        ("time", "latitude", "longitude"),
        da.stack(directions).rechunk((6, 100, 100)).compute(),
    )

    # UV is an average from zero to 6, repeating throughout the time series.
    # Correct this to 1-hour average

    # Read out hours 1-120, reshape to 6 hour steps
    uvProc = xarray_hist_merged["DUVB_surface"].values

    n = np.arange(1, 7)
    n = n[:, np.newaxis, np.newaxis]

    # Save first step to concatonate later
    first_step = uvProc[0, :, :]
    first_step = first_step[np.newaxis, :, :]

    # Create numpy array of processed UV
    uvProcHour = np.concatenate(
        (first_step, np.diff(uvProc, axis=0) * n[1:, :, :] + uvProc[0:5, :, :]), axis=0
    )

    # Remove zero values
    uvProcHour[uvProcHour < 0] = 0

    # (average_series[1:] - average_series[0:-1]) * np.array([2, 3, 4, 5]) +  average_series[0:-1]
    # From https://math.stackexchange.com/questions/106700/incremental-averaging

    xarray_hist_merged["DUVB_surface"] = xarray_hist_merged["DUVB_surface"].copy(
        data=uvProcHour
    )

    # Clear memory
    del (
        uvProc,
        uvProcHour,
        apcpProc,
        apcpProcHour,
        xarray_his_wgrib_merged,
        xarray_his_wgribUV_merged,
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

    # No chunking since only one time step
    encoding = {
        vname: {"chunks": (6, processChunk, processChunk)} for vname in zarrVars[1:-2]
    }

    # with ProgressBar():
    xarray_hist_merged.to_zarr(
        store=zarrStore, mode="w", consolidated=False, encoding=encoding
    )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged.nc")
    os.remove(hist_process_path + "_wgrib2_merged_UV.nc")

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
    + "/GFS_Hist_v2"
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

    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

    if dask_var == "Storm_Distance":
        daskForecastArray = da.from_zarr(forecast_process_path + "_stormDist.zarr")
    elif dask_var == "Storm_Direction":
        daskForecastArray = da.from_zarr(forecast_process_path + "_stormDir.zarr")
    else:
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
        ).rechunk((len(stacked_timesUnix), processChunk, processChunk))

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
            .rechunk((len(stacked_timesUnix), processChunk, processChunk))
            .astype("float32")
        )

    daskVarArrays = []

    print(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Mask out invalid data
# Ignore storm distance, since it can reach very high values that are still correct
daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge, ignore_axis=[19])

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
        forecast_process_dir + "/GFS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/GFS.zarr")

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
# 4 (TMP)
# 8 (UGRD)
# 9 (VGRD)
# 10 (PRATE)
# 11 (PACCUM)
# 12:15 (PTYPE)
# 21 (REFC)

# Loop through variables, creating a new one with a name and 36 x 100 x 100 chunks
# Save -12:24 hours, aka steps 24:60
# Create a Zarr array in the store with zstd compression

if saveType == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/GFS_Maps.zarr.zip", mode="a"
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(forecast_process_dir + "/GFS_Maps.zarr")

for z in [0, 4, 8, 9, 10, 11, 12, 13, 14, 15, 21]:
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

    with ProgressBar():
        da.rechunk(daskVarArrayStackDisk[z, 36:72, :, :], (36, 100, 100)).to_zarr(
            zarr_array, overwrite=True, compute=True
        )

    print(zarrVars[z])


if saveType == "S3":
    zarr_store_maps.close()

# %% Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/GFS.zarr.zip",
        forecast_path + "/" + ingestVersion + "/GFS.zarr.zip",
    )
    s3.put_file(
        forecast_process_dir + "/GFS_Maps.zarr.zip",
        forecast_path + "/" + ingestVersion + "/GFS_Maps.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/GFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/GFS.time.pickle",
        forecast_path + "/" + ingestVersion + "/GFS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/GFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/GFS.time.pickle",
        forecast_path + "/" + ingestVersion + "/GFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GFS.zarr",
        forecast_path + "/" + ingestVersion + "/GFS.zarr",
        dirs_exist_ok=True,
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GFS_Maps.zarr",
        forecast_path + "/" + ingestVersion + "/GFS_Maps.zarr",
        dirs_exist_ok=True,
    )
# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
