# %% Script to test FastHerbie.py to download GFS data
# Alexander Rey, September 2023

# %% Import modules
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
import zarr.storage
from herbie import FastHerbie, HerbieLatest, Path
from numcodecs import BitRound, Blosc
from rechunker import rechunk
from scipy.interpolate import make_interp_spline
from xrspatial import direction, proximity


# Scipy Interp Function
def linInterp(block, T_in, T_out):
    interp = make_interp_spline(T_in, block, 3, axis=0)
    interpOut = interp(T_out)
    return interpOut


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Define base time from the most recent run
T0 = time.time()


# To be changed in the Docker version
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2b/grib2/wgrib2/wgrib2 "
)
forecast_process_path = os.getenv(
    "forecast_process_path", default="/home/ubuntu/data/GFS_forecast"
)
hist_process_path = os.getenv(
    "hist_process_path", default="/home/ubuntu/data/GFS_historic"
)
merge_process_dir = os.getenv("merge_process_dir", default="/home/ubuntu/data/")
ncForecastWorking_path = forecast_process_path + "_Working.nc"
tmpDIR = os.getenv("tmp_dir", default="~/data")
saveType = os.getenv("save_type", default="S3")
s3_bucket = os.getenv("save_path", default="s3://piratezarr2")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


latestRun = HerbieLatest(
    model="gfs",
    n=3,
    freq="6h",
    fxx=240,
    product="pgrb2.0p25",
    verbose=False,
    priority=["aws"],
)

base_time = latestRun.date
# base_time = pd.Timestamp("2024-02-27 06:00:00Z")

print(base_time)


# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(s3_bucket + "/ForecastTar/GFS.time.pickle"):
        with s3.open(s3_bucket + "/ForecastTar/GFS.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GFS, ending")
            sys.exit()

else:
    if os.path.exists(s3_bucket + "/ForecastTar/GFS.time.pickle"):
        # Open the file in binary mode
        with open(s3_bucket + "/ForecastTar/GFS.time.pickle", "rb") as file:
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
)


print(forecast_process_path)

s3_save_path = "/ForecastProd/GFS/GFS_"

redis_host = "zarrforecastprodb.v7xcty.ng.0001.use1.cache.amazonaws.com"
redis_db = 2

redis_prefix = "GFS"
hisPeriod = 36

# Create new directory for processing if it does not exist
if not os.path.exists(merge_process_dir):
    os.makedirs(merge_process_dir)
if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)
if saveType == "Download":
    if not os.path.exists(s3_bucket):
        os.makedirs(s3_bucket)
    if not os.path.exists(s3_bucket + "/ForecastTar"):
        os.makedirs(s3_bucket + "/ForecastTar")


#####################################################################################################
# %% Download forecast data using Herbie Latest
# %% Find the latest run with 240 hours


# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
matchstring_su = ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|PRES|VIS|GUST):surface:.*hour fcst)"
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_oz = "(:TOZNE:)"
matchstring_cl = "(:TCDC:entire atmosphere:.*hour fcst)"
matchstring_ap = "(:APCP:surface:0-[1-9]*)"
matchstring_sl = "(:PRMSL:)"


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

# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

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
)

# Download UV subsets
UVmatchString = ":DUVB:surface:"
FH_forecastUV.download(UVmatchString, verbose=False)

# Create list of downloaded grib files
gribListUV = [
    str(Path(x.get_localFilePath(UVmatchString)).expand())
    for x in FH_forecastUV.file_exists
]

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
APCP_surface_tmp = xarray_forecast_merged["APCP_surface"].copy(
    data=np.diff(
        xarray_forecast_merged["APCP_surface"],
        axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
        prepend=0,
    )
)

# Convert 3-hourly to 1-hourly
APCP_surface_tmp[120:, :, :] = APCP_surface_tmp[120:, :, :] / 3

xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged["APCP_surface"].copy(
    data=APCP_surface_tmp
)


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
# len(xarray_forecast_distance_interp.time)

# Find nearest storm distance and direction for first 12 hours
for t in range(0, 160):
    distances.append(
        dask.delayed(proximity)(
            xarray_forecast_distance["APCP_surface"].isel(time=t),
            distance_metric="GREAT_CIRCLE",
            x="longitude",
            y="latitude",
        )
    )

    directions.append(
        dask.delayed(direction)(
            xarray_forecast_distance["APCP_surface"].isel(time=t),
            distance_metric="GREAT_CIRCLE",
            x="longitude",
            y="latitude",
        )
    )

# Set to zero for rest of range
# for t in range(12, 160):
#     distances.append(np.zeros(xarray_forecast_distance['APCP_surface'].shape[1:]))
#     directions.append(np.zeros(xarray_forecast_distance['APCP_surface'].shape[1:]))

distanced_stacked = dask.delayed(da.stack)(distances)
directions_stacked = dask.delayed(da.stack)(directions)

distanced_chunked = distanced_stacked.rechunk(160, 60, 60)
directions_chunked = directions_stacked.rechunk(160, 60, 60)


# with ProgressBar():
distanced_chunked.to_zarr(
    forecast_process_path + "_stormDist.zarr", overwrite=True
).compute()
directions_chunked.to_zarr(
    forecast_process_path + "_stormDir.zarr", overwrite=True
).compute()


# UV is an average from zero to 6, repeating throughout the time series.
# Correct this to 1-hour average

# Read out hours 1-120, reshape to 6 hour steps
uvProc = (
    xarray_forecast_merged["DUVB_surface"]
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
    xarray_forecast_merged["DUVB_surface"]
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
xarray_forecast_merged["DUVB_surface"] = xarray_forecast_merged["DUVB_surface"].copy(
    data=np.concatenate((uvProcHour3D, uvProcHour3DB), axis=0)
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
)

# %% Save merged and processed xarray dataset to disk using zarr with compression
# Save the dataset with compression and filters for all variables
compressor = Blosc(cname="lz4", clevel=1)
filters = [BitRound(keepbits=12)]

# No chunking since only one time step
encoding = {
    vname: {"compressor": compressor, "filters": filters} for vname in zarrVars[1:-2]
}

# xarray_forecast_merged = xarray_forecast_merged.chunk(chunks ={'time':240, 'latitude':60, 'longitude':60})
xarray_forecast_merged.to_zarr(ncForecastWorking_path + "_.zarr", mode="w")

# Clear the xaarray dataset from memory
del xarray_forecast_merged

T1 = time.time()

print(T1 - T0)
os.remove(forecast_process_path + "_wgrib_merged_UV.nc")
os.remove(forecast_process_path + "_wgrib2_merged.nc")


################################################################################################
# Historic data

# %% Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(hisPeriod, 0, -6):
    if saveType == "S3":
        # S3 Path Setup
        s3_path = (
            s3_bucket
            + "/GFS/GFS_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Try to open the zarr file to check if it has already been saved
        if s3.exists(s3_path):
            continue

    else:
        # Local Path Setup
        local_path = (
            s3_bucket
            + "/GFS/GFS_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check if local file exists
        if os.path.exists(local_path):
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
        DATES, model="gfs", fxx=fxx, product="pgrb2.0p25", verbose=False
    )

    # Download the subsets
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

    # %% Download and add UV data from the pgrib2b product
    FH_histsubUV = FastHerbie(
        DATES, model="gfs", fxx=fxx, product="pgrb2b.0p25", verbose=False
    )

    # Download the subsets
    FH_histsubUV.download(UVmatchString, verbose=False)

    # Create list of downloaded grib files
    gribListUV = [
        str(Path(x.get_localFilePath(UVmatchString)).expand())
        for x in FH_histsubUV.file_exists
    ]

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

    # %% Merge the UV data and xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib_merged = xr.open_dataset(hist_process_path + "_wgrib2_merged.nc")
    xarray_his_wgribUV_merged = xr.open_dataset(
        hist_process_path + "_wgrib2_merged_UV.nc"
    )

    xarray_hist_merged = xr.merge(
        [xarray_his_wgrib_merged, xarray_his_wgribUV_merged], compat="override"
    )

    # %% Fix things
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
            dask.delayed(proximity)(
                xarray_hist_distance["APCP_surface"].isel(time=t),
                distance_metric="GREAT_CIRCLE",
                x="longitude",
                y="latitude",
            )
        )

        directions.append(
            dask.delayed(direction)(
                xarray_hist_distance["APCP_surface"].isel(time=t),
                distance_metric="GREAT_CIRCLE",
                x="longitude",
                y="latitude",
            )
        )

    # Copy back to main array
    # with ProgressBar():
    xarray_hist_merged["Storm_Distance"] = (
        ("time", "latitude", "longitude"),
        dask.delayed(da.stack)(distances).rechunk((6, 100, 100)).compute(),
    )
    xarray_hist_merged["Storm_Direction"] = (
        ("time", "latitude", "longitude"),
        dask.delayed(da.stack)(directions).rechunk((6, 100, 100)).compute(),
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

    # %% Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    # Save as Zarr to s3 for Time Machine
    if saveType == "S3":
        zarrStore = s3fs.S3Map(root=s3_path, s3=s3, create=True)
    else:
        # Create local Zarr store
        zarrStore = zarr.storage.LocalStore(local_path)

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times
    compressor = Blosc(cname="lz4", clevel=3)
    filters = [BitRound(keepbits=9)]

    # No chunking since only one time step
    encoding = {
        vname: {"compressor": compressor, "filters": filters, "chunks": (6, 100, 100)}
        for vname in zarrVars[1:-2]
    }

    # with ProgressBar():
    xarray_hist_merged.to_zarr(
        store=zarrStore, mode="w", consolidated=True, encoding=encoding
    )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged.nc")
    os.remove(hist_process_path + "_wgrib2_merged_UV.nc")

    print((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))

#####################################################################################################
# %% Map Blocks Approach

# Create a zarr backed dask array
zarr_store = zarr.storage.LocalStore(merge_process_dir + "/GFS_UnChunk.zarr")

compressor = Blosc(cname="lz4", clevel=1)
filters = [BitRound(keepbits=12)]

# Create a Zarr array in the store with zstd compression
zarr_array = zarr.zeros(
    (21, 276, 721, 1440),
    chunks=(1, 276, 20, 20),
    store=zarr_store,
    compressor=compressor,
    dtype="float32",
)

# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    s3_bucket
    + "/GFS/GFS_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, 0, -6)
]

# Dask
# daskArrays = []
daskInterpArrays = []
daskVarArrays = []
for daskVarIDX, dask_var in enumerate(zarrVars):
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
            ncForecastWorking_path + "_.zarr", component=dask_var, inline_array=True
        )

    if dask_var == "time":
        # For some reason this is much faster when using numpy?

        daskVarArraysShape = da.reshape(daskVarArraysStack, (36, 1), merge_chunks=False)
        daskCatTimes = da.concatenate(
            (da.squeeze(daskVarArraysShape), daskForecastArray), axis=0
        ).astype("float32")

        # with ProgressBar():
        interpTimes = da.map_blocks(
            linInterp,
            daskCatTimes.rechunk((196)),
            stacked_timesUnix,
            hourly_timesUnix,
            dtype="float32",
        ).compute()

        daskArrayOut = np.tile(
            np.expand_dims(np.expand_dims(interpTimes, axis=1), axis=1), (1, 721, 1440)
        )

        da.to_zarr(
            da.from_array(np.expand_dims(daskArrayOut, axis=0)),
            zarr_array,
            region=(
                slice(daskVarIDX, daskVarIDX + 1),
                slice(0, 276),
                slice(0, 721),
                slice(0, 1440),
            ),
        )

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack, (36, 721, 1440), merge_chunks=False
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        # with ProgressBar():
        da.to_zarr(
            da.from_array(
                da.expand_dims(
                    da.map_blocks(
                        linInterp,
                        daskArrayOut[:, :, :].rechunk((196, 20, 20)).astype("float32"),
                        stacked_timesUnix,
                        hourly_timesUnix,
                        dtype="float32",
                    ).compute(),
                    axis=0,
                )
            ),
            zarr_array,
            region=(
                slice(daskVarIDX, daskVarIDX + 1),
                slice(0, 276),
                slice(0, 721),
                slice(0, 1440),
            ),
        )

    daskVarArrays = []

    print(dask_var)

# Rechunk the zarr array

encoding = {"compressor": Blosc(cname="lz4", clevel=1)}

source = zarr.open(merge_process_dir + "/GFS_UnChunk.zarr")
intermediate = merge_process_dir + "/GFS_Mid.zarr"
target = zarr.storage.ZipStore(merge_process_dir + "/GFS.zarr.zip", compression=0)
rechunked = rechunk(
    source,
    target_chunks=(21, 276, 1, 1),
    target_store=target,
    max_mem="1G",
    temp_store=intermediate,
    target_options=encoding,
)

# with ProgressBar():
result = rechunked.execute()

target.close()

if saveType == "S3":
    # Upload to S3
    s3.put_file(
        merge_process_dir + "/GFS.zarr.zip", s3_bucket + "/ForecastTar/GFS.zarr.zip"
    )

    # Write most recent forecast time
    with open(merge_process_dir + "/GFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        merge_process_dir + "/GFS.time.pickle",
        s3_bucket + "/ForecastTar/GFS.time.pickle",
    )

else:
    # Move to local
    shutil.move(
        merge_process_dir + "/GFS.zarr.zip", s3_bucket + "/ForecastTar/GFS.zarr.zip"
    )

    # Write most recent forecast time
    with open(merge_process_dir + "/GFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        merge_process_dir + "/GFS.time.pickle",
        s3_bucket + "/ForecastTar/GFS.time.pickle",
    )

    # Clean up
    shutil.rmtree(merge_process_dir)

T2 = time.time()

print(T2 - T0)
