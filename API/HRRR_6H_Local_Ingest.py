# %% HRRR 6-hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023
# Note that because the hourly script saves the 1-h forecast to S3, this script doesn't have to do this


# %% Import modules
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

import pandas as pd
import s3fs

import zarr
from numcodecs import Blosc, BitRound

import dask.array as da

import numpy as np
import xarray as xr
import time

import subprocess
import os
import shutil
import sys
import pickle


import dask

import warnings

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
# To be changed in the Docker version
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/grib2/wgrib2/wgrib2 "
)
forecast_process_path = os.getenv(
    "forecast_process_path", default="/home/ubuntu/data/HRRR_forecast"
)
merge_process_dir = os.getenv("merge_process_dir", default="/home/ubuntu/data/")
tmpDIR = os.getenv("tmp_dir", default="~/data")
saveType = os.getenv("save_type", default="S3")
s3_bucket = os.getenv("save_path", default="s3://piratezarr2")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(
    key=aws_access_key_id, secret=aws_secret_access_key
)


# s3_bucket = 's3://pirate-s3-azb--use1-az4--x-s3'
s3_save_path = "/ForecastProd/HRRR/HRRR_"
hisPeriod = 36


# %% Define base time from the most recent run
# base_time = pd.Timestamp("2023-07-01 00:00")
T0 = time.time()

latestRun = Herbie_latest(
    model="hrrr", n=3, freq="6H", fxx=[48], product="sfc", verbose=False, priority="aws"
)

base_time = latestRun.date

print(base_time)
# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(s3_bucket + "/ForecastTar/HRRR_6H.time.pickle"):
        with s3.open(s3_bucket + "/ForecastTar/HRRR_6H.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to HRRR_6H, ending")
            sys.exit()

else:
    if os.path.exists(s3_bucket + "/ForecastTar/HRRR_6H.time.pickle"):
        # Open the file in binary mode
        with open(s3_bucket + "/ForecastTar/HRRR_6H.time.pickle", "rb") as file:
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
)


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
matchstring_8m = ":(MASSDEN:8 m above ground:)"
matchstring_su = ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST):surface:.*hour fcst)"
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_cl = "(:TCDC:entire atmosphere:.*hour fcst)"
matchstring_ap = "(:APCP:surface:0-[1-9]*)"
matchstring_sl = "(:MSLMA:)"

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
    pd.date_range(start=base_time, periods=1, freq="1H"),
    model="hrrr",
    fxx=hrrr_range1,
    product="sfc",
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

# %% Create XArray
# Read the netcdf file using xarray
xarray_forecast_merged = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")

# print(xarray_forecast_merged.variables)

assert len(xarray_forecast_merged.time) == len(
    hrrr_range1
), "Incorrect number of timesteps! Exiting"

# %% Fix things
# Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged["APCP_surface"].copy(
    data=np.diff(
        xarray_forecast_merged["APCP_surface"],
        axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
        prepend=0,
    )
)

# Save the dataset with compression and filters for all variables
compressor = Blosc(cname="lz4", clevel=1)
filters = [BitRound(keepbits=12)]

# No chunking since only one time step
encoding = {
    vname: {"compressor": compressor, "filters": filters} for vname in zarrVars[1:]
}

# with ProgressBar():
# xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 31, "x": 90, "y": 90}
)
xarray_forecast_merged.to_zarr(
    forecast_process_path + "_xr_merged.zarr", encoding=encoding, mode="w"
)

# Remove wgrib2 temp files
os.remove(forecast_process_path + "_wgrib2_merged.grib2")
os.remove(forecast_process_path + "_wgrib2_merged.regrid")
os.remove(forecast_process_path + "_wgrib2_merged.nc")

# %% Format as dask and save as zarr
#####################################################################################################
# Convert from xarray to dask dataframe
daskArrays = []

for dask_var in zarrVars:
    # Concatenate the dask arrays through time
    if dask_var == "time":
        daskArrays.append(
            da.from_array(
                np.tile(
                    np.expand_dims(
                        np.expand_dims(
                            dask.delayed(da.from_zarr)(
                                forecast_process_path + "_xr_merged.zarr",
                                component=dask_var,
                                inline_array=True,
                            )
                            .compute()
                            .astype("float32")
                            .compute(),
                            axis=1,
                        ),
                        axis=1,
                    ),
                    (1, 1059, 1799),
                )
            )
        )
    else:
        # Concatenate the dask arrays through time
        daskArrays.append(
            dask.delayed(da.from_zarr)(
                forecast_process_path + "_xr_merged.zarr",
                component=dask_var,
                inline_array=True,
            )
        )

# Stack the DataArrays into a Dask array
stacked_dask_array = dask.delayed(da.stack)(daskArrays, axis=0)

# Chunk the Dask for fast reads at one point
# Variable, time, lat, lon
chunked_dask_array = stacked_dask_array.rechunk((17, 31, 3, 3)).astype("float32")

# Setup S3 for dask array save
# s3_path = s3_bucket + s3_save_path + base_time.strftime('%Y%m%dT%H%M%SZ') + '_TEST'
# s3store = s3fs.S3Map(root=s3_path, s3=s3, create=True)

# Define the compressor
# filters = [BitRound(keepbits=9)] # Only keep ~ 3 significant digits
compressor = Blosc(cname="zstd", clevel=3)  # Use zstd compression


# Save to Zip
zip_store = zarr.ZipStore(merge_process_dir + "/HRRR_6H.zarr.zip", compression=0)
chunked_dask_array.to_zarr(zip_store, compressor=compressor, overwrite=True).compute()
zip_store.close()

# Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        merge_process_dir + "/HRRR_6H.zarr.zip",
        s3_bucket + "/ForecastTar/HRRR_6H.zarr.zip",
    )

    # Write most recent forecast time
    with open(merge_process_dir + "/HRRR_6H.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        merge_process_dir + "/HRRR_6H.time.pickle",
        s3_bucket + "/ForecastTar/HRRR_6H.time.pickle",
    )

else:
    # Move to local
    shutil.move(
        merge_process_dir + "/HRRR_6H.zarr.zip",
        s3_bucket + "/ForecastTar/HRRR_6H.zarr.zip",
    )
    # Write most recent forecast time
    with open(merge_process_dir + "/HRRR_6H.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        merge_process_dir + "/HRRR_6H.time.pickle",
        s3_bucket + "/ForecastTar/HRRR_6H.time.pickle",
    )
    # Clean up
    shutil.rmtree(merge_process_dir)

T1 = time.time()
print(T1 - T0)
