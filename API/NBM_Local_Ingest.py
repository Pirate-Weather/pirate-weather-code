# %% NBM Hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023

# %% Import modules
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest
import pandas as pd
import s3fs

import zarr
import dask

from numcodecs import Blosc, BitRound

import dask.array as da
from rechunker import rechunk

import numpy as np
import xarray as xr
import time

import subprocess

import os
import shutil
import sys
import pickle


import netCDF4 as nc

from itertools import chain

from scipy.interpolate import make_interp_spline

import warnings


# Scipy Interp Function
def linInterp1D(block, T_in, T_out):
    interp = make_interp_spline(T_in, block, 3, axis=0)
    interp.extrapolate = False
    interpOut = interp(T_out)
    return interpOut


def linInterp3D(block, T_in, T_out):
    # Filter large values
    bMask = block[:, 1, 1] < 1e10

    interp = make_interp_spline(T_in[bMask], block[bMask, :, :], 3, axis=0)
    interp.extrapolate = False
    interpOut = interp(T_out)
    return interpOut


def getGribList(FH_forecastsub, matchStrings):
    try:
        gribList = [
            str(Path(x.get_localFilePath(matchStrings)).expand())
            for x in FH_forecastsub.file_exists
        ]
    except:
        print("Download Failure 1, wait 20 seconds and retry")
        time.sleep(20)
        FH_forecastsub.download(matchStrings, verbose=False)
        try:
            gribList = [
                str(Path(x.get_localFilePath(matchStrings)).expand())
                for x in FH_forecastsub.file_exists
            ]
        except:
            print("Download Failure 2, wait 20 seconds and retry")
            time.sleep(20)
            FH_forecastsub.download(matchStrings, verbose=False)
            try:
                gribList = [
                    str(Path(x.get_localFilePath(matchStrings)).expand())
                    for x in FH_forecastsub.file_exists
                ]
            except:
                print("Download Failure 3, wait 20 seconds and retry")
                time.sleep(20)
                FH_forecastsub.download(matchStrings, verbose=False)
                try:
                    gribList = [
                        str(Path(x.get_localFilePath(matchStrings)).expand())
                        for x in FH_forecastsub.file_exists
                    ]
                except:
                    print("Download Failure 4, wait 20 seconds and retry")
                    time.sleep(20)
                    FH_forecastsub.download(matchStrings, verbose=False)
                    try:
                        gribList = [
                            str(Path(x.get_localFilePath(matchStrings)).expand())
                            for x in FH_forecastsub.file_exists
                        ]
                    except:
                        print("Download Failure 5, wait 20 seconds and retry")
                        time.sleep(20)
                        FH_forecastsub.download(matchStrings, verbose=False)
                        try:
                            gribList = [
                                str(Path(x.get_localFilePath(matchStrings)).expand())
                                for x in FH_forecastsub.file_exists
                            ]
                        except:
                            print("Download Failure 6, Fail")
                            exit(1)
    return gribList


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
# To be changed in the Docker version
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/grib2/wgrib2/wgrib2 "
)
forecast_process_path = os.getenv(
    "forecast_process_path", default="/home/ubuntu/data/NBM_forecast"
)
hist_process_path = os.getenv("hist_process_path", default="/home/ubuntu/data/NBM_hist")
merge_process_dir = os.getenv("merge_process_dir", default="/home/ubuntu/data/")
tmpDIR = os.getenv("tmp_dir", default="~/data")
saveType = os.getenv("save_type", default="S3")
s3_bucket = os.getenv("save_path", default="s3://piratezarr2")


# s3_bucket = 's3://pirate-s3-azb--use1-az4--x-s3'
s3_save_path = "/ForecastProd/NBM/NBM_"

hisPeriod = 36

s3 = s3fs.S3FileSystem(
    key="AKIA2HTALZ5LWRCTHC5F", secret="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"
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

# %% Define base time from the most recent run
T0 = time.time()

latestRun = Herbie_latest(
    model="nbm",
    n=5,
    freq="1H",
    fxx=[190, 191, 192, 193, 194, 195],
    product="co",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

base_time = latestRun.date


# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(s3_bucket + "/ForecastTar/NBM.time.pickle"):
        with s3.open(s3_bucket + "/ForecastTar/NBM.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM, ending")
            sys.exit()

else:
    if os.path.exists(s3_bucket + "/ForecastTar/NBM.time.pickle"):
        # Open the file in binary mode
        with open(s3_bucket + "/ForecastTar/NBM.time.pickle", "rb") as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM, ending")
            sys.exit()

# base_time = pd.Timestamp("2024-03-05 16:00")
# base_time = base_time - pd.Timedelta(1,'h')
print(base_time)

zarrVars = (
    "time",
    "GUST_10maboveground",
    "TMP_2maboveground",
    "APTMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "WIND_10maboveground",
    "WDIR_10maboveground",
    "APCP_surface",
    "TCDC_surface",
    "VIS_surface",
    "PWTHER_surfaceMreserved",
    "PPROB",
    "PACCUM",
    "PTYPE_prob_GE_1_LT_2_prob_fcst_1_1_surface",
    "PTYPE_prob_GE_3_LT_4_prob_fcst_1_1_surface",
    "PTYPE_prob_GE_5_LT_7_prob_fcst_1_1_surface",
    "PTYPE_prob_GE_8_LT_9_prob_fcst_1_1_surface",
)


#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours
# Set download rannges
if base_time.hour % 6 == 0:
    nbm_range1 = range(1, 37, 1)
    nbm_range2 = range(42, 193, 3)
elif base_time.hour % 6 == 1:
    nbm_range1 = range(1, 36, 1)
    nbm_range2 = range(41, 192, 3)
elif base_time.hour % 6 == 2:
    nbm_range1 = range(1, 35, 1)
    nbm_range2 = range(40, 191, 3)
elif base_time.hour % 6 == 3:
    nbm_range1 = range(1, 34, 1)
    nbm_range2 = range(39, 190, 3)
elif base_time.hour % 6 == 4:
    nbm_range1 = range(1, 33, 1)
    nbm_range2 = range(38, 189, 3)
elif base_time.hour % 6 == 5:
    nbm_range1 = range(1, 32, 1)
    nbm_range2 = range(37, 188, 3)

nbm_range = list(chain(nbm_range1, nbm_range2))

# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:.*fcst:nan)"
matchstring_su = ":((PTYPE):surface:.*)"
matchstring_10m = "(:(GUST|WIND|WDIR):10 m above ground:.*fcst:nan)"
matchstring_pr = "(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:nan)"
matchstring_re = (
    ":((TCDC|VIS):surface:.*fcst:nan)"  # This gets the correct surface param
)
matchstring_pw = ":(PWTHER:)"  # This gets the correct surface param

# Merge matchstrings for download
matchStrings = (
    matchstring_2m
    + "|"
    + matchstring_su
    + "|"
    + matchstring_10m
    + "|"
    + matchstring_pr
    + "|"
    + matchstring_re
    + "|"
    + matchstring_pw
)


# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1H"),
    model="nbm",
    fxx=nbm_range,
    product="co",
    verbose=False,
    priority=["aws"],
    max_threads=1,
    save_dir=tmpDIR,
)

# for i in nbm_range:
#     response = requests.get("https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.20240515/13/core/blend.t13z.core.f{:03d}.co.grib2.idx".format(i))
#     print(response.status_code)

# Download the subsets
FH_forecastsub.download(matchStrings, verbose=False)

# Create list of downloaded grib files
gribList = getGribList(FH_forecastsub, matchStrings)

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
os.remove(forecast_process_path + "_wgrib2_merged_order.grib")


##### PPROB
# Set download ranges. Note the 6 hourly second step instead of 3, since that's where the prob data is saved
if base_time.hour % 6 == 0:
    nbm_range1 = range(1, 37, 1)
    nbm_range2 = range(42, 193, 6)
elif base_time.hour % 6 == 1:
    nbm_range1 = range(1, 36, 1)
    nbm_range2 = range(41, 192, 6)
elif base_time.hour % 6 == 2:
    nbm_range1 = range(1, 35, 1)
    nbm_range2 = range(40, 191, 6)
elif base_time.hour % 6 == 3:
    nbm_range1 = range(1, 34, 1)
    nbm_range2 = range(39, 190, 6)
elif base_time.hour % 6 == 4:
    nbm_range1 = range(1, 33, 1)
    nbm_range2 = range(38, 189, 6)
elif base_time.hour % 6 == 5:
    nbm_range1 = range(1, 32, 1)
    nbm_range2 = range(37, 188, 6)

# Download PPROB as 1-Hour Prob
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1H"),
    model="nbm",
    fxx=nbm_range1,
    product="co",
    verbose=False,
    priority=["aws"],
)

matchstring_po = "(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|[0-9]*0-[0-9]{1,2}1|[0-9]*1-[0-9]{1,2}2|[0-9]*2-[0-9]{1,2}3|[0-9]*3-[0-9]{1,2}4|[0-9]*4-[0-9]{1,2}5|[0-9]*5-[0-9]{1,2}6|[0-9]*6-[0-9]{1,2}7|[0-9]*7-[0-9]{1,2}8|[0-9]*8-[0-9]{1,2}9|[0-9]*9-[0-9]{1,2}0).*fcst:prob.*)"
# Download the subsets
FH_forecastsub.download(matchstring_po, verbose=False)

# Create list of downloaded grib files
gribList1 = getGribList(FH_forecastsub, matchstring_po)
#####
# Download PPROB as 6-Hour Accum for hours 036-190

# Create FastHerbie object
FH_forecastsub2 = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1H"),
    model="nbm",
    fxx=nbm_range2,
    product="co",
    verbose=False,
    priority=["aws"],
)

# Match 6-hour probs
matchstring_po2 = ":APCP:surface:(0-6|[0-9]*0-[0-9]{1,2}6|[0-9]*1-[0-9]{1,2}7|[0-9]*2-[0-9]{1,2}8|[0-9]*3-[0-9]{1,2}9|[0-9]*4-[0-9]{1,2}0|[0-9]*5-[0-9]{1,2}1|[0-9]*6-[0-9]{1,2}2|[0-9]*7-[0-9]{1,2}3|[0-9]*8-[0-9]{1,2}4|[0-9]*9-[0-9]{1,2}5).*fcst:prob"
# Download the subsets
FH_forecastsub2.download(matchstring_po2, verbose=False)

# Create list of downloaded grib files
gribList2 = getGribList(FH_forecastsub2, matchstring_po2)

gribList = gribList1 + gribList2

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(gribList)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -grib "
    + forecast_process_path
    + "_prob_wgrib2_merged.grib2"
)
# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

# Use wgrib2 to change the order
cmd2 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_prob_wgrib2_merged.grib2 "
    + " -ijsmall_grib "
    + " 1:2345 1:1597 "
    + forecast_process_path
    + "_prob_wgrib2_merged_order.grib"
)
spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")
os.remove(forecast_process_path + "_prob_wgrib2_merged.grib2")

# Convert to NetCDF
cmd4 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_prob_wgrib2_merged_order.grib "
    + " -set_ext_name 1 -netcdf "
    + forecast_process_path
    + "_prob_wgrib2_merged.nc"
)

# Run wgrib2 to save as  NetCDF
spOUT4 = subprocess.run(cmd4, shell=True, capture_output=True, encoding="utf-8")
os.remove(forecast_process_path + "_prob_wgrib2_merged_order.grib")


# Download PACCUM as 1-Hour and 6-hour Accum
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1H"),
    model="nbm",
    fxx=nbm_range1,
    product="co",
    verbose=True,
    priority=["aws"],
)

matchstring_pa = "(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:nan)"
# Download the subsets
FH_forecastsub.download(matchstring_pa, verbose=True)

# Create list of downloaded grib files
gribList1 = getGribList(FH_forecastsub, matchstring_pa)

#####
# Download PPROB as 6-Hour Accum for hours 036-190

# Create FastHerbie object
FH_forecastsub2 = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1H"),
    model="nbm",
    fxx=nbm_range2,
    product="co",
    verbose=True,
    priority=["aws"],
)

# Match 6-hour probs
matchstring_pa2 = "(:APCP:surface:(0-6|\d*0-\d{1,2}6|\d*1-\d{1,2}7|\d*2-\d{1,2}8|\d*3-\d{1,2}9|\d*4-\d{1,2}0|\d*5-\d{1,2}1|\d*6-\d{1,2}2|\d*7-\d{1,2}3|\d*8-\d{1,2}4|\d*9-\d{1,2}5).*fcst:nan)"
# Download the subsets
FH_forecastsub2.download(matchstring_pa2, verbose=True)

# Create list of downloaded grib files
gribList2 = getGribList(FH_forecastsub2, matchstring_pa2)
gribList = gribList1 + gribList2


# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(gribList)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -grib "
    + forecast_process_path
    + "_accum_wgrib2_merged.grib2"
)
# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

# Use wgrib2 to change the order
cmd2 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_accum_wgrib2_merged.grib2 "
    + " -ijsmall_grib "
    + " 1:2345 1:1597 "
    + forecast_process_path
    + "_accum_wgrib2_merged_order.grib"
)
spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")
os.remove(forecast_process_path + "_accum_wgrib2_merged.grib2")

# Convert to NetCDF
cmd4 = (
    f"{wgrib2_path}"
    + "  "
    + forecast_process_path
    + "_accum_wgrib2_merged_order.grib "
    + " -set_ext_name 1 -netcdf "
    + forecast_process_path
    + "_accum_wgrib2_merged.nc"
)

# Run wgrib2 to save as NetCDF
spOUT4 = subprocess.run(cmd4, shell=True, capture_output=True, encoding="utf-8")
os.remove(forecast_process_path + "_accum_wgrib2_merged_order.grib")


#######
# Use Dask to create a merged array (too large for xarray)
# Dask
chunkx = 100
chunky = 100

# Create base xarray for time interpolation
xarray_forecast_base = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")


# Create a new time series
start = xarray_forecast_base.time.min().values  # Adjust as necessary
end = xarray_forecast_base.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hisPeriod + 1, "H"),
    end=start + pd.Timedelta(192, "H"),
    freq="H",
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hisPeriod + 1, "H"),
            end=start - pd.Timedelta(1, "H"),
            freq="H",
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
        if dask_var == "PPROB":
            # Interpolate over nan's in APCP
            xarray_forecast = xarray_forecast_base.copy()
            # Import and allign times
            xarray_forecast["PPROB"] = xr.open_mfdataset(
                forecast_process_path + "_prob_wgrib2_merged.nc"
            )["APCP_prob_GT_0D254_prob_fcst_255_255_surface"]
            xarray_forecast["PPROB"] = xarray_forecast["PPROB"].bfill(dim="time")
            daskArray = xarray_forecast["PPROB"].data

            del xarray_forecast

        elif dask_var == "PACCUM":
            xarray_forecast = xarray_forecast_base.copy()
            xarray_forecast["PACCUM"] = xr.open_mfdataset(
                forecast_process_path + "_accum_wgrib2_merged.nc"
            )["APCP_surface"]
            # Fix precipitation accumulation timing to account for everything after the hourly data being a 6-hour accumulation,
            # Change to 1 hour accum
            xarray_forecast["PACCUM"][nbm_range1[-1] :, :, :] = (
                xarray_forecast["PACCUM"][nbm_range1[-1] :, :, :] / 6
            )

            # Interpolate over nan's in APCP
            xarray_forecast["PACCUM"] = xarray_forecast["PACCUM"].bfill(dim="time")

            daskArray = xarray_forecast["PACCUM"].data

            del xarray_forecast

        else:
            daskArray = da.from_array(
                nc.Dataset(forecast_process_path + "_wgrib2_merged.nc")[dask_var],
                lock=True,
            )

        # Check length for errors

        if len(daskArray) != len(nbm_range):
            print(len(daskArray))
            print(len(nbm_range))
            print(dask_var)
            assert len(daskArray) == len(
                nbm_range
            ), "Incorrect number of timesteps! Exiting"

        # Rechunk
        daskArray = daskArray.rechunk(chunks=(len(nbm_range), chunkx, chunky))

        # %% Save merged and processed xarray dataset to disk using zarr with compression
        # Define the path to save the zarr dataset
        # Save the dataset with compression and filters for all variables
        if dask_var == "time":
            # Save the dataset without compression and filters for all variable
            daskArray.to_zarr(
                forecast_process_path + "_zarrs/" + dask_var + ".zarr", overwrite=True
            )
        else:
            filters = [BitRound(keepbits=12)]  # Only keep ~ 3 significant digits
            compressor = Blosc(cname="zstd", clevel=1)  # Use zstd compression
            # Save the dataset with compression and filters for all variable
            daskArray.to_zarr(
                forecast_process_path + "_zarrs/" + dask_var + ".zarr",
                filters=filters,
                compression=compressor,
                overwrite=True,
            )


# Del to free memory
del daskArray, xarray_forecast_base

# Remove wgrib2 temp files
os.remove(forecast_process_path + "_wgrib2_merged.nc")
os.remove(forecast_process_path + "_prob_wgrib2_merged.nc")
os.remove(forecast_process_path + "_accum_wgrib2_merged.nc")

T1 = time.time()
print(T0 - T1)

################################################################################################
# Historic data
# %% Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
# %% Loop through the runs and check if they have already been processed to s3

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
for i in range(hisPeriod, -1, -1):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    if saveType == "S3":
        s3_path = (
            s3_bucket
            + "/NBM/NBM_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # # Try to open the zarr file to check if it has already been saved
        if s3.exists(s3_path):
            continue

    else:
        # Local Path Setup
        local_path = (
            s3_bucket
            + "/NBM/NBM_Hist"
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
    # Since the first hour forecast is used, then the time is an hour behind
    # So data for 18:00 would be the 1st hour of the 17:00 forecast.
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i + 1) + "h"),
        periods=1,
        freq="1H",
    )

    # Create a range of forecast lead times
    # Only want forecast at hour 1- SLightly less accurate than initializing at hour 0 but much avoids precipitation accumulation issues
    fxx = range(1, 2)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="nbm",
        fxx=fxx,
        product="co",
        verbose=False,
        priority=["aws"],
        save_dir=tmpDIR,
    )

    # Main Vars + Accum
    # Download the subsets
    FH_histsub.download(matchStrings + "|" + matchstring_po, verbose=False)

    # Use wgrib2 to change the order
    cmd1 = (
        f"{wgrib2_path}"
        + "  "
        + str(
            FH_histsub.file_exists[0].get_localFilePath(
                matchStrings + "|" + matchstring_po
            )
        )
        + " -ijsmall_grib "
        + " 1:2345 1:1597 "
        + hist_process_path
        + "_wgrib2_merged_order.grib"
    )
    spOUT1 = subprocess.run(cmd1, shell=True, capture_output=True, encoding="utf-8")

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

    # %% Merge the  xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib = xr.open_dataset(hist_process_path + "_wgrib_merge.nc")

    # Rename PPROB
    xarray_his_wgrib["PPROB"] = xarray_his_wgrib[
        "APCP_prob_GT_0D254_prob_fcst_255_255_surface"
    ]

    # Add PACCUM
    xarray_his_wgrib["PACCUM"] = xarray_his_wgrib["APCP_surface"]

    # Drop raw ptypes
    xarray_his_wgrib = xarray_his_wgrib.drop_vars(
        ["APCP_prob_GT_0D254_prob_fcst_255_255_surface"]
    )

    # %% Save merged and processed xarray dataset to disk using zarr with compression
    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times
    compressor = Blosc(cname="lz4", clevel=1)
    filters = [BitRound(keepbits=12)]

    # No chunking since only one time step
    encoding = {
        vname: {"compressor": compressor, "filters": filters} for vname in zarrVars[1:]
    }

    # Save as Zarr to s3 for Time Machine
    if saveType == "S3":
        zarrStore = s3fs.S3Map(root=s3_path, s3=s3, create=True)
    else:
        # Create local Zarr store
        zarrStore = zarr.DirectoryStore(local_path)

    # with ProgressBar():
    xarray_his_wgrib.to_zarr(
        store=zarrStore, mode="w", consolidated=True, encoding=encoding
    )

    # Clear the xarray dataset from memory
    del xarray_his_wgrib

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged_order.grib")
    os.remove(hist_process_path + "_wgrib_merge.nc")
    # os.remove(hist_process_path + '_ncTemp.nc')

    print((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))


# %% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################
# %% Merge the historic and forecast datasets and then squash using dask

# Create a zarr backed dask array
zarr_store = zarr.DirectoryStore(merge_process_dir + "/NBM_UnChunk.zarr")

compressor = Blosc(cname="zstd", clevel=3)
filters = [BitRound(keepbits=12)]

# Create a Zarr array in the store with zstd compression. Max length is 195 Forecast Hours  37
zarr_array = zarr.zeros(
    (len(zarrVars), 230, 1597, 2345),
    chunks=(1, 230, 100, 100),
    store=zarr_store,
    compressor=compressor,
    dtype="float32",
    overwrite=True,
)

# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    s3_bucket
    + "/NBM/NBM_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, -1, -1)
]

# Dask
daskArrays = []
daskVarArrays = []

with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    for daskVarIDX, dask_var in enumerate(zarrVars):
        for local_ncpath in ncLocalWorking_paths:
            if saveType == "S3":
                daskVarArrays.append(
                    da.from_zarr(
                        local_ncpath,
                        component=dask_var,
                        inline_array=True,
                        storage_options={
                            "key": "AKIA2HTALZ5LWRCTHC5F",
                            "secret": "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",
                        },
                    )
                )
            else:
                daskVarArrays.append(
                    da.from_zarr(local_ncpath, component=dask_var, inline_array=True)
                )

        # Add NC Forecast
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_zarrs/" + dask_var + ".zarr", inline_array=True
        )

        if dask_var == "time":
            daskVarArraysStack = da.stack(daskVarArrays)

            # Doesn't like being delayed?
            # Convert to float32 to match the other types
            daskCatTimes = da.concatenate(
                (da.squeeze(daskVarArraysStack), daskForecastArray), axis=0
            ).astype("float32")

            # with ProgressBar():
            interpTimes = da.map_blocks(
                linInterp1D,
                daskCatTimes.rechunk(len(daskCatTimes)),
                stacked_timesUnix,
                hourly_timesUnix,
                dtype="float32",
            ).compute()

            daskArrayOut = np.tile(
                np.expand_dims(np.expand_dims(interpTimes, axis=1), axis=1),
                (1, 1597, 2345),
            )

            da.to_zarr(
                da.from_array(np.expand_dims(daskArrayOut, axis=0)),
                zarr_array,
                region=(
                    slice(daskVarIDX, daskVarIDX + 1),
                    slice(0, 230),
                    slice(0, 1597),
                    slice(0, 2345),
                ),
            )

        else:
            daskVarArraysAppend = daskVarArrays.append(daskForecastArray)
            varMerged = da.concatenate(daskVarArrays, axis=0)

            # with ProgressBar():
            da.to_zarr(
                da.from_array(
                    da.expand_dims(
                        da.map_blocks(
                            linInterp3D,
                            varMerged.rechunk(
                                (len(stacked_timesUnix), 100, 100)
                            ).astype("float32"),
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
                    slice(0, 230),
                    slice(0, 1597),
                    slice(0, 2345),
                ),
            )

        daskVarArrays = []
        varMerged = []

        print(dask_var)

    zarr_store.close()

    shutil.rmtree(forecast_process_path + "_zarrs")

    # Rechunk the zarr array
    encoding = {"compressor": Blosc(cname="zstd", clevel=3)}

    source = zarr.open(merge_process_dir + "/NBM_UnChunk.zarr")
    intermediate = merge_process_dir + "/NBM_Mid.zarr"
    target = zarr.ZipStore(merge_process_dir + "/NBM.zarr.zip", compression=0)
    rechunked = rechunk(
        source,
        target_chunks=(len(zarrVars), 230, 2, 2),
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
        merge_process_dir + "/NBM.zarr.zip", s3_bucket + "/ForecastTar/NBM.zarr.zip"
    )

    # Write most recent forecast time
    with open(merge_process_dir + "/NBM.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        merge_process_dir + "/NBM.time.pickle",
        s3_bucket + "/ForecastTar/NBM.time.pickle",
    )


else:
    # Move to local
    shutil.move(
        merge_process_dir + "/NBM.zarr.zip", s3_bucket + "/ForecastTar/NBM.zarr.zip"
    )

    # Write most recent forecast time
    with open(merge_process_dir + "/NBM.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        merge_process_dir + "/NBM.time.pickle",
        s3_bucket + "/ForecastTar/NBM.time.pickle",
    )

    # Clean up
    shutil.rmtree(merge_process_dir)
T2 = time.time()

print(T2 - T1)
print(T2 - T0)
