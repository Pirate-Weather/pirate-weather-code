# %% NBM Hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023

# %% Import modules
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from itertools import chain

import dask
import dask.array as da
import netCDF4 as nc
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest
from scipy.interpolate import make_interp_spline


# Scipy Interp Function
def linInterp1D(block, T_in, T_out):
    interp = make_interp_spline(T_in, block, 3, axis=1)
    interpOut = interp(T_out)
    return interpOut


def getGribList(FH_forecastsub, matchStrings):
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
    return gribList


warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/Process/NBM"
)
forecast_process_path = forecast_process_dir + "/NBM_Process"
hist_process_path = forecast_process_dir + "/NBM_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/NBM")
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/Hist/NBM")


saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunk size
finalChunk = 3

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
T0 = time.time()

latestRun = Herbie_latest(
    model="nbm",
    n=5,
    freq="1h",
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
    if s3.exists(forecast_path + "/NBM.time.pickle"):
        with s3.open(forecast_path + "/NBM.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/NBM.time.pickle"):
        # Open the file in binary mode
        with open(forecast_path + "/NBM.time.pickle", "rb") as file:
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
matchstring_2m = r":((DPT|TMP|APTMP|RH):2 m above ground:.*fcst:nan)"
matchstring_su = r":((PTYPE):surface:.*)"
matchstring_10m = r"(:(GUST|WIND|WDIR):10 m above ground:.*fcst:nan)"
matchstring_pr = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:nan)"
matchstring_re = (
    r":((TCDC|VIS):surface:.*fcst:nan)"  # This gets the correct surface param
)

matchstring_pw = r":(PWTHER:)"  # This gets the correct surface param

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
# Create FastHerbieFastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
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
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

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
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="nbm",
    fxx=nbm_range1,
    product="co",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

matchstring_po = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|[0-9]*0-[0-9]{1,2}1|[0-9]*1-[0-9]{1,2}2|[0-9]*2-[0-9]{1,2}3|[0-9]*3-[0-9]{1,2}4|[0-9]*4-[0-9]{1,2}5|[0-9]*5-[0-9]{1,2}6|[0-9]*6-[0-9]{1,2}7|[0-9]*7-[0-9]{1,2}8|[0-9]*8-[0-9]{1,2}9|[0-9]*9-[0-9]{1,2}0).*fcst:prob.*)"
# Download the subsets
FH_forecastsub.download(matchstring_po, verbose=False)

# Create list of downloaded grib files
gribList1 = getGribList(FH_forecastsub, matchstring_po)
#####
# Download PPROB as 6-Hour Accum for hours 036-190

# Create FastHerbie object
FH_forecastsub2 = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="nbm",
    fxx=nbm_range2,
    product="co",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

# Match 6-hour probs
matchstring_po2 = r":APCP:surface:(0-6|[0-9]*0-[0-9]{1,2}6|[0-9]*1-[0-9]{1,2}7|[0-9]*2-[0-9]{1,2}8|[0-9]*3-[0-9]{1,2}9|[0-9]*4-[0-9]{1,2}0|[0-9]*5-[0-9]{1,2}1|[0-9]*6-[0-9]{1,2}2|[0-9]*7-[0-9]{1,2}3|[0-9]*8-[0-9]{1,2}4|[0-9]*9-[0-9]{1,2}5).*fcst:prob"
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
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()


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
if spOUT2.returncode != 0:
    print(spOUT2.stderr)
    sys.exit()
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
if spOUT4.returncode != 0:
    print(spOUT4.stderr)
    sys.exit()
os.remove(forecast_process_path + "_prob_wgrib2_merged_order.grib")


# Download PACCUM as 1-Hour and 6-hour Accum
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="nbm",
    fxx=nbm_range1,
    product="co",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

matchstring_pa = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:nan)"
# Download the subsets
FH_forecastsub.download(matchstring_pa, verbose=False)

# Create list of downloaded grib files
gribList1 = getGribList(FH_forecastsub, matchstring_pa)

#####
# Download PPROB as 6-Hour Accum for hours 036-190

# Create FastHerbie object
FH_forecastsub2 = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="nbm",
    fxx=nbm_range2,
    product="co",
    verbose=False,
    priority=["aws"],
    save_dir=tmpDIR,
)

# Match 6-hour probs
matchstring_pa2 = r"(:APCP:surface:(0-6|\d*0-\d{1,2}6|\d*1-\d{1,2}7|\d*2-\d{1,2}8|\d*3-\d{1,2}9|\d*4-\d{1,2}0|\d*5-\d{1,2}1|\d*6-\d{1,2}2|\d*7-\d{1,2}3|\d*8-\d{1,2}4|\d*9-\d{1,2}5).*fcst:nan)"
# Download the subsets
FH_forecastsub2.download(matchstring_pa2, verbose=False)

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
if spOUT.returncode != 0:
    print(spOUT.stderr)
    sys.exit()

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
if spOUT2.returncode != 0:
    print(spOUT2.stderr)
    sys.exit()
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
if spOUT4.returncode != 0:
    print(spOUT4.stderr)
    sys.exit()
os.remove(forecast_process_path + "_accum_wgrib2_merged_order.grib")


#######
# Use Dask to create a merged array (too large for xarray)
# Dask

# Create base xarray for time interpolation
xarray_forecast_base = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")


# Create a new time series
start = xarray_forecast_base.time.min().values  # Adjust as necessary
end = xarray_forecast_base.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hisPeriod + 1, "h"),
    end=start + pd.Timedelta(192, "h"),
    freq="h",
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hisPeriod + 1, "h"),
            end=start - pd.Timedelta(1, "h"),
            freq="h",
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

# Open NC Dataset
ncForecast = nc.Dataset(forecast_process_path + "_wgrib2_merged.nc")

# Disable masking
ncForecast.set_auto_mask(False)

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
                ncForecast[dask_var],
                lock=True,
            )

        # Check length for errors

        if len(daskArray) != len(nbm_range):
            print(len(daskArray))
            print(len(nbm_range))
            print(dask_var)
            assert len(daskArray) == len(nbm_range), (
                "Incorrect number of timesteps! Exiting"
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
                codecs=[
                    zarr.codecs.BytesCodec(),
                    zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                ],
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
# %% Historic data
# Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
# Loop through the runs and check if they have already been processed to s3

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
for i in range(hisPeriod, -1, -1):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    if saveType == "S3":
        s3_path = (
            historic_path
            + "/NBM_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # # Try to open the zarr file to check if it has already been saved
        if s3.exists(s3_path):
            continue

    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/NBM_Hist"
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
        freq="1h",
    )

    # Create a range of forecast lead times
    # Only want forecast at hour 1- SLightly less accurate than initializing at hour 0 but much avoids precipitation accumulation issues
    fxx = range(1, 2)

    print(DATES)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="nbm",
        fxx=fxx,
        product="co",
        verbose=True,
        priority=["aws"],
        save_dir=tmpDIR,
    )

    # Main Vars + Accum
    # Download the subsets
    FH_histsub.download(matchStrings + "|" + matchstring_po, verbose=True)

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

    # Save merged and processed xarray dataset to disk using zarr
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

    # with ProgressBar():
    xarray_his_wgrib.chunk(
        chunks={"time": 1, "x": processChunk, "y": processChunk}
    ).to_zarr(store=zarrStore, mode="w", consolidated=False)

    # Clear the xarray dataset from memory
    del xarray_his_wgrib

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged_order.grib")
    os.remove(hist_process_path + "_wgrib_merge.nc")
    # os.remove(hist_process_path + '_ncTemp.nc')

    print((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))


# %% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    historic_path
    + "/NBM_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, -1, -1)
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
        forecast_process_dir + "/NBM.zarr.zip", mode="w", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/NBM.zarr")

    # Check if the store exists, if so, open itm otherwise create it
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
stackInterp = da.rechunk(
    da.map_blocks(
        linInterp1D,
        daskVarArrayStackDisk,
        stacked_timesUnix,
        hourly_timesUnix,
        dtype="float32",
        chunks=(1, len(stacked_timesUnix), processChunk, processChunk),
    ),
    (len(zarrVars), len(hourly_timesUnix), finalChunk, finalChunk),
).to_zarr(zarr_array, overwrite=True, compute=True)


if saveType == "S3":
    zarr_store.close()


# Rechunk subset of data for maps!
# Want variables:
# 0 (time)
# 2 (TMP)
# 6 (WIND)
# 7 (WDIR)
# 8 (APCP)
# 13 (PACCUM)
# 14:17 (PTYPE)

# Loop through variables, creating a new one with a name and 36 x 100 x 100 chunks
# Save -12:24 hours, aka steps 24:60
# Create a Zarr array in the store with zstd compression
if saveType == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/NBM_Maps.zarr.zip", mode="a"
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(forecast_process_dir + "/NBM_Maps.zarr")

for z in [0, 2, 6, 7, 8, 13, 14, 15, 16, 17]:
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

    # with ProgressBar():
    da.rechunk(daskVarArrayStackDisk[z, 24:60, :, :], (36, 100, 100)).to_zarr(
        zarr_array, overwrite=True, compute=True
    )

    print(zarrVars[z])

if saveType == "S3":
    zarr_store_maps.close()

# %% Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(forecast_process_dir + "/NBM.zarr.zip", forecast_path + "/NBM.zarr.zip")
    s3.put_file(
        forecast_process_dir + "/NBM_Maps.zarr.zip",
        forecast_path + "/NBM_Maps.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/NBM.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/NBM.time.pickle",
        forecast_path + "/NBM.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/NBM.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/NBM.time.pickle",
        forecast_path + "/NBM.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/NBM.zarr",
        forecast_path + "/NBM.zarr",
        dirs_exist_ok=True,
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/NBM_Maps.zarr",
        forecast_path + "/NBM_Maps.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
