# %% NBM Hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023

# %% Import modules
import fcntl
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from itertools import chain

import dask
import dask.array as da
import netCDF4 as nc
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import FastHerbie
from herbie.fast import Herbie_latest

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    configure_zarr_limits,
    getGribList,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
    validate_grib_stats,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")


def _timing_log(message: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[TIMING] {ts} {message}")


def _close_store(store: object) -> None:
    """Close a zarr store if it exposes a close method."""
    close_fn = getattr(store, "close", None)
    if callable(close_fn):
        close_fn()


def _copytree_for_publish(src: str, dst: str) -> None:
    """Copy a directory tree for publish in a low-FD child process."""
    os.makedirs(dst, exist_ok=True)
    try:
        subprocess.run(["cp", "-a", os.path.join(src, "."), dst], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(
            f"Warning: cp -a publish copy failed ({exc}); falling back to shutil.copytree"
        )
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _publish_staged_dir(staged_path: str, final_path: str) -> None:
    """Promote a staged directory to final path with a same-dir rename when possible."""
    parent = os.path.dirname(final_path)
    backup_path = final_path + ".prev"
    os.makedirs(parent, exist_ok=True)

    if os.path.exists(backup_path):
        shutil.rmtree(backup_path, ignore_errors=True)

    try:
        if os.path.exists(final_path):
            os.replace(final_path, backup_path)
        os.replace(staged_path, final_path)
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)
    except OSError as exc:
        print(f"Warning: atomic promote failed ({exc}); falling back to copy")
        if os.path.exists(final_path):
            shutil.rmtree(final_path, ignore_errors=True)
        _copytree_for_publish(staged_path, final_path)
        shutil.rmtree(staged_path, ignore_errors=True)
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)


def _acquire_single_run_lock(lock_path: str):
    """Acquire a non-blocking process lock so overlapping ingests do not clobber temp data."""
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(
            "Another NBM ingest process is already running; "
            f"skipping this run (lock: {lock_path})"
        )
        lock_file.close()
        sys.exit(0)

    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/NBM")
forecast_process_path = forecast_process_dir + "/NBM_Process"
hist_process_path = forecast_process_dir + "/NBM_Historic"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/NBM")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/Hist/NBM")
local_forecast_version_dir = forecast_path + "/" + ingest_version
nbm_staged_path: str | None = None
nbm_maps_staged_path: str | None = None


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)
herbie_max_threads = positive_int_env("herbie_max_threads", 1)

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["NBM"]

# Define the final x/y chunk size
final_chunk = FINAL_CHUNK_SIZES["NBM"]

his_period = HISTORY_PERIODS["NBM"]
_ingest_lock_handle = _acquire_single_run_lock(forecast_process_dir + ".lock")

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download":
    if not os.path.exists(local_forecast_version_dir):
        os.makedirs(local_forecast_version_dir)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

# %% Define base time from the most recent run
T0 = time.time()

latest_run = Herbie_latest(
    model="nbm",
    n=5,
    freq="1h",
    fxx=[190, 191, 192, 193, 194, 195],
    product="co",
    verbose=False,
    priority=["aws", "nomads"],
    save_dir=tmp_dir,
)

base_time = latest_run.date
# base_time = pd.Timestamp("2025-07-03 17:00:00")

# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/NBM.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/NBM.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/NBM.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/NBM.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to NBM, ending")
            sys.exit()

# base_time = pd.Timestamp("2024-03-05 16:00")
# base_time = base_time - pd.Timedelta(hours=1)
print(base_time)
_timing_log(f"NBM_FORECAST run_start base_time={base_time}")

zarr_vars = (
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
    "DSWRF_surface",
    "CAPE_surface",
)


#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours
# Set download rannges
forecast_iter_start = time.perf_counter()
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
matchstring_2m = r":((DPT|TMP|APTMP|RH):2 m above ground:.*fcst:$)"
matchstring_su = r":((PTYPE):surface:.*)"
matchstring_10m = r"(:(GUST|WIND|WDIR):10 m above ground:.*fcst:$)"
matchstring_pr = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:$)"
matchstring_re = (
    r":((TCDC|VIS|DSWRF|CAPE):surface:.*fcst:$)"  # This gets the correct surface param
)

matchstring_pw = r":(PWTHER:)"  # This gets the correct surface param

# Merge matchstrings for download
match_strings = (
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
    priority=["aws", "nomads"],
    max_threads=herbie_max_threads,
    save_dir=tmp_dir,
)

# Download the subsets
stage_start = time.perf_counter()
FH_forecastsub.download(match_strings, verbose=False)
_timing_log(
    f"NBM_FORECAST stage=download_main elapsed={time.perf_counter() - stage_start:.2f}s"
)


# Check for download length
if len(FH_forecastsub.file_exists) != len(nbm_range):
    print(
        "Download failed, expected "
        + str(len(nbm_range))
        + " files, but got "
        + str(len(FH_forecastsub.file_exists))
    )
    sys.exit(1)


# Create list of downloaded grib files
grib_list = getGribList(FH_forecastsub, match_strings)

# Perform a check if any data seems to be invalid
stage_start = time.perf_counter()
cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

validate_grib_stats(grib_check)
print("Grib files passed validation, proceeding with processing")
_timing_log(
    f"NBM_FORECAST stage=validate_main elapsed={time.perf_counter() - stage_start:.2f}s"
)


# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(grib_list)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -grib "
    + forecast_process_path
    + "_wgrib2_merged.grib2"
)
# Run wgrib2
stage_start = time.perf_counter()
substage_start = time.perf_counter()
sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    print(sp_out.stderr)
    sys.exit()
_timing_log(
    f"NBM_FORECAST stage=main_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
)

# Use wgrib2 to change the order
substage_start = time.perf_counter()
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
_timing_log(
    f"NBM_FORECAST stage=main_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
)
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
substage_start = time.perf_counter()
spOUT4 = subprocess.run(cmd4, shell=True, capture_output=True, encoding="utf-8")
if spOUT4.returncode != 0:
    print(spOUT4.stderr)
    sys.exit()
os.remove(forecast_process_path + "_wgrib2_merged_order.grib")
_timing_log(
    f"NBM_FORECAST stage=main_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
)
_timing_log(
    f"NBM_FORECAST stage=main_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
)


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
    priority=["aws", "nomads"],
    max_threads=herbie_max_threads,
    save_dir=tmp_dir,
)

matchstring_po = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|[0-9]*0-[0-9]{1,2}1|[0-9]*1-[0-9]{1,2}2|[0-9]*2-[0-9]{1,2}3|[0-9]*3-[0-9]{1,2}4|[0-9]*4-[0-9]{1,2}5|[0-9]*5-[0-9]{1,2}6|[0-9]*6-[0-9]{1,2}7|[0-9]*7-[0-9]{1,2}8|[0-9]*8-[0-9]{1,2}9|[0-9]*9-[0-9]{1,2}0).*fcst:prob.*)"
# Download the subsets
stage_start = time.perf_counter()
FH_forecastsub.download(matchstring_po, verbose=False)
_timing_log(
    "NBM_FORECAST "
    f"stage=download_pprob_hourly elapsed={time.perf_counter() - stage_start:.2f}s"
)

# Check for download length
if len(FH_forecastsub.file_exists) != len(nbm_range1):
    print(
        "Download failed, expected "
        + str(len(nbm_range1))
        + " files, but got "
        + str(len(FH_forecastsub.file_exists))
    )
    sys.exit(1)


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
    priority=["aws", "nomads"],
    max_threads=herbie_max_threads,
    save_dir=tmp_dir,
)

# Match 6-hour probs
matchstring_po2 = r":APCP:surface:(0-6|[0-9]*0-[0-9]{1,2}6|[0-9]*1-[0-9]{1,2}7|[0-9]*2-[0-9]{1,2}8|[0-9]*3-[0-9]{1,2}9|[0-9]*4-[0-9]{1,2}0|[0-9]*5-[0-9]{1,2}1|[0-9]*6-[0-9]{1,2}2|[0-9]*7-[0-9]{1,2}3|[0-9]*8-[0-9]{1,2}4|[0-9]*9-[0-9]{1,2}5).*fcst:prob"
# Download the subsets
stage_start = time.perf_counter()
FH_forecastsub2.download(matchstring_po2, verbose=False)
_timing_log(
    "NBM_FORECAST "
    f"stage=download_pprob_6hour elapsed={time.perf_counter() - stage_start:.2f}s"
)

# Check for download length
if len(FH_forecastsub2.file_exists) != len(nbm_range2):
    print(
        "Download failed, expected "
        + str(len(nbm_range2))
        + " files, but got "
        + str(len(FH_forecastsub2.file_exists))
    )
    sys.exit(1)

# Create list of downloaded grib files
gribList2 = getGribList(FH_forecastsub2, matchstring_po2)

grib_list = gribList1 + gribList2

# Perform a check if any data seems to be invalid
stage_start = time.perf_counter()
cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

validate_grib_stats(grib_check)
print("Grib files passed validation, proceeding with processing")
_timing_log(
    f"NBM_FORECAST stage=validate_pprob elapsed={time.perf_counter() - stage_start:.2f}s"
)


# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(grib_list)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -grib "
    + forecast_process_path
    + "_prob_wgrib2_merged.grib2"
)
# Run wgrib2
stage_start = time.perf_counter()
substage_start = time.perf_counter()
sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    print(sp_out.stderr)
    sys.exit()
_timing_log(
    f"NBM_FORECAST stage=pprob_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
)


# Use wgrib2 to change the order
substage_start = time.perf_counter()
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
_timing_log(
    f"NBM_FORECAST stage=pprob_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
)
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
substage_start = time.perf_counter()
spOUT4 = subprocess.run(cmd4, shell=True, capture_output=True, encoding="utf-8")
if spOUT4.returncode != 0:
    print(spOUT4.stderr)
    sys.exit()
os.remove(forecast_process_path + "_prob_wgrib2_merged_order.grib")
_timing_log(
    f"NBM_FORECAST stage=pprob_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
)
_timing_log(
    f"NBM_FORECAST stage=pprob_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
)


# Download PACCUM as 1-Hour and 6-hour Accum
# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="nbm",
    fxx=nbm_range1,
    product="co",
    verbose=False,
    priority=["aws", "nomads"],
    max_threads=herbie_max_threads,
    save_dir=tmp_dir,
)

matchstring_pa = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:$)"
# Download the subsets
stage_start = time.perf_counter()
FH_forecastsub.download(matchstring_pa, verbose=False)
_timing_log(
    "NBM_FORECAST "
    f"stage=download_paccum_hourly elapsed={time.perf_counter() - stage_start:.2f}s"
)

# Check for download length
if len(FH_forecastsub.file_exists) != len(nbm_range1):
    print(
        "Download failed, expected "
        + str(len(nbm_range1))
        + " files, but got "
        + str(len(FH_forecastsub.file_exists))
    )
    sys.exit(1)

# Create list of downloaded grib files
gribList1 = getGribList(FH_forecastsub, matchstring_pa)

#####
# Download 6-Hour Accum for hours 036-190

# Create FastHerbie object
FH_forecastsub2 = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="nbm",
    fxx=nbm_range2,
    product="co",
    verbose=False,
    priority=["aws", "nomads"],
    max_threads=herbie_max_threads,
    save_dir=tmp_dir,
)

# Match 6-hour accumulation
matchstring_pa2 = r"(:APCP:surface:(0-6|\d*0-\d{1,2}6|\d*1-\d{1,2}7|\d*2-\d{1,2}8|\d*3-\d{1,2}9|\d*4-\d{1,2}0|\d*5-\d{1,2}1|\d*6-\d{1,2}2|\d*7-\d{1,2}3|\d*8-\d{1,2}4|\d*9-\d{1,2}5).*fcst:$)"
# Download the subsets
stage_start = time.perf_counter()
FH_forecastsub2.download(matchstring_pa2, verbose=False)
_timing_log(
    "NBM_FORECAST "
    f"stage=download_paccum_6hour elapsed={time.perf_counter() - stage_start:.2f}s"
)

# Check for download length
if len(FH_forecastsub2.file_exists) != len(nbm_range2):
    print(
        "Download failed, expected "
        + str(len(nbm_range2))
        + " files, but got "
        + str(len(FH_forecastsub2.file_exists))
    )
    sys.exit(1)

# Create list of downloaded grib files
gribList2 = getGribList(FH_forecastsub2, matchstring_pa2)
grib_list = gribList1 + gribList2


# Perform a check if any data seems to be invalid
stage_start = time.perf_counter()
cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

validate_grib_stats(grib_check)
print("Grib files passed validation, proceeding with processing")
_timing_log(
    f"NBM_FORECAST stage=validate_paccum elapsed={time.perf_counter() - stage_start:.2f}s"
)


# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(grib_list)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -grib "
    + forecast_process_path
    + "_accum_wgrib2_merged.grib2"
)
# Run wgrib2
stage_start = time.perf_counter()
substage_start = time.perf_counter()
sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    print(sp_out.stderr)
    sys.exit()
_timing_log(
    f"NBM_FORECAST stage=paccum_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
)

# Use wgrib2 to change the order
substage_start = time.perf_counter()
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
_timing_log(
    f"NBM_FORECAST stage=paccum_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
)
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
substage_start = time.perf_counter()
spOUT4 = subprocess.run(cmd4, shell=True, capture_output=True, encoding="utf-8")
if spOUT4.returncode != 0:
    print(spOUT4.stderr)
    sys.exit()
os.remove(forecast_process_path + "_accum_wgrib2_merged_order.grib")
_timing_log(
    f"NBM_FORECAST stage=paccum_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
)
_timing_log(
    f"NBM_FORECAST stage=paccum_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
)


#######
# Use Dask to create a merged array (too large for xarray)
# Dask
stage_start = time.perf_counter()

# Create base xarray for time interpolation
xarray_forecast_base = xr.open_mfdataset(forecast_process_path + "_wgrib2_merged.nc")


# Create a new time series
start = xarray_forecast_base.time.min().values  # Adjust as necessary
end = xarray_forecast_base.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hours=his_period + 1),
    end=start + pd.Timedelta(hours=192),
    freq="h",
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hours=his_period + 1),
            end=start - pd.Timedelta(hours=1),
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
forecast_var_zarr_dir = forecast_process_path + "_zarrs"
if os.path.exists(forecast_var_zarr_dir):
    shutil.rmtree(forecast_var_zarr_dir)
os.makedirs(forecast_var_zarr_dir, exist_ok=True)

with dask.config.set(**{"array.slicing.split_large_chunks": True}):
    for dask_var in zarr_vars:
        var_stage_start = time.perf_counter()
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
            chunks=(len(nbm_range), process_chunk, process_chunk)
        )

        # Save merged and processed xarray dataset to disk using zarr with compression
        # Define the path to save the zarr dataset
        # Save the dataset with compression and filters for all variables
        forecast_var_zarr_path = forecast_var_zarr_dir + "/" + dask_var + ".zarr"
        if dask_var == "time":
            # Save the dataset without compression and filters for all variable
            daskArray.to_zarr(forecast_var_zarr_path, overwrite=True)
        else:
            # Save the dataset with compression and filters for all variable
            daskArray.to_zarr(
                forecast_var_zarr_path,
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                overwrite=True,
            )
        try:
            zarr.open_array(forecast_var_zarr_path, mode="r")
        except Exception as exc:
            raise RuntimeError(
                "Failed to open forecast zarr store for "
                f"{dask_var} at {forecast_var_zarr_path}"
            ) from exc
        _timing_log(
            "NBM_FORECAST "
            f"stage=forecast_var_to_zarr var={dask_var} elapsed={time.perf_counter() - var_stage_start:.2f}s"
        )

_timing_log(
    f"NBM_FORECAST stage=forecast_vars_to_zarr elapsed={time.perf_counter() - stage_start:.2f}s"
)


# Del to free memory
ncForecast.close()
xarray_forecast_base.close()
del daskArray, xarray_forecast_base

# Remove wgrib2 temp files
cleanup_stage_start = time.perf_counter()
os.remove(forecast_process_path + "_wgrib2_merged.nc")
os.remove(forecast_process_path + "_prob_wgrib2_merged.nc")
os.remove(forecast_process_path + "_accum_wgrib2_merged.nc")
_timing_log(
    f"NBM_FORECAST stage=forecast_temp_cleanup elapsed={time.perf_counter() - cleanup_stage_start:.2f}s"
)

T1 = time.time()
print(T0 - T1)

################################################################################################
# %% Historic data
# Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
# Loop through the runs and check if they have already been processed to s3

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
hist_processed_count = 0
hist_total_seconds = 0.0
for i in range(his_period, -1, -1):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601
    hist_target = (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    hist_iter_start = time.perf_counter()
    _timing_log(f"NBM_HIST {hist_target} start")

    if save_type == "S3":
        s3_path = historic_path + "/NBM_Hist" + hist_target + ".zarr"
        # Check for a done file in S3
        if s3.exists(s3_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + s3_path)
            _timing_log(f"NBM_HIST {hist_target} skip (already exists)")

            # If the file exists, check that it works
            try:
                hisCheckStore = zarr.storage.FsspecStore.from_url(
                    s3_path,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                )
                zarr.open(hisCheckStore)[zarr_vars[-1]][-1, -1, -1]
                continue  # If it exists, skip to the next iteration
            except Exception:
                print("### Historic Data Failure!")
                print(traceback.print_exc())

                # Delete the file if it exists
                if s3.exists(s3_path):
                    s3.rm(s3_path)

    else:
        # Local Path Setup
        local_path = historic_path + "/NBM_Hist" + hist_target + ".zarr"
        # Check for a loca done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + local_path)
            _timing_log(f"NBM_HIST {hist_target} skip (already exists)")
            continue

    print("Downloading: " + hist_target)

    # Create a range of dates for historic data going back 48 hours
    # Since the first hour forecast is used, then the time is an hour behind
    # So data for 18:00 would be the 1st hour of the 17:00 forecast.
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(hours=i + 1),
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
        verbose=False,
        priority=["aws", "nomads"],
        max_threads=herbie_max_threads,
        save_dir=tmp_dir,
    )

    # Main Vars + Accum
    # Download the subsets
    stage_start = time.perf_counter()
    FH_histsub.download(match_strings + "|" + matchstring_po, verbose=False)
    _timing_log(
        f"NBM_HIST {hist_target} stage=download_subsets elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Perform a check if any data seems to be invalid
    stage_start = time.perf_counter()
    cmd = (
        f"{wgrib2_path}"
        + "  "
        + str(
            FH_histsub.file_exists[0].get_localFilePath(
                match_strings + "|" + matchstring_po
            )
        )
        + " -s -stats"
    )

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

    validate_grib_stats(grib_check)
    print("Grib files passed validation, proceeding with processing")
    _timing_log(
        f"NBM_HIST {hist_target} stage=validate_grib elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Use wgrib2 to change the order
    stage_start = time.perf_counter()
    cmd1 = (
        f"{wgrib2_path}"
        + "  "
        + str(
            FH_histsub.file_exists[0].get_localFilePath(
                match_strings + "|" + matchstring_po
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
    _timing_log(
        f"NBM_HIST {hist_target} stage=wgrib_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Merge the  xarrays
    # Read the netcdf file using xarray
    stage_start = time.perf_counter()
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
    _timing_log(
        f"NBM_HIST {hist_target} stage=open_transform elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Save merged and processed xarray dataset to disk using zarr
    # Save as Zarr to s3 for Time Machine
    if save_type == "S3":
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

    # Limit worker fan-out for local zarr writes to avoid "too many open files".
    stage_start = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_his_wgrib.chunk(
            chunks={"time": 1, "x": process_chunk, "y": process_chunk}
        ).to_zarr(
            store=zarrStore,
            mode="w",
            consolidated=False,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    _timing_log(
        f"NBM_HIST {hist_target} stage=write_zarr elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Clear the xarray dataset from memory
    xarray_his_wgrib.close()
    del xarray_his_wgrib

    # Remove temp file created by wgrib2
    stage_start = time.perf_counter()
    os.remove(hist_process_path + "_wgrib2_merged_order.grib")
    os.remove(hist_process_path + "_wgrib_merge.nc")
    _timing_log(
        f"NBM_HIST {hist_target} stage=cleanup elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        done_file = s3_path.replace(".zarr", ".done")
        s3.touch(done_file)
    else:
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    print(hist_target)
    hist_elapsed = time.perf_counter() - hist_iter_start
    hist_processed_count += 1
    hist_total_seconds += hist_elapsed
    _timing_log(f"NBM_HIST {hist_target} total_elapsed={hist_elapsed:.2f}s")

if hist_processed_count:
    _timing_log(
        "NBM_HIST summary "
        f"processed={hist_processed_count} "
        f"total_elapsed={hist_total_seconds:.2f}s "
        f"avg_per_file={hist_total_seconds / hist_processed_count:.2f}s"
    )


# %% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################

print("Merge and interpolate arrays.")
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    historic_path
    + "/NBM_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(his_period, -1, -1)
]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(zarr_vars[:]):
    for local_ncpath in ncLocalWorking_paths:
        if save_type == "S3":
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
    forecast_var_zarr_path = forecast_var_zarr_dir + "/" + dask_var + ".zarr"
    if not os.path.exists(forecast_var_zarr_path):
        raise FileNotFoundError(
            "Missing forecast zarr variable store: "
            f"{forecast_var_zarr_path}. This usually means another ingest "
            "run cleared the shared processing directory while this run was active."
        )
    daskForecastArray = da.from_zarr(forecast_var_zarr_path, inline_array=True)

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
        ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskArrayOut = da.concatenate((daskVarArraysStack, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
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
stage_start = time.perf_counter()
daskVarArrayListMergeNaN.to_zarr(
    forecast_process_path + "_stack.zarr", overwrite=True, compute=True
)
_timing_log(
    f"NBM_FORECAST stage=stack_to_zarr elapsed={time.perf_counter() - stage_start:.2f}s"
)

print("Stacked 4D array saved to disk.")

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/NBM.zarr.zip", mode="a", compression=0
    )
else:
    nbm_staged_path = local_forecast_version_dir + "/NBM.zarr.partial"
    if os.path.exists(nbm_staged_path):
        shutil.rmtree(nbm_staged_path)
    zarr_store = zarr.storage.LocalStore(nbm_staged_path)


with ProgressBar():
    stage_start = time.perf_counter()
    # 1. Interpolate the stacked array to be hourly along the time axis
    daskVarArrayStackDiskInterp = interp_time_take_blend(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
    )

    # 2. Pad to chunk size
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    # 3. Create the zarr array
    zarr_array = zarr.create_array(
        store=zarr_store,
        shape=(
            len(zarr_vars),
            len(hourly_timesUnix),
            daskVarArrayStackDiskInterpPad.shape[2],
            daskVarArrayStackDiskInterpPad.shape[3],
        ),
        chunks=(len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
    )

    # 4. Rechunk it to match the final array
    # 5. Write it out to the zarr array
    daskVarArrayStackDiskInterpPad.round(5).rechunk(
        (len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk)
    ).to_zarr(zarr_array, overwrite=True, compute=True)
    _timing_log(
        f"NBM_FORECAST stage=interpolate_and_write elapsed={time.perf_counter() - stage_start:.2f}s"
    )

print("Interpolate complete")

_close_store(zarr_store)


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

# Add padding for map chunking (100x100)
daskVarArrayStackDisk_maps = pad_to_chunk_size(daskVarArrayStackDisk, 100)

if save_type == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/NBM_Maps.zarr.zip", mode="a"
    )
else:
    nbm_maps_staged_path = local_forecast_version_dir + "/NBM_Maps.zarr.partial"
    if os.path.exists(nbm_maps_staged_path):
        shutil.rmtree(nbm_maps_staged_path)
    zarr_store_maps = zarr.storage.LocalStore(nbm_maps_staged_path)

stage_start = time.perf_counter()
for z in [0, 2, 6, 7, 8, 13, 14, 15, 16, 17]:
    # Create a zarr backed dask array
    zarr_array = zarr.create_array(
        store=zarr_store_maps,
        name=zarr_vars[z],
        shape=(
            36,
            daskVarArrayStackDisk_maps.shape[2],
            daskVarArrayStackDisk_maps.shape[3],
        ),
        chunks=(36, 100, 100),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
    )

    # with ProgressBar():
    da.rechunk(daskVarArrayStackDisk_maps[z, 36:72, :, :], (36, 100, 100)).to_zarr(
        zarr_array, overwrite=True, compute=True
    )

    print(zarr_vars[z])

_close_store(zarr_store_maps)
_timing_log(
    f"NBM_FORECAST stage=maps_write elapsed={time.perf_counter() - stage_start:.2f}s"
)

print("Map complete")

# %% Upload to S3
stage_start = time.perf_counter()
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/NBM.zarr.zip",
        forecast_path + "/" + ingest_version + "/NBM.zarr.zip",
    )
    s3.put_file(
        forecast_process_dir + "/NBM_Maps.zarr.zip",
        forecast_path + "/" + ingest_version + "/NBM_Maps.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/NBM.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/NBM.time.pickle",
        forecast_path + "/" + ingest_version + "/NBM.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(local_forecast_version_dir + "/NBM.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    if nbm_staged_path is None or nbm_maps_staged_path is None:
        raise RuntimeError(
            "Expected staged local zarr paths for Download publish, but they were not set"
        )

    _publish_staged_dir(
        nbm_staged_path,
        local_forecast_version_dir + "/NBM.zarr",
    )

    _publish_staged_dir(
        nbm_maps_staged_path,
        local_forecast_version_dir + "/NBM_Maps.zarr",
    )
_timing_log(
    f"NBM_FORECAST stage=publish_output elapsed={time.perf_counter() - stage_start:.2f}s"
)
_timing_log(
    f"NBM_FORECAST total_elapsed={time.perf_counter() - forecast_iter_start:.2f}s"
)

# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
