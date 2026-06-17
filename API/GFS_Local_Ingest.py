"""GFS Local Data Ingestion Script

Downloads, processes, and stores GFS forecast and historic data using wgrib2,
Herbie, and xarray. Generates hourly interpolated datasets with multiple variables.

Author: Alexander Rey
Date: September 2023
"""
# ruff: noqa: E402  # dotenv must be loaded before downstream imports

import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da

# Env setup
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest
from tqdm import tqdm

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR, MISSING_DATA
from API.ingest_grib_utils import (
    awk_path,
    cat_gribs,
    download_and_validate_gfs_subset,
    has_records,
    output_path,
    quote_path,
    run_checked,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    archive_tmp_zarr_and_upload,
    close_store,
    configure_zarr_limits,
    download_extract_historic_archive,
    interp_time_take_blend,
    make_herbie_save_dir,
    mask_invalid_data,
    mask_invalid_refc,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
    validate_stacked_time_alignment,
)
from API.utils.storm_proc import compute_storm_fields_from_apcp_dataarray

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Constants
# ============================================================================
# Grid dimensions for GFS 0.25° global grid
GRID_LAT = 721
GRID_LON = 1440
# Time dimensions
FORECAST_TIME_STEPS = 160
HISTORIC_TIME_STEPS = 6
MAP_TIME_STEPS = 36
MAP_CHUNK_SIZE = 100
# Zarr store settings
DEFAULT_ZARR_WORKERS = 12
DEFAULT_ZARR_CONCURRENCY = 12
DEFAULT_HERBIE_RETRIES = 5
DEFAULT_HERBIE_RETRY_SLEEP = 20

# ============================================================================
# Environment Configuration
# ============================================================================

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

# Paths
forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/reya/Weather/GFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "GFS_Process")
hist_process_path = os.path.join(forecast_process_dir, "GFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod/GFS")
historic_path = os.getenv("historic_path", default="/home/reya/Weather/GFS")

ingest_version = INGEST_VERSION_STR

# Save and upload settings
save_type = os.getenv("save_type", default="Download")


def _parse_bool_env(value: str) -> bool:
    """Parse environment variable as boolean."""
    return value.lower() in ("1", "true", "yes", "on")


no_upload = _parse_bool_env(os.getenv("NO_UPLOAD", os.getenv("no_upload", "")))
force_update = _parse_bool_env(os.getenv("force_update", ""))

# AWS/S3 settings
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", DEFAULT_ZARR_WORKERS)
zarr_async_concurrency = positive_int_env(
    "zarr_async_concurrency", DEFAULT_ZARR_CONCURRENCY
)
herbie_download_retries = positive_int_env(
    "herbie_download_retries", DEFAULT_HERBIE_RETRIES
)
herbie_retry_sleep_seconds = positive_int_env(
    "herbie_retry_sleep_seconds", DEFAULT_HERBIE_RETRY_SLEEP
)

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["GFS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GFS"]

his_period = HISTORY_PERIODS["GFS"]

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
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

herbie_save_dir = make_herbie_save_dir(tmp_dir)

################################################################################################
# Forecast Data Processing
################################################################################################

T0 = time.time()

latest_run = HerbieLatest(
    model="gfs",
    n=3,
    freq="6h",
    fxx=240,
    product="pgrb2.0p25",
    verbose=False,
    priority=["aws", "nomads"],
    save_dir=herbie_save_dir,
)

base_time = latest_run.date
# base_time = pd.Timestamp("2024-03-24 06:00:00Z")

logger.info(base_time)


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/GFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/GFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if not force_update and previous_base_time >= base_time:
            logger.info("No Update to GFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/GFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/GFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if not force_update and previous_base_time >= base_time:
            logger.info("No Update to GFS, ending")
            sys.exit()

if force_update:
    logger.info("force_update enabled, bypassing No Update check")

zarr_vars = (
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
    "PRES_station",
)

#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours


# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
matchstring_su = (
    ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|PRES|VIS|GUST|CAPE|PRES):surface:.*hour fcst)"
)
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_oz = "(:TOZNE:)"
matchstring_cl = "(:(TCDC|REFC):entire atmosphere:.*hour fcst)"
matchstring_ap = "(:APCP:surface:0-[1-9]*)"
matchstring_sl = "(:(PRMSL|DSWRF):)"


# Merge matchstrings for download
match_strings = (
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

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach

# %% GFS pgrb2.0p25 download and validation

gfs_range_1 = FORECAST_LEAD_RANGES["GFS_1"]
gfs_range_2 = FORECAST_LEAD_RANGES["GFS_2"]
gfs_forecast_hours = [*gfs_range_1, *gfs_range_2]

wgrib2_exe = wgrib2_path.strip()


pgrb2_grib_files = download_and_validate_gfs_subset(
    product="pgrb2.0p25",
    search=match_strings,
    dataset_name="GFS forecast pgrb2.0p25",
    base_time=base_time,
    wgrib2_exe=wgrib2_exe,
    gfs_forecast_hours=gfs_forecast_hours,
    herbie_save_dir=herbie_save_dir,
    herbie_download_retries=herbie_download_retries,
    herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
)


# %% File names

pgrb2_merged_grib = output_path(forecast_process_path, "pgrb2_0p25_merged.grib")

apcp_norm_grib = output_path(forecast_process_path, "apcp_norm.grib")
apcp_rate_grib = output_path(forecast_process_path, "apcp_rate_mmhr.grib")
apcp_rate_nc4 = output_path(forecast_process_path, "apcp_rate_mmhr.nc4")

apcp_dt1_inv = output_path(forecast_process_path, "apcp_dt1.inv")
apcp_dt3_inv = output_path(forecast_process_path, "apcp_dt3.inv")
apcp_dt_other_inv = output_path(forecast_process_path, "apcp_dt_other.inv")

dswrf_norm_grib = output_path(forecast_process_path, "dswrf_norm.grib")
dswrf_norm_nc4 = output_path(forecast_process_path, "dswrf_norm.nc4")

other_fields_nc4 = output_path(forecast_process_path, "other_fields.nc4")

duvb_merged_grib = output_path(forecast_process_path, "duvb_merged.grib")
duvb_merged_nc4 = output_path(forecast_process_path, "duvb_merged.nc4")
duvb_norm_grib = output_path(forecast_process_path, "duvb_norm.grib")


# Clean old intermediate/output files that may be appended to or regenerated.
for path in [
    pgrb2_merged_grib,
    apcp_norm_grib,
    apcp_rate_grib,
    apcp_rate_nc4,
    apcp_dt1_inv,
    apcp_dt3_inv,
    apcp_dt_other_inv,
    dswrf_norm_grib,
    dswrf_norm_nc4,
    other_fields_nc4,
    duvb_merged_grib,
    duvb_merged_nc4,
    duvb_norm_grib,
]:
    if os.path.exists(path):
        os.remove(path)


# %% Merge pgrb2.0p25 GRIBs

cmd_merge_pgrb2 = (
    f"{cat_gribs(pgrb2_grib_files)} | "
    f"{quote_path(wgrib2_exe)} - "
    f"-grib {quote_path(pgrb2_merged_grib)}"
)

run_checked(cmd_merge_pgrb2, "Merge GFS pgrb2.0p25 GRIB files")


# %% Normalize APCP, convert accumulated precipitation to mm/hr, and write NetCDF

cmd_norm_apcp = (
    f"{quote_path(wgrib2_exe)} {quote_path(pgrb2_merged_grib)} "
    "-match ':APCP:surface:' "
    "-set_grib_type c1 "
    f"-ncep_norm {quote_path(apcp_norm_grib)}"
)

run_checked(cmd_norm_apcp, "Normalize APCP")


# Split normalized APCP inventory by accumulation interval length.
awk_prog = (
    "{ "
    "ftime = $6; "
    'split(ftime, p, " "); '
    'split(p[1], h, "-"); '
    'if (p[3] != "acc" || p[4] != "fcst") next; '
    "dt = h[2] - h[1]; "
    'if (p[2] == "day") dt = dt * 24; '
    f'if (dt == 1) print > "{awk_path(apcp_dt1_inv)}"; '
    f'else if (dt == 3) print > "{awk_path(apcp_dt3_inv)}"; '
    f'else print > "{awk_path(apcp_dt_other_inv)}"; '
    "}"
)

cmd_split_apcp_intervals = (
    f"{quote_path(wgrib2_exe)} {quote_path(apcp_norm_grib)} "
    "-s "
    "-match ':APCP:surface:' "
    "| "
    f"awk -F: {quote_path(awk_prog)}"
)

run_checked(cmd_split_apcp_intervals, "Split APCP inventory by accumulation interval")


if has_records(apcp_dt_other_inv):
    logger.warning(
        "Found APCP records with accumulation intervals other than 1 or 3 hours: %s",
        apcp_dt_other_inv,
    )


apcp_rate_written = False

for interval_hours, inventory_file in [
    (1, apcp_dt1_inv),
    (3, apcp_dt3_inv),
]:
    if not has_records(inventory_file):
        logger.warning(
            "No %s-hour APCP accumulation records found in %s.",
            interval_hours,
            inventory_file,
        )
        continue

    append_arg = " -append" if apcp_rate_written else ""

    cmd_apcp_to_rate = (
        f"cat {quote_path(inventory_file)} | "
        f"{quote_path(wgrib2_exe)} {quote_path(apcp_norm_grib)} "
        "-i "
        f"-rpn '{interval_hours}:/' "
        "-set_grib_type c1 "
        f"{append_arg} "
        f"-grib_out {quote_path(apcp_rate_grib)}"
    )

    run_checked(
        cmd_apcp_to_rate,
        f"Convert {interval_hours}-hour APCP accumulations to mm/hr",
    )

    apcp_rate_written = True


if not apcp_rate_written:
    logger.error("No APCP records were converted to precipitation rate.")
    sys.exit(1)


cmd_apcp_rate_to_nc4 = (
    f"{quote_path(wgrib2_exe)} {quote_path(apcp_rate_grib)} "
    "-nc4 "
    f"-netcdf {quote_path(apcp_rate_nc4)}"
)

run_checked(cmd_apcp_rate_to_nc4, "Convert APCP rate GRIB to NetCDF")


# %% Normalize DSWRF and write NetCDF

cmd_norm_dswrf = (
    f"{quote_path(wgrib2_exe)} {quote_path(pgrb2_merged_grib)} "
    "-match ':DSWRF:' "
    "-set_grib_type c1 "
    f"-ncep_norm {quote_path(dswrf_norm_grib)}"
)

run_checked(cmd_norm_dswrf, "Normalize DSWRF")


cmd_dswrf_to_nc4 = (
    f"{quote_path(wgrib2_exe)} {quote_path(dswrf_norm_grib)} "
    "-nc4 "
    f"-netcdf {quote_path(dswrf_norm_nc4)}"
)

run_checked(cmd_dswrf_to_nc4, "Convert DSWRF normalized GRIB to NetCDF")


# %% Convert remaining pgrb2.0p25 fields to NetCDF

cmd_other_fields_to_nc4 = (
    f"{quote_path(wgrib2_exe)} {quote_path(pgrb2_merged_grib)} "
    "-not ':APCP:surface:' "
    "-not ':DSWRF:' "
    "-nc4 "
    f"-netcdf {quote_path(other_fields_nc4)}"
)

run_checked(cmd_other_fields_to_nc4, "Convert remaining pgrb2.0p25 fields to NetCDF")


# %% Download, validate, and process UV data from pgrb2b.0p25

duvb_match_string = ":DUVB:surface:"

duvb_grib_files = download_and_validate_gfs_subset(
    product="pgrb2b.0p25",
    search=duvb_match_string,
    dataset_name="GFS forecast pgrb2b.0p25 DUVB",
    base_time=base_time,
    wgrib2_exe=wgrib2_exe,
    gfs_forecast_hours=gfs_forecast_hours,
    herbie_save_dir=herbie_save_dir,
    herbie_download_retries=herbie_download_retries,
    herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
)


cmd_merge_duvb = (
    f"{cat_gribs(duvb_grib_files)} | "
    f"{quote_path(wgrib2_exe)} - "
    f"-grib {quote_path(duvb_merged_grib)}"
)

run_checked(cmd_merge_duvb, "Merge DUVB GRIB files")


cmd_norm_duvb = (
    f"{quote_path(wgrib2_exe)} {quote_path(duvb_merged_grib)} "
    "-set_grib_type c1 "
    f"-ncep_norm {quote_path(duvb_norm_grib)}"
)

run_checked(cmd_norm_duvb, "Normalize DUVB")


cmd_duvb_to_nc4 = (
    f"{quote_path(wgrib2_exe)} {quote_path(duvb_norm_grib)} "
    "-nc4 "
    f"-netcdf {quote_path(duvb_merged_nc4)}"
)

run_checked(cmd_duvb_to_nc4, "Convert DUVB normalized GRIB to NetCDF")


# %% Merge NetCDF datasets with xarray

ds_other_fields = xr.open_mfdataset(other_fields_nc4, parallel=True)
ds_apcp_rate = xr.open_mfdataset(apcp_rate_nc4, parallel=True)
ds_duvb = xr.open_mfdataset(duvb_merged_nc4, parallel=True)
ds_dswrf = xr.open_mfdataset(dswrf_norm_nc4, parallel=True)

xarray_forecast_merged = xr.merge(
    [
        ds_dswrf,
        ds_other_fields,
        ds_apcp_rate,
        ds_duvb,
    ],
    compat="override",
)

assert len(xarray_forecast_merged.time) == len(gfs_forecast_hours), (
    "Incorrect number of timesteps! Exiting"
)

# Create a new time series
start = xarray_forecast_merged.time.min().values  # Adjust as necessary
end = xarray_forecast_merged.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hours=his_period), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hours=his_period),
            end=start - pd.Timedelta(hours=1),
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
# Set REFC values < 5 to 0
xarray_forecast_merged["REFC_entireatmosphere"] = mask_invalid_refc(
    xarray_forecast_merged["REFC_entireatmosphere"]
)

# Clean up APCP/DSWF/DUVB rates that are below 0
xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged["APCP_surface"].clip(
    min=0
)
xarray_forecast_merged["DSWRF_surface"] = xarray_forecast_merged["DSWRF_surface"].clip(
    min=0
)
xarray_forecast_merged["DUVB_surface"] = xarray_forecast_merged["DUVB_surface"].clip(
    min=0
)


# Compute nearest storm distance and direction from shared scipy/dask utilities.
distanced_stacked, directions_stacked = compute_storm_fields_from_apcp_dataarray(
    apcp_dataarray=xarray_forecast_merged["APCP_surface"],
    threshold=0.2,
    max_distance_m=None,
)

distanced_chunked = distanced_stacked.rechunk((160, process_chunk, process_chunk))
directions_chunked = directions_stacked.rechunk((160, process_chunk, process_chunk))


with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        distanced_chunked.to_zarr(
            forecast_process_path + "_stormDist.zarr",
            overwrite=True,
            compute=True,
        )
        directions_chunked.to_zarr(
            forecast_process_path + "_stormDir.zarr",
            overwrite=True,
            compute=True,
        )


# %% Save merged and processed xarray dataset to disk using zarr with compression

# Rename PRES_surface to PRES_station for clarity
# From here on out, it'll be referred to as PRES_station
xarray_forecast_merged = xarray_forecast_merged.rename({"PRES_surface": "PRES_station"})

# Save the dataset with compression and filters for all variables
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 160, "latitude": process_chunk, "longitude": process_chunk}
)

with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
    xarray_forecast_merged.to_zarr(
        forecast_process_path + "_.zarr",
        mode="w",
        consolidated=False,
        compute=True,
        chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
    )

# %% Delete to free memory
del (
    ds_other_fields,
    ds_apcp_rate,
    ds_duvb,
    ds_dswrf,
    directions_chunked,
    distanced_chunked,
    distanced_stacked,
    directions_stacked,
    xarray_forecast_merged,
)
T1 = time.time()

logger.info(T1 - T0)

# Remove forecast intermediate GRIB/NetCDF/inventory artifacts.
for path in [
    pgrb2_merged_grib,
    apcp_norm_grib,
    apcp_rate_grib,
    apcp_rate_nc4,
    apcp_dt1_inv,
    apcp_dt3_inv,
    apcp_dt_other_inv,
    dswrf_norm_grib,
    dswrf_norm_nc4,
    other_fields_nc4,
    duvb_merged_grib,
    duvb_merged_nc4,
    duvb_norm_grib,
]:
    if os.path.exists(path):
        os.remove(path)

################################################################################################
# Historic Data Processing
# Loop through historical runs (6-hour intervals) and process if not already done
################################################################################################

HISTORICAL_FORECAST_HOURS = range(1, 7)

for hours_offset in range(his_period, 0, -6):
    if save_type == "S3":
        timestamp = (base_time - pd.Timedelta(hours=hours_offset)).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        s3_path = f"{historic_path}/GFS_Hist_v3{timestamp}.zarr.tar.gz"

        # Check for a done file in S3
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            logger.info("S3 file exists, skipping: %s", s3_path)
            continue
    else:
        # Local Path Setup
        timestamp = (base_time - pd.Timedelta(hours=hours_offset)).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        local_path = f"{historic_path}/GFS_Hist_v3{timestamp}.zarr"

        # Check for a local done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info("Local file exists, skipping: %s", local_path)
            continue

    hist_run_date = base_time - pd.Timedelta(hours=hours_offset)
    hist_forecast_hours = HISTORICAL_FORECAST_HOURS

    hist_pgrb2_merged_grib = hist_process_path + "_wgrib2_merged.grib"
    hist_apcp_norm_grib = hist_process_path + "_apcp_norm.grib"
    hist_apcp_rate_nc4 = hist_process_path + "_apcp_rate_mmhr.nc4"
    hist_dswrf_norm_grib = hist_process_path + "_dswrf_norm.grib"
    hist_dswrf_norm_nc4 = hist_process_path + "_dswrf_norm.nc4"
    hist_other_fields_nc4 = hist_process_path + "_other_fields.nc4"
    hist_pgrb2_uv_merged_grib = hist_process_path + "_wgrib2_merged_UV.grib"
    hist_duvb_norm_grib = hist_process_path + "_duvb_norm.grib"
    hist_pgrb2_uv_merged_nc4 = hist_process_path + "_wgrib2_merged_UV.nc4"

    logger.info("Downloading historic data: %s", timestamp)

    # Create a range of dates for historic data going back 48 hours
    grib_list = download_and_validate_gfs_subset(
        product="pgrb2.0p25",
        search=match_strings,
        dataset_name="GFS historic pgrb2.0p25",
        run_date=hist_run_date,
        forecast_hours=hist_forecast_hours,
        base_time=base_time,
        wgrib2_exe=wgrib2_exe,
        gfs_forecast_hours=gfs_forecast_hours,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
    )

    cmd_merge_hist_pgrb2 = (
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_exe)} - "
        f"-grib {quote_path(hist_pgrb2_merged_grib)}"
    )

    run_checked(cmd_merge_hist_pgrb2, "Merge historic GFS pgrb2.0p25 GRIB files")

    cmd_norm_hist_apcp = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_pgrb2_merged_grib)} "
        "-match ':APCP:surface:' "
        "-set_grib_type c1 "
        f"-ncep_norm {quote_path(hist_apcp_norm_grib)}"
    )

    run_checked(cmd_norm_hist_apcp, "Normalize historic APCP")

    cmd_hist_apcp_rate_to_nc4 = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_apcp_norm_grib)} "
        "-nc4 "
        f"-netcdf {quote_path(hist_apcp_rate_nc4)}"
    )

    run_checked(cmd_hist_apcp_rate_to_nc4, "Convert historic APCP rate GRIB to NetCDF")

    cmd_norm_hist_dswrf = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_pgrb2_merged_grib)} "
        "-match ':DSWRF:' "
        "-set_grib_type c1 "
        f"-ncep_norm {quote_path(hist_dswrf_norm_grib)}"
    )

    run_checked(cmd_norm_hist_dswrf, "Normalize historic DSWRF")

    cmd_hist_dswrf_to_nc4 = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_dswrf_norm_grib)} "
        "-nc4 "
        f"-netcdf {quote_path(hist_dswrf_norm_nc4)}"
    )

    run_checked(
        cmd_hist_dswrf_to_nc4, "Convert historic DSWRF normalized GRIB to NetCDF"
    )

    cmd_hist_other_fields_to_nc4 = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_pgrb2_merged_grib)} "
        "-not ':APCP:surface:' "
        "-not ':DSWRF:' "
        "-nc4 "
        f"-netcdf {quote_path(hist_other_fields_nc4)}"
    )

    run_checked(
        cmd_hist_other_fields_to_nc4,
        "Convert historic non-APCP fields to NetCDF",
    )

    # Download and add UV data from the pgrb2b product
    duvb_match_string = ":DUVB:surface:"
    grib_list_uv = download_and_validate_gfs_subset(
        product="pgrb2b.0p25",
        search=duvb_match_string,
        dataset_name="GFS historic pgrb2b.0p25 DUVB",
        run_date=hist_run_date,
        forecast_hours=hist_forecast_hours,
        base_time=base_time,
        wgrib2_exe=wgrib2_exe,
        gfs_forecast_hours=gfs_forecast_hours,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
    )

    cmd_merge_hist_pgrb2_uv = (
        f"{cat_gribs(grib_list_uv)} | "
        f"{quote_path(wgrib2_exe)} - "
        f"-grib {quote_path(hist_pgrb2_uv_merged_grib)}"
    )

    run_checked(
        cmd_merge_hist_pgrb2_uv,
        "Merge historic GFS pgrb2b.0p25 DUVB GRIB files",
    )

    cmd_norm_hist_duvb = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_pgrb2_uv_merged_grib)} "
        "-set_grib_type c1 "
        f"-ncep_norm {quote_path(hist_duvb_norm_grib)}"
    )

    run_checked(cmd_norm_hist_duvb, "Normalize historic DUVB")

    cmd_hist_duvb_to_nc4 = (
        f"{quote_path(wgrib2_exe)} {quote_path(hist_duvb_norm_grib)} "
        "-nc4 "
        f"-netcdf {quote_path(hist_pgrb2_uv_merged_nc4)}"
    )

    run_checked(
        cmd_hist_duvb_to_nc4,
        "Convert historic DUVB normalized GRIB to NetCDF",
    )

    # Merge the UV data and xarrays
    # Read the netcdf file using xarray
    ds_other_fields = xr.open_mfdataset(hist_other_fields_nc4, parallel=True)
    ds_apcp_rate = xr.open_mfdataset(hist_apcp_rate_nc4, parallel=True)
    ds_historic_duvb = xr.open_mfdataset(hist_pgrb2_uv_merged_nc4, parallel=True)
    ds_historic_dswrf = xr.open_mfdataset(hist_dswrf_norm_nc4, parallel=True)

    xarray_hist_merged = xr.merge(
        [
            ds_historic_dswrf,
            ds_other_fields,
            ds_apcp_rate,
            ds_historic_duvb,
        ],
        compat="override",
    )

    # Fix things
    # Historic APCP has already been deaccumulated by wgrib2.
    xarray_hist_merged["APCP_surface"] = xarray_hist_merged["APCP_surface"].clip(min=0)

    # Storm distance and direction
    storm_distance_hist, storm_direction_hist = (
        compute_storm_fields_from_apcp_dataarray(
            apcp_dataarray=xarray_hist_merged["APCP_surface"],
            threshold=0.2,
            max_distance_m=None,
        )
    )

    # Set REFC values < 5 to 0
    xarray_hist_merged["REFC_entireatmosphere"] = mask_invalid_refc(
        xarray_hist_merged["REFC_entireatmosphere"]
    )

    # Copy back to main array
    # with ProgressBar():
    xarray_hist_merged["Storm_Distance"] = (
        ("time", "latitude", "longitude"),
        storm_distance_hist.rechunk((6, process_chunk, process_chunk)).compute(),
    )
    xarray_hist_merged["Storm_Direction"] = (
        ("time", "latitude", "longitude"),
        storm_direction_hist.rechunk((6, process_chunk, process_chunk)).compute(),
    )

    # Rechunk the rest of the variables in the merged dataset to the processing chunk size
    xarray_hist_merged = xarray_hist_merged.chunk(
        {
            "time": 6,
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    # Clear memory
    del (
        ds_other_fields,
        ds_apcp_rate,
        ds_historic_dswrf,
        ds_historic_duvb,
    )

    # Rename PRES_surface to PRES_station for clarity
    # From here on out, it'll be referred to as PRES_station
    xarray_hist_merged = xarray_hist_merged.rename({"PRES_surface": "PRES_station"})

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chunks to speed up read times
    encoding = {
        vname: {"chunks": (6, process_chunk, process_chunk)} for vname in zarr_vars[1:]
    }

    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_hist_merged.to_zarr(
            hist_process_path + "_GFS_Hist_TMP.zarr",
            mode="w",
            consolidated=False,
            encoding=encoding,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    os.remove(hist_pgrb2_merged_grib)
    os.remove(hist_apcp_norm_grib)
    os.remove(hist_apcp_rate_nc4)
    os.remove(hist_dswrf_norm_grib)
    os.remove(hist_dswrf_norm_nc4)
    os.remove(hist_other_fields_nc4)
    os.remove(hist_pgrb2_uv_merged_grib)
    os.remove(hist_duvb_norm_grib)
    os.remove(hist_pgrb2_uv_merged_nc4)

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        if no_upload:
            logger.info(
                "NO_UPLOAD enabled, skipping historic S3 archive upload: %s", s3_path
            )
            shutil.rmtree(hist_process_path + "_GFS_Hist_TMP.zarr", ignore_errors=True)
        else:
            archive_tmp_zarr_and_upload(
                tmp_zarr_path=hist_process_path + "_GFS_Hist_TMP.zarr",
                s3_path=s3_path,
                archive_member_name="GFS_Hist.zarr",
                s3=s3,
            )
    else:
        # Move to Local Path
        os.rename(hist_process_path + "_GFS_Hist_TMP.zarr", local_path)

        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info("Completed historic data processing: %s", timestamp)


################################################################################################
# Merge Historic and Forecast Datasets
################################################################################################
if save_type == "S3":
    local_temp_dir = forecast_process_path + "_s3_temp_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)

    # The function that downloads and extracts a single timestamp
    def download_and_extract(timestamp):
        # Names expected locally
        final_zarr_name = f"GFS_Hist_v3{timestamp}.zarr"
        extracted_path = download_extract_historic_archive(
            s3=s3,
            historic_path=historic_path,
            final_zarr_name=final_zarr_name,
            extracted_store_name="GFS_Hist.zarr",
            local_temp_dir=local_temp_dir,
            expected_vars=zarr_vars,
        )
        if extracted_path is None:
            tqdm.write(f"Error: GFS_Hist.zarr not found inside archive for {timestamp}")
        return extracted_path

    # Generate target timestamps
    timestamps = [
        (base_time - pd.Timedelta(hours=hours_offset)).strftime("%Y%m%dT%H%M%SZ")
        for hours_offset in range(his_period, 1, -6)
    ]

    logger.info("Downloading and extracting %d archives from S3...", len(timestamps))

    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(
            tqdm(
                executor.map(download_and_extract, timestamps),
                total=len(timestamps),
                desc="S3 Archive Sync",
            )
        )

    # Filter out the missing files (None values) and keep the valid paths
    historic_zarr_paths = [path for path in results if path is not None]
else:
    historic_zarr_paths = [
        f"{historic_path}/GFS_Hist_v3{(base_time - pd.Timedelta(hours=hours_offset)).strftime('%Y%m%dT%H%M%SZ')}.zarr"
        for hours_offset in range(his_period, 1, -6)
    ]

# Dask Setup
dask_var_arrays_list = []
dask_interp_arrays = []

for var_idx, dask_var in enumerate(zarr_vars[:]):
    for historic_zarr_path in historic_zarr_paths:
        # If not found in array, use MISSING_DATA to show missing
        try:
            dask_var_arrays_list.append(
                da.from_zarr(historic_zarr_path, component=dask_var, inline_array=True)
            )
        # Add a fallback in case of a FileNotFoundError
        except FileNotFoundError:
            logger.info("File not found, adding NaN array for: %s", historic_zarr_path)
            dask_var_arrays_list.append(
                da.full(
                    (HISTORIC_TIME_STEPS, GRID_LAT, GRID_LON), MISSING_DATA
                ).rechunk((HISTORIC_TIME_STEPS, process_chunk, process_chunk))
            )

    dask_var_arrays_stacked = da.stack(
        dask_var_arrays_list, allow_unknown_chunksizes=True
    )

    if dask_var == "Storm_Distance":
        dask_forecast_array = da.from_zarr(forecast_process_path + "_stormDist.zarr")
    elif dask_var == "Storm_Direction":
        dask_forecast_array = da.from_zarr(forecast_process_path + "_stormDir.zarr")
    else:
        dask_forecast_array = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

    if dask_var == "time":
        # Create a time array with the same shape
        # This is because multiple steps are stored in each file
        dask_var_arrays_reshaped = da.reshape(
            dask_var_arrays_stacked,
            (dask_var_arrays_stacked.shape[0] * dask_var_arrays_stacked.shape[1], 1),
            merge_chunks=False,
        )
        dask_times_concatenated = da.concatenate(
            (da.squeeze(dask_var_arrays_reshaped), dask_forecast_array), axis=0
        ).astype("float32")

        # Get times as numpy
        times_array = dask_times_concatenated.compute()
        validate_stacked_time_alignment(stacked_timesUnix, times_array)

        output_array = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(times_array, axis=1), axis=1),
                (1, GRID_LAT, GRID_LON),
            )
        ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

        dask_interp_arrays.append(output_array)

    else:
        dask_var_arrays_reshaped = da.reshape(
            dask_var_arrays_stacked,
            (
                dask_var_arrays_stacked.shape[0] * dask_var_arrays_stacked.shape[1],
                GRID_LAT,
                GRID_LON,
            ),
            merge_chunks=False,
        )
        output_array = da.concatenate(
            (dask_var_arrays_reshaped, dask_forecast_array), axis=0
        )

        dask_interp_arrays.append(
            output_array[:, :, :]
            .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
            .astype("float32")
        )

    dask_var_arrays_list = []
    logger.info("Processed variable: %s", dask_var)

# Merge the arrays into a single 4D array
merged_arrays = da.stack(dask_interp_arrays, axis=0)

# Mask out invalid data
# Ignore storm distance, since it can reach very high values that are still correct
merged_arrays_masked = mask_invalid_data(
    merged_arrays, ignoreAxis=[zarr_vars.index("Storm_Distance")]
)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
    merged_arrays_masked.to_zarr(
        forecast_process_path + "_stack.zarr",
        overwrite=True,
        compute=True,
    )

# Read in stacked 4D array back in
stacked_array_disk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/GFS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/GFS.zarr")


#
# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Pad to chunk size
# 3. Create the zarr array
# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array

with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        # 1. Interpolate the stacked array to be hourly along the time axis
        stacked_array_interp = interp_time_take_blend(
            stacked_array_disk,
            stacked_timesUnix=stacked_timesUnix,
            hourly_timesUnix=hourly_timesUnix,
            dtype="float32",
            fill_value=np.nan,
        )

        # 2. Pad to chunk size
        stacked_array_padded = pad_to_chunk_size(stacked_array_interp, final_chunk)

        # 3. Create the zarr array
        zarr_array = zarr.create_array(
            store=zarr_store,
            shape=(
                len(zarr_vars),
                len(hourly_timesUnix),
                stacked_array_padded.shape[2],
                stacked_array_padded.shape[3],
            ),
            chunks=(len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )

        # 4. Rechunk it to match the final array
        # 5. Write it out to the zarr array
        stacked_array_padded.round(5).rechunk(
            (len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk)
        ).to_zarr(zarr_array, overwrite=True, compute=True)


close_store(zarr_store)

# Rechunk map data for faster web access
# Variables included: time(0), TMP(4), UGRD(8), VGRD(9), PRATE(10), PACCUM(11),
#                     CRAIN(12), CICEP(13), CSNOW(14), CFRZR(15), REFC(21)
# Map extent: -12 to +24 hours (36 hours total)
# Map chunk size: 100x100 pixels for fast tiling

MAP_VAR_INDICES = [0, 4, 8, 9, 10, 11, 12, 13, 14, 15, 21]

# Add padding for map chunking (100x100)
stacked_array_maps = pad_to_chunk_size(stacked_array_disk, MAP_CHUNK_SIZE)

if save_type == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/GFS_Maps.zarr.zip", mode="a"
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(forecast_process_dir + "/GFS_Maps.zarr")

for var_idx in MAP_VAR_INDICES:
    # Create a zarr backed dask array
    zarr_array = zarr.create_array(
        store=zarr_store_maps,
        name=zarr_vars[var_idx],
        shape=(
            MAP_TIME_STEPS,
            stacked_array_maps.shape[2],
            stacked_array_maps.shape[3],
        ),
        chunks=(MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
    )

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            da.rechunk(
                stacked_array_maps[var_idx, his_period - 12 : his_period + 24, :, :],
                (MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
            ).to_zarr(zarr_array, overwrite=True, compute=True)

    logger.info("Created map data for %s", zarr_vars[var_idx])


close_store(zarr_store_maps)

# %% Upload to S3
if save_type == "S3":
    # Write most recent forecast time
    with open(forecast_process_dir + "/GFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)
    if no_upload:
        logger.info("NO_UPLOAD enabled, skipping forecast S3 publish")
    else:
        # Upload to S3
        s3.put_file(
            forecast_process_dir + "/GFS.zarr.zip",
            forecast_path + "/" + ingest_version + "/GFS.zarr.zip",
        )
        s3.put_file(
            forecast_process_dir + "/GFS_Maps.zarr.zip",
            forecast_path + "/" + ingest_version + "/GFS_Maps.zarr.zip",
        )

        s3.put_file(
            forecast_process_dir + "/GFS.time.pickle",
            forecast_path + "/" + ingest_version + "/GFS.time.pickle",
        )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/GFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/GFS.time.pickle",
        forecast_path + "/" + ingest_version + "/GFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GFS.zarr",
        forecast_path + "/" + ingest_version + "/GFS.zarr",
        dirs_exist_ok=True,
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GFS_Maps.zarr",
        forecast_path + "/" + ingest_version + "/GFS_Maps.zarr",
        dirs_exist_ok=True,
    )
# Clean up
if no_upload:
    logger.info("NO_UPLOAD enabled, skipping cleanup of %s", forecast_process_dir)
else:
    shutil.rmtree(forecast_process_dir)

# Timing
T1 = time.time()
logger.info(T1 - T0)

# Test Read
# G = zarr.open(forecast_path + "/" + ingest_version + "/GFS.zarr", read_only=True)
# G.info
