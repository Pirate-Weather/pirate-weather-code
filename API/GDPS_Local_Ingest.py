# %% Script to test FastHerbie.py to download GDPS data
# Alexander Rey, April 2026

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import re

# import os
# from pathlib import Path
#
# defs = Path("/mnt/c/Users/REYA/Documents/Repo/pirate-weather-code/.build/ingest-test/toolchain/share/eccodes/definitions")
# samples = Path("/mnt/c/Users/REYA/Documents/Repo/pirate-weather-code/.build/ingest-test/toolchain/share/eccodes/samples")
#
# print("defs exists:", defs.exists())
# print("boot exists:", (defs / "boot.def").exists())
# print("boot readable:", (defs / "boot.def").read_text()[:20])
# print("samples exists:", samples.exists())
# print("GRIB2 sample exists:", (samples / "GRIB2.tmpl").exists())
#
# os.environ["ECCODES_DIR"] = "/mnt/c/Users/REYA/Documents/Repo/pirate-weather-code/.build/ingest-test/toolchain"
# os.environ["ECCODES_DEFINITION_PATH"] = str(defs)
# os.environ["ECCODES_SAMPLES_PATH"] = str(samples)
# os.environ["ECCODES_PYTHON_USE_FINDLIBS"] = "1"
#
# import eccodes
#
# print("api:", eccodes.codes_get_api_version())
#
# h = eccodes.codes_grib_new_from_samples("GRIB2")
# print("sample handle:", h)
# eccodes.codes_release(h)

# Env setup
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest
from tqdm import tqdm

from API.api_utils import estimate_visibility_from_rh_pr
from API.constants.api_const import U_REF
from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR, MISSING_DATA
from API.ingest_grib_utils import (
    cat_gribs,
    download_and_validate_gfs_subset,
    quote_path,
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
    pad_to_chunk_size,
    positive_int_env,
    run_command,
    tune_nofile_limit,
)
from API.utils.storm_proc import compute_storm_fields_from_apcp_dataarray

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GDPS")
forecast_process_path = forecast_process_dir + "/GDPS_Process"
hist_process_path = forecast_process_dir + "/GDPS_Historic"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GDPS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/GDPS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)
herbie_download_retries = positive_int_env("herbie_download_retries", 5)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)
skip_gdps_wgrib2_validation = os.getenv(
    "skip_gdps_wgrib2_validation", "true"
).lower() in {"1", "true", "yes", "on"}

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["GDPS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GDPS"]

his_period = HISTORY_PERIODS["GDPS"]

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


T0 = time.time()

latest_run = HerbieLatest(
    model="gdps",
    priority=["msc"],
    periods=7,
    fxx=240,
    product="15km",
    verbose=True,
    variable="AirTemp",
    level="AGL-2m",
    save_dir=herbie_save_dir,
)

base_time = latest_run.date


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/GDPS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/GDPS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to GDPS, ending")
            raise

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/GDPS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/GDPS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to GDPS, ending")
            raise

zarr_vars = (
    "time",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "WIND_10maboveground",
    "WDIR_10maboveground",
    "GUST_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "UVI_surface",
    "DSWRF_surface",
    "PRES_surface",
    "TCDC_surface",
    "PRMSL_meansealevel",
    "TCIOZ_surface",
    "PTYPE_surface",
    "CAPE_surface",
    "CIN_surface",
    "VVEL_500mb",
    "KX_surface",
)

#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours

# Define the variables to download as a dictionary of variable and level pairs to match in the grib files
match_strings = [
    {"variable": "AirTemp", "level": "AGL-2m"},
    {"variable": "DewPoint", "level": "AGL-2m"},
    {"variable": "RelativeHumidity", "level": "AGL-2m"},
    {"variable": "WindSpeed", "level": "AGL-10m"},
    {"variable": "WindDir", "level": "AGL-10m"},
    {"variable": "WindGust", "level": "AGL-10m"},
    {"variable": "PrecipRate", "level": "Sfc"},
    {"variable": "Precip-Accum", "level": "Sfc"},
    {"variable": "PrecipType-Instant", "level": "Sfc"},
    {"variable": "UVIndex", "level": "Sfc"},
    {"variable": "DownwardShortwaveRadiationFlux-Accum", "level": "Sfc"},
    {"variable": "CAPE", "level": "Sfc"},
    {"variable": "Pressure", "level": "Sfc"},
    {"variable": "TotalCloudCover", "level": "Sfc"},
    {"variable": "CIN", "level": "Sfc"},
    {"variable": "Pressure", "level": "MSL"},
    {"variable": "VerticalVelocity", "level": "IsbL-0500"},
    {"variable": "KIndex", "level": "Sfc"},
    {"variable": "O3", "level": "EAtm"},
]
# Range should be 1 hour through hour 83, then 3 hourly through hour 240.
gdps_range_1 = FORECAST_LEAD_RANGES["GDPS_1"]
gdps_range_2 = FORECAST_LEAD_RANGES["GDPS_2"]
gdps_file_range = [*gdps_range_1, *gdps_range_2]
gdps_reduced_file_range = [
    hour
    for hour in gdps_file_range
    if hour % 3 == 0 and (hour <= 168 or hour % 6 == 0)
]
gdps_3_hour_file_range = [hour for hour in gdps_file_range if hour % 3 == 0]
gdps_expected_forecast_hours = {
    ("PrecipType-Instant", "Sfc"): gdps_reduced_file_range,
    ("CAPE", "Sfc"): gdps_reduced_file_range,
    ("CIN", "Sfc"): gdps_reduced_file_range,
    ("KIndex", "Sfc"): gdps_reduced_file_range,
    ("VerticalVelocity", "IsbL-0500"): gdps_3_hour_file_range,
}
gdps_excluded_stats_variables = {
    ("DownwardShortwaveRadiationFlux-Accum", "Sfc"): ["DSWRF"],
    ("CIN", "Sfc"): ["CIN"],
}

# MSC models have each variable in a separate file, so we loop through the variables and levels to download each one and then merge them later
all_files = []
expected_total = 0
for g in match_strings:
    expected_forecast_hours = gdps_expected_forecast_hours.get(
        (g["variable"], g["level"]), gdps_file_range
    )
    excluded_stats_variables = gdps_excluded_stats_variables.get(
        (g["variable"], g["level"])
    )
    grib_files = download_and_validate_gfs_subset(
        model="gdps",
        product="15km",
        search=None,
        dataset_name=f"GDPS forecast {g['variable']}:{g['level']}",
        base_time=base_time,
        wgrib2_exe=wgrib2_path.strip(),
        forecast_hours=gdps_file_range,
        expected_forecast_hours=expected_forecast_hours,
        excluded_stats_variables=excluded_stats_variables,
        skip_wgrib2_validation=skip_gdps_wgrib2_validation,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
        herbie_kwargs={
            "variable": g["variable"],
            "level": g["level"],
            "verbose": True,
        },
    )

    all_files += grib_files
    expected_total += len(expected_forecast_hours)

    # Log that the download was completed for this variable and level
    logger.info(
        f"Download completed for GDPS forecast {g['variable']}:{g['level']}, "
        f"{len(grib_files)} files downloaded."
    )

# Deduplicate and sanity-check total files
all_files = sorted(set(all_files))
if len(all_files) < expected_total:
    logger.error(
        f"Download incomplete, expected at least {expected_total} files but got {len(all_files)}"
    )
    sys.exit(1)

# Create ordered list of downloaded grib files from collected paths
grib_list = all_files


# Sort the list
grib_list_sort = sorted(
    grib_list,
    key=lambda f: int(re.search(r"_PT(\d+)H", str(f)).group(1)),
)

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    f"{cat_gribs(grib_list_sort)} | "
    f"{quote_path(wgrib2_path.strip())} - "
    f"-netcdf {quote_path(forecast_process_path + '_wgrib2_merged.nc')}"
)


# Run wgrib2
sp_out = run_command(cmd)
if sp_out.returncode != 0:
    logger.error(sp_out.stderr)
    raise


#%% Read the merged netcdf file using xarray (single combined file)
xarray_forecast_merged = xr.open_dataset(forecast_process_path + "_wgrib2_merged.nc")

if len(xarray_forecast_merged.time) != len(gdps_file_range):
    raise ValueError("Incorrect number of timesteps! Exiting")

# Determine grid size from merged dataset (supports rotated grids)
NY = xarray_forecast_merged.dims.get(
    "latitude", xarray_forecast_merged["latitude"].size
)
NX = xarray_forecast_merged.dims.get(
    "longitude", xarray_forecast_merged["longitude"].size
)

# Create a new time series
start = xarray_forecast_merged.time.min().values  # Adjust as necessary
end = xarray_forecast_merged.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(his_period, "h"), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(his_period, "h"),
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


#TODO:
# Daccum DownwardShortwaveRadiationFlux
# Check for 3 hourly DownwardShortwaveRadiationFlux issues
# Something about UVIndex
# During merge, ensure nan gets interpoalted over


# Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
APCP_surface_tmp = da.diff(
    xarray_forecast_merged["APCP_surface"],
    axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
    prepend=0,
)

# Using the difference between times in the xarray, convert from 3-hourly to 1-hourly
forecast_time_steps = (
    xarray_forecast_merged.time.diff("time")
    / np.timedelta64(1, "h")
).values.astype("float32")
forecast_time_steps = np.insert(forecast_time_steps, 0, forecast_time_steps[0])
APCP_surface_tmp = APCP_surface_tmp / forecast_time_steps[:, None, None]

xarray_forecast_merged["APCP_surface"].data = APCP_surface_tmp

# Clip precipitation to >= 0
xarray_forecast_merged["APCP_surface"].data = da.clip(
    xarray_forecast_merged["APCP_surface"].data, 0, None
)


# Save the dataset with compression and filters for all variables
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 136, "latitude": process_chunk, "longitude": process_chunk}
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
    APCP_surface_tmp,
    xarray_forecast_merged,
)
T1 = time.time()

logger.info(T1 - T0)
os.remove(forecast_process_path + "_wgrib2_merged.nc")

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 12 hour runs
for i in range(his_period, 0, -12):
    if save_type == "S3":
        s3_path = (
            historic_path
            + "/GDPS_Hist_v3"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr.tar.gz"
        )

        # Check for a done file in S3
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            logger.info("File already exists in S3, skipping download for: %s", s3_path)
            continue
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/GDPS_Hist_v3"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a local done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info(
                "File already exists locally, skipping download for: %s", local_path
            )
            continue

    logger.info(
        "Downloading: %s",
        (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"),
    )

    hist_run_date = base_time - pd.Timedelta(hours=i)

    # Create a range of forecast lead times
    # Go from 1 to 12 to account for the weird prate approach
    all_files = []
    for g in match_strings:
        expected_forecast_hours = gdps_expected_forecast_hours.get(
            (g["variable"], g["level"]), gdps_file_range
        )

        expected_forecast_hours = [hour for hour in expected_forecast_hours if hour <= 12]

        grib_files = download_and_validate_gfs_subset(
            model="gdps",
            product="15km",
            search=None,
            dataset_name=f"GDPS forecast {g['variable']}:{g['level']}",
            base_time=hist_run_date,
            wgrib2_exe=wgrib2_path.strip(),
            forecast_hours=expected_forecast_hours,
            skip_wgrib2_validation=skip_gdps_wgrib2_validation,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
            herbie_kwargs={
                "variable": g["variable"],
                "level": g["level"],
                "verbose": True,
            },
        )

        all_files += grib_files
        expected_total += len(expected_forecast_hours)

        # Log that the download was completed for this variable and level
        logger.info(
            f"Download completed for GDPS forecast {g['variable']}:{g['level']}, "
            f"{len(grib_files)} files downloaded."
        )

    # Create ordered list of downloaded grib files from collected paths
    grib_list = all_files

    # Sort the list
    grib_list_sort = sorted(
        grib_list,
        key=lambda f: int(re.search(r"_PT(\d+)H", str(f)).group(1)),
    )

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        f"{cat_gribs(grib_list_sort)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-netcdf {quote_path(hist_process_path + '_wgrib2_merged.nc')}"
    )

    # Run wgrib2
    sp_out = run_command(cmd)
    if sp_out.returncode != 0:
        logger.error(sp_out.stderr)
        raise

    # Read the merged netcdf file using xarray (single combined file)
    xarray_hist_merged = xr.open_dataset(hist_process_path + "_wgrib2_merged.nc")

    # Fix things
    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time, every 6 hours
    apcpProc = xarray_hist_merged["APCP_surface"].values

    apcpProcHour = np.diff(apcpProc, axis=0, prepend=0)

    xarray_hist_merged["APCP_surface"] = xarray_hist_merged[
        "APCP_surface"
    ].copy(data=apcpProcHour)

    # Clip precipitation to >= 0
    xarray_hist_merged["APCP_surface"] = xarray_hist_merged[
        "APCP_surface"
    ].clip(min=0)
    
    # Clear memory
    del (apcpProc, apcpProcHour)

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chunks to speed up read times
    # Small fix for PRES_station/ PRES_surface
    encoding = {
        vname: {"chunks": (12, process_chunk, process_chunk)} for vname in zarr_vars[1:]
    }


    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_hist_merged.to_zarr(
            hist_process_path + "_GDPS_Hist_TMP.zarr",
            mode="w",
            consolidated=False,
            encoding=encoding,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged.nc")

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        archive_tmp_zarr_and_upload(
            tmp_zarr_path=hist_process_path + "_GDPS_Hist_TMP.zarr",
            s3_path=s3_path,
            archive_member_name="GDPS_Hist.zarr",
            s3=s3,
        )
    else:
        # Move to Local Path
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.rename(hist_process_path + "_GDPS_Hist_TMP.zarr", local_path)

        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))


# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
if save_type == "S3":
    local_temp_dir = forecast_process_path + "_s3_temp_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)

    # The function that downloads and extracts a single timestamp
    def download_and_extract(timestamp):
        # Names expected locally
        final_zarr_name = f"GDPS_Hist_v3{timestamp}.zarr"
        extracted_path = download_extract_historic_archive(
            s3=s3,
            historic_path=historic_path,
            final_zarr_name=final_zarr_name,
            extracted_store_name="GDPS_Hist.zarr",
            local_temp_dir=local_temp_dir,
        )
        if extracted_path is None:
            tqdm.write(
                f"Error: GDPS_Hist.zarr not found inside archive for {timestamp}"
            )
        return extracted_path

    # Generate target timestamps
    timestamps = [
        (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        for i in range(his_period, 0, -12)
    ]

    logger.info("Phase 1: Downloading and extracting %s archives...", len(timestamps))

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
    ncLocalWorking_paths = [path for path in results if path is not None]
else:
    ncLocalWorking_paths = [
        historic_path
        + "/GDPS_Hist_v3"
        + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        + ".zarr"
        for i in range(his_period, 0, -12)
    ]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(zarr_vars[:]):
    for local_ncpath in ncLocalWorking_paths:
        daskVarArrays.append(
            da.from_zarr(local_ncpath, component=dask_var, inline_array=True)
        )


    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

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
                (1, NY, NX),
            )
        ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack,
            (daskVarArraysStack.shape[0] * daskVarArraysStack.shape[1], NY, NX),
            merge_chunks=False,
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
            .astype("float32")
        )

    daskVarArrays = []

    logger.info(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Mask out invalid data
# Ignore storm distance, since it can reach very high values that are still correct
daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/GDPS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/GDPS.zarr")


#
# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Pad to chunk size
# 3. Create the zarr array
# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array

with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
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


close_store(zarr_store)

# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/GDPS.zarr.zip",
        forecast_path + "/" + ingest_version + "/GDPS.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/GDPS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/GDPS.time.pickle",
        forecast_path + "/" + ingest_version + "/GDPS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/GDPS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/GDPS.time.pickle",
        forecast_path + "/" + ingest_version + "/GDPS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GDPS.zarr",
        forecast_path + "/" + ingest_version + "/GDPS.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

# Timing
T1 = time.time()
logger.info(T1 - T0)
