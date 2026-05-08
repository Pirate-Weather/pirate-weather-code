# %% Script to test FastHerbie.py to download ECMWF data
# Alexander Rey, March 2025

# %% Import modules
import logging
import os

# os.environ["ECCODES_DEFINITION_PATH"] = (
#    "/home/ubuntu/eccodes-2.40.0-Source/definitions/"
# )
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
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import FastHerbie, HerbieLatest, Path

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    archive_tmp_zarr_and_upload,
    configure_zarr_limits,
    download_extract_historic_archive,
    close_store,
    interp_time_take_blend,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
    validate_grib_stats,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/ECMWF_AIFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "ECMWF_AIFS_Process")
hist_process_path = os.path.join(forecast_process_dir, "ECMWF_AIFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/ECMWF_AIFS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/ECMWF_AIFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["ECMWF"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["ECMWF"]

his_period = HISTORY_PERIODS["ECMWF_AIFS"]

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

# %% Define base time from the most recent run
T0 = time.time()

latest_run = HerbieLatest(
    model="aifs",
    n=3,
    freq="6h",
    fxx=240,
    product="oper",
    verbose=True,
    priority=["aws", "ecmwf"],
    save_dir=tmp_dir,
)

base_time = latest_run.date
# Base date for testing
# base_time = pd.Timestamp("2025-11-05 00:00:00")

logger.info(base_time)


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to ECMWF AIFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to AIFS, ending")
            sys.exit()

zarr_vars = (
    "time",
    "msl",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "ptype",
    "tcc",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
)


#####################################################################################################
# %% Download AIFS data using Herbie Latest
# Start with the ensemble forecast
# Find the latest run with 240 hours


# Create a range of forecast lead times
aifs_range = FORECAST_LEAD_RANGES["ECMWF_AIFS"]

# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aifs",
    fxx=aifs_range,
    product="enfo",
    verbose=True,
    save_dir=tmp_dir,
)

match_string_enfo = r":((tp|sf):sfc:\d+):"
ens_paths = FH_forecastsub.download(match_string_enfo, verbose=False)

grib_list = [
    str(Path(x.get_localFilePath(match_string_enfo)).expand())
    for x in FH_forecastsub.file_exists
]

# Perform a check if any data seems to be invalid
cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
validate_grib_stats(grib_check)
logger.info("Grib files passed validation, proceeding with processing")


ens_mf = xr.open_mfdataset(
    ens_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
).sortby("step")

ens_mf["tpd"] = ens_mf["tp"].diff(dim="step")

# Set the first difference to the first accumulation
ens_mf["tpd"] = xr.where(
    ens_mf.step == ens_mf.step.isel(step=0), ens_mf["tp"].isel(step=0), ens_mf["tpd"]
)

ens_mf["sfd"] = ens_mf["sf"].diff(dim="step")

# Set the first difference to the first accumulation
ens_mf["sfd"] = xr.where(
    ens_mf.step == ens_mf.step.isel(step=0), ens_mf["sf"].isel(step=0), ens_mf["sfd"]
)

# AIFS outputs 6-hour accumulations; convert to hourly rate
ens_mf["tpd"] = ens_mf["tpd"] / 6
ens_mf["sfd"] = ens_mf["sfd"] / 6

# Find the probability of precipitation greater than 0.1 mm/h (0.0001) m/h across all members
X3_Precipitation_Prob = (ens_mf["tpd"] > 0.0001).sum(dim="number") / ens_mf.sizes[
    "number"
]

# Find the standard deviation of precipitation accumulation across all members
X3_Precipitation_StdDev = ens_mf["tpd"].std(dim="number")

# Find the average precipitation accumulation across all members
X3_Precipitation_Mean = ens_mf["tpd"].mean(dim="number")

# Find the average snowfall accumulation across all members
X3_Snowfall_Mean = ens_mf["sfd"].mean(dim="number")

# Find the type of precipitation based on the snowfall and total precipitation
# If mean snow is greater than 50% of total precipitation, classify as snow, else rain, 0 if no precipitation
# Use grib types: https://codes.ecmwf.int/grib/format/grib2/ctables/4/201/, 1 is rain, 5 is snow
X3_pType = xr.where(
    X3_Precipitation_Mean > 0,
    xr.where(X3_Snowfall_Mean > X3_Precipitation_Mean *0.5 , 5, 1),
    0)



# Merge into a new xarray dataset
xr_ensoOut = xr.Dataset(
    {
        "Precipitation_Prob": X3_Precipitation_Prob,
        "APCP_StdDev": X3_Precipitation_StdDev,
        "APCP_Mean": X3_Precipitation_Mean,
        "ptype": X3_pType,
    },
    coords={
        "time": ens_mf["step"],
        "latitude": ens_mf["latitude"],
        "longitude": ens_mf["longitude"],
    },
)

#####################################################################################################
# %% Download Base IFS data using Herbie Latest
# Find the latest run with 240 hours

# Define the subset of variables to download as a list of strings
# 2 m level – use ECMWF’s 2t (temperature) and 2d (dew point)
matchstring_2m = "(:(2d|2t):)"
matchstring_10m = "(:(10u|10v):)"
matchstring_sl = "(:(msl):)"
matchstring_tcc = "(:(tcc):)"


# Merge matchstrings for download
match_strings = (
    matchstring_2m
    + "|"
    + matchstring_10m
    + "|"
    + matchstring_sl
    + "|"
    + matchstring_tcc
)


# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aifs",
    fxx=aifs_range,
    product="oper",
    verbose=False,
    priority=["aws", "ecmwf"],
    save_dir=tmp_dir,
)

# Download the subsets
aifs_paths = FH_forecastsub.download(match_strings, verbose=False)

grib_list = [
    str(Path(x.get_localFilePath(match_strings)).expand())
    for x in FH_forecastsub.file_exists
]

# Perform a check if any data seems to be invalid
cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
validate_grib_stats(grib_check)
logger.info("Grib files passed validation, proceeding with processing")


aifs_mf_2 = xr.open_mfdataset(
    aifs_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [2]}
    },
).sortby("step")
aifs_mf_10 = xr.open_mfdataset(
    aifs_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [10]}
    },
).sortby("step")

aifs_mf_msl = xr.open_mfdataset(
    aifs_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"}},
).sortby("step")

aifs_mf_atm = xr.open_mfdataset(
    aifs_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={"filter_by_keys": {"typeOfLevel": "entireAtmosphere"}},
).sortby("step")

# Combine the datasets
aifs_mf = xr.merge(
    [aifs_mf_2, aifs_mf_10, aifs_mf_msl, aifs_mf_atm], compat="override"
)


# %% Merge the ENSO and OPER data

xarray_forecast_merged = xr.merge(
    [aifs_mf, xr_ensoOut], compat="override", join="outer"
)


assert len(xarray_forecast_merged.step) == len(aifs_range), (
    "Incorrect number of timesteps! Exiting"
)

# Replace the step variable with a time variable using the start time and the step values
xarray_forecast_merged = (
    xarray_forecast_merged.assign_coords(
        time=(
            "step",
            pd.to_datetime(xarray_forecast_merged.time.values)
            + pd.to_timedelta(xarray_forecast_merged.step.values, unit="h"),
        )
    )
    .swap_dims({"step": "time"})
    .drop_vars("step")
)

## Create a new time series for the interpolation target
end = xarray_forecast_merged.time[-1].values
new_hourly_time = pd.date_range(
    start=pd.to_datetime(base_time) - pd.Timedelta(hours=his_period), end=end, freq="h"
)

# Get the actual stacked times from the concatenated dataset (to be created later)
# This will be computed after loading and concatenating historical + forecast data
# For now, just compute the hourly target times
unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second


# Chunk and save as zarr
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": len(xarray_forecast_merged.time), "latitude": process_chunk, "longitude": process_chunk}
)

with ProgressBar():
    xarray_forecast_merged.to_zarr(
        forecast_process_path + "_merged.zarr",
        mode="w",
        consolidated=False,
        compute=True,
    )


# %% Delete to free memory
del (
    xarray_forecast_merged,
    aifs_mf,
    aifs_mf_2,
    aifs_mf_10,
    aifs_mf_msl,
    aifs_mf_atm,
    ens_mf,
    xr_ensoOut,
)

T1 = time.time()
logger.info(T1 - T0)

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(his_period, -1, -6):
    if save_type == "S3":
        # S3 Path Setup
        s3_path = (
            historic_path
            + "/ECMWF_AIFS_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr.tar.gz"
        )
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            logger.info("File already exists in S3, skipping download for: %s", s3_path)
            continue

    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/ECMWF_AIFS_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a loca done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info(
                "File already exists in S3, skipping download for: %s", local_path
            )
            continue

    logger.info(
        "Downloading: %s",
        (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"),
    )

    # Create a range of dates for historic data going back 48 hours
    # Add the extra 6 hours to ensure we get the full 48 hours of data since AIFS outputs 6 hour accumulations
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(hours=i) - pd.Timedelta(hours=6),
        periods=1,
        freq="6h",
    )
    # Create a range of forecast lead times
    # Only want step 6
    fxx = [6]


    ## Ensemble
    # Create FastHerbie object
    FH_histsub_ens = FastHerbie(
        DATES,
        model="aifs",
        fxx=fxx,
        product="enfo",
        verbose=True,
        save_dir=tmp_dir,
    )

    match_string_enfo = r":((tp|sf):sfc:\d+):"
    ens_paths_his = FH_histsub_ens.download(match_string_enfo, verbose=False)

    grib_list = [
        str(Path(x.get_localFilePath(match_string_enfo)).expand())
        for x in FH_histsub_ens.file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    validate_grib_stats(grib_check)
    logger.info("Grib files passed validation, proceeding with processing")


    ens_mf_his = xr.open_mfdataset(
        ens_paths_his,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
    ).sortby("step")

    # AIFS outputs 6-hour accumulations; convert to hourly rate.
    # No difference since only one step
    ens_mf_his["tpd"] = ens_mf_his["tp"] / 6
    ens_mf_his["sfd"] = ens_mf_his["sf"] / 6

    # Find the probability of precipitation greater than 0.1 mm/h (0.0001) m/h across all members
    X3_Precipitation_Prob_His = (ens_mf_his["tpd"] > 0.0001).sum(dim="number") / ens_mf_his.sizes[
        "number"
    ]

    # Find the standard deviation of precipitation accumulation across all members
    X3_Precipitation_StdDev_His = ens_mf_his["tpd"].std(dim="number")

    # Find the average precipitation accumulation across all members
    X3_Precipitation_Mean_His = ens_mf_his["tpd"].mean(dim="number")

    # Find the average snowfall accumulation across all members
    X3_Snowfall_Mean_His = ens_mf_his["sfd"].mean(dim="number")

    # Find the type of precipitation based on the snowfall and total precipitation
    # If mean snow is greater than 50% of total precipitation, classify as snow, else rain, 0 if no precipitation
    # Use grib types: https://codes.ecmwf.int/grib/format/grib2/ctables/4/201/, 1 is rain, 5 is snow
    X3_pType_His = xr.where(
        X3_Precipitation_Mean_His > 0,
        xr.where(X3_Snowfall_Mean_His > X3_Precipitation_Mean_His *0.5 , 5, 1),
        0)



    # Merge into a new xarray dataset
    xr_ensoOut_His = xr.Dataset(
        {
            "Precipitation_Prob": X3_Precipitation_Prob_His,
            "APCP_StdDev": X3_Precipitation_StdDev_His,
            "APCP_Mean": X3_Precipitation_Mean_His,
            "ptype": X3_pType_His,
        },
        coords={
            "time": ens_mf_his["step"],
            "latitude": ens_mf_his["latitude"],
            "longitude": ens_mf_his["longitude"],
        },
    )



    ## Deterministic
    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES, model="aifs", fxx=fxx, product="oper", verbose=False, save_dir=tmp_dir
    )

    # Download the subsets
    # Start with oper
    aifs_hisgribs = FH_histsub.download(match_strings, verbose=False)

    # Check for download length
    if len(FH_histsub.file_exists) != len(fxx):
        logger.error(
            "Download failed, expected %d files but got %d",
            len(fxx),
            len(FH_histsub.file_exists),
        )
        sys.exit(1)

    # Create list of downloaded grib files
    grib_list = [
        str(Path(x.get_localFilePath(match_strings)).expand())
        for x in FH_histsub.file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + " - " + " -s -stats"

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    validate_grib_stats(grib_check)
    logger.info("Grib files passed validation, proceeding with processing")

    # Created merged xarray object for the ifs data
    aifs_his_mf_10 = xr.open_mfdataset(
        aifs_hisgribs,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [10]}
        },
    ).sortby("step")

    aifs_his_mf_2 = xr.open_mfdataset(
        aifs_hisgribs,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [2]}
        },
    ).sortby("step")

    aifs_his_mf_msl = xr.open_mfdataset(
        aifs_hisgribs,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"}},
    ).sortby("step")

    aifs_his_mf_atm = xr.open_mfdataset(
        aifs_hisgribs,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "entireAtmosphere"}},
    ).sortby("step")

    # Combine the datasets
    aifs_his_mf = xr.merge(
        [
            aifs_his_mf_2,
            aifs_his_mf_10,
            aifs_his_mf_msl,
            aifs_his_mf_atm,
        ],
        compat="override",
    )

    # Merge the xarray objects
    xarray_hist_merged = xr.merge(
        [aifs_his_mf, xr_ensoOut_His], compat="override", join="outer"
    )

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    hist_tmp_zarr_path = hist_process_path + "_ECMWF_AIFS_Hist_TMP.zarr"

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chunks to speed up read times

    # Replace the step variable with a time variable using the start time and the step values
    xarray_hist_merged = (
        xarray_hist_merged.assign_coords(
            time=(
                "step",
                pd.to_datetime(xarray_hist_merged.time.values)
                + pd.to_timedelta(xarray_hist_merged.step.values, unit="h"),
            )
        )
        .swap_dims({"step": "time"})
        .drop_vars("step")
    )

    # Chunk and save as zarr
    xarray_hist_merged = xarray_hist_merged.chunk(
        chunks={"time": 1, "latitude": process_chunk, "longitude": process_chunk}
    )

    with ProgressBar():
        xarray_hist_merged.to_zarr(
            hist_tmp_zarr_path,
            mode="w",
            consolidated=False,
        )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    # os.remove(hist_process_path + "_wgrib2_merged.nc")
    # os.remove(hist_process_path + "_wgrib2_merged_UV.nc")

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        archive_tmp_zarr_and_upload(
            tmp_zarr_path=hist_tmp_zarr_path,
            s3_path=s3_path,
            archive_member_name="ECMWF_AIFS_Hist.zarr",
            s3=s3,
        )
    else:
        os.rename(hist_tmp_zarr_path, local_path)
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))

# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
if save_type == "S3":
    local_temp_dir = forecast_process_path + "_s3_temp_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)

    ncLocalWorking_paths = []
    for i in range(his_period, 1, -12):
        timestamp = (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        final_zarr_name = f"ECMWF_AIFS_Hist{timestamp}.zarr"
        extracted_path = download_extract_historic_archive(
            s3=s3,
            historic_path=historic_path,
            final_zarr_name=final_zarr_name,
            extracted_store_name="ECMWF_AIFS_Hist.zarr",
            local_temp_dir=local_temp_dir,
        )
        if extracted_path is not None:
            ncLocalWorking_paths.append(extracted_path)
else:
    ncLocalWorking_paths = [
        historic_path
        + "/ECMWF_AIFS_Hist"
        + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        + ".zarr"
        for i in range(his_period, -1, -6)
    ]

# Read in the zarr arrays
hist = [xr.open_zarr(p, consolidated=False) for p in ncLocalWorking_paths]

fcst = xr.open_zarr(f"{forecast_process_path}_merged.zarr", consolidated=False)
ds = xr.concat(
    [*hist, fcst],
    dim="time",
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="override",
)

# Clip to valid data ranges
for var in zarr_vars:
    if var in ds.data_vars:
        ds_clip = ds[var]
        if np.issubdtype(ds_clip.dtype, np.number):
            mask = (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
            ds[var] = ds_clip.where(mask)  # out-of-range → NaN

# Get the actual stacked times from the concatenated dataset
# This contains the real data times, not artificial times
stacked_times = ds.time.values
stacked_timesUnix = (stacked_times - unix_epoch) / one_second

# Rename time dimension to match later processing
ds_rename = ds.rename({"time": "stacked_time"})

# Add a 3D time array
time3d = (
    ((ds_rename["stacked_time"] - unix_epoch) / np.timedelta64(1, "s"))
    .astype("float32")  # 1D ('time',)
    .expand_dims(
        latitude=ds_rename.latitude,  # add ('latitude',)
        longitude=ds_rename.longitude,
    )  # add ('longitude',)
    .transpose("stacked_time", "latitude", "longitude")  # order dims
)

# Add the time array to the dataset
ds_rename["time"] = time3d

# Set the order correctly
vars_in = [
    v for v in zarr_vars if v in ds_rename.data_vars
]  # keep only those that exist
ds_stack = ds_rename[vars_in].to_array(dim="var", name="var")

# Rechunk the data to be more manageable for processing
ds_chunk = ds_stack.chunk(
    {
        "var": 1,
        "stacked_time": len(stacked_timesUnix),
        "latitude": process_chunk,
        "longitude": process_chunk,
    }
)

# Interim zarr save of the stacked array. Not necessary for local, but speeds things up on S3
with ProgressBar():
    ds_chunk.to_zarr(forecast_process_path + "_stack.zarr", mode="w")

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(
    forecast_process_path + "_stack.zarr", component="__xarray_dataarray_variable__"
)


# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/ECMWF_AIFS.zarr.zip", mode="a"
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/ECMWF_AIFS.zarr")




# Define which variables are integers and need special handling
int_vars = ["ptype"]
# Find the index of these variables in the zarr_vars list
int_var_indices = [i for i, v in enumerate(zarr_vars) if v in int_vars]

# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Pad to chunk size
# 3. Create the zarr array
# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array

with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=4):
        # 1. Interpolate the stacked array to be hourly along the time axis
        daskVarArrayStackDiskInterp = interp_time_take_blend(
            daskVarArrayStackDisk,
            stacked_timesUnix=stacked_timesUnix,
            hourly_timesUnix=hourly_timesUnix,
            nearest_vars=int_var_indices,
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
        forecast_process_dir + "/ECMWF_AIFS.zarr.zip",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/ECMWF_AIFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/ECMWF_AIFS.time.pickle",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/ECMWF_AIFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/ECMWF_AIFS.time.pickle",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/ECMWF_AIFS.zarr",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

T2 = time.time()
logger.info(T2 - T0)
