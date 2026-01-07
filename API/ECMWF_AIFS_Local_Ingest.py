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
    calculate_freezing_level,
    pad_to_chunk_size,
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
    "tp",
    "tcc",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
    "freezing_level",
)


#####################################################################################################
# %% Download ENS data using Herbie Latest
# Needed for tcc
# Find the latest run with 240 hours


# Create a range of forecast lead times
aifs_range = FORECAST_LEAD_RANGES["ECMWF_AIFS"]

# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aifs",
    fxx=aifs_range,
    product="enfo",
    verbose=False,
    save_dir=tmp_dir,
)


match_string_enfo = r":(tp:sfc:\d+):"
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

# AIFS outputs 6-hour accumulations; convert to hourly rate
ens_mf["tpd"] = ens_mf["tpd"] / 6

# Find the probability of precipitation greater than 0.1 mm/h (0.0001) m/h across all members
X3_Precipitation_Prob = (ens_mf["tpd"] > 0.0001).sum(dim="number") / ens_mf.sizes[
    "number"
]

# Find the standard deviation of precipitation accumulation across all members
X3_Precipitation_StdDev = ens_mf["tpd"].std(dim="number")

# Find the average precipitation accumulation across all members
X3_Precipitation_Mean = ens_mf["tpd"].mean(dim="number")

# Merge into a new xarray dataset
xr_ensoOut = xr.Dataset(
    {
        "Precipitation_Prob": X3_Precipitation_Prob,
        "APCP_StdDev": X3_Precipitation_StdDev,
        "APCP_Mean": X3_Precipitation_Mean,
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
matchstring_ap = "(:(tp):)"
matchstring_sl = "(:(msl):)"


# Merge matchstrings for download
match_strings = (
    matchstring_2m + "|" + matchstring_10m + "|" + matchstring_ap + "|" + matchstring_sl
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
ifs_paths = FH_forecastsub.download(match_strings, verbose=False)

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
    ifs_paths,
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
    ifs_paths,
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

aifs_mf_surf = xr.open_mfdataset(
    ifs_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
).sortby("step")

aifs_mf_msl = xr.open_mfdataset(
    ifs_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"}},
).sortby("step")

# Combine the datasets
aifs_mf = xr.merge(
    [aifs_mf_2, aifs_mf_10, aifs_mf_surf, aifs_mf_msl], compat="override"
)


#####################################################################################################
# %% Download pressure level data for freezing level calculation
logger.info("Downloading pressure level data for freezing level calculation")

# Define pressure levels (in hPa) - using same levels as available in AIFS
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Create match string for temperature and geopotential at pressure levels
# ECMWF uses "t" for temperature and "z" for geopotential
matchstring_pressure = "(:(t|z):)"

# Download pressure level data
FH_pressure = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aifs",
    fxx=aifs_range,
    product="oper",
    verbose=False,
    priority=["aws", "ecmwf"],
    save_dir=tmp_dir,
)

pressure_paths = FH_pressure.download(matchstring_pressure, verbose=False)

# Open pressure level data for each variable separately
# Temperature at pressure levels
aifs_temp_pressure = xr.open_mfdataset(
    pressure_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "t"}
    },
).sortby("step")

# Geopotential at pressure levels
aifs_z_pressure = xr.open_mfdataset(
    pressure_paths,
    engine="cfgrib",
    combine="nested",
    concat_dim="step",
    decode_timedelta=False,
    join="outer",
    coords="minimal",
    compat="override",
    backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "z"}
    },
).sortby("step")

# Calculate freezing level
logger.info("Calculating freezing level height")
freezing_level = calculate_freezing_level(
    temperature_levels=aifs_temp_pressure["t"],
    geopotential_levels=aifs_z_pressure["z"],
    pressure_levels=pressure_levels,
)

# Add freezing level to the dataset
aifs_mf["freezing_level"] = freezing_level

# Clean up pressure level data from memory
del aifs_temp_pressure, aifs_z_pressure

logger.info("Freezing level calculation complete")


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


# Create a new time series for the interpolation target
start = xarray_forecast_merged.time[0].values
end = xarray_forecast_merged.time[-1].values
new_hourly_time = pd.date_range(
    start=pd.to_datetime(start) - pd.Timedelta(his_period, "h"), end=end, freq="h"
)

# Get the actual stacked times from the concatenated dataset (to be created later)
# This will be computed after loading and concatenating historical + forecast data
# For now, just compute the hourly target times
unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second


# Chunk and save as zarr
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 64, "latitude": process_chunk, "longitude": process_chunk}
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
    aifs_mf_surf,
    ens_mf,
    xr_ensoOut,
)

T1 = time.time()
logger.info(T1 - T0)

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(his_period, 1, -12):
    if save_type == "S3":
        # S3 Path Setup
        s3_path = (
            historic_path
            + "/ECMWF_AIFS_Hist"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )
        if s3.exists(s3_path.replace(".zarr", ".done")):
            logger.info("File already exists in S3, skipping download for: %s", s3_path)
            # If the file exists, check that it works

            # Try to open and read data from the last variable of the zarr file to check if it has already been saved
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
                logger.error("### Historic Data Failure!")
                logger.exception("Exception processing historic data", exc_info=True)

                # Delete the file if it exists
                if s3.exists(s3_path):
                    s3.rm(s3_path)

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
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="6h",
    )
    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach
    fxx = range(0, 13, 6)

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

    aifs_his_mf_surf = xr.open_mfdataset(
        aifs_hisgribs,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
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

    # Combine the datasets
    aifs_his_mf = xr.merge(
        [aifs_his_mf_2, aifs_his_mf_10, aifs_his_mf_surf, aifs_his_mf_msl],
        compat="override",
    )

    ########################################################################
    ### Download pressure level data for historical freezing level calculation
    logger.info("Downloading historical pressure level data for freezing level")

    # Download pressure level data for the same time period
    FH_histsub_pressure = FastHerbie(
        DATES, model="aifs", fxx=fxx, product="oper", verbose=False, save_dir=tmp_dir
    )

    pressure_his_paths = FH_histsub_pressure.download(
        matchstring_pressure, verbose=False
    )

    # Temperature at pressure levels
    aifs_his_temp_pressure = xr.open_mfdataset(
        pressure_his_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "t"}
        },
    ).sortby("step")

    # Geopotential at pressure levels
    aifs_his_z_pressure = xr.open_mfdataset(
        pressure_his_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "z"}
        },
    ).sortby("step")

    # Calculate freezing level for historical data
    freezing_level_his = calculate_freezing_level(
        temperature_levels=aifs_his_temp_pressure["t"],
        geopotential_levels=aifs_his_z_pressure["z"],
        pressure_levels=pressure_levels,
    )

    # Add freezing level to the historical dataset
    aifs_his_mf["freezing_level"] = freezing_level_his

    # Clean up
    del aifs_his_temp_pressure, aifs_his_z_pressure

    ########################################################################
    ### Download the enfo data
    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES, model="aifs", fxx=fxx, product="enfo", verbose=False, save_dir=tmp_dir
    )

    ens_his_paths = FH_histsub.download(match_string_enfo, verbose=False)

    ens_his_mf = xr.open_mfdataset(
        ens_his_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
    ).sortby("step")

    ens_his_mf["tpd"] = ens_his_mf["tp"].diff(dim="step")

    # Set the first difference value to the first value
    ens_his_mf["tpd"] = xr.where(
        ens_his_mf.step == ens_his_mf.step.isel(step=0),
        ens_his_mf["tp"].isel(step=0),
        ens_his_mf["tpd"],
    )

    # AIFS historic outputs are 6-hour accumulations; convert to hourly
    ens_his_mf["tpd"] = ens_his_mf["tpd"] / 6

    # Find the probability of precipitation greater than 0.1 mm/h (0.0001) m/h across all members
    X3_Precipitation_Prob_His = (ens_his_mf["tpd"] > 0.0001).sum(
        dim="number"
    ) / ens_his_mf.sizes["number"]

    # Find the standard deviation of precipitation accumulation across all members
    X3_Precipitation_StdDev_His = ens_his_mf["tpd"].std(dim="number")

    # Find the average precipitation accumulation across all members
    X3_Precipitation_Mean_His = ens_his_mf["tpd"].mean(dim="number")

    # Merge into a new xarray dataset
    xr_enso_hisOut = xr.Dataset(
        {
            "Precipitation_Prob": X3_Precipitation_Prob_His,
            "APCP_StdDev": X3_Precipitation_StdDev_His,
            "APCP_Mean": X3_Precipitation_Mean_His,
        },
        coords={
            "step": ens_his_mf["step"],
            "latitude": ens_his_mf["latitude"],
            "longitude": ens_his_mf["longitude"],
        },
    )

    # Merge the xarray objects
    xarray_hist_merged = xr.merge(
        [aifs_his_mf, xr_enso_hisOut], compat="override", join="outer"
    )

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

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
        chunks={"time": 64, "latitude": process_chunk, "longitude": process_chunk}
    )

    with ProgressBar():
        xarray_hist_merged.to_zarr(
            store=zarrStore,
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
        done_file = s3_path.replace(".zarr", ".done")
        s3.touch(done_file)
    else:
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))

# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    historic_path
    + "/ECMWF_AIFS_Hist"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(his_period, 1, -12)
]

# Read in the zarr arrays
if save_type == "S3":
    hist = [
        xr.open_zarr(
            p,
            consolidated=False,
            storage_options={
                "key": aws_access_key_id,
                "secret": aws_secret_access_key,
            },
        )
        for p in ncLocalWorking_paths
    ]
else:
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


# TEST READ
# Z = zarr.storage.LocalStore(forecast_process_dir + "/ECMWF.zarr", read_only='r')
# Z2 = zarr.open(Z)

# Rechunk subset of data for maps!
# Want variables:
# 0 (time)
# 2 (t2m)
# 4 (u10)
# 5 (v10)
# 6 (tp)

# Loop through variables, creating a new one with a name and 36 x 100 x 100 chunks
# Save -12:24 hours, aka steps 24:60
# Create a Zarr array in the store with zstd compression

# Add padding for map chunking (100x100)
daskVarArrayStackDisk_maps = pad_to_chunk_size(daskVarArrayStackDisk, 100)

if save_type == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/ECMWF_AIFS_Maps.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(
        forecast_process_dir + "/ECMWF_AIFS_Maps.zarr"
    )

for z in [0, 2, 4, 5, 6]:
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

    da.rechunk(daskVarArrayStackDisk_maps[z, 36:72, :, :], (36, 100, 100)).to_zarr(
        zarr_array, overwrite=True, compute=True
    )

    logger.info(zarr_vars[z])

if save_type == "S3":
    zarr_store_maps.close()

# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/ECMWF_AIFS.zarr.zip",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS.zarr.zip",
    )
    s3.put_file(
        forecast_process_dir + "/ECMWF_AIFS_Maps.zarr.zip",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS_Maps.zarr.zip",
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

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/ECMWF_AIFS_Maps.zarr",
        forecast_path + "/" + ingest_version + "/ECMWF_AIFS_Maps.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

T2 = time.time()
logger.info(T2 - T0)
