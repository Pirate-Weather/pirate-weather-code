# %% Script to ingest HGEFS (Hybrid Global Ensemble Forecast System) data
# Combines AIGEFS ensemble precipitation with AIGFS deterministic variables
# Created February 2026

# %% Import modules
import logging
import os
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
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import FastHerbie, HerbieLatest, Path

from API.constants.shared_const import (
    HISTORY_PERIODS,
    INGEST_VERSION_STR,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    calculate_cloud_cover_from_rh,
    calculate_freezing_level,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    validate_grib_stats,
)

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

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/HGEFS")
forecast_process_path = os.path.join(forecast_process_dir, "HGEFS_Process")
hist_process_path = os.path.join(forecast_process_dir, "HGEFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/HGEFS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/HGEFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


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


T0 = time.time()

# Use AIGEFS to check for latest run (ensemble members)
latest_run = HerbieLatest(
    model="aigefs",
    n=3,
    freq="6h",
    fxx=240,
    product="sfc",
    verbose=False,
    member="avg",
    priority=["nomads"],
    save_dir=tmp_dir,
)

base_time = latest_run.date
# base_time = pd.Timestamp("2024-03-24 06:00:00Z")

logger.info("Base time: %s", base_time)


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/HGEFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/HGEFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to HGEFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/HGEFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/HGEFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to HGEFS, ending")
            sys.exit()

# Define output zarr variables
zarr_vars = (
    "time",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "APCP_Mean",
    "APCP_StdDev",
    "Precipitation_Prob",
    "freezing_level",
    "cloud_cover",
)

#####################################################################################################
# %% PART 1: Download AIGEFS ensemble members for precipitation statistics
logger.info("===== Starting AIGEFS ensemble member downloads for precipitation =====")

# Merge matchstrings for download (precipitation only)
match_strings_precip = "(:APCP:surface:0-[1-9]*)"

# Create a range of forecast lead times
hgefs_range = FORECAST_LEAD_RANGES["AIGEFS"]

# Create FastHerbie object for all 31 members
FH_forecastsubMembers = []
mem = 0
failCount = 0
while mem < 31:
    FH_IN = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="6h"),
        model="aigefs",
        fxx=hgefs_range,
        member=mem + 1,
        product="sfc",
        verbose=False,
        priority=["nomads"],
        save_dir=tmp_dir,
    )

    # Check for download length
    if len(FH_IN.file_exists) != 80:
        logger.warning("Member %d has not downloaded all files, trying again", mem + 1)
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            logger.error(
                "Failed to download files for member %d after multiple attempts.",
                mem + 1,
            )
            sys.exit(1)

        continue

    FH_forecastsubMembers.append(FH_IN)

    # Download and process the subsets
    FH_forecastsubMembers[mem].download(match_strings_precip, verbose=False)

    # Create list of downloaded grib files
    grib_list = [
        str(Path(x.get_localFilePath(match_strings_precip)).expand())
        for x in FH_forecastsubMembers[mem].file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

    validate_grib_stats(grib_check)
    logger.info("Member %d: Grib files passed validation", mem + 1)

    # Create a string to pass to wgrib2 to merge all gribs into one grib
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + "-netcdf "
        + forecast_process_path
        + "_wgrib2_merged_m"
        + str(mem + 1)
        + ".nc"
    )

    # Run wgrib2 to merge all the grib files
    sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if sp_out.returncode != 0:
        logger.error(sp_out.stderr)
        sys.exit()

    # Fix precip and chunk each member
    xarray_wgrib = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc"
    )

    # Save coordinates and time from the first member for consistency
    if mem == 0:
        first_member_lat = xarray_wgrib["latitude"]
        first_member_lon = xarray_wgrib["longitude"]
        first_member_time_min = xarray_wgrib.time.min().values
        first_member_time_max = xarray_wgrib.time.max().values

    # Sometimes there will be weird tiny negative values, set them to zero
    xarray_wgrib["APCP_surface"] = np.maximum(xarray_wgrib["APCP_surface"], 0)

    # Divide by 6 to get hourly accumulation for precipitation only
    # AIGEFS provides 6-hourly accumulation, convert to hourly rate
    xarray_wgrib["APCP_surface"] = xarray_wgrib["APCP_surface"] / 6

    xarray_wgrib = xarray_wgrib.chunk(
        chunks={"time": 80, "latitude": process_chunk, "longitude": process_chunk}
    )

    xarray_wgrib.to_zarr(
        forecast_process_path + "_xr_m" + str(mem + 1) + ".zarr",
        consolidated=False,
        mode="w",
    )

    # Delete the wgrib netcdf to save space
    subprocess.run(
        "rm " + forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc",
        shell=True,
        capture_output=True,
        encoding="utf-8",
    )

    mem += 1

logger.info("Completed downloading all 31 AIGEFS ensemble members")

# Create a new time series using the first member's time range
start = first_member_time_min
end = first_member_time_max
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(hours=his_period), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hours=his_period),
            end=start - pd.Timedelta(hours=1),
            freq="6h",
        ),
        xarray_wgrib.time.values,
    )
)

unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

ncLocalWorking_paths = [
    forecast_process_path + "_xr_m" + str(i) + ".zarr" for i in range(1, 32)
]

# Dask
daskArrays = dict()

# Combine NetCDF files into a Dask Array
for dask_var in ["APCP_surface", "time"]:
    daskVarArrays = []
    for local_ncpath in ncLocalWorking_paths:
        daskVarArrays.append(da.from_zarr(local_ncpath, dask_var))

    # Stack times together, keeping variables separate
    daskArrays[dask_var] = da.stack(daskVarArrays, axis=0)

    daskVarArrays = []


# Dict to hold output dask arrays for precipitation
daskPrecipOutput = dict()

# Find the probability of precipitation greater than 0.0001 mm/h across all members
daskPrecipOutput["Precipitation_Prob"] = ((daskArrays["APCP_surface"]) > 0.0001).sum(
    axis=0
) / daskArrays["APCP_surface"].shape[0]

# Find the standard deviation of precipitation accumulation across all members
daskPrecipOutput["APCP_StdDev"] = daskArrays["APCP_surface"].std(axis=0)

# Find the average precipitation accumulation across all members
daskPrecipOutput["APCP_Mean"] = daskArrays["APCP_surface"].mean(axis=0)

# Copy time over (use first member [0] for consistency)
daskPrecipOutput["time"] = daskArrays["time"][0, :]

logger.info("Calculated ensemble precipitation statistics")

# Save precipitation statistics to zarr for later merging
for precip_var in ["Precipitation_Prob", "APCP_Mean", "APCP_StdDev", "time"]:
    if precip_var == "time":
        daskPrecipOutput[precip_var].to_zarr(
            forecast_process_path + "_" + precip_var + ".zarr",
            codecs=[
                zarr.codecs.BytesCodec(),
                zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            ],
            overwrite=True,
            compute=True,
        )
    else:
        daskPrecipOutput[precip_var].rechunk(
            (80, process_chunk, process_chunk)
        ).to_zarr(
            forecast_process_path + "_" + precip_var + ".zarr",
            codecs=[
                zarr.codecs.BytesCodec(),
                zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            ],
            overwrite=True,
            compute=True,
        )

# Clean up ensemble member files to save space
for i in range(1, 32):
    shutil.rmtree(forecast_process_path + "_xr_m" + str(i) + ".zarr")

del FH_forecastsubMembers, daskArrays, daskPrecipOutput, xarray_wgrib

logger.info("===== Completed AIGEFS precipitation ensemble processing =====")

#####################################################################################################
# %% PART 2: Download AIGFS deterministic data for other variables
logger.info("===== Starting AIGFS deterministic download for other variables =====")

# Define the subset of variables to download as a list of strings
matchstring_2m = ":((TMP):2 m above ground:)"
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_sl = "(:(PRMSL):)"

# Merge matchstrings for download (no APCP since we have it from ensemble)
match_strings_aigfs = matchstring_2m + "|" + matchstring_10m + "|" + matchstring_sl

# Create a range of forecast lead times
aigfs_range = FORECAST_LEAD_RANGES["AIGFS"]

# Create FastHerbie object for deterministic AIGFS
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aigfs",
    fxx=aigfs_range,
    product="sfc",
    verbose=False,
    priority=["nomads"],
    save_dir=tmp_dir,
)

# Download the subsets
FH_forecastsub.download(match_strings_aigfs, verbose=False)

# Check for download length
if len(FH_forecastsub.file_exists) != len(aigfs_range):
    logger.error(
        "Download failed, expected %d files but got %d",
        len(aigfs_range),
        len(FH_forecastsub.file_exists),
    )
    sys.exit(1)


# Create list of downloaded grib files
grib_list = [
    str(Path(x.get_localFilePath(match_strings_aigfs)).expand())
    for x in FH_forecastsub.file_exists
]

# Perform a check if any data seems to be invalid
cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

# Validate the grib files
validate_grib_stats(grib_check)
logger.info("AIGFS: Grib validation complete, no errors found.")


# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(grib_list)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_aigfs_merged.nc"
)


# Run wgrib2
sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    logger.error(sp_out.stderr)
    sys.exit()

# Read the netcdf file using xarray
xarray_aigfs_merged = xr.open_mfdataset(forecast_process_path + "_aigfs_merged.nc")

assert len(xarray_aigfs_merged.time) == len(aigfs_range), (
    "Incorrect number of timesteps! Exiting"
)

logger.info("Completed AIGFS deterministic download")

#####################################################################################################
# %% Download pressure level data for freezing level and cloud cover calculation
logger.info("Downloading pressure level data for derived parameters")

# Define pressure levels (in hPa)
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Match strings for pressure level data
matchstring_pressure_tmp = (
    "(:TMP:(1000|925|850|700|600|500|400|300|250|200|150|100|50) mb:)"
)
matchstring_pressure_hgt = (
    "(:HGT:(1000|925|850|700|600|500|400|300|250|200|150|100|50) mb:)"
)
matchstring_pressure_spfh = (
    "(:SPFH:(1000|925|850|700|600|500|400|300|250|200|150|100|50) mb:)"
)

match_strings_pressure = (
    matchstring_pressure_tmp
    + "|"
    + matchstring_pressure_hgt
    + "|"
    + matchstring_pressure_spfh
)

# Download pressure level data
FH_pressure = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aigfs",
    fxx=aigfs_range,
    product="pres",
    verbose=False,
    priority=["nomads"],
    save_dir=tmp_dir,
)

FH_pressure.download(match_strings_pressure, verbose=False)

# Check for download length
if len(FH_pressure.file_exists) != len(aigfs_range):
    logger.error(
        "Pressure download failed, expected %d files but got %d",
        len(aigfs_range),
        len(FH_pressure.file_exists),
    )
    sys.exit(1)

# Create list of downloaded grib files
grib_list_pressure = [
    str(Path(x.get_localFilePath(match_strings_pressure)).expand())
    for x in FH_pressure.file_exists
]

# Merge pressure level grib files
cmd = (
    "cat "
    + " ".join(grib_list_pressure)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_pressure_merged.nc"
)

sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    logger.error(sp_out.stderr)
    sys.exit()

# Read pressure level data
xarray_pressure = xr.open_mfdataset(forecast_process_path + "_pressure_merged.nc")

logger.info("Completed pressure level data download")

#####################################################################################################
# %% Calculate derived parameters using AIGFS pressure data
logger.info("Calculating derived parameters (freezing level, cloud cover)")

# Calculate freezing level
freezing_level_data = calculate_freezing_level(
    temp_profiles=xarray_pressure["TMP_isobaricInhPa"],
    height_profiles=xarray_pressure["HGT_isobaricInhPa"],
    pressure_levels=xarray_pressure.isobaricInhPa,
)

# Calculate cloud cover from relative humidity
cloud_cover_data = calculate_cloud_cover_from_rh(
    temp_profiles=xarray_pressure["TMP_isobaricInhPa"],
    spfh_profiles=xarray_pressure["SPFH_isobaricInhPa"],
    pressure_levels=xarray_pressure.isobaricInhPa,
)

logger.info("Completed derived parameter calculations")

# Create combined forecast dataset with all variables
xarray_forecast_merged = xr.Dataset(
    {
        "PRMSL_meansealevel": xarray_aigfs_merged["PRMSL_meansealevel"],
        "TMP_2maboveground": xarray_aigfs_merged["TMP_2maboveground"],
        "UGRD_10maboveground": xarray_aigfs_merged["UGRD_10maboveground"],
        "VGRD_10maboveground": xarray_aigfs_merged["VGRD_10maboveground"],
        "freezing_level": (["time", "latitude", "longitude"], freezing_level_data),
        "cloud_cover": (["time", "latitude", "longitude"], cloud_cover_data),
    },
    coords={
        "time": xarray_aigfs_merged.time,
        "latitude": xarray_aigfs_merged.latitude,
        "longitude": xarray_aigfs_merged.longitude,
    },
)

# Load precipitation statistics from zarr and add to dataset
precip_time = da.from_zarr(forecast_process_path + "_time.zarr")
xarray_forecast_merged["APCP_Mean"] = xr.DataArray(
    da.from_zarr(forecast_process_path + "_APCP_Mean.zarr"),
    dims=["time", "latitude", "longitude"],
    coords={
        "time": precip_time,
        "latitude": first_member_lat,
        "longitude": first_member_lon,
    },
)
xarray_forecast_merged["APCP_StdDev"] = xr.DataArray(
    da.from_zarr(forecast_process_path + "_APCP_StdDev.zarr"),
    dims=["time", "latitude", "longitude"],
    coords={
        "time": precip_time,
        "latitude": first_member_lat,
        "longitude": first_member_lon,
    },
)
xarray_forecast_merged["Precipitation_Prob"] = xr.DataArray(
    da.from_zarr(forecast_process_path + "_Precipitation_Prob.zarr"),
    dims=["time", "latitude", "longitude"],
    coords={
        "time": precip_time,
        "latitude": first_member_lat,
        "longitude": first_member_lon,
    },
)

logger.info("Merged all variables into unified dataset")

# Clean up intermediate files
del xarray_aigfs_merged, xarray_pressure, FH_forecastsub, FH_pressure

#####################################################################################################
# %% Time handling and interpolation setup

# Create a new time series
start = xarray_forecast_merged.time.min().values
end = xarray_forecast_merged.time.max().values
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

#####################################################################################################
# %% Historical data processing
logger.info("===== Starting historical data processing =====")

# Create a list of times to download, starting from 48 hours ago
# For HGEFS, we'll use AIGFS for historical deterministic data
hist_download_times = pd.date_range(
    start=base_time - pd.Timedelta(hours=his_period), end=base_time, freq="6h"
)[:-1]

historical_datasets = []

for hist_time in hist_download_times:
    hist_zarr_path = historic_path + "/HGEFS_Hist_v2" + str(hist_time.value) + ".zarr"

    # Check if historical data already exists
    if save_type == "S3":
        if s3.exists(hist_zarr_path):
            logger.info("Historical data exists for %s, skipping", hist_time)
            try:
                hist_ds = xr.open_zarr(s3fs.S3Map(hist_zarr_path, s3=s3))
                historical_datasets.append(hist_ds)
                continue
            except Exception as e:
                logger.warning("Error loading historical data: %s", e)
    else:
        if os.path.exists(hist_zarr_path):
            logger.info("Historical data exists for %s, skipping", hist_time)
            try:
                hist_ds = xr.open_zarr(hist_zarr_path)
                historical_datasets.append(hist_ds)
                continue
            except Exception as e:
                logger.warning("Error loading historical data: %s", e)

    # Download historical data using AIGFS deterministic model
    logger.info("Downloading historical data for %s", hist_time)

    FH_hist = FastHerbie(
        pd.date_range(start=hist_time, periods=1, freq="6h"),
        model="aigfs",
        fxx=[0, 6],  # Only get first two time steps
        product="sfc",
        verbose=False,
        priority=["nomads"],
        save_dir=tmp_dir,
    )

    # Download surface variables
    FH_hist.download(match_strings_aigfs, verbose=False)

    # Check download
    if len(FH_hist.file_exists) < 2:
        logger.warning("Historical download incomplete for %s, skipping", hist_time)
        continue

    # Create list of grib files
    grib_list_hist = [
        str(Path(x.get_localFilePath(match_strings_aigfs)).expand())
        for x in FH_hist.file_exists
    ]

    # Merge historical grib files
    cmd = (
        "cat "
        + " ".join(grib_list_hist)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + hist_process_path
        + "_hist_"
        + str(hist_time.value)
        + ".nc"
    )

    sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if sp_out.returncode != 0:
        logger.warning("Historical grib merge failed for %s", hist_time)
        continue

    # Load historical data
    xarray_hist = xr.open_mfdataset(
        hist_process_path + "_hist_" + str(hist_time.value) + ".nc"
    )

    # Download historical pressure data for derived parameters
    FH_hist_pressure = FastHerbie(
        pd.date_range(start=hist_time, periods=1, freq="6h"),
        model="aigfs",
        fxx=[0, 6],
        product="pres",
        verbose=False,
        priority=["nomads"],
        save_dir=tmp_dir,
    )

    FH_hist_pressure.download(match_strings_pressure, verbose=False)

    if len(FH_hist_pressure.file_exists) >= 2:
        grib_list_hist_pressure = [
            str(Path(x.get_localFilePath(match_strings_pressure)).expand())
            for x in FH_hist_pressure.file_exists
        ]

        cmd = (
            "cat "
            + " ".join(grib_list_hist_pressure)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + " -netcdf "
            + hist_process_path
            + "_hist_pressure_"
            + str(hist_time.value)
            + ".nc"
        )

        sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        if sp_out.returncode == 0:
            xarray_hist_pressure = xr.open_mfdataset(
                hist_process_path + "_hist_pressure_" + str(hist_time.value) + ".nc"
            )

            # Calculate derived parameters for historical data
            hist_freezing_level = calculate_freezing_level(
                temp_profiles=xarray_hist_pressure["TMP_isobaricInhPa"],
                height_profiles=xarray_hist_pressure["HGT_isobaricInhPa"],
                pressure_levels=xarray_hist_pressure.isobaricInhPa,
            )

            hist_cloud_cover = calculate_cloud_cover_from_rh(
                temp_profiles=xarray_hist_pressure["TMP_isobaricInhPa"],
                spfh_profiles=xarray_hist_pressure["SPFH_isobaricInhPa"],
                pressure_levels=xarray_hist_pressure.isobaricInhPa,
            )

            # Add derived parameters to historical dataset
            xarray_hist["freezing_level"] = (
                ["time", "latitude", "longitude"],
                hist_freezing_level,
            )
            xarray_hist["cloud_cover"] = (
                ["time", "latitude", "longitude"],
                hist_cloud_cover,
            )

    # For historical precipitation, use NaN (no ensemble data available)
    # This is acceptable as historical data is less critical
    xarray_hist["APCP_Mean"] = xr.full_like(
        xarray_hist["TMP_2maboveground"], fill_value=np.nan
    )
    xarray_hist["APCP_StdDev"] = xr.full_like(
        xarray_hist["TMP_2maboveground"], fill_value=np.nan
    )
    xarray_hist["Precipitation_Prob"] = xr.full_like(
        xarray_hist["TMP_2maboveground"], fill_value=np.nan
    )

    # Save historical data to zarr
    if save_type == "S3":
        xarray_hist.to_zarr(
            s3fs.S3Map(hist_zarr_path, s3=s3),
            mode="w",
            consolidated=True,
        )
    else:
        xarray_hist.to_zarr(hist_zarr_path, mode="w", consolidated=True)

    historical_datasets.append(xarray_hist)

    # Clean up
    subprocess.run(
        f"rm {hist_process_path}_hist_{hist_time.value}.nc",
        shell=True,
        capture_output=True,
        encoding="utf-8",
    )
    subprocess.run(
        f"rm {hist_process_path}_hist_pressure_{hist_time.value}.nc",
        shell=True,
        capture_output=True,
        encoding="utf-8",
    )

logger.info("Completed historical data processing")

#####################################################################################################
# %% Combine historical and forecast data
logger.info("===== Combining historical and forecast data =====")

# Combine historical datasets with forecast
if historical_datasets:
    combined_dataset = xr.concat(
        historical_datasets + [xarray_forecast_merged], dim="time"
    )
else:
    combined_dataset = xarray_forecast_merged

logger.info(
    "Combined dataset time range: %s to %s",
    combined_dataset.time.min().values,
    combined_dataset.time.max().values,
)

#####################################################################################################
# %% Interpolate to hourly and save final output
logger.info("===== Interpolating to hourly resolution =====")

# Convert to dask arrays for processing
daskVarArrayStack = []
for var in zarr_vars:
    if var == "time":
        continue
    daskVarArrayStack.append(combined_dataset[var].data)

daskVarArrayStackDisk = da.stack(daskVarArrayStack, axis=0)

# Get time coordinates
combined_times = combined_dataset.time.values
combined_timesUnix = (combined_times - unix_epoch) / one_second

# Interpolate to hourly
daskVarArrayStackDiskInterp = interp_time_take_blend(
    daskVarArrayStackDisk, combined_timesUnix, hourly_timesUnix
)

# Pad to chunk size
daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
    daskVarArrayStackDiskInterp, final_chunk
)

# Mask invalid data
daskVarArrayStackDiskInterpPad = mask_invalid_data(daskVarArrayStackDiskInterpPad)

logger.info("Completed interpolation and padding")

#####################################################################################################
# %% Save final zarr output
logger.info("===== Saving final HGEFS zarr output =====")

# Create output zarr array
final_shape = (
    len(zarr_vars) - 1,  # Exclude time from count
    len(hourly_timesUnix),
    daskVarArrayStackDiskInterpPad.shape[2],
    daskVarArrayStackDiskInterpPad.shape[3],
)

final_chunks = (len(zarr_vars) - 1, len(hourly_timesUnix), final_chunk, final_chunk)

output_path = forecast_path + "/" + ingest_version + "/HGEFS.zarr"

if save_type == "S3":
    store = s3fs.S3Map(output_path, s3=s3)
else:
    store = output_path

# Create zarr array
z = zarr.open(
    store,
    mode="w",
    shape=final_shape,
    chunks=final_chunks,
    dtype="f4",
)

# Write data
logger.info("Writing data to zarr (this may take a while)...")
with ProgressBar():
    z[:] = daskVarArrayStackDiskInterpPad.compute()

logger.info("Completed writing zarr data")

# Save metadata
attrs = {
    "variables": list(zarr_vars),
    "time": hourly_timesUnix.tolist(),
    "version": ingest_version,
    "model": "HGEFS",
    "description": "Hybrid Global Ensemble Forecast System - AIGEFS precipitation + AIGFS deterministic",
}

if save_type == "S3":
    with s3.open(
        forecast_path + "/" + ingest_version + "/HGEFS.attrs.pickle", "wb"
    ) as f:
        pickle.dump(attrs, f)
    with s3.open(
        forecast_path + "/" + ingest_version + "/HGEFS.time.pickle", "wb"
    ) as f:
        pickle.dump(base_time, f)
else:
    with open(forecast_path + "/" + ingest_version + "/HGEFS.attrs.pickle", "wb") as f:
        pickle.dump(attrs, f)
    with open(forecast_path + "/" + ingest_version + "/HGEFS.time.pickle", "wb") as f:
        pickle.dump(base_time, f)

#####################################################################################################
# %% Cleanup and completion
logger.info("===== Cleaning up temporary files =====")

# Remove temporary files
subprocess.run(
    f"rm {forecast_process_path}_aigfs_merged.nc",
    shell=True,
    capture_output=True,
    encoding="utf-8",
)
subprocess.run(
    f"rm {forecast_process_path}_pressure_merged.nc",
    shell=True,
    capture_output=True,
    encoding="utf-8",
)

# Remove precipitation zarr files
for precip_var in ["time", "APCP_Mean", "APCP_StdDev", "Precipitation_Prob"]:
    zarr_path = forecast_process_path + "_" + precip_var + ".zarr"
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

T1 = time.time()
total_time = T1 - T0

logger.info("===== HGEFS INGEST COMPLETE =====")
logger.info(
    "Total processing time: %.2f seconds (%.2f minutes)", total_time, total_time / 60
)
logger.info("Base time: %s", base_time)
logger.info("Output path: %s", output_path)
