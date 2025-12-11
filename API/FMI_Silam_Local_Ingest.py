# %% FMI SILAM Global Air Quality Processing script
# This script downloads the latest FMI SILAM global air quality forecast data
# from the THREDDS server, processes it, and saves to a Zarr store for API consumption.
#
# SILAM (System for Integrated modeLling of Atmospheric coMposition) provides
# global air quality forecasts including PM2.5, PM10, O3, NO2, SO2, and CO.
#
# Data source: https://thredds.silam.fmi.fi/thredds/catalog/catalog.html
#
# Author: Alexander Rey
# Date: December 2024

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import s3fs
import xarray as xr
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import CHUNK_SIZES, calculate_aqi

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingestVersion = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/SILAM"
)
forecast_process_path = os.path.join(forecast_process_dir, "SILAM_Process")
tmpDIR = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/SILAM")

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing chunk size - use GFS chunk sizes as SILAM is global data
processChunk = CHUNK_SIZES.get("GFS", 50)

# Define the variables to be saved in the final Zarr store
# These match the SILAM output variables for air quality
zarrVars = (
    "time",
    "cnc_PM2_5",  # PM2.5 concentration (µg/m³)
    "cnc_PM10",  # PM10 concentration (µg/m³)
    "cnc_O3",  # Ozone concentration (µg/m³)
    "cnc_NO2",  # Nitrogen dioxide concentration (µg/m³)
    "cnc_SO2",  # Sulfur dioxide concentration (µg/m³)
    "cnc_CO",  # Carbon monoxide concentration (µg/m³)
    "AQI",  # Air Quality Index (calculated)
)

# Alternative SILAM variable names (different SILAM versions may use different names)
silam_alt_names = {
    "cnc_PM2_5": ["cnc_PM2_5_gas", "cnc_PM25", "pm2_5", "PM2.5", "pm25"],
    "cnc_PM10": ["cnc_PM10_gas", "cnc_pm10", "pm10", "PM10"],
    "cnc_O3": ["cnc_O3_gas", "cnc_o3", "o3", "O3", "ozone"],
    "cnc_NO2": ["cnc_NO2_gas", "cnc_no2", "no2", "NO2"],
    "cnc_SO2": ["cnc_SO2_gas", "cnc_so2", "so2", "SO2"],
    "cnc_CO": ["cnc_CO_gas", "cnc_co", "co", "CO"],
}


def get_latest_silam_run():
    """
    Determines the latest available SILAM model run time.
    SILAM air quality data is updated once daily at 00 UTC.

    Returns:
        datetime: The latest model run time.
    """
    now_utc = datetime.now(timezone.utc)

    # SILAM updates once daily at 00 UTC
    # Allow 3 hours for data availability after the 00 UTC run
    if now_utc.hour >= 3:
        # Use today's 00 UTC run
        latest_origintime = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Use previous day's 00 UTC run (today's not yet available)
        latest_origintime = (now_utc - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    return latest_origintime


def find_variable(ds, target_var):
    """
    Find a variable in the dataset, trying alternative names.

    Args:
        ds: xarray Dataset
        target_var: Target variable name

    Returns:
        str or None: Found variable name, or None if not found
    """
    # Check if target var exists directly
    if target_var in ds.data_vars:
        return target_var

    # Try alternative names
    alt_names = silam_alt_names.get(target_var, [])
    for alt in alt_names:
        if alt in ds.data_vars:
            return alt

    return None


# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)

if saveType == "Download":
    if not os.path.exists(forecast_path + "/" + ingestVersion):
        os.makedirs(forecast_path + "/" + ingestVersion)

start_time = time.time()

# Get the latest model run time
origintime = get_latest_silam_run()

# Construct the OPeNDAP URL for SILAM global forecast
# SILAM data is available via THREDDS OPeNDAP service
# Using SILAM version 6.1 (silam_glob06_v6_1) - the latest available version
# The URL pattern follows: base_url/silam_glob06_v6_1/runs/silam_glob06_v6_1_YYYYMMDDHH.nc
base_opendap_url = "https://thredds.silam.fmi.fi/thredds/dodsC"
silam_dataset_path = "silam_glob06_v6_1/runs"
run_filename = f"silam_glob06_v6_1_{origintime.strftime('%Y%m%d%H')}.nc"

opendap_url = f"{base_opendap_url}/{silam_dataset_path}/{run_filename}"

logger.info(f"Attempting to access SILAM data from: {opendap_url}")
logger.info(f"Model run time: {origintime}")

# Check if this is newer than the current file
if saveType == "S3":
    if s3.exists(forecast_path + "/" + ingestVersion + "/SILAM.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingestVersion + "/SILAM.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= origintime:
            logger.info("No Update to SILAM, ending")
            sys.exit()
else:
    if os.path.exists(forecast_path + "/" + ingestVersion + "/SILAM.time.pickle"):
        with open(
            forecast_path + "/" + ingestVersion + "/SILAM.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= origintime:
            logger.info("No Update to SILAM, ending")
            sys.exit()


# %% Load the SILAM data via OPeNDAP
try:
    # Open the dataset via OPeNDAP
    # SILAM data has dimensions: time, height (usually surface level), lat, lon
    xarray_silam_data = xr.open_dataset(
        opendap_url,
        engine="netcdf4",
        chunks={"time": 24, "lat": processChunk, "lon": processChunk},
    )

    logger.info("Successfully opened SILAM dataset via OPeNDAP")
    logger.info(f"Dataset dimensions: {xarray_silam_data.dims}")
    logger.info(f"Available variables: {list(xarray_silam_data.data_vars.keys())}")

except (IOError, OSError, ValueError) as e:
    logger.error(f"Error opening SILAM data via OPeNDAP: {e}")
    logger.info("Attempting fallback URL pattern...")

    # Try alternative URL patterns (including older versions as fallback)
    alt_urls = [
        f"{base_opendap_url}/silam_glob06_v6_1/silam_glob06_v6_1_{origintime.strftime('%Y%m%d%H')}.nc",
        f"{base_opendap_url}/silam_glob06_v6_1/latest.nc",
        # Fallback to older v5.9 if v6.1 is unavailable
        f"{base_opendap_url}/silam_glob05_v5_9/runs/silam_glob05_v5_9_{origintime.strftime('%Y%m%d%H')}.nc",
        f"{base_opendap_url}/silam_glob05_v5_9/silam_glob05_v5_9_{origintime.strftime('%Y%m%d%H')}.nc",
    ]

    xarray_silam_data = None
    for alt_url in alt_urls:
        try:
            logger.info(f"Trying: {alt_url}")
            xarray_silam_data = xr.open_dataset(
                alt_url,
                engine="netcdf4",
                chunks={"time": 24, "lat": processChunk, "lon": processChunk},
            )
            logger.info(f"Successfully opened SILAM from: {alt_url}")
            break
        except Exception as alt_e:
            logger.warning(f"Failed to open {alt_url}: {alt_e}")
            continue

    if xarray_silam_data is None:
        logger.critical("Failed to access SILAM data from any URL. Exiting.")
        sys.exit(1)


# %% Process the SILAM data
# Select surface level if height dimension exists
if "height" in xarray_silam_data.dims:
    xarray_silam_data = xarray_silam_data.isel(height=0)
elif "level" in xarray_silam_data.dims:
    xarray_silam_data = xarray_silam_data.isel(level=0)

# Rename coordinates if needed to standardize
if "lat" in xarray_silam_data.coords:
    xarray_silam_data = xarray_silam_data.rename({"lat": "latitude"})
if "lon" in xarray_silam_data.coords:
    xarray_silam_data = xarray_silam_data.rename({"lon": "longitude"})

# Process and rename variables to our standard naming convention
processed_vars = {}

for target_var in zarrVars[1:-1]:  # Skip 'time' and 'AQI' (calculated later)
    found_var = find_variable(xarray_silam_data, target_var)
    if found_var:
        processed_vars[target_var] = xarray_silam_data[found_var]
        logger.info(f"Found variable {found_var} -> {target_var}")
    else:
        logger.warning(
            f"Variable {target_var} not found in dataset, will be filled with NaN"
        )

# Create processed dataset
xarray_processed = xr.Dataset(coords=xarray_silam_data.coords)
xarray_processed["time"] = xarray_silam_data["time"]

for var_name, var_data in processed_vars.items():
    xarray_processed[var_name] = var_data

# Fill missing variables with NaN
for var in zarrVars[1:-1]:
    if var not in xarray_processed.data_vars:
        xarray_processed[var] = xr.DataArray(
            np.full(
                (
                    len(xarray_processed.time),
                    len(xarray_processed.latitude),
                    len(xarray_processed.longitude),
                ),
                np.nan,
                dtype=np.float32,
            ),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": xarray_processed.time,
                "latitude": xarray_processed.latitude,
                "longitude": xarray_processed.longitude,
            },
        )

# Calculate AQI from pollutant concentrations
logger.info("Calculating Air Quality Index (AQI) using EPA NowCast...")

# Create fallback shape for missing data (3D: time, latitude, longitude)
fallback_shape = (
    len(xarray_processed.time),
    len(xarray_processed.latitude),
    len(xarray_processed.longitude),
)

pm25_data = (
    xarray_processed["cnc_PM2_5"].values
    if "cnc_PM2_5" in xarray_processed
    else np.full(fallback_shape, np.nan, dtype=np.float32)
)
pm10_data = (
    xarray_processed["cnc_PM10"].values
    if "cnc_PM10" in xarray_processed
    else np.full(fallback_shape, np.nan, dtype=np.float32)
)
o3_data = (
    xarray_processed["cnc_O3"].values
    if "cnc_O3" in xarray_processed
    else np.full(fallback_shape, np.nan, dtype=np.float32)
)
no2_data = (
    xarray_processed["cnc_NO2"].values
    if "cnc_NO2" in xarray_processed
    else np.full(fallback_shape, np.nan, dtype=np.float32)
)
so2_data = (
    xarray_processed["cnc_SO2"].values
    if "cnc_SO2" in xarray_processed
    else np.full(fallback_shape, np.nan, dtype=np.float32)
)
co_data = (
    xarray_processed["cnc_CO"].values
    if "cnc_CO" in xarray_processed
    else np.full(fallback_shape, np.nan, dtype=np.float32)
)

# Calculate AQI using EPA NowCast algorithm for PM2.5 and PM10
aqi_values = calculate_aqi(
    pm25_data, pm10_data, o3_data, no2_data, so2_data, co_data, use_nowcast=True
)

xarray_processed["AQI"] = xr.DataArray(
    aqi_values.astype(np.float32),
    dims=["time", "latitude", "longitude"],
    coords={
        "time": xarray_processed.time,
        "latitude": xarray_processed.latitude,
        "longitude": xarray_processed.longitude,
    },
    attrs={
        "long_name": "Air Quality Index",
        "units": "1",
        "method": "EPA NowCast for PM2.5/PM10",
    },
)

logger.info("AQI calculation complete (using EPA NowCast for PM2.5/PM10)")

# %% Save the processed data to Zarr
xarray_processed = xarray_processed.chunk(
    chunks={
        "time": xarray_processed.time.size,
        "latitude": processChunk,
        "longitude": processChunk,
    }
)

logger.info(f"Saving processed data to: {forecast_process_path}_.zarr")

with ProgressBar():
    xarray_processed.to_zarr(
        forecast_process_path + "_.zarr", mode="w", consolidated=False, compute=True
    )
logger.info("Saved Zarr data to disk.")


# %% Final output handling and cleanup
if saveType == "S3":
    s3_bucket_name = os.getenv("s3_bucket", "your-s3-bucket")
    s3_zarr_key = os.path.join("ForecastTar_v2", "SILAM.zarr")
    s3_url = f"s3://{s3_bucket_name}/{s3_zarr_key}"

    # Write directly to S3-backed zarr store
    zarr_store = zarr.storage.FsspecStore.from_url(
        s3_url,
        storage_options={"key": aws_access_key_id, "secret": aws_secret_access_key},
    )
    xarray_processed.to_zarr(store=zarr_store, mode="w", consolidated=False)

    # Save time pickle to S3
    with open(forecast_process_dir + "/SILAM.time.pickle", "wb") as file:
        pickle.dump(origintime, file)

    s3.put_file(
        forecast_process_dir + "/SILAM.time.pickle",
        forecast_path + "/" + ingestVersion + "/SILAM.time.pickle",
    )
    logger.info("Uploaded SILAM data to S3.")
else:
    # Save time pickle locally
    with open(forecast_process_dir + "/SILAM.time.pickle", "wb") as file:
        pickle.dump(origintime, file)

    shutil.move(
        forecast_process_dir + "/SILAM.time.pickle",
        forecast_path + "/" + ingestVersion + "/SILAM.time.pickle",
    )

    # Copy Zarr to final location
    shutil.copytree(
        forecast_process_path + "_.zarr",
        forecast_path + "/" + ingestVersion + "/SILAM.zarr",
        dirs_exist_ok=True,
    )
    logger.info(
        f"Saved SILAM data locally to {forecast_path}/{ingestVersion}/SILAM.zarr"
    )

# Clean up temporary files and directories
shutil.rmtree(forecast_process_dir)

end_time = time.time()
logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
logger.info("SILAM ingest script finished successfully.")
