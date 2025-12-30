# %% FMI IS4FIRES Wildfire Smoke Processing script
# This script downloads the latest FMI IS4FIRES wildfire smoke forecast data
# from the THREDDS server, processes it, and saves to a Zarr store for API consumption.
#
# IS4FIRES (Integrated System for wildland FIRES) provides global wildfire smoke
# forecasts including PM2.5 from fire emissions (PM_FRP).
#
# Data source: https://thredds.silam.fmi.fi/thredds/catalog/catalog.html
#
# Author: Alexander Rey
# Date: December 2025

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
from dask.diagnostics import ProgressBar

from API.constants.shared_const import INGEST_VERSION_STR
from API.silam_conversion import KG_M3_TO_UG_M3

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingestVersion = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/IS4FIRES"
)
forecast_process_path = os.path.join(forecast_process_dir, "IS4FIRES_Process")
tmpDIR = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/IS4FIRES")

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing chunk size - use 50 for IS4FIRES global data (same as GFS)
processChunk = 50

# IS4FIRES variable names and units:
# - cnc_PM_FRP: Fire-related particulate matter in kg/m³ (need conversion to µg/m³)
#   This represents PM2.5 from wildfire smoke emissions


def get_latest_is4fires_run():
    """Determines the latest available IS4FIRES model run time.

    IS4FIRES wildfire smoke data is updated once daily at 00 UTC.
    The data is approximately 2 days behind realtime.

    Returns:
        datetime: The latest model run time.
    """
    now_utc = datetime.now(timezone.utc)

    # IS4FIRES updates once daily at 00 UTC with the date being ~2 days behind current time
    latest_origintime = (now_utc - timedelta(days=2)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    return latest_origintime


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
origintime = get_latest_is4fires_run()

# Construct the OPeNDAP URL for IS4FIRES wildfire smoke forecast
# IS4FIRES data is available via THREDDS OPeNDAP service
# Using IS4FIRES version 2.0 (IS4FIRES_v2_0)
# The URL pattern follows: base_url/i4f20-fc/runs/IS4FIRES-fc_RUN_YYYYMMDDHH.nc
base_opendap_url = "https://thredds.silam.fmi.fi/thredds/dodsC"
is4fires_dataset_path = "i4f20-fc/runs"
run_filename = f"IS4FIRES-fc_RUN_{origintime.strftime('%Y%m%d%H')}.nc"

opendap_url = f"{base_opendap_url}/{is4fires_dataset_path}/{run_filename}"

logger.info(f"Attempting to access IS4FIRES data from: {opendap_url}")
logger.info(f"Model run time: {origintime}")

# Check if this is newer than the current file
if saveType == "S3":
    if s3.exists(forecast_path + "/" + ingestVersion + "/IS4FIRES.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingestVersion + "/IS4FIRES.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= origintime:
            logger.info("No Update to IS4FIRES, ending")
            sys.exit()
else:
    if os.path.exists(forecast_path + "/" + ingestVersion + "/IS4FIRES.time.pickle"):
        with open(
            forecast_path + "/" + ingestVersion + "/IS4FIRES.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= origintime:
            logger.info("No Update to IS4FIRES, ending")
            sys.exit()


# %% Load the IS4FIRES data via OPeNDAP
try:
    # Open the dataset via OPeNDAP
    # IS4FIRES data has dimensions: time, height (usually surface level), lat, lon
    xarray_is4fires_data = xr.open_dataset(
        opendap_url,
        engine="netcdf4",
        chunks={"time": 24, "lat": processChunk, "lon": processChunk},
    )

    logger.info("Successfully opened IS4FIRES dataset via OPeNDAP")
    logger.info(f"Dataset dimensions: {xarray_is4fires_data.dims}")
    logger.info(f"Available variables: {list(xarray_is4fires_data.data_vars.keys())}")

except (IOError, OSError, ValueError) as e:
    logger.error(f"Error opening IS4FIRES data via OPeNDAP: {e}")
    logger.critical("Failed to access IS4FIRES data. Exiting.")
    sys.exit(1)


# %% Process the IS4FIRES data
# Select surface level if height dimension exists
if "height" in xarray_is4fires_data.dims:
    xarray_is4fires_data = xarray_is4fires_data.isel(height=0)
elif "level" in xarray_is4fires_data.dims:
    xarray_is4fires_data = xarray_is4fires_data.isel(level=0)

# Rename coordinates if needed to standardize
if "lat" in xarray_is4fires_data.coords:
    xarray_is4fires_data = xarray_is4fires_data.rename({"lat": "latitude"})
if "lon" in xarray_is4fires_data.coords:
    xarray_is4fires_data = xarray_is4fires_data.rename({"lon": "longitude"})

# Create processed dataset
xarray_processed = xr.Dataset(coords=xarray_is4fires_data.coords)
xarray_processed["time"] = xarray_is4fires_data["time"]


def _make_nan_dataarray(
    time_coord, lat_coord, lon_coord, fill=np.nan, dtype=np.float32
):
    """Return a 3D DataArray filled with `fill` matching provided coords."""
    shape = (len(time_coord), len(lat_coord), len(lon_coord))
    return xr.DataArray(
        np.full(shape, fill, dtype=dtype),
        dims=["time", "latitude", "longitude"],
        coords={"time": time_coord, "latitude": lat_coord, "longitude": lon_coord},
    )


# Process PM_FRP variable (Fire-related particulate matter from wildfires)
# cnc_PM_FRP represents PM2.5 from wildfire smoke
if "cnc_PM_FRP" in xarray_is4fires_data:
    xarray_processed["cnc_PM_FRP"] = (
        xarray_is4fires_data["cnc_PM_FRP"] * KG_M3_TO_UG_M3
    ).astype(np.float32)
    xarray_processed["cnc_PM_FRP"].attrs["units"] = "µg/m³"
    xarray_processed["cnc_PM_FRP"].attrs["long_name"] = (
        "Fire-related particulate matter (PM2.5) from wildfire smoke"
    )
    xarray_processed["cnc_PM_FRP"].attrs["source"] = "FMI IS4FIRES v2.0"
    logger.info("Loaded and converted cnc_PM_FRP from kg/m³ to µg/m³")
else:
    logger.warning("cnc_PM_FRP not found in dataset")
    xarray_processed["cnc_PM_FRP"] = _make_nan_dataarray(
        xarray_processed.time, xarray_processed.latitude, xarray_processed.longitude
    )

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
# Save the time pickle locally first
pickle_file_path = os.path.join(forecast_process_dir, "IS4FIRES.time.pickle")
with open(pickle_file_path, "wb") as file:
    pickle.dump(origintime, file)

if saveType == "S3":
    # Zip the Zarr directory and upload the zip to S3 (pattern used by other ingests)
    zip_base = os.path.join(forecast_process_dir, "IS4FIRES.zarr")
    # This will create IS4FIRES.zarr.zip in forecast_process_dir
    shutil.make_archive(zip_base, "zip", forecast_process_path + "_.zarr")
    zip_path = zip_base + ".zip"

    # Upload the zarr zip and time pickle to S3
    s3.put_file(
        zip_path,
        os.path.join(forecast_path, ingestVersion, "IS4FIRES.zarr.zip"),
    )

    s3.put_file(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "IS4FIRES.time.pickle"),
    )

    logger.info("Uploaded IS4FIRES zarr zip and time pickle to S3.")
else:
    # Move the time pickle to final local location
    shutil.move(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "IS4FIRES.time.pickle"),
    )

    # Copy Zarr to final location
    shutil.copytree(
        forecast_process_path + "_.zarr",
        forecast_path + "/" + ingestVersion + "/IS4FIRES.zarr",
        dirs_exist_ok=True,
    )
    logger.info(
        f"Saved IS4FIRES data locally to {forecast_path}/{ingestVersion}/IS4FIRES.zarr"
    )

# Clean up temporary files and directories
try:
    shutil.rmtree(forecast_process_dir)
except FileNotFoundError:
    logger.debug(
        f"Cleanup directory {forecast_process_dir} not found; nothing to remove."
    )
except PermissionError as e:
    logger.warning(f"Permission denied removing {forecast_process_dir}: {e}")
except OSError as e:
    logger.warning(f"OS error while removing {forecast_process_dir}: {e}")

end_time = time.time()
logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
logger.info("IS4FIRES ingest script finished successfully.")
