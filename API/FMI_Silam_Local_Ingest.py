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
import requests
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import CHUNK_SIZES
from API.silam_conversion import (
    KG_M3_TO_UG_M3,
    MOLAR_MASS_CO,
    MOLAR_MASS_NO2,
    MOLAR_MASS_O3,
    MOLAR_MASS_SO2,
    convert_vmr_to_concentration,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingestVersion = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/SILAM"
)
forecast_process_path = os.path.join(forecast_process_dir, "SILAM_Process")
tmpDIR = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/SILAM")

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing chunk size - use GFS chunk sizes as SILAM is global data
processChunk = CHUNK_SIZES.get("GFS", 50)

# Standard air density constant
STANDARD_AIR_DENSITY = 1.225  # kg/m³ at sea level (used as fallback)

# SILAM variable names and units:
# - cnc_PM2_5, cnc_PM10: Particulate matter in kg/m³ (need conversion to µg/m³)
# - vmr_*_gas: Gas species as volume mixing ratios in mole/mole (need conversion using air density and molar mass)
# - air_dens: Air density in kg/m³ (used for volume mixing ratio conversions)


def get_latest_silam_run():
    """Determines the latest available SILAM model run time.

    SILAM air quality data is updated once daily at 00 UTC. This function
    accounts for a delay in data availability.

    Returns:
        datetime: The latest model run time.
    """
    now_utc = datetime.now(timezone.utc)

    # SILAM updates once daily at 00 UTC with the date being one day behind current time
    latest_origintime = (now_utc - timedelta(days=1)).replace(
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
    if not os.path.exists(os.path.join(forecast_path, ingestVersion)):
        os.makedirs(os.path.join(forecast_path, ingestVersion))

start_time = time.time()

# Get the latest model run time
origintime = get_latest_silam_run()

# Construct the download URL for SILAM global forecast
# SILAM data is available via THREDDS fileServer (plain HTTP) service, which
# is used here instead of OPeNDAP so the full NetCDF file is downloaded to
# local disk before being opened with xarray. This avoids flaky/slow remote
# reads over OPeNDAP and lets us retry the download independently of xarray.
# Using SILAM version 6.1 (silam_glob_v6_1_sfc) - the latest available version
base_fileserver_url = "https://thredds.silam.fmi.fi/thredds/fileServer"
silam_dataset_path = "silam_glob_v6_1_sfc/files"
run_filename = f"silam_glob_v6_1_sfc_RUN_{origintime.strftime('%Y-%m-%dT%H:%M:%SZ')}"
run_filename = f"SILAM-AQ-sfc-glob_v6_1_{origintime.strftime('%Y%m%d%H.nc4')}"

download_url = f"{base_fileserver_url}/{silam_dataset_path}/{run_filename}"
local_nc_path = os.path.join(tmpDIR, "SILAM_latest.nc")

logger.info(f"Attempting to download SILAM data from: {download_url}")
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


# %% Download the SILAM NetCDF file to disk, then load it with xarray
def download_silam_file(url: str, local_path: str, max_retries: int = 3) -> bool:
    """Downloads the SILAM NetCDF file from THREDDS to a local path.

    Args:
        url: The fileServer URL of the NetCDF file to download.
        local_path: The local filesystem path to save the file to.
        max_retries: Number of attempts before giving up.

    Returns:
        True if the download succeeded, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                if response.status_code == 404:
                    logger.warning(f"File not found: {url}")
                    return False
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            logger.info(f"Downloaded SILAM file to: {local_path}")
            return True
        except (requests.RequestException, IOError, OSError) as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed downloading {url}: {e}")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(5)

    return False


if not download_silam_file(download_url, local_nc_path):
    logger.critical(f"Failed to download SILAM data from {download_url}. Exiting.")
    sys.exit(1)

try:
    # Open the downloaded NetCDF file from local disk
    # SILAM data has dimensions: time, height (usually surface level), lat, lon
    xarray_silam_data = xr.open_dataset(
        local_nc_path,
        engine="netcdf4",
        chunks={"time": 24, "lat": processChunk, "lon": processChunk},
    )

    logger.info("Successfully opened SILAM dataset from local file")
    logger.info(f"Dataset dimensions: {xarray_silam_data.dims}")
    logger.info(f"Available variables: {list(xarray_silam_data.data_vars.keys())}")

except (IOError, OSError, ValueError) as e:
    logger.error(f"Error opening local SILAM NetCDF file: {e}")
    logger.critical("Failed to access SILAM data. Exiting.")
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

# Create processed dataset
xarray_processed = xr.Dataset(coords=xarray_silam_data.coords)
xarray_processed["time"] = xarray_silam_data["time"]


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


def _process_pm(var_in_name, var_out_name):
    """Process particulate mass variables (kg/m3 -> µg/m3) or create NaN array."""
    if var_in_name in xarray_silam_data:
        xarray_processed[var_out_name] = (
            xarray_silam_data[var_in_name] * KG_M3_TO_UG_M3
        ).astype(np.float32)
        xarray_processed[var_out_name].attrs["units"] = "µg/m³"
        xarray_processed[var_out_name].attrs["long_name"] = (
            "PM2.5 concentration" if "2_5" in var_out_name else "PM10 concentration"
        )
        logger.info(f"Loaded and converted {var_in_name} from kg/m³ to µg/m³")
    else:
        logger.warning(f"{var_in_name} not found in dataset")
        xarray_processed[var_out_name] = _make_nan_dataarray(
            xarray_processed.time, xarray_processed.latitude, xarray_processed.longitude
        )


# Process PM variables
_process_pm("cnc_PM2_5", "cnc_PM2_5")
_process_pm("cnc_PM10", "cnc_PM10")

# Load air density for volume mixing ratio conversions
if "air_dens" in xarray_silam_data:
    air_density = xarray_silam_data["air_dens"].astype(np.float32)
    logger.info("Loaded air_dens for volume mixing ratio conversions")
else:
    logger.warning(
        f"air_dens not found, using standard air density of {STANDARD_AIR_DENSITY} kg/m³"
    )
    air_density = _make_nan_dataarray(
        xarray_processed.time,
        xarray_processed.latitude,
        xarray_processed.longitude,
        fill=STANDARD_AIR_DENSITY,
    )


# Process gas volume mixing ratio variables (convert to µg/m³)
gas_variables = {
    "vmr_O3_gas": ("cnc_O3", MOLAR_MASS_O3, "Ozone"),
    "vmr_NO2_gas": ("cnc_NO2", MOLAR_MASS_NO2, "Nitrogen dioxide"),
    "vmr_SO2_gas": ("cnc_SO2", MOLAR_MASS_SO2, "Sulfur dioxide"),
    "vmr_CO_gas": ("cnc_CO", MOLAR_MASS_CO, "Carbon monoxide"),
}

for silam_var, (output_var, molar_mass, long_name) in gas_variables.items():
    if silam_var in xarray_silam_data:
        xarray_processed[output_var] = convert_vmr_to_concentration(
            xarray_silam_data[silam_var], air_density, molar_mass
        ).astype(np.float32)
        xarray_processed[output_var].attrs["units"] = "µg/m³"
        xarray_processed[output_var].attrs["long_name"] = f"{long_name} concentration"
        logger.info(f"Loaded and converted {silam_var} to {output_var} in µg/m³")
    else:
        logger.warning(f"{silam_var} not found in dataset, {output_var} will be NaN")
        xarray_processed[output_var] = _make_nan_dataarray(
            xarray_processed.time, xarray_processed.latitude, xarray_processed.longitude
        )


def _values_or_nan(ds, var_name, fallback_shape):
    return (
        ds[var_name].values
        if var_name in ds
        else np.full(fallback_shape, np.nan, dtype=np.float32)
    )


# Create fallback shape for missing data (3D: time, latitude, longitude)
fallback_shape = (
    len(xarray_processed.time),
    len(xarray_processed.latitude),
    len(xarray_processed.longitude),
)

pm25_data = _values_or_nan(xarray_processed, "cnc_PM2_5", fallback_shape)
pm10_data = _values_or_nan(xarray_processed, "cnc_PM10", fallback_shape)
o3_data = _values_or_nan(xarray_processed, "cnc_O3", fallback_shape)
no2_data = _values_or_nan(xarray_processed, "cnc_NO2", fallback_shape)
so2_data = _values_or_nan(xarray_processed, "cnc_SO2", fallback_shape)
co_data = _values_or_nan(xarray_processed, "cnc_CO", fallback_shape)

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
pickle_file_path = os.path.join(forecast_process_dir, "SILAM.time.pickle")
with open(pickle_file_path, "wb") as file:
    pickle.dump(origintime, file)

if saveType == "S3":
    # Zip the Zarr directory and upload the zip to S3 (pattern used by other ingests)
    zip_base = os.path.join(forecast_process_dir, "SILAM.zarr")
    # This will create SILAM.zarr.zip in forecast_process_dir
    shutil.make_archive(zip_base, "zip", forecast_process_path + "_.zarr")
    zip_path = zip_base + ".zip"

    # Upload the zarr zip and time pickle to S3
    s3.put_file(
        zip_path,
        os.path.join(forecast_path, ingestVersion, "SILAM.zarr.zip"),
    )

    s3.put_file(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "SILAM.time.pickle"),
    )

    logger.info("Uploaded SILAM zarr zip and time pickle to S3.")
else:
    # Move the time pickle to final local location
    shutil.move(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "SILAM.time.pickle"),
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
logger.info("SILAM ingest script finished successfully.")
