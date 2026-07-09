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
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import dask
import dask.array as da
import numpy as np
import pandas as pd
import requests
import s3fs
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    archive_tmp_zarr_and_upload,
    close_store,
    configure_zarr_limits,
    download_extract_historic_archive,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
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
    "forecast_process_dir", default="/home/reya/Weather/Process/SILAM"
)
forecast_process_path = os.path.join(forecast_process_dir, "SILAM_Process")
hist_process_path = os.path.join(forecast_process_dir, "SILAM_Historic")
tmpDIR = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod")
historic_path = os.getenv("historic_path", default="/home/reya/Weather/History/SILAM")

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)

# Define the processing chunk size - use GFS-style chunks as SILAM is global data
processChunk = CHUNK_SIZES["SILAM"]
finalChunk = FINAL_CHUNK_SIZES["SILAM"]
hisPeriod = HISTORY_PERIODS["SILAM"]
MAP_TIME_STEPS = 36
MAP_CHUNK_SIZE = 100

HISTORIC_STEP_HOURS = 24
KG_M3_TO_UG_M3 = 1e9
KG_M2_TO_UG_M2 = 1e9
MOL_MOL_TO_PPB = 1e9

zarr_vars = (
    "time",
    "cnc_PM2_5",
    "cnc_PM10",
    "PM_FRP_column",
    "BLH",
    "cnc_O3",
    "cnc_NO2",
    "cnc_SO2",
    "cnc_CO",
)
MAP_VAR_INDICES = list(range(len(zarr_vars)))

# SILAM variable names and units:
# - cnc_PM2_5, cnc_PM10: Particulate matter in kg/m³ (need conversion to µg/m³)
# - PM_FRP_column: Fire PM column in kg/m² (need conversion to µg/m²)
# - BLH: Boundary layer height in m
# - vmr_*_gas: Gas species as volume mixing ratios in mole/mole
#   (need conversion to ppb)

base_fileserver_url = "https://thredds.silam.fmi.fi/thredds/fileServer"
base_catalog_url = "https://thredds.silam.fmi.fi/thredds/catalog"
silam_dataset_path = "silam_glob_v6_1_sfc/files"
silam_filename_pattern = re.compile(r"SILAM-AQ-sfc-glob_v6_1_(\d{10})\.nc4$")


def convert_vmr_to_ppb(vmr):
    """Convert volume mixing ratio in mole/mole to ppb."""
    return vmr * MOL_MOL_TO_PPB


def get_latest_silam_run():
    """Determines the latest available SILAM model run time from THREDDS.

    SILAM availability can lag behind wall-clock time, so inspect the server
    catalog and choose the newest listed SILAM forecast file rather than
    assuming a fixed delay.

    Returns:
        datetime: The latest model run time.
    """
    catalog_urls = (
        f"{base_catalog_url}/{silam_dataset_path}/catalog.xml",
        f"{base_catalog_url}/{silam_dataset_path}/latest.xml",
    )

    for catalog_url in catalog_urls:
        try:
            response = requests.get(catalog_url, timeout=30)
            response.raise_for_status()
            catalog = ElementTree.fromstring(response.content)
        except (ElementTree.ParseError, requests.RequestException) as e:
            logger.warning(f"Unable to read SILAM catalog {catalog_url}: {e}")
            continue

        run_times = []
        for dataset in catalog.iter():
            for dataset_value in (dataset.get("name", ""), dataset.get("urlPath", "")):
                match = silam_filename_pattern.search(dataset_value)
                if match:
                    run_times.append(
                        datetime.strptime(match.group(1), "%Y%m%d%H").replace(
                            tzinfo=timezone.utc
                        )
                    )

        if run_times:
            latest_origintime = max(run_times)
            logger.info(
                f"Latest SILAM run from server catalog {catalog_url}: "
                f"{latest_origintime}"
            )
            return latest_origintime

        logger.warning(f"No SILAM run files found in catalog {catalog_url}")

    # Fall back to the previous fixed-delay behavior if the catalog is unavailable.
    latest_origintime = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    logger.warning(f"Falling back to estimated SILAM run time: {latest_origintime}")

    return latest_origintime


def build_silam_download_url(run_time: datetime) -> str:
    """Build the SILAM THREDDS fileServer URL for a model run."""
    run_filename = f"SILAM-AQ-sfc-glob_v6_1_{run_time.strftime('%Y%m%d%H.nc4')}"
    return f"{base_fileserver_url}/{silam_dataset_path}/{run_filename}"


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
        except (requests.RequestException, OSError) as e:
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed downloading {url}: {e}"
            )
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(5)

    return False


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


def process_silam_file(local_nc_path: str) -> xr.Dataset:
    """Open and convert a SILAM NetCDF file to the standard processed dataset."""
    try:
        # Open the downloaded NetCDF file from local disk
        # SILAM data has dimensions: time, height (usually surface level), lat, lon
        xarray_silam_data = xr.open_dataset(
            local_nc_path,
            engine="netcdf4",
        )

        logger.info("Successfully opened SILAM dataset from local file")
        logger.info(f"Dataset dimensions: {xarray_silam_data.dims}")
        logger.info(f"Available variables: {list(xarray_silam_data.data_vars.keys())}")

    except (OSError, ValueError) as e:
        logger.error(f"Error opening local SILAM NetCDF file: {e}")
        raise

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
    xarray_processed = xr.Dataset(
        coords={
            "time": xarray_silam_data["time"],
            "latitude": xarray_silam_data["latitude"],
            "longitude": xarray_silam_data["longitude"],
        }
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
                xarray_processed.time,
                xarray_processed.latitude,
                xarray_processed.longitude,
            )

    # Process PM variables
    _process_pm("cnc_PM2_5", "cnc_PM2_5")
    _process_pm("cnc_PM10", "cnc_PM10")

    if "PM_FRP_column" in xarray_silam_data:
        xarray_processed["PM_FRP_column"] = (
            xarray_silam_data["PM_FRP_column"] * KG_M2_TO_UG_M2
        ).astype(np.float32)
        xarray_processed["PM_FRP_column"].attrs["units"] = "µg/m²"
        xarray_processed["PM_FRP_column"].attrs["long_name"] = "Total PM_FRP column"
        logger.info("Loaded and converted PM_FRP_column from kg/m² to µg/m²")
    else:
        logger.warning("PM_FRP_column not found in dataset")
        xarray_processed["PM_FRP_column"] = _make_nan_dataarray(
            xarray_processed.time,
            xarray_processed.latitude,
            xarray_processed.longitude,
        )

    if "BLH" in xarray_silam_data:
        xarray_processed["BLH"] = xarray_silam_data["BLH"].astype(np.float32)
        xarray_processed["BLH"].attrs["units"] = "m"
        xarray_processed["BLH"].attrs["long_name"] = "Boundary layer height"
        logger.info("Loaded BLH in m")
    else:
        logger.warning("BLH not found in dataset")
        xarray_processed["BLH"] = _make_nan_dataarray(
            xarray_processed.time,
            xarray_processed.latitude,
            xarray_processed.longitude,
        )

    # Process gas volume mixing ratio variables (convert to ppb)
    gas_variables = {
        "vmr_O3_gas": ("cnc_O3", "Ozone"),
        "vmr_NO2_gas": ("cnc_NO2", "Nitrogen dioxide"),
        "vmr_SO2_gas": ("cnc_SO2", "Sulfur dioxide"),
        "vmr_CO_gas": ("cnc_CO", "Carbon monoxide"),
    }

    for silam_var, (output_var, long_name) in gas_variables.items():
        if silam_var in xarray_silam_data:
            xarray_processed[output_var] = convert_vmr_to_ppb(
                xarray_silam_data[silam_var]
            ).astype(np.float32)
            xarray_processed[output_var].attrs["units"] = "ppb"
            xarray_processed[output_var].attrs["long_name"] = (
                f"{long_name} volume mixing ratio"
            )
            logger.info(f"Loaded and converted {silam_var} to {output_var} in ppb")
        else:
            logger.warning(
                f"{silam_var} not found in dataset, {output_var} will be NaN"
            )
            xarray_processed[output_var] = _make_nan_dataarray(
                xarray_processed.time,
                xarray_processed.latitude,
                xarray_processed.longitude,
            )

    return xarray_processed


def convert_time_coord_to_unix(xarray_processed: xr.Dataset) -> xr.Dataset:
    """Return a dataset with numeric Unix-second time coordinates."""
    time_unix = (
        pd.to_datetime(xarray_processed.time.values).astype("int64") // 1_000_000_000
    )
    return xarray_processed.assign_coords(time=time_unix.astype(np.int64))


def save_processed_zarr(xarray_processed: xr.Dataset, zarr_path: str) -> None:
    """Chunk and save a processed SILAM dataset as a zarr group."""
    xarray_processed = xarray_processed.chunk(
        chunks={
            "time": xarray_processed.time.size,
            "latitude": processChunk,
            "longitude": processChunk,
        }
    )

    with ProgressBar():
        xarray_processed.to_zarr(
            zarr_path,
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )


# Create new directory for processing if it does not exist
shutil.rmtree(forecast_process_dir, ignore_errors=True)
os.makedirs(forecast_process_dir, exist_ok=True)

os.makedirs(tmpDIR, exist_ok=True)
os.makedirs(hist_process_path, exist_ok=True)
if saveType == "S3":
    s3.mkdirs(historic_path, exist_ok=True)
else:
    os.makedirs(historic_path, exist_ok=True)

if saveType == "Download":
    os.makedirs(os.path.join(forecast_path, ingestVersion), exist_ok=True)

start_time = time.time()

# Get the latest model run time
origintime = get_latest_silam_run()
download_url = build_silam_download_url(origintime)
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

# %% Download and process the latest SILAM NetCDF file
if not download_silam_file(download_url, local_nc_path):
    logger.critical(f"Failed to download SILAM data from {download_url}. Exiting.")
    sys.exit(1)

try:
    xarray_processed = process_silam_file(local_nc_path)
except (OSError, ValueError):
    logger.critical("Failed to access SILAM data. Exiting.")
    sys.exit(1)

logger.info(f"Saving processed forecast data to: {forecast_process_path}_.zarr")
xarray_forecast_processed = convert_time_coord_to_unix(xarray_processed)
save_processed_zarr(xarray_forecast_processed, forecast_process_path + "_.zarr")
logger.info("Saved forecast Zarr data to disk.")

# %% Historic data
# Store prior daily SILAM slices separately, then merge them with the current forecast
# using the same dask stacking approach used by GFS/HRRR.
for hours_offset in range(hisPeriod, 0, -HISTORIC_STEP_HOURS):
    hist_start = origintime - timedelta(hours=hours_offset)
    hist_end = hist_start + timedelta(hours=HISTORIC_STEP_HOURS + 1)

    timestamp = hist_start.strftime("%Y%m%dT%H%M%SZ")

    if saveType == "S3":
        s3_path = f"{historic_path}/SILAM_Hist_v3{timestamp}.zarr.tar.gz"
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            logger.info("Historic SILAM file already exists in S3: %s", s3_path)
            continue
    else:
        local_path = f"{historic_path}/SILAM_Hist_v3{timestamp}.zarr"
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info("Historic SILAM file already exists locally: %s", local_path)
            continue

    hist_url = build_silam_download_url(hist_start)
    hist_nc_path = os.path.join(tmpDIR, f"SILAM_hist_{timestamp}.nc")
    logger.info("Downloading historic SILAM run: %s", timestamp)

    if not download_silam_file(hist_url, hist_nc_path):
        logger.warning("Skipping missing historic SILAM run: %s", timestamp)
        continue

    try:
        xarray_hist_processed = process_silam_file(hist_nc_path)
    except (OSError, ValueError):
        logger.warning("Skipping unreadable historic SILAM run: %s", timestamp)
        continue

    # Keep only the non-overlapping historic valid-time slice for this run.
    slice_end = hist_end - timedelta(seconds=1)
    xarray_hist_processed = xarray_hist_processed.sel(
        time=slice(hist_start.replace(tzinfo=None), slice_end.replace(tzinfo=None))
    )

    if xarray_hist_processed.sizes.get("time", 0) == 0:
        logger.warning("No historic SILAM times found for slice: %s", timestamp)
        continue

    hist_tmp_zarr_path = hist_process_path + "_SILAM_Hist_TMP.zarr"
    xarray_hist_processed = convert_time_coord_to_unix(xarray_hist_processed)
    save_processed_zarr(xarray_hist_processed, hist_tmp_zarr_path)

    if saveType == "S3":
        archive_tmp_zarr_and_upload(
            tmp_zarr_path=hist_tmp_zarr_path,
            s3_path=s3_path,
            archive_member_name="SILAM_Hist.zarr",
            s3=s3,
        )
    else:
        os.rename(hist_tmp_zarr_path, local_path)
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info("Saved historic SILAM slice: %s", timestamp)

# %% Merge historic and forecast datasets into final stacked zarr
if saveType == "S3":
    local_temp_dir = forecast_process_path + "_s3_temp_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)
    historic_zarr_paths = []
    for hours_offset in range(hisPeriod, 0, -HISTORIC_STEP_HOURS):
        timestamp = (origintime - timedelta(hours=hours_offset)).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        extracted_path = download_extract_historic_archive(
            s3=s3,
            historic_path=historic_path,
            final_zarr_name=f"SILAM_Hist_v3{timestamp}.zarr",
            extracted_store_name="SILAM_Hist.zarr",
            local_temp_dir=local_temp_dir,
            expected_vars=zarr_vars,
        )
        if extracted_path is not None:
            historic_zarr_paths.append(extracted_path)
else:
    historic_zarr_paths = [
        f"{historic_path}/SILAM_Hist_v3"
        f"{(origintime - timedelta(hours=hours_offset)).strftime('%Y%m%dT%H%M%SZ')}"
        ".zarr"
        for hours_offset in range(hisPeriod, 0, -HISTORIC_STEP_HOURS)
        if os.path.exists(
            f"{historic_path}/SILAM_Hist_v3"
            f"{(origintime - timedelta(hours=hours_offset)).strftime('%Y%m%dT%H%M%SZ')}"
            ".zarr"
        )
    ]

lat_count = xarray_forecast_processed.sizes["latitude"]
lon_count = xarray_forecast_processed.sizes["longitude"]
dask_var_arrays_list = []
dask_interp_arrays = []

for dask_var in zarr_vars:
    for historic_zarr_path in historic_zarr_paths:
        try:
            dask_var_arrays_list.append(
                da.from_zarr(historic_zarr_path, component=dask_var, inline_array=True)
            )
        except (FileNotFoundError, KeyError):
            logger.info("Missing %s in historic zarr: %s", dask_var, historic_zarr_path)

    dask_forecast_array = da.from_zarr(
        forecast_process_path + "_.zarr", component=dask_var, inline_array=True
    )

    if dask_var == "time":
        dask_time_arrays = [da.squeeze(array) for array in dask_var_arrays_list]
        dask_time_arrays.append(dask_forecast_array)
        dask_times_concatenated = da.concatenate(dask_time_arrays, axis=0).astype(
            "float32"
        )

        output_array = da.broadcast_to(
            dask_times_concatenated[:, None, None],
            (dask_times_concatenated.shape[0], lat_count, lon_count),
        ).rechunk((dask_times_concatenated.shape[0], processChunk, processChunk))
        dask_interp_arrays.append(output_array)
    else:
        dask_data_arrays = dask_var_arrays_list + [dask_forecast_array]
        output_array = da.concatenate(dask_data_arrays, axis=0)
        dask_interp_arrays.append(
            output_array[:, :, :]
            .rechunk((output_array.shape[0], processChunk, processChunk))
            .astype("float32")
        )

    dask_var_arrays_list = []
    logger.info("Processed variable: %s", dask_var)

# Merge the arrays into a single 4D array
merged_arrays = da.stack(dask_interp_arrays, axis=0)
merged_arrays_masked = mask_invalid_data(
    merged_arrays, ignoreAxis=[zarr_vars.index("PM_FRP_column")]
)

# Write out to disk. This intermediate step avoids memory overflow.
with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
    merged_arrays_masked.to_zarr(
        forecast_process_path + "_stack.zarr",
        overwrite=True,
        compute=True,
    )

stacked_array_disk = da.from_zarr(forecast_process_path + "_stack.zarr")
stacked_array_padded = pad_to_chunk_size(stacked_array_disk, finalChunk)

if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/SILAM.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/SILAM.zarr")

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=stacked_array_padded.shape,
    chunks=(
        len(zarr_vars),
        stacked_array_padded.shape[1],
        finalChunk,
        finalChunk,
    ),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)

with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        stacked_array_padded.round(5).rechunk(
            (
                len(zarr_vars),
                stacked_array_padded.shape[1],
                finalChunk,
                finalChunk,
            )
        ).to_zarr(zarr_array, overwrite=True, compute=True)

close_store(zarr_store)

# Rechunk map data for faster web access.
# Mirrors the GFS map zarr layout: one named array per variable, with all map
# times in a single time chunk and 100x100 spatial chunks for fast tile reads.
# Map extent: -12 to +24 hours around the forecast origin (36 hours total).
stacked_array_maps = pad_to_chunk_size(stacked_array_disk, MAP_CHUNK_SIZE)

if saveType == "S3":
    zarr_store_maps = zarr.storage.ZipStore(
        forecast_process_dir + "/SILAM_Maps.zarr.zip", mode="a"
    )
else:
    zarr_store_maps = zarr.storage.LocalStore(forecast_process_dir + "/SILAM_Maps.zarr")

for var_idx in MAP_VAR_INDICES:
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
                stacked_array_maps[var_idx, hisPeriod - 12 : hisPeriod + 24, :, :],
                (MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
            ).to_zarr(zarr_array, overwrite=True, compute=True)

    logger.info("Created SILAM map data for %s", zarr_vars[var_idx])

close_store(zarr_store_maps)

# %% Final output handling and cleanup
pickle_file_path = os.path.join(forecast_process_dir, "SILAM.time.pickle")
with open(pickle_file_path, "wb") as file:
    pickle.dump(origintime, file)

if saveType == "S3":
    s3.put_file(
        forecast_process_dir + "/SILAM.zarr.zip",
        os.path.join(forecast_path, ingestVersion, "SILAM.zarr.zip"),
    )

    s3.put_file(
        forecast_process_dir + "/SILAM_Maps.zarr.zip",
        os.path.join(forecast_path, ingestVersion, "SILAM_Maps.zarr.zip"),
    )

    s3.put_file(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "SILAM.time.pickle"),
    )

    logger.info("Uploaded SILAM zarrs and time pickle to S3.")
else:
    shutil.move(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "SILAM.time.pickle"),
    )

    shutil.copytree(
        forecast_process_dir + "/SILAM.zarr",
        forecast_path + "/" + ingestVersion + "/SILAM.zarr",
        dirs_exist_ok=True,
    )

    shutil.copytree(
        forecast_process_dir + "/SILAM_Maps.zarr",
        forecast_path + "/" + ingestVersion + "/SILAM_Maps.zarr",
        dirs_exist_ok=True,
    )
    logger.info(
        f"Saved SILAM data locally to {forecast_path}/{ingestVersion}/SILAM.zarr "
        f"and {forecast_path}/{ingestVersion}/SILAM_Maps.zarr"
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
