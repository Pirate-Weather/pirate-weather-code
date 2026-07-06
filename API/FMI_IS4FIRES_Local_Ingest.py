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
    "forecast_process_dir", default="/home/reya/Weather/IS4FIRES"
)
forecast_process_path = os.path.join(forecast_process_dir, "IS4FIRES_Process")
hist_process_path = os.path.join(forecast_process_dir, "IS4FIRES_Historic")
tmpDIR = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod/IS4FIRES")
historic_path = os.getenv(
    "historic_path", default="/home/reya/Weather/History/IS4FIRES"
)

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

# Define the processing chunk size - use SILAM-style chunks as IS4FIRES is global data.
processChunk = CHUNK_SIZES["IS4FIRES"]
finalChunk = FINAL_CHUNK_SIZES["IS4FIRES"]
hisPeriod = HISTORY_PERIODS["IS4FIRES"]
HISTORIC_STEP_HOURS = 24

zarr_vars = ("time", "cnc_PM_FRP")

# IS4FIRES variable names and units:
# - cnc_PM_FRP: Fire-related particulate matter in kg/m³ (need conversion to µg/m³)
#   This represents PM2.5 from wildfire smoke emissions

base_fileserver_url = "https://thredds.silam.fmi.fi/thredds/fileServer"
base_catalog_url = "https://thredds.silam.fmi.fi/thredds/catalog"
is4fires_files_dataset_path = "i4f20-fc/files"
is4fires_runs_dataset_path = "i4f20-fc/runs"
is4fires_file_pattern = re.compile(r"is4fires_cnc_(\d{8})fc\.nc4$")
is4fires_run_pattern = re.compile(r"IS4FIRES-fc_RUN_(\d{4}-\d{2}-\d{2})T00:00:00Z$")


def get_latest_is4fires_run():
    """Determines the latest available IS4FIRES model run time from THREDDS."""
    catalog_urls = (
        f"{base_catalog_url}/{is4fires_files_dataset_path}/catalog.xml",
        f"{base_catalog_url}/{is4fires_files_dataset_path}/latest.xml",
        f"{base_catalog_url}/{is4fires_runs_dataset_path}/catalog.xml",
    )

    for catalog_url in catalog_urls:
        try:
            response = requests.get(catalog_url, timeout=30)
            response.raise_for_status()
            catalog = ElementTree.fromstring(response.content)
        except (ElementTree.ParseError, requests.RequestException) as e:
            logger.warning(f"Unable to read IS4FIRES catalog {catalog_url}: {e}")
            continue

        run_times = []
        for dataset in catalog.iter():
            for dataset_value in (dataset.get("name", ""), dataset.get("urlPath", "")):
                file_match = is4fires_file_pattern.search(dataset_value)
                if file_match:
                    run_times.append(
                        datetime.strptime(file_match.group(1), "%Y%m%d").replace(
                            tzinfo=timezone.utc
                        )
                    )
                    continue

                run_match = is4fires_run_pattern.search(dataset_value)
                if run_match:
                    run_times.append(
                        datetime.strptime(run_match.group(1), "%Y-%m-%d").replace(
                            tzinfo=timezone.utc
                        )
                    )

        if run_times:
            latest_origintime = max(run_times)
            logger.info(
                f"Latest IS4FIRES run from server catalog {catalog_url}: "
                f"{latest_origintime}"
            )
            return latest_origintime

        logger.warning(f"No IS4FIRES run files found in catalog {catalog_url}")

    latest_origintime = (datetime.now(timezone.utc) - timedelta(days=2)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    logger.warning(f"Falling back to estimated IS4FIRES run time: {latest_origintime}")

    return latest_origintime


def build_is4fires_download_url(run_time: datetime) -> str:
    """Build the IS4FIRES THREDDS fileServer URL for a model run."""
    run_filename = f"is4fires_cnc_{run_time.strftime('%Y%m%d')}fc.nc4"
    return f"{base_fileserver_url}/{is4fires_files_dataset_path}/{run_filename}"


def download_is4fires_file(url: str, local_path: str, max_retries: int = 3) -> bool:
    """Download an IS4FIRES NetCDF file from THREDDS to a local path."""
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
            logger.info(f"Downloaded IS4FIRES file to: {local_path}")
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


def decode_is4fires_time_coord(xarray_is4fires_data: xr.Dataset) -> xr.Dataset:
    """Decode IS4FIRES numeric time without xarray's native time decoder.

    The IS4FIRES raw NetCDF files use units like
    ``seconds since 2026-07-04 00:00:00 UTC``. In this environment, xarray's
    netCDF4 time decoder segfaults on these files, while opening with
    ``decode_times=False`` works. Decode the coordinate explicitly instead.
    """
    time_units = xarray_is4fires_data["time"].attrs.get("units", "")
    match = re.match(
        r"^(seconds|minutes|hours|days) since "
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?(?: UTC)?$",
        time_units,
    )
    if not match:
        raise ValueError(f"Unsupported IS4FIRES time units: {time_units!r}")

    unit_name, base_time_text = match.groups()
    timedelta_units = {
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
    }
    base_time = pd.Timestamp(base_time_text)
    decoded_times = (
        base_time
        + pd.to_timedelta(
            xarray_is4fires_data["time"].values,
            unit=timedelta_units[unit_name],
        )
    ).to_numpy(dtype="datetime64[ns]")

    return xarray_is4fires_data.assign_coords(time=decoded_times)


def process_is4fires_file(local_nc_path: str) -> xr.Dataset:
    """Open and convert an IS4FIRES NetCDF file to the standard processed dataset."""
    try:
        xarray_is4fires_data = xr.open_dataset(
            local_nc_path,
            engine="netcdf4",
            decode_times=False,
            chunks={"time": 24, "lat": processChunk, "lon": processChunk},
        )
        xarray_is4fires_data = decode_is4fires_time_coord(xarray_is4fires_data)

        logger.info("Successfully opened IS4FIRES dataset from local file")
        logger.info(f"Dataset dimensions: {xarray_is4fires_data.dims}")
        logger.info(
            f"Available variables: {list(xarray_is4fires_data.data_vars.keys())}"
        )

    except (OSError, ValueError) as e:
        logger.error(f"Error opening local IS4FIRES NetCDF file: {e}")
        raise

    if "height" in xarray_is4fires_data.dims:
        xarray_is4fires_data = xarray_is4fires_data.isel(height=0)
    elif "level" in xarray_is4fires_data.dims:
        xarray_is4fires_data = xarray_is4fires_data.isel(level=0)

    if "lat" in xarray_is4fires_data.coords:
        xarray_is4fires_data = xarray_is4fires_data.rename({"lat": "latitude"})
    if "lon" in xarray_is4fires_data.coords:
        xarray_is4fires_data = xarray_is4fires_data.rename({"lon": "longitude"})

    xarray_processed = xr.Dataset(
        coords={
            "time": xarray_is4fires_data["time"],
            "latitude": xarray_is4fires_data["latitude"],
            "longitude": xarray_is4fires_data["longitude"],
        }
    )

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

    return xarray_processed


def convert_time_coord_to_unix(xarray_processed: xr.Dataset) -> xr.Dataset:
    """Return a dataset with numeric Unix-second time coordinates."""
    time_unix = (
        pd.to_datetime(xarray_processed.time.values).astype("int64") // 1_000_000_000
    )
    return xarray_processed.assign_coords(time=time_unix.astype(np.int64))


def save_processed_zarr(xarray_processed: xr.Dataset, zarr_path: str) -> None:
    """Chunk and save a processed IS4FIRES dataset as a zarr group."""
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
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

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
origintime = get_latest_is4fires_run()
download_url = build_is4fires_download_url(origintime)
local_nc_path = os.path.join(tmpDIR, "IS4FIRES_latest.nc")

logger.info(f"Attempting to download IS4FIRES data from: {download_url}")
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

# %% Download and process the latest IS4FIRES NetCDF file
if not download_is4fires_file(download_url, local_nc_path):
    logger.critical(f"Failed to download IS4FIRES data from {download_url}. Exiting.")
    sys.exit(1)


xarray_processed = process_is4fires_file(local_nc_path)


logger.info(f"Saving processed forecast data to: {forecast_process_path}_.zarr")
xarray_forecast_processed = convert_time_coord_to_unix(xarray_processed)
save_processed_zarr(xarray_forecast_processed, forecast_process_path + "_.zarr")
logger.info("Saved forecast Zarr data to disk.")

# %% Historic data
# Store prior daily IS4FIRES slices separately, then merge with the current forecast.
for hours_offset in range(hisPeriod, 0, -HISTORIC_STEP_HOURS):
    hist_start = origintime - timedelta(hours=hours_offset)
    hist_end = min(
        hist_start + timedelta(hours=HISTORIC_STEP_HOURS),
        origintime,
    )
    timestamp = hist_start.strftime("%Y%m%dT%H%M%SZ")

    if saveType == "S3":
        s3_path = f"{historic_path}/IS4FIRES_Hist_v3{timestamp}.zarr.tar.gz"
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            logger.info("Historic IS4FIRES file already exists in S3: %s", s3_path)
            continue
    else:
        local_path = f"{historic_path}/IS4FIRES_Hist_v3{timestamp}.zarr"
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info("Historic IS4FIRES file already exists locally: %s", local_path)
            continue

    hist_url = build_is4fires_download_url(hist_start)
    hist_nc_path = os.path.join(tmpDIR, f"IS4FIRES_hist_{timestamp}.nc")
    logger.info("Downloading historic IS4FIRES run: %s", timestamp)

    if not download_is4fires_file(hist_url, hist_nc_path):
        logger.warning("Skipping missing historic IS4FIRES run: %s", timestamp)
        continue

    try:
        xarray_hist_processed = process_is4fires_file(hist_nc_path)
    except (OSError, ValueError):
        logger.warning("Skipping unreadable historic IS4FIRES run: %s", timestamp)
        continue

    slice_end = hist_end - timedelta(seconds=1)
    xarray_hist_processed = xarray_hist_processed.sel(
        time=slice(hist_start.replace(tzinfo=None), slice_end.replace(tzinfo=None))
    )

    if xarray_hist_processed.sizes.get("time", 0) == 0:
        logger.warning("No historic IS4FIRES times found for slice: %s", timestamp)
        continue

    hist_tmp_zarr_path = hist_process_path + "_IS4FIRES_Hist_TMP.zarr"
    xarray_hist_processed = convert_time_coord_to_unix(xarray_hist_processed)
    save_processed_zarr(xarray_hist_processed, hist_tmp_zarr_path)

    if saveType == "S3":
        archive_tmp_zarr_and_upload(
            tmp_zarr_path=hist_tmp_zarr_path,
            s3_path=s3_path,
            archive_member_name="IS4FIRES_Hist.zarr",
            s3=s3,
        )
    else:
        os.rename(hist_tmp_zarr_path, local_path)
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info("Saved historic IS4FIRES slice: %s", timestamp)

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
            final_zarr_name=f"IS4FIRES_Hist_v3{timestamp}.zarr",
            extracted_store_name="IS4FIRES_Hist.zarr",
            local_temp_dir=local_temp_dir,
            expected_vars=zarr_vars,
        )
        if extracted_path is not None:
            historic_zarr_paths.append(extracted_path)
else:
    historic_zarr_paths = [
        f"{historic_path}/IS4FIRES_Hist_v3"
        f"{(origintime - timedelta(hours=hours_offset)).strftime('%Y%m%dT%H%M%SZ')}"
        ".zarr"
        for hours_offset in range(hisPeriod, 0, -HISTORIC_STEP_HOURS)
        if os.path.exists(
            f"{historic_path}/IS4FIRES_Hist_v3"
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

        times_array = dask_times_concatenated.compute()
        output_array = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(times_array, axis=1), axis=1),
                (1, lat_count, lon_count),
            )
        ).rechunk((len(times_array), processChunk, processChunk))
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

merged_arrays = da.stack(dask_interp_arrays, axis=0)
merged_arrays_masked = mask_invalid_data(merged_arrays)

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
        forecast_process_dir + "/IS4FIRES.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/IS4FIRES.zarr")

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

# %% Final output handling and cleanup
pickle_file_path = os.path.join(forecast_process_dir, "IS4FIRES.time.pickle")
with open(pickle_file_path, "wb") as file:
    pickle.dump(origintime, file)

if saveType == "S3":
    s3.put_file(
        forecast_process_dir + "/IS4FIRES.zarr.zip",
        os.path.join(forecast_path, ingestVersion, "IS4FIRES.zarr.zip"),
    )

    s3.put_file(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "IS4FIRES.time.pickle"),
    )

    logger.info("Uploaded IS4FIRES zarr zip and time pickle to S3.")
else:
    shutil.move(
        pickle_file_path,
        os.path.join(forecast_path, ingestVersion, "IS4FIRES.time.pickle"),
    )

    shutil.copytree(
        forecast_process_dir + "/IS4FIRES.zarr",
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
