# %% Regional Deterministic Air Quality Analysis (RDAQA) Processing Script
# Author: Alexander Rey
# Date: June 2026

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
import zipfile
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

import dask.array as da
import numpy as np
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    close_store,
    mask_invalid_data,
    pad_to_chunk_size,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Constants
RDAQA_BASE_URL = "https://dd.weather.gc.ca"
RDAQA_VARS = ["PM2.5", "PM10", "NO2", "O3", "SO2"]

# Molecular volume of an ideal gas at 25°C and 1 atm is ~24.465 Liters/mol
MOLAR_VOLUME_25C = 24.465

# Conversion factors to µg/m³
KG_M3_TO_UG_M3 = 1e9
O3_PPB_TO_UG_M3 = 48.00 / MOLAR_VOLUME_25C  # ~1.962
NO2_PPB_TO_UG_M3 = 46.01 / MOLAR_VOLUME_25C  # ~1.881
SO2_PPB_TO_UG_M3 = 64.06 / MOLAR_VOLUME_25C  # ~2.618

# %% Helper functions


def get_latest_rdaqa_run():
    """Finds the latest available RDAQA-Prelim run.
    The model runs hourly, but preliminary files emerge ~1h 30m after real-time.
    We verify availability iteratively looking backwards.
    """
    now_utc = datetime.now(timezone.utc)
    for hour_offset in range(0, 5):
        test_time = now_utc - timedelta(hours=hour_offset)
        test_time = test_time.replace(minute=0, second=0, microsecond=0)

        url = build_rdaqa_url(test_time, "O3")
        try:
            req = Request(url, method="HEAD")
            with urlopen(req) as response:
                if response.status == 200:
                    logger.info(f"Found latest valid RDAQA runtime at: {test_time}")
                    return test_time
        except Exception:
            continue

    fallback = (now_utc - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    return fallback


def build_rdaqa_url(run_time, variable):
    """Constructs the canonical ECCC MSC Datamart URL for RDAQA-Prelim."""
    hh = run_time.strftime("%H")
    yyyymmdd = run_time.strftime("%Y%m%d")
    filename = (
        f"{yyyymmdd}T{hh}Z_MSC_RDAQA-Prelim_{variable}_Sfc_RLatLon0.09_PT0H.grib2"
    )
    return f"{RDAQA_BASE_URL}/{yyyymmdd}/WXO-DD/model_rdaqa/10km/{hh}/{filename}"


def download_rdaqa_file(url, dest_path):
    """Downloads a GRIB2 target file locally."""
    try:
        logger.info(f"Downloading: {url}")
        with urlopen(url, timeout=30) as response, open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        logger.warning(f"Failed to download from {url}: {e}")
        return False


def convert_to_ug_m3(da, var_name):
    """Converts variables from GRIB native units (kg/m3 or ppb) to µg/m³."""
    if var_name in ["PM2.5", "PM10"]:
        logger.info(f"Converting {var_name} from kg/m³ to µg/m³")
        return da * KG_M3_TO_UG_M3
    elif var_name == "O3":
        logger.info("Converting O3 from ppb to µg/m³")
        return da * O3_PPB_TO_UG_M3
    elif var_name == "NO2":
        logger.info("Converting NO2 from ppb to µg/m³")
        return da * NO2_PPB_TO_UG_M3
    elif var_name == "SO2":
        logger.info("Converting SO2 from ppb to µg/m³")
        return da * SO2_PPB_TO_UG_M3
    return da


def download_extract_historic_zip(s3, s3_zip_path, local_temp_dir):
    """Download and extract an existing RDAQA historic zip archive."""
    os.makedirs(local_temp_dir, exist_ok=True)
    local_zarr_path = os.path.join(
        local_temp_dir, os.path.basename(s3_zip_path).removesuffix(".zip")
    )

    if os.path.exists(local_zarr_path):
        return local_zarr_path

    if not s3.exists(s3_zip_path):
        logger.warning(
            "Historic done marker exists, but archive is missing: %s", s3_zip_path
        )
        return None

    local_zip_path = os.path.join(local_temp_dir, os.path.basename(s3_zip_path))
    extract_dir = local_zarr_path + "_extract"

    try:
        s3.get_file(s3_zip_path, local_zip_path)
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(local_zip_path) as zip_file:
            zip_file.extractall(extract_dir)

        shutil.move(extract_dir, local_zarr_path)
        return local_zarr_path
    finally:
        if os.path.exists(local_zip_path):
            os.remove(local_zip_path)
        shutil.rmtree(extract_dir, ignore_errors=True)


ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/reya/Weather/RDAQA"
)
forecast_process_path = os.path.join(forecast_process_dir, "RDAQA_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")
forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod")
historic_path = os.getenv(
    "historic_path", default=os.path.join(forecast_path, "RDAQA_Hist")
)
save_type = os.getenv("save_type", default="Download")

if os.path.exists(forecast_process_dir):
    shutil.rmtree(forecast_process_dir)
os.makedirs(tmp_dir, exist_ok=True)

if save_type == "Download":
    os.makedirs(os.path.join(forecast_path, ingest_version), exist_ok=True)

s3 = s3fs.S3FileSystem(
    key=os.environ.get("AWS_KEY", ""), secret=os.environ.get("AWS_SECRET", "")
)

base_time = get_latest_rdaqa_run()

time_pickle_path = os.path.join(forecast_path, ingest_version, "RDAQA.time.pickle")
if save_type == "S3":
    s3_pickle_path = os.path.join(forecast_path, ingest_version, "RDAQA.time.pickle")
    if s3.exists(s3_pickle_path):
        with s3.open(s3_pickle_path, "rb") as f:
            if pickle.load(f) >= base_time:
                logger.info("RDAQA store is already up to date. Exiting.")
                sys.exit()
else:
    if os.path.exists(time_pickle_path):
        with open(time_pickle_path, "rb") as f:
            if pickle.load(f) >= base_time:
                logger.info("RDAQA store is already up to date. Exiting.")
                sys.exit()

downloaded_datasets = {}

for var in RDAQA_VARS:
    url = build_rdaqa_url(base_time, var)
    local_grib = os.path.join(tmp_dir, f"{var}.grib2")

    if download_rdaqa_file(url, local_grib):
        try:
            with xr.open_dataset(local_grib, engine="cfgrib", decode_times=False) as ds:
                grib_var_name = list(ds.data_vars)[0]
                da_var = ds[grib_var_name].astype(np.float32)

                # Perform unit normalization here
                da_converted = convert_to_ug_m3(da_var, var)
                downloaded_datasets[var] = da_converted
        except Exception as e:
            logger.error(f"Error reading dataset variable {var}: {e}")

if not downloaded_datasets:
    logger.error("No valid RDAQA variables could be parsed. Aborting.")
    sys.exit(1)

sample_da = list(downloaded_datasets.values())[0]
time_unix = np.array([int(base_time.timestamp())], dtype=np.int64)

zarr_store_path = forecast_process_path + "_forecast.zarr"
zarr_store = zarr.storage.LocalStore(zarr_store_path)

process_chunk = CHUNK_SIZES.get("RDAQA", 250)
final_chunk = FINAL_CHUNK_SIZES.get("RDAQA", 500)
his_period = HISTORY_PERIODS.get("RDAQA", 24)

zarr_output_vars = ["time"] + RDAQA_VARS

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarr_output_vars),
        len(time_unix),
        sample_da.shape[0],
        sample_da.shape[1],
    ),
    chunks=(1, len(time_unix), final_chunk, final_chunk),
    dtype=np.float32,
    overwrite=True,
)

dask_variable_list = []

time_da = da.from_array(time_unix, chunks=(1,)).reshape(1, len(time_unix), 1, 1)
time_da_broadcasted = da.broadcast_to(
    time_da, (1, len(time_unix), sample_da.shape[0], sample_da.shape[1])
)
dask_variable_list.append(time_da_broadcasted)
start_time = time.time()

for var in RDAQA_VARS:
    if var in downloaded_datasets:
        var_data = da.from_array(
            downloaded_datasets[var].values, chunks=(process_chunk, process_chunk)
        )
        var_data = var_data.reshape(1, 1, sample_da.shape[0], sample_da.shape[1])
    else:
        var_data = da.full(
            (1, 1, sample_da.shape[0], sample_da.shape[1]), np.nan, dtype=np.float32
        )
    dask_variable_list.append(var_data)

final_stacked_dask = da.concatenate(dask_variable_list, axis=0)

logger.info("Writing normalized µg/m³ arrays directly to Zarr...")
with ProgressBar():
    final_stacked_dask.rechunk(
        (len(zarr_output_vars), len(time_unix), final_chunk, final_chunk)
    ).to_zarr(zarr_array, overwrite=True, compute=True)

close_store(zarr_store)

################################################################################################
# %% Historic data

# Loop through previous hours to download and cache historical single-hour datasets
historic_zarr_paths = []
local_temp_historic_dir = os.path.join(forecast_process_dir, "Historic_Downloads")

for i in range(his_period, 0, -1):
    hist_time = base_time - timedelta(hours=i)
    timestamp_str = hist_time.strftime("%Y%m%dT%H%M%SZ")

    if save_type == "S3":
        s3_path = f"{historic_path}/RDAQA_Hist_{timestamp_str}.zarr.zip"
        s3_done_path = s3_path.replace(".zarr.zip", ".done")
        if s3.exists(s3_done_path):
            logger.info(
                f"File already exists in S3, skipping download for: {timestamp_str}"
            )
            extracted_path = download_extract_historic_zip(
                s3, s3_path, local_temp_historic_dir
            )
            if extracted_path is not None:
                historic_zarr_paths.append(extracted_path)
            continue
    else:
        local_hist_dir = historic_path
        os.makedirs(local_hist_dir, exist_ok=True)
        local_path = os.path.join(local_hist_dir, f"RDAQA_Hist_{timestamp_str}.zarr")
        if os.path.exists(local_path + ".done"):
            logger.info(
                f"File already exists locally, skipping download for: {timestamp_str}"
            )
            if os.path.exists(local_path):
                historic_zarr_paths.append(local_path)
            else:
                logger.warning(
                    "Historic done marker exists, but zarr is missing: %s", local_path
                )
            continue

    logger.info(f"Processing historical data for timestamp: {timestamp_str}")

    hist_datasets = {}
    for var in RDAQA_VARS:
        url = build_rdaqa_url(hist_time, var)
        local_grib = os.path.join(tmp_dir, f"hist_{timestamp_str}_{var}.grib2")

        if download_rdaqa_file(url, local_grib):
            try:
                ds = xr.open_dataset(local_grib, engine="cfgrib", decode_times=False)
                grib_var_name = list(ds.data_vars)[0]
                da_var = ds[grib_var_name].astype(np.float32)

                # Maintain unit conversions consistent with live forecasting
                da_converted = convert_to_ug_m3(da_var, var)
                hist_datasets[var] = da_converted

                # Cleanup immediate raw files
                os.remove(local_grib)
            except Exception as e:
                logger.error(
                    f"Error reading historical variable {var} for {timestamp_str}: {e}"
                )

    if not hist_datasets:
        logger.warning(
            f"No valid variables parsed for historical hour {timestamp_str}. Skipping."
        )
        continue

    # Create temporary single-hour Zarr file structure
    hist_zarr_path = os.path.join(
        forecast_process_dir, f"RDAQA_Hist_{timestamp_str}_TMP.zarr"
    )
    hist_store = zarr.storage.LocalStore(hist_zarr_path)

    hist_zarr_array = zarr.create_array(
        store=hist_store,
        shape=(
            len(zarr_output_vars),
            1,
            sample_da.shape[0],
            sample_da.shape[1],
        ),
        chunks=(1, 1, final_chunk, final_chunk),
        dtype=np.float32,
        overwrite=True,
    )

    hist_dask_list = []
    hist_time_unix = np.array([int(hist_time.timestamp())], dtype=np.int64)
    h_time_da = da.from_array(hist_time_unix, chunks=(1,)).reshape(1, 1, 1, 1)
    h_time_broadcasted = da.broadcast_to(
        h_time_da, (1, 1, sample_da.shape[0], sample_da.shape[1])
    )
    hist_dask_list.append(h_time_broadcasted)

    for var in RDAQA_VARS:
        if var in hist_datasets:
            v_data = da.from_array(
                hist_datasets[var].values, chunks=(process_chunk, process_chunk)
            )
            v_data = v_data.reshape(1, 1, sample_da.shape[0], sample_da.shape[1])
        else:
            v_data = da.full(
                (1, 1, sample_da.shape[0], sample_da.shape[1]), np.nan, dtype=np.float32
            )
        hist_dask_list.append(v_data)

    hist_stacked_dask = da.concatenate(hist_dask_list, axis=0)

    # Write to cached historical storage
    hist_stacked_dask.rechunk(
        (len(zarr_output_vars), 1, final_chunk, final_chunk)
    ).to_zarr(hist_zarr_array, overwrite=True, compute=True)

    close_store(hist_store)

    # Finalize target deployment and create tracking markers (.done)
    if save_type == "S3":
        zip_export_path = hist_zarr_path.replace("_TMP.zarr", "")
        shutil.make_archive(zip_export_path, "zip", hist_zarr_path)
        s3.put_file(zip_export_path + ".zip", s3_path)
        historic_zarr_paths.append(hist_zarr_path)

        # Write S3 signaling .done file
        tmp_done = os.path.join(tmp_dir, f"{timestamp_str}.done")
        with open(tmp_done, "w") as f:
            f.write("Done")
        s3.put_file(tmp_done, s3_done_path)
        os.remove(tmp_done)
    else:
        final_local_path = os.path.join(
            historic_path, f"RDAQA_Hist_{timestamp_str}.zarr"
        )
        shutil.copytree(hist_zarr_path, final_local_path, dirs_exist_ok=True)
        with open(final_local_path + ".done", "w") as f:
            f.write("Done")
        historic_zarr_paths.append(final_local_path)

        # Clean up processing directory for the hour
        shutil.rmtree(hist_zarr_path)

# %% Merge historic and forecast datasets into final stacked zarr
logger.info("Merging historic and current RDAQA datasets before production save.")

dask_arrays = []
for historic_zarr_path in historic_zarr_paths:
    try:
        dask_arrays.append(da.from_zarr(historic_zarr_path, inline_array=True))
    except (FileNotFoundError, KeyError):
        logger.info("Missing historic zarr: %s", historic_zarr_path)

dask_arrays.append(da.from_zarr(zarr_store_path, inline_array=True))
merged_arrays = da.concatenate(dask_arrays, axis=1).astype("float32")
merged_arrays_masked = mask_invalid_data(merged_arrays)

# Write out to disk. This intermediate step avoids memory overflow.
merged_arrays_masked.to_zarr(
    forecast_process_path + "_stack.zarr",
    overwrite=True,
    compute=True,
)

stacked_array_disk = da.from_zarr(forecast_process_path + "_stack.zarr")
stacked_array_padded = pad_to_chunk_size(stacked_array_disk, final_chunk)

if save_type == "S3":
    final_zarr_path = os.path.join(forecast_process_dir, "RDAQA.zarr.zip")
    zarr_store = zarr.storage.ZipStore(final_zarr_path, mode="a", compression=0)
else:
    final_zarr_path = os.path.join(forecast_process_dir, "RDAQA.zarr")
    zarr_store = zarr.storage.LocalStore(final_zarr_path)

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=stacked_array_padded.shape,
    chunks=(
        len(zarr_output_vars),
        stacked_array_padded.shape[1],
        final_chunk,
        final_chunk,
    ),
    dtype=np.float32,
    overwrite=True,
)

with ProgressBar():
    stacked_array_padded.round(5).rechunk(
        (
            len(zarr_output_vars),
            stacked_array_padded.shape[1],
            final_chunk,
            final_chunk,
        )
    ).to_zarr(zarr_array, overwrite=True, compute=True)

close_store(zarr_store)

pickle_file_path = os.path.join(tmp_dir, "RDAQA.time.pickle")
with open(pickle_file_path, "wb") as f:
    pickle.dump(base_time, f)

if save_type == "S3":
    s3.put_file(
        final_zarr_path,
        os.path.join(forecast_path, ingest_version, "RDAQA.zarr.zip"),
    )
    s3.put_file(
        pickle_file_path,
        os.path.join(forecast_path, ingest_version, "RDAQA.time.pickle"),
    )
    logger.info("S3 target deployment complete.")
else:
    shutil.move(
        pickle_file_path,
        os.path.join(forecast_path, ingest_version, "RDAQA.time.pickle"),
    )
    shutil.copytree(
        final_zarr_path,
        os.path.join(forecast_path, ingest_version, "RDAQA.zarr"),
        dirs_exist_ok=True,
    )
    logger.info("Saved RDAQA datasets to local environment.")

end_time = time.time()
logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
logger.info("RDAQA ingest script finished successfully.")

try:
    shutil.rmtree(forecast_process_dir)
except Exception:
    pass
