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
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, urllib

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import dask.array as da
import zarr.storage
from dask.diagnostics import ProgressBar

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import CHUNK_SIZES, FINAL_CHUNK_SIZES, close_store

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Constants
RDAQA_BASE_URL = "https://dd.weather.gc.ca/today/model_rdaqa/10km"
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
            req = urllib.request.Request(url, method="HEAD")
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
    return f"{RDAQA_BASE_URL}/{hh}/{filename}"


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
        logger.info(f"Converting O3 from ppb to µg/m³")
        return da * O3_PPB_TO_UG_M3
    elif var_name == "NO2":
        logger.info(f"Converting NO2 from ppb to µg/m³")
        return da * NO2_PPB_TO_UG_M3
    elif var_name == "SO2":
        logger.info(f"Converting SO2 from ppb to µg/m³")
        return da * SO2_PPB_TO_UG_M3
    return da


ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/RDAQA")
forecast_process_path = os.path.join(forecast_process_dir, "RDAQA_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")
forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RDAQA")
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
            ds = xr.open_dataset(local_grib, engine="cfgrib")
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
time_unix = np.array([int(pd.Timestamp(base_time).timestamp())], dtype=np.int64)

zarr_store_path = forecast_process_path + ".zarr"
zarr_store = zarr.storage.DirectoryStore(zarr_store_path)

process_chunk = CHUNK_SIZES.get("RDAQA", 250)
final_chunk = FINAL_CHUNK_SIZES.get("RDAQA", 500)

zarr_output_vars = ["time"] + RDAQA_VARS

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarr_output_vars),
        len(time_unix),
        sample_da.shape[0],
        sample_da.shape[1],
    ),
    chunks=(1, 1, final_chunk, final_chunk),
    dtype=np.float32,
    overwrite=True,
)

dask_variable_list = []

time_da = da.from_array(time_unix, chunks=(1,)).reshape(1, len(time_unix), 1, 1)
time_da_broadcasted = da.broadcast_to(
    time_da, (1, len(time_unix), sample_da.shape[0], sample_da.shape[1])
)
dask_variable_list.append(time_da_broadcasted)

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
        (len(zarr_output_vars), 1, final_chunk, final_chunk)
    ).to_zarr(zarr_array, overwrite=True, compute=True)

close_store(zarr_store)

pickle_file_path = os.path.join(tmp_dir, "RDAQA.time.pickle")
with open(pickle_file_path, "wb") as f:
    pickle.dump(base_time, f)

if save_type == "S3":
    shutil.make_archive(forecast_process_path, "zip", zarr_store_path)
    s3.put_file(
        forecast_process_path + ".zip",
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
        zarr_store_path,
        os.path.join(forecast_path, ingest_version, "RDAQA.zarr"),
        dirs_exist_ok=True,
    )
    logger.info(f"Saved RDAQA datasets to local environment.")

try:
    shutil.rmtree(forecast_process_dir)
except Exception:
    pass
