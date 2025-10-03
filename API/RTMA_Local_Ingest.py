# %% RTMA Rapid Update Processing script using Dask, FastHerbie, and MetPy
# Alexander Rey, September 2025

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings

# Define ECCODES_DEFINITION_PATH env variable for eccodes
# This is needed in my testing instance- should not be required for the docker image
os.environ["ECCODES_DEFINITION_PATH"] = "/home/ubuntu/eccodes-2.40.0-Source/definitions/"


import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from herbie import FastHerbie, Path, Herbie
from herbie.fast import Herbie_latest
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import CHUNK_SIZES, FINAL_CHUNK_SIZES, mask_invalid_data

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR
forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/RTMA_RU"
)
forecast_process_path = forecast_process_dir + "/RTMA_RU_Process"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RTMA_RU")

# Define the processing and final chunk size
process_chunk = CHUNK_SIZES["RTMA"]
final_chunk = FINAL_CHUNK_SIZES["RTMA"]

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)

if save_type == "Download":
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)


# %% Define base time from the most recent run
t0 = time.time()

latest_run = Herbie_latest(
    model="rtma_ru",
    n=5,
    freq="15min",
    product="anl",
    verbose=True,
    priority="aws",
    save_dir=tmpDIR,
)

base_time = latest_run.date
logging.info(f"Checking for new RTMA_RU data for base time: {base_time}")

# Check if this is newer than the current file
if save_type == "S3":
    if s3.exists(forecast_path + "/" + ingest_version + "/RTMA_RU.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/RTMA_RU.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= base_time:
            logging.info("No Update to RTMA_RU, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/RTMA_RU.time.pickle"):
        with open(
            forecast_path + "/" + ingest_version + "/RTMA_RU.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= base_time:
            logging.info("No Update to RTMA_RU, ending")
            sys.exit()


zarr_vars = (
    "time",
    "VIS_surface",
    "GUST_10maboveground",
    "PRES_station",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "TCDC_entireatmosphere",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
)

# %% Download RTMA analysis data using Herbie Latest
match_strings = (
    ":((DPT|TMP|SPFH):2 m above ground:)"
    "|:(GUST:10 m above ground:)"
    "|:(UGRD:10 m above ground:)"
    "|:(VGRD:10 m above ground:)"
    "|:((VIS|PRES):surface:)"
    "|:TCDC:entire atmosphere"
)

fh_analysis = Herbie(
    base_time,
    model="rtma_ru",
    product="anl",
    verbose=False,
    priority="aws",
    save_dir=tmpDIR,
)

fh_analysis.download(match_strings, verbose=False)

logging.info("RTMA_RU GRIB file downloaded successfully.")

xarray_herbie_list = fh_analysis.xarray(match_strings)

# Merge the three datasets into one
xarray_analysis_merged = xr.merge(xarray_herbie_list, compat="override")

# Convert RH from specific humidity and pressure and add it to the dataset
rh_2m = relative_humidity_from_specific_humidity(
    pressure = xarray_analysis_merged["sp"] * units.Pa,
    temperature = xarray_analysis_merged["t2m"]  * units.degK,
    specific_humidity = xarray_analysis_merged["sh2"],
)

xarray_analysis_merged["rh"] = rh_2m.metpy.dequantify()


# Rename variables
rename_dict = {
    "t2m": "TMP_2maboveground",
    "d2m": "DPT_2maboveground",
    "rh": "RH_2maboveground",
    "u10": "UGRD_10maboveground",
    "v10": "VGRD_10maboveground",
    "i10fg": "GUST_10maboveground",
    "vis": "VIS_surface",
    "sp": "PRES_station",
    "tcc": "TCDC_entireatmosphere",
}

rename_dict = {
    k: v for k, v in rename_dict.items() if k in xarray_analysis_merged.data_vars
}
xarray_analysis_merged = xarray_analysis_merged.rename(rename_dict)
logging.info("Variables renamed for consistency.")

# Drop time as a coordinate
model_UNIX_time = xarray_analysis_merged.time.data.astype("datetime64[s]").astype(int)
xarray_analysis_merged = xarray_analysis_merged.reset_coords("time", drop=True)

# Add a new data variables for time
# Same X Y shape as the rest, identical values
# UNIX time
xarray_analysis_merged["time"] = (
    ("y", "x"),
    np.full(
        (xarray_analysis_merged.dims["y"], xarray_analysis_merged.dims["x"]),
        model_UNIX_time,
    ),
)


# Only keep required variables
keep_vars = [v for v in zarr_vars if v in xarray_analysis_merged.data_vars]
xarray_analysis_merged = xarray_analysis_merged[keep_vars]

# Merge the arrays into a single 3D array
xarray_analysis_stack = xarray_analysis_merged.to_stacked_array(new_dim='var',
                                                                    sample_dims=['y', 'x']).chunk(
    chunks={"var": -1, "x": final_chunk, "y": final_chunk}
).transpose("var", "y", "x")

# Convert from xarray to the raw dask array
dask_var_array = xarray_analysis_stack.data

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/RTMA_RU3.zarr.zip", mode="w", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/RTMA_RU.zarr")

# Create zarr array
zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarr_vars),
        dask_var_array.shape[1],
        dask_var_array.shape[2],
    ),
    chunks=(len(zarr_vars), final_chunk, final_chunk),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)

dask_var_array.to_zarr(
    zarr_array, overwrite=True, compute=True)

if save_type == "S3":
    zarr_store.close()
    logging.info("Zarr zip store closed.")


# %% Upload to S3 or move to final location

# Save to Production Path (Existing Logic)
if save_type == "S3":
    s3.put_file(
        forecast_process_dir + "/RTMA_RU.zarr.zip",
        forecast_path + "/" + ingest_version + "/RTMA_RU.zarr.zip",
    )
    logging.info("Final Zarr zip file uploaded to S3.")

    with open(forecast_process_dir + "/RTMA_RU.time.pickle", "wb") as file:
        pickle.dump(base_time, file)
    s3.put_file(
        forecast_process_dir + "/RTMA_RU.time.pickle",
        forecast_path + "/" + ingest_version + "/RTMA_RU.time.pickle",
    )
    logging.info("Time pickle file uploaded to S3.")

else:
    with open(forecast_process_dir + "/RTMA_RU.time.pickle", "wb") as file:
        pickle.dump(base_time, file)
    shutil.move(
        forecast_process_dir + "/RTMA_RU.time.pickle",
        forecast_path + "/" + ingest_version + "/RTMA_RU.time.pickle",
    )
    shutil.copytree(
        forecast_process_dir + "/RTMA_RU.zarr",
        forecast_path + "/" + ingest_version + "/RTMA_RU.zarr",
        dirs_exist_ok=True,
    )
    logging.info("Final Zarr and time pickle files moved to local storage.")


# Clean up
shutil.rmtree(forecast_process_dir)
logging.info("Cleaning up temporary processing directories.")

# Test Read
t1 = time.time()
logging.info(f"Total script execution time: {t1 - t0:.2f} seconds.")
