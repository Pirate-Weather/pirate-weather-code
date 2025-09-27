# %% RTMA Processing script using Dask, FastHerbie, and MetPy
# Alexander Rey, September 2025

# %% Import modules
import os
import pickle
import shutil
import sys
import time
import warnings
import zarr
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest
from API.ingest_utils import mask_invalid_data, CHUNK_SIZES, FINAL_CHUNK_SIZES
from metpy.calc import relative_humidity_from_dewpoint
import logging

from API.constants.shared_const import INGEST_VERSION_STR

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR
analysis_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/RTMA"
)
analysis_process_path = analysis_process_dir + "/RTMA_Process"
tmp_dir = analysis_process_dir + "/Downloads"
analysis_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/RTMA")
historic_base_path = os.getenv(
    "historic_base_path", default="/home/ubuntu/Weather/Hist/RTMA"
)

# Define the processing and final chunk size
process_chunk = CHUNK_SIZES["RTMA"]
final_chunk = FINAL_CHUNK_SIZES["RTMA"]

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Create new directory for processing if it does not exist
if not os.path.exists(analysis_process_dir):
    os.makedirs(analysis_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(analysis_process_dir)
    os.makedirs(analysis_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download":
    if not os.path.exists(analysis_path + "/" + ingest_version):
        os.makedirs(analysis_path + "/" + ingest_version)


# %% Define base time from the most recent run
t0 = time.time()

latest_run = Herbie_latest(
    model="rtma",
    n=1,
    freq="1h",
    fxx=[0],
    product="anl",
    verbose=False,
    priority="aws",
    save_dir=tmp_dir,
)

base_time = latest_run.date
logging.info(f"Checking for new RTMA data for base time: {base_time}")

# Check if this is newer than the current file
if save_type == "S3":
    if s3.exists(analysis_path + "/" + ingest_version + "/RTMA.time.pickle"):
        with s3.open(
            analysis_path + "/" + ingest_version + "/RTMA.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= base_time:
            logging.info("No Update to RTMA, ending")
            sys.exit()

else:
    if os.path.exists(analysis_path + "/" + ingest_version + "/RTMA.time.pickle"):
        with open(
            analysis_path + "/" + ingest_version + "/RTMA.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= base_time:
            logging.info("No Update to RTMA, ending")
            sys.exit()


zarr_vars = (
    "time",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "GUST_10maboveground",
    "VIS_surface",
    "TCCC_entireatmosphere",
    "PRES_station",
)


# %% Download RTMA analysis data using Herbie Latest
match_strings = (
    ":((DPT|TMP):2 m above ground:)"
    "|:(GUST:10 m above ground:)"
    "|:(UGRD:10 m above ground:)"
    "|:(VGRD:10 m above ground:)"
    "|:((VIS|PRES):surface:)"
    "|:TCC:entire atmosphere:"
)

fh_analysis = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="1h"),
    model="rtma",
    fxx=[0],
    product="anl",
    verbose=False,
    priority="aws",
    save_dir=tmp_dir,
)

fh_analysis.download(match_strings, verbose=False)

if len(fh_analysis.file_exists) != 1:
    logging.error(
        f"Download failed, expected 1 file but got {len(fh_analysis.file_exists)}"
    )
    sys.exit(1)
logging.info("RTMA GRIB file downloaded successfully.")

grib_list = [
    str(Path(x.get_localFilePath(match_strings)).expand())
    for x in fh_analysis.file_exists
]

# Create XArray
xarray_analysis_merged = xr.open_mfdataset(grib_list, engine="cfgrib")
logging.info("RTMA dataset loaded with xarray.")

# Convert RH from temperature & dewpoint and add it to the dataset
try:
    t = xarray_analysis_merged["t2m"].metpy.quantify()
    td = xarray_analysis_merged["d2m"].metpy.quantify()
    rh = relative_humidity_from_dewpoint(t, td)
    xarray_analysis_merged["rh"] = rh
    xarray_analysis_merged["rh"].attrs.update(units="%", long_name="Relative Humidity")
    logging.info("Relative Humidity calculated and added to dataset.")
except Exception as e:
    logging.warning(f"Failed to calculate RH: {e}")
    xarray_analysis_merged["rh"] = xr.full_like(xarray_analysis_merged["t2m"], np.nan)
    xarray_analysis_merged["rh"].attrs.update(
        units="%", long_name="Relative Humidity (Failed)"
    )


# Rename variables
rename_dict = {
    "t2m": "TMP_2maboveground",
    "d2m": "DPT_2maboveground",
    "rh": "RH_2maboveground",
    "u10": "UGRD_10maboveground",
    "v10": "VGRD_10maboveground",
    "i10fg": "GUST_10maboveground",
    "vis": "VIS_surface",
    "pres": "PRES_station",
    "tcc": "TCCC_entireatmosphere",
}

rename_dict = {
    k: v for k, v in rename_dict.items() if k in xarray_analysis_merged.data_vars
}
xarray_analysis_merged = xarray_analysis_merged.rename(rename_dict)
logging.info("Variables renamed for consistency.")


# Only keep required variables
keep_vars = [v for v in zarr_vars if v in xarray_analysis_merged.data_vars]
xarray_analysis_merged = xarray_analysis_merged[keep_vars]

# Mask invalid data
xarray_analysis_merged = mask_invalid_data(xarray_analysis_merged)

# Save to Zarr
xarray_analysis_merged = xarray_analysis_merged.chunk(
    chunks={"time": 1, "x": process_chunk, "y": process_chunk}
)
xarray_analysis_merged.to_zarr(
    analysis_process_path + "_xr_merged.zarr", mode="w", consolidated=False
)
logging.info("Intermediate xarray dataset saved to Zarr.")

del xarray_analysis_merged


# %% Format as dask and save as zarr
dask_var_array_list = []
dask_var_arrays = []

logging.info("Starting Dask array creation and processing.")
for dask_var_idx, dask_var in enumerate(zarr_vars[:]):
    if dask_var in ["time", "latitude", "longitude"]:
        continue
    logging.info(f"Processing variable: {dask_var}")
    dask_analysis_array = da.from_zarr(
        analysis_process_path + "_xr_merged.zarr", component=dask_var, inline_array=True
    )
    dask_var_array_list.append(
        dask_analysis_array.rechunk(
            (dask_analysis_array.shape[0], process_chunk, process_chunk)
        ).astype("float32")
    )


# Merge the arrays into a single 4D array
dask_var_array_list_merge = da.stack(dask_var_array_list, axis=0)

# Mask out invalid data
dask_var_array_list_merge_nan = mask_invalid_data(dask_var_array_list_merge)
logging.info("Stacked Dask array created and invalid data masked.")

# Write out to disk
dask_var_array_list_merge_nan.to_zarr(
    analysis_process_path + "_stack.zarr", overwrite=True, compute=True
)
logging.info("Intermediate stacked Zarr file created.")

# Read in stacked 4D array back in
dask_var_array_stack_disk = da.from_zarr(analysis_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        analysis_process_dir + "/RTMA.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(analysis_process_dir + "/RTMA.zarr")

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=dask_var_array_stack_disk.shape,
    chunks=(
        dask_var_array_stack_disk.shape[0],
        dask_var_array_stack_disk.shape[1],
        final_chunk,
        final_chunk,
    ),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)

logging.info("Writing final Zarr array to disk.")
da.rechunk(
    dask_var_array_stack_disk.round(3),
    (
        dask_var_array_stack_disk.shape[0],
        dask_var_array_stack_disk.shape[1],
        final_chunk,
        final_chunk,
    ),
).to_zarr(zarr_array, compute=True)

if save_type == "S3":
    zarr_store.close()
    logging.info("Zarr zip store closed.")


# %% Upload to S3 or move to final location
# Define Historical Save Path and Ensure Directory Exists
time_str = base_time.strftime("%Y%m%d_%H%M")
historic_save_path = f"{historic_base_path}/{ingest_version}/{time_str}"

if save_type == "Download":
    if not os.path.exists(historic_save_path):
        os.makedirs(historic_save_path)

# Save to Production Path (Existing Logic)
if save_type == "S3":
    s3.put_file(
        analysis_process_dir + "/RTMA.zarr.zip",
        analysis_path + "/" + ingest_version + "/RTMA.zarr.zip",
    )
    logging.info("Final Zarr zip file uploaded to S3.")

    with open(analysis_process_dir + "/RTMA.time.pickle", "wb") as file:
        pickle.dump(base_time, file)
    s3.put_file(
        analysis_process_dir + "/RTMA.time.pickle",
        analysis_path + "/" + ingest_version + "/RTMA.time.pickle",
    )
    logging.info("Time pickle file uploaded to S3.")

else:
    with open(analysis_process_dir + "/RTMA.time.pickle", "wb") as file:
        pickle.dump(base_time, file)
    shutil.move(
        analysis_process_dir + "/RTMA.time.pickle",
        analysis_path + "/" + ingest_version + "/RTMA.time.pickle",
    )
    shutil.copytree(
        analysis_process_dir + "/RTMA.zarr",
        analysis_path + "/" + ingest_version + "/RTMA.zarr",
        dirs_exist_ok=True,
    )
    logging.info("Final Zarr and time pickle files moved to local storage.")

# Save to Historical Path
logging.info(f"Saving to Historical Archive: {historic_save_path}")

if save_type == "S3":
    # Upload Zarr
    s3.put_file(
        analysis_process_dir + "/RTMA.zarr.zip",
        historic_save_path + "/RTMA.zarr.zip",
    )
    logging.info("Historical Zarr zip file uploaded to S3.")

    # Upload Time Pickle
    # The pickle file is needed in the historic path for verification
    s3.put_file(
        analysis_process_dir + "/RTMA.time.pickle",
        historic_save_path + "/RTMA.time.pickle",
    )
    logging.info("Historical Time pickle file uploaded to S3.")

else:
    # Copy Time Pickle
    # Note: Use copy since the production move already happened above.
    shutil.copy(
        analysis_process_dir + "/RTMA.time.pickle",
        historic_save_path + "/RTMA.time.pickle",
    )
    logging.info("Historical Time pickle file copied to local storage.")

    # Copy Zarr
    shutil.copytree(
        analysis_process_dir + "/RTMA.zarr",
        historic_save_path + "/RTMA.zarr",
        dirs_exist_ok=True,
    )
    logging.info("Historical Zarr file copied to local storage.")


# Clean up
shutil.rmtree(analysis_process_dir)
logging.info("Cleaning up temporary processing directories.")

# Test Read
t1 = time.time()
logging.info(f"Total script execution time: {t1 - t0:.2f} seconds.")
