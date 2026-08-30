# %% URMA Processing script using Dask, FastHerbie, and MetPy
# Alexander Rey, August 2026

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings

import dask as da

# Define ECCODES_DEFINITION_PATH env variable for eccodes
# This is needed in my testing instance- should not be required for the docker image
# os.environ["ECCODES_DEFINITION_PATH"] = (
#    "/home/ubuntu/eccodes-2.40.0-Source/definitions/"
# )
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from herbie import Herbie
from herbie.fast import Herbie_latest
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    archive_tmp_zarr_and_upload,
    close_store,
    configure_zarr_limits,
    download_extract_historic_archive,
    earth_relative_wind_components,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

historic_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/URMA")
tmp_dir = historic_process_dir + "/Downloads"

historic_path = os.getenv("historic_path", default="/mnt/nvme/data/Prod/URMA")
hist_process_path = historic_path + "/URMA_Historic"

# Define the processing and final chunk size
process_chunk = CHUNK_SIZES["URMA"]
final_chunk = FINAL_CHUNK_SIZES["URMA"]
his_period = HISTORY_PERIODS["URMA"]

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)

# Create new directory for processing if it does not exist
if not os.path.exists(historic_process_dir):
    os.makedirs(historic_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(historic_process_dir)
    os.makedirs(historic_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download" and not os.path.exists(historic_path + "/" + ingest_version):
    os.makedirs(historic_path + "/" + ingest_version)

herbie_save_dir = make_herbie_save_dir(tmp_dir)

# %% Define base time from the most recent run
t0 = time.time()

latest_run = Herbie_latest(
    model="urma",
    n=5,
    product="anl",
    verbose=True,
    priority=["aws", "nomdas"],
    save_dir=herbie_save_dir,
)

base_time = latest_run.date
logger.info(f"Checking for new URMA data for base time: {base_time}")

# Check if this is newer than the current file
if save_type == "S3":
    if s3.exists(historic_path + "/" + ingest_version + "/URMA.time.pickle"):
        with s3.open(
            historic_path + "/" + ingest_version + "/URMA.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= base_time:
            logger.info("No Update to URMA, ending")
            sys.exit()

else:
    if os.path.exists(historic_path + "/" + ingest_version + "/URMA.time.pickle"):
        with open(
            historic_path + "/" + ingest_version + "/URMA.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= base_time:
            logger.info("No Update to URMA, ending")
            sys.exit()


zarr_vars = (
    "time",
    "vis",
    "i10fg",
    "sp",
    "t2m",
    "d2m",
    "rh",
    "tcc",
    "u10",
    "v10",
)

# %% Download URMA analysis data using Herbie Latest
match_strings = (
    ":((DPT|TMP|SPFH):2 m above ground:)"
    "|:(GUST:10 m above ground:)"
    "|:(UGRD:10 m above ground:)"
    "|:(VGRD:10 m above ground:)"
    "|:((VIS|PRES):surface:)"
    "|:TCDC:entire atmosphere"
)
# Historical loop to download the last his_period hours of URMA data, starting from base_time
for i in range(his_period, -1, -1):
    timestamp_str = (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")

    if save_type == "S3":
        s3_path = f"{historic_path}/URMA_Hist_v1_{timestamp_str}.zarr.tar.gz"
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            logger.info("File already exists in S3, skipping: %s", s3_path)
            continue
    else:
        local_path = f"{historic_path}/URMA_Hist_v1_{timestamp_str}.zarr"
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info("File already exists locally, skipping: %s", local_path)
            continue

    logger.info("Downloading URMA historic timestep: %s", timestamp_str)

    # Directly target the historic hour without the -1 hour shift
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(hours=i),
        periods=1,
        freq="1h",
    )

    fh_analysis = Herbie(
        DATES[0],
        model="urma",
        product="anl",
        verbose=False,
        priority=["aws", "nomdas"],
        save_dir=herbie_save_dir,
    )

    fh_analysis.download(match_strings, verbose=True)

    logger.info("URMA GRIB file downloaded successfully.")

    xarray_herbie_list = fh_analysis.xarray(match_strings)

    # Merge the three datasets into one
    xarray_analysis_merged = xr.merge(xarray_herbie_list, compat="override")

    # Assign coordinates from one of the datasets to the merged dataset
    xarray_analysis_merged = xarray_analysis_merged.assign_coords(
        xarray_herbie_list[0].metpy.parse_cf().coords
    )

    # Convert RH from specific humidity and pressure and add it to the dataset
    # relative_humidity_from_specific_humidity returns a dimensionless fraction (0-1)
    rh_2m = relative_humidity_from_specific_humidity(
        pressure=xarray_analysis_merged["sp"] * units.Pa,
        temperature=xarray_analysis_merged["t2m"] * units.degK,
        specific_humidity=xarray_analysis_merged["sh2"] * units("kg/kg"),
    )

    xarray_analysis_merged["rh"] = rh_2m.metpy.dequantify()

    # Convert winds from grid relative to earth relative
    u_earth, v_earth = earth_relative_wind_components(
        xarray_analysis_merged["u10"], xarray_analysis_merged["v10"]
    )

    # Put U and V back into the dataset, replacing the grid relative versions
    xarray_analysis_merged["u10"].data = u_earth
    xarray_analysis_merged["v10"].data = v_earth

    # Drop time as a coordinate
    model_UNIX_time = xarray_analysis_merged.time.data.astype("datetime64[s]").astype(
        int
    )
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

    # Clip to valid data ranges
    for var in zarr_vars:
        if var == "time":
            continue
        elif var in xarray_analysis_merged.data_vars:
            ds_clip = xarray_analysis_merged[var]
            if np.issubdtype(ds_clip.dtype, np.number):
                mask = (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
                xarray_analysis_merged[var] = ds_clip.where(mask)  # out-of-range → NaN

    # Drop the sh2 variable as we no longer need it
    xarray_analysis_merged = xarray_analysis_merged.drop_vars("sh2")

    # Set the order correctly
    vars_in = [v for v in zarr_vars if v in xarray_analysis_merged.data_vars]

    # Merge the arrays into a single 3D array with the correct order, add a 1 length time dimension, and rechunk
    xarray_analysis_stack = (
        xarray_analysis_merged[vars_in]
        .to_stacked_array(new_dim="var", sample_dims=["y", "x"])
        .expand_dims("time", axis=1)
        .chunk(chunks={"var": -1, "time": 1, "x": final_chunk, "y": final_chunk})
        .transpose("var", "time", "y", "x")
    )

    # Mask out invalid data
    dask_var_array = mask_invalid_data(xarray_analysis_stack)

    # Add padding to the zarr store
    dask_var_array = pad_to_chunk_size(dask_var_array, final_chunk)

    # Create a zarr backed dask array
    if save_type == "S3":
        zarr_store = zarr.storage.ZipStore(
            historic_path + "/URMA_Hist_TMP.zarr.zip", mode="a", compression=0
        )
    else:
        zarr_store = zarr.storage.LocalStore(historic_path + "/URMA_Hist_TMP.zarr")

    # Create zarr array
    zarr_array = zarr.create_array(
        store=zarr_store,
        shape=(
            len(zarr_vars),
            1,
            dask_var_array.shape[2],
            dask_var_array.shape[3],
        ),
        chunks=(len(zarr_vars), 1, final_chunk, final_chunk),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
    )

    with da.config.set(scheduler="threads", num_workers=zarr_store_workers):
        dask_var_array.to_zarr(zarr_array, overwrite=True, compute=True)

    close_store(zarr_store)
    if save_type == "S3":
        logger.info("Zarr zip store closed.")
        archive_tmp_zarr_and_upload(
            tmp_zarr_path=historic_path + "/URMA_Hist_TMP.zarr.zip",
            s3_path=s3_path,
            archive_member_name="URMA_Hist.zarr",
            s3=s3,
        )
    else:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.rename(historic_path + "/URMA_Hist_TMP.zarr", local_path)
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

# %% Upload to S3 or move to final location

# Save to Historic Path
if save_type == "S3":
    local_temp_dir = hist_process_path + "_s3_temp_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)
    ncHistWorking_paths = []
    for i in range(his_period, -1, -1):
        timestamp = (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        final_zarr_name = f"URMA_Hist_{timestamp}.zarr"
        extracted_path = download_extract_historic_archive(
            s3=s3,
            historic_path=historic_path,
            final_zarr_name=final_zarr_name,
            extracted_store_name="URMA.zarr",
            local_temp_dir=local_temp_dir,
            expected_vars=zarr_vars,
        )
        if extracted_path is not None:
            ncHistWorking_paths.append(extracted_path)
else:
    ncHistWorking_paths = [
        historic_path
        + "/URMA_Hist_"
        + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        + ".zarr"
        for i in range(his_period, -1, -1)
    ]

dask_var_array_list = []

for dask_var in zarr_vars:
    daskVarArrays = [
        da.from_zarr(local_path, component=dask_var, inline_array=True)
        for local_path in ncHistWorking_paths
    ]

    # Stack along time dimension (axis 0)
    dask_var_arrays_stack = da.stack(daskVarArrays, axis=0)

    if dask_var == "time":
        # Compute exact time matrix dimensions
        np_cat_times = dask_var_arrays_stack[:, 0, 0].compute()
        ny, nx = daskVarArrays[0].shape[1], daskVarArrays[0].shape[2]

        daskArrayOut = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(np_cat_times, axis=1), axis=1),
                (1, ny, nx),
            )
        ).rechunk((len(np_cat_times), process_chunk, process_chunk))

        dask_var_array_list.append(daskArrayOut)
    else:
        daskArrayOut = (
            dask_var_arrays_stack.squeeze(axis=1)
            .rechunk((len(ncHistWorking_paths), process_chunk, process_chunk))
            .astype("float32")
        )

        dask_var_array_list.append(daskArrayOut)

# Merge variables into a single 4D array (var, time, y, x)
dask_var_array_list_merge = da.stack(dask_var_array_list, axis=0)

# Clean up
shutil.rmtree(historic_process_dir)
logger.info("Cleaning up temporary processing directories.")

# Test Read
t1 = time.time()
logger.info(f"Total script execution time: {t1 - t0:.2f} seconds.")
