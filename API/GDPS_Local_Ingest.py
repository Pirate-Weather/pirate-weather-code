# %% Script to test FastHerbie.py to download GDPS data
# Alexander Rey, April 2026

# %% Import modules
import os
import pickle
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import FastHerbie, HerbieLatest
from tqdm import tqdm

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR, MISSING_DATA
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    archive_tmp_zarr_and_upload,
    build_herbie_grib_list,
    close_store,
    configure_zarr_limits,
    download_extract_historic_archive,
    interp_time_take_blend,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    run_command,
    tune_nofile_limit,
    validate_grib_stats,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GDPS")
forecast_process_path = forecast_process_dir + "/GDPS_Process"
hist_process_path = forecast_process_dir + "/GDPS_Historic"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GDPS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/GDPS")


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


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["GDPS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GDPS"]

his_period = HISTORY_PERIODS["GDPS"]

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download":
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

herbie_save_dir = make_herbie_save_dir(tmp_dir)


T0 = time.time()

latest_run = HerbieLatest(
    model="gdps",
    n=3,
    freq="12h",
    fxx=240,
    product="15km",
    verbose=False,
    save_dir=herbie_save_dir,
)

base_time = latest_run.date


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/GDPS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/GDPS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GDPS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/GDPS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/GDPS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GDPS, ending")
            sys.exit()

zarr_vars = (
    "time",
    "WindGust_AGL_10m",
    "Pressure_MSL",
    "AirTemp_AGL_2m",
    "DewPoint_AGL_2m",
    "RelativeHumidity_AGL_2m",
    "WindU_AGL_10m",
    "WindV_AGL_10m",
    "PrecipRate_Sfc",
    "Precip_Accum_Sfc",
    "PrecipType_Instant_Sfc",
    "TotalCloudCover_Sfc",
    "UVIndex_Sfc",
    "DownwardShortwaveRadiationFlux_Accum_Sfc",
    "CAPE_Sfc",
    "Pressure_Sfc",
    "CIN_Sfc",
    "VerticalVelocity_IsbL_0500",
    "KIndex_Sfc",
)

#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours

# Define the variables to download as a dictionary of variable and level pairs to match in the grib files
match_strings = [
    {"variable": "AirTemp", "level": "AGL-2m"},
    {"variable": "DewPoint", "level": "AGL-2m"},
    {"variable": "RelativeHumidity", "level": "AGL-2m"},
    {"variable": "WindU", "level": "AGL-10m"},
    {"variable": "WindV", "level": "AGL-10m"},
    {"variable": "WindGust", "level": "AGL-10m"},
    {"variable": "PrecipRate", "level": "Sfc"},
    {"variable": "Precip-Accum", "level": "Sfc"},
    {"variable": "PrecipType-Instant", "level": "Sfc"},
    {"variable": "UVIndex", "level": "Sfc"},
    {"variable": "DownwardShortwaveRadiationFlux-Accum", "level": "Sfc"},
    {"variable": "CAPE", "level": "Sfc"},
    {"variable": "Pressure", "level": "Sfc"},
    {"variable": "TotalCloudCover", "level": "Sfc"},
    {"variable": "CIN", "level": "Sfc"},
    {"variable": "Pressure", "level": "MSL"},
    {"variable": "VerticalVelocity", "level": "IsbL-0500"},
    {"variable": "KIndex", "level": "Sfc"},
]

gdps_file_range = FORECAST_LEAD_RANGES["GDPS"]

# Create FastHerbie object
FH_forecastsub = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="12h"),
    model="gdps",
    fxx=gdps_file_range,
    product="15km",
    verbose=False,
    save_dir=herbie_save_dir,
)

# MSC models have each variable in a separate file, so we loop through the variables and levels to download each one and then merge them later
all_files = []
for g in match_strings:
    FH = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="12h"),
        model="gdps",
        fxx=gdps_file_range,
        product="15km",
        variable=g["variable"],
        level=g["level"],
        save_dir=herbie_save_dir,
        verbose=False,
    )
    FH.download()

    # Ensure each variable produced the expected number of lead files
    if len(FH.file_exists) != len(gdps_file_range):
        print(
            f"Download failed for {g['variable']}:{g['level']}, expected "
            f"{len(gdps_file_range)} files but got {len(FH.file_exists)}"
        )
        sys.exit(1)

    all_files += FH.file_exists

# Deduplicate and sanity-check total files
all_files = sorted(set(all_files))
expected_total = len(gdps_file_range) * len(match_strings)
if len(all_files) < expected_total:
    print(
        f"Download incomplete, expected at least {expected_total} files but got {len(all_files)}"
    )
    sys.exit(1)

# Create ordered/filtered list of downloaded grib files from collected paths
grib_list = build_herbie_grib_list(all_files, match_strings)

# Perform a check if any data seems to be invalid
cmd = f"cat {' '.join(grib_list)} | {wgrib2_path.strip()} - -s -stats"

grib_check = run_command(cmd)

# Validate the grib files
validate_grib_stats(grib_check)
print("Grib validation complete, no errors found.")


# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = (
    "cat "
    + " ".join(grib_list)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_wgrib2_merged.nc"
)


# Run wgrib2
sp_out = run_command(cmd)
if sp_out.returncode != 0:
    print(sp_out.stderr)
    sys.exit()

# Read the merged netcdf file using xarray (single combined file)
xarray_forecast_merged = xr.open_dataset(forecast_process_path + "_wgrib2_merged.nc")

assert len(xarray_forecast_merged.time) == len(gdps_file_range), (
    "Incorrect number of timesteps! Exiting"
)

# Determine grid size from merged dataset (supports rotated grids)
NY = xarray_forecast_merged.dims.get(
    "latitude", xarray_forecast_merged["latitude"].size
)
NX = xarray_forecast_merged.dims.get(
    "longitude", xarray_forecast_merged["longitude"].size
)

# Create a new time series
start = xarray_forecast_merged.time.min().values  # Adjust as necessary
end = xarray_forecast_merged.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(his_period, "h"), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(his_period, "h"),
            end=start - pd.Timedelta(1, "h"),
            freq="h",
        ),
        xarray_forecast_merged.time.values,
    )
)
unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

# Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
APCP_surface_tmp = da.diff(
    xarray_forecast_merged["Precip_Accum_Sfc"],
    axis=xarray_forecast_merged["Precip_Accum_Sfc"].get_axis_num("time"),
    prepend=0,
)

# Convert 3-hourly to 1-hourly
APCP_surface_tmp = APCP_surface_tmp / 3

xarray_forecast_merged["Precip_Accum_Sfc"].data = APCP_surface_tmp

# Save the dataset with compression and filters for all variables
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 240, "latitude": process_chunk, "longitude": process_chunk}
)
with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
    xarray_forecast_merged.to_zarr(
        forecast_process_path + "_.zarr",
        mode="w",
        consolidated=False,
        compute=True,
        chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
    )

# %% Delete to free memory
del (
    APCP_surface_tmp,
    xarray_forecast_merged,
)
T1 = time.time()

print(T1 - T0)
os.remove(forecast_process_path + "_wgrib2_merged.nc")

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(his_period, 0, -6):
    if save_type == "S3":
        s3_path = (
            historic_path
            + "/GDPS_Hist_v3"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr.tar.gz"
        )

        # Check for a done file in S3
        if s3.exists(s3_path.replace(".tar.gz", ".done")):
            print("File already exists in S3, skipping download for: " + s3_path)
            continue
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/GDPS_Hist_v3"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a local done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            print("File already exists locally, skipping download for: " + local_path)
            continue

    print(
        "Downloading: " + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    )

    # Create a range of dates for historic data going back 240 hours
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="12h",
    )
    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach
    fxx = [3, 6]

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="gdps",
        fxx=fxx,
        product="15km",
        verbose=False,
        save_dir=herbie_save_dir,
    )

    # Download the subsets
    FH_histsub.download(match_strings, verbose=False)

    # Check for download length
    if len(FH_histsub.file_exists) != len(fxx):
        print(
            f"Download failed, expected {len(fxx)} files but got "
            + str(len(FH_histsub.file_exists))
        )
        sys.exit(1)

    # Create list of downloaded grib files
    grib_list = build_herbie_grib_list(FH_histsub.file_exists, match_strings)

    # Perform a check if any data seems to be invalid
    cmd = f"cat {' '.join(grib_list)} | {wgrib2_path.strip()} - -s -stats"

    grib_check = run_command(cmd)

    validate_grib_stats(grib_check)
    print("Grib files passed validation, proceeding with processing")

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + hist_process_path
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2
    sp_out = run_command(cmd)
    if sp_out.returncode != 0:
        print(sp_out.stderr)
        sys.exit()

    # Read the merged netcdf file using xarray (single combined file)
    xarray_hist_merged = xr.open_dataset(hist_process_path + "_wgrib2_merged.nc")

    # Fix things
    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time, every 6 hours
    apcpProc = xarray_hist_merged["Precip_Accum_Sfc"].values

    apcpProcHour = np.diff(apcpProc, axis=0, prepend=0)

    xarray_hist_merged["Precip_Accum_Sfc"] = xarray_hist_merged[
        "Precip_Accum_Sfc"
    ].copy(data=apcpProcHour)

    # Clear memory
    del (apcpProc, apcpProcHour)

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chunks to speed up read times
    # Small fix for PRES_station/ PRES_surface
    encoding = {
        vname: {"chunks": (6, process_chunk, process_chunk)} for vname in zarr_vars[1:]
    }

    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_hist_merged.to_zarr(
            hist_process_path + "_GDPS_Hist_TMP.zarr",
            mode="w",
            consolidated=False,
            encoding=encoding,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged.nc")

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        archive_tmp_zarr_and_upload(
            tmp_zarr_path=hist_process_path + "_GDPS_Hist_TMP.zarr",
            s3_path=s3_path,
            archive_member_name="GDPS_Hist.zarr",
            s3=s3,
        )
    else:
        # Move to Local Path
        os.rename(hist_process_path + "_GDPS_Hist_TMP.zarr", local_path)

        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    print((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))


# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
if save_type == "S3":
    local_temp_dir = forecast_process_path + "_s3_temp_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)

    # The function that downloads and extracts a single timestamp
    def download_and_extract(timestamp):
        # Names expected locally
        final_zarr_name = f"GDPS_Hist_v3{timestamp}.zarr"
        extracted_path = download_extract_historic_archive(
            s3=s3,
            historic_path=historic_path,
            final_zarr_name=final_zarr_name,
            extracted_store_name="GDPS_Hist.zarr",
            local_temp_dir=local_temp_dir,
        )
        if extracted_path is None:
            tqdm.write(
                f"Error: GDPS_Hist.zarr not found inside archive for {timestamp}"
            )
        return extracted_path

    # Generate target timestamps
    timestamps = [
        (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        for i in range(his_period, 1, -6)
    ]

    print(f"Phase 1: Downloading and extracting {len(timestamps)} archives...")

    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(
            tqdm(
                executor.map(download_and_extract, timestamps),
                total=len(timestamps),
                desc="S3 Archive Sync",
            )
        )

    # Filter out the missing files (None values) and keep the valid paths
    ncLocalWorking_paths = [path for path in results if path is not None]
else:
    ncLocalWorking_paths = [
        historic_path
        + "/GDPS_Hist_v3"
        + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
        + ".zarr"
        for i in range(his_period, 1, -6)
    ]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(zarr_vars[:]):
    for local_ncpath in ncLocalWorking_paths:
        # If not found in array, use MISSING_DATA to show missing
        try:
            daskVarArrays.append(
                da.from_zarr(local_ncpath, component=dask_var, inline_array=True)
            )
        # Add a fallback in case of a FileNotFoundError
        except FileNotFoundError:
            print("File not found, adding NaN array for: " + local_ncpath)
            daskVarArrays.append(
                da.full((len(fxx), NY, NX), MISSING_DATA).rechunk(
                    (len(fxx), process_chunk, process_chunk)
                )
            )

    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

    daskForecastArray = da.from_zarr(
        forecast_process_path + "_.zarr", component=dask_var, inline_array=True
    )

    if dask_var == "time":
        # Create a time array with the same shape
        # This is because multiple steps are stored in each file
        daskVarArraysShape = da.reshape(
            daskVarArraysStack,
            (daskVarArraysStack.shape[0] * daskVarArraysStack.shape[1], 1),
            merge_chunks=False,
        )
        daskCatTimes = da.concatenate(
            (da.squeeze(daskVarArraysShape), daskForecastArray), axis=0
        ).astype("float32")

        # Get times as numpy
        npCatTimes = daskCatTimes.compute()

        daskArrayOut = da.from_array(
            np.tile(
                np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                (1, NY, NX),
            )
        ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack,
            (daskVarArraysStack.shape[0] * daskVarArraysStack.shape[1], NY, NX),
            merge_chunks=False,
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
            .astype("float32")
        )

    daskVarArrays = []

    print(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Mask out invalid data
# Ignore storm distance, since it can reach very high values that are still correct
daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
# with ProgressBar():
with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
    daskVarArrayListMergeNaN.to_zarr(
        forecast_process_path + "_stack.zarr",
        overwrite=True,
        compute=True,
    )

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/GDPS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/GDPS.zarr")


#
# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Pad to chunk size
# 3. Create the zarr array
# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array

with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        # 1. Interpolate the stacked array to be hourly along the time axis
        daskVarArrayStackDiskInterp = interp_time_take_blend(
            daskVarArrayStackDisk,
            stacked_timesUnix=stacked_timesUnix,
            hourly_timesUnix=hourly_timesUnix,
            dtype="float32",
            fill_value=np.nan,
        )

        # 2. Pad to chunk size
        daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
            daskVarArrayStackDiskInterp, final_chunk
        )

        # 3. Create the zarr array
        zarr_array = zarr.create_array(
            store=zarr_store,
            shape=(
                len(zarr_vars),
                len(hourly_timesUnix),
                daskVarArrayStackDiskInterpPad.shape[2],
                daskVarArrayStackDiskInterpPad.shape[3],
            ),
            chunks=(len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )

        # 4. Rechunk it to match the final array
        # 5. Write it out to the zarr array
        daskVarArrayStackDiskInterpPad.round(5).rechunk(
            (len(zarr_vars), len(hourly_timesUnix), final_chunk, final_chunk)
        ).to_zarr(zarr_array, overwrite=True, compute=True)


close_store(zarr_store)

# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/GDPS.zarr.zip",
        forecast_path + "/" + ingest_version + "/GDPS.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/GDPS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/GDPS.time.pickle",
        forecast_path + "/" + ingest_version + "/GDPS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/GDPS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/GDPS.time.pickle",
        forecast_path + "/" + ingest_version + "/GDPS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GDPS.zarr",
        forecast_path + "/" + ingest_version + "/GDPS.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)

# Timing
T1 = time.time()
print(T1 - T0)
