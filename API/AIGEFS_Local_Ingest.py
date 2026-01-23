# %% Script to test FastHerbie.py to download GFS data
# Alexander Rey, September 2023

# %% Import modules
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import FastHerbie, HerbieLatest, Path

from API.constants.shared_const import (
    GRAVITY,
    HISTORY_PERIODS,
    INGEST_VERSION_STR,
    MISSING_DATA,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    calculate_freezing_level,
    derive_precip_type,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    validate_grib_stats,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/AIGEFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "AIGEFS_Process")
hist_process_path = os.path.join(forecast_process_dir, "AIGEFS_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/AIGEFS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/AIGEFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["GFS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GFS"]

his_period = HISTORY_PERIODS["GFS"]

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


T0 = time.time()

latest_run = HerbieLatest(
    model="aigefs",
    n=3,
    freq="6h",
    fxx=240,
    product="sfc",
    verbose=False,
    member="avg",
    priority=["nomads"],
    save_dir=tmp_dir,
)

base_time = latest_run.date
# base_time = pd.Timestamp("2024-03-24 06:00:00Z")

logger.info(base_time)


# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to AIGEFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            logger.info("No Update to AIGEFS, ending")
            sys.exit()

zarr_vars = (
    "time",
    "APCP_surface",
    "freezing_level",
)

prob_vars = (
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
)

#####################################################################################################
# %% Download forecast data for all 31 members to find percentages

# Merge matchstrings for download
match_strings = "(:APCP:surface:0-[1-9]*)"

# Create a range of forecast lead times
aigefs_range = FORECAST_LEAD_RANGES["AIGEFS"]

# Create FastHerbie object for all 31 members
FH_forecastsubMembers = []
mem = 0
failCount = 0
while mem < 31:
    FH_IN = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="6h"),
        model="aigefs",
        fxx=aigefs_range,
        member=mem + 1,
        product="sfc",
        verbose=False,
        priority=["nomads"],
        save_dir=tmp_dir,
    )

    # Check for download length
    if len(FH_IN.file_exists) != 80:
        logger.warning("Member %d has not downloaded all files, trying again", mem + 1)
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            logger.error(
                "Failed to download files for member %d after multiple attempts.",
                mem + 1,
            )
            sys.exit(1)

        continue

    FH_forecastsubMembers.append(FH_IN)

    # Download and process the subsets
    FH_forecastsubMembers[mem].download(match_strings, verbose=False)

    # Create list of downloaded grib files
    grib_list = [
        str(Path(x.get_localFilePath(match_strings)).expand())
        for x in FH_forecastsubMembers[mem].file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + "- -s -stats"

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

    validate_grib_stats(grib_check)
    logger.info("Grib files passed validation, proceeding with processing")

    # Create a string to pass to wgrib2 to merge all gribs into one grib
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + "-netcdf "
        + forecast_process_path
        + "_wgrib2_merged_m"
        + str(mem + 1)
        + ".nc"
    )

    # Run wgrib2 to megre all the grib files
    sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if sp_out.returncode != 0:
        logger.error(sp_out.stderr)
        sys.exit()

    # Fix precip and chunk each member
    xarray_wgrib = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc"
    )

    # Save coordinates and time from the first member for consistency
    if mem == 0:
        first_member_lat = xarray_wgrib["latitude"]
        first_member_lon = xarray_wgrib["longitude"]
        first_member_time_min = xarray_wgrib.time.min().values
        first_member_time_max = xarray_wgrib.time.max().values

    # Sometimes there will be weird tiny negative values, set them to zero
    xarray_wgrib["APCP_surface"] = np.maximum(xarray_wgrib["APCP_surface"], 0)

    # Divide by 6 to get hourly accumulation for precipitation only
    # AIGEFS provides 6-hourly accumulation, convert to hourly rate
    xarray_wgrib["APCP_surface"] = xarray_wgrib["APCP_surface"] / 6

    # Get a list of all variables in the dataset
    wgribVars = list(xarray_wgrib.data_vars)

    # Define compression and chunking for each variable
    # Compress to save space
    # Save the dataset as a nc file with compression
    # encoding = {
    #     vname: {"zlib": True, "complevel": 1, "chunksizes": (80, 60, 60)}
    #     for vname in wgribVars
    # }
    # xarray_wgrib.to_netcdf(
    #     forecast_process_path + "_xr_m" + str(mem + 1) + ".nc", encoding=encoding
    # )

    xarray_wgrib = xarray_wgrib.chunk(
        chunks={"time": 80, "latitude": process_chunk, "longitude": process_chunk}
    )

    xarray_wgrib.to_zarr(
        forecast_process_path + "_xr_m" + str(mem + 1) + ".zarr",
        consolidated=False,
        mode="w",
    )

    # Delete the wgrib netcdf to save space
    subprocess.run(
        "rm " + forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc",
        shell=True,
        capture_output=True,
        encoding="utf-8",
    )

    mem += 1

# Create a new time series using the first member's time range
start = first_member_time_min  # Adjust as necessary
end = first_member_time_max  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(his_period, "h"), end=end, freq="h"
)

stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(his_period, "h"),
            end=start - pd.Timedelta(1, "h"),
            freq="6h",
        ),
        xarray_wgrib.time.values,
    )
)

unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

ncLocalWorking_paths = [
    forecast_process_path + "_xr_m" + str(i) + ".zarr" for i in range(1, 31, 1)
]

# Dask
daskArrays = dict()


# Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
# Note that the chunks
for dask_var in zarr_vars:
    daskVarArrays = []
    for local_ncpath in ncLocalWorking_paths:
        daskVarArrays.append(da.from_zarr(local_ncpath, dask_var))

    # Stack times together, keeping variables separate
    daskArrays[dask_var] = da.stack(daskVarArrays, axis=0)

    daskVarArrays = []


# Dict to hold output dask arrays
daskOutput = dict()

# Find the probability of precipitation greater than 0.0001 mm/h  across all members
daskOutput["Precipitation_Prob"] = ((daskArrays["APCP_surface"]) > 0.0001).sum(
    axis=0
) / daskArrays["APCP_surface"].shape[0]

# Find the standard deviation of precipitation accumulation across all members
daskOutput["APCP_StdDev"] = daskArrays["APCP_surface"].std(axis=0)

# Find the average precipitation accumulation across all members
daskOutput["APCP_Mean"] = daskArrays["APCP_surface"].mean(axis=0)

# Copy time over (use first member [0] for consistency)
daskOutput["time"] = daskArrays["time"][0, :]


# filters = [BitRound(keepbits=12)]  # Only keep ~ 3 significant digits
# compressor = Blosc(cname="zstd", clevel=1)  # Use zstd compression

for dask_var in prob_vars:
    # with ProgressBar():
    if dask_var == "time":
        daskOutput[dask_var].to_zarr(
            forecast_process_path + "_" + dask_var + ".zarr",
            codecs=[
                zarr.codecs.BytesCodec(),
                zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            ],
            overwrite=True,
            compute=True,
        )
    else:
        daskOutput[dask_var].rechunk((80, process_chunk, process_chunk)).to_zarr(
            forecast_process_path + "_" + dask_var + ".zarr",
            codecs=[
                zarr.codecs.BytesCodec(),
                zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            ],
            overwrite=True,
            compute=True,
        )


# %% Delete to free memory (keep objects needed for building the merged forecast)
# We keep `xarray_wgrib` and `daskOutput` here because they are used
# to construct `xarray_forecast_merged` below.
del FH_forecastsubMembers, daskArrays

# Build an xarray Dataset from the dask outputs so downstream code can
# add derived variables (freezing level) and write a single forecast zarr.
logger.info("Constructing merged forecast xarray dataset from dask outputs")

# Use coordinate arrays from the first member (saved earlier) for consistency
coords = {
    "time": ("time", daskOutput["time"]),
    "latitude": ("latitude", first_member_lat),
    "longitude": ("longitude", first_member_lon),
}

# Create Dataset using DataArray wrappers so dask arrays are preserved lazily
xarray_forecast_merged = xr.Dataset(
    {
        "APCP_Mean": xr.DataArray(
            daskOutput["APCP_Mean"], dims=("time", "latitude", "longitude")
        ),
        "APCP_StdDev": xr.DataArray(
            daskOutput["APCP_StdDev"], dims=("time", "latitude", "longitude")
        ),
        "Precipitation_Prob": xr.DataArray(
            daskOutput["Precipitation_Prob"], dims=("time", "latitude", "longitude")
        ),
        "time": xr.DataArray(daskOutput["time"], dims=("time",)),
    },
    coords={
        "time": daskOutput["time"],
        "latitude": first_member_lat,
        "longitude": first_member_lon,
    },
)


#####################################################################################################
# %% Download pressure level data for freezing level and cloud cover calculation
logger.info("Downloading pressure level data for derived parameters")

# Define pressure levels (in hPa)
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Match strings for pressure level data
# AIGEFS uses TMP, HGT, SPFH at pressure levels
matchstring_pressure_tmp = (
    "(:TMP:(1000|925|850|700|600|500|400|300|250|200|150|100|50) mb:)"
)
matchstring_pressure_hgt = (
    "(:HGT:(1000|925|850|700|600|500|400|300|250|200|150|100|50) mb:)"
)

match_strings_pressure = matchstring_pressure_tmp + "|" + matchstring_pressure_hgt

# Also download surface temperature for precipitation type calculation
matchstring_surface_tmp = "(:TMP:2 m above ground:)"

# Download pressure level data
FH_pressure = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aigefs",
    fxx=aigefs_range,
    product="pres",
    verbose=False,
    member="avg",
    priority=["nomads"],
    save_dir=tmp_dir,
)

FH_pressure.download(match_strings_pressure, verbose=False)

# Download surface temperature from avg member
FH_surface = FastHerbie(
    pd.date_range(start=base_time, periods=1, freq="6h"),
    model="aigefs",
    fxx=aigefs_range,
    product="sfc",
    verbose=False,
    member="avg",
    priority=["nomads"],
    save_dir=tmp_dir,
)

FH_surface.download(matchstring_surface_tmp, verbose=False)

# Create list of downloaded grib files
grib_list_pressure = [
    str(Path(x.get_localFilePath(match_strings_pressure)).expand())
    for x in FH_pressure.file_exists
]

# Create list of downloaded surface temperature grib files
grib_list_surface = [
    str(Path(x.get_localFilePath(matchstring_surface_tmp)).expand())
    for x in FH_surface.file_exists
]

# Merge into netcdf
cmd = (
    "cat "
    + " ".join(grib_list_pressure)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_pressure.nc"
)

sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    logger.error(sp_out.stderr)
    sys.exit()

# Merge surface temperature into netcdf
cmd_surface = (
    "cat "
    + " ".join(grib_list_surface)
    + " | "
    + f"{wgrib2_path}"
    + " - "
    + " -netcdf "
    + forecast_process_path
    + "_surface.nc"
)

sp_out = subprocess.run(cmd_surface, shell=True, capture_output=True, encoding="utf-8")
if sp_out.returncode != 0:
    logger.error(sp_out.stderr)
    sys.exit()

# Read pressure level data
xarray_pressure = xr.open_dataset(forecast_process_path + "_pressure.nc")

# Read surface temperature data
xarray_surface = xr.open_dataset(forecast_process_path + "_surface.nc")

# Calculate freezing level
logger.info("Calculating freezing level height")
# Need to convert variables to match expected format
# GFS uses "TMP_<level>mb" format, need to stack into DataArray with level dimension
temp_vars = [f"TMP_{level}mb" for level in pressure_levels]
hgt_vars = [f"HGT_{level}mb" for level in pressure_levels]

# Stack into DataArrays with level dimension
temp_data = xr.concat(
    [xarray_pressure[var] for var in temp_vars], dim="level"
).assign_coords(level=pressure_levels)

hgt_data = xr.concat(
    [xarray_pressure[var] for var in hgt_vars], dim="level"
).assign_coords(level=pressure_levels)

# Calculate freezing level (convert HGT to geopotential: z = g * h)
geopotential_data = hgt_data * GRAVITY

freezing_level = calculate_freezing_level(
    temperature_levels=temp_data,
    geopotential_levels=geopotential_data,
    pressure_levels=pressure_levels,
)

# Add derived variables to the forecast dataset
xarray_forecast_merged["freezing_level"] = freezing_level

# Derive precipitation type (1=snow,2=freezing rain,3=sleet,4=rain)
try:
    # Check if surface temperature is available
    temp_surface = xarray_surface.get("TMP_2maboveground", None)
    if temp_surface is None:
        logger.warning("Surface temperature (TMP_2maboveground) not found, precip type may be less accurate")

    xarray_forecast_merged["precip_type"] = derive_precip_type(
        apcp=xarray_forecast_merged["APCP_Mean"],
        temp_surface=temp_surface,
        temp_levels=temp_data,
        geopotential_levels=geopotential_data,
        pressure_levels=pressure_levels,
    )
except Exception:
    logger.exception("Could not derive precip_type; skipping")

# Clean up
del xarray_pressure, temp_data, hgt_data, geopotential_data
if os.path.exists(forecast_process_path + "_pressure.nc"):
    os.remove(forecast_process_path + "_pressure.nc")
if os.path.exists(forecast_process_path + "_surface.nc"):
    os.remove(forecast_process_path + "_surface.nc")
if 'xarray_surface' in locals():
    del xarray_surface

logger.info("Derived parameter calculation complete")

# %% Save merged and processed xarray dataset to disk using zarr with compression

# Save the dataset with compression and filters for all variables
xarray_forecast_merged = xarray_forecast_merged.chunk(
    chunks={"time": 240, "latitude": process_chunk, "longitude": process_chunk}
)
xarray_forecast_merged.to_zarr(
    forecast_process_path + "_.zarr", mode="w", consolidated=False, compute=True
)

# %% Delete to free memory (keep only existing variables)
del xarray_forecast_merged, freezing_level, xarray_wgrib
T1 = time.time()

logger.info(T1 - T0)
if os.path.exists(forecast_process_path + "_wgrib2_merged.nc"):
    os.remove(forecast_process_path + "_wgrib2_merged.nc")

################################################################################################
# %% Historic data
# Loop through the runs and check if they have already been processed to s3

# 6 hour runs
for i in range(his_period, 0, -6):
    if save_type == "S3":
        s3_path = (
            historic_path
            + "/AIGEFS_Hist_v2"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a done file in S3
        if s3.exists(s3_path.replace(".zarr", ".done")):
            logger.info("File already exists in S3, skipping download for: %s", s3_path)
            # If the file exists, check that it works
            try:
                hisCheckStore = zarr.storage.FsspecStore.from_url(
                    s3_path,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                )
                zarr.open(hisCheckStore)[zarr_vars[-1]][-1, -1, -1]
                continue  # If it exists, skip to the next iteration
            except Exception:
                logger.error("### Historic Data Failure!")
                logger.exception("Exception processing historic data", exc_info=True)

                # Delete the file if it exists
                if s3.exists(s3_path):
                    s3.rm(s3_path)
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/AIGEFS_Hist_v2"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a loca done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            logger.info(
                "File already exists in S3, skipping download for: %s", local_path
            )
            continue

    logger.info(
        "Downloading: %s",
        (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"),
    )

    # Create a range of dates for historic data going back 48 hours
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="6h",
    )
    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach
    fxx = range(1, 7)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(
        DATES,
        model="aigefs",
        fxx=fxx,
        product="sfc",
        verbose=False,
        member="avg",
        priority=["nomads"],
        save_dir=tmp_dir,
    )

    # Download the subsets
    FH_histsub.download(match_strings, verbose=False)

    # Check for download length
    if len(FH_histsub.file_exists) != len(fxx):
        logger.error(
            "Download failed, expected 6 files but got %d", len(FH_histsub.file_exists)
        )
        sys.exit(1)

    # Create list of downloaded grib files
    grib_list = [
        str(Path(x.get_localFilePath(match_strings)).expand())
        for x in FH_histsub.file_exists
    ]

    # Perform a check if any data seems to be invalid
    cmd = "cat " + " ".join(grib_list) + " | " + f"{wgrib2_path}" + " - " + " -s -stats"

    grib_check = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

    validate_grib_stats(grib_check)
    logger.info("Grib files passed validation, proceeding with processing")

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
    sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if sp_out.returncode != 0:
        logger.error(sp_out.stderr)
        sys.exit()

    # Read the netcdf file using xarray (no UV product for Graphcast)
    xarray_his_wgrib_merged = xr.open_dataset(hist_process_path + "_wgrib2_merged.nc")

    xarray_hist_merged = xarray_his_wgrib_merged

    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time, every 6 hours
    apcpProc = xarray_hist_merged["APCP_surface"].values

    apcpProcHour = np.diff(apcpProc, axis=0, prepend=0)

    xarray_hist_merged["APCP_surface"] = xarray_hist_merged["APCP_surface"].copy(
        data=apcpProcHour
    )

    # Clear memory of temporary inputs
    del xarray_his_wgrib_merged

    ########################################################################
    ### Download pressure level data for historical freezing level and cloud cover
    logger.info("Downloading historical pressure level data")

    # Download pressure level data
    FH_histsub_pressure = FastHerbie(
        DATES,
        model="aigefs",
        fxx=fxx,
        product="pres",
        verbose=False,
        member="avg",
        priority=["nomads"],
        save_dir=tmp_dir,
    )

    FH_histsub_pressure.download(match_strings_pressure, verbose=False)

    # Create list of downloaded grib files
    grib_list_pressure_his = [
        str(Path(x.get_localFilePath(match_strings_pressure)).expand())
        for x in FH_histsub_pressure.file_exists
    ]

    # Merge into netcdf
    cmd = (
        "cat "
        + " ".join(grib_list_pressure_his)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -netcdf "
        + hist_process_path
        + "_pressure.nc"
    )

    sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if sp_out.returncode != 0:
        logger.error(sp_out.stderr)
        sys.exit()

    # Read pressure level data
    xarray_pressure_his = xr.open_dataset(hist_process_path + "_pressure.nc")

    # Stack into DataArrays
    temp_data_his = xr.concat(
        [xarray_pressure_his[var] for var in temp_vars], dim="level"
    ).assign_coords(level=pressure_levels)

    hgt_data_his = xr.concat(
        [xarray_pressure_his[var] for var in hgt_vars], dim="level"
    ).assign_coords(level=pressure_levels)

    # Calculate derived parameters
    geopotential_data_his = hgt_data_his * GRAVITY

    freezing_level_his = calculate_freezing_level(
        temperature_levels=temp_data_his,
        geopotential_levels=geopotential_data_his,
        pressure_levels=pressure_levels,
    )

    # Add to historical dataset
    xarray_hist_merged["freezing_level"] = freezing_level_his

    # Clean up
    del xarray_pressure_his, temp_data_his, hgt_data_his
    del geopotential_data_his, freezing_level_his
    os.remove(hist_process_path + "_pressure.nc")

    # PRES_surface not used for Graphcast; skip renaming

    # Save merged and processed xarray dataset to disk using zarr with compression
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    # Save as Zarr to s3 for Time Machine
    if save_type == "S3":
        zarrStore = zarr.storage.FsspecStore.from_url(
            s3_path,
            storage_options={
                "key": aws_access_key_id,
                "secret": aws_secret_access_key,
            },
        )
    else:
        # Create local Zarr store
        zarrStore = zarr.storage.LocalStore(local_path)

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chunks to speed up read times
    # Small fix for pressure variable naming not required for Graphcast
    encoding = {
        vname: {"chunks": (6, process_chunk, process_chunk)} for vname in zarr_vars[1:]
    }

    # with ProgressBar():
    xarray_hist_merged.to_zarr(
        store=zarrStore, mode="w", consolidated=False, encoding=encoding
    )

    # Clear the xarray dataset from memory
    del xarray_hist_merged

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + "_wgrib2_merged.nc")

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        done_file = s3_path.replace(".zarr", ".done")
        s3.touch(done_file)
    else:
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

    logger.info((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))


# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
ncLocalWorking_paths = [
    historic_path
    + "/AIGEFS_Hist_v2"
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
            if save_type == "S3":
                daskVarArrays.append(
                    da.from_zarr(
                        local_ncpath,
                        component=dask_var,
                        inline_array=True,
                        storage_options={
                            "key": aws_access_key_id,
                            "secret": aws_secret_access_key,
                        },
                    )
                )

            else:
                daskVarArrays.append(
                    da.from_zarr(local_ncpath, component=dask_var, inline_array=True)
                )
        # Add a fallback in case of a FileNotFoundError
        except FileNotFoundError:
            logger.warning("File not found, adding NaN array for: %s", local_ncpath)
            daskVarArrays.append(
                da.full((6, 721, 1440), MISSING_DATA).rechunk(
                    (6, process_chunk, process_chunk)
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
                (1, 721, 1440),
            )
        ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack,
            (daskVarArraysStack.shape[0] * daskVarArraysStack.shape[1], 721, 1440),
            merge_chunks=False,
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
            .astype("float32")
        )

    daskVarArrays = []

    logger.info(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Mask out invalid data
# Ignore storm distance, since it can reach very high values that are still correct
daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
# with ProgressBar():
daskVarArrayListMergeNaN.to_zarr(
    forecast_process_path + "_stack.zarr", overwrite=True, compute=True
)

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/AIGEFS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/AIGEFS.zarr")


#
# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Pad to chunk size
# 3. Create the zarr array
# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array

with ProgressBar():
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


if save_type == "S3":
    zarr_store.close()

# Maps generation removed for Graphcast ingest (not required)

# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/AIGEFS.zarr.zip",
        forecast_path + "/" + ingest_version + "/AIGEFS.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/AIGEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/AIGEFS.time.pickle",
        forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/AIGEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/AIGEFS.time.pickle",
        forecast_path + "/" + ingest_version + "/AIGEFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/AIGEFS.zarr",
        forecast_path + "/" + ingest_version + "/AIGEFS.zarr",
        dirs_exist_ok=True,
    )

    # Maps not generated for Graphcast ingest
# Clean up
shutil.rmtree(forecast_process_dir)

# Timing
T1 = time.time()
logger.info(T1 - T0)
