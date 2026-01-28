# %% GEFS Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, September 2023

# %% Import modules
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    validate_grib_stats,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GEFS")
forecast_process_path = forecast_process_dir + "/GEFS_Process"
hist_process_path = forecast_process_dir + "/GEFS_Historic"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GEFS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/GEFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["GEFS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GEFS"]

his_period = HISTORY_PERIODS["GEFS"]

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


# %% Find the most recent run of the GEFS model
T0 = time.time()

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
latest_run = Herbie_latest(
    model="gefs",
    n=3,
    freq="6h",
    fxx=[240],
    product="atmos.25",
    verbose=True,
    member="avg",
    priority=["aws", "nomads"],
)

base_time = latest_run.date
# Check if this is newer than the current file
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/GEFS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/GEFS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GEFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/GEFS.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/GEFS.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GEFS, ending")
            sys.exit()


print(base_time)
# base_time = pd.Timestamp("2024-02-29 18:00:00")

zarr_vars = (
    "time",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
)

probVars = (
    "time",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
    "CSNOW_Prob",
    "CICEP_Prob",
    "CFRZR_Prob",
    "CRAIN_Prob",
)

s3_save_path = "/ForecastProd/GEFS/GEFS_Prob_"

#####################################################################################################
# %% Download forecast data for all 30 members to find percentages

# Define the subset of variables to download as a list of strings
matchstring_su = "(:(CRAIN|CICEP|CSNOW|CFRZR):)"
matchstring_ap = "(:APCP:)"

# Merge matchstrings for download
match_strings = matchstring_su + "|" + matchstring_ap

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach

gefs_range = FORECAST_LEAD_RANGES["GEFS"]

# Create FastHerbie object for all 30 members
FH_forecastsubMembers = []
mem = 0
failCount = 0
while mem < 30:
    FH_IN = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="6h"),
        model="gefs",
        fxx=gefs_range,
        member=mem + 1,
        product="atmos.25",
        verbose=False,
        priority=["aws", "nomads"],
        save_dir=tmp_dir,
    )

    # Check for download length
    if len(FH_IN.file_exists) != 80:
        print("Member " + str(mem + 1) + " has not downloaded all files, trying again")
        failCount += 1

        # Break after 10 failed attempts
        if failCount > 10:
            break

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
    print("Grib files passed validation, proceeding with processing")

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
        print(sp_out.stderr)
        sys.exit()

    # Fix precip and chunk each member
    xarray_wgrib = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc"
    )

    # Change from 3 and 6 hour accumulation to 3 hour accumulation
    # Use the difference between 3 and 6 for every other timestep
    apcp_diff_xr = xarray_wgrib["APCP_surface"].diff(dim="time")
    xarray_wgrib["APCP_surface"][slice(1, None, 2), :, :] = apcp_diff_xr[
        slice(0, None, 2), :, :
    ]

    # Sometimes there will be weird tiny negative values, set them to zero
    xarray_wgrib["APCP_surface"] = np.maximum(xarray_wgrib["APCP_surface"], 0)

    # Divide by 3 to get hourly accum
    xarray_wgrib["APCP_surface"] = xarray_wgrib["APCP_surface"] / 3

    # NOTE: Because the cateogical vars are mixed (0-3 and 0-6) intervals, there can be values even when there's no precip
    for var in ["CRAIN", "CSNOW", "CFRZR", "CICEP"]:
        xarray_wgrib[var + "_surface"] = xarray_wgrib[var + "_surface"].where(
            xarray_wgrib["APCP_surface"] != 0, 0
        )

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

# Create a new time series
start = xarray_wgrib.time.min().values  # Adjust as necessary
end = xarray_wgrib.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(
    start=start - pd.Timedelta(his_period, "h"), end=end, freq="h"
)

# Plus 2 since we start at Hour 3
stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(his_period, "h"),
            end=start - pd.Timedelta(1, "h"),
            freq="3h",
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

# Find the probability of precipitation greater than 0.1 mm/h  across all members
daskOutput["Precipitation_Prob"] = ((daskArrays["APCP_surface"]) > 0.1).sum(axis=0) / 30

# Find the average precipitation accumulation and categorical parameters across all members
for var in ["CRAIN", "CSNOW", "CFRZR", "CICEP"]:
    daskOutput[var + "_Prob"] = daskArrays[var + "_surface"].sum(axis=0) / 30

# Find the standard deviation of precipitation accumulation across all members
daskOutput["APCP_StdDev"] = daskArrays["APCP_surface"].std(axis=0)

# Find the average precipitation accumulation across all members
daskOutput["APCP_Mean"] = daskArrays["APCP_surface"].mean(axis=0)

# Copy time over
daskOutput["time"] = daskArrays["time"][1, :]


# filters = [BitRound(keepbits=12)]  # Only keep ~ 3 significant digits
# compressor = Blosc(cname="zstd", clevel=1)  # Use zstd compression

for dask_var in probVars:
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


# %% Delete to free memory
del xarray_wgrib, FH_forecastsubMembers, daskOutput, daskArrays, apcp_diff_xr

################################################################################################
# %% Historic data
# Create a range of dates for historic data going back 48 hours
# Loop through the runs and check if they have already been processed to s3


# Preprocess function for xarray to add a member dimension
def preprocess(ds):
    return ds.expand_dims("member", axis=0)


# Note: since these files are only 6 hour segments (aka 2 timesteps instead of 80), all the fancy dask stuff isn't necessary
# 6 hour runs
for i in range(his_period, 0, -6):
    # s3_path_NC = s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ') + '.nc'

    # Try to open the zarr file to check if it has already been saved
    if save_type == "S3":
        # Create the S3 filesystem
        s3_path = (
            historic_path
            + "/GEFS_HistProb_"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a done file in S3
        if s3.exists(s3_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + s3_path)

            # If the file exists, check that it works
            try:
                hisCheckStore = zarr.storage.FsspecStore.from_url(
                    s3_path,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                )
                zarr.open(hisCheckStore)[probVars[-1]][-1, -1, -1]
                continue  # If it exists, skip to the next iteration
            except Exception:
                print("### Historic Data Failure!")
                print(traceback.print_exc())

                # Delete the file if it exists
                if s3.exists(s3_path):
                    s3.rm(s3_path)
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/GEFS_HistProb_"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check for a loca done file
        if os.path.exists(local_path.replace(".zarr", ".done")):
            print("File already exists in S3, skipping download for: " + local_path)
            continue
    print(
        "Downloading: " + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    )

    # Create a range of dates for historic data
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="6h",
    )

    # Create a range of forecast lead times
    # Forward looking, so 00Z forecast is from 03Z
    # This is what we want for accumilation variables
    FH_forecastsubMembers = []
    for mem in range(0, 30):
        FH_forecastsubMembers.append(
            FastHerbie(
                DATES,
                model="gefs",
                fxx=range(3, 7, 3),
                member=mem + 1,
                product="atmos.25",
                verbose=False,
                priority=["aws", "nomads"],
                save_dir=tmp_dir,
            )
        )
    # Download the subsets
    for mem in range(0, 30):
        # Download the subsets
        FH_forecastsubMembers[mem].download(match_strings, verbose=False)
        # Create list of downloaded grib files
        grib_list = [
            str(Path(x.get_localFilePath(match_strings)).expand())
            for x in FH_forecastsubMembers[mem].file_exists
        ]

        # Perform a check if any data seems to be invalid
        cmd = (
            "cat "
            + " ".join(grib_list)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + " -s -stats"
        )

        grib_check = subprocess.run(
            cmd, shell=True, capture_output=True, encoding="utf-8"
        )

        validate_grib_stats(grib_check)
        print("Grib files passed validation, proceeding with processing")

        # Create a string to pass to wgrib2 to merge all gribs into one grib
        cmd = (
            "cat "
            + " ".join(grib_list)
            + " | "
            + f"{wgrib2_path}"
            + " - "
            + "-netcdf "
            + hist_process_path
            + "_wgrib2_merged_m"
            + str(mem + 1)
            + ".nc"
        )

        # Run wgrib2 to merge all the grib files
        sp_out = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        if sp_out.returncode != 0:
            print(sp_out.stderr)
            sys.exit()

        # Open the NetCDF file with xarray to process and compress
        xarray_hist_wgrib = xr.open_dataset(
            hist_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc"
        )

        # Change from 3 and 6 hour accumulations to 3 hour accumulations
        apcp_hist_diff_xr = xarray_hist_wgrib["APCP_surface"].diff(dim="time")
        xarray_hist_wgrib["APCP_surface"][slice(1, None, 2), :, :] = apcp_hist_diff_xr[
            slice(0, None, 2), :, :
        ]

        # Sometimes there will be weird negative values, set them to zero
        xarray_hist_wgrib["APCP_surface"] = np.maximum(
            xarray_hist_wgrib["APCP_surface"], 0
        )

        # Divide by 3 to get hourly accumilations
        xarray_hist_wgrib["APCP_surface"] = xarray_hist_wgrib["APCP_surface"] / 3

        # NOTE: Because the cateogical vars are mixed (0-3 and 0-6) intervals, there can be values even when there's no precip
        for var in ["CRAIN", "CSNOW", "CFRZR", "CICEP"]:
            xarray_hist_wgrib[var + "_surface"] = xarray_hist_wgrib[
                var + "_surface"
            ].where(xarray_hist_wgrib["APCP_surface"] != 0, 0)

        # Get a list of all variables in the dataset
        wgribVars = list(xarray_hist_wgrib.data_vars)

        # Save to NetCDF for prob process
        # encoding = {
        #     vname: {"zlib": True, "complevel": 1, "chunksizes": (2, 90, 90)}
        #     for vname in zarr_vars[1:]
        # }
        # xarray_hist_wgrib.to_netcdf(
        #     hist_process_path + "_xr_merged_m" + str(mem + 1) + ".nc", encoding=encoding
        # )

        xarray_wgrib = xarray_hist_wgrib.chunk(
            chunks={"time": 2, "latitude": 100, "longitude": 100}
        )

        xarray_wgrib.to_zarr(
            hist_process_path + "_xr_merged_m" + str(mem + 1) + ".zarr",
            consolidated=False,
            mode="w",
        )

        # Delete the netcdf to save space
        subprocess.run(
            "rm " + hist_process_path + "_wgrib2_merged_m" + str(mem + 1) + ".nc",
            shell=True,
            capture_output=True,
            encoding="utf-8",
        )

    # Calculate probabilities and standard deviation for historic data
    # Read the merged netcdf files into xarray using the preprocess function, concatenating along the member dimension
    xarray_hist_wgrib_merged = xr.open_mfdataset(
        [
            hist_process_path + "_xr_merged_m" + str(mem + 1) + ".zarr"
            for mem in range(0, 30)
        ],
        engine="zarr",
        preprocess=preprocess,
        combine="nested",
        concat_dim="member",
        chunks={"member": 30, "time": 2, "latitude": 100, "longitude": 100},
        consolidated=False,
    )

    # Create an empty xarray dataset to store the probability of precipitation greater than 1 mm
    xarray_hist_wgrib_prob = xr.Dataset()

    # Find the probably of precipitation greater than 0.1 mm/h across all members
    xarray_hist_wgrib_prob["Precipitation_Prob"] = (
        (xarray_hist_wgrib_merged["APCP_surface"]) > 0.1
    ).sum(dim="member") / 30

    # Find the average precipitation accumulation and categorical parameters across all members
    for var in ["CRAIN", "CSNOW", "CFRZR", "CICEP"]:
        xarray_hist_wgrib_prob[var + "_Prob"] = (
            xarray_hist_wgrib_merged[var + "_surface"].sum(dim="member") / 30
        )

    # Find the standard deviation of precipitation accumulation across all members
    xarray_hist_wgrib_prob["APCP_StdDev"] = xarray_hist_wgrib_merged[
        "APCP_surface"
    ].std(dim="member")

    # Find the average precipitation accumulation across all members
    xarray_hist_wgrib_prob["APCP_Mean"] = xarray_hist_wgrib_merged["APCP_surface"].mean(
        dim="member"
    )

    # Chunk the xarray dataset to speed up processing
    xarray_hist_wgrib_prob = xarray_hist_wgrib_prob.chunk(
        {"time": 2, "latitude": process_chunk, "longitude": process_chunk}
    )

    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times
    # Get a list of all variables in the dataset
    # compressor = Blosc(cname="lz4", clevel=1)
    # filters = [BitRound(keepbits=9)]

    # Don't filter time
    # encoding = {
    #     vname: {"compressor": compressor, "filters": filters} for vname in probVars[1:]
    # }

    # Save as zarr for timemachine
    # with ProgressBar():
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

    xarray_hist_wgrib_prob.to_zarr(store=zarrStore, mode="w", consolidated=False)

    # Clear memory
    del xarray_hist_wgrib_prob, xarray_hist_wgrib, xarray_hist_wgrib_merged

    # Save a done file to s3 to indicate that the historic data has been processed
    if save_type == "S3":
        done_file = s3_path.replace(".zarr", ".done")
        s3.touch(done_file)
    else:
        done_file = local_path.replace(".zarr", ".done")
        with open(done_file, "w") as f:
            f.write("Done")

#####################################################################################################
# %% Merge the historic and forecast datasets and then squash using dask
ncLocalWorking_paths = [
    historic_path
    + "/GEFS_HistProb_"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(his_period, 0, -6)
]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(probVars[:]):
    for local_ncpath in ncLocalWorking_paths:
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

    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

    # Add forecast as dask array
    daskForecastArray = da.from_zarr(
        forecast_process_path + "_" + dask_var + ".zarr", inline_array=True
    )

    if dask_var == "time":
        # Create a time array with the same shape
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

    print(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Mask out invalid data
daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
# with ProgressBar():
# Read in stacked 4D array back in
daskVarArrayListMergeNaN.to_zarr(
    forecast_process_path + "_stack.zarr", overwrite=True, compute=True
)

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/GEFS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/GEFS.zarr")


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
        len(probVars),
        len(hourly_timesUnix),
        daskVarArrayStackDiskInterpPad.shape[2],
        daskVarArrayStackDiskInterpPad.shape[3],
    ),
    chunks=(len(probVars), len(hourly_timesUnix), final_chunk, final_chunk),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)

# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array
daskVarArrayStackDiskInterpPad.round(5).rechunk(
    (len(probVars), len(hourly_timesUnix), final_chunk, final_chunk)
).to_zarr(zarr_array, overwrite=True, compute=True)

# Close the zarr
if save_type == "S3":
    zarr_store.close()


# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/GEFS.zarr.zip",
        forecast_path + "/" + ingest_version + "/GEFS.zarr.zip",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/GEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/GEFS.time.pickle",
        forecast_path + "/" + ingest_version + "/GEFS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/GEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/GEFS.time.pickle",
        forecast_path + "/" + ingest_version + "/GEFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GEFS.zarr",
        forecast_path + "/" + ingest_version + "/GEFS.zarr",
        dirs_exist_ok=True,
    )


# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
