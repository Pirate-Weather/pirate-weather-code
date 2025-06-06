# %% GEFS Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, September 2023

# %% Import modules
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
import zarr.storage
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore", "This pattern is interpreted")


# Scipy Interp Function
def linInterp(block, T_in, T_out):
    interp = make_interp_spline(T_in, block, 3, axis=1)
    interpOut = interp(T_out)
    return interpOut


# %% Setup paths and parameters
wgrib2_path = os.getenv("wgrib2_path", default="/home/ubuntu/wgrib2_build/bin/wgrib2 ")

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/GEFS"
)
forecast_process_path = forecast_process_dir + "/GEFS_Process"
hist_process_path = forecast_process_dir + "/GEFS_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/GEFS")
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/History/GEFS")


saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunksize
finalChunk = 3

hisPeriod = 36

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)

if saveType == "Download":
    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)


# %% Find the most recent run of the GEFS model
T0 = time.time()

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
latestRun = Herbie_latest(
    model="gefs",
    n=3,
    freq="6h",
    fxx=[240],
    product="atmos.25",
    verbose=False,
    member="avg",
    priority="aws",
)

base_time = latestRun.date
# Check if this is newer than the current file
if saveType == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/GEFS.time.pickle"):
        with s3.open(forecast_path + "/GEFS.time.pickle", "rb") as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GEFS, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/GEFS.time.pickle"):
        # Open the file in binary mode
        with open(forecast_path + "/GEFS.time.pickle", "rb") as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to GEFS, ending")
            sys.exit()


print(base_time)
# base_time = pd.Timestamp("2024-02-29 18:00:00")

zarrVars = (
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


hisPeriod = 36
s3_save_path = "/ForecastProd/GEFS/GEFS_Prob_"

#####################################################################################################
# %% Download forecast data for all 30 members to find percentages

# Define the subset of variables to download as a list of strings
matchstring_su = "(:(CRAIN|CICEP|CSNOW|CFRZR):)"
matchstring_ap = "(:APCP:)"

# Merge matchstrings for download
matchStrings = matchstring_su + "|" + matchstring_ap

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
gefs_range = range(3, 241, 3)

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
        priority="aws",
        save_dir=tmpDIR,
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
    FH_forecastsubMembers[mem].download(matchStrings, verbose=False)

    # Create list of downloaded grib files
    gribList = [
        str(Path(x.get_localFilePath(matchStrings)).expand())
        for x in FH_forecastsubMembers[mem].file_exists
    ]

    # Create a string to pass to wgrib2 to merge all gribs into one grib
    cmd = (
        "cat "
        + " ".join(gribList)
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
    spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
    if spOUT.returncode != 0:
        print(spOUT.stderr)
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
        chunks={"time": 80, "latitude": processChunk, "longitude": processChunk}
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
    start=start - pd.Timedelta(hisPeriod, "h"), end=end, freq="h"
)

# Plus 2 since we start at Hour 3
stacked_times = np.concatenate(
    (
        pd.date_range(
            start=start - pd.Timedelta(hisPeriod, "h"),
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
for dask_var in zarrVars:
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
        daskOutput[dask_var].rechunk((80, processChunk, processChunk)).to_zarr(
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
for i in range(hisPeriod, 0, -6):
    # s3_path_NC = s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ') + '.nc'

    # Try to open the zarr file to check if it has already been saved
    if saveType == "S3":
        # Create the S3 filesystem
        s3_path = (
            historic_path
            + "/GEFS_HistProb_"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        if s3.exists(s3_path):
            continue
    else:
        # Local Path Setup
        local_path = (
            historic_path
            + "/GEFS_HistProb_"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
        )

        # Check if local file exists
        if os.path.exists(local_path):
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
                priority="aws",
                save_dir=tmpDIR,
            )
        )
    # Download the subsets
    for mem in range(0, 30):
        # Download the subsets
        FH_forecastsubMembers[mem].download(matchStrings, verbose=False)
        # Create list of downloaded grib files
        gribList = [
            str(Path(x.get_localFilePath(matchStrings)).expand())
            for x in FH_forecastsubMembers[mem].file_exists
        ]

        # Create a string to pass to wgrib2 to merge all gribs into one grib
        cmd = (
            "cat "
            + " ".join(gribList)
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
        spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        if spOUT.returncode != 0:
            print(spOUT.stderr)
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
        #     for vname in zarrVars[1:]
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
        {"time": 2, "latitude": processChunk, "longitude": processChunk}
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
    if saveType == "S3":
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

#####################################################################################################
# %% Merge the historic and forecast datasets and then squash using dask
ncLocalWorking_paths = [
    historic_path
    + "/GEFS_HistProb_"
    + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
    + ".zarr"
    for i in range(hisPeriod, 0, -6)
]

# Dask Setup
daskInterpArrays = []
daskVarArrays = []
daskVarArrayList = []

for daskVarIDX, dask_var in enumerate(probVars[:]):
    for local_ncpath in ncLocalWorking_paths:
        if saveType == "S3":
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
        daskVarArraysShape = da.reshape(daskVarArraysStack, (12, 1), merge_chunks=False)
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
        ).rechunk((len(stacked_timesUnix), processChunk, processChunk))

        daskVarArrayList.append(daskArrayOut)

    else:
        daskVarArraysShape = da.reshape(
            daskVarArraysStack, (12, 721, 1440), merge_chunks=False
        )
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        daskVarArrayList.append(
            daskArrayOut[:, :, :]
            .rechunk((len(stacked_timesUnix), processChunk, processChunk))
            .astype("float32")
        )

    daskVarArrays = []

    print(dask_var)

# Merge the arrays into a single 4D array
daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

# Write out to disk
# This intermediate step is necessary to avoid memory overflow
# with ProgressBar():
daskVarArrayListMerge.to_zarr(
    forecast_process_path + "GEFS_stack.zarr", overwrite=True, compute=True
)

# Read in stacked 4D array back in
daskVarArrayListMerge.to_zarr(
    forecast_process_path + "_stack.zarr", overwrite=True, compute=True
)

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

# Create a zarr backed dask array
if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/GEFS.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/GEFS.zarr")

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarrVars),
        len(hourly_timesUnix),
        daskVarArrayStackDisk.shape[2],
        daskVarArrayStackDisk.shape[3],
    ),
    chunks=(len(zarrVars), len(hourly_timesUnix), finalChunk, finalChunk),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)

#
# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Rechunk it to match the final array
# 3. Write it out to the zarr array
stackInterp = da.rechunk(
    da.map_blocks(
        linInterp,
        daskVarArrayStackDisk,
        stacked_timesUnix,
        hourly_timesUnix,
        dtype="float32",
        chunks=(1, len(stacked_timesUnix), processChunk, processChunk),
    ).round(3),
    (len(zarrVars), len(hourly_timesUnix), finalChunk, finalChunk),
).to_zarr(zarr_array, overwrite=True, compute=True)


# Close the zarr
if saveType == "S3":
    zarr_store.close()


# %% Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/GEFS.zarr.zip", forecast_path + "/GEFS.zarr.zip"
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/GEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/GEFS.time.pickle",
        forecast_path + "/GEFS.time.pickle",
    )
else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/GEFS.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/GEFS.time.pickle",
        forecast_path + "/GEFS.time.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/GEFS.zarr",
        forecast_path + "/GEFS.zarr",
        dirs_exist_ok=True,
    )


# Clean up
shutil.rmtree(forecast_process_dir)

# Test Read
T1 = time.time()
print(T1 - T0)
