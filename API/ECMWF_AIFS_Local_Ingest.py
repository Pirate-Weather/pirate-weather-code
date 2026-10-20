# %% Script to test FastHerbie.py to download ECMWF data
# Alexander Rey, March 2025

# %% Import modules
import logging
import os

# os.environ["ECCODES_DEFINITION_PATH"] = (
#    "/home/ubuntu/eccodes-2.40.0-Source/definitions/"
# )
import pickle
import shutil
import sys
import time
import warnings
from contextlib import nullcontext

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_gfs_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    close_store,
    configure_zarr_limits,
    interp_time_take_blend,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

AIFS_SOURCE_PRIORITY = ["google", "ecmwf", "aws"]
AIFS_MAX_THREADS = positive_int_env("aifs_download_threads", 4)
AIFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
AIFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)
AIFS_VERBOSE_LOGS = os.getenv("aifs_verbose_logs", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/ECMWF_AIFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "ECMWF_AIFS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/ECMWF_AIFS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


# Define the processing chunk size
process_chunk = CHUNK_SIZES["ECMWF"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["ECMWF"]


def download_aifs_subset(
    product: str,
    search: str,
    dataset_name: str,
    run_date: pd.Timestamp,
    forecast_hours: list[int],
) -> list[str]:
    """Download and validate an AIFS GRIB subset with bounded concurrency."""
    return download_and_validate_gfs_subset(
        model="aifs",
        product=product,
        search=search,
        dataset_name=dataset_name,
        base_time=run_date,
        forecast_hours=forecast_hours,
        wgrib2_exe=wgrib2_path.rstrip(),
        herbie_save_dir=tmp_dir,
        herbie_download_retries=AIFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=AIFS_RETRY_SLEEP_SECONDS,
        priority=AIFS_SOURCE_PRIORITY,
        herbie_kwargs={"max_threads": AIFS_MAX_THREADS},
        download_max_threads=AIFS_MAX_THREADS,
        download_verbose=AIFS_VERBOSE_LOGS,
    )


def aifs_progress_bar():
    """Show Dask progress only when verbose AIFS logging is enabled."""
    return ProgressBar() if AIFS_VERBOSE_LOGS else nullcontext()


def main() -> None:
    """Run the ECMWF AIFS forecast ingest."""
    logging.basicConfig(level=logging.INFO)

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = positive_int_env("zarr_store_workers", 2)
    zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    tune_nofile_limit()
    zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
        zarr_store_workers, zarr_async_concurrency
    )

    # Create new directory for processing if it does not exist
    if not os.path.exists(forecast_process_dir):
        os.makedirs(forecast_process_dir)
    else:
        # If it does exist, remove it
        shutil.rmtree(forecast_process_dir)
        os.makedirs(forecast_process_dir)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if save_type == "Download" and not os.path.exists(
        forecast_path + "/" + ingest_version
    ):
        os.makedirs(forecast_path + "/" + ingest_version)

    # %% Define base time from the most recent run
    T0 = time.time()

    latest_run = HerbieLatest(
        model="aifs",
        n=3,
        freq="6h",
        fxx=240,
        product="oper",
        verbose=AIFS_VERBOSE_LOGS,
        priority=AIFS_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )

    base_time = latest_run.date
    # Base date for testing
    # base_time = pd.Timestamp("2025-11-05 00:00:00")

    logger.info(base_time)

    # Check if this is newer than the current file
    if save_type == "S3":
        # Check if the file exists and load it
        if s3.exists(forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle"):
            with s3.open(
                forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle", "rb"
            ) as f:
                previous_base_time = pickle.load(f)

            # Compare timestamps and download if the S3 object is more recent
            if previous_base_time >= base_time:
                logger.info("No Update to ECMWF AIFS, ending")
                sys.exit()

    else:
        if os.path.exists(
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle"
        ):
            # Open the file in binary mode
            with open(
                forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle", "rb"
            ) as file:
                # Deserialize and retrieve the variable from the file
                previous_base_time = pickle.load(file)

            # Compare timestamps and download if the S3 object is more recent
            if previous_base_time >= base_time:
                logger.info("No Update to AIFS, ending")
                sys.exit()

    zarr_vars = (
        "time",
        "msl",
        "t2m",
        "d2m",
        "u10",
        "v10",
        "ptype",
        "tcc",
        "Precipitation_Prob",
        "APCP_Mean",
        "APCP_StdDev",
    )

    #####################################################################################################
    # %% Download AIFS data using Herbie Latest
    # Start with the ensemble forecast
    # Find the latest run with 240 hours

    # Create a range of forecast lead times
    aifs_range = FORECAST_LEAD_RANGES["ECMWF_AIFS"]

    match_string_enfo = r":((tp|sf):sfc:\d+):"
    ens_paths = download_aifs_subset(
        product="enfo",
        search=match_string_enfo,
        dataset_name="AIFS ensemble forecast",
        run_date=base_time,
        forecast_hours=aifs_range,
    )

    ens_mf = xr.open_mfdataset(
        ens_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
    ).sortby("step")

    ens_mf["tpd"] = ens_mf["tp"].diff(dim="step")

    # Set the first difference to the first accumulation
    ens_mf["tpd"] = xr.where(
        ens_mf.step == ens_mf.step.isel(step=0),
        ens_mf["tp"].isel(step=0),
        ens_mf["tpd"],
    )

    ens_mf["sfd"] = ens_mf["sf"].diff(dim="step")

    # Set the first difference to the first accumulation
    ens_mf["sfd"] = xr.where(
        ens_mf.step == ens_mf.step.isel(step=0),
        ens_mf["sf"].isel(step=0),
        ens_mf["sfd"],
    )

    # AIFS ensemble tp and sf are in kg m**-2 (= mm), not meters like IFS.
    # Divide by 6000 (6 hours × 1000 mm/m) to convert mm/6h → m/h so that
    # the downstream * 1000 conversion (m/h → mm/h) yields the correct result.
    ens_mf["tpd"] = ens_mf["tpd"] / 6000
    ens_mf["sfd"] = ens_mf["sfd"] / 6000

    # Find the probability of precipitation greater than 0.1 mm/h (0.0001) m/h across all members
    X3_Precipitation_Prob = (ens_mf["tpd"] > 0.0001).sum(dim="number") / ens_mf.sizes[
        "number"
    ]

    # Find the standard deviation of precipitation accumulation across all members
    X3_Precipitation_StdDev = ens_mf["tpd"].std(dim="number")

    # Find the average precipitation accumulation across all members
    X3_Precipitation_Mean = ens_mf["tpd"].mean(dim="number")

    # Find the average snowfall accumulation across all members
    X3_Snowfall_Mean = ens_mf["sfd"].mean(dim="number")

    # Find the type of precipitation based on the snowfall and total precipitation
    # If mean snow is greater than 50% of total precipitation, classify as snow, else rain, 0 if no precipitation
    # Use grib types: https://codes.ecmwf.int/grib/format/grib2/ctables/4/201/, 1 is rain, 5 is snow
    X3_pType = xr.where(
        X3_Precipitation_Mean > 0,
        xr.where(X3_Snowfall_Mean > X3_Precipitation_Mean * 0.5, 5, 1),
        0,
    )

    # Merge into a new xarray dataset
    xr_ensoOut = xr.Dataset(
        {
            "Precipitation_Prob": X3_Precipitation_Prob,
            "APCP_StdDev": X3_Precipitation_StdDev,
            "APCP_Mean": X3_Precipitation_Mean,
            "ptype": X3_pType,
        },
        coords={
            "time": ens_mf["step"],
            "latitude": ens_mf["latitude"],
            "longitude": ens_mf["longitude"],
        },
    )

    #####################################################################################################
    # %% Download Base IFS data using Herbie Latest
    # Find the latest run with 240 hours

    # Define the subset of variables to download as a list of strings
    # 2 m level – use ECMWF’s 2t (temperature) and 2d (dew point)
    matchstring_2m = "(:(2d|2t):)"
    matchstring_10m = "(:(10u|10v):)"
    matchstring_sl = "(:(msl):)"
    matchstring_tcc = "(:(tcc):)"

    # Merge matchstrings for download
    match_strings = (
        matchstring_2m
        + "|"
        + matchstring_10m
        + "|"
        + matchstring_sl
        + "|"
        + matchstring_tcc
    )

    aifs_paths = download_aifs_subset(
        product="oper",
        search=match_strings,
        dataset_name="AIFS deterministic forecast",
        run_date=base_time,
        forecast_hours=aifs_range,
    )

    aifs_mf_2 = xr.open_mfdataset(
        aifs_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [2]}
        },
    ).sortby("step")
    aifs_mf_10 = xr.open_mfdataset(
        aifs_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": [10]}
        },
    ).sortby("step")

    aifs_mf_msl = xr.open_mfdataset(
        aifs_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"}},
    ).sortby("step")

    aifs_mf_atm = xr.open_mfdataset(
        aifs_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "entireAtmosphere"}},
    ).sortby("step")

    # Combine the datasets
    aifs_mf = xr.merge(
        [aifs_mf_2, aifs_mf_10, aifs_mf_msl, aifs_mf_atm], compat="override"
    )

    # %% Merge the ENSO and OPER data

    xarray_forecast_merged = xr.merge(
        [aifs_mf, xr_ensoOut], compat="override", join="outer"
    )

    assert len(xarray_forecast_merged.step) == len(aifs_range), (
        "Incorrect number of timesteps! Exiting"
    )

    # Replace the step variable with a time variable using the start time and the step values
    xarray_forecast_merged = (
        xarray_forecast_merged.assign_coords(
            time=(
                "step",
                pd.to_datetime(xarray_forecast_merged.time.values)
                + pd.to_timedelta(xarray_forecast_merged.step.values, unit="h"),
            )
        )
        .swap_dims({"step": "time"})
        .drop_vars("step")
    )

    ## Create a new time series for the interpolation target
    # Hourly target axis runs from the first stored forecast valid time
    # (base_time + 6 h) to the last one. AIFS output starts at step 6, so
    # hours base_time+1 to +5 are not in this store; the API falls back to
    # other sources for them.
    start = xarray_forecast_merged.time[0].values
    end = xarray_forecast_merged.time[-1].values
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Chunk and save as zarr
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(xarray_forecast_merged.time),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    with aifs_progress_bar():
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_merged.zarr",
            mode="w",
            consolidated=False,
            compute=True,
        )

    # %% Delete to free memory
    del (
        xarray_forecast_merged,
        aifs_mf,
        aifs_mf_2,
        aifs_mf_10,
        aifs_mf_msl,
        aifs_mf_atm,
        ens_mf,
        xr_ensoOut,
    )

    T1 = time.time()
    logger.info(T1 - T0)

    # %% Read the forecast back in and squash using dask
    ds = xr.open_zarr(f"{forecast_process_path}_merged.zarr", consolidated=False)

    # Clip to valid data ranges
    for var in zarr_vars:
        if var in ds.data_vars:
            ds_clip = ds[var]
            if np.issubdtype(ds_clip.dtype, np.number):
                mask = (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
                ds[var] = ds_clip.where(mask)  # out-of-range → NaN

    # Get the actual stacked times from the forecast dataset
    stacked_times = ds.time.values
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second

    # Rename time dimension to match later processing
    ds_rename = ds.rename({"time": "stacked_time"})

    # Add a 3D time array
    time3d = (
        ((ds_rename["stacked_time"] - unix_epoch) / np.timedelta64(1, "s"))
        .astype("float32")  # 1D ('time',)
        .expand_dims(
            latitude=ds_rename.latitude,  # add ('latitude',)
            longitude=ds_rename.longitude,
        )  # add ('longitude',)
        .transpose("stacked_time", "latitude", "longitude")  # order dims
    )

    # Add the time array to the dataset
    ds_rename["time"] = time3d

    # Set the order correctly
    vars_in = [
        v for v in zarr_vars if v in ds_rename.data_vars
    ]  # keep only those that exist
    ds_stack = ds_rename[vars_in].to_array(dim="var", name="var")

    # Rechunk the data to be more manageable for processing
    ds_chunk = ds_stack.chunk(
        {
            "var": 1,
            "stacked_time": len(stacked_timesUnix),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    # Interim zarr save of the stacked array. Not necessary for local, but speeds things up on S3
    with aifs_progress_bar():
        ds_chunk.to_zarr(forecast_process_path + "_stack.zarr", mode="w")

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(
        forecast_process_path + "_stack.zarr", component="__xarray_dataarray_variable__"
    )

    # Create a zarr backed dask array
    if save_type == "S3":
        zarr_store = zarr.storage.ZipStore(
            forecast_process_dir + "/ECMWF_AIFS.zarr.zip", mode="a"
        )
    else:
        zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/ECMWF_AIFS.zarr")

    # Define which variables are integers and need special handling
    int_vars = ["ptype"]
    # Find the index of these variables in the zarr_vars list
    int_var_indices = [i for i, v in enumerate(zarr_vars) if v in int_vars]

    # 1. Interpolate the stacked array to be hourly along the time axis
    # 2. Pad to chunk size
    # 3. Create the zarr array
    # 4. Rechunk it to match the final array
    # 5. Write it out to the zarr array

    with aifs_progress_bar(), dask.config.set(scheduler="threads", num_workers=4):
        # 1. Interpolate the stacked array to be hourly along the time axis
        daskVarArrayStackDiskInterp = interp_time_take_blend(
            daskVarArrayStackDisk,
            stacked_timesUnix=stacked_timesUnix,
            hourly_timesUnix=hourly_timesUnix,
            nearest_vars=int_var_indices,
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
            forecast_process_dir + "/ECMWF_AIFS.zarr.zip",
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.zarr.zip",
        )

        # Write most recent forecast time
        with open(forecast_process_dir + "/ECMWF_AIFS.time.pickle", "wb") as file:
            # Serialize and write the variable to the file
            pickle.dump(base_time, file)

        s3.put_file(
            forecast_process_dir + "/ECMWF_AIFS.time.pickle",
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle",
        )
    else:
        # Write most recent forecast time
        with open(forecast_process_dir + "/ECMWF_AIFS.time.pickle", "wb") as file:
            # Serialize and write the variable to the file
            pickle.dump(base_time, file)

        shutil.move(
            forecast_process_dir + "/ECMWF_AIFS.time.pickle",
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.time.pickle",
        )

        # Copy the zarr file to the final location
        shutil.copytree(
            forecast_process_dir + "/ECMWF_AIFS.zarr",
            forecast_path + "/" + ingest_version + "/ECMWF_AIFS.zarr",
            dirs_exist_ok=True,
        )

    # Clean up
    shutil.rmtree(forecast_process_dir)

    T2 = time.time()
    logger.info(T2 - T0)


if __name__ == "__main__":
    main()
