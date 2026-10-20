# %% Script to test FastHerbie.py to download REPS data
# Alexander Rey, April 2026

# %% Import modules
import logging
import os
import re
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
from dask.diagnostics import ProgressBar
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR, MISSING_DATA
from API.ingest_grib_utils import (
    cat_gribs,
    download_and_validate_subset,
    quote_path,
    run_checked,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    archive_tmp_zarr_and_upload,
    clear_directory,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    download_extract_historic_archive,
    interp_time_map_blocks_nan,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    run_command,
    write_final_zarr,
)

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

from herbie import HerbieLatest  # noqa: E402

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/REPS")
forecast_process_path = forecast_process_dir + "/REPS_Process"
hist_process_path = forecast_process_dir + "/REPS_Historic"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/REPS")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/REPS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
herbie_download_retries = positive_int_env("herbie_download_retries", 5)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)
skip_reps_wgrib2_validation = os.getenv(
    "skip_reps_wgrib2_validation", "true"
).lower() in {"1", "true", "yes", "on"}


# Define the processing and history chunk size
process_chunk = CHUNK_SIZES["REPS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["REPS"]

his_period = HISTORY_PERIODS["REPS"]


# Ensemble statistics output variables (written to zarr, read by the API)
probVars = (
    "time",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
    "AFRAIN_Mean",
    "AICEP_Mean",
    "ARAIN_Mean",
    "ASNOW_Mean",
)

# Minimum precipitation rate (mm/h) used for the precipitation probability calculation
PRECIP_THRESHOLD = 0.1

# Base variable names as produced by wgrib2 for REPS GRIB2 files. The rate
# fields use ECCC's FPRATE/IPRATE/RPRATE/SPRATE parameter names in some builds.
base_var_name_aliases = {
    "APCP": ["APCP_SFC"],
    "AFRAIN": ["AFRAIN_SFC", "FPRATE_surface"],
    "AICEP": ["AICEP_SFC", "IPRATE_surface"],
    "ARAIN": ["ARAIN_SFC", "RPRATE_surface"],
    "ASNOW": ["ASNOW_SFC", "SPRATE_surface"],
}


def find_member_variables(ds, base_name):
    """Return all ensemble member variables for *base_name* found in *ds*.

    wgrib2 names members as VARNAME_LEVEL (control/first), VARNAME_LEVEL.1,
    VARNAME_LEVEL.2, or extended ensemble names sorted in member-number order.
    """
    var_prefix = base_name.split("_", maxsplit=1)[0]
    extended_control_name = f"{var_prefix}_ENS_EQ_lowMres_ctl_surface"
    extended_member_pattern = re.compile(
        rf"^{re.escape(var_prefix)}_MMMENS_EQ_(\d+)_surface$"
    )

    def member_number(v):
        if v == extended_control_name:
            return -1
        extended_match = extended_member_pattern.match(v)
        if extended_match is not None:
            return int(extended_match.group(1))

        suffix = v[len(base_name) + 1 :]
        return int(suffix) if suffix.isdigit() else -1

    return sorted(
        [
            v
            for v in ds.data_vars
            if v == base_name
            or (v.startswith(base_name + ".") and v[len(base_name) + 1 :].isdigit())
            or v == extended_control_name
            or extended_member_pattern.match(v) is not None
        ],
        key=member_number,
    )


def main() -> None:
    """Run the REPS forecast and historic ingest."""
    configure_herbie_request_timeouts()
    logging.basicConfig(level=logging.INFO)
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)
        if not os.path.exists(historic_path):
            os.makedirs(historic_path)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    latest_run = HerbieLatest(
        model="reps",
        priority=["msc"],
        periods=7,
        fxx=72,
        product="10km/grib2",
        variable="APCP",
        level="SFC",
        verbose=False,
        save_dir=herbie_save_dir,
    )

    base_time = latest_run.date

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/REPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to REPS, ending")
        return
    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 72 hours

    # Define the variables to download as a dictionary of variable and level pairs to match in the grib files
    match_strings = [
        {"variable": "AFRAIN", "level": "SFC"},
        {"variable": "AICEP", "level": "SFC"},
        {"variable": "APCP", "level": "SFC"},
        {"variable": "ARAIN", "level": "SFC"},
        {"variable": "ASNOW", "level": "SFC"},
    ]

    reps_file_range = FORECAST_LEAD_RANGES["REPS"]

    # MSC models have each variable in a separate file, so we loop through the variables and levels to download each one and then merge them later
    all_files = []
    expected_total = 0
    for g in match_strings:
        grib_files = download_and_validate_subset(
            model="reps",
            product="10km/grib2",
            search=None,
            dataset_name=f"REPS forecast {g['variable']}:{g['level']}",
            base_time=base_time,
            wgrib2_exe=wgrib2_path.strip(),
            forecast_hours=reps_file_range,
            expected_forecast_hours=reps_file_range,
            skip_wgrib2_validation=skip_reps_wgrib2_validation,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
            herbie_kwargs={
                "variable": g["variable"],
                "level": g["level"],
                "verbose": True,
            },
        )
        all_files += grib_files
        expected_total += len(reps_file_range)

        logger.info(
            "Download completed for REPS forecast %s:%s, %s files downloaded.",
            g["variable"],
            g["level"],
            len(grib_files),
        )

    # Deduplicate and sanity-check total files
    all_files = sorted(set(all_files))
    if len(all_files) < expected_total:
        logger.error(
            f"Download incomplete, expected at least {expected_total} files but got {len(all_files)}"
        )
        sys.exit(1)

    # Create ordered list of downloaded GRIB files from collected paths.
    grib_list = all_files

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-set_ext_name 1 -netcdf {quote_path(forecast_process_path + '_wgrib2_merged.nc')}"
    )

    # Run wgrib2
    run_checked(cmd, "REPS wgrib2 NetCDF conversion")

    # Note: UV (DUVB) is included in the main product for REPS; no separate UV download required

    # %% Merge the UV data and xarrays
    # Read the merged netcdf file using xarray (single combined file)
    xarray_forecast_merged = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged.nc"
    )

    if len(xarray_forecast_merged.time) != len(reps_file_range):
        raise ValueError("Incorrect number of timesteps! Exiting")

    # REPS latitude and longitude are 2-D coordinates over the rotated y/x grid.
    spatial_dims = next(iter(xarray_forecast_merged.data_vars.values())).dims[-2:]
    NY, NX = (xarray_forecast_merged.sizes[dim] for dim in spatial_dims)

    # Create a new time series
    start = xarray_forecast_merged.time.min().values  # Adjust as necessary
    end = xarray_forecast_merged.time.max().values  # Adjust as necessary
    new_hourly_time = pd.date_range(
        start=start - pd.Timedelta(his_period, "h"), end=end, freq="h"
    )

    stacked_times = pd.date_range(
        start=start - pd.Timedelta(his_period, "h"),
        end=end,
        freq="3h",
    )
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Chunk the merged dataset for efficient ensemble processing
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(reps_file_range),
            spatial_dims[0]: process_chunk,
            spatial_dims[1]: process_chunk,
        }
    )

    # Calculate ensemble statistics for every accumulation variable.
    # REPS stores all members in a single file; wgrib2 outputs them as
    # VARNAME_SFC (control), VARNAME_SFC.1, VARNAME_SFC.2, … per time step.
    stats_vars = {}
    for var_prefix, base_var_candidates in base_var_name_aliases.items():
        member_vars = []
        found_base_var = None
        for base_var in base_var_candidates:
            member_vars = find_member_variables(xarray_forecast_merged, base_var)
            if member_vars:
                found_base_var = base_var
                break

        if not member_vars:
            logger.warning("No member variables found for %s, skipping", var_prefix)
            continue

        n_members = len(member_vars)
        logger.info(
            "Processing %s: found %s ensemble members", found_base_var, n_members
        )

        # Stack all members → shape (n_members, n_times, ny, nx)
        raw_stacked = da.stack(
            [xarray_forecast_merged[v].data for v in member_vars],
            axis=0,
        )

        # REPS stores totals since run start; diff converts to per-step accumulation.
        # Prepend zeros so the first step equals its own 3-hour accumulation.
        stacked = da.diff(
            raw_stacked, axis=1, prepend=da.zeros_like(raw_stacked[:, :1, :, :])
        )

        # Divide by 3 to convert 3-hourly accumulation to mm/h
        stacked = stacked / 3

        # Clamp negatives that can arise from the diff at boundaries
        stacked = da.maximum(stacked, 0)

        # Mean across all members for every accumulation variable
        stats_vars[f"{var_prefix}_Mean"] = stacked.mean(axis=0)

        # APCP only: standard deviation and precipitation probability
        if var_prefix == "APCP":
            stats_vars["APCP_StdDev"] = stacked.std(axis=0)
            stats_vars["Precipitation_Prob"] = (stacked > PRECIP_THRESHOLD).sum(
                axis=0
            ) / n_members

    # Build an xarray Dataset from the ensemble statistics
    stats_ds = xr.Dataset(
        {
            key: xr.DataArray(val, dims=("time", *spatial_dims))
            for key, val in stats_vars.items()
        },
        coords=xarray_forecast_merged.coords,
    )

    stats_ds = stats_ds.chunk(
        chunks={
            "time": len(reps_file_range),
            spatial_dims[0]: process_chunk,
            spatial_dims[1]: process_chunk,
        }
    )

    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        stats_ds.to_zarr(
            forecast_process_path + "_.zarr",
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # %% Delete to free memory
    del xarray_forecast_merged, stats_vars, stats_ds
    T1 = time.time()

    logger.info(T1 - T0)
    os.remove(forecast_process_path + "_wgrib2_merged.nc")

    ################################################################################################
    # %% Historic data
    # Loop through the runs and check if they have already been processed to s3

    # 6 hour runs
    for i in range(his_period, 0, -6):
        if save_type == "S3":
            s3_path = (
                historic_path
                + "/REPS_Hist_v3"
                + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
                + ".zarr.tar.gz"
            )

            # Check for a done file in S3
            if s3.exists(s3_path.replace(".tar.gz", ".done")):
                logger.info(
                    "File already exists in S3, skipping download for: %s", s3_path
                )
                continue
        else:
            # Local Path Setup
            local_path = (
                historic_path
                + "/REPS_Hist_v3"
                + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
                + ".zarr"
            )

            # Check for a local done file
            if os.path.exists(local_path.replace(".zarr", ".done")):
                logger.info(
                    "File already exists locally, skipping download for: %s", local_path
                )
                continue

        logger.info(
            "Downloading: %s",
            (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"),
        )

        hist_run_date = base_time - pd.Timedelta(hours=i)
        historic_forecast_hours = [3, 6]
        all_files = []
        expected_total = 0

        for g in match_strings:
            grib_files = download_and_validate_subset(
                model="reps",
                product="10km/grib2",
                search=None,
                dataset_name=f"REPS historic {g['variable']}:{g['level']}",
                base_time=hist_run_date,
                wgrib2_exe=wgrib2_path.strip(),
                forecast_hours=historic_forecast_hours,
                expected_forecast_hours=historic_forecast_hours,
                skip_wgrib2_validation=skip_reps_wgrib2_validation,
                herbie_save_dir=herbie_save_dir,
                herbie_download_retries=herbie_download_retries,
                herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
                herbie_kwargs={
                    "variable": g["variable"],
                    "level": g["level"],
                    "verbose": True,
                },
            )
            all_files += grib_files
            expected_total += len(historic_forecast_hours)

            logger.info(
                "Download completed for REPS historic %s:%s, %s files downloaded.",
                g["variable"],
                g["level"],
                len(grib_files),
            )

        grib_list = sorted(set(all_files))
        if len(grib_list) < expected_total:
            logger.error(
                "Historic download incomplete, expected at least %s files but got %s",
                expected_total,
                len(grib_list),
            )
            sys.exit(1)

        # Create a string to pass to wgrib2 to merge all gribs into one netcdf
        cmd = (
            f"{cat_gribs(grib_list)} | "
            f"{quote_path(wgrib2_path.strip())} - "
            f"-set_ext_name 1 -netcdf {quote_path(hist_process_path + '_wgrib2_merged.nc')}"
        )

        # Run wgrib2
        sp_out = run_command(cmd)
        if sp_out.returncode != 0:
            logger.error(sp_out.stderr)
            sys.exit(1)

        # Read the merged netcdf file using xarray (single combined file)
        xarray_hist_merged = xr.open_dataset(hist_process_path + "_wgrib2_merged.nc")
        hist_spatial_dims = next(iter(xarray_hist_merged.data_vars.values())).dims[-2:]

        # Chunk for efficient ensemble processing
        xarray_hist_merged = xarray_hist_merged.chunk(
            chunks={
                "time": 6,
                hist_spatial_dims[0]: process_chunk,
                hist_spatial_dims[1]: process_chunk,
            }
        )

        # Calculate ensemble statistics for historic data (same approach as forecast)
        hist_stats_vars = {}
        for var_prefix, base_var_candidates in base_var_name_aliases.items():
            member_vars = []
            for base_var in base_var_candidates:
                member_vars = find_member_variables(xarray_hist_merged, base_var)
                if member_vars:
                    break

            if not member_vars:
                continue

            n_members = len(member_vars)

            raw_stacked = da.stack(
                [xarray_hist_merged[v].data for v in member_vars],
                axis=0,
            )

            stacked = da.diff(
                raw_stacked, axis=1, prepend=da.zeros_like(raw_stacked[:, :1, :, :])
            )
            stacked = stacked / 3
            stacked = da.maximum(stacked, 0)

            hist_stats_vars[f"{var_prefix}_Mean"] = stacked.mean(axis=0)

            if var_prefix == "APCP":
                hist_stats_vars["APCP_StdDev"] = stacked.std(axis=0)
                hist_stats_vars["Precipitation_Prob"] = (
                    stacked > PRECIP_THRESHOLD
                ).sum(axis=0) / n_members

        hist_stats_ds = xr.Dataset(
            {
                key: xr.DataArray(val, dims=("time", *hist_spatial_dims))
                for key, val in hist_stats_vars.items()
            },
            coords=xarray_hist_merged.coords,
        )

        hist_stats_ds = hist_stats_ds.chunk(
            chunks={
                "time": 6,
                hist_spatial_dims[0]: process_chunk,
                hist_spatial_dims[1]: process_chunk,
            }
        )

        encoding = {
            vname: {"chunks": (6, process_chunk, process_chunk)}
            for vname in probVars[1:]
        }

        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            hist_stats_ds.to_zarr(
                hist_process_path + "_REPS_Hist_TMP.zarr",
                mode="w",
                consolidated=False,
                encoding=encoding,
                compute=True,
                chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
            )

        # Clear the xarray dataset from memory
        del xarray_hist_merged, hist_stats_vars, hist_stats_ds

        # Remove temp file created by wgrib2
        os.remove(hist_process_path + "_wgrib2_merged.nc")

        # Save a done file to s3 to indicate that the historic data has been processed
        if save_type == "S3":
            archive_tmp_zarr_and_upload(
                tmp_zarr_path=hist_process_path + "_REPS_Hist_TMP.zarr",
                s3_path=s3_path,
                archive_member_name="REPS_Hist.zarr",
                s3=s3,
            )
        else:
            # Move to Local Path
            os.rename(hist_process_path + "_REPS_Hist_TMP.zarr", local_path)

            done_file = local_path.replace(".zarr", ".done")
            with open(done_file, "w") as f:
                f.write("Done")

        logger.info((base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ"))

    # %% Merge the historic and forecast datasets and then squash using dask
    # Get the s3 paths to the historic data
    if save_type == "S3":
        local_temp_dir = forecast_process_path + "_s3_temp_downloads"
        os.makedirs(local_temp_dir, exist_ok=True)

        # The function that downloads and extracts a single timestamp
        def download_and_extract(timestamp):
            # Names expected locally
            final_zarr_name = f"REPS_Hist_v3{timestamp}.zarr"
            extracted_path = download_extract_historic_archive(
                s3=s3,
                historic_path=historic_path,
                final_zarr_name=final_zarr_name,
                extracted_store_name="REPS_Hist.zarr",
                local_temp_dir=local_temp_dir,
            )
            if extracted_path is None:
                tqdm.write(
                    f"Error: REPS_Hist.zarr not found inside archive for {timestamp}"
                )
            return extracted_path

        # Generate target timestamps
        timestamps = [
            (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            for i in range(his_period, 1, -6)
        ]

        logger.info(
            "Phase 1: Downloading and extracting %s archives...", len(timestamps)
        )

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
            + "/REPS_Hist_v3"
            + (base_time - pd.Timedelta(hours=i)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
            for i in range(his_period, 1, -6)
        ]

    # Dask Setup
    daskVarArrays = []
    daskVarArrayList = []

    for daskVarIDX, dask_var in enumerate(probVars[:]):
        for local_ncpath in ncLocalWorking_paths:
            # If not found in array, use MISSING_DATA to show missing
            try:
                daskVarArrays.append(
                    da.from_zarr(local_ncpath, component=dask_var, inline_array=True)
                )
            # Add a fallback in case of a FileNotFoundError
            except FileNotFoundError:
                logger.warning("File not found, adding NaN array for: %s", local_ncpath)
                daskVarArrays.append(
                    da.full((6, NY, NX), MISSING_DATA).rechunk(
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
            daskArrayOut = da.concatenate(
                (daskVarArraysShape, daskForecastArray), axis=0
            )

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
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    reps_store_name = "REPS.zarr.zip" if save_type == "S3" else "REPS.zarr"

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            # 1. Interpolate the stacked array to be hourly along the time axis
            daskVarArrayStackDiskInterp = interp_time_map_blocks_nan(
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

        # 3. Write the final store in source-aligned tiles
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + reps_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [reps_store_name],
        base_time,
        "REPS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Timing
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
