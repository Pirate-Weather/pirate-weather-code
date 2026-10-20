# %% Script to download GEPS data
# Alexander Rey, April 2026

# %% Import modules
import logging
import os
import re
import sys
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from API.constants.shared_const import INGEST_VERSION_STR
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
    clear_directory,
    close_store,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_to_hourly_rate,
    interp_time_map_blocks_nan,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
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

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GEPS")
forecast_process_path = forecast_process_dir + "/GEPS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GEPS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
herbie_download_retries = positive_int_env("herbie_download_retries", 5)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)
skip_geps_wgrib2_validation = os.getenv(
    "skip_geps_wgrib2_validation", "false"
).lower() in {"1", "true", "yes", "on"}


def forecast_hour_sort_key(path: str) -> tuple[int, str]:
    """Sort MSC GRIB files by forecast lead hour, then path for deterministic merges."""
    match = re.search(r"_PT(\d+)H", str(path))
    if match is None:
        return (0, str(path))
    return (int(match.group(1)), str(path))


# Define the processing chunk size
process_chunk = CHUNK_SIZES["GEPS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GEPS"]


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

# Base variable names as produced by wgrib2 for GEPS GRIB2 files.
# GEPS stores all ensemble members in a single file; wgrib2 names them
# VARNAME_SFC (first/control member), VARNAME_SFC.1, VARNAME_SFC.2, etc.
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
    """Run the GEPS forecast ingest."""
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

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    latest_run = HerbieLatest(
        model="geps",
        priority=["msc"],
        periods=7,
        fxx=240,
        product="geps-raw",
        variable="APCP",
        level="SFC_0",
        verbose=False,
        save_dir=herbie_save_dir,
    )

    base_time = latest_run.date

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/GEPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to GEPS, ending")
        return
    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 240 hours

    # Define the variables to download as a dictionary of variable and level pairs to match in the grib files
    match_strings = [
        {"variable": "AFRAIN", "level": "SFC_0"},
        {"variable": "AICEP", "level": "SFC_0"},
        {"variable": "APCP", "level": "SFC_0"},
        {"variable": "ARAIN", "level": "SFC_0"},
        {"variable": "ASNOW", "level": "SFC_0"},
    ]

    geps_file_range = FORECAST_LEAD_RANGES["GEPS"]

    # MSC models have each variable in a separate file, so we loop through the variables and levels to download each one and then merge them later
    all_files = []
    expected_total = 0
    for g in match_strings:
        grib_files = download_and_validate_subset(
            model="geps",
            product="geps-raw",
            search=None,
            dataset_name=f"GEPS forecast {g['variable']}:{g['level']}",
            base_time=base_time,
            wgrib2_exe=wgrib2_path.strip(),
            forecast_hours=geps_file_range,
            priority=["msc"],
            skip_wgrib2_validation=skip_geps_wgrib2_validation,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
            herbie_kwargs={
                "variable": g["variable"],
                "level": g["level"],
                "verbose": False,
            },
        )

        all_files += grib_files
        expected_total += len(geps_file_range)

        logger.info(
            f"Download completed for GEPS forecast {g['variable']}:{g['level']}, "
            f"{len(grib_files)} files downloaded."
        )

    # Deduplicate and sanity-check total files
    all_files = sorted(set(all_files))
    if len(all_files) < expected_total:
        logger.error(
            f"Download incomplete, expected at least {expected_total} files but got {len(all_files)}"
        )
        sys.exit(1)

    # Create ordered list of downloaded grib files from collected paths
    grib_list = sorted(all_files, key=forecast_hour_sort_key)

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-set_ext_name 1 -netcdf {quote_path(forecast_process_path + '_wgrib2_merged.nc')}"
    )

    # Run wgrib2
    run_checked(cmd, "GEPS wgrib2 NetCDF conversion")

    # Read the merged netcdf file using xarray (single combined file)
    xarray_forecast_merged = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged.nc"
    )

    if len(xarray_forecast_merged.time) != len(geps_file_range):
        raise ValueError("Incorrect number of timesteps! Exiting")

    # Determine grid size from merged dataset (supports rotated grids)
    NY = xarray_forecast_merged.dims.get(
        "latitude", xarray_forecast_merged["latitude"].size
    )
    NX = xarray_forecast_merged.dims.get(
        "longitude", xarray_forecast_merged["longitude"].size
    )

    # Create a new hourly time series from the first to the last forecast valid
    # time; the 3- and 6-hourly steps are interpolated onto it below.
    start = xarray_forecast_merged.time.min().values
    end = xarray_forecast_merged.time.max().values
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    # Use the exact forecast valid times as the interpolation source axis. The
    # stored time variable is cast to float32 (~1 minute resolution), which could
    # put the first or last hourly target just outside the source range.
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    source_timesUnix = (xarray_forecast_merged.time.values - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Chunk the merged dataset for efficient ensemble processing
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(geps_file_range),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    # Calculate ensemble statistics for every accumulation variable.
    # GEPS stores all members in a single file; wgrib2 outputs them as
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
            raise RuntimeError(
                f"No ensemble member variables found for required GEPS variable "
                f"{var_prefix}; tried aliases: {', '.join(base_var_candidates)}"
            )

        n_members = len(member_vars)
        logger.info(
            "Processing %s: found %s ensemble members", found_base_var, n_members
        )

        # Stack all members → shape (n_members, n_times, ny, nx)
        raw_stacked = da.stack(
            [xarray_forecast_merged[v].data for v in member_vars],
            axis=0,
        )

        # GEPS stores totals since run start. Normalize each difference by its
        # corresponding lead interval (3 hours through 192, then 6 hours).
        stacked = deaccumulate_to_hourly_rate(
            raw_stacked,
            geps_file_range,
            time_axis=1,
        )

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
            key: xr.DataArray(val, dims=["time", "latitude", "longitude"])
            for key, val in stats_vars.items()
        },
        coords={
            "time": xarray_forecast_merged["time"],
            "latitude": xarray_forecast_merged["latitude"],
            "longitude": xarray_forecast_merged["longitude"],
        },
    )

    stats_ds = stats_ds.chunk(
        chunks={
            "time": len(geps_file_range),
            "latitude": process_chunk,
            "longitude": process_chunk,
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

    # %% Stack the ensemble statistics into a single 4D array using dask
    daskVarArrayList = []
    source_time_count = len(source_timesUnix)

    for dask_var in probVars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            # Broadcast the 1D time axis over the grid so it is stored like a variable
            npForecastTimes = daskForecastArray.astype("float32").compute()

            daskArrayOut = da.from_array(
                np.tile(
                    np.expand_dims(np.expand_dims(npForecastTimes, axis=1), axis=1),
                    (1, NY, NX),
                )
            ).rechunk((source_time_count, process_chunk, process_chunk))

            daskVarArrayList.append(daskArrayOut)

        else:
            daskVarArrayList.append(
                daskForecastArray[:, :, :]
                .rechunk((source_time_count, process_chunk, process_chunk))
                .astype("float32")
            )

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

    # Create a zarr backed dask array
    geps_store_name = "GEPS.zarr.zip" if save_type == "S3" else "GEPS.zarr"
    if save_type == "S3":
        zarr_store = zarr.storage.ZipStore(
            forecast_process_dir + "/" + geps_store_name, mode="a", compression=0
        )
    else:
        zarr_store = zarr.storage.LocalStore(
            forecast_process_dir + "/" + geps_store_name
        )

    #
    # 1. Interpolate the stacked array to be hourly along the time axis
    # 2. Pad to chunk size
    # 3. Create the zarr array
    # 4. Rechunk it to match the final array
    # 5. Write it out to the zarr array

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            # 1. Interpolate the stacked array to be hourly along the time axis
            daskVarArrayStackDiskInterp = interp_time_map_blocks_nan(
                daskVarArrayStackDisk,
                stacked_timesUnix=source_timesUnix,
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
            overwrite=True,
        )

        # 4. Write source-aligned spatial tiles. A single whole-array rechunk to the
        # final 3x3 spatial chunks builds a very large store graph and retains too
        # much memory during execution.
        source_y = daskVarArrayStackDiskInterp.shape[2]
        source_x = daskVarArrayStackDiskInterp.shape[3]
        target_y = zarr_array.shape[2]
        target_x = zarr_array.shape[3]
        y_starts = range(0, target_y, process_chunk)
        x_starts = range(0, target_x, process_chunk)

        for y_start in tqdm(y_starts, desc="Writing GEPS.zarr rows"):
            y_stop = min(y_start + process_chunk, target_y)
            y_source_stop = min(y_stop, source_y)
            for x_start in x_starts:
                x_stop = min(x_start + process_chunk, target_x)
                x_source_stop = min(x_stop, source_x)

                with dask.config.set(
                    scheduler="threads", num_workers=zarr_store_workers
                ):
                    tile = daskVarArrayStackDiskInterp[
                        :, :, y_start:y_source_stop, x_start:x_source_stop
                    ].compute()

                tile = np.round(tile, 5).astype("float32", copy=False)
                if (
                    tile.shape[2] != y_stop - y_start
                    or tile.shape[3] != x_stop - x_start
                ):
                    padded_tile = np.full(
                        (
                            tile.shape[0],
                            tile.shape[1],
                            y_stop - y_start,
                            x_stop - x_start,
                        ),
                        np.nan,
                        dtype="float32",
                    )
                    padded_tile[:, :, : tile.shape[2], : tile.shape[3]] = tile
                    tile = padded_tile

                zarr_array[:, :, y_start:y_stop, x_start:x_stop] = tile

    close_store(zarr_store)

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [geps_store_name],
        base_time,
        "GEPS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    # shutil.rmtree(forecast_process_dir)
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Timing
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
