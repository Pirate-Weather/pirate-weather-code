"""REPS ensemble precipitation ingest.

Downloads the precipitation fields of every member of the newest MSC Regional
Ensemble Prediction System run that reaches 72 h (10 km rotated
latitude/longitude grid over North America; 3-hourly leads to 72 h, 6-hourly
cycles) from the MSC Datamart, reduces them to ensemble statistics
(precipitation probability, mean and standard deviation, and the mean of each
precipitation type), interpolates them to an hourly axis and publishes
``REPS.zarr`` plus ``REPS.time.pickle``. The store starts at the first
forecast lead (run + 3 h).

Alexander Rey, April 2026
"""

import logging
import os
import re
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dotenv import find_dotenv, load_dotenv

from API.constants.shared_const import (
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    INGEST_VERSION_STR,
)
from API.ingest_grib_utils import (
    cat_gribs,
    download_and_validate_subset,
    quote_path,
    run_checked,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    DOWNLOAD_CACHE_MAX_AGE_HOURS,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    broadcast_times_to_grid,
    clear_directory,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    interp_time_take_blend,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    prune_old_files,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

# Load .env before importing Herbie, which reads its configuration from the
# environment at import time
dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

from herbie import HerbieLatest  # noqa: E402

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# 3-hourly leads 3-72 h
REPS_FORECAST_HOURS = FORECAST_LEAD_RANGES["REPS"]
REPS_LEAD_STEP_HOURS = 3
REPS_PRODUCT = "10km/grib2"
# HerbieLatest steps back 6 hours per period; 7 periods covers the last
# 42 hours.
REPS_LATEST_RUN_SEARCH_PERIODS = 7
# The run is complete once the last lead of precipitation is on the Datamart
REPS_LATEST_RUN_PROBE = {"variable": "APCP", "level": "SFC"}

# MSC publishes one GRIB file per variable and lead holding every member. Each
# entry is the (Datamart variable, level) pair of one downloaded field: total
# precipitation and the freezing rain, ice pellet, rain and snow accumulations.
REPS_MATCH_STRINGS = [
    ("AFRAIN", "SFC"),
    ("AICEP", "SFC"),
    ("APCP", "SFC"),
    ("ARAIN", "SFC"),
    ("ASNOW", "SFC"),
]

# wgrib2 variable names for each field. Some wgrib2 builds use ECCC's
# FPRATE/IPRATE/RPRATE/SPRATE parameter names for the precipitation types.
base_var_name_aliases = {
    "APCP": ["APCP_SFC"],
    "AFRAIN": ["AFRAIN_SFC", "FPRATE_surface"],
    "AICEP": ["AICEP_SFC", "IPRATE_surface"],
    "ARAIN": ["ARAIN_SFC", "RPRATE_surface"],
    "ASNOW": ["ASNOW_SFC", "SPRATE_surface"],
}

# Variables in the published store; the order is the API's REPS index map
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

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# When running the docker container, this should be "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/REPS")
forecast_process_path = forecast_process_dir + "/REPS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/REPS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

REPS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
REPS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)
# The wgrib2 stats check is on by default; set this only to bypass it
skip_reps_wgrib2_validation = os.getenv(
    "skip_reps_wgrib2_validation", "false"
).lower() in {"1", "true", "yes", "on"}


def download_reps_variable(
    run_time: pd.Timestamp,
    variable: str,
    level: str,
    forecast_hours: list[int],
    dataset_label: str,
    herbie_save_dir: str,
) -> list[str]:
    """Download and validate the given leads of one REPS field with bounded retries."""
    return download_and_validate_subset(
        model="reps",
        product=REPS_PRODUCT,
        search=None,
        dataset_name=f"REPS {dataset_label} {variable}:{level}",
        base_time=run_time,
        forecast_hours=forecast_hours,
        skip_wgrib2_validation=skip_reps_wgrib2_validation,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=REPS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=REPS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"variable": variable, "level": level},
        wgrib2_exe=wgrib2_path.strip(),
    )


def grib_lead_hour(path: str) -> int:
    """Return the forecast lead of an MSC REPS file (``..._PT003H.grib2``)."""
    match = re.search(r"_PT(\d+)H\.grib2$", os.path.basename(path))
    if match is None:
        raise ValueError(f"No forecast lead in REPS GRIB file name {path}")
    return int(match.group(1))


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


def write_ensemble_stats(
    xarray_merged: xr.Dataset,
    forecast_hours: list[int],
    zarr_path: str,
    process_chunk: int,
    zarr_store_workers: int,
) -> None:
    """Reduce the converted members to the ensemble statistics and write them."""
    spatial_dims = next(iter(xarray_merged.data_vars.values())).dims[-2:]
    xarray_merged = xarray_merged.chunk(
        chunks={
            "time": len(forecast_hours),
            spatial_dims[0]: process_chunk,
            spatial_dims[1]: process_chunk,
        }
    )

    stats_vars = {}
    member_count = None
    for var_prefix, base_var_candidates in base_var_name_aliases.items():
        member_vars = []
        found_base_var = None
        for base_var in base_var_candidates:
            member_vars = find_member_variables(xarray_merged, base_var)
            if member_vars:
                found_base_var = base_var
                break

        if not member_vars:
            raise RuntimeError(
                f"No ensemble member variables found for required REPS variable "
                f"{var_prefix}; tried aliases: {', '.join(base_var_candidates)}"
            )
        # Every field comes from the same members; a shorter list means a member
        # is missing from the downloaded files.
        if member_count is not None and len(member_vars) != member_count:
            raise RuntimeError(
                f"REPS {found_base_var} has {len(member_vars)} members, "
                f"expected {member_count} like the other fields"
            )
        member_count = len(member_vars)
        logger.info(
            "Processing %s: found %s ensemble members", found_base_var, member_count
        )

        # Stack all members -> (member, time, y, x)
        raw_stacked = da.stack([xarray_merged[v].data for v in member_vars], axis=0)

        # REPS stores totals since the run start and every lead is 3 h after
        # the previous one, starting at 3 h. Difference
        # consecutive leads and divide by 3 h to get nonnegative hourly rates.
        # This equals deaccumulate_to_hourly_rate, but stays in float32: the
        # helper works in float64, which changes the ensemble mean and spread
        # in the last bit.
        stacked = da.diff(
            raw_stacked, axis=1, prepend=da.zeros_like(raw_stacked[:, :1, :, :])
        )
        stacked = da.maximum(stacked / REPS_LEAD_STEP_HOURS, 0)

        stats_vars[f"{var_prefix}_Mean"] = stacked.mean(axis=0)

        # Spread and probability only for total precipitation
        if var_prefix == "APCP":
            stats_vars["APCP_StdDev"] = stacked.std(axis=0)
            stats_vars["Precipitation_Prob"] = (
                stacked > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H
            ).sum(axis=0) / member_count

    stats_ds = xr.Dataset(
        {
            key: xr.DataArray(val, dims=("time", *spatial_dims))
            for key, val in stats_vars.items()
        },
        coords=xarray_merged.coords,
    ).chunk(
        chunks={
            "time": len(forecast_hours),
            spatial_dims[0]: process_chunk,
            spatial_dims[1]: process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        stats_ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )


def convert_reps_run(
    run_time: pd.Timestamp,
    forecast_hours: list[int],
    dataset_label: str,
    process_path: str,
    herbie_save_dir: str,
) -> tuple[xr.Dataset, str]:
    """Download one REPS run's leads and convert them to a validated NetCDF.

    Returns the opened dataset and the NetCDF path, which the caller removes.
    """
    grib_list = []
    for variable, level in REPS_MATCH_STRINGS:
        grib_files = download_reps_variable(
            run_time, variable, level, forecast_hours, dataset_label, herbie_save_dir
        )
        grib_list += grib_files
        logger.info(
            "Download completed for REPS %s %s:%s, %s files downloaded.",
            dataset_label,
            variable,
            level,
            len(grib_files),
        )

    # Feed wgrib2 the files in lead order so every message arrives in valid-time
    # order; the path breaks ties so the merge is deterministic.
    grib_list = sorted(grib_list, key=lambda path: (grib_lead_hour(path), path))

    # -set_ext_name gives each member its own variable name (see
    # find_member_variables) instead of overwriting one another
    merged_nc_path = process_path + "_wgrib2_merged.nc"
    run_checked(
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-set_ext_name 1 -netcdf {quote_path(merged_nc_path)}",
        f"REPS {dataset_label} run {run_time} wgrib2 NetCDF conversion",
    )
    xarray_merged = xr.open_dataset(merged_nc_path)

    expected_valid_times = (
        pd.Timestamp(run_time) + pd.to_timedelta(forecast_hours, unit="h")
    ).values
    if not np.array_equal(xarray_merged["time"].values, expected_valid_times):
        raise RuntimeError(
            f"REPS {dataset_label} run {run_time} times "
            f"{xarray_merged['time'].values} do not match the expected leads "
            f"{forecast_hours}"
        )
    return xarray_merged, merged_nc_path


def main() -> None:
    """Run the REPS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["REPS"]
    final_chunk = FINAL_CHUNK_SIZES["REPS"]

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    herbie_save_dir = make_herbie_save_dir(tmp_dir)
    prune_old_files(herbie_save_dir, DOWNLOAD_CACHE_MAX_AGE_HOURS)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 72 h lead exists
    latest_run = HerbieLatest(
        model="reps",
        priority=["msc"],
        periods=REPS_LATEST_RUN_SEARCH_PERIODS,
        fxx=REPS_FORECAST_HOURS[-1],
        product=REPS_PRODUCT,
        verbose=False,
        save_dir=herbie_save_dir,
        **REPS_LATEST_RUN_PROBE,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/REPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to REPS, ending")
        return
    logger.info("Processing REPS run %s", base_time)

    # %% Forecast hours: 3-hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(REPS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download, convert and reduce the forecast run
    xarray_forecast_merged, merged_nc_path = convert_reps_run(
        base_time,
        REPS_FORECAST_HOURS,
        "forecast",
        forecast_process_path,
        herbie_save_dir,
    )

    # REPS is on a rotated latitude/longitude grid: latitude and longitude are
    # 2-D coordinates over the y/x dimensions, so the grid size comes from a
    # field's dimensions (either coordinate's size would be NY * NX).
    spatial_dims = next(iter(xarray_forecast_merged.data_vars.values())).dims[-2:]
    NY, NX = (xarray_forecast_merged.sizes[dim] for dim in spatial_dims)

    write_ensemble_stats(
        xarray_forecast_merged,
        REPS_FORECAST_HOURS,
        forecast_process_path + "_.zarr",
        process_chunk,
        zarr_store_workers,
    )
    xarray_forecast_merged.close()

    # The merged NetCDF is large and no longer needed
    os.remove(merged_nc_path)

    # %% Stack the forecast into the (variable, time, y, x) layout the API reads
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    # Hourly target axis from the first to the last forecast lead
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0], end=forecast_valid_times[-1], freq="h"
    )
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    daskVarArrayList = []
    for dask_var in probVars:
        daskCatArray = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            npCatTimes = daskCatArray.astype("float32").compute()
            # Check that the stored times are aligned with the expected times
            validate_stacked_time_alignment(stacked_timesUnix, npCatTimes)
            # The API reads time from the store like any other variable
            daskVarArrayList.append(
                broadcast_times_to_grid(npCatTimes, NY, NX, process_chunk)
            )
        else:
            daskVarArrayList.append(
                daskCatArray.rechunk(
                    (len(stacked_timesUnix), process_chunk, process_chunk)
                ).astype("float32")
            )

    daskVarArrayListMergeNaN = mask_invalid_data(da.stack(daskVarArrayList, axis=0))

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr", overwrite=True, compute=True
        )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Interpolate to hourly and write the final store
    daskVarArrayStackDiskInterp = interp_time_take_blend(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    reps_store_name = "REPS.zarr.zip" if save_type == "S3" else "REPS.zarr"
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

    # %% Clean up, keeping the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    logger.info("REPS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
