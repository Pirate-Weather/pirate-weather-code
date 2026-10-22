"""GEPS ensemble precipitation ingest.

Downloads the precipitation fields of every member of the newest MSC Global
Ensemble Prediction System run that reaches 240 h (0.5 degree global
latitude/longitude grid; 3-hourly leads to 192 h, then 6-hourly to 240 h) from
the MSC Datamart, reduces them to ensemble statistics (precipitation
probability, mean and standard deviation, and the mean of each precipitation
type), interpolates them to an hourly axis and publishes ``GEPS.zarr`` plus
``GEPS.time.pickle``. The store starts at the first forecast lead (run + 3 h).

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
    deaccumulate_to_hourly_rate,
    interp_time_take_blend,
    make_herbie_save_dir,
    mask_invalid_data,
    open_remote,
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
# 3-hourly leads through 192 h, then 6-hourly through 240 h
GEPS_FORECAST_HOURS = FORECAST_LEAD_RANGES["GEPS"]
# Raw member output (not the Datamart's pre-computed ensemble products)
GEPS_PRODUCT = "geps-raw"
# HerbieLatest steps back 6 hours per period; 7 periods covers the last
# 42 hours, enough to always reach a run with a 240 h lead.
GEPS_LATEST_RUN_SEARCH_PERIODS = 7
# The run is complete once the last lead of precipitation is on the Datamart
GEPS_LATEST_RUN_PROBE = {"variable": "APCP", "level": "SFC_0"}

# MSC publishes one GRIB file per variable and lead holding every member. Each
# entry is the (Datamart variable, level) pair of one downloaded field: total
# precipitation and the freezing rain, ice pellet, rain and snow accumulations.
GEPS_MATCH_STRINGS = [
    ("AFRAIN", "SFC_0"),
    ("AICEP", "SFC_0"),
    ("APCP", "SFC_0"),
    ("ARAIN", "SFC_0"),
    ("ASNOW", "SFC_0"),
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

# Variables in the published store; the order is the API's GEPS index map
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

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GEPS")
forecast_process_path = forecast_process_dir + "/GEPS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GEPS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

GEPS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
GEPS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)
# The wgrib2 stats check is on by default; set this only to bypass it
skip_geps_wgrib2_validation = os.getenv(
    "skip_geps_wgrib2_validation", "false"
).lower() in {"1", "true", "yes", "on"}


def download_geps_variable(
    base_time: pd.Timestamp, variable: str, level: str, herbie_save_dir: str
) -> list[str]:
    """Download and validate every lead of one GEPS field with bounded retries."""
    return download_and_validate_subset(
        model="geps",
        product=GEPS_PRODUCT,
        search=None,
        dataset_name=f"GEPS forecast {variable}:{level}",
        base_time=base_time,
        forecast_hours=GEPS_FORECAST_HOURS,
        priority=["msc"],
        skip_wgrib2_validation=skip_geps_wgrib2_validation,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=GEPS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=GEPS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"variable": variable, "level": level},
        wgrib2_exe=wgrib2_path.strip(),
    )


def grib_lead_hour(path: str) -> int:
    """Return the forecast lead of an MSC GEPS file (``..._P003_allmbrs.grib2``)."""
    match = re.search(r"_P(\d+)_allmbrs\.grib2$", os.path.basename(path))
    if match is None:
        raise ValueError(f"No forecast lead in GEPS GRIB file name {path}")
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
                f"No ensemble member variables found for required GEPS variable "
                f"{var_prefix}; tried aliases: {', '.join(base_var_candidates)}"
            )
        # Every field comes from the same members; a shorter list means a member
        # is missing from the downloaded files.
        if member_count is not None and len(member_vars) != member_count:
            raise RuntimeError(
                f"GEPS {found_base_var} has {len(member_vars)} members, "
                f"expected {member_count} like the other fields"
            )
        member_count = len(member_vars)
        logger.info(
            "Processing %s: found %s ensemble members", found_base_var, member_count
        )

        # Stack all members -> (member, time, y, x)
        raw_stacked = da.stack([xarray_merged[v].data for v in member_vars], axis=0)

        # GEPS stores totals since the run start. Difference consecutive leads
        # and divide by each lead interval (3 h through 192 h, then 6 h) to get
        # nonnegative hourly rates.
        stacked = deaccumulate_to_hourly_rate(raw_stacked, forecast_hours, time_axis=1)

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


def main() -> None:
    """Run the GEPS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["GEPS"]
    final_chunk = FINAL_CHUNK_SIZES["GEPS"]

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    herbie_save_dir = make_herbie_save_dir(tmp_dir)
    prune_old_files(herbie_save_dir, DOWNLOAD_CACHE_MAX_AGE_HOURS)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="geps",
        priority=["msc"],
        periods=GEPS_LATEST_RUN_SEARCH_PERIODS,
        fxx=GEPS_FORECAST_HOURS[-1],
        product=GEPS_PRODUCT,
        verbose=False,
        save_dir=herbie_save_dir,
        **GEPS_LATEST_RUN_PROBE,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "GEPS.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to GEPS, ending")
        return
    logger.info("Processing GEPS run %s", base_time)

    # %% Forecast hours: 3- then 6-hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(GEPS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download each field, then convert all of them to one NetCDF
    grib_list = []
    for variable, level in GEPS_MATCH_STRINGS:
        grib_files = download_geps_variable(base_time, variable, level, herbie_save_dir)
        grib_list += grib_files
        logger.info(
            "Download completed for GEPS forecast %s:%s, %s files downloaded.",
            variable,
            level,
            len(grib_files),
        )

    # Feed wgrib2 the files in lead order so every message arrives in valid-time
    # order; the path breaks ties so the merge is deterministic.
    grib_list = sorted(grib_list, key=lambda path: (grib_lead_hour(path), path))

    # -set_ext_name gives each member its own variable name (see
    # find_member_variables) instead of overwriting one another
    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-set_ext_name 1 -netcdf {quote_path(merged_nc_path)}",
        "GEPS wgrib2 NetCDF conversion",
    )
    xarray_forecast_merged = xr.open_dataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"GEPS run {base_time} times {xarray_forecast_merged['time'].values} "
            f"do not match the expected forecast leads {GEPS_FORECAST_HOURS}"
        )

    # GEPS is on a regular latitude/longitude grid, so the spatial dimensions
    # are (latitude, longitude); read them from a field like REPS does.
    spatial_dims = next(iter(xarray_forecast_merged.data_vars.values())).dims[-2:]
    NY, NX = (xarray_forecast_merged.sizes[dim] for dim in spatial_dims)

    # %% Ensemble statistics across members
    write_ensemble_stats(
        xarray_forecast_merged,
        GEPS_FORECAST_HOURS,
        forecast_process_path + "_.zarr",
        process_chunk,
        zarr_store_workers,
    )
    xarray_forecast_merged.close()

    # The merged NetCDF is large and no longer needed
    os.remove(merged_nc_path)

    # %% Stack the statistics into the (variable, time, y, x) layout the API reads
    # The interpolation source axis is the exact forecast valid times: the stored
    # time variable is float32 (~2 minute resolution), which could put the first
    # or last hourly target just outside the source range.
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    # Hourly target axis from the first lead (run + 3 h) to the last; the 3- and
    # 6-hourly steps are interpolated onto it below.
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0], end=forecast_valid_times[-1], freq="h"
    )
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    daskVarArrayList = []
    for dask_var in probVars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            npForecastTimes = daskForecastArray.astype("float32").compute()
            # Check that the stored times are aligned with the expected forecast times
            validate_stacked_time_alignment(stacked_timesUnix, npForecastTimes)
            # The API reads time from the store like any other variable
            daskVarArrayList.append(
                broadcast_times_to_grid(npForecastTimes, NY, NX, process_chunk)
            )
        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
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

    geps_store_name = "GEPS.zarr"
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/" + geps_store_name,
        final_chunk,
        zarr_store_workers,
        probVars,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [geps_store_name],
        base_time,
        "GEPS.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up, keeping the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    logger.info("GEPS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
