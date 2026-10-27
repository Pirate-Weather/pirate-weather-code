"""AIGEFS ensemble precipitation ingest.

Downloads the precipitation field of all 30 members of NOAA's AI Global Ensemble
Forecast System (AIGEFS, 0.25 degree global grid, 6-hourly leads to 240 h) for
the newest 6-hourly cycle, reduces them to ensemble statistics (precipitation
probability, mean and standard deviation), interpolates them to an hourly axis
and publishes ``AIGEFS.zarr`` plus ``AIGEFS.time.pickle``. The store starts at
the first forecast lead (run + 6 h); the API falls back to other sources for
earlier hours.

Alexander Rey, September 2023
"""

import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import dask.array as da
import numpy as np
import pandas as pd
import requests
import xarray as xr
import zarr
from herbie import HerbieLatest

from API.constants.shared_const import (
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    INGEST_VERSION_STR,
)
from API.herbie_custom_templates import register_aws_aigfs_aigefs_templates
from API.ingest_grib_utils import check_grib_stats, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    download_grib_file,
    interp_time_take_blend,
    mask_invalid_data,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
AIGEFS_MEMBER_COUNT = 30
AIGEFS_FORECAST_HOURS = FORECAST_LEAD_RANGES["AIGEFS"]
# Each lead holds the precipitation total over the preceding 6 hours (not a
# run total), so it only needs dividing into an hourly rate.
AIGEFS_LEAD_STEP_HOURS = 6
# The AWS EAGLE ensemble is experimental; operational AIGEFS is on NOMADS.
# (On 2026-10-23 the AWS control member's precipitation correlated only 0.78
# with the operational one: a different model, not a mirror.)
AIGEFS_SOURCE_PRIORITY = ["nomads"]
AIGEFS_NOMADS_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod"
# The full member files are downloaded; wgrib2 keeps only precipitation
AIGEFS_WGRIB2_MATCH = ":APCP:surface:"

# Per-member variables, in the order they are read back after conversion; the
# member times are only used to check the time axis before publishing
zarr_vars = (
    "time",
    "APCP_surface",
)

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so probVars[i] is column i + 1 of the
# API's AIGEFS index map.
probVars = (
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# When running the docker container, this should be "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/AIGEFS"
)
forecast_process_path = os.path.join(forecast_process_dir, "AIGEFS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/AIGEFS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

# NOMADS rate-limits per IP; requests are paced across threads, so a few
# threads only overlap the transfers.
AIGEFS_MAX_THREADS = positive_int_env("aigefs_download_threads", 4)
AIGEFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
AIGEFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)
AIGEFS_REQUEST_TIMEOUT_SECONDS = positive_int_env("herbie_request_timeout_seconds", 120)


def aigefs_member_url(run_date: pd.Timestamp, member: int, fxx: int) -> str:
    """NOMADS URL of one member's surface GRIB (the full file is downloaded)."""
    return (
        f"{AIGEFS_NOMADS_ROOT}/aigefs.{run_date:%Y%m%d/%H}/mem{member:03d}/model/"
        f"atmos/grib2/aigefs.t{run_date:%H}z.sfc.f{fxx:03d}.grib2"
    )


def download_aigefs_member(
    run_date: pd.Timestamp,
    forecast_hours: list[int],
    member: int,
    dataset_name: str,
) -> list[str]:
    """Download and validate a complete member with bounded retries.

    Each file is one paced GET. Going through Herbie cost three NOMADS requests
    per file (a HEAD on the GRIB, a HEAD on its unused .idx, then the GET),
    about 13 s per file and ~5 h per run at 5 s pacing; the run is already
    known to exist (HerbieLatest found member 0's last lead), and a file that
    is not posted yet fails its GET and is retried.
    """
    member_dir = os.path.join(
        tmp_dir, f"aigefs_{run_date:%Y%m%d%H}", f"mem{member:03d}"
    )
    targets = [
        (
            aigefs_member_url(run_date, member, fxx),
            os.path.join(member_dir, f"aigefs.sfc.f{fxx:03d}.grib2"),
        )
        for fxx in forecast_hours
    ]

    for attempt in range(1, AIGEFS_DOWNLOAD_RETRIES + 1):
        try:
            with ThreadPoolExecutor(max_workers=AIGEFS_MAX_THREADS) as executor:
                grib_files = list(
                    executor.map(
                        lambda target: download_grib_file(
                            target[0],
                            target[1],
                            overwrite=False,
                            request_timeout_s=AIGEFS_REQUEST_TIMEOUT_SECONDS,
                        ),
                        targets,
                    )
                )
            break
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == AIGEFS_DOWNLOAD_RETRIES:
                raise
            sleep_s = AIGEFS_RETRY_SLEEP_SECONDS * attempt
            logger.warning(
                "%s download attempt %d/%d failed (%s). Retrying in %ss",
                dataset_name,
                attempt,
                AIGEFS_DOWNLOAD_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    check_grib_stats(grib_files, wgrib2_path.strip(), dataset_name)
    return grib_files


def main() -> None:
    """Run the AIGEFS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    # Use the repository Herbie templates for the current AWS/NOMADS paths
    register_aws_aigfs_aigefs_templates()
    # The final write used dask's default thread pool (one thread per CPU).
    zarr_store_workers = configure_zarr_writes(default_workers=os.cpu_count() or 2)

    process_chunk = CHUNK_SIZES["GEFS"]
    final_chunk = FINAL_CHUNK_SIZES["GEFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="aigefs",
        n=3,
        freq="6h",
        fxx=AIGEFS_FORECAST_HOURS[-1],
        product="sfc",
        verbose=False,
        member="mem000",
        priority=AIGEFS_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "AIGEFS.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to AIGEFS, ending")
        return
    logger.info("Processing AIGEFS run %s", base_time)

    # %% Forecast hours: 6-hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(AIGEFS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download, convert and correct each member
    member_zarr_paths = []
    for member in range(1, AIGEFS_MEMBER_COUNT + 1):
        grib_list = download_aigefs_member(
            base_time,
            list(AIGEFS_FORECAST_HOURS),
            member,
            f"AIGEFS member {member} forecast",
        )

        member_nc_path = forecast_process_path + f"_wgrib2_merged_m{member}.nc"
        run_checked(
            "cat "
            + " ".join(grib_list)
            + f' | {wgrib2_path} - -match "{AIGEFS_WGRIB2_MATCH}"'
            + f" -netcdf {member_nc_path}",
            f"AIGEFS member {member} forecast GRIB conversion",
        )
        xarray_wgrib = xr.open_dataset(member_nc_path)

        if not np.array_equal(xarray_wgrib["time"].values, forecast_valid_times):
            raise RuntimeError(
                f"AIGEFS member {member} times {xarray_wgrib['time'].values} do not "
                f"match the expected forecast leads {AIGEFS_FORECAST_HOURS}"
            )

        # The source can contain small negative precipitation totals; clip them
        # (as the old history path did) before converting to an hourly rate.
        xarray_wgrib["APCP_surface"] = (
            np.maximum(xarray_wgrib["APCP_surface"], 0) / AIGEFS_LEAD_STEP_HOURS
        )

        xarray_wgrib_chunked = xarray_wgrib.chunk(
            chunks={
                "time": len(AIGEFS_FORECAST_HOURS),
                "latitude": process_chunk,
                "longitude": process_chunk,
            }
        )
        member_zarr_path = forecast_process_path + f"_xr_m{member}.zarr"
        xarray_wgrib_chunked.to_zarr(member_zarr_path, consolidated=False, mode="w")
        # Close the dataset open_dataset returned; closing the chunked copy
        # leaves the member NetCDF open until the variable is reassigned.
        xarray_wgrib.close()
        member_zarr_paths.append(member_zarr_path)

        # The merged NetCDF is large and no longer needed
        os.remove(member_nc_path)

    # %% Ensemble statistics across members
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    # Hourly target axis from the first lead (run + 6 h) to the last
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0], end=forecast_valid_times[-1], freq="h"
    )
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Stack members along a new leading axis, one dask array per variable
    # (this works much better than xarray's open_mfdataset here)
    daskArrays = {
        dask_var: da.stack(
            [da.from_zarr(path, dask_var) for path in member_zarr_paths], axis=0
        )
        for dask_var in zarr_vars
    }

    # Every member was checked against the expected valid times above; check the
    # stored (encoded) times too, from member 0
    validate_stacked_time_alignment(
        stacked_timesUnix, daskArrays["time"][0, :].compute().astype("float64")
    )

    daskOutput = {
        "Precipitation_Prob": (
            daskArrays["APCP_surface"] > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H
        ).sum(axis=0)
        / AIGEFS_MEMBER_COUNT,
        "APCP_Mean": daskArrays["APCP_surface"].mean(axis=0),
        "APCP_StdDev": daskArrays["APCP_surface"].std(axis=0),
    }

    # Write each statistic to disk; computing them all at once overflows memory
    for dask_var in probVars:
        stat_array = daskOutput[dask_var].rechunk(
            (len(AIGEFS_FORECAST_HOURS), process_chunk, process_chunk)
        )
        stat_array.to_zarr(
            forecast_process_path + "_" + dask_var + ".zarr",
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            overwrite=True,
            compute=True,
            dtype="float32",
        )
    del daskOutput, daskArrays

    # %% Stack the statistics into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in probVars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_" + dask_var + ".zarr", inline_array=True
        )
        daskVarArrayList.append(
            daskForecastArray.rechunk(
                (len(stacked_timesUnix), process_chunk, process_chunk)
            ).astype("float32")
        )

    daskVarArrayListMergeNaN = mask_invalid_data(da.stack(daskVarArrayList, axis=0))

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
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

    aigefs_store_name = "AIGEFS.zarr"
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/" + aigefs_store_name,
        final_chunk,
        FINAL_SHARD_SIZES["GEFS"],
        zarr_store_workers,
        probVars,
        times=hourly_timesUnix,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [aigefs_store_name],
        base_time,
        "AIGEFS.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("AIGEFS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
