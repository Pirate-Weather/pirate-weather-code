# %% NBM Hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023

# %% Import modules
import fcntl
import logging
import os
import shutil
import sys
import time
import warnings
from itertools import chain

import dask
import dask.array as da
import netCDF4 as nc
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from dotenv import find_dotenv, load_dotenv

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    close_store,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    promote_staged_dir,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
)

# Env setup required before importing Herbie for ECCodes paths
dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

from herbie.fast import Herbie_latest  # noqa: E402

from API.ingest_grib_utils import (  # noqa: E402
    check_grib_stats,
    download_and_validate_subset,
    run_checked,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging
logger = logging.getLogger(__name__)


def _timing_log(message: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[TIMING] {ts} {message}")


def _acquire_single_run_lock(lock_path: str):
    """Acquire a non-blocking process lock so overlapping ingests do not clobber temp data."""
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock_file = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(
            "Another NBM ingest process is already running; "
            f"skipping this run (lock: {lock_path})"
        )
        lock_file.close()
        sys.exit(0)

    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def _download_nbm_subset(
    forecast_hours,
    search: str,
    dataset_name: str,
    run_date=None,
    *,
    base_time: pd.Timestamp,
) -> list[str]:
    """Use the shared retry and local-file checks before NBM processing."""
    return download_and_validate_subset(
        model="nbm",
        product="co",
        search=search,
        dataset_name=dataset_name,
        base_time=base_time,
        run_date=run_date,
        forecast_hours=list(forecast_hours),
        priority=["aws", "nomads"],
        herbie_save_dir=tmp_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
        herbie_kwargs={"max_threads": herbie_max_threads},
        download_max_threads=herbie_max_threads,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
        # Combined probability and accumulation subsets are validated below.
        skip_wgrib2_validation=True,
    )


# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/NBM")
forecast_process_path = forecast_process_dir + "/NBM_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/NBM")
local_forecast_version_dir = forecast_path + "/" + ingest_version


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
herbie_max_threads = positive_int_env("herbie_max_threads", 1)
herbie_download_retries = positive_int_env("herbie_download_retries", 3)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)


# Define the processing chunk size
process_chunk = CHUNK_SIZES["NBM"]

# Define the final x/y chunk size
final_chunk = FINAL_CHUNK_SIZES["NBM"]


def main() -> None:
    """Run the NBM forecast ingest."""
    logging.basicConfig(level=logging.INFO)
    nbm_staged_path: str | None = None
    nbm_maps_staged_path: str | None = None
    configure_herbie_request_timeouts(
        nomads_request_interval_s=positive_int_env("nomads_request_interval_seconds", 5)
    )

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    configure_zarr_writes()
    _ingest_lock_handle = _acquire_single_run_lock(forecast_process_dir + ".lock")

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    if save_type == "Download":
        os.makedirs(local_forecast_version_dir, exist_ok=True)

    # %% Define base time from the most recent run
    T0 = time.time()

    latest_run = Herbie_latest(
        model="nbm",
        n=5,
        freq="1h",
        fxx=[190, 191, 192, 193, 194, 195],
        product="co",
        verbose=False,
        priority=["aws", "nomads"],
        save_dir=tmp_dir,
    )

    base_time = latest_run.date

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        local_forecast_version_dir + "/NBM.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to NBM, ending")
        return

    print(base_time)
    _timing_log(f"NBM_FORECAST run_start base_time={base_time}")

    zarr_vars = (
        "time",
        "GUST_10maboveground",
        "TMP_2maboveground",
        "APTMP_2maboveground",
        "DPT_2maboveground",
        "RH_2maboveground",
        "WIND_10maboveground",
        "WDIR_10maboveground",
        "APCP_surface",
        "TCDC_surface",
        "VIS_surface",
        "PWTHER_surfaceMreserved",
        "PPROB",
        "PACCUM",
        "PTYPE_prob_GE_1_LT_2_prob_fcst_1_1_surface",
        "PTYPE_prob_GE_3_LT_4_prob_fcst_1_1_surface",
        "PTYPE_prob_GE_5_LT_7_prob_fcst_1_1_surface",
        "PTYPE_prob_GE_8_LT_9_prob_fcst_1_1_surface",
        "DSWRF_surface",
        "CAPE_surface",
    )

    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 240 hours
    # Set download rannges
    forecast_iter_start = time.perf_counter()
    if base_time.hour % 6 == 0:
        nbm_range1 = range(1, 37, 1)
        nbm_range2 = range(42, 193, 3)
    elif base_time.hour % 6 == 1:
        nbm_range1 = range(1, 36, 1)
        nbm_range2 = range(41, 192, 3)
    elif base_time.hour % 6 == 2:
        nbm_range1 = range(1, 35, 1)
        nbm_range2 = range(40, 191, 3)
    elif base_time.hour % 6 == 3:
        nbm_range1 = range(1, 34, 1)
        nbm_range2 = range(39, 190, 3)
    elif base_time.hour % 6 == 4:
        nbm_range1 = range(1, 33, 1)
        nbm_range2 = range(38, 189, 3)
    elif base_time.hour % 6 == 5:
        nbm_range1 = range(1, 32, 1)
        nbm_range2 = range(37, 188, 3)

    nbm_range = list(chain(nbm_range1, nbm_range2))

    # Define the subset of variables to download as a list of strings
    matchstring_2m = r":((DPT|TMP|APTMP|RH):2 m above ground:.*fcst:$)"
    matchstring_su = r":((PTYPE):surface:.*)"
    matchstring_10m = r"(:(GUST|WIND|WDIR):10 m above ground:.*fcst:$)"
    matchstring_pr = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:$)"
    matchstring_re = r":((TCDC|VIS|DSWRF|CAPE):surface:.*fcst:$)"  # This gets the correct surface param

    # Merge matchstrings for download
    # Note: PWTHER (predominant weather) is intentionally excluded here because NBM v5 drops
    # the variable intermittently. It is not used by the API so we fill it with NaN values
    # during processing to keep the array shape unchanged.
    match_strings = (
        matchstring_2m
        + "|"
        + matchstring_su
        + "|"
        + matchstring_10m
        + "|"
        + matchstring_pr
        + "|"
        + matchstring_re
    )

    # Download the subsets
    stage_start = time.perf_counter()
    grib_list = _download_nbm_subset(
        nbm_range, match_strings, "NBM main forecast", base_time=base_time
    )
    _timing_log(
        f"NBM_FORECAST stage=download_main elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Perform a check if any data seems to be invalid
    stage_start = time.perf_counter()
    check_grib_stats(grib_list, wgrib2_path, "NBM main forecast")
    _timing_log(
        f"NBM_FORECAST stage=validate_main elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -grib "
        + forecast_process_path
        + "_wgrib2_merged.grib2"
    )
    # Run wgrib2
    stage_start = time.perf_counter()
    substage_start = time.perf_counter()
    run_checked(cmd, "NBM main wgrib2 GRIB merge")
    _timing_log(
        f"NBM_FORECAST stage=main_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )

    # Use wgrib2 to change the order
    substage_start = time.perf_counter()
    cmd2 = (
        f"{wgrib2_path}"
        + "  "
        + forecast_process_path
        + "_wgrib2_merged.grib2 "
        + " -ijsmall_grib "
        + " 1:2345 1:1597 "
        + forecast_process_path
        + "_wgrib2_merged_order.grib"
    )
    run_checked(cmd2, "NBM main wgrib2 grid reorder")
    _timing_log(
        f"NBM_FORECAST stage=main_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    os.remove(forecast_process_path + "_wgrib2_merged.grib2")

    # Convert to NetCDF
    cmd4 = (
        f"{wgrib2_path}"
        + "  "
        + forecast_process_path
        + "_wgrib2_merged_order.grib "
        + " -set_ext_name 1 -netcdf "
        + forecast_process_path
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2 to rotate winds and save as NetCDF
    substage_start = time.perf_counter()
    run_checked(cmd4, "NBM main wgrib2 NetCDF conversion")
    os.remove(forecast_process_path + "_wgrib2_merged_order.grib")
    _timing_log(
        f"NBM_FORECAST stage=main_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    _timing_log(
        f"NBM_FORECAST stage=main_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    ##### PPROB
    # Set download ranges. Note the 6 hourly second step instead of 3, since that's where the prob data is saved
    if base_time.hour % 6 == 0:
        nbm_range1 = range(1, 37, 1)
        nbm_range2 = range(42, 193, 6)
    elif base_time.hour % 6 == 1:
        nbm_range1 = range(1, 36, 1)
        nbm_range2 = range(41, 192, 6)
    elif base_time.hour % 6 == 2:
        nbm_range1 = range(1, 35, 1)
        nbm_range2 = range(40, 191, 6)
    elif base_time.hour % 6 == 3:
        nbm_range1 = range(1, 34, 1)
        nbm_range2 = range(39, 190, 6)
    elif base_time.hour % 6 == 4:
        nbm_range1 = range(1, 33, 1)
        nbm_range2 = range(38, 189, 6)
    elif base_time.hour % 6 == 5:
        nbm_range1 = range(1, 32, 1)
        nbm_range2 = range(37, 188, 6)

    # Download PPROB as 1-Hour Prob
    matchstring_po = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|[0-9]*0-[0-9]{1,2}1|[0-9]*1-[0-9]{1,2}2|[0-9]*2-[0-9]{1,2}3|[0-9]*3-[0-9]{1,2}4|[0-9]*4-[0-9]{1,2}5|[0-9]*5-[0-9]{1,2}6|[0-9]*6-[0-9]{1,2}7|[0-9]*7-[0-9]{1,2}8|[0-9]*8-[0-9]{1,2}9|[0-9]*9-[0-9]{1,2}0).*fcst:prob.*)"
    # Download the subsets
    stage_start = time.perf_counter()
    gribList1 = _download_nbm_subset(
        nbm_range1,
        matchstring_po,
        "NBM hourly precipitation probability",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_pprob_hourly elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    #####
    # Download PPROB as 6-Hour Accum for hours 036-190

    # Match 6-hour probs
    matchstring_po2 = r":APCP:surface:(0-6|[0-9]*0-[0-9]{1,2}6|[0-9]*1-[0-9]{1,2}7|[0-9]*2-[0-9]{1,2}8|[0-9]*3-[0-9]{1,2}9|[0-9]*4-[0-9]{1,2}0|[0-9]*5-[0-9]{1,2}1|[0-9]*6-[0-9]{1,2}2|[0-9]*7-[0-9]{1,2}3|[0-9]*8-[0-9]{1,2}4|[0-9]*9-[0-9]{1,2}5).*fcst:prob"
    # Download the subsets
    stage_start = time.perf_counter()
    gribList2 = _download_nbm_subset(
        nbm_range2,
        matchstring_po2,
        "NBM six-hour precipitation probability",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_pprob_6hour elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    grib_list = gribList1 + gribList2

    # Perform a check if any data seems to be invalid
    stage_start = time.perf_counter()
    check_grib_stats(grib_list, wgrib2_path, "NBM PPROB")
    _timing_log(
        f"NBM_FORECAST stage=validate_pprob elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -grib "
        + forecast_process_path
        + "_prob_wgrib2_merged.grib2"
    )
    # Run wgrib2
    stage_start = time.perf_counter()
    substage_start = time.perf_counter()
    run_checked(cmd, "NBM PPROB wgrib2 GRIB merge")
    _timing_log(
        f"NBM_FORECAST stage=pprob_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )

    # Use wgrib2 to change the order
    substage_start = time.perf_counter()
    cmd2 = (
        f"{wgrib2_path}"
        + "  "
        + forecast_process_path
        + "_prob_wgrib2_merged.grib2 "
        + " -ijsmall_grib "
        + " 1:2345 1:1597 "
        + forecast_process_path
        + "_prob_wgrib2_merged_order.grib"
    )
    run_checked(cmd2, "NBM PPROB wgrib2 grid reorder")
    _timing_log(
        f"NBM_FORECAST stage=pprob_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    os.remove(forecast_process_path + "_prob_wgrib2_merged.grib2")

    # Convert to NetCDF
    cmd4 = (
        f"{wgrib2_path}"
        + "  "
        + forecast_process_path
        + "_prob_wgrib2_merged_order.grib "
        + " -set_ext_name 1 -netcdf "
        + forecast_process_path
        + "_prob_wgrib2_merged.nc"
    )

    # Run wgrib2 to save as  NetCDF
    substage_start = time.perf_counter()
    run_checked(cmd4, "NBM PPROB wgrib2 NetCDF conversion")
    os.remove(forecast_process_path + "_prob_wgrib2_merged_order.grib")
    _timing_log(
        f"NBM_FORECAST stage=pprob_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    _timing_log(
        f"NBM_FORECAST stage=pprob_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Download PACCUM as 1-Hour and 6-hour Accum
    matchstring_pa = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:$)"
    # Download the subsets
    stage_start = time.perf_counter()
    gribList1 = _download_nbm_subset(
        nbm_range1,
        matchstring_pa,
        "NBM hourly precipitation accumulation",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_paccum_hourly elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    #####
    # Download 6-Hour Accum for hours 036-190

    # Match 6-hour accumulation
    matchstring_pa2 = r"(:APCP:surface:(0-6|\d*0-\d{1,2}6|\d*1-\d{1,2}7|\d*2-\d{1,2}8|\d*3-\d{1,2}9|\d*4-\d{1,2}0|\d*5-\d{1,2}1|\d*6-\d{1,2}2|\d*7-\d{1,2}3|\d*8-\d{1,2}4|\d*9-\d{1,2}5).*fcst:$)"
    # Download the subsets
    stage_start = time.perf_counter()
    gribList2 = _download_nbm_subset(
        nbm_range2,
        matchstring_pa2,
        "NBM six-hour precipitation accumulation",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_paccum_6hour elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    grib_list = gribList1 + gribList2

    # Perform a check if any data seems to be invalid
    stage_start = time.perf_counter()
    check_grib_stats(grib_list, wgrib2_path, "NBM PACCUM")
    _timing_log(
        f"NBM_FORECAST stage=validate_paccum elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -grib "
        + forecast_process_path
        + "_accum_wgrib2_merged.grib2"
    )
    # Run wgrib2
    stage_start = time.perf_counter()
    substage_start = time.perf_counter()
    run_checked(cmd, "NBM PACCUM wgrib2 GRIB merge")
    _timing_log(
        f"NBM_FORECAST stage=paccum_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )

    # Use wgrib2 to change the order
    substage_start = time.perf_counter()
    cmd2 = (
        f"{wgrib2_path}"
        + "  "
        + forecast_process_path
        + "_accum_wgrib2_merged.grib2 "
        + " -ijsmall_grib "
        + " 1:2345 1:1597 "
        + forecast_process_path
        + "_accum_wgrib2_merged_order.grib"
    )
    run_checked(cmd2, "NBM PACCUM wgrib2 grid reorder")
    _timing_log(
        f"NBM_FORECAST stage=paccum_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    os.remove(forecast_process_path + "_accum_wgrib2_merged.grib2")

    # Convert to NetCDF
    cmd4 = (
        f"{wgrib2_path}"
        + "  "
        + forecast_process_path
        + "_accum_wgrib2_merged_order.grib "
        + " -set_ext_name 1 -netcdf "
        + forecast_process_path
        + "_accum_wgrib2_merged.nc"
    )

    # Run wgrib2 to save as NetCDF
    substage_start = time.perf_counter()
    run_checked(cmd4, "NBM PACCUM wgrib2 NetCDF conversion")
    os.remove(forecast_process_path + "_accum_wgrib2_merged_order.grib")
    _timing_log(
        f"NBM_FORECAST stage=paccum_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    _timing_log(
        f"NBM_FORECAST stage=paccum_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    #######
    # Use Dask to create a merged array (too large for xarray)
    # Dask
    stage_start = time.perf_counter()

    # Create base xarray for time interpolation
    xarray_forecast_base = xr.open_mfdataset(
        forecast_process_path + "_wgrib2_merged.nc"
    )

    # Create a new time series
    start = xarray_forecast_base.time.min().values  # Adjust as necessary
    end = xarray_forecast_base.time.max().values  # noqa: F841  # Adjust as necessary
    # The hourly target axis starts at the first forecast valid time
    # (base_time + 1 h); the store holds no history rows.
    new_hourly_time = pd.date_range(
        start=start,
        end=start + pd.Timedelta(hours=192),
        freq="h",
    )

    stacked_times = xarray_forecast_base.time.values
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Drop all variables
    xarray_forecast_base = xarray_forecast_base.drop_vars(
        [i for i in xarray_forecast_base.data_vars]
    )

    # Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
    # Note: don't chunk on loading since we don't know how wgrib2 chunked the files. Intead, read the variable into memory and chunk later

    # Open NC Dataset
    ncForecast = nc.Dataset(forecast_process_path + "_wgrib2_merged.nc")

    # Disable masking
    ncForecast.set_auto_mask(False)
    forecast_var_zarr_dir = forecast_process_path + "_zarrs"
    if os.path.exists(forecast_var_zarr_dir):
        shutil.rmtree(forecast_var_zarr_dir)
    os.makedirs(forecast_var_zarr_dir, exist_ok=True)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        for dask_var in zarr_vars:
            var_stage_start = time.perf_counter()
            if dask_var == "PPROB":
                # Interpolate over nan's in APCP
                xarray_forecast = xarray_forecast_base.copy()
                # Import and allign times
                xarray_forecast["PPROB"] = xr.open_mfdataset(
                    forecast_process_path + "_prob_wgrib2_merged.nc"
                )["APCP_prob_GT_0D254_prob_fcst_255_255_surface"]
                xarray_forecast["PPROB"] = xarray_forecast["PPROB"].bfill(dim="time")
                daskArray = xarray_forecast["PPROB"].data

                del xarray_forecast

            elif dask_var == "PACCUM":
                xarray_forecast = xarray_forecast_base.copy()
                xarray_forecast["PACCUM"] = xr.open_mfdataset(
                    forecast_process_path + "_accum_wgrib2_merged.nc"
                )["APCP_surface"]
                # Fix precipitation accumulation timing to account for everything after the hourly data being a 6-hour accumulation,
                # Change to 1 hour accum
                xarray_forecast["PACCUM"][nbm_range1[-1] :, :, :] = (
                    xarray_forecast["PACCUM"][nbm_range1[-1] :, :, :] / 6
                )

                # Interpolate over nan's in APCP
                xarray_forecast["PACCUM"] = xarray_forecast["PACCUM"].bfill(dim="time")

                daskArray = xarray_forecast["PACCUM"].data

                del xarray_forecast

            elif dask_var == "PWTHER_surfaceMreserved":
                # PWTHER is not used by the API and is intermittently absent from NBM v5 grib
                # files. Fill the slot with NaN values so the downstream array shape is preserved.
                ref_shape = ncForecast["TMP_2maboveground"].shape
                daskArray = da.full(ref_shape, np.nan, dtype=np.float32)

            else:
                daskArray = da.from_array(
                    ncForecast[dask_var],
                    lock=True,
                )

            # Check length for errors

            if len(daskArray) != len(nbm_range):
                logger.error("daskArray length: %d", len(daskArray))
                logger.error("nbm_range length: %d", len(nbm_range))
                logger.error("dask_var: %s", dask_var)
                assert len(daskArray) == len(nbm_range), (
                    "Incorrect number of timesteps! Exiting"
                )

            # Rechunk
            daskArray = daskArray.rechunk(
                chunks=(len(nbm_range), process_chunk, process_chunk)
            )

            # Save merged and processed xarray dataset to disk using zarr with compression
            # Define the path to save the zarr dataset
            # Save the dataset with compression and filters for all variables
            forecast_var_zarr_path = forecast_var_zarr_dir + "/" + dask_var + ".zarr"
            if dask_var == "time":
                # Save the dataset without compression and filters for all variable
                daskArray.to_zarr(forecast_var_zarr_path, overwrite=True)
            else:
                # Save the dataset with compression and filters for all variable
                daskArray.to_zarr(
                    forecast_var_zarr_path,
                    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                    overwrite=True,
                )
            try:
                zarr.open_array(forecast_var_zarr_path, mode="r")
            except Exception as exc:
                raise RuntimeError(
                    "Failed to open forecast zarr store for "
                    f"{dask_var} at {forecast_var_zarr_path}"
                ) from exc
            _timing_log(
                "NBM_FORECAST "
                f"stage=forecast_var_to_zarr var={dask_var} elapsed={time.perf_counter() - var_stage_start:.2f}s"
            )

    _timing_log(
        f"NBM_FORECAST stage=forecast_vars_to_zarr elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # Del to free memory
    ncForecast.close()
    xarray_forecast_base.close()
    del daskArray, xarray_forecast_base

    # Remove wgrib2 temp files
    cleanup_stage_start = time.perf_counter()
    os.remove(forecast_process_path + "_wgrib2_merged.nc")
    os.remove(forecast_process_path + "_prob_wgrib2_merged.nc")
    os.remove(forecast_process_path + "_accum_wgrib2_merged.nc")
    _timing_log(
        f"NBM_FORECAST stage=forecast_temp_cleanup elapsed={time.perf_counter() - cleanup_stage_start:.2f}s"
    )

    T1 = time.time()
    logger.info(T0 - T1)

    # %% Squash the forecast variables using dask and interpolate to hourly
    #####################################################################################################

    logger.info("Merge and interpolate arrays.")
    # Dask Setup
    daskInterpArrays = []  # noqa: F841
    daskVarArrayList = []

    for daskVarIDX, dask_var in enumerate(zarr_vars[:]):
        forecast_var_zarr_path = forecast_var_zarr_dir + "/" + dask_var + ".zarr"
        if not os.path.exists(forecast_var_zarr_path):
            raise FileNotFoundError(
                "Missing forecast zarr variable store: "
                f"{forecast_var_zarr_path}. This usually means another ingest "
                "run cleared the shared processing directory while this run was active."
            )
        daskForecastArray = da.from_zarr(forecast_var_zarr_path, inline_array=True)

        if dask_var == "time":
            # Get times as numpy
            npCatTimes = daskForecastArray.astype("float32").compute()
            validate_stacked_time_alignment(stacked_timesUnix, npCatTimes)

            daskArrayOut = da.from_array(
                np.tile(
                    np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                    (1, 1597, 2345),
                )
            ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

            daskVarArrayList.append(daskArrayOut)

        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
                    (len(stacked_timesUnix), process_chunk, process_chunk)
                ).astype("float32")
            )

        logger.info(dask_var)

    # Merge the arrays into a single 4D array
    daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

    # Mask out invalid data
    daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

    # Write out to disk
    # This intermediate step is necessary to avoid memory overflow
    # with ProgressBar():
    stage_start = time.perf_counter()
    daskVarArrayListMergeNaN.to_zarr(
        forecast_process_path + "_stack.zarr", overwrite=True, compute=True
    )
    _timing_log(
        f"NBM_FORECAST stage=stack_to_zarr elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    logger.info("Stacked 4D array saved to disk.")

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # Create a zarr backed dask array
    if save_type == "S3":
        zarr_store = zarr.storage.ZipStore(
            forecast_process_dir + "/NBM.zarr.zip", mode="a", compression=0
        )
    else:
        nbm_staged_path = local_forecast_version_dir + "/NBM.zarr.partial"
        if os.path.exists(nbm_staged_path):
            shutil.rmtree(nbm_staged_path)
        zarr_store = zarr.storage.LocalStore(nbm_staged_path)

    with ProgressBar():
        stage_start = time.perf_counter()
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
        _timing_log(
            f"NBM_FORECAST stage=interpolate_and_write elapsed={time.perf_counter() - stage_start:.2f}s"
        )

    logger.info("Interpolate complete")

    close_store(zarr_store)

    # Rechunk subset of data for maps!
    # Want variables:
    # 0 (time)
    # 2 (TMP)
    # 6 (WIND)
    # 7 (WDIR)
    # 8 (APCP)
    # 13 (PACCUM)
    # 14:17 (PTYPE)

    # Loop through variables, creating a new one with a name and 36 x 100 x 100 chunks
    # Save the first 36 stored forecast steps (base+1 h onward, before hourly
    # interpolation). These were steps 36:72 when 48 h of history preceded them.
    # Create a Zarr array in the store with zstd compression

    # Add padding for map chunking (100x100)
    daskVarArrayStackDisk_maps = pad_to_chunk_size(daskVarArrayStackDisk, 100)

    if save_type == "S3":
        zarr_store_maps = zarr.storage.ZipStore(
            forecast_process_dir + "/NBM_Maps.zarr.zip", mode="a"
        )
    else:
        nbm_maps_staged_path = local_forecast_version_dir + "/NBM_Maps.zarr.partial"
        if os.path.exists(nbm_maps_staged_path):
            shutil.rmtree(nbm_maps_staged_path)
        zarr_store_maps = zarr.storage.LocalStore(nbm_maps_staged_path)

    stage_start = time.perf_counter()
    for z in [0, 2, 6, 7, 8, 13, 14, 15, 16, 17]:
        # Create a zarr backed dask array
        zarr_array = zarr.create_array(
            store=zarr_store_maps,
            name=zarr_vars[z],
            shape=(
                36,
                daskVarArrayStackDisk_maps.shape[2],
                daskVarArrayStackDisk_maps.shape[3],
            ),
            chunks=(36, 100, 100),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )

        # with ProgressBar():
        da.rechunk(daskVarArrayStackDisk_maps[z, :36, :, :], (36, 100, 100)).to_zarr(
            zarr_array, overwrite=True, compute=True
        )

        logger.info(zarr_vars[z])

    close_store(zarr_store_maps)
    _timing_log(
        f"NBM_FORECAST stage=maps_write elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    logger.info("Map complete")

    # %% Publish the stores, then the run time the API uses to detect them
    stage_start = time.perf_counter()
    if save_type == "S3":
        publish_ingest_run(
            ["NBM.zarr.zip", "NBM_Maps.zarr.zip"],
            base_time,
            "NBM.time.pickle",
            process_dir=forecast_process_dir,
            publish_dir=local_forecast_version_dir,
            save_type=save_type,
            s3=s3,
        )
    else:
        if nbm_staged_path is None or nbm_maps_staged_path is None:
            raise RuntimeError(
                "Expected staged local zarr paths for Download publish, but they were not set"
            )

        # Local stores were written straight into "<publish dir>/*.partial" to avoid
        # copying a large store, so only the rename and the run time remain.
        promote_staged_dir(nbm_staged_path, local_forecast_version_dir + "/NBM.zarr")
        promote_staged_dir(
            nbm_maps_staged_path, local_forecast_version_dir + "/NBM_Maps.zarr"
        )
        publish_ingest_run(
            [],
            base_time,
            "NBM.time.pickle",
            process_dir=forecast_process_dir,
            publish_dir=local_forecast_version_dir,
            save_type=save_type,
            s3=s3,
        )
    _timing_log(
        f"NBM_FORECAST stage=publish_output elapsed={time.perf_counter() - stage_start:.2f}s"
    )
    _timing_log(
        f"NBM_FORECAST total_elapsed={time.perf_counter() - forecast_iter_start:.2f}s"
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)

    # Test Read
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
