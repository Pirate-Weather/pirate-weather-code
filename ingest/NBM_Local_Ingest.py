"""NBM (National Blend of Models) forecast ingest.

Downloads the newest hourly NBM CONUS cycle (2.5 km Lambert conformal grid,
2345 x 1597, hourly cycles, hourly leads to 36 h then 3-hourly to 192 h, shifted
earlier for off-synoptic cycles), plus the 1-hour / 6-hour precipitation
probability (PPROB) and accumulation (PACCUM) fields, interpolates everything to
an hourly axis and publishes ``NBM.zarr`` and ``NBM.time.pickle``. The store starts at the first forecast lead (run + 1 h).

Local (Download) runs write the stores straight into ``<publish dir>/*.partial``
and promote them with a rename, so the large store is never copied.

Alexander Rey, November 2023
"""

import fcntl
import logging
import os
import shutil
import time
import warnings

import dask
import dask.array as da
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dotenv import find_dotenv, load_dotenv

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
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

logger = logging.getLogger(__name__)

# %% Model constants
# wgrib2 -ijsmall_grib i and j ranges covering the full NBM CONUS grid. Cutting
# the full extent rewrites every record on the same grid before the NetCDF export.
NBM_IJSMALL_GRID = "1:2345 1:1597"
NBM_SOURCE_PRIORITY = ["aws", "nomads"]
# Leads checked to find the newest complete cycle; the last lead shifts with the
# cycle hour (see below), so any of these marks a finished run.
NBM_LATEST_CHECK_HOURS = [190, 191, 192, 193, 194, 195]

# Lead layout of the synoptic (00/06/12/18 UTC) cycles: hourly to 36 h, then
# 3-hourly from 42 h to 192 h. Precipitation probabilities and 6-hour
# accumulations only exist every 6 hours past the hourly section. Off-synoptic
# cycles end each section earlier by (cycle hour % 6) hours so their long-range
# valid times line up with the synoptic cycles'.
NBM_SYNOPTIC_CYCLE_HOURS = 6
NBM_HOURLY_LAST_LEAD = 36
NBM_LONG_FIRST_LEAD = 42
NBM_LAST_LEAD = 192
NBM_LONG_STEP_HOURS = 3
NBM_LONG_PRECIP_STEP_HOURS = 6
# The long-range PACCUM values are 6-hour totals
NBM_LONG_ACCUM_HOURS = 6
# Hourly output axis length: run + 1 h to run + 193 h
NBM_HOURLY_OUTPUT_HOURS = 192

# Main forecast fields. PWTHER (predominant weather) is intentionally excluded
# because NBM v5 drops it intermittently; it is not used by the API, so its slot
# is filled with NaN during processing to keep the array shape unchanged.
NBM_MATCH_STRINGS = (
    r":((DPT|TMP|APTMP|RH):2 m above ground:.*fcst:$)"
    r"|:((PTYPE):surface:.*)"
    r"|(:(GUST|WIND|WDIR):10 m above ground:.*fcst:$)"
    r"|(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:$)"
    # This gets the correct surface parameter
    r"|:((TCDC|VIS|DSWRF|CAPE):surface:.*fcst:$)"
)
# 1-hour precipitation probabilities (hourly section)
NBM_PPROB_1H_MATCH = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|[0-9]*0-[0-9]{1,2}1|[0-9]*1-[0-9]{1,2}2|[0-9]*2-[0-9]{1,2}3|[0-9]*3-[0-9]{1,2}4|[0-9]*4-[0-9]{1,2}5|[0-9]*5-[0-9]{1,2}6|[0-9]*6-[0-9]{1,2}7|[0-9]*7-[0-9]{1,2}8|[0-9]*8-[0-9]{1,2}9|[0-9]*9-[0-9]{1,2}0).*fcst:prob.*)"
# 6-hour precipitation probabilities (long-range section)
NBM_PPROB_6H_MATCH = r":APCP:surface:(0-6|[0-9]*0-[0-9]{1,2}6|[0-9]*1-[0-9]{1,2}7|[0-9]*2-[0-9]{1,2}8|[0-9]*3-[0-9]{1,2}9|[0-9]*4-[0-9]{1,2}0|[0-9]*5-[0-9]{1,2}1|[0-9]*6-[0-9]{1,2}2|[0-9]*7-[0-9]{1,2}3|[0-9]*8-[0-9]{1,2}4|[0-9]*9-[0-9]{1,2}5).*fcst:prob"
# 1-hour precipitation accumulations (hourly section)
NBM_PACCUM_1H_MATCH = r"(:APCP:surface:(0-1|1-2|2-3|3-4|4-5|5-6|6-7|7-8|8-9|9-10|\d*0-\d{1,2}1|\d*1-\d{1,2}2|\d*2-\d{1,2}3|\d*3-\d{1,2}4|\d*4-\d{1,2}5|\d*5-\d{1,2}6|\d*6-\d{1,2}7|\d*7-\d{1,2}8|\d*8-\d{1,2}9|\d*9-\d{1,2}0).*fcst:$)"
# 6-hour precipitation accumulations (long-range section)
NBM_PACCUM_6H_MATCH = r"(:APCP:surface:(0-6|\d*0-\d{1,2}6|\d*1-\d{1,2}7|\d*2-\d{1,2}8|\d*3-\d{1,2}9|\d*4-\d{1,2}0|\d*5-\d{1,2}1|\d*6-\d{1,2}2|\d*7-\d{1,2}3|\d*8-\d{1,2}4|\d*9-\d{1,2}5).*fcst:$)"
# Probability of more than 0.254 mm (0.01 in) in the PPROB NetCDF
NBM_PPROB_VAR = "APCP_prob_GT_0D254_prob_fcst_255_255_surface"

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so zarr_vars[i] is column i + 1 of the
# API's NBM index map.
zarr_vars = (
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

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/NBM")
forecast_process_path = forecast_process_dir + "/NBM_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/NBM")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
herbie_max_threads = positive_int_env("herbie_max_threads", 1)
herbie_download_retries = positive_int_env("herbie_download_retries", 3)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)


def _timing_log(message: str) -> None:
    """Log a stage timing line with a UTC timestamp for run-time monitoring."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info("[TIMING] %s %s", ts, message)


def _acquire_single_run_lock(lock_path: str):
    """Take a non-blocking process lock so overlapping ingests do not clobber temp data.

    Returns the open lock file (keep it open for the run), or None if another NBM
    ingest holds the lock.
    """
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock_file = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return None

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
    """Download one NBM subset with the shared bounded retries and file checks."""
    return download_and_validate_subset(
        model="nbm",
        product="co",
        search=search,
        dataset_name=dataset_name,
        base_time=base_time,
        run_date=run_date,
        forecast_hours=list(forecast_hours),
        priority=NBM_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=herbie_download_retries,
        herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
        herbie_kwargs={"max_threads": herbie_max_threads},
        download_max_threads=herbie_max_threads,
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
        # The probability and accumulation subsets are downloaded in two parts
        # and validated together in main().
        skip_wgrib2_validation=True,
    )


def _nbm_grib_to_netcdf(
    grib_list: list[str],
    file_tag: str,
    stage_tag: str,
    dataset_name: str,
    expected_valid_times: np.ndarray,
) -> str:
    """Merge GRIBs, rewrite them over the full grid and export one NetCDF.

    Raises if the NetCDF time axis is not the expected valid times.
    """
    merged_grib_path = forecast_process_path + file_tag + "_wgrib2_merged.grib2"
    order_grib_path = forecast_process_path + file_tag + "_wgrib2_merged_order.grib"
    nc_path = forecast_process_path + file_tag + "_wgrib2_merged.nc"

    stage_start = time.perf_counter()
    substage_start = time.perf_counter()
    run_checked(
        "cat " + " ".join(grib_list) + f" | {wgrib2_path} - -grib {merged_grib_path}",
        f"{dataset_name} wgrib2 GRIB merge",
    )
    _timing_log(
        f"NBM_FORECAST stage={stage_tag}_merge_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )

    substage_start = time.perf_counter()
    run_checked(
        f"{wgrib2_path} {merged_grib_path} -ijsmall_grib {NBM_IJSMALL_GRID} "
        f"{order_grib_path}",
        f"{dataset_name} wgrib2 grid reorder",
    )
    _timing_log(
        f"NBM_FORECAST stage={stage_tag}_reorder_grib elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    os.remove(merged_grib_path)

    substage_start = time.perf_counter()
    run_checked(
        f"{wgrib2_path} {order_grib_path} -set_ext_name 1 -netcdf {nc_path}",
        f"{dataset_name} wgrib2 NetCDF conversion",
    )
    os.remove(order_grib_path)
    _timing_log(
        f"NBM_FORECAST stage={stage_tag}_write_netcdf elapsed={time.perf_counter() - substage_start:.2f}s"
    )
    _timing_log(
        f"NBM_FORECAST stage={stage_tag}_to_netcdf elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    with xr.open_dataset(nc_path) as converted:
        nc_times = converted["time"].values
    if not np.array_equal(nc_times, expected_valid_times):
        raise RuntimeError(
            f"{dataset_name} times {nc_times} do not match the expected valid "
            f"times {expected_valid_times}"
        )
    return nc_path


def main() -> None:
    """Run the NBM forecast ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()

    # The final write used dask's default thread pool (one thread per CPU).
    zarr_store_workers = configure_zarr_writes(default_workers=os.cpu_count() or 2)

    process_chunk = CHUNK_SIZES["NBM"]
    final_chunk = FINAL_CHUNK_SIZES["NBM"]

    # The processing directory is shared between runs, and an NBM run can outlast
    # the hourly schedule, so hold the lock until main() returns.
    lock_path = forecast_process_dir + ".lock"
    ingest_lock_handle = _acquire_single_run_lock(lock_path)
    if ingest_lock_handle is None:
        logger.info(
            "Another NBM ingest process is already running; skipping this run "
            "(lock: %s)",
            lock_path,
        )
        return

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

    # %% Determine the model run: the newest complete hourly cycle
    latest_run = Herbie_latest(
        model="nbm",
        n=5,
        freq="1h",
        fxx=NBM_LATEST_CHECK_HOURS,
        product="co",
        verbose=False,
        priority=NBM_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "NBM.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to NBM, ending")
        return
    logger.info("Processing NBM run %s", base_time)
    _timing_log(f"NBM_FORECAST run_start base_time={base_time}")
    forecast_iter_start = time.perf_counter()

    # %% Forecast hours: hourly section, then 3-hourly (6-hourly for precipitation
    # probabilities and accumulations), shortened for off-synoptic cycles
    cycle_offset_hours = base_time.hour % NBM_SYNOPTIC_CYCLE_HOURS
    nbm_range1 = range(1, NBM_HOURLY_LAST_LEAD + 1 - cycle_offset_hours)
    nbm_range2 = range(
        NBM_LONG_FIRST_LEAD - cycle_offset_hours,
        NBM_LAST_LEAD + 1 - cycle_offset_hours,
        NBM_LONG_STEP_HOURS,
    )
    nbm_range2_6h = range(
        NBM_LONG_FIRST_LEAD - cycle_offset_hours,
        NBM_LAST_LEAD + 1 - cycle_offset_hours,
        NBM_LONG_PRECIP_STEP_HOURS,
    )
    nbm_range = [*nbm_range1, *nbm_range2]
    nbm_precip_range = [*nbm_range1, *nbm_range2_6h]

    run_time = pd.Timestamp(base_time)
    forecast_valid_times = (run_time + pd.to_timedelta(nbm_range, unit="h")).values
    precip_valid_times = (run_time + pd.to_timedelta(nbm_precip_range, unit="h")).values

    # %% Download, validate and convert the main forecast fields
    stage_start = time.perf_counter()
    grib_list = _download_nbm_subset(
        nbm_range, NBM_MATCH_STRINGS, "NBM main forecast", base_time=base_time
    )
    _timing_log(
        f"NBM_FORECAST stage=download_main elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    stage_start = time.perf_counter()
    check_grib_stats(grib_list, wgrib2_path, "NBM main forecast")
    _timing_log(
        f"NBM_FORECAST stage=validate_main elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    main_nc_path = _nbm_grib_to_netcdf(
        grib_list, "", "main", "NBM main", forecast_valid_times
    )

    # %% Download, validate and convert the precipitation probabilities (PPROB)
    stage_start = time.perf_counter()
    gribList1 = _download_nbm_subset(
        nbm_range1,
        NBM_PPROB_1H_MATCH,
        "NBM hourly precipitation probability",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_pprob_hourly elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    stage_start = time.perf_counter()
    gribList2 = _download_nbm_subset(
        nbm_range2_6h,
        NBM_PPROB_6H_MATCH,
        "NBM six-hour precipitation probability",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_pprob_6hour elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    grib_list = gribList1 + gribList2
    stage_start = time.perf_counter()
    check_grib_stats(grib_list, wgrib2_path, "NBM PPROB")
    _timing_log(
        f"NBM_FORECAST stage=validate_pprob elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    prob_nc_path = _nbm_grib_to_netcdf(
        grib_list, "_prob", "pprob", "NBM PPROB", precip_valid_times
    )

    # %% Download, validate and convert the precipitation accumulations (PACCUM)
    stage_start = time.perf_counter()
    gribList1 = _download_nbm_subset(
        nbm_range1,
        NBM_PACCUM_1H_MATCH,
        "NBM hourly precipitation accumulation",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_paccum_hourly elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    stage_start = time.perf_counter()
    gribList2 = _download_nbm_subset(
        nbm_range2_6h,
        NBM_PACCUM_6H_MATCH,
        "NBM six-hour precipitation accumulation",
        base_time=base_time,
    )
    _timing_log(
        "NBM_FORECAST "
        f"stage=download_paccum_6hour elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    grib_list = gribList1 + gribList2
    stage_start = time.perf_counter()
    check_grib_stats(grib_list, wgrib2_path, "NBM PACCUM")
    _timing_log(
        f"NBM_FORECAST stage=validate_paccum elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    accum_nc_path = _nbm_grib_to_netcdf(
        grib_list, "_accum", "paccum", "NBM PACCUM", precip_valid_times
    )

    # %% Correct NBM-specific quirks and write one processing zarr per variable
    # The NetCDF is read through netCDF4 + dask rather than xarray's open_mfdataset,
    # which works significantly better here. It is not chunked on load since wgrib2's
    # NetCDF chunking is unknown; each variable is rechunked before writing.
    stage_start = time.perf_counter()

    # Time-only copy of the main dataset, used to align PPROB and PACCUM (which
    # are 6-hourly past the hourly section) onto the main forecast time axis
    xarray_forecast_base = xr.open_mfdataset(main_nc_path)
    xarray_forecast_base = xarray_forecast_base.drop_vars(
        list(xarray_forecast_base.data_vars)
    )

    # Hourly target axis from the first lead (run + 1 h); the store holds no history
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0],
        end=forecast_valid_times[0] + pd.Timedelta(hours=NBM_HOURLY_OUTPUT_HOURS),
        freq="h",
    )
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    ncForecast = nc.Dataset(main_nc_path)
    ncForecast.set_auto_mask(False)
    # Check that the NetCDF times (Unix seconds) are the expected forecast times
    validate_stacked_time_alignment(stacked_timesUnix, ncForecast["time"][:])
    forecast_var_zarr_dir = forecast_process_path + "_zarrs"
    if os.path.exists(forecast_var_zarr_dir):
        shutil.rmtree(forecast_var_zarr_dir)
    os.makedirs(forecast_var_zarr_dir, exist_ok=True)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        for dask_var in zarr_vars:
            var_stage_start = time.perf_counter()
            if dask_var == "PPROB":
                xarray_forecast = xarray_forecast_base.copy()
                xarray_forecast["PPROB"] = xr.open_mfdataset(prob_nc_path)[
                    NBM_PPROB_VAR
                ]
                # Fill the 3-hourly steps between the 6-hourly probabilities with
                # the probability of the 6-hour window they fall in
                xarray_forecast["PPROB"] = xarray_forecast["PPROB"].bfill(dim="time")
                daskArray = xarray_forecast["PPROB"].data
                del xarray_forecast

            elif dask_var == "PACCUM":
                xarray_forecast = xarray_forecast_base.copy()
                xarray_forecast["PACCUM"] = xr.open_mfdataset(accum_nc_path)[
                    "APCP_surface"
                ]
                # Past the hourly section the accumulations are 6-hour totals;
                # convert them to hourly amounts
                xarray_forecast["PACCUM"][len(nbm_range1) :, :, :] = (
                    xarray_forecast["PACCUM"][len(nbm_range1) :, :, :]
                    / NBM_LONG_ACCUM_HOURS
                )
                # Fill the 3-hourly steps between the 6-hour totals with the
                # hourly amount of the window they fall in
                xarray_forecast["PACCUM"] = xarray_forecast["PACCUM"].bfill(dim="time")
                daskArray = xarray_forecast["PACCUM"].data
                del xarray_forecast

            elif dask_var == "PWTHER_surfaceMreserved":
                # PWTHER is not downloaded (see NBM_MATCH_STRINGS); fill its slot
                # with NaN so the downstream array shape is preserved.
                ref_shape = ncForecast["TMP_2maboveground"].shape
                daskArray = da.full(ref_shape, np.nan, dtype=np.float32)

            else:
                daskArray = da.from_array(ncForecast[dask_var], lock=True)

            if len(daskArray) != len(nbm_range):
                raise RuntimeError(
                    f"NBM run {base_time} {dask_var} has {len(daskArray)} time "
                    f"steps, expected {len(nbm_range)}"
                )

            daskArray = daskArray.rechunk(
                chunks=(len(nbm_range), process_chunk, process_chunk)
            )

            forecast_var_zarr_path = forecast_var_zarr_dir + "/" + dask_var + ".zarr"
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

    ncForecast.close()
    xarray_forecast_base.close()
    del daskArray, xarray_forecast_base

    # The wgrib2 NetCDFs are large and no longer needed
    cleanup_stage_start = time.perf_counter()
    os.remove(main_nc_path)
    os.remove(prob_nc_path)
    os.remove(accum_nc_path)
    _timing_log(
        f"NBM_FORECAST stage=forecast_temp_cleanup elapsed={time.perf_counter() - cleanup_stage_start:.2f}s"
    )
    logger.info(
        "NBM run %s downloaded and converted in %.1f s", base_time, time.time() - T0
    )

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in zarr_vars:
        forecast_var_zarr_path = forecast_var_zarr_dir + "/" + dask_var + ".zarr"
        if not os.path.exists(forecast_var_zarr_path):
            raise FileNotFoundError(
                "Missing forecast zarr variable store: "
                f"{forecast_var_zarr_path}. This usually means another ingest "
                "run cleared the shared processing directory while this run was active."
            )
        daskForecastArray = da.from_zarr(forecast_var_zarr_path, inline_array=True)
        daskVarArrayList.append(
            daskForecastArray.rechunk(
                (len(stacked_timesUnix), process_chunk, process_chunk)
            ).astype("float32")
        )

    daskVarArrayListMergeNaN = mask_invalid_data(da.stack(daskVarArrayList, axis=0))

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
    stage_start = time.perf_counter()
    daskVarArrayListMergeNaN.to_zarr(
        forecast_process_path + "_stack.zarr", overwrite=True, compute=True
    )
    _timing_log(
        f"NBM_FORECAST stage=stack_to_zarr elapsed={time.perf_counter() - stage_start:.2f}s"
    )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Interpolate to hourly and write the final store
    stage_start = time.perf_counter()
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
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/NBM.zarr",
        final_chunk,
        FINAL_SHARD_SIZES["NBM"],
        zarr_store_workers,
        zarr_vars,
        times=hourly_timesUnix,
    )
    _timing_log(
        f"NBM_FORECAST stage=interpolate_and_write elapsed={time.perf_counter() - stage_start:.2f}s"
    )

    # %% Publish the stores, then the run time the API uses to detect them
    stage_start = time.perf_counter()
    publish_ingest_run(
        ["NBM.zarr"],
        base_time,
        "NBM.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )
    _timing_log(
        f"NBM_FORECAST stage=publish_output elapsed={time.perf_counter() - stage_start:.2f}s"
    )
    _timing_log(
        f"NBM_FORECAST total_elapsed={time.perf_counter() - forecast_iter_start:.2f}s"
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    ingest_lock_handle.close()
    logger.info("NBM run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
