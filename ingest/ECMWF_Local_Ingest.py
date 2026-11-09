"""ECMWF IFS forecast ingest.

Downloads the newest 12-hourly ECMWF IFS open-data cycle (0.1 degree / 9 km global
grid, 3-hourly leads to 144 h then 6-hourly to 240 h): the deterministic ``oper``
surface fields plus the ``enfo`` ensemble total precipitation, which is reduced
to precipitation probability, mean and standard deviation. The forecast is
merged with the first 12 h of the previous 48 h of runs (short-term history,
archived per run), interpolated to an hourly axis, given a derived 2 m relative
humidity and published as ``ECMWF.zarr`` plus ``ECMWF.time.pickle``. The store
starts 48 h before the run time.

Only the IFS has a 0.1 degree feed; AIFS stays at 0.25 degrees.

Alexander Rey, March 2025
"""

import logging
import os
import re
import shutil
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest

from API.api_utils import relative_humidity_from_dewpoint_c
from API.constants.shared_const import (
    ENSEMBLE_MEMBER_COUNTS,
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    HISTORY_PERIODS,
    INGEST_VERSION_STR,
    KELVIN_TO_CELSIUS,
    member_mask_var_names,
)
from API.constants.unit_const import MM_PER_M
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    FORECAST_LEAD_RANGES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    configure_zarr_writes,
    download_history_stores,
    history_store_exists,
    interp_time_take_blend,
    map_history_stores,
    open_history_store,
    open_remote,
    pack_member_mask,
    pad_to_chunk_size,
    positive_int_env,
    publish_history_store,
    publish_ingest_run,
    read_published_run_time,
    shard_whole_variables,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# Herbie picks the first source holding each lead. Google is the fastest (0.2 s/MB,
# against 0.8 for ECMWF) but lags: the 240 h 0.1 degree ensemble file of the 00Z run
# of 2026-11-04 reached ECMWF at 09:40, AWS at 09:56 and Google at 12:27. AWS is last
# because it answered most GETs with 503 SlowDown; Azure no longer serves the feed.
ECMWF_SOURCE_PRIORITY = ["google", "ecmwf", "aws"]
# Named explicitly: Herbie's default depends on the date, and the store layout
# and the API's grid lookup (ECMWF_LATS/ECMWF_LONS) depend on this resolution.
ECMWF_RESOLUTION = "0p1"
# IFS open data only runs to 240 h at 00 and 12 UTC
ECMWF_RUN_FREQ = "12h"
ECMWF_FORECAST_HOURS = [
    *FORECAST_LEAD_RANGES["ECMWF_IFS_1"],
    *FORECAST_LEAD_RANGES["ECMWF_IFS_2"],
]

# Each history run contributes its first 12 h (3-hourly) until the next run
ECMWF_HISTORY_RUN_STEP_HOURS = 12
ECMWF_HISTORY_LEAD_HOURS = list(range(3, ECMWF_HISTORY_RUN_STEP_HOURS + 1, 3))
HISTORY_PERIOD_HOURS = HISTORY_PERIODS["ECMWF"]
# Named for the grid: a 0.25 degree store from before the 0.1 degree switch must
# not be found and merged. Old stores age out through the prune-history job.
HIST_STORE_PREFIX = "ECMWF_Hist_0p1_"

# Deterministic fields: 2 m temperature/dew point, 10 m wind, precipitation
# rate/accumulation/type, mean sea level pressure and total cloud cover
ECMWF_OPER_MATCH_STRINGS = (
    "(:(2d|2t):)|(:(10u|10v):)|(:(ptype|tprate|tp):)|(:(msl):)|(:(tcc):)"
)
# Ensemble total precipitation (accumulated from the run start, in m)
ECMWF_ENFO_MATCH_STRING = r":(tp:sfc:\d+):"

# cfgrib can only open one hypercube per call, so the oper files are opened once
# per level type and merged (in this order).
IFS_OPER_CFGRIB_GROUPS = (
    {"typeOfLevel": "heightAboveGround", "level": [2]},
    {"typeOfLevel": "heightAboveGround", "level": [10]},
    {"typeOfLevel": "surface"},
    {"typeOfLevel": "meanSea"},
    {"typeOfLevel": "entireAtmosphere"},
)

# Time chunk of the intermediate merged zarrs. A multi-step chunk makes dask hold
# every lead of the whole grid while it assembles tiles (~20 GB at 0.1 degrees).
PROCESS_TIME_CHUNK = 1

# Ensemble members in a run (control plus 50 perturbed); the run fails before
# publishing when the GRIB holds a different number
ECMWF_IFS_MEMBER_COUNT = ENSEMBLE_MEMBER_COUNTS["ECMWF_IFS"]
# Per-hour member bitmask words (pack_member_mask), from which the API gets the
# share of members with precipitation at any hour of a day
MEMBER_MASK_VARS = member_mask_var_names(ECMWF_IFS_MEMBER_COUNT)

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so zarr_vars[i] is column i + 1 of the
# API's ECMWF index map. The mask words come last so the earlier columns keep
# their positions.
zarr_vars = (
    "msl",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "tprate",
    "tp",
    "ptype",
    "tcc",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
    "rh",
    *MEMBER_MASK_VARS,
)
# rh is derived after the hourly interpolation (append_relative_humidity); the
# rest are read from GRIB and kept in the intermediate and history stores.
PROCESS_VARS = tuple(var for var in zarr_vars if var != "rh")
# The history stores carry no member masks: the API only uses them for hours
# still ahead, so the history runs skip the work and older stores stay valid
HISTORY_VARS = tuple(var for var in PROCESS_VARS if var not in MEMBER_MASK_VARS)
# Categorical precipitation type is interpolated with nearest neighbour; the
# member masks take the lead whose accumulation interval contains the hour
NEAREST_INTERP_VARS = ("ptype",)

# Peak memory scales with this count: each thread holds a final-store shard, which
# spans every variable and forecast hour. 4 matches the ingest instance's vCPUs.
DEFAULT_ZARR_WORKERS = 4
DEFAULT_ZARR_CONCURRENCY = 4

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/ECMWF")
forecast_process_path = forecast_process_dir + "/ECMWF_Process"
hist_process_path = forecast_process_dir + "/ECMWF_Historic"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/ECMWF")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/ECMWF")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

ECMWF_MAX_THREADS = positive_int_env("ecmwf_download_threads", 4)
ECMWF_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
ECMWF_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)


def download_ecmwf_subset(
    run_time: pd.Timestamp,
    forecast_hours: list[int],
    product: str,
    search: str,
) -> list[str]:
    """Download and validate one IFS GRIB subset with bounded retries."""
    return download_and_validate_subset(
        model="ifs",
        product=product,
        search=search,
        dataset_name=f"ECMWF IFS {product} {run_time}",
        base_time=run_time,
        forecast_hours=list(forecast_hours),
        wgrib2_exe=wgrib2_path.rstrip(),
        herbie_save_dir=tmp_dir,
        herbie_download_retries=ECMWF_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=ECMWF_RETRY_SLEEP_SECONDS,
        priority=ECMWF_SOURCE_PRIORITY,
        herbie_kwargs={
            "max_threads": ECMWF_MAX_THREADS,
            "resolution": ECMWF_RESOLUTION,
        },
        download_max_threads=ECMWF_MAX_THREADS,
    )


def open_ifs_cfgrib(grib_paths: list[str], filter_by_keys=None) -> xr.Dataset:
    """Open one cfgrib message group across all lead files, sorted by step."""
    backend_kwargs = {"filter_by_keys": filter_by_keys} if filter_by_keys else {}
    return xr.open_mfdataset(
        grib_paths,
        engine="cfgrib",
        combine="nested",
        concat_dim="step",
        decode_timedelta=False,
        join="outer",
        coords="minimal",
        compat="override",
        backend_kwargs=backend_kwargs,
    ).sortby("step")


def open_ifs_oper(grib_paths: list[str]) -> xr.Dataset:
    """Open and merge the deterministic IFS fields from every level type."""
    ifs_mf = xr.merge(
        [open_ifs_cfgrib(grib_paths, group) for group in IFS_OPER_CFGRIB_GROUPS],
        compat="override",
    )
    # Convert from 0-1 to 0-100 for consistency with AIFS data
    ifs_mf["tcc"] = ifs_mf["tcc"] * 100
    return ifs_mf


def write_ifs_ensemble_precip_stats(
    grib_paths: list[str],
    lead_hours: list[int],
    zarr_path: str,
    member_count: int = ECMWF_IFS_MEMBER_COUNT,
    member_masks: bool = True,
    run_label: str = "ECMWF IFS ensemble",
) -> xr.Dataset:
    """Reduce the IFS ensemble total precipitation to hourly-rate statistics
    and, for the forecast run, per-lead member bitmasks (``member_masks``; the
    history runs skip them, see HISTORY_VARS).

    At 0.1 degrees one lead is 50 members x 6.5 M cells (1.3 GB as float32), and
    a lazy multi-file cfgrib open peaked at 11 GB for a single lead. So each lead
    is decoded, reduced and appended to ``zarr_path`` on its own, holding only the
    previous lead's totals for the differencing.
    """
    paths_by_lead = {
        int(re.search(r"-(\d+)h-enfo-", os.path.basename(path)).group(1)): path
        for path in grib_paths
    }
    if sorted(paths_by_lead) != list(lead_hours):
        raise RuntimeError(
            f"IFS ensemble files cover leads {sorted(paths_by_lead)}, "
            f"expected {lead_hours}"
        )

    previous_total = None
    previous_lead = 0
    for lead in lead_hours:
        with xr.open_dataset(
            paths_by_lead[lead],
            engine="cfgrib",
            decode_timedelta=False,
            backend_kwargs={"indexpath": ""},
        ) as ens:
            total = ens["tp"].values.astype(np.float32, copy=False)
            latitude = ens["latitude"].values
            longitude = ens["longitude"].values
            if float(ens["step"]) != lead:
                raise RuntimeError(
                    f"{paths_by_lead[lead]} holds step {float(ens['step'])}, "
                    f"expected {lead}"
                )

        # tp accumulates from the run start: difference consecutive leads, keeping
        # the first lead's 0-to-lead total as is. Dividing by the interval length
        # (3 h, or 6 h after 144 h) gives an hourly rate in m/h.
        interval_total = total if previous_total is None else total - previous_total
        rate = interval_total / (lead - previous_lead)
        del interval_total
        previous_total = total
        previous_lead = lead

        # Fraction of members above the shared ensemble threshold (mm/h -> m/h),
        # and which members they are, for the daily probability
        wet_members = rate > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H / MM_PER_M
        precip_prob = wet_members.sum(axis=0, dtype=np.float32) / member_count
        mask_words = (
            pack_member_mask(wet_members, member_count, label=run_label)
            if member_masks
            else []
        )
        del wet_members

        dims = ("step", "latitude", "longitude")
        lead_stats = xr.Dataset(
            {
                "Precipitation_Prob": (dims, precip_prob[np.newaxis]),
                "APCP_StdDev": (dims, rate.std(axis=0)[np.newaxis]),
                "APCP_Mean": (dims, rate.mean(axis=0)[np.newaxis]),
                **{
                    name: (dims, word[np.newaxis])
                    for name, word in zip(
                        member_mask_var_names(member_count) if member_masks else (),
                        mask_words,
                        strict=True,
                    )
                },
            },
            coords={
                "step": [float(lead)],
                "latitude": latitude,
                "longitude": longitude,
            },
        )
        if lead == lead_hours[0]:
            chunks = (
                PROCESS_TIME_CHUNK,
                CHUNK_SIZES["ECMWF_IFS"],
                CHUNK_SIZES["ECMWF_IFS"],
            )
            lead_stats.to_zarr(
                zarr_path,
                mode="w",
                consolidated=False,
                encoding={name: {"chunks": chunks} for name in lead_stats.data_vars},
            )
        else:
            # The grid was written with the first lead
            lead_stats.drop_vars(["latitude", "longitude"]).to_zarr(
                zarr_path, append_dim="step", consolidated=False
            )
        # Only the totals carry over; these would otherwise sit in memory (1.3 GB
        # each) while the next lead decodes
        del total, rate, precip_prob, mask_words, lead_stats

    return xr.open_zarr(zarr_path, consolidated=False)


def append_relative_humidity(stacked: da.Array) -> da.Array:
    """Insert 2 m relative humidity (percent, like GFS and HRRR) at its
    ``zarr_vars`` position, ahead of the member mask words.

    Derived from the hourly temperature and dew point (K in the stored fields) so
    each stored hour equals what the API used to derive per request, and reads no
    longer pay for the saturation vapour pressure calculation.
    """

    def relative_humidity_percent(t2m_k, d2m_k):
        return (
            relative_humidity_from_dewpoint_c(
                t2m_k - KELVIN_TO_CELSIUS, d2m_k - KELVIN_TO_CELSIUS
            )
            * 100.0
        ).astype(np.float32)

    relative_humidity = da.map_blocks(
        relative_humidity_percent,
        stacked[PROCESS_VARS.index("t2m")],
        stacked[PROCESS_VARS.index("d2m")],
        dtype="float32",
    )
    rh_index = zarr_vars.index("rh")
    return da.concatenate(
        [stacked[:rh_index], relative_humidity[np.newaxis], stacked[rh_index:]],
        axis=0,
    )


def step_to_valid_time(
    ds: xr.Dataset, run_time: pd.Timestamp, lead_hours: list[int], dataset_name: str
) -> xr.Dataset:
    """Index ``ds`` by valid time (run time + step) and check it is complete."""
    valid_times = pd.to_datetime(ds.time.values) + pd.to_timedelta(
        ds.step.values, unit="h"
    )
    expected_times = pd.Timestamp(run_time) + pd.to_timedelta(lead_hours, unit="h")
    if not np.array_equal(valid_times.values, expected_times.values):
        raise RuntimeError(
            f"{dataset_name} valid times {valid_times.values} do not match the "
            f"expected leads {lead_hours} from {run_time}"
        )
    return (
        ds.assign_coords(time=("step", valid_times))
        .swap_dims({"step": "time"})
        .drop_vars("step")
    )


def write_process_zarr(
    ds: xr.Dataset,
    zarr_path: str,
    zarr_store_workers: int,
    shard_variables: bool = False,
):
    """Chunk and write an intermediate dataset to a local zarr; history stores
    get one shard per variable (``shard_whole_variables``)."""
    ds = ds.chunk(
        chunks={
            "time": PROCESS_TIME_CHUNK,
            "latitude": CHUNK_SIZES["ECMWF_IFS"],
            "longitude": CHUNK_SIZES["ECMWF_IFS"],
        }
    )
    encoding = None
    if shard_variables:
        ds, encoding = shard_whole_variables(ds)
    with (
        ProgressBar(),
        dask.config.set(scheduler="threads", num_workers=zarr_store_workers),
    ):
        ds.to_zarr(
            zarr_path,
            mode="w",
            write_empty_chunks=True,
            consolidated=False,
            compute=True,
            encoding=encoding,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )


def main() -> None:
    """Run the ECMWF IFS forecast and history ingest."""
    logging.basicConfig(level=logging.INFO)

    zarr_store_workers = configure_zarr_writes(
        DEFAULT_ZARR_WORKERS, DEFAULT_ZARR_CONCURRENCY
    )

    process_chunk = CHUNK_SIZES["ECMWF_IFS"]
    final_chunk = FINAL_CHUNK_SIZES["ECMWF"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )
    history_remote = open_remote(
        historic_path, save_type, aws_access_key_id, aws_secret_access_key
    )

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists. The
    # ensemble's last file is published after the deterministic one, so it decides.
    latest_run = HerbieLatest(
        model="ifs",
        n=3,
        freq=ECMWF_RUN_FREQ,
        fxx=ECMWF_FORECAST_HOURS[-1],
        product="enfo",
        resolution=ECMWF_RESOLUTION,
        verbose=True,
        priority=ECMWF_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "ECMWF.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to ECMWF, ending")
        return
    logger.info("Processing ECMWF run %s", base_time)

    # %% Download and convert the forecast: ensemble precipitation statistics
    ens_paths = download_ecmwf_subset(
        base_time, ECMWF_FORECAST_HOURS, "enfo", ECMWF_ENFO_MATCH_STRING
    )
    xr_ensoOut = write_ifs_ensemble_precip_stats(
        ens_paths,
        ECMWF_FORECAST_HOURS,
        forecast_process_path + "_ens.zarr",
        run_label=f"ECMWF IFS ensemble run {base_time}",
    )

    # Deterministic fields
    ifs_paths = download_ecmwf_subset(
        base_time, ECMWF_FORECAST_HOURS, "oper", ECMWF_OPER_MATCH_STRINGS
    )
    ifs_mf = open_ifs_oper(ifs_paths)

    xarray_forecast_merged = step_to_valid_time(
        xr.merge([ifs_mf, xr_ensoOut], compat="override", join="outer"),
        base_time,
        ECMWF_FORECAST_HOURS,
        "ECMWF forecast",
    )

    # Hourly target axis from the start of the history period to the last lead
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    new_hourly_time = pd.date_range(
        start=pd.to_datetime(xarray_forecast_merged.time[0].values)
        - pd.Timedelta(hours=HISTORY_PERIOD_HOURS),
        end=xarray_forecast_merged.time[-1].values,
        freq="h",
    )
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    write_process_zarr(
        xarray_forecast_merged,
        forecast_process_path + "_merged.zarr",
        zarr_store_workers,
    )
    del xarray_forecast_merged, ifs_mf, xr_ensoOut

    logger.info("ECMWF forecast converted in %.1f s", time.time() - T0)

    # %% History: archive the first 12 h of each earlier run not yet processed
    history_offsets = range(HISTORY_PERIOD_HOURS, 0, -ECMWF_HISTORY_RUN_STEP_HOURS)
    for hours_back in history_offsets:
        hist_run_time = base_time - pd.Timedelta(hours=hours_back)
        hist_timestamp = hist_run_time.strftime("%Y%m%dT%H%M%SZ")
        hist_store_name = f"{HIST_STORE_PREFIX}{hist_timestamp}.zarr"
        if history_store_exists(history_remote, hist_store_name):
            logger.info("ECMWF history already stored, skipping %s", hist_store_name)
            continue

        logger.info("Downloading ECMWF history run %s", hist_timestamp)

        ifs_his_paths = download_ecmwf_subset(
            hist_run_time, ECMWF_HISTORY_LEAD_HOURS, "oper", ECMWF_OPER_MATCH_STRINGS
        )
        ifs_his_mf = open_ifs_oper(ifs_his_paths)

        ens_his_paths = download_ecmwf_subset(
            hist_run_time, ECMWF_HISTORY_LEAD_HOURS, "enfo", ECMWF_ENFO_MATCH_STRING
        )
        # No member masks: the API only uses them for hours still ahead
        xr_enso_hisOut = write_ifs_ensemble_precip_stats(
            ens_his_paths,
            ECMWF_HISTORY_LEAD_HOURS,
            hist_process_path + "_ens.zarr",
            member_masks=False,
            run_label=f"ECMWF IFS ensemble history run {hist_timestamp}",
        )

        xarray_hist_merged = step_to_valid_time(
            xr.merge([ifs_his_mf, xr_enso_hisOut], compat="override", join="outer"),
            hist_run_time,
            ECMWF_HISTORY_LEAD_HOURS,
            f"ECMWF history run {hist_timestamp}",
        )

        hist_tmp_zarr_path = hist_process_path + "_ECMWF_Hist_TMP.zarr"
        write_process_zarr(
            xarray_hist_merged,
            hist_tmp_zarr_path,
            zarr_store_workers,
            shard_variables=True,
        )
        del xarray_hist_merged

        publish_history_store(hist_tmp_zarr_path, history_remote, hist_store_name)

        logger.info("Saved ECMWF history run %s", hist_timestamp)

    # %% Merge the history and forecast datasets
    # Every history run was stored above, so a missing one means it was lost;
    # dropping it would let the hourly interpolation bridge the gap with
    # invented values, so open_history_store fails instead. The history stores
    # keep their own time coordinate, which the concatenation below indexes by.
    hist_store_names = [
        HIST_STORE_PREFIX
        + (base_time - pd.Timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M%SZ")
        + ".zarr"
        for hours_back in history_offsets
    ]
    hist_local_paths = download_history_stores(
        history_remote, hist_store_names, forecast_process_dir + "/History"
    )

    def open_history_dataset(name_and_path):
        name, local_path = name_and_path
        open_history_store(history_remote, name, ("time", *HISTORY_VARS), local_path)
        ds = xr.open_zarr(local_path, consolidated=False)
        # Member masks are NaN over the history period (see HISTORY_VARS)
        for var in MEMBER_MASK_VARS:
            if var not in ds.data_vars:
                ds[var] = xr.full_like(ds["Precipitation_Prob"], np.nan)
        return ds

    hist = map_history_stores(
        open_history_dataset, list(zip(hist_store_names, hist_local_paths, strict=True))
    )
    fcst = xr.open_zarr(f"{forecast_process_path}_merged.zarr", consolidated=False)
    ds = xr.concat(
        [*hist, fcst],
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )

    # The API reads variables by position, so a missing one must not shift the rest
    missing_vars = [v for v in PROCESS_VARS if v not in ds.data_vars]
    if missing_vars:
        raise RuntimeError(f"ECMWF run {base_time} is missing {missing_vars}")

    # Out-of-range values become NaN; the mask words are integers up to 2^24,
    # above the range's ceiling
    for var in PROCESS_VARS:
        ds_clip = ds[var]
        if var not in MEMBER_MASK_VARS and np.issubdtype(ds_clip.dtype, np.number):
            ds[var] = ds_clip.where(
                (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
            )

    # The stacked (native 3/6-hourly) times the interpolation starts from
    stacked_timesUnix = (ds.time.values - unix_epoch) / one_second

    # %% Stack into the (variable, time, y, x) layout the API reads
    ds_stack = ds[list(PROCESS_VARS)].to_array(dim="var", name="var")
    ds_chunk = ds_stack.chunk(
        {
            "var": 1,
            "time": len(stacked_timesUnix),
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    # Interim zarr save of the stacked array. Not necessary for local, but speeds
    # things up on S3
    with (
        ProgressBar(),
        dask.config.set(scheduler="threads", num_workers=zarr_store_workers),
    ):
        ds_chunk.to_zarr(
            forecast_process_path + "_stack.zarr",
            mode="w",
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    daskVarArrayStackDisk = da.from_zarr(
        forecast_process_path + "_stack.zarr", component="__xarray_dataarray_variable__"
    )

    # %% Interpolate to hourly and write the final store
    daskVarArrayStackDiskInterp = interp_time_take_blend(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        nearest_vars=[PROCESS_VARS.index(v) for v in NEAREST_INTERP_VARS],
        next_lead_vars=[PROCESS_VARS.index(v) for v in MEMBER_MASK_VARS],
        dtype="float32",
        fill_value=np.nan,
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        append_relative_humidity(daskVarArrayStackDiskInterp), final_chunk
    )

    ecmwf_store_name = "ECMWF.zarr"
    with ProgressBar():
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + ecmwf_store_name,
            final_chunk,
            FINAL_SHARD_SIZES["ECMWF"],
            zarr_store_workers,
            zarr_vars,
            times=new_hourly_time.values.astype("datetime64[s]").astype("int64"),
        )

    # %% Publish the stores, then the run time the API uses to detect them
    publish_ingest_run(
        [ecmwf_store_name],
        base_time,
        "ECMWF.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("ECMWF run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
