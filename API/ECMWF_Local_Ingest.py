"""ECMWF IFS forecast ingest.

Downloads the newest 12-hourly ECMWF IFS open-data cycle (0.25 degree global
grid, 3-hourly leads to 144 h then 6-hourly to 240 h): the deterministic ``oper``
surface fields plus the ``enfo`` ensemble total precipitation, which is reduced
to precipitation probability, mean and standard deviation. The forecast is
merged with the first 12 h of the previous 48 h of runs (short-term history,
archived per run), interpolated to an hourly axis and published as
``ECMWF.zarr`` and ``ECMWF_Maps.zarr`` plus ``ECMWF.time.pickle``. The store
starts 48 h before the run time.

Alexander Rey, March 2025
"""

import logging
import os
import shutil
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from herbie import HerbieLatest

from API.constants.shared_const import (
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    HISTORY_PERIODS,
    INGEST_VERSION_STR,
)
from API.constants.unit_const import MM_PER_M
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    archive_tmp_zarr_and_upload,
    close_store,
    configure_zarr_writes,
    download_extract_historic_archive,
    interp_time_take_blend,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
ECMWF_SOURCE_PRIORITY = ["google", "ecmwf", "aws", "azure"]
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

# Time chunk of the intermediate merged zarrs
PROCESS_TIME_CHUNK = 64

# Variables in the published store; the order is the API's ECMWF index map
zarr_vars = (
    "time",
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
)
# Categorical precipitation type is interpolated with nearest neighbour
NEAREST_INTERP_VARS = ("ptype",)

# Map store: temperature, wind, precipitation rate/accumulation/type
MAP_VARS = ("time", "t2m", "u10", "v10", "tprate", "tp", "ptype")
MAP_VAR_INDICES = [zarr_vars.index(map_var) for map_var in MAP_VARS]
MAP_TIME_STEPS = 36
MAP_CHUNK_SIZE = 100
# Index into the stacked (history + forecast, native 3/6-hourly) time axis
MAP_FIRST_STEP = 36

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
        herbie_kwargs={"max_threads": ECMWF_MAX_THREADS},
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


def ifs_ensemble_precip_stats(
    grib_paths: list[str], lead_hours: list[int]
) -> xr.Dataset:
    """Reduce the IFS ensemble total precipitation to hourly-rate statistics."""
    ens_mf = open_ifs_cfgrib(grib_paths)

    # tp accumulates from the run start: difference consecutive leads, keeping
    # the first lead's 0-to-lead total as is
    ens_mf["tpd"] = ens_mf["tp"].diff(dim="step")
    ens_mf["tpd"] = xr.where(
        ens_mf.step == ens_mf.step.isel(step=0),
        ens_mf["tp"].isel(step=0),
        ens_mf["tpd"],
    )

    # Divide each interval total by its length (3 h, or 6 h after 144 h) to get
    # an hourly rate in m/h
    interval_hours = xr.DataArray(
        np.diff([0, *lead_hours]).astype(np.float32), dims="step"
    )
    ens_mf["tpd"] = ens_mf["tpd"] / interval_hours

    # Fraction of members above the shared ensemble threshold (mm/h -> m/h)
    precip_prob = (ens_mf["tpd"] > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H / MM_PER_M).sum(
        dim="number"
    ) / ens_mf.sizes["number"]

    return xr.Dataset(
        {
            "Precipitation_Prob": precip_prob,
            "APCP_StdDev": ens_mf["tpd"].std(dim="number"),
            "APCP_Mean": ens_mf["tpd"].mean(dim="number"),
        },
        coords={
            "step": ens_mf["step"],
            "latitude": ens_mf["latitude"],
            "longitude": ens_mf["longitude"],
        },
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


def write_process_zarr(ds: xr.Dataset, zarr_path: str, zarr_store_workers: int):
    """Chunk and write an intermediate dataset to a local zarr."""
    ds = ds.chunk(
        chunks={
            "time": PROCESS_TIME_CHUNK,
            "latitude": CHUNK_SIZES["ECMWF"],
            "longitude": CHUNK_SIZES["ECMWF"],
        }
    )
    with (
        ProgressBar(),
        dask.config.set(scheduler="threads", num_workers=zarr_store_workers),
    ):
        ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )


def main() -> None:
    """Run the ECMWF IFS forecast and history ingest."""
    logging.basicConfig(level=logging.INFO)

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["ECMWF"]
    final_chunk = FINAL_CHUNK_SIZES["ECMWF"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)
        os.makedirs(historic_path, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="ifs",
        n=3,
        freq=ECMWF_RUN_FREQ,
        fxx=ECMWF_FORECAST_HOURS[-1],
        product="oper",
        verbose=True,
        priority=ECMWF_SOURCE_PRIORITY,
        save_dir=tmp_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/ECMWF.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to ECMWF, ending")
        return
    logger.info("Processing ECMWF run %s", base_time)

    # %% Download and convert the forecast: ensemble precipitation statistics
    ens_paths = download_ecmwf_subset(
        base_time, ECMWF_FORECAST_HOURS, "enfo", ECMWF_ENFO_MATCH_STRING
    )
    xr_ensoOut = ifs_ensemble_precip_stats(ens_paths, ECMWF_FORECAST_HOURS)

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
        if save_type == "S3":
            s3_path = historic_path + "/ECMWF_Hist_v3" + hist_timestamp + ".zarr.tar.gz"
            if s3.exists(s3_path.replace(".tar.gz", ".done")):
                logger.info("ECMWF history already in S3, skipping %s", s3_path)
                continue
        else:
            local_path = historic_path + "/ECMWF_Hist_v3" + hist_timestamp + ".zarr"
            if os.path.exists(local_path.replace(".zarr", ".done")):
                logger.info("ECMWF history already local, skipping %s", local_path)
                continue

        logger.info("Downloading ECMWF history run %s", hist_timestamp)

        ifs_his_paths = download_ecmwf_subset(
            hist_run_time, ECMWF_HISTORY_LEAD_HOURS, "oper", ECMWF_OPER_MATCH_STRINGS
        )
        ifs_his_mf = open_ifs_oper(ifs_his_paths)

        ens_his_paths = download_ecmwf_subset(
            hist_run_time, ECMWF_HISTORY_LEAD_HOURS, "enfo", ECMWF_ENFO_MATCH_STRING
        )
        xr_enso_hisOut = ifs_ensemble_precip_stats(
            ens_his_paths, ECMWF_HISTORY_LEAD_HOURS
        )

        xarray_hist_merged = step_to_valid_time(
            xr.merge([ifs_his_mf, xr_enso_hisOut], compat="override", join="outer"),
            hist_run_time,
            ECMWF_HISTORY_LEAD_HOURS,
            f"ECMWF history run {hist_timestamp}",
        )

        hist_tmp_zarr_path = hist_process_path + "_ECMWF_Hist_TMP.zarr"
        write_process_zarr(xarray_hist_merged, hist_tmp_zarr_path, zarr_store_workers)
        del xarray_hist_merged

        # Archive the run and write its done marker
        if save_type == "S3":
            archive_tmp_zarr_and_upload(
                tmp_zarr_path=hist_tmp_zarr_path,
                s3_path=s3_path,
                archive_member_name="ECMWF_Hist.zarr",
                s3=s3,
            )
        else:
            os.rename(hist_tmp_zarr_path, local_path)
            with open(local_path.replace(".zarr", ".done"), "w") as f:
                f.write("Done")

        logger.info("Saved ECMWF history run %s", hist_timestamp)

    # %% Merge the history and forecast datasets
    if save_type == "S3":
        local_temp_dir = forecast_process_path + "_s3_temp_downloads"
        os.makedirs(local_temp_dir, exist_ok=True)
        ncLocalWorking_paths = []
        for hours_back in history_offsets:
            hist_timestamp = (base_time - pd.Timedelta(hours=hours_back)).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            extracted_path = download_extract_historic_archive(
                s3=s3,
                historic_path=historic_path,
                final_zarr_name=f"ECMWF_Hist_v3{hist_timestamp}.zarr",
                extracted_store_name="ECMWF_Hist.zarr",
                local_temp_dir=local_temp_dir,
                expected_vars=zarr_vars,
            )
            # Every history run was archived above, so a missing one means the
            # archive was lost; dropping it would let the hourly interpolation
            # bridge the gap with invented values.
            if extracted_path is None:
                raise RuntimeError(
                    f"ECMWF history archive ECMWF_Hist_v3{hist_timestamp} is "
                    f"missing from {historic_path}"
                )
            ncLocalWorking_paths.append(extracted_path)
    else:
        ncLocalWorking_paths = [
            historic_path
            + "/ECMWF_Hist_v3"
            + (base_time - pd.Timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M%SZ")
            + ".zarr"
            for hours_back in history_offsets
        ]

    hist = [xr.open_zarr(p, consolidated=False) for p in ncLocalWorking_paths]
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
    missing_vars = [v for v in zarr_vars[1:] if v not in ds.data_vars]
    if missing_vars:
        raise RuntimeError(f"ECMWF run {base_time} is missing {missing_vars}")

    # Out-of-range values become NaN
    for var in zarr_vars[1:]:
        ds_clip = ds[var]
        if np.issubdtype(ds_clip.dtype, np.number):
            ds[var] = ds_clip.where(
                (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
            )

    # The stacked (native 3/6-hourly) times the interpolation starts from
    stacked_timesUnix = (ds.time.values - unix_epoch) / one_second

    # %% Stack into the (variable, time, y, x) layout the API reads
    ds_rename = ds.rename({"time": "stacked_time"})
    # The API reads time from the store like any other variable
    ds_rename["time"] = (
        ((ds_rename["stacked_time"] - unix_epoch) / one_second)
        .astype("float32")
        .expand_dims(latitude=ds_rename.latitude, longitude=ds_rename.longitude)
        .transpose("stacked_time", "latitude", "longitude")
    )
    ds_stack = ds_rename[list(zarr_vars)].to_array(dim="var", name="var")
    ds_chunk = ds_stack.chunk(
        {
            "var": 1,
            "stacked_time": len(stacked_timesUnix),
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
        nearest_vars=[zarr_vars.index(v) for v in NEAREST_INTERP_VARS],
        dtype="float32",
        fill_value=np.nan,
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    ecmwf_store_name = "ECMWF.zarr.zip" if save_type == "S3" else "ECMWF.zarr"
    with ProgressBar():
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + ecmwf_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # %% Map store: one named array per map variable, all map times in one chunk
    # and 100x100 spatial chunks for fast tile reads. Taken from the native
    # (not interpolated) stacked axis: with the full 16-step history, steps
    # 36-71 cover run +63 h to +192 h.
    daskVarArrayStackDisk_maps = pad_to_chunk_size(
        daskVarArrayStackDisk, MAP_CHUNK_SIZE
    )

    ecmwf_maps_store_name = (
        "ECMWF_Maps.zarr.zip" if save_type == "S3" else "ECMWF_Maps.zarr"
    )
    if save_type == "S3":
        # Uncompressed zip like the main store; the arrays are Blosc-compressed
        zarr_store_maps = zarr.storage.ZipStore(
            forecast_process_dir + "/" + ecmwf_maps_store_name, mode="a", compression=0
        )
    else:
        zarr_store_maps = zarr.storage.LocalStore(
            forecast_process_dir + "/" + ecmwf_maps_store_name
        )

    for var_idx in MAP_VAR_INDICES:
        zarr_array = zarr.create_array(
            store=zarr_store_maps,
            name=zarr_vars[var_idx],
            shape=(
                MAP_TIME_STEPS,
                daskVarArrayStackDisk_maps.shape[2],
                daskVarArrayStackDisk_maps.shape[3],
            ),
            chunks=(MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            dtype="float32",
        )
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            da.rechunk(
                daskVarArrayStackDisk_maps[
                    var_idx, MAP_FIRST_STEP : MAP_FIRST_STEP + MAP_TIME_STEPS, :, :
                ],
                (MAP_TIME_STEPS, MAP_CHUNK_SIZE, MAP_CHUNK_SIZE),
            ).to_zarr(zarr_array, overwrite=True, compute=True)
        logger.info("Created ECMWF map data for %s", zarr_vars[var_idx])

    close_store(zarr_store_maps)

    # %% Publish the stores, then the run time the API uses to detect them
    publish_ingest_run(
        [ecmwf_store_name, ecmwf_maps_store_name],
        base_time,
        "ECMWF.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("ECMWF run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
