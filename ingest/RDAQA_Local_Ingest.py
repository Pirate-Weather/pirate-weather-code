# ruff: noqa: E402
"""ECCC RDAQA air quality analysis ingest with short-term history.

Downloads the hourly preliminary analyses of Environment and Climate Change
Canada's Regional Deterministic Air Quality Analysis from the MSC Datamart. RDAQA blends RAQDPS with hourly AirNow and Canadian monitor
observations on RAQDPS's 10 km rotated lat/lon grid; the preliminary analysis is
posted about 40 minutes after its hour (the final one about 2 h after, with the
same values to 0.1 ug/m3 in nearly every hour checked).

Each hour holds RAQDPS's pollutants, converted the same way (PM2.5/PM10 to
ug/m3, NO2/O3/SO2 to ppb), and is archived as its own
``RDAQA_Hist_<time>.zarr``. The last ``HISTORY_PERIODS["RDAQA"]`` hours are then
merged into ``RDAQA.zarr``, published with ``RDAQA.time.pickle`` (the newest
hour). The grid is RAQDPS's, so the API locates cells with
``RAQDPS.lat_lon.pickle``.

The Datamart sometimes posts an hour late or not at all. Such an hour is stored
as NaN and retried on every run while it is in the window, so the store keeps
one row per hour (the API reads the rows by position) and RAQDPS or SILAM fill
it until it arrives.
"""

import logging
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Configure ecCodes before importing cfgrib-backed readers.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECCODES_DIR = WORKSPACE_ROOT / ".build" / "ingest-test" / "toolchain"
os.environ.setdefault("ECCODES_DIR", str(DEFAULT_ECCODES_DIR))
os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    f"{DEFAULT_ECCODES_DIR / 'lib'}:{DEFAULT_ECCODES_DIR / 'lib64'}",
)

import dask.array as da
import numpy as np
import requests
import xarray as xr
import zarr

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    close_store,
    configure_zarr_writes,
    history_store_exists,
    map_history_stores,
    mask_invalid_data,
    open_history_arrays,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_history_store,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)
from API.raqdps_utils import (
    RAQDPS_VARIABLES,
    as_float32_array,
    convert_to_output_units,
    herbie_naive_utc,
    normalize_utc,
)
from API.rdaqa_herbie_template import build_rdaqa_url

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", "In a future version")

logger = logging.getLogger(__name__)

# %% Model constants
# RAQDPS's pollutants, order and units, so the API's RDAQA index map is RAQDPS's.
# The API reader puts the valid times (the store's time attribute) in column 0,
# so zarr_vars[i] is column i + 1 of the API's RDAQA index map.
zarr_vars = RAQDPS_VARIABLES
# Prelim is posted ~40 min after its hour against ~2 h for final, and missed
# fewer hours in the 2026-11-17/18 checks, with the same values.
RDAQA_ANALYSIS = "prelim"
# How far back from the current hour to look for the newest complete hour
RDAQA_LATEST_SEARCH_HOURS = 6
# Hours of history before the newest analysis kept in the published store
his_period = HISTORY_PERIODS["RDAQA"]
HOURLY_STORE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
RDAQA_STORE_NAME = "RDAQA.zarr"
RDAQA_TIME_PICKLE = "RDAQA.time.pickle"

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/RDAQA")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RDAQA")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/Prod/RDAQA")
save_type = os.getenv("save_type", default="Download")

aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

RDAQA_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RDAQA_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)
# Files downloaded at once, as the other MSC ingests (msc_download_fields)
RDAQA_DOWNLOAD_WORKERS = positive_int_env("msc_download_fields", 4)
# (connect, read) seconds; a file is ~0.5 MB
RDAQA_REQUEST_TIMEOUT = (10, 60)


class NotPosted(Exception):
    """The Datamart answered 404: the file is not posted (yet)."""


def hourly_store_name(valid_time: datetime) -> str:
    """Name of the history store holding one analysis hour."""
    return f"RDAQA_Hist_{valid_time:{HOURLY_STORE_TIMESTAMP_FORMAT}}.zarr"


def window_hours(base_time: datetime) -> list[datetime]:
    """The hours of the published store, oldest first, ending at ``base_time``."""
    return [base_time - timedelta(hours=h) for h in range(his_period, -1, -1)]


def rdaqa_url(valid_time, variable) -> str:
    return build_rdaqa_url(normalize_utc(valid_time), variable, RDAQA_ANALYSIS)


def with_retries(action, description):
    """Run ``action()``, retrying network errors and non-404 HTTP errors.

    ``NotPosted`` (a 404) is raised at once. Only a 404 means a file is not
    posted: Herbie's existence check also returns "not found" when a request
    to the Datamart fails, which stored posted hours as missing.
    """
    for attempt in range(1, RDAQA_DOWNLOAD_RETRIES + 1):
        try:
            return action()
        except NotPosted:
            raise
        except (OSError, requests.RequestException) as e:
            if attempt == RDAQA_DOWNLOAD_RETRIES:
                raise RuntimeError(
                    f"RDAQA request failed after {attempt} attempts: {description}"
                ) from e
            sleep_s = RDAQA_RETRY_SLEEP_SECONDS * attempt
            logger.warning(
                "RDAQA request attempt %d/%d failed for %s (%s). Retrying in %ss",
                attempt,
                RDAQA_DOWNLOAD_RETRIES,
                description,
                e,
                sleep_s,
            )
            time.sleep(sleep_s)


def _raise_for_status(response: requests.Response, url: str) -> None:
    if response.status_code == 404:
        raise NotPosted(url)
    response.raise_for_status()


def file_is_posted(url: str) -> bool:
    """True if the Datamart has ``url`` (False only on a 404)."""

    def head():
        response = requests.head(url, timeout=RDAQA_REQUEST_TIMEOUT)
        _raise_for_status(response, url)

    try:
        with_retries(head, url)
    except NotPosted:
        return False
    return True


def get_latest_rdaqa_hour(now: datetime | None = None) -> datetime:
    """The newest hour whose every pollutant is posted.

    Requiring every pollutant keeps a half-posted hour from becoming the
    published run time, after which the next run would see nothing new.
    """
    now = normalize_utc(now or datetime.now(UTC))
    for back in range(RDAQA_LATEST_SEARCH_HOURS + 1):
        candidate = now - timedelta(hours=back)
        if all(file_is_posted(rdaqa_url(candidate, v)) for v in zarr_vars):
            logger.info("Found latest RDAQA hour: %s", candidate)
            return candidate
    raise RuntimeError(
        f"No complete RDAQA {RDAQA_ANALYSIS} hour in the last "
        f"{RDAQA_LATEST_SEARCH_HOURS} hours"
    )


def download_rdaqa_file(valid_time, variable) -> str | None:
    """Download one RDAQA GRIB2 file into ``tmp_dir``, with bounded retries.

    Returns None when the file is not on the Datamart (a late or skipped hour),
    and raises when it is there but cannot be downloaded.
    """
    url = rdaqa_url(valid_time, variable)
    path = os.path.join(tmp_dir, url.rsplit("/", 1)[1])

    def get():
        # Written to a temporary name and renamed, so a failed attempt never
        # leaves a partial file behind
        partial = path + ".part"
        with requests.get(url, timeout=RDAQA_REQUEST_TIMEOUT, stream=True) as response:
            _raise_for_status(response, url)
            with open(partial, "wb") as fh:
                fh.writelines(response.iter_content(chunk_size=1 << 20))
        os.replace(partial, path)
        return path

    try:
        return with_retries(get, url)
    except NotPosted:
        return None


def read_rdaqa_grib(local_grib_path, variable, valid_time) -> np.ndarray:
    """Read one RDAQA field in output units as float32, checking its hour."""
    ds = xr.open_dataset(
        local_grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
    )
    try:
        grib_time = np.datetime64(ds["valid_time"].values, "s")
        expected = np.datetime64(herbie_naive_utc(valid_time), "s")
        if grib_time != expected:
            raise RuntimeError(
                f"RDAQA {variable} file {local_grib_path} is for {grib_time}, "
                f"expected {expected}"
            )
        data_array = ds[next(iter(ds.data_vars))].astype(np.float32)
        return as_float32_array(convert_to_output_units(data_array, variable).values)
    finally:
        ds.close()


def load_hours(valid_times) -> dict:
    """Download and read every pollutant of each hour.

    Returns ``{hour: {variable: field}}`` for the hours whose every pollutant
    is posted; an hour with any file missing is left out (and logged).
    """
    pairs = [(t, v) for t in valid_times for v in zarr_vars]
    with ThreadPoolExecutor(max_workers=RDAQA_DOWNLOAD_WORKERS) as pool:
        paths = list(pool.map(lambda pair: download_rdaqa_file(*pair), pairs))

    by_hour = {t: {} for t in valid_times}
    for (valid_time, variable), path in zip(pairs, paths, strict=True):
        if path is not None:
            by_hour[valid_time][variable] = path

    hours = {}
    for valid_time, files in by_hour.items():
        missing = [v for v in zarr_vars if v not in files]
        if missing:
            logger.warning(
                "RDAQA hour %s is not posted yet (missing %s); stored as NaN",
                valid_time,
                missing,
            )
            continue
        fields = {
            variable: read_rdaqa_grib(files[variable], variable, valid_time)
            for variable in zarr_vars
        }
        shapes = {field.shape for field in fields.values()}
        if len(shapes) != 1:
            raise RuntimeError(f"RDAQA hour {valid_time} has mixed grids {shapes}")
        hours[valid_time] = fields
    return hours


def write_hour_store(valid_time, fields, path) -> None:
    """Write one hour as a Zarr group: one ``(1, y, x)`` array per pollutant,
    each one object, with the hour (Unix seconds) as its ``time`` attribute."""
    shutil.rmtree(path, ignore_errors=True)
    store = zarr.storage.LocalStore(path)
    group = zarr.open_group(store, mode="w")
    unix_time = int(normalize_utc(valid_time).timestamp())
    for variable in zarr_vars:
        field = fields[variable]
        array = group.create_array(
            variable,
            shape=(1, *field.shape),
            chunks=(1, *field.shape),
            dtype="float32",
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
            attributes={"time": unix_time},
        )
        array[0] = field
    close_store(store)


def archive_new_hours(base_time, history_remote) -> None:
    """Archive every hour of the window that is not in the history yet."""
    hours = window_hours(base_time)
    archived = map_history_stores(
        lambda t: history_store_exists(history_remote, hourly_store_name(t)), hours
    )
    to_fetch = [t for t, done in zip(hours, archived, strict=True) if not done]
    logger.info("Fetching %d RDAQA hours not archived yet", len(to_fetch))

    tmp_store = os.path.join(forecast_process_dir, "RDAQA_Hist_TMP.zarr")
    for valid_time, fields in load_hours(to_fetch).items():
        write_hour_store(valid_time, fields, tmp_store)
        publish_history_store(tmp_store, history_remote, hourly_store_name(valid_time))


def merge_history(base_time, history_remote, local_dir):
    """Stack the window's archived hours into a ``(variable, time, y, x)`` array.

    Hours with no archived store are NaN, so the rows stay one per hour.
    Returns the array and the hours' Unix times.
    """
    hours = window_hours(base_time)
    names = [hourly_store_name(t) for t in hours]
    present = map_history_stores(
        lambda name: history_store_exists(history_remote, name), names
    )
    if not present[-1]:
        raise RuntimeError(f"Newest RDAQA hour {base_time} was not archived")
    present_names = [name for name, ok in zip(names, present, strict=True) if ok]
    stores = dict(
        zip(
            present_names,
            open_history_arrays(history_remote, present_names, zarr_vars, local_dir),
            strict=True,
        )
    )

    times_unix = np.array([int(normalize_utc(t).timestamp()) for t in hours])
    grid_shape = stores[names[-1]][zarr_vars[0]].shape[-2:]
    stacked = np.full(
        (len(zarr_vars), len(hours), *grid_shape), np.nan, dtype=np.float32
    )
    for time_index, (name, expected_time) in enumerate(
        zip(names, times_unix, strict=True)
    ):
        store = stores.get(name)
        if store is None:
            continue
        for var_index, variable in enumerate(zarr_vars):
            array = store[variable]
            if array.attrs.get("time") != expected_time:
                raise RuntimeError(
                    f"{name} {variable} holds time {array.attrs.get('time')}, "
                    f"expected {expected_time}"
                )
            if array.shape[-2:] != grid_shape:
                raise RuntimeError(
                    f"{name} grid {array.shape[-2:]} differs from {grid_shape}"
                )
            stacked[var_index, time_index] = array[0]

    missing = [t for t, ok in zip(hours, present, strict=True) if not ok]
    if missing:
        logger.warning(
            "%d of %d RDAQA hours are missing and stored as NaN: %s",
            len(missing),
            len(hours),
            ", ".join(f"{t:%Y-%m-%d %HZ}" for t in missing),
        )
    return stacked, times_unix


def already_processed(base_time, forecast_remote) -> bool:
    """True if the published store already ends at ``base_time``."""
    previous = read_published_run_time(forecast_remote, RDAQA_TIME_PICKLE)
    return previous is not None and normalize_utc(previous) >= base_time


def main() -> None:
    """Run the RDAQA ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    zarr_store_workers = configure_zarr_writes()

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )
    history_remote = open_remote(
        historic_path, save_type, aws_access_key_id, aws_secret_access_key
    )

    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    T0 = time.time()

    try:
        # %% The newest hour with every pollutant posted
        base_time = get_latest_rdaqa_hour()
        if already_processed(base_time, forecast_remote):
            logger.info("RDAQA store already ends at %s. Exiting.", base_time)
            return
        logger.info("Processing RDAQA through %s", base_time)

        # %% Archive the hours not archived yet, retrying late ones
        archive_new_hours(base_time, history_remote)

        # %% Stack the window, pad, and write the final store
        stacked, times_unix = merge_history(
            base_time, history_remote, os.path.join(forecast_process_dir, "History")
        )
        process_chunk = CHUNK_SIZES["RDAQA"]
        final_chunk = FINAL_CHUNK_SIZES["RDAQA"]
        stacked_dask = da.from_array(
            stacked,
            chunks=(len(zarr_vars), len(times_unix), process_chunk, process_chunk),
        )
        write_final_zarr(
            pad_to_chunk_size(mask_invalid_data(stacked_dask), final_chunk),
            os.path.join(forecast_process_dir, RDAQA_STORE_NAME),
            final_chunk,
            FINAL_SHARD_SIZES["RDAQA"],
            zarr_store_workers,
            zarr_vars,
            times=times_unix,
        )

        # %% Publish the store, then the run time the API uses to detect it
        publish_ingest_run(
            [RDAQA_STORE_NAME],
            base_time,
            RDAQA_TIME_PICKLE,
            process_dir=forecast_process_dir,
            remote=forecast_remote,
        )
        logger.info("RDAQA through %s published in %.1f s", base_time, time.time() - T0)
    finally:
        # %% Clean up
        shutil.rmtree(forecast_process_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
