# %% Script to contain the helper functions as part of the data ingest for Pirate Weather
# Alexander Rey. July 2025

import logging
import os
import pickle
import re
import resource
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import cartopy.crs as ccrs
import dask.array as da
import numpy as np
import requests
import xarray as xr

# Import atmospheric calculation constants
from API.constants.shared_const import (
    MISSING_DATA,
    REFC_THRESHOLD,
)

# Logging
logger = logging.getLogger(__name__)

# Shared ingest constants
CHUNK_SIZES = {
    "NBM": 200,
    "HRRR": 200,
    "HRRR_6H": 200,
    "GFS": 100,
    "GEFS": 200,
    "ECMWF": 200,
    "NBM_Fire": 200,
    "RTMA": 200,
    "DWD": 200,
    "HRDPS": 200,
    "RDPS": 200,
    "REPS": 200,
    "GDPS": 200,
    "GEPS": 200,
    "RDAQA": 250,
    "RAQDPS": 250,
    "SILAM": 200,
    "URMA": 200,
}

FINAL_CHUNK_SIZES = {
    "NBM": 3,
    "HRRR": 5,
    "HRRR_6H": 5,
    "GFS": 3,
    "GEFS": 3,
    "HGEFS": 3,
    "ECMWF": 3,
    "NBM_Fire": 5,
    "RTMA": 25,
    # Hourly RTMA keeps 48 rows; small tiles keep a point read to one small chunk
    "RTMA_HOURLY": 5,
    "DWD": 5,
    # These divide evenly into the Canadian weather ingests' 200-cell processing tiles.
    "HRDPS": 5,
    "RDPS": 5,
    "REPS": 5,
    "GDPS": 5,
    "GEPS": 5,
    "RDAQA": 25,
    "RAQDPS": 25,
    "SILAM": 5,
    "URMA": 25,
}

FORECAST_LEAD_RANGES = {
    "GFS_1": list(range(1, 121)),
    "GFS_2": list(range(123, 241, 3)),
    "GEFS": list(range(3, 241, 3)),
    "GDPS_1": list(range(1, 84, 1)),
    "GDPS_2": list(range(84, 241, 3)),
    "NBM_FIRE": list(range(6, 192, 6)),
    "HRRR_1H": list(range(1, 19)),
    "HRRR_6H": list(range(18, 49)),
    "ECMWF_AIFS": list(range(6, 241, 6)),
    "ECMWF_IFS_1": list(range(3, 144, 3)),
    "ECMWF_IFS_2": list(range(144, 241, 6)),
    "AIGFS": list(range(6, 241, 6)),
    "AIGEFS": list(range(6, 241, 6)),
    "HRDPS": list(range(1, 49)),
    "RDPS": list(range(1, 85)),
    "REPS": list(range(3, 73, 3)),
    "GEPS": [*range(3, 193, 3), *range(198, 241, 6)],
}

# Radius, in km, used for DWD model nearest-neighbor selection
DWD_RADIUS = 50

VALID_DATA_MIN = -100
VALID_DATA_MAX = 120000

# Cached CMC GRIB downloads older than this are pruned at the start of a run.
# REPS history reuses files from cycles up to ~48 h old, so keep a margin.
DOWNLOAD_CACHE_MAX_AGE_HOURS = 72


def deaccumulate_to_hourly_rate(
    values: da.Array | np.ndarray,
    forecast_hours: Iterable[float],
    time_axis: int = 0,
) -> da.Array:
    """Convert run-total accumulations to nonnegative hourly rates."""
    forecast_hours = np.asarray(tuple(forecast_hours), dtype="float64")
    if forecast_hours.ndim != 1 or len(forecast_hours) == 0:
        raise ValueError("forecast_hours must be a nonempty one-dimensional sequence.")

    interval_hours = np.diff(np.insert(forecast_hours, 0, 0.0))
    if not np.all(np.isfinite(interval_hours)) or np.any(interval_hours <= 0):
        raise ValueError(
            "forecast_hours must be finite, positive, and strictly increasing."
        )

    accumulated = da.asarray(values)
    normalized_time_axis = time_axis % accumulated.ndim
    if accumulated.shape[normalized_time_axis] != len(forecast_hours):
        raise ValueError(
            "values time-axis length must match the number of forecast hours."
        )

    broadcast_shape = [1] * accumulated.ndim
    broadcast_shape[normalized_time_axis] = len(interval_hours)
    increments = da.diff(accumulated, axis=normalized_time_axis, prepend=0)
    hourly_rate = increments / interval_hours.reshape(broadcast_shape)
    return da.maximum(hourly_rate, 0)


def deaccumulate_energy_to_flux(
    values: da.Array | np.ndarray,
    times: np.ndarray,
    time_axis: int = 0,
) -> da.Array:
    """Convert accumulated energy into nonnegative interval-average flux.

    The first accumulation interval is assumed to have the same duration as
    the first interval represented in ``times``. This matches forecast series
    that begin at their first positive lead time.
    """
    times = np.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if len(times) < 2:
        raise ValueError("At least two times are required for de-accumulation.")

    time_deltas_seconds = (np.diff(times) / np.timedelta64(1, "s")).astype("float64")
    if not np.all(np.isfinite(time_deltas_seconds)) or np.any(time_deltas_seconds <= 0):
        raise ValueError("times must be finite and strictly increasing.")

    interval_seconds = np.insert(time_deltas_seconds, 0, time_deltas_seconds[0])
    accumulated = da.asarray(values)
    normalized_time_axis = time_axis % accumulated.ndim
    if accumulated.shape[normalized_time_axis] != len(times):
        raise ValueError(
            "values time-axis length must match the number of entries in times."
        )

    broadcast_shape = [1] * accumulated.ndim
    broadcast_shape[normalized_time_axis] = len(interval_seconds)
    increments = da.diff(accumulated, axis=normalized_time_axis, prepend=0)
    flux = increments / interval_seconds.reshape(broadcast_shape)
    return da.maximum(flux, 0)


def run_command(command: str, encoding: str = "utf-8") -> subprocess.CompletedProcess:
    """Execute a command string without shell=True, including a single pipe."""
    command = command.strip()
    if not command:
        raise ValueError("Cannot execute an empty command string")

    if "|" not in command:
        return subprocess.run(
            shlex.split(command),
            capture_output=True,
            encoding=encoding,
            check=False,
        )

    left, right = command.split("|", maxsplit=1)
    left_args = shlex.split(left)
    right_args = shlex.split(right)
    if not left_args or not right_args:
        raise ValueError(f"Invalid piped command: {command!r}")

    left_proc = subprocess.Popen(
        left_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        result = subprocess.run(
            right_args,
            stdin=left_proc.stdout,
            capture_output=True,
            encoding=encoding,
            check=False,
        )
    finally:
        if left_proc.stdout is not None:
            left_proc.stdout.close()

    _, left_stderr = left_proc.communicate()
    if left_proc.returncode not in (0, None):
        left_err_text = left_stderr.decode(encoding, errors="replace")
        combined_stderr = left_err_text
        if result.stderr:
            combined_stderr = f"{combined_stderr}\n{result.stderr}"
        return subprocess.CompletedProcess(
            args=result.args,
            returncode=left_proc.returncode,
            stdout=result.stdout,
            stderr=combined_stderr,
        )

    return result


def tune_nofile_limit(target: int = 65535) -> None:
    """Increase soft nofile limit when possible to avoid zarr write exhaustion."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard == resource.RLIM_INFINITY:
            new_soft = max(soft, target)
        else:
            new_soft = min(max(soft, target), hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            logger.info("Raised nofile soft limit from %s to %s", soft, new_soft)
    except (ValueError, OSError) as exc:
        logger.warning("Unable to tune nofile limit: %s", exc)


def positive_int_env(name: str, default: int) -> int:
    """Read an integer env var and fall back to a safe positive default."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%s must be >= 1; using %s", name, value, default)
        return default
    return value


def configure_zarr_limits(
    requested_workers: int, requested_async_concurrency: int
) -> tuple[int, int]:
    """Clamp zarr write parallelism so local stores do not exhaust open files."""
    workers = requested_workers
    async_concurrency = requested_async_concurrency
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY:
            # Reserve descriptor headroom for downloads/netcdf/wgrib IO.
            fd_budget = max(soft - 256, 1)
            worker_cap = max(1, fd_budget // 256)
            async_cap = max(1, fd_budget // 512)
            workers = min(workers, worker_cap)
            async_concurrency = min(async_concurrency, async_cap)
    except (ValueError, OSError) as exc:
        logger.warning("Unable to read nofile limit for zarr tuning: %s", exc)

    async_concurrency = min(async_concurrency, workers)
    if workers < requested_workers:
        logger.info(
            "Clamped zarr_store_workers from %s to %s based on nofile limits",
            requested_workers,
            workers,
        )
    if async_concurrency < requested_async_concurrency:
        logger.info(
            "Clamped zarr_async_concurrency from %s to %s based on nofile limits",
            requested_async_concurrency,
            async_concurrency,
        )

    # Local import keeps helper lightweight for non-zarr callers.
    import zarr

    zarr.config.set({"async.concurrency": async_concurrency})
    logger.info(
        "Configured zarr write parallelism: workers=%s, async_concurrency=%s",
        workers,
        async_concurrency,
    )
    return workers, async_concurrency


def configure_zarr_writes(
    default_workers: int = 2, default_async_concurrency: int = 2
) -> int:
    """Set up zarr write parallelism for an ingest run and return the worker count.

    Reads ``zarr_store_workers`` / ``zarr_async_concurrency`` from the environment,
    raises the open-file limit, and clamps both so large local stores do not run
    out of file descriptors.
    """
    tune_nofile_limit()
    zarr_store_workers, _ = configure_zarr_limits(
        positive_int_env("zarr_store_workers", default_workers),
        positive_int_env("zarr_async_concurrency", default_async_concurrency),
    )
    return zarr_store_workers


def make_herbie_save_dir(tmp_dir: str, prefix: str = "herbie") -> str:
    """Create a stable Herbie cache directory."""
    save_dir = os.path.join(tmp_dir, prefix)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


class _HerbieRequests:
    """Pace NOMADS probes, index reads, and GRIB byte-range requests together."""

    RequestException = requests.RequestException

    def __init__(self, interval_s: float, timeout_s: int) -> None:
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._lock = Lock()
        self._last_request: float | None = None

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", (10, self.timeout_s))
        request = getattr(requests, method)
        if (urlparse(str(url)).hostname or "").lower() != "nomads.ncep.noaa.gov":
            return request(url, **kwargs)

        # A NOMADS rate-limit redirect must not become a successful HTML download.
        kwargs["allow_redirects"] = False
        with self._lock:
            if self._last_request is not None:
                delay = self.interval_s - (time.monotonic() - self._last_request)
                if delay > 0:
                    time.sleep(delay)
            self._last_request = time.monotonic()
            response = request(url, **kwargs)
        if 300 <= response.status_code < 400:
            response.close()
            raise requests.HTTPError(
                f"NOMADS returned HTTP {response.status_code} for {url}",
                response=response,
            )
        return response

    def get(self, url: str, **kwargs) -> requests.Response:
        return self._request("get", url, **kwargs)

    def head(self, url: str, **kwargs) -> requests.Response:
        return self._request("head", url, **kwargs)


def configure_herbie_request_timeouts(
    request_timeout_s: int | None = None,
    *,
    nomads_request_interval_s: float | None = None,
) -> None:
    """Bound Herbie request times and pace every Herbie request to NOMADS.

    NOMADS rate-limits by client IP across all ingests, answering aggressive
    clients with redirects to an HTML page (#772), so pacing is always on. The
    interval comes from ``nomads_request_interval_seconds`` (default 5 s) unless
    given; calling again without an interval keeps the pacing already set up.
    Requests to other sources (AWS, Google, MSC, ECMWF) are not slowed down.
    """
    if request_timeout_s is None:
        request_timeout_s = positive_int_env("herbie_request_timeout_seconds", 120)

    from herbie import core

    Herbie = core.Herbie
    if nomads_request_interval_s is not None and nomads_request_interval_s <= 0:
        raise ValueError("NOMADS request interval must be positive")
    if isinstance(core.requests, _HerbieRequests):
        if nomads_request_interval_s is not None:
            core.requests.interval_s = nomads_request_interval_s
        core.requests.timeout_s = request_timeout_s
    else:
        if nomads_request_interval_s is None:
            nomads_request_interval_s = positive_int_env(
                "nomads_request_interval_seconds", 5
            )
        core.requests = _HerbieRequests(nomads_request_interval_s, request_timeout_s)

    if getattr(Herbie, "_pirate_weather_request_timeout_s", None) == request_timeout_s:
        return

    def _check_grib_with_timeout(self, url: str, min_content_length: int = 10) -> bool:
        try:
            head = core.requests.head(url, timeout=(10, request_timeout_s))
        except requests.RequestException:
            return False

        if not head.ok:
            return False
        content_length = head.headers.get("Content-Length")
        return content_length is not None and int(content_length) > min_content_length

    def _check_idx_with_timeout(
        self, url: str, verbose: bool = False
    ) -> tuple[bool, str | None]:
        for suffix in self.IDX_SUFFIX:
            if Path(url).suffix in {".grb", ".grib", ".grb2", ".grib2"}:
                idx_url = url.rsplit(".", maxsplit=1)[0] + suffix
            else:
                idx_url = url + suffix

            try:
                hostname = (urlparse(idx_url).hostname or "").lower()
                if hostname == "blob.core.windows.net" or hostname.endswith(
                    ".blob.core.windows.net"
                ):
                    dl_url = (
                        "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
                        + idx_url
                    )
                    response = core.requests.get(
                        dl_url, timeout=(10, request_timeout_s)
                    )
                    idx_url = response.json()["href"]
                idx_exists = core.requests.head(
                    idx_url, timeout=(10, request_timeout_s)
                ).ok
            except requests.RequestException as exc:
                idx_exists = False
                if verbose:
                    print(f"Unable to get index file from {idx_url} due to error {exc}")

            if verbose:
                print(f"Herbie index URL: {idx_url}")
                print(f"Herbie index exists: {idx_exists}")
            if idx_exists:
                return idx_exists, idx_url

        return False, None

    Herbie._check_grib = _check_grib_with_timeout
    Herbie._check_idx = _check_idx_with_timeout
    Herbie._pirate_weather_request_timeout_s = request_timeout_s


def _is_remote_url(path_or_url: Any) -> bool:
    return str(path_or_url).startswith(("http://", "https://"))


def _download_full_herbie_file(
    herbie_ref,
    *,
    overwrite: bool,
    request_timeout_s: int,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download a full Herbie GRIB with request timeouts and atomic replacement."""
    out_file = Path(herbie_ref.get_localFilePath(None)).expanduser()
    if out_file.exists() and out_file.stat().st_size > 0 and not overwrite:
        return str(out_file)

    grib_source = getattr(herbie_ref, "grib", None)
    if grib_source is None:
        raise RuntimeError(f"GRIB2 file not found: {herbie_ref!r}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_name(f"{out_file.name}.tmp.{os.getpid()}")
    if tmp_file.exists():
        tmp_file.unlink()

    try:
        if _is_remote_url(grib_source):
            request_client = requests
            if (
                urlparse(str(grib_source)).hostname or ""
            ).lower() == "nomads.ncep.noaa.gov":
                from herbie import core

                request_client = core.requests
            with request_client.get(
                str(grib_source),
                stream=True,
                timeout=(10, request_timeout_s),
            ) as response:
                response.raise_for_status()
                with open(tmp_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
        else:
            source_path = Path(grib_source).expanduser()
            if source_path.resolve() == out_file.resolve():
                if out_file.exists() and out_file.stat().st_size > 0:
                    return str(out_file)
                raise RuntimeError(f"Local GRIB file is empty or missing: {out_file}")
            shutil.copyfile(source_path, tmp_file)

        if not tmp_file.exists() or tmp_file.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty GRIB file: {out_file}")

        tmp_file.replace(out_file)
        return str(out_file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise


def _download_full_herbie_files(
    herbie_obj,
    *,
    overwrite: bool,
    max_threads: int,
    request_timeout_s: int,
    verbose: bool,
) -> list[str]:
    """Download full-file Herbie refs without relying on Herbie's no-timeout downloader."""
    refs = list(herbie_obj.file_exists)
    downloaded_paths: list[str] = []
    threads = min(len(refs), max_threads)
    if threads < 1:
        return downloaded_paths

    if verbose:
        logger.info(
            "Downloading %d full Herbie files with %d threads and %ss read timeout.",
            len(refs),
            threads,
            request_timeout_s,
        )
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                _download_full_herbie_file,
                ref,
                overwrite=overwrite,
                request_timeout_s=request_timeout_s,
            )
            for ref in refs
        ]
        for future in as_completed(futures):
            downloaded_paths.append(future.result())

    return downloaded_paths


def download_herbie_with_retry(
    herbie_obj,
    expected_count: int,
    dataset_name: str,
    retries: int,
    retry_sleep_s: int,
    search: Any = None,
    max_threads: int | None = None,
    overwrite_first_attempt: bool = False,
    verbose: bool = True,
) -> None:
    """Retry transient Herbie download failures and enforce expected file count."""
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            # Overwrite on retries to avoid keeping partial/corrupt files.
            download_kwargs = {
                "verbose": verbose,
                "overwrite": overwrite_first_attempt or attempt > 1,
            }
            if max_threads is not None:
                download_kwargs["max_threads"] = max_threads
            if search is None:
                downloaded = _download_full_herbie_files(
                    herbie_obj,
                    overwrite=download_kwargs["overwrite"],
                    max_threads=max_threads
                    if max_threads is not None
                    else positive_int_env("herbie_download_threads", 20),
                    request_timeout_s=positive_int_env(
                        "herbie_request_timeout_seconds", 120
                    ),
                    verbose=verbose,
                )
            else:
                downloaded = herbie_obj.download(search, **download_kwargs)

            downloaded_paths = []
            if downloaded is not None:
                if not isinstance(downloaded, list):
                    downloaded = [downloaded]
                downloaded_paths = [
                    str(Path(path).expanduser()) for path in downloaded if path
                ]

                if len(downloaded_paths) != expected_count:
                    raise RuntimeError(
                        f"Expected {expected_count} downloaded {dataset_name} paths "
                        f"but got {len(downloaded_paths)}"
                    )

            matched_refs = list(herbie_obj.file_exists)
            matched_count = len(matched_refs)
            if matched_count != expected_count:
                raise RuntimeError(
                    f"Expected {expected_count} {dataset_name} references "
                    f"but got {matched_count}"
                )

            # Herbie can report matched references even when local downloads fail;
            # verify every expected local GRIB path is present and non-empty.
            local_paths = downloaded_paths or build_herbie_grib_list(
                matched_refs, search
            )
            valid_local_paths = [
                p for p in local_paths if os.path.isfile(p) and os.path.getsize(p) > 0
            ]

            if len(valid_local_paths) == expected_count:
                if attempt > 1:
                    logger.info(
                        "%s download succeeded on retry %d/%d",
                        dataset_name,
                        attempt,
                        attempts,
                    )
                return

            missing_count = expected_count - len(valid_local_paths)
            raise RuntimeError(
                f"Expected {expected_count} downloaded {dataset_name} files but found "
                f"{len(valid_local_paths)} valid local files (missing {missing_count})"
            )
        except Exception as exc:
            if attempt == attempts:
                logger.exception(
                    "%s download failed after %d attempts",
                    dataset_name,
                    attempts,
                )
                raise

            sleep_s = retry_sleep_s * attempt
            logger.warning(
                "%s download attempt %d/%d failed (%s). Retrying in %ss",
                dataset_name,
                attempt,
                attempts,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)


def safe_herbie_local_file_path(
    herbie_obj, search: Any = None, retries: int = 3, retry_sleep_s: float = 0.1
) -> str:
    """Resolve a local Herbie path and repair file-vs-dir cache collisions."""
    from herbie import Path

    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return str(Path(herbie_obj.get_localFilePath(search)).expand())
        except FileExistsError as exc:
            conflict_path = getattr(exc, "filename", None)
            if conflict_path and os.path.isfile(conflict_path):
                logger.info(
                    "Repairing Herbie cache path collision at: %s", conflict_path
                )
                os.remove(conflict_path)
                os.makedirs(conflict_path, exist_ok=True)
            elif conflict_path and not os.path.exists(conflict_path):
                os.makedirs(conflict_path, exist_ok=True)
            else:
                time.sleep(retry_sleep_s)

            if attempt == attempts - 1:
                raise

    raise RuntimeError("Unreachable Herbie local path resolution state")


def build_herbie_grib_list(
    file_refs, search: Any = None, retries: int = 3
) -> list[str]:
    """Build a list of local GRIB paths from Herbie file references."""
    return [
        safe_herbie_local_file_path(ref, search, retries=retries) for ref in file_refs
    ]


def close_store(store: object) -> None:
    """Close a zarr-like store if it exposes a close method."""
    close_fn = getattr(store, "close", None)
    if callable(close_fn):
        close_fn()


def clear_directory(path: str, keep: str | None = None) -> None:
    """Empty ``path`` (creating it if needed), optionally keeping one entry.

    The CMC ingests keep their Herbie download cache (``keep``) between runs so
    unchanged GRIB files are not downloaded again.
    """
    os.makedirs(path, exist_ok=True)
    keep_path = os.path.abspath(keep) if keep else None
    for entry in os.scandir(path):
        if os.path.abspath(entry.path) == keep_path:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)


def prune_old_files(path: str, max_age_hours: float) -> None:
    """Delete files under ``path`` last modified more than ``max_age_hours`` ago.

    The CMC ingests keep their Herbie download cache between runs so unchanged
    GRIB files are not downloaded again; without pruning, one directory per
    run date accumulates forever. Directories left empty are removed too.
    """
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.getmtime(file_path) < cutoff:
                os.remove(file_path)
                removed += 1
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
    if removed:
        logger.info(
            "Pruned %d cached files older than %s h from %s",
            removed,
            max_age_hours,
            path,
        )


# Published stores are uploaded this many objects at a time. A 9,601-object store
# (8 KiB chunks) took 43 / 27 / 22 / 32 s to OVH at 64 / 128 / 256 / 512.
UPLOAD_CONCURRENCY = positive_int_env("upload_concurrency", 256)
# Optional storage class for every S3 object, e.g. EXPRESS_ONEZONE for OVH High
# Performance (a default header is not signed, so it is set per upload).
S3_STORAGE_CLASS = os.getenv("s3_storage_class") or None


def open_remote(path: str, save_type: str, key: str = "", secret: str = ""):
    """Open the place an ingest publishes to as an obstore store.

    With ``save_type`` "S3", ``path`` is ``bucket/prefix``; ``s3_endpoint`` and
    ``s3_region`` point it at a non-AWS store such as OVH, and empty keys fall
    back to obstore's ``AWS_*`` environment variables. Anything else is a local
    directory (created if missing), so publishing and history use one code path
    whether the remote is a bucket or a disk.
    """
    import obstore.store

    if save_type == "S3":
        bucket, _, prefix = path.strip("/").partition("/")
        options = {
            "endpoint": os.getenv("s3_endpoint"),
            "region": os.getenv("s3_region"),
            "access_key_id": key,
            "secret_access_key": secret,
        }
        return obstore.store.S3Store(
            bucket,
            prefix=prefix or None,
            **{name: value for name, value in options.items() if value},
        )
    os.makedirs(path, exist_ok=True)
    return obstore.store.LocalStore(path)


def _put_attributes(remote) -> dict | None:
    import obstore.store

    if S3_STORAGE_CLASS and isinstance(remote, obstore.store.S3Store):
        return {"Storage-Class": S3_STORAGE_CLASS}
    return None


def _put_files(remote, uploads: list[tuple[str, str]]) -> None:
    """Upload ``(local_file, key)`` pairs, ``UPLOAD_CONCURRENCY`` at a time.

    obstore retries each request itself (bounded by its default retry config).
    """
    import asyncio

    import obstore

    attributes = _put_attributes(remote)

    async def put_all() -> None:
        semaphore = asyncio.Semaphore(UPLOAD_CONCURRENCY)

        async def put(local_file: str, key: str) -> None:
            async with semaphore:
                await obstore.put_async(
                    remote, key, Path(local_file), attributes=attributes
                )

        await asyncio.gather(*(put(local, key) for local, key in uploads))

    asyncio.run(put_all())


def upload_store(local_path: str, remote, name: str) -> None:
    """Upload a local zarr directory store to ``name`` on the remote.

    Chunks go first, then the arrays' ``zarr.json``, then the store's root
    ``zarr.json``, so a store whose root metadata exists is complete (the history
    ingests check that). Chunk keys are the same every run and empty chunks are
    written, so the upload replaces every object of the previous run. It is not
    atomic, but a point read is one chunk holding every variable and time step,
    so a reader during it sees one run or the other at each point.
    """
    chunk_files, array_metadata, root_metadata = [], [], []
    for dirpath, _, filenames in os.walk(local_path):
        for filename in filenames:
            local_file = os.path.join(dirpath, filename)
            relative = Path(local_file).relative_to(local_path).as_posix()
            if relative == "zarr.json":
                group = root_metadata
            elif filename == "zarr.json":
                group = array_metadata
            else:
                group = chunk_files
            group.append((local_file, f"{name}/{relative}"))
    for uploads in (chunk_files, array_metadata, root_metadata):
        _put_files(remote, uploads)


def read_published_run_time(remote, time_pickle_name: str) -> Any | None:
    """Return the run time recorded by the last published ingest, or None.

    Every ingest publishes ``<MODEL>.time.pickle`` next to its store; comparing it
    with the newest available run is how a scheduled ingest decides whether there
    is anything new to process.
    """
    import obstore

    try:
        return pickle.loads(obstore.get(remote, time_pickle_name).bytes())
    except (FileNotFoundError, obstore.exceptions.NotFoundError):
        return None


def publish_ingest_run(
    output_names: Iterable[str],
    run_time: Any,
    time_pickle_name: str,
    *,
    process_dir: str,
    remote,
) -> None:
    """Publish a finished run: its outputs first, then ``<MODEL>.time.pickle``.

    ``output_names`` are entries in ``process_dir``: zarr directory stores such as
    ``GEFS.zarr`` (see ``upload_store``) or side files the API reads. The
    run-time pickle is how the API and the next scheduled ingest see that a new
    run exists, so it is only written once every output is in place.
    """
    import obstore

    for name in output_names:
        source_path = os.path.join(process_dir, name)
        if os.path.isdir(source_path):
            upload_store(source_path, remote, name)
        else:
            _put_files(remote, [(source_path, name)])
        logger.info("Published %s", name)

    obstore.put(
        remote,
        time_pickle_name,
        pickle.dumps(run_time),
        attributes=_put_attributes(remote),
    )


def write_final_zarr(
    stacked: da.Array,
    store_path: str,
    final_chunk: int,
    zarr_store_workers: int,
) -> None:
    """Write the final ``(variable, time, y, x)`` array the API reads.

    One chunk holds every variable and time step for a ``final_chunk`` square of
    grid cells, so a point forecast is a single chunk read. ``stacked`` must
    already be padded to a multiple of ``final_chunk``. ``store_path`` is a local
    directory store, published with ``publish_ingest_run``. Values are rounded to
    5 decimals to help compression. Empty (all-NaN) chunks are written too, so a
    publish over the previous run replaces every chunk.

    Dask writes tiles about the size of a source chunk and zarr splits each one
    into the small final chunks. Rechunking dask all the way down to 3x3 or 5x5
    chunks builds a very large task graph: on a GFS-sized array this write took
    102 s / 7.1 GB peak versus 153 s / 7.9 GB (GEFS-sized: 86 s / 3.3 GB versus
    135 s / 4.3 GB), with identical output. Tiles are a multiple of
    ``final_chunk`` so every zarr chunk is written exactly once.

    Stores are not sharded: 200 x 200 shards made every local point read about
    0.4 ms slower (zarr's sharding codec), and uploading the small chunk files
    with ``UPLOAD_CONCURRENCY`` parallel requests is fast enough (HRRR's 76k
    objects in about 4 min to OVH).
    """
    import dask
    import zarr

    store = zarr.storage.LocalStore(store_path)
    zarr_array = zarr.create_array(
        store=store,
        shape=stacked.shape,
        chunks=(stacked.shape[0], stacked.shape[1], final_chunk, final_chunk),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        dtype="float32",
        config={"write_empty_chunks": True},
    )

    tile_y = final_chunk * max(1, round(stacked.chunks[2][0] / final_chunk))
    tile_x = final_chunk * max(1, round(stacked.chunks[3][0] / final_chunk))
    tiles = (stacked.shape[0], stacked.shape[1], tile_y, tile_x)
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        stacked.round(5).rechunk(tiles).to_zarr(
            zarr_array, overwrite=True, compute=True
        )
    close_store(store)


def remote_object_exists(remote, key: str) -> bool:
    """True if ``key`` exists on the remote (a local remote raises a different error)."""
    import obstore

    try:
        obstore.head(remote, key)
    except (FileNotFoundError, obstore.exceptions.NotFoundError):
        return False
    return True


def history_store_exists(remote, name: str) -> bool:
    """True once history store ``name`` is fully published (root metadata last)."""
    return remote_object_exists(remote, f"{name}/zarr.json")


def publish_history_store(local_path: str, remote, name: str) -> None:
    """Upload one finished history store to the history remote, then drop it."""
    upload_store(local_path, remote, name)
    shutil.rmtree(local_path, ignore_errors=True)
    logger.info("Published history store %s", name)


def open_history_store(remote, name: str, expected_vars: Iterable[str]):
    """Open a published history store read-only, straight from the remote.

    Every history run is published before the merge, so a missing store means it
    was lost and the run fails rather than publishing a forecast with a gap. A
    store missing an expected variable (written by an older ingest) is deleted so
    the next run rebuilds it, and the run fails.
    """
    import obstore
    import zarr

    if not history_store_exists(remote, name):
        raise RuntimeError(f"History store {name} is missing from {remote}")

    group = zarr.open_group(
        zarr.storage.ObjectStore(remote, read_only=True), path=name, mode="r"
    )
    missing_vars = sorted(set(expected_vars) - set(group.keys()))
    if missing_vars:
        stale_keys = [
            meta["path"]
            for batch in obstore.list(remote, prefix=f"{name}/")
            for meta in batch
        ]
        obstore.delete(remote, stale_keys)
        raise ValueError(
            f"History store {name} is missing variables {missing_vars}; "
            "deleted it so the next run rebuilds it"
        )
    return group


def mask_invalid_data(daskArray, ignoreAxis=None):
    """Masks invalid data in a dask array, ignoring the time dimension."""
    # TODO: Update to mask for each variable according to reasonable values, as opposed to this global mask
    valid_mask = (daskArray >= VALID_DATA_MIN) & (daskArray <= VALID_DATA_MAX)
    # Ignore times by setting first dimension to True
    valid_mask[0, :, :, :] = True

    # Also ignore the specified axis if provided
    if ignoreAxis is not None:
        for i in ignoreAxis:
            valid_mask[i, :, :, :] = True
    return da.where(valid_mask, daskArray, MISSING_DATA)


def mask_invalid_refc(xrArr: "xr.DataArray") -> "xr.DataArray":
    """Masks REFC values less than 5, setting them to 0.

    Args:
        xrArr: The input xarray DataArray with REFC values.

    Returns:
        The masked xarray DataArray.
    """
    return xrArr.where(xrArr >= REFC_THRESHOLD, 0)


# Function to get the list of GRIB files from the forecast subscription, used by NBM
def validate_grib_stats(gribCheck, excluded_variables: Iterable[str] | None = None):
    """
    Inspect gribCheck.stdout (from `wgrib2 … -stats`) for min/max values,
    print any out-of-range records, and exit(10) if invalid data is found.

    Expects:
      - gribCheck.stdout: the full stdout string
      - globals: VALID_DATA_MIN, VALID_DATA_MAX
    """
    excluded_variables = set(excluded_variables or [])

    # extract all mins and maxs
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    minValues = [
        float(m) for m in re.findall(rf"min=({number_pattern})", gribCheck.stdout)
    ]
    maxValues = [
        float(M) for M in re.findall(rf"max=({number_pattern})", gribCheck.stdout)
    ]

    # extract variable names (4th field)
    varNames = re.findall(r"(?m)^(?:[^:]+:){3}([^:]+):", gribCheck.stdout)
    # ensure we found at least one variable
    if not varNames:
        logger.error("Error: no variables found in GRIB stats output.")
        return False

    # extract forecast lead times (6th field)
    varTimes = re.findall(r"(?m)^(?:[^:]+:){5}([^:]+):", gribCheck.stdout)

    # find any indices where data is out of range
    # TODO: This would be better if we checked against a dictionary of valid ranges defined per variable
    invalidIdxs = [
        i
        for i, (var, mn, mx) in enumerate(zip(varNames, minValues, maxValues))
        if var not in excluded_variables
        and (mn < VALID_DATA_MIN or mx > VALID_DATA_MAX)
    ]

    if invalidIdxs:
        logger.error("Invalid data found in grib files:")
        for i in invalidIdxs:
            logger.error("  Variable : %s", varNames[i])
            logger.error("  Time     : %s", varTimes[i])
            logger.error("  Min/Max  : %s / %s", minValues[i], maxValues[i])
            logger.error("---")
        logger.error("Exiting due to invalid data in grib files.")

        # Return False to indicate validation failure, allowing caller to handle exit or retry logic
        return False

    else:
        logger.info("All grib files passed validation checks.")
        # compute overall min/max for each variable across all times
        varExtremes = {}
        for var, mn, mx in zip(varNames, minValues, maxValues):
            lo, hi = varExtremes.setdefault(var, [mn, mx])
            varExtremes[var][0] = min(lo, mn)
            varExtremes[var][1] = max(hi, mx)

        # print overall extremes
        logger.info("Overall min/max for each variable across all times:")
        for var, (mn, mx) in varExtremes.items():
            logger.info("  %s: min=%s, max=%s", var, mn, mx)

    # all good
    return True


def pad_to_chunk_size(dask_array: da.Array, final_chunk: int) -> da.Array:
    """Pad a 4D dask array so its Y and X dimensions are multiples of final_chunk.

    This ensures efficient zarr storage by aligning spatial dimensions to chunk boundaries.
    Padding is done with NaN values on the right and bottom edges.

    Args:
        dask_array: 4D dask array with shape (var, time, y, x).
        final_chunk: The target chunk size for spatial dimensions.

    Returns:
        Padded dask array with y and x dimensions that are multiples of final_chunk.
        If no padding is needed, returns the original array.
    """
    y, x = dask_array.shape[2], dask_array.shape[3]
    pad_y = (-y) % final_chunk  # 0..(final_chunk - 1)
    pad_x = (-x) % final_chunk  # 0..(final_chunk - 1)

    # Only pad if necessary
    if pad_y or pad_x:
        return da.pad(
            dask_array,
            ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=np.nan,
        )
    return dask_array


def broadcast_times_to_grid(
    times: np.ndarray, ny: int, nx: int, spatial_chunk: int
) -> da.Array:
    """Lazily broadcast one timestamp per timestep across a spatial grid.

    Args:
        times: One-dimensional array of timestamps.
        ny: Number of grid rows.
        nx: Number of grid columns.
        spatial_chunk: Chunk size for both spatial dimensions.

    Returns:
        A lazily broadcast Dask array with shape ``(time, y, x)``.

    Raises:
        ValueError: If the timestamps are not one-dimensional or any output
            dimension or chunk size is not positive.
    """
    times = np.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if len(times) < 1 or ny < 1 or nx < 1 or spatial_chunk < 1:
        raise ValueError("time and spatial dimensions and chunk size must be positive")

    time_column = da.from_array(
        times[:, np.newaxis, np.newaxis],
        chunks=(len(times), 1, 1),
    )
    return da.broadcast_to(
        time_column,
        (len(times), ny, nx),
        chunks=(len(times), spatial_chunk, spatial_chunk),
    )


def earth_relative_wind_components(
    ugrd: xr.DataArray, vgrd: xr.DataArray
) -> tuple[
    np.ndarray, np.ndarray
]:  # Based off: https://unidata.github.io/python-gallery/examples/500hPa_Absolute_Vorticity_winds.html#function-to-compute-earth-relative-winds
    """Calculate north-relative wind components from grid-relative components.

    Uses Cartopy to transform vectors from the model's grid-relative projection
    to a standard Plate Carree projection (earth-relative).

    Args:
        ugrd: Xarray DataArray of the grid-relative u-component of the wind.
        vgrd: Xarray DataArray of the grid-relative v-component of the wind.

    Returns:
        A tuple containing two numpy arrays: the earth-relative u-component (ut)
        and v-component (vt) of the wind.
    """
    data_crs = ugrd.metpy_crs.metpy.cartopy_crs

    x = ugrd.x.values
    y = ugrd.y.values

    xx, yy = np.meshgrid(x, y)

    ut, vt = ccrs.PlateCarree().transform_vectors(
        data_crs, xx, yy, ugrd.values, vgrd.values
    )

    return ut, vt


def interp_time_take_blend(
    arr: da.Array,
    stacked_timesUnix: np.ndarray,
    hourly_timesUnix: np.ndarray,
    nearest_vars: int | Iterable[int] | None = None,  # var indices using NN
    dtype: str = "float32",
    fill_value: float = np.nan,
    time_axis: int = 1,
) -> da.Array:
    r"""Interpolate model data along the time dimension via gather-and-blend.

    The helper assumes the input has shape \(V, T, Y, X\) and that time is
    the second axis. It gathers the values for the two surrounding stored
    times using ``da.take``, then linearly blends them using the fractional
    offset between ``stacked_timesUnix`` and ``hourly_timesUnix``. Points that
    fall outside the time range defined by ``stacked_timesUnix`` are filled with
    ``fill_value``. Optionally, a subset of variable indices may be overridden
    with nearest-neighbor interpolation instead of the blended values.

    Args:
        arr: Chunked Dask array containing the forecast in \(V, T, Y, X\)
            order. The time axis must already be a single chunk so gather
            operations stay within chunk boundaries.
        stacked_timesUnix: Known source timestamps to interpolate from (monotonic
            increasing unix seconds).
        hourly_timesUnix: Desired target timestamps; must lie within the range
            spanned by ``stacked_timesUnix`` when possible.
        nearest_vars: Optional variable indices that should use the closer
            neighbor directly instead of linear blending (e.g., categorical
            flags). Can be a single index or an iterable.
        dtype: Output dtype for the interpolated array.
        fill_value: Value used for times outside the available range.
        time_axis: Axis index for time (must be 1 in this helper).

    Returns:
        A dask array shaped \(V, T_new, Y, X\) with interpolated (or overridden)
        values and ``dtype``.
    """
    if arr.ndim != 4:
        raise ValueError("Expected arr with dims (V, T, Y, X).")

    VAX, TAX = 0, time_axis
    if TAX != 1:
        raise NotImplementedError("This helper assumes time_axis == 1 for (V,T,Y,X).")

    # Precompute the two neighbor‐indices and the weights
    x_a = np.array(stacked_timesUnix)
    x_b = np.array(hourly_timesUnix)

    idx = np.searchsorted(x_a, x_b) - 1
    idx0 = np.clip(idx, 0, len(x_a) - 2)
    idx1 = idx0 + 1
    w = (x_b - x_a[idx0]) / (x_a[idx1] - x_a[idx0])  # float array, shape (T_new,)

    # boolean mask of “in‐range” points
    valid = (x_b >= x_a[0]) & (x_b <= x_a[-1])  # shape (T_new,)

    T_new = len(idx0)
    if not (len(idx1) == T_new and len(w) == T_new and len(valid) == T_new):
        raise ValueError("idx0, idx1, w, and valid must all have length T_new.")

    # Ensure time axis already fits in one chunk so gather (`da.take`) stays within chunk boundaries
    time_chunks = arr.chunks[TAX]
    if len(time_chunks) != 1:
        raise ValueError(
            "time axis must be a single chunk; please rechunk with ``arr.rechunk({time_axis: -1})`` "
            f"before calling (got {len(time_chunks)} chunks)."
        )
    arr_t = arr

    # Gather neighbors along time
    y0 = da.take(arr_t, idx0, axis=TAX)  # (V, T_new, Y, X)
    y1 = da.take(arr_t, idx1, axis=TAX)

    # Weighted blend
    w_r = da.asarray(w, chunks=(T_new,))[None, :, None, None]
    out = (1 - w_r) * y0 + w_r * y1
    out = out.astype(dtype, copy=False)

    # Mask invalid new times
    if (~valid).any():
        valid_r = da.asarray(valid, chunks=(T_new,))[None, :, None, None]
        out = da.where(valid_r, out, fill_value)

    # Optional nearest-neighbor override for specified variable indices
    if nearest_vars is not None:
        # Precompute nearest indices once: closer of idx0 / idx1
        nearest_idx = np.where(w < 0.5, idx0, idx1).astype(idx0.dtype)

        # Normalize indices as a sorted, unique list
        if isinstance(nearest_vars, int):
            nearest_vars = [nearest_vars]
        nv = sorted({int(i) for i in nearest_vars})

        # Compute nearest only for needed variables (cheap if few vars)
        # Shape of each nearest slice: (1, T_new, Y, X)
        take_nn = da.take(arr_t, nearest_idx, axis=TAX)
        # Replace in 'out' per variable index
        pieces = []
        prev = 0
        for i in nv:
            if i < 0 or i >= arr.shape[VAX]:
                raise IndexError(
                    f"nearest_vars index {i} out of range for V={arr.shape[VAX]}"
                )
            if i > prev:
                pieces.append(out[prev:i])  # unchanged segment
            pieces.append(
                take_nn[i : i + 1].astype(dtype, copy=False)
            )  # nearest segment
            prev = i + 1
        if prev < arr.shape[VAX]:
            pieces.append(out[prev:])
        out = da.concatenate(pieces, axis=VAX)

    return out


def validate_stacked_time_alignment(
    stacked_times_unix: np.ndarray,
    concatenated_times_unix: np.ndarray,
    tolerance_seconds: float = 300,
) -> None:
    """Ensure concatenated stored times stay close to the expected stacked times."""
    expected_times = np.asarray(stacked_times_unix, dtype=np.float64).reshape(-1)
    actual_times = np.asarray(concatenated_times_unix, dtype=np.float64).reshape(-1)

    if expected_times.shape != actual_times.shape:
        raise ValueError(
            "Time alignment check failed due to shape mismatch: "
            f"expected {expected_times.shape}, got {actual_times.shape}."
        )

    time_deltas = np.abs(expected_times - actual_times)
    mismatched_indices = np.flatnonzero(time_deltas > tolerance_seconds)
    if mismatched_indices.size:
        first_idx = int(mismatched_indices[0])
        raise ValueError(
            "Time alignment check failed: "
            f"{mismatched_indices.size} timestamps differ by more than "
            f"{tolerance_seconds} seconds. First mismatch at index {first_idx}: "
            f"expected={expected_times[first_idx]}, "
            f"actual={actual_times[first_idx]}, "
            f"delta={time_deltas[first_idx]}."
        )


# --- Air quality helpers (NowCast & EPA AQI) ---
def calculate_nowcast_concentration(
    concentrations: np.ndarray, num_hours: int = 12
) -> np.ndarray:
    """
    Calculate the EPA NowCast weighted concentration for PM2.5 and PM10.
    The NowCast algorithm weights recent hours more heavily than older hours,
    making it more responsive to changing air quality conditions than a
    simple average.
    Args:
        concentrations: Array of concentrations with time as the first dimension.
                       Shape: (time, latitude, longitude)
        num_hours: Number of hours to use in NowCast calculation (default 12)
    Returns:
        NowCast weighted concentration array with same shape as input
    """
    if concentrations.shape[0] < 3:
        return concentrations

    hours_to_use = min(num_hours, concentrations.shape[0])
    nowcast_result = np.full_like(concentrations, np.nan)

    for t in range(concentrations.shape[0]):
        start_idx = max(0, t - hours_to_use + 1)
        window = concentrations[start_idx : t + 1]

        if window.shape[0] < 3:
            nowcast_result[t] = concentrations[t]
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            c_max = np.nanmax(window, axis=0)
            c_min = np.nanmin(window, axis=0)
            c_range = c_max - c_min
            weight_factor = np.where(
                c_max > 0, np.maximum(1 - c_range / c_max, 0.5), 0.5
            )

        num_window_hours = window.shape[0]
        weights = np.zeros_like(window)
        for i in range(num_window_hours):
            hours_ago = num_window_hours - 1 - i
            weights[i] = weight_factor**hours_ago

        with np.errstate(invalid="ignore"):
            weighted_sum = np.nansum(window * weights, axis=0)
            weight_sum = np.nansum(np.where(~np.isnan(window), weights, 0), axis=0)
            nowcast_result[t] = np.where(
                weight_sum > 0, weighted_sum / weight_sum, np.nan
            )

    return nowcast_result


def trailing_mean(conc: np.ndarray | None, window: int) -> np.ndarray | None:
    """
    Compute trailing window mean along time axis for array with shape (T, Y, X).
    If window <= 1 returns conc. Handles NaNs by using nanmean over available points.
    """
    if conc is None:
        return None
    if window is None or window <= 1:
        return conc

    T = conc.shape[0]
    out = np.full_like(conc, np.nan)
    for t in range(T):
        start = max(0, t - window + 1)
        # nanmean handles NaNs and short windows
        with np.errstate(invalid="ignore"):
            out[t] = np.nanmean(conc[start : t + 1], axis=0)
    return out


def interpolate_temporal_gaps_efficiently(
    ds_chunked, nearest_vars=None, max_gap_hours=3, time_dim="time"
):
    """
    Interpolates temporal gaps and extrapolates edges efficiently in a sparse Dask/Xarray dataset.

    Logic:
    1. Re-chunks data to be contiguous in time ('pencils').
    2. Short-circuits empty (all-NaN) spatial chunks.
    3. Interpolates internal gaps (Linear by default, Nearest for specific vars).
    4. Extrapolates edges (Nearest neighbor / Forward & Back fill) for ALL vars.

    Args:
        ds_chunked (xr.Dataset): Input dataset. Must be chunked with time as a single chunk.
        nearest_vars (list): List of variable names to use 'nearest' interpolation for
                             (e.g., flags, codes). Default is 'linear' for others.
        max_gap_hours (int): Max gap size to interpolate internally.
                             Extrapolation is applied to the ends regardless of gap size.
        time_dim (str): Name of the time dimension.

    Returns:
        xr.Dataset: Processed dataset.
    """

    if nearest_vars is None:
        nearest_vars = []

    # Helper function applied to each Dask block
    def _interpolate_block(block, time_coords, method):
        # OPTIMIZATION: Short-circuit empty blocks
        # If the entire spatial chunk is NaN, return immediately.
        if np.all(np.isnan(block)):
            return block

        # Wrap numpy block in DataArray for convenient Xarray methods
        da_temp = xr.DataArray(
            block, dims=(time_dim, "y", "x"), coords={time_dim: time_coords}
        )

        # 1. Interpolate Internal Gaps
        # use_coordinate=True ensures we respect actual time steps, not just index count
        filled = da_temp.interpolate_na(
            dim=time_dim, method=method, limit=max_gap_hours, use_coordinate=True
        )

        # 2. Extrapolate Edges (Nearest Neighbour)
        # We use ffill (forward) and bfill (backward) to extend the last known
        # valid value to the start/end of the series.
        filled = filled.ffill(time_dim).bfill(time_dim)

        return filled.values

    # Function to map over every variable
    def _process_variable(da_var):
        # Skip coordinate variables or vars without time
        if time_dim not in da_var.dims:
            return da_var

        # Determine interpolation method for this specific variable
        # Default is 'linear', unless specified in nearest_vars
        interp_method = "nearest" if da_var.name in nearest_vars else "linear"

        time_coords = da_var[time_dim].values

        # dask.map_blocks lets us run the logic on every chunk in parallel
        processed_data = da_var.data.map_blocks(
            _interpolate_block,
            time_coords=time_coords,
            method=interp_method,
            dtype=da_var.dtype,
            chunks=da_var.chunks,
        )

        return da_var.copy(data=processed_data)

    # Execute
    return ds_chunked.map(_process_variable)
