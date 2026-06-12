# %% Script to contain the helper functions as part of the data ingest for Pirate Weather
# Alexander Rey. July 2025

import logging
import os
import re
import resource
import shlex
import shutil
import subprocess
import tarfile
import time
from typing import Iterable, Optional, Union

import cartopy.crs as ccrs
import dask.array as da
import numpy as np
import xarray as xr
from herbie import Path

# Import atmospheric calculation constants
from API.constants.aqi_const import (
    CO_AQI,
    CO_BP,
    NO2_AQI,
    NO2_BP,
    O3_AQI,
    O3_BP,
    PM10_AQI,
    PM10_BP,
    PM25_AQI,
    PM25_BP,
    SO2_AQI,
    SO2_BP,
)
from API.constants.shared_const import (
    MISSING_DATA,
    REFC_THRESHOLD,
)

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    "DWD": 5,
}

FORECAST_LEAD_RANGES = {
    "GFS_1": list(range(1, 121)),
    "GFS_2": list(range(123, 241, 3)),
    "GEFS": list(range(3, 241, 3)),
    "NBM_FIRE": list(range(6, 192, 6)),
    "HRRR_1H": list(range(1, 19)),
    "HRRR_6H": list(range(18, 49)),
    "ECMWF_AIFS": list(range(6, 241, 6)),
    "ECMWF_IFS_1": list(range(3, 144, 3)),
    "ECMWF_IFS_2": list(range(144, 241, 6)),
    "AIGFS": list(range(6, 241, 6)),
    "AIGEFS": list(range(6, 241, 6)),
}

# Radius, in km, used for DWD model nearest-neighbor selection
DWD_RADIUS = 50

VALID_DATA_MIN = -100
VALID_DATA_MAX = 120000


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
            print(f"Raised nofile soft limit from {soft} to {new_soft}")
    except (ValueError, OSError) as exc:
        print(f"Warning: unable to tune nofile limit: {exc}")


def positive_int_env(name: str, default: int) -> int:
    """Read an integer env var and fall back to a safe positive default."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        print(f"Warning: invalid {name}={raw!r}; using {default}")
        return default
    if value < 1:
        print(f"Warning: {name}={value} must be >= 1; using {default}")
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
        print(f"Warning: unable to read nofile limit for zarr tuning: {exc}")

    async_concurrency = min(async_concurrency, workers)
    if workers < requested_workers:
        print(
            "Clamped zarr_store_workers from "
            f"{requested_workers} to {workers} based on nofile limits"
        )
    if async_concurrency < requested_async_concurrency:
        print(
            "Clamped zarr_async_concurrency from "
            f"{requested_async_concurrency} to {async_concurrency} based on nofile limits"
        )

    # Local import keeps helper lightweight for non-zarr callers.
    import zarr

    zarr.config.set({"async.concurrency": async_concurrency})
    print(
        f"Configured zarr write parallelism: workers={workers}, "
        f"async_concurrency={async_concurrency}"
    )
    return workers, async_concurrency


def make_herbie_save_dir(tmp_dir: str, prefix: str = "herbie") -> str:
    """Create a per-run Herbie cache directory to avoid path collisions."""
    save_dir = os.path.join(tmp_dir, f"{prefix}_{int(time.time())}_{os.getpid()}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def download_herbie_with_retry(
    herbie_obj,
    search: str,
    expected_count: int,
    dataset_name: str,
    retries: int,
    retry_sleep_s: int,
) -> None:
    """Retry transient Herbie download failures and enforce expected file count."""
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            # Overwrite on retries to avoid keeping partial/corrupt files.
            herbie_obj.download(
                search,
                verbose=False,
                overwrite=(attempt > 1),
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
            local_paths = build_herbie_grib_list(matched_refs, search)
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
    herbie_obj, search: str, retries: int = 3, retry_sleep_s: float = 0.1
) -> str:
    """Resolve a local Herbie path and repair file-vs-dir cache collisions."""
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return str(Path(herbie_obj.get_localFilePath(search)).expand())
        except FileExistsError as exc:
            conflict_path = getattr(exc, "filename", None)
            if conflict_path and os.path.isfile(conflict_path):
                print(f"Repairing Herbie cache path collision at: {conflict_path}")
                os.remove(conflict_path)
                os.makedirs(conflict_path, exist_ok=True)
            elif conflict_path and not os.path.exists(conflict_path):
                os.makedirs(conflict_path, exist_ok=True)
            else:
                time.sleep(retry_sleep_s)

            if attempt == attempts - 1:
                raise

    raise RuntimeError("Unreachable Herbie local path resolution state")


def build_herbie_grib_list(file_refs, search: str, retries: int = 3) -> list[str]:
    """Build a list of local GRIB paths from Herbie file references."""
    return [
        safe_herbie_local_file_path(ref, search, retries=retries) for ref in file_refs
    ]


def close_store(store: object) -> None:
    """Close a zarr-like store if it exposes a close method."""
    close_fn = getattr(store, "close", None)
    if callable(close_fn):
        close_fn()


def archive_tmp_zarr_and_upload(
    *,
    tmp_zarr_path: str,
    s3_path: str,
    archive_member_name: str,
    s3,
) -> None:
    """Tar/gzip a temporary zarr directory, upload to S3, and write done marker."""
    tmp_tar_path = f"{tmp_zarr_path}.tar.gz"
    with tarfile.open(tmp_tar_path, "w:gz") as tar:
        tar.add(tmp_zarr_path, arcname=archive_member_name)

    s3.put_file(tmp_tar_path, s3_path)
    if os.path.exists(tmp_tar_path):
        os.remove(tmp_tar_path)
    shutil.rmtree(tmp_zarr_path, ignore_errors=True)
    s3.touch(s3_path.replace(".tar.gz", ".done"))


def _delete_historic_archive_from_s3(s3, s3_tar_path: str) -> None:
    """Delete a historic zarr archive and its completion marker from S3."""
    s3_zarr_path = s3_tar_path.removesuffix(".tar.gz")
    for path in (
        s3_tar_path,
        s3_zarr_path,
        s3_tar_path.replace(".tar.gz", ".done"),
    ):
        if s3.exists(path):
            s3.rm(path, recursive=True)


def _validate_historic_zarr_variables(
    zarr_path: str,
    expected_vars: Iterable[str] | None,
) -> None:
    """Validate that a local historic zarr contains every expected variable."""
    if expected_vars is None:
        return

    import zarr

    z = zarr.open(zarr_path, mode="r")
    store_vars = set(z.keys())
    expected_set = set(expected_vars)
    missing_vars = sorted(expected_set - store_vars)
    if missing_vars:
        raise ValueError(
            f"Missing variables in extracted historic zarr {zarr_path}: "
            f"{missing_vars}. Found variables: {sorted(store_vars)}"
        )


def download_extract_historic_archive(
    *,
    s3,
    historic_path: str,
    final_zarr_name: str,
    extracted_store_name: str,
    local_temp_dir: str,
    expected_vars: Iterable[str] | None = None,
) -> Optional[str]:
    """Helper to download and extract a historic archive to a local zarr path."""
    os.makedirs(local_temp_dir, exist_ok=True)
    local_zarr_path = os.path.join(local_temp_dir, final_zarr_name)

    tar_name = f"{final_zarr_name}.tar.gz"
    s3_tar_path = f"{historic_path}/{tar_name}"

    if os.path.exists(local_zarr_path):
        try:
            _validate_historic_zarr_variables(local_zarr_path, expected_vars)
        except Exception:
            shutil.rmtree(local_zarr_path, ignore_errors=True)
            _delete_historic_archive_from_s3(s3, s3_tar_path)
            raise
        return local_zarr_path

    if not s3.exists(s3_tar_path):
        return None

    local_tar_path = os.path.join(local_temp_dir, tar_name)
    timestamp_tag = final_zarr_name.replace(".zarr", "")
    extract_dir = os.path.join(local_temp_dir, f"extract_{timestamp_tag}")

    try:
        s3.get_file(s3_tar_path, local_tar_path)
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")

        extracted_source = os.path.join(extract_dir, extracted_store_name)
        if os.path.exists(extracted_source):
            shutil.move(extracted_source, local_zarr_path)

        if os.path.exists(local_zarr_path):
            _validate_historic_zarr_variables(local_zarr_path, expected_vars)
    except Exception:
        shutil.rmtree(local_zarr_path, ignore_errors=True)
        _delete_historic_archive_from_s3(s3, s3_tar_path)
        raise
    finally:
        if os.path.exists(local_tar_path):
            os.remove(local_tar_path)
        shutil.rmtree(extract_dir, ignore_errors=True)

    return local_zarr_path if os.path.exists(local_zarr_path) else None


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
def getGribList(FH_forecastsub, matchStrings):
    try:
        gribList = [
            str(Path(x.get_localFilePath(matchStrings)).expand())
            for x in FH_forecastsub.file_exists
        ]
    except (ValueError, OSError, KeyError, IndexError, RuntimeError):
        print("Download Failure 1, wait 20 seconds and retry")
        time.sleep(20)
        FH_forecastsub.download(matchStrings, verbose=False)
        try:
            gribList = [
                str(Path(x.get_localFilePath(matchStrings)).expand())
                for x in FH_forecastsub.file_exists
            ]
        except (ValueError, OSError, KeyError, IndexError, RuntimeError):
            print("Download Failure 2, wait 20 seconds and retry")
            time.sleep(20)
            FH_forecastsub.download(matchStrings, verbose=False)
            try:
                gribList = [
                    str(Path(x.get_localFilePath(matchStrings)).expand())
                    for x in FH_forecastsub.file_exists
                ]
            except (ValueError, OSError, KeyError, IndexError, RuntimeError):
                print("Download Failure 3, wait 20 seconds and retry")
                time.sleep(20)
                FH_forecastsub.download(matchStrings, verbose=False)
                try:
                    gribList = [
                        str(Path(x.get_localFilePath(matchStrings)).expand())
                        for x in FH_forecastsub.file_exists
                    ]
                except (ValueError, OSError, KeyError, IndexError, RuntimeError):
                    print("Download Failure 4, wait 20 seconds and retry")
                    time.sleep(20)
                    FH_forecastsub.download(matchStrings, verbose=False)
                    try:
                        gribList = [
                            str(Path(x.get_localFilePath(matchStrings)).expand())
                            for x in FH_forecastsub.file_exists
                        ]
                    except (ValueError, OSError, KeyError, IndexError, RuntimeError):
                        print("Download Failure 5, wait 20 seconds and retry")
                        time.sleep(20)
                        FH_forecastsub.download(matchStrings, verbose=False)
                        try:
                            gribList = [
                                str(Path(x.get_localFilePath(matchStrings)).expand())
                                for x in FH_forecastsub.file_exists
                            ]
                        except (
                            ValueError,
                            OSError,
                            KeyError,
                            IndexError,
                            RuntimeError,
                        ):
                            print("Download Failure 6, Fail")
                            exit(1)
    return gribList


def validate_grib_stats(gribCheck):
    """
    Inspect gribCheck.stdout (from `wgrib2 … -stats`) for min/max values,
    print any out-of-range records, and exit(10) if invalid data is found.

    Expects:
      - gribCheck.stdout: the full stdout string
      - globals: VALID_DATA_MIN, VALID_DATA_MAX
    """
    # extract all mins and maxs
    minValues = [float(m) for m in re.findall(r"min=([-\d\.eE]+)", gribCheck.stdout)]
    maxValues = [float(M) for M in re.findall(r"max=([-\d\.eE]+)", gribCheck.stdout)]

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
        for i, (mn, mx) in enumerate(zip(minValues, maxValues))
        if mn < VALID_DATA_MIN or mx > VALID_DATA_MAX
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
    nearest_vars: Optional[Union[int, Iterable[int]]] = None,  # var indices using NN
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

    T_new = int(len(idx0))
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
        nv = sorted(set(int(i) for i in nearest_vars))

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


def trailing_mean(conc: Optional[np.ndarray], window: int) -> Optional[np.ndarray]:
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


def calculate_aqi(
    pm25: np.ndarray,
    pm10: np.ndarray,
    o3: np.ndarray,
    no2: np.ndarray,
    so2: np.ndarray,
    co: np.ndarray,
    use_nowcast: bool = True,
) -> np.ndarray:
    """Calculate Air Quality Index (AQI) based on EPA standards.

    Returns the maximum AQI value among all pollutants for each grid cell and time.

    Args:
        pm25: PM2.5 concentration in µg/m³ (time, lat, lon).
        pm10: PM10 concentration in µg/m³ (time, lat, lon).
        o3: Ozone concentration in µg/m³ (time, lat, lon).
        no2: NO2 concentration in µg/m³ (time, lat, lon).
        so2: SO2 concentration in µg/m³ (time, lat, lon).
        co: CO concentration in µg/m³ (time, lat, lon).
        use_nowcast: Whether to use EPA NowCast for PM2.5/PM10 (default True).

    Returns:
        A numpy array of AQI values (0-500+ scale) with shape (time, lat, lon).
    """

    if use_nowcast:
        pm25_nowcast = calculate_nowcast_concentration(pm25, num_hours=12)
        pm10_nowcast = calculate_nowcast_concentration(pm10, num_hours=12)
        aqi_pm25 = np.interp(pm25_nowcast, PM25_BP, PM25_AQI)
        aqi_pm10 = np.interp(pm10_nowcast, PM10_BP, PM10_AQI)
    else:
        # When NowCast is disabled, use a 24-hour trailing average for PM2.5/PM10
        pm25_avg = trailing_mean(pm25, 24)
        pm10_avg = trailing_mean(pm10, 24)
        aqi_pm25 = np.interp(pm25_avg, PM25_BP, PM25_AQI)
        aqi_pm10 = np.interp(pm10_avg, PM10_BP, PM10_AQI)

    # Apply trailing averages appropriate for pollutant averaging windows
    # EPA AQI uses 8-hour averages for O3 and CO for hourly index values.
    # Use 1-hour trailing mean for NO2 / SO2 (effectively the instantaneous value).
    o3_avg = trailing_mean(o3, 8)
    o3_1h = trailing_mean(o3, 1)
    co_avg = trailing_mean(co, 8)
    no2_avg = trailing_mean(no2, 1)
    so2_avg = trailing_mean(so2, 1)

    def _interp_or_nan(arr, bp, aqi_arr, ref_shape):
        if arr is None:
            return np.full(ref_shape, np.nan, dtype=np.float32)
        return np.interp(arr, bp, aqi_arr)

    ref_shape = aqi_pm25.shape
    aqi_o3_8h = _interp_or_nan(o3_avg, O3_BP, O3_AQI, ref_shape)
    aqi_o3_1h = _interp_or_nan(o3_1h, O3_BP, O3_AQI, ref_shape)
    aqi_no2 = _interp_or_nan(no2_avg, NO2_BP, NO2_AQI, ref_shape)
    aqi_so2 = _interp_or_nan(so2_avg, SO2_BP, SO2_AQI, ref_shape)
    aqi_co = _interp_or_nan(co_avg, CO_BP, CO_AQI, ref_shape)

    # Include both 8-hour and 1-hour ozone AQI values (take the max later)
    stack = [
        aqi_pm25,
        aqi_pm10,
        aqi_o3_8h,
        aqi_o3_1h,
        aqi_no2,
        aqi_so2,
        aqi_co,
    ]

    return np.nanmax(np.stack(stack, axis=0), axis=0)


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


def check_historic_zarr(
    zarr_path: str,
    save_type: str,
    expected_vars: tuple,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
) -> bool:
    """
    Validates a historic Zarr store.

    Checks that the store exists, can be opened, contains all expected variables,
    and that data can be read from the last variable.
    If the store is invalid, it is deleted along with any corresponding .done file.

    Parameters:
    - zarr_path (str): Path to the Zarr store.
    - save_type (str): "S3" or "local" / "Download"
    - expected_vars (tuple): Tuple of expected variable names.
    - aws_access_key_id (str): AWS access key for S3.
    - aws_secret_access_key (str): AWS secret key for S3.

    Returns:
    - bool: True if the store is valid, False otherwise.
    """
    import os
    import shutil
    import traceback

    import zarr

    s3 = None
    try:
        if save_type == "S3":
            import s3fs

            s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
            if not s3.exists(zarr_path):
                return False

            store = zarr.storage.FsspecStore.from_url(
                zarr_path,
                storage_options={
                    "key": aws_access_key_id,
                    "secret": aws_secret_access_key,
                },
            )
        else:
            if not os.path.exists(zarr_path):
                return False

            store = zarr.storage.LocalStore(zarr_path)

        # Open the zarr group.
        z = zarr.open(store, mode="r")

        # Check if all expected variables exist
        store_vars = set(z.keys())
        expected_set = set(expected_vars)
        if not expected_set.issubset(store_vars):
            print(
                f"Missing variables in {zarr_path}. Expected subset {expected_set}, found {store_vars}"
            )
            raise ValueError("Missing variables in Zarr store")

        # Check the last variable has data by reading its last value
        last_var = expected_vars[-1]
        _ = z[last_var][-1, -1, -1]

        return True

    except (
        ValueError,
        IndexError,
        KeyError,
        zarr.errors.GroupNotFoundError,
        zarr.errors.NodeNotFoundError,
        zarr.errors.ArrayNotFoundError,
    ):
        print(f"### Historic Data Failure for {zarr_path}!")
        print(traceback.print_exc())

        # Delete the invalid store
        try:
            if save_type == "S3":
                if s3 is not None:
                    if s3.exists(zarr_path):
                        s3.rm(zarr_path, recursive=True)
                    done_file = zarr_path.replace(".zarr", ".done")
                    if s3.exists(done_file):
                        s3.rm(done_file)
            else:
                if os.path.exists(zarr_path):
                    shutil.rmtree(zarr_path, ignore_errors=True)
                done_file = zarr_path.replace(".zarr", ".done")
                if os.path.exists(done_file):
                    os.remove(done_file)
        except OSError as e:
            print(f"Failed to delete corrupt store {zarr_path}: {e}")

        return False
