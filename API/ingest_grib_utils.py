"""Ingest GRIB download and wgrib2 utilities.

Shared helpers for downloading model GRIB subsets with Herbie, checking them with
``wgrib2 -stats``, and running wgrib2 commands with checked exit status.

Author: Alexander Rey
"""

import logging
import shlex
import time
from datetime import datetime
from typing import Any, cast

import pandas as pd
from herbie import FastHerbie

from API.ingest_utils import (
    build_herbie_grib_list,
    configure_herbie_request_timeouts,
    download_herbie_with_retry,
    positive_int_env,
    run_command,
    validate_grib_stats,
)

logger = logging.getLogger(__name__)


def quote_path(path: str) -> str:
    """Shell-quote a file path for safe use in shell commands.

    Args:
        path: File path to quote

    Returns:
        Shell-quoted path string
    """
    return shlex.quote(str(path))


def cat_gribs(grib_files: list[str]) -> str:
    """Build a shell command to concatenate GRIB files.

    Args:
        grib_files: List of GRIB file paths

    Returns:
        Shell cat command with quoted paths
    """
    return "cat " + " ".join(quote_path(path) for path in grib_files)


def run_checked(cmd: str, description: str):
    """Run a command (optionally with one pipe) and raise if it fails.

    Args:
        cmd: Command to execute, e.g. ``cat a b | wgrib2 - -netcdf out.nc``
        description: What the command does, including the model, for the error

    Raises:
        RuntimeError: If the command returns a non-zero exit code, so a failed
            conversion stops the ingest instead of surfacing later as an
            unrelated array error.
    """
    sp_out = run_command(cmd)
    if sp_out.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {sp_out.returncode}.\n"
            f"Command: {cmd}\n{sp_out.stderr}"
        )
    return sp_out


def check_grib_stats(
    grib_files: list[str],
    wgrib2_exe: str,
    dataset_name: str,
    excluded_variables: list[str] | None = None,
) -> None:
    """Run ``wgrib2 -s -stats`` over GRIB files and raise if any record is out of range.

    This is the gate that keeps corrupt upstream GRIB data from being published.
    """
    cmd_stats = (
        f"{cat_gribs(grib_files)} | {quote_path(wgrib2_exe.strip())} - -s -stats"
    )
    grib_check = run_checked(cmd_stats, f"{dataset_name} GRIB validation")
    if not validate_grib_stats(grib_check, excluded_variables=excluded_variables):
        raise RuntimeError(f"{dataset_name} failed GRIB validation.")
    logger.info("%s passed GRIB validation.", dataset_name)


def download_and_validate_subset(
    *,
    product: str,
    dataset_name: str,
    base_time: pd.Timestamp,
    wgrib2_exe: str,
    herbie_save_dir: str,
    herbie_download_retries: int,
    herbie_retry_sleep_seconds: int,
    model: str = "gfs",
    search=None,
    forecast_hours: list[int],
    run_date=None,
    priority=None,
    save_dir=None,
    path_search=None,
    expected_count: int | None = None,
    expected_forecast_hours: list[int] | None = None,
    excluded_stats_variables: list[str] | None = None,
    skip_wgrib2_validation: bool = False,
    herbie_kwargs: dict[str, Any] | None = None,
    download_max_threads: int | None = None,
    download_verbose: bool = True,
) -> list[str]:
    """Download a model GRIB subset, validate file count, and run wgrib2 checks.

    Args:
        model: Herbie model identifier (e.g., 'gfs', 'gdps').
        product: GRIB product name (e.g., 'pgrb2.0p25')
        search: Optional search pattern passed to ``Herbie.download``.
        dataset_name: Human-readable name for dataset
        base_time: Base forecast time
        wgrib2_exe: Path to wgrib2 executable
        forecast_hours: Forecast lead hours to download.
        herbie_save_dir: Directory to save Herbie downloads
        herbie_download_retries: Number of retries for Herbie downloads
        herbie_retry_sleep_seconds: Sleep time between retries
        run_date: Override base_time with specific date
        priority: Source priority for Herbie downloads
        save_dir: Override default save directory
        path_search: Optional search pattern used to resolve local Herbie paths.
        expected_count: Optional expected local file count.
        expected_forecast_hours: Optional forecast hours expected to be available when
            this differs from the requested forecast hours.
        excluded_stats_variables: Optional GRIB variable names to skip in generic
            min/max stats validation.
        skip_wgrib2_validation: Skip the ``wgrib2 -s -stats`` validation step.
        herbie_kwargs: Additional model-specific ``FastHerbie`` keyword arguments.
        download_max_threads: Maximum concurrent subset downloads.
        download_verbose: Whether Herbie should log each downloaded file.

    Returns:
        List of downloaded GRIB file paths
    """
    if run_date is None:
        run_date = base_time
    if priority is None:
        priority = ["aws", "google", "nomads"] if model == "gfs" else None
    if save_dir is None:
        save_dir = herbie_save_dir
    if herbie_kwargs is None:
        herbie_kwargs = {}
    if path_search is None:
        path_search = search

    configure_herbie_request_timeouts()

    run_date_dt = cast(datetime, pd.Timestamp(run_date).to_pydatetime())
    herbie_dates: list[datetime] = [run_date_dt]

    fast_herbie_kwargs: dict[str, Any] = {
        "model": model,
        "fxx": forecast_hours,
        "product": product,
        "verbose": False,
        "save_dir": save_dir,
        "max_threads": positive_int_env("herbie_download_threads", 20),
    }
    if priority is not None:
        fast_herbie_kwargs["priority"] = priority
    fast_herbie_kwargs.update(herbie_kwargs)

    if expected_count is None:
        expected_count = len(expected_forecast_hours or forecast_hours)

    attempts = max(1, herbie_download_retries)
    herbie_obj = None
    for attempt in range(1, attempts + 1):
        try:
            herbie_obj = FastHerbie(herbie_dates, **fast_herbie_kwargs)
            download_herbie_with_retry(
                herbie_obj=herbie_obj,
                search=search,
                expected_count=expected_count,
                dataset_name=dataset_name,
                retries=1,
                retry_sleep_s=herbie_retry_sleep_seconds,
                max_threads=download_max_threads,
                overwrite_first_attempt=attempt > 1,
                verbose=download_verbose,
            )
            break
        except Exception as exc:
            if attempt == attempts:
                logger.exception(
                    "%s download failed after %d attempts",
                    dataset_name,
                    attempts,
                )
                raise

            sleep_s = herbie_retry_sleep_seconds * attempt
            logger.warning(
                "%s download attempt %d/%d failed (%s). Retrying in %ss",
                dataset_name,
                attempt,
                attempts,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    if herbie_obj is None:
        raise RuntimeError(f"{dataset_name} download did not create a Herbie object.")

    downloaded_count = len(herbie_obj.file_exists)

    if downloaded_count != expected_count:
        raise RuntimeError(
            f"Download failed for {dataset_name}: expected {expected_count} files "
            f"but got {downloaded_count}."
        )

    grib_files = build_herbie_grib_list(herbie_obj.file_exists, path_search)

    if skip_wgrib2_validation:
        logger.info("%s skipped GRIB validation.", dataset_name)
        return grib_files

    check_grib_stats(grib_files, wgrib2_exe, dataset_name, excluded_stats_variables)
    return grib_files
