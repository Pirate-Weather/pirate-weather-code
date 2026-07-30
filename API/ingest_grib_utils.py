"""GFS Ingest GRIB Processing Utilities

Helper functions for GRIB file processing, validation, and data transformations
used in the GFS Local Data Ingestion Script.

Author: Alexander Rey
"""

import logging
import os
import shlex
import sys
from datetime import datetime
from typing import Any, cast

import pandas as pd
from herbie import FastHerbie

from API.ingest_utils import (
    build_herbie_grib_list,
    download_herbie_with_retry,
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


def output_path(forecast_process_path: str, suffix: str) -> str:
    """Create output file path with consistent naming convention.

    Args:
        forecast_process_path: Base path for forecast processing
        suffix: File suffix to append (e.g., 'pgrb2_0p25_merged.grib')

    Returns:
        Full output path: {forecast_process_path}_{suffix}
    """
    return f"{forecast_process_path}_{suffix}"


def run_checked(cmd: str, description: str):
    """Execute shell command and exit on failure.

    Args:
        cmd: Shell command to execute
        description: Human-readable description for logging

    Raises:
        SystemExit: If command returns non-zero exit code
    """
    sp_out = run_command(cmd)
    if sp_out.returncode != 0:
        logger.error("%s failed.", description)
        logger.error(sp_out.stderr)
        sys.exit(1)
    return sp_out


def has_records(path: str) -> bool:
    """Check if inventory file exists and contains records.

    Args:
        path: Path to inventory file

    Returns:
        True if file exists and is non-empty
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def awk_path(path: str) -> str:
    """Escape a path for safe use inside double-quoted awk strings.

    Args:
        path: Path to escape

    Returns:
        Escaped path string
    """
    return path.replace("\\", "\\\\").replace('"', '\\"')


def download_and_validate_gfs_subset(
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
    gfs_forecast_hours: list[int] | None = None,
    default_forecast_hours: list[int] | None = None,
    run_date=None,
    forecast_hours: list[int] | None = None,
    priority=None,
    save_dir=None,
    path_search=None,
    expected_count: int | None = None,
    herbie_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Download a model GRIB subset, validate file count, and run wgrib2 checks.

    Args:
        model: Herbie model identifier (e.g., 'gfs', 'gdps').
        product: GRIB product name (e.g., 'pgrb2.0p25')
        search: Optional search pattern passed to ``Herbie.download``.
        dataset_name: Human-readable name for dataset
        base_time: Base forecast time
        wgrib2_exe: Path to wgrib2 executable
        gfs_forecast_hours: Backward-compatible default forecast hours for GFS callers.
        herbie_save_dir: Directory to save Herbie downloads
        herbie_download_retries: Number of retries for Herbie downloads
        herbie_retry_sleep_seconds: Sleep time between retries
        default_forecast_hours: Model-agnostic default forecast hours.
        run_date: Override base_time with specific date
        forecast_hours: Override default forecast hours
        priority: Source priority for Herbie downloads
        save_dir: Override default save directory
        path_search: Optional search pattern used to resolve local Herbie paths.
        expected_count: Optional expected local file count.
        herbie_kwargs: Additional model-specific ``FastHerbie`` keyword arguments.

    Returns:
        List of downloaded GRIB file paths
    """
    if run_date is None:
        run_date = base_time
    if forecast_hours is None:
        forecast_hours = default_forecast_hours or gfs_forecast_hours
    if forecast_hours is None:
        raise ValueError("forecast_hours or default_forecast_hours must be provided.")
    if priority is None:
        priority = ["aws", "google", "nomads"] if model == "gfs" else None
    if save_dir is None:
        save_dir = herbie_save_dir
    if herbie_kwargs is None:
        herbie_kwargs = {}
    if path_search is None:
        path_search = search

    run_date_dt = cast(datetime, pd.Timestamp(run_date).to_pydatetime())
    herbie_dates: list[datetime] = [run_date_dt]

    fast_herbie_kwargs: dict[str, Any] = {
        "model": model,
        "fxx": forecast_hours,
        "product": product,
        "verbose": False,
        "save_dir": save_dir,
    }
    if priority is not None:
        fast_herbie_kwargs["priority"] = priority
    fast_herbie_kwargs.update(herbie_kwargs)

    herbie_obj = FastHerbie(herbie_dates, **fast_herbie_kwargs)

    if expected_count is None:
        expected_count = len(forecast_hours)

    download_herbie_with_retry(
        herbie_obj=herbie_obj,
        search=search,
        expected_count=expected_count,
        dataset_name=dataset_name,
        retries=herbie_download_retries,
        retry_sleep_s=herbie_retry_sleep_seconds,
    )

    downloaded_count = len(herbie_obj.file_exists)

    if downloaded_count != expected_count:
        logger.error(
            "Download failed for %s: expected %s files but got %s.",
            dataset_name,
            expected_count,
            downloaded_count,
        )
        sys.exit(1)

    grib_files = build_herbie_grib_list(herbie_obj.file_exists, path_search)

    cmd_stats = f"{cat_gribs(grib_files)} | {quote_path(wgrib2_exe)} - -s -stats"

    grib_check = run_checked(cmd_stats, f"{dataset_name} GRIB validation")
    validate_grib_stats(grib_check)

    logger.info("%s passed GRIB validation.", dataset_name)

    return grib_files
