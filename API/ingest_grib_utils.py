"""GFS Ingest GRIB Processing Utilities

Helper functions for GRIB file processing, validation, and data transformations
used in the GFS Local Data Ingestion Script.

Author: Alexander Rey
"""

import logging
import os
import shlex
import sys
from typing import cast
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from herbie import FastHerbie

from API.ingest_utils import (
    build_herbie_grib_list,
    run_command,
    download_herbie_with_retry,
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


def deaverage_historic_duvb_hourly(duvb_dataarray: xr.DataArray) -> xr.DataArray:
    """Convert cumulative historic DUVB averages into hourly values.
    
    Deaccumulates DUVB (downward UV-B radiation) from 6-hour cumulative
    averages stored in the GFS historic data.
    
    Args:
        duvb_dataarray: Input xarray DataArray with shape (6, lat, lon)
        
    Returns:
        xarray DataArray with hourly DUVB values
    """
    uv_values = duvb_dataarray.values
    # Deaccumulation multipliers for each time step
    time_multipliers = np.arange(1, 7)
    time_multipliers = time_multipliers[:, np.newaxis, np.newaxis]

    first_timestep = uv_values[0, :, :]
    first_timestep = first_timestep[np.newaxis, :, :]

    # Deaccumulate: diff * multiplier + first value
    uv_hourly = np.concatenate(
        (first_timestep, np.diff(uv_values, axis=0) * time_multipliers[1:, :, :] + uv_values[0:5, :, :]),
        axis=0,
    )

    # Ensure no negative values
    uv_hourly[uv_hourly < 0] = 0
    return duvb_dataarray.copy(data=uv_hourly)


def download_and_validate_gfs_subset(
    *,
    product: str,
    search,
    dataset_name: str,
    base_time: pd.Timestamp,
    wgrib2_exe: str,
    gfs_forecast_hours: list[int],
    herbie_save_dir: str,
    herbie_download_retries: int,
    herbie_retry_sleep_seconds: int,
    run_date=None,
    forecast_hours=None,
    priority=None,
    save_dir=None,
) -> list[str]:
    """Download a GFS subset, validate file count, and run wgrib2 stats checks.
    
    Args:
        product: GRIB product name (e.g., 'pgrb2.0p25')
        search: Search pattern for variables to download
        dataset_name: Human-readable name for dataset
        base_time: Base forecast time
        wgrib2_exe: Path to wgrib2 executable
        gfs_forecast_hours: Default forecast hours
        herbie_save_dir: Directory to save Herbie downloads
        herbie_download_retries: Number of retries for Herbie downloads
        herbie_retry_sleep_seconds: Sleep time between retries
        run_date: Override base_time with specific date
        forecast_hours: Override default forecast hours
        priority: Source priority for Herbie downloads
        save_dir: Override default save directory
        
    Returns:
        List of downloaded GRIB file paths
    """
    if run_date is None:
        run_date = base_time
    if forecast_hours is None:
        forecast_hours = gfs_forecast_hours
    if priority is None:
        priority = ["aws", "google", "nomads"]
    if save_dir is None:
        save_dir = herbie_save_dir

    run_date_dt = cast(datetime, pd.Timestamp(run_date).to_pydatetime())
    herbie_dates: list[datetime] = [run_date_dt]

    herbie_obj = FastHerbie(
        herbie_dates,
        model="gfs",
        fxx=forecast_hours,
        product=product,
        verbose=False,
        priority=priority,
        save_dir=save_dir,
    )

    download_herbie_with_retry(
        herbie_obj=herbie_obj,
        search=search,
        expected_count=len(forecast_hours),
        dataset_name=dataset_name,
        retries=herbie_download_retries,
        retry_sleep_s=herbie_retry_sleep_seconds,
    )

    downloaded_count = len(herbie_obj.file_exists)
    expected_count = len(forecast_hours)

    if downloaded_count != expected_count:
        logger.error(
            "Download failed for %s: expected %s files but got %s.",
            dataset_name,
            expected_count,
            downloaded_count,
        )
        sys.exit(1)

    grib_files = build_herbie_grib_list(herbie_obj.file_exists, search)

    cmd_stats = (
        f"{cat_gribs(grib_files)} | "
        f"{quote_path(wgrib2_exe)} - -s -stats"
    )

    grib_check = run_checked(cmd_stats, f"{dataset_name} GRIB validation")
    validate_grib_stats(grib_check)

    logger.info("%s passed GRIB validation.", dataset_name)

    return grib_files
