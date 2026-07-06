"""Utilities for ECCC RAQDPS ingest processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

RAQDPS_BASE_URL = "https://dd.weather.gc.ca"
RAQDPS_MODEL_NAME = "RAQDPS"
RAQDPS_PRODUCT = "10km/grib2"
RAQDPS_GRID = "RLatLon0.09"
RAQDPS_LEVEL = "Sfc"
RAQDPS_VARIABLES = ("PM2.5", "PM10", "NO2", "O3", "SO2")
RAQDPS_OUTPUT_VARS = ("time",) + RAQDPS_VARIABLES
RAQDPS_FORECAST_HOURS = tuple(range(73))
RAQDPS_HISTORY_MERGE_HOURS = 12

# Molecular volume of an ideal gas at 25°C and 1 atm is ~24.465 L/mol.
MOLAR_VOLUME_25C = 24.465
KG_M3_TO_UG_M3 = 1e9
O3_PPB_TO_UG_M3 = 48.00 / MOLAR_VOLUME_25C
NO2_PPB_TO_UG_M3 = 46.01 / MOLAR_VOLUME_25C
SO2_PPB_TO_UG_M3 = 64.06 / MOLAR_VOLUME_25C


def normalize_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime with sub-hour fields removed."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def build_raqdps_filename(
    run_time: datetime,
    variable: str,
    forecast_hour: int,
    level: str = RAQDPS_LEVEL,
) -> str:
    """Build the canonical RAQDPS Datamart GRIB2 filename."""
    run_time = normalize_utc(run_time)
    return (
        f"{run_time:%Y%m%dT%HZ}_MSC_RAQDPS_{variable}_{level}_"
        f"{RAQDPS_GRID}_PT{forecast_hour:03d}H.grib2"
    )


def build_raqdps_url(
    run_time: datetime,
    variable: str,
    forecast_hour: int,
    level: str = RAQDPS_LEVEL,
) -> str:
    """Build the canonical RAQDPS Datamart GRIB2 URL."""
    run_time = normalize_utc(run_time)
    filename = build_raqdps_filename(run_time, variable, forecast_hour, level=level)
    return (
        f"{RAQDPS_BASE_URL}/{run_time:%Y%m%d}/WXO-DD/model_raqdps/"
        f"{RAQDPS_PRODUCT}/{run_time:%H}/{forecast_hour:03d}/{filename}"
    )


def candidate_raqdps_runs(now: datetime | None = None, count: int = 8) -> list[datetime]:
    """Return recent 00/12 UTC RAQDPS run candidates, newest first."""
    if now is None:
        now = datetime.now(timezone.utc)
    now = normalize_utc(now)
    cycle_hour = 12 if now.hour >= 12 else 0
    candidate = now.replace(hour=cycle_hour)
    return [candidate - timedelta(hours=12 * offset) for offset in range(count)]


def history_run_for_valid_time(valid_time: datetime) -> tuple[datetime, int]:
    """Map a valid time to the newest RAQDPS run with lead 000-012."""
    valid_time = normalize_utc(valid_time)
    cycle_hour = 12 if valid_time.hour >= 12 else 0
    run_time = valid_time.replace(hour=cycle_hour)
    forecast_hour = int((valid_time - run_time).total_seconds() // 3600)
    if forecast_hour < 0 or forecast_hour > RAQDPS_HISTORY_MERGE_HOURS:
        raise ValueError(f"Unable to map valid time to 0-12h RAQDPS lead: {valid_time}")
    return run_time, forecast_hour


def history_valid_times(base_time: datetime, hours: int) -> list[datetime]:
    """Return historical valid times ending one hour before base_time."""
    base_time = normalize_utc(base_time)
    return [base_time - timedelta(hours=offset) for offset in range(hours, 0, -1)]


def convert_to_ug_m3(values, variable: str):
    """Convert RAQDPS native pollutant units to µg/m³."""
    if variable in {"PM2.5", "PM10"}:
        return values * KG_M3_TO_UG_M3
    if variable == "O3":
        return values * O3_PPB_TO_UG_M3
    if variable == "NO2":
        return values * NO2_PPB_TO_UG_M3
    if variable == "SO2":
        return values * SO2_PPB_TO_UG_M3
    return values


def as_float32_array(values) -> np.ndarray:
    """Convert array-like values to float32 without copying unnecessarily."""
    return np.asarray(values, dtype=np.float32)
