"""Utilities for ECCC RAQDPS ingest processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from API.constants.unit_const import KG_M3_TO_UG_M3, MOL_MOL_TO_PPB

RAQDPS_BASE_URL = "https://dd.weather.gc.ca"
RAQDPS_MODEL_NAME = "RAQDPS"
RAQDPS_PRODUCT = "10km/grib2"
RAQDPS_GRID = "RLatLon0.09"
RAQDPS_LEVEL = "Sfc"
RAQDPS_VARIABLES = ("PM2.5", "PM10", "NO2", "O3", "SO2")
RAQDPS_OUTPUT_VARS = ("time",) + RAQDPS_VARIABLES
RAQDPS_FORECAST_HOURS = tuple(range(73))


def normalize_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime with sub-hour fields removed."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def herbie_naive_utc(dt: datetime) -> datetime:
    """Return a timezone-naive UTC datetime for Herbie internals."""
    return normalize_utc(dt).replace(tzinfo=None)


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


def candidate_raqdps_runs(
    now: datetime | None = None, count: int = 8
) -> list[datetime]:
    """Return recent 00/12 UTC RAQDPS run candidates, newest first."""
    if now is None:
        now = datetime.now(timezone.utc)
    now = normalize_utc(now)
    cycle_hour = 12 if now.hour >= 12 else 0
    candidate = now.replace(hour=cycle_hour)
    return [candidate - timedelta(hours=12 * offset) for offset in range(count)]


def convert_to_output_units(values, variable: str):
    """Convert RAQDPS native pollutant units to output units.

    Particulate matter remains mass concentration in µg/m³. Gaseous species are
    stored as ppb rather than mass concentration.
    """
    if variable in {"PM2.5", "PM10"}:
        return values * KG_M3_TO_UG_M3
    if variable in {"O3", "NO2", "SO2"}:
        return values * MOL_MOL_TO_PPB
    return values


def output_units_for_variable(variable: str) -> str | None:
    """Return the RAQDPS output unit label for a pollutant variable."""
    if variable in {"PM2.5", "PM10"}:
        return "µg m-3"
    if variable in {"O3", "NO2", "SO2"}:
        return "ppb"
    return None


def as_float32_array(values) -> np.ndarray:
    """Convert array-like values to float32 without copying unnecessarily."""
    return np.asarray(values, dtype=np.float32)
