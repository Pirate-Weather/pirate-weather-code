"""Helpers for building day/hour index arrays."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
from pytz import utc

from API.constants.shared_const import MISSING_DATA


@dataclass
class TimeIndexing:
    """Pre-computed day arrays and hourly-to-day index mappings."""

    day_array_grib: np.ndarray
    day_array_4am_grib: np.ndarray
    day_array_4pm_grib: np.ndarray
    day_array_5pm_grib: np.ndarray
    day_array_6am_grib: np.ndarray
    day_array_6pm_grib: np.ndarray
    hourly_day_index: np.ndarray
    hourly_day_4am_index: np.ndarray
    hourly_high_index: np.ndarray
    hourly_low_index: np.ndarray
    hourly_day_4pm_index: np.ndarray
    hourly_night_4am_index: np.ndarray


def _build_day_array(
    *,
    base_time: datetime.datetime,
    timezone_localizer,
    hour: int,
    days: int = 10,
) -> np.ndarray:
    base = datetime.datetime(
        year=base_time.year, month=base_time.month, day=base_time.day, hour=hour
    )
    return np.array(
        [
            timezone_localizer.localize(base + datetime.timedelta(days=i))
            .astimezone(utc)
            .timestamp()
            for i in range(days)
        ]
    ).astype(np.int32)


def _assign_range(
    target: np.ndarray,
    source: np.ndarray,
    *,
    start: int,
    end: int,
    include_start: bool,
    include_end: bool,
    value: int,
) -> None:
    lower_cmp = source >= start if include_start else source > start
    upper_cmp = source <= end if include_end else source < end
    target[lower_cmp & upper_cmp] = value


def _finalize_index(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=-999).astype(int)


def calculate_time_indexing(
    *,
    base_time: datetime.datetime,
    timezone_localizer,
    hour_array_grib: np.ndarray,
    time_machine: bool,
    existing_day_array_grib: np.ndarray | None = None,
) -> TimeIndexing:
    """Create day arrays and map each hour to the correct day bucket."""
    day_array_grib = (
        existing_day_array_grib.astype(np.int32)
        if existing_day_array_grib is not None
        else _build_day_array(
            base_time=base_time, timezone_localizer=timezone_localizer, hour=0
        )
    )
    day_array_4am_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=4
    )
    day_array_4pm_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=16
    )
    day_array_5pm_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=17
    )
    day_array_6am_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=6
    )
    day_array_6pm_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=18
    )

    if time_machine:
        zero_index = np.full(len(hour_array_grib), int(0))
        return TimeIndexing(
            day_array_grib=day_array_grib,
            day_array_4am_grib=day_array_4am_grib,
            day_array_4pm_grib=day_array_4pm_grib,
            day_array_5pm_grib=day_array_5pm_grib,
            day_array_6am_grib=day_array_6am_grib,
            day_array_6pm_grib=day_array_6pm_grib,
            hourly_day_index=zero_index.copy(),
            hourly_day_4am_index=zero_index.copy(),
            hourly_high_index=zero_index.copy(),
            hourly_low_index=zero_index.copy(),
            hourly_day_4pm_index=zero_index.copy(),
            hourly_night_4am_index=zero_index.copy(),
        )

    hourly_day_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_day_4am_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_high_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_low_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_day_4pm_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_night_4am_index = np.full(len(hour_array_grib), MISSING_DATA)

    # Zero to 9 to account for the four hours in day 8
    for d in range(0, 9):
        _assign_range(
            hourly_day_index,
            hour_array_grib,
            start=day_array_grib[d],
            end=day_array_grib[d + 1],
            include_start=True,
            include_end=False,
            value=d,
        )
        _assign_range(
            hourly_day_4am_index,
            hour_array_grib,
            start=day_array_4am_grib[d],
            end=day_array_4am_grib[d + 1],
            include_start=True,
            include_end=False,
            value=d,
        )
        _assign_range(
            hourly_day_4pm_index,
            hour_array_grib,
            start=day_array_4am_grib[d],
            end=day_array_4pm_grib[d],
            include_start=True,
            include_end=True,
            value=d,
        )
        _assign_range(
            hourly_night_4am_index,
            hour_array_grib,
            start=day_array_5pm_grib[d],
            end=day_array_4am_grib[d + 1],
            include_start=True,
            include_end=False,
            value=d,
        )
        _assign_range(
            hourly_high_index,
            hour_array_grib,
            start=day_array_6am_grib[d],
            end=day_array_6pm_grib[d],
            include_start=False,
            include_end=True,
            value=d,
        )
        _assign_range(
            hourly_low_index,
            hour_array_grib,
            start=day_array_6pm_grib[d],
            end=day_array_6am_grib[d + 1],
            include_start=False,
            include_end=True,
            value=d,
        )

    return TimeIndexing(
        day_array_grib=day_array_grib,
        day_array_4am_grib=day_array_4am_grib,
        day_array_4pm_grib=day_array_4pm_grib,
        day_array_5pm_grib=day_array_5pm_grib,
        day_array_6am_grib=day_array_6am_grib,
        day_array_6pm_grib=day_array_6pm_grib,
        hourly_day_index=_finalize_index(hourly_day_index),
        hourly_day_4am_index=_finalize_index(hourly_day_4am_index),
        hourly_high_index=_finalize_index(hourly_high_index),
        hourly_low_index=_finalize_index(hourly_low_index),
        hourly_day_4pm_index=_finalize_index(hourly_day_4pm_index),
        hourly_night_4am_index=_finalize_index(hourly_night_4am_index),
    )
