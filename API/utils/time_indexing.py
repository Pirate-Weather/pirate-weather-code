"""Helpers for building day/hour index arrays."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
from pytz import utc
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from API.constants.shared_const import MISSING_DATA


@dataclass
class TimeIndexing:
    """Pre-computed day arrays and hourly-to-day index mappings."""

    day_array_grib: np.ndarray
    day_array_6am_grib: np.ndarray
    day_array_6pm_grib: np.ndarray
    # Midnight to midnight days (daily accumulations and means)
    hourly_day_index: np.ndarray
    # 6am to 6am days (daily summary, icon and precipitation type)
    hourly_day_6am_index: np.ndarray
    hourly_high_index: np.ndarray
    hourly_low_index: np.ndarray
    # 6am to 6pm and 6pm to 6am halves of each day (day/night forecast)
    hourly_half_day_index: np.ndarray
    hourly_half_night_index: np.ndarray


def localize_day_start(timezone_localizer, midnight: datetime.datetime):
    """Return the first instant of a local day from its naive midnight.

    Most zones change their clocks overnight, but some do it at midnight. Where
    clocks fall back onto midnight (Havana) it happens twice and the day starts at
    the first. Where they spring forward from it (Beirut, Santiago) it never
    happens: the standard-time reading of 00:00 is the instant the clocks jump to
    01:00, when the day starts.
    """
    try:
        return timezone_localizer.localize(midnight, is_dst=None)
    except AmbiguousTimeError:
        return timezone_localizer.localize(midnight, is_dst=True)
    except NonExistentTimeError:
        return timezone_localizer.localize(midnight, is_dst=False)


def _build_day_array(
    *,
    base_time: datetime.datetime,
    timezone_localizer,
    hour: int,
    days: int = 10,
) -> np.ndarray:
    base = datetime.datetime(
        year=base_time.year,
        month=base_time.month,
        day=base_time.day,
        hour=hour,
    )
    return np.array(
        [
            (
                localize_day_start(timezone_localizer, day)
                if hour == 0
                else timezone_localizer.localize(day)
            )
            .astimezone(utc)
            .timestamp()
            for day in (base + datetime.timedelta(days=i) for i in range(days))
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
    daily_days: int = 1,
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
    day_array_6am_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=6
    )
    day_array_6pm_grib = _build_day_array(
        base_time=base_time, timezone_localizer=timezone_localizer, hour=18
    )

    if time_machine and daily_days == 1:
        # Every daily statistic covers the whole requested local day (the grid has
        # no 6am-6am window to offer). Grid rows past its end belong to the next
        # day, an hour on a normal day and two on a 23-hour one.
        zero_index = (hour_array_grib >= day_array_grib[1]).astype(int)
        return TimeIndexing(
            day_array_grib=day_array_grib,
            day_array_6am_grib=day_array_6am_grib,
            day_array_6pm_grib=day_array_6pm_grib,
            hourly_day_index=zero_index.copy(),
            hourly_day_6am_index=zero_index.copy(),
            hourly_high_index=zero_index.copy(),
            hourly_low_index=zero_index.copy(),
            hourly_half_day_index=zero_index.copy(),
            hourly_half_night_index=zero_index.copy(),
        )

    hourly_day_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_day_6am_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_high_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_low_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_half_day_index = np.full(len(hour_array_grib), MISSING_DATA)
    hourly_half_night_index = np.full(len(hour_array_grib), MISSING_DATA)

    # Zero to 9 to account for the hours before 6am in day 8
    for d in range(9):
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
            hourly_day_6am_index,
            hour_array_grib,
            start=day_array_6am_grib[d],
            end=day_array_6am_grib[d + 1],
            include_start=True,
            include_end=False,
            value=d,
        )
        _assign_range(
            hourly_half_day_index,
            hour_array_grib,
            start=day_array_6am_grib[d],
            end=day_array_6pm_grib[d],
            include_start=True,
            include_end=False,
            value=d,
        )
        _assign_range(
            hourly_half_night_index,
            hour_array_grib,
            start=day_array_6pm_grib[d],
            end=day_array_6am_grib[d + 1],
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

    # Replace missing values by forward-filling from the last valid day index.
    # This handles hours that fall outside the expected day ranges, ensuring they are assigned to the most recent valid day index rather than being left as missing.
    # For DST
    if np.any(np.isnan(hourly_day_index)):
        mask = np.isnan(hourly_day_index)
        idx = np.where(~mask, np.arange(len(hourly_day_index)), 0)
        np.maximum.accumulate(idx, out=idx)
        hourly_day_index[mask] = hourly_day_index[idx[mask]]
        # If there are still NaNs (at the beginning), fill with 0.
        hourly_day_index[np.isnan(hourly_day_index)] = 0

    return TimeIndexing(
        day_array_grib=day_array_grib,
        day_array_6am_grib=day_array_6am_grib,
        day_array_6pm_grib=day_array_6pm_grib,
        hourly_day_index=_finalize_index(hourly_day_index),
        hourly_day_6am_index=_finalize_index(hourly_day_6am_index),
        hourly_high_index=_finalize_index(hourly_high_index),
        hourly_low_index=_finalize_index(hourly_low_index),
        hourly_half_day_index=_finalize_index(hourly_half_day_index),
        hourly_half_night_index=_finalize_index(hourly_half_night_index),
    )
