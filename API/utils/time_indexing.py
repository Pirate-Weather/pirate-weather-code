"""Helpers for building day/hour index arrays."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

# Zero to 9 to account for the hours before 6am in day 8
INDEXED_DAYS = 9
# Hours outside every window of a daily statistic
NO_DAY_INDEX = -999


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


def _build_day_array(
    *,
    base_time: datetime.datetime,
    local_tz,
    hour: int,
    days: int = 10,
) -> np.ndarray:
    # Adding days to a zone-aware time moves the wall clock, so each stays at this
    # local hour across a DST change. Fold 0 is the start of a day where clocks change
    # at midnight: see prepare_initial_request.
    base = datetime.datetime(
        year=base_time.year,
        month=base_time.month,
        day=base_time.day,
        hour=hour,
        tzinfo=local_tz,
    )
    return np.array(
        [
            (base + datetime.timedelta(days=i)).astimezone(datetime.UTC).timestamp()
            for i in range(days)
        ]
    ).astype(np.int32)


def _window_index(
    hour_array_grib: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    include_start: bool,
) -> np.ndarray:
    """The day whose window holds each hour, or NO_DAY_INDEX.

    A day's window never overlaps the next day's, so the only candidate is the
    latest window starting at or before the hour. Windows include their start and
    exclude their end, or the reverse when ``include_start`` is False.
    """
    side = "right" if include_start else "left"
    latest_start = (
        np.searchsorted(starts[:INDEXED_DAYS], hour_array_grib, side=side) - 1
    )
    candidate = np.clip(latest_start, 0, INDEXED_DAYS - 1)
    before_end = (
        hour_array_grib < ends[candidate]
        if include_start
        else hour_array_grib <= ends[candidate]
    )
    return np.where((latest_start >= 0) & before_end, candidate, NO_DAY_INDEX)


def calculate_time_indexing(
    *,
    base_time: datetime.datetime,
    local_tz,
    hour_array_grib: np.ndarray,
    time_machine: bool,
    daily_days: int = 1,
    existing_day_array_grib: np.ndarray | None = None,
) -> TimeIndexing:
    """Create day arrays and map each hour to the correct day bucket."""
    day_array_grib = (
        existing_day_array_grib.astype(np.int32)
        if existing_day_array_grib is not None
        else _build_day_array(base_time=base_time, local_tz=local_tz, hour=0)
    )
    day_array_6am_grib = _build_day_array(
        base_time=base_time, local_tz=local_tz, hour=6
    )
    day_array_6pm_grib = _build_day_array(
        base_time=base_time, local_tz=local_tz, hour=18
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

    next_day_6am = day_array_6am_grib[1:]
    hourly_day_index = _window_index(
        hour_array_grib, day_array_grib, day_array_grib[1:], include_start=True
    )
    hourly_day_6am_index = _window_index(
        hour_array_grib, day_array_6am_grib, next_day_6am, include_start=True
    )
    hourly_half_day_index = _window_index(
        hour_array_grib, day_array_6am_grib, day_array_6pm_grib, include_start=True
    )
    hourly_half_night_index = _window_index(
        hour_array_grib, day_array_6pm_grib, next_day_6am, include_start=True
    )
    hourly_high_index = _window_index(
        hour_array_grib, day_array_6am_grib, day_array_6pm_grib, include_start=False
    )
    hourly_low_index = _window_index(
        hour_array_grib, day_array_6pm_grib, next_day_6am, include_start=False
    )

    # Replace missing values by forward-filling from the last valid day index.
    # This handles hours that fall outside the expected day ranges, ensuring they are assigned to the most recent valid day index rather than being left as missing.
    # For DST
    missing_day = hourly_day_index == NO_DAY_INDEX
    if np.any(missing_day):
        idx = np.where(~missing_day, np.arange(len(hourly_day_index)), 0)
        np.maximum.accumulate(idx, out=idx)
        hourly_day_index[missing_day] = hourly_day_index[idx[missing_day]]
        # If there are still missing days (at the beginning), use the first day.
        hourly_day_index[hourly_day_index == NO_DAY_INDEX] = 0

    return TimeIndexing(
        day_array_grib=day_array_grib,
        day_array_6am_grib=day_array_6am_grib,
        day_array_6pm_grib=day_array_6pm_grib,
        hourly_day_index=hourly_day_index,
        hourly_day_6am_index=hourly_day_6am_index,
        hourly_high_index=hourly_high_index,
        hourly_low_index=hourly_low_index,
        hourly_half_day_index=hourly_half_day_index,
        hourly_half_night_index=hourly_half_night_index,
    )
