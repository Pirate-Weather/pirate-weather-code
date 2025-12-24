"""Time grid initialization helpers for hourly/minutely processing."""

from __future__ import annotations

import datetime
from typing import Tuple

import numpy as np
from pytz import utc

from API.constants.forecast_const import DATA_HOURLY, DATA_MINUTELY
from API.constants.shared_const import MISSING_DATA


def initialize_time_grids(
    *,
    base_time: datetime.datetime,
    base_day: datetime.datetime,
    daily_days: int,
    daily_day_hours: int,
    timezone_localizer,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create minute/hour arrays and initialize interpolation buffers."""
    minute_array = np.arange(
        base_time.astimezone(utc).replace(tzinfo=None),
        base_time.astimezone(utc).replace(tzinfo=None) + datetime.timedelta(minutes=61),
        datetime.timedelta(minutes=1),
    )

    minute_array_grib = (
        (minute_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    InterTminute = np.zeros((61, 5))
    InterPminute = np.full((61, max(DATA_MINUTELY.values()) + 1), MISSING_DATA)

    hour_array = np.arange(
        base_day.astimezone(utc).replace(tzinfo=None),
        base_day.astimezone(utc).replace(tzinfo=None)
        + datetime.timedelta(days=daily_days)
        + datetime.timedelta(hours=daily_day_hours),
        datetime.timedelta(hours=1),
    )

    InterPhour = np.full((len(hour_array), max(DATA_HOURLY.values()) + 1), MISSING_DATA)

    hour_array_grib = (
        (hour_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Pre-compute day arrays to avoid repeating in callers
    day_array_grib = np.array(
        [
            timezone_localizer.localize(
                datetime.datetime(
                    year=base_time.year, month=base_time.month, day=base_time.day
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    return (
        minute_array_grib,
        minute_array,
        InterTminute,
        InterPminute,
        InterPhour,
        hour_array_grib,
        hour_array,
        day_array_grib,
    )
