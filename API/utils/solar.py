"""
Utilities for calculating sunrise/sunset information.
"""

from __future__ import annotations

import datetime

import numpy as np
from astral import LocationInfo
from astral.sun import sun
from pytz import utc

from API.utils.geo import _polar_is_all_day

UNIX_EPOCH = np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))


def datetime_to_unix_seconds(value: datetime.datetime) -> np.int32:
    """Convert a timezone-aware datetime to unix seconds in UTC."""
    return (
        (np.datetime64(value.astimezone(utc).replace(tzinfo=None)) - UNIX_EPOCH)
        .astype("timedelta64[s]")
        .astype(np.int32)
    )


def timedelta_to_seconds(delta: np.timedelta64) -> np.int32:
    """Convert a numpy timedelta to integer seconds."""
    return delta.astype("timedelta64[s]").astype(np.int32)


def calculate_solar_times(
    *,
    day_index: int,
    base_day: datetime.datetime,
    location: LocationInfo,
    day_array_grib: np.ndarray,
    latitude: float,
) -> tuple[np.int32, np.int32, np.int32, np.int32, bool, bool]:
    """Return sunrise, sunset, dawn, dusk in unix seconds plus polar flags."""
    try:
        solar_times = sun(
            location.observer, date=base_day + datetime.timedelta(days=day_index)
        )
        return (
            datetime_to_unix_seconds(solar_times["sunrise"]),
            datetime_to_unix_seconds(solar_times["sunset"]),
            datetime_to_unix_seconds(solar_times["dawn"]),
            datetime_to_unix_seconds(solar_times["dusk"]),
            False,
            False,
        )
    except ValueError:
        # Handle polar day/night fallbacks using the existing heuristic.
        day_start = day_array_grib[day_index]
        one_second = timedelta_to_seconds(np.timedelta64(1, "s"))
        full_day = timedelta_to_seconds(np.timedelta64(1, "D"))

        if _polar_is_all_day(latitude, base_day.month):
            sunrise = day_start + one_second
            sunset = day_start + full_day - one_second
            return sunrise, sunset, sunrise, sunset, True, False

        sunrise = day_start + full_day - (one_second * 2)
        sunset = day_start + full_day - one_second
        return sunrise, sunset, sunrise, sunset, False, True
