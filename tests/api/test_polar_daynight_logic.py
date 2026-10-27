"""Sunrise/sunset fallback for polar day and polar night."""

import datetime

import numpy as np
import pytest
from astral import LocationInfo
from pytz import timezone

from API.utils.solar import calculate_solar_times

LONGYEARBYEN = LocationInfo(
    "Longyearbyen", "Svalbard", "Arctic/Longyearbyen", 78.22, 15.65
)


@pytest.mark.parametrize(
    ("month", "all_day", "all_night"),
    [
        pytest.param(6, True, False, id="june-polar-day"),
        pytest.param(12, False, True, id="december-polar-night"),
    ],
)
def test_polar_fallback_flags_and_sun_times(month, all_day, all_night):
    """astral raises when the sun never rises or sets; the fallback must flag the
    day correctly and place sunrise/sunset inside it, otherwise daily icons and
    day/night splits are wrong for every high-Arctic location for months.
    """
    base_day = timezone(LONGYEARBYEN.timezone).localize(
        datetime.datetime(2024, month, 15)
    )
    day_start = int(base_day.timestamp())

    sunrise, sunset, dawn, dusk, is_all_day, is_all_night = calculate_solar_times(
        day_index=0,
        base_day=base_day,
        location=LONGYEARBYEN,
        day_array_grib=np.array([day_start]),
        latitude=LONGYEARBYEN.latitude,
    )

    assert (is_all_day, is_all_night) == (all_day, all_night)
    if all_day:
        # Sun "rises" at the first second and "sets" at the last of the day.
        assert (sunrise, sunset) == (day_start + 1, day_start + 86399)
    else:
        # Sunrise and sunset collapse into the final seconds of the day.
        assert (sunrise, sunset) == (day_start + 86398, day_start + 86399)
    assert (dawn, dusk) == (sunrise, sunset)
