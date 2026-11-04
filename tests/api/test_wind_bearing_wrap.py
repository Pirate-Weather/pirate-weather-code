"""Wind bearings are angles: averages and blends must respect the wrap at north."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.current.metrics import InterpolationState, _interp_direction
from API.daily.builder import build_daily_section
from tests.helpers.text import make_hour


def _build_one_day(InterPhour, **kwargs):
    """Run the whole daily section for one day of hourly rows."""
    hours_per_day = len(InterPhour)
    day_index = np.zeros(hours_per_day, dtype=int)
    defaults = {
        "InterPhour": InterPhour,
        "hourlyDayIndex": day_index,
        "hourlyDay6amIndex": day_index,
        "hourlyHalfDayIndex": day_index,
        "hourlyHalfNightIndex": day_index,
        "hourlyHighIndex": day_index,
        "hourlyLowIndex": day_index,
        "daily_days": 1,
        "prepAccumUnit": 1.0,
        "prepIntensityUnit": 1.0,
        "windUnit": 1.0,
        "visUnits": 1.0,
        "tempUnits": 1.0,
        "extraVars": [],
        "summaryText": False,
        "translation": MagicMock(),
        "is_all_night": False,
        "is_all_day": False,
        "tz_name": "UTC",
        "icon": "default",
        "unitSystem": "si",
        "version": 2,
        "timeMachine": False,
        "tmExtra": False,
        "day_array_grib": np.array([0]),
        "day_array_6am_grib": np.array([6 * 3600, 30 * 3600]),
        "day_array_6pm_grib": np.array([18 * 3600]),
        "InterSday": np.zeros((1, max(DATA_DAY.values()) + 1)),
        "hourList_si": [
            {"time": i * 3600, "icon": "clear-day", "summary": "Clear"}
            for i in range(hours_per_day + 48)
        ],
        "pTypeMap": np.array(["none", "rain", "snow", "sleet", "ice"]),
        "pTextMap": np.array(["None", "Rain", "Snow", "Sleet", "Ice"]),
        "logger": MagicMock(),
        "loc_tag": "test_loc",
    }
    return build_daily_section(**(defaults | kwargs))


def test_daily_bearing_is_a_circular_mean():
    """A day of northerly winds either side of 0 degrees was averaged
    arithmetically and reported as southerly (San Francisco, GFS only: 161 for a
    true 360). Run the whole daily section so the response field is what's
    checked, along with a plain mean that must not change.
    """
    hours_per_day = 24
    InterPhour = np.zeros((hours_per_day, max(DATA_HOURLY.values()) + 1))
    InterPhour[:, DATA_HOURLY["time"]] = np.arange(hours_per_day) * 3600
    InterPhour[:, DATA_HOURLY["bearing"]] = [350.0, 20.0] * (hours_per_day // 2)
    InterPhour[:, DATA_HOURLY["wind"]] = 4.0

    result = _build_one_day(InterPhour)

    day = result.day_list[0]
    # The circular mean of 350 and 20 is 5; the bearing is truncated to an int
    assert day["windBearing"] in (4, 5)
    assert day["windSpeed"] == pytest.approx(4.0)


def test_current_direction_blends_across_north():
    """HRDPS and GDPS supply currently.windBearing from a stored direction,
    blended between the two surrounding hours; a linear blend turns a shift from
    350 to 10 degrees into a southerly wind.
    """
    merged = np.array([[350.0], [10.0]])
    state = InterpolationState(idx1=0, idx2=1, fac1=0.25, fac2=0.75)

    assert _interp_direction(merged, 0, state) == pytest.approx(5.0)


def test_todays_summary_only_describes_the_hours_left():
    """Past hours have their precipitation zeroed but keep fog and cloud, so a day
    that rained until now read "Rain starting in the evening", and morning fog was
    still in the afternoon's summary. Today starts at the current hour, and the
    following day keeps its whole window. Run the whole daily section, as the
    window is chosen there.
    """
    hours = 54
    InterPhour = np.zeros((hours, max(DATA_HOURLY.values()) + 1))
    InterPhour[:, DATA_HOURLY["time"]] = np.arange(hours) * 3600
    InterPhour[:, DATA_HOURLY["cloud"]] = 0.0
    InterPhour[:, DATA_HOURLY["vis"]] = 16000.0
    InterPhour[:, DATA_HOURLY["temp"]] = 15.0
    InterPhour[:, DATA_HOURLY["dew"]] = 5.0
    fog_hours = np.arange(6, 12)
    hourList_si = [
        make_hour(
            i * 3600,
            cloudCover=0.0,
            windSpeed=1.0,
            visibility=100 if i in fog_hours or i - 24 in fog_hours else 16000,
            temperature=15.0,
            dewPoint=14.9 if i in fog_hours or i - 24 in fog_hours else 5.0,
        )
        for i in range(hours)
    ]
    day_index = (np.arange(hours) // 24).astype(int)
    base = {
        "hourlyDayIndex": day_index,
        "hourlyDay6amIndex": ((np.arange(hours) - 6) // 24).astype(int),
        "hourlyHalfDayIndex": day_index,
        "hourlyHalfNightIndex": day_index,
        "hourlyHighIndex": day_index,
        "hourlyLowIndex": day_index,
        "daily_days": 2,
        "summaryText": True,
        "day_array_grib": np.array([0, 86400, 172800]),
        "day_array_6am_grib": np.array([6, 30, 54]) * 3600,
        "day_array_6pm_grib": np.array([18, 42]) * 3600,
        "InterSday": np.zeros((3, max(DATA_DAY.values()) + 1)),
        "hourList_si": hourList_si,
    }

    full_day = _build_one_day(InterPhour, **base)
    afternoon = _build_one_day(InterPhour, **base, current_hour_time=14 * 3600)

    assert "fog" in str(full_day.day_list_si[0]["summaryText"])
    today = str(afternoon.day_list_si[0]["summaryText"])
    assert "fog" not in today
    assert "for-day" not in today
    assert "fog" in str(afternoon.day_list_si[1]["summaryText"])
