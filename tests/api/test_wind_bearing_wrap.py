"""Wind bearings are angles: averages and blends must respect the wrap at north."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.current.metrics import InterpolationState, _interp_direction
from API.daily.builder import build_daily_section


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
    day_index = np.zeros(hours_per_day, dtype=int)

    result = build_daily_section(
        InterPhour=InterPhour,
        hourlyDayIndex=day_index,
        hourlyDay6amIndex=day_index,
        hourlyHalfDayIndex=day_index,
        hourlyHalfNightIndex=day_index,
        hourlyHighIndex=day_index,
        hourlyLowIndex=day_index,
        daily_days=1,
        prepAccumUnit=1.0,
        prepIntensityUnit=1.0,
        windUnit=1.0,
        visUnits=1.0,
        tempUnits=1.0,
        extraVars=[],
        summaryText=False,
        translation=MagicMock(),
        is_all_night=False,
        is_all_day=False,
        tz_name="UTC",
        icon="default",
        unitSystem="si",
        version=2,
        timeMachine=False,
        tmExtra=False,
        day_array_grib=np.array([0]),
        day_array_6am_grib=np.array([6 * 3600, 30 * 3600]),
        day_array_6pm_grib=np.array([18 * 3600]),
        InterSday=np.zeros((1, max(DATA_DAY.values()) + 1)),
        hourList_si=[
            {"time": i * 3600, "icon": "clear-day", "summary": "Clear"}
            for i in range(hours_per_day + 48)
        ],
        pTypeMap=np.array(["none", "rain", "snow", "sleet", "ice"]),
        pTextMap=np.array(["None", "Rain", "Snow", "Sleet", "Ice"]),
        logger=MagicMock(),
        loc_tag="test_loc",
    )

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
