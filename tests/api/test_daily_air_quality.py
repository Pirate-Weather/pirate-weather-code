"""Daily air quality summary built from the hourly AQI column."""

from unittest.mock import MagicMock

import numpy as np

from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.daily.builder import build_daily_section


def test_daily_air_quality_index_uses_hourly_aqi_column():
    """The daily AQI fields must summarise the hourly AQI values (mean, max with
    its time, min), not a pollutant concentration column such as PM2.5."""
    hours_per_day = 24
    InterPhour = np.zeros((hours_per_day, max(DATA_HOURLY.values()) + 1))
    InterPhour[:, DATA_HOURLY["time"]] = np.arange(hours_per_day) * 3600
    InterPhour[:, DATA_HOURLY["pm25"]] = 1
    InterPhour[:, DATA_HOURLY["aqi"]] = 10
    InterPhour[7, DATA_HOURLY["aqi"]] = 120
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
    # Mean = (23 * 10 + 120) / 24 = 14.6, reported as 15.
    assert day["airQualityIndex"] == 15
    assert day["airQualityIndexMax"] == 120
    assert day["airQualityIndexMaxTime"] == 7 * 3600
    assert day["airQualityIndexMin"] == 10
