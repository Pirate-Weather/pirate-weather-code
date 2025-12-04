
import pytest
import numpy as np
from unittest.mock import MagicMock
from API.daily.builder import build_daily_section, DailySection
from API.constants.forecast_const import DATA_HOURLY, DATA_DAY

def test_build_daily_section_structure():
    # Mock inputs
    daily_days = 2
    hours_per_day = 24
    total_hours = daily_days * hours_per_day
    
    # Mock InterPhour (hours, vars)
    # Assuming max index in DATA_HOURLY is around 20-30
    InterPhour = np.zeros((total_hours, max(DATA_HOURLY.values()) + 1))
    
    # Mock indices
    # hourlyDayIndex maps each hour to a day index (0 to daily_days-1)
    hourlyDayIndex = np.repeat(np.arange(daily_days), hours_per_day)
    
    # Other indices can be same for simplicity in this structural test
    hourlyDay4amIndex = hourlyDayIndex
    hourlyDay4pmIndex = hourlyDayIndex
    hourlyNight4amIndex = hourlyDayIndex
    hourlyHighIndex = hourlyDayIndex
    hourlyLowIndex = hourlyDayIndex
    
    # Mock day arrays
    day_array_grib = np.arange(daily_days) * 86400
    day_array_4am_grib = day_array_grib + 4 * 3600
    day_array_5pm_grib = day_array_grib + 17 * 3600
    
    # Mock InterSday
    InterSday = np.zeros((daily_days, max(DATA_DAY.values()) + 1))
    
    # Mock hourList_si
    hourList_si = [{"time": i * 3600, "icon": "clear-day", "summary": "Clear"} for i in range(total_hours + 48)] # Add buffer
    
    # Mock maps
    pTypeMap = np.array(["none", "rain", "snow", "sleet", "ice"])
    pTextMap = np.array(["None", "Rain", "Snow", "Sleet", "Ice"])
    
    kwargs = {
        "InterPhour": InterPhour,
        "hourlyDayIndex": hourlyDayIndex,
        "hourlyDay4amIndex": hourlyDay4amIndex,
        "hourlyDay4pmIndex": hourlyDay4pmIndex,
        "hourlyNight4amIndex": hourlyNight4amIndex,
        "hourlyHighIndex": hourlyHighIndex,
        "hourlyLowIndex": hourlyLowIndex,
        "daily_days": daily_days,
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
        "day_array_grib": day_array_grib,
        "day_array_4am_grib": day_array_4am_grib,
        "day_array_5pm_grib": day_array_5pm_grib,
        "InterSday": InterSday,
        "hourList_si": hourList_si,
        "pTypeMap": pTypeMap,
        "pTextMap": pTextMap,
        "logger": MagicMock(),
        "loc_tag": "test_loc",
    }

    result = build_daily_section(**kwargs)
    
    assert isinstance(result, DailySection)
    assert len(result.day_list) == daily_days
    assert len(result.day_list_si) == daily_days
    assert isinstance(result.day_list[0], dict)
    assert "time" in result.day_list[0]
