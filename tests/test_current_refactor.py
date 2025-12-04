
import pytest
import numpy as np
from unittest.mock import MagicMock
from API.current.metrics import build_current_section, CurrentSection
from API.constants.forecast_const import DATA_CURRENT

def test_build_current_section_structure():
    # Mock inputs
    sourceList = ["hrrr_0-18", "hrrr_18-48"]
    hour_array_grib = np.array([100, 200, 300])
    minute_array_grib = np.array([150])
    InterPminute = np.zeros((1, 10))
    minuteItems = [{"precipIntensity": 0, "precipProbability": 0, "precipIntensityError": 0, "precipType": "none"}]
    minuteRainIntensity = np.zeros(1)
    minuteSnowIntensity = np.zeros(1)
    minuteSleetIntensity = np.zeros(1)
    InterSday = np.zeros((1, 21))
    
    # Mock other arguments with defaults or simple values
    kwargs = {
        "sourceList": sourceList,
        "hour_array_grib": hour_array_grib,
        "minute_array_grib": minute_array_grib,
        "InterPminute": InterPminute,
        "minuteItems": minuteItems,
        "minuteRainIntensity": minuteRainIntensity,
        "minuteSnowIntensity": minuteSnowIntensity,
        "minuteSleetIntensity": minuteSleetIntensity,
        "InterSday": InterSday,
        "dayZeroRain": 0.0,
        "dayZeroSnow": 0.0,
        "dayZeroIce": 0.0,
        "prepAccumUnit": 1.0,
        "prepIntensityUnit": 1.0,
        "windUnit": 1.0,
        "visUnits": 1.0,
        "tempUnits": 1.0,
        "humidUnit": 1.0,
        "extraVars": [],
        "summaryText": False,
        "translation": MagicMock(),
        "icon": "default",
        "unitSystem": "si",
        "version": 2,
        "timeMachine": False,
        "tmExtra": False,
        "lat": 0.0,
        "lon_IN": 0.0,
        "tz_name": "UTC",
        "tz_offset": 0.0,
        "ETOPO": 0.0,
        "elevUnit": 1.0,
        "dataOut_rtma_ru": None,
        "hrrrSubHInterpolation": None,
        "HRRR_Merged": np.zeros((3, 20)), # Mock merged data
        "NBM_Merged": None,
        "ECMWF_Merged": None,
        "GFS_Merged": None,
        "ERA5_MERGED": None,
        "NBM_Fire_Merged": None,
        "logger": MagicMock(),
        "loc_tag": "test_loc",
    }

    result = build_current_section(**kwargs)
    
    assert isinstance(result, CurrentSection)
    assert isinstance(result.currently, dict)
    assert isinstance(result.interp_current, np.ndarray)
    
    # Check if time is correctly set (interpolated or closest)
    # In this case, 150 is exactly between 100 and 200.
    # The logic in build_current_section handles this.
    assert result.currently["time"] == 150

