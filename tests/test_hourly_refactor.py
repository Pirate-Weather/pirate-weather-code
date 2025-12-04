
import pytest
import numpy as np
from unittest.mock import MagicMock
from API.hourly.block import build_hourly_block
from API.constants.forecast_const import DATA_HOURLY, DATA_DAY

def test_build_hourly_block_structure():
    # Mock inputs
    num_hours = 48
    
    # Mock InterPhour (hours, vars)
    InterPhour = np.zeros((num_hours, max(DATA_HOURLY.values()) + 1))
    
    # Mock arrays
    hour_array_grib = np.arange(num_hours) * 3600
    hour_array = np.arange(num_hours)
    
    # Mock InterSday
    InterSday = np.zeros((2, max(DATA_DAY.values()) + 1))
    
    # Mock indices
    hourlyDayIndex = np.zeros(num_hours, dtype=int)
    
    # Mock inputs dictionaries
    InterThour_inputs = {
        "nbm_snow": np.zeros(num_hours),
        "nbm_ice": np.zeros(num_hours),
        "nbm_freezing_rain": np.zeros(num_hours),
        "nbm_rain": np.zeros(num_hours),
    }
    
    prcipIntensity_inputs = {}
    prcipProbability_inputs = {}
    temperature_inputs = np.zeros((num_hours, 1))
    dew_inputs = np.zeros((num_hours, 1))
    humidity_inputs = np.zeros((num_hours, 1))
    pressure_inputs = np.zeros((num_hours, 1))
    wind_inputs = np.zeros((num_hours, 1))
    gust_inputs = np.zeros((num_hours, 1))
    bearing_inputs = np.zeros((num_hours, 1))
    cloud_inputs = np.zeros((num_hours, 1))
    uv_inputs = np.zeros((num_hours, 1))
    vis_inputs = np.zeros((num_hours, 1))
    ozone_inputs = np.zeros((num_hours, 1))
    smoke_inputs = np.zeros((num_hours, 1))
    accum_inputs = np.zeros((num_hours, 1))
    nearstorm_inputs = {"dist": np.zeros((num_hours, 1)), "dir": np.zeros((num_hours, 1))}
    station_pressure_inputs = np.zeros((num_hours, 1))
    fire_inputs = np.zeros((num_hours, 1))
    feels_like_inputs = np.zeros((num_hours, 1))
    solar_inputs = np.zeros((num_hours, 1))
    cape_inputs = np.zeros((num_hours, 1))
    error_inputs = np.zeros((num_hours, 1))
    
    kwargs = {
        "source_list": ["nbm"],
        "InterPhour": InterPhour,
        "hour_array_grib": hour_array_grib,
        "hour_array": hour_array,
        "InterSday": InterSday,
        "hourlyDayIndex": hourlyDayIndex,
        "baseTimeOffset": 0,
        "timeMachine": False,
        "tmExtra": False,
        "prepIntensityUnit": 1.0,
        "prepAccumUnit": 1.0,
        "windUnit": 1.0,
        "visUnits": 1.0,
        "tempUnits": 1, # Celsius
        "humidUnit": 1.0,
        "extraVars": [],
        "summaryText": False,
        "icon": "default",
        "translation": MagicMock(),
        "unitSystem": "si",
        "is_all_night": False,
        "tz_name": "UTC",
        "InterThour_inputs": InterThour_inputs,
        "prcipIntensity_inputs": prcipIntensity_inputs,
        "prcipProbability_inputs": prcipProbability_inputs,
        "temperature_inputs": temperature_inputs,
        "dew_inputs": dew_inputs,
        "humidity_inputs": humidity_inputs,
        "pressure_inputs": pressure_inputs,
        "wind_inputs": wind_inputs,
        "gust_inputs": gust_inputs,
        "bearing_inputs": bearing_inputs,
        "cloud_inputs": cloud_inputs,
        "uv_inputs": uv_inputs,
        "vis_inputs": vis_inputs,
        "ozone_inputs": ozone_inputs,
        "smoke_inputs": smoke_inputs,
        "accum_inputs": accum_inputs,
        "nearstorm_inputs": nearstorm_inputs,
        "station_pressure_inputs": station_pressure_inputs,
        "era5_rain_intensity": None,
        "era5_snow_water_equivalent": None,
        "fire_inputs": fire_inputs,
        "feels_like_inputs": feels_like_inputs,
        "solar_inputs": solar_inputs,
        "cape_inputs": cape_inputs,
        "error_inputs": error_inputs,
        "version": 2,
    }

    result = build_hourly_block(**kwargs)
    
    assert isinstance(result, tuple)
    assert len(result) == 11
    
    hourList, hourList_si, hourIconList, hourTextList, dayZeroRain, dayZeroSnow, dayZeroIce, hourly_display, PTypeHour, PTextHour, InterPhour_res = result
    
    assert len(hourList) == num_hours
    assert isinstance(hourList[0], dict)
    assert "time" in hourList[0]
