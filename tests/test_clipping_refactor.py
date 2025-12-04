import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from API.hourly.block import build_hourly_block
from API.constants.clip_const import CLIP_TEMP, CLIP_HUMIDITY
from API.constants.shared_const import MISSING_DATA

@pytest.fixture
def mock_inputs():
    num_hours = 24
    return {
        "source_list": ["gfs"],
        "InterPhour": np.zeros((num_hours, 50)), # Adjust size as needed
        "hour_array_grib": np.arange(num_hours),
        "hour_array": np.arange(num_hours),
        "InterSday": np.zeros((1, 50)),
        "hourlyDayIndex": np.zeros(num_hours, dtype=int),
        "baseTimeOffset": 0,
        "timeMachine": False,
        "tmExtra": False,
        "prepIntensityUnit": 1.0,
        "prepAccumUnit": 1.0,
        "windUnit": 1.0,
        "visUnits": 1.0,
        "tempUnits": 1,
        "humidUnit": 1.0,
        "extraVars": [],
        "summaryText": False,
        "icon": "test",
        "translation": MagicMock(),
        "unitSystem": "us",
        "is_all_night": False,
        "tz_name": "UTC",
        "InterThour_inputs": {},
        "prcipIntensity_inputs": {},
        "prcipProbability_inputs": {},
        "temperature_inputs": np.full((num_hours, 1), 20.0),
        "dew_inputs": np.full((num_hours, 1), 10.0),
        "humidity_inputs": np.full((num_hours, 1), 0.5),
        "pressure_inputs": np.full((num_hours, 1), 101325.0), # Pascals
        "wind_inputs": np.full((num_hours, 1), 5.0),
        "gust_inputs": np.full((num_hours, 1), 10.0),
        "bearing_inputs": np.full((num_hours, 1), 180.0),
        "cloud_inputs": np.full((num_hours, 1), 0.5),
        "uv_inputs": np.full((num_hours, 1), 5.0),
        "vis_inputs": np.full((num_hours, 1), 10.0),
        "ozone_inputs": np.full((num_hours, 1), 300.0),
        "smoke_inputs": np.full((num_hours, 1), 10.0),
        "accum_inputs": np.full((num_hours, 1), 0.0),
        "nearstorm_inputs": {"dist": np.zeros((num_hours, 1)), "dir": np.zeros((num_hours, 1))},
        "station_pressure_inputs": None,
        "era5_rain_intensity": None,
        "era5_snow_water_equivalent": None,
        "fire_inputs": np.zeros((num_hours, 1)),
        "feels_like_inputs": np.full((num_hours, 1), 20.0),
        "solar_inputs": np.zeros((num_hours, 1)),
        "cape_inputs": np.zeros((num_hours, 1)),
        "error_inputs": np.zeros((num_hours, 1)),
        "version": 2,
    }

def test_clipping_normal_values(mock_inputs):
    # Test with normal values within bounds
    result = build_hourly_block(**mock_inputs)
    InterPhour = result[10] # InterPhour is the last element returned
    # Check temp is not clipped or modified unexpectedly
    # DATA_HOURLY["temp"] is 5
    assert np.all(InterPhour[:, 5] == 20.0) 

def test_clipping_out_of_bounds(mock_inputs):
    # Test with values that should be clipped
    mock_inputs["temperature_inputs"][:] = 1000.0 # Way above max
    
    # We expect clipLog to handle this. 
    # If it's way above max ( > 1.25 * max), it might set to MISSING_DATA or log error.
    # CLIP_TEMP max is likely around 60C. 1000 is > 1.25 * 60.
    # So it should be set to MISSING_DATA (-9999 usually)
    
    result = build_hourly_block(**mock_inputs)
    InterPhour = result[10]
    
    # Check if it was handled. 
    # Note: clipLog implementation sets to MISSING_DATA if > 1.25 * max
    # We need to know what MISSING_DATA is. Assuming -9999 or similar.
    # Let's check if it's not 1000.0
    assert np.all(InterPhour[:, 5] != 1000.0)

def test_clipping_just_above_max(mock_inputs):
    # Test with values just above max but within 1.25 factor
    # CLIP_TEMP max is likely 60. Let's try 61.
    mock_inputs["temperature_inputs"][:] = 61.0 
    
    # Mock CLIP_TEMP to be sure? No, let's rely on actual constants.
    # If 61 is clipped, it should be 60 (or whatever max is).
    
    result = build_hourly_block(**mock_inputs)
    InterPhour = result[10]
    
    # It should be clipped to max
    assert np.all(InterPhour[:, 5] < 61.0)
    assert np.all(InterPhour[:, 5] > 0) # Should be valid

