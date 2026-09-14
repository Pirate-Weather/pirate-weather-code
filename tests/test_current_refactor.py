from unittest.mock import MagicMock

import numpy as np

from API.constants.aqi_const import compute_aqi_array
from API.constants.forecast_const import DATA_CURRENT
from API.constants.model_const import GDPS, GFS, HRDPS, NBM
from API.constants.shared_const import MISSING_DATA
from API.current.metrics import (
    CurrentSection,
    InterpolationState,
    _get_fire,
    _get_ozone,
    _get_temp,
    _get_uv,
    build_current_section,
)


def test_current_temperature_prefers_hrdps_over_nbm_in_canada():
    hrdps = np.full((2, max(HRDPS.values()) + 1), np.nan)
    nbm = np.full((2, max(NBM.values()) + 1), np.nan)
    hrdps[:, HRDPS["temp"]] = 5.0
    nbm[:, NBM["temp"]] = 10.0
    model_data = {
        "HRDPS_Merged": hrdps,
        "NBM_Merged": nbm,
    }

    value = _get_temp(
        ["hrdps", "nbm"],
        model_data,
        InterpolationState(idx1=0, idx2=1, fac1=0.5, fac2=0.5),
        49.2827,
        -123.1207,
    )

    assert value == 5.0


def test_current_canada_prioritizes_canadian_uv_and_ozone_over_gfs():
    hrdps = np.full((2, max(HRDPS.values()) + 1), np.nan)
    gdps = np.full((2, max(GDPS.values()) + 1), np.nan)
    gfs = np.full((2, max(GFS.values()) + 1), np.nan)
    hrdps[:, HRDPS["uv"]] = 7.0
    gdps[:, GDPS["uv"]] = 6.0
    gdps[:, GDPS["ozone"]] = 350.0
    gfs[:, GFS["uv"]] = 9.0 / (18.9 * 0.025)
    gfs[:, GFS["ozone"]] = 300.0
    model_data = {
        "HRDPS_Merged": hrdps,
        "GDPS_Merged": gdps,
        "GFS_Merged": gfs,
    }
    state = InterpolationState(idx1=0, idx2=1, fac1=0.5, fac2=0.5)

    uv = _get_uv(["gfs", "hrdps", "gdps"], model_data, state, 45.4215, -75.6972)
    ozone = _get_ozone(["gfs", "hrdps", "gdps"], model_data, state, 45.4215, -75.6972)

    assert uv == 7.0
    assert ozone == 350.0


def test_build_current_section_structure():
    # Mock inputs
    sourceList = ["hrrr_0-18", "hrrr_18-48"]
    hour_array_grib = np.array([100, 200, 300])
    minute_array_grib = np.array([150])
    InterPminute = np.zeros((1, 10))
    minuteItems = [
        {
            "precipIntensity": 0,
            "precipProbability": 0,
            "precipIntensityError": 0,
            "precipType": "none",
        }
    ]
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
        "HRRR_Merged": np.zeros((3, 20)),  # Mock merged data
        "NBM_Merged": None,
        "DWD_MOSMIX_Merged": None,
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


def test_build_current_section_interpolates_current_aq_inputs():
    sourceList = ["hrrr_0-18", "hrrr_18-48"]
    hour_array_grib = np.array([0, 3600, 7200])
    minute_array_grib = np.array([1800])
    InterPminute = np.zeros((1, 10))
    minuteItems = [
        {
            "precipIntensity": 0,
            "precipProbability": 0,
            "precipIntensityError": 0,
            "precipType": "none",
        }
    ]
    aq_inputs = {
        "pm25": np.array([10.0, 20.0, 30.0]),
        "pm10": np.array([20.0, 30.0, 40.0]),
        "o3": np.array([30.0, 40.0, 50.0]),
        "no2": np.array([40.0, 50.0, 60.0]),
        "so2": np.array([1.0, 2.0, 3.0]),
        "co": np.array([100.0, 200.0, 300.0]),
    }

    result = build_current_section(
        sourceList=sourceList,
        hour_array_grib=hour_array_grib,
        minute_array_grib=minute_array_grib,
        InterPminute=InterPminute,
        minuteItems=minuteItems,
        minuteRainIntensity=np.zeros(1),
        minuteSnowIntensity=np.zeros(1),
        minuteSleetIntensity=np.zeros(1),
        InterSday=np.zeros((1, 21)),
        dayZeroRain=0.0,
        dayZeroSnow=0.0,
        dayZeroIce=0.0,
        prepAccumUnit=1.0,
        prepIntensityUnit=1.0,
        windUnit=1.0,
        visUnits=1.0,
        tempUnits=1.0,
        humidUnit=1.0,
        extraVars=[],
        summaryText=False,
        translation=MagicMock(),
        icon="default",
        unitSystem="si",
        version=2,
        timeMachine=False,
        tmExtra=False,
        lat=0.0,
        lon_IN=0.0,
        tz_name="UTC",
        tz_offset=0.0,
        ETOPO=0.0,
        elevUnit=1.0,
        dataOut_rtma_ru=None,
        hrrrSubHInterpolation=None,
        HRRR_Merged=np.zeros((3, 20)),
        NBM_Merged=None,
        DWD_MOSMIX_Merged=None,
        ECMWF_Merged=None,
        GFS_Merged=None,
        ERA5_MERGED=None,
        NBM_Fire_Merged=None,
        logger=MagicMock(),
        loc_tag="test_loc",
        aq_inputs=aq_inputs,
        inc_airqualitydetails=1,
    )

    aqi_arr = compute_aqi_array(unit_system="si", **aq_inputs)
    expected_aqi = aqi_arr[0] * 0.5 + aqi_arr[1] * 0.5

    assert result.interp_current[DATA_CURRENT["pm25"]] == 15.0
    assert result.currently["pm25"] == 15.0
    assert result.interp_current[DATA_CURRENT["aqi"]] == expected_aqi
    assert result.currently["airQualityIndex"] == round(float(expected_aqi))


def test_get_fire_derived_from_si_inputs():
    value = _get_fire(30.0, 0.5, 5.0)
    assert np.isclose(value, 19.56673537, atol=1e-6)


def test_get_fire_returns_missing_when_input_missing():
    assert np.isnan(_get_fire(MISSING_DATA, 0.5, 5.0))
