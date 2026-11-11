"""Currently-block values: air quality interpolation and Canadian source priority."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from API.constants.forecast_const import DATA_CURRENT
from API.constants.model_const import GDPS, GFS, HRDPS
from API.constants.shared_const import MISSING_DATA
from API.current.metrics import (
    InterpolationState,
    _get_ozone,
    _get_uv,
    build_current_section,
)

AQ_FIELDS = ("pm25", "pm10", "o3", "no2", "so2", "co")


@pytest.mark.parametrize(
    ("aq_inputs", "expected_pm25", "expected_aqi", "expected_display_aqi"),
    [
        # Hourly CAQI is 68 at 00:00 (NO2 75 µg/m³) and 78 at 01:00 (NO2 96
        # µg/m³); the request time 00:30 sits halfway between them.
        pytest.param(
            {
                "pm25": np.array([10.0, 20.0, 30.0]),
                "pm10": np.array([20.0, 30.0, 40.0]),
                "o3": np.array([30.0, 40.0, 50.0]),
                "no2": np.array([40.0, 51.0, 60.0]),
                "so2": np.array([1.0, 2.0, 3.0]),
                "co": np.array([100.0, 200.0, 300.0]),
            },
            15.0,
            73.0,
            73,
            id="interpolated_between_hours",
        ),
        pytest.param(
            {field: np.full(3, np.nan) for field in AQ_FIELDS},
            np.nan,
            np.nan,
            np.nan,
            id="no_aq_data_reports_nan",
        ),
    ],
)
def test_build_current_section_interpolates_current_aq_inputs(
    aq_inputs, expected_pm25, expected_aqi, expected_display_aqi
):
    """Regression for #689: current AQ values must be interpolated to the request
    time from the surrounding hours (it used to take the next hour's value), and missing
    AQ data must surface as NaN rather than a clean-air 0."""
    result = build_current_section(
        sourceList=["hrrr_0-18", "hrrr_18-48"],
        hour_array_grib=np.array([0, 3600, 7200]),
        minute_array_grib=np.array([1800]),
        InterPminute=np.zeros((1, 10)),
        minuteItems=[
            {
                "precipIntensity": 0,
                "precipProbability": 0,
                "precipIntensityError": 0,
                "precipType": "none",
            }
        ],
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
        logger=MagicMock(),
        loc_tag="test_loc",
        aq_inputs=aq_inputs,
        inc_airqualitydetails=1,
    )

    np.testing.assert_equal(result.interp_current[DATA_CURRENT["pm25"]], expected_pm25)
    np.testing.assert_equal(result.currently["pm25"], expected_pm25)
    np.testing.assert_equal(result.interp_current[DATA_CURRENT["aqi"]], expected_aqi)
    np.testing.assert_equal(result.currently["airQualityIndex"], expected_display_aqi)


def test_current_canada_prioritizes_canadian_uv_and_ozone_over_gfs():
    """Regression for #734: in Canada the currently UV index and ozone must come
    from HRDPS/GDPS even when GFS is listed first in the source list."""
    hrdps = np.full((2, max(HRDPS.values()) + 1), np.nan)
    gdps = np.full((2, max(GDPS.values()) + 1), np.nan)
    gfs = np.full((2, max(GFS.values()) + 1), np.nan)
    hrdps[:, HRDPS["uv"]] = 7.0
    gdps[:, GDPS["uv"]] = 6.0
    gdps[:, GDPS["ozone"]] = 350.0
    gfs[:, GFS["uv"]] = 9.0 / (18.9 * 0.025)  # UV index 9 after GFS scaling
    gfs[:, GFS["ozone"]] = 300.0
    model_data = {"HRDPS_Merged": hrdps, "GDPS_Merged": gdps, "GFS_Merged": gfs}
    state = InterpolationState(idx1=0, idx2=1, fac1=0.5, fac2=0.5)
    sources = ["gfs", "hrdps", "gdps"]
    ottawa = (45.4215, -75.6972)

    assert _get_uv(sources, model_data, state, *ottawa) == 7.0
    assert _get_ozone(sources, model_data, state, *ottawa) == 350.0


@pytest.mark.parametrize("requested_hour", [3, 18])
def test_future_day_currently_precipitation_is_the_requested_hourly_item(
    requested_hour,
):
    """Regression for #689 with CFS: on a future day `currently` showed the first
    minute's precipitation, and the minutely block picks one source per element
    for its whole window. On the day GFS/GEFS end (here at 12Z) and CFS covers
    the rest, that disagreed with the hourly row, which falls back per hour
    (late hours came out -999 and "mixed" while hourly showed CFS's rain).
    `currently` must equal the hourly item of the hour holding the requested
    time, both while GFS still covers it and after only CFS does.

    The handler itself can't run offline, so this chains the real builders:
    inputs, minutely and hourly blocks, the overwrite the handler applies on a
    future day, then the current section.
    """
    from API.constants.forecast_const import DATA_MINUTELY
    from API.constants.model_const import CFS, GEFS
    from API.current.metrics import use_hourly_precip_for_first_minute
    from API.hourly.block import build_hourly_block
    from API.minutely.builder import build_minutely_block
    from tests.helpers.hourly import hourly_block_kwargs
    from tests.helpers.model_arrays import hourly_inputs

    num_hours = 25
    base = 1_800_000_000 - 1_800_000_000 % 86400
    hour_times = base + 3600.0 * np.arange(num_hours)
    lat, lon = -33.87, 151.21
    source_list = ["gfs", "gefs", "cfs"]
    intensity_unit = 0.0394  # in/h, so display and SI values differ

    # GFS and GEFS end at 12Z: wet, GEFS probability 0.8
    gfs = np.full((num_hours, max(GFS.values()) + 1), np.nan)
    gefs = np.full((num_hours, max(GEFS.values()) + 1), np.nan)
    gfs[:, 0] = gefs[:, 0] = hour_times
    gfs[:13, [GFS["temp"], GFS["dew"], GFS["humidity"]]] = [15.0, 10.0, 70.0]
    gfs[:13, [GFS["wind_u"], GFS["wind_v"], GFS["cloud"]]] = [3.0, 4.0, 90.0]
    gfs[:13, [GFS["intensity"], GFS["accum"], GFS["rain"]]] = [2.0, 2.0, 1.0]
    gfs[:13, [GFS["snow"], GFS["ice"], GFS["freezing_rain"]]] = 0
    gefs[:13, [GEFS["accum"], GEFS["prob"], GEFS["error"]]] = [2.0, 0.8, 0.3]
    gefs[:13, [GEFS["rain"], GEFS["snow"], GEFS["ice"]]] = [1.0, 0, 0]
    gefs[:13, GEFS["freezing_rain"]] = 0
    # CFS covers the whole day with lighter rain, past its 4-member range
    # (no probability or spread)
    cfs = np.full((num_hours, max(CFS.values()) + 1), np.nan)
    cfs[:, 0] = hour_times
    cfs[:, [CFS["temp"], CFS["dew"], CFS["humidity"]]] = [12.0, 9.0, 80.0]
    cfs[:, [CFS["wind_u"], CFS["wind_v"], CFS["cloud"]]] = [2.0, 1.0, 70.0]
    cfs[:, [CFS["intensity"], CFS["solar"]]] = [0.6, 0.0]

    inputs = hourly_inputs(
        lat=lat, lon=lon, source_list=source_list, gfs=gfs, gefs=gefs, cfs=cfs
    )
    minute_times = hour_times[requested_hour] + 1800 + 60 * np.arange(61.0)
    InterPminute, _, minuteItems, *_ = build_minutely_block(
        minute_array_grib=minute_times,
        source_list=source_list,
        hrrr_subh_data=None,
        hrrr_merged=None,
        nbm_data=None,
        dwd_mosmix_data=None,
        gefs_data=gefs,
        gfs_data=gfs,
        ecmwf_data=None,
        era5_data=None,
        prep_intensity_unit=intensity_unit,
        version=2,
        lat=lat,
        lon=lon,
    )
    hourly = build_hourly_block(
        **hourly_block_kwargs(
            num_hours,
            source_list=source_list,
            hour_array_grib=hour_times,
            prepIntensityUnit=intensity_unit,
            currentHourOffset=requested_hour,
            **inputs,
        )
    )
    hourList, InterPhour = hourly[0], hourly[-1]

    use_hourly_precip_for_first_minute(
        InterPminute, minuteItems, InterPhour, hourList[requested_hour], requested_hour
    )
    currently = build_current_section(
        sourceList=source_list,
        hour_array_grib=hour_times,
        minute_array_grib=minute_times,
        InterPminute=InterPminute,
        minuteItems=minuteItems,
        minuteRainIntensity=InterPminute[:, DATA_MINUTELY["rain_intensity"]],
        minuteSnowIntensity=InterPminute[:, DATA_MINUTELY["snow_intensity"]],
        minuteSleetIntensity=InterPminute[:, DATA_MINUTELY["ice_intensity"]],
        InterSday=np.zeros((1, 21)),
        dayZeroRain=0.0,
        dayZeroSnow=0.0,
        dayZeroIce=0.0,
        prepAccumUnit=1.0,
        prepIntensityUnit=intensity_unit,
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
        lat=lat,
        lon_IN=lon,
        tz_name="UTC",
        tz_offset=0.0,
        ETOPO=0.0,
        elevUnit=1.0,
        dataOut_rtma_ru=None,
        hrrrSubHInterpolation=None,
        HRRR_Merged=None,
        NBM_Merged=None,
        DWD_MOSMIX_Merged=None,
        ECMWF_Merged=None,
        GFS_Merged=gfs,
        ERA5_MERGED=None,
        CFS_Merged=cfs,
        logger=MagicMock(),
        loc_tag="test_loc",
    ).currently

    hour_item = hourList[requested_hour]
    for key in (
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
        "rainIntensity",
        "snowIntensity",
        "iceIntensity",
    ):
        np.testing.assert_equal(currently[key], hour_item[key], err_msg=key)
    assert hour_item["precipType"] == "rain"
    # The early hour is GFS's rain, the late one CFS's, probability missing
    expected = (2.0, 0.8) if requested_hour < 13 else (0.6, np.nan)
    np.testing.assert_allclose(
        [currently["precipIntensity"], currently["precipProbability"]],
        [round(expected[0] * intensity_unit, 4), expected[1]],
    )
    if requested_hour >= 13:
        # CFS has no visibility field: `currently` must carry the same estimate
        # as its hour rather than -999
        assert currently["visibility"] != MISSING_DATA
        np.testing.assert_allclose(currently["visibility"], hour_item["visibility"])
