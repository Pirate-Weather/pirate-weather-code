"""Currently-block values: air quality interpolation and Canadian source priority."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from API.constants.forecast_const import DATA_CURRENT
from API.constants.model_const import GDPS, GFS, HRDPS
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
