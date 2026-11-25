"""Currently-block values: air quality interpolation and Canadian source priority."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from API.constants.aqi_const import AQ_LEAD_IN_HOURS
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
# Concentrations at 00:00, 01:00 and 02:00 of the test grid
_AQ_HOURS = {
    "pm25": np.array([10.0, 20.0, 30.0]),
    "pm10": np.array([20.0, 30.0, 40.0]),
    "o3": np.array([30.0, 40.0, 50.0]),
    "no2": np.array([40.0, 51.0, 60.0]),
    "so2": np.array([1.0, 2.0, 3.0]),
    "co": np.array([100.0, 200.0, 300.0]),
}


def _with_rdaqa(analysis_times, *, lead_in=False):
    """``_AQ_HOURS`` with RDAQA analysis times ({lead-in + grid row: unix s});
    ``lead_in`` adds a lead-in at 00:00's levels but PM2.5 12."""
    rdaqa_times = np.full(AQ_LEAD_IN_HOURS + 3, np.nan)
    for row, analysis_time in analysis_times.items():
        rdaqa_times[row] = analysis_time
    inputs = {**_AQ_HOURS, "rdaqa_times": rdaqa_times}
    if lead_in:
        inputs["lead_in"] = {
            field: np.full(AQ_LEAD_IN_HOURS, 12.0 if field == "pm25" else hours[0])
            for field, hours in _AQ_HOURS.items()
        }
    return inputs


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
        # RDAQA analysed 00:00 (30 min before the request) and 01:00 (after it):
        # currently reports 00:00 (PM2.5 10, CAQI 68), not the blend with 01:00.
        pytest.param(
            _with_rdaqa({AQ_LEAD_IN_HOURS: 0.0, AQ_LEAD_IN_HOURS + 1: 3600.0}),
            10.0,
            68.0,
            68,
            id="newest_rdaqa_hour_at_or_before_request",
        ),
        # Only 23:00 the day before, 1.5 h old: currently reports that lead-in
        # hour (PM2.5 12; NO2 still sets CAQI 68) instead of the forecast.
        pytest.param(
            _with_rdaqa({AQ_LEAD_IN_HOURS - 1: -3600.0}, lead_in=True),
            12.0,
            68.0,
            68,
            id="rdaqa_from_before_midnight",
        ),
        # The analysis behind 00:00 is over 3 h old by its own time, though the
        # grid hour it was matched to is not: currently interpolates.
        pytest.param(
            _with_rdaqa({AQ_LEAD_IN_HOURS: 1800.0 - 3 * 3600 - 60}),
            15.0,
            73.0,
            73,
            id="rdaqa_too_old_by_analysis_time",
        ),
    ],
)
def test_build_current_section_interpolates_current_aq_inputs(
    aq_inputs, expected_pm25, expected_aqi, expected_display_aqi
):
    """Regression for #689: current AQ values must be interpolated to the request
    time from the surrounding hours (it used to take the next hour's value), and missing
    AQ data must surface as NaN rather than a clean-air 0. Where RDAQA has analysed a
    recent hour, currently must be that measured hour: blending in the next hour's
    RAQDPS forecast would undo it (RAQDPS ran 6-13 ug/m3 low in Ottawa/Kingston).
    The analysis must be at or before the requested time, may be the previous
    evening's just after midnight, and must be judged by its own time, not the
    grid hour it was matched to (30 min apart at fractional UTC offsets)."""
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
        # Nor gust or UV fields: gusts are estimated from the wind rather than
        # copying it, and UV from the sun's height and cloud cover
        np.testing.assert_allclose(currently["windGust"], hour_item["windGust"])
        assert hour_item["windGust"] > hour_item["windSpeed"]
        assert np.isfinite(currently["uvIndex"])


def _currently_with_lightning(librewxr, version, lightning_age_s=300.0):
    """The currently block of a wet GFS/GEFS point (GFS storm 50 km east) with
    LibreWXR rows."""
    from API.constants.forecast_const import DATA_MINUTELY
    from API.constants.model_const import GEFS
    from API.minutely.builder import build_minutely_block

    num_hours = 3
    hour_times = 1_800_000_000.0 + 3600.0 * np.arange(num_hours)
    minute_times = hour_times[0] + 60 * np.arange(61.0)
    source_list = ["gfs", "gefs", "librewxr"]
    gfs = np.full((num_hours, max(GFS.values()) + 1), np.nan)
    gefs = np.full((num_hours, max(GEFS.values()) + 1), np.nan)
    gfs[:, 0] = gefs[:, 0] = hour_times
    gefs[:, [GEFS["accum"], GEFS["prob"], GEFS["error"]]] = [3.0, 0.9, 0.3]
    gefs[:, [GEFS["rain"], GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"]]] = [
        1.0,
        0,
        0,
        0,
    ]
    gfs[:, [GFS["temp"], GFS["dew"], GFS["humidity"]]] = [25.0, 20.0, 70.0]
    gfs[:, [GFS["wind_u"], GFS["wind_v"], GFS["cloud"]]] = [3.0, 4.0, 90.0]
    gfs[:, [GFS["intensity"], GFS["accum"], GFS["rain"]]] = [3.0, 3.0, 1.0]
    gfs[:, [GFS["snow"], GFS["ice"], GFS["freezing_rain"]]] = 0
    gfs[:, [GFS["storm_dist"], GFS["storm_dir"]]] = [50_000.0, 90.0]
    if librewxr is not None:
        librewxr[:, 0] += minute_times[0]
    InterPminute, _, minuteItems, *_ = build_minutely_block(
        minute_array_grib=minute_times,
        source_list=source_list,
        hrrr_subh_data=None,
        librewxr_data=librewxr,
        hrrr_merged=None,
        nbm_data=None,
        dwd_mosmix_data=None,
        gefs_data=gefs,
        gfs_data=gfs,
        ecmwf_data=None,
        era5_data=None,
        prep_intensity_unit=1.0,
        version=version,
        lat=25.8,
        lon=-80.2,
    )
    translation = MagicMock()
    translation.translate.side_effect = lambda parts: parts[1]
    return build_current_section(
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
        prepIntensityUnit=1.0,
        windUnit=1.0,
        visUnits=0.001,  # km
        tempUnits=1.0,
        humidUnit=1.0,
        extraVars=[],
        summaryText=True,
        translation=translation,
        icon="darksky",
        unitSystem="si",
        version=version,
        timeMachine=False,
        tmExtra=False,
        lat=25.8,
        lon_IN=-80.2,
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
        librewxr_data=librewxr,
        librewxr_lightning_time=minute_times[0] - lightning_age_s,
        logger=MagicMock(),
        loc_tag="test_loc",
    ).currently


def _librewxr_lightning(distance_m, bearing, count=0.0):
    """LibreWXR rows (T+0 and one nowcast step) outside radar coverage, with the
    nearest strike at T+0; times are offsets from the first minute."""
    from API.constants.model_const import LIBREWXR

    rows = np.full((2, max(LIBREWXR.values()) + 1), np.nan)
    rows[:, 0] = [-600.0, 0.0]
    rows[
        0,
        [
            LIBREWXR["lightning_dist"],
            LIBREWXR["lightning_dir"],
            LIBREWXR["lightning_count"],
        ],
    ] = [distance_m, bearing, count]
    return rows


def test_satellite_lightning_sets_nearest_storm_and_thunderstorm_summary():
    """Satellite lightning replaces the GFS nearest storm in `currently` for every
    version, and in version 2 a strike within 15 km turns current rain into a
    thunderstorm even when the models have no instability data (here, none) and
    reports the strike count within 15 km as `lightningCount`. It
    must not leak where it is not valid: no strike within the 300 km search only
    floors the GFS distance, and lightning fields computed over 20 minutes ago
    (a stalled export) fall back to GFS entirely, so a dead feed never reads as
    "storm overhead" or "no storm".
    """
    from API.api_utils import remove_conditional_fields

    # Three strikes, the nearest 8 km south: lightning wins, summary is a
    # thunderstorm in v2 only, and lightningCount is a v2 field
    strikes = _librewxr_lightning(8_000.0, 180.0, count=3.0)
    v2 = _currently_with_lightning(strikes, version=2)
    assert v2["nearestStormDistance"] == pytest.approx(8.0)
    assert v2["nearestStormBearing"] == 180
    assert v2["lightningCount"] == 3
    assert v2["icon"] == "thunderstorm"
    assert v2["summary"] == "thunderstorm"
    v1 = _currently_with_lightning(_librewxr_lightning(8_000.0, 180.0), version=1)
    assert v1["nearestStormDistance"] == pytest.approx(8.0)
    assert v1["icon"] != "thunderstorm"
    assert "lightningCount" not in remove_conditional_fields(
        dict(v1), version=1, time_machine=False, tm_extra=False
    )
    # Very light rain has summary text but no precipitation icon; the observed
    # lightning must still set the icon, not leave a cloud icon under a
    # thunderstorm summary
    from API.PirateText import calculate_text

    drizzle = dict(v2, precipIntensity=0.02, rainIntensity=0.02)
    assert calculate_text(
        drizzle, True, "current", "darksky", lightning_nearby=True
    ) == (
        "thunderstorm",
        "thunderstorm",
    )

    # Covered, nothing within 300 km: GFS's 50 km storm is ruled out
    none_near = _currently_with_lightning(
        _librewxr_lightning(np.inf, np.nan), version=2
    )
    assert none_near["nearestStormDistance"] == pytest.approx(300.0)
    assert none_near["nearestStormBearing"] == 90
    assert none_near["lightningCount"] == 0
    assert none_near["icon"] != "thunderstorm"

    # Lightning computed 40 minutes ago, or no LibreWXR data: GFS as before
    for librewxr, age in ((_librewxr_lightning(8_000.0, 180.0), 2400.0), (None, 0)):
        fallback = _currently_with_lightning(librewxr, version=2, lightning_age_s=age)
        assert fallback["nearestStormDistance"] == pytest.approx(50.0)
        assert fallback["nearestStormBearing"] == 90
        assert np.isnan(fallback["lightningCount"])
        assert fallback["icon"] != "thunderstorm"
