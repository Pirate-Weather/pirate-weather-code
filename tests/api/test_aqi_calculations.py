"""AQI index values per unit system, the averaging applied before lookup, and the
RAQDPS/SILAM source selection that feeds them."""

import math
import struct

import numpy as np
import pytest

from API.constants import aqi_const
from API.constants.aqi_const import (
    AQ_LEAD_IN_HOURS,
    compute_aqi_array,
    compute_aqi_for_unit_system,
    compute_aqi_with_lead_in,
    compute_daqi,
)
from API.constants.model_const import RAQDPS, SILAM
from API.data_inputs import prepare_aq_inputs

# ---------------------------------------------------------------------------
# Index values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit_system", "pollutants", "expected"),
    [
        # US EPA AQI, PM2.5 table as revised Feb 2024 (#686): 9.0 µg/m³ is the top
        # of "Good" (AQI 50). The pre-2024 table put 50 at 12.0 µg/m³ (-> 38 here).
        # O3 30 ppb (-> 28) and NO2 20 ppb (-> 19) stay below it.
        pytest.param(
            "us",
            {"pm25_ug": 9.0, "o3_ppb": 30.0, "no2_ppb": 20.0},
            50.0,
            id="us-epa-pm25-good-ceiling",
        ),
        # Health Canada AQHI formula: (1000/10.4) * sum(exp(beta*C) - 1) with
        # beta O3 0.000537, NO2 0.000871, PM2.5 0.000487 -> 2.872, reported to 0.1.
        pytest.param(
            "ca",
            {"pm25_ug": 10.0, "o3_ppb": 30.0, "no2_ppb": 10.0},
            2.9,
            id="ca-aqhi-formula",
        ),
        # EEA European AQI: PM2.5 15 µg/m³ is the fair/moderate boundary (40 on
        # the 0-100 CAQI scale); PM10 15 µg/m³ is good/fair (20).
        pytest.param(
            "si",
            {"pm25_ug": 15.0, "pm10_ug": 15.0, "o3_ppb": 20.0, "no2_ppb": 5.0},
            40.0,
            id="si-caqi-pm25-moderate-boundary",
        ),
        # UK DEFRA DAQI: PM2.5 36 µg/m³ is the first value of band 4; NO2 40 ppb
        # (75 µg/m³) is band 2 and PM10 17 µg/m³ is band 2.
        pytest.param(
            "uk",
            {"pm25_ug": 36.0, "pm10_ug": 17.0, "o3_ppb": 20.0, "no2_ppb": 40.0},
            4.0,
            id="uk-daqi-pm25-band4",
        ),
        # Hong Kong EPD AQHI: added risk O3 3.11 + NO2 4.28 + SO2 0.18 +
        # max(PM2.5 0.55, PM10 1.42) = 9.00 %, inside the 7.53-9.41 band -> 5.
        pytest.param(
            "hk",
            {
                "pm25_ug": 25.0,
                "pm10_ug": 50.0,
                "o3_ppb": 30.0,
                "no2_ppb": 50.0,
                "so2_ppb": 5.0,
            },
            5,
            id="hk-aqhi-added-risk-band5",
        ),
        # Ireland EPA AQIH: same as DAQI except SO2. 25 ppb SO2 = 65.5 µg/m³ is
        # band 3 on the AQIH table (60-89) but band 1 on DAQI's.
        pytest.param(
            "ie",
            {"pm25_ug": 5.0, "so2_ppb": 25.0},
            3.0,
            id="ie-aqih-so2-band3",
        ),
        # Israel AQI (inverted, 100 = clean): PM2.5 37 µg/m³ is the 100 sub-index
        # breakpoint, so the reported value is 100 - 100 = 0.
        pytest.param(
            "il",
            {"pm25_ug": 37.0, "pm10_ug": 65.0},
            0.0,
            id="il-iaqi-inverted-scale",
        ),
        # Indonesia ISPU (KLHK P.14/2020): PM2.5 55.4 µg/m³ -> 100.
        pytest.param(
            "id",
            {"pm25_ug": 55.4, "pm10_ug": 50.0},
            100.0,
            id="id-ispu-pm25-breakpoint",
        ),
        # China HJ 633-2012: PM2.5 115 µg/m³ -> IAQI 150; PM10 50 µg/m³ -> 50.
        pytest.param(
            "cn",
            {"pm25_ug": 115.0, "pm10_ug": 50.0},
            150.0,
            id="cn-aqi-pm25-breakpoint",
        ),
        # Malaysia API, breakpoints from aqihub.info (the reference aqi_const
        # cites): PM2.5 153 µg/m³ -> 100, PM10 190 µg/m³ -> 50.
        pytest.param(
            "my",
            {"pm25_ug": 153.0, "pm10_ug": 190.0},
            100.0,
            id="my-api-pm25-breakpoint",
        ),
        # Vietnam VN_AQI (VEA Decision 1459/QD-TCMT): PM2.5 25 µg/m³ and PM10
        # 50 µg/m³ are both the top of "Good" (50).
        pytest.param(
            "vn",
            {"pm25_ug": 25.0, "pm10_ug": 50.0},
            50.0,
            id="vn-aqi-good-ceiling",
        ),
    ],
)
def test_aqi_value_for_each_unit_system(unit_system, pollutants, expected):
    """Pin one official-table value per AQI system so a changed breakpoint,
    conversion factor, or unit-system dispatch shows up as a changed number
    instead of being hidden by a loose range check."""
    assert compute_aqi_for_unit_system(unit_system, **pollutants) == expected


def test_aqi_is_nan_without_any_pollutant_data():
    """Missing AQ data must surface as NaN, never as a clean-air index of 0."""
    assert math.isnan(compute_aqi_for_unit_system("us"))


@pytest.mark.parametrize("unit_system", ["us", "si"])
def test_hourly_aqi_array_matches_per_hour_index(unit_system):
    """The us (EPA) and si (CAQI) hourly AQI are computed for all hours at once
    for speed; every hour must equal the per-hour reference calculation, across
    breakpoint edges, gaps between tables, the 1-hour O3 and 24-hour SO2
    special cases, negative values and missing data."""
    rng = np.random.default_rng(0)
    n_hours = 2000
    pollutants = {
        "pm25_ug": rng.uniform(-5, 400, n_hours),
        "pm10_ug": rng.uniform(-5, 700, n_hours),
        "o3_ppb": rng.uniform(-5, 700, n_hours),
        "no2_ppb": rng.uniform(-5, 2500, n_hours),
        "so2_ppb": rng.uniform(-5, 1200, n_hours),
        "co_ppb": rng.uniform(-5, 60000, n_hours),
    }
    for values in pollutants.values():
        values[rng.random(n_hours) < 0.15] = np.nan
        values[:7] = [0.0, 9.0, 54.0, 70.0, 124.0, 304.0, 305.0]

    if unit_system == "us":
        vectorised = aqi_const.compute_epa_aqi_array(
            pollutants["pm25_ug"],
            pollutants["pm10_ug"],
            pollutants["o3_ppb"],
            pollutants["o3_ppb"],
            pollutants["no2_ppb"],
            pollutants["so2_ppb"],
            pollutants["so2_ppb"],
            pollutants["co_ppb"],
        )
    else:
        vectorised = aqi_const.compute_caqi_array(
            pollutants["pm25_ug"],
            pollutants["pm10_ug"],
            pollutants["o3_ppb"],
            pollutants["no2_ppb"],
            pollutants["so2_ppb"],
        )
    per_hour = [
        compute_aqi_for_unit_system(
            unit_system, **{name: float(v[i]) for name, v in pollutants.items()}
        )
        for i in range(n_hours)
    ]

    np.testing.assert_array_equal(vectorised, per_hour)


# ---------------------------------------------------------------------------
# Averaging applied before breakpoint lookup
# ---------------------------------------------------------------------------


def _aqi_series(unit_system, **pollutants):
    arrays = dict.fromkeys(("pm25", "pm10", "o3", "no2", "so2", "co"))
    arrays.update(pollutants)
    return compute_aqi_array(unit_system, **arrays)


def test_epa_uses_nowcast_not_instantaneous():
    """EPA PM AQI is defined on the NowCast average; using the raw hour would
    make a one-hour spike jump straight to "Very Unhealthy"."""
    conc = np.concatenate([np.full(12, 5.0), [200.0]])

    result = _aqi_series("us", pm25=conc)

    # NowCast weight is 0.5 (min/max floor), giving 102.5 µg/m³ -> AQI 184.
    # The raw 200 µg/m³ hour alone would be AQI 275.
    assert result[-1] == 184
    assert compute_aqi_for_unit_system("us", pm25_ug=200.0) == 275


def test_daqi_pm25_uses_24h_rolling_mean():
    """DAQI PM2.5 bands are defined on a 24 h mean; a single spike hour must not
    push the index to "Very High"."""
    conc = np.concatenate([np.full(23, 5.0), [80.0]])

    result = _aqi_series("uk", pm25=conc)

    # 24 h mean = (23 * 5 + 80) / 24 = 8.1 µg/m³ -> band 1; raw 80 µg/m³ -> band 10.
    assert result[-1] == 1
    assert compute_daqi(pm25_ug=80.0) == 10


def test_daqi_o3_uses_8h_rolling_mean():
    """DAQI O3 bands are defined on an 8 h mean, not the instantaneous value."""
    conc = np.concatenate([np.full(7, 6.0), [105.0]])

    result = _aqi_series("uk", o3=conc)

    # 8 h mean = (7 * 6 + 105) / 8 = 18.4 ppb = 36.7 µg/m³ -> band 2.
    # The raw 105 ppb (210 µg/m³) hour alone would be band 8.
    assert result[-1] == 2


# ---------------------------------------------------------------------------
# prepare_aq_inputs source selection
# ---------------------------------------------------------------------------

N_HOURS = 24
HOURS = np.arange(N_HOURS, dtype=float) * 3600


def _zarr_rows(var_idx, times, **columns):
    """Build a fake zarr read result: one row per time, column 0 = unix time."""
    arr = np.full((len(times), max(var_idx.values()) + 1), np.nan)
    arr[:, 0] = times
    for name, value in columns.items():
        arr[:, var_idx[name]] = value
    return arr


def test_a_nan_time_does_not_lose_the_air_quality_model():
    """A row with no data gets a NaN time. np.argmin stops at the first NaN, so
    one short or padded run used to unmatch every hour and silently drop the whole
    model, losing the AQI for that location.
    """
    silam = _zarr_rows(SILAM, HOURS, pm25=12.0)
    silam[5, 0] = np.nan

    result = prepare_aq_inputs(N_HOURS, None, silam, HOURS)

    assert np.all(result["pm25"][:5] == 12.0)
    assert np.all(result["pm25"][6:] == 12.0)


def test_raqdps_wins_when_both_have_data():
    """RAQDPS is the higher-resolution North American model and must take
    priority over global SILAM hour by hour, with SILAM filling hours RAQDPS
    does not cover."""
    raqdps_pm25 = np.full(N_HOURS, 30.0)
    raqdps_pm25[-6:] = np.nan  # past the end of the RAQDPS forecast
    raqdps = _zarr_rows(
        RAQDPS, HOURS, pm25=raqdps_pm25, pm10=40.0, no2=10.0, o3=20.0, so2=5.0
    )
    silam = _zarr_rows(
        SILAM, HOURS, pm25=99.0, pm10=99.0, no2=99.0, o3=99.0, so2=99.0, co=99.0
    )

    result = prepare_aq_inputs(N_HOURS, raqdps, silam, HOURS)

    expected_pm25 = np.where(np.isnan(raqdps_pm25), 99.0, 30.0)
    np.testing.assert_array_equal(result["pm25"], expected_pm25)
    for key, value in {"pm10": 40.0, "no2": 10.0, "o3": 20.0, "so2": 5.0}.items():
        np.testing.assert_array_equal(result[key], np.full(N_HOURS, value))


@pytest.mark.parametrize(
    ("use_raqdps", "use_silam", "expected"),
    [
        pytest.param(
            False,
            True,
            {"pm25": 15.0, "co": 200.0},
            id="raqdps_missing_falls_back_to_silam",
        ),
        # RAQDPS has no CO field, so CO always comes from SILAM.
        pytest.param(True, True, {"pm25": 5.0, "co": 200.0}, id="co_from_silam"),
        pytest.param(
            False,
            False,
            dict.fromkeys(
                ("pm25", "pm10", "o3", "no2", "so2", "co", "smoke_frp"), np.nan
            ),
            id="no_aq_sources_gives_nan",
        ),
    ],
)
def test_aq_source_selection(use_raqdps, use_silam, expected):
    """Each pollutant must come from the best available source, and with no AQ
    source at all every field (including smoke) must be NaN rather than 0."""
    raqdps = (
        _zarr_rows(RAQDPS, HOURS, pm25=5.0, pm10=10.0, no2=3.0, o3=8.0, so2=1.0)
        if use_raqdps
        else False
    )
    silam = (
        _zarr_rows(
            SILAM, HOURS, pm25=15.0, pm10=25.0, no2=8.0, o3=35.0, so2=3.0, co=200.0
        )
        if use_silam
        else False
    )

    result = prepare_aq_inputs(N_HOURS, raqdps, silam, HOURS)

    for key, value in expected.items():
        np.testing.assert_array_equal(result[key], np.full(N_HOURS, value), err_msg=key)


@pytest.mark.parametrize(
    ("blh", "frp_column", "expected_smoke"),
    [
        # 10 m * 3 is below the 2000 m floor: 1000 / 2000 * weight 0.2 = 0.1.
        pytest.param(10.0, 1000.0, 0.1, id="shallow_blh_clamped_to_2000m"),
        # No BLH: neutral 3000 m depth: 5000 / 3000 * 0.2 = 1/3.
        pytest.param(
            np.nan, 5000.0, 5000.0 / 3000.0 * 0.2, id="missing_blh_uses_3000m"
        ),
    ],
)
def test_smoke_frp_enforces_minimum_blh(blh, frp_column, expected_smoke):
    """SILAM fire PM is a column burden; dividing by a very shallow (or missing)
    boundary layer would produce absurd surface smoke, so the plume depth is
    floored at 2000 m and defaults to 3000 m. PM2.5 of 5 µg/m³ gives a 0.2 weight.
    """
    silam = _zarr_rows(
        SILAM, HOURS, pm25=5.0, pm_frp_column=frp_column, blh=blh, co=100.0
    )

    result = prepare_aq_inputs(N_HOURS, False, silam, HOURS)

    np.testing.assert_allclose(result["smoke_frp"], expected_smoke, rtol=1e-12)


def test_fractional_timezone_30min_offset_matches():
    """Regression for #684: SILAM times are stored as float32 (about ±64 s at
    1.75e9), and 30-minute UTC offsets (India, Newfoundland, Adelaide) put request
    times exactly 1800 s from a SILAM hour, so a < 1800 s match tolerance dropped
    roughly half the hours to NaN."""
    n = 48
    base = 1752004800.0  # 2025-07-08 20:00 UTC
    request_hours = base + np.arange(n) * 3600 + 1800
    silam_hours_f32 = np.array(
        [struct.unpack("f", struct.pack("f", base + i * 3600))[0] for i in range(n)]
    )
    silam = _zarr_rows(SILAM, silam_hours_f32, pm25=10.0, co=150.0)

    result = prepare_aq_inputs(n, False, silam, request_hours)

    np.testing.assert_array_equal(result["pm25"], np.full(n, 10.0))


def _rows_from(model, start, hours, **columns):
    """Hourly store rows from ``start``, like one model's read at a point."""
    return _zarr_rows(model, start + np.arange(hours) * 3600.0, **columns)


def test_aqi_averages_reach_back_before_local_midnight():
    """The AQ series starts at local midnight, and the averages stop at the start of
    the array, so a smoky evening was forgotten overnight: just after midnight the
    NowCast only saw one clean hour. The earlier hours must count, without changing
    the concentrations reported for the grid or the length of the result.
    """
    midnight = 1752019200.0
    silam_hours = 72
    silam = _rows_from(SILAM, midnight - 24 * 3600.0, silam_hours, pm25=5.0, co=100.0)
    # A smoky day before midnight, in the 48 h SILAM keeps
    silam[:24, SILAM["pm25"]] = 150.0
    grid = midnight + np.arange(N_HOURS) * 3600.0

    inputs = prepare_aq_inputs(N_HOURS, False, silam, grid)
    series = {name: inputs[name] for name in ("pm25", "pm10", "o3", "no2", "so2", "co")}

    assert inputs["pm25"][0] == 5.0
    assert len(inputs["lead_in"]["pm25"]) == AQ_LEAD_IN_HOURS

    with_lead_in = compute_aqi_with_lead_in("us", inputs["lead_in"], **series)
    without = compute_aqi_with_lead_in("us", None, **series)

    assert len(with_lead_in) == N_HOURS
    assert with_lead_in[0] > 2 * without[0]
    # Once the window has passed the smoke, the history makes no difference
    np.testing.assert_allclose(with_lead_in[23:], without[23:])


def test_aqi_lead_in_prefers_raqdps_then_falls_back_to_silam():
    """RAQDPS keeps no history, so a window crossing its run time must use SILAM
    before it, per hour, rather than losing those hours. With no earlier data at all
    the result matches the old behaviour of averaging what there is.
    """
    midnight = 1752019200.0
    grid = midnight + np.arange(N_HOURS) * 3600.0
    raqdps = _rows_from(RAQDPS, midnight - 3 * 3600.0, N_HOURS, pm25=40.0)
    silam = _rows_from(SILAM, midnight - 24 * 3600.0, N_HOURS + 24, pm25=8.0)

    lead_in = prepare_aq_inputs(N_HOURS, raqdps, silam, grid)["lead_in"]
    np.testing.assert_array_equal(lead_in["pm25"][-3:], [40.0, 40.0, 40.0])
    np.testing.assert_array_equal(lead_in["pm25"][:21], np.full(21, 8.0))

    no_history = _rows_from(SILAM, midnight, N_HOURS, pm25=8.0)
    inputs = prepare_aq_inputs(N_HOURS, False, no_history, grid)
    assert np.isnan(inputs["lead_in"]["pm25"]).all()
    series = {name: inputs[name] for name in ("pm25", "pm10", "o3", "no2", "so2", "co")}
    np.testing.assert_allclose(
        compute_aqi_with_lead_in("us", inputs["lead_in"], **series),
        compute_aqi_with_lead_in("us", None, **series),
    )
