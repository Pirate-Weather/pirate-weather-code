"""Tests for AQI calculation functions (EPA / AQHI / CAQI) and source priority."""

import math

import numpy as np
import pytest

from API.constants.aqi_const import (
    AQI_SYSTEM_MAP,
    compute_aqhi,
    compute_aqi_array,
    compute_aqi_for_unit_system,
    compute_caqi,
    compute_epa_aqi,
)
from API.data_inputs import prepare_aq_inputs

# ---------------------------------------------------------------------------
# EPA AQI tests
# ---------------------------------------------------------------------------


class TestEPAAQI:
    def test_good_air_quality(self):
        """Low PM2.5 → AQI in 'Good' range (0–50)."""
        aqi = compute_epa_aqi(pm25_ug=5.0)
        assert 0 <= aqi <= 50

    def test_moderate_pm25(self):
        """PM2.5 around 12.5 µg/m³ sits near AQI 50 boundary."""
        aqi = compute_epa_aqi(pm25_ug=12.0)
        assert 0 <= aqi <= 50

    def test_unhealthy_pm25(self):
        """PM2.5 of 150 µg/m³ → AQI ≥ 200 (Very Unhealthy)."""
        aqi = compute_epa_aqi(pm25_ug=150.0)
        assert aqi >= 200

    def test_nan_inputs_return_nan(self):
        """All-NaN inputs should return NaN."""
        aqi = compute_epa_aqi()
        assert math.isnan(aqi)

    def test_pm10_drives_aqi(self):
        """High PM10 alone should produce a positive AQI."""
        aqi = compute_epa_aqi(pm10_ug=200.0)
        assert aqi > 0

    def test_multiple_pollutants_max(self):
        """AQI should reflect the dominant pollutant (highest sub-index)."""
        aqi_pm25_only = compute_epa_aqi(pm25_ug=100.0)
        aqi_both = compute_epa_aqi(pm25_ug=100.0, pm10_ug=300.0)
        # With high PM10, result should be >= PM2.5 only
        assert aqi_both >= aqi_pm25_only


# ---------------------------------------------------------------------------
# AQHI tests
# ---------------------------------------------------------------------------


class TestAQHI:
    def test_clean_air_returns_low_aqhi(self):
        """Very low concentrations → AQHI close to 1."""
        aqhi = compute_aqhi(pm25_ug=2.0, o3_ppb=10.0, no2_ppb=5.0)
        assert aqhi >= 1

    def test_high_concentrations_return_high_aqhi(self):
        """Elevated O3/NO2 should push AQHI above 7."""
        aqhi = compute_aqhi(pm25_ug=50.0, o3_ppb=200.0, no2_ppb=100.0)
        assert aqhi >= 7

    def test_nan_inputs_return_nan(self):
        """All NaN → NaN."""
        aqhi = compute_aqhi()
        assert math.isnan(aqhi)


# ---------------------------------------------------------------------------
# CAQI tests
# ---------------------------------------------------------------------------


class TestCAQI:
    def test_low_pollutants_return_low_caqi(self):
        caqi = compute_caqi(pm25_ug=5.0, pm10_ug=10.0, o3_ppb=20.0, no2_ppb=10.0)
        assert 0 <= caqi <= 25

    def test_high_pollutants_return_high_caqi(self):
        caqi = compute_caqi(pm25_ug=80.0, pm10_ug=100.0, o3_ppb=300.0, no2_ppb=200.0)
        assert caqi >= 75

    def test_nan_inputs_return_nan(self):
        caqi = compute_caqi()
        assert math.isnan(caqi)


# ---------------------------------------------------------------------------
# Unit-system dispatch tests
# ---------------------------------------------------------------------------


class TestAQISystemDispatch:
    @pytest.mark.parametrize(
        "unit_system,expected_system",
        [
            ("us", "EPA"),
            ("ca", "AQHI"),
            ("uk", "CAQI"),
            ("si", "CAQI"),
            ("unknown", "EPA"),  # default
        ],
    )
    def test_system_map(self, unit_system, expected_system):
        assert AQI_SYSTEM_MAP.get(unit_system, "EPA") == expected_system

    def test_us_returns_epa_value(self):
        epa = compute_epa_aqi(pm25_ug=35.0)
        dispatched = compute_aqi_for_unit_system("us", pm25_ug=35.0)
        assert abs(dispatched - epa) < 1e-6

    def test_ca_returns_aqhi_value(self):
        aqhi = compute_aqhi(pm25_ug=20.0, o3_ppb=50.0, no2_ppb=30.0)
        dispatched = compute_aqi_for_unit_system(
            "ca", pm25_ug=20.0, o3_ppb=50.0, no2_ppb=30.0
        )
        assert abs(dispatched - aqhi) < 1e-6

    def test_si_returns_caqi_value(self):
        caqi = compute_caqi(pm25_ug=20.0, pm10_ug=40.0, o3_ppb=60.0, no2_ppb=30.0)
        dispatched = compute_aqi_for_unit_system(
            "si", pm25_ug=20.0, pm10_ug=40.0, o3_ppb=60.0, no2_ppb=30.0
        )
        assert abs(dispatched - caqi) < 1e-6


# ---------------------------------------------------------------------------
# Vectorised AQI array
# ---------------------------------------------------------------------------


class TestAQIArray:
    def test_returns_correct_length(self):
        n = 10
        pm25 = np.full(n, 15.0)
        result = compute_aqi_array(
            "us", pm25=pm25, pm10=None, o3=None, no2=None, so2=None, co=None
        )
        assert len(result) == n

    def test_nan_propagates(self):
        pm25 = np.array([np.nan, 10.0, 20.0])
        result = compute_aqi_array(
            "us", pm25=pm25, pm10=None, o3=None, no2=None, so2=None, co=None
        )
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_all_none_inputs_returns_empty(self):
        result = compute_aqi_array(
            "us", pm25=None, pm10=None, o3=None, no2=None, so2=None, co=None
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# prepare_aq_inputs source-priority tests
# ---------------------------------------------------------------------------


def _make_zarr_data(n_rows, var_idx_map, time_values, data_values):
    """Build a minimal fake zarr read result (rows × n_vars)."""
    n_vars = max(var_idx_map.values()) + 1
    arr = np.full((n_rows, n_vars), np.nan)
    for i, t in enumerate(time_values):
        arr[i, 0] = t
        for k, v in data_values.items():
            arr[i, var_idx_map[k]] = v[i]
    return arr


class TestPrepareAQInputs:
    """Tests for prepare_aq_inputs RAQDPS-over-SILAM priority."""

    def _hour_array(self, n=24, start=0):
        return np.array([start + i * 3600 for i in range(n)], dtype=float)

    def test_raqdps_wins_when_both_have_data(self):
        """RAQDPS PM2.5 should take priority over SILAM PM2.5."""
        from API.constants.model_const import RAQDPS, SILAM

        n = 24
        hours = self._hour_array(n)

        raqdps_data = _make_zarr_data(
            n,
            RAQDPS,
            hours,
            {
                "pm25": np.full(n, 30.0),
                "pm10": np.full(n, 40.0),
                "no2": np.full(n, 10.0),
                "o3": np.full(n, 20.0),
                "so2": np.full(n, 5.0),
            },
        )
        silam_data = _make_zarr_data(
            n,
            SILAM,
            hours,
            {
                "pm25": np.full(n, 99.0),
                "pm10": np.full(n, 99.0),
                "no2": np.full(n, 99.0),
                "o3": np.full(n, 99.0),
                "so2": np.full(n, 99.0),
                "co": np.full(n, 99.0),
            },
        )

        result = prepare_aq_inputs(n, raqdps_data, silam_data, hours)
        # RAQDPS PM2.5 (30) should win over SILAM (99)
        assert np.nanmean(result["pm25"]) == pytest.approx(30.0, abs=1.0)

    def test_silam_fallback_when_raqdps_missing(self):
        """When RAQDPS is unavailable, SILAM data should be used."""
        from API.constants.model_const import SILAM

        n = 24
        hours = self._hour_array(n)

        silam_data = _make_zarr_data(
            n,
            SILAM,
            hours,
            {
                "pm25": np.full(n, 15.0),
                "pm10": np.full(n, 25.0),
                "no2": np.full(n, 8.0),
                "o3": np.full(n, 35.0),
                "so2": np.full(n, 3.0),
                "co": np.full(n, 200.0),
            },
        )

        result = prepare_aq_inputs(n, False, silam_data, hours)
        assert np.nanmean(result["pm25"]) == pytest.approx(15.0, abs=1.0)
        assert np.nanmean(result["co"]) == pytest.approx(200.0, abs=5.0)

    def test_co_always_from_silam(self):
        """CO is only available in SILAM, not RAQDPS."""
        from API.constants.model_const import RAQDPS, SILAM

        n = 24
        hours = self._hour_array(n)
        raqdps_data = _make_zarr_data(
            n,
            RAQDPS,
            hours,
            {
                "pm25": np.full(n, 5.0),
                "pm10": np.full(n, 10.0),
                "no2": np.full(n, 3.0),
                "o3": np.full(n, 8.0),
                "so2": np.full(n, 1.0),
            },
        )
        silam_data = _make_zarr_data(
            n,
            SILAM,
            hours,
            {
                "pm25": np.full(n, 5.0),
                "pm10": np.full(n, 10.0),
                "no2": np.full(n, 3.0),
                "o3": np.full(n, 8.0),
                "so2": np.full(n, 1.0),
                "co": np.full(n, 300.0),
            },
        )
        result = prepare_aq_inputs(n, raqdps_data, silam_data, hours)
        assert np.nanmean(result["co"]) == pytest.approx(300.0, abs=5.0)

    def test_no_aq_data_returns_nans(self):
        """Both excluded → all NaN."""
        n = 24
        hours = self._hour_array(n)
        result = prepare_aq_inputs(n, False, False, hours)
        for key in ("pm25", "pm10", "o3", "no2", "so2", "co"):
            assert np.all(np.isnan(result[key]))
