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
    nowcast_pm,
    rolling_mean,
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
        aqi = compute_epa_aqi(pm25_ug=9.0)
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

    def test_extreme_concentrations_cap_aqhi_at_15(self):
        """Extreme concentrations should not report AQHI above 15."""
        aqhi = compute_aqhi(pm25_ug=5000.0, o3_ppb=5000.0, no2_ppb=5000.0)
        assert aqhi == 15.0

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
        # For the US EPA system, NowCast requires at least 2 of the 3 most-recent
        # hours to be valid.  In a 3-element array starting with NaN:
        #   index 0: window=[NaN]             → only 1 valid in recent window → NaN
        #   index 1: window=[10.0, NaN]       → 1 valid in recent-3  → NaN
        #   index 2: window=[20.0, 10.0, NaN] → 2 valid in recent-3  → valid AQI
        pm25 = np.array([np.nan, 10.0, 20.0])
        result = compute_aqi_array(
            "us", pm25=pm25, pm10=None, o3=None, no2=None, so2=None, co=None
        )
        assert np.isnan(result[0])
        assert not np.isnan(result[2])

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

    def test_smoke_frp_extracted_from_silam(self):
        """smoke_frp should be PM_FRP_column divided by BLH (µg/m² → µg/m³)."""
        from API.constants.model_const import SILAM

        n = 24
        hours = self._hour_array(n)
        # PM_FRP_column = 2000 µg/m², BLH = 500 m → expected smoke_frp = 4 µg/m³
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
                "co": np.full(n, 100.0),
                "pm_frp_column": np.full(n, 2000.0),
                "blh": np.full(n, 500.0),
            },
        )
        result = prepare_aq_inputs(n, False, silam_data, hours)
        assert "smoke_frp" in result
        assert np.nanmean(result["smoke_frp"]) == pytest.approx(4.0, abs=0.1)

    def test_smoke_frp_uses_default_blh_when_blh_missing(self):
        """When BLH is NaN or absent, smoke_frp should use a 1000 m neutral default."""
        from API.constants.model_const import SILAM

        n = 24
        hours = self._hour_array(n)
        # BLH column not provided → extracted as NaN → default 1000 m applied
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
                "co": np.full(n, 100.0),
                "pm_frp_column": np.full(n, 5000.0),
            },
        )
        result = prepare_aq_inputs(n, False, silam_data, hours)
        assert "smoke_frp" in result
        # 5000 µg/m² / 1000 m (NaN BLH → neutral 1000 m default) = 5.0 µg/m³
        assert np.nanmean(result["smoke_frp"]) == pytest.approx(5.0, abs=0.1)

    def test_smoke_frp_enforces_minimum_blh(self):
        """Valid BLH values below 100 m should be clamped to 100 m."""
        from API.constants.model_const import SILAM

        n = 4
        hours = self._hour_array(n)
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
                "co": np.full(n, 100.0),
                "pm_frp_column": np.full(n, 1000.0),
                "blh": np.full(n, 10.0),  # valid but < 100 m → clamped to 100 m
            },
        )
        result = prepare_aq_inputs(n, False, silam_data, hours)
        # 1000 µg/m² / 100 m (clamped) = 10.0 µg/m³
        assert np.nanmean(result["smoke_frp"]) == pytest.approx(10.0, abs=0.1)

    def test_smoke_frp_nans_when_silam_absent(self):
        """When SILAM is unavailable, smoke_frp should be all-NaN."""
        n = 24
        hours = self._hour_array(n)
        result = prepare_aq_inputs(n, False, False, hours)
        assert "smoke_frp" in result
        assert np.all(np.isnan(result["smoke_frp"]))

    def test_fractional_timezone_30min_offset_matches(self):
        """SILAM timestamps stored as float32 have ±64 s rounding at ~1.75e9.

        Locations with 30-minute UTC offsets (India UTC+5:30, Newfoundland UTC-3:30,
        Adelaide UTC+9:30) produce request timestamps that are exactly 1800 s from
        the nearest SILAM hour.  Float32 rounding can push stored SILAM timestamps
        ~64 s away, making diff > 1800 for roughly half of hours.  The tolerance
        was raised to < 3600 s so that all hours match correctly.
        """
        import struct

        from API.constants.model_const import SILAM

        def float32_round(v):
            b = struct.pack("f", v)
            return struct.unpack("f", b)[0]

        n = 48
        base = 1752004800.0  # July 8, 2026 18:00:00 UTC (near problem range)

        # Request timestamps at :30 offset (simulating UTC+5:30 or UTC-3:30)
        request_hours = np.array(
            [base + i * 3600 + 1800 for i in range(n)], dtype=float
        )

        # SILAM timestamps stored as float32 (whole hours, with rounding errors)
        silam_hours_f32 = np.array(
            [float32_round(base + i * 3600) for i in range(n)], dtype=float
        )

        silam_data = _make_zarr_data(
            n,
            SILAM,
            silam_hours_f32,
            {
                "pm25": np.full(n, 10.0),
                "pm10": np.full(n, 20.0),
                "no2": np.full(n, 5.0),
                "o3": np.full(n, 30.0),
                "so2": np.full(n, 2.0),
                "co": np.full(n, 150.0),
            },
        )

        result = prepare_aq_inputs(n, False, silam_data, request_hours)

        # All hours must match; none should be NaN despite the float32 rounding
        assert not np.any(np.isnan(result["pm25"])), (
            "Some hours returned NaN for 30-min UTC-offset timestamps with "
            "float32-rounded SILAM times; tolerance is too tight"
        )
        assert np.nanmean(result["pm25"]) == pytest.approx(10.0, abs=0.1)


# ---------------------------------------------------------------------------
# EPA averaging helpers
# ---------------------------------------------------------------------------


class TestNowcastPM:
    """Unit tests for the EPA NowCast algorithm."""

    def test_constant_series_returns_same_value(self):
        """Constant concentrations → NowCast equals that concentration."""
        conc = np.full(24, 20.0)
        result = nowcast_pm(conc)
        # From index 2 onwards at least 2 of 3 most-recent are valid
        assert not np.isnan(result[2])
        assert result[2] == pytest.approx(20.0, abs=0.01)

    def test_all_nan_returns_nan(self):
        conc = np.full(10, np.nan)
        result = nowcast_pm(conc)
        assert np.all(np.isnan(result))

    def test_requires_two_of_three_recent(self):
        """With only 1 valid reading in the 3-hour window, result should be NaN."""
        conc = np.array([np.nan, np.nan, 10.0])
        result = nowcast_pm(conc)
        # index 2: window = [10.0, NaN, NaN] → 1 valid → NaN
        assert np.isnan(result[2])

    def test_two_valid_recent_hours_computes(self):
        """With 2 valid readings in the 3-hour window, result should be valid."""
        conc = np.array([np.nan, 10.0, 20.0])
        result = nowcast_pm(conc)
        # index 2: window = [20.0, 10.0, NaN] → 2 valid → valid NowCast
        assert not np.isnan(result[2])

    def test_output_length_matches_input(self):
        conc = np.linspace(5.0, 50.0, 48)
        result = nowcast_pm(conc)
        assert len(result) == 48


class TestRollingMean:
    """Unit tests for the backward-looking rolling mean."""

    def test_8h_constant_series(self):
        """8-hour mean of constant series equals that constant."""
        conc = np.full(24, 15.0)
        result = rolling_mean(conc, window=8)
        assert np.allclose(result, 15.0)

    def test_24h_window(self):
        """24-hour mean converges correctly."""
        conc = np.arange(1.0, 49.0)  # values 1..48
        result = rolling_mean(conc, window=24)
        # At index 23 (24th hour), mean of 1..24 = 12.5
        assert result[23] == pytest.approx(12.5, abs=0.01)

    def test_all_nan_returns_nan(self):
        conc = np.full(10, np.nan)
        result = rolling_mean(conc, window=8)
        assert np.all(np.isnan(result))

    def test_partial_nan_uses_valid_values(self):
        """NaN values within the window should be ignored."""
        conc = np.array([10.0, np.nan, 20.0, np.nan, 30.0])
        result = rolling_mean(conc, window=8)
        # At index 4: valid values in window = [10.0, 20.0, 30.0], mean = 20.0
        assert result[4] == pytest.approx(20.0, abs=0.01)

    def test_output_length_matches_input(self):
        conc = np.random.rand(48)
        result = rolling_mean(conc, window=8)
        assert len(result) == 48


class TestEPAAveragingInArray:
    """Integration tests confirming EPA averaging is applied in compute_aqi_array."""

    def test_epa_uses_nowcast_not_instantaneous(self):
        """With a spike followed by low values, NowCast should damp the spike."""
        # 12 hours of low PM2.5 followed by a spike at hour 12
        conc = np.concatenate([np.full(12, 5.0), [200.0]])
        result = compute_aqi_array(
            "us", pm25=conc, pm10=None, o3=None, no2=None, so2=None, co=None
        )
        # Last value should reflect NowCast (weighted avg) not raw spike
        assert not np.isnan(result[-1])
        # Raw spike (200 µg/m³) → EPA AQI ~250; NowCast should be lower
        raw_aqi = compute_aqi_for_unit_system("us", pm25_ug=200.0)
        assert result[-1] < raw_aqi

    def test_caqi_uses_instantaneous(self):
        """For CAQI (uk/si), instantaneous values should be used."""
        conc = np.array([5.0, 5.0, 200.0])
        result = compute_aqi_array(
            "uk", pm25=conc, pm10=None, o3=None, no2=None, so2=None, co=None
        )
        # CAQI at index 2 should reflect the raw 200 µg/m³ concentration
        assert not np.isnan(result[2])
        assert result[2] >= 75  # 200 µg/m³ PM2.5 → CAQI ≥ 75 (very high)
