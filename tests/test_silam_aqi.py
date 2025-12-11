"""Tests for SILAM AQI calculation functions.
Tests the EPA NowCast algorithm and AQI calculation functions
used in the FMI SILAM air quality ingest script.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import EPA AQI breakpoints from shared constants
from API.constants.aqi_const import PM10_AQI, PM10_BP, PM25_AQI, PM25_BP

# Import the functions to test from the shared ingest utilities
from API.ingest_utils import calculate_aqi, calculate_nowcast_concentration


class TestNowCastAlgorithm:
    """Tests for the EPA NowCast algorithm."""

    def test_nowcast_steady_concentration(self):
        """Test NowCast with constant concentrations returns same value."""
        # Create 12 hours of constant concentration data
        steady_data = np.full((12, 3, 3), 25.0, dtype=np.float32)
        result = calculate_nowcast_concentration(steady_data, num_hours=12)

        # With constant data, NowCast should return approximately the same value
        np.testing.assert_allclose(result[-1], 25.0, rtol=0.01)

    def test_nowcast_rising_concentration(self):
        """Test NowCast weights recent values more heavily for rising concentrations."""
        # Create rising concentration data
        rising_data = np.zeros((12, 3, 3), dtype=np.float32)
        for t in range(12):
            rising_data[t] = 10 + t * 5  # 10, 15, 20, ... 65

        result = calculate_nowcast_concentration(rising_data, num_hours=12)

        # Simple average would be 37.5
        simple_avg = np.mean(rising_data, axis=0)[0, 0]
        nowcast_val = result[-1, 0, 0]

        # NowCast should be higher than simple average for rising values
        assert nowcast_val > simple_avg

    def test_nowcast_falling_concentration(self):
        """Test NowCast weights recent values more heavily for falling concentrations."""
        # Create falling concentration data
        falling_data = np.zeros((12, 3, 3), dtype=np.float32)
        for t in range(12):
            falling_data[t] = 65 - t * 5  # 65, 60, 55, ... 10

        result = calculate_nowcast_concentration(falling_data, num_hours=12)

        # Simple average would be 37.5
        simple_avg = np.mean(falling_data, axis=0)[0, 0]
        nowcast_val = result[-1, 0, 0]

        # NowCast should be lower than simple average for falling values
        assert nowcast_val < simple_avg

    def test_nowcast_minimum_hours(self):
        """Test NowCast with less than 3 hours returns original data."""
        short_data = np.full((2, 3, 3), 25.0, dtype=np.float32)
        result = calculate_nowcast_concentration(short_data, num_hours=12)

        # With only 2 hours, should return original data
        np.testing.assert_array_equal(result, short_data)

    def test_nowcast_handles_nan(self):
        """Test NowCast handles NaN values correctly."""
        data_with_nan = np.full((12, 3, 3), 25.0, dtype=np.float32)
        data_with_nan[5, 1, 1] = np.nan

        result = calculate_nowcast_concentration(data_with_nan, num_hours=12)

        # Result should not be all NaN
        assert not np.all(np.isnan(result))

    def test_nowcast_weight_factor_minimum(self):
        """Test that weight factor doesn't go below 0.5."""
        # Create data with large range (weight factor would be 0 without minimum)
        large_range_data = np.zeros((12, 3, 3), dtype=np.float32)
        for t in range(12):
            large_range_data[t] = 10 + t * 10  # 10, 20, 30, ... 120

        result = calculate_nowcast_concentration(large_range_data, num_hours=12)

        # Result should not be NaN even with large range
        assert not np.any(np.isnan(result[-1]))


class TestAQICalculation:
    """Tests for AQI calculation using EPA breakpoints."""

    def test_aqi_pm25_good(self):
        """Test AQI calculation for good PM2.5 levels (0-12 µg/m³)."""
        # PM2.5 of 6.0 should give AQI of 25
        aqi = np.interp(6.0, PM25_BP, PM25_AQI)
        np.testing.assert_allclose(aqi, 25.0, rtol=0.01)

    def test_aqi_pm25_moderate(self):
        """Test AQI calculation for moderate PM2.5 levels (12.1-35.4 µg/m³)."""
        # PM2.5 of 24.0 should give AQI between 50-100
        aqi = np.interp(24.0, PM25_BP, PM25_AQI)
        assert 50 < aqi <= 100

    def test_aqi_pm25_unhealthy_sensitive(self):
        """Test AQI for unhealthy for sensitive groups (35.5-55.4 µg/m³)."""
        # PM2.5 of 45.0 should give AQI between 100-150
        aqi = np.interp(45.0, PM25_BP, PM25_AQI)
        assert 100 < aqi <= 150

    def test_aqi_pm25_unhealthy(self):
        """Test AQI for unhealthy levels (55.5-150.4 µg/m³)."""
        # PM2.5 of 100.0 should give AQI between 150-200
        aqi = np.interp(100.0, PM25_BP, PM25_AQI)
        assert 150 < aqi <= 200

    def test_aqi_pm10_breakpoints(self):
        """Test PM10 AQI calculation at breakpoints."""
        # PM10 of 54 should give AQI of 50
        aqi = np.interp(54.0, PM10_BP, PM10_AQI)
        np.testing.assert_allclose(aqi, 50.0, rtol=0.01)

    def test_aqi_zero_concentration(self):
        """Test AQI for zero concentration."""
        aqi = np.interp(0.0, PM25_BP, PM25_AQI)
        assert aqi == 0

    def test_aqi_nan_concentration(self):
        """Test AQI for NaN concentration."""
        aqi = np.interp(np.nan, PM25_BP, PM25_AQI)
        assert np.isnan(aqi)

    def test_aqi_exceeds_breakpoints(self):
        """Test AQI for concentration exceeding highest breakpoint."""
        # Very high PM2.5 should return max AQI
        aqi = np.interp(600.0, PM25_BP, PM25_AQI)
        assert aqi == 500

    def test_aqi_linear_interpolation(self):
        """Test that AQI interpolation is linear within breakpoints."""
        # PM2.5 at exact breakpoint
        aqi_at_bp = np.interp(12.0, PM25_BP, PM25_AQI)
        np.testing.assert_allclose(aqi_at_bp, 50.0, rtol=0.01)

        # PM2.5 at midpoint between 0 and 12 should give AQI of 25
        aqi_mid = np.interp(6.0, PM25_BP, PM25_AQI)
        np.testing.assert_allclose(aqi_mid, 25.0, rtol=0.01)


class TestAQICategorization:
    """Tests for AQI category thresholds."""

    def test_aqi_good_category(self):
        """AQI 0-50 is Good."""
        aqi = np.interp(10.0, PM25_BP, PM25_AQI)
        assert 0 <= aqi <= 50, "PM2.5 of 10 should be in Good category"

    def test_aqi_moderate_category(self):
        """AQI 51-100 is Moderate."""
        aqi = np.interp(30.0, PM25_BP, PM25_AQI)
        assert 50 < aqi <= 100, "PM2.5 of 30 should be in Moderate category"

    def test_aqi_unhealthy_sensitive_category(self):
        """AQI 101-150 is Unhealthy for Sensitive Groups."""
        aqi = np.interp(50.0, PM25_BP, PM25_AQI)
        assert 100 < aqi <= 150, "PM2.5 of 50 should be Unhealthy for Sensitive Groups"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_point_spatial_data(self):
        """Test NowCast with single spatial point."""
        data = np.full((12, 1, 1), 25.0, dtype=np.float32)
        result = calculate_nowcast_concentration(data, num_hours=12)
        assert result.shape == data.shape

    def test_large_spatial_data(self):
        """Test NowCast with larger spatial dimensions."""
        data = np.full((12, 100, 100), 25.0, dtype=np.float32)
        result = calculate_nowcast_concentration(data, num_hours=12)
        assert result.shape == data.shape

    def test_varying_spatial_values(self):
        """Test NowCast with spatially varying data."""
        data = np.zeros((12, 5, 5), dtype=np.float32)
        for t in range(12):
            data[t] = np.random.uniform(10, 50, (5, 5))

        result = calculate_nowcast_concentration(data, num_hours=12)
        assert result.shape == data.shape
        assert not np.all(np.isnan(result[-1]))

    def test_negative_concentration(self):
        """Test AQI with negative concentration (should return 0)."""
        aqi = np.interp(-5.0, PM25_BP, PM25_AQI)
        assert aqi == 0


class TestCalculateAQI:
    """Tests for the calculate_aqi function."""

    def test_aqi_with_nowcast_enabled(self):
        """Test AQI calculation with NowCast enabled."""
        # Create test data with 12 time steps
        pm25 = np.full((12, 3, 3), 25.0, dtype=np.float32)
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=True)

        # Check shape
        assert aqi.shape == pm25.shape
        # Check no NaN values
        assert not np.any(np.isnan(aqi))
        # Check positive values
        assert np.all(aqi >= 0)

    def test_aqi_with_nowcast_disabled(self):
        """Test AQI calculation with NowCast disabled."""
        # Create test data with 12 time steps
        pm25 = np.full((12, 3, 3), 25.0, dtype=np.float32)
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # Check shape
        assert aqi.shape == pm25.shape
        # Check no NaN values
        assert not np.any(np.isnan(aqi))
        # Check positive values
        assert np.all(aqi >= 0)

    def test_aqi_returns_maximum_pollutant(self):
        """Test that AQI returns the maximum AQI across all pollutants."""
        # Create data where PM2.5 has highest AQI
        pm25 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI ~200
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)  # AQI ~50
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI ~50
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)  # AQI ~40
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)  # AQI ~37

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be dominated by PM2.5 which is ~200
        assert np.all(aqi[-1] > 150)

    def test_aqi_with_zero_concentrations(self):
        """Test AQI calculation with all zero concentrations."""
        pm25 = np.zeros((12, 3, 3), dtype=np.float32)
        pm10 = np.zeros((12, 3, 3), dtype=np.float32)
        o3 = np.zeros((12, 3, 3), dtype=np.float32)
        no2 = np.zeros((12, 3, 3), dtype=np.float32)
        so2 = np.zeros((12, 3, 3), dtype=np.float32)

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be 0 for all zero concentrations
        assert np.all(aqi == 0)

    def test_aqi_with_nan_values(self):
        """Test AQI calculation handles NaN values correctly."""
        pm25 = np.full((12, 3, 3), 25.0, dtype=np.float32)
        pm25[5, 1, 1] = np.nan
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should handle NaN gracefully with nanmax
        # Most values should not be NaN
        assert np.sum(~np.isnan(aqi)) > 0

    def test_aqi_good_category_threshold(self):
        """Test AQI in Good category (0-50)."""
        # All concentrations in Good range
        pm25 = np.full((12, 3, 3), 10.0, dtype=np.float32)  # AQI ~42
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)  # AQI ~50
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI ~46
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)  # AQI ~40
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)  # AQI ~38

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be in Good category
        assert np.all(aqi[-1] <= 50)

    def test_aqi_moderate_category_threshold(self):
        """Test AQI in Moderate category (51-100)."""
        # PM2.5 in Moderate range
        pm25 = np.full((12, 3, 3), 30.0, dtype=np.float32)  # AQI ~89
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)  # AQI ~50
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI ~46
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)  # AQI ~40
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)  # AQI ~38

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be in Moderate category
        assert np.all(aqi[-1] > 50)
        assert np.all(aqi[-1] <= 100)

    def test_aqi_unhealthy_sensitive_category_threshold(self):
        """Test AQI in Unhealthy for Sensitive Groups category (101-150)."""
        # PM2.5 in Unhealthy for Sensitive Groups range
        pm25 = np.full((12, 3, 3), 45.0, dtype=np.float32)  # AQI ~125
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)  # AQI ~50
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI ~46
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)  # AQI ~40
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)  # AQI ~38

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be in Unhealthy for Sensitive Groups category
        assert np.all(aqi[-1] > 100)
        assert np.all(aqi[-1] <= 150)

    def test_aqi_very_unhealthy_category_threshold(self):
        """Test AQI in Unhealthy category (151-200)."""
        # PM2.5 in Unhealthy range
        pm25 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI ~172
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)  # AQI ~50
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI ~46
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)  # AQI ~40
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)  # AQI ~38

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be in Unhealthy category
        assert np.all(aqi[-1] > 150)
        assert np.all(aqi[-1] <= 200)

    def test_aqi_multiple_high_pollutants(self):
        """Test AQI with multiple pollutants at high levels."""
        # Multiple pollutants at moderate to unhealthy levels
        pm25 = np.full((12, 3, 3), 40.0, dtype=np.float32)  # AQI ~112
        pm10 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI ~99
        o3 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI ~112
        no2 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI ~75
        so2 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI ~71

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be around 112 (max of pm25 and o3)
        assert np.all(aqi[-1] > 100)
        assert np.all(aqi[-1] <= 150)

    def test_aqi_spatial_variation(self):
        """Test AQI with spatially varying concentrations."""
        # Create spatially varying PM2.5
        pm25 = np.zeros((12, 5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                pm25[:, i, j] = 10 + (i + j) * 5  # Range from 10 to 50

        pm10 = np.full((12, 5, 5), 50.0, dtype=np.float32)
        o3 = np.full((12, 5, 5), 100.0, dtype=np.float32)
        no2 = np.full((12, 5, 5), 80.0, dtype=np.float32)
        so2 = np.full((12, 5, 5), 70.0, dtype=np.float32)

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should vary spatially
        assert aqi[-1].min() < aqi[-1].max()
        # All values should be positive
        assert np.all(aqi >= 0)
