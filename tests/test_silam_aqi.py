"""Tests for SILAM AQI calculation functions.

Tests the EPA NowCast algorithm and AQI calculation functions
used in the FMI SILAM air quality ingest script.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the functions to test - these need to be extracted from the ingest script
# For now, we'll copy the core calculation logic for isolated testing


def calculate_nowcast_concentration(concentrations, num_hours=12):
    """
    Calculate the EPA NowCast weighted concentration for PM2.5 and PM10.

    The NowCast algorithm weights recent hours more heavily than older hours,
    making it more responsive to changing air quality conditions than a
    simple 24-hour average.

    Args:
        concentrations: Array of concentrations with time as the first dimension.
                       Shape: (time, latitude, longitude)
        num_hours: Number of hours to use in NowCast calculation (default 12)

    Returns:
        NowCast weighted concentration array with same shape as input
    """
    if concentrations.shape[0] < 3:
        # Need at least 3 hours for NowCast, return original if not enough data
        return concentrations

    # Limit to available hours
    hours_to_use = min(num_hours, concentrations.shape[0])

    # Initialize output array with same shape as input
    nowcast_result = np.full_like(concentrations, np.nan)

    # Process each time step
    for t in range(concentrations.shape[0]):
        # Determine the window of hours to use (looking back from current hour)
        start_idx = max(0, t - hours_to_use + 1)
        window = concentrations[start_idx : t + 1]

        if window.shape[0] < 3:
            # Not enough hours, use instantaneous value
            nowcast_result[t] = concentrations[t]
            continue

        # Calculate weight factor based on concentration range
        # Weight = 1 - (range / max), minimum 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            c_max = np.nanmax(window, axis=0)
            c_min = np.nanmin(window, axis=0)
            c_range = c_max - c_min

            # Avoid division by zero - weight_factor is 2D (lat, lon)
            weight_factor = np.where(
                c_max > 0, np.maximum(1 - c_range / c_max, 0.5), 0.5
            )

        # Calculate weighted average with spatially varying weight factor
        # Weights: w^0, w^1, w^2, ... w^(n-1) from most recent to oldest
        num_window_hours = window.shape[0]

        # Build weights array with shape (time, lat, lon) for broadcasting
        weights = np.zeros_like(window)
        for i in range(num_window_hours):
            # i=0 is oldest, i=num_window_hours-1 is most recent
            hours_ago = num_window_hours - 1 - i
            weights[i] = weight_factor**hours_ago

        # Calculate weighted concentration
        with np.errstate(invalid="ignore"):
            weighted_sum = np.nansum(window * weights, axis=0)
            weight_sum = np.nansum(np.where(~np.isnan(window), weights, 0), axis=0)
            nowcast_result[t] = np.where(
                weight_sum > 0, weighted_sum / weight_sum, np.nan
            )

    return nowcast_result


def _calc_aqi_for_pollutant(conc, bp, aqi_vals):
    """Calculate AQI for a single pollutant using linear interpolation."""
    if np.isnan(conc):
        return np.nan
    if conc <= 0:
        return 0

    for i in range(len(bp) - 1):
        if bp[i] <= conc < bp[i + 1]:
            aqi = ((aqi_vals[i + 1] - aqi_vals[i]) / (bp[i + 1] - bp[i])) * (
                conc - bp[i]
            ) + aqi_vals[i]
            return aqi

    # If concentration exceeds highest breakpoint
    return aqi_vals[-1]


# EPA AQI breakpoints
PM25_BP = [0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]
PM25_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

PM10_BP = [0, 54, 154, 254, 354, 424, 504, 604]
PM10_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

O3_BP = [0, 108, 140, 170, 210, 400, 504, 604]
O3_AQI = [0, 50, 100, 150, 200, 300, 400, 500]


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
        aqi = _calc_aqi_for_pollutant(6.0, PM25_BP, PM25_AQI)
        assert 0 < aqi <= 50

    def test_aqi_pm25_moderate(self):
        """Test AQI calculation for moderate PM2.5 levels (12.1-35.4 µg/m³)."""
        # PM2.5 of 24.0 should give AQI between 50-100
        aqi = _calc_aqi_for_pollutant(24.0, PM25_BP, PM25_AQI)
        assert 50 < aqi <= 100

    def test_aqi_pm25_unhealthy_sensitive(self):
        """Test AQI for unhealthy for sensitive groups (35.5-55.4 µg/m³)."""
        # PM2.5 of 45.0 should give AQI between 100-150
        aqi = _calc_aqi_for_pollutant(45.0, PM25_BP, PM25_AQI)
        assert 100 < aqi <= 150

    def test_aqi_pm25_unhealthy(self):
        """Test AQI for unhealthy levels (55.5-150.4 µg/m³)."""
        # PM2.5 of 100.0 should give AQI between 150-200
        aqi = _calc_aqi_for_pollutant(100.0, PM25_BP, PM25_AQI)
        assert 150 < aqi <= 200

    def test_aqi_pm10_breakpoints(self):
        """Test PM10 AQI calculation at breakpoints."""
        # PM10 of 54 should give AQI of 50
        aqi = _calc_aqi_for_pollutant(54.0, PM10_BP, PM10_AQI)
        np.testing.assert_allclose(aqi, 50.0, rtol=0.01)

    def test_aqi_zero_concentration(self):
        """Test AQI for zero concentration."""
        aqi = _calc_aqi_for_pollutant(0.0, PM25_BP, PM25_AQI)
        assert aqi == 0

    def test_aqi_nan_concentration(self):
        """Test AQI for NaN concentration."""
        aqi = _calc_aqi_for_pollutant(np.nan, PM25_BP, PM25_AQI)
        assert np.isnan(aqi)

    def test_aqi_exceeds_breakpoints(self):
        """Test AQI for concentration exceeding highest breakpoint."""
        # Very high PM2.5 should return max AQI
        aqi = _calc_aqi_for_pollutant(600.0, PM25_BP, PM25_AQI)
        assert aqi == 500

    def test_aqi_linear_interpolation(self):
        """Test that AQI interpolation is linear within breakpoints."""
        # PM2.5 at exact breakpoint
        aqi_at_bp = _calc_aqi_for_pollutant(12.0, PM25_BP, PM25_AQI)
        np.testing.assert_allclose(aqi_at_bp, 50.0, rtol=0.01)

        # PM2.5 at midpoint between 0 and 12 should give AQI of 25
        aqi_mid = _calc_aqi_for_pollutant(6.0, PM25_BP, PM25_AQI)
        np.testing.assert_allclose(aqi_mid, 25.0, rtol=0.01)


class TestAQICategorization:
    """Tests for AQI category thresholds."""

    def test_aqi_good_category(self):
        """AQI 0-50 is Good."""
        aqi = _calc_aqi_for_pollutant(10.0, PM25_BP, PM25_AQI)
        assert 0 <= aqi <= 50, "PM2.5 of 10 should be in Good category"

    def test_aqi_moderate_category(self):
        """AQI 51-100 is Moderate."""
        aqi = _calc_aqi_for_pollutant(30.0, PM25_BP, PM25_AQI)
        assert 50 < aqi <= 100, "PM2.5 of 30 should be in Moderate category"

    def test_aqi_unhealthy_sensitive_category(self):
        """AQI 101-150 is Unhealthy for Sensitive Groups."""
        aqi = _calc_aqi_for_pollutant(50.0, PM25_BP, PM25_AQI)
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
        aqi = _calc_aqi_for_pollutant(-5.0, PM25_BP, PM25_AQI)
        assert aqi == 0
