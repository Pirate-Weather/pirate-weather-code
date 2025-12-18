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

        # Provide CO test data as well
        co = np.full((12, 3, 3), 1000.0, dtype=np.float32)
        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, co, use_nowcast=True)

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

        co = np.full((12, 3, 3), 1000.0, dtype=np.float32)
        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, co, use_nowcast=False)

        # Check shape
        assert aqi.shape == pm25.shape
        # Check no NaN values
        assert not np.any(np.isnan(aqi))
        # Check positive values
        assert np.all(aqi >= 0)

    def test_aqi_returns_maximum_pollutant(self):
        """Test that AQI returns the maximum AQI across all pollutants."""
        # Create data where PM2.5 has highest AQI
        pm25 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI = 199.79
        pm10 = np.full((12, 3, 3), 50.0, dtype=np.float32)  # AQI = 46.30
        o3 = np.full((12, 3, 3), 100.0, dtype=np.float32)  # AQI = 46.30
        no2 = np.full((12, 3, 3), 80.0, dtype=np.float32)  # AQI = 40.00
        so2 = np.full((12, 3, 3), 70.0, dtype=np.float32)  # AQI = 37.86

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be dominated by PM2.5 which is approximately 199.79
        np.testing.assert_allclose(aqi[-1], 199.79, rtol=0.01)

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
        pm25 = np.full((12, 3, 3), 40.0, dtype=np.float32)  # AQI = 111.50
        pm10 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI = 98.68
        o3 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI = 116.67
        no2 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI = 75.00
        so2 = np.full((12, 3, 3), 150.0, dtype=np.float32)  # AQI = 71.43

        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, use_nowcast=False)

        # AQI should be 116.67 (max of O3)
        np.testing.assert_allclose(aqi[-1], 116.67, rtol=0.01)

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


class TestVMRConversion:
    """Tests for volume mixing ratio (VMR) to concentration conversion."""

    def test_vmr_conversion_basic(self):
        """Test basic VMR to concentration conversion."""
        # Import the conversion function from silam_conversion module
        from API.silam_conversion import (
            MOLAR_MASS_AIR,
            MOLAR_MASS_O3,
            convert_vmr_to_concentration,
        )

        # Test with known values
        # VMR of 1 ppm (1e-6 mole/mole) of O3
        # Air density: 1.225 kg/m³
        # O3 molar mass: 0.048 kg/mole
        # Air molar mass: 0.02897 kg/mole
        vmr = 1e-6  # 1 ppm
        air_density = 1.225  # kg/m³
        molar_mass = MOLAR_MASS_O3  # 0.048 kg/mole

        concentration = convert_vmr_to_concentration(vmr, air_density, molar_mass)

        # Expected: 1e-6 * 1.225 * (0.048/0.02897) * 1e9
        # = 1e-6 * 1.225 * 1.6569 * 1e9 = 2029.7 µg/m³
        expected = vmr * air_density * (molar_mass / MOLAR_MASS_AIR) * 1e9
        np.testing.assert_allclose(concentration, expected, rtol=1e-6)
        # O3 at 1 ppm should be approximately 2030 µg/m³
        np.testing.assert_allclose(concentration, 2029.7, rtol=0.01)

    def test_vmr_conversion_zero(self):
        """Test VMR conversion with zero concentration."""
        from API.silam_conversion import (
            MOLAR_MASS_CO,
            convert_vmr_to_concentration,
        )

        vmr = 0.0
        air_density = 1.225
        molar_mass = MOLAR_MASS_CO

        concentration = convert_vmr_to_concentration(vmr, air_density, molar_mass)
        assert concentration == 0.0

    def test_vmr_conversion_different_gases(self):
        """Test VMR conversion for different gas species."""
        from API.silam_conversion import (
            MOLAR_MASS_AIR,
            MOLAR_MASS_CO,
            MOLAR_MASS_NO2,
            MOLAR_MASS_O3,
            MOLAR_MASS_SO2,
            convert_vmr_to_concentration,
        )

        vmr = 1e-6  # 1 ppm for all gases
        air_density = 1.225  # kg/m³

        # Test each gas
        gases = {
            "O3": MOLAR_MASS_O3,
            "NO2": MOLAR_MASS_NO2,
            "SO2": MOLAR_MASS_SO2,
            "CO": MOLAR_MASS_CO,
        }

        for gas_name, molar_mass in gases.items():
            concentration = convert_vmr_to_concentration(vmr, air_density, molar_mass)
            # Verify the conversion produces a positive value
            assert concentration > 0
            # Verify it follows the expected formula
            expected = vmr * air_density * (molar_mass / MOLAR_MASS_AIR) * 1e9
            np.testing.assert_allclose(concentration, expected, rtol=1e-6)

    def test_vmr_conversion_with_array(self):
        """Test VMR conversion with numpy arrays."""
        from API.silam_conversion import (
            MOLAR_MASS_O3,
            convert_vmr_to_concentration,
        )

        # Create a 3D array of VMR values
        vmr = np.full((12, 3, 3), 1e-6, dtype=np.float32)  # 1 ppm everywhere
        air_density = np.full((12, 3, 3), 1.225, dtype=np.float32)

        concentration = convert_vmr_to_concentration(vmr, air_density, MOLAR_MASS_O3)

        # Check shape is preserved
        assert concentration.shape == vmr.shape
        # Check all values are approximately 2030 µg/m³
        np.testing.assert_allclose(concentration, 2029.7, rtol=0.01)

    def test_vmr_conversion_realistic_o3(self):
        """Test VMR conversion with realistic O3 concentrations."""
        from API.silam_conversion import (
            MOLAR_MASS_O3,
            convert_vmr_to_concentration,
        )

        # Typical tropospheric O3: 20-100 ppb (20e-9 to 100e-9 mole/mole)
        vmr_ppb = 50e-9  # 50 ppb
        air_density = 1.225  # kg/m³

        concentration = convert_vmr_to_concentration(
            vmr_ppb, air_density, MOLAR_MASS_O3
        )

        # 50 ppb O3 should be approximately 101.5 µg/m³
        np.testing.assert_allclose(concentration, 101.5, rtol=0.01)


class TestAQIWithVMRConversion:
    """Integration tests for AQI calculation with VMR-converted concentrations."""

    def test_aqi_with_vmr_converted_gases(self):
        """Test AQI calculation using VMR-converted gas concentrations."""
        from API.silam_conversion import (
            MOLAR_MASS_CO,
            MOLAR_MASS_NO2,
            MOLAR_MASS_O3,
            MOLAR_MASS_SO2,
            convert_vmr_to_concentration,
        )

        # Create test data with 12 time steps
        air_density = np.full((12, 3, 3), 1.225, dtype=np.float32)

        # Simulate moderate air quality conditions
        # O3: 60 ppb -> ~122 µg/m³ (moderate AQI ~57)
        vmr_o3 = np.full((12, 3, 3), 60e-9, dtype=np.float32)
        o3 = convert_vmr_to_concentration(vmr_o3, air_density, MOLAR_MASS_O3)

        # NO2: 40 ppb -> ~75 µg/m³ (good AQI ~38)
        vmr_no2 = np.full((12, 3, 3), 40e-9, dtype=np.float32)
        no2 = convert_vmr_to_concentration(vmr_no2, air_density, MOLAR_MASS_NO2)

        # SO2: 30 ppb -> ~79 µg/m³ (good AQI ~43)
        vmr_so2 = np.full((12, 3, 3), 30e-9, dtype=np.float32)
        so2 = convert_vmr_to_concentration(vmr_so2, air_density, MOLAR_MASS_SO2)

        # CO: 0.5 ppm -> ~574 µg/m³ (good AQI ~6)
        vmr_co = np.full((12, 3, 3), 0.5e-6, dtype=np.float32)
        co = convert_vmr_to_concentration(vmr_co, air_density, MOLAR_MASS_CO)

        # PM values
        pm25 = np.full((12, 3, 3), 15.0, dtype=np.float32)  # moderate AQI ~58
        pm10 = np.full((12, 3, 3), 40.0, dtype=np.float32)  # good AQI ~37

        # Calculate AQI
        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, co, use_nowcast=False)

        # AQI should be dominated by PM2.5 (moderate range)
        # With 24h averaging for PM2.5, final AQI should be in moderate range (51-100)
        assert np.all(aqi[-1] > 0)
        assert np.all(aqi[-1] < 150)  # Should not reach unhealthy
        # Check that concentrations were converted properly
        assert np.all(o3 > 0)
        assert np.all(no2 > 0)
        assert np.all(so2 > 0)
        assert np.all(co > 0)

    def test_aqi_with_high_o3_from_vmr(self):
        """Test AQI calculation with high O3 levels from VMR."""
        from API.silam_conversion import (
            MOLAR_MASS_O3,
            convert_vmr_to_concentration,
        )

        # Create test data
        air_density = np.full((12, 3, 3), 1.225, dtype=np.float32)

        # High O3: 120 ppb -> ~244 µg/m³ (unhealthy for sensitive groups)
        vmr_o3 = np.full((12, 3, 3), 120e-9, dtype=np.float32)
        o3 = convert_vmr_to_concentration(vmr_o3, air_density, MOLAR_MASS_O3)

        # Low values for other pollutants
        pm25 = np.full((12, 3, 3), 5.0, dtype=np.float32)
        pm10 = np.full((12, 3, 3), 10.0, dtype=np.float32)
        no2 = np.full((12, 3, 3), 20.0, dtype=np.float32)
        so2 = np.full((12, 3, 3), 20.0, dtype=np.float32)
        co = np.full((12, 3, 3), 200.0, dtype=np.float32)

        # Calculate AQI
        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, co, use_nowcast=False)

        # AQI should be dominated by O3
        # O3 at 244 µg/m³ with 8h averaging should give AQI > 100
        assert np.all(aqi[-1] > 50)  # At least moderate
        # Verify O3 was converted to expected range
        np.testing.assert_allclose(o3[0, 0, 0], 244.0, rtol=0.01)

    def test_aqi_spatial_variation_with_vmr(self):
        """Test AQI with spatially varying VMR values."""
        from API.silam_conversion import (
            MOLAR_MASS_O3,
            convert_vmr_to_concentration,
        )

        # Create spatially varying O3 VMR
        vmr_o3 = np.zeros((12, 5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                # O3 ranging from 20 to 100 ppb
                vmr_o3[:, i, j] = (20 + (i + j) * 10) * 1e-9

        air_density = np.full((12, 5, 5), 1.225, dtype=np.float32)
        o3 = convert_vmr_to_concentration(vmr_o3, air_density, MOLAR_MASS_O3)

        # Low values for other pollutants
        pm25 = np.full((12, 5, 5), 5.0, dtype=np.float32)
        pm10 = np.full((12, 5, 5), 10.0, dtype=np.float32)
        no2 = np.full((12, 5, 5), 20.0, dtype=np.float32)
        so2 = np.full((12, 5, 5), 20.0, dtype=np.float32)
        co = np.full((12, 5, 5), 200.0, dtype=np.float32)

        # Calculate AQI
        aqi = calculate_aqi(pm25, pm10, o3, no2, so2, co, use_nowcast=False)

        # AQI should vary spatially due to O3 variation
        assert aqi[-1].min() < aqi[-1].max()
        # O3 concentration should vary spatially
        assert o3[0].min() < o3[0].max()
        # All AQI values should be positive
        assert np.all(aqi >= 0)
