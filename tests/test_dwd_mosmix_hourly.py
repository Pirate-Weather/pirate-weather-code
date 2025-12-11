"""Tests for DWD MOSMIX data in hourly forecasts."""

import logging

import numpy as np

from API.constants.model_const import DWD_MOSMIX
from API.constants.shared_const import KELVIN_TO_CELSIUS
from API.data_inputs import prepare_data_inputs
from API.forecast_sources import SourceMetadata, merge_hourly_models


def test_dwd_mosmix_temperature_conversion():
    """Test that DWD MOSMIX temperature is properly converted from Kelvin to Celsius."""
    # Simulate DWD MOSMIX data with temperature in Kelvin
    # Using actual Ottawa data: 262.85K for first hour
    num_hours = 10
    dwd_data = np.full((num_hours + 50, max(DWD_MOSMIX.values()) + 1), np.nan)

    # Set timestamps (Unix epoch seconds)
    base_time = 1700000000
    dwd_data[:, 0] = np.arange(num_hours + 50) * 3600 + base_time

    # Set temperatures in Kelvin using actual Ottawa data
    ottawa_temps_k = [
        262.85,
        261.55,
        261.25,
        261.05,
        260.55,
        260.45,
        259.85,
        259.55,
        258.85,
        258.25,
    ]
    dwd_data[:num_hours, DWD_MOSMIX["temp"]] = ottawa_temps_k

    # Set dew point in Kelvin
    dwd_data[:, DWD_MOSMIX["dew"]] = 260.0

    # Set cloud cover in percent (95%)
    dwd_data[:, DWD_MOSMIX["cloud"]] = 95.0

    # Set visibility in meters (4500m)
    dwd_data[:, DWD_MOSMIX["vis"]] = 4500.0

    # Simulate the conversion that happens in convert_data_to_celsius
    dwd_data[:, DWD_MOSMIX["temp"]] -= KELVIN_TO_CELSIUS
    dwd_data[:, DWD_MOSMIX["dew"]] -= KELVIN_TO_CELSIUS

    # Now merge the data
    metadata = SourceMetadata(["dwd_mosmix"], {}, {})
    logger = logging.getLogger(__name__)
    merge_result = merge_hourly_models(
        metadata=metadata,
        num_hours=num_hours,
        base_day_utc_grib=base_time,
        data_hrrrh=None,
        data_h2=None,
        data_nbm=None,
        data_nbm_fire=None,
        data_gfs=None,
        data_ecmwf=None,
        data_gefs=None,
        data_dwd_mosmix=dwd_data,
        logger=logger,
        loc_tag="test",
    )

    dwd_merged = merge_result.dwd_mosmix

    # Verify the merged data has converted temperatures
    assert dwd_merged is not None, "DWD MOSMIX merged data should not be None"
    assert dwd_merged.shape[0] == num_hours, (
        f"Expected {num_hours} hours, got {dwd_merged.shape[0]}"
    )

    # Check first temperature is correctly converted: 262.85K - 273.15 = -10.3°C
    temp_celsius = dwd_merged[0, DWD_MOSMIX["temp"]]
    assert not np.isnan(temp_celsius), "Temperature should not be NaN"
    assert np.isclose(temp_celsius, -10.3, atol=0.1), (
        f"Temperature should be -10.3°C (from 262.85K), got {temp_celsius}°C"
    )

    # Check second temperature: 261.55K - 273.15 = -11.6°C
    temp_celsius_2 = dwd_merged[1, DWD_MOSMIX["temp"]]
    assert not np.isnan(temp_celsius_2), "Second temperature should not be NaN"
    assert np.isclose(temp_celsius_2, -11.6, atol=0.1), (
        f"Second temperature should be -11.6°C (from 261.55K), got {temp_celsius_2}°C"
    )

    # Check cloud cover is in percent (should be 95)
    cloud_pct = dwd_merged[0, DWD_MOSMIX["cloud"]]
    assert not np.isnan(cloud_pct), "Cloud cover should not be NaN"
    assert np.isclose(cloud_pct, 95.0, atol=0.1), (
        f"Cloud cover should be 95%, got {cloud_pct}%"
    )

    # Check visibility is in meters (should be 4500)
    vis_m = dwd_merged[0, DWD_MOSMIX["vis"]]
    assert not np.isnan(vis_m), "Visibility should not be NaN"
    assert np.isclose(vis_m, 4500.0, atol=1.0), (
        f"Visibility should be 4500m, got {vis_m}m"
    )


def test_dwd_mosmix_hourly_inputs():
    """Test that DWD MOSMIX data is properly used in hourly inputs."""
    num_hours = 10

    # Create DWD MOSMIX merged data (already converted to Celsius)
    dwd_merged = np.full((num_hours, max(DWD_MOSMIX.values()) + 1), np.nan)
    dwd_merged[:, 0] = np.arange(num_hours) * 3600  # timestamps
    dwd_merged[:, DWD_MOSMIX["temp"]] = -10.7  # Celsius
    dwd_merged[:, DWD_MOSMIX["dew"]] = -13.15  # Celsius
    dwd_merged[:, DWD_MOSMIX["cloud"]] = 95.0  # percent
    dwd_merged[:, DWD_MOSMIX["vis"]] = 4500.0  # meters
    dwd_merged[:, DWD_MOSMIX["humidity"]] = 85.0  # percent
    dwd_merged[:, DWD_MOSMIX["pressure"]] = 101325.0  # Pa

    # Prepare inputs (no other sources)
    inputs = prepare_data_inputs(
        source_list=["dwd_mosmix"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=dwd_merged,
        ecmwf_merged=None,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=45.0,  # Arbitrary location for testing
        lon=-75.0,
    )

    # Verify temperature_inputs uses DWD MOSMIX data
    temp_inputs = inputs["temperature_inputs"]
    assert temp_inputs.shape[0] == num_hours, (
        "Temperature inputs should have correct number of hours"
    )
    assert temp_inputs.shape[1] >= 1, (
        "Temperature inputs should have at least one source"
    )

    # DWD MOSMIX should be the first (and only) source, so check column 0
    first_temp = temp_inputs[0, 0]
    assert not np.isnan(first_temp), "Temperature should not be NaN"
    assert np.isclose(first_temp, -10.7, atol=0.1), (
        f"First temperature should be -10.7°C from DWD MOSMIX, got {first_temp}°C"
    )

    # Verify cloud_inputs uses DWD MOSMIX data (converted to fraction)
    cloud_inputs = inputs["cloud_inputs"]
    first_cloud = cloud_inputs[0, 0]
    assert not np.isnan(first_cloud), "Cloud cover should not be NaN"
    assert np.isclose(first_cloud, 0.95, atol=0.01), (
        f"First cloud cover should be 0.95 from DWD MOSMIX, got {first_cloud}"
    )

    # Verify vis_inputs uses DWD MOSMIX data
    vis_inputs = inputs["vis_inputs"]
    first_vis = vis_inputs[0, 0]
    assert not np.isnan(first_vis), "Visibility should not be NaN"
    assert np.isclose(first_vis, 4500.0, atol=1.0), (
        f"First visibility should be 4500m from DWD MOSMIX, got {first_vis}m"
    )


def test_dwd_mosmix_with_multiple_sources():
    """Test that DWD MOSMIX is used when higher priority sources are excluded."""
    num_hours = 10

    # Create DWD MOSMIX merged data
    dwd_merged = np.full((num_hours, max(DWD_MOSMIX.values()) + 1), np.nan)
    dwd_merged[:, 0] = np.arange(num_hours) * 3600
    dwd_merged[:, DWD_MOSMIX["temp"]] = -10.7
    dwd_merged[:, DWD_MOSMIX["cloud"]] = 95.0
    dwd_merged[:, DWD_MOSMIX["vis"]] = 4500.0

    # Prepare inputs with only DWD MOSMIX and GFS (GFS has lower priority)
    # In this case, DWD MOSMIX should be used
    inputs = prepare_data_inputs(
        source_list=["dwd_mosmix", "gfs"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=dwd_merged,
        ecmwf_merged=None,
        gefs_merged=None,
        gfs_merged=None,  # Even if GFS is in source_list, pass None to simulate exclusion
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=45.0,  # Arbitrary location for testing
        lon=-75.0,
    )

    # Verify DWD MOSMIX data is used (should be in first column)
    temp_inputs = inputs["temperature_inputs"]
    first_temp = temp_inputs[0, 0]
    assert not np.isnan(first_temp), "Temperature should not be NaN"
    assert np.isclose(first_temp, -10.7, atol=0.1), (
        f"Should use DWD MOSMIX temperature (-10.7°C), got {first_temp}°C"
    )


def test_dwd_mosmix_timestamp_alignment():
    """Test that DWD MOSMIX data is properly aligned by timestamp, not by index."""
    num_hours = 12
    base_time = 1700000000  # Arbitrary base time (represents start of day in test)

    # Create DWD MOSMIX data that starts 3 hours AFTER base_time
    # This simulates a forecast run that doesn't include the first few hours
    dwd_data = np.full((20, max(DWD_MOSMIX.values()) + 1), np.nan)

    # Timestamps start at base_time + 3 hours
    offset_hours = 3
    dwd_data[:, 0] = np.arange(20) * 3600 + base_time + offset_hours * 3600

    # Set temperatures for hours 3-14 (indices 0-11 in dwd_data)
    # Use a simple pattern: -10 - hour_number
    for i in range(12):
        actual_hour = offset_hours + i
        dwd_data[i, DWD_MOSMIX["temp"]] = -10.0 - actual_hour

    # Merge the data
    metadata = SourceMetadata(["dwd_mosmix"], {}, {})
    logger = logging.getLogger(__name__)
    merge_result = merge_hourly_models(
        metadata=metadata,
        num_hours=num_hours,
        base_day_utc_grib=base_time,
        data_hrrrh=None,
        data_h2=None,
        data_nbm=None,
        data_nbm_fire=None,
        data_gfs=None,
        data_ecmwf=None,
        data_gefs=None,
        data_dwd_mosmix=dwd_data,
        logger=logger,
        loc_tag="test_offset",
    )

    dwd_merged = merge_result.dwd_mosmix
    assert dwd_merged is not None, "DWD MOSMIX merged data should not be None"

    # Hours 0-2 should be NaN (no data available)
    for hour in range(offset_hours):
        temp = dwd_merged[hour, DWD_MOSMIX["temp"]]
        assert np.isnan(temp), (
            f"Hour {hour} should be NaN (before data starts), got {temp}"
        )

    # Hours 3-11 should have correct temperatures aligned by timestamp
    for hour in range(offset_hours, num_hours):
        temp = dwd_merged[hour, DWD_MOSMIX["temp"]]
        expected_temp = -10.0 - hour
        assert not np.isnan(temp), f"Hour {hour} should not be NaN"
        assert np.isclose(temp, expected_temp, atol=0.1), (
            f"Hour {hour} should be {expected_temp}°C, got {temp}°C"
        )
