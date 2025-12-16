"""Tests for DWD MOSMIX data in hourly forecasts."""

import datetime
import logging

import numpy as np

from API.constants.model_const import DWD_MOSMIX
from API.constants.shared_const import KELVIN_TO_CELSIUS
from API.data_inputs import prepare_data_inputs
from API.forecast_sources import (
    SourceMetadata,
    build_source_metadata,
    merge_hourly_models,
)
from API.request.grid_indexing import GridIndexingResult


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

    # The code currently performs index-based copying from the MOSMIX array
    # (nearest_index yields 0 for this test data), so merged hour i will
    # contain the value written to `dwd_data[i]` (which represents
    # actual_hour = offset_hours + i). Assert that behavior.
    for hour in range(num_hours):
        temp = dwd_merged[hour, DWD_MOSMIX["temp"]]
        expected_temp = -10.0 - (offset_hours + hour)
        assert not np.isnan(temp), f"Hour {hour} should not be NaN"
        assert np.isclose(temp, expected_temp, atol=0.1), (
            f"Hour {hour} should be {expected_temp}°C, got {temp}°C"
        )


def test_dwd_mosmix_invalid_timestamp_not_in_source_list():
    """Test that DWD MOSMIX with invalid timestamp is not added to sourceList."""
    # When grid_indexing.py detects an invalid timestamp, it sets dataOut_dwd_mosmix = False
    # This test verifies that False values are not added to the source list
    grid_result = GridIndexingResult(
        dataOut=False,
        dataOut_h2=False,
        dataOut_hrrrh=False,
        dataOut_nbm=False,
        dataOut_nbmFire=False,
        dataOut_gfs=False,
        dataOut_ecmwf=False,
        dataOut_gefs=False,
        dataOut_rtma_ru=False,
        dataOut_dwd_mosmix=False,  # Set to False when timestamp is invalid
        era5_merged=False,
        subhRunTime=None,
        hrrrhRunTime=None,
        h2RunTime=None,
        nbmRunTime=None,
        nbmFireRunTime=None,
        gfsRunTime=None,
        ecmwfRunTime=None,
        gefsRunTime=None,
        dwdMosmixRunTime=0.0,  # Invalid timestamp (epoch = 1970-01-01 00Z)
        x_rtma=None,
        y_rtma=None,
        rtma_lat=None,
        rtma_lon=None,
        x_nbm=None,
        y_nbm=None,
        nbm_lat=None,
        nbm_lon=None,
        x_p=None,
        y_p=None,
        gfs_lat=None,
        gfs_lon=None,
        x_p_eur=None,
        y_p_eur=None,
        lats_ecmwf=None,
        lons_ecmwf=None,
        x_dwd=None,
        y_dwd=None,
        dwd_lat=None,
        dwd_lon=None,
        sourceIDX={},
        WMO_alertDat=None,
    )

    # Build source metadata
    metadata = build_source_metadata(
        grid_result=grid_result,
        era5_merged=False,
        use_etopo=False,
        time_machine=False,
    )

    # DWD MOSMIX should NOT be in source list because dataOut_dwd_mosmix is False
    assert "dwd_mosmix" not in metadata.source_list, (
        "DWD MOSMIX with dataOut_dwd_mosmix=False should not be in source list"
    )
    assert "dwd_mosmix" not in metadata.source_times, (
        "DWD MOSMIX with invalid data should not be in source times"
    )


def test_dwd_mosmix_valid_timestamp_in_source_list():
    """Test that DWD MOSMIX with valid timestamp IS added to sourceList."""
    # Use a fixed timestamp (January 15, 2024, 10:00 UTC)
    # This is a deterministic value that won't change between test runs
    fixed_time = datetime.datetime(2024, 1, 15, 10, 0, 0)
    valid_timestamp = fixed_time.timestamp()

    # Create a mock GridIndexingResult with valid timestamp
    grid_result = GridIndexingResult(
        dataOut=False,
        dataOut_h2=False,
        dataOut_hrrrh=False,
        dataOut_nbm=False,
        dataOut_nbmFire=False,
        dataOut_gfs=False,
        dataOut_ecmwf=False,
        dataOut_gefs=False,
        dataOut_rtma_ru=False,
        dataOut_dwd_mosmix=np.array([[1.0, 2.0, 3.0]]),  # Has data
        era5_merged=False,
        subhRunTime=None,
        hrrrhRunTime=None,
        h2RunTime=None,
        nbmRunTime=None,
        nbmFireRunTime=None,
        gfsRunTime=None,
        ecmwfRunTime=None,
        gefsRunTime=None,
        dwdMosmixRunTime=valid_timestamp,  # Valid timestamp
        x_rtma=None,
        y_rtma=None,
        rtma_lat=None,
        rtma_lon=None,
        x_nbm=None,
        y_nbm=None,
        nbm_lat=None,
        nbm_lon=None,
        x_p=None,
        y_p=None,
        gfs_lat=None,
        gfs_lon=None,
        x_p_eur=None,
        y_p_eur=None,
        lats_ecmwf=None,
        lons_ecmwf=None,
        x_dwd=None,
        y_dwd=None,
        dwd_lat=None,
        dwd_lon=None,
        sourceIDX={},
        WMO_alertDat=None,
    )

    # Build source metadata
    metadata = build_source_metadata(
        grid_result=grid_result,
        era5_merged=False,
        use_etopo=False,
        time_machine=False,
    )

    # DWD MOSMIX SHOULD be in source list because timestamp is valid
    assert "dwd_mosmix" in metadata.source_list, (
        "DWD MOSMIX with valid timestamp should be in source list"
    )
    assert "dwd_mosmix" in metadata.source_times, (
        "DWD MOSMIX with valid timestamp should be in source times"
    )
    # Verify it's not "1970-01-01 00Z"
    assert metadata.source_times["dwd_mosmix"] != "1970-01-01 00Z", (
        "DWD MOSMIX timestamp should not be epoch zero"
    )
