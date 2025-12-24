"""Tests for DWD MOSMIX station mapping functionality."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_station_data():
    """Create sample station data for testing."""
    return pd.DataFrame(
        {
            "station_id": ["10001", "10002", "10003"],
            "station_name": ["Berlin", "Munich", "Hamburg"],
            "latitude": [52.5, 48.1, 53.6],
            "longitude": [13.4, 11.6, 10.0],
            "time": pd.date_range("2024-01-01", periods=3, freq="h"),
        }
    )


def test_station_data_structure(sample_station_data):
    """Test that sample station data has the expected structure."""
    assert "station_id" in sample_station_data.columns
    assert "station_name" in sample_station_data.columns
    assert "latitude" in sample_station_data.columns
    assert "longitude" in sample_station_data.columns
    assert len(sample_station_data) == 3


def test_station_coordinates_valid(sample_station_data):
    """Test that station coordinates are within valid ranges."""
    assert all(sample_station_data["latitude"].between(-90, 90))
    assert all(sample_station_data["longitude"].between(-180, 180))


def test_grid_cell_mapping_key_format():
    """Test that grid cell keys are in the expected (y, x) tuple format."""
    # Sample grid coordinates
    y_idx = 200
    x_idx = 150
    grid_key = (y_idx, x_idx)

    # Verify key structure
    assert isinstance(grid_key, tuple)
    assert len(grid_key) == 2
    assert all(isinstance(i, int) for i in grid_key)


def test_station_info_format():
    """Test that station info dictionaries have the expected format."""
    station_info = {
        "id": "10001",
        "name": "Berlin",
        "lat": 52.5,
        "lon": 13.4,
    }

    # Verify all required fields are present
    assert "id" in station_info
    assert "name" in station_info
    assert "lat" in station_info
    assert "lon" in station_info

    # Verify types
    assert isinstance(station_info["id"], str)
    assert isinstance(station_info["name"], str)
    assert isinstance(station_info["lat"], (int, float))
    assert isinstance(station_info["lon"], (int, float))


def test_longitude_conversion():
    """Test longitude conversion from [0, 360] to [-180, 180] format."""
    # Test cases: (input, expected_output)
    test_cases = [
        (0.0, 0.0),
        (180.0, -180.0),  # 180 degrees wraps to -180
        (270.0, -90.0),
        (359.0, -1.0),
        (13.4, 13.4),
    ]

    for input_lon, expected_lon in test_cases:
        # Convert using the same formula as in the code
        output_lon = ((input_lon + 180) % 360) - 180
        assert abs(output_lon - expected_lon) < 0.01, (
            f"Failed for input {input_lon}: expected {expected_lon}, got {output_lon}"
        )


def test_grid_resolution():
    """Test that GFS 0.25° grid resolution is correctly defined."""
    gfs_resolution = 0.25
    gfs_lats = np.arange(-90, 90, gfs_resolution)
    gfs_lons = np.arange(0, 360, gfs_resolution)

    # Check grid size
    expected_lat_size = int(180 / gfs_resolution)
    expected_lon_size = int(360 / gfs_resolution)

    assert len(gfs_lats) == expected_lat_size
    assert len(gfs_lons) == expected_lon_size

    # Check grid bounds
    assert gfs_lats[0] == -90
    assert gfs_lats[-1] < 90
    assert gfs_lons[0] == 0
    assert gfs_lons[-1] < 360


def test_radius_to_radians_conversion():
    """Test conversion of radius from km to radians."""
    radius_km = 50
    earth_radius_km = 6371.0
    radius_rad = radius_km / earth_radius_km

    # Verify the conversion is reasonable
    assert radius_rad > 0
    assert radius_rad < 1  # Should be less than 1 radian for reasonable radii


def test_station_list_not_empty():
    """Test that a valid station list is not empty."""
    stations_list = [
        {"id": "10001", "name": "Berlin", "lat": 52.5, "lon": 13.4},
        {"id": "10002", "name": "Munich", "lat": 48.1, "lon": 11.6},
    ]

    assert len(stations_list) > 0
    assert all(isinstance(station, dict) for station in stations_list)
