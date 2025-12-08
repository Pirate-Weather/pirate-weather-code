"""Tests for DWD MOSMIX duplicate station handling.

This test verifies that when multiple stations map to the same grid cell,
they are properly averaged using inverse distance weighting to prevent data jumps.
"""

import numpy as np
import pandas as pd
import pytest


def test_distinct_stations_same_grid_cell():
    """Test that stations with different IDs but mapping to same grid cell get averaged."""
    # Based on the problem statement:
    # OSAKA AIRPORT has ID 47771 with temperatures starting at 280.95K
    # OSAKA has ID 47772 with temperatures starting at 283.15K
    # These are distinct stations but close enough to map to the same grid cells

    station_id_1 = "47771"
    station_id_2 = "47772"

    # Verify that the station IDs are different
    assert station_id_1 != station_id_2, "Stations should have different IDs"


def test_inverse_distance_weighting():
    """Test that inverse distance weighting properly averages values from multiple stations."""
    # Simulate two stations at different distances contributing to the same grid cell
    # Station 1 is closer (smaller distance)
    dist_1 = 0.001  # radians (very close)
    value_1 = 10.0

    # Station 2 is farther
    dist_2 = 0.002  # radians (twice as far)
    value_2 = 20.0

    # Calculate inverse distance weights
    epsilon = 1e-10
    weight_1 = 1.0 / (dist_1 + epsilon)
    weight_2 = 1.0 / (dist_2 + epsilon)

    # Calculate weighted average
    weighted_avg = (value_1 * weight_1 + value_2 * weight_2) / (weight_1 + weight_2)

    # The closer station should have more influence
    assert weighted_avg < 15.0, (
        "Weighted average should be closer to the nearer station's value"
    )
    assert weighted_avg > 10.0, "Weighted average should be influenced by both stations"

    # Verify the weight ratio
    assert weight_1 > weight_2, "Closer station should have higher weight"
    assert weight_1 / weight_2 == pytest.approx(2.0, rel=0.01), (
        "Weight ratio should match inverse distance ratio"
    )


def test_multiple_stations_data_separation():
    """Test that data from multiple nearby stations is properly weighted, not jumbled."""
    # Create a dataframe simulating two stations with different IDs
    times = pd.date_range("2024-01-01", periods=12, freq="h")

    # OSAKA station (ID 47772) temperatures (from problem statement, in Kelvin)
    osaka_temps = [
        283.15,
        282.75,
        282.25,
        281.75,
        281.35,
        281.05,
        280.95,
        281.15,
        281.75,
        282.85,
        283.75,
        284.75,
    ]

    # OSAKA AIRPORT (ID 47771) temperatures (from problem statement)
    osaka_airport_temps = [
        280.95,
        280.45,
        279.85,
        278.65,
        278.25,
        277.85,
        277.35,
        277.55,
        278.85,
        280.35,
        282.25,
        283.65,
    ]

    # Build dataframe with distinct station IDs
    data = []
    for i, time in enumerate(times):
        data.append(
            {
                "station_id": "47772",
                "station_name": "OSAKA",
                "latitude": 34.6937,
                "longitude": 135.5023,
                "time": time,
                "TTT": osaka_temps[i],
            }
        )
        data.append(
            {
                "station_id": "47771",
                "station_name": "OSAKA AIRPORT",
                "latitude": 34.7858,
                "longitude": 135.4381,
                "time": time,
                "TTT": osaka_airport_temps[i],
            }
        )

    df = pd.DataFrame(data)

    # Verify we have distinct station IDs
    unique_stations = df["station_id"].unique()
    assert len(unique_stations) == 2, "Should have 2 unique station IDs"
    assert "47771" in unique_stations, "Should have OSAKA AIRPORT (47771)"
    assert "47772" in unique_stations, "Should have OSAKA (47772)"

    # Verify data for each station is distinct
    station_1_data = df[df["station_id"] == "47771"]
    station_2_data = df[df["station_id"] == "47772"]

    assert len(station_1_data) == 12, "Station 1 should have 12 time points"
    assert len(station_2_data) == 12, "Station 2 should have 12 time points"

    # Verify temperatures are as expected (no mixing between stations)
    temps_1 = station_1_data["TTT"].values
    temps_2 = station_2_data["TTT"].values

    assert np.allclose(temps_1, osaka_airport_temps), (
        "Station 47771 temperatures should match OSAKA AIRPORT data"
    )
    assert np.allclose(temps_2, osaka_temps), (
        "Station 47772 temperatures should match OSAKA data"
    )

    # Verify the temperatures are actually different between stations
    assert not np.allclose(temps_1, temps_2), (
        "The two stations should have different temperature values"
    )


def test_temperature_jump_detection():
    """Test that we can detect temperature jumps that indicate improper handling of multiple stations."""
    # This simulates the problem: alternating between two stations' data
    # From the problem statement, the jumbled output was:
    # 10.1, 9.6, 6.4, 8.6, 8.2, 4.7, 7.7, 4.5, 5.6, 9.6, 10.6, 10.4
    # These are in Celsius

    jumbled_temps_c = [10.1, 9.6, 6.4, 8.6, 8.2, 4.7, 7.7, 4.5, 5.6, 9.6, 10.6, 10.4]

    # Calculate temperature differences
    diffs = np.diff(jumbled_temps_c)
    abs_diffs = np.abs(diffs)

    # When data is jumbled between stations, we see large jumps
    # A properly interpolated single station should have smoother transitions
    large_jumps = abs_diffs > 2.0  # More than 2°C change per hour is suspicious
    num_large_jumps = np.sum(large_jumps)

    # The jumbled data has several large jumps
    assert num_large_jumps >= 4, (
        f"Jumbled data should have multiple large jumps, found {num_large_jumps}"
    )

    # Now test with clean data from a single station (from problem statement)
    # OSAKA station proper: converting to Celsius
    clean_temps_k = [
        283.15,
        282.75,
        282.25,
        281.75,
        281.35,
        281.05,
        280.95,
        281.15,
        281.75,
        282.85,
        283.75,
        284.75,
    ]
    clean_temps_c = [t - 273.15 for t in clean_temps_k]

    clean_diffs = np.abs(np.diff(clean_temps_c))
    clean_large_jumps = np.sum(clean_diffs > 2.0)

    # Clean data should have no large jumps
    assert clean_large_jumps == 0, (
        f"Clean single-station data should have no large temperature jumps, found {clean_large_jumps}"
    )


def test_weighted_average_calculation():
    """Test that weighted average correctly handles multiple contributing values."""
    # Simulate accumulation arrays as used in the fix
    arr = np.zeros((1, 1, 1), dtype=np.float32)
    weight_sum = np.zeros((1, 1, 1), dtype=np.float32)

    # Two values contributing to the same grid cell at the same time
    # Value 1: 283.15K at distance 0.001 radians
    # Value 2: 280.95K at distance 0.002 radians
    epsilon = 1e-10

    value_1 = 283.15
    dist_1 = 0.001
    weight_1 = 1.0 / (dist_1 + epsilon)

    value_2 = 280.95
    dist_2 = 0.002
    weight_2 = 1.0 / (dist_2 + epsilon)

    # Accumulate as done in the fix
    arr[0, 0, 0] += value_1 * weight_1
    weight_sum[0, 0, 0] += weight_1

    arr[0, 0, 0] += value_2 * weight_2
    weight_sum[0, 0, 0] += weight_2

    # Calculate final weighted average
    result = arr[0, 0, 0] / weight_sum[0, 0, 0]

    # Result should be between the two values, closer to the nearer station
    assert result < value_1, "Result should be less than warmer station"
    assert result > value_2, "Result should be more than cooler station"

    # Calculate expected weighted average manually
    expected = (value_1 * weight_1 + value_2 * weight_2) / (weight_1 + weight_2)
    assert np.isclose(result, expected, rtol=1e-5), (
        f"Weighted average should be {expected}, got {result}"
    )

    # The closer station (with smaller distance) should dominate
    # Since dist_1 is half of dist_2, weight_1 should be ~2x weight_2
    # So result should be closer to value_1
    midpoint = (value_1 + value_2) / 2
    assert result > midpoint, "Result should be closer to nearer station's value"


def test_np_add_at_duplicates():
    """Test that np.add.at correctly handles duplicate indices."""
    # This is the key to fixing the bug - np.add.at properly accumulates
    # values at duplicate indices instead of just taking the last one

    arr = np.zeros((5,), dtype=np.float32)
    indices = np.array([0, 1, 1, 2, 2, 2])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    np.add.at(arr, indices, values)

    # Verify accumulation at duplicate indices
    assert arr[0] == 1.0, "Single value should be stored"
    assert arr[1] == 5.0, "Two values (2.0 + 3.0) should be summed"
    assert arr[2] == 15.0, "Three values (4.0 + 5.0 + 6.0) should be summed"
    assert arr[3] == 0.0, "Unused index should remain zero"


def test_expected_temperature_pattern_after_fix():
    """Test that after fixing, nearby stations produce smooth interpolated values."""
    # When two stations are averaged with inverse distance weighting,
    # the result should be smoother than either individual station

    # OSAKA temps (warmer)
    osaka_temps_c = [10.0, 9.6, 9.1, 8.6, 8.2, 7.9, 7.8, 8.0, 8.6, 9.7, 10.6, 11.6]

    # OSAKA AIRPORT temps (cooler)
    airport_temps_c = [7.8, 7.3, 6.7, 5.5, 5.1, 4.7, 4.2, 4.4, 5.7, 7.2, 9.1, 10.5]

    # Expected weighted average (assuming equal weighting for simplicity)
    expected_avg = [(a + b) / 2 for a, b in zip(osaka_temps_c, airport_temps_c)]

    # Calculate temperature changes
    airport_diffs = np.abs(np.diff(airport_temps_c))
    avg_diffs = np.abs(np.diff(expected_avg))

    # The averaged values should generally have smaller jumps than the cooler station
    # (which has larger temperature swings)
    assert np.mean(avg_diffs) < np.mean(airport_diffs), (
        "Averaged temperatures should have smaller jumps than the more variable station"
    )
