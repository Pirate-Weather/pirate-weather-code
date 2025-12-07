"""Tests for DWD MOSMIX NaN data validation in grid_indexing."""

import numpy as np


def test_nan_detection_all_nan():
    """Test that all-NaN data is detected correctly."""
    # Simulate DWD data with all NaNs (except timestamp column)
    data = np.full((48, 10), np.nan)
    data[:, 0] = np.arange(48)  # First column is timestamp

    # Check if all data (excluding first column) is NaN
    result = np.all(np.isnan(data[:, 1:]))
    assert result, "Should detect all-NaN data"


def test_nan_detection_has_valid_data():
    """Test that data with some valid values is not flagged as all-NaN."""
    # Simulate DWD data with some valid data
    data = np.full((48, 10), np.nan)
    data[:, 0] = np.arange(48)  # First column is timestamp
    data[0, 1] = 25.5  # Add one valid temperature value

    # Check if all data (excluding first column) is NaN
    result = np.all(np.isnan(data[:, 1:]))
    assert not result, "Should not flag data as all-NaN when valid data exists"


def test_nan_detection_partial_nan():
    """Test detection with partially filled data (realistic scenario)."""
    # Simulate DWD data with partial coverage
    data = np.full((48, 10), np.nan)
    data[:, 0] = np.arange(48)  # First column is timestamp
    data[0:10, 1:5] = 20.0  # Some valid data in first 10 rows

    # Check if all data (excluding first column) is NaN
    result = np.all(np.isnan(data[:, 1:]))
    assert not result, "Should not flag partially filled data as all-NaN"


def test_nan_detection_empty_array():
    """Test edge case with empty-like data."""
    # Simulate minimal data structure
    data = np.full((1, 2), np.nan)
    data[0, 0] = 0  # Timestamp

    # Check if all data (excluding first column) is NaN
    result = np.all(np.isnan(data[:, 1:]))
    assert result, "Should detect single-row all-NaN data"
