"""Fixtures shared by API and ingest tests."""

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
