import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PW_API = os.environ.get("PW_API")

# Cache the client to avoid reloading the module multiple times
_cached_client = None


def _get_client():
    """Load ``responseLocal`` and return a :class:`TestClient`."""
    global _cached_client

    if _cached_client is not None:
        return _cached_client

    os.environ["STAGE"] = "TESTING"
    os.environ["save_type"] = "S3"
    os.environ.setdefault("useETOPO", "FALSE")

    api_path = Path(__file__).resolve().parents[1] / "API"
    sys.path.insert(0, str(api_path))
    response_path = api_path / "responseLocal.py"
    spec = importlib.util.spec_from_file_location("responseLocal", response_path)
    response_local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(response_local)
    _cached_client = TestClient(response_local.app)
    return _cached_client


def _check_forecast_structure(data: dict) -> None:
    """Validate that the key forecast blocks contain realistic data.

    The checks below use the Pirate Weather OpenAPI spec as reference for the
    expected fields in each forecast block.
    """

    assert "currently" in data
    assert "minutely" in data
    assert "hourly" in data
    assert "daily" in data

    curr = data["currently"]
    assert isinstance(curr.get("summary"), str)
    assert curr.get("summary")
    assert isinstance(curr.get("icon"), str)
    assert curr.get("icon")
    assert isinstance(curr.get("temperature"), (int, float))
    assert -100 <= curr["temperature"] <= 150
    assert isinstance(curr.get("humidity"), (int, float))
    assert 0 <= curr["humidity"] <= 1
    assert isinstance(curr.get("pressure"), (int, float))
    assert 800 <= curr["pressure"] <= 1100
    assert isinstance(curr.get("windSpeed"), (int, float))
    assert curr["windSpeed"] >= 0
    assert isinstance(curr.get("windBearing"), int)
    assert 0 <= curr["windBearing"] <= 360
    assert isinstance(curr.get("time"), int)
    # The timestamp should be within three hours of "now"
    assert abs(curr["time"] - int(time.time())) < 3 * 3600

    minute = data["minutely"]
    assert isinstance(minute.get("summary"), str)
    minute_data = minute["data"]
    assert isinstance(minute_data, list)
    assert 0 < len(minute_data) <= 61
    first_minute = minute_data[0]
    assert isinstance(first_minute.get("time"), int)
    assert isinstance(first_minute.get("precipIntensity"), (int, float))
    assert isinstance(first_minute.get("precipProbability"), (int, float))
    assert 0 <= first_minute["precipProbability"] <= 1

    hourly = data["hourly"]
    assert isinstance(hourly.get("summary"), str)
    hour_data = hourly["data"]
    assert isinstance(hour_data, list)
    assert len(hour_data) >= 24
    first_hour = hour_data[0]
    assert isinstance(first_hour.get("time"), int)
    assert isinstance(first_hour.get("temperature"), (int, float))
    assert isinstance(first_hour.get("humidity"), (int, float))
    assert 0 <= first_hour["humidity"] <= 1
    assert isinstance(first_hour.get("windSpeed"), (int, float))

    daily = data["daily"]
    assert isinstance(daily.get("summary"), str)
    day_data = daily["data"]
    assert isinstance(day_data, list)
    assert len(day_data) >= 1
    first_day = day_data[0]
    assert isinstance(first_day.get("time"), int)
    assert isinstance(first_day.get("sunriseTime"), int)
    assert isinstance(first_day.get("sunsetTime"), int)
    assert isinstance(first_day.get("temperatureHigh"), (int, float))
    assert isinstance(first_day.get("temperatureLow"), (int, float))
    assert isinstance(first_day.get("humidity"), (int, float))
    assert 0 <= first_day["humidity"] <= 1


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
@pytest.mark.parametrize(
    "location",
    [
        (45.0, -75.0),
        (10.0, 10.0),
        (47.28, -53.13),
        (28.64, 77.09),
        (-34.92, 138.60),
        (32.73, -117.192),
        (-15.83, -47.90),
        (-33.91, 18.32),
    ],
)
def test_live_s3_forecast_blocks(location):
    client = _get_client()

    lat, lon = location
    response = client.get(f"/forecast/{PW_API}/{lat},{lon}")
    assert response.status_code == 200

    data = response.json()
    assert data["latitude"] == pytest.approx(lat, abs=0.5)
    assert data["longitude"] == pytest.approx(lon, abs=0.5)
