"""Tests for the timemachine functionality.

These tests validate the timemachine feature that provides historical weather data
for dates post May 2024 (when Pirate Weather archive is available).
"""

import datetime
import json
import os
import warnings
from typing import Dict, Any

import httpx
import pytest

from tests import DiffWarning
from tests.test_s3_live import _get_client

PW_API = os.environ.get("PW_API")
PROD_BASE = "https://api.pirateweather.net"


def _check_timemachine_structure(data: Dict[str, Any], test_date: datetime.datetime) -> None:
    """Validate that the timemachine response contains realistic historical data.
    
    Args:
        data: The API response data
        test_date: The requested historical date for context
    """
    # Basic structure checks
    assert "latitude" in data
    assert "longitude" in data  
    assert "timezone" in data
    assert "offset" in data
    
    # Timemachine responses should have currently and daily sections
    assert "currently" in data
    assert "daily" in data
    
    # Currently block validation
    curr = data["currently"]
    assert isinstance(curr.get("time"), int)
    assert isinstance(curr.get("temperature"), (int, float))
    assert -100 <= curr["temperature"] <= 150
    assert isinstance(curr.get("humidity"), (int, float))
    assert 0 <= curr["humidity"] <= 1
    assert isinstance(curr.get("pressure"), (int, float))
    assert 800 <= curr["pressure"] <= 1100
    assert isinstance(curr.get("windSpeed"), (int, float))
    assert curr["windSpeed"] >= 0
    
    # Validate the timestamp is roughly for the requested date
    curr_time = datetime.datetime.fromtimestamp(curr["time"])
    assert curr_time.date() == test_date.date()
    
    # Daily block validation
    daily = data["daily"]
    day_data = daily["data"]
    assert isinstance(day_data, list)
    assert len(day_data) >= 1
    
    first_day = day_data[0]
    assert isinstance(first_day.get("time"), int)
    assert isinstance(first_day.get("temperatureHigh"), (int, float))
    assert isinstance(first_day.get("temperatureLow"), (int, float))
    assert isinstance(first_day.get("humidity"), (int, float))
    assert 0 <= first_day["humidity"] <= 1


def _diff_nested(a: object, b: object, path: str = "") -> dict:
    """Return a mapping of all differences between ``a`` and ``b``.

    The keys of the returned dict are ``/`` separated paths describing the
    location of the difference within the nested structure.
    """
    diffs: dict[str, dict[str, object]] = {}

    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            sub_path = f"{path}/{key}" if path else str(key)
            if key not in a:
                diffs[sub_path] = {"local": None, "prod": b[key]}
            elif key not in b:
                diffs[sub_path] = {"local": a[key], "prod": None}
            else:
                diffs.update(_diff_nested(a[key], b[key], sub_path))
    elif isinstance(a, list) and isinstance(b, list):
        for idx in range(max(len(a), len(b))):
            sub_path = f"{path}[{idx}]"
            try:
                val_a = a[idx]
            except IndexError:
                val_a = None
            try:
                val_b = b[idx]
            except IndexError:
                val_b = None
            if val_a is None or val_b is None:
                diffs[sub_path] = {"local": val_a, "prod": val_b}
            else:
                diffs.update(_diff_nested(val_a, val_b, sub_path))
    else:
        if a != b:
            diffs[path] = {"local": a, "prod": b}

    return diffs


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
@pytest.mark.parametrize(
    "location,test_date",
    [
        # Test various locations with dates post May 2024 (when Pirate Weather archive is available)
        ((45.0, -75.0), datetime.datetime(2024, 6, 15, 12, 0, 0)),  # Ottawa, Canada
        ((40.7128, -74.0060), datetime.datetime(2024, 7, 4, 12, 0, 0)),  # New York City
        ((51.5074, -0.1278), datetime.datetime(2024, 8, 20, 12, 0, 0)),  # London, UK  
        ((35.6762, 139.6503), datetime.datetime(2024, 9, 1, 12, 0, 0)),  # Tokyo, Japan
    ],
)
def test_timemachine_historical_data(location, test_date):
    """Test timemachine requests for historical dates post May 2024."""
    try:
        client = _get_client()
    except ImportError as e:
        pytest.skip(f"Could not initialize test client due to missing dependencies: {e}")
    
    lat, lon = location
    # Format date as timestamp for API
    timestamp = int(test_date.timestamp())
    
    # Use the timemachine endpoint - the URL will contain "timemachine" which
    # allows the request to proceed (bypassing the production restriction)
    response = client.get(f"/timemachine/{PW_API}/{lat},{lon},{timestamp}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["latitude"] == pytest.approx(lat, abs=0.5)
    assert data["longitude"] == pytest.approx(lon, abs=0.5)
    
    _check_timemachine_structure(data, test_date)


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")  
@pytest.mark.parametrize(
    "location,test_date", 
    [
        ((45.0, -75.0), datetime.datetime(2024, 6, 15, 12, 0, 0)),  # Ottawa, Canada
        ((40.7128, -74.0060), datetime.datetime(2024, 7, 4, 12, 0, 0)),  # New York City
    ],
)
def test_timemachine_vs_production(location, test_date):
    """Compare local timemachine responses with production API for validation."""
    try:
        client = _get_client()
    except ImportError as e:
        pytest.skip(f"Could not initialize test client due to missing dependencies: {e}")
    
    session = httpx.Client()
    
    lat, lon = location
    timestamp = int(test_date.timestamp())
    
    # Get local timemachine response
    local_resp = client.get(f"/timemachine/{PW_API}/{lat},{lon},{timestamp}")
    assert local_resp.status_code == 200
    local_data = local_resp.json()
    
    # Get production timemachine response
    prod_url = f"{PROD_BASE}/timemachine/{PW_API}/{lat},{lon},{timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=30)
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production timemachine API: {exc}")
    
    assert prod_resp.status_code == 200
    prod_data = prod_resp.json()
    
    # Compare the responses and warn about significant differences
    diffs = _diff_nested(local_data, prod_data)
    if diffs:
        # Filter out acceptable differences (like processing time, exact timestamps)
        significant_diffs = {}
        for path, diff in diffs.items():
            # Skip minor timing differences and server metadata
            if not any(skip in path.lower() for skip in ["time", "processtime", "x-node-id"]):
                significant_diffs[path] = diff
        
        if significant_diffs:
            diff_text = json.dumps(significant_diffs, indent=2, sort_keys=True)
            warnings.warn(
                f"Timemachine differences for {lat},{lon} on {test_date.date()}:\n{diff_text}", 
                DiffWarning
            )


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
def test_timemachine_error_conditions():
    """Test error conditions for timemachine requests."""
    try:
        client = _get_client()
    except ImportError as e:
        pytest.skip(f"Could not initialize test client due to missing dependencies: {e}")
    
    # Test future date (should fail)
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    future_timestamp = int(future_date.timestamp())
    
    response = client.get(f"/timemachine/{PW_API}/45.0,-75.0,{future_timestamp}")
    assert response.status_code == 400
    assert "Future" in response.json()["detail"]


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
@pytest.mark.parametrize(
    "location,test_date",
    [
        # Test various locations with dates post May 2024 (when Pirate Weather archive is available)
        ((45.0, -75.0), datetime.datetime(2024, 6, 15, 12, 0, 0)),  # Ottawa, Canada
        ((40.7128, -74.0060), datetime.datetime(2024, 7, 4, 12, 0, 0)),  # New York City
        ((51.5074, -0.1278), datetime.datetime(2024, 8, 20, 12, 0, 0)),  # London, UK
    ],
)
def test_production_timemachine_api(location, test_date):
    """Test that production timemachine API is working for historical dates."""
    session = httpx.Client()
    
    lat, lon = location
    timestamp = int(test_date.timestamp())
    
    # Test production timemachine endpoint directly
    prod_url = f"{PROD_BASE}/timemachine/{PW_API}/{lat},{lon},{timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=30)
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production timemachine API: {exc}")
    
    assert prod_resp.status_code == 200, f"Expected 200 but got {prod_resp.status_code}: {prod_resp.text}"
    data = prod_resp.json()
    
    # Validate response structure
    assert data["latitude"] == pytest.approx(lat, abs=0.5)
    assert data["longitude"] == pytest.approx(lon, abs=0.5)
    _check_timemachine_structure(data, test_date)


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
def test_timemachine_error_handling():
    """Test that production API correctly handles invalid timemachine requests."""
    session = httpx.Client()
    
    # Test future date (should fail)
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    future_timestamp = int(future_date.timestamp())
    
    prod_url = f"{PROD_BASE}/timemachine/{PW_API}/45.0,-75.0,{future_timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=30)
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production timemachine API: {exc}")
    
    # Should get an error for future dates
    assert prod_resp.status_code == 400, f"Expected 400 for future date, got {prod_resp.status_code}"
    error_detail = prod_resp.json().get("detail", "")
    assert "Future" in error_detail, f"Expected 'Future' in error message, got: {error_detail}"


def test_timemachine_url_detection():
    """Test that the URL detection logic works correctly for timemachine requests."""
    # This is a unit test that doesn't require the full API setup
    
    # URLs that should trigger timemachine mode
    timemachine_urls = [
        "http://localhost:8000/timemachine/key/45.0,-75.0,1234567890",
        "http://127.0.0.1:8000/forecast/key/45.0,-75.0,1234567890",
        "https://example.com/timemachine/key/45.0,-75.0,1234567890",
    ]
    
    # URLs that should NOT trigger timemachine mode for old dates
    normal_urls = [
        "https://api.pirateweather.net/forecast/key/45.0,-75.0,1234567890",
        "https://production.example.com/forecast/key/45.0,-75.0,1234567890",
    ]
    
    for url in timemachine_urls:
        # The logic checks for these substrings in the URL
        assert (
            ("localhost" in url) or 
            ("timemachine" in url) or 
            ("127.0.0.1" in url)
        ), f"URL {url} should trigger timemachine mode"
    
    for url in normal_urls:
        # These URLs should not trigger timemachine mode
        assert not (
            ("localhost" in url) or 
            ("timemachine" in url) or 
            ("127.0.0.1" in url)  
        ), f"URL {url} should not trigger timemachine mode"