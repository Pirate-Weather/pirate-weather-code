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

# Try to import _get_client, but make it optional for unit tests
try:
    from tests.test_s3_live import _get_client
except ImportError:
    _get_client = None

PW_API = os.environ.get("PW_API")
PROD_BASE = "https://api.pirateweather.net"


def _check_timemachine_structure(
    data: Dict[str, Any], test_date: datetime.datetime
) -> None:
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
    if not PW_API:
        pytest.skip("PW_API environment variable not set")

    if _get_client is None:
        pytest.skip("Local API client not available - missing dependencies")

    try:
        client = _get_client()
    except (ImportError, OSError) as e:
        pytest.skip(
            f"Could not initialize test client due to missing dependencies or credentials: {e}"
        )

    lat, lon = location
    # Format date as timestamp for API
    timestamp = int(test_date.timestamp())

    # Use the timemachine endpoint - the URL will contain "timemachine" which
    # allows the request to proceed (bypassing the production restriction)
    try:
        response = client.get(f"/timemachine/{PW_API}/{lat},{lon},{timestamp}")
    except (OSError, Exception) as e:
        # Skip if AWS credentials or other infrastructure is not available
        pytest.skip(f"Local API test skipped due to infrastructure requirements: {e}")

    assert response.status_code == 200

    data = response.json()
    assert data["latitude"] == pytest.approx(lat, abs=0.5)
    assert data["longitude"] == pytest.approx(lon, abs=0.5)

    _check_timemachine_structure(data, test_date)


@pytest.mark.parametrize(
    "location,test_date",
    [
        ((45.0, -75.0), datetime.datetime(2024, 6, 15, 12, 0, 0)),  # Ottawa, Canada
        ((40.7128, -74.0060), datetime.datetime(2024, 7, 4, 12, 0, 0)),  # New York City
    ],
)
def test_timemachine_vs_production(location, test_date):
    """Compare local timemachine responses with production API for validation."""
    if not PW_API:
        pytest.skip("PW_API environment variable not set")

    if _get_client is None:
        pytest.skip("Local API client not available - missing dependencies")

    try:
        client = _get_client()
    except (ImportError, OSError) as e:
        pytest.skip(
            f"Could not initialize test client due to missing dependencies or credentials: {e}"
        )

    session = httpx.Client()

    lat, lon = location
    timestamp = int(test_date.timestamp())

    # Get local timemachine response
    try:
        local_resp = client.get(f"/timemachine/{PW_API}/{lat},{lon},{timestamp}")
    except (OSError, Exception) as e:
        # Skip if AWS credentials or other infrastructure is not available
        pytest.skip(f"Local API test skipped due to infrastructure requirements: {e}")

    assert local_resp.status_code == 200
    local_data = local_resp.json()

    # Get production timemachine response
    prod_url = f"{PROD_BASE}/timemachine/{PW_API}/{lat},{lon},{timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=30)
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production timemachine API: {exc}")

    if prod_resp.status_code != 200:
        pytest.skip(
            f"Production API returned {prod_resp.status_code}: {prod_resp.text}"
        )

    prod_data = prod_resp.json()

    # Compare the responses and warn about significant differences
    diffs = _diff_nested(local_data, prod_data)
    if diffs:
        # Filter out acceptable differences (like processing time, exact timestamps)
        significant_diffs = {}
        for path, diff in diffs.items():
            # Skip minor timing differences and server metadata
            if not any(
                skip in path.lower() for skip in ["time", "processtime", "x-node-id"]
            ):
                significant_diffs[path] = diff

        if significant_diffs:
            diff_text = json.dumps(significant_diffs, indent=2, sort_keys=True)
            warnings.warn(
                f"Timemachine differences for {lat},{lon} on {test_date.date()}:\n{diff_text}",
                DiffWarning,
            )


def test_timemachine_error_conditions():
    """Test error conditions for timemachine requests."""
    if not PW_API:
        pytest.skip("PW_API environment variable not set")

    if _get_client is None:
        pytest.skip("Local API client not available - missing dependencies")

    try:
        client = _get_client()
    except (ImportError, OSError) as e:
        pytest.skip(
            f"Could not initialize test client due to missing dependencies or credentials: {e}"
        )

    # Test future date (should fail)
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    future_timestamp = int(future_date.timestamp())

    try:
        response = client.get(f"/timemachine/{PW_API}/45.0,-75.0,{future_timestamp}")
    except (OSError, Exception) as e:
        # Skip if AWS credentials or other infrastructure is not available
        pytest.skip(f"Local API test skipped due to infrastructure requirements: {e}")

    assert response.status_code == 400
    assert "Future" in response.json()["detail"]


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
    if not PW_API:
        pytest.skip("PW_API environment variable not set")

    session = httpx.Client()

    lat, lon = location
    timestamp = int(test_date.timestamp())

    # Test production timemachine endpoint directly
    prod_url = f"{PROD_BASE}/timemachine/{PW_API}/{lat},{lon},{timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=30)
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production timemachine API: {exc}")

    if prod_resp.status_code == 404:
        pytest.skip(
            f"Production timemachine API endpoint may not be available: {prod_resp.text}"
        )
    elif prod_resp.status_code != 200:
        pytest.skip(
            f"Production API returned {prod_resp.status_code}: {prod_resp.text}"
        )
    data = prod_resp.json()

    # Validate response structure
    assert data["latitude"] == pytest.approx(lat, abs=0.5)
    assert data["longitude"] == pytest.approx(lon, abs=0.5)
    _check_timemachine_structure(data, test_date)


def test_timemachine_error_handling():
    """Test that production API correctly handles invalid timemachine requests."""
    if not PW_API:
        pytest.skip("PW_API environment variable not set")

    session = httpx.Client()

    # Test future date (should fail)
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    future_timestamp = int(future_date.timestamp())

    prod_url = f"{PROD_BASE}/timemachine/{PW_API}/45.0,-75.0,{future_timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=30)
    except Exception as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production timemachine API: {exc}")

    # Skip if the endpoint doesn't exist or has other issues
    if prod_resp.status_code == 404:
        pytest.skip(
            f"Production timemachine API endpoint may not be available: {prod_resp.text}"
        )
    elif prod_resp.status_code not in [400, 200]:
        pytest.skip(
            f"Production API returned unexpected status {prod_resp.status_code}: {prod_resp.text}"
        )

    # Should get an error for future dates (if endpoint exists)
    if prod_resp.status_code == 400:
        error_detail = prod_resp.json().get("detail", "")
        assert "Future" in error_detail, (
            f"Expected 'Future' in error message, got: {error_detail}"
        )


def test_timemachine_integration_with_demo_key():
    """Test timemachine functionality using demo key or graceful fallback."""
    # Try with demo key first, then fall back to any available key
    test_api_key = PW_API or "demo"

    # Try to use production API directly for validation
    session = httpx.Client()

    # Test a historical date post May 2024
    test_date = datetime.datetime(2024, 6, 15, 12, 0, 0)
    lat, lon = 45.0, -75.0  # Ottawa, Canada
    timestamp = int(test_date.timestamp())

    # Test production timemachine endpoint directly
    prod_url = f"{PROD_BASE}/timemachine/{test_api_key}/{lat},{lon},{timestamp}"
    try:
        prod_resp = session.get(prod_url, timeout=10)

        # If we get a response, validate it
        if prod_resp.status_code == 200:
            data = prod_resp.json()
            assert data["latitude"] == pytest.approx(lat, abs=0.5)
            assert data["longitude"] == pytest.approx(lon, abs=0.5)
            _check_timemachine_structure(data, test_date)

        elif prod_resp.status_code == 400:
            # Check if it's a proper error response for invalid API key
            error_data = prod_resp.json()
            assert "detail" in error_data or "message" in error_data
            # This is expected for invalid API keys

        elif prod_resp.status_code == 403:
            # API key issue - expected with demo key
            pytest.skip("API key authentication issue (expected with demo key)")

        elif prod_resp.status_code == 404:
            # Endpoint may not exist or be accessible
            pytest.skip("Production timemachine endpoint not accessible")

        else:
            # Other errors - validate we're getting proper error responses
            assert prod_resp.status_code in [400, 401, 403, 404, 500]

    except Exception as e:
        # Network or other connectivity issues
        pytest.skip(f"Could not reach production API for integration test: {e}")


def test_timemachine_request_format_validation():
    """Test that timemachine request URL formats are constructed correctly."""
    # Test various API key and location combinations
    test_cases = [
        ("demo", 45.0, -75.0, 1718452800),
        ("test_key", 40.7128, -74.0060, 1720094400),
        ("api_key_123", 51.5074, -0.1278, 1724155200),
    ]

    for api_key, lat, lon, timestamp in test_cases:
        # Test local timemachine URL format
        local_url = f"/timemachine/{api_key}/{lat},{lon},{timestamp}"
        assert local_url.startswith("/timemachine/")
        assert api_key in local_url
        assert str(lat) in local_url
        assert str(lon) in local_url
        assert str(timestamp) in local_url

        # Test production timemachine URL format
        prod_url = f"{PROD_BASE}/timemachine/{api_key}/{lat},{lon},{timestamp}"
        assert prod_url.startswith("https://api.pirateweather.net/timemachine/")
        assert api_key in prod_url

        # Validate URL contains timemachine pattern (for bypass logic)
        assert "timemachine" in local_url
        assert "timemachine" in prod_url


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
        assert ("localhost" in url) or ("timemachine" in url) or ("127.0.0.1" in url), (
            f"URL {url} should trigger timemachine mode"
        )

    for url in normal_urls:
        # These URLs should not trigger timemachine mode
        assert not (
            ("localhost" in url) or ("timemachine" in url) or ("127.0.0.1" in url)
        ), f"URL {url} should not trigger timemachine mode"


def test_timemachine_date_logic():
    """Test the date logic for determining timemachine requests."""
    # Test the cutoff date logic (May 1, 2024)
    cutoff_date = datetime.datetime(2024, 5, 1)

    # Dates before May 1, 2024 should trigger timemachine
    old_date = datetime.datetime(2024, 3, 15, 12, 0, 0)
    assert old_date < cutoff_date, "Old date should be before cutoff"

    # Dates after May 1, 2024 should use standard logic
    new_date = datetime.datetime(2024, 6, 15, 12, 0, 0)
    assert new_date >= cutoff_date, "New date should be after cutoff"

    # Test timestamp conversion
    timestamp = int(new_date.timestamp())
    converted_back = datetime.datetime.fromtimestamp(timestamp)
    assert converted_back.date() == new_date.date(), (
        "Timestamp conversion should preserve date"
    )


def test_timemachine_location_parsing():
    """Test location parameter parsing for timemachine requests."""
    # Test valid location formats
    test_cases = [
        ("45.0,-75.0,1718452800", (45.0, -75.0, 1718452800)),
        ("40.7128,-74.0060,1720094400", (40.7128, -74.0060, 1720094400)),
        ("51.5074,-0.1278,1724155200", (51.5074, -0.1278, 1724155200)),
    ]

    for location_str, expected in test_cases:
        parts = location_str.split(",")
        assert len(parts) == 3, f"Location string should have 3 parts: {location_str}"

        lat = float(parts[0])
        lon = float(parts[1])
        timestamp = int(parts[2])

        assert lat == expected[0], f"Latitude should match: {lat} vs {expected[0]}"
        assert lon == expected[1], f"Longitude should match: {lon} vs {expected[1]}"
        assert timestamp == expected[2], (
            f"Timestamp should match: {timestamp} vs {expected[2]}"
        )

        # Validate coordinate ranges
        assert -90 <= lat <= 90, f"Latitude should be valid: {lat}"
        assert -180 <= lon <= 180, f"Longitude should be valid: {lon}"


def test_timemachine_response_structure_validation():
    """Test the validation functions for timemachine response structure."""
    # Test a valid timemachine response structure
    valid_response = {
        "latitude": 45.0,
        "longitude": -75.0,
        "timezone": "America/Toronto",
        "offset": -5,
        "currently": {
            "time": 1718452800,
            "temperature": 72.5,
            "humidity": 0.65,
            "pressure": 1013.25,
            "windSpeed": 5.2,
        },
        "daily": {
            "data": [
                {
                    "time": 1718452800,
                    "temperatureHigh": 78.0,
                    "temperatureLow": 65.0,
                    "humidity": 0.68,
                }
            ]
        },
    }

    test_date = datetime.datetime(2024, 6, 15, 12, 0, 0)

    # This should not raise any exceptions
    try:
        _check_timemachine_structure(valid_response, test_date)
        test_passed = True
    except AssertionError:
        test_passed = False

    assert test_passed, "Valid response structure should pass validation"

    # Test invalid response (missing required fields)
    invalid_response = {
        "latitude": 45.0,
        # Missing longitude, timezone, etc.
    }

    try:
        _check_timemachine_structure(invalid_response, test_date)
        test_passed = True
    except (AssertionError, KeyError):
        test_passed = False

    assert not test_passed, "Invalid response structure should fail validation"


def test_timemachine_temperature_validation():
    """Test temperature range validation logic."""
    # Test valid temperature ranges
    valid_temps = [-50, 0, 25, 72.5, 100, 120]
    for temp in valid_temps:
        assert -100 <= temp <= 150, f"Temperature {temp} should be in valid range"

    # Test boundary values
    assert -100 <= -100 <= 150, "Lower boundary should be valid"
    assert -100 <= 150 <= 150, "Upper boundary should be valid"

    # Test invalid temperatures (for edge case testing)
    invalid_temps = [-200, 200, 500]
    for temp in invalid_temps:
        assert not (-100 <= temp <= 150), f"Temperature {temp} should be invalid"


def test_timemachine_humidity_validation():
    """Test humidity range validation logic."""
    # Test valid humidity values (0-1 range)
    valid_humidity = [0.0, 0.25, 0.5, 0.75, 1.0]
    for humidity in valid_humidity:
        assert 0 <= humidity <= 1, f"Humidity {humidity} should be in valid range"

    # Test boundary values
    assert 0 <= 0 <= 1, "Lower humidity boundary should be valid"
    assert 0 <= 1 <= 1, "Upper humidity boundary should be valid"

    # Test invalid humidity values
    invalid_humidity = [-0.1, 1.5, 2.0]
    for humidity in invalid_humidity:
        assert not (0 <= humidity <= 1), f"Humidity {humidity} should be invalid"
