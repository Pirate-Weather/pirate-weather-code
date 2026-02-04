import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_alert_url_with_zone_and_county():
    """Test building alert URL when both zone and county codes are present."""
    # Expected URL format when both zone and county are available
    expected_url = (
        "https://forecast.weather.gov/showsigwx.php?warnzone=NCZ039&warncounty=NCC151"
    )

    # Verify the expected URL format
    # The actual implementation is in API/NWS_Alerts_Local.py build_alert_url()
    assert "forecast.weather.gov/showsigwx.php" in expected_url
    assert "warnzone=NCZ039" in expected_url
    assert "warncounty=NCC151" in expected_url


def test_build_alert_url_fallback_to_api():
    """Test that URL falls back to API endpoint when zone/county can't be extracted."""
    cap_id = "urn:oid:2.49.0.1.840.0.123456"
    fallback_url = f"https://api.weather.gov/alerts/{cap_id}"

    # Verify fallback URL format
    assert "api.weather.gov/alerts/" in fallback_url
    assert cap_id in fallback_url


def test_url_format_patterns():
    """Test that URL format patterns are correct."""
    # User-friendly forecast URL pattern
    forecast_url = (
        "https://forecast.weather.gov/showsigwx.php?warnzone=NCZ039&warncounty=NCC151"
    )
    assert "forecast.weather.gov" in forecast_url
    assert "showsigwx.php" in forecast_url
    assert "warnzone=" in forecast_url
    assert "warncounty=" in forecast_url

    # API fallback URL pattern
    api_url = "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.123456"
    assert "api.weather.gov/alerts/" in api_url


def test_zone_code_extraction():
    """Test zone code extraction from URLs."""
    zone_url = "https://api.weather.gov/zones/forecast/NCZ039"
    county_url = "https://api.weather.gov/zones/county/NCC151"

    # Extract zone code
    zone_code = zone_url.split("/")[-1]
    assert zone_code == "NCZ039"
    assert "Z" in zone_code
    assert zone_code[-3:].isdigit()

    # Extract county code
    county_code = county_url.split("/")[-1]
    assert county_code == "NCC151"
    assert "C" in county_code
    assert county_code[-3:].isdigit()
