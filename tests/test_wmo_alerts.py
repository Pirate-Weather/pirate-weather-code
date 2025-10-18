import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime
import re

from pytz import utc


def parse_wmo_alert(wmo_alert_string):
    """
    Parse a WMO alert string and return a formatted alert dictionary.
    This is extracted from the logic in responseLocal.py for testing purposes.

    Args:
        wmo_alert_string: String in format "event}{description}{area_desc}{effective}{expires}{severity}{URL"

    Returns:
        dict: Formatted alert dictionary
    """
    # Extract alert details
    # Format: event}{description}{area_desc}{effective}{expires}{severity}{URL
    wmo_alertDetails = wmo_alert_string.split("}{")

    # Parse times - WMO times are in ISO format
    alertOnset = datetime.datetime.strptime(
        wmo_alertDetails[3], "%Y-%m-%dT%H:%M:%S%z"
    ).astimezone(utc)
    alertEnd = datetime.datetime.strptime(
        wmo_alertDetails[4], "%Y-%m-%dT%H:%M:%S%z"
    ).astimezone(utc)

    # Format description newlines
    alertDescript = wmo_alertDetails[1]
    # Step 1: Replace double newlines with a single newline
    formatted_text = re.sub(r"(?<!\n)\n(?!\n)", " ", alertDescript)

    # Step 2: Replace remaining single newlines with a space
    formatted_text = re.sub(r"\n\n", "\n", formatted_text)

    wmo_alertDict = {
        "title": wmo_alertDetails[0],
        "regions": [s.lstrip() for s in wmo_alertDetails[2].split(";")],
        "severity": wmo_alertDetails[5],
        "time": int(
            (
                alertOnset - datetime.datetime(1970, 1, 1, 0, 0, 0).astimezone(utc)
            ).total_seconds()
        ),
        "expires": int(
            (
                alertEnd - datetime.datetime(1970, 1, 1, 0, 0, 0).astimezone(utc)
            ).total_seconds()
        ),
        "description": formatted_text,
        "uri": wmo_alertDetails[6],
    }

    return wmo_alertDict


def test_parse_wmo_alert_basic():
    """Test parsing a basic WMO alert string."""
    # Sample alert string matching the format from WMO_Alerts_Local.py
    alert_string = "Heavy Rain Warning}{Heavy rainfall expected in the area. Please take precautions.}{Test Region; Another Region}{2025-01-15T12:00:00+00:00}{2025-01-16T12:00:00+00:00}{Severe}{https://example.com/alert/12345"

    result = parse_wmo_alert(alert_string)

    assert result["title"] == "Heavy Rain Warning"
    assert (
        result["description"]
        == "Heavy rainfall expected in the area. Please take precautions."
    )
    assert result["regions"] == ["Test Region", "Another Region"]
    assert result["severity"] == "Severe"
    assert result["uri"] == "https://example.com/alert/12345"

    # Check that times are converted correctly
    # 2025-01-15T12:00:00+00:00 = 1736942400 seconds since epoch
    assert result["time"] == 1736942400
    # 2025-01-16T12:00:00+00:00 = 1737028800 seconds since epoch
    assert result["expires"] == 1737028800


def test_parse_wmo_alert_with_newlines():
    """Test parsing a WMO alert with newlines in the description."""
    alert_string = "Thunderstorm Alert}{Severe thunderstorms are expected.\n\nPlease stay indoors.}{City Center}{2025-02-01T18:00:00+00:00}{2025-02-02T06:00:00+00:00}{Moderate}{https://example.com/alert/67890"

    result = parse_wmo_alert(alert_string)

    # Check that double newlines are converted to single newlines
    assert "\n\n" not in result["description"]
    assert (
        "Severe thunderstorms are expected.\nPlease stay indoors."
        in result["description"]
    )


def test_parse_wmo_alert_single_region():
    """Test parsing a WMO alert with a single region."""
    alert_string = "Wind Advisory}{Strong winds expected.}{Downtown Area}{2025-03-10T09:00:00+00:00}{2025-03-10T21:00:00+00:00}{Minor}{https://example.com/alert/11111"

    result = parse_wmo_alert(alert_string)

    assert result["regions"] == ["Downtown Area"]
    assert len(result["regions"]) == 1


def test_parse_multiple_wmo_alerts():
    """Test parsing multiple WMO alerts separated by pipe character."""
    alert_data = "Heavy Rain Warning}{Heavy rainfall expected.}{Region A}{2025-01-15T12:00:00+00:00}{2025-01-16T12:00:00+00:00}{Severe}{https://example.com/alert/1|Wind Advisory}{Strong winds.}{Region B}{2025-03-10T09:00:00+00:00}{2025-03-10T21:00:00+00:00}{Minor}{https://example.com/alert/2"

    alerts = alert_data.split("|")
    assert len(alerts) == 2

    # Parse first alert
    result1 = parse_wmo_alert(alerts[0])
    assert result1["title"] == "Heavy Rain Warning"
    assert result1["severity"] == "Severe"

    # Parse second alert
    result2 = parse_wmo_alert(alerts[1])
    assert result2["title"] == "Wind Advisory"
    assert result2["severity"] == "Minor"
