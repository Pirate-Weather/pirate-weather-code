import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def test_nws_alert_url_transformation():
    """Test that NWS alert URLs are transformed from API to user-friendly format."""
    # Create a sample dataframe similar to what's in NWS_Alerts_Local.py
    test_data = {
        "URL": [
            "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.123456",
            "https://api.weather.gov/alerts/urn:oid:2.49.0.1.840.0.789012",
            "https://api.weather.gov/alerts/some-other-id",
        ]
    }
    df = pd.DataFrame(test_data)

    # Apply the transformation (same as in NWS_Alerts_Local.py)
    df["URL"] = (
        df["URL"]
        .astype(str)
        .str.replace("api.weather.gov", "www.weather.gov", regex=False)
    )

    # Verify the transformation
    assert (
        df["URL"][0] == "https://www.weather.gov/alerts/urn:oid:2.49.0.1.840.0.123456"
    )
    assert (
        df["URL"][1] == "https://www.weather.gov/alerts/urn:oid:2.49.0.1.840.0.789012"
    )
    assert df["URL"][2] == "https://www.weather.gov/alerts/some-other-id"

    # Verify no api.weather.gov remains
    assert not any(df["URL"].str.contains("api.weather.gov"))

    # Verify all contain www.weather.gov
    assert all(df["URL"].str.contains("www.weather.gov"))


def test_nws_alert_url_transformation_handles_edge_cases():
    """Test that URL transformation handles edge cases properly."""
    # Test with various edge cases
    test_data = {
        "URL": [
            "https://api.weather.gov/alerts/test",  # Normal case
            "nan",  # String 'nan'
            "",  # Empty string
            "https://some-other-site.com/alerts/123",  # Different domain
        ]
    }
    df = pd.DataFrame(test_data)

    # Apply the transformation
    df["URL"] = (
        df["URL"]
        .astype(str)
        .str.replace("api.weather.gov", "www.weather.gov", regex=False)
    )

    # Verify the results
    assert df["URL"][0] == "https://www.weather.gov/alerts/test"
    assert df["URL"][1] == "nan"  # Should remain unchanged
    assert df["URL"][2] == ""  # Should remain empty
    assert (
        df["URL"][3] == "https://some-other-site.com/alerts/123"
    )  # Should remain unchanged
