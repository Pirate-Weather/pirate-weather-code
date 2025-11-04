import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from API.PirateDailyText import calculate_day_text


def create_hour_with_snow(snow_accum, snow_error=0, time_offset=0):
    """Create an hour object with snow accumulation.

    Args:
        snow_accum (float): Snow accumulation in mm
        snow_error (float): Precipitation intensity error in mm
        time_offset (int): Time offset in seconds

    Returns:
        dict: Hour object with snow data
    """
    return {
        "time": 1609459200 + time_offset,
        "precipType": "snow",
        "rainIntensity": 0.0,
        "snowIntensity": 1.0,
        "iceIntensity": 0.0,
        "liquidAccumulation": 0.0,
        "snowAccumulation": snow_accum,
        "iceAccumulation": 0.0,
        "precipProbability": 0.8,
        "cloudCover": 0.9,
        "windSpeed": 5.0,
        "temperature": -2.0,
        "humidity": 0.8,
        "visibility": 5000,
        "dewPoint": -4.0,
        "smoke": 0,
        "cape": 0,
        "precipIntensityError": snow_error,
    }


def test_snow_range_with_error_si_units():
    """Test that snow accumulation shows range when error data is available (SI units - cm)."""
    # Create 24 hours with 1mm snow each and 0.5mm error each = 24mm total, 12mm total error
    # This should result in 2.4cm ± 1.2cm = range of 1-4 cm (floor/ceil applied)
    hours = [create_hour_with_snow(1.0, 0.5, i * 3600) for i in range(24)]

    icon, summary = calculate_day_text(
        hours,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
        unit_system="si",
    )

    # The summary should contain a snow sentence with a range
    assert summary is not None
    assert isinstance(summary, list)

    # Recursively search for the snow sentence
    def find_snow_sentence(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, list) and len(item) >= 2:
                    if (
                        item[0] == "centimeters"
                        and isinstance(item[1], list)
                        and item[1][0] == "range"
                    ):
                        return item
                result = find_snow_sentence(item)
                if result:
                    return result
        return None

    snow_sentence = find_snow_sentence(summary)
    assert snow_sentence is not None, f"Snow sentence not found in summary: {summary}"
    assert snow_sentence[0] == "centimeters"
    assert isinstance(snow_sentence[1], list)
    assert snow_sentence[1][0] == "range", (
        f"Expected range format, got: {snow_sentence[1]}"
    )

    # Check that we have a valid range (low should be >= 0, high should be > low)
    low = snow_sentence[1][1]
    high = snow_sentence[1][2]
    assert low >= 0
    assert high > low
    print(f"✓ Snow range with error (SI): {low}-{high} cm")


def test_snow_range_without_error_si_units():
    """Test that snow accumulation shows less-than format when error is 0 (SI units - cm)."""
    # Create 24 hours with 1mm snow each and 0.0 error (not missing) = 24mm total, 0mm error
    # With error = 0 (ECMWF/GEFS), should show "< 3 cm" format
    hours = [create_hour_with_snow(1.0, 0.0, i * 3600) for i in range(24)]

    icon, summary = calculate_day_text(
        hours,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
        unit_system="si",
    )

    assert summary is not None
    assert isinstance(summary, list)

    # Recursively search for the less-than snow sentence
    def find_less_than_sentence(obj):
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                if item == "less-than" and i + 1 < len(obj):
                    next_item = obj[i + 1]
                    if (
                        isinstance(next_item, list)
                        and len(next_item) >= 2
                        and next_item[0] == "centimeters"
                    ):
                        return obj[i:]  # Return from "less-than" onwards
                result = find_less_than_sentence(item)
                if result:
                    return result
        return None

    snow_sentence = find_less_than_sentence(summary)
    assert snow_sentence is not None, f"Snow sentence not found in summary: {summary}"

    # Should use less-than format when error is 0
    assert snow_sentence[0] == "less-than", (
        f"Expected less-than format, got: {snow_sentence}"
    )
    assert isinstance(snow_sentence[1], list)
    assert snow_sentence[1][0] == "centimeters"
    assert snow_sentence[1][1] > 0  # Should have a positive upper bound
    print(f"✓ Snow with error=0 (SI): < {snow_sentence[1][1]} cm")


def test_snow_range_with_error_us_units():
    """Test that snow accumulation shows range when error data is available (US units - inches)."""
    # Create 24 hours with 2.54mm snow each and 1.27mm error each = 60.96mm total, 30.48mm error
    # This converts to ~2.4 inches total ± ~1.2 inches error = range with floor/ceil applied
    hours = [create_hour_with_snow(2.54, 1.27, i * 3600) for i in range(24)]

    icon, summary = calculate_day_text(
        hours,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
        unit_system="us",
    )

    assert summary is not None
    assert isinstance(summary, list)

    # Recursively search for the snow sentence with inches
    def find_snow_sentence(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, list) and len(item) >= 2:
                    if (
                        item[0] == "inches"
                        and isinstance(item[1], list)
                        and item[1][0] == "range"
                    ):
                        return item
                result = find_snow_sentence(item)
                if result:
                    return result
        return None

    snow_sentence = find_snow_sentence(summary)
    assert snow_sentence is not None, f"Snow sentence not found in summary: {summary}"
    assert snow_sentence[0] == "inches"
    assert isinstance(snow_sentence[1], list)
    assert snow_sentence[1][0] == "range", (
        f"Expected range format, got: {snow_sentence[1]}"
    )

    low = snow_sentence[1][1]
    high = snow_sentence[1][2]
    assert low >= 0
    assert high > low
    print(f"✓ Snow range with error (US): {low}-{high} inches")


def test_snow_small_accumulation_with_error():
    """Test that small snow accumulation with error still shows appropriate format."""
    # Create 24 hours with 0.5mm snow and 0.3mm error = 12mm total, 7.2mm total error
    # This should result in 1.2cm ± 0.72cm = range of 0-2 cm, which becomes "< 2 cm"
    hours = [create_hour_with_snow(0.5, 0.3, i * 3600) for i in range(24)]

    icon, summary = calculate_day_text(
        hours,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
        unit_system="si",
    )

    assert summary is not None
    assert isinstance(summary, list)

    # Recursively search for either less-than or range format
    def find_snow_sentence(obj):
        if isinstance(obj, list):
            for i, item in enumerate(obj):
                # Check for less-than format
                if item == "less-than" and i + 1 < len(obj):
                    next_item = obj[i + 1]
                    if (
                        isinstance(next_item, list)
                        and len(next_item) >= 2
                        and next_item[0] == "centimeters"
                    ):
                        return obj[i:]
                # Check for range format
                if isinstance(item, list) and len(item) >= 2:
                    if item[0] == "centimeters" and isinstance(item[1], list):
                        return item
                result = find_snow_sentence(item)
                if result:
                    return result
        return None

    snow_sentence = find_snow_sentence(summary)
    assert snow_sentence is not None, f"Snow sentence not found in summary: {summary}"

    # Should use less-than format when lower range is 0
    if snow_sentence[0] == "less-than":
        assert isinstance(snow_sentence[1], list)
        assert snow_sentence[1][0] == "centimeters"
        print(f"✓ Small snow accumulation: < {snow_sentence[1][1]} cm")
    else:
        # Could also be a range if the calculation gives a non-zero lower bound
        assert snow_sentence[0] == "centimeters"
        assert isinstance(snow_sentence[1], list)
        if snow_sentence[1][0] == "range":
            print(
                f"✓ Small snow accumulation: {snow_sentence[1][1]}-{snow_sentence[1][2]} cm"
            )
        else:
            print("✓ Small snow accumulation in correct format")


def test_snow_exact_value_when_error_missing():
    """Test that snow accumulation shows exact value when error is missing (np.nan)."""
    import numpy as np

    # Create 24 hours with 1mm snow each but error is np.nan (ERA5 data)
    # With missing error data, should show exact value "3 cm" (not range or <)
    hours = [create_hour_with_snow(1.0, np.nan, i * 3600) for i in range(24)]

    icon, summary = calculate_day_text(
        hours,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
        unit_system="si",
    )

    assert summary is not None
    assert isinstance(summary, list)

    # Recursively search for the snow sentence with exact value
    def find_exact_value_sentence(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, list) and len(item) >= 2:
                    # Look for ["centimeters", <number>] (exact value, not range or less-than)
                    if (
                        item[0] == "centimeters"
                        and isinstance(item[1], (int, float))
                        and not isinstance(item[1], list)
                    ):
                        return item
                result = find_exact_value_sentence(item)
                if result:
                    return result
        return None

    snow_sentence = find_exact_value_sentence(summary)
    assert snow_sentence is not None, f"Snow sentence not found in summary: {summary}"

    # Should use exact value format when error is missing (np.nan)
    assert snow_sentence[0] == "centimeters", (
        f"Expected centimeters format, got: {snow_sentence}"
    )
    assert isinstance(snow_sentence[1], (int, float))
    assert not isinstance(snow_sentence[1], list), (
        f"Expected exact value, not range or less-than: {snow_sentence}"
    )
    assert snow_sentence[1] > 0  # Should have a positive value
    print(f"✓ Snow with missing error (SI): {snow_sentence[1]} cm (exact)")
