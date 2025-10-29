import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from API.PirateMinutelyText import calculate_minutely_text


def create_minute_with_rain(intensity=5.0, precip_type="rain"):
    """Helper to create a minute with precipitation."""
    return {
        "precipIntensity": intensity,
        "precipType": precip_type,
    }


def create_minute_no_precip():
    """Helper to create a minute without precipitation."""
    return {
        "precipIntensity": 0.0,
        "precipType": "none",
    }


def test_minutely_thunderstorm_replaces_rain_for_hour():
    """
    Test that when CAPE > 2500 and there's rain for the entire hour,
    the summary shows "Thunderstorms for the hour." instead of rain.
    """
    # Create 61 minutes of rain (minutely data includes minute 0-60)
    minute_arr = [create_minute_with_rain(5.0) for _ in range(61)]

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=2600,  # Above 2500 threshold
    )

    # Should show thunderstorms for the hour
    assert text == ["for-hour", "thunderstorm"]
    assert icon == "thunderstorm"


def test_minutely_thunderstorm_starting_in():
    """
    Test that when CAPE > 2500 and rain starts later,
    the summary shows "Thunderstorms starting in X minutes."
    """
    # No rain for first 15 minutes, then rain until minute 60
    minute_arr = [create_minute_no_precip() for _ in range(15)]
    minute_arr.extend([create_minute_with_rain(5.0) for _ in range(46)])  # 15 + 46 = 61

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=3000,  # Well above threshold
    )

    # Should show thunderstorms starting in 15 minutes
    assert text == ["starting-in", "thunderstorm", ["minutes", 15]]
    assert icon == "thunderstorm"


def test_minutely_thunderstorm_stopping_in():
    """
    Test that when CAPE > 2500 and rain stops during the hour,
    the summary shows "Thunderstorms stopping in X minutes."
    """
    # Rain for first 30 minutes, then stops
    minute_arr = [create_minute_with_rain(5.0) for _ in range(30)]
    minute_arr.extend([create_minute_no_precip() for _ in range(31)])  # 30 + 31 = 61

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="medium-rain",
        currentIcon="rain",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=2800,
    )

    # Should show thunderstorms stopping in 30 minutes (last minute with rain is index 29, so 29+1=30)
    assert text == ["stopping-in", "thunderstorm", ["minutes", 30]]
    assert icon == "thunderstorm"


def test_minutely_normal_rain_when_cape_low():
    """
    Test that when CAPE < 2500, the summary shows normal rain text.
    """
    # Create 61 minutes of rain
    minute_arr = [create_minute_with_rain(5.0) for _ in range(61)]

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=1500,  # Below 2500 threshold
    )

    # Should show normal rain, not thunderstorms
    assert text == ["for-hour", "medium-rain"]
    assert icon == "rain"


def test_minutely_no_thunderstorm_without_precipitation():
    """
    Test that even with high CAPE, if there's no precipitation,
    thunderstorms don't appear in the summary.
    """
    # No precipitation for the entire hour
    minute_arr = [create_minute_no_precip() for _ in range(61)]

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=3000,  # High CAPE but no precipitation
    )

    # Should show current conditions, not thunderstorms
    assert text == ["for-hour", "clear"]
    assert icon == "clear-day"


def test_minutely_thunderstorm_with_snow():
    """
    Test that thunderstorms work with snow as well.
    """
    # Create 61 minutes of snow
    minute_arr = [create_minute_with_rain(3.0, "snow") for _ in range(61)]

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=2600,
    )

    # Should show thunderstorms (replacing snow)
    assert text == ["for-hour", "thunderstorm"]
    assert icon == "thunderstorm"


def test_minutely_thunderstorm_stopping_then_starting():
    """
    Test thunderstorms with a gap in the middle.
    """
    # Rain for 20 minutes, gap for 21 minutes, rain for 20 minutes (total 61)
    minute_arr = [create_minute_with_rain(5.0) for _ in range(20)]
    minute_arr.extend([create_minute_no_precip() for _ in range(21)])
    minute_arr.extend([create_minute_with_rain(5.0) for _ in range(20)])

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="medium-rain",
        currentIcon="rain",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=2700,
    )

    # Should show thunderstorms stopping then starting again
    assert text[0] == "stopping-then-starting-later"
    assert text[1] == "thunderstorm"
    assert icon == "thunderstorm"


def test_minutely_default_cape_zero():
    """
    Test that when no CAPE is provided (default 0), normal precipitation is shown.
    """
    # Create 61 minutes of rain
    minute_arr = [create_minute_with_rain(5.0) for _ in range(61)]

    # Call without maxCAPE parameter (should default to 0)
    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        # maxCAPE not provided, defaults to 0
    )

    # Should show normal rain since CAPE=0 < 2500
    assert text == ["for-hour", "medium-rain"]
    assert icon == "rain"


def test_minutely_thunderstorm_at_exact_threshold():
    """
    Test that CAPE exactly at 2500 threshold triggers thunderstorms.
    """
    # Create 61 minutes of rain
    minute_arr = [create_minute_with_rain(5.0) for _ in range(61)]

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=2500,  # Exactly at threshold
    )

    # Should show thunderstorms
    assert text == ["for-hour", "thunderstorm"]
    assert icon == "thunderstorm"


def test_minutely_thunderstorm_just_below_threshold():
    """
    Test that CAPE just below 2500 doesn't trigger thunderstorms.
    """
    # Create 61 minutes of rain
    minute_arr = [create_minute_with_rain(5.0) for _ in range(61)]

    text, icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
        maxCAPE=2499,  # Just below threshold
    )

    # Should show normal rain, not thunderstorms
    assert text == ["for-hour", "medium-rain"]
    assert icon == "rain"
