import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from API.constants.shared_const import MISSING_DATA
from API.PirateDailyText import calculate_day_text
from API.PirateText import calculate_text
from API.PirateWeeklyText import calculate_weekly_text


def test_currently_hourly_thunderstorm_with_precipitation():
    """
    Test that thunderstorms are combined with precipitation in currently/hourly summaries.
    When both precipitation and CAPE >= 1000 exist, thunderstorm text should appear.
    """
    hour_object = {
        "precipType": "rain",
        "cloudCover": 0.8,
        "windSpeed": 5.0,
        "temperature": 25.0,
        "humidity": 0.7,
        "visibility": 10000,
        "precipProbability": 0.8,
        "precipIntensity": 5.0,
        "precipAccumulation": 5.0,
        "dewPoint": 20.0,
        "smoke": 0,
        "cape": 2600,  # Above high threshold for icon
        "liftedIndex": -5,  # Indicates thunderstorms
    }

    text, icon = calculate_text(
        hourObject=hour_object,
        prepAccumUnit=1.0,
        visUnits=1.0,
        windUnit=1.0,
        tempUnits=1,
        isDayTime=True,
        rainPrep=5.0,
        snowPrep=0.0,
        icePrep=0.0,
        type="current",
        precipIntensity=5.0,
        icon="darksky",
    )

    # Thunderstorm text should be combined with precipitation
    assert text is not None
    assert isinstance(text, list)
    assert "and" in text
    # Thunderstorm should come first
    assert "possible-thunderstorm" in text or "thunderstorm" in text
    # Icon should be thunderstorm when CAPE >= 2500
    assert icon == "thunderstorm"


def test_currently_hourly_no_thunderstorm_without_precipitation():
    """
    Test that thunderstorms don't appear without precipitation, even with high CAPE.
    """
    hour_object = {
        "precipType": "none",
        "cloudCover": 0.3,
        "windSpeed": 5.0,
        "temperature": 25.0,
        "humidity": 0.5,
        "visibility": 10000,
        "precipProbability": 0.0,
        "precipIntensity": 0.0,
        "precipAccumulation": 0.0,
        "dewPoint": 15.0,
        "smoke": 0,
        "cape": 2000,  # High CAPE but no precipitation
        "liftedIndex": -6,
    }

    text, icon = calculate_text(
        hourObject=hour_object,
        prepAccumUnit=1.0,
        visUnits=1.0,
        windUnit=1.0,
        tempUnits=1,
        isDayTime=True,
        rainPrep=0.0,
        snowPrep=0.0,
        icePrep=0.0,
        type="current",
        precipIntensity=0.0,
        icon="darksky",
    )

    # Should not have thunderstorm text
    assert text is not None
    if isinstance(text, list):
        assert "thunderstorm" not in str(text).lower()
    else:
        assert "thunderstorm" not in text.lower()


def test_currently_hourly_no_thunderstorm_low_cape():
    """
    Test that thunderstorms don't appear with precipitation if CAPE < 1000.
    """
    hour_object = {
        "precipType": "rain",
        "cloudCover": 0.8,
        "windSpeed": 5.0,
        "temperature": 25.0,
        "humidity": 0.7,
        "visibility": 10000,
        "precipProbability": 0.8,
        "precipIntensity": 3.0,
        "precipAccumulation": 3.0,
        "dewPoint": 20.0,
        "smoke": 0,
        "cape": 500,  # Below low threshold
        "liftedIndex": 0,
    }

    text, icon = calculate_text(
        hourObject=hour_object,
        prepAccumUnit=1.0,
        visUnits=1.0,
        windUnit=1.0,
        tempUnits=1,
        isDayTime=True,
        rainPrep=3.0,
        snowPrep=0.0,
        icePrep=0.0,
        type="current",
        precipIntensity=3.0,
        icon="darksky",
    )

    # Should not have thunderstorm text
    assert text is not None
    if isinstance(text, list):
        text_str = str(text).lower()
        assert "thunderstorm" not in text_str
    else:
        assert "thunderstorm" not in text.lower()


def test_daily_thunderstorms_joined_with_precipitation():
    """
    Test daily summary where thunderstorms and precipitation occur in the same periods.
    Thunderstorms should be combined with precipitation.
    """
    # Create hourly data for morning with rain and thunderstorms
    hours = []
    for i in range(8):  # 8 hours of rain with thunderstorms
        hours.append(
            {
                "time": 1609459200 + (i * 3600),  # Starting from 8 AM
                "precipType": "rain",
                "precipIntensity": 5.0,
                "precipAccumulation": 5.0,
                "precipProbability": 0.8,
                "cloudCover": 0.9,
                "windSpeed": 8.0,
                "temperature": 22.0,
                "humidity": 0.75,
                "visibility": 8000,
                "dewPoint": 18.0,
                "smoke": 0,
                "cape": 2600,  # Above high threshold for icon
                "liftedIndex": -5,
                "precipIntensityError": 0.5,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
    )

    # Should have combined thunderstorm and precipitation
    assert summary_text is not None
    summary_str = str(summary_text).lower()
    assert "thunderstorm" in summary_str
    assert "rain" in summary_str or "precipitation" in summary_str
    # Icon should be thunderstorm
    assert icon == "thunderstorm"


def test_daily_thunderstorms_not_joined_with_precipitation():
    """
    Test daily summary where rain occurs in morning and thunderstorms in afternoon (different periods).
    They should be shown separately.
    """
    hours = []

    # Morning: rain without thunderstorms (4 hours)
    for i in range(4):
        hours.append(
            {
                "time": 1609459200 + (i * 3600),  # 8 AM - 12 PM
                "precipType": "rain",
                "precipIntensity": 3.0,
                "precipAccumulation": 3.0,
                "precipProbability": 0.7,
                "cloudCover": 0.8,
                "windSpeed": 5.0,
                "temperature": 20.0,
                "humidity": 0.7,
                "visibility": 9000,
                "dewPoint": 15.0,
                "smoke": 0,
                "cape": 500,  # Below threshold - no thunderstorms
                "liftedIndex": 0,
                "precipIntensityError": 0.3,
            }
        )

    # Afternoon: rain with thunderstorms (4 hours)
    for i in range(4, 8):
        hours.append(
            {
                "time": 1609459200 + (i * 3600),  # 12 PM - 4 PM
                "precipType": "rain",
                "precipIntensity": 6.0,
                "precipAccumulation": 6.0,
                "precipProbability": 0.9,
                "cloudCover": 0.95,
                "windSpeed": 10.0,
                "temperature": 24.0,
                "humidity": 0.8,
                "visibility": 7000,
                "dewPoint": 20.0,
                "smoke": 0,
                "cape": 2000,  # Above threshold - thunderstorms
                "liftedIndex": -7,
                "precipIntensityError": 0.5,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
    )

    # Should mention both rain and thunderstorms
    assert summary_text is not None
    summary_str = str(summary_text).lower()
    # Both should be present in the summary
    assert "rain" in summary_str or "precipitation" in summary_str
    # Thunderstorm should be present
    assert "thunderstorm" in summary_str


def test_24hour_thunderstorms_starting_later():
    """
    Test 24-hour summary where it's afternoon and thunderstorms start later.
    Should show "starting" language for thunderstorms.
    """
    hours = []

    # First period: clear/cloudy (3 hours from current time)
    for i in range(3):
        hours.append(
            {
                "time": 1609470000 + (i * 3600),  # Starting from 11 AM
                "precipType": "none",
                "precipIntensity": 0.0,
                "precipAccumulation": 0.0,
                "precipProbability": 0.0,
                "cloudCover": 0.5,
                "windSpeed": 5.0,
                "temperature": 22.0,
                "humidity": 0.6,
                "visibility": 15000,
                "dewPoint": 14.0,
                "smoke": 0,
                "cape": 0,
                "liftedIndex": MISSING_DATA,
                "precipIntensityError": 0,
            }
        )

    # Later period: thunderstorms (4 hours)
    for i in range(3, 7):
        hours.append(
            {
                "time": 1609470000 + (i * 3600),  # 2 PM - 6 PM
                "precipType": "rain",
                "precipIntensity": 8.0,
                "precipAccumulation": 8.0,
                "precipProbability": 0.85,
                "cloudCover": 0.95,
                "windSpeed": 12.0,
                "temperature": 25.0,
                "humidity": 0.8,
                "visibility": 6000,
                "dewPoint": 21.0,
                "smoke": 0,
                "cape": 2600,  # High CAPE for thunderstorm icon
                "liftedIndex": -8,
                "precipIntensityError": 0.7,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609470000,
        mode="hour",  # 24-hour mode
        icon_set="darksky",
    )

    # Should show thunderstorms
    assert summary_text is not None
    summary_str = str(summary_text).lower()
    assert "thunderstorm" in summary_str
    # Icon should be thunderstorm
    assert icon == "thunderstorm"


def test_weekly_thunderstorms():
    """
    Test weekly summary with thunderstorms on multiple days.
    """
    week_array = []

    # Day 1: Rain with thunderstorms
    week_array.append(
        {
            "time": 1609459200,
            "icon": "thunderstorm",
            "precipType": "rain",
            "precipAccumulation": 10.0,
            "precipIntensityMax": 5.0,
            "precipProbability": 0.8,
            "temperatureHigh": 25.0,
            "cape": 2000,
            "liftedIndex": -6,
        }
    )

    # Day 2: Rain with thunderstorms
    week_array.append(
        {
            "time": 1609545600,
            "icon": "thunderstorm",
            "precipType": "rain",
            "precipAccumulation": 12.0,
            "precipIntensityMax": 6.0,
            "precipProbability": 0.85,
            "temperatureHigh": 26.0,
            "cape": 2500,
            "liftedIndex": -7,
        }
    )

    # Day 3-8: No precipitation
    for i in range(2, 8):
        week_array.append(
            {
                "time": 1609459200 + (i * 86400),
                "icon": "partly-cloudy-day",
                "precipType": "none",
                "precipAccumulation": 0.0,
                "precipIntensityMax": 0.0,
                "precipProbability": 0.0,
                "temperatureHigh": 24.0 + i,
                "cape": MISSING_DATA,
                "liftedIndex": MISSING_DATA,
            }
        )

    text, icon = calculate_weekly_text(
        weekArr=week_array,
        intensityUnit=1.0,
        tempUnit=1,
        timeZone="UTC",
        icon="darksky",
    )

    # Should mention thunderstorms
    assert text is not None
    text_str = str(text).lower()
    assert "thunderstorm" in text_str
    # Icon should be thunderstorm since more than half of precipitation days have them
    assert icon == "thunderstorm"


def test_daily_uses_max_cape_with_precipitation():
    """
    Test that daily summary uses max CAPE that occurs with precipitation,
    not overall max CAPE. If max CAPE is 3000 but doesn't occur with precipitation
    but 2000 does, use 2000.
    """
    hours = []

    # First hours: High CAPE but no precipitation
    for i in range(4):
        hours.append(
            {
                "time": 1609459200 + (i * 3600),
                "precipType": "none",
                "precipIntensity": 0.0,
                "precipAccumulation": 0.0,
                "precipProbability": 0.0,
                "cloudCover": 0.6,
                "windSpeed": 5.0,
                "temperature": 28.0,
                "humidity": 0.5,
                "visibility": 15000,
                "dewPoint": 15.0,
                "smoke": 0,
                "cape": 3000,  # Very high CAPE but no precipitation
                "liftedIndex": -8,
                "precipIntensityError": 0,
            }
        )

    # Later hours: Precipitation with high CAPE (but less than above)
    for i in range(4, 8):
        hours.append(
            {
                "time": 1609459200 + (i * 3600),
                "precipType": "rain",
                "precipIntensity": 7.0,
                "precipAccumulation": 7.0,
                "precipProbability": 0.85,
                "cloudCover": 0.9,
                "windSpeed": 10.0,
                "temperature": 24.0,
                "humidity": 0.8,
                "visibility": 8000,
                "dewPoint": 20.0,
                "smoke": 0,
                "cape": 2600,  # High enough for thunderstorm icon, but less than 3000
                "liftedIndex": -5,
                "precipIntensityError": 0.5,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=1609459200,
        mode="daily",
        icon_set="darksky",
    )

    # Should show thunderstorms (using CAPE=2600 which occurs with precipitation)
    assert summary_text is not None
    summary_str = str(summary_text).lower()
    assert "thunderstorm" in summary_str
    # Icon should be thunderstorm
    assert icon == "thunderstorm"
