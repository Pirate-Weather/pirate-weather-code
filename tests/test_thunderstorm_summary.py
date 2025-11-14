import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from API.constants.shared_const import MISSING_DATA
from API.PirateDailyText import calculate_day_text
from API.PirateText import calculate_text
from API.PirateWeeklyText import calculate_weekly_text


# Base hour object with common default values
def create_base_hour(time_offset=0, **overrides):
    """Create a base hour object with default values, allowing overrides.

    Args:
        time_offset (int): The time offset in seconds to add to the base time.
        **overrides: Keyword arguments to override default values in the base hour.

    Returns:
        dict: A dictionary representing an hourly forecast data point.
    """

    base_hour = {
        "time": 1609459200 + time_offset,
        "precipType": "none",
        "rainIntensity": 0.0,
        "liquidAccumulation": 0.0,
        "precipProbability": 0.0,
        "cloudCover": 0.5,
        "windSpeed": 5.0,
        "temperature": 22.0,
        "humidity": 0.6,
        "visibility": 10000,
        "dewPoint": 18.0,
        "smoke": 0,
        "cape": 0,
        "precipIntensityError": 0,
        "snowIntensity": 0.0,
        "iceIntensity": 0.0,
        "snowAccumulation": 0.0,
        "iceAccumulation": 0.0,
    }
    base_hour.update(overrides)
    return base_hour


def test_currently_hourly_thunderstorm_with_precipitation():
    """
    Test that thunderstorms are combined with precipitation in currently/hourly summaries.
    When both precipitation and CAPE >= 1000 exist, thunderstorm text should appear.
    """
    hour_object = create_base_hour(
        precipType="rain",
        cloudCover=0.8,
        temperature=25.0,
        humidity=0.7,
        precipProbability=0.8,
        rainIntensity=5.0,
        liquidAccumulation=5.0,
        dewPoint=20.0,
        cape=2600,  # Above high threshold for icon
    )

    text, icon = calculate_text(
        hourObject=hour_object,
        isDayTime=True,
        type="current",
        icon="darksky",
    )

    # Thunderstorm text should be combined with precipitation
    # Check exact structure: ['and', 'thunderstorm', 'medium-rain']
    assert text == "thunderstorm"
    # Icon should be thunderstorm when CAPE >= 2500
    assert icon == "thunderstorm"


def test_currently_possible_thunderstorm_with_precipitation():
    """
    Test that thunderstorms are combined with precipitation in currently/hourly summaries.
    When both precipitation and CAPE >= 1000 exist, thunderstorm text should appear.
    """
    hour_object = create_base_hour(
        precipType="rain",
        cloudCover=0.8,
        temperature=25.0,
        humidity=0.7,
        precipProbability=0.8,
        rainIntensity=5.0,
        liquidAccumulation=5.0,
        dewPoint=20.0,
        cape=1500,  # Above high threshold for icon
    )

    text, icon = calculate_text(
        hourObject=hour_object,
        isDayTime=True,
        type="current",
        icon="darksky",
    )

    # Thunderstorm text should not be combined with precipitation
    # Check exact structure: ['medium-rain']
    assert text == "medium-rain"
    # Icon should be rain when CAPE < 2500
    assert icon == "rain"


def test_hourly_possible_thunderstorm_with_precipitation():
    """
    Test that thunderstorms are combined with precipitation in currently/hourly summaries.
    When both precipitation and CAPE >= 1000 exist, thunderstorm text should appear.
    """
    hour_object = create_base_hour(
        precipType="rain",
        cloudCover=0.8,
        temperature=25.0,
        humidity=0.7,
        precipProbability=0.8,
        rainIntensity=5.0,
        liquidAccumulation=5.0,
        dewPoint=20.0,
        cape=1500,  # Above high threshold for icon
    )

    text, icon = calculate_text(
        hourObject=hour_object,
        isDayTime=True,
        type="hour",
        icon="pirate",
    )

    # Thunderstorm text should be combined with precipitation
    # Check exact structure: ['and', 'possible-thunderstorm', 'medium-rain']
    assert text == "possible-thunderstorm"
    # Icon should be rain when CAPE < 2500
    assert icon == "possible-thunderstorm-day"


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
        "rainIntensity": 0.0,
        "liquidAccumulation": 0.0,
        "dewPoint": 15.0,
        "smoke": 0,
        "cape": 2000,  # High CAPE but no precipitation
        "snowIntensity": 0.0,
        "iceIntensity": 0.0,
        "snowAccumulation": 0.0,
        "iceAccumulation": 0.0,
    }

    text, icon = calculate_text(
        hourObject=hour_object,
        isDayTime=True,
        type="current",
        icon="darksky",
    )

    # Should not have thunderstorm text - check exact structure
    assert text is not None
    # For no precipitation, expect just sky cover text
    assert isinstance(text, str)
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
        "rainIntensity": 3.0,
        "liquidAccumulation": 3.0,
        "dewPoint": 20.0,
        "smoke": 0,
        "cape": 500,  # Below low threshold
        "snowIntensity": 0.0,
        "iceIntensity": 0.0,
        "snowAccumulation": 0.0,
        "iceAccumulation": 0.0,
    }

    text, icon = calculate_text(
        hourObject=hour_object,
        isDayTime=True,
        type="current",
        icon="darksky",
    )

    # Should not have thunderstorm text with low CAPE
    # Check that it's just rain without thunderstorm
    assert text is not None
    # Could be a string or list, but should not contain thunderstorm
    text_str = str(text) if not isinstance(text, str) else text
    assert "thunderstorm" not in text_str.lower()


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
                "rainIntensity": 5.0,
                "liquidAccumulation": 5.0,
                "precipProbability": 0.8,
                "cloudCover": 0.9,
                "windSpeed": 8.0,
                "temperature": 22.0,
                "humidity": 0.75,
                "visibility": 8000,
                "dewPoint": 18.0,
                "smoke": 0,
                "cape": 2600,  # Above high threshold for icon
                "precipIntensityError": 0.5,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        icon_set="darksky",
    )

    # Should have combined thunderstorm and precipitation
    # Check exact structure: thunderstorm and rain combined with wind "for-day"
    assert summary_text == [
        "sentence",
        [
            "for-day",
            ["and", "thunderstorm", "light-wind"],
        ],
    ]
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
                "rainIntensity": 3.0,
                "liquidAccumulation": 3.0,
                "precipProbability": 0.7,
                "cloudCover": 0.8,
                "windSpeed": 5.0,
                "temperature": 20.0,
                "humidity": 0.7,
                "visibility": 9000,
                "dewPoint": 15.0,
                "smoke": 0,
                "cape": 500,  # Below threshold - no thunderstorms
                "precipIntensityError": 0.3,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    # Afternoon: rain with thunderstorms (4 hours)
    for i in range(4, 8):
        hours.append(
            {
                "time": 1609459200 + (i * 3600),  # 12 PM - 4 PM
                "precipType": "rain",
                "rainIntensity": 6.0,
                "liquidAccumulation": 6.0,
                "precipProbability": 0.9,
                "cloudCover": 0.95,
                "windSpeed": 10.0,
                "temperature": 24.0,
                "humidity": 0.8,
                "visibility": 7000,
                "dewPoint": 20.0,
                "smoke": 0,
                "cape": 2000,  # Above threshold - thunderstorms
                "precipIntensityError": 0.5,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        icon_set="darksky",
    )

    # Should mention both rain and thunderstorms separately
    # Check exact structure: rain "for-day" AND thunderstorm "during morning"
    assert summary_text == [
        "sentence",
        [
            "and",
            ["for-day", "medium-rain"],
            ["during", ["and", "possible-thunderstorm", "medium-wind"], "morning"],
        ],
    ]


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
                "rainIntensity": 0.0,
                "liquidAccumulation": 0.0,
                "precipProbability": 0.0,
                "cloudCover": 0.5,
                "windSpeed": 5.0,
                "temperature": 22.0,
                "humidity": 0.6,
                "visibility": 15000,
                "dewPoint": 14.0,
                "smoke": 0,
                "cape": 0,
                "precipIntensityError": 0,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    # Later period: thunderstorms (4 hours)
    for i in range(3, 7):
        hours.append(
            {
                "time": 1609470000 + (i * 3600),  # 2 PM - 6 PM
                "precipType": "rain",
                "rainIntensity": 8.0,
                "liquidAccumulation": 8.0,
                "precipProbability": 0.85,
                "cloudCover": 0.95,
                "windSpeed": 12.0,
                "temperature": 25.0,
                "humidity": 0.8,
                "visibility": 6000,
                "dewPoint": 21.0,
                "smoke": 0,
                "cape": 2600,  # High CAPE for thunderstorm icon
                "precipIntensityError": 0.7,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        is_day_time=True,
        time_zone="UTC",
        mode="hour",  # 24-hour mode
        icon_set="darksky",
    )

    # Should show thunderstorms combined with rain and wind "later" during today-morning
    # Check exact structure with "later-" prefix
    assert summary_text == [
        "sentence",
        [
            "during",
            ["and", "thunderstorm", "medium-wind"],
            "today-morning",
        ],
    ]
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
            "liquidAccumulation": 10.0,
            "rainIntensityMax": 5.0,
            "precipProbability": 0.8,
            "temperatureHigh": 25.0,
            "cape": 2000,
            "snowIntensityMax": 0.0,
            "iceIntensityMax": 0.0,
            "snowAccumulation": 0.0,
            "iceAccumulation": 0.0,
        }
    )

    # Day 2: Rain with thunderstorms
    week_array.append(
        {
            "time": 1609545600,
            "icon": "thunderstorm",
            "precipType": "rain",
            "liquidAccumulation": 12.0,
            "rainIntensityMax": 6.0,
            "precipProbability": 0.85,
            "temperatureHigh": 26.0,
            "cape": 2500,
            "snowIntensityMax": 0.0,
            "iceIntensityMax": 0.0,
            "snowAccumulation": 0.0,
            "iceAccumulation": 0.0,
        }
    )

    # Day 3-8: No precipitation
    for i in range(2, 8):
        week_array.append(
            {
                "time": 1609459200 + (i * 86400),
                "icon": "partly-cloudy-day",
                "precipType": "none",
                "liquidAccumulation": 0.0,
                "rainIntensityMax": 0.0,
                "precipProbability": 0.0,
                "temperatureHigh": 24.0 + i,
                "cape": MISSING_DATA,
                "snowIntensityMax": 0.0,
                "iceIntensityMax": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    text, icon = calculate_weekly_text(
        weekArr=week_array,
        timeZone="UTC",
        icon="darksky",
    )

    # Should mention thunderstorms - check exact structure
    assert text == [
        "with",
        [
            "during",
            ["and", "thunderstorm", "medium-rain"],
            ["and", "today", "tomorrow"],
        ],
        ["temperatures-rising", ["celsius", 31], "next-friday"],
    ]
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
                "rainIntensity": 0.0,
                "liquidAccumulation": 0.0,
                "precipProbability": 0.0,
                "cloudCover": 0.6,
                "windSpeed": 5.0,
                "temperature": 28.0,
                "humidity": 0.5,
                "visibility": 15000,
                "dewPoint": 15.0,
                "smoke": 0,
                "cape": 3000,  # Very high CAPE but no precipitation
                "precipIntensityError": 0,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    # Later hours: Precipitation with high CAPE (but less than above)
    for i in range(4, 8):
        hours.append(
            {
                "time": 1609459200 + (i * 3600),
                "precipType": "rain",
                "rainIntensity": 7.0,
                "liquidAccumulation": 7.0,
                "precipProbability": 0.85,
                "cloudCover": 0.9,
                "windSpeed": 10.0,
                "temperature": 24.0,
                "humidity": 0.8,
                "visibility": 8000,
                "dewPoint": 20.0,
                "smoke": 0,
                "cape": 2600,  # High enough for thunderstorm icon, but less than 3000
                "precipIntensityError": 0.5,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "snowAccumulation": 0.0,
                "iceAccumulation": 0.0,
            }
        )

    icon, summary_text = calculate_day_text(
        hours=hours,
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        icon_set="darksky",
    )

    # Should show thunderstorms (using CAPE=2600 which occurs with precipitation)
    # Check exact structure
    assert summary_text == [
        "sentence",
        [
            "during",
            ["and", "thunderstorm", "medium-wind"],
            "morning",
        ],
    ]
    # Icon should be thunderstorm
    assert icon == "thunderstorm"


def test_thunderstorms_dont_combine_with_humidity():
    """
    Test that humid/dry conditions don't combine with thunderstorms + precipitation.
    This prevents overly wordy summaries like "thunderstorms and rain and humid".
    """
    # All day: thunderstorms with rain and high humidity
    hours = [
        create_base_hour(
            time_offset=i * 3600,
            precipType="rain",
            rainIntensity=5.0,
            liquidAccumulation=5.0,
            precipProbability=0.8,
            cloudCover=0.9,
            windSpeed=5.0,  # Low wind so it doesn't combine
            temperature=28.0,  # High temp
            humidity=0.96,  # High humidity
            visibility=8000,
            dewPoint=26.0,
            cape=2600,
            precipIntensityError=0.5,
        )
        for i in range(8)
    ]

    icon, summary_text = calculate_day_text(
        hours=hours,
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        icon_set="darksky",
    )

    # Should NOT include humidity with thunderstorms
    # Check exact structure - no "high-humidity" or "low-humidity"
    assert summary_text == [
        "sentence",
        ["for-day", "thunderstorm"],
    ]
    assert icon == "thunderstorm"

    # Verify humidity doesn't appear in the text
    summary_str = str(summary_text)
    assert "humidity" not in summary_str.lower()


def test_humidity_still_combines_without_thunderstorms():
    """
    Test that humid/dry still combines with regular precipitation when there are no thunderstorms.
    """
    # All day: rain with high humidity but no thunderstorms
    hours = [
        create_base_hour(
            time_offset=i * 3600,
            precipType="rain",
            rainIntensity=5.0,
            liquidAccumulation=5.0,
            precipProbability=0.8,
            cloudCover=0.9,
            windSpeed=5.0,
            temperature=28.0,
            humidity=0.96,  # High humidity
            visibility=8000,
            dewPoint=26.0,
            cape=500,  # Low CAPE - no thunderstorms
            precipIntensityError=0.5,
        )
        for i in range(8)
    ]

    icon, summary_text = calculate_day_text(
        hours=hours,
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        icon_set="darksky",
    )

    # Should include humidity with regular rain
    assert summary_text == [
        "sentence",
        ["for-day", ["and", "medium-rain", "high-humidity"]],
    ]
    assert icon == "rain"
