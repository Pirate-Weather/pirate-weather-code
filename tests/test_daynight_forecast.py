import datetime

from dateutil import tz

from API.PirateDayNightText import calculate_half_day_text


def make_hour(dt, **overrides):
    """Creates a default hourly data structure for tests.

    Parameters:
        - dt (datetime.datetime): The datetime object for the hour.
        - **overrides: Keyword arguments to override any of the default values.

    Returns:
        - dict: A dictionary representing an hour of weather data.
    """
    defaults = {
        "time": int(dt.timestamp()),
        "cloudCover": 0.0,
        "windSpeed": 0.0,
        "rainIntensity": 0.0,
        "snowIntensity": 0.0,
        "iceIntensity": 0.0,
        "visibility": 16090,
        "humidity": 0.5,
        "precipIntensityError": float("nan"),
        "precipProbability": 0.0,
        "smoke": 0.0,
        "temperature": 10.0,
        "dewPoint": 10.0,
        "liquidAccumulation": 0.0,
        "snowAccumulation": 0.0,
        "iceAccumulation": 0.0,
        "precipType": "none",
        "cape": -999,
    }
    defaults.update(overrides)
    return defaults


def test_half_day_clear_day_icon_and_summary():
    """Tests that a clear day scenario produces the correct icon and summary."""
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 12, 0, tzinfo=zone)
    hours = [
        make_hour(base + datetime.timedelta(hours=i), cloudCover=0.1) for i in range(3)
    ]

    icon, summary = calculate_half_day_text(
        hours, True, "UTC", hours[0]["time"], mode="hour", icon_set="darksky"
    )

    assert icon == "clear-day"
    # summary should include the word 'clear' in its nested structure
    assert "clear" in str(summary)


def test_all_day_half_day_clear_day_icon_and_summary():
    """Tests that a full day clear day scenario produces the correct icon and summary."""
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 4, 0, tzinfo=zone)
    hours = [
        make_hour(base + datetime.timedelta(hours=i), cloudCover=0.1) for i in range(13)
    ]

    icon, summary = calculate_half_day_text(
        hours, True, "UTC", hours[0]["time"], mode="hour", icon_set="darksky"
    )

    assert icon == "clear-day"
    # summary should include the word 'clear' in its nested structure
    assert "clear" in str(summary)


def test_half_day_cloudy_night_icon_and_summary():
    """Tests that a cloudy night scenario produces the correct icon and summary."""
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 23, 0, tzinfo=zone)
    hours = [
        make_hour(base + datetime.timedelta(hours=i), cloudCover=0.9) for i in range(3)
    ]

    icon, summary = calculate_half_day_text(
        hours, False, "UTC", hours[0]["time"], mode="hour", icon_set="darksky"
    )

    assert icon == "cloudy"
    # summary should mention heavy-clouds (cloudy -> heavy-clouds text)
    assert "heavy-clouds" in str(summary)


def test_precipitation_pirate_possible_day_and_night_icons():
    """Tests a light rain scenario with low POP using the pirate icon set."""
    zone = tz.gettz("UTC")
    base_day = datetime.datetime(2025, 11, 10, 13, 0, tzinfo=zone)
    hour_day = make_hour(
        base_day,
        cloudCover=0.5,
        rainIntensity=0.01,
        liquidAccumulation=1.0,
        precipProbability=0.1,
        precipType="rain",
    )

    icon_day, summary_day = calculate_half_day_text(
        [hour_day], True, "UTC", hour_day["time"], mode="hour", icon_set="pirate"
    )
    assert icon_day == "possible-rain-day"

    base_night = datetime.datetime(2025, 11, 10, 23, 0, tzinfo=zone)
    hour_night = make_hour(
        base_night,
        cloudCover=0.5,
        rainIntensity=0.01,
        liquidAccumulation=1.0,
        precipProbability=0.1,
        precipType="rain",
    )

    icon_night, summary_night = calculate_half_day_text(
        [hour_night], False, "UTC", hour_night["time"], mode="hour", icon_set="pirate"
    )
    assert icon_night == "possible-rain-night"


def test_too_many_hours_returns_unavailable():
    """Tests that providing more than MAX_HOURS returns 'unavailable'."""
    # MAX_HOURS is 15, so we use 16 hours
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 12, 0, tzinfo=zone)
    hours = [make_hour(base + datetime.timedelta(hours=i)) for i in range(16)]

    icon, summary = calculate_half_day_text(
        hours, True, "UTC", hours[0]["time"], mode="hour", icon_set="darksky"
    )

    assert icon == "none"
    assert summary == "unavailable"
