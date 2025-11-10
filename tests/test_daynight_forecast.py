import datetime

from dateutil import tz

from API.PirateDayNightText import calculate_half_day_text


def make_hour(dt, **overrides):
    # default hour structure used by calculate_half_day_text
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
    # Construct 3 hourly records around local noon
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 12, 0, tzinfo=zone)
    hours = [make_hour(base + datetime.timedelta(hours=i), cloudCover=0.1) for i in range(3)]

    icon, summary = calculate_half_day_text(hours, True, "UTC", hours[0]["time"], mode="hour", icon_set="darksky")

    assert icon == "clear-day"
    # summary should include the word 'clear' in its nested structure
    assert "clear" in str(summary)

def test_all_day_half_day_clear_day_icon_and_summary():
    # Construct 3 hourly records around local noon
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 4, 0, tzinfo=zone)
    hours = [make_hour(base + datetime.timedelta(hours=i), cloudCover=0.1) for i in range(13)]

    icon, summary = calculate_half_day_text(hours, True, "UTC", hours[0]["time"], mode="hour", icon_set="darksky")

    assert icon == "clear-day"
    # summary should include the word 'clear' in its nested structure
    assert "clear" in str(summary)


def test_half_day_cloudy_night_icon_and_summary():
    # Construct 3 hourly records around local 23:00 (night)
    zone = tz.gettz("UTC")
    base = datetime.datetime(2025, 11, 10, 23, 0, tzinfo=zone)
    hours = [make_hour(base + datetime.timedelta(hours=i), cloudCover=0.9) for i in range(3)]

    icon, summary = calculate_half_day_text(hours, False, "UTC", hours[0]["time"], mode="hour", icon_set="darksky")

    assert icon == "cloudy"
    # summary should mention heavy-clouds (cloudy -> heavy-clouds text)
    assert "heavy-clouds" in str(summary)


def test_precipitation_pirate_possible_day_and_night_icons():
    # Create a light rain scenario with low POP so it becomes "possible-" and uses pirate day/night icons
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

    icon_day, summary_day = calculate_half_day_text([hour_day], True, "UTC", hour_day["time"], mode="hour", icon_set="pirate")
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

    icon_night, summary_night = calculate_half_day_text([hour_night], False, "UTC", hour_night["time"], mode="hour", icon_set="pirate")
    assert icon_night == "possible-rain-night"
