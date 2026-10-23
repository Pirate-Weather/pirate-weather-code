"""Summary windows: the 6am-6am daily summary and the minutely hour."""

from API.PirateDailyText import calculate_day_text
from API.PirateMinutelyText import calculate_minutely_text
from tests.helpers.text import BASE_TIME, HOUR, make_hours

RAIN = {
    "precipType": "rain",
    "rainIntensity": 1.0,
    "liquidAccumulation": 1.0,
    "precipProbability": 0.9,
    "cloudCover": 0.9,
}


def test_daily_summary_night_runs_until_6am():
    """Daily summaries cover 6am to 6am, so rain at 4-6am the next morning must
    be described as overnight. Daily periods have no today-/tomorrow- prefix,
    so with a 4am morning those hours would be merged into the day's morning.
    """
    start = BASE_TIME + 6 * HOUR
    hours = make_hours(22, start, cloudCover=0.9) + make_hours(
        2, start + 22 * HOUR, **RAIN
    )

    assert calculate_day_text(hours, True, "UTC", "day") == (
        "rain",
        ["sentence", ["during", "light-rain", "night"]],
    )


def test_minutely_summary_without_precipitation():
    """A dry hour must say there is no precipitation (keeping the current
    icon) instead of repeating the current conditions as the minutely summary.
    """
    minutes = [
        {"time": BASE_TIME + 60 * i, "precipIntensity": 0.0, "precipType": "none"}
        for i in range(61)
    ]

    assert calculate_minutely_text(minutes, "partly-cloudy-day", "darksky") == (
        ["for-hour", "no-precipitation"],
        "partly-cloudy-day",
    )
