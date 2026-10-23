"""Hour-dict builder shared by the text-summary tests.

``calculate_text`` and ``calculate_day_text`` (including half-day summaries) read
the same SI-unit hourly keys. This builds one dry, partly cloudy hour that tests
override only where the scenario differs.
"""

# 2021-01-01 00:00 UTC
BASE_TIME = 1609459200
HOUR = 3600


def make_hour(time=BASE_TIME, **overrides):
    """Return an hourly forecast dict (SI units) with ``overrides`` applied."""
    hour = {
        "time": time,
        "precipType": "none",
        "rainIntensity": 0.0,
        "snowIntensity": 0.0,
        "iceIntensity": 0.0,
        "liquidAccumulation": 0.0,
        "snowAccumulation": 0.0,
        "iceAccumulation": 0.0,
        "precipProbability": 0.0,
        "precipIntensityError": 0.0,
        "cloudCover": 0.5,
        "windSpeed": 5.0,
        "temperature": 22.0,
        "humidity": 0.6,
        "visibility": 10000,
        "dewPoint": 18.0,
        "smoke": 0.0,
        "cape": 0,
    }
    hour.update(overrides)
    return hour


def make_hours(count, start=BASE_TIME, **overrides):
    """Return ``count`` consecutive hours from ``start``, all with ``overrides``."""
    return [make_hour(start + i * HOUR, **overrides) for i in range(count)]
