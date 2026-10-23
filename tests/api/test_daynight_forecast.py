"""Day/night (half-day) summary text from ``calculate_day_text(half_day=True)``."""

import datetime

import pytest

from API.PirateDailyText import MAX_HALF_DAY_HOURS, calculate_day_text
from tests.helpers.text import make_hour, make_hours


def _utc(hour):
    """Unix time for ``hour``:00 UTC on 2025-11-10."""
    return int(datetime.datetime(2025, 11, 10, hour, tzinfo=datetime.UTC).timestamp())


def _half_day(hours, is_day=True, icon_set="darksky"):
    return calculate_day_text(
        hours, is_day, "UTC", mode="hour", icon_set=icon_set, half_day=True
    )


@pytest.mark.parametrize(
    ("start_hour", "count"),
    [
        pytest.param(12, 3, id="short-afternoon"),
        pytest.param(4, 13, id="full-half-day"),
    ],
)
def test_clear_half_day(start_hour, count):
    """A clear half-day must be a bare "clear" sentence: no "for-day" wrapper
    (removed in #385) however many hours the window spans.
    """
    hours = make_hours(count, _utc(start_hour), cloudCover=0.1)

    assert _half_day(hours) == ("clear-day", ["sentence", "clear"])


def test_cloudy_night():
    """Overcast nights must use the cloudy icon and "heavy-clouds" text."""
    hours = make_hours(3, _utc(23), cloudCover=0.9)

    assert _half_day(hours, is_day=False) == ("cloudy", ["sentence", "heavy-clouds"])


@pytest.mark.parametrize(
    ("hour", "is_day", "expected_icon"),
    [
        pytest.param(13, True, "possible-rain-day", id="day"),
        pytest.param(23, False, "possible-rain-night", id="night"),
    ],
)
def test_pirate_possible_rain_icon(hour, is_day, expected_icon):
    """Low-PoP rain must use the pirate set's day/night "possible" icons and
    "possible" wording, not a definite rain icon.
    """
    hours = [
        make_hour(
            _utc(hour),
            precipType="rain",
            rainIntensity=0.01,
            liquidAccumulation=1.0,
            precipProbability=0.1,
        )
    ]

    assert _half_day(hours, is_day=is_day, icon_set="pirate") == (
        expected_icon,
        ["sentence", "possible-very-light-rain"],
    )


def test_more_than_max_hours_is_unavailable():
    """Slices longer than a half-day window must return the plain
    "unavailable" text (#385) instead of a misleading summary; a slice of
    exactly MAX_HALF_DAY_HOURS is still summarised.
    """
    assert _half_day(make_hours(MAX_HALF_DAY_HOURS + 1, _utc(12))) == (
        "none",
        "unavailable",
    )
    assert _half_day(make_hours(MAX_HALF_DAY_HOURS, _utc(12))) == (
        "partly-cloudy-day",
        ["sentence", "light-clouds"],
    )


def test_three_period_precip_summary_stays_valid():
    """A slice spanning three periods (e.g. DST-shifted) must join them into
    one "during" phrase instead of returning None (#624).
    """
    rain = {
        "precipType": "rain",
        "rainIntensity": 1.0,
        "liquidAccumulation": 1.0,
        "precipProbability": 0.8,
        "cloudCover": 0.8,
    }
    hours = [make_hour(_utc(3))] + make_hours(14, _utc(4), **rain)

    assert _half_day(hours) == (
        "rain",
        [
            "sentence",
            [
                "during",
                "light-rain",
                ["and", "today-morning", ["and", "today-afternoon", "today-evening"]],
            ],
        ],
    )
