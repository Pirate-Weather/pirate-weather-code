"""Summary text windows, period names and fallbacks."""

from unittest.mock import MagicMock

import pytest

from API.legacy.summary import build_minutely_summary
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
    so a night ending before 6am would merge those hours into the day's morning.
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


@pytest.mark.parametrize(
    ("start_hour", "rain_offset", "expected_period"),
    [
        pytest.param(20, 5, "today-night", id="evening-start-tonight"),
        pytest.param(2, 22, "tomorrow-night", id="night-start-next-night"),
    ],
)
def test_next_24_hours_night_names(start_hour, rain_offset, expected_period):
    """The night after midnight belongs to the evening before it: from an
    evening start it is "tonight", but from a start during the night the
    following night is "tomorrow night". Getting this wrong names the wrong night.
    """
    start = BASE_TIME + start_hour * HOUR
    hours = make_hours(24, start, cloudCover=0.9)
    for hour in hours[rain_offset : rain_offset + 2]:
        hour.update(RAIN)

    assert calculate_day_text(hours, True, "UTC", "hour") == (
        "rain",
        ["sentence", ["during", "light-rain", expected_period]],
    )


def test_minutely_stop_and_return_times():
    """Minutely times must agree: rain in minutes 0-19 that returns at minute 41
    stops in 20 minutes and returns 21 minutes later (not 22).
    """
    minutes = [
        {
            "time": BASE_TIME + 60 * i,
            "precipIntensity": 1.0 if i < 20 or i >= 41 else 0.0,
            "precipType": "rain",
        }
        for i in range(61)
    ]

    text, _ = calculate_minutely_text(minutes, "rain", "darksky")

    assert text == [
        "stopping-then-starting-later",
        "light-rain",
        ["minutes", 20],
        ["minutes", 21],
    ]


def test_minutely_summary_falls_back_when_text_fails():
    """If minutely text generation fails, the summary must fall back to the most
    common precipitation text instead of raising and failing the request.
    """
    translation = MagicMock()
    translation.translate.side_effect = ValueError("translation failed")

    assert build_minutely_summary(
        summary_text=True,
        translation=translation,
        minute_items_si=[
            {"time": BASE_TIME, "precipIntensity": 0.0, "precipType": "none"}
        ],
        current_icon="cloudy",
        icon="darksky",
        max_p_chance=[1, 1, 0],
        p_types_text=["None", "Rain"],
        p_types_icon=["none", "rain"],
        logger=MagicMock(),
        loc_tag="test",
    ) == ("Rain", "rain")
