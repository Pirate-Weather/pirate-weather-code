"""Summary text windows, period names and fallbacks."""

import math
from unittest.mock import MagicMock

import pytest

from API.legacy.summary import build_minutely_summary
from API.PirateDailyText import calculate_day_text
from API.PirateMinutelyText import calculate_minutely_text
from API.PirateWeeklyText import calculate_weekly_text
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


def test_weekly_summary_includes_days_with_a_rain_icon():
    """A day whose daily summary shows rain must appear in the weekly summary,
    even with less rain (0.8 mm) than the weekly accumulation threshold, and
    with a 60% chance it is not "possible" rain however little falls.
    """
    week = [
        {
            "time": BASE_TIME + day * 86400,
            "icon": "partly-cloudy-day",
            "precipType": "none",
            "precipProbability": 0.0,
            "liquidAccumulation": 0.0,
            "snowAccumulation": 0.0,
            "iceAccumulation": 0.0,
            "rainIntensityMax": 0.0,
            "snowIntensityMax": 0.0,
            "iceIntensityMax": 0.0,
            "temperatureHigh": 24.0,
        }
        for day in range(8)
    ]
    week[2].update(
        icon="rain",
        precipType="rain",
        precipProbability=0.6,
        liquidAccumulation=0.8,
        rainIntensityMax=0.8,
    )

    text, icon = calculate_weekly_text(week, "UTC")

    # text[2] is the week's temperature trend, which is not under test here.
    assert text[:2] == ["with", ["during", "light-rain", "sunday"]]
    assert icon == "rain"


DRIZZLE = {
    "precipType": "rain",
    "rainIntensity": 0.2,
    "liquidAccumulation": 0.2,
    "precipProbability": 0.9,
    "cloudCover": 0.95,
}


@pytest.mark.parametrize(
    ("wind_hours", "expected"),
    [
        pytest.param(
            range(12, 18),
            [
                "multiple-sentences",
                ["sentence", ["until", "very-light-rain", "evening"]],
                ["sentence", ["during", "light-wind", "evening"]],
            ],
            id="different-periods",
        ),
        pytest.param(
            range(12),
            [
                "sentence",
                ["until", ["and", "very-light-rain", "light-wind"], "evening"],
            ],
            id="same-periods",
        ),
    ],
)
def test_daily_conditions_split_into_sentences(wind_hours, expected):
    """Conditions in different periods must be separate sentences ("Drizzle
    until evening. Breezy in the evening.") rather than one long "and" chain,
    while conditions in exactly the same periods stay joined ("Drizzle and
    breezy until evening.").
    """
    hours = make_hours(24, BASE_TIME + 6 * HOUR, cloudCover=0.95)
    for hour in hours[:12]:
        hour.update(DRIZZLE)
    for i in wind_hours:
        hours[i]["windSpeed"] = 8.0

    assert calculate_day_text(hours, True, "UTC", "day")[1] == expected


@pytest.mark.parametrize(
    ("wet_hours", "amounts", "expected"),
    [
        # One wet hour in every period reaches each period's accumulation
        # threshold, but must not read as rain "throughout the day".
        pytest.param(
            [2, 8, 14, 20],
            None,
            ["for-day", ["occasional", "light-rain"]],
            id="occasional",
        ),
        pytest.param(
            [*range(4), *range(10, 14), *range(19, 23)],
            None,
            ["for-day", ["periods-of", "light-rain"]],
            id="periods-of",
        ),
        # Every period is half wet, but rain every other hour isn't all day.
        pytest.param(
            list(range(0, 24, 2)),
            None,
            ["for-day", ["intermittent", "light-rain"]],
            id="intermittent",
        ),
        # One continuous spell that only clips the morning and overnight.
        pytest.param(
            list(range(5, 20)),
            None,
            ["starting-continuing-until", "light-rain", "afternoon", "night"],
            id="single-spell-trimmed",
        ),
        pytest.param(
            list(range(10)),
            [3, 3, 3, 3, 3, 3, 2.5, 1.5, 0.8, 0.3],
            ["during", ["tapering-off", "medium-rain"], "afternoon"],
            id="tapering-off",
        ),
        # Rain that stops abruptly isn't tapering off just because a later
        # shower, too light to be reported, weakens.
        pytest.param(
            [*range(6), 14, 15, 16],
            [3, 3, 3, 3, 3, 3, 0.08, 0.05, 0.03],
            ["during", "medium-rain", "morning"],
            id="abrupt-stop-then-light-shower",
        ),
    ],
)
def test_daily_precipitation_pattern(wet_hours, amounts, expected):
    """For-day must only be used when precipitation lasts all day; showers
    through the day are occasional, periods of, or intermittent precipitation,
    and rain that weakens and stops is "tapering off" rather than "until".
    """
    hours = make_hours(24, BASE_TIME + 6 * HOUR, cloudCover=0.95)
    for i, hour_idx in enumerate(wet_hours):
        amount = amounts[i] if amounts else 1.0
        hours[hour_idx].update(
            precipType="rain",
            rainIntensity=amount,
            liquidAccumulation=amount,
            precipProbability=0.9,
        )

    summary = calculate_day_text(hours, True, "UTC", "day")[1]

    # summary[1] is the precipitation sentence; an amount sentence may follow
    if summary[0] == "multiple-sentences":
        summary = summary[1]
    assert summary == ["sentence", expected]


@pytest.mark.parametrize(
    ("cloud_cover", "expected"),
    [
        # HRRR/HRDPS-style two-hour dip must not turn an overcast day into
        # "clearing" or "becoming".
        pytest.param(
            [100] * 7 + [0, 14] + [100] * 15,
            ["for-day", "heavy-clouds"],
            id="model-spike-ignored",
        ),
        # A quick drop that then holds is a step change, not "clearing".
        pytest.param(
            [75] * 6 + [15, 13, 20] * 6,
            ["during", ["becoming", "medium-clouds", "very-light-clouds"], "afternoon"],
            id="step-change-becoming",
        ),
        # A steady rise over most of the day is "increasing clouds".
        pytest.param(
            [5, 10, 15, 25, 30, 40, 50, 55, 65, 70, 80, 90] + [95] * 12,
            ["until", "increasing", "evening"],
            id="gradual-increasing",
        ),
        # Mostly cloudy for 18 of 24 hours, partly cloudy in the morning.
        pytest.param(
            [60] * 6 + [70] * 18,
            ["for-day", ["generally", "medium-clouds"]],
            id="generally",
        ),
        pytest.param(
            [60] * 12 + [70] * 12,
            ["for-day", ["to", "light-clouds", "medium-clouds"]],
            id="to",
        ),
        # One missing hour must not turn its whole period clear; the period
        # uses the hours it has.
        pytest.param(
            [math.nan] + [70] * 23,
            ["for-day", "medium-clouds"],
            id="one-hour-missing",
        ),
        # With no cloud cover at all the day is described as clear.
        pytest.param([math.nan] * 24, ["for-day", "clear"], id="all-missing"),
    ],
)
def test_daily_cloud_cover_wording(cloud_cover, expected):
    """Cloud transitions must come from sustained changes in the smoothed
    hourly cloud cover, so short model spikes don't produce clearing or
    becoming text, and quick step changes aren't described as trends. Cloud
    split between two neighbouring levels is "generally X" or "X to Y" rather
    than one level with a time phrase.
    """
    hours = make_hours(24, BASE_TIME + 6 * HOUR)
    for hour, cover in zip(hours, cloud_cover, strict=True):
        hour["cloudCover"] = cover / 100

    assert calculate_day_text(hours, True, "UTC", "day")[1] == ["sentence", expected]


def test_weekly_lists_days_and_steady_highs():
    """Three or more separate groups of rainy days must use "list" rather than
    nested "and"s, and highs within a few degrees all week are "steady around"
    their mean rather than an arbitrary peak or valley.
    """
    highs = [12.0, 13.0, 12.0, 14.0, 13.0, 12.0, 13.0, 12.0]
    week = [
        {
            "time": BASE_TIME + day * 86400,
            "icon": "partly-cloudy-day",
            "precipType": "none",
            "precipProbability": 0.0,
            "liquidAccumulation": 0.0,
            "snowAccumulation": 0.0,
            "iceAccumulation": 0.0,
            "rainIntensityMax": 0.0,
            "snowIntensityMax": 0.0,
            "iceIntensityMax": 0.0,
            "temperatureHigh": highs[day],
        }
        for day in range(8)
    ]
    for day in (0, 1, 2, 4, 6):
        week[day].update(
            icon="rain",
            precipType="rain",
            precipProbability=0.8,
            liquidAccumulation=5.0,
            rainIntensityMax=3.0,
        )

    text, _ = calculate_weekly_text(week, "UTC")

    assert text == [
        "with",
        [
            "during",
            "medium-rain",
            ["list", ["through", "today", "sunday"], "tuesday", "thursday"],
        ],
        ["temperatures-steady", ["celsius", 13]],
    ]
