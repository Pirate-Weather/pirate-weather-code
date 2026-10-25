"""Thunderstorm wording in currently/hourly, daily, 24-hour and weekly summaries.

Thunderstorms are decided by the multi-index score in
``PirateTextHelper.calculate_thunderstorm_text`` (CAPE, lifted index, K index and
vertical velocity, damped by CIN, dewpoint depression and PoP). These tests pin the
user-visible outcome of that score in each summary type.
"""

import pytest

from API.constants.shared_const import MISSING_DATA
from API.PirateDailyText import calculate_day_text
from API.PirateText import calculate_text
from API.PirateWeeklyText import calculate_weekly_text
from tests.helpers.text import BASE_TIME, HOUR, make_hour, make_hours

# 5 mm/h rain with 80% PoP: a clearly wet hour whose storm wording depends on
# the stability indices alone.
RAIN = {
    "precipType": "rain",
    "rainIntensity": 5.0,
    "liquidAccumulation": 5.0,
    "precipProbability": 0.8,
    "cloudCover": 0.8,
}

# Moderate CAPE plus unstable lifted index, high K index and ascent. CAPE alone
# (1500 J/kg) only reaches "possible"; the other indices push it to a storm.
UNSTABLE = {
    "cape": 1500,
    "liftedIndex": -5.0,
    "kIndex": 35.0,
    "verticalVelocity": -0.5,
}


@pytest.mark.parametrize(
    ("overrides", "summary_type", "icon_set", "expected"),
    [
        pytest.param(
            {"cape": 2600},
            "hour",
            "darksky",
            ("thunderstorm", "thunderstorm"),
            id="high-cape-storm",
        ),
        pytest.param(
            {**UNSTABLE, "convectiveInhibition": -30.0},
            "hour",
            "darksky",
            ("thunderstorm", "thunderstorm"),
            id="multi-index-storm-weak-cin",
        ),
        pytest.param(
            {**UNSTABLE, "convectiveInhibition": -200.0},
            "hour",
            "darksky",
            ("medium-rain", "rain"),
            id="strong-cin-cap-suppresses",
        ),
        pytest.param(
            {"cape": 2600, "precipProbability": 0.2},
            "hour",
            "darksky",
            ("possible-medium-rain", "partly-cloudy-day"),
            id="low-pop-suppresses",
        ),
        pytest.param(
            {"cape": 1500},
            "current",
            "darksky",
            ("medium-rain", "rain"),
            id="currently-drops-possible-storm",
        ),
        pytest.param(
            {"cape": 1500},
            "hour",
            "pirate",
            ("possible-thunderstorm", "possible-thunderstorm-day"),
            id="hourly-possible-storm-pirate-icon",
        ),
    ],
)
def test_hourly_thunderstorm_text(overrides, summary_type, icon_set, expected):
    """Pin when an hour's rain text becomes a thunderstorm.

    Guards the score thresholds and suppressors (a strong CIN cap or PoP below
    PRECIP_PROB_THRESHOLD must veto even high CAPE), and the rule that
    currently never shows a merely "possible" storm.
    """
    hour = make_hour(**{**RAIN, **overrides})

    assert (
        calculate_text(hour, isDayTime=True, type=summary_type, icon=icon_set)
        == expected
    )


def test_daily_thunderstorms_joined_with_precipitation():
    """When storms cover every rainy period, the storm text replaces the rain
    text instead of listing both. The 40 mm of rain is still given in its own
    sentence.
    """
    hours = make_hours(8, **RAIN, cape=2600)

    icon, summary = calculate_day_text(
        hours=hours, is_day_time=True, time_zone="UTC", mode="daily"
    )

    assert summary == [
        "multiple-sentences",
        ["sentence", ["for-day", "thunderstorm"]],
        [
            "sentence",
            ["of-expected", ["less-than", ["millimeters", 40]], "medium-rain"],
        ],
    ]
    assert icon == "thunderstorm"


def test_daily_thunderstorms_accompany_precipitation():
    """When storms cover only some of the rainy periods, they are attached to
    the rain with "accompanied-by" ("Rain, with thunderstorms in the
    morning.") instead of a separate clause, which read awkwardly and got very
    long. All-day rain is the bare condition, and definite storms take the icon.

    00:00-05:00 UTC (the night, which runs until the 6am start of the daily
    morning) is rain with low CAPE; 06:00-09:00 UTC adds CAPE 2000.
    """
    rain = {**RAIN, "rainIntensity": 3.0, "liquidAccumulation": 3.0}
    hours = make_hours(6, **rain, cape=500) + make_hours(
        4, BASE_TIME + 6 * HOUR, **rain, cape=2000
    )

    icon, summary = calculate_day_text(
        hours=hours, is_day_time=True, time_zone="UTC", mode="daily"
    )

    assert summary == [
        "multiple-sentences",
        [
            "sentence",
            [
                "accompanied-by",
                "medium-rain",
                ["during", "thunderstorm", "morning"],
            ],
        ],
        [
            "sentence",
            ["of-expected", ["less-than", ["millimeters", 30]], "medium-rain"],
        ],
    ]
    assert icon == "thunderstorm"


def test_24hour_thunderstorms_starting_later():
    """The 24-hour ("hour" mode) summary must name the later period where storms
    occur rather than describing the whole window.

    Starts dry at 03:00 UTC; storms from 06:00 UTC.
    """
    start = BASE_TIME + 3 * HOUR
    hours = make_hours(3, start) + make_hours(4, start + 3 * HOUR, **RAIN, cape=2600)

    icon, summary = calculate_day_text(
        hours=hours, is_day_time=True, time_zone="UTC", mode="hour"
    )

    assert summary == [
        "multiple-sentences",
        ["sentence", ["during", "thunderstorm", "today-morning"]],
        [
            "sentence",
            ["of-expected", ["less-than", ["millimeters", 20]], "medium-rain"],
        ],
    ]
    assert icon == "thunderstorm"


def test_weekly_thunderstorms():
    """Weekly text must put storms in front of the rain for the days that have
    them, and use the storm icon when most rainy days are stormy.
    """
    week = [
        {
            "time": BASE_TIME + day * 86400,
            "icon": "partly-cloudy-day",
            "precipType": "none",
            "liquidAccumulation": 0.0,
            "rainIntensityMax": 0.0,
            "precipProbability": 0.0,
            "temperatureHigh": 24.0,
            "cape": MISSING_DATA,
            "snowIntensityMax": 0.0,
            "iceIntensityMax": 0.0,
            "snowAccumulation": 0.0,
            "iceAccumulation": 0.0,
        }
        for day in range(8)
    ]
    for day, cape in ((0, 2000), (1, 2500)):
        week[day].update(
            icon="thunderstorm",
            precipType="rain",
            liquidAccumulation=10.0,
            rainIntensityMax=5.0,
            precipProbability=0.8,
            cape=cape,
        )

    text, icon = calculate_weekly_text(weekArr=week, timeZone="UTC", icon="darksky")

    # text[2] is the week's temperature trend, which is not under test here.
    assert text[:2] == [
        "with",
        [
            "during",
            ["and", "thunderstorm", "medium-rain"],
            ["and", "today", "tomorrow"],
        ],
    ]
    assert icon == "thunderstorm"


def test_daily_storm_ignores_cape_from_low_pop_hours():
    """Daily storm strength must use only CAPE from hours with PoP >=
    PRECIP_PROB_THRESHOLD.

    Low-PoP rain hours carry CAPE 3000 and likely-rain hours carry CAPE 1500
    (a "possible" storm on its own). If the PoP guard in the daily aggregation is
    dropped, the 3000 J/kg leaks in and the day is upgraded to a full
    thunderstorm.
    """
    low_pop = {**RAIN, "precipProbability": 0.1, "cape": 3000}
    likely = {**RAIN, "cape": 1500}
    hours = [
        make_hour(BASE_TIME + i * HOUR, **(low_pop if i % 2 == 0 else likely))
        for i in range(8)
    ]

    icon, summary = calculate_day_text(
        hours=hours, is_day_time=True, time_zone="UTC", mode="daily"
    )

    assert summary == [
        "multiple-sentences",
        ["sentence", ["for-day", "possible-thunderstorm"]],
        [
            "sentence",
            ["of-expected", ["less-than", ["millimeters", 40]], "medium-rain"],
        ],
    ]
    assert icon == "rain"
