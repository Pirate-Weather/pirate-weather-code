"""Thunderstorm wording in the minutely summary.

Minutely uses a raw ``maxCAPE >= CAPE_THRESHOLDS["high"]`` gate (see TODO.md:
it disagrees with the currently/hourly score). When it fires, the precipitation
word in the timing sentence is replaced by "thunderstorm".
"""

from unittest.mock import ANY

import pytest

from API.constants.text_const import CAPE_THRESHOLDS
from API.PirateMinutelyText import calculate_minutely_text

RAIN = {"precipIntensity": 5.0, "precipType": "rain"}
DRY = {"precipIntensity": 0.0, "precipType": "none"}

# The second duration of the two "-later" shapes looks off by one minute (rain
# from minute 41 is reported as "returning 22 min later" after ending at minute
# 20). It is reported for review rather than pinned here.
LATER = ["minutes", ANY]


def _minutes(*runs):
    """Build a 61-minute array from (minute_dict, count) runs."""
    minutes = [dict(minute) for minute, count in runs for _ in range(count)]
    assert len(minutes) == 61
    return minutes


def _summarise(minutes, max_cape):
    return calculate_minutely_text(
        minuteArr=minutes,
        currentText="clear",
        currentIcon="clear-day",
        icon="darksky",
        maxCAPE=max_cape,
    )


@pytest.mark.parametrize(
    ("minutes", "expected_text"),
    [
        pytest.param(_minutes((RAIN, 61)), ["for-hour", "thunderstorm"], id="for-hour"),
        pytest.param(
            _minutes((DRY, 15), (RAIN, 46)),
            ["starting-in", "thunderstorm", ["minutes", 15]],
            id="starting-in",
        ),
        pytest.param(
            _minutes((RAIN, 30), (DRY, 31)),
            ["stopping-in", "thunderstorm", ["minutes", 30]],
            id="stopping-in",
        ),
        pytest.param(
            _minutes((RAIN, 20), (DRY, 21), (RAIN, 20)),
            [
                "stopping-then-starting-later",
                "thunderstorm",
                ["minutes", 20],
                LATER,
            ],
            id="stopping-then-starting-later",
        ),
        pytest.param(
            _minutes((DRY, 10), (RAIN, 20), (DRY, 31)),
            [
                "starting-then-stopping-later",
                "thunderstorm",
                ["minutes", 10],
                LATER,
            ],
            id="starting-then-stopping-later",
        ),
    ],
)
def test_minutely_thunderstorm_keeps_timing(minutes, expected_text):
    """High CAPE must swap only the precipitation word for "thunderstorm" and
    keep the start/stop timing for every sentence shape minutely can produce.
    """
    assert _summarise(minutes, max_cape=3000) == (expected_text, "thunderstorm")


@pytest.mark.parametrize(
    ("max_cape", "expected"),
    [
        pytest.param(
            CAPE_THRESHOLDS["high"] - 1,
            (["for-hour", "medium-rain"], "rain"),
            id="just-below",
        ),
        pytest.param(
            CAPE_THRESHOLDS["high"],
            (["for-hour", "thunderstorm"], "thunderstorm"),
            id="at-threshold",
        ),
    ],
)
def test_minutely_thunderstorm_cape_boundary(max_cape, expected):
    """The minutely gate is inclusive at CAPE_THRESHOLDS["high"]; an off-by-one
    here would flip storms on or off for borderline hours.
    """
    assert _summarise(_minutes((RAIN, 61)), max_cape) == expected


def test_minutely_no_thunderstorm_without_precipitation():
    """High CAPE alone must not produce a storm when no precipitation is
    forecast in the next hour.
    """
    assert _summarise(_minutes((DRY, 61)), max_cape=3000) == (
        ["for-hour", "clear"],
        "clear-day",
    )
