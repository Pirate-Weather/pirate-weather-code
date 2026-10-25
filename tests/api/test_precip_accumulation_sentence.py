"""Expected precipitation amount sentence in the daily summary.

A significant amount of precipitation is given in its own "of-expected"
sentence after the precipitation summary. Its form depends on the model's
precipitation error: a "low-high" range, "< N", or an exact value when the
source has no error field (ERA5).
"""

import numpy as np
import pytest

from API.PirateDailyText import calculate_day_text
from tests.helpers.text import make_hours

# 24 hours of light snow at -2 C / 5 m/s. Each hour's precipIntensityError
# (liquid mm) is turned into snow depth at about 7.2 mm per mm by the density
# formula, then summed into the daily error.
SNOW = {
    "precipType": "snow",
    "snowIntensity": 1.0,
    "precipProbability": 0.8,
    "cloudCover": 0.9,
    "temperature": -2.0,
    "dewPoint": -4.0,
}


@pytest.mark.parametrize(
    ("snow_mm_per_hour", "error_mm_per_hour", "unit_system", "expected_amount"),
    [
        # 2.4 cm +/- 0.9 cm
        pytest.param(1.0, 0.1, "si", ["centimeters", ["range", 1, 4]], id="range-si"),
        # 2.4 in +/- 0.9 in
        pytest.param(2.54, 0.25, "us", ["inches", ["range", 1, 4]], id="range-us"),
        # Models with a zero error field: no lower bound to show.
        pytest.param(
            1.0, 0.0, "si", ["less-than", ["centimeters", 3]], id="zero-error"
        ),
        # ERA5 has no error field: show the rounded-up amount itself.
        pytest.param(1.0, np.nan, "si", ["centimeters", 3], id="missing-error"),
        # 2.4 cm +/- 1.5 cm: range would start at 0 and the top (4 cm) is below
        # LESS_THAN_TOLERANCE (5 cm), so "< 4 cm".
        pytest.param(
            1.0,
            0.175,
            "si",
            ["less-than", ["centimeters", 4]],
            id="zero-low-below-tolerance",
        ),
        # 2.4 cm +/- 3 cm: top reaches the 5 cm tolerance, so the 0 lower bound
        # is bumped to 1 and a range is kept (#510).
        pytest.param(
            1.0,
            0.35,
            "si",
            ["centimeters", ["range", 1, 6]],
            id="zero-low-bumped-510",
        ),
    ],
)
def test_daily_snow_accumulation_sentence(
    snow_mm_per_hour, error_mm_per_hour, unit_system, expected_amount
):
    """Pin the exact snow amount phrase for each error regime and unit system.

    Guards the unit conversion, the floor/ceil range rule, the missing-error
    (ERA5) path, and the #510 rule that large uncertain totals show "1-N"
    instead of a misleading "< N".
    """
    hours = make_hours(
        24,
        **SNOW,
        snowAccumulation=snow_mm_per_hour,
        precipIntensityError=error_mm_per_hour,
    )

    icon, summary = calculate_day_text(
        hours,
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        icon_set="darksky",
        unit_system=unit_system,
    )

    assert summary == [
        "multiple-sentences",
        ["sentence", ["for-day", "very-light-snow"]],
        ["sentence", ["of-expected", expected_amount, "medium-snow"]],
    ]
    assert icon == "snow"


def _rain_and_ice_hours(rain_mm, ice_mm):
    """12 hours of rain then 12 of freezing rain, rain_mm/ice_mm in total,
    each with a 0.1 mm/h precipitation error."""
    rain = make_hours(
        12,
        precipType="rain",
        rainIntensity=rain_mm / 12,
        liquidAccumulation=rain_mm / 12,
        precipProbability=0.9,
        precipIntensityError=0.1,
    )
    ice = make_hours(
        12,
        rain[-1]["time"] + 3600,
        precipType="ice",
        iceIntensity=ice_mm / 12,
        iceAccumulation=ice_mm / 12,
        precipProbability=0.9,
        precipIntensityError=0.1,
    )
    return rain + ice


@pytest.mark.parametrize(
    ("rain_mm", "ice_mm", "unit_system", "expected"),
    [
        # 20 mm +/- 1.2 mm of rain
        pytest.param(
            20,
            0,
            "si",
            ["of-expected", ["millimeters", ["range", 19, 21]], "medium-rain"],
            id="rain-mm",
        ),
        # The translations take whole numbers only, so in US units 20 mm
        # (0.8 in) can only be "< 1 in".
        pytest.param(
            20,
            0,
            "us",
            ["of-expected", ["less-than", ["inches", 1]], "medium-rain"],
            id="rain-us-whole-inches",
        ),
        # Ice over 1 mm is shown ahead of a larger amount of rain
        pytest.param(
            30,
            3,
            "si",
            ["of-expected", ["millimeters", ["range", 2, 4]], "medium-freezing-rain"],
            id="ice-before-rain",
        ),
        # 12 mm of rain is below the 15 mm threshold for the sentence
        pytest.param(12, 0, "si", None, id="rain-below-threshold"),
    ],
)
def test_daily_rain_and_ice_accumulation_sentence(
    rain_mm, ice_mm, unit_system, expected
):
    """The amount sentence covers every precipitation type, not just snow: guard
    the type chosen when several fall (ice first), the per-type thresholds,
    and millimetres for rain and ice outside US units.
    """
    _, summary = calculate_day_text(
        _rain_and_ice_hours(rain_mm, ice_mm),
        is_day_time=True,
        time_zone="UTC",
        mode="daily",
        unit_system=unit_system,
    )

    if expected is None:
        assert summary[0] == "sentence"
    else:
        assert summary[0] == "multiple-sentences"
        assert summary[2] == ["sentence", expected]
