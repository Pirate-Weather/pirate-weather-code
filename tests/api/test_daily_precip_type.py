"""Precipitation type selection for daily blocks and GEPS/REPS ensembles."""

import numpy as np
import pytest

from API.api_utils import map_ensemble_precip_rates_to_ptype, select_daily_precip_type
from API.constants.api_const import PRECIP_IDX
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.shared_const import MISSING_DATA
from API.hourly.block import _calculate_derived_metrics
from API.PirateDailyText import calculate_day_text
from API.PirateTextHelper import calculate_precip_text
from tests.helpers.text import make_hours


def test_daily_precip_type_follows_dominant_accumulation():
    """Each day's precipType must be the type with the most accumulation, and
    "mixed" only when rain, snow and ice all fall.

    Regression for #582: 13.6 mm rain + 3.6 mm snow was reported as "snow"
    because of threshold overrides; it must be "rain".
    """
    days = [
        # (rain, snow, ice) mm, expected type
        ((13.6, 3.6, 0.0), "rain"),  # #582
        ((3.0, 6.0, 0.0), "snow"),
        ((6.0, 6.0, 0.0), "rain"),  # tie resolves to rain
        ((1.0, 1.0, 1.0), "mixed"),
    ]
    day_sums = np.zeros((len(days), max(DATA_DAY.values()) + 1))
    for row, ((rain, snow, ice), _) in enumerate(days):
        day_sums[row, [DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]]] = (
            rain,
            snow,
            ice,
        )

    result = select_daily_precip_type(
        day_sums, DATA_DAY, np.zeros(len(days), dtype=int), PRECIP_IDX, 1.0
    )

    assert result.tolist() == [PRECIP_IDX[expected] for _, expected in days]


def test_ensemble_rates_map_to_precip_type():
    """GEPS/REPS have no categorical type, only per-type rates. A single channel
    above the threshold must map to its type (ice pellets -> sleet, freezing
    rain -> ice), nothing above it to "none", and all-NaN to MISSING_DATA so
    missing data is not reported as dry.

    Several strong channels are mixed, not any one of them.
    """
    nan = np.nan
    result = map_ensemble_precip_rates_to_ptype(
        snow=np.array([4.0, 0.0, 0.0, 0.0, 0.05, nan, 2.0]),
        ice=np.array([0.0, 3.0, 0.0, 0.0, 0.0, nan, 0.0]),
        freezing_rain=np.array([0.0, 0.0, 1.0, 0.0, 0.0, nan, 0.0]),
        rain=np.array([0.0, 0.0, 0.0, 2.0, 0.05, nan, 3.0]),
        threshold=0.1,
    )

    np.testing.assert_array_equal(
        result,
        [
            PRECIP_IDX["snow"],
            PRECIP_IDX["sleet"],
            PRECIP_IDX["ice"],
            PRECIP_IDX["rain"],
            PRECIP_IDX["none"],
            MISSING_DATA,
            PRECIP_IDX["mixed"],
        ],
    )


@pytest.mark.parametrize(
    ("precip_type", "other_accum", "icon_set", "expected"),
    [
        pytest.param(
            "ice", 0.0, "darksky", ("light-freezing-rain", "sleet"), id="ice-darksky"
        ),
        pytest.param(
            "ice",
            0.0,
            "pirate",
            ("light-freezing-rain", "light-freezing-rain"),
            id="ice-pirate",
        ),
        pytest.param(
            "sleet", 0.0, "pirate", ("light-sleet", "light-sleet"), id="sleet"
        ),
        pytest.param(
            "mixed", 1.0, "pirate", ("mixed-precipitation", "mixed"), id="mixed"
        ),
    ],
)
def test_ice_sleet_and_mixed_precip_text(precip_type, other_accum, icon_set, expected):
    """Freezing rain, sleet and mixed precipitation must keep distinct wording
    and icons; Dark Sky has no freezing-rain icon so it falls back to sleet.

    ``other_accum`` is the rain and snow accumulation (mm) next to 2 mm of ice.
    """
    assert (
        calculate_precip_text(
            precipType=precip_type,
            type="hourly",
            rainAccum=other_accum,
            snowAccum=other_accum,
            sleetAccum=2.0,
            pop=1.0,
            icon=icon_set,
            mode="both",
            eff_ice_intensity=0.5,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("hourly_type", "accumulation", "expected"),
    [
        pytest.param("sleet", (0.0, 0.0, 3.0), "sleet", id="ice-pellets-stay-sleet"),
        pytest.param("ice", (0.0, 0.0, 3.0), "ice", id="freezing-rain-stays-ice"),
        # Ice is the most accumulation, but the hours weren't sleet
        pytest.param("rain", (1.0, 0.0, 3.0), "ice", id="other-hours-default-to-ice"),
        # A little sleet doesn't outrank the rain that fell
        pytest.param("sleet", (5.0, 0.0, 1.0), "rain", id="dominant-rain-wins"),
    ],
)
def test_daily_sleet_is_kept_apart_from_freezing_rain(
    hourly_type, accumulation, expected
):
    """Sleet and freezing rain share one accumulation column, so only the hourly
    type tells them apart. It was overwritten before the check that keeps it, and
    every ice-pellet day was reported as freezing rain, a more hazardous forecast.
    """
    day_sums = np.zeros((1, max(DATA_DAY.values()) + 1))
    day_sums[0, [DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]]] = accumulation

    result = select_daily_precip_type(
        day_sums, DATA_DAY, np.array([PRECIP_IDX[hourly_type]]), PRECIP_IDX, 1.0
    )

    assert result.tolist() == [PRECIP_IDX[expected]]


def test_mixed_ensemble_hours_keep_their_precipitation():
    """GEPS and REPS mark an hour with several strong channels as mixed before the
    accumulation is split by type, which only knew rain, snow, ice and sleet: the
    hour's precipitation vanished from the accumulation, intensity, and the daily
    text and type. It must count as ice and still read as mixed.
    """
    hours = 24
    InterPhour = np.zeros((hours, max(DATA_HOURLY.values()) + 1))
    InterPhour[:, DATA_HOURLY["type"]] = PRECIP_IDX["mixed"]
    InterPhour[:, DATA_HOURLY["accum"]] = 2.0
    InterPhour[:, DATA_HOURLY["intensity"]] = 2.0
    InterPhour[:, DATA_HOURLY["prob"]] = 0.9
    InterPhour[:, DATA_HOURLY["temp"]] = 0.0

    _calculate_derived_metrics(
        InterPhour,
        hourlyDayIndex=np.zeros(hours, dtype=int),
        baseTimeOffset=0,
        timeMachine=True,
    )

    assert np.all(InterPhour[:, DATA_HOURLY["ice"]] == 2.0)
    assert np.all(InterPhour[:, DATA_HOURLY["ice_intensity"]] == 2.0)

    day_sum = np.zeros((1, max(DATA_DAY.values()) + 1))
    day_sum[0, DATA_DAY["ice"]] = InterPhour[:, DATA_HOURLY["ice"]].sum()
    daily_type = select_daily_precip_type(
        day_sum, DATA_DAY, np.array([PRECIP_IDX["mixed"]]), PRECIP_IDX, 1.0
    )
    assert daily_type.tolist() == [PRECIP_IDX["mixed"]]

    hour_list = make_hours(
        hours,
        precipType="mixed",
        iceAccumulation=2.0,
        iceIntensity=2.0,
        precipProbability=0.9,
    )
    _, text = calculate_day_text(hour_list, True, "UTC", "day")
    assert "mixed-precipitation" in str(text)
