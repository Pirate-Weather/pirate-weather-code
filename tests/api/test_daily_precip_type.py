"""Precipitation type selection for daily blocks and GEPS/REPS ensembles."""

import numpy as np
import pytest

from API.api_utils import map_ensemble_precip_rates_to_ptype, select_daily_precip_type
from API.constants.api_const import PRECIP_IDX
from API.constants.forecast_const import DATA_DAY
from API.constants.shared_const import MISSING_DATA
from API.PirateTextHelper import calculate_precip_text


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

    Rows with several strong channels are not pinned: whether they should be
    sleet or mixed is an open question (TODO.md).
    """
    nan = np.nan
    result = map_ensemble_precip_rates_to_ptype(
        snow=np.array([4.0, 0.0, 0.0, 0.0, 0.05, nan]),
        ice=np.array([0.0, 3.0, 0.0, 0.0, 0.0, nan]),
        freezing_rain=np.array([0.0, 0.0, 1.0, 0.0, 0.0, nan]),
        rain=np.array([0.0, 0.0, 0.0, 2.0, 0.05, nan]),
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
