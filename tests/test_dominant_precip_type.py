"""
Tests for dominant precipitation type selection.

Ensures that the most dominant precipitation type by accumulation is used
for the day icon and summary, without arbitrary threshold overrides.
"""

import numpy as np

from API.api_utils import select_daily_precip_type
from API.constants.api_const import PRECIP_IDX

# Shared DATA_DAY layout used by select_daily_precip_type
DATA_DAY = {
    "rain": 0,
    "snow": 1,
    "ice": 2,
}


class TestDominantPrecipType:
    """Dominant-by-accumulation type selection with no threshold overrides."""

    def test_rain_dominant_over_snow(self):
        """Rain > snow: type should be rain regardless of snow amount.

        Reproduces the reported bug where e.g. 13.6 mm rain + 3.6 mm snow
        incorrectly resolved to 'snow'.
        """
        InterPdaySum = np.array([[13.6, 3.6, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], "Rain-dominant day should be 'rain'"

    def test_snow_dominant_over_rain(self):
        """Snow > rain: type should be snow."""
        InterPdaySum = np.array([[3.0, 6.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["snow"], "Snow-dominant day should be 'snow'"

    def test_snow_only(self):
        """Only snow present: type should be snow."""
        InterPdaySum = np.array([[0.0, 6.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["snow"], "Snow-only day should be 'snow'"

    def test_rain_only(self):
        """Only rain present: type should be rain."""
        InterPdaySum = np.array([[15.0, 0.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], "Rain-only day should be 'rain'"

    def test_rain_wins_on_tie(self):
        """When rain == snow accumulation, rain wins (first-index tie-break)."""
        InterPdaySum = np.array([[6.0, 6.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], (
            "Tied accumulations should resolve to rain (first-index tie-break)"
        )

    def test_multiple_days_mixed_dominance(self):
        """Multi-day array: each day independently uses its dominant type."""
        # Day 0: rain dominant (13.6 rain, 3.6 snow) → rain
        # Day 1: snow dominant (3.0 rain, 6.0 snow) → snow
        # Day 2: rain only (10.0 rain, 0.0 snow) → rain
        InterPdaySum = np.array(
            [
                [13.6, 3.6, 0.0],
                [3.0, 6.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        maxPchanceDay = np.array([0, 0, 0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], "Day 0 should be rain-dominant"
        assert result[1] == PRECIP_IDX["snow"], "Day 1 should be snow-dominant"
        assert result[2] == PRECIP_IDX["rain"], "Day 2 should be rain"
