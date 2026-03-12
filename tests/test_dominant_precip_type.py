"""
Tests for dominant precipitation type selection.

Ensures that when rain accumulation exceeds snow accumulation, the day icon
and summary correctly show "rain" rather than "snow", even when snow
accumulation surpasses the snow promotion threshold.

See: https://github.com/Pirate-Weather/pirate-weather-code/issues (precipType dominant type bug)
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


class TestSnowThresholdDoesNotOverrideDominantRain:
    """
    The snow threshold override must not flip the precip type to 'snow' when
    rain accumulation is the actual dominant type for the day.
    """

    def test_rain_dominant_snow_above_threshold(self):
        """Rain > snow: type should be rain even when snow exceeds the threshold.

        Reproduces the reported bug where e.g. 13.6 mm rain + 3.6 mm snow
        incorrectly resolved to 'snow'.
        """
        # snow > 5mm threshold, but rain > snow
        InterPdaySum = np.array([[13.6, 3.6, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], (
            "Rain-dominant day should remain 'rain' even when snow exceeds the threshold"
        )

    def test_snow_dominant_above_threshold(self):
        """Snow > rain: type should be snow when snow also exceeds the threshold."""
        # snow > 5mm threshold AND snow > rain
        InterPdaySum = np.array([[3.0, 6.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["snow"], (
            "Snow-dominant day should be 'snow' when snow exceeds the threshold"
        )

    def test_snow_only_above_threshold(self):
        """Only snow present and above threshold: type should be snow."""
        InterPdaySum = np.array([[0.0, 6.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["snow"], "Snow-only day should be 'snow'"

    def test_rain_only_above_threshold(self):
        """Only rain present and above threshold: type should be rain."""
        InterPdaySum = np.array([[15.0, 0.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], "Rain-only day should be 'rain'"

    def test_snow_equals_rain_above_threshold(self):
        """When snow == rain, snow threshold override is allowed (tie goes to snow)."""
        InterPdaySum = np.array([[6.0, 6.0, 0.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        # Both exceed their thresholds and are equal — rain override fires first,
        # then snow override is allowed (snow >= rain), so result should be snow.
        assert result[0] == PRECIP_IDX["snow"], (
            "Tied accumulations with both above threshold should resolve to snow"
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
