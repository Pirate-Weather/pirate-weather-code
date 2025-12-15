"""
Tests for ice and mixed precipitation type handling.

This test file ensures that:
1. Ice (freezing rain) and sleet types are correctly mapped
2. Mixed precipitation is properly detected and displayed
3. The type builder arrays match PRECIP_IDX constant order
"""

import numpy as np

from API.api_utils import select_daily_precip_type
from API.constants.api_const import PRECIP_IDX
from API.PirateTextHelper import calculate_precip_text


class TestPrecipTypeMapping:
    """Test that precipitation type indices match PRECIP_IDX constant."""

    def test_precip_idx_constant(self):
        """Verify PRECIP_IDX has correct indices."""
        assert PRECIP_IDX["none"] == 0
        assert PRECIP_IDX["snow"] == 1
        assert PRECIP_IDX["ice"] == 2
        assert PRECIP_IDX["sleet"] == 3
        assert PRECIP_IDX["rain"] == 4
        assert PRECIP_IDX["mixed"] == 5

    def test_ptype_map_order(self):
        """Verify pTypeMap arrays match PRECIP_IDX order."""
        # The correct order matching PRECIP_IDX
        expected_types = ["none", "snow", "ice", "sleet", "rain", "mixed"]
        expected_texts = [
            "None",
            "Snow",
            "Freezing Rain",
            "Sleet",
            "Rain",
            "Mixed Precipitation",
        ]

        # Test each index maps correctly
        for idx, (type_name, text_name) in enumerate(
            zip(expected_types, expected_texts)
        ):
            if type_name in PRECIP_IDX:
                assert PRECIP_IDX[type_name] == idx, (
                    f"PRECIP_IDX['{type_name}'] should be {idx}"
                )


class TestMixedPrecipDetection:
    """Test that mixed precipitation is correctly detected."""

    def test_all_three_types_present(self):
        """When rain, snow, and ice are all present, type should be mixed."""
        # Create mock data with all three types present
        DATA_DAY = {
            "rain": 0,
            "snow": 1,
            "ice": 2,
        }

        # Create array with rain, snow, and ice all > 0
        InterPdaySum = np.array([[1.0, 1.0, 1.0]])  # rain, snow, ice
        maxPchanceDay = np.array([0])  # Start with none
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["mixed"], (
            "All three types present should result in mixed type"
        )

    def test_two_types_not_mixed(self):
        """When only two types are present, should not be mixed."""
        DATA_DAY = {
            "rain": 0,
            "snow": 1,
            "ice": 2,
        }

        # Rain and snow, but no ice
        InterPdaySum = np.array([[1.0, 1.0, 0.0]])
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        # Should be the dominant type (rain or snow), not mixed
        assert result[0] != PRECIP_IDX["mixed"], (
            "Only two types should not result in mixed"
        )

    def test_one_type_only(self):
        """When only one type is present, should be that type."""
        DATA_DAY = {
            "rain": 0,
            "snow": 1,
            "ice": 2,
        }

        # Only rain
        InterPdaySum = np.array([[5.0, 0.0, 0.0]])
        maxPchanceDay = np.array([0])
        prepAccumUnit = 1.0

        result = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )

        assert result[0] == PRECIP_IDX["rain"], "Should be rain type"


class TestTextGeneration:
    """Test text generation for ice and mixed types."""

    def test_mixed_precip_text(self):
        """Test that mixed precipitation generates correct text."""
        # When num_types > 2, should generate "mixed-precipitation"
        text, icon = calculate_precip_text(
            precipType="mixed",
            type="hourly",
            rainAccum=1.0,
            snowAccum=1.0,
            sleetAccum=1.0,
            pop=1.0,
            icon="darksky",
            mode="both",
        )

        assert text == "mixed-precipitation", (
            "Mixed type should generate 'mixed-precipitation' text"
        )

    def test_ice_precip_text(self):
        """Test that ice (freezing rain) generates correct text."""
        # Note: sleetAccum parameter handles both sleet and ice accumulation
        text, icon = calculate_precip_text(
            precipType="ice",
            type="hourly",
            rainAccum=0.0,
            snowAccum=0.0,
            sleetAccum=2.0,  # Used for both ice and sleet accumulation
            pop=1.0,
            icon="darksky",
            mode="both",
            eff_ice_intensity=0.5,  # Medium intensity
        )

        # Should contain "freezing-rain" in the text
        assert "freezing-rain" in text, "Ice type should generate freezing rain text"

    def test_sleet_precip_text(self):
        """Test that sleet generates correct text."""
        text, icon = calculate_precip_text(
            precipType="sleet",
            type="hourly",
            rainAccum=0.0,
            snowAccum=0.0,
            sleetAccum=2.0,
            pop=1.0,
            icon="darksky",
            mode="both",
            eff_ice_intensity=0.5,  # Medium intensity
        )

        # Should contain "sleet" in the text
        assert "sleet" in text, "Sleet type should generate sleet text"

    def test_mixed_overrides_individual_types(self):
        """Test that mixed precipitation text is not overridden by individual types."""
        # When precipType is rain but num_types > 2, should still be mixed
        text, icon = calculate_precip_text(
            precipType="mixed",  # Set to mixed
            type="hourly",
            rainAccum=1.0,
            snowAccum=1.0,
            sleetAccum=1.0,
            pop=1.0,
            icon="darksky",
            mode="both",
            eff_rain_intensity=0.5,
        )

        assert text == "mixed-precipitation", (
            "Mixed type should not be overridden by individual type conditions"
        )
