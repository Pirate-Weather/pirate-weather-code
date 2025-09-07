import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

from API.PirateTextHelper import (
    calculate_precip_text,
    calculate_sky_icon,
    estimateSnowDensity,
    estimateSnowHeight,
    humidity_sky_text,
)


def test_calculate_sky_icon_day_night():
    assert calculate_sky_icon(0.9, True) == "cloudy"
    assert calculate_sky_icon(0.9, False) == "cloudy"
    assert calculate_sky_icon(0.5, True) == "partly-cloudy-day"
    assert calculate_sky_icon(0.5, False) == "partly-cloudy-night"
    assert calculate_sky_icon(0.1, True, icon="pirate") == "clear-day"


def test_calculate_precip_text_light_rain():
    text, icon = calculate_precip_text(
        prepIntensity=0.1,
        prepAccumUnit=1.0,
        prepType="rain",
        type="hour",
        rainPrep=0.1,
        snowPrep=0,
        icePrep=0,
        pop=0.5,
        icon="pirate",
        isDayTime=True,
    )
    assert text == "possible-very-light-rain"
    assert icon == "possible-rain-day"


def test_estimate_snow_density_and_height():
    density = estimateSnowDensity(0, 0)
    assert math.isclose(density, 118.381, rel_tol=1e-3)
    height = estimateSnowHeight(10, 0, 0)
    expected_height = 10 * 10 / density
    assert math.isclose(height, expected_height, rel_tol=1e-6)


def test_humidity_sky_text():
    assert humidity_sky_text(25, 1, 0.96) == "high-humidity"
    assert humidity_sky_text(10, 1, 0.1) == "low-humidity"
    assert humidity_sky_text(15, 1, 0.5) is None
