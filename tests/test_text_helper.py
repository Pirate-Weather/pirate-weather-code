import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from API.PirateTextHelper import (
    calculate_precip_text,
    calculate_sky_icon,
    estimate_snow_density,
    estimate_snow_height,
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
        precipType="rain",
        type="hour",
        rainAccum=0.1,
        snowAccum=0,
        sleetAccum=0,
        pop=0.5,
        icon="pirate",
        isDayTime=True,
        eff_rain_intensity=0.01,
    )
    assert text == "possible-very-light-rain"
    assert icon == "possible-rain-day"


def test_estimate_snow_density_and_height_vectorized():
    temperatures_c = np.array([0.0, -5.0, 2.0])
    wind_speeds_mps = np.array([0.0, 5.0, 10.0])
    liquid_mm = np.array([10.0, 5.0, 15.0])

    # Expected densities (calculated based on the vectorized function logic)
    expected_densities = np.array([118.3810, 119.3624, 285.6412])
    calculated_densities = estimate_snow_density(temperatures_c, wind_speeds_mps)
    np.testing.assert_allclose(calculated_densities, expected_densities, rtol=1e-3)

    # Expected heights
    expected_heights = np.array([84.47300, 41.88923, 52.51342])
    calculated_heights = estimate_snow_height(
        liquid_mm, temperatures_c, wind_speeds_mps
    )
    np.testing.assert_allclose(calculated_heights, expected_heights, rtol=1e-6)


def test_humidity_sky_text():
    assert humidity_sky_text(25, 0.96) == "high-humidity"
    assert humidity_sky_text(10, 0.1) == "low-humidity"
    assert humidity_sky_text(15, 0.5) is None
