"""Precipitation and heat metrics helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from API.constants.api_const import DBZ_CONVERSION_CONST, DBZ_CONST, GLOBE_TEMP_CONST
from API.constants.shared_const import REFC_THRESHOLD
from API.constants.api_const import WBGT_CONST, WBGT_PERCENTAGE_DIVISOR


def calculate_globe_temperature(
    air_temperature: float,
    solar_radiation: float,
    wind_speed: float,
    globe_diameter: float = 0.15,
    emissivity: float = 0.95,
) -> float:
    """Estimate globe temperature given ambient conditions."""
    globe_temperature = air_temperature + (
        GLOBE_TEMP_CONST["factor"] * (solar_radiation ** GLOBE_TEMP_CONST["temp_exp"])
    ) / (
        emissivity
        * (globe_diameter ** GLOBE_TEMP_CONST["diam_exp"])
        * (wind_speed ** GLOBE_TEMP_CONST["wind_exp"])
    )
    return globe_temperature


def calculate_wbgt(
    temperature: float,
    humidity: float,
    wind_speed: Optional[float] = None,
    solar_radiation: Optional[float] = None,
    globe_temperature: Optional[float] = None,
    in_sun: bool = False,
) -> float:
    """Calculate the Wet-Bulb Globe Temperature (WBGT)."""
    if in_sun:
        if globe_temperature is None:
            if wind_speed is None or solar_radiation is None:
                raise ValueError(
                    "Wind speed and solar radiation must be provided if globe temperature is not provided for outdoor WBGT calculation."
                )
            globe_temperature = calculate_globe_temperature(
                temperature, solar_radiation, wind_speed
            )
        wbgt = (
            WBGT_CONST["temp_weight"] * temperature
            + WBGT_CONST["globe_weight"] * globe_temperature
            + WBGT_CONST["wind_weight"] * wind_speed
        )
    else:
        wbgt = WBGT_CONST["temp_weight"] * temperature + WBGT_CONST[
            "humidity_weight"
        ] * (humidity / WBGT_PERCENTAGE_DIVISOR * temperature)

    return wbgt


def dbz_to_rate(
    dbz_array: np.ndarray,
    precip_type_array: np.ndarray,
    min_dbz: float = REFC_THRESHOLD,
) -> np.ndarray:
    """
    Convert dBZ to precipitation rate (mm/h) using a Z-R relationship with soft threshold.
    """
    dbz_array = np.maximum(dbz_array, DBZ_CONVERSION_CONST["min_value"])

    # Convert dBZ to Z
    z_array = 10 ** (dbz_array / DBZ_CONVERSION_CONST["divisor"])

    # Initialize rate coefficients for rain
    a_array = np.full_like(dbz_array, DBZ_CONST["rain_a"], dtype=float)
    b_array = np.full_like(dbz_array, DBZ_CONST["rain_b"], dtype=float)
    snow_mask = precip_type_array == "snow"
    a_array[snow_mask] = DBZ_CONST["snow_a"]
    b_array[snow_mask] = DBZ_CONST["snow_b"]

    # Compute precipitation rate
    rate_array = (z_array / a_array) ** (DBZ_CONVERSION_CONST["exponent"] / b_array)

    # Apply soft threshold for sub-threshold dBZ values
    below_threshold = dbz_array < min_dbz
    rate_array[below_threshold] *= dbz_array[below_threshold] / min_dbz

    # Final check: ensure no negative rates
    rate_array = np.maximum(rate_array, DBZ_CONVERSION_CONST["min_value"])
    return rate_array
