"""Fire weather index calculation utilities."""

from __future__ import annotations

import numpy as np

from API.constants.clip_const import CLIP_FIRE


def calculate_fosberg_fire_index(
    temperature_c: np.ndarray,
    humidity_fraction: np.ndarray,
    wind_speed_ms: np.ndarray,
) -> np.ndarray:
    """Calculate the Fosberg Fire Weather Index from SI meteorological inputs.

    Converts SI inputs to US customary units internally and applies the
    three-regime equilibrium moisture content model before computing the
    final index.  Output is clipped to the configured ``CLIP_FIRE`` bounds.

    Reference:
        https://a.atmos.washington.edu/wrfrt/descript/definitions/fosbergindex.html

    Args:
        temperature_c: Air temperature in degrees Celsius.  May be a scalar
            or any shape ndarray.
        humidity_fraction: Relative humidity as a fraction in [0, 1].  Values
            are clamped to [0 %, 100 %] before use.
        wind_speed_ms: Wind speed in metres per second.  Negative values are
            treated as calm (zero).

    Returns:
        Fosberg Fire Weather Index array with the same shape as the inputs,
        clipped to ``[CLIP_FIRE["min"], CLIP_FIRE["max"]]``.  Elements where
        any input is non-finite (NaN / ±Inf) are returned as ``NaN``.
    """
    # Convert SI inputs to US customary units required by the Fosberg formula.
    temp_f = (np.asarray(temperature_c) * 9 / 5) + 32
    rh = np.clip(np.asarray(humidity_fraction) * 100, 0, 100)
    wind_mph = np.maximum(np.asarray(wind_speed_ms) * 2.2369362921, 0)

    fire_index = np.full(temp_f.shape, np.nan, dtype=float)

    # Only compute for grid points where all three inputs are finite.
    valid = np.isfinite(temp_f) & np.isfinite(rh) & np.isfinite(wind_mph)
    if not np.any(valid):
        return fire_index

    # Equilibrium moisture content uses three RH regimes (low / mid / high).
    fuel_moisture = np.full(temp_f.shape, np.nan, dtype=float)
    low_rh = valid & (rh < 10)
    mid_rh = valid & (rh >= 10) & (rh <= 50)
    high_rh = valid & (rh > 50)

    fuel_moisture[low_rh] = (
        0.03229 + 0.281073 * rh[low_rh] - 0.000578 * rh[low_rh] * temp_f[low_rh]
    )
    fuel_moisture[mid_rh] = 2.22749 + 0.160107 * rh[mid_rh] - 0.014784 * temp_f[mid_rh]
    fuel_moisture[high_rh] = (
        21.0606
        + 0.005565 * rh[high_rh] ** 2
        - 0.00035 * rh[high_rh] * temp_f[high_rh]
        - 0.483199 * rh[high_rh]
    )

    # Moisture damping coefficient η (eta) reduces fire potential as fuel moisture rises.
    eta = (
        1
        - 2 * (fuel_moisture / 30)
        + 1.5 * (fuel_moisture / 30) ** 2
        - 0.5 * (fuel_moisture / 30) ** 3
    )

    # Final index: η × √(1 + U²) / 0.3002, where U is wind speed in mph.
    fire_index[valid] = eta[valid] * np.sqrt(1 + wind_mph[valid] ** 2) / 0.3002

    return np.clip(fire_index, CLIP_FIRE["min"], CLIP_FIRE["max"])
