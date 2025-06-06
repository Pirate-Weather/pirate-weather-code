"""Utility functions for time and solar calculations."""

import datetime
import math
import numpy as np
from typing import Tuple

from pytz import timezone, utc
from timezonefinder import TimezoneFinder


tf = TimezoneFinder(in_memory=True)


def get_offset(
    *, lat: float, lng: float, utcTime: datetime.datetime, tf: TimezoneFinder = tf
) -> Tuple[float, timezone]:
    """Return the local timezone offset in minutes and the timezone object."""
    today = utcTime
    tz_target = timezone(tf.timezone_at(lng=lng, lat=lat))
    today_target = tz_target.localize(today)
    today_utc = utc.localize(today)
    return (today_utc - today_target).total_seconds() / 60, tz_target


def rounder(t: datetime.datetime) -> datetime.datetime:
    """Round a ``datetime`` to the nearest hour."""
    if t.minute >= 30:
        return t.replace(second=0, microsecond=0, minute=0) + datetime.timedelta(hours=1)
    return t.replace(second=0, microsecond=0, minute=0)


def unix_to_day_of_year_and_lst(dt: datetime.datetime, longitude: float) -> Tuple[int, float]:
    """Convert a UTC time to day-of-year and local solar time."""
    day_of_year = dt.timetuple().tm_yday
    utc_time = dt.hour + dt.minute / 60 + dt.second / 3600
    lst = utc_time + (longitude / 15)
    return day_of_year, lst


def solar_rad(D_t: int, lat: float, t_t: float) -> float:
    """Calculate theoretical clear-sky short wave radiation."""
    d = 1 + 0.0167 * math.sin((2 * math.pi * (D_t - 93.5365)) / 365)
    r = 0.75
    S_0 = 1367
    delta = 0.4096 * math.sin((2 * math.pi * (D_t + 284)) / 365)
    radLat = math.radians(lat)
    solarHour = math.pi * ((t_t - 12) / 12)
    cosTheta = math.sin(delta) * math.sin(radLat) + math.cos(delta) * math.cos(radLat) * math.cos(solarHour)
    R_s = r * (S_0 / d**2) * cosTheta
    return max(R_s, 0)


def solar_irradiance(latitude: float, longitude: float, unix_time: datetime.datetime) -> float:
    """Estimate clear-sky solar irradiance in W/m^2."""
    G_sc = 1367
    day_of_year, local_solar_time = unix_to_day_of_year_and_lst(unix_time, longitude)
    delta = math.radians(23.45) * math.sin(math.radians(360 / 365 * (284 + day_of_year)))
    H = math.radians(15 * (local_solar_time - 12))
    phi = math.radians(latitude)
    sin_alpha = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(H)
    AM = 1 / sin_alpha if sin_alpha > 0 else float("inf")
    G_0 = G_sc * (1 + 0.033 * math.cos(math.radians(360 * day_of_year / 365)))
    return G_0 * sin_alpha * math.exp(-0.14 * AM) if sin_alpha > 0 else 0


def calculate_globe_temperature(
    air_temperature: float,
    solar_radiation: float,
    wind_speed: float,
    globe_diameter: float = 0.15,
    emissivity: float = 0.95,
) -> float:
    """Estimate globe temperature used for WBGT calculations."""
    return air_temperature + (1.5 * 10**8 * (solar_radiation ** 0.6)) / (
        emissivity * (globe_diameter ** 0.4) * (wind_speed ** 0.6)
    )


def calculate_wbgt(
    temperature: float,
    humidity: float,
    wind_speed: float | None = None,
    solar_radiation: float | None = None,
    globe_temperature: float | None = None,
    in_sun: bool = False,
) -> float:
    """Calculate the Wet-Bulb Globe Temperature (WBGT)."""
    if in_sun:
        if globe_temperature is None:
            if wind_speed is None or solar_radiation is None:
                raise ValueError(
                    "Wind speed and solar radiation must be provided if globe temperature is not provided for outdoor WBGT calculation."
                )
            globe_temperature = calculate_globe_temperature(temperature, solar_radiation, wind_speed)
        wbgt = 0.7 * temperature + 0.2 * globe_temperature + 0.1 * wind_speed
    else:
        wbgt = 0.7 * temperature + 0.3 * (humidity / 100.0 * temperature)
    return wbgt



def map_times_to_day_indices(times: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return the day index each time belongs to based on ascending boundaries."""
    idx = np.searchsorted(boundaries, times, side="right") - 1
    return np.clip(idx, 0, len(boundaries) - 2).astype(int)
