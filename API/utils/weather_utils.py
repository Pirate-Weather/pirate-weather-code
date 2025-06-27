"""Reusable weather math utilities."""

from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Tuple

import numpy as np
from astral import LocationInfo, moon
from astral.sun import sun
from pytz import timezone, utc
from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

tf = TimezoneFinder(in_memory=True)


def solar_rad(day_of_year: int, latitude: float, hour: float) -> float:
    """Return theoretical clear sky short wave radiation."""
    d = 1 + 0.0167 * math.sin((2 * math.pi * (day_of_year - 93.5365)) / 365)
    r = 0.75
    s0 = 1367
    delta = 0.4096 * math.sin((2 * math.pi * (day_of_year + 284)) / 365)
    rad_lat = np.deg2rad(latitude)
    solar_hour = math.pi * ((hour - 12) / 12)
    cos_theta = (
        math.sin(delta) * math.sin(rad_lat)
        + math.cos(delta) * math.cos(rad_lat) * math.cos(solar_hour)
    )
    rs = r * (s0 / d ** 2) * cos_theta
    return max(rs, 0)


def to_timestamp(d: _dt.datetime) -> float:
    """Convenience wrapper for ``datetime.timestamp``."""
    return d.timestamp()


def get_offset(*, lat: float, lng: float, utcTime: _dt.datetime, tf: TimezoneFinder = tf) -> Tuple[float, timezone]:
    """Return timezone offset in minutes and tz object."""
    tz_target = timezone(tf.timezone_at(lng=lng, lat=lat))
    today_target = tz_target.localize(utcTime)
    today_utc = utc.localize(utcTime)
    return (today_utc - today_target).total_seconds() / 60, tz_target


def array_interp(hour_array_grib: np.ndarray, modelData: np.ndarray, modelIndex: int) -> np.ndarray:
    """Simple wrapper around ``np.interp``."""
    return np.interp(
        hour_array_grib,
        modelData[:, 0],
        modelData[:, modelIndex],
        left=np.nan,
        right=np.nan,
    )


def cull(lng: float, lat: float) -> int:
    """Return 1 if coordinates fall within the CONUS bounding box."""
    top = 49.3457868
    left = -124.7844079
    right = -66.9513812
    bottom = 24.7433195
    return int(bottom <= lat <= top and left <= lng <= right)


def lambert_grid_match(
    central_longitude: float,
    central_latitude: float,
    standard_parallel: float,
    semimajor_axis: float,
    lat: float,
    lon: float,
    min_x: float,
    min_y: float,
    delta: float,
) -> Tuple[float, float, int, int]:
    """Convert lat/lon to Lambert conformal grid coordinates."""
    hrr_n = math.sin(standard_parallel)
    hrr_f = (
        math.cos(standard_parallel)
        * (math.tan(0.25 * math.pi + 0.5 * standard_parallel)) ** hrr_n
    ) / hrr_n
    hrr_p = semimajor_axis * hrr_f / (
        math.tan(0.25 * math.pi + 0.5 * math.radians(lat)) ** hrr_n
    )
    hrr_p0 = semimajor_axis * hrr_f / (
        math.tan(0.25 * math.pi + 0.5 * central_latitude) ** hrr_n
    )
    x_loc = hrr_p * math.sin(hrr_n * (math.radians(lon) - central_longitude))
    y_loc = hrr_p0 - hrr_p * math.cos(
        hrr_n * (math.radians(lon) - central_longitude)
    )
    x_idx = round((x_loc - min_x) / delta)
    y_idx = round((y_loc - min_y) / delta)

    return (
        math.degrees(
            2 * math.atan((semimajor_axis * hrr_f / math.copysign(math.sqrt(x_loc ** 2 + (hrr_p0 - y_loc) ** 2), hrr_n)) ** (1 / hrr_n)) - math.pi / 2
        ),
        math.degrees(central_longitude + math.atan(x_loc / (hrr_p0 - y_loc)) / hrr_n),
        x_idx,
        y_idx,
    )


def rounder(t: _dt.datetime) -> _dt.datetime:
    """Round ``t`` to the nearest hour."""
    if t.minute >= 30:
        return t.replace(second=0, microsecond=0, minute=0) + _dt.timedelta(hours=1)
    return t.replace(second=0, microsecond=0, minute=0)


def unix_to_day_of_year_and_lst(dt: _dt.datetime, longitude: float) -> Tuple[int, float]:
    day_of_year = dt.timetuple().tm_yday
    utc_time = dt.hour + dt.minute / 60 + dt.second / 3600
    lst = utc_time + (longitude / 15)
    return day_of_year, lst


def solar_irradiance(latitude: float, longitude: float, unix_time: _dt.datetime) -> float:
    g_sc = 1367
    day_of_year, local_solar_time = unix_to_day_of_year_and_lst(unix_time, longitude)
    delta = math.radians(23.45) * math.sin(math.radians(360 / 365 * (284 + day_of_year)))
    h = math.radians(15 * (local_solar_time - 12))
    phi = math.radians(latitude)
    sin_alpha = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)
    am = 1 / sin_alpha if sin_alpha > 0 else float("inf")
    g0 = g_sc * (1 + 0.033 * math.cos(math.radians(360 * day_of_year / 365)))
    return g0 * sin_alpha * math.exp(-0.14 * am) if sin_alpha > 0 else 0


def calculate_globe_temperature(
    air_temperature: float,
    solar_radiation: float,
    wind_speed: float,
    globe_diameter: float = 0.15,
    emissivity: float = 0.95,
) -> float:
    """Estimate the globe temperature given solar radiation and wind."""
    return air_temperature + (1.5 * 10 ** 8 * (solar_radiation ** 0.6)) / (
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
    """Calculate the Wet-Bulb Globe Temperature."""
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


def clip_log(data: np.ndarray | float, min_val: float, max_val: float, name: str) -> np.ndarray | float:
    """Clamp ``data`` between ``min_val`` and ``max_val`` logging when clipping occurs."""
    arr = np.array(data)
    if arr.min() < min_val:
        logger.error("Min clipping required for %s", name)
        if arr.size > 1:
            logger.error("Min Value: %s", arr.min())
            logger.error("Min Index: %s", np.where(arr == arr.min()))
    if arr.max() > max_val:
        logger.error("Max clipping required for %s", name)
        if arr.size > 1:
            logger.error("Max Value: %s", arr.max())
            logger.error("Max Index: %s", np.where(arr == arr.max()))
    clipped = np.clip(arr, min_val, max_val)
    return clipped if isinstance(data, np.ndarray) else float(clipped)


def calculate_apparent_temperature(air_temp: float, humidity: float, wind: float) -> float:
    """Return the apparent temperature in Kelvin."""
    air_temp_c = air_temp - 273.15
    e = humidity * 6.105 * np.exp(17.27 * air_temp_c / (237.7 + air_temp_c))
    apparent_c = air_temp_c + 0.33 * e - 0.70 * wind - 4.00
    return clip_log(apparent_c + 273.15, -183, 333, "Apparent Temperature Current")
