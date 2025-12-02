"""Geospatial helpers shared across forecast builders."""

from __future__ import annotations

import datetime
import math

from pytz import timezone, utc
from timezonefinder import TimezoneFinder

from API.constants.api_const import (
    DEFAULT_ROUNDING_INTERVAL,
    LAMBERT_CONST,
    SOLAR_CALC_CONST,
    SOLAR_IRRADIANCE_CONST,
    UNIT_CONVERSION_CONST,
)
from API.constants.grid_const import US_BOUNDING_BOX


def get_offset(*, lat, lng, utcTime, tf: TimezoneFinder):
    """
    returns a location's time zone offset from UTC in minutes.
    """

    today = utcTime
    tz_target = timezone(tf.timezone_at(lng=lng, lat=lat))
    today_target = tz_target.localize(today)
    today_utc = utc.localize(today)
    return (today_utc - today_target).total_seconds() / UNIT_CONVERSION_CONST[
        "seconds_to_minutes"
    ], tz_target


def _polar_is_all_day(lat_val: float, month_val: int) -> bool:
    """Determine whether a given latitude and month fall inside the
    "polar day" season.

    This helper encapsulates the heuristic used for locations inside the
    polar circles where the Astral library may raise a ``ValueError``
    (sun never rises or never sets). The heuristic is based on the
    hemisphere and month:

    - Northern hemisphere (lat > 0): months April (4) through September (9)
      are treated as the polar-day season.
    - Southern hemisphere (lat < 0): months October (10) through March (3)
      are treated as the polar-day season.
    """
    return (lat_val > 0 and month_val >= 4 and month_val <= 9) or (
        lat_val < 0 and (month_val >= 10 or month_val <= 3)
    )


def cull(lng, lat):
    """Accepts a list of lat/lng tuples.
    returns the list of tuples that are within the bounding box for the US.
    NB. THESE ARE NOT NECESSARILY WITHIN THE US BORDERS!
    https://gist.github.com/jsundram/1251783
    """

    top = US_BOUNDING_BOX["top"]
    left = US_BOUNDING_BOX["left"]
    right = US_BOUNDING_BOX["right"]
    bottom = US_BOUNDING_BOX["bottom"]

    inside_box = 0
    if (bottom <= lat <= top) and (left <= lng <= right):
        inside_box = 1

    return inside_box


def lambertGridMatch(
    central_longitude,
    central_latitude,
    standard_parallel,
    semimajor_axis,
    lat,
    lon,
    hrrr_minX,
    hrrr_minY,
    hrrr_delta,
):
    # From https://en.wikipedia.org/wiki/Lambert_conformal_conic_projection

    hrr_n = math.sin(standard_parallel)
    hrrr_F = (
        math.cos(standard_parallel)
        * (
            math.tan(
                LAMBERT_CONST["pi_factor"] * math.pi
                + LAMBERT_CONST["half_pi_factor"] * standard_parallel
            )
        )
        ** hrr_n
    ) / hrr_n
    hrrr_p = (
        semimajor_axis
        * hrrr_F
        * 1
        / (
            math.tan(
                LAMBERT_CONST["pi_factor"] * math.pi
                + LAMBERT_CONST["half_pi_factor"] * math.radians(lat)
            )
            ** hrr_n
        )
    )
    hrrr_p0 = (
        semimajor_axis
        * hrrr_F
        * 1
        / (
            math.tan(
                LAMBERT_CONST["pi_factor"] * math.pi
                + LAMBERT_CONST["half_pi_factor"] * central_latitude
            )
            ** hrr_n
        )
    )

    x_hrrrLoc = hrrr_p * math.sin(hrr_n * (math.radians(lon) - central_longitude))
    y_hrrrLoc = hrrr_p0 - hrrr_p * math.cos(
        hrr_n * (math.radians(lon) - central_longitude)
    )

    x_hrrr = round((x_hrrrLoc - hrrr_minX) / hrrr_delta)
    y_hrrr = round((y_hrrrLoc - hrrr_minY) / hrrr_delta)

    x_grid = x_hrrr * hrrr_delta + hrrr_minX
    y_grid = y_hrrr * hrrr_delta + hrrr_minY

    hrrr_p2 = math.copysign(math.sqrt(x_grid**2 + (hrrr_p0 - y_grid) ** 2), hrr_n)

    lat_grid = math.degrees(
        2 * math.atan((semimajor_axis * hrrr_F / hrrr_p2) ** (1 / hrr_n)) - math.pi / 2
    )

    hrrr_theta = math.atan((x_grid) / (hrrr_p0 - y_grid))

    lon_grid = math.degrees(central_longitude + hrrr_theta / hrr_n)

    return lat_grid, lon_grid, x_hrrr, y_hrrr


def rounder(
    t: datetime.datetime, to: int = DEFAULT_ROUNDING_INTERVAL
) -> datetime.datetime:
    """Rounds a datetime object to the nearest interval in minutes."""
    discard = datetime.timedelta(
        minutes=t.minute % to, seconds=t.second, microseconds=t.microsecond
    )
    t -= discard
    if discard >= datetime.timedelta(minutes=to / 2):
        t += datetime.timedelta(minutes=to)
    return t.replace(second=0, microsecond=0)


def unix_to_day_of_year_and_lst(dt, longitude):
    """Convert Unix time to day of year and local solar time."""
    day_of_year = dt.timetuple().tm_yday

    utc_time = (
        dt.hour
        + dt.minute / UNIT_CONVERSION_CONST["hours_to_minutes"]
        + dt.second / UNIT_CONVERSION_CONST["seconds_to_hours"]
    )

    lst = utc_time + (longitude / UNIT_CONVERSION_CONST["longitude_to_hours"])
    return day_of_year, lst


def solar_irradiance(latitude, longitude, unix_time):
    G_sc = SOLAR_IRRADIANCE_CONST["GSC"]

    # Get the day of the year and Local Solar Time (LST)
    day_of_year, local_solar_time = unix_to_day_of_year_and_lst(unix_time, longitude)

    # Calculate solar declination (delta) in radians
    delta = math.radians(SOLAR_IRRADIANCE_CONST["declination"]) * math.sin(
        math.radians(
            SOLAR_CALC_CONST["degrees_per_year"]
            / SOLAR_CALC_CONST["days_per_year"]
            * (SOLAR_CALC_CONST["day_of_year_base"] + day_of_year)
        )
    )

    # Calculate hour angle (H) in degrees, then convert to radians
    H = math.radians(
        SOLAR_CALC_CONST["hour_factor"]
        * (local_solar_time - SOLAR_CALC_CONST["hour_offset"])
    )

    # Convert latitude to radians
    phi = math.radians(latitude)

    # Calculate solar elevation angle (alpha)
    sin_alpha = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(
        delta
    ) * math.cos(H)

    # Calculate air mass (AM)
    AM = 1 / sin_alpha if sin_alpha > 0 else float("inf")
    G_0 = G_sc * (
        1
        + SOLAR_IRRADIANCE_CONST["g0_coeff"]
        * math.cos(
            math.radians(
                SOLAR_CALC_CONST["degrees_per_year"]
                * day_of_year
                / SOLAR_CALC_CONST["days_per_year"]
            )
        )
    )
    G = (
        G_0 * sin_alpha * math.exp(-SOLAR_IRRADIANCE_CONST["am_coeff"] * AM)
        if sin_alpha > 0
        else 0
    )

    return G
