"""Geospatial helpers shared across forecast builders."""

from __future__ import annotations

import datetime
import functools
import math

import reverse_geocode
from timezonefinder import TimezoneFinder

from API.constants.api_const import (
    DEFAULT_ROUNDING_INTERVAL,
    LAMBERT_CONST,
    SOLAR_CALC_CONST,
    SOLAR_IRRADIANCE_CONST,
    UNIT_CONVERSION_CONST,
)
from API.constants.grid_const import US_BOUNDING_BOX

# Time zones that are only in Canada (tz database zone.tab). Canadian-first source
# priority is decided from these.
CANADA_TIMEZONES = frozenset(
    {
        "America/Atikokan",
        "America/Blanc-Sablon",
        "America/Cambridge_Bay",
        "America/Creston",
        "America/Dawson",
        "America/Dawson_Creek",
        "America/Edmonton",
        "America/Fort_Nelson",
        "America/Glace_Bay",
        "America/Goose_Bay",
        "America/Halifax",
        "America/Inuvik",
        "America/Iqaluit",
        "America/Moncton",
        "America/Rankin_Inlet",
        "America/Regina",
        "America/Resolute",
        "America/St_Johns",
        "America/Swift_Current",
        "America/Toronto",
        "America/Vancouver",
        "America/Whitehorse",
        "America/Winnipeg",
    }
)
# Alaska west of the North America box: the western Aleutians, across the
# antimeridian to Attu (172.4 E), and St Lawrence Island.
ALASKA_WEST_TIMEZONES = frozenset({"America/Adak", "America/Nome"})
ALASKA_WEST_MIN_LAT = 50.0
ALASKA_WEST_MIN_EAST_LON = 170.0
# is_in_canada is called many times per request, for a few points at a time
REGION_CACHE_POINTS = 256


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Uses the Haversine formula to calculate the distance in kilometers.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert coordinates from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def get_offset(*, lat, lng, utc_time, tf: TimezoneFinder):
    """
    Get a location's time zone offset from UTC in minutes.

    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        utc_time (datetime.datetime): The UTC datetime to convert.
        tf (TimezoneFinder): TimezoneFinder instance.

    Returns:
        tuple: (offset_minutes (float), tz_target (pytz.timezone))
            offset_minutes: The time zone offset from UTC in minutes.
            tz_target: The pytz timezone object for the location.
    """

    from pytz import timezone, utc

    today = utc_time
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

    Accepts individual longitude and latitude values.
    Returns 1 if the point is within the bounding box for the US, 0 otherwise.
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


@functools.cache
def get_timezone_finder() -> TimezoneFinder:
    """The process's TimezoneFinder (about 34 MB in memory, read-only once built)."""
    return TimezoneFinder(in_memory=True)


def is_in_north_america(lat: float, lon: float) -> bool:
    """
    Determine if a location is within North America (USA, Canada, Mexico).

    Args:
        lat (float): Latitude in degrees (-90 to 90).
        lon (float): Longitude in degrees (-180 to 180).

    Returns:
        bool: True if the location is in North America, False otherwise.
    """
    # Normalize longitude to -180 to 180 range
    lon_normalized = ((lon + 180) % 360) - 180

    # Main North America bounding box
    # Includes USA (including Alaska), Canada, and Mexico
    # Latitude: 14°N (southern Mexico) to 83°N (northern Canada/Greenland)
    # Longitude: -168°W (western Alaska) to -52°W (eastern Newfoundland)
    # Small Caribbean territories (e.g., Navassa Island) are within the box.
    if 14.0 <= lat <= 83.0 and -168.0 <= lon_normalized <= -52.0:
        return True

    # Alaska continues west of the box, across the antimeridian to Attu. A box
    # can't separate it from Chukotka and the US Minor Outlying Islands (Midway,
    # Wake), but their time zones do.
    return (
        lat >= ALASKA_WEST_MIN_LAT
        and (lon_normalized < -168.0 or lon_normalized >= ALASKA_WEST_MIN_EAST_LON)
        and get_timezone_finder().timezone_at(lat=lat, lng=lon_normalized)
        in ALASKA_WEST_TIMEZONES
    )


@functools.lru_cache(maxsize=REGION_CACHE_POINTS)
def is_in_canada(lat: float, lon: float) -> bool:
    """Determine if a location is in Canada, for Canadian-first source priority.

    Time zone polygons follow the national borders, including the Great Lakes, and
    the API already looks them up on every request (about 2 us), so this needs no
    other geometry. Open water is in an Etc/GMT zone, so offshore points take the
    country of the nearest city. The bounding box skips both lookups for most of
    the world. Called many times per request, hence the cache.
    """
    lon_normalized = ((lon + 180) % 360) - 180
    if not (41.7 <= lat <= 83.0 and -141.0 <= lon_normalized <= -52.0):
        return False

    zone = get_timezone_finder().timezone_at(lat=lat, lng=lon_normalized)
    if zone is None or zone.startswith("Etc/"):
        return reverse_geocode.get((lat, lon_normalized))["country_code"] == "CA"
    return zone in CANADA_TIMEZONES
