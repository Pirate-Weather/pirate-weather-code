import datetime
import logging
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
from pytz import timezone, utc
from starlette.middleware.base import BaseHTTPMiddleware

from API.constants.api_const import (
    DBZ_CONST,
    DBZ_CONVERSION_CONST,
    DEFAULT_ROUNDING_INTERVAL,
    GLOBE_TEMP_CONST,
    LAMBERT_CONST,
    SOLAR_CALC_CONST,
    SOLAR_IRRADIANCE_CONST,
    SOLAR_RAD_CONST,
    UNIT_CONVERSION_CONST,
    WBGT_CONST,
    WBGT_PERCENTAGE_DIVISOR,
)
from API.constants.grid_const import US_BOUNDING_BOX
from API.constants.shared_const import MISSING_DATA, REFC_THRESHOLD

TIMING = os.environ.get("TIMING", False)
logger = logging.getLogger("pirate-weather-api")


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        total_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Server-Time"] = f"{total_ms:.1f}"
        return response


def solar_rad(D_t, lat, t_t):
    """
    returns The theortical clear sky short wave radiation
    https://www.mdpi.com/2072-4292/5/10/4735/htm
    """

    d = 1 + SOLAR_RAD_CONST["eccentricity"] * math.sin(
        (2 * math.pi * (D_t - SOLAR_RAD_CONST["offset"])) / 365
    )
    r = SOLAR_RAD_CONST["r"]
    S_0 = SOLAR_RAD_CONST["S0"]
    delta = SOLAR_RAD_CONST["delta_factor"] * math.sin(
        (2 * math.pi * (D_t + SOLAR_RAD_CONST["delta_offset"])) / 365
    )
    rad_lat = np.deg2rad(lat)
    solar_hour = math.pi * ((t_t - SOLAR_RAD_CONST["hour_offset"]) / 12)
    cos_theta = math.sin(delta) * math.sin(rad_lat) + math.cos(delta) * math.cos(
        rad_lat
    ) * math.cos(solar_hour)
    R_s = r * (S_0 / d**2) * cos_theta

    if R_s < 0:
        R_s = 0

    return R_s


def toTimestamp(d):
    """Convert datetime to Unix timestamp.

    Args:
        d: datetime object

    Returns:
        Unix timestamp (float)
    """
    return d.timestamp()


def get_offset(*, lat, lng, utcTime, tf):
    # tf = TimezoneFinder()
    """
    returns a location's time zone offset from UTC in minutes.
    """

    today = utcTime
    tz_target = timezone(tf.timezone_at(lng=lng, lat=lat))
    # ATTENTION: tz_target could be None! handle error case
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

    Args:
        lat_val (float): Latitude in decimal degrees. Positive values are
            north of the equator, negative values are south.
        month_val (int): Month as an integer in the range 1..12.

    Returns:
        bool: True when the (latitude, month) pair corresponds to a
        polar-day season (i.e. the sun would effectively be "always up"
        for that date), False otherwise.

    Notes:
        This is a simple heuristic and does not compute astronomical
        sunrise/sunset times; it is only used as a fallback when
        Astral cannot compute sun times for polar conditions.
    """
    return (lat_val > 0 and month_val >= 4 and month_val <= 9) or (
        lat_val < 0 and (month_val >= 10 or month_val <= 3)
    )


def has_interior_nan_holes(arr: np.ndarray) -> Tuple[bool, Optional[int]]:
    """
    Detect an interior block of NaNs in a 2D array.

    Args:
        arr (np.ndarray): Array shaped as ``rows × cols``.

    Returns:
        Tuple[bool, Optional[int]]: ``(True, row_index)`` if a contiguous
        NaN block that does *not* touch the first or last column is found in
        the specified 0-based ``row_index``; otherwise ``(False, None)``.
    """
    # 1) make a mask of NaNs
    mask = np.isnan(arr)

    # 2) pad left/right with False so that edges never count as run boundaries
    #    padded.shape == (rows, cols+2)
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)

    # 3) compute a 1D diff along each row:
    #    diff == +1  → run *start* (False→True)
    #    diff == -1  → run *end*   (True→False)
    #    diff.shape == (rows, cols+1)
    diff = padded[:, 1:].astype(int) - padded[:, :-1].astype(int)
    starts = diff == 1  # potential run‐starts
    ends = diff == -1  # potential run‐ends

    # 4) ignore any that occur at the very first or last original column:
    #    we only want starts/ends in columns 1…(cols-2)
    interior_starts = starts[:, 1:-1]
    interior_ends = ends[:, 1:-1]

    # 5) a row has an interior hole iff it has at least one interior start
    #    *and* at least one interior end.  If any row meets that, we’re done.
    row_has_start = interior_starts.any(axis=1)
    row_has_end = interior_ends.any(axis=1)

    matching_rows = np.flatnonzero(row_has_start & row_has_end)
    if matching_rows.size:
        return True, int(matching_rows[0])

    return False, None


# Interpolation function to interpolate nans in a row, keeping nan's at the start and end
def _interp_row(row: np.ndarray) -> np.ndarray:
    """
    Fill only strictly interior NaN‐runs in a 1D array
    (i.e. ignore any NaNs at index 0 or -1) by linear interpolation.
    """
    n = row.size
    x = np.arange(n)

    # mask of all NaNs
    mask = np.isnan(row)

    if mask.any() and not mask.all():
        good = ~mask

        # interp only at mask positions, using the remaining points
        row[mask] = np.interp(
            x[mask], x[good], row[good], left=MISSING_DATA, right=MISSING_DATA
        )

    return row


def cull(lng, lat):
    """Accepts a list of lat/lng tuples.
    returns the list of tuples that are within the bounding box for the US.
    NB. THESE ARE NOT NECESSARILY WITHIN THE US BORDERS!
    https://gist.github.com/jsundram/1251783
    """

    ### TODO: Add Alaska somehow

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
    """Rounds a datetime object to the nearest interval in minutes.

    Parameters:
        t (datetime.datetime): The datetime to round.
        to (int): The interval in minutes to round to (e.g., 60 for hour, 15 for quarter-hour).

    Returns:
        datetime.datetime: The rounded datetime.
    """
    discard = datetime.timedelta(
        minutes=t.minute % to, seconds=t.second, microseconds=t.microsecond
    )
    t -= discard
    if discard >= datetime.timedelta(minutes=to / 2):
        t += datetime.timedelta(minutes=to)
    return t.replace(second=0, microsecond=0)


def unix_to_day_of_year_and_lst(dt, longitude):
    """Convert Unix time to day of year and local solar time.

    Args:
        dt: datetime object
        longitude: Longitude in degrees

    Returns:
        tuple: (day_of_year, local_solar_time)
    """
    # Calculate the day of the year
    day_of_year = dt.timetuple().tm_yday

    # Calculate UTC time in hours
    utc_time = (
        dt.hour
        + dt.minute / UNIT_CONVERSION_CONST["hours_to_minutes"]
        + dt.second / UNIT_CONVERSION_CONST["seconds_to_hours"]
    )
    if TIMING:
        logger.debug(f"UTC time: {utc_time}")

    # Calculate Local Solar Time (LST) considering the longitude
    lst = utc_time + (longitude / UNIT_CONVERSION_CONST["longitude_to_hours"])
    if TIMING:
        logger.debug(f"LST: {lst}")

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


def calculate_globe_temperature(
    air_temperature, solar_radiation, wind_speed, globe_diameter=0.15, emissivity=0.95
):
    """
    Estimate the globe temperature based on ambient temperature, solar radiation, and wind speed.

    Parameters:
    air_temperature (float): Ambient air temperature in degrees Celsius.
    solar_radiation (float): Solar radiation in watts per square meter (W/m²).
    wind_speed (float): Wind speed in meters per second (m/s).
    globe_diameter (float, optional): Diameter of the globe thermometer in meters (default is 0.15m).
    emissivity (float, optional): Emissivity of the globe (default is 0.95 for a black globe).

    Returns:
    float: Estimated globe temperature in degrees Celsius.
    """
    globe_temperature = air_temperature + (
        GLOBE_TEMP_CONST["factor"] * (solar_radiation ** GLOBE_TEMP_CONST["temp_exp"])
    ) / (
        emissivity
        * (globe_diameter ** GLOBE_TEMP_CONST["diam_exp"])
        * (wind_speed ** GLOBE_TEMP_CONST["wind_exp"])
    )
    return globe_temperature


def calculate_wbgt(
    temperature,
    humidity,
    wind_speed=None,
    solar_radiation=None,
    globe_temperature=None,
    in_sun=False,
):
    """
    Calculate the Wet-Bulb Globe Temperature (WBGT).

    Parameters:
    temperature (float): The ambient air temperature in degrees Celsius.
    humidity (float): The relative humidity as a percentage (0-100).
    wind_speed (float, optional): The wind speed in meters per second. Required if `in_sun` is True.
    solar_radiation (float, optional): Solar radiation in watts per square meter (W/m²). Used to calculate globe temperature if `globe_temperature` is not provided.
    globe_temperature (float, optional): The globe temperature in degrees Celsius. Required if `in_sun` is True and `solar_radiation` is not provided.
    in_sun (bool, optional): If True, calculates WBGT for sunny conditions using wind_speed and globe_temperature.

    Returns:
    float: The Wet-Bulb Globe Temperature in degrees Celsius.
    """
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


def dbz_to_rate(dbz_array, precip_type_array, min_dbz=REFC_THRESHOLD):
    """
    Convert dBZ to precipitation rate (mm/h) using a Z-R relationship with soft threshold.

    Args:
        dbz_array (np.ndarray): Radar reflectivity in dBZ.
        precip_type_array (np.ndarray): Array of precipitation types ('rain' or 'snow').
        min_dbz (float): Minimum dBZ for soft thresholding. Values below this are scaled linearly.

    Returns:
        np.ndarray: Precipitation rate in mm/h.
    """
    # Ensure no negative dBZ values
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
