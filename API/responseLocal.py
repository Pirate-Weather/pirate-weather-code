"""
Response Local Module for Pirate Weather API.

This module handles the local weather data processing and API responses.
It includes functions for reading weather data from zarr files, processing
weather forecasts, and generating API responses.
"""

import asyncio
import datetime
import logging

# Standard library imports
import math
import os
import platform
import re
import sys
import threading
import time
from collections import Counter
from typing import Union

# Third-party imports
import metpy as mp
import numpy as np
import reverse_geocode
import s3fs
import xarray as xr
import zarr
from astral import LocationInfo, moon
from astral.sun import sun
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from metpy.calc import relative_humidity_from_dewpoint
from pirateweather_translations.dynamic_loader import load_all_translations
from pytz import timezone, utc
from starlette.middleware.base import BaseHTTPMiddleware
from timezonefinder import TimezoneFinder

from API.api_utils import (
    calculate_apparent_temperature,
    clipLog,
    estimate_visibility_gultepe_rh_pr_numpy,
    fast_nearest_interp,
    replace_nan,
    select_daily_precip_type,
)
from API.constants.api_const import (
    API_VERSION,
    COORDINATE_CONST,
    DBZ_CONST,
    DBZ_CONVERSION_CONST,
    DEFAULT_ROUNDING_INTERVAL,
    ETOPO_CONST,
    GLOBE_TEMP_CONST,
    LAMBERT_CONST,
    MAX_ZARR_READ_RETRIES,
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    ROUNDING_RULES,
    SOLAR_CALC_CONST,
    SOLAR_IRRADIANCE_CONST,
    SOLAR_RAD_CONST,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
    TEMPERATURE_UNITS_THRESH,
    TIME_MACHINE_CONST,
    UNIT_CONVERSION_CONST,
    WBGT_CONST,
    WBGT_PERCENTAGE_DIVISOR,
)
from API.constants.clip_const import (
    CLIP_CAPE,
    CLIP_CLOUD,
    CLIP_FEELS_LIKE,
    CLIP_FIRE,
    CLIP_GLOBAL,
    CLIP_HUMIDITY,
    CLIP_OZONE,
    CLIP_PRESSURE,
    CLIP_PROB,
    CLIP_SMOKE,
    CLIP_SOLAR,
    CLIP_TEMP,
    CLIP_UV,
    CLIP_VIS,
    CLIP_WIND,
)
from API.constants.forecast_const import (
    DATA_CURRENT,
    DATA_DAY,
    DATA_HOURLY,
    DATA_MINUTELY,
)
from API.constants.grid_const import (
    HRRR_X_MAX,
    HRRR_X_MIN,
    HRRR_Y_MAX,
    HRRR_Y_MIN,
    NBM_X_MAX,
    NBM_X_MIN,
    NBM_Y_MAX,
    NBM_Y_MIN,
    RTMA_RU_AXIS,
    RTMA_RU_CENTRAL_LAT,
    RTMA_RU_CENTRAL_LONG,
    RTMA_RU_DELTA,
    RTMA_RU_MIN_X,
    RTMA_RU_MIN_Y,
    RTMA_RU_PARALLEL,
    RTMA_RU_X_MAX,
    RTMA_RU_X_MIN,
    RTMA_RU_Y_MAX,
    RTMA_RU_Y_MIN,
    US_BOUNDING_BOX,
)

# Project imports
from API.constants.model_const import (
    ECMWF,
    ERA5,
    GEFS,
    GFS,
    HRRR,
    HRRR_SUBH,
    NBM,
    NBM_FIRE_INDEX,
    RTMA_RU,
)
from API.constants.shared_const import (
    HISTORY_PERIODS,
    INGEST_VERSION_STR,
    KELVIN_TO_CELSIUS,
    MISSING_DATA,
    REFC_THRESHOLD,
)
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    PRECIP_PROB_THRESHOLD,
    WIND_THRESHOLDS,
)
from API.constants.unit_const import country_units
from API.PirateDailyText import calculate_day_text
from API.PirateDayNightText import calculate_half_day_text
from API.PirateMinutelyText import calculate_minutely_text
from API.PirateText import calculate_text
from API.PirateTextHelper import estimate_snow_height
from API.PirateWeeklyText import calculate_weekly_text
from API.ZarrHelpers import (
    _add_custom_header,
    init_ERA5,
    setup_testing_zipstore,
)

Translations = load_all_translations()

lock = threading.Lock()

# Keep these for TESTING/TM_TESTING stages that still use S3
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
s3_bucket = os.getenv("s3_bucket", default="piratezarr2")
ingest_version = INGEST_VERSION_STR
save_type = os.getenv("save_type", default="S3")

pw_api_key = os.environ.get("PW_API", "")
save_dir = os.getenv("save_dir", default="/tmp")
use_etopo = os.getenv("use_etopo", default=True)
TIMING = os.environ.get("TIMING", False)

force_now = os.getenv("force_now", default=False)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    # include timestamp, level, logger name, module, line number, message
    fmt = "%(asctime)s %(levelname)s [%(name)s:%(module)s:%(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


# Define TimingMiddleware for performance measurement
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        total_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Server-Time"] = f"{total_ms:.1f}"
        return response


logger = logging.getLogger("pirate-weather-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Initialize Zarr stores
ETOPO_f = None
SubH_Zarr = None
HRRR_6H_Zarr = None
GFS_Zarr = None
ECMWF_Zarr = None
NBM_Zarr = None
NBM_Fire_Zarr = None
GEFS_Zarr = None
HRRR_Zarr = None
NWS_Alerts_Zarr = None
WMO_Alerts_Zarr = None
RTMA_RU_Zarr = None
ERA5_Data = None


def update_zarr_store(initialRun):
    """Load zarr data stores from static file paths.

    File syncing and download is now handled by a separate container.
    This function simply opens the zarr stores at their expected paths.

    Args:
        initialRun: Whether this is the initial run (kept for compatibility)
    """
    global ETOPO_f
    global SubH_Zarr
    global HRRR_6H_Zarr
    global GFS_Zarr
    global ECMWF_Zarr
    global NBM_Zarr
    global NBM_Fire_Zarr
    global GEFS_Zarr
    global HRRR_Zarr
    global NWS_Alerts_Zarr
    global WMO_Alerts_Zarr
    global RTMA_RU_Zarr
    global ERA5_Data

    # Get stage
    STAGE = os.environ.get("STAGE", "PROD")

    # Always load GFS
    gfs_path = os.path.join(save_dir, "GFS.zarr")
    if os.path.exists(gfs_path):
        GFS_Zarr = zarr.open(zarr.storage.LocalStore(gfs_path), mode="r")
        logger.info("Loaded GFS from: " + gfs_path)

    # Load ETOPO on initial run if enabled
    if (initialRun) and (use_etopo):
        etopo_path = os.path.join(save_dir, "ETOPO_DA_C.zarr")
        if os.path.exists(etopo_path):
            ETOPO_f = zarr.open(zarr.storage.LocalStore(etopo_path), mode="r")
            logger.info("Loaded ETOPO from: " + etopo_path)

    # Open the Google ERA5 dataset for Dev and TimeMachine
    if STAGE in ("DEV", "TIMEMACHINE"):
        ERA5_Data = init_ERA5()

    # Don't open the other files in TimeMachine to reduce memory
    if STAGE in ("DEV", "PROD"):
        # Load NWS Alerts
        nws_alerts_path = os.path.join(save_dir, "NWS_Alerts.zarr")
        if os.path.exists(nws_alerts_path):
            NWS_Alerts_Zarr = zarr.open(
                zarr.storage.LocalStore(nws_alerts_path), mode="r"
            )
            logger.info("Loaded NWS_Alerts from: " + nws_alerts_path)

        # Load SubH
        subh_path = os.path.join(save_dir, "SubH.zarr")
        if os.path.exists(subh_path):
            SubH_Zarr = zarr.open(zarr.storage.LocalStore(subh_path), mode="r")
            logger.info("Loaded SubH from: " + subh_path)

        # Load HRRR_6H
        hrrr_6h_path = os.path.join(save_dir, "HRRR_6H.zarr")
        if os.path.exists(hrrr_6h_path):
            HRRR_6H_Zarr = zarr.open(zarr.storage.LocalStore(hrrr_6h_path), mode="r")
            logger.info("Loaded HRRR_6H from: " + hrrr_6h_path)

        # Load ECMWF
        ecmwf_path = os.path.join(save_dir, "ECMWF.zarr")
        if os.path.exists(ecmwf_path):
            try:
                ECMWF_Zarr = zarr.open(zarr.storage.LocalStore(ecmwf_path), mode="r")
                logger.info("Loaded ECMWF from: " + ecmwf_path)
            except Exception as e:
                logger.info(f"ECMWF not available: {e}")
                ECMWF_Zarr = None

        # Load NBM
        nbm_path = os.path.join(save_dir, "NBM.zarr")
        if os.path.exists(nbm_path):
            NBM_Zarr = zarr.open(zarr.storage.LocalStore(nbm_path), mode="r")
            logger.info("Loaded NBM from: " + nbm_path)

        # Load NBM_Fire
        nbm_fire_path = os.path.join(save_dir, "NBM_Fire.zarr")
        if os.path.exists(nbm_fire_path):
            NBM_Fire_Zarr = zarr.open(zarr.storage.LocalStore(nbm_fire_path), mode="r")
            logger.info("Loaded NBM_Fire from: " + nbm_fire_path)

        # Load GEFS
        gefs_path = os.path.join(save_dir, "GEFS.zarr")
        if os.path.exists(gefs_path):
            GEFS_Zarr = zarr.open(zarr.storage.LocalStore(gefs_path), mode="r")
            logger.info("Loaded GEFS from: " + gefs_path)

        # Load HRRR
        hrrr_path = os.path.join(save_dir, "HRRR.zarr")
        if os.path.exists(hrrr_path):
            HRRR_Zarr = zarr.open(zarr.storage.LocalStore(hrrr_path), mode="r")
            logger.info("Loaded HRRR from: " + hrrr_path)

        # Load WMO_Alerts
        wmo_alerts_path = os.path.join(save_dir, "WMO_Alerts.zarr")
        if os.path.exists(wmo_alerts_path):
            WMO_Alerts_Zarr = zarr.open(
                zarr.storage.LocalStore(wmo_alerts_path), mode="r"
            )
            logger.info("Loaded WMO_Alerts from: " + wmo_alerts_path)

        # Load RTMA_RU
        rtma_ru_path = os.path.join(save_dir, "RTMA_RU.zarr")
        if os.path.exists(rtma_ru_path):
            RTMA_RU_Zarr = zarr.open(zarr.storage.LocalStore(rtma_ru_path), mode="r")
            logger.info("Loaded RTMA_RU from: " + rtma_ru_path)

    logger.info("Zarr stores loaded")


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(TimingMiddleware)


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


# If testing, read zarrs directly from S3 zip files
# This should be implemented as a fallback at some point
STAGE = os.environ.get("STAGE", "PROD")
if (STAGE == "TESTING") or (STAGE == "TM_TESTING"):
    logger.info("Setting up S3 zarrs")
    # If S3, use that, otherwise use local
    if save_type == "S3":
        s3 = s3fs.S3FileSystem(
            anon=True,
            asynchronous=False,
            endpoint_url="https://api.pirateweather.net/files/",
        )
        s3.s3.meta.events.register("before-sign.s3.*", _add_custom_header)
    elif save_type == "S3Zarr":
        s3 = s3fs.S3FileSystem(
            key=aws_access_key_id, secret=aws_secret_access_key, version_aware=True
        )
    else:
        s3 = None

    GFS_store = setup_testing_zipstore(s3, s3_bucket, ingest_version, save_type, "GFS")
    GFS_Zarr = zarr.open(GFS_store, mode="r")
    logger.info("GFS Read")

    ERA5_Data = init_ERA5()
    logger.info("ERA5 Read")

    if STAGE == "TESTING":
        NWS_Alerts_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "NWS_Alerts"
        )
        NWS_Alerts_Zarr = zarr.open(NWS_Alerts_store, mode="r")

        SubH_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "SubH"
        )
        SubH_Zarr = zarr.open(SubH_store, mode="r")
        logger.info("SubH Read")

        HRRR_6H_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "HRRR_6H"
        )
        HRRR_6H_Zarr = zarr.open(HRRR_6H_store, mode="r")
        logger.info("HRRR_6H Read")

        GEFS_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "GEFS"
        )
        GEFS_Zarr = zarr.open(GEFS_store, mode="r")
        logger.info("GEFS Read")

        NBM_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "NBM"
        )
        NBM_Zarr = zarr.open(NBM_store, mode="r")
        logger.info("NBM Read")

        NBM_Fire_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "NBM_Fire"
        )
        NBM_Fire_Zarr = zarr.open(NBM_Fire_store, mode="r")
        logger.info("NBM Fire Read")

        HRRR_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "HRRR"
        )
        HRRR_Zarr = zarr.open(HRRR_store, mode="r")
        logger.info("HRRR Read")

        WMO_Alerts_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "WMO_Alerts"
        )
        WMO_Alerts_Zarr = zarr.open(WMO_Alerts_store, mode="r")
        logger.info("WMO_Alerts Read")

        RTMA_RU_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "RTMA_RU"
        )
        RTMA_RU_Zarr = zarr.open(RTMA_RU_store, mode="r")
        logger.info("RTMA_RU Read")

        ECMWF_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "ECMWF"
        )
        ECMWF_Zarr = zarr.open(ECMWF_store, mode="r")
        logger.info("ECMWF Read")

        if use_etopo:
            ETOPO_store = setup_testing_zipstore(
                s3, s3_bucket, ingest_version, save_type, "ETOPO_DA_C"
            )
            ETOPO_f = zarr.open(ETOPO_store, mode="r")
            logger.info("ETOPO Read")


async def get_zarr(store, X, Y):
    """Asynchronously retrieve zarr data at given coordinates.

    Args:
        store: Zarr store to read from
        X: X coordinate
        Y: Y coordinate

    Returns:
        Zarr data at the specified coordinates
    """
    return store[:, :, X, Y]


lats_etopo = np.arange(
    COORDINATE_CONST["latitude_min"],
    COORDINATE_CONST["latitude_max"],
    ETOPO_CONST["lat_resolution"],
)
lons_etopo = np.arange(
    COORDINATE_CONST["longitude_min"],
    COORDINATE_CONST["longitude_offset"],
    ETOPO_CONST["lon_resolution"],
)

tf = TimezoneFinder(in_memory=True)


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


def has_interior_nan_holes(arr: np.ndarray) -> bool:
    """
    Return True if `arr` (2D: rows × cols) contains at least one
    contiguous block of NaNs that:
      - does *not* touch the first or last column
      - has at least one NaN
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

    return bool(np.any(row_has_start & row_has_end))


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


class WeatherParallel(object):
    """Helper class for parallel zarr reading operations."""

    def __init__(self, loc_tag: str = "") -> None:
        self.loc_tag = loc_tag

    async def zarr_read(self, model, opened_zarr, x, y):
        if TIMING:
            logger.debug(f"### {model} Reading!")
            logger.debug(datetime.datetime.now(datetime.UTC).replace(tzinfo=None))

        err_count = 0
        data_out = False
        # Try to read Zarr file
        while err_count < MAX_ZARR_READ_RETRIES:
            try:
                data_out = await asyncio.to_thread(lambda: opened_zarr[:, :, y, x].T)

                # Check for missing/ bad data and interpolate
                # This should not occur, but good to have a fallback
                if has_interior_nan_holes(data_out.T):
                    logger.warning(f"### {model} Interpolating missing data!")

                    # Print the location of the missing data
                    if TIMING:
                        logger.debug(
                            f"### {model} Missing data at: {np.argwhere(np.isnan(data_out))}"
                        )

                    data_out = np.apply_along_axis(_interp_row, 0, data_out)

                if TIMING:
                    logger.debug(f"### {model} Done!")
                    logger.debug(
                        datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                    )
                return data_out

            except Exception:
                logger.exception("### %s Failure! %s", model, self.loc_tag)
                err_count += 1

        logger.error("### %s Failure! %s", model, self.loc_tag)
        data_out = False
        return data_out


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


@app.get("/timemachine/{apikey}/{location}", response_class=ORJSONResponse)
@app.get("/forecast/{apikey}/{location}", response_class=ORJSONResponse)
async def PW_Forecast(
    request: Request,
    location: str,
    units: Union[str, None] = None,
    extend: Union[str, None] = None,
    exclude: Union[str, None] = None,
    include: Union[str, None] = None,
    lang: Union[str, None] = None,
    version: Union[str, None] = None,
    tmextra: Union[str, None] = None,
    apikey: Union[str, None] = None,
    icon: Union[str, None] = None,
    extraVars: Union[str, None] = None,
) -> dict:
    global ETOPO_f
    global SubH_Zarr
    global HRRR_6H_Zarr
    global GFS_Zarr
    global ECMWF_Zarr
    global NBM_Zarr
    global NBM_Fire_Zarr
    global GEFS_Zarr
    global HRRR_Zarr
    global NWS_Alerts_Zarr
    global WMO_Alerts_Zarr
    global RTMA_RU_Zarr
    global ERA5_Data

    readHRRR = False
    readGFS = False
    readECMWF = False
    readNBM = False
    readGEFS = False
    readERA5 = False

    STAGE = os.environ.get("STAGE", "PROD")

    # Timing Check
    T_Start = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

    # Current time
    if force_now is False:
        nowTime = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    else:
        # Force now for testing with static inputs
        nowTime = datetime.datetime.fromtimestamp(int(force_now), datetime.UTC).replace(
            tzinfo=None
        )

        logger.info("Forced Current Time to:")
        logger.info(nowTime)

    ### If developing in REPL, uncomment to provide static variables
    # location = "47.1756,27.594,1741126460"
    # units = "ca"
    # extend = None
    # exclude = None
    # lang = "en"
    # version = "2"
    # tmextra: None
    # apikey: None

    locationReq = location.split(",")

    # Get the location
    try:
        lat = float(locationReq[0])
        lon_IN = float(locationReq[1])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Location Specification")
        # return {
        #     'statusCode': 400,
        #     'body': json.dumps('Invalid Location Specification')
        # }
    lon = lon_IN % COORDINATE_CONST["longitude_max"]  # 0-360
    az_Lon = (
        (lon + COORDINATE_CONST["longitude_offset"]) % COORDINATE_CONST["longitude_max"]
    ) - COORDINATE_CONST["longitude_offset"]  # -180-180

    if (lon_IN < COORDINATE_CONST["longitude_min"]) or (
        lon > COORDINATE_CONST["longitude_max"]
    ):
        # logger.error('Invalid Longitude')
        raise HTTPException(status_code=400, detail="Invalid Longitude")
    if (lat < COORDINATE_CONST["latitude_min"]) or (
        lat > COORDINATE_CONST["latitude_max"]
    ):
        # logger.error('Invalid Latitude')
        raise HTTPException(status_code=400, detail="Invalid Latitude")

    # Debug tag for logging with location
    loc_tag = f"[loc={lat:.4f},{az_Lon:.4f}]"

    if len(locationReq) == 2:
        if STAGE == "TIMEMACHINE":
            raise HTTPException(status_code=400, detail="Missing Time Specification")

        else:
            utcTime = nowTime

    elif len(locationReq) == 3:
        # If time is specified as a unix time
        if locationReq[2].lstrip("-+").isnumeric():
            if float(locationReq[2]) > 0:
                utcTime = datetime.datetime.fromtimestamp(
                    float(locationReq[2]), datetime.UTC
                ).replace(tzinfo=None)
            elif (
                float(locationReq[2]) < TIME_MACHINE_CONST["very_negative_threshold"]
            ):  # Very negative time
                utcTime = datetime.datetime.fromtimestamp(
                    float(locationReq[2]), datetime.UTC
                ).replace(tzinfo=None)
            elif float(locationReq[2]) < 0:  # Negative time
                utcTime = nowTime + datetime.timedelta(seconds=float(locationReq[2]))

        else:
            try:
                utcTime = datetime.datetime.strptime(
                    locationReq[2], "%Y-%m-%dT%H:%M:%S%z"
                )
                # Since it is in UTC time already
                utcTime = utcTime.replace(tzinfo=None)
            except Exception:
                try:
                    utcTime = datetime.datetime.strptime(
                        locationReq[2], "%Y-%m-%dT%H:%M:%S%Z"
                    )
                    # Since it is in UTC time already
                    utcTime = utcTime.replace(tzinfo=None)
                except Exception:
                    try:
                        localTime = datetime.datetime.strptime(
                            locationReq[2], "%Y-%m-%dT%H:%M:%S"
                        )

                        # If no time zone specified, assume local time, and convert
                        tz_offsetLocIN = {
                            "lat": lat,
                            "lng": az_Lon,
                            "utcTime": localTime,
                            "tf": tf,
                        }

                        tz_offsetIN, tz_name = get_offset(**tz_offsetLocIN)
                        utcTime = localTime - datetime.timedelta(minutes=tz_offsetIN)

                    except Exception:
                        # logger.error('Invalid Time Specification')
                        raise HTTPException(
                            status_code=400, detail="Invalid Time Specification"
                        )

    else:
        raise HTTPException(
            status_code=400, detail="Invalid Time or Location Specification"
        )

    timeMachine = False
    # Set up translations
    if not lang:
        lang = "en"

    if icon != "pirate":
        icon = "darksky"

    # Check if langugage is supported
    if lang not in Translations:
        # Throw an error
        raise HTTPException(status_code=400, detail="Language Not Supported")

    translation = Translations[lang]

    if (nowTime - utcTime) > datetime.timedelta(hours=25):
        # More than 10 days ago must be time machine request
        if (
            ("localhost" in str(request.url))
            or ("timemachine" in str(request.url))
            or ("127.0.0.1" in str(request.url))
            or ("dev" in str(request.url))
        ):
            timeMachine = True

        else:
            raise HTTPException(
                status_code=400,
                detail="Requested Time is in the Past. Please Use Timemachine.",
            )

    elif nowTime < utcTime:
        if (utcTime - nowTime) < datetime.timedelta(hours=1):
            utcTime = nowTime
        else:
            raise HTTPException(
                status_code=400, detail="Requested Time is in the Future"
            )
    elif (nowTime - utcTime) < datetime.timedelta(
        hours=TIME_MACHINE_CONST["threshold_hours"]
    ):
        # If within the last 25 hours, it may or may not be a timemachine request
        # If it is, then only return 24h of data
        if "timemachine" in str(request.url):
            timeMachine = True
            # This results in the API using the live zip file, but only doing a 24 hour forecast from midnight of the requested day
            if TIMING:
                logger.debug("Near term timemachine request")
                # Log how far in the past it is
                logger.debug(nowTime - utcTime)
        # Otherwise, just a normal request

    # Timing Check
    if TIMING:
        logger.debug("Request process time")
        logger.debug(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Calculate the timezone offset
    tz_offsetLoc = {"lat": lat, "lng": az_Lon, "utcTime": utcTime, "tf": tf}
    tz_offset, tz_name = get_offset(**tz_offsetLoc)

    tzReq = tf.timezone_at(lat=lat, lng=az_Lon)

    # Reverse geocode the location to return a city name and approx unit system
    loc_name = await asyncio.to_thread(reverse_geocode.get, (lat, az_Lon))

    # Timing Check
    if TIMING:
        print("Timezone offset time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Set defaults
    if not extend:
        extendFlag = 0
    else:
        if extend == "hourly":
            extendFlag = 1
        else:
            extendFlag = 0

    if not version:
        version = 1

    version = float(version)

    # Check if extra information should be included with time machine
    if not tmextra:
        tmExtra = False
    else:
        tmExtra = True

    if not exclude:
        excludeParams = ""
    else:
        excludeParams = exclude

    if not include:
        includeParams = ""
    else:
        includeParams = include

    if not extraVars:
        extraVars = []
    else:
        extraVars = extraVars.split(",")

    exCurrently = 0
    exMinutely = 0
    exHourly = 0
    exDaily = 0
    exFlags = 0
    exAlerts = 0
    exNBM = 0
    exHRRR = 0
    exGEFS = 0
    exGFS = 0
    exRTMA_RU = 0
    exECMWF = 0
    inc_day_night = 0

    summaryText = True

    if "currently" in excludeParams:
        exCurrently = 1
    if "minutely" in excludeParams:
        exMinutely = 1
    if "hourly" in excludeParams:
        exHourly = 1
    if "daily" in excludeParams:
        exDaily = 1
    if "flags" in excludeParams:
        exFlags = 1
    if "alerts" in excludeParams:
        exAlerts = 1
    if "nbm" in excludeParams:
        exNBM = 1
    if "hrrr" in excludeParams:
        exHRRR = 1
    if "gefs" in excludeParams:
        exGEFS = 1
    if "gfs" in excludeParams:
        exGFS = 1
    if "rtma_ru" in excludeParams:
        exRTMA_RU = 1
    if "ecmwf_ifs" in excludeParams:
        exECMWF = 1
    if "summary" in excludeParams:
        summaryText = False

    if "day_night_forecast" in includeParams:
        inc_day_night = 1

    # If more than 25 hours in the past, exclude everything except gfs
    if (nowTime - utcTime) > datetime.timedelta(hours=25):
        exNBM = 1
        exAlerts = 1
        exHRRR = 1
        exGEFS = 1
        exRTMA_RU = 1
        exECMWF = 1

    readRTMA_RU = False

    if timeMachine or exAlerts == 1:
        readWMOAlerts = False
    else:
        readWMOAlerts = True

    # Set up timemachine params
    if timeMachine and not tmExtra:
        exMinutely = 1

    if timeMachine:
        exAlerts = 1

    # Default to US
    unitSystem = "us"
    windUnit = 2.234  # mph
    prepIntensityUnit = 0.0394  # inches/hour
    prepAccumUnit = 0.0394  # inches
    tempUnits = 0  # F. This is harder
    # pressUnits removed - pressure kept in Pascals, converted to hPa at output
    visUnits = 0.00062137  # miles
    humidUnit = 0.01  # %
    elevUnit = 3.28084  # ft

    if units:
        if units == "auto":
            unitSystem = country_units.get(loc_name["country_code"], "us").lower()
        else:
            unitSystem = units[0:2]

        if unitSystem == "ca":
            windUnit = 3.600  # kph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            # pressUnits removed - pressure kept in Pascals, converted to hPa at output
            visUnits = 0.001  # km
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        elif unitSystem == "uk":
            windUnit = 2.234  # mph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            # pressUnits removed - pressure kept in Pascals, converted to hPa at output
            visUnits = 0.00062137  # miles
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        elif unitSystem == "si":
            windUnit = 1  # m/s
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            # pressUnits removed - pressure kept in Pascals, converted to hPa at output
            visUnits = 0.001  # km
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        else:
            unitSystem = "us"

    weather = WeatherParallel(loc_tag=loc_tag)

    zarrTasks = dict()

    # Base times
    pytzTZ = timezone(tzReq)

    # utcTime  = datetime.datetime(year=2024, month=3, day=8, hour=6, minute=15)
    baseTime = utc.localize(
        datetime.datetime(
            year=utcTime.year,
            month=utcTime.month,
            day=utcTime.day,
            hour=utcTime.hour,
            minute=utcTime.minute,
        )
    ).astimezone(pytzTZ)
    baseHour = pytzTZ.localize(
        datetime.datetime(
            year=baseTime.year,
            month=baseTime.month,
            day=baseTime.day,
            hour=baseTime.hour,
        )
    )

    baseDay = baseTime.replace(hour=0, minute=0, second=0, microsecond=0)

    baseDayUTC = baseDay.astimezone(utc)

    # Find UTC time for the base day
    baseDayUTC_Grib = (
        (
            np.datetime64(baseDay.astimezone(utc).replace(tzinfo=None))
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
        )
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Setup the time parameters for output and processing
    if timeMachine:
        daily_days = 1  # Number of days to output
        daily_day_hours = 1  # Additional hours to use in the processing
        ouputHours = 24
        ouputDays = 1

    else:
        daily_days = 8
        daily_day_hours = 5

        if extendFlag:
            ouputHours = 168
        else:
            ouputHours = 48
        ouputDays = 8

    hour_array = np.arange(
        baseDay.astimezone(utc).replace(tzinfo=None),
        baseDay.astimezone(utc).replace(tzinfo=None)
        + datetime.timedelta(days=daily_days)
        + datetime.timedelta(hours=daily_day_hours),
        datetime.timedelta(hours=1),
    )

    numHours = len(hour_array)

    # Timing Check
    if TIMING:
        print("### HRRR Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    sourceIDX = dict()

    # Ignore areas outside of HRRR coverage
    if (
        az_Lon < -134
        or az_Lon > -61
        or lat < 21
        or lat > 53
        or exHRRR == 1
        or timeMachine
    ):
        dataOut = False
        dataOut_hrrrh = False
        dataOut_h2 = False

    else:
        # HRRR
        central_longitude_hrrr = math.radians(262.5)
        central_latitude_hrrr = math.radians(38.5)
        standard_parallel_hrrr = math.radians(38.5)
        semimajor_axis_hrrr = 6371229
        hrrr_minX = -2697500
        hrrr_minY = -1587300
        hrrr_delta = 3000

        hrrr_lat, hrrr_lon, x_hrrr, y_hrrr = lambertGridMatch(
            central_longitude_hrrr,
            central_latitude_hrrr,
            standard_parallel_hrrr,
            semimajor_axis_hrrr,
            lat,
            lon,
            hrrr_minX,
            hrrr_minY,
            hrrr_delta,
        )

        if (
            (x_hrrr < HRRR_X_MIN)
            or (y_hrrr < HRRR_Y_MIN)
            or (x_hrrr > HRRR_X_MAX)
            or (y_hrrr > HRRR_Y_MAX)
        ):
            dataOut = False
            dataOut_h2 = False
            dataOut_hrrrh = False
        else:
            # Read HRRR if within bounds
            readHRRR = True

        sourceIDX["hrrr"] = dict()
        sourceIDX["hrrr"]["x"] = int(x_hrrr)
        sourceIDX["hrrr"]["y"] = int(y_hrrr)
        sourceIDX["hrrr"]["lat"] = round(hrrr_lat, 2)
        sourceIDX["hrrr"]["lon"] = round(((hrrr_lon + 180) % 360) - 180, 2)

    # Timing Check
    if TIMING:
        print("### RTMA_RU Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # RTMA_RU - only for currently, not for time machine
    # Uses same grid as NBM (Lambert conformal conic projection, ~2.54km resolution)
    if (
        az_Lon < -138.3
        or az_Lon > -59
        or lat < 19.3
        or lat > 57
        or timeMachine
        or exRTMA_RU == 1
    ):
        dataOut_rtma_ru = False
    else:
        # RTMA_RU uses same Lambert Conformal Conic projection as NBM
        central_longitude_rtma = math.radians(RTMA_RU_CENTRAL_LONG)
        central_latitude_rtma = math.radians(RTMA_RU_CENTRAL_LAT)
        standard_parallel_rtma = math.radians(RTMA_RU_PARALLEL)
        semimajor_axis_rtma = RTMA_RU_AXIS
        rtma_minX = RTMA_RU_MIN_X
        rtma_minY = RTMA_RU_MIN_Y
        rtma_delta = RTMA_RU_DELTA  # 2539.703m grid matching NBM

        rtma_lat, rtma_lon, x_rtma, y_rtma = lambertGridMatch(
            central_longitude_rtma,
            central_latitude_rtma,
            standard_parallel_rtma,
            semimajor_axis_rtma,
            lat,
            lon,
            rtma_minX,
            rtma_minY,
            rtma_delta,
        )

        if (
            (x_rtma < RTMA_RU_X_MIN)
            or (y_rtma < RTMA_RU_Y_MIN)
            or (x_rtma > RTMA_RU_X_MAX)
            or (y_rtma > RTMA_RU_Y_MAX)
        ):
            dataOut_rtma_ru = False
        else:
            readRTMA_RU = True

    # Timing Check
    if TIMING:
        print("### NBM Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)
    # Ignore areas outside of NBM coverage
    if (
        az_Lon < -138.3
        or az_Lon > -59
        or lat < 19.3
        or lat > 57
        or exNBM == 1
        or timeMachine
    ):
        dataOut_nbm = False
        dataOut_nbmFire = False
    else:
        # NBM
        central_longitude_nbm = math.radians(265)
        central_latitude_nbm = math.radians(25)
        standard_parallel_nbm = math.radians(25.0)
        semimajor_axis_nbm = 6371200
        nbm_minX = -3271152.8
        nbm_minY = -263793.46
        nbm_delta = 2539.703000

        nbm_lat, nbm_lon, x_nbm, y_nbm = lambertGridMatch(
            central_longitude_nbm,
            central_latitude_nbm,
            standard_parallel_nbm,
            semimajor_axis_nbm,
            lat,
            lon,
            nbm_minX,
            nbm_minY,
            nbm_delta,
        )

        if (
            (x_nbm < NBM_X_MIN)
            or (y_nbm < NBM_Y_MIN)
            or (x_nbm > NBM_X_MAX)
            or (y_nbm > NBM_Y_MAX)
        ):
            dataOut_nbm = False
            dataOut_nbmFire = False
        else:
            # Timing Check
            if TIMING:
                print("### NBM Detail Start ###")
                print(
                    datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
                )

            readNBM = True

    # Timing Check
    if TIMING:
        print("### GFS/GEFS Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # GFS
    lats_gfs = np.arange(-90, 90, 0.25)
    lons_gfs = np.arange(0, 360, 0.25)

    abslat = np.abs(lats_gfs - lat)
    abslon = np.abs(lons_gfs - lon)
    y_p = np.argmin(abslat)
    x_p = np.argmin(abslon)

    gfs_lat = lats_gfs[y_p]
    gfs_lon = lons_gfs[x_p]

    # Timing Check
    if TIMING:
        print("### GFS Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # If more than 10 days ago, read ERA5
    if (nowTime - utcTime) > datetime.timedelta(hours=10 * 24):
        dataOut_gfs = False
        readERA5 = True
        readGFS = False
        exGFS = 1  # Force exclude GFS
    elif exGFS:
        dataOut_gfs = False
        readGFS = False
    else:
        readGFS = True

    # Timing Check
    if TIMING:
        print("### GFS Detail END ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # ECMWF - only for non-timemachine requests and if data is available
    if TIMING:
        print("### ECMWF Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    dataOut_ecmwf = False
    if exECMWF == 1:
        dataOut_ecmwf = False
    elif timeMachine:
        dataOut_ecmwf = False
    elif ECMWF_Zarr is None:
        dataOut_ecmwf = False
    else:
        readECMWF = True
        lats_ecmwf = np.arange(90, -90, -0.25)
        lons_ecmwf = np.arange(-180, 180, 0.25)

        abslat_ecmwf = np.abs(lats_ecmwf - lat)
        abslon_ecmwf = np.abs(lons_ecmwf - az_Lon)
        y_p_eur = np.argmin(abslat_ecmwf)
        x_p_eur = np.argmin(abslon_ecmwf)

    if TIMING:
        print("### ECMWF Detail END ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # GEFS
    # Timing Check
    if TIMING:
        print("### GEFS Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    if exGEFS == 1:
        dataOut_gefs = False
    elif timeMachine:
        dataOut_gefs = False
    else:
        readGEFS = True

    # Timing Check
    if TIMING:
        print("### GEFS Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # If a timemachine request for more than 10 days ago, read ERA5
    if readERA5:
        # Get nearest lat and lon for the ERA5 model

        # Find nearest latitude and longitude in ERA5 data
        # Same as GFS
        abslat = np.abs(ERA5_Data["ERA5_lats"] - lat)
        abslon = np.abs(ERA5_Data["ERA5_lons"] - lon)  # 0-360
        y_p = np.argmin(abslat)
        x_p = np.argmin(abslon)

        # Find closest date to baseDayUTC
        t_p = np.argmin(
            np.abs(
                ERA5_Data["ERA5_times"] - np.datetime64(baseDayUTC.replace(tzinfo=None))
            )
        )

        # Read the ERA5 data for the location and time
        # isel is significantly faster than sel for this operation
        dataOut_ERA5_xr = ERA5_Data["dsERA5"][ERA5.keys()].isel(
            latitude=y_p, longitude=x_p, time=slice(t_p, t_p + 25)
        )

        # Stack into a 2D (var, time) array
        dataOut_ERA5 = xr.concat(
            [dataOut_ERA5_xr[var] for var in ERA5.keys()], dim="variable"
        )

        # Add unix time as first row
        unix_times_era5 = (
            dataOut_ERA5_xr["time"].astype("datetime64[s]")
            - np.datetime64("1970-01-01T00:00:00")
        ).astype(np.int64)
        ERA5_MERGED = np.vstack((unix_times_era5, dataOut_ERA5.values)).T
    else:
        ERA5_MERGED = False
        dataOut_ERA5 = False

    if readHRRR:
        zarrTasks["SubH"] = weather.zarr_read("SubH", SubH_Zarr, x_hrrr, y_hrrr)

        # HRRR_6H
        zarrTasks["HRRR_6H"] = weather.zarr_read(
            "HRRR_6H", HRRR_6H_Zarr, x_hrrr, y_hrrr
        )

        # HRRR
        zarrTasks["HRRR"] = weather.zarr_read("HRRR", HRRR_Zarr, x_hrrr, y_hrrr)

    if readNBM:
        zarrTasks["NBM"] = weather.zarr_read("NBM", NBM_Zarr, x_nbm, y_nbm)
        zarrTasks["NBM_Fire"] = weather.zarr_read(
            "NBM_Fire", NBM_Fire_Zarr, x_nbm, y_nbm
        )

    if readGFS:
        zarrTasks["GFS"] = weather.zarr_read("GFS", GFS_Zarr, x_p, y_p)

    if readECMWF:
        zarrTasks["ECMWF"] = weather.zarr_read("ECMWF", ECMWF_Zarr, x_p_eur, y_p_eur)

    if readGEFS:
        zarrTasks["GEFS"] = weather.zarr_read("GEFS", GEFS_Zarr, x_p, y_p)

    if readRTMA_RU:
        zarrTasks["RTMA_RU"] = weather.zarr_read(
            "RTMA_RU", RTMA_RU_Zarr, x_rtma, y_rtma
        )

    # Initialize WMO alert data
    WMO_alertDat = None

    if readWMOAlerts:
        wmo_alerts_lats = np.arange(-60, 85, 0.0625)
        wmo_alerts_lons = np.arange(-180, 180, 0.0625)
        wmo_abslat = np.abs(wmo_alerts_lats - lat)
        wmo_abslon = np.abs(wmo_alerts_lons - az_Lon)
        wmo_alerts_y_p = np.argmin(wmo_abslat)
        wmo_alerts_x_p = np.argmin(wmo_abslon)

        WMO_alertDat = WMO_Alerts_Zarr[wmo_alerts_y_p, wmo_alerts_x_p]

        if TIMING:
            # Temp until added to response
            print(WMO_alertDat)

    results = await asyncio.gather(*zarrTasks.values())
    zarr_results = {key: result for key, result in zip(zarrTasks.keys(), results)}

    if readHRRR:
        dataOut = zarr_results["SubH"]
        dataOut_h2 = zarr_results["HRRR_6H"]
        dataOut_hrrrh = zarr_results["HRRR"]

        if (
            (dataOut is not False)
            and (dataOut_h2 is not False)
            and (dataOut_hrrrh is not False)
        ):
            # Calculate run times from specific time step for each model
            subhRunTime = dataOut[0, 0]

            # Check if the model times are valid for the request time
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    subhRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=4):
                dataOut = False
                print("OLD SubH")

            hrrrhRunTime = dataOut_hrrrh[HISTORY_PERIODS["HRRR"], 0]
            # print( datetime.datetime.fromtimestamp(dataOut_hrrrh[35, 0].astype(int)))
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    hrrrhRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=16):
                dataOut_hrrrh = False
                print("OLD HRRRH")

            h2RunTime = dataOut_h2[0, 0]
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    h2RunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=46):
                dataOut_h2 = False
                print("OLD HRRR_6H")
        else:  # Set all to false if any failed
            dataOut = False
            dataOut_h2 = False
            dataOut_hrrrh = False

    if readNBM:
        dataOut_nbm = zarr_results["NBM"]
        dataOut_nbmFire = zarr_results["NBM_Fire"]

        if dataOut_nbm is not False:
            nbmRunTime = dataOut_nbm[HISTORY_PERIODS["NBM"], 0]

        sourceIDX["nbm"] = dict()
        sourceIDX["nbm"]["x"] = int(x_nbm)
        sourceIDX["nbm"]["y"] = int(y_nbm)
        sourceIDX["nbm"]["lat"] = round(nbm_lat, 2)
        sourceIDX["nbm"]["lon"] = round(((nbm_lon + 180) % 360) - 180, 2)

        # Timing Check
        if TIMING:
            print("### NMB Detail End ###")
            print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

        if dataOut_nbmFire is not False:
            nbmFireRunTime = dataOut_nbmFire[HISTORY_PERIODS["NBM"] - 6, 0]

    if readGFS:
        dataOut_gfs = zarr_results["GFS"]
        if dataOut_gfs is not False:
            gfsRunTime = dataOut_gfs[HISTORY_PERIODS["GFS"] - 1, 0]

    if readECMWF:
        dataOut_ecmwf = zarr_results["ECMWF"]
        if dataOut_ecmwf is not False:
            # ECMWF forecast starts at hour +3, so base_time is at HISTORY_PERIODS - 3
            ecmwfRunTime = dataOut_ecmwf[HISTORY_PERIODS["ECMWF"] - 3, 0]
            sourceIDX["ecmwf_ifs"] = dict()
            sourceIDX["ecmwf_ifs"]["x"] = int(x_p_eur)
            sourceIDX["ecmwf_ifs"]["y"] = int(y_p_eur)
            sourceIDX["ecmwf_ifs"]["lat"] = round(lats_ecmwf[y_p_eur], 2)
            sourceIDX["ecmwf_ifs"]["lon"] = round(lons_ecmwf[x_p_eur], 2)

    if readGEFS:
        dataOut_gefs = zarr_results["GEFS"]
        gefsRunTime = dataOut_gefs[HISTORY_PERIODS["GEFS"] - 3, 0]

    if readRTMA_RU:
        dataOut_rtma_ru = zarr_results["RTMA_RU"]

        # Check if RTMA_RU data is valid (not too old)
        if dataOut_rtma_ru is not False:
            rtma_ru_time = dataOut_rtma_ru[0, 0]
            # RTMA-RU is updated every 15 minutes, so data older than 1 hour is stale
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    rtma_ru_time.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=1):
                dataOut_rtma_ru = False
                logger.warning("OLD RTMA_RU")

    sourceTimes = dict()
    sourceList = []
    if use_etopo:
        sourceList.append("ETOPO1")

    # If ERA5 data was read and merged
    if isinstance(ERA5_MERGED, np.ndarray):
        sourceList.append("era5")

    # Timing Check
    if TIMING:
        print("### Sources Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # If point is not in HRRR coverage or HRRR-subh is more than 4 hours old, the fallback to GFS
    if isinstance(dataOut, np.ndarray):
        sourceList.append("hrrrsubh")
        sourceTimes["hrrr_subh"] = rounder(
            datetime.datetime.fromtimestamp(
                subhRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

    # Add RTMA_RU to source list if available (only for currently, not time machine)
    if (isinstance(dataOut_rtma_ru, np.ndarray)) & (not timeMachine):
        sourceList.append("rtma_ru")
        rtma_timestamp = datetime.datetime.fromtimestamp(
            dataOut_rtma_ru[0, 0].astype(int), datetime.UTC
        ).replace(tzinfo=None)
        rounded_rtma_time = rounder(rtma_timestamp, to=15)
        sourceTimes["rtma_ru"] = rounded_rtma_time.strftime("%Y-%m-%d %H:%MZ")

        sourceIDX["rtma_ru"] = {
            "x": int(x_rtma),
            "y": int(y_rtma),
            "lat": round(rtma_lat, 2),
            "lon": round(((rtma_lon + 180) % 360) - 180, 2),
        }

    if (isinstance(dataOut_hrrrh, np.ndarray)) & (not timeMachine):
        sourceList.append("hrrr_0-18")
        sourceTimes["hrrr_0-18"] = rounder(
            datetime.datetime.fromtimestamp(
                hrrrhRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_hrrrh, np.ndarray)) & (timeMachine):
        sourceList.append("hrrr")

    if (isinstance(dataOut_nbm, np.ndarray)) & (not timeMachine):
        sourceList.append("nbm")
        sourceTimes["nbm"] = rounder(
            datetime.datetime.fromtimestamp(
                nbmRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_nbm, np.ndarray)) & (timeMachine):
        sourceList.append("nbm")

    if (isinstance(dataOut_nbmFire, np.ndarray)) & (not timeMachine):
        sourceList.append("nbm_fire")
        sourceTimes["nbm_fire"] = rounder(
            datetime.datetime.fromtimestamp(
                nbmFireRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

    # Add ECMWF IFS after NBM, before GFS
    if (isinstance(dataOut_ecmwf, np.ndarray)) and (not timeMachine):
        sourceList.append("ecmwf_ifs")
        sourceTimes["ecmwf_ifs"] = rounder(
            datetime.datetime.fromtimestamp(
                ecmwfRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

    # If point is not in HRRR coverage or HRRR-hrrrh is more than 16 hours old, the fallback to GFS
    if isinstance(dataOut_h2, np.ndarray):
        sourceList.append("hrrr_18-48")
        # Subtract 18 hours since we're using the 18h time steo
        sourceTimes["hrrr_18-48"] = rounder(
            datetime.datetime.fromtimestamp(
                h2RunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
            - datetime.timedelta(hours=18)
        ).strftime("%Y-%m-%d %HZ")

    if isinstance(dataOut_gfs, np.ndarray):
        sourceTimes["gfs"] = rounder(
            datetime.datetime.fromtimestamp(
                gfsRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

        sourceList.append("gfs")
        sourceIDX["gfs"] = dict()
        sourceIDX["gfs"]["x"] = int(x_p)
        sourceIDX["gfs"]["y"] = int(y_p)
        sourceIDX["gfs"]["lat"] = round(gfs_lat, 2)
        sourceIDX["gfs"]["lon"] = round(((gfs_lon + 180) % 360) - 180, 2)

    if isinstance(dataOut_gefs, np.ndarray):
        sourceList.append("gefs")
        sourceTimes["gefs"] = rounder(
            datetime.datetime.fromtimestamp(
                gefsRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

    # Timing Check
    if TIMING:
        print("### ETOPO Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    ## ELEVATION
    abslat = np.abs(lats_etopo - lat)
    abslon = np.abs(lons_etopo - az_Lon)
    y_p_etopo = np.argmin(abslat)
    x_p_etopo = np.argmin(abslon)

    if (use_etopo) and ((STAGE == "PROD") or (STAGE == "DEV")):
        ETOPO = int(ETOPO_f[y_p_etopo, x_p_etopo])
    else:
        ETOPO = 0

    if ETOPO < 0:
        ETOPO = 0

    if use_etopo:
        sourceIDX["etopo"] = dict()
        sourceIDX["etopo"]["x"] = int(x_p_etopo)
        sourceIDX["etopo"]["y"] = int(y_p_etopo)
        sourceIDX["etopo"]["lat"] = round(lats_etopo[y_p_etopo], 4)
        sourceIDX["etopo"]["lon"] = round(lons_etopo[x_p_etopo], 4)

    # Timing Check
    if TIMING:
        print("Base Times")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Number of hours to start at
    if timeMachine:
        baseTimeOffset = 0
    else:
        baseTimeOffset = (baseHour - baseDay).seconds / 3600

    # Merge hourly models onto a consistent time grid, starting from midnight on the requested day
    # Note that baseTime is the requested time, in TZ aware datetime format
    ### Minutely
    minute_array = np.arange(
        baseTime.astimezone(utc).replace(tzinfo=None),
        baseTime.astimezone(utc).replace(tzinfo=None) + datetime.timedelta(minutes=61),
        datetime.timedelta(minutes=1),
    )

    minute_array_grib = (
        (minute_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    InterTminute = np.zeros((61, 5))  # Type
    InterPminute = np.full((61, max(DATA_MINUTELY.values()) + 1), MISSING_DATA)

    # Create the hourly time and main data arrays
    # InterPhour is the main data array
    InterPhour = np.full(
        (numHours, max(DATA_HOURLY.values()) + 1), MISSING_DATA
    )  # Time, Intensity,Probability

    hour_array_grib = (
        (hour_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Timing Check
    if TIMING:
        print("Nearest IDX Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # HRRR
    # Since the forecast files are pre-processed, they'll always be hourly and the same length. This avoids interpolation
    try:  # Add a fallback to GFS if these don't work
        # HRRR
        if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
            HRRR_StartIDX = nearest_index(dataOut_hrrrh[:, 0], baseDayUTC_Grib)
            H2_StartIDX = nearest_index(dataOut_h2[:, 0], dataOut_hrrrh[-1, 0]) + 1

            if (H2_StartIDX < 1) or (HRRR_StartIDX < 2):
                if "hrrr_18-48" in sourceList:
                    sourceTimes.pop("hrrr_18-48", None)
                    sourceList.remove("hrrr_18-48")
                if "hrrr_0-18" in sourceList:
                    sourceTimes.pop("hrrr_0-18", None)
                    sourceList.remove("hrrr_0-18")

                # Log the error
                logger.error("HRRR data not available for the requested time range.")

            else:
                HRRR_Merged = np.full((numHours, dataOut_h2.shape[1]), MISSING_DATA)
                # The 0-18 hour HRRR data (dataOut_hrrrh) has fewer columns than the 18-48 hour data (dataOut_h2)
                # when in timeMachine mode. Only concatenate the common columns (0-17).
                common_cols = min(dataOut_hrrrh.shape[1], dataOut_h2.shape[1])
                # Calculate actual concatenated size dynamically
                hrrr_rows = len(dataOut_hrrrh) - HRRR_StartIDX
                h2_rows = len(dataOut_h2) - H2_StartIDX
                total_rows = min(hrrr_rows + h2_rows, numHours)
                HRRR_Merged[0:total_rows, 0:common_cols] = np.concatenate(
                    (
                        dataOut_hrrrh[HRRR_StartIDX:, 0:common_cols],
                        dataOut_h2[H2_StartIDX:, 0:common_cols],
                    ),
                    axis=0,
                )[0:total_rows, :]

        # NBM
        if "nbm" in sourceList:
            NBM_StartIDX = nearest_index(dataOut_nbm[:, 0], baseDayUTC_Grib)

            if NBM_StartIDX < 1:
                if "nbm" in sourceList:
                    sourceList.remove("nbm")
                if "nbm" in sourceTimes:
                    sourceTimes.pop("nbm", None)
                logger.error("NBM data not available for the requested time range.")
            else:
                NBM_Merged = np.full((numHours, dataOut_nbm.shape[1]), MISSING_DATA)
                NBM_EndIDX = min((len(dataOut_nbm), (numHours + NBM_StartIDX)))
                NBM_Merged[0 : (NBM_EndIDX - NBM_StartIDX), :] = dataOut_nbm[
                    NBM_StartIDX:NBM_EndIDX, :
                ]

        # NBM FIre
        if "nbm_fire" in sourceList:
            NBM_Fire_StartIDX = nearest_index(dataOut_nbmFire[:, 0], baseDayUTC_Grib)

            if NBM_Fire_StartIDX < 1:
                if "nbm_fire" in sourceList:
                    sourceList.remove("nbm_fire")
                if "nbm_fire" in sourceTimes:
                    sourceTimes.pop("nbm_fire", None)

                logger.error(
                    "NBM Fire data not available for the requested time range."
                )
            else:
                NBM_Fire_Merged = np.full(
                    (numHours, dataOut_nbmFire.shape[1]), MISSING_DATA
                )

                NBM_Fire_EndIDX = min(
                    (len(dataOut_nbmFire), (numHours + NBM_Fire_StartIDX))
                )
                NBM_Fire_Merged[0 : (NBM_Fire_EndIDX - NBM_Fire_StartIDX), :] = (
                    dataOut_nbmFire[NBM_Fire_StartIDX:NBM_Fire_EndIDX, :]
                )

    except Exception:
        logger.exception(
            "HRRR or NBM data not available, falling back to GFS %s", loc_tag
        )
        if "hrrr_18-48" in sourceTimes:
            sourceTimes.pop("hrrr_18-48", None)
        if "nbm_fire" in sourceTimes:
            sourceTimes.pop("nbm_fire", None)
        if "nbm" in sourceTimes:
            sourceTimes.pop("nbm", None)
        if "hrrr_0-18" in sourceTimes:
            sourceTimes.pop("hrrr_0-18", None)
        if "hrrr_subh" in sourceTimes:
            sourceTimes.pop("hrrr_subh", None)

        if "hrrrsubh" in sourceList:
            sourceList.remove("hrrrsubh")
        if "hrrr_0-18" in sourceList:
            sourceList.remove("hrrr_0-18")
        if "nbm" in sourceList:
            sourceList.remove("nbm")
        if "nbm_fire" in sourceList:
            sourceList.remove("nbm_fire")
        if "hrrr_18-48" in sourceList:
            sourceList.remove("hrrr_18-48")

    # GFS
    if "gfs" in sourceList:
        GFS_StartIDX = nearest_index(dataOut_gfs[:, 0], baseDayUTC_Grib)
        GFS_EndIDX = min((len(dataOut_gfs), (numHours + GFS_StartIDX)))
        GFS_Merged = np.full((numHours, max(GFS.values()) + 1), MISSING_DATA)
        GFS_Merged[0 : (GFS_EndIDX - GFS_StartIDX), 0 : dataOut_gfs.shape[1]] = (
            dataOut_gfs[GFS_StartIDX:GFS_EndIDX, 0 : dataOut_gfs.shape[1]]
        )

    # ECMWF
    if "ecmwf_ifs" in sourceList:
        ECMWF_StartIDX = nearest_index(dataOut_ecmwf[:, 0], baseDayUTC_Grib)
        ECMWF_EndIDX = min((len(dataOut_ecmwf), (numHours + ECMWF_StartIDX)))
        ECMWF_Merged = np.full((numHours, max(ECMWF.values()) + 1), MISSING_DATA)
        ECMWF_Merged[
            0 : (ECMWF_EndIDX - ECMWF_StartIDX), 0 : dataOut_ecmwf.shape[1]
        ] = dataOut_ecmwf[ECMWF_StartIDX:ECMWF_EndIDX, 0 : dataOut_ecmwf.shape[1]]

    # GEFS
    if "gefs" in sourceList:
        GEFS_StartIDX = nearest_index(dataOut_gefs[:, 0], baseDayUTC_Grib)
        GEFS_EndIDX = min((len(dataOut_gefs), (numHours + GEFS_StartIDX)))
        GEFS_Merged = np.full((numHours, dataOut_gefs.shape[1]), MISSING_DATA)
        GEFS_Merged[0 : (GEFS_EndIDX - GEFS_StartIDX), :] = dataOut_gefs[
            GEFS_StartIDX:GEFS_EndIDX, :
        ]

    # Timing Check
    if TIMING:
        print("Array start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    InterPhour[:, DATA_HOURLY["time"]] = hour_array_grib

    # Daily array, 12 to 12
    # Have to redo the localize because of daylight saving time
    day_array_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_4am_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=4
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_4pm_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=16
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_5pm_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=17
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_6am_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=6
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_6pm_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=18
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    # Which hours map to which days
    hourlyDayIndex = np.full(len(hour_array_grib), MISSING_DATA)
    hourlyDay4amIndex = np.full(len(hour_array_grib), MISSING_DATA)
    hourlyHighIndex = np.full(len(hour_array_grib), MISSING_DATA)
    hourlyLowIndex = np.full(len(hour_array_grib), MISSING_DATA)
    hourlyDay4pmIndex = np.full(len(hour_array_grib), MISSING_DATA)
    hourlyNight4amIndex = np.full(len(hour_array_grib), MISSING_DATA)

    # Zero to 9 to account for the four horus in day 8
    for d in range(0, 9):
        hourlyDayIndex[
            np.where(
                (hour_array_grib >= day_array_grib[d])
                & (hour_array_grib < day_array_grib[d + 1])
            )
        ] = d
        hourlyDay4amIndex[
            np.where(
                (hour_array_grib >= day_array_4am_grib[d])
                & (hour_array_grib < day_array_4am_grib[d + 1])
            )
        ] = d
        hourlyDay4pmIndex[
            np.where(
                (hour_array_grib >= day_array_4am_grib[d])
                & (hour_array_grib <= day_array_4pm_grib[d])
            )
        ] = d
        hourlyNight4amIndex[
            np.where(
                (hour_array_grib >= day_array_5pm_grib[d])
                & (hour_array_grib < day_array_4am_grib[d + 1])
            )
        ] = d
        hourlyHighIndex[
            np.where(
                (hour_array_grib > day_array_6am_grib[d])
                & (hour_array_grib <= day_array_6pm_grib[d])
            )
        ] = d
        hourlyLowIndex[
            np.where(
                (hour_array_grib > day_array_6pm_grib[d])
                & (hour_array_grib <= day_array_6am_grib[d + 1])
            )
        ] = d

    if not timeMachine:
        # Replace NaN values with -999 before casting to int to avoid RuntimeWarning
        hourlyDayIndex = np.nan_to_num(hourlyDayIndex, nan=-999).astype(int)
        hourlyDay4amIndex = np.nan_to_num(hourlyDay4amIndex, nan=-999).astype(int)
        hourlyHighIndex = np.nan_to_num(hourlyHighIndex, nan=-999).astype(int)
        hourlyLowIndex = np.nan_to_num(hourlyLowIndex, nan=-999).astype(int)
        hourlyDay4pmIndex = np.nan_to_num(hourlyDay4pmIndex, nan=-999).astype(int)
        hourlyNight4amIndex = np.nan_to_num(hourlyNight4amIndex, nan=-999).astype(int)
    else:
        # When running in timemachine mode, don't try to parse through different times, use the current 24h day for everything
        hourlyDayIndex = np.full(len(hour_array_grib), int(0))
        hourlyDay4amIndex = np.full(len(hour_array_grib), int(0))
        hourlyHighIndex = np.full(len(hour_array_grib), int(0))
        hourlyLowIndex = np.full(len(hour_array_grib), int(0))
        hourlyDay4pmIndex = np.full(len(hour_array_grib), int(0))
        hourlyNight4amIndex = np.full(len(hour_array_grib), int(0))

    # +1 to account for the extra 4 hours of summary
    InterSday = np.zeros(shape=(daily_days + 1, 21))

    # Timing Check
    if TIMING:
        print("Sunrise start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    loc = LocationInfo("name", "region", tz_name, lat, az_Lon)

    is_all_day = False
    is_all_night = False

    # Calculate Sunrise, Sunset, Moon Phase
    for i in range(0, daily_days + 1):
        try:
            s = sun(
                loc.observer, date=baseDay + datetime.timedelta(days=i)
            )  # Use local to get the correct date

            InterSday[i, DATA_DAY["sunrise"]] = (
                (
                    np.datetime64(s["sunrise"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
            InterSday[i, DATA_DAY["sunset"]] = (
                (
                    np.datetime64(s["sunset"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )

            InterSday[i, DATA_DAY["dawn"]] = (
                (
                    np.datetime64(s["dawn"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
            InterSday[i, DATA_DAY["dusk"]] = (
                (
                    np.datetime64(s["dusk"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )

        except ValueError:
            # If always sunny (polar day) or always dark (polar night) we need to
            # determine which case applies based on hemisphere and month ranges.
            # Use boolean operators with explicit grouping to avoid accidental
            # precedence issues from bitwise operators.
            # Northern hemisphere: roughly April (4) through September (9) -> polar day
            # Southern hemisphere: roughly October (10) through March (3) -> polar day
            if _polar_is_all_day(lat, baseDay.month):
                # Set sunrise to one second after midnight
                InterSday[i, DATA_DAY["sunrise"]] = day_array_grib[i] + np.timedelta64(
                    1, "s"
                ).astype("timedelta64[s]").astype(np.int32)
                # Set sunset to one second before midnight the following day
                InterSday[i, DATA_DAY["sunset"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                # Set sunrise to one second after midnight
                InterSday[i, DATA_DAY["dawn"]] = day_array_grib[i] + np.timedelta64(
                    1, "s"
                ).astype("timedelta64[s]").astype(np.int32)
                # Set sunset to one second before midnight the following day
                InterSday[i, DATA_DAY["dusk"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                is_all_day = True
            else:
                # Set sunrise to two seconds before midnight
                InterSday[i, DATA_DAY["sunrise"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(2, "s").astype("timedelta64[s]").astype(np.int32)
                )
                # Set sunset to one seconds before midnight
                InterSday[i, DATA_DAY["sunset"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                InterSday[i, DATA_DAY["dawn"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(2, "s").astype("timedelta64[s]").astype(np.int32)
                )
                # Set sunset to one seconds before midnight
                InterSday[i, DATA_DAY["dusk"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                is_all_night = True

        m = moon.phase(baseDay + datetime.timedelta(days=i))
        InterSday[i, DATA_DAY["moon_phase"]] = m / 27.99

    # Timing Check
    if TIMING:
        print("Interpolation Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Interpolate for minutely
    # Concatenate HRRR and HRRR2
    if "gefs" in sourceList:
        gefsMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_gefs[0, :]))
        )
    if "gfs" in sourceList:
        gfsMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_gfs[0, :]))
        )

    if "ecmwf_ifs" in sourceList:
        ecmwfMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_ecmwf[0, :]))
        )

    nbmMinuteInterpolation = np.zeros((len(minute_array_grib), 18))

    if "hrrrsubh" in sourceList:
        hrrrSubHInterpolation = np.zeros((len(minute_array_grib), len(dataOut[0, :])))
        for i in range(len(dataOut[0, :]) - 1):
            hrrrSubHInterpolation[:, i + 1] = np.interp(
                minute_array_grib,
                dataOut[:, 0].squeeze(),
                dataOut[:, i + 1],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        # Check for nan, which means SubH is out of range, and fall back to regular HRRR
        if np.isnan(hrrrSubHInterpolation[1, 1]):
            hrrrSubHInterpolation[:, HRRR_SUBH["gust"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["gust"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["pressure"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["pressure"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["temp"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["temp"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["dew"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["dew"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["wind_u"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["wind_u"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["wind_v"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["wind_v"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["intensity"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["intensity"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["snow"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["snow"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["ice"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["ice"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["freezing_rain"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["freezing_rain"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["rain"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["rain"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["refc"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["refc"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["solar"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["solar"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

            # Visibility is at a weird index
            hrrrSubHInterpolation[:, HRRR_SUBH["vis"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["vis"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
        if "gefs" in sourceList:
            gefsMinuteInterpolation[:, GEFS["error"]] = np.interp(
                minute_array_grib,
                dataOut_gefs[:, 0].squeeze(),
                dataOut_gefs[:, GEFS["error"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

    else:  # Usse GFS/GEFS
        if "gefs" in sourceList:
            for i in range(len(dataOut_gefs[0, :]) - 1):
                gefsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_gefs[:, 0].squeeze(),
                    dataOut_gefs[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

        if "gfs" in sourceList:  # GFS Fallback
            # This could be optimized by only interpolating the necessary columns
            for i in range(len(dataOut_gfs[0, :]) - 1):
                gfsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_gfs[:, 0].squeeze(),
                    dataOut_gfs[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

    if "ecmwf_ifs" in sourceList:
        for i in range(len(dataOut_ecmwf[0, :]) - 1):
            # Switch to nearest for precipitation type
            if i + 1 == ECMWF["ptype"]:
                ecmwfMinuteInterpolation[:, i + 1] = fast_nearest_interp(
                    minute_array_grib,
                    dataOut_ecmwf[:, 0].squeeze(),
                    dataOut_ecmwf[:, i + 1],
                )
            else:
                ecmwfMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_ecmwf[:, 0].squeeze(),
                    dataOut_ecmwf[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

    if "nbm" in sourceList:
        for i in [
            NBM["accum"],
            NBM["prob"],
            NBM["rain"],
            NBM["freezing_rain"],
            NBM["snow"],
            NBM["ice"],
        ]:
            nbmMinuteInterpolation[:, i] = np.interp(
                minute_array_grib,
                dataOut_nbm[:, 0].squeeze(),
                dataOut_nbm[:, i],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

    era5_MinuteInterpolation = np.zeros((len(minute_array_grib), max(ERA5.values())))

    if "era5" in sourceList:
        for i in [
            ERA5["large_scale_rain_rate"],
            ERA5["convective_rain_rate"],
            ERA5["large_scale_snowfall_rate_water_equivalent"],
            ERA5["convective_snowfall_rate_water_equivalent"],
        ]:
            era5_MinuteInterpolation[:, i] = np.interp(
                minute_array_grib,
                ERA5_MERGED[:, 0].squeeze(),
                ERA5_MERGED[:, i],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        # Precipitation type should be nearest, not linear
        era5_MinuteInterpolation[:, ERA5["precipitation_type"]] = fast_nearest_interp(
            minute_array_grib,
            ERA5_MERGED[:, 0].squeeze(),
            ERA5_MERGED[:, ERA5["precipitation_type"]],
        )

    # Timing Check
    if TIMING:
        print("Minutely Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    InterPminute[:, DATA_MINUTELY["time"]] = minute_array_grib

    # "precipProbability"
    # Use NBM where available, then ECMWF, then GEFS
    if "nbm" in sourceList:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            nbmMinuteInterpolation[:, NBM["prob"]] * 0.01
        )
    elif "ecmwf_ifs" in sourceList:
        InterPminute[:, DATA_MINUTELY["prob"]] = ecmwfMinuteInterpolation[
            :, ECMWF["prob"]
        ]
    elif "gefs" in sourceList:
        InterPminute[:, DATA_MINUTELY["prob"]] = gefsMinuteInterpolation[
            :, GEFS["prob"]
        ]
    else:  # Missing (-999) fallback
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    # Less than 5% set to 0
    InterPminute[
        InterPminute[:, DATA_MINUTELY["prob"]] < 0.05, DATA_MINUTELY["prob"]
    ] = 0

    # Precipitation Type
    # IF HRRR, use that, then NBM, then ECMWF, then GEFS, else GFS
    if "hrrrsubh" in sourceList:
        for i in [
            HRRR_SUBH["snow"],
            HRRR_SUBH["ice"],
            HRRR_SUBH["freezing_rain"],
            HRRR_SUBH["rain"],
        ]:
            InterTminute[:, i - 7] = hrrrSubHInterpolation[:, i]
    elif "nbm" in sourceList:
        # 14 = Rain (1,2), 15 = Freezing Rain/ Ice (3,4), 16 = Snow (5,6,7), 17 = Ice (8,9)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-201.shtml

        # Snow
        InterTminute[:, 1] = nbmMinuteInterpolation[:, NBM["snow"]]
        # Ice
        InterTminute[:, 2] = nbmMinuteInterpolation[:, NBM["ice"]]
        # Freezing Rain
        InterTminute[:, 3] = nbmMinuteInterpolation[:, NBM["freezing_rain"]]
        # Rain
        InterTminute[:, 4] = nbmMinuteInterpolation[:, NBM["rain"]]
    elif "ecmwf_ifs" in sourceList:
        # ECMWF precipitation type codes:
        # 0=No precip, 1=Rain, 2=Thunderstorm, 3=Freezing rain, 4=Mixed/ice, 5=Snow,
        # 6=Wet snow, 7=Mix of rain/snow, 8=Ice pellets, 9=Graupel, 10=Hail,
        # 11=Drizzle, 12=Freezing drizzle, 255=Missing
        ptype_ecmwf = ecmwfMinuteInterpolation[:, ECMWF["ptype"]]

        # Map ECMWF ptype to InterTminute columns:
        # InterTminute[:, 0] = none (not set here, default)
        # InterTminute[:, 1] = snow (codes 5, 6, 9)
        # InterTminute[:, 2] = ice/sleet (codes 4, 8, 10)
        # InterTminute[:, 3] = freezing rain (codes 3, 12)
        # InterTminute[:, 4] = rain (codes 1, 2, 7, 11)

        InterTminute[:, 1] = np.where(
            np.isin(ptype_ecmwf, [5, 6, 9]), 1, 0
        )  # Snow, wet snow, graupel
        InterTminute[:, 2] = np.where(
            np.isin(ptype_ecmwf, [4, 8, 10]), 1, 0
        )  # Mixed/ice, ice pellets, hail
        InterTminute[:, 3] = np.where(
            np.isin(ptype_ecmwf, [3, 12]), 1, 0
        )  # Freezing rain, freezing drizzle
        InterTminute[:, 4] = np.where(
            np.isin(ptype_ecmwf, [1, 2, 7, 11]), 1, 0
        )  # Rain, thunderstorm, rain/snow mix, drizzle

    elif "gefs" in sourceList:
        for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
            InterTminute[:, i - 3] = gefsMinuteInterpolation[:, i]
    elif "gfs" in sourceList:  # GFS Fallback
        for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
            InterTminute[:, i - 11] = gfsMinuteInterpolation[:, i]
    elif "era5" in sourceList:
        # ERA5 precipitation type codes:
        # 0=No precip, 1=Rain, 2=Thunderstorm, 3=Freezing rain, 4=Mixed/ice, 5=Snow,
        # 6=Wet snow, 7=Mix of rain/snow, 8=Ice pellets, 9=Graupel, 10=Hail,
        # 11=Drizzle, 12=Freezing drizzle, 255=Missing
        ptype_era5 = era5_MinuteInterpolation[:, ERA5["precipitation_type"]]

        InterTminute[:, 1] = np.where(
            np.isin(ptype_era5, [5, 6, 9]), 1, 0
        )  # Snow, wet snow, graupel
        InterTminute[:, 2] = np.where(
            np.isin(ptype_era5, [4, 8, 10]), 1, 0
        )  # Mixed/ice, ice pellets, hail
        InterTminute[:, 3] = np.where(
            np.isin(ptype_era5, [3, 12]), 1, 0
        )  # Freezing rain, freezing drizzle
        InterTminute[:, 4] = np.where(
            np.isin(ptype_era5, [1, 2, 7, 11]), 1, 0
        )  # Rain, thunderstorm, rain/snow mix, drizzle

    # If all nan, set pchance to -999, otherwise determine the predominant type
    maxPchance = (
        np.argmax(InterTminute, axis=1)
        if not np.any(np.isnan(InterTminute))
        else np.full(len(minute_array_grib), 5)
    )
    pTypes = ["none", "snow", "sleet", "sleet", "rain", MISSING_DATA]
    pTypesText = ["Clear", "Snow", "Sleet", "Sleet", "Rain", MISSING_DATA]
    pTypesIcon = ["clear", "snow", "sleet", "sleet", "rain", MISSING_DATA]

    minuteType = [pTypes[maxPchance[idx]] for idx in range(61)]

    precipTypes = np.array(minuteType)

    if "hrrrsubh" in sourceList:
        # Get temperature and reflectivity arrays first.
        temp_arr = hrrrSubHInterpolation[:, HRRR_SUBH["temp"]]
        refc_arr = hrrrSubHInterpolation[:, HRRR_SUBH["refc"]]

        # Mask: only assign type if current type is "none" AND reflectivity shows precip
        mask = (precipTypes == "none") & (refc_arr > 0)

        # Assign rain, snow, sleet based on temperature thresholds
        precipTypes[mask] = np.where(
            temp_arr[mask] >= TEMP_THRESHOLD_RAIN_C,
            "rain",
            np.where(temp_arr[mask] <= TEMP_THRESHOLD_SNOW_C, "snow", "sleet"),
        )

        # Update lists and arrays
        minuteType = precipTypes.tolist()
        precipTypes = np.array(minuteType)

        # Now convert reflectivity to precipitation intensity using estimated types
        InterPminute[:, DATA_MINUTELY["intensity"]] = dbz_to_rate(refc_arr, precipTypes)
    elif "nbm" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = nbmMinuteInterpolation[
            :, NBM["accum"]
        ]
    elif "ecmwf_ifs" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            ecmwfMinuteInterpolation[:, ECMWF["intensity"]] * 3600
        )
    elif "gefs" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = gefsMinuteInterpolation[
            :, GEFS["accum"]
        ]
    elif "gfs" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = dbz_to_rate(
            gfsMinuteInterpolation[:, GFS["refc"]], precipTypes
        )
    elif "era5" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            era5_MinuteInterpolation[
                :, ERA5["large_scale_snowfall_rate_water_equivalent"]
            ]
            + era5_MinuteInterpolation[
                :, ERA5["convective_snowfall_rate_water_equivalent"]
            ]
            + era5_MinuteInterpolation[:, ERA5["large_scale_rain_rate"]]
            + era5_MinuteInterpolation[:, ERA5["convective_rain_rate"]]
        ) * 3600

    # "precipIntensityError"
    if "ecmwf_ifs" in sourceList:
        InterPminute[:, DATA_MINUTELY["error"]] = (
            ecmwfMinuteInterpolation[:, ECMWF["accum_stddev"]] * 1000
        )  # Accum stddev is in meters
    elif "gefs" in sourceList:
        InterPminute[:, DATA_MINUTELY["error"]] = gefsMinuteInterpolation[
            :, GEFS["error"]
        ]
    else:  # Missing
        InterPminute[:, DATA_MINUTELY["error"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    # Create list of icons based off of maxPchance
    minuteKeys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
    ]
    if version >= 2:
        minuteKeys += ["rainIntensity", "snowIntensity", "sleetIntensity"]

    # Calculate type-specific intensities for minutely (in SI units - mm/h liquid equivalent)
    # Initialize all to zero
    InterPminute[:, DATA_MINUTELY["rain_intensity"]] = 0
    InterPminute[:, DATA_MINUTELY["snow_intensity"]] = 0
    InterPminute[:, DATA_MINUTELY["ice_intensity"]] = 0

    # Rain intensity (direct from intensity)
    rain_mask_min = maxPchance == PRECIP_IDX["rain"]
    InterPminute[rain_mask_min, DATA_MINUTELY["rain_intensity"]] = InterPminute[
        rain_mask_min, DATA_MINUTELY["intensity"]
    ]

    # Snow intensity - for minutely, we don't have temperature/wind readily available
    # So use a default 10:1 ratio
    snow_mask_min = maxPchance == PRECIP_IDX["snow"]
    InterPminute[snow_mask_min, DATA_MINUTELY["snow_intensity"]] = (
        InterPminute[snow_mask_min, DATA_MINUTELY["intensity"]] * 10
    )

    # Sleet intensity (direct from intensity)
    sleet_mask_min = (maxPchance == PRECIP_IDX["ice"]) | (
        maxPchance == PRECIP_IDX["sleet"]
    )
    InterPminute[sleet_mask_min, DATA_MINUTELY["ice_intensity"]] = InterPminute[
        sleet_mask_min, DATA_MINUTELY["intensity"]
    ]

    minuteTimes = InterPminute[:, DATA_MINUTELY["time"]]
    minuteIntensity = np.maximum(InterPminute[:, DATA_MINUTELY["intensity"]], 0)
    minuteProbability = np.minimum(
        np.maximum(InterPminute[:, DATA_MINUTELY["prob"]], 0), 1
    )
    minuteIntensityError = np.maximum(InterPminute[:, DATA_MINUTELY["error"]], 0)

    # Prepare minutely intensity arrays for output
    minuteRainIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["rain_intensity"]], 0
    )
    minuteSnowIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["snow_intensity"]], 0
    )
    minuteSleetIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["ice_intensity"]], 0
    )

    # Set values below 0.01 mm/h to zero to reduce noise. This value can be tuned if needed.
    minuteRainIntensity[np.abs(minuteRainIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0
    minuteSnowIntensity[np.abs(minuteSnowIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0
    minuteSleetIntensity[np.abs(minuteSleetIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = (
        0.0
    )
    minuteProbability[np.abs(minuteProbability) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0
    minuteIntensityError[np.abs(minuteIntensityError) < PRECIP_NOISE_THRESHOLD_MMH] = (
        0.0
    )
    minuteIntensity[np.abs(minuteIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0

    # Pre-calculate all unit conversions for minutely block (vectorized approach)
    # Convert to display units and round
    minuteIntensity_display = np.round(minuteIntensity * prepIntensityUnit, 4)
    minuteIntensityError_display = np.round(minuteIntensityError * prepIntensityUnit, 4)
    minuteRainIntensity_display = np.round(minuteRainIntensity * prepIntensityUnit, 4)
    minuteSnowIntensity_display = np.round(minuteSnowIntensity * prepIntensityUnit, 4)
    minuteSleetIntensity_display = np.round(minuteSleetIntensity * prepIntensityUnit, 4)
    minuteProbability_display = np.round(minuteProbability, 4)

    minuteItems = []
    minuteItems_si = []
    all_minute_keys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
        "rainIntensity",
        "snowIntensity",
        "sleetIntensity",
    ]
    for idx in range(61):
        values = [
            int(minuteTimes[idx]),
            float(minuteIntensity_display[idx]),
            float(minuteProbability_display[idx]),
            float(minuteIntensityError_display[idx]),
            minuteType[idx],
        ]
        if version >= 2:
            values += [
                float(minuteRainIntensity_display[idx]),
                float(minuteSnowIntensity_display[idx]),
                float(minuteSleetIntensity_display[idx]),
            ]
        minuteItems.append(dict(zip(minuteKeys, values)))

        # SI object always includes all keys
        values_si = [
            int(minuteTimes[idx]),
            float(minuteIntensity[idx]),
            float(minuteProbability[idx]),
            float(minuteIntensityError[idx]),
            minuteType[idx],
            float(minuteRainIntensity[idx]),
            float(minuteSnowIntensity[idx]),
            float(minuteSleetIntensity[idx]),
        ]
        minuteItems_si.append(dict(zip(all_minute_keys, values_si)))

    # Timing Check
    if TIMING:
        print("Hourly start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    ## Approach
    # Use NBM where available
    # Use HRRR for some other variables
    # Use ECMWF where HRRR not available
    # If ECMWF is not available, use GFS

    # Precipitation Type
    # NBM, HRRR, ECMWF, GEFS/GFS, ERA5
    maxPchanceHour = np.full((len(hour_array_grib), 5), MISSING_DATA)

    if "nbm" in sourceList:
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        InterThour[:, 1] = NBM_Merged[:, NBM["snow"]]
        InterThour[:, 2] = NBM_Merged[:, NBM["ice"]]
        InterThour[:, 3] = NBM_Merged[:, NBM["freezing_rain"]]
        InterThour[:, 4] = NBM_Merged[:, NBM["rain"]]

        # 14 = Rain (1,2), 15 = Freezing Rain/ Ice (3,4), 16 = Snow (5,6,7), 17 = Ice (8,9)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-201.shtml

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 0] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 0] = MISSING_DATA

    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterThour = np.zeros(shape=(len(hour_array), 5))
        InterThour[:, 1] = HRRR_Merged[:, HRRR["snow"]]
        InterThour[:, 2] = HRRR_Merged[:, HRRR["ice"]]
        InterThour[:, 3] = HRRR_Merged[:, HRRR["freezing_rain"]]
        InterThour[:, 4] = HRRR_Merged[:, HRRR["rain"]]

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0
        maxPchanceHour[:, 1] = np.argmax(InterThour, axis=1)
        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 1] = MISSING_DATA

    # ECMWF - convert ptype codes to categorical indices
    if "ecmwf_ifs" in sourceList:
        # ECMWF precipitation type codes:
        # 0=No precip, 1=Rain, 2=Thunderstorm, 3=Freezing rain, 4=Mixed/ice, 5=Snow,
        # 6=Wet snow, 7=Mix of rain/snow, 8=Ice pellets, 9=Graupel, 10=Hail,
        # 11=Drizzle, 12=Freezing drizzle, 255=Missing
        #
        # Map to indices: 0=none, 1=snow, 2=sleet, 3=freezing rain, 4=rain
        ptype_ecmwf_hour = ECMWF_Merged[:, ECMWF["ptype"]]

        # Initialize with 0 (none)
        conditions = [
            np.isin(ptype_ecmwf_hour, [5, 6, 9]),  # snow
            np.isin(ptype_ecmwf_hour, [4, 8, 10]),  # sleet
            np.isin(ptype_ecmwf_hour, [3, 12]),  # freezing rain
            np.isin(ptype_ecmwf_hour, [1, 2, 7, 11]),  # rain
        ]
        choices = [1, 2, 3, 4]
        mapped_ptype = np.select(conditions, choices, default=0)

        maxPchanceHour[:, 2] = mapped_ptype
        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(ptype_ecmwf_hour), 2] = MISSING_DATA

    # GEFS
    if "gefs" in sourceList:
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
            InterThour[:, i - 3] = GEFS_Merged[:, i]

        # 4 = Snow, 5 = Sleet, 6 = Freezing Rain, 7 = Rain

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 3] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 3] = MISSING_DATA
    elif "gfs" in sourceList:  # GFS Fallback
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
            InterThour[:, i - 11] = GFS_Merged[:, i]

        # 12 = Snow, 13 = Sleet, 14 = Freezing Rain, 15 = Rain

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 3] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 3] = MISSING_DATA

    # ERA5 for Timemachine
    if "era5" in sourceList:
        ptype_era5_hour = ERA5_MERGED[:, ERA5["precipitation_type"]]

        # Round to nearest integer
        ptype_era5_hour = np.round(ptype_era5_hour).astype(int)

        # Initialize with 0 (none)
        conditions = [
            np.isin(ptype_era5_hour, [5, 6, 9]),  # snow
            np.isin(ptype_era5_hour, [4, 8, 10]),  # sleet
            np.isin(ptype_era5_hour, [3, 12]),  # freezing rain
            np.isin(ptype_era5_hour, [1, 2, 7, 11]),  # rain
        ]
        choices = [1, 2, 3, 4]
        mapped_ptype = np.select(conditions, choices, default=0)

        maxPchanceHour[:, 4] = mapped_ptype
        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(ptype_era5_hour), 4] = MISSING_DATA

    # Intensity
    # NBM, HRRR, ECMWF, GEFS/GFS
    prcipIntensityHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    if "nbm" in sourceList:
        prcipIntensityHour[:, 0] = NBM_Merged[:, NBM["intensity"]]
    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        prcipIntensityHour[:, 1] = HRRR_Merged[:, HRRR["intensity"]] * 3600
    # ECMWF
    if "ecmwf_ifs" in sourceList:
        # Use tprate (total precipitation rate) for intensity
        prcipIntensityHour[:, 2] = ECMWF_Merged[:, ECMWF["intensity"]] * 3600
    # GEFS or GFS
    if "gefs" in sourceList:
        prcipIntensityHour[:, 3] = GEFS_Merged[:, GEFS["accum"]]
    elif "gfs" in sourceList:  # GFS Fallback
        prcipIntensityHour[:, 3] = GFS_Merged[:, GFS["intensity"]] * 3600

    # ERA5
    if "era5" in sourceList:
        # This isn't perfect, since ERA5 only has instant rates for rain and snow, not ice.
        prcipIntensityHour[:, 4] = (
            ERA5_MERGED[:, ERA5["large_scale_rain_rate"]]
            + ERA5_MERGED[:, ERA5["convective_rain_rate"]]
            + ERA5_MERGED[:, ERA5["large_scale_snowfall_rate_water_equivalent"]]
            + ERA5_MERGED[:, ERA5["convective_snowfall_rate_water_equivalent"]]
        ) * 3600

        # Calculate separate rain and snow intensities for ERA5
        # Rain intensity from ERA5 rain rates (mm/h liquid)
        era5_rain_intensity = (
            ERA5_MERGED[:, ERA5["large_scale_rain_rate"]]
            + ERA5_MERGED[:, ERA5["convective_rain_rate"]]
        ) * 3600  # Convert from m/s to mm/h

        # Snow intensity from ERA5 snow rates (mm/h water equivalent)
        era5_snow_water_equivalent = (
            ERA5_MERGED[:, ERA5["large_scale_snowfall_rate_water_equivalent"]]
            + ERA5_MERGED[:, ERA5["convective_snowfall_rate_water_equivalent"]]
        ) * 3600  # Convert from m/s to mm/h

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["intensity"]] = (
        np.choose(np.argmin(np.isnan(prcipIntensityHour), axis=1), prcipIntensityHour.T)
        * prepIntensityUnit
    )

    # Set zero as the floor
    InterPhour[:, DATA_HOURLY["intensity"]] = np.maximum(
        InterPhour[:, DATA_HOURLY["intensity"]], 0
    )

    # Set all values below 0.0005 (0.5 mm of snow/ 0.05 mm of rain) to zero
    InterPhour[
        InterPhour[:, DATA_HOURLY["intensity"]] < (0.0005 * prepIntensityUnit),
        DATA_HOURLY["intensity"],
    ] = 0

    # Use the same type value as the intensity
    InterPhour[:, DATA_HOURLY["type"]] = np.choose(
        np.argmin(np.isnan(prcipIntensityHour), axis=1), maxPchanceHour.T
    )

    # Probability
    # NBM, ECMWF, GEFS priority order
    prcipProbabilityHour = np.full((len(hour_array_grib), 3), MISSING_DATA)
    if "nbm" in sourceList:
        prcipProbabilityHour[:, 0] = NBM_Merged[:, NBM["prob"]] * 0.01
    # ECMWF
    if "ecmwf_ifs" in sourceList:
        prcipProbabilityHour[:, 1] = ECMWF_Merged[:, ECMWF["prob"]]
    # GEFS
    if "gefs" in sourceList:
        prcipProbabilityHour[:, 2] = GEFS_Merged[:, GEFS["prob"]]

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["prob"]] = np.choose(
        np.argmin(np.isnan(prcipProbabilityHour), axis=1), prcipProbabilityHour.T
    )

    # Cap at 1
    InterPhour[:, DATA_HOURLY["prob"]] = clipLog(
        InterPhour[:, DATA_HOURLY["prob"]],
        CLIP_PROB["min"],
        CLIP_PROB["max"],
        "Probability Hour",
    )

    # Less than 5% set to 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] < 0.05, DATA_HOURLY["prob"]] = 0

    # Set intensity to zero if POP == 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] == 0, 2] = 0

    # Intensity Error
    # ECMWF, then GEFS
    if "ecmwf_ifs" in sourceList:
        InterPhour[:, DATA_HOURLY["error"]] = np.maximum(
            ECMWF_Merged[:, ECMWF["accum_stddev"]] * 1000, 0
        )
    elif "gefs" in sourceList:
        InterPhour[:, DATA_HOURLY["error"]] = np.maximum(
            GEFS_Merged[:, GEFS["error"]], 0
        )

    ### Temperature
    TemperatureHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    if "nbm" in sourceList:
        TemperatureHour[:, 0] = NBM_Merged[:, NBM["temp"]]

    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        TemperatureHour[:, 1] = HRRR_Merged[:, HRRR["temp"]]

    if "ecmwf_ifs" in sourceList:
        TemperatureHour[:, 2] = ECMWF_Merged[:, ECMWF["temp"]]

    if "gfs" in sourceList:
        TemperatureHour[:, 3] = GFS_Merged[:, GFS["temp"]]

    if "era5" in sourceList:
        TemperatureHour[:, 4] = ERA5_MERGED[:, ERA5["2m_temperature"]]

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["temp"]] = np.choose(
        np.argmin(np.isnan(TemperatureHour), axis=1), TemperatureHour.T
    )

    # Clip between -90 and 60
    InterPhour[:, DATA_HOURLY["temp"]] = clipLog(
        InterPhour[:, DATA_HOURLY["temp"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Temperature Hour",
    )

    ### Dew Point
    DewPointHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    if "nbm" in sourceList:
        DewPointHour[:, 0] = NBM_Merged[:, NBM["dew"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        DewPointHour[:, 1] = HRRR_Merged[:, HRRR["dew"]]
    if "ecmwf_ifs" in sourceList:
        DewPointHour[:, 2] = ECMWF_Merged[:, ECMWF["dew"]]
    if "gfs" in sourceList:
        DewPointHour[:, 3] = GFS_Merged[:, GFS["dew"]]
    if "era5" in sourceList:
        DewPointHour[:, 4] = ERA5_MERGED[:, ERA5["2m_dewpoint_temperature"]]

    InterPhour[:, DATA_HOURLY["dew"]] = np.choose(
        np.argmin(np.isnan(DewPointHour), axis=1), DewPointHour.T
    )

    # Clip between -90 and 60 C
    InterPhour[:, DATA_HOURLY["dew"]] = clipLog(
        InterPhour[:, DATA_HOURLY["dew"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Dew Point Hour",
    )

    ### Humidity
    HumidityHour = np.full((len(hour_array_grib), 4), MISSING_DATA)
    if "nbm" in sourceList:
        HumidityHour[:, 0] = NBM_Merged[:, NBM["humidity"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        HumidityHour[:, 1] = HRRR_Merged[:, HRRR["humidity"]]
    if "gfs" in sourceList:
        HumidityHour[:, 2] = GFS_Merged[:, GFS["humidity"]]
    if "era5" in sourceList:
        HumidityHour[:, 3] = (
            relative_humidity_from_dewpoint(
                ERA5_MERGED[:, ERA5["2m_temperature"]] * mp.units.units.degK,
                ERA5_MERGED[:, ERA5["2m_dewpoint_temperature"]] * mp.units.units.degK,
                phase="auto",
            ).magnitude
            * 100
        )  # Convert to percentage

    InterPhour[:, DATA_HOURLY["humidity"]] = (
        np.choose(np.argmin(np.isnan(HumidityHour), axis=1), HumidityHour.T) * humidUnit
    )

    # Clip between 0 and 1
    InterPhour[:, DATA_HOURLY["humidity"]] = clipLog(
        InterPhour[:, DATA_HOURLY["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
        "Humidity Hour",
    )

    ### Pressure
    PressureHour = np.full((len(hour_array_grib), 4), MISSING_DATA)
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        PressureHour[:, 0] = HRRR_Merged[:, HRRR["pressure"]]
    if "ecmwf_ifs" in sourceList:
        PressureHour[:, 1] = ECMWF_Merged[:, ECMWF["pressure"]]
    if "gfs" in sourceList:
        PressureHour[:, 2] = GFS_Merged[:, GFS["pressure"]]
    if "era5" in sourceList:
        PressureHour[:, 3] = ERA5_MERGED[:, ERA5["mean_sea_level_pressure"]]
    InterPhour[:, DATA_HOURLY["pressure"]] = np.choose(
        np.argmin(np.isnan(PressureHour), axis=1), PressureHour.T
    )

    # Clip between 800 and 1100 hPa (80000-110000 Pascals)
    InterPhour[:, DATA_HOURLY["pressure"]] = clipLog(
        InterPhour[:, DATA_HOURLY["pressure"]],
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Pressure Hour",
    )

    ### Wind Speed
    WindSpeedHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    if "nbm" in sourceList:
        WindSpeedHour[:, 0] = NBM_Merged[:, NBM["wind"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindSpeedHour[:, 1] = np.sqrt(
            HRRR_Merged[:, HRRR["wind_u"]] ** 2 + HRRR_Merged[:, HRRR["wind_v"]] ** 2
        )
    if "ecmwf_ifs" in sourceList:
        WindSpeedHour[:, 2] = np.sqrt(
            ECMWF_Merged[:, ECMWF["wind_u"]] ** 2
            + ECMWF_Merged[:, ECMWF["wind_v"]] ** 2
        )
    if "gfs" in sourceList:
        WindSpeedHour[:, 3] = np.sqrt(
            GFS_Merged[:, GFS["wind_u"]] ** 2 + GFS_Merged[:, GFS["wind_v"]] ** 2
        )
    if "era5" in sourceList:
        WindSpeedHour[:, 3] = np.sqrt(
            ERA5_MERGED[:, ERA5["10m_u_component_of_wind"]] ** 2
            + ERA5_MERGED[:, ERA5["10m_v_component_of_wind"]] ** 2
        )

    InterPhour[:, DATA_HOURLY["wind"]] = np.choose(
        np.argmin(np.isnan(WindSpeedHour), axis=1), WindSpeedHour.T
    )

    # Clip between 0 and 400, keep in m/s (SI units)
    InterPhour[:, DATA_HOURLY["wind"]] = clipLog(
        InterPhour[:, DATA_HOURLY["wind"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "Wind Speed",
    )

    ### Wind Gust
    WindGustHour = np.full((len(hour_array_grib), 4), MISSING_DATA)
    if "nbm" in sourceList:
        WindGustHour[:, 0] = NBM_Merged[:, NBM["gust"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindGustHour[:, 1] = HRRR_Merged[:, HRRR["gust"]]
    if "gfs" in sourceList:
        WindGustHour[:, 2] = GFS_Merged[:, GFS["gust"]]
    if "era5" in sourceList:
        WindGustHour[:, 3] = ERA5_MERGED[:, ERA5["instantaneous_10m_wind_gust"]]

    InterPhour[:, DATA_HOURLY["gust"]] = np.choose(
        np.argmin(np.isnan(WindGustHour), axis=1), WindGustHour.T
    )
    # Clip between 0 and 400, keep in m/s (SI units)
    InterPhour[:, DATA_HOURLY["gust"]] = clipLog(
        InterPhour[:, DATA_HOURLY["gust"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "Wind Gust Hour",
    )

    ### Wind Bearing
    WindBearingHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    if "nbm" in sourceList:
        WindBearingHour[:, 0] = NBM_Merged[:, NBM["bearing"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindBearingHour[:, 1] = np.rad2deg(
            np.mod(
                np.arctan2(
                    HRRR_Merged[:, HRRR["wind_u"]], HRRR_Merged[:, HRRR["wind_v"]]
                )
                + np.pi,
                2 * np.pi,
            )
        )
    if "ecmwf_ifs" in sourceList:
        WindBearingHour[:, 2] = np.rad2deg(
            np.mod(
                np.arctan2(
                    ECMWF_Merged[:, ECMWF["wind_u"]], ECMWF_Merged[:, ECMWF["wind_v"]]
                )
                + np.pi,
                2 * np.pi,
            )
        )
    if "gfs" in sourceList:
        WindBearingHour[:, 3] = np.rad2deg(
            np.mod(
                np.arctan2(GFS_Merged[:, GFS["wind_u"]], GFS_Merged[:, GFS["wind_v"]])
                + np.pi,
                2 * np.pi,
            )
        )
    if "era5" in sourceList:
        WindBearingHour[:, 4] = np.rad2deg(
            np.mod(
                np.arctan2(
                    ERA5_MERGED[:, ERA5["10m_u_component_of_wind"]],
                    ERA5_MERGED[:, ERA5["10m_v_component_of_wind"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )

    InterPhour[:, DATA_HOURLY["bearing"]] = np.mod(
        np.choose(np.argmin(np.isnan(WindBearingHour), axis=1), WindBearingHour.T), 360
    )

    ### Cloud Cover
    CloudCoverHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    if "nbm" in sourceList:
        CloudCoverHour[:, 0] = NBM_Merged[:, NBM["cloud"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        CloudCoverHour[:, 1] = HRRR_Merged[:, HRRR["cloud"]]
    if "ecmwf_ifs" in sourceList:
        CloudCoverHour[:, 2] = ECMWF_Merged[:, ECMWF["cloud"]]
    if "gfs" in sourceList:
        CloudCoverHour[:, 3] = GFS_Merged[:, GFS["cloud"]]
    if "era5" in sourceList:
        CloudCoverHour[:, 4] = ERA5_MERGED[:, ERA5["total_cloud_cover"]] * 100

    InterPhour[:, DATA_HOURLY["cloud"]] = np.maximum(
        np.choose(np.argmin(np.isnan(CloudCoverHour), axis=1), CloudCoverHour.T) * 0.01,
        0,
    )
    # Clip between 0 and 1
    InterPhour[:, DATA_HOURLY["cloud"]] = clipLog(
        InterPhour[:, DATA_HOURLY["cloud"]],
        CLIP_CLOUD["min"],
        CLIP_CLOUD["max"],
        "Cloud Cover Hour",
    )

    ### UV Index
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["uv"]] = clipLog(
            GFS_Merged[:, GFS["uv"]] * 18.9 * 0.025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Hour",
        )

    elif "era5" in sourceList:
        # TODO: Implement a more accurate uv index
        InterPhour[:, DATA_HOURLY["uv"]] = clipLog(
            ERA5_MERGED[:, ERA5["downward_uv_radiation_at_the_surface"]]
            / 3600
            * 40
            * 0.0025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )

    ### Visibility
    VisibilityHour = np.full((len(hour_array_grib), 4), MISSING_DATA)
    if "nbm" in sourceList:
        VisibilityHour[:, 0] = NBM_Merged[:, NBM["vis"]]

        # Filter out missing visibility values
        # VisibilityHour[VisibilityHour[:, 0] < -1, 0] = MISSING_DATA
        # VisibilityHour[VisibilityHour[:, 0] > 1e6, 0] = MISSING_DATA
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        VisibilityHour[:, 1] = HRRR_Merged[:, HRRR["vis"]]
    if "gfs" in sourceList:
        VisibilityHour[:, 2] = GFS_Merged[:, GFS["vis"]]
    if "era5" in sourceList:
        VisibilityHour[:, 3] = estimate_visibility_gultepe_rh_pr_numpy(
            ERA5_MERGED, var_index=ERA5, var_axis=1
        )

    # Keep visibility in meters (SI units)
    InterPhour[:, DATA_HOURLY["vis"]] = np.clip(
        np.choose(np.argmin(np.isnan(VisibilityHour), axis=1), VisibilityHour.T),
        CLIP_VIS["min"],
        CLIP_VIS["max"],
    )

    ### Ozone Index
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["ozone"]] = clipLog(
            GFS_Merged[:, GFS["ozone"]],
            CLIP_OZONE["min"],
            CLIP_OZONE["max"],
            "Ozone Hour",
        )
    elif "era5" in sourceList:
        # Conversion from: https://sacs.aeronomie.be/info/dobson.php
        InterPhour[:, DATA_HOURLY["ozone"]] = clipLog(
            ERA5_MERGED[:, ERA5["total_column_ozone"]] * 46696,
            CLIP_OZONE["min"],
            CLIP_OZONE["max"],
            "Ozone Hour",
        )  # To convert to dobson units

    ### Precipitation Accumulation
    PrecpAccumHour = np.full((len(hour_array_grib), 6), MISSING_DATA)
    # NBM
    if "nbm" in sourceList:
        PrecpAccumHour[:, 0] = NBM_Merged[:, NBM["intensity"]]
    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        PrecpAccumHour[:, 1] = HRRR_Merged[:, HRRR["accum"]]
    # ECMWF
    if "ecmwf_ifs" in sourceList:
        # Use APCP_Mean for accumulation.
        # APCP_Mean is in m/h, convert to mm for accumulation units.
        PrecpAccumHour[:, 2] = ECMWF_Merged[:, ECMWF["accum_mean"]] * 1000
    # GEFS
    if "gefs" in sourceList:
        PrecpAccumHour[:, 3] = GEFS_Merged[:, GEFS["accum"]]
    # GFS
    if "gfs" in sourceList:
        PrecpAccumHour[:, 4] = GFS_Merged[:, GFS["accum"]]
    if "era5" in sourceList:
        PrecpAccumHour[:, 5] = (
            ERA5_MERGED[:, ERA5["total_precipitation"]] * 1000
        )  # m to mm

    # Set all values below 0.0005 (0.5 mm of snow/ 0.05 mm of rain) to zero
    PrecpAccumHour[PrecpAccumHour < 0.0005] = 0

    InterPhour[:, DATA_HOURLY["accum"]] = np.maximum(
        np.choose(np.argmin(np.isnan(PrecpAccumHour), axis=1), PrecpAccumHour.T),
        0,
    )

    # Set accumulation to zero if POP == 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] == 0, DATA_HOURLY["accum"]] = 0

    ### Near Storm Distance
    # Keep in meters (SI units)
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["storm_dist"]] = np.maximum(
            GFS_Merged[:, GFS["storm_dist"]], 0
        )

    ### Near Storm Direction
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["storm_dir"]] = GFS_Merged[:, GFS["storm_dir"]]

    # Air quality/ smoke
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPhour[:, DATA_HOURLY["smoke"]] = clipLog(
            HRRR_Merged[:, HRRR["smoke"]],
            CLIP_SMOKE["min"],
            CLIP_SMOKE["max"],
            "Air quality Hour",
        )  # Maximum US AQI value for PM2.5 (smoke) is 500 which corresponds to 500 PM2.5
    else:
        InterPhour[:, DATA_HOURLY["smoke"]] = MISSING_DATA

    # Fire Index
    if "nbm_fire" in sourceList:
        InterPhour[:, DATA_HOURLY["fire"]] = clipLog(
            NBM_Fire_Merged[:, NBM_FIRE_INDEX],
            CLIP_FIRE["min"],
            CLIP_FIRE["max"],
            "Fire Hour",
        )

    # Solar
    if "nbm" in sourceList:
        InterPhour[:, DATA_HOURLY["solar"]] = NBM_Merged[:, NBM["solar"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPhour[:, DATA_HOURLY["solar"]] = HRRR_Merged[:, HRRR["solar"]]
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["solar"]] = GFS_Merged[:, GFS["solar"]]
    if "era5" in sourceList:
        InterPhour[:, DATA_HOURLY["solar"]] = (
            ERA5_MERGED[:, ERA5["surface_solar_radiation_downwards"]] / 3600
        )  # J/m2 to W/m2

    InterPhour[:, DATA_HOURLY["solar"]] = clipLog(
        InterPhour[:, DATA_HOURLY["solar"]],
        CLIP_SOLAR["min"],
        CLIP_SOLAR["max"],
        "Solar Hour",
    )

    # Wind speed is already in m/s (SI units)
    windSpeedMps = InterPhour[:, DATA_HOURLY["wind"]]

    # Calculate the apparent temperature
    InterPhour[:, DATA_HOURLY["apparent"]] = calculate_apparent_temperature(
        InterPhour[:, DATA_HOURLY["temp"]],  # Air temperature in Kelvin
        InterPhour[:, DATA_HOURLY["humidity"]],  # Relative humidity (0.0 to 1.0)
        windSpeedMps,  # Wind speed in meters per second
        solar=InterPhour[:, DATA_HOURLY["solar"]],  # Solar radiation in W/m^2
    )

    ### Feels Like Temperature
    AppTemperatureHour = np.full((len(hour_array_grib), 2), MISSING_DATA)
    if "nbm" in sourceList:
        AppTemperatureHour[:, 0] = NBM_Merged[:, NBM["apparent"]]

    if "gfs" in sourceList:
        AppTemperatureHour[:, 1] = GFS_Merged[:, GFS["apparent"]]

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["feels_like"]] = np.choose(
        np.argmin(np.isnan(AppTemperatureHour), axis=1), AppTemperatureHour.T
    )

    # Clip between -90 and 60
    InterPhour[:, DATA_HOURLY["feels_like"]] = clipLog(
        InterPhour[:, DATA_HOURLY["feels_like"]],
        CLIP_FEELS_LIKE["min"],
        CLIP_FEELS_LIKE["max"],
        "Feels Like Hour",
    )

    # Station Pressure
    station_pressure_hour = np.full((len(hour_array_grib), 2), MISSING_DATA)
    if "gfs" in sourceList:
        station_pressure_hour[:, 0] = GFS_Merged[:, GFS["station_pressure"]]
    elif "era5" in sourceList:
        station_pressure_hour[:, 1] = ERA5_MERGED[:, ERA5["surface_pressure"]]

    InterPhour[:, DATA_HOURLY["station_pressure"]] = np.choose(
        np.argmin(np.isnan(station_pressure_hour), axis=1), station_pressure_hour.T
    )

    InterPhour[:, DATA_HOURLY["station_pressure"]] = clipLog(
        InterPhour[:, DATA_HOURLY["station_pressure"]],
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Station Pressure Hour",
    )

    # CAPE
    if "nbm" in sourceList:
        InterPhour[:, DATA_HOURLY["cape"]] = NBM_Merged[:, NBM["cape"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPhour[:, DATA_HOURLY["cape"]] = HRRR_Merged[:, HRRR["cape"]]
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["cape"]] = GFS_Merged[:, GFS["cape"]]
    if "era5" in sourceList:
        InterPhour[:, DATA_HOURLY["cape"]] = ERA5_MERGED[
            :, ERA5["convective_available_potential_energy"]
        ]

    InterPhour[:, DATA_HOURLY["cape"]] = clipLog(
        InterPhour[:, DATA_HOURLY["cape"]],
        CLIP_CAPE["min"],
        CLIP_CAPE["max"],
        "CAPE Hour",
    )

    # Keep temperatures in SI units (Celsius) - conversion happens when building output
    # Convert from Kelvin to Celsius for internal use
    # From here on out, temperature should be in Celsius
    InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["humidity"]] = (
        InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["humidity"]] - KELVIN_TO_CELSIUS
    )
    InterPhour[:, DATA_HOURLY["feels_like"]] = (
        InterPhour[:, DATA_HOURLY["feels_like"]] - KELVIN_TO_CELSIUS
    )

    # Add a global check for weird values, since nothing should ever be greater than 10000
    # Keep time col
    InterPhourData = InterPhour[:, DATA_HOURLY["type"] :]
    InterPhourData[InterPhourData > CLIP_GLOBAL["max"]] = MISSING_DATA
    InterPhourData[InterPhourData < CLIP_GLOBAL["min"]] = MISSING_DATA
    InterPhour[:, 1:] = InterPhourData

    hourList = []
    hourList_si = []
    hourIconList = []
    hourTextList = []

    # Find snow and liquid precip
    # Set to zero as baseline
    InterPhour[:, DATA_HOURLY["rain"]] = 0
    InterPhour[:, DATA_HOURLY["snow"]] = 0
    InterPhour[:, DATA_HOURLY["ice"]] = 0

    # Accumulations in liquid equivalent
    InterPhour[InterPhour[:, DATA_HOURLY["type"]] == 4, DATA_HOURLY["rain"]] = (
        InterPhour[InterPhour[:, DATA_HOURLY["type"]] == 4, DATA_HOURLY["accum"]]
    )  # rain

    # Use the new snow height estimation for snow accumulation.
    # Keep in SI units (mm)
    snow_indices = np.where(InterPhour[:, DATA_HOURLY["type"]] == 1)[0]
    if snow_indices.size > 0:
        # Extract data for all snow events - already in SI units (mm, Celsius, m/s)
        liquid_mm = InterPhour[snow_indices, DATA_HOURLY["accum"]]
        temp_c = InterPhour[snow_indices, DATA_HOURLY["temp"]]
        # windSpeedMps is already calculated as m/s above at line 3520
        wind_mps = windSpeedMps[snow_indices]
        # Calculate snow height for all snow indices in a vectorized operation (returns mm)
        snow_mm_values = estimate_snow_height(liquid_mm, temp_c, wind_mps)
        # Keep in mm (SI units)
        InterPhour[snow_indices, DATA_HOURLY["snow"]] = snow_mm_values

    InterPhour[
        (
            (InterPhour[:, DATA_HOURLY["type"]] == 2)
            | (InterPhour[:, DATA_HOURLY["type"]] == 3)
        ),
        DATA_HOURLY["ice"],
    ] = (
        InterPhour[
            (
                (InterPhour[:, DATA_HOURLY["type"]] == 2)
                | (InterPhour[:, DATA_HOURLY["type"]] == 3)
            ),
            DATA_HOURLY["accum"],
        ]
        * 1
    )  # Ice

    # Rain
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepRain = InterPhour[:, DATA_HOURLY["rain"]].copy()
    # Everything that isn't the current day
    dayZeroPrepRain[hourlyDayIndex != 0] = 0
    # Everything after the request time
    if not timeMachine:
        dayZeroPrepRain[int(baseTimeOffset) :] = 0

    # Snow
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepSnow = InterPhour[:, DATA_HOURLY["snow"]].copy()
    # Everything that isn't the current day
    dayZeroPrepSnow[hourlyDayIndex != 0] = 0
    # Everything after the request time
    if not timeMachine:
        dayZeroPrepSnow[int(baseTimeOffset) :] = 0

    # Sleet
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepSleet = InterPhour[:, DATA_HOURLY["ice"]].copy()
    # Everything that isn't the current day
    dayZeroPrepSleet[hourlyDayIndex != 0] = 0
    # Everything after the request time
    if not timeMachine:
        dayZeroPrepSleet[int(baseTimeOffset) :] = 0

    # Accumulations in liquid equivalent
    dayZeroRain = dayZeroPrepRain.sum()  # rain
    dayZeroSnow = dayZeroPrepSnow.sum()  # Snow
    dayZeroIce = dayZeroPrepSleet.sum()  # Ice

    # Zero prep intensity and accum before forecast time
    if not timeMachine:
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["intensity"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["accum"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["rain"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["snow"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["ice"]] = 0

        # Zero prep prob before forecast time
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["prob"]] = 0

    # Calculate type-specific intensities (in SI units - mm/h liquid equivalent)
    # Initialize all to zero
    InterPhour[:, DATA_HOURLY["rain_intensity"]] = 0
    InterPhour[:, DATA_HOURLY["snow_intensity"]] = 0
    InterPhour[:, DATA_HOURLY["ice_intensity"]] = 0

    # For ERA5 source, use separate rain and snow rates directly
    if "era5" in sourceList:
        # Rain intensity from ERA5 rain rates (already in mm/h)
        InterPhour[:, DATA_HOURLY["rain_intensity"]] = era5_rain_intensity

        # Snow intensity: convert ERA5 snow water equivalent to snow depth using temperature and wind
        era5_snow_intensity_si = estimate_snow_height(
            era5_snow_water_equivalent,  # mm/h of water equivalent
            InterPhour[:, DATA_HOURLY["temp"]],  # Celsius
            windSpeedMps,  # m/s
        )
        InterPhour[:, DATA_HOURLY["snow_intensity"]] = era5_snow_intensity_si

        # ERA5 doesn't provide separate ice/sleet rates, so sleet intensity remains 0
    else:
        # For non-ERA5 sources, derive type-specific intensities from main intensity
        # Rain intensity (direct from intensity)
        rain_mask = InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["rain"]
        InterPhour[rain_mask, DATA_HOURLY["rain_intensity"]] = InterPhour[
            rain_mask, DATA_HOURLY["intensity"]
        ]

        # Snow intensity (use liquid water conversion)
        snow_mask = InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["snow"]
        snow_indices = np.where(snow_mask)[0]
        if snow_indices.size > 0:
            # Convert snow accumulation to intensity using liquid water conversion
            snow_intensity_si = estimate_snow_height(
                InterPhour[
                    snow_indices, DATA_HOURLY["intensity"]
                ],  # mm/h of water equivalent
                InterPhour[snow_indices, DATA_HOURLY["temp"]],  # Celsius
                windSpeedMps[snow_indices],  # m/s
            )
            InterPhour[snow_indices, DATA_HOURLY["snow_intensity"]] = snow_intensity_si

        # Sleet intensity (direct from intensity for types 2 and 3)
        sleet_mask = (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["ice"]) | (
            InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["sleet"]
        )
        InterPhour[sleet_mask, DATA_HOURLY["ice_intensity"]] = InterPhour[
            sleet_mask, DATA_HOURLY["intensity"]
        ]

    # pTypeMap = {0: 'none', 1: 'snow', 2: 'sleet', 3: 'sleet', 4: 'rain'}
    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])
    PTypeHour = pTypeMap[
        np.nan_to_num(InterPhour[:, DATA_HOURLY["type"]], 0).astype(int)
    ]
    PTextHour = pTextMap[
        np.nan_to_num(InterPhour[:, DATA_HOURLY["type"]], 0).astype(int)
    ]

    # Fix very small neg from interp to solve -0
    InterPhour[((InterPhour >= -0.01) & (InterPhour <= 0.01))] = 0

    # Timing Check
    if TIMING:
        print("Hourly Loop start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # ===== OPTIMIZATION: Convert all units and apply rounding BEFORE the loop =====
    # This significantly improves performance by:
    # 1. Moving unit conversions out of the per-hour loop (vectorized operations)
    # 2. Applying rounding once to all values before object generation
    # This reduces the overhead from O(n*m) to O(n) where n=hours, m=fields
    # Create display arrays with all unit conversions applied at once
    hourly_display = np.zeros((numHours, max(DATA_HOURLY.values()) + 1))

    # Temperature conversions - vectorized
    if tempUnits == 0:  # Fahrenheit
        hourly_display[:, DATA_HOURLY["temp"]] = (
            InterPhour[:, DATA_HOURLY["temp"]] * 9 / 5 + 32
        )
        hourly_display[:, DATA_HOURLY["apparent"]] = (
            InterPhour[:, DATA_HOURLY["apparent"]] * 9 / 5 + 32
        )
        hourly_display[:, DATA_HOURLY["dew"]] = (
            InterPhour[:, DATA_HOURLY["dew"]] * 9 / 5 + 32
        )
        hourly_display[:, DATA_HOURLY["feels_like"]] = (
            InterPhour[:, DATA_HOURLY["feels_like"]] * 9 / 5 + 32
        )
    else:  # Celsius (already in Celsius)
        hourly_display[:, DATA_HOURLY["temp"]] = InterPhour[:, DATA_HOURLY["temp"]]
        hourly_display[:, DATA_HOURLY["apparent"]] = InterPhour[
            :, DATA_HOURLY["apparent"]
        ]
        hourly_display[:, DATA_HOURLY["dew"]] = InterPhour[:, DATA_HOURLY["dew"]]
        hourly_display[:, DATA_HOURLY["feels_like"]] = InterPhour[
            :, DATA_HOURLY["feels_like"]
        ]

    # Wind conversions - vectorized
    hourly_display[:, DATA_HOURLY["wind"]] = (
        InterPhour[:, DATA_HOURLY["wind"]] * windUnit
    )
    hourly_display[:, DATA_HOURLY["gust"]] = (
        InterPhour[:, DATA_HOURLY["gust"]] * windUnit
    )

    # Visibility conversion - vectorized
    hourly_display[:, DATA_HOURLY["vis"]] = InterPhour[:, DATA_HOURLY["vis"]] * visUnits

    # Precipitation conversions - vectorized
    hourly_display[:, DATA_HOURLY["intensity"]] = (
        InterPhour[:, DATA_HOURLY["intensity"]] * prepIntensityUnit
    )
    hourly_display[:, DATA_HOURLY["error"]] = (
        InterPhour[:, DATA_HOURLY["error"]] * prepIntensityUnit
    )
    hourly_display[:, DATA_HOURLY["rain"]] = (
        InterPhour[:, DATA_HOURLY["rain"]] * prepAccumUnit
    )
    hourly_display[:, DATA_HOURLY["snow"]] = (
        InterPhour[:, DATA_HOURLY["snow"]] * prepAccumUnit
    )
    hourly_display[:, DATA_HOURLY["ice"]] = (
        InterPhour[:, DATA_HOURLY["ice"]] * prepAccumUnit
    )
    hourly_display[:, DATA_HOURLY["rain_intensity"]] = (
        InterPhour[:, DATA_HOURLY["rain_intensity"]] * prepIntensityUnit
    )
    hourly_display[:, DATA_HOURLY["snow_intensity"]] = (
        InterPhour[:, DATA_HOURLY["snow_intensity"]] * prepIntensityUnit
    )
    hourly_display[:, DATA_HOURLY["ice_intensity"]] = (
        InterPhour[:, DATA_HOURLY["ice_intensity"]] * prepIntensityUnit
    )

    # Pressure conversion - vectorized (Pascals to hectopascals)
    hourly_display[:, DATA_HOURLY["pressure"]] = (
        InterPhour[:, DATA_HOURLY["pressure"]] / 100
    )

    # Storm distance conversion - vectorized
    hourly_display[:, DATA_HOURLY["storm_dist"]] = (
        InterPhour[:, DATA_HOURLY["storm_dist"]] * visUnits
    )

    # Copy unchanged fields
    hourly_display[:, DATA_HOURLY["prob"]] = InterPhour[:, DATA_HOURLY["prob"]]
    hourly_display[:, DATA_HOURLY["humidity"]] = InterPhour[:, DATA_HOURLY["humidity"]]
    hourly_display[:, DATA_HOURLY["bearing"]] = InterPhour[:, DATA_HOURLY["bearing"]]
    hourly_display[:, DATA_HOURLY["cloud"]] = InterPhour[:, DATA_HOURLY["cloud"]]
    hourly_display[:, DATA_HOURLY["uv"]] = InterPhour[:, DATA_HOURLY["uv"]]
    hourly_display[:, DATA_HOURLY["ozone"]] = InterPhour[:, DATA_HOURLY["ozone"]]
    hourly_display[:, DATA_HOURLY["smoke"]] = InterPhour[:, DATA_HOURLY["smoke"]]
    hourly_display[:, DATA_HOURLY["storm_dir"]] = InterPhour[
        :, DATA_HOURLY["storm_dir"]
    ]
    hourly_display[:, DATA_HOURLY["fire"]] = InterPhour[:, DATA_HOURLY["fire"]]
    hourly_display[:, DATA_HOURLY["solar"]] = InterPhour[:, DATA_HOURLY["solar"]]
    hourly_display[:, DATA_HOURLY["cape"]] = InterPhour[:, DATA_HOURLY["cape"]]
    if "stationPressure" in extraVars:
        hourly_display[:, DATA_HOURLY["station_pressure"]] = (
            InterPhour[:, DATA_HOURLY["station_pressure"]] / 100
        )

    # Apply rounding to the converted values - define mapping for hourly fields
    hourly_rounding_map = {
        DATA_HOURLY["temp"]: ROUNDING_RULES.get("temperature", 2),
        DATA_HOURLY["apparent"]: ROUNDING_RULES.get("apparentTemperature", 2),
        DATA_HOURLY["dew"]: ROUNDING_RULES.get("dewPoint", 2),
        DATA_HOURLY["feels_like"]: ROUNDING_RULES.get("feelsLike", 2),
        DATA_HOURLY["wind"]: ROUNDING_RULES.get("windSpeed", 2),
        DATA_HOURLY["gust"]: ROUNDING_RULES.get("windGust", 2),
        DATA_HOURLY["vis"]: ROUNDING_RULES.get("visibility", 2),
        DATA_HOURLY["intensity"]: ROUNDING_RULES.get("precipIntensity", 4),
        DATA_HOURLY["error"]: ROUNDING_RULES.get("precipIntensityError", 4),
        DATA_HOURLY["rain"]: ROUNDING_RULES.get("liquidAccumulation", 2),
        DATA_HOURLY["snow"]: ROUNDING_RULES.get("snowAccumulation", 2),
        DATA_HOURLY["ice"]: ROUNDING_RULES.get("iceAccumulation", 2),
        DATA_HOURLY["rain_intensity"]: ROUNDING_RULES.get("rainIntensity", 4),
        DATA_HOURLY["snow_intensity"]: ROUNDING_RULES.get("snowIntensity", 4),
        DATA_HOURLY["ice_intensity"]: ROUNDING_RULES.get("iceIntensity", 4),
        DATA_HOURLY["pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_HOURLY["storm_dist"]: ROUNDING_RULES.get("nearestStormDistance", 2),
        DATA_HOURLY["prob"]: ROUNDING_RULES.get("precipProbability", 2),
        DATA_HOURLY["humidity"]: ROUNDING_RULES.get("humidity", 2),
        DATA_HOURLY["cloud"]: ROUNDING_RULES.get("cloudCover", 2),
        DATA_HOURLY["uv"]: ROUNDING_RULES.get("uvIndex", 0),
        DATA_HOURLY["ozone"]: ROUNDING_RULES.get("ozone", 2),
        DATA_HOURLY["smoke"]: ROUNDING_RULES.get("smoke", 2),
        DATA_HOURLY["fire"]: ROUNDING_RULES.get("fireIndex", 2),
        DATA_HOURLY["solar"]: ROUNDING_RULES.get("solar", 2),
        DATA_HOURLY["cape"]: ROUNDING_RULES.get("cape", 0),
        DATA_HOURLY["windBearing"]: ROUNDING_RULES.get("cape", 0),
    }

    # Apply rounding in-place to the hourly_display array
    for idx_field, decimals in hourly_rounding_map.items():
        if decimals == 0:
            hourly_display[:, idx_field] = np.round(
                hourly_display[:, idx_field]
            ).astype(int)
        else:
            hourly_display[:, idx_field] = np.round(
                hourly_display[:, idx_field], decimals
            )

    # for idx in range(int(baseTimeOffset), hourly_hours + int(baseTimeOffset)):
    # For day 0 summary, need to calculate hourly data from midnight local
    for idx in range(0, numHours):
        # Check if day or night
        if hour_array_grib[idx] < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]:
            isDay = False
        elif (
            hour_array_grib[idx] >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            and hour_array_grib[idx]
            <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
        ):
            isDay = True
        elif hour_array_grib[idx] > InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]:
            isDay = False

        # Set text
        # Thresholds are in SI units (mm for precipitation, meters for visibility, m/s for wind)
        if InterPhour[idx, DATA_HOURLY["prob"]] >= PRECIP_PROB_THRESHOLD and (
            (
                (
                    InterPhour[idx, DATA_HOURLY["rain"]]
                    + InterPhour[idx, DATA_HOURLY["ice"]]
                )
                > HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM
            )
            or (
                InterPhour[idx, DATA_HOURLY["snow"]]
                > HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM
            )
        ):
            # If more than 30% chance of precip at any point throughout the day, then the icon for whatever is happening
            # Thresholds set in mm
            hourIcon = PTypeHour[idx]
            hourText = PTextHour[idx]
        # If visibility < FOG_THRESHOLD_METERS
        elif InterPhour[idx, DATA_HOURLY["vis"]] < FOG_THRESHOLD_METERS:
            hourIcon = "fog"
            hourText = "Fog"
        # If wind is greater than light wind threshold (m/s)
        elif InterPhour[idx, DATA_HOURLY["wind"]] > WIND_THRESHOLDS["light"]:
            hourIcon = "wind"
            hourText = "Windy"
        elif InterPhour[idx, DATA_HOURLY["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
            hourIcon = "cloudy"
            hourText = "Cloudy"
        elif (
            InterPhour[idx, DATA_HOURLY["cloud"]]
            > CLOUD_COVER_THRESHOLDS["partly_cloudy"]
        ):
            hourText = "Partly Cloudy"

            if (
                hour_array_grib[idx]
                < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            ):
                # Before sunrise
                hourIcon = "partly-cloudy-night"
            elif (
                hour_array_grib[idx]
                >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
                and hour_array_grib[idx]
                <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunrise before sunset
                hourIcon = "partly-cloudy-day"
            elif (
                hour_array_grib[idx]
                > InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunset
                hourIcon = "partly-cloudy-night"
        else:
            hourText = "Clear"

            if (
                hour_array_grib[idx]
                < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            ):
                # Before sunrise
                hourIcon = "clear-night"
            elif (
                hour_array_grib[idx]
                >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
                and hour_array_grib[idx]
                <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunrise before sunset
                hourIcon = "clear-day"
            elif (
                hour_array_grib[idx]
                > InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunset
                hourIcon = "clear-night"

        # Use pre-converted and rounded values from hourly_display
        accum_display = (
            hourly_display[idx, DATA_HOURLY["rain"]]
            + hourly_display[idx, DATA_HOURLY["snow"]]
            + hourly_display[idx, DATA_HOURLY["ice"]]
        )

        hourItem = {
            "time": int(hour_array_grib[idx])
            if not np.isnan(hour_array_grib[idx])
            else 0,
            "summary": hourText,
            "icon": hourIcon,
            "precipIntensity": hourly_display[idx, DATA_HOURLY["intensity"]],
            "precipProbability": hourly_display[idx, DATA_HOURLY["prob"]],
            "precipIntensityError": hourly_display[idx, DATA_HOURLY["error"]],
            "precipAccumulation": accum_display,
            "precipType": PTypeHour[idx],
            "rainIntensity": hourly_display[idx, DATA_HOURLY["rain_intensity"]],
            "snowIntensity": hourly_display[idx, DATA_HOURLY["snow_intensity"]],
            "iceIntensity": hourly_display[idx, DATA_HOURLY["ice_intensity"]],
            "temperature": hourly_display[idx, DATA_HOURLY["temp"]],
            "apparentTemperature": hourly_display[idx, DATA_HOURLY["apparent"]],
            "dewPoint": hourly_display[idx, DATA_HOURLY["dew"]],
            "humidity": hourly_display[idx, DATA_HOURLY["humidity"]],
            "pressure": hourly_display[idx, DATA_HOURLY["pressure"]],
            "windSpeed": hourly_display[idx, DATA_HOURLY["wind"]],
            "windGust": hourly_display[idx, DATA_HOURLY["gust"]],
            "windBearing": int(hourly_display[idx, DATA_HOURLY["bearing"]])
            if not np.isnan(hourly_display[idx, DATA_HOURLY["bearing"]])
            else 0,
            "cloudCover": hourly_display[idx, DATA_HOURLY["cloud"]],
            "uvIndex": hourly_display[idx, DATA_HOURLY["uv"]],
            "visibility": hourly_display[idx, DATA_HOURLY["vis"]],
            "ozone": hourly_display[idx, DATA_HOURLY["ozone"]],
            "smoke": hourly_display[idx, DATA_HOURLY["smoke"]],
            "liquidAccumulation": hourly_display[idx, DATA_HOURLY["rain"]],
            "snowAccumulation": hourly_display[idx, DATA_HOURLY["snow"]],
            "iceAccumulation": hourly_display[idx, DATA_HOURLY["ice"]],
            "nearestStormDistance": hourly_display[idx, DATA_HOURLY["storm_dist"]],
            "nearestStormBearing": int(hourly_display[idx, DATA_HOURLY["storm_dir"]])
            if not np.isnan(hourly_display[idx, DATA_HOURLY["storm_dir"]])
            else 0,
            "fireIndex": hourly_display[idx, DATA_HOURLY["fire"]],
            "feelsLike": hourly_display[idx, DATA_HOURLY["feels_like"]],
            "solar": hourly_display[idx, DATA_HOURLY["solar"]],
            "cape": int(hourly_display[idx, DATA_HOURLY["cape"]])
            if not np.isnan(hourly_display[idx, DATA_HOURLY["cape"]])
            else 0,
        }

        # Add station pressure if requested
        if "stationPressure" in extraVars:
            hourItem["stationPressure"] = hourly_display[
                idx, DATA_HOURLY["station_pressure"]
            ]

        # Create SI version of hourItem for text generation (values already in SI units in InterPhour)
        hourItem_si = {
            "time": int(hour_array_grib[idx]),
            "temperature": InterPhour[idx, DATA_HOURLY["temp"]],
            "dewPoint": InterPhour[idx, DATA_HOURLY["dew"]],
            "humidity": InterPhour[idx, DATA_HOURLY["humidity"]],
            "windSpeed": InterPhour[idx, DATA_HOURLY["wind"]],
            "visibility": InterPhour[idx, DATA_HOURLY["vis"]],
            "cloudCover": InterPhour[idx, DATA_HOURLY["cloud"]],
            "smoke": InterPhour[idx, DATA_HOURLY["smoke"]],
            "precipType": PTypeHour[idx],
            "precipProbability": InterPhour[idx, DATA_HOURLY["prob"]],
            "cape": InterPhour[idx, DATA_HOURLY["cape"]],
            "liquidAccumulation": InterPhour[idx, DATA_HOURLY["rain"]],
            "snowAccumulation": InterPhour[idx, DATA_HOURLY["snow"]],
            "iceAccumulation": InterPhour[idx, DATA_HOURLY["ice"]],
            "rainIntensity": InterPhour[idx, DATA_HOURLY["rain_intensity"]],
            "snowIntensity": InterPhour[idx, DATA_HOURLY["snow_intensity"]],
            "iceIntensity": InterPhour[idx, DATA_HOURLY["ice_intensity"]],
            "precipIntensity": InterPhour[
                idx, DATA_HOURLY["intensity"]
            ],  # mm/h, SI, liquid equivalent
            "precipIntensityError": InterPhour[
                idx, DATA_HOURLY["error"]
            ],  # mm, SI, accumulation error
        }

        try:
            if summaryText:
                hourText, hourIcon = calculate_text(
                    hourItem_si,
                    isDay,
                    "hour",
                    icon,
                )
                hourItem["summary"] = translation.translate(["title", hourText])
                hourItem["icon"] = hourIcon

        except Exception:
            logger.exception("HOURLY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            hourItem.pop("liquidAccumulation", None)
            hourItem.pop("snowAccumulation", None)
            hourItem.pop("iceAccumulation", None)
            hourItem.pop("nearestStormDistance", None)
            hourItem.pop("nearestStormBearing", None)
            hourItem.pop("fireIndex", None)
            hourItem.pop("feelsLike", None)
            hourItem.pop("solar", None)
            hourItem.pop("rainIntensity", None)
            hourItem.pop("snowIntensity", None)
            hourItem.pop("iceIntensity", None)
            hourItem.pop("cape", None)

        if timeMachine and not tmExtra:
            hourItem.pop("uvIndex", None)
            hourItem.pop("ozone", None)

        hourList.append(hourItem)
        hourList_si.append(hourItem_si)

        hourIconList.append(hourIcon)
        hourTextList.append(hourItem["summary"])

    # Daily calculations #################################################
    # Timing Check
    if TIMING:
        print("Daily start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    mean_results = []
    sum_results = []
    max_results = []
    min_results = []
    argmax_results = []
    argmin_results = []
    high_results = []
    low_results = []
    arghigh_results = []
    arglow_results = []
    mean_4am_results = []
    sum_4am_results = []
    max_4am_results = []
    mean_day_results = []
    sum_day_results = []
    max_day_results = []
    mean_night_results = []
    sum_night_results = []
    max_night_results = []
    maxPchanceDay = np.zeros((daily_days))
    max_precip_chance_day = np.zeros((daily_days))
    max_precip_chance_night = np.zeros((daily_days))

    # Pre-calculate masks for each group to avoid redundant computation
    masks = [hourlyDayIndex == day_index for day_index in range(daily_days)]
    for mask in masks:
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        mean_results.append(np.mean(filtered_data, axis=0))
        sum_results.append(np.sum(filtered_data, axis=0))
        max_results.append(np.max(filtered_data, axis=0))
        min_results.append(np.min(filtered_data, axis=0))
        maxTime = np.argmax(filtered_data, axis=0)
        minTime = np.argmin(filtered_data, axis=0)
        argmax_results.append(filtered_data[maxTime, 0])
        argmin_results.append(filtered_data[minTime, 0])

    # Icon/ summary parameters go from 4 am to 4 am
    masks = [hourlyDay4amIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        mean_4am_results.append(np.mean(filtered_data, axis=0))
        sum_4am_results.append(np.sum(filtered_data, axis=0))
        max_4am_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        # Check if the most common type is zero, in that case return the second most common
        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                maxPchanceDay[mIDX] = dailyTypeCount[1][0]
            else:
                maxPchanceDay[mIDX] = dailyTypeCount[0][
                    0
                ]  # If all ptypes are none, then really shouldn't be any precipitation

        else:
            maxPchanceDay[mIDX] = dailyTypeCount[0][0]

    # Day portion of half day runs from 4am to 4pm
    masks = [hourlyDay4pmIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        mean_day_results.append(np.mean(filtered_data, axis=0))
        sum_day_results.append(np.sum(filtered_data, axis=0))
        max_day_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        # Check if the most common type is zero, in that case return the second most common
        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                max_precip_chance_day[mIDX] = dailyTypeCount[1][0]
            else:
                max_precip_chance_day[mIDX] = dailyTypeCount[0][
                    0
                ]  # If all ptypes are none, then really shouldn't be any precipitation

        else:
            max_precip_chance_day[mIDX] = dailyTypeCount[0][0]

    # Night portion of half day runs from 5pm to 4am the next day
    masks = [hourlyNight4amIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        mean_night_results.append(np.mean(filtered_data, axis=0))
        sum_night_results.append(np.sum(filtered_data, axis=0))
        max_night_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        # Check if the most common type is zero, in that case return the second most common
        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                max_precip_chance_night[mIDX] = dailyTypeCount[1][0]
            else:
                max_precip_chance_night[mIDX] = dailyTypeCount[0][
                    0
                ]  # If all ptypes are none, then really shouldn't be any precipitation

        else:
            max_precip_chance_night[mIDX] = dailyTypeCount[0][0]

    # Daily High
    masks = [hourlyHighIndex == day_index for day_index in range(daily_days)]

    for mask in masks:
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        high_results.append(np.max(filtered_data, axis=0))
        maxTime = np.argmax(filtered_data, axis=0)
        arghigh_results.append(filtered_data[maxTime, 0])

    # Daily Low
    masks = [hourlyLowIndex == day_index for day_index in range(daily_days)]

    for mask in masks:
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        low_results.append(np.min(filtered_data, axis=0))
        minTime = np.argmin(filtered_data, axis=0)
        arglow_results.append(filtered_data[minTime, 0])

    # Convert lists to numpy arrays if necessary
    InterPday = np.array(mean_results)
    InterPdaySum = np.array(sum_results)
    InterPdayMax = np.array(max_results)
    InterPdayMin = np.array(min_results)
    InterPdayMaxTime = np.array(argmax_results)
    InterPdayMinTime = np.array(argmin_results)
    InterPdayHigh = np.array(high_results)
    InterPdayLow = np.array(low_results)
    InterPdayHighTime = np.array(arghigh_results)
    InterPdayLowTime = np.array(arglow_results)
    InterPday4am = np.array(mean_4am_results)
    InterPdaySum4am = np.array(sum_4am_results)
    InterPdayMax4am = np.array(max_4am_results)
    interp_half_day_sum = np.array(sum_day_results)
    interp_half_day_mean = np.array(mean_day_results)
    interp_half_day_max = np.array(max_day_results)
    interp_half_night_sum = np.array(sum_night_results)
    interp_half_night_mean = np.array(mean_night_results)
    interp_half_night_max = np.array(max_night_results)

    # Determine the daily precipitation type (encapsulated helper)
    try:
        maxPchanceDay = select_daily_precip_type(
            InterPdaySum, DATA_DAY, maxPchanceDay, PRECIP_IDX, prepAccumUnit
        )
        max_precip_chance_day = select_daily_precip_type(
            interp_half_day_sum,
            DATA_DAY,
            max_precip_chance_day,
            PRECIP_IDX,
            prepAccumUnit,
        )
        max_precip_chance_night = select_daily_precip_type(
            interp_half_night_sum,
            DATA_DAY,
            max_precip_chance_night,
            PRECIP_IDX,
            prepAccumUnit,
        )
    except Exception:
        # Fallback: preserve original inline logic if helper fails (shouldn't happen)
        logger.exception("select_daily_precip_type error %s", loc_tag)

    # Process Day/Night data for output
    day_night_list = []
    max_precip_chance_day = np.array(max_precip_chance_day).astype(int)
    precip_type_half_day = pTypeMap[max_precip_chance_day]
    precip_text_half_day = pTextMap[max_precip_chance_day]
    max_precip_chance_night = np.array(max_precip_chance_night).astype(int)
    precip_type_half_night = pTypeMap[max_precip_chance_night]
    precip_text_half_night = pTextMap[max_precip_chance_night]

    # Process Daily Data for ouput
    dayList = []
    dayList_si = []
    dayIconList = []
    dayTextList = []

    maxPchanceDay = np.array(maxPchanceDay).astype(int)
    PTypeDay = pTypeMap[maxPchanceDay]
    PTextDay = pTextMap[maxPchanceDay]

    if TIMING:
        print("Daily Loop start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # ===== OPTIMIZATION: Convert all units and apply rounding BEFORE the loop =====
    # This significantly improves performance by:
    # 1. Moving unit conversions out of the per-day loop (vectorized operations)
    # 2. Applying rounding once to all values before object generation
    # Similar to hourly optimization, reduces overhead from O(n*m) to O(n) where n=days, m=fields

    # Helper for temperature conversion (C to F) if needed
    def _conv_temp(arr):
        return arr * 9 / 5 + 32 if tempUnits == 0 else arr

    # Build enumerated display arrays for daily mean/max/min/high/low with unit conversions
    daily_display_mean = InterPday.copy()
    daily_display_mean[:, DATA_DAY["dew"]] = _conv_temp(InterPday[:, DATA_DAY["dew"]])
    daily_display_mean[:, DATA_DAY["pressure"]] = (
        InterPday[:, DATA_DAY["pressure"]] / 100
    )
    daily_display_mean[:, DATA_DAY["wind"]] = InterPday[:, DATA_DAY["wind"]] * windUnit
    daily_display_mean[:, DATA_DAY["gust"]] = InterPday[:, DATA_DAY["gust"]] * windUnit
    daily_display_mean[:, DATA_DAY["vis"]] = InterPday[:, DATA_DAY["vis"]] * visUnits
    daily_display_mean[:, DATA_DAY["intensity"]] = (
        InterPday[:, DATA_DAY["intensity"]] * prepIntensityUnit
    )
    daily_display_mean[:, DATA_DAY["rain_intensity"]] = (
        InterPday[:, DATA_DAY["rain_intensity"]] * prepIntensityUnit
    )
    daily_display_mean[:, DATA_DAY["snow_intensity"]] = (
        InterPday[:, DATA_DAY["snow_intensity"]] * prepIntensityUnit
    )
    daily_display_mean[:, DATA_DAY["ice_intensity"]] = (
        InterPday[:, DATA_DAY["ice_intensity"]] * prepIntensityUnit
    )

    daily_display_high = InterPdayHigh.copy()
    daily_display_high[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayHigh[:, DATA_DAY["temp"]]
    )
    daily_display_high[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayHigh[:, DATA_DAY["apparent"]]
    )

    daily_display_low = InterPdayLow.copy()
    daily_display_low[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayLow[:, DATA_DAY["temp"]]
    )
    daily_display_low[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayLow[:, DATA_DAY["apparent"]]
    )

    daily_display_min = InterPdayMin.copy()
    daily_display_min[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayMin[:, DATA_DAY["temp"]]
    )
    daily_display_min[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayMin[:, DATA_DAY["apparent"]]
    )

    daily_display_max = InterPdayMax.copy()
    daily_display_max[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayMax[:, DATA_DAY["temp"]]
    )
    daily_display_max[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayMax[:, DATA_DAY["apparent"]]
    )
    daily_display_max[:, DATA_DAY["intensity"]] = (
        InterPdayMax[:, DATA_DAY["intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["rain_intensity"]] = (
        InterPdayMax[:, DATA_DAY["rain_intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["snow_intensity"]] = (
        InterPdayMax[:, DATA_DAY["snow_intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["ice_intensity"]] = (
        InterPdayMax[:, DATA_DAY["ice_intensity"]] * prepIntensityUnit
    )

    daily_display_sum = InterPdaySum.copy()
    daily_display_sum[:, DATA_DAY["rain"]] = (
        InterPdaySum[:, DATA_DAY["rain"]] * prepAccumUnit
    )
    daily_display_sum[:, DATA_DAY["snow"]] = (
        InterPdaySum[:, DATA_DAY["snow"]] * prepAccumUnit
    )
    daily_display_sum[:, DATA_DAY["ice"]] = (
        InterPdaySum[:, DATA_DAY["ice"]] * prepAccumUnit
    )

    # Half-day display arrays (use hourly indices)
    half_day_display_mean = interp_half_day_mean.copy()
    half_day_display_mean[:, DATA_HOURLY["dew"]] = _conv_temp(
        interp_half_day_mean[:, DATA_HOURLY["dew"]]
    )
    half_day_display_mean[:, DATA_HOURLY["pressure"]] = (
        interp_half_day_mean[:, DATA_HOURLY["pressure"]] / 100
    )
    half_day_display_mean[:, DATA_HOURLY["wind"]] = (
        interp_half_day_mean[:, DATA_HOURLY["wind"]] * windUnit
    )
    half_day_display_mean[:, DATA_HOURLY["gust"]] = (
        interp_half_day_mean[:, DATA_HOURLY["gust"]] * windUnit
    )
    half_day_display_mean[:, DATA_HOURLY["vis"]] = (
        interp_half_day_mean[:, DATA_HOURLY["vis"]] * visUnits
    )
    half_day_display_mean[:, DATA_HOURLY["intensity"]] = (
        interp_half_day_mean[:, DATA_HOURLY["intensity"]] * prepIntensityUnit
    )
    half_day_display_mean[:, DATA_HOURLY["rain"]] = (
        interp_half_day_mean[:, DATA_HOURLY["rain"]] * prepIntensityUnit
    )
    half_day_display_mean[:, DATA_HOURLY["snow"]] = (
        interp_half_day_mean[:, DATA_HOURLY["snow"]] * prepIntensityUnit
    )
    half_day_display_mean[:, DATA_HOURLY["ice"]] = (
        interp_half_day_mean[:, DATA_HOURLY["ice"]] * prepIntensityUnit
    )

    half_day_display_max = interp_half_day_max.copy()
    half_day_display_max[:, DATA_HOURLY["intensity"]] = (
        interp_half_day_max[:, DATA_HOURLY["intensity"]] * prepIntensityUnit
    )
    half_day_display_max[:, DATA_HOURLY["rain"]] = (
        interp_half_day_max[:, DATA_HOURLY["rain"]] * prepIntensityUnit
    )
    half_day_display_max[:, DATA_HOURLY["snow"]] = (
        interp_half_day_max[:, DATA_HOURLY["snow"]] * prepIntensityUnit
    )
    half_day_display_max[:, DATA_HOURLY["ice"]] = (
        interp_half_day_max[:, DATA_HOURLY["ice"]] * prepIntensityUnit
    )

    half_day_display_sum = interp_half_day_sum.copy()
    half_day_display_sum[:, DATA_HOURLY["rain"]] = (
        interp_half_day_sum[:, DATA_HOURLY["rain"]] * prepAccumUnit
    )
    half_day_display_sum[:, DATA_HOURLY["snow"]] = (
        interp_half_day_sum[:, DATA_HOURLY["snow"]] * prepAccumUnit
    )
    half_day_display_sum[:, DATA_HOURLY["ice"]] = (
        interp_half_day_sum[:, DATA_HOURLY["ice"]] * prepAccumUnit
    )

    half_night_display_mean = interp_half_night_mean.copy()
    half_night_display_mean[:, DATA_HOURLY["dew"]] = _conv_temp(
        interp_half_night_mean[:, DATA_HOURLY["dew"]]
    )
    half_night_display_mean[:, DATA_HOURLY["pressure"]] = (
        interp_half_night_mean[:, DATA_HOURLY["pressure"]] / 100
    )
    half_night_display_mean[:, DATA_HOURLY["wind"]] = (
        interp_half_night_mean[:, DATA_HOURLY["wind"]] * windUnit
    )
    half_night_display_mean[:, DATA_HOURLY["gust"]] = (
        interp_half_night_mean[:, DATA_HOURLY["gust"]] * windUnit
    )
    half_night_display_mean[:, DATA_HOURLY["vis"]] = (
        interp_half_night_mean[:, DATA_HOURLY["vis"]] * visUnits
    )
    half_night_display_mean[:, DATA_HOURLY["intensity"]] = (
        interp_half_night_mean[:, DATA_HOURLY["intensity"]] * prepIntensityUnit
    )
    half_night_display_mean[:, DATA_HOURLY["rain"]] = (
        interp_half_night_mean[:, DATA_HOURLY["rain"]] * prepIntensityUnit
    )
    half_night_display_mean[:, DATA_HOURLY["snow"]] = (
        interp_half_night_mean[:, DATA_HOURLY["snow"]] * prepIntensityUnit
    )
    half_night_display_mean[:, DATA_HOURLY["ice"]] = (
        interp_half_night_mean[:, DATA_HOURLY["ice"]] * prepIntensityUnit
    )

    half_night_display_max = interp_half_night_max.copy()
    half_night_display_max[:, DATA_HOURLY["intensity"]] = (
        interp_half_night_max[:, DATA_HOURLY["intensity"]] * prepIntensityUnit
    )
    half_night_display_max[:, DATA_HOURLY["rain"]] = (
        interp_half_night_max[:, DATA_HOURLY["rain"]] * prepIntensityUnit
    )
    half_night_display_max[:, DATA_HOURLY["snow"]] = (
        interp_half_night_max[:, DATA_HOURLY["snow"]] * prepIntensityUnit
    )
    half_night_display_max[:, DATA_HOURLY["ice"]] = (
        interp_half_night_max[:, DATA_HOURLY["ice"]] * prepIntensityUnit
    )

    half_night_display_sum = interp_half_night_sum.copy()
    half_night_display_sum[:, DATA_HOURLY["rain"]] = (
        interp_half_night_sum[:, DATA_HOURLY["rain"]] * prepAccumUnit
    )
    half_night_display_sum[:, DATA_HOURLY["snow"]] = (
        interp_half_night_sum[:, DATA_HOURLY["snow"]] * prepAccumUnit
    )
    half_night_display_sum[:, DATA_HOURLY["ice"]] = (
        interp_half_night_sum[:, DATA_HOURLY["ice"]] * prepAccumUnit
    )

    if "stationPressure" in extraVars:
        daily_display_mean[:, DATA_DAY["station_pressure"]] = (
            InterPday[:, DATA_DAY["station_pressure"]] / 100
        )
        half_day_display_mean[:, DATA_HOURLY["station_pressure"]] = (
            interp_half_day_mean[:, DATA_HOURLY["station_pressure"]] / 100
        )
        half_night_display_mean[:, DATA_HOURLY["station_pressure"]] = (
            interp_half_night_mean[:, DATA_HOURLY["station_pressure"]] / 100
        )

    # Rounding maps using enumerated indices
    daily_mean_rounding_map = {
        DATA_DAY["dew"]: ROUNDING_RULES.get("dewPoint", 2),
        DATA_DAY["pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_DAY["wind"]: ROUNDING_RULES.get("windSpeed", 2),
        DATA_DAY["gust"]: ROUNDING_RULES.get("windGust", 2),
        DATA_DAY["vis"]: ROUNDING_RULES.get("visibility", 2),
        DATA_DAY["intensity"]: ROUNDING_RULES.get("precipIntensity", 4),
        DATA_DAY["rain_intensity"]: ROUNDING_RULES.get("rainIntensity", 4),
        DATA_DAY["snow_intensity"]: ROUNDING_RULES.get("snowIntensity", 4),
        DATA_DAY["ice_intensity"]: ROUNDING_RULES.get("iceIntensity", 4),
        DATA_DAY["prob"]: ROUNDING_RULES.get("precipProbability", 2),
        DATA_DAY["humidity"]: ROUNDING_RULES.get("humidity", 2),
        DATA_DAY["cloud"]: ROUNDING_RULES.get("cloudCover", 2),
        DATA_DAY["uv"]: ROUNDING_RULES.get("uvIndex", 0),
        DATA_DAY["smoke"]: ROUNDING_RULES.get("smoke", 2),
        DATA_DAY["fire"]: ROUNDING_RULES.get("fireIndex", 2),
        DATA_DAY["solar"]: ROUNDING_RULES.get("solar", 2),
        DATA_DAY["station_pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_DAY["cape"]: ROUNDING_RULES.get("cape", 0),
        DATA_DAY["windBearing"]: ROUNDING_RULES.get("cape", 0),
    }

    for idx_field, decimals in daily_mean_rounding_map.items():
        if decimals == 0:
            daily_display_mean[:, idx_field] = np.round(
                daily_display_mean[:, idx_field]
            ).astype(int)
        else:
            daily_display_mean[:, idx_field] = np.round(
                daily_display_mean[:, idx_field], decimals
            )

    # Rounding for high/low/min/max temps and apparents
    temp_dec = ROUNDING_RULES.get("temperature", 2)
    app_dec = ROUNDING_RULES.get("apparentTemperature", 2)
    daily_display_high[:, DATA_DAY["temp"]] = np.round(
        daily_display_high[:, DATA_DAY["temp"]], temp_dec
    )
    daily_display_low[:, DATA_DAY["temp"]] = np.round(
        daily_display_low[:, DATA_DAY["temp"]], temp_dec
    )
    daily_display_min[:, DATA_DAY["temp"]] = np.round(
        daily_display_min[:, DATA_DAY["temp"]], temp_dec
    )
    daily_display_max[:, DATA_DAY["temp"]] = np.round(
        daily_display_max[:, DATA_DAY["temp"]], temp_dec
    )
    daily_display_high[:, DATA_DAY["apparent"]] = np.round(
        daily_display_high[:, DATA_DAY["apparent"]], app_dec
    )
    daily_display_low[:, DATA_DAY["apparent"]] = np.round(
        daily_display_low[:, DATA_DAY["apparent"]], app_dec
    )
    daily_display_min[:, DATA_DAY["apparent"]] = np.round(
        daily_display_min[:, DATA_DAY["apparent"]], app_dec
    )
    daily_display_max[:, DATA_DAY["apparent"]] = np.round(
        daily_display_max[:, DATA_DAY["apparent"]], app_dec
    )

    # Rounding for max intensities
    for idx_field in (
        DATA_DAY["intensity"],
        DATA_DAY["rain_intensity"],
        DATA_DAY["snow_intensity"],
        DATA_DAY["ice_intensity"],
        DATA_DAY["uv"],
        DATA_DAY["smoke"],
        DATA_DAY["fire"],
        DATA_DAY["solar"],
        DATA_DAY["prob"],
    ):
        dec = daily_mean_rounding_map.get(idx_field, 2)
        if dec == 0:
            daily_display_max[:, idx_field] = np.round(
                daily_display_max[:, idx_field]
            ).astype(int)
        else:
            daily_display_max[:, idx_field] = np.round(
                daily_display_max[:, idx_field], dec
            )

    # Rounding for accumulations (sum)
    accum_dec = ROUNDING_RULES.get("precipAccumulation", 2)
    for idx_field in (DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]):
        daily_display_sum[:, idx_field] = np.round(
            daily_display_sum[:, idx_field], accum_dec
        )

    # Half-day rounding maps (use hourly indices similar to hourly_rounding_map)
    half_rounding_map = {
        DATA_HOURLY["dew"]: ROUNDING_RULES.get("dewPoint", 2),
        DATA_HOURLY["pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_HOURLY["wind"]: ROUNDING_RULES.get("windSpeed", 2),
        DATA_HOURLY["gust"]: ROUNDING_RULES.get("windGust", 2),
        DATA_HOURLY["vis"]: ROUNDING_RULES.get("visibility", 2),
        DATA_HOURLY["intensity"]: ROUNDING_RULES.get("precipIntensity", 4),
        DATA_HOURLY["rain"]: ROUNDING_RULES.get("rainIntensity", 4),
        DATA_HOURLY["snow"]: ROUNDING_RULES.get("snowIntensity", 4),
        DATA_HOURLY["ice"]: ROUNDING_RULES.get("iceIntensity", 4),
        DATA_HOURLY["prob"]: ROUNDING_RULES.get("precipProbability", 2),
        DATA_HOURLY["humidity"]: ROUNDING_RULES.get("humidity", 2),
        DATA_HOURLY["cloud"]: ROUNDING_RULES.get("cloudCover", 2),
        DATA_HOURLY["uv"]: ROUNDING_RULES.get("uvIndex", 0),
        DATA_HOURLY["ozone"]: ROUNDING_RULES.get("ozone", 2),
        DATA_HOURLY["smoke"]: ROUNDING_RULES.get("smoke", 2),
        DATA_HOURLY["fire"]: ROUNDING_RULES.get("fireIndex", 2),
        DATA_HOURLY["solar"]: ROUNDING_RULES.get("solar", 2),
        DATA_HOURLY["station_pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_HOURLY["cape"]: ROUNDING_RULES.get("cape", 0),
        DATA_HOURLY["windBearing"]: ROUNDING_RULES.get("windBearing", 0),
    }

    def _apply_rounding_to(arr, rounding_map):
        for idx_field, decimals in rounding_map.items():
            if decimals == 0:
                arr[:, idx_field] = np.round(arr[:, idx_field]).astype(int)
            else:
                arr[:, idx_field] = np.round(arr[:, idx_field], decimals)

    _apply_rounding_to(half_day_display_mean, half_rounding_map)
    _apply_rounding_to(half_day_display_max, half_rounding_map)
    _apply_rounding_to(half_night_display_mean, half_rounding_map)
    _apply_rounding_to(half_night_display_max, half_rounding_map)

    # Accum rounding for half-day sums
    half_day_display_sum[:, DATA_HOURLY["rain"]] = np.round(
        half_day_display_sum[:, DATA_HOURLY["rain"]], accum_dec
    )
    half_day_display_sum[:, DATA_HOURLY["snow"]] = np.round(
        half_day_display_sum[:, DATA_HOURLY["snow"]], accum_dec
    )
    half_day_display_sum[:, DATA_HOURLY["ice"]] = np.round(
        half_day_display_sum[:, DATA_HOURLY["ice"]], accum_dec
    )
    half_night_display_sum[:, DATA_HOURLY["rain"]] = np.round(
        half_night_display_sum[:, DATA_HOURLY["rain"]], accum_dec
    )
    half_night_display_sum[:, DATA_HOURLY["snow"]] = np.round(
        half_night_display_sum[:, DATA_HOURLY["snow"]], accum_dec
    )
    half_night_display_sum[:, DATA_HOURLY["ice"]] = np.round(
        half_night_display_sum[:, DATA_HOURLY["ice"]], accum_dec
    )

    def _pick_day_icon_and_summary(
        max_arr,
        mean_arr,
        sum_arr,
        precip_type_arr,
        precip_text_arr,
        idx,
        is_night=False,
        mode="hourly",
    ):
        """
        Select an icon and summary text for a day/half-day based on arrays and thresholds.
        Legacy approach encapsulated in a helper function.

        Args:
            max_arr: array used for max/probability checks (indexable by [idx, ...]).
            mean_arr: array used for mean-based checks (indexable by [idx, ...]).
            sum_arr: array used for sum/accumulation checks (indexable by [idx, ...]).
            precip_type_arr: array mapping most-likely precip type per period.
            precip_text_arr: array mapping summary text for precip types per period.
            idx: integer index for the current period.
            is_night: if True, use night-specific icons for partly-cloudy/clear.
            mode: "hourly" (default) uses hourly accumulation thresholds and mean-based checks;
                  "daily" uses daily accumulation thresholds and sum-based checks.

        Returns:
            (icon:str, text:str)
        """

        # Precipitation check (probability + accumulation threshold). Use different thresholds for hourly vs daily.
        if mode == "hourly":
            prob = max_arr[idx, DATA_HOURLY["prob"]]
            rain = mean_arr[idx, DATA_HOURLY["rain"]]
            ice = mean_arr[idx, DATA_HOURLY["ice"]]
            snow = mean_arr[idx, DATA_HOURLY["snow"]]
            accum_thresh = HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit
            precip_type = precip_type_arr[idx]
            precip_text = precip_text_arr[idx]
        else:
            prob = max_arr[idx, DATA_DAY["prob"]]
            rain = sum_arr[idx, DATA_DAY["rain"]]
            ice = sum_arr[idx, DATA_DAY["ice"]]
            snow = sum_arr[idx, DATA_DAY["snow"]]
            accum_thresh = DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit
            # daily snow uses a larger separate threshold
            snow_thresh = DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit
            precip_type = precip_type_arr[idx]
            precip_text = precip_text_arr[idx]

        if prob >= PRECIP_PROB_THRESHOLD and (
            (mode == "hourly" and ((rain + ice) > accum_thresh or snow > accum_thresh))
            or (mode == "daily" and ((rain + ice) > accum_thresh or snow > snow_thresh))
        ):
            return precip_type, precip_text

        # Fog check
        vis_val = (
            mean_arr[idx, DATA_HOURLY["vis"]]
            if mode == "hourly"
            else mean_arr[idx, DATA_DAY["vis"]]
        )
        if vis_val < (FOG_THRESHOLD_METERS * visUnits):
            return "fog", "Fog"

        # Wind check
        wind_val = (
            mean_arr[idx, DATA_HOURLY["wind"]]
            if mode == "hourly"
            else mean_arr[idx, DATA_DAY["wind"]]
        )
        if wind_val > (WIND_THRESHOLDS["light"] * windUnit):
            return "wind", "Windy"

        # Cloud checks
        cloud_val = (
            mean_arr[idx, DATA_HOURLY["cloud"]]
            if mode == "hourly"
            else mean_arr[idx, DATA_DAY["cloud"]]
        )
        if cloud_val > CLOUD_COVER_THRESHOLDS["cloudy"]:
            return "cloudy", "Cloudy"
        if cloud_val > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
            return (
                ("partly-cloudy-night", "Partly Cloudy")
                if is_night
                else ("partly-cloudy-day", "Partly Cloudy")
            )

        # Clear fallback
        return ("clear-night", "Clear") if is_night else ("clear-day", "Clear")

    for idx in range(0, daily_days):

        def _build_half_day_item(
            idx,
            time_val,
            icon,
            text,
            precip_type_val,
            temp_val,
            apparent_val,
            display_mean,
            display_max,
            display_sum,
            interp_mean,
        ):
            """
            Build the half-day forecast item dict using provided pre-converted arrays.

            Args:
                idx: index for period arrays
                time_val: integer timestamp for the period
                icon: selected icon string
                text: selected summary string
                precip_type_val: precipitation type for this period
                temp_val: temperature (high for day, low for night)
                apparent_val: apparent temperature (high for day, low for night)
                display_mean: mean values display array (half_day or half_night)
                display_max: max values display array
                display_sum: sum values display array
                interp_mean: interpolated mean array (for bearing, cape)

            Returns:
                dict: the half-day item matching the original structure
            """
            liquid_accum = display_sum[idx, DATA_HOURLY["rain"]]
            snow_accum = display_sum[idx, DATA_HOURLY["snow"]]
            ice_accum = display_sum[idx, DATA_HOURLY["ice"]]
            precip_accum = liquid_accum + snow_accum + ice_accum

            item = {
                "time": int(time_val),
                "summary": text,
                "icon": icon,
                "precipIntensity": display_mean[idx, DATA_HOURLY["intensity"]],
                "precipIntensityMax": display_max[idx, DATA_HOURLY["intensity"]],
                "rainIntensity": display_mean[idx, DATA_HOURLY["rain"]],
                "rainIntensityMax": display_max[idx, DATA_HOURLY["rain"]],
                "snowIntensity": display_mean[idx, DATA_HOURLY["snow"]],
                "snowIntensityMax": display_max[idx, DATA_HOURLY["snow"]],
                "iceIntensity": display_mean[idx, DATA_HOURLY["ice"]],
                "iceIntensityMax": display_max[idx, DATA_HOURLY["ice"]],
                "precipProbability": display_max[idx, DATA_HOURLY["prob"]],
                "precipAccumulation": precip_accum,
                "precipType": precip_type_val,
                "temperature": temp_val,
                "apparentTemperature": apparent_val,
                "dewPoint": display_mean[idx, DATA_HOURLY["dew"]],
                "humidity": display_mean[idx, DATA_HOURLY["humidity"]],
                "pressure": display_mean[idx, DATA_HOURLY["pressure"]],
                "windSpeed": display_mean[idx, DATA_HOURLY["wind"]],
                "windGust": display_mean[idx, DATA_HOURLY["gust"]],
                "windBearing": int(interp_mean[idx, DATA_HOURLY["bearing"]]),
                "cloudCover": display_mean[idx, DATA_HOURLY["cloud"]],
                "uvIndex": display_mean[idx, DATA_HOURLY["uv"]],
                "visibility": display_mean[idx, DATA_HOURLY["vis"]],
                "ozone": display_mean[idx, DATA_HOURLY["ozone"]],
                "smoke": display_mean[idx, DATA_HOURLY["smoke"]],
                "liquidAccumulation": liquid_accum,
                "snowAccumulation": snow_accum,
                "iceAccumulation": ice_accum,
                "fireIndex": display_mean[idx, DATA_HOURLY["fire"]],
                "solar": display_mean[idx, DATA_HOURLY["solar"]],
                "cape": int(interp_mean[idx, DATA_HOURLY["cape"]]),
            }

            if "stationPressure" in extraVars:
                item["stationPressure"] = display_mean[
                    idx, DATA_HOURLY["station_pressure"]
                ]

            return item

        # Day
        # Set text (select icon and summary)
        day_icon, day_text = _pick_day_icon_and_summary(
            interp_half_day_max,
            interp_half_day_mean,
            interp_half_day_sum,
            precip_type_half_day,
            precip_text_half_day,
            idx,
            is_night=is_all_night,
            mode="hourly",
        )

        day_item = _build_half_day_item(
            idx,
            day_array_4am_grib[idx],
            day_icon,
            day_text,
            precip_type_half_day[idx],
            daily_display_high[idx, DATA_DAY["temp"]],
            daily_display_high[idx, DATA_DAY["apparent"]],
            half_day_display_mean,
            half_day_display_max,
            half_day_display_sum,
            interp_half_day_mean,
        )

        try:
            if idx < 8:
                # Translate the text
                if summaryText:
                    # Calculate the day summary from 4am to 4pm (13 hours)
                    dayIcon, dayText = calculate_half_day_text(
                        hourList_si[(idx * 24) + 4 : (idx * 24) + 17],
                        not is_all_night,
                        str(tz_name),
                        icon_set=icon,
                        unit_system=unitSystem,
                    )
                    day_item["summary"] = translation.translate(["sentence", dayText])
                    day_item["icon"] = dayIcon
        except Exception:
            logger.exception("DAY HALF DAY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            day_item.pop("liquidAccumulation", None)
            day_item.pop("snowAccumulation", None)
            day_item.pop("iceAccumulation", None)
            day_item.pop("fireIndex", None)
            day_item.pop("feelsLike", None)
            day_item.pop("solar", None)

        if timeMachine and not tmExtra:
            day_item.pop("uvIndex", None)
            day_item.pop("ozone", None)

        day_night_list.append(day_item)

        # Night
        # Set text (select icon and summary)
        day_icon, day_text = _pick_day_icon_and_summary(
            interp_half_night_max,
            interp_half_night_mean,
            interp_half_night_sum,
            precip_type_half_night,
            precip_text_half_night,
            idx,
            is_night=not is_all_day,
            mode="hourly",
        )

        day_item = _build_half_day_item(
            idx,
            day_array_5pm_grib[idx],
            day_icon,
            day_text,
            precip_type_half_night[idx],
            daily_display_low[idx, DATA_DAY["temp"]],
            daily_display_low[idx, DATA_DAY["apparent"]],
            half_night_display_mean,
            half_night_display_max,
            half_night_display_sum,
            interp_half_night_mean,
        )

        try:
            if idx < 8:
                # Calculate the night summary from 5pm to 4am (11 hours)

                # Translate the text
                if summaryText:
                    dayIcon, dayText = calculate_half_day_text(
                        hourList_si[(idx * 24) + 17 : ((idx + 1) * 24) + 4],
                        is_all_day,
                        str(tz_name),
                        icon_set=icon,
                        unit_system=unitSystem,
                    )

                    day_item["summary"] = translation.translate(["sentence", dayText])
                    day_item["icon"] = dayIcon
        except Exception:
            logger.exception("NIGHT HALF DAY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            day_item.pop("liquidAccumulation", None)
            day_item.pop("snowAccumulation", None)
            day_item.pop("iceAccumulation", None)
            day_item.pop("fireIndex", None)
            day_item.pop("feelsLike", None)
            day_item.pop("solar", None)

        if timeMachine and not tmExtra:
            day_item.pop("uvIndex", None)
            day_item.pop("ozone", None)

        day_night_list.append(day_item)

        # Select icon and summary for the full-day object
        dayIcon, dayText = _pick_day_icon_and_summary(
            InterPdayMax4am,
            InterPday4am,
            InterPdaySum4am,
            PTypeDay,
            PTextDay,
            idx,
            is_night=is_all_night,
            mode="daily",
        )

        # Fallback if no ptype for some reason. This should only apply when precipitation selection returned 'none'
        if dayIcon == "none":
            if tempUnits == 0:
                tempThresh = TEMPERATURE_UNITS_THRESH["f"]
            else:
                tempThresh = TEMPERATURE_UNITS_THRESH["c"]

            if InterPday[idx, DATA_DAY["temp"]] > tempThresh:
                dayIcon = "rain"
                dayText = "Rain"
            else:
                dayIcon = "snow"
                dayText = "Snow"

        # Temperature High is daytime high, so 6 am to 6 pm
        # First index is 6 am, then index 2
        # Nightime is index 1, 3, etc.

        # Use pre-converted and rounded temperature values
        temp_high = daily_display_high[idx, DATA_DAY["temp"]]
        temp_low = daily_display_low[idx, DATA_DAY["temp"]]
        temp_min = daily_display_min[idx, DATA_DAY["temp"]]
        temp_max = daily_display_max[idx, DATA_DAY["temp"]]
        apparent_high = daily_display_high[idx, DATA_DAY["apparent"]]
        apparent_low = daily_display_low[idx, DATA_DAY["apparent"]]
        apparent_min = daily_display_min[idx, DATA_DAY["apparent"]]
        apparent_max = daily_display_max[idx, DATA_DAY["apparent"]]
        dew_point = daily_display_mean[idx, DATA_DAY["dew"]]
        pressure_hpa = daily_display_mean[idx, DATA_DAY["pressure"]]

        dayObject = {
            "time": int(day_array_grib[idx]),
            "summary": dayText,
            "icon": dayIcon,
            "dawnTime": int(InterSday[idx, DATA_DAY["dawn"]]),
            "sunriseTime": int(InterSday[idx, DATA_DAY["sunrise"]]),
            "sunsetTime": int(InterSday[idx, DATA_DAY["sunset"]]),
            "duskTime": int(InterSday[idx, DATA_DAY["dusk"]]),
            "moonPhase": InterSday[idx, DATA_DAY["moon_phase"]],
            "precipIntensity": daily_display_mean[idx, DATA_DAY["intensity"]],
            "precipIntensityMax": daily_display_max[idx, DATA_DAY["intensity"]],
            "precipIntensityMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["intensity"]]),
            "precipProbability": daily_display_max[idx, DATA_DAY["prob"]],
            "precipAccumulation": (
                daily_display_sum[idx, DATA_DAY["rain"]]
                + daily_display_sum[idx, DATA_DAY["snow"]]
                + daily_display_sum[idx, DATA_DAY["ice"]]
            ),
            "precipType": PTypeDay[idx],
            "rainIntensity": daily_display_mean[idx, DATA_DAY["rain_intensity"]],
            "rainIntensityMax": daily_display_max[idx, DATA_DAY["rain_intensity"]],
            "snowIntensity": daily_display_mean[idx, DATA_DAY["snow_intensity"]],
            "snowIntensityMax": daily_display_max[idx, DATA_DAY["snow_intensity"]],
            "iceIntensity": daily_display_mean[idx, DATA_DAY["ice_intensity"]],
            "iceIntensityMax": daily_display_max[idx, DATA_DAY["ice_intensity"]],
            "temperatureHigh": temp_high,
            "temperatureHighTime": int(InterPdayHighTime[idx, DATA_DAY["temp"]]),
            "temperatureLow": temp_low,
            "temperatureLowTime": int(InterPdayLowTime[idx, DATA_DAY["temp"]]),
            "apparentTemperatureHigh": apparent_high,
            "apparentTemperatureHighTime": int(
                InterPdayHighTime[idx, DATA_DAY["apparent"]]
            ),
            "apparentTemperatureLow": apparent_low,
            "apparentTemperatureLowTime": int(
                InterPdayLowTime[idx, DATA_DAY["apparent"]]
            ),
            "dewPoint": dew_point,
            "humidity": daily_display_mean[idx, DATA_DAY["humidity"]],
            "pressure": pressure_hpa,
            "windSpeed": daily_display_mean[idx, DATA_DAY["wind"]],
            "windGust": daily_display_mean[idx, DATA_DAY["gust"]],
            "windGustTime": int(InterPdayMaxTime[idx, DATA_DAY["gust"]]),
            "windBearing": int(InterPday[idx, DATA_DAY["bearing"]]),
            "cloudCover": daily_display_mean[idx, DATA_DAY["cloud"]],
            "uvIndex": daily_display_max[idx, DATA_DAY["uv"]],
            "uvIndexTime": int(InterPdayMaxTime[idx, DATA_DAY["uv"]]),
            "visibility": daily_display_mean[idx, DATA_DAY["vis"]],
            "temperatureMin": temp_min,
            "temperatureMinTime": int(InterPdayMinTime[idx, DATA_DAY["temp"]]),
            "temperatureMax": temp_max,
            "temperatureMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["temp"]]),
            "apparentTemperatureMin": apparent_min,
            "apparentTemperatureMinTime": int(
                InterPdayMinTime[idx, DATA_DAY["apparent"]]
            ),
            "apparentTemperatureMax": apparent_max,
            "apparentTemperatureMaxTime": int(
                InterPdayMaxTime[idx, DATA_DAY["apparent"]]
            ),
            "smokeMax": daily_display_max[idx, DATA_DAY["smoke"]],
            "smokeMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["smoke"]])
            if not np.isnan(InterPdayMax[idx, DATA_DAY["smoke"]])
            else MISSING_DATA,
            "liquidAccumulation": daily_display_sum[idx, DATA_DAY["rain"]],
            "snowAccumulation": daily_display_sum[idx, DATA_DAY["snow"]],
            "iceAccumulation": daily_display_sum[idx, DATA_DAY["ice"]],
            "fireIndexMax": daily_display_max[idx, DATA_DAY["fire"]],
            "fireIndexMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["fire"]])
            if not np.isnan(InterPdayMax[idx, DATA_DAY["fire"]])
            else MISSING_DATA,
            "solarMax": daily_display_max[idx, DATA_DAY["solar"]],
            "solarMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["solar"]]),
            "capeMax": InterPdayMax[idx, DATA_DAY["cape"]],
            "capeMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["cape"]]),
        }

        # Add station pressure if requested
        if "stationPressure" in extraVars:
            dayObject["stationPressure"] = daily_display_mean[
                idx, DATA_DAY["station_pressure"]
            ]

        try:
            if idx < 8:
                # Calculate the day summary from 4 to 4

                # Translate the text
                if summaryText:
                    dayIcon, dayText = calculate_day_text(
                        hourList_si[((idx) * 24) + 4 : ((idx + 1) * 24) + 4],
                        not is_all_night,
                        str(tz_name),
                        "day",
                        icon,
                        unitSystem,
                    )

                    dayObject["summary"] = translation.translate(["sentence", dayText])
                    dayObject["icon"] = dayIcon
        except Exception:
            logger.exception("DAILY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            dayObject.pop("dawnTime", None)
            dayObject.pop("duskTime", None)
            dayObject.pop("smokeMax", None)
            dayObject.pop("smokeMaxTime", None)
            dayObject.pop("liquidAccumulation", None)
            dayObject.pop("snowAccumulation", None)
            dayObject.pop("iceAccumulation", None)
            dayObject.pop("fireIndexMax", None)
            dayObject.pop("fireIndexMaxTime", None)
            dayObject.pop("solarMax", None)
            dayObject.pop("solarMaxTime", None)
            dayObject.pop("capeMax", None)
            dayObject.pop("capeMaxTime", None)
            dayObject.pop("rainIntensity", None)
            dayObject.pop("snowIntensity", None)
            dayObject.pop("iceIntensity", None)
            dayObject.pop("liquidIntensityMax", None)
            dayObject.pop("snowIntensityMax", None)
            dayObject.pop("iceIntensityMax", None)

        if timeMachine and not tmExtra:
            dayObject.pop("precipProbability", None)
            dayObject.pop("humidity", None)
            dayObject.pop("uvIndex", None)
            dayObject.pop("uvIndexTime", None)
            dayObject.pop("visibility", None)

        dayList.append(dayObject)

        # Create a SI version of dayObject for text generation (values already in SI units in InterPday)
        dayObject_si = {
            "time": int(day_array_grib[idx]),
            "icon": dayIcon,
            "precipType": PTypeDay[idx],
            "precipProbability": InterPdayMax[idx, DATA_DAY["prob"]],
            "precipIntensity": InterPday[idx, DATA_DAY["intensity"]],
            "snowAccumulation": InterPdaySum[idx, DATA_DAY["snow"]],
            "iceAccumulation": InterPdaySum[idx, DATA_DAY["ice"]],
            "liquidAccumulation": InterPdaySum[idx, DATA_DAY["rain"]],
            "rainIntensityMax": InterPdayMax[idx, DATA_DAY["rain_intensity"]],
            "snowIntensityMax": InterPdayMax[idx, DATA_DAY["snow_intensity"]],
            "iceIntensityMax": InterPdayMax[idx, DATA_DAY["ice_intensity"]],
            "temperatureHigh": InterPdayHigh[idx, DATA_DAY["temp"]],
            "temperatureLow": InterPdayLow[idx, DATA_DAY["temp"]],
            "apparentTemperatureHigh": InterPdayHigh[idx, DATA_DAY["apparent"]],
            "apparentTemperatureLow": InterPdayLow[idx, DATA_DAY["apparent"]],
            "dewPoint": InterPday[idx, DATA_DAY["dew"]],
            "humidity": InterPday[idx, DATA_DAY["humidity"]],
            "windSpeed": InterPday[idx, DATA_DAY["wind"]],
            "cloudCover": InterPday[idx, DATA_DAY["cloud"]],
            "visibility": InterPday[idx, DATA_DAY["vis"]],
        }
        dayList_si.append(dayObject_si)

        dayTextList.append(dayObject["summary"])
        dayIconList.append(dayIcon)

    # Timing Check
    if TIMING:
        print("Alert Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    alertDict = []
    alertList = []
    now_utc = datetime.datetime.now(datetime.UTC).astimezone(utc)

    # If alerts are requested and in the US
    try:
        if (
            (not timeMachine)
            and (exAlerts == 0)
            and (az_Lon > -127)
            and (az_Lon < -65)
            and (lat > 24)
            and (lat < 50)
        ):
            # Read in NetCDF
            # Find NetCDF Point based on alerts grid
            alerts_lons = np.arange(-127, -65, 0.025)
            alerts_lats = np.arange(24, 50, 0.025)

            abslat = np.abs(alerts_lats - lat)
            abslon = np.abs(alerts_lons - az_Lon)
            alerts_y_p = np.argmin(abslat)
            alerts_x_p = np.argmin(abslon)

            alertDat = NWS_Alerts_Zarr[alerts_y_p, alerts_x_p]

            if alertDat != "":
                # Match if any alerts
                alerts = str(alertDat).split("|")
                # Loop through each alert
                for alert in alerts:
                    # Extract alert details
                    alertDetails = alert.split("}{")

                    alertOnset = datetime.datetime.strptime(
                        alertDetails[3], "%Y-%m-%dT%H:%M:%S%z"
                    ).astimezone(utc)
                    alertEnd = datetime.datetime.strptime(
                        alertDetails[4], "%Y-%m-%dT%H:%M:%S%z"
                    ).astimezone(utc)

                    # Format description newlines
                    alertDescript = alertDetails[1]
                    # Step 1: Replace double newlines with a single newline
                    formatted_text = re.sub(r"(?<!\n)\n(?!\n)", " ", alertDescript)

                    # Step 2: Replace remaining single newlines with a space
                    formatted_text = re.sub(r"\n\n", "\n", formatted_text)

                    alertDict = {
                        "title": alertDetails[0],
                        "regions": [s.lstrip() for s in alertDetails[2].split(";")],
                        "severity": alertDetails[5],
                        "time": int(
                            (
                                alertOnset
                                - datetime.datetime(1970, 1, 1, 0, 0, 0).astimezone(utc)
                            ).total_seconds()
                        ),
                        "expires": int(
                            (
                                alertEnd
                                - datetime.datetime(1970, 1, 1, 0, 0, 0).astimezone(utc)
                            ).total_seconds()
                        ),
                        "description": formatted_text,
                        "uri": alertDetails[6],
                    }
                    # Only append if alert has not already expired
                    if alertEnd is None or alertEnd > now_utc:
                        alertList.append(dict(alertDict))
                    else:
                        logger.debug("Skipping expired NWS alert: %s", alertDetails[0])

    except Exception:
        logger.exception("An Alert error occurred %s", loc_tag)

    # Process WMO alerts for non-US locations
    try:
        if (
            (not timeMachine)
            and (exAlerts == 0)
            and readWMOAlerts
            and WMO_alertDat is not None
            and WMO_alertDat != ""
        ):
            # WMO alerts use the same grid as was calculated earlier:
            # wmo_alerts_lats = np.arange(-60, 85, 0.0625)
            # wmo_alerts_lons = np.arange(-180, 180, 0.0625)
            # WMO_alertDat was already read at line 2125

            # Match if any alerts
            wmo_alerts = str(WMO_alertDat).split("~")
            # Loop through each alert
            for wmo_alert in wmo_alerts:
                # Extract alert details
                # Format: event}{description}{area_desc}{effective}{expires}{severity}{URL
                wmo_alertDetails = wmo_alert.split("}{")
                # Ensure there are enough parts to parse basic info, preventing IndexError.
                if len(wmo_alertDetails) < 3:
                    continue

                alertEnd = None
                expires_ts = -999
                alertOnset = None
                onset_ts = -999
                alert_severity = "Unknown"
                alert_uri = ""

                # Parse times - WMO times are in ISO format
                if len(wmo_alertDetails) > 3 and wmo_alertDetails[3].strip():
                    alertOnset = datetime.datetime.strptime(
                        wmo_alertDetails[3], "%Y-%m-%dT%H:%M:%S%z"
                    ).astimezone(utc)
                    onset_ts = int(alertOnset.timestamp())

                if len(wmo_alertDetails) > 4 and wmo_alertDetails[4].strip():
                    alertEnd = datetime.datetime.strptime(
                        wmo_alertDetails[4], "%Y-%m-%dT%H:%M:%S%z"
                    ).astimezone(utc)
                    expires_ts = int(alertEnd.timestamp())

                if len(wmo_alertDetails) > 5:
                    alert_severity = wmo_alertDetails[5]

                if len(wmo_alertDetails) > 6:
                    alert_uri = wmo_alertDetails[6]

                wmo_alertDict = {
                    "title": wmo_alertDetails[0],
                    "regions": [
                        s.lstrip() for s in wmo_alertDetails[2].split(";") if s.strip()
                    ],
                    "severity": alert_severity,
                    "time": onset_ts,
                    "expires": expires_ts,
                    "description": wmo_alertDetails[1],
                    "uri": alert_uri,
                }

                # Only append if alert has not already expired
                if alertEnd is None or alertEnd > now_utc:
                    alertList.append(dict(wmo_alertDict))
                else:
                    logger.debug("Skipping expired WMO alert: %s", alertDetails[0])

    except Exception:
        logger.exception("A WMO Alert error occurred %s", loc_tag)

    # Timing Check
    if TIMING:
        print("Current Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Currently data, find points for linear averaging
    # If within 2 minutes of a hour, do not using rounding
    if np.min(np.abs(hour_array_grib - minute_array_grib[0])) < 120:
        currentIDX_hrrrh = np.argmin(np.abs(hour_array_grib - minute_array_grib[0]))
        interpFac1 = 0
        interpFac2 = 1
    else:
        currentIDX_hrrrh = np.searchsorted(
            hour_array_grib, minute_array_grib[0], side="left"
        )

        # Find weighting factors for hourly data
        # Weighting factors for linear interpolation
        interpFac1 = 1 - (
            abs(minute_array_grib[0] - hour_array_grib[currentIDX_hrrrh - 1])
            / (
                hour_array_grib[currentIDX_hrrrh]
                - hour_array_grib[currentIDX_hrrrh - 1]
            )
        )

        interpFac2 = 1 - (
            abs(minute_array_grib[0] - hour_array_grib[currentIDX_hrrrh])
            / (
                hour_array_grib[currentIDX_hrrrh]
                - hour_array_grib[currentIDX_hrrrh - 1]
            )
        )

    currentIDX_hrrrh_A = np.max((currentIDX_hrrrh - 1, 0))

    InterPcurrent = np.zeros(shape=max(DATA_CURRENT.values()) + 1)
    InterPcurrent[DATA_CURRENT["time"]] = int(minute_array_grib[0])

    # Temperature from RTMA_RU (highest priority), then subH, then NBM, then ECMWF, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = dataOut_rtma_ru[0, RTMA_RU["temp"]]
    elif "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = hrrrSubHInterpolation[
            0, HRRR_SUBH["temp"]
        ]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["temp"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["temp"]] * interpFac2
        )
    elif "ecmwf_ifs" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["temp"]] * interpFac1
            + ECMWF_Merged[currentIDX_hrrrh, ECMWF["temp"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["temp"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["temp"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["2m_temperature"]] * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["2m_temperature"]] * interpFac2
        )

    # Clip between -90 and 60
    InterPcurrent[DATA_CURRENT["temp"]] = clipLog(
        InterPcurrent[DATA_CURRENT["temp"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Temperature Current",
    )

    # Dewpoint from RTMA_RU (highest priority), then subH, then NBM, then ECMWF, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = dataOut_rtma_ru[0, RTMA_RU["dew"]]
    elif "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = hrrrSubHInterpolation[0, HRRR_SUBH["dew"]]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["dew"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["dew"]] * interpFac2
        )
    elif "ecmwf_ifs" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = (
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["dew"]] * interpFac1
            + ECMWF_Merged[currentIDX_hrrrh, ECMWF["dew"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["dew"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["dew"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["2m_dewpoint_temperature"]]
            * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["2m_dewpoint_temperature"]]
            * interpFac2
        )

    # Clip between -90 and 60
    InterPcurrent[DATA_CURRENT["dew"]] = clipLog(
        InterPcurrent[DATA_CURRENT["dew"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Dewpoint Current",
    )

    # humidity, RTMA_RU then NBM then HRRR, then GFS
    # Note: RTMA_RU humidity is already a fraction so no need to convert
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["humidity"]] = dataOut_rtma_ru[
            0, RTMA_RU["humidity"]
        ]
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["humidity"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["humidity"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["humidity"]] * interpFac2
        ) * humidUnit
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["humidity"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["humidity"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["humidity"]] * interpFac2
        ) * humidUnit
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["humidity"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["humidity"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["humidity"]] * interpFac2
        ) * humidUnit
    elif "ecmwf_ifs" in sourceList:
        # ECMWF humidity needs to be calculated from dewpoint and temperature
        ECMWF_humidFac1 = relative_humidity_from_dewpoint(
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["temp"]] * mp.units.units.degK,
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["dew"]] * mp.units.units.degK,
            phase="auto",
        ).magnitude
        ECMWF_humidFac2 = relative_humidity_from_dewpoint(
            ECMWF_Merged[currentIDX_hrrrh, ECMWF["temp"]] * mp.units.units.degK,
            ECMWF_Merged[currentIDX_hrrrh, ECMWF["dew"]] * mp.units.units.degK,
            phase="auto",
        ).magnitude

        InterPcurrent[DATA_CURRENT["humidity"]] = (
            (ECMWF_humidFac1 * interpFac1 + ECMWF_humidFac2 * interpFac2)
            * 100
            * humidUnit
        )
    elif "era5" in sourceList:
        ERA5_humidFac1 = relative_humidity_from_dewpoint(
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["2m_temperature"]]
            * mp.units.units.degK,
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["2m_dewpoint_temperature"]]
            * mp.units.units.degK,
            phase="auto",
        ).magnitude
        ERA5_humidFac2 = relative_humidity_from_dewpoint(
            ERA5_MERGED[currentIDX_hrrrh, ERA5["2m_temperature"]] * mp.units.units.degK,
            ERA5_MERGED[currentIDX_hrrrh, ERA5["2m_dewpoint_temperature"]]
            * mp.units.units.degK,
            phase="auto",
        ).magnitude

        InterPcurrent[DATA_CURRENT["humidity"]] = (
            (ERA5_humidFac1 * interpFac1 + ERA5_humidFac2 * interpFac2)
            * 100
            * humidUnit
        )

    # Clip between 0 and 1
    InterPcurrent[DATA_CURRENT["humidity"]] = clipLog(
        InterPcurrent[DATA_CURRENT["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
        "Humidity Current",
    )

    # Pressure from HRRR, then ECMWF, then GFS (RTMA_RU has surface pressure, not mean sea level pressure)
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["pressure"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["pressure"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["pressure"]] * interpFac2
        )
    elif "ecmwf_ifs" in sourceList:
        InterPcurrent[DATA_CURRENT["pressure"]] = (
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["pressure"]] * interpFac1
            + ECMWF_Merged[currentIDX_hrrrh, ECMWF["pressure"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["pressure"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["pressure"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["pressure"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["pressure"]] = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["mean_sea_level_pressure"]]
            * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["mean_sea_level_pressure"]]
            * interpFac2
        )

    # Clip between 800 and 1100 hPa (80000-110000 Pascals)
    InterPcurrent[DATA_CURRENT["pressure"]] = clipLog(
        InterPcurrent[DATA_CURRENT["pressure"]],
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Pressure Current",
    )

    # WindSpeed from RTMA_RU, then subH, then NBM, then ECMWF, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            dataOut_rtma_ru[0, RTMA_RU["wind_u"]] ** 2
            + dataOut_rtma_ru[0, RTMA_RU["wind_v"]] ** 2
        )
    elif "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            hrrrSubHInterpolation[0, HRRR_SUBH["wind_u"]] ** 2
            + hrrrSubHInterpolation[0, HRRR_SUBH["wind_v"]] ** 2
        )
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["wind"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["wind"]] * interpFac2
        )
    elif "ecmwf_ifs" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            (
                ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["wind_u"]] * interpFac1
                + ECMWF_Merged[currentIDX_hrrrh, ECMWF["wind_u"]] * interpFac2
            )
            ** 2
            + (
                ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["wind_v"]] * interpFac1
                + ECMWF_Merged[currentIDX_hrrrh, ECMWF["wind_v"]] * interpFac2
            )
            ** 2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["wind_u"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["wind_u"]] * interpFac2
            )
            ** 2
            + (
                GFS_Merged[currentIDX_hrrrh_A, GFS["wind_v"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["wind_v"]] * interpFac2
            )
            ** 2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            (
                ERA5_MERGED[currentIDX_hrrrh_A, ERA5["10m_u_component_of_wind"]]
                * interpFac1
                + ERA5_MERGED[currentIDX_hrrrh, ERA5["10m_u_component_of_wind"]]
                * interpFac2
            )
            ** 2
            + (
                ERA5_MERGED[currentIDX_hrrrh_A, ERA5["10m_v_component_of_wind"]]
                * interpFac1
                + ERA5_MERGED[currentIDX_hrrrh, ERA5["10m_v_component_of_wind"]]
                * interpFac2
            )
            ** 2
        )

    # Keep wind speed in m/s (SI units)
    InterPcurrent[DATA_CURRENT["wind"]] = clipLog(
        InterPcurrent[DATA_CURRENT["wind"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "WindSpeed Current",
    )

    # Gust from RTMA_RU, then subH, then NBM, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = dataOut_rtma_ru[0, RTMA_RU["gust"]]
    elif "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = hrrrSubHInterpolation[
            0, HRRR_SUBH["gust"]
        ]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["gust"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["gust"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["gust"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["gust"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["instantaneous_10m_wind_gust"]]
            * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["instantaneous_10m_wind_gust"]]
            * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["gust"]] = MISSING_DATA

    # Clip between 0 and 400, keep in m/s (SI units)
    InterPcurrent[DATA_CURRENT["gust"]] = clipLog(
        InterPcurrent[DATA_CURRENT["gust"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "Gust Current",
    )

    # Get prep probability, intensity and error from minutely
    if "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["intensity"]] = (
            (
                ERA5_MERGED[currentIDX_hrrrh_A, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh_A, ERA5["convective_rain_rate"]]
                + ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["large_scale_snowfall_rate_water_equivalent"],
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["convective_snowfall_rate_water_equivalent"],
                ]
            )
            * interpFac1
            + (
                ERA5_MERGED[currentIDX_hrrrh, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh, ERA5["convective_rain_rate"]]
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["large_scale_snowfall_rate_water_equivalent"]
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["convective_snowfall_rate_water_equivalent"]
                ]
            )
            * interpFac2
        ) * 3600  # Convert from mm/s to mm/hr

        # Calculate separate rain and snow intensities for ERA5
        # Rain intensity (mm/h)
        InterPcurrent[DATA_CURRENT["rain_intensity"]] = (
            (
                ERA5_MERGED[currentIDX_hrrrh_A, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh_A, ERA5["convective_rain_rate"]]
            )
            * interpFac1
            + (
                ERA5_MERGED[currentIDX_hrrrh, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh, ERA5["convective_rain_rate"]]
            )
            * interpFac2
        ) * 3600  # Convert from mm/s to mm/hr

        # Snow water equivalent (mm/h)
        era5_current_snow_we = (
            (
                ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["large_scale_snowfall_rate_water_equivalent"],
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["convective_snowfall_rate_water_equivalent"],
                ]
            )
            * interpFac1
            + (
                ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["large_scale_snowfall_rate_water_equivalent"]
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["convective_snowfall_rate_water_equivalent"]
                ]
            )
            * interpFac2
        ) * 3600  # Convert from mm/s to mm/hr

        # Convert snow water equivalent to snow depth (cm/h)
        InterPcurrent[DATA_CURRENT["snow_intensity"]] = estimate_snow_height(
            np.array([era5_current_snow_we]),  # mm/h water equivalent
            np.array([InterPcurrent[DATA_CURRENT["temp"]]])
            - KELVIN_TO_CELSIUS,  # Celsius
            np.array([InterPcurrent[DATA_CURRENT["wind"]]]),  # m/s
        )[0]

        # ERA5 doesn't provide sleet/ice rates
        InterPcurrent[DATA_CURRENT["ice_intensity"]] = 0
    else:
        InterPcurrent[DATA_CURRENT["intensity"]] = InterPminute[
            0, DATA_MINUTELY["intensity"]
        ]
        InterPcurrent[DATA_CURRENT["prob"]] = InterPminute[
            0, DATA_MINUTELY["prob"]
        ]  # "precipProbability"
        InterPcurrent[DATA_CURRENT["error"]] = InterPminute[
            0, DATA_MINUTELY["error"]
        ]  # "precipIntensityError"

    # WindDir from RTMA_RU, then subH, then NBM, then ECMWF, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    dataOut_rtma_ru[0, RTMA_RU["wind_u"]],
                    dataOut_rtma_ru[0, RTMA_RU["wind_v"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )
    elif "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_u"]],
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_v"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = NBM_Merged[
            currentIDX_hrrrh, NBM["bearing"]
        ]
    elif "ecmwf_ifs" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    ECMWF_Merged[currentIDX_hrrrh, ECMWF["wind_u"]],
                    ECMWF_Merged[currentIDX_hrrrh, ECMWF["wind_v"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    GFS_Merged[currentIDX_hrrrh, GFS["wind_u"]],
                    GFS_Merged[currentIDX_hrrrh, GFS["wind_v"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    ERA5_MERGED[currentIDX_hrrrh, ERA5["10m_u_component_of_wind"]],
                    ERA5_MERGED[currentIDX_hrrrh, ERA5["10m_v_component_of_wind"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )
    else:
        InterPcurrent[DATA_CURRENT["bearing"]] = MISSING_DATA

    # Cloud, RTMA_RU, then NBM, then ECMWF, then HRRR, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            (dataOut_rtma_ru[0, RTMA_RU["cloud"]]) * 0.01
        )
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["cloud"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["cloud"]] * interpFac2
        ) * 0.01
    elif "ecmwf_ifs" in sourceList:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["cloud"]] * interpFac1
            + ECMWF_Merged[currentIDX_hrrrh, ECMWF["cloud"]] * interpFac2
        ) * 0.01
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["cloud"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["cloud"]] * interpFac2
        ) * 0.01
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["cloud"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["cloud"]] * interpFac2
        ) * 0.01
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["total_cloud_cover"]] * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["total_cloud_cover"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["cloud"]] = MISSING_DATA

    # Clip
    InterPcurrent[DATA_CURRENT["cloud"]] = clipLog(
        InterPcurrent[DATA_CURRENT["cloud"]],
        CLIP_CLOUD["min"],
        CLIP_CLOUD["max"],
        "Cloud Current",
    )

    # UV Index from GFS
    if "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["uv"]] = clipLog(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["uv"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["uv"]] * interpFac2
            )
            * 18.9
            * 0.025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )
    elif "era5" in sourceList:
        # TODO: Implement a more accurate uv index
        InterPcurrent[DATA_CURRENT["uv"]] = clipLog(
            (
                ERA5_MERGED[
                    currentIDX_hrrrh_A, ERA5["downward_uv_radiation_at_the_surface"]
                ]
                * interpFac1
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["downward_uv_radiation_at_the_surface"]
                ]
                * interpFac2
            )
            / 3600
            * 40
            * 0.0025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )
    else:
        InterPcurrent[DATA_CURRENT["uv"]] = MISSING_DATA

    # Station Pressure from RTMA_RU (surface pressure), then GFS
    station_pressure_value = MISSING_DATA
    if "rtma_ru" in sourceList:
        station_pressure_value = dataOut_rtma_ru[0, RTMA_RU["pressure"]]
    elif "gfs" in sourceList:
        station_pressure_value = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["station_pressure"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["station_pressure"]] * interpFac2
        )
    elif "era5" in sourceList:
        station_pressure_value = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["surface_pressure"]] * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["surface_pressure"]] * interpFac2
        )

    InterPcurrent[DATA_CURRENT["station_pressure"]] = clipLog(
        station_pressure_value,
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Station Pressure Current",
    )

    # VIS, RTMA_RU, then SubH, then NBM then HRRR, then GFS
    if "rtma_ru" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = dataOut_rtma_ru[0, RTMA_RU["vis"]]
        # RTMA_RU has max visibility of 16000m, convert to 16090m for exact 10 miles
        # Use threshold to handle floating point precision
        if InterPcurrent[DATA_CURRENT["vis"]] >= 15999:
            InterPcurrent[DATA_CURRENT["vis"]] = 16090
    elif "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = hrrrSubHInterpolation[0, HRRR_SUBH["vis"]]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["vis"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["vis"]] * interpFac2
        )
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["vis"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["vis"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["vis"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["vis"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["vis"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = estimate_visibility_gultepe_rh_pr_numpy(
            ERA5_MERGED[currentIDX_hrrrh_A, :] * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, :] * interpFac2,
            var_index=ERA5,
            var_axis=1,
        )

    else:
        InterPcurrent[DATA_CURRENT["vis"]] = MISSING_DATA

    # Keep visibility in meters (SI units)
    InterPcurrent[DATA_CURRENT["vis"]] = np.clip(
        InterPcurrent[DATA_CURRENT["vis"]], CLIP_VIS["min"], CLIP_VIS["max"]
    )

    # Ozone from GFS or ERA5
    ozone_value = MISSING_DATA
    if "gfs" in sourceList:
        ozone_value = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["ozone"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["ozone"]] * interpFac2
        )
    elif "era5" in sourceList:
        # Conversion from: https://sacs.aeronomie.be/info/dobson.php
        ozone_value = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["total_column_ozone"]] * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["total_column_ozone"]] * interpFac2
        ) * 46696  # To convert to dobson units

    InterPcurrent[DATA_CURRENT["ozone"]] = clipLog(
        ozone_value,
        CLIP_OZONE["min"],
        CLIP_OZONE["max"],
        "Ozone Current",
    )

    # Storm Distance from GFS
    if "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["storm_dist"]] = np.maximum(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["storm_dist"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["storm_dist"]] * interpFac2
            ),
            0,
        )
        # Storm Bearing from GFS
        InterPcurrent[DATA_CURRENT["storm_dir"]] = GFS_Merged[
            currentIDX_hrrrh, GFS["storm_dir"]
        ]
    else:
        InterPcurrent[DATA_CURRENT["storm_dist"]] = MISSING_DATA
        InterPcurrent[DATA_CURRENT["storm_dir"]] = MISSING_DATA

    # Smoke from HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["smoke"]] = clipLog(
            (
                HRRR_Merged[currentIDX_hrrrh_A, HRRR["smoke"]] * interpFac1
                + HRRR_Merged[currentIDX_hrrrh, HRRR["smoke"]] * interpFac2
            ),
            CLIP_SMOKE["min"],
            CLIP_SMOKE["max"],
            "Smoke Current",
        )
    else:
        InterPcurrent[DATA_CURRENT["smoke"]] = MISSING_DATA

    # Solar from subH, then NBM, then, HRRR, then GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["solar"]] = hrrrSubHInterpolation[
            0, HRRR_SUBH["solar"]
        ]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["solar"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["solar"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["solar"]] * interpFac2
        )
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["solar"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["solar"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["solar"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["solar"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["solar"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["solar"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["solar"]] = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["surface_solar_radiation_downwards"]]
            * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["surface_solar_radiation_downwards"]]
            * interpFac2
        ) / 3600  # Convert from J/m2 to W/m2

    InterPcurrent[DATA_CURRENT["solar"]] = clipLog(
        InterPcurrent[DATA_CURRENT["solar"]],
        CLIP_SOLAR["min"],
        CLIP_SOLAR["max"],
        "Solar Current",
    )

    # CAPE from NBM, then, HRRR, then GFS
    if "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["cape"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["cape"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["cape"]] * interpFac2
        )
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["cape"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["cape"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["cape"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["cape"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["cape"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["cape"]] * interpFac2
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["cape"]] = (
            ERA5_MERGED[
                currentIDX_hrrrh_A, ERA5["convective_available_potential_energy"]
            ]
            * interpFac1
            + ERA5_MERGED[
                currentIDX_hrrrh, ERA5["convective_available_potential_energy"]
            ]
            * interpFac2
        )

    InterPcurrent[DATA_CURRENT["cape"]] = clipLog(
        InterPcurrent[DATA_CURRENT["cape"]],
        CLIP_CAPE["min"],
        CLIP_CAPE["max"],
        "CAPE Current",
    )

    # Calculate the apparent temperature
    InterPcurrent[DATA_CURRENT["apparent"]] = calculate_apparent_temperature(
        InterPcurrent[DATA_CURRENT["temp"]],  # Air temperature in Kelvin
        InterPcurrent[DATA_CURRENT["humidity"]],  # Relative humidity (0.0 to 1.0)
        InterPcurrent[DATA_CURRENT["wind"]],  # Wind speed in meters per second
        InterPcurrent[DATA_CURRENT["solar"]],  # Solar radiation in W/m^2
    )

    if "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["feels_like"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["apparent"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["apparent"]] * interpFac2
        )
    elif "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["feels_like"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["apparent"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["apparent"]] * interpFac2
        )
    elif timeMachine:
        # If timemachine, use the calculated value
        InterPcurrent[DATA_CURRENT["feels_like"]] = InterPcurrent[
            DATA_CURRENT["apparent"]
        ]
    else:
        InterPcurrent[DATA_CURRENT["feels_like"]] = MISSING_DATA

    # Clip
    InterPcurrent[DATA_CURRENT["feels_like"]] = clipLog(
        InterPcurrent[DATA_CURRENT["feels_like"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )

    # Fire index from NBM Fire
    if "nbm_fire" in sourceList:
        InterPcurrent[DATA_CURRENT["fire"]] = clipLog(
            (
                NBM_Fire_Merged[currentIDX_hrrrh_A, NBM_FIRE_INDEX] * interpFac1
                + NBM_Fire_Merged[currentIDX_hrrrh, NBM_FIRE_INDEX] * interpFac2
            ),
            CLIP_FIRE["min"],
            CLIP_FIRE["max"],
            "Fire index Current",
        )

    else:
        InterPcurrent[DATA_CURRENT["fire"]] = MISSING_DATA

    # Save SI unit values for text generation before converting to requested units
    curr_temp_si = InterPcurrent[DATA_CURRENT["temp"]] - KELVIN_TO_CELSIUS
    curr_dew_si = InterPcurrent[DATA_CURRENT["dew"]] - KELVIN_TO_CELSIUS
    curr_wind_si = InterPcurrent[DATA_CURRENT["wind"]]
    curr_vis_si = InterPcurrent[DATA_CURRENT["vis"]]

    # Pre-calculate all unit conversions for currently block (vectorized approach)
    # Temperature conversions
    if tempUnits == 0:
        curr_temp_display = np.round(
            (InterPcurrent[DATA_CURRENT["temp"]] - KELVIN_TO_CELSIUS) * 9 / 5 + 32, 2
        )
        curr_apparent_display = np.round(
            (InterPcurrent[DATA_CURRENT["apparent"]] - KELVIN_TO_CELSIUS) * 9 / 5 + 32,
            2,
        )
        curr_dew_display = np.round(
            (InterPcurrent[DATA_CURRENT["dew"]] - KELVIN_TO_CELSIUS) * 9 / 5 + 32, 2
        )
        curr_feels_like_display = np.round(
            (InterPcurrent[DATA_CURRENT["feels_like"]] - KELVIN_TO_CELSIUS) * 9 / 5
            + 32,
            2,
        )
    else:
        curr_temp_display = np.round(InterPcurrent[DATA_CURRENT["temp"]] - tempUnits, 2)
        curr_apparent_display = np.round(
            InterPcurrent[DATA_CURRENT["apparent"]] - tempUnits, 2
        )
        curr_dew_display = np.round(InterPcurrent[DATA_CURRENT["dew"]] - tempUnits, 2)
        curr_feels_like_display = np.round(
            InterPcurrent[DATA_CURRENT["feels_like"]] - tempUnits, 2
        )

    # Other unit conversions
    curr_storm_dist_display = np.round(
        InterPcurrent[DATA_CURRENT["storm_dist"]] * visUnits, 2
    )
    curr_rain_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["rain_intensity"]] * prepIntensityUnit, 2
    )
    curr_snow_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["snow_intensity"]] * prepIntensityUnit, 2
    )
    curr_ice_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["ice_intensity"]] * prepIntensityUnit, 2
    )
    curr_pressure_display = np.round(InterPcurrent[DATA_CURRENT["pressure"]] / 100, 2)
    curr_wind_display = np.round(InterPcurrent[DATA_CURRENT["wind"]] * windUnit, 2)
    curr_gust_display = np.round(InterPcurrent[DATA_CURRENT["gust"]] * windUnit, 2)
    curr_vis_display = np.round(InterPcurrent[DATA_CURRENT["vis"]] * visUnits, 2)
    curr_station_pressure_display = np.round(
        InterPcurrent[DATA_CURRENT["station_pressure"]] / 100, 2
    )

    # Fields that don't need unit conversion but do need rounding
    curr_humidity_display = np.round(InterPcurrent[DATA_CURRENT["humidity"]], 2)
    curr_cloud_display = np.round(InterPcurrent[DATA_CURRENT["cloud"]], 2)
    curr_uv_display = np.round(InterPcurrent[DATA_CURRENT["uv"]], 2)
    curr_ozone_display = np.round(InterPcurrent[DATA_CURRENT["ozone"]], 2)
    curr_smoke_display = np.round(InterPcurrent[DATA_CURRENT["smoke"]], 2)
    curr_fire_display = np.round(InterPcurrent[DATA_CURRENT["fire"]], 2)
    curr_solar_display = np.round(InterPcurrent[DATA_CURRENT["solar"]], 2)
    curr_bearing_display = int(
        np.round(np.mod(InterPcurrent[DATA_CURRENT["bearing"]], 360), 0)
    )
    curr_cape_display = (
        int(np.round(InterPcurrent[DATA_CURRENT["cape"]], 0))
        if not np.isnan(InterPcurrent[DATA_CURRENT["cape"]])
        else 0
    )

    # Round current day accumulations to 4 decimal places
    dayZeroIce = float(np.round(dayZeroIce * prepAccumUnit, 4))
    dayZeroRain = float(np.round(dayZeroRain * prepAccumUnit, 4))
    dayZeroSnow = float(np.round(dayZeroSnow * prepAccumUnit, 4))

    if (
        (minuteItems[0]["precipIntensity"])
        > (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepIntensityUnit)
    ) & (minuteItems[0]["precipType"] is not None):
        # If more than 25% chance of precip, then the icon for whatever is happening, so long as the icon exists
        cIcon = minuteItems[0]["precipType"]
        cText = (
            minuteItems[0]["precipType"][0].upper() + minuteItems[0]["precipType"][1:]
        )

        # Because soemtimes there's precipitation not no type, don't use an icon in those cases

    # If visibility < FOG_THRESHOLD_METERS and during the day
    elif InterPcurrent[DATA_CURRENT["vis"]] < FOG_THRESHOLD_METERS:
        cIcon = "fog"
        cText = "Fog"
    elif InterPcurrent[DATA_CURRENT["wind"]] > WIND_THRESHOLDS["light"]:
        cIcon = "wind"
        cText = "Windy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
        cIcon = "cloudy"
        cText = "Cloudy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        cText = "Partly Cloudy"

        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            # Before sunrise
            cIcon = "partly-cloudy-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            # After sunrise before sunset
            cIcon = "partly-cloudy-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            # After sunset
            cIcon = "partly-cloudy-night"
    else:
        cText = "Clear"
        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            # Before sunrise
            cIcon = "clear-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            # After sunrise before sunset
            cIcon = "clear-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            # After sunset
            cIcon = "clear-night"

    # Timing Check
    if TIMING:
        print("Object Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Calculate type-specific intensities for currently (in SI units - mm/h)
    # Initialize all to zero
    InterPcurrent[DATA_CURRENT["rain_intensity"]] = 0
    InterPcurrent[DATA_CURRENT["snow_intensity"]] = 0
    InterPcurrent[DATA_CURRENT["ice_intensity"]] = 0

    # Get the current precip intensity from the minuteRainIntensity
    InterPcurrent[DATA_CURRENT["rain_intensity"]] = minuteRainIntensity[0]
    InterPcurrent[DATA_CURRENT["snow_intensity"]] = minuteSnowIntensity[0]
    InterPcurrent[DATA_CURRENT["ice_intensity"]] = minuteSleetIntensity[0]

    # Fix small neg zero
    InterPcurrent[((InterPcurrent > -0.01) & (InterPcurrent < 0.01))] = 0

    ### RETURN ###
    returnOBJ = dict()

    returnOBJ["latitude"] = round(float(lat), 4)
    returnOBJ["longitude"] = round(float(lon_IN), 4)
    returnOBJ["timezone"] = str(tz_name)
    returnOBJ["offset"] = float(tz_offset / 60)
    returnOBJ["elevation"] = int(round(float(ETOPO * elevUnit), 0))

    if exCurrently != 1:
        returnOBJ["currently"] = dict()
        returnOBJ["currently"]["time"] = int(minute_array_grib[0])
        returnOBJ["currently"]["summary"] = cText
        returnOBJ["currently"]["icon"] = cIcon
        returnOBJ["currently"]["nearestStormDistance"] = curr_storm_dist_display
        returnOBJ["currently"]["nearestStormBearing"] = (
            int(InterPcurrent[DATA_CURRENT["storm_dir"]])
            if not np.isnan(InterPcurrent[DATA_CURRENT["storm_dir"]])
            else np.nan
        )
        returnOBJ["currently"]["precipIntensity"] = minuteItems[0]["precipIntensity"]
        returnOBJ["currently"]["precipProbability"] = minuteItems[0][
            "precipProbability"
        ]
        returnOBJ["currently"]["precipIntensityError"] = minuteItems[0][
            "precipIntensityError"
        ]
        returnOBJ["currently"]["precipType"] = minuteItems[0]["precipType"]
        returnOBJ["currently"]["rainIntensity"] = curr_rain_intensity_display
        returnOBJ["currently"]["snowIntensity"] = curr_snow_intensity_display
        returnOBJ["currently"]["iceIntensity"] = curr_ice_intensity_display
        returnOBJ["currently"]["temperature"] = curr_temp_display
        returnOBJ["currently"]["apparentTemperature"] = curr_apparent_display
        returnOBJ["currently"]["dewPoint"] = curr_dew_display
        returnOBJ["currently"]["humidity"] = curr_humidity_display
        returnOBJ["currently"]["pressure"] = curr_pressure_display
        returnOBJ["currently"]["windSpeed"] = curr_wind_display
        returnOBJ["currently"]["windGust"] = curr_gust_display
        returnOBJ["currently"]["windBearing"] = curr_bearing_display
        returnOBJ["currently"]["cloudCover"] = curr_cloud_display
        returnOBJ["currently"]["uvIndex"] = curr_uv_display
        returnOBJ["currently"]["visibility"] = curr_vis_display
        returnOBJ["currently"]["ozone"] = curr_ozone_display
        returnOBJ["currently"]["smoke"] = curr_smoke_display
        returnOBJ["currently"]["fireIndex"] = curr_fire_display
        returnOBJ["currently"]["feelsLike"] = curr_feels_like_display
        returnOBJ["currently"]["currentDayIce"] = dayZeroIce
        returnOBJ["currently"]["currentDayLiquid"] = dayZeroRain
        returnOBJ["currently"]["currentDaySnow"] = dayZeroSnow
        returnOBJ["currently"]["solar"] = curr_solar_display
        returnOBJ["currently"]["cape"] = curr_cape_display

        if "stationPressure" in extraVars:
            returnOBJ["currently"]["stationPressure"] = curr_station_pressure_display

        # Update the text
        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            # Before sunrise
            currentDay = False
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            # After sunrise before sunset
            currentDay = True
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            # After sunset
            currentDay = False

        # Create SI unit version of currently object for text generation
        currently_si = dict(returnOBJ["currently"])
        # Replace converted values with SI values
        currently_si["icon"] = returnOBJ["currently"]["icon"]
        currently_si["precipType"] = returnOBJ["currently"]["precipType"]
        currently_si["windSpeed"] = curr_wind_si
        currently_si["visibility"] = curr_vis_si
        currently_si["temperature"] = curr_temp_si
        currently_si["dewPoint"] = curr_dew_si
        currently_si["cloudCover"] = InterPcurrent[DATA_CURRENT["cloud"]]
        currently_si["humidity"] = InterPcurrent[DATA_CURRENT["humidity"]]
        currently_si["smoke"] = InterPcurrent[DATA_CURRENT["smoke"]]
        currently_si["cape"] = InterPcurrent[DATA_CURRENT["cape"]]
        currently_si["rainIntensity"] = InterPcurrent[DATA_CURRENT["rain_intensity"]]
        currently_si["snowIntensity"] = InterPcurrent[DATA_CURRENT["snow_intensity"]]
        currently_si["iceIntensity"] = InterPcurrent[DATA_CURRENT["ice_intensity"]]
        # No accumulation in current period
        currently_si["liquidAccumulation"] = 0
        currently_si["snowAccumulation"] = 0
        currently_si["iceAccumulation"] = 0

        try:
            if summaryText:
                currentText, currentIcon = calculate_text(
                    currently_si,
                    currentDay,
                    "current",
                    icon,
                )
                returnOBJ["currently"]["summary"] = translation.translate(
                    ["title", currentText]
                )
                returnOBJ["currently"]["icon"] = currentIcon
        except Exception:
            logger.exception("CURRENTLY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            returnOBJ["currently"].pop("smoke", None)
            returnOBJ["currently"].pop("currentDayIce", None)
            returnOBJ["currently"].pop("currentDayLiquid", None)
            returnOBJ["currently"].pop("currentDaySnow", None)
            returnOBJ["currently"].pop("fireIndex", None)
            returnOBJ["currently"].pop("feelsLike", None)
            returnOBJ["currently"].pop("solar", None)
            returnOBJ["currently"].pop("cape", None)
            returnOBJ["currently"].pop("rainIntensity", None)
            returnOBJ["currently"].pop("snowIntensity", None)
            returnOBJ["currently"].pop("iceIntensity", None)

        if timeMachine and not tmExtra:
            returnOBJ["currently"].pop("nearestStormDistance", None)
            returnOBJ["currently"].pop("nearestStormBearing", None)
            returnOBJ["currently"].pop("precipProbability", None)
            returnOBJ["currently"].pop("precipIntensityError", None)
            returnOBJ["currently"].pop("humidity", None)
            returnOBJ["currently"].pop("uvIndex", None)
            returnOBJ["currently"].pop("visibility", None)
            returnOBJ["currently"].pop("ozone", None)

    if exMinutely != 1:
        returnOBJ["minutely"] = dict()
        try:
            if summaryText:
                # Get max CAPE for the next hour to determine if thunderstorms should be shown
                # Use the maximum of current CAPE and first hourly CAPE
                currentCAPE = np.nan_to_num(InterPcurrent[DATA_CURRENT["cape"]], nan=0)
                # Get CAPE from first hourly entry if available
                hourlyCAPE = (
                    np.nan_to_num(InterPhour[0, DATA_HOURLY["cape"]], nan=0)
                    if len(InterPhour) > 0
                    else 0
                )
                maxCAPE = max(currentCAPE, hourlyCAPE)

                minuteText, minuteIcon = calculate_minutely_text(
                    minuteItems_si,
                    currentText,
                    currentIcon,
                    icon,
                    maxCAPE,
                )
                returnOBJ["minutely"]["summary"] = translation.translate(
                    ["sentence", minuteText]
                )
                returnOBJ["minutely"]["icon"] = minuteIcon
            else:
                returnOBJ["minutely"]["summary"] = pTypesText[
                    int(Counter(maxPchance).most_common(1)[0][0])
                ]
                returnOBJ["minutely"]["icon"] = pTypesIcon[
                    int(Counter(maxPchance).most_common(1)[0][0])
                ]

        except Exception:
            logger.exception("MINUTELY TEXT GEN ERROR %s", loc_tag)
            returnOBJ["minutely"]["summary"] = pTypesText[
                int(Counter(maxPchance).most_common(1)[0][0])
            ]
            returnOBJ["minutely"]["icon"] = pTypesIcon[
                int(Counter(maxPchance).most_common(1)[0][0])
            ]

        returnOBJ["minutely"]["data"] = minuteItems

    if exHourly != 1:
        returnOBJ["hourly"] = dict()
        # Compute int conversion once for reuse
        base_time_offset_int = int(baseTimeOffset)
        if not timeMachine:
            try:
                if summaryText:
                    hourIcon, hourText = calculate_day_text(
                        hourList_si[base_time_offset_int : base_time_offset_int + 24],
                        not is_all_night,
                        str(tz_name),
                        "hour",
                        icon,
                        unitSystem,
                    )

                    returnOBJ["hourly"]["summary"] = translation.translate(
                        ["sentence", hourText]
                    )
                    returnOBJ["hourly"]["icon"] = hourIcon
                else:
                    returnOBJ["hourly"]["summary"] = max(
                        set(hourTextList), key=hourTextList.count
                    )
                    returnOBJ["hourly"]["icon"] = max(
                        set(hourIconList), key=hourIconList.count
                    )

            except Exception:
                logger.exception("TEXT GEN ERROR %s", loc_tag)
                returnOBJ["hourly"]["summary"] = max(
                    set(hourTextList), key=hourTextList.count
                )
                returnOBJ["hourly"]["icon"] = max(
                    set(hourIconList), key=hourIconList.count
                )
        else:  # Timemachine
            # Use simplified text for timemachine since this is covered in daily
            returnOBJ["hourly"]["summary"] = max(
                set(hourTextList), key=hourTextList.count
            )
            returnOBJ["hourly"]["icon"] = max(set(hourIconList), key=hourIconList.count)

        # Final hourly cleanup.
        fieldsToRemove = []

        # Remove 'smoke' if the version is less than 2.
        if version < 2:
            fieldsToRemove.extend(["smoke", "cape"])

        # Remove extra fields for basic Time Machine requests.
        if timeMachine and not tmExtra:
            fieldsToRemove.extend(
                [
                    "precipProbability",
                    "precipIntensityError",
                    "humidity",
                    "visibility",
                    "cape",
                    "solar",
                ]
            )

        # Apply all identified removals to the final hourList.
        if fieldsToRemove:
            for hourItem in hourList:
                for field in fieldsToRemove:
                    hourItem.pop(field, None)

        # If a timemachine request, do not offset to now
        if timeMachine:
            returnOBJ["hourly"]["data"] = hourList[0:ouputHours]
        else:
            returnOBJ["hourly"]["data"] = hourList[
                base_time_offset_int : base_time_offset_int + ouputHours
            ]

    if inc_day_night == 1 and not timeMachine:
        returnOBJ["day_night"] = dict()
        returnOBJ["day_night"]["data"] = day_night_list[0 : (ouputDays * 2)]

    if exDaily != 1:
        returnOBJ["daily"] = dict()
        if (
            not timeMachine
        ):  # Since TimeMachine Requests only have 24 hours of data, skip weekly summary
            try:
                if summaryText:
                    weekText, weekIcon = calculate_weekly_text(
                        dayList_si, str(tz_name), unitSystem, icon
                    )
                    returnOBJ["daily"]["summary"] = translation.translate(
                        ["sentence", weekText]
                    )
                    returnOBJ["daily"]["icon"] = weekIcon
                else:
                    returnOBJ["daily"]["summary"] = max(
                        set(dayTextList), key=dayTextList.count
                    )
                    returnOBJ["daily"]["icon"] = max(
                        set(dayIconList), key=dayIconList.count
                    )

            except Exception:
                logger.exception("DAILY SUMMARY TEXT GEN ERROR %s", loc_tag)
                returnOBJ["daily"]["summary"] = max(
                    set(dayTextList), key=dayTextList.count
                )
                returnOBJ["daily"]["icon"] = max(
                    set(dayIconList), key=dayIconList.count
                )
        else:
            # Timemachine fallback
            returnOBJ["daily"]["summary"] = max(set(dayTextList), key=dayTextList.count)
            returnOBJ["daily"]["icon"] = max(set(dayIconList), key=dayIconList.count)

        returnOBJ["daily"]["data"] = dayList[0:ouputDays]

    if exAlerts != 1:
        returnOBJ["alerts"] = alertList

    # Timing Check
    if TIMING:
        print("Final Time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    if exFlags != 1:
        returnOBJ["flags"] = dict()
        returnOBJ["flags"]["sources"] = sourceList
        returnOBJ["flags"]["sourceTimes"] = sourceTimes
        returnOBJ["flags"]["nearest-station"] = int(0)
        returnOBJ["flags"]["units"] = unitSystem
        returnOBJ["flags"]["version"] = API_VERSION
        if version >= 2:
            returnOBJ["flags"]["sourceIDX"] = sourceIDX
            returnOBJ["flags"]["processTime"] = (
                datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
            ).microseconds
            returnOBJ["flags"]["ingestVersion"] = ingest_version
            # Return the approx location names, if they are found
            returnOBJ["flags"]["nearestCity"] = loc_name.get("city") or None
            returnOBJ["flags"]["nearestCountry"] = loc_name.get("country") or None
            returnOBJ["flags"]["nearestSubNational"] = loc_name.get("state") or None

    # Timing Check
    if TIMING:
        print("Flags Time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Replace all MISSING_DATA with -999
    returnOBJ = replace_nan(returnOBJ, -999)

    if TIMING:
        print("Replace NaN Time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

        handler_ms = (
            datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
        ).total_seconds() * 1000

    else:
        handler_ms = (
            datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
        ).total_seconds() * 1000

    return ORJSONResponse(
        content=returnOBJ,
        headers={
            "X-Node-ID": platform.node(),
            "X-Handler-Time": f"{handler_ms:.1f}",
            "X-Response-Time": str(
                (
                    datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
                ).total_seconds()
                * 1000
            ),
            "Cache-Control": "max-age=900, must-revalidate",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


@app.on_event("startup")
def initialDataLoad() -> None:
    """Load zarr stores on startup.

    File syncing is now handled by a separate container.
    This just loads the zarr stores from their expected paths.
    """
    global zarrReady

    zarrReady = False
    logger.info("Initial data load")

    STAGE = os.environ.get("STAGE", "PROD")

    if STAGE in ("PROD", "DEV", "TIMEMACHINE"):
        update_zarr_store(True)

    zarrReady = True

    logger.info("Initial data load complete")


def nearest_index(a, v):
    """Find the nearest index in array to value using binary search.

    Slightly faster than a simple linear search for large arrays.

    Args:
        a: Sorted array
        v: Value to find

    Returns:
        Index of nearest value in array
    """
    # Find insertion point
    idx = np.searchsorted(a, v)
    # Clip so we don’t run off the ends
    idx = np.clip(idx, 1, len(a) - 1)
    # Look at neighbors, pick the closer one
    left, right = a[idx - 1], a[idx]
    return idx if abs(right - v) < abs(v - left) else idx - 1
