"""
Response Local Module for Pirate Weather API.

This module handles the local weather data processing and API responses.
It includes functions for reading weather data from zarr files, processing
weather forecasts, and generating API responses.
"""

import asyncio
import datetime
import logging
import math
import os
import platform
import re
import sys
import threading
from typing import Union

import metpy as mp
import numpy as np
import reverse_geocode
import xarray as xr
from astral import LocationInfo, moon
from astral.sun import sun
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from metpy.calc import relative_humidity_from_dewpoint
from pirateweather_translations.dynamic_loader import load_all_translations
from pytz import timezone, utc
from timezonefinder import TimezoneFinder

from API.api_utils import (
    clipLog,
    estimate_visibility_gultepe_rh_pr_numpy,
    fast_nearest_interp,
    replace_nan,
)
from API.constants.api_const import (
    API_VERSION,
    COORDINATE_CONST,
    ETOPO_CONST,
    ROUNDING_RULES,
    TIME_MACHINE_CONST,
)
from API.constants.clip_const import (
    CLIP_OZONE,
    CLIP_SMOKE,
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
)
from API.constants.shared_const import (
    HISTORY_PERIODS,
    INGEST_VERSION_STR,
    KELVIN_TO_CELSIUS,
    MISSING_DATA,
)
from API.constants.unit_const import country_units
from API.current.metrics import build_current_section
from API.daily.builder import build_daily_section
from API.hourly.block import build_hourly_block
from API.hourly.builder import initialize_time_grids
from API.io.zarr_reader import WeatherParallel, update_zarr_store
from API.legacy.summary import (
    build_daily_summary,
    build_hourly_summary,
    build_minutely_summary,
)
from API.minutely.builder import build_minutely_block
from API.utils.geo import (
    _polar_is_all_day,
    get_offset,
    lambertGridMatch,
    rounder,
)
from API.utils.timing import TimingMiddleware, TimingTracker

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
use_etopo = str(os.getenv("use_etopo", "True")).lower() not in {"0", "false", "no"}
TIMING = str(os.environ.get("TIMING", "0")).lower() not in {"0", "false", "no"}

force_now = os.getenv("force_now", default=False)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    # include timestamp, level, logger name, module, line number, message
    fmt = "%(asctime)s %(levelname)s [%(name)s:%(module)s:%(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


logger = logging.getLogger("pirate-weather-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Initialize Zarr stores via helper module
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


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(TimingMiddleware)

STAGE = os.environ.get("STAGE", "PROD")
logger.info("OS: %s Stage: %s", platform.system(), STAGE)

zarr_stores = update_zarr_store(
    True,
    stage=STAGE,
    save_dir=save_dir,
    use_etopo=use_etopo,
    save_type=save_type,
    s3_bucket=s3_bucket,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    logger=logger,
)

ETOPO_f = zarr_stores.ETOPO_f
SubH_Zarr = zarr_stores.SubH_Zarr
HRRR_6H_Zarr = zarr_stores.HRRR_6H_Zarr
GFS_Zarr = zarr_stores.GFS_Zarr
ECMWF_Zarr = zarr_stores.ECMWF_Zarr
NBM_Zarr = zarr_stores.NBM_Zarr
NBM_Fire_Zarr = zarr_stores.NBM_Fire_Zarr
GEFS_Zarr = zarr_stores.GEFS_Zarr
HRRR_Zarr = zarr_stores.HRRR_Zarr
NWS_Alerts_Zarr = zarr_stores.NWS_Alerts_Zarr
WMO_Alerts_Zarr = zarr_stores.WMO_Alerts_Zarr
RTMA_RU_Zarr = zarr_stores.RTMA_RU_Zarr
ERA5_Data = zarr_stores.ERA5_Data

logger.info("Initial data load complete")


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

    timing_tracker = TimingTracker(
        logger=logger,
        enabled=TIMING,
        prefix=f"{loc_tag} ",
    )

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

    (
        minute_array_grib,
        minute_array,
        InterTminute,
        InterPminute,
        InterPhour,
        hour_array_grib,
        hour_array,
        day_array_grib,
    ) = initialize_time_grids(
        base_time=baseTime,
        base_day=baseDay,
        daily_days=daily_days,
        daily_day_hours=daily_day_hours,
        timezone_localizer=pytzTZ,
    )

    numHours = len(hour_array)
    HRRR_Merged = None
    NBM_Merged = None
    NBM_Fire_Merged = None
    GFS_Merged = None
    ECMWF_Merged = None
    GEFS_Merged = None

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

    with timing_tracker.track("Minutely block"):
        (
            InterPminute,
            InterTminute,
            minuteItems,
            minuteItems_si,
            maxPchance,
            pTypesText,
            pTypesIcon,
            hrrrSubHInterpolation,
        ) = build_minutely_block(
            minute_array_grib=minute_array_grib,
            source_list=sourceList,
            hrrr_subh_data=dataOut if isinstance(dataOut, np.ndarray) else None,
            hrrr_merged=HRRR_Merged
            if ("hrrr_0-18" in sourceList and "hrrr_18-48" in sourceList)
            else None,
            nbm_data=dataOut_nbm if "nbm" in sourceList else None,
            gefs_data=dataOut_gefs if "gefs" in sourceList else None,
            gfs_data=dataOut_gfs if "gfs" in sourceList else None,
            ecmwf_data=dataOut_ecmwf if "ecmwf_ifs" in sourceList else None,
            era5_data=ERA5_MERGED if isinstance(ERA5_MERGED, np.ndarray) else None,
            prep_intensity_unit=prepIntensityUnit,
            version=version,
        )
    minuteRainIntensity = InterPminute[:, DATA_MINUTELY["rain_intensity"]]
    minuteSnowIntensity = InterPminute[:, DATA_MINUTELY["snow_intensity"]]
    minuteSleetIntensity = InterPminute[:, DATA_MINUTELY["ice_intensity"]]

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
        moon_phase_value = np.clip(m / 27.99, 0.0, 1.0)
        InterSday[i, DATA_DAY["moon_phase"]] = np.round(
            moon_phase_value, ROUNDING_RULES.get("moonPhase", 2)
        )

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

    def _stack_fields(*arrays):
        valid = [np.asarray(arr) for arr in arrays if arr is not None]
        if not valid:
            return np.full((numHours, 1), np.nan)
        return np.column_stack(valid)

    InterThour_inputs = {}
    if "nbm" in sourceList and NBM_Merged is not None:
        InterThour_inputs["nbm_snow"] = NBM_Merged[:, NBM["snow"]]
        InterThour_inputs["nbm_ice"] = NBM_Merged[:, NBM["ice"]]
        InterThour_inputs["nbm_freezing_rain"] = NBM_Merged[:, NBM["freezing_rain"]]
        InterThour_inputs["nbm_rain"] = NBM_Merged[:, NBM["rain"]]
    if (
        ("hrrr_0-18" in sourceList)
        and ("hrrr_18-48" in sourceList)
        and (HRRR_Merged is not None)
    ):
        InterThour_inputs["hrrr_snow"] = HRRR_Merged[:, HRRR["snow"]]
        InterThour_inputs["hrrr_ice"] = HRRR_Merged[:, HRRR["ice"]]
        InterThour_inputs["hrrr_freezing_rain"] = HRRR_Merged[:, HRRR["freezing_rain"]]
        InterThour_inputs["hrrr_rain"] = HRRR_Merged[:, HRRR["rain"]]
    if "ecmwf_ifs" in sourceList and ECMWF_Merged is not None:
        InterThour_inputs["ecmwf_ptype"] = ECMWF_Merged[:, ECMWF["ptype"]]
    if "gefs" in sourceList and GEFS_Merged is not None:
        InterThour_inputs["gefs_snow"] = GEFS_Merged[:, GEFS["snow"]]
        InterThour_inputs["gefs_ice"] = GEFS_Merged[:, GEFS["ice"]]
        InterThour_inputs["gefs_freezing_rain"] = GEFS_Merged[:, GEFS["freezing_rain"]]
        InterThour_inputs["gefs_rain"] = GEFS_Merged[:, GEFS["rain"]]
    elif "gfs" in sourceList and GFS_Merged is not None:
        InterThour_inputs["gefs_snow"] = GFS_Merged[:, GFS["snow"]]
        InterThour_inputs["gefs_ice"] = GFS_Merged[:, GFS["ice"]]
        InterThour_inputs["gefs_freezing_rain"] = GFS_Merged[:, GFS["freezing_rain"]]
        InterThour_inputs["gefs_rain"] = GFS_Merged[:, GFS["rain"]]
    if "era5" in sourceList and isinstance(ERA5_MERGED, np.ndarray):
        InterThour_inputs["era5_ptype"] = ERA5_MERGED[:, ERA5["precipitation_type"]]

    prcipIntensity_inputs = {}
    if "nbm" in sourceList and NBM_Merged is not None:
        prcipIntensity_inputs["nbm"] = NBM_Merged[:, NBM["intensity"]]
    if (
        ("hrrr_0-18" in sourceList)
        and ("hrrr_18-48" in sourceList)
        and (HRRR_Merged is not None)
    ):
        prcipIntensity_inputs["hrrr"] = HRRR_Merged[:, HRRR["intensity"]] * 3600
    if "ecmwf_ifs" in sourceList and ECMWF_Merged is not None:
        prcipIntensity_inputs["ecmwf"] = ECMWF_Merged[:, ECMWF["intensity"]] * 3600
    if "gefs" in sourceList and GEFS_Merged is not None:
        prcipIntensity_inputs["gfs_gefs"] = GEFS_Merged[:, GEFS["accum"]]
    elif "gfs" in sourceList and GFS_Merged is not None:
        prcipIntensity_inputs["gfs_gefs"] = GFS_Merged[:, GFS["intensity"]] * 3600
    if "era5" in sourceList and isinstance(ERA5_MERGED, np.ndarray):
        prcipIntensity_inputs["era5"] = (
            ERA5_MERGED[:, ERA5["large_scale_rain_rate"]]
            + ERA5_MERGED[:, ERA5["convective_rain_rate"]]
            + ERA5_MERGED[:, ERA5["large_scale_snowfall_rate_water_equivalent"]]
            + ERA5_MERGED[:, ERA5["convective_snowfall_rate_water_equivalent"]]
        ) * 3600
        era5_rain_intensity = (
            ERA5_MERGED[:, ERA5["large_scale_rain_rate"]]
            + ERA5_MERGED[:, ERA5["convective_rain_rate"]]
        ) * 3600
        era5_snow_water_equivalent = (
            ERA5_MERGED[:, ERA5["large_scale_snowfall_rate_water_equivalent"]]
            + ERA5_MERGED[:, ERA5["convective_snowfall_rate_water_equivalent"]]
        ) * 3600
    else:
        era5_rain_intensity = None
        era5_snow_water_equivalent = None

    prcipProbability_inputs = {}
    if "nbm" in sourceList and NBM_Merged is not None:
        prcipProbability_inputs["nbm"] = NBM_Merged[:, NBM["prob"]] * 0.01
    if "ecmwf_ifs" in sourceList and ECMWF_Merged is not None:
        prcipProbability_inputs["ecmwf"] = ECMWF_Merged[:, ECMWF["prob"]]
    if "gefs" in sourceList and GEFS_Merged is not None:
        prcipProbability_inputs["gefs"] = GEFS_Merged[:, GEFS["prob"]]

    temperature_inputs = _stack_fields(
        NBM_Merged[:, NBM["temp"]] if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["temp"]] if HRRR_Merged is not None else None,
        ECMWF_Merged[:, ECMWF["temp"]] if ECMWF_Merged is not None else None,
        GFS_Merged[:, GFS["temp"]] if GFS_Merged is not None else None,
        ERA5_MERGED[:, ERA5["2m_temperature"]]
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )
    dew_inputs = _stack_fields(
        NBM_Merged[:, NBM["dew"]] if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["dew"]] if HRRR_Merged is not None else None,
        ECMWF_Merged[:, ECMWF["dew"]] if ECMWF_Merged is not None else None,
        GFS_Merged[:, GFS["dew"]] if GFS_Merged is not None else None,
        ERA5_MERGED[:, ERA5["2m_dewpoint_temperature"]]
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    era5_humidity = None
    if isinstance(ERA5_MERGED, np.ndarray):
        era5_humidity = (
            relative_humidity_from_dewpoint(
                ERA5_MERGED[:, ERA5["2m_temperature"]] * mp.units.units.degK,
                ERA5_MERGED[:, ERA5["2m_dewpoint_temperature"]] * mp.units.units.degK,
                phase="auto",
            ).magnitude
            * 100
        )
    humidity_inputs = _stack_fields(
        NBM_Merged[:, NBM["humidity"]] if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["humidity"]] if HRRR_Merged is not None else None,
        GFS_Merged[:, GFS["humidity"]] if GFS_Merged is not None else None,
        era5_humidity,
    )

    pressure_inputs = _stack_fields(
        HRRR_Merged[:, HRRR["pressure"]] if HRRR_Merged is not None else None,
        ECMWF_Merged[:, ECMWF["pressure"]] if ECMWF_Merged is not None else None,
        GFS_Merged[:, GFS["pressure"]] if GFS_Merged is not None else None,
        ERA5_MERGED[:, ERA5["mean_sea_level_pressure"]]
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    def _wind_speed(u, v):
        if u is None or v is None:
            return None
        return np.sqrt(u**2 + v**2)

    wind_inputs = _stack_fields(
        NBM_Merged[:, NBM["wind"]] if NBM_Merged is not None else None,
        _wind_speed(HRRR_Merged[:, HRRR["wind_u"]], HRRR_Merged[:, HRRR["wind_v"]])
        if HRRR_Merged is not None
        else None,
        _wind_speed(ECMWF_Merged[:, ECMWF["wind_u"]], ECMWF_Merged[:, ECMWF["wind_v"]])
        if ECMWF_Merged is not None
        else None,
        _wind_speed(GFS_Merged[:, GFS["wind_u"]], GFS_Merged[:, GFS["wind_v"]])
        if GFS_Merged is not None
        else None,
        _wind_speed(
            ERA5_MERGED[:, ERA5["10m_u_component_of_wind"]],
            ERA5_MERGED[:, ERA5["10m_v_component_of_wind"]],
        )
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    gust_inputs = _stack_fields(
        NBM_Merged[:, NBM["gust"]] if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["gust"]] if HRRR_Merged is not None else None,
        GFS_Merged[:, GFS["gust"]] if GFS_Merged is not None else None,
        ERA5_MERGED[:, ERA5["instantaneous_10m_wind_gust"]]
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    def _bearing(u, v):
        if u is None or v is None:
            return None
        return np.rad2deg(np.mod(np.arctan2(u, v) + np.pi, 2 * np.pi))

    bearing_inputs = _stack_fields(
        NBM_Merged[:, NBM["bearing"]] if NBM_Merged is not None else None,
        _bearing(HRRR_Merged[:, HRRR["wind_u"]], HRRR_Merged[:, HRRR["wind_v"]])
        if HRRR_Merged is not None
        else None,
        _bearing(ECMWF_Merged[:, ECMWF["wind_u"]], ECMWF_Merged[:, ECMWF["wind_v"]])
        if ECMWF_Merged is not None
        else None,
        _bearing(GFS_Merged[:, GFS["wind_u"]], GFS_Merged[:, GFS["wind_v"]])
        if GFS_Merged is not None
        else None,
        _bearing(
            ERA5_MERGED[:, ERA5["10m_u_component_of_wind"]],
            ERA5_MERGED[:, ERA5["10m_v_component_of_wind"]],
        )
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    cloud_inputs = _stack_fields(
        NBM_Merged[:, NBM["cloud"]] * 0.01 if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["cloud"]] * 0.01 if HRRR_Merged is not None else None,
        ECMWF_Merged[:, ECMWF["cloud"]] * 0.01 if ECMWF_Merged is not None else None,
        GFS_Merged[:, GFS["cloud"]] * 0.01 if GFS_Merged is not None else None,
        ERA5_MERGED[:, ERA5["total_cloud_cover"]]
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    uv_inputs = _stack_fields(
        (GFS_Merged[:, GFS["uv"]] * 18.9 * 0.025) if GFS_Merged is not None else None,
        (
            ERA5_MERGED[:, ERA5["downward_uv_radiation_at_the_surface"]]
            / 3600
            * 40
            * 0.0025
        )
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    vis_inputs = _stack_fields(
        NBM_Merged[:, NBM["vis"]] if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["vis"]] if HRRR_Merged is not None else None,
        GFS_Merged[:, GFS["vis"]] if GFS_Merged is not None else None,
        estimate_visibility_gultepe_rh_pr_numpy(ERA5_MERGED, var_index=ERA5, var_axis=1)
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    ozone_inputs = _stack_fields(
        clipLog(
            GFS_Merged[:, GFS["ozone"]],
            CLIP_OZONE["min"],
            CLIP_OZONE["max"],
            "Ozone Hour",
        )
        if GFS_Merged is not None
        else None,
        clipLog(
            ERA5_MERGED[:, ERA5["total_column_ozone"]] * 46696,
            CLIP_OZONE["min"],
            CLIP_OZONE["max"],
            "Ozone Hour",
        )
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    smoke_inputs = _stack_fields(
        clipLog(
            HRRR_Merged[:, HRRR["smoke"]],
            CLIP_SMOKE["min"],
            CLIP_SMOKE["max"],
            "Air quality Hour",
        )
        if HRRR_Merged is not None
        else None
    )

    accum_inputs = _stack_fields(
        NBM_Merged[:, NBM["intensity"]] if NBM_Merged is not None else None,
        HRRR_Merged[:, HRRR["accum"]] if HRRR_Merged is not None else None,
        ECMWF_Merged[:, ECMWF["accum_mean"]] * 1000
        if ECMWF_Merged is not None
        else None,
        GEFS_Merged[:, GEFS["accum"]] if GEFS_Merged is not None else None,
        GFS_Merged[:, GFS["accum"]] if GFS_Merged is not None else None,
        ERA5_MERGED[:, ERA5["total_precipitation"]] * 1000
        if isinstance(ERA5_MERGED, np.ndarray)
        else None,
    )

    nearstorm_inputs = {
        "dist": _stack_fields(
            np.maximum(GFS_Merged[:, GFS["storm_dist"]], 0)
            if GFS_Merged is not None
            else None
        ),
        "dir": _stack_fields(
            GFS_Merged[:, GFS["storm_dir"]] if GFS_Merged is not None else None
        ),
    }

    apparent_inputs = _stack_fields(
        NBM_Merged[:, NBM["apparent"]] if NBM_Merged is not None else None,
        GFS_Merged[:, GFS["apparent"]] if GFS_Merged is not None else None,
    )

    station_pressure_inputs = None
    if "stationPressure" in extraVars:
        station_pressure_inputs = _stack_fields(
            GFS_Merged[:, GFS["station_pressure"]] if GFS_Merged is not None else None,
            ERA5_MERGED[:, ERA5["surface_pressure"]]
            if isinstance(ERA5_MERGED, np.ndarray)
            else None,
        )

    with timing_tracker.track("Hourly block"):
        (
            hourList,
            hourList_si,
            hourIconList,
            hourTextList,
            dayZeroRain,
            dayZeroSnow,
            dayZeroIce,
            hourly_display,
            PTypeHour,
            PTextHour,
        ) = build_hourly_block(
            source_list=sourceList,
            InterPhour=InterPhour,
            hour_array_grib=hour_array_grib,
            hour_array=hour_array,
            InterSday=InterSday,
            hourlyDayIndex=hourlyDayIndex,
            baseTimeOffset=baseTimeOffset,
            timeMachine=timeMachine,
            prepIntensityUnit=prepIntensityUnit,
            prepAccumUnit=prepAccumUnit,
            windUnit=windUnit,
            visUnits=visUnits,
            tempUnits=tempUnits,
            extraVars=extraVars,
            summaryText=summaryText,
            icon=icon,
            translation=translation,
            unitSystem=unitSystem,
            is_all_night=is_all_night,
            tz_name=tz_name,
            InterThour_inputs=InterThour_inputs,
            prcipIntensity_inputs=prcipIntensity_inputs,
            prcipProbability_inputs=prcipProbability_inputs,
            temperature_inputs=temperature_inputs,
            dew_inputs=dew_inputs,
            humidity_inputs=humidity_inputs,
            pressure_inputs=pressure_inputs,
            wind_inputs=wind_inputs,
            gust_inputs=gust_inputs,
            bearing_inputs=bearing_inputs,
            cloud_inputs=cloud_inputs,
            uv_inputs=uv_inputs,
            vis_inputs=vis_inputs,
            ozone_inputs=ozone_inputs,
            smoke_inputs=smoke_inputs,
            accum_inputs=accum_inputs,
            nearstorm_inputs=nearstorm_inputs,
            apparent_inputs=apparent_inputs,
            station_pressure_inputs=station_pressure_inputs,
            era5_rain_intensity=era5_rain_intensity,
            era5_snow_water_equivalent=era5_snow_water_equivalent,
        )

    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])

    with timing_tracker.track("Daily block"):
        daily_section = build_daily_section(
            InterPhour=InterPhour,
            hourlyDayIndex=hourlyDayIndex,
            hourlyDay4amIndex=hourlyDay4amIndex,
            hourlyDay4pmIndex=hourlyDay4pmIndex,
            hourlyNight4amIndex=hourlyNight4amIndex,
            hourlyHighIndex=hourlyHighIndex,
            hourlyLowIndex=hourlyLowIndex,
            daily_days=daily_days,
            prepAccumUnit=prepAccumUnit,
            prepIntensityUnit=prepIntensityUnit,
            windUnit=windUnit,
            visUnits=visUnits,
            tempUnits=tempUnits,
            extraVars=extraVars,
            summaryText=summaryText,
            translation=translation,
            is_all_night=is_all_night,
            is_all_day=is_all_day,
            tz_name=tz_name,
            icon=icon,
            unitSystem=unitSystem,
            version=version,
            timeMachine=timeMachine,
            tmExtra=tmExtra,
            day_array_grib=day_array_grib,
            day_array_4am_grib=day_array_4am_grib,
            day_array_5pm_grib=day_array_5pm_grib,
            InterSday=InterSday,
            hourList_si=hourList_si,
            pTypeMap=pTypeMap,
            pTextMap=pTextMap,
            logger=logger,
            loc_tag=loc_tag,
        )

    dayList = daily_section.day_list
    dayList_si = daily_section.day_list_si
    dayIconList = daily_section.day_icon_list
    dayTextList = daily_section.day_text_list
    day_night_list = daily_section.day_night_list

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

    with timing_tracker.track("Current block"):
        current_section = build_current_section(
            sourceList=sourceList,
            hour_array_grib=hour_array_grib,
            minute_array_grib=minute_array_grib,
            InterPminute=InterPminute,
            minuteItems=minuteItems,
            minuteRainIntensity=minuteRainIntensity,
            minuteSnowIntensity=minuteSnowIntensity,
            minuteSleetIntensity=minuteSleetIntensity,
            InterSday=InterSday,
            dayZeroRain=dayZeroRain,
            dayZeroSnow=dayZeroSnow,
            dayZeroIce=dayZeroIce,
            prepAccumUnit=prepAccumUnit,
            prepIntensityUnit=prepIntensityUnit,
            windUnit=windUnit,
            visUnits=visUnits,
            tempUnits=tempUnits,
            humidUnit=humidUnit,
            extraVars=extraVars,
            summaryText=summaryText,
            translation=translation,
            icon=icon,
            unitSystem=unitSystem,
            version=version,
            timeMachine=timeMachine,
            tmExtra=tmExtra,
            lat=lat,
            lon_IN=lon_IN,
            tz_name=tz_name,
            tz_offset=tz_offset,
            ETOPO=ETOPO,
            elevUnit=elevUnit,
            dataOut_rtma_ru=dataOut_rtma_ru,
            hrrrSubHInterpolation=hrrrSubHInterpolation,
            HRRR_Merged=HRRR_Merged,
            NBM_Merged=NBM_Merged,
            ECMWF_Merged=ECMWF_Merged,
            GFS_Merged=GFS_Merged,
            ERA5_MERGED=ERA5_MERGED,
            NBM_Fire_Merged=NBM_Fire_Merged,
            logger=logger,
            loc_tag=loc_tag,
            include_currently=exCurrently != 1,
        )

    ### RETURN ###
    returnOBJ = dict()

    returnOBJ["latitude"] = round(float(lat), 4)
    returnOBJ["longitude"] = round(float(lon_IN), 4)
    returnOBJ["timezone"] = str(tz_name)
    returnOBJ["offset"] = float(tz_offset / 60)
    returnOBJ["elevation"] = int(round(float(ETOPO * elevUnit), 0))

    if exCurrently != 1:
        returnOBJ["currently"] = dict(current_section.currently)

    if exMinutely != 1:
        returnOBJ["minutely"] = dict()
        current_cape = float(
            np.nan_to_num(
                current_section.interp_current[DATA_CURRENT["cape"]],
                nan=0,
            )
        )
        hourly_cape = 0.0
        if len(InterPhour) > 0:
            hourly_cape = float(
                np.nan_to_num(InterPhour[0, DATA_HOURLY["cape"]], nan=0)
            )
        minute_summary, minute_icon = build_minutely_summary(
            summary_text=summaryText,
            translation=translation,
            inter_p_current=current_cape,
            inter_p_hour=hourly_cape,
            minute_items_si=minuteItems_si,
            current_text=(
                current_section.summary_key
                or current_section.currently.get("summary", "")
            ),
            current_icon=current_section.currently.get("icon", ""),
            icon=icon,
            max_p_chance=maxPchance,
            p_types_text=pTypesText,
            p_types_icon=pTypesIcon,
            logger=logger,
            loc_tag=loc_tag,
        )
        returnOBJ["minutely"]["summary"] = minute_summary
        returnOBJ["minutely"]["icon"] = minute_icon
        returnOBJ["minutely"]["data"] = minuteItems

    if exHourly != 1:
        returnOBJ["hourly"] = dict()
        # Compute int conversion once for reuse
        base_time_offset_int = int(baseTimeOffset)
        hour_summary, hour_icon = build_hourly_summary(
            summary_text=summaryText,
            translation=translation,
            hour_list_si=hourList_si,
            is_all_night=is_all_night,
            tz_name=tz_name,
            icon=icon,
            unit_system=unitSystem,
            hour_text_list=hourTextList,
            hour_icon_list=hourIconList,
            time_machine=timeMachine,
            base_time_offset_int=base_time_offset_int,
            logger=logger,
            loc_tag=loc_tag,
        )
        returnOBJ["hourly"]["summary"] = hour_summary
        returnOBJ["hourly"]["icon"] = hour_icon

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
        daily_summary, daily_icon = build_daily_summary(
            summary_text=summaryText,
            translation=translation,
            day_list_si=dayList_si,
            tz_name=tz_name,
            unit_system=unitSystem,
            icon=icon,
            day_text_list=dayTextList,
            day_icon_list=dayIconList,
            time_machine=timeMachine,
            logger=logger,
            loc_tag=loc_tag,
        )
        returnOBJ["daily"]["summary"] = daily_summary
        returnOBJ["daily"]["icon"] = daily_icon

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
