"""
Response Local Module for Pirate Weather API.

This module handles the local weather data processing and API responses.
It includes functions for reading weather data from zarr files, processing
weather forecasts, and generating API responses.
"""

import datetime
import logging
import os
import platform
import sys
import threading
from typing import Union

import numpy as np
from astral import LocationInfo, moon
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from pirateweather_translations.dynamic_loader import load_all_translations
from timezonefinder import TimezoneFinder

from API.alerts import build_alerts
from API.api_utils import (
    replace_nan,
)
from API.constants.api_const import (
    API_VERSION,
    COORDINATE_CONST,
    ETOPO_CONST,
    ROUNDING_RULES,
)
from API.constants.forecast_const import (
    DATA_CURRENT,
    DATA_DAY,
    DATA_HOURLY,
    DATA_MINUTELY,
)
from API.constants.model_const import (
    ECMWF,
    ERA5,
    GFS,
    HRRR,
    HRRR_SUBH,
    NBM,
    RTMA_RU,
)

# Project imports
from API.constants.shared_const import INGEST_VERSION_STR, KELVIN_TO_CELSIUS
from API.current.metrics import build_current_section
from API.daily.builder import build_daily_section
from API.data_inputs import prepare_data_inputs
from API.forecast_sources import (
    add_etopo_source,
    build_source_metadata,
    merge_hourly_models,
)
from API.hourly.block import build_hourly_block
from API.io.zarr_reader import update_zarr_store
from API.legacy.summary import (
    build_daily_summary,
    build_hourly_summary,
    build_minutely_summary,
)
from API.minutely.builder import build_minutely_block
from API.request.grid_indexing import ZarrSources, calculate_grid_indexing
from API.request.preprocess import prepare_initial_request
from API.utils.solar import calculate_solar_times
from API.utils.time_indexing import calculate_time_indexing
from API.utils.timing import TimingMiddleware

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

"""Load zarr stores on startup.

File syncing is now handled by a separate container.
This just loads the zarr stores from their expected paths.
"""
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

logger.info("Initial data load complete")

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


def convert_data_to_celsius(
    dataOut,
    dataOut_h2,
    dataOut_hrrrh,
    dataOut_nbm,
    dataOut_gfs,
    dataOut_ecmwf,
    dataOut_rtma_ru,
    era5_merged,
):
    """
    Converts temperature, dew point, and apparent temperature from Kelvin to Celsius
    for the provided data arrays.
    """
    # HRRR Sub-Hourly
    if isinstance(dataOut, np.ndarray):
        dataOut[:, HRRR_SUBH["temp"]] -= KELVIN_TO_CELSIUS
        dataOut[:, HRRR_SUBH["dew"]] -= KELVIN_TO_CELSIUS

    # HRRR 6H
    if isinstance(dataOut_h2, np.ndarray):
        dataOut_h2[:, HRRR["temp"]] -= KELVIN_TO_CELSIUS
        dataOut_h2[:, HRRR["dew"]] -= KELVIN_TO_CELSIUS

    # HRRR Hourly
    if isinstance(dataOut_hrrrh, np.ndarray):
        dataOut_hrrrh[:, HRRR["temp"]] -= KELVIN_TO_CELSIUS
        dataOut_hrrrh[:, HRRR["dew"]] -= KELVIN_TO_CELSIUS

    # NBM
    if isinstance(dataOut_nbm, np.ndarray):
        dataOut_nbm[:, NBM["temp"]] -= KELVIN_TO_CELSIUS
        dataOut_nbm[:, NBM["dew"]] -= KELVIN_TO_CELSIUS
        dataOut_nbm[:, NBM["apparent"]] -= KELVIN_TO_CELSIUS

    # GFS
    if isinstance(dataOut_gfs, np.ndarray):
        dataOut_gfs[:, GFS["temp"]] -= KELVIN_TO_CELSIUS
        dataOut_gfs[:, GFS["dew"]] -= KELVIN_TO_CELSIUS
        dataOut_gfs[:, GFS["apparent"]] -= KELVIN_TO_CELSIUS

    # ECMWF
    if isinstance(dataOut_ecmwf, np.ndarray):
        dataOut_ecmwf[:, ECMWF["temp"]] -= KELVIN_TO_CELSIUS
        dataOut_ecmwf[:, ECMWF["dew"]] -= KELVIN_TO_CELSIUS

    # RTMA
    if isinstance(dataOut_rtma_ru, np.ndarray):
        dataOut_rtma_ru[:, RTMA_RU["temp"]] -= KELVIN_TO_CELSIUS
        dataOut_rtma_ru[:, RTMA_RU["dew"]] -= KELVIN_TO_CELSIUS

    # ERA5
    if isinstance(era5_merged, np.ndarray):
        era5_merged[:, ERA5["2m_temperature"]] -= KELVIN_TO_CELSIUS
        era5_merged[:, ERA5["2m_dewpoint_temperature"]] -= KELVIN_TO_CELSIUS


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
    """
    Main entry point for the Pirate Weather API forecast.

    This function handles the entire forecast generation process:
    1. Parses request parameters.
    2. Initializes Zarr data stores.
    3. Calculates grid indexing for the requested location.
    4. Merges data from various weather models (HRRR, GFS, etc.).
    5. Generates minutely, hourly, daily, and current weather sections.
    6. Processes alerts.
    7. Constructs and returns the final JSON response.

    Args:
        request: The FastAPI request object.
        location: The location string (lat,lon).
        units: Unit system (us, si, ca, uk2).
        extend: Extend hourly forecast (hourly).
        exclude: Blocks to exclude (currently, minutely, hourly, daily, alerts, flags).
        include: Blocks to include (overrides exclude).
        lang: Language for text summaries.
        version: API version.
        tmextra: Extra time machine parameters.
        apikey: The API key used for the request.
        icon: Icon set to use.
        extraVars: Extra variables to include.

    Returns:
        dict: The complete weather forecast JSON object.
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

    # Timing Check
    T_Start = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

    # Parse request parameters and initialize variables
    # This function handles all the input validation and setup
    initial = await prepare_initial_request(
        request=request,
        location=location,
        units=units,
        extend=extend,
        exclude=exclude,
        include=include,
        lang=lang,
        version=version,
        tmextra=tmextra,
        icon=icon,
        extraVars=extraVars,
        tf=tf,
        translations=Translations,
        timing_enabled=TIMING,
        force_now=force_now,
        logger=logger,
        start_time=T_Start,
    )

    STAGE = initial.stage
    lat = initial.lat
    lon_IN = initial.lon_in
    lon = initial.lon
    az_Lon = initial.az_lon
    nowTime = initial.now_time
    utcTime = initial.utc_time
    timeMachine = initial.time_machine
    loc_tag = initial.loc_tag
    timing_tracker = initial.timing_tracker
    tz_offset = initial.tz_offset
    tz_name = initial.tz_name
    loc_name = initial.loc_name
    icon = initial.icon
    translation = initial.translation
    version = initial.version
    tmExtra = initial.tm_extra
    extraVars = initial.extra_vars
    exCurrently = initial.ex_currently
    exMinutely = initial.ex_minutely
    exHourly = initial.ex_hourly
    exDaily = initial.ex_daily
    exFlags = initial.ex_flags
    exAlerts = initial.ex_alerts
    exNBM = initial.ex_nbm
    exHRRR = initial.ex_hrrr
    exGEFS = initial.ex_gefs
    exGFS = initial.ex_gfs
    exRTMA_RU = initial.ex_rtma_ru
    exECMWF = initial.ex_ecmwf
    inc_day_night = initial.inc_day_night
    summaryText = initial.summary_text
    unitSystem = initial.unit_system
    windUnit = initial.wind_unit
    prepIntensityUnit = initial.prep_intensity_unit
    prepAccumUnit = initial.prep_accum_unit
    tempUnits = initial.temp_units
    visUnits = initial.vis_units
    humidUnit = initial.humid_unit
    elevUnit = initial.elev_unit
    weather = initial.weather
    pytzTZ = initial.pytz_tz
    baseTime = initial.base_time
    baseHour = initial.base_hour
    baseDay = initial.base_day
    baseDayUTC = initial.base_day_utc
    baseDayUTC_Grib = initial.base_day_utc_grib
    daily_days = initial.daily_days
    ouputHours = initial.output_hours
    ouputDays = initial.output_days
    minute_array_grib = initial.minute_array_grib
    InterTminute = initial.inter_tminute
    InterPminute = initial.inter_pminute
    InterPhour = initial.inter_phour
    hour_array_grib = initial.hour_array_grib
    hour_array = initial.hour_array
    day_array_grib = initial.day_array_grib
    numHours = initial.num_hours
    readWMOAlerts = initial.read_wmo_alerts

    HRRR_Merged = None
    NBM_Merged = None
    NBM_Fire_Merged = None
    GFS_Merged = None
    ECMWF_Merged = None
    GEFS_Merged = None

    if TIMING:
        print("### HRRR Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    zarr_sources = ZarrSources(
        subh=SubH_Zarr,
        hrrr_6h=HRRR_6H_Zarr,
        hrrr=HRRR_Zarr,
        nbm=NBM_Zarr,
        nbm_fire=NBM_Fire_Zarr,
        gfs=GFS_Zarr,
        ecmwf=ECMWF_Zarr,
        gefs=GEFS_Zarr,
        rtma_ru=RTMA_RU_Zarr,
        wmo_alerts=WMO_Alerts_Zarr,
        era5_data=ERA5_Data,
    )

    # Calculate grid indices for the requested location to retrieve data from Zarr stores
    # This determines which grid points in the model data correspond to the user's lat/lon
    grid_result = await calculate_grid_indexing(
        lat=lat,
        lon=lon,
        az_lon=az_Lon,
        utc_time=utcTime,
        now_time=nowTime,
        time_machine=timeMachine,
        ex_hrrr=exHRRR,
        ex_nbm=exNBM,
        ex_gfs=exGFS,
        ex_ecmwf=exECMWF,
        ex_gefs=exGEFS,
        ex_rtma_ru=exRTMA_RU,
        read_wmo_alerts=readWMOAlerts,
        base_day_utc=baseDayUTC,
        zarr_sources=zarr_sources,
        weather=weather,
        timing_start=T_Start,
        timing_enabled=TIMING,
        logger=logger,
    )

    dataOut = grid_result.dataOut
    dataOut_h2 = grid_result.dataOut_h2
    dataOut_hrrrh = grid_result.dataOut_hrrrh
    dataOut_nbm = grid_result.dataOut_nbm
    dataOut_nbmFire = grid_result.dataOut_nbmFire
    dataOut_gfs = grid_result.dataOut_gfs
    dataOut_ecmwf = grid_result.dataOut_ecmwf
    dataOut_gefs = grid_result.dataOut_gefs
    dataOut_rtma_ru = grid_result.dataOut_rtma_ru
    WMO_alertDat = grid_result.WMO_alertDat

    ERA5_MERGED = grid_result.era5_merged

    # Convert temperature columns to Celsius based on model index
    convert_data_to_celsius(
        dataOut,
        dataOut_h2,
        dataOut_hrrrh,
        dataOut_nbm,
        dataOut_gfs,
        dataOut_ecmwf,
        dataOut_rtma_ru,
        ERA5_MERGED,
    )

    source_metadata = build_source_metadata(
        grid_result=grid_result,
        era5_merged=ERA5_MERGED,
        use_etopo=use_etopo,
        time_machine=timeMachine,
    )

    # Convert temperature columns to Celsius based on model index

    if TIMING:
        print("### Sources Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

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
        add_etopo_source(
            metadata=source_metadata,
            x_idx=x_p_etopo,
            y_idx=y_p_etopo,
            lat_val=lats_etopo[y_p_etopo],
            lon_val=lons_etopo[x_p_etopo],
        )

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
    # This combines data from multiple weather models (HRRR, GFS, etc.) to create a unified forecast
    if TIMING:
        print("Nearest IDX Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    merge_result = merge_hourly_models(
        metadata=source_metadata,
        num_hours=numHours,
        base_day_utc_grib=baseDayUTC_Grib,
        data_hrrrh=dataOut_hrrrh if isinstance(dataOut_hrrrh, np.ndarray) else None,
        data_h2=dataOut_h2 if isinstance(dataOut_h2, np.ndarray) else None,
        data_nbm=dataOut_nbm if isinstance(dataOut_nbm, np.ndarray) else None,
        data_nbm_fire=dataOut_nbmFire
        if isinstance(dataOut_nbmFire, np.ndarray)
        else None,
        data_gfs=dataOut_gfs if isinstance(dataOut_gfs, np.ndarray) else None,
        data_ecmwf=dataOut_ecmwf if isinstance(dataOut_ecmwf, np.ndarray) else None,
        data_gefs=dataOut_gefs if isinstance(dataOut_gefs, np.ndarray) else None,
        logger=logger,
        loc_tag=loc_tag,
    )

    HRRR_Merged = merge_result.hrrr
    NBM_Merged = merge_result.nbm
    NBM_Fire_Merged = merge_result.nbm_fire
    GFS_Merged = merge_result.gfs
    ECMWF_Merged = merge_result.ecmwf
    GEFS_Merged = merge_result.gefs
    sourceList = merge_result.metadata.source_list
    sourceTimes = merge_result.metadata.source_times
    sourceIDX = merge_result.metadata.source_idx

    # Generate the minutely forecast section (precipitation intensity, etc.)
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

    # Calculate time indices to align data with the requested time zone and intervals
    time_indexing = calculate_time_indexing(
        base_time=baseTime,
        timezone_localizer=pytzTZ,
        hour_array_grib=hour_array_grib,
        time_machine=timeMachine,
        existing_day_array_grib=day_array_grib,
    )

    day_array_grib = time_indexing.day_array_grib
    day_array_4am_grib = time_indexing.day_array_4am_grib
    day_array_5pm_grib = time_indexing.day_array_5pm_grib
    hourlyDayIndex = time_indexing.hourly_day_index
    hourlyDay4amIndex = time_indexing.hourly_day_4am_index
    hourlyHighIndex = time_indexing.hourly_high_index
    hourlyLowIndex = time_indexing.hourly_low_index
    hourlyDay4pmIndex = time_indexing.hourly_day_4pm_index
    hourlyNight4amIndex = time_indexing.hourly_night_4am_index

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
        (
            sunrise_value,
            sunset_value,
            dawn_value,
            dusk_value,
            saw_all_day,
            saw_all_night,
        ) = calculate_solar_times(
            day_index=i,
            base_day=baseDay,
            location=loc,
            day_array_grib=day_array_grib,
            latitude=lat,
        )
        InterSday[i, DATA_DAY["sunrise"]] = sunrise_value
        InterSday[i, DATA_DAY["sunset"]] = sunset_value
        InterSday[i, DATA_DAY["dawn"]] = dawn_value
        InterSday[i, DATA_DAY["dusk"]] = dusk_value
        is_all_day = is_all_day or saw_all_day
        is_all_night = is_all_night or saw_all_night

        m = moon.phase(baseDay + datetime.timedelta(days=i))
        moon_phase_value = np.clip(m / 27.99, 0.0, 1.0)
        InterSday[i, DATA_DAY["moon_phase"]] = np.round(
            moon_phase_value, ROUNDING_RULES.get("moonPhase", 2)
        )

    # Extract and prepare specific weather variables (temperature, wind, etc.) from the merged model data
    inputs = prepare_data_inputs(
        source_list=sourceList,
        nbm_merged=NBM_Merged,
        nbm_fire_merged=NBM_Fire_Merged,
        hrrr_merged=HRRR_Merged,
        ecmwf_merged=ECMWF_Merged,
        gefs_merged=GEFS_Merged,
        gfs_merged=GFS_Merged,
        era5_merged=ERA5_MERGED,
        extra_vars=extraVars,
        num_hours=numHours,
    )

    InterThour_inputs = inputs["InterThour_inputs"]
    prcipIntensity_inputs = inputs["prcipIntensity_inputs"]
    prcipProbability_inputs = inputs["prcipProbability_inputs"]
    temperature_inputs = inputs["temperature_inputs"]
    dew_inputs = inputs["dew_inputs"]
    humidity_inputs = inputs["humidity_inputs"]
    pressure_inputs = inputs["pressure_inputs"]
    wind_inputs = inputs["wind_inputs"]
    gust_inputs = inputs["gust_inputs"]
    bearing_inputs = inputs["bearing_inputs"]
    cloud_inputs = inputs["cloud_inputs"]
    uv_inputs = inputs["uv_inputs"]
    vis_inputs = inputs["vis_inputs"]
    ozone_inputs = inputs["ozone_inputs"]
    smoke_inputs = inputs["smoke_inputs"]
    accum_inputs = inputs["accum_inputs"]
    nearstorm_inputs = inputs["nearstorm_inputs"]
    station_pressure_inputs = inputs["station_pressure_inputs"]
    era5_rain_intensity = inputs["era5_rain_intensity"]
    era5_snow_water_equivalent = inputs["era5_snow_water_equivalent"]
    fire_inputs = inputs["fire_inputs"]
    feels_like_inputs = inputs["feels_like_inputs"]
    solar_inputs = inputs["solar_inputs"]
    cape_inputs = inputs["cape_inputs"]
    error_inputs = inputs["error_inputs"]

    # Generate the hourly forecast section
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
            InterPhour,
        ) = build_hourly_block(
            source_list=sourceList,
            InterPhour=InterPhour,
            hour_array_grib=hour_array_grib,
            hour_array=hour_array,
            InterSday=InterSday,
            hourlyDayIndex=hourlyDayIndex,
            baseTimeOffset=baseTimeOffset,
            timeMachine=timeMachine,
            tmExtra=tmExtra,
            prepIntensityUnit=prepIntensityUnit,
            prepAccumUnit=prepAccumUnit,
            windUnit=windUnit,
            visUnits=visUnits,
            tempUnits=tempUnits,
            humidUnit=humidUnit,
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
            station_pressure_inputs=station_pressure_inputs,
            era5_rain_intensity=era5_rain_intensity,
            era5_snow_water_equivalent=era5_snow_water_equivalent,
            fire_inputs=fire_inputs,
            feels_like_inputs=feels_like_inputs,
            solar_inputs=solar_inputs,
            cape_inputs=cape_inputs,
            error_inputs=error_inputs,
            version=version,
        )

    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])

    # Generate the daily forecast section
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

    # Process weather alerts for the location
    alertList = build_alerts(
        time_machine=timeMachine,
        ex_alerts=exAlerts,
        lat=lat,
        az_lon=az_Lon,
        nws_alerts_zarr=NWS_Alerts_Zarr,
        wmo_alert_data=WMO_alertDat,
        read_wmo_alerts=readWMOAlerts,
        logger=logger,
        loc_tag=loc_tag,
    )

    # Generate the current weather conditions section
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
    # Construct the final JSON response object
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
