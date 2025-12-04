"""Helpers for early request processing in the forecast handler."""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import reverse_geocode
from fastapi import HTTPException, Request
from pytz import timezone, utc
from timezonefinder import TimezoneFinder

from API.constants.api_const import COORDINATE_CONST, TIME_MACHINE_CONST
from API.constants.shared_const import KELVIN_TO_CELSIUS
from API.constants.unit_const import country_units
from API.hourly.builder import initialize_time_grids
from API.io.zarr_reader import WeatherParallel
from API.utils.geo import get_offset
from API.utils.timing import TimingTracker


def parse_request_time(
    time_str: str,
    now_time: datetime.datetime,
    lat: float,
    az_lon: float,
    tf: TimezoneFinder,
) -> datetime.datetime:
    """
    Parse the time string from the request URL.

    Handles:
    - Unix timestamps (positive and negative)
    - Relative time (seconds offset from now)
    - ISO 8601 strings (with and without timezone)
    - Local time strings (requires timezone lookup)
    """
    if time_str.lstrip("-+").isnumeric():
        val = float(time_str)
        if val > 0:
            return datetime.datetime.fromtimestamp(val, datetime.UTC).replace(
                tzinfo=None
            )
        elif val < TIME_MACHINE_CONST["very_negative_threshold"]:
            return datetime.datetime.fromtimestamp(val, datetime.UTC).replace(
                tzinfo=None
            )
        elif val < 0:
            return now_time + datetime.timedelta(seconds=val)

    # Try parsing as ISO format
    try:
        utc_time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S%z")
        return utc_time.replace(tzinfo=None)
    except Exception:
        pass

    try:
        utc_time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S%Z")
        return utc_time.replace(tzinfo=None)
    except Exception:
        pass

    # Try parsing as local time
    try:
        local_time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
        tz_offset_loc_in = {
            "lat": lat,
            "lng": az_lon,
            "utc_time": local_time,
            "tf": tf,
        }
        tz_offset_in, _ = get_offset(**tz_offset_loc_in)
        return local_time - datetime.timedelta(minutes=tz_offset_in)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Time Specification")


@dataclass
class InitialRequestContext:
    """Container for the light-weight request processing stage."""

    lat: float
    lon_in: float
    lon: float
    az_lon: float
    now_time: datetime.datetime
    utc_time: datetime.datetime
    time_machine: bool
    stage: str
    loc_tag: str
    timing_tracker: TimingTracker
    tz_offset: float
    tz_name: timezone
    tz_req: str
    loc_name: Dict[str, str]
    icon: str
    translation: dict
    extend_flag: int
    version: float
    tm_extra: bool
    exclude_params: str
    include_params: str
    extra_vars: List[str]
    ex_currently: int
    ex_minutely: int
    ex_hourly: int
    ex_daily: int
    ex_flags: int
    ex_alerts: int
    ex_nbm: int
    ex_hrrr: int
    ex_gefs: int
    ex_gfs: int
    ex_rtma_ru: int
    ex_ecmwf: int
    inc_day_night: int
    summary_text: bool
    unit_system: str
    wind_unit: float
    prep_intensity_unit: float
    prep_accum_unit: float
    temp_units: float
    vis_units: float
    humid_unit: float
    elev_unit: float
    read_wmo_alerts: bool
    weather: WeatherParallel
    pytz_tz: timezone
    base_time: datetime.datetime
    base_hour: datetime.datetime
    base_day: datetime.datetime
    base_day_utc: datetime.datetime
    base_day_utc_grib: int
    daily_days: int
    daily_day_hours: int
    output_hours: int
    output_days: int
    minute_array_grib: np.ndarray
    minute_array: np.ndarray
    inter_tminute: np.ndarray
    inter_pminute: np.ndarray
    inter_phour: np.ndarray
    hour_array_grib: np.ndarray
    hour_array: np.ndarray
    day_array_grib: np.ndarray
    num_hours: int


async def prepare_initial_request(
    *,
    request: Request,
    location: str,
    units: Union[str, None],
    extend: Union[str, None],
    exclude: Union[str, None],
    include: Union[str, None],
    lang: Union[str, None],
    version: Union[str, None],
    tmextra: Union[str, None],
    icon: Union[str, None],
    extraVars: Union[str, None],
    tf: TimezoneFinder,
    translations: dict,
    timing_enabled: bool,
    force_now: Union[str, bool, None],
    logger: logging.Logger,
    start_time: datetime.datetime,
) -> InitialRequestContext:
    """Run the initial request parsing and time grid setup."""
    stage = os.environ.get("STAGE", "PROD")

    if force_now is False:
        now_time = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    else:
        now_time = datetime.datetime.fromtimestamp(
            int(force_now), datetime.UTC
        ).replace(tzinfo=None)
        logger.info("Forced Current Time to:")
        logger.info(now_time)

    location_req = location.split(",")

    try:
        lat = float(location_req[0])
        lon_in = float(location_req[1])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Location Specification")

    lon = lon_in % COORDINATE_CONST["longitude_max"]  # 0-360
    az_lon = (
        (lon + COORDINATE_CONST["longitude_offset"]) % COORDINATE_CONST["longitude_max"]
    ) - COORDINATE_CONST["longitude_offset"]  # -180-180

    if (lon_in < COORDINATE_CONST["longitude_min"]) or (
        lon > COORDINATE_CONST["longitude_max"]
    ):
        raise HTTPException(status_code=400, detail="Invalid Longitude")
    if (lat < COORDINATE_CONST["latitude_min"]) or (
        lat > COORDINATE_CONST["latitude_max"]
    ):
        raise HTTPException(status_code=400, detail="Invalid Latitude")

    loc_tag = f"[loc={lat:.4f},{az_lon:.4f}]"
    timing_tracker = TimingTracker(
        logger=logger,
        enabled=timing_enabled,
        prefix=f"{loc_tag} ",
    )

    if len(location_req) == 2:
        if stage == "TIMEMACHINE":
            raise HTTPException(status_code=400, detail="Missing Time Specification")
        utc_time = now_time
    elif len(location_req) == 3:
        utc_time = parse_request_time(location_req[2], now_time, lat, az_lon, tf)
    else:
        raise HTTPException(
            status_code=400, detail="Invalid Time or Location Specification"
        )

    time_machine = False
    if not lang:
        lang = "en"

    if icon != "pirate":
        icon = "darksky"

    if lang not in translations:
        raise HTTPException(status_code=400, detail="Language Not Supported")

    translation = translations[lang]

    if (now_time - utc_time) > datetime.timedelta(hours=25):
        if (
            ("localhost" in str(request.url))
            or ("timemachine" in str(request.url))
            or ("127.0.0.1" in str(request.url))
            or ("dev" in str(request.url))
        ):
            time_machine = True
        else:
            raise HTTPException(
                status_code=400,
                detail="Requested Time is in the Past. Please Use Timemachine.",
            )
    elif now_time < utc_time:
        if (utc_time - now_time) < datetime.timedelta(hours=1):
            utc_time = now_time
        else:
            raise HTTPException(
                status_code=400, detail="Requested Time is in the Future"
            )
    elif (now_time - utc_time) < datetime.timedelta(
        hours=TIME_MACHINE_CONST["threshold_hours"]
    ):
        if "timemachine" in str(request.url):
            time_machine = True
            if timing_enabled:
                logger.debug("Near term timemachine request")
                logger.debug(now_time - utc_time)

    if timing_enabled:
        logger.debug("Request process time")
        logger.debug(
            datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - start_time
        )

    tz_offset_loc = {"lat": lat, "lng": az_lon, "utc_time": utc_time, "tf": tf}
    tz_offset, tz_name = get_offset(**tz_offset_loc)
    tz_req = tf.timezone_at(lat=lat, lng=az_lon)
    loc_name = await asyncio.to_thread(reverse_geocode.get, (lat, az_lon))

    extend_flag = 0 if not extend else int(extend == "hourly")
    version_val = float(version) if version else 1.0
    tm_extra = bool(tmextra)
    exclude_params = exclude or ""
    include_params = include or ""
    extra_vars = extraVars.split(",") if extraVars else []

    ex_currently = int("currently" in exclude_params)
    ex_minutely = int("minutely" in exclude_params)
    ex_hourly = int("hourly" in exclude_params)
    ex_daily = int("daily" in exclude_params)
    ex_flags = int("flags" in exclude_params)
    ex_alerts = int("alerts" in exclude_params)
    ex_nbm = int("nbm" in exclude_params)
    ex_hrrr = int("hrrr" in exclude_params)
    ex_gefs = int("gefs" in exclude_params)
    ex_gfs = int("gfs" in exclude_params)
    ex_rtma_ru = int("rtma_ru" in exclude_params)
    ex_ecmwf = int("ecmwf_ifs" in exclude_params)
    summary_text = "summary" not in exclude_params
    inc_day_night = int("day_night_forecast" in include_params)

    if (now_time - utc_time) > datetime.timedelta(hours=25):
        ex_nbm = 1
        ex_alerts = 1
        ex_hrrr = 1
        ex_gefs = 1
        ex_rtma_ru = 1
        ex_ecmwf = 1

    read_wmo_alerts = not (time_machine or ex_alerts == 1)

    if time_machine and not tm_extra:
        ex_minutely = 1
    if time_machine:
        ex_alerts = 1

    unit_system = "us"
    # Default US units
    unit_config = {
        "wind_unit": 2.234,
        "prep_intensity_unit": 0.0394,
        "prep_accum_unit": 0.0394,
        "temp_units": 0,
        "vis_units": 0.00062137,
        "humid_unit": 0.01,
        "elev_unit": 3.28084,
    }

    if units:
        if units == "auto":
            unit_system = country_units.get(loc_name["country_code"], "us").lower()
        else:
            unit_system = units[0:2]

        # Define unit overrides
        unit_overrides = {
            "ca": {
                "wind_unit": 3.600,
                "prep_intensity_unit": 1,
                "prep_accum_unit": 0.1,
                "temp_units": KELVIN_TO_CELSIUS,
                "vis_units": 0.001,
                "elev_unit": 1,
            },
            "uk": {
                "wind_unit": 2.234,
                "prep_intensity_unit": 1,
                "prep_accum_unit": 0.1,
                "temp_units": KELVIN_TO_CELSIUS,
                "vis_units": 0.00062137,
                "elev_unit": 1,
            },
            "si": {
                "wind_unit": 1,
                "prep_intensity_unit": 1,
                "prep_accum_unit": 0.1,
                "temp_units": KELVIN_TO_CELSIUS,
                "vis_units": 0.001,
                "elev_unit": 1,
            },
        }

        if unit_system in unit_overrides:
            unit_config.update(unit_overrides[unit_system])
        else:
            unit_system = "us"

    wind_unit = unit_config["wind_unit"]
    prep_intensity_unit = unit_config["prep_intensity_unit"]
    prep_accum_unit = unit_config["prep_accum_unit"]
    temp_units = unit_config["temp_units"]
    vis_units = unit_config["vis_units"]
    humid_unit = unit_config["humid_unit"]
    elev_unit = unit_config["elev_unit"]

    weather = WeatherParallel(loc_tag=loc_tag)
    pytz_tz = timezone(tz_req)

    base_time = utc.localize(
        datetime.datetime(
            year=utc_time.year,
            month=utc_time.month,
            day=utc_time.day,
            hour=utc_time.hour,
            minute=utc_time.minute,
        )
    ).astimezone(pytz_tz)
    base_hour = pytz_tz.localize(
        datetime.datetime(
            year=base_time.year,
            month=base_time.month,
            day=base_time.day,
            hour=base_time.hour,
        )
    )
    base_day = base_time.replace(hour=0, minute=0, second=0, microsecond=0)
    base_day_utc = base_day.astimezone(utc)
    base_day_utc_grib = (
        (
            np.datetime64(base_day.astimezone(utc).replace(tzinfo=None))
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
        )
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    if time_machine:
        daily_days = 1
        daily_day_hours = 1
        output_hours = 24
        output_days = 1
    else:
        daily_days = 8
        daily_day_hours = 5
        output_hours = 168 if extend_flag else 48
        output_days = 8

    (
        minute_array_grib,
        minute_array,
        inter_tminute,
        inter_pminute,
        inter_phour,
        hour_array_grib,
        hour_array,
        day_array_grib,
    ) = initialize_time_grids(
        base_time=base_time,
        base_day=base_day,
        daily_days=daily_days,
        daily_day_hours=daily_day_hours,
        timezone_localizer=pytz_tz,
    )

    num_hours = len(hour_array)

    return InitialRequestContext(
        lat=lat,
        lon_in=lon_in,
        lon=lon,
        az_lon=az_lon,
        now_time=now_time,
        utc_time=utc_time,
        time_machine=time_machine,
        stage=stage,
        loc_tag=loc_tag,
        timing_tracker=timing_tracker,
        tz_offset=tz_offset,
        tz_name=tz_name,
        tz_req=tz_req,
        loc_name=loc_name,
        icon=icon,
        translation=translation,
        extend_flag=extend_flag,
        version=version_val,
        tm_extra=tm_extra,
        exclude_params=exclude_params,
        include_params=include_params,
        extra_vars=extra_vars,
        ex_currently=ex_currently,
        ex_minutely=ex_minutely,
        ex_hourly=ex_hourly,
        ex_daily=ex_daily,
        ex_flags=ex_flags,
        ex_alerts=ex_alerts,
        ex_nbm=ex_nbm,
        ex_hrrr=ex_hrrr,
        ex_gefs=ex_gefs,
        ex_gfs=ex_gfs,
        ex_rtma_ru=ex_rtma_ru,
        ex_ecmwf=ex_ecmwf,
        inc_day_night=inc_day_night,
        summary_text=summary_text,
        unit_system=unit_system,
        wind_unit=wind_unit,
        prep_intensity_unit=prep_intensity_unit,
        prep_accum_unit=prep_accum_unit,
        temp_units=temp_units,
        vis_units=vis_units,
        humid_unit=humid_unit,
        elev_unit=elev_unit,
        read_wmo_alerts=read_wmo_alerts,
        weather=weather,
        pytz_tz=pytz_tz,
        base_time=base_time,
        base_hour=base_hour,
        base_day=base_day,
        base_day_utc=base_day_utc,
        base_day_utc_grib=base_day_utc_grib,
        daily_days=daily_days,
        daily_day_hours=daily_day_hours,
        output_hours=output_hours,
        output_days=output_days,
        minute_array_grib=minute_array_grib,
        minute_array=minute_array,
        inter_tminute=inter_tminute,
        inter_pminute=inter_pminute,
        inter_phour=inter_phour,
        hour_array_grib=hour_array_grib,
        hour_array=hour_array,
        day_array_grib=day_array_grib,
        num_hours=num_hours,
    )
