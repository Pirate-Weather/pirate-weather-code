import asyncio
import datetime
import logging
import math
import os
import platform
from typing import Union

import numpy as np
import zarr
from astral import LocationInfo, moon
from astral.sun import sun
from dateutil.relativedelta import relativedelta
from fastapi import HTTPException
from fastapi.responses import ORJSONResponse
from PirateSimpleDayText import calculate_simple_day_text
from PirateText import calculate_text
from pirateweather_translations.dynamic_loader import load_all_translations
from pytz import timezone, utc

from API.constants.api_const import SOLAR_RAD_CONST
from API.constants.forecast_const import DATA_TIMEMACHINE
from API.constants.shared_const import KELVIN_TO_CELSIUS
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    WIND_THRESHOLDS,
)
from API.constants.timemachine_const import (
    APPARENT_TEMP_WINDCHILL_CONST,
    DAILY_PRECIP_THRESHOLD,
    ICE_ACCUMULATION,
)

Translations = load_all_translations()


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
    radLat = np.deg2rad(lat)
    solarHour = math.pi * (
        (t_t - SOLAR_RAD_CONST["hour_offset"]) / SOLAR_RAD_CONST["hour_offset"]
    )
    cosTheta = math.sin(delta) * math.sin(radLat) + math.cos(delta) * math.cos(
        radLat
    ) * math.cos(solarHour)
    R_s = r * (S_0 / d**2) * cosTheta

    if R_s < 0:
        R_s = 0

    return R_s


def get_offset(*, lat, lng, utcTime, tf):
    """
    returns a location's time zone offset from UTC in minutes.
    """

    today = utcTime
    tz_target = timezone(tf.timezone_at(lng=lng, lat=lat))
    # ATTENTION: tz_target could be None! handle error case
    today_target = tz_target.localize(today)
    today_utc = utc.localize(today)
    return (today_utc - today_target).total_seconds() / 60, tz_target


def toTimestamp(d):
    return d.timestamp()


def round_time(dt=None, round_to=60):
    if dt is None:
        dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def find_nearest(array, value):
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array) - 1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx - 1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx - 1]
        else:
            idx_nearest = idx_sorted[idx]
    return idx_nearest


def x_round(x):
    rounded = round(x * 4) / 4
    return rounded


kerchunkERA5Dir = os.environ.get("ERADIR", "/efs/kerchunk/ERA5_V4/")


async def TimeMachine(
    lat: float,
    lon: float,
    az_Lon: float,
    utcTime: int,
    tf,
    units: Union[str, None] = None,
    exclude: Union[str, None] = None,
    lang: Union[str, None] = None,
    apiVersion: Union[str, None] = None,
) -> dict:
    kerchunkERA5Dir = os.environ.get("ERADIR", "/efs/kerchunk/ERA5_V4/")

    logging.info("Starting ERA5 Request")

    T_Start = datetime.datetime.utcnow()

    baseTime = utcTime

    # Calculate the timezone offset
    tz_offsetLoc = {"lat": lat, "lng": az_Lon, "utcTime": utcTime, "tf": tf}
    tz_offset, tz_name = get_offset(**tz_offsetLoc)

    # Default to US
    unitSystem = "us"
    windUnit = 2.234  # mph
    prepIntensityUnit = 0.0394  # inches/hour
    prepAccumUnit = 0.0394  # inches
    tempUnits = 0  # F. This is harder
    pressUnits = 0.01  # Hectopascals
    visUnits = 0.00062137  # miles
    # humidUnit = 0.01  # %
    # elevUnit = 3.28084  # ft

    if units:
        unitSystem = units[0:2]

        if unitSystem == "ca":
            windUnit = 3.600  # kph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km
            # humidUnit = 0.01  # %
            # elevUnit = 1  # m
        elif unitSystem == "uk":
            windUnit = 2.234  # mph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.00062137  # miles
            # humidUnit = 0.01  # %
            # elevUnit = 1  # m
        elif unitSystem == "si":
            windUnit = 1  # m/s
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km

    if not exclude:
        excludeParams = ""
    else:
        excludeParams = exclude

    # Check if language is supported
    if lang is None:
        lang = "en"  # Default to English

    if lang not in Translations:
        # Throw an error
        raise HTTPException(status_code=400, detail="Language Not Supported")

    translation = Translations[lang]

    exCurrently = 0
    exHourly = 0
    exDaily = 0
    exFlags = 0

    if "currently" in excludeParams:
        exCurrently = 1
    if "hourly" in excludeParams:
        exHourly = 1
    if "daily" in excludeParams:
        exDaily = 1
    if "flags" in excludeParams:
        exFlags = 1

    varList_inst = ["2t", "10u", "10v", "2d", "msl", "tcc"]
    varList_accum = ["lsp", "cp", "sf"]  # Large Scale Precp, Convective Precp, Snow
    varList_rate = [
        "i10fg"
    ]  # Large scale instant rate, convective instant rate, wing guest
    # weatherVars = varList_inst + varList_accum + varList_rate

    # dataOut = np.zeros(shape=(24, 8))
    # weatherVarCount = 1

    baseTimeLocal = utcTime + datetime.timedelta(minutes=tz_offset)

    # Adjust half hour time zones
    if baseTimeLocal.minute == 30:
        baseTimeLocal = baseTimeLocal - datetime.timedelta(minutes=30)
        halfTZ = 30
    else:
        halfTZ = 0

    # Midnight local in UTC
    baseDayLocalMN = datetime.datetime(
        year=baseTimeLocal.year, month=baseTimeLocal.month, day=baseTimeLocal.day
    ) - datetime.timedelta(minutes=tz_offset)

    # ERA5 File
    instantFile = datetime.datetime(
        year=baseDayLocalMN.year, month=baseDayLocalMN.month, day=1
    )

    # Index
    lats_era = np.arange(90, -90, -0.25)
    lons_era = np.arange(0, 360, 0.25)

    abslat = np.abs(lats_era - lat)
    abslon = np.abs(lons_era - lon)
    x = np.argmin(abslat)
    y = np.argmin(abslon)

    # era_lat = lats_era[x]
    # era_lon = lons_era[y]

    dataDict = dict()

    async def kerchunkRead(kerchunkERA5Dir, instantFile, v, x, y, tIDX_start, tIDX_end):
        if v[0].isnumeric():
            varName = "VAR_" + v.upper()
        else:
            varName = v.upper()

        if tIDX_end == 0:
            dataOut = await asyncio.to_thread(
                lambda: zarr.open(
                    "reference://",
                    storage_options={
                        "fo": kerchunkERA5Dir
                        + instantFile.strftime("%Y%m")
                        + "/"
                        + v
                        + "_"
                        + instantFile.strftime("%Y%m%d%H")
                        + ".parq",
                        "remote_protocol": "s3",
                        "remote_options": {"anon": True},
                        "asynchronous": False,
                    },
                )[varName][tIDX_start:, x, y]
            )
        else:
            dataOut = await asyncio.to_thread(
                lambda: zarr.open(
                    "reference://",
                    storage_options={
                        "fo": kerchunkERA5Dir
                        + instantFile.strftime("%Y%m")
                        + "/"
                        + v
                        + "_"
                        + instantFile.strftime("%Y%m%d%H")
                        + ".parq",
                        "remote_protocol": "s3",
                        "remote_options": {"anon": True},
                        "asynchronous": False,
                    },
                )[varName][tIDX_start:tIDX_end, x, y]
            )
        return dataOut

    async def kerchunkReadAccum(
        kerchunkERA5Dir, instantFile, v, x, y, tIDX_start, tIDX_end
    ):
        if v[0].isnumeric():
            varName = "VAR_" + v.upper()
        else:
            varName = v.upper()

        if tIDX_end == 0:
            dataOut = await asyncio.to_thread(
                lambda: zarr.open(
                    "reference://",
                    storage_options={
                        "fo": kerchunkERA5Dir
                        + instantFile.strftime("%Y%m")
                        + "/"
                        + v
                        + "_"
                        + instantFile.strftime("%Y%m%d%H")
                        + ".parq",
                        "remote_protocol": "s3",
                        "remote_options": {"anon": True},
                        "asynchronous": False,
                    },
                )[varName][tIDX_start:, :, x, y]
            )
        else:
            dataOut = await asyncio.to_thread(
                lambda: zarr.open(
                    "reference://",
                    storage_options={
                        "fo": kerchunkERA5Dir
                        + instantFile.strftime("%Y%m")
                        + "/"
                        + v
                        + "_"
                        + instantFile.strftime("%Y%m%d%H")
                        + ".parq",
                        "remote_protocol": "s3",
                        "remote_options": {"anon": True},
                        "asynchronous": False,
                    },
                )[varName][tIDX_start:tIDX_end, :, x, y]
            )
        return dataOut

    ####### Instant Vars
    # If requesting the last timestep in a file, also need the following one
    if (baseDayLocalMN + datetime.timedelta(days=1)).month == baseDayLocalMN.month:
        tasks = dict()
        for v in varList_inst:
            tIDX_start = int((baseDayLocalMN - instantFile).total_seconds() / 3600)
            tIDX_end = int(
                (
                    baseDayLocalMN + datetime.timedelta(days=1) - instantFile
                ).total_seconds()
                / 3600
            )

            tasks["VAR_" + v] = kerchunkRead(
                kerchunkERA5Dir, instantFile, v, x, y, tIDX_start, tIDX_end
            )

        results = await asyncio.gather(*tasks.values())

        # Create a dictionary to match keys to results
        dataDict = {key: result for key, result in zip(tasks.keys(), results)}

    else:
        tasks = dict()
        for v in varList_inst:
            instantFile_b = instantFile + relativedelta(months=1)

            tIDX_a_start = int((baseDayLocalMN - instantFile).total_seconds() / 3600)
            tIDX_b_end = int(
                (
                    baseDayLocalMN + datetime.timedelta(days=1) - instantFile_b
                ).total_seconds()
                / 3600
            )

            tasks["A_VAR_" + v] = kerchunkRead(
                kerchunkERA5Dir, instantFile, v, x, y, tIDX_a_start, 0
            )
            tasks["B_VAR_" + v] = kerchunkRead(
                kerchunkERA5Dir, instantFile_b, v, x, y, 0, tIDX_b_end
            )

        results = await asyncio.gather(*tasks.values())
        resultsDict = {key: result for key, result in zip(tasks.keys(), results)}

        for v in varList_inst:
            dataDict["VAR_" + v] = np.hstack(
                (resultsDict["A_VAR_" + v], resultsDict["B_VAR_" + v])
            )

    #### Accum/ Flux
    tasks = dict()
    for v in varList_accum + varList_rate:
        if ((baseDayLocalMN.day == 16) & (baseDayLocalMN.hour < 6)) | (
            baseDayLocalMN.day < 16
        ):
            if (baseDayLocalMN.day == 1) & (baseDayLocalMN.hour < 6):
                if baseDayLocalMN.month == 1:
                    accumFile = datetime.datetime(
                        year=baseDayLocalMN.year - 1,
                        month=12,
                        day=16,
                        hour=6,
                        minute=0,
                        second=0,
                    )
                else:
                    accumFile = datetime.datetime(
                        year=baseDayLocalMN.year,
                        month=baseDayLocalMN.month - 1,
                        day=16,
                        hour=6,
                        minute=0,
                        second=0,
                    )
            else:
                accumFile = datetime.datetime(
                    year=baseDayLocalMN.year,
                    month=baseDayLocalMN.month,
                    day=1,
                    hour=6,
                    minute=0,
                    second=0,
                )

        else:
            accumFile = datetime.datetime(
                year=baseDayLocalMN.year,
                month=baseDayLocalMN.month,
                day=16,
                hour=6,
                minute=0,
                second=0,
            )

        tIDX_start = int(
            np.floor((baseDayLocalMN - accumFile).total_seconds() / 3600) / 12
        )
        tStep_start = int(
            np.floor((baseDayLocalMN - accumFile).total_seconds() / 3600) % 12
        )

        numHours = int(
            (
                (baseDayLocalMN + datetime.timedelta(days=1)) - baseDayLocalMN
            ).total_seconds()
            / 3600
        )

        if v == varList_accum[0]:
            f = (
                kerchunkERA5Dir
                + accumFile.strftime("%Y%m")
                + "/"
                + v
                + "_"
                + accumFile.strftime("%Y%m%d%H")
                + ".parq"
            )

            ds = zarr.open(
                "reference://",
                storage_options={
                    "fo": f,
                    "remote_protocol": "s3",
                    "remote_options": {"anon": True},
                    "asynchronous": False,
                },
            )

        # If end of file
        if tIDX_start < (ds[varList_accum[0].upper()].shape[0] - 2):
            tasks["VAR_" + v] = kerchunkReadAccum(
                kerchunkERA5Dir, accumFile, v, x, y, tIDX_start, tIDX_start + 3
            )
        else:
            tasks["A_VAR_" + v] = kerchunkReadAccum(
                kerchunkERA5Dir, accumFile, v, x, y, tIDX_start, 0
            )

            baseDayLocalMN_b = baseDayLocalMN + datetime.timedelta(days=1)

            if ((baseDayLocalMN_b.day == 16) & (baseDayLocalMN_b.hour < 6)) | (
                baseDayLocalMN_b.day < 16
            ):
                if (baseDayLocalMN_b.day == 1) & (baseDayLocalMN_b.hour < 6):
                    if baseDayLocalMN.month == 1:
                        accumFile_b = datetime.datetime(
                            year=baseDayLocalMN.year - 1,
                            month=12,
                            day=16,
                            hour=6,
                            minute=0,
                            second=0,
                        )
                    else:
                        accumFile_b = datetime.datetime(
                            year=baseDayLocalMN.year,
                            month=baseDayLocalMN.month - 1,
                            day=16,
                            hour=6,
                            minute=0,
                            second=0,
                        )
                else:
                    accumFile_b = datetime.datetime(
                        year=baseDayLocalMN_b.year,
                        month=baseDayLocalMN_b.month,
                        day=1,
                        hour=6,
                        minute=0,
                        second=0,
                    )

            else:
                accumFile_b = datetime.datetime(
                    year=baseDayLocalMN_b.year,
                    month=baseDayLocalMN_b.month,
                    day=16,
                    hour=6,
                    minute=0,
                    second=0,
                )

            tIDX_end = int(
                np.floor(
                    (
                        baseDayLocalMN
                        + datetime.timedelta(days=1)
                        + datetime.timedelta(hours=12)
                        - accumFile_b
                    ).total_seconds()
                    / 3600
                )
                / 12
            )
            tStep_end = int(
                (
                    (baseDayLocalMN + datetime.timedelta(days=1)) - baseDayLocalMN
                ).total_seconds()
                / 3600
            )

            tasks["B_VAR_" + v] = kerchunkReadAccum(
                kerchunkERA5Dir, accumFile, v, x, y, 0, tIDX_end
            )

        dataDict["hours"] = np.arange(
            baseDayLocalMN,
            baseDayLocalMN + datetime.timedelta(days=1),
            datetime.timedelta(hours=1),
        )

    # Await all the tasks and store the results in the same dictionary
    results = await asyncio.gather(*tasks.values())

    # Create a dictionary to match keys to results
    zarr_results = {key: result for key, result in zip(tasks.keys(), results)}

    for v in varList_accum + varList_rate:
        if tIDX_start < (ds[varList_accum[0].upper()].shape[0] - 2):
            data_a = zarr_results["VAR_" + v].flatten()

            dataDict["VAR_" + v] = data_a[tStep_start : tStep_start + numHours]
        else:
            data_a = zarr_results["A_VAR_" + v].flatten()
            data_b = zarr_results["B_VAR_" + v].flatten()

            dataDict["VAR_" + v] = np.hstack(
                (data_a[tStep_start:], data_b[: tStep_end - (12 - tStep_start)])
            )

    ### Hourly convert to list ####
    hourList = []
    pTypeList = []
    pTextList = []
    hTextList = []
    pIconList = []
    InterPhour = np.zeros(shape=(len(dataDict["hours"]), 16))
    dayRainAccum = 0
    daySnowAccum = 0

    InterPhour[:, 0] = (
        (dataDict["hours"] - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    for idx in range(0, len(dataDict["hours"]), 1):
        if dataDict["VAR_lsp"][idx] + dataDict["VAR_cp"][idx] == 0:
            pTypeList.append("none")
            pTextList.append("None")
            pFac = 0
        else:
            if (dataDict["VAR_lsp"][idx] + dataDict["VAR_cp"][idx]) * 0.5 > dataDict[
                "VAR_sf"
            ][idx]:
                pTypeList.append("rain")
                pTextList.append("Rain")
                pFac = 1000  # Units in m, convert to mm
            else:
                pTypeList.append("snow")
                pTextList.append("Snow")
                pFac = 10000  # Units in m, convert to mm * 10 snow/water ratio

        ## Add Temperature
        InterPhour[idx, DATA_TIMEMACHINE["temp"]] = dataDict["VAR_2t"][idx]
        ## Add Precip
        InterPhour[idx, DATA_TIMEMACHINE["precip"]] = (
            dataDict["VAR_lsp"][idx] + dataDict["VAR_cp"][idx]
        ) * pFac
        ## Add Dew Point
        InterPhour[idx, DATA_TIMEMACHINE["dew"]] = dataDict["VAR_2d"][idx]
        # Pressure
        InterPhour[idx, DATA_TIMEMACHINE["pressure"]] = dataDict["VAR_msl"][idx]
        ## Add wind speed
        InterPhour[idx, DATA_TIMEMACHINE["wind"]] = np.sqrt(
            dataDict["VAR_10u"][idx] ** 2 + dataDict["VAR_10v"][idx] ** 2
        )
        # Add Wind Bearing
        InterPhour[idx, DATA_TIMEMACHINE["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(dataDict["VAR_10u"][idx], dataDict["VAR_10v"][idx]) + np.pi,
                2 * np.pi,
            )
        )
        # Add Cloud Cover
        InterPhour[idx, DATA_TIMEMACHINE["cloud"]] = dataDict["VAR_tcc"][idx]
        # Add Snow
        InterPhour[idx, DATA_TIMEMACHINE["snow"]] = dataDict["VAR_sf"][idx] * 10000
        # Add Wind Gust
        InterPhour[idx, DATA_TIMEMACHINE["gust"]] = dataDict["VAR_i10fg"][idx]

        # Add Apparent Temperature based on https://en.wikipedia.org/wiki/Wind_chill
        wc = APPARENT_TEMP_WINDCHILL_CONST
        if dataDict["VAR_2t"][idx] < wc["threshold_k"]:
            # Convert to C, then back to K
            InterPhour[idx, DATA_TIMEMACHINE["apparent"]] = (
                wc["windchill_1"]
                + wc["windchill_2"] * (dataDict["VAR_2t"][idx] - KELVIN_TO_CELSIUS)
                - wc["windchill_3"]
                * (InterPhour[idx, DATA_TIMEMACHINE["wind"]] * wc["windchill_kph_conv"])
                ** wc["windchill_exp"]
                + wc["windchill_4"]
                * (dataDict["VAR_2t"][idx] - KELVIN_TO_CELSIUS)
                * (InterPhour[idx, DATA_TIMEMACHINE["wind"]] * wc["windchill_kph_conv"])
                ** wc["windchill_exp"]
            )
        else:
            InterPhour[idx, DATA_TIMEMACHINE["apparent"]] = (
                dataDict["VAR_2t"][idx] - KELVIN_TO_CELSIUS
            ) + wc["apparent_temp_const"] * (
                wc["apparent_temp_2"]
                * math.exp(
                    wc["apparent_temp_3"]
                    * (
                        (1 / wc["apparent_temp_4"])
                        - (
                            1
                            / (
                                KELVIN_TO_CELSIUS
                                + (dataDict["VAR_2d"][idx] - KELVIN_TO_CELSIUS)
                            )
                        )
                    )
                )
                - wc["apparent_temp_5"]
            )

        InterPhour[idx, DATA_TIMEMACHINE["apparent"]] = (
            InterPhour[idx, DATA_TIMEMACHINE["apparent"]] + KELVIN_TO_CELSIUS
        )

    # Put temperature into units
    if tempUnits == 0:
        for k in [
            DATA_TIMEMACHINE["temp"],
            DATA_TIMEMACHINE["apparent"],
            DATA_TIMEMACHINE["dew"],
        ]:
            InterPhour[:, k] = (InterPhour[:, k] - KELVIN_TO_CELSIUS) * 9 / 5 + 32
    else:
        for k in [
            DATA_TIMEMACHINE["temp"],
            DATA_TIMEMACHINE["apparent"],
            DATA_TIMEMACHINE["dew"],
        ]:
            InterPhour[:, k] = InterPhour[:, k] - tempUnits

    ## Daily setup
    baseDay = datetime.datetime(
        year=baseTime.year, month=baseTime.month, day=baseTime.day
    )
    InterPday = np.zeros(shape=(19, 1))
    InterPdayMax = np.zeros(shape=(16, 1))
    InterPdayMaxTime = np.zeros(shape=(16, 1))
    InterPdayMin = np.zeros(shape=(16, 1))
    InterPdayMinTime = np.zeros(shape=(16, 1))
    InterPdaySum = np.zeros(shape=(16, 1))

    ## Sunrise sunset
    loc = LocationInfo("name", "region", tz_name, lat, lon - 360)
    s = sun(loc.observer, date=baseDay, tzinfo=tz_name)
    m = moon.phase(baseDay)

    InterPday[16, 0] = (
        (
            np.datetime64(s["sunrise"])
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
        )
        .astype("timedelta64[s]")
        .astype(np.int32)
    )
    InterPday[17, 0] = (
        (
            np.datetime64(s["sunset"])
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
        )
        .astype("timedelta64[s]")
        .astype(np.int32)
    )
    InterPday[18, 0] = m / 27.99

    for idx in range(0, len(dataDict["hours"]), 1):
        # Calculate type-based accumulation for text summaries
        hourRainAccum = 0
        hourSnowAccum = 0
        if pTypeList[idx] == "snow":
            hourSnowAccum += (
                InterPhour[idx, DATA_TIMEMACHINE["intensity"]] * prepAccumUnit
            )
        else:
            hourRainAccum += (
                InterPhour[idx, DATA_TIMEMACHINE["intensity"]] * prepAccumUnit
            )

        dayRainAccum += hourRainAccum
        daySnowAccum += hourSnowAccum

        # Check if day or night
        sunrise_ts = InterPday[DATA_TIMEMACHINE["sunrise"], 0]
        sunset_ts = InterPday[DATA_TIMEMACHINE["sunset"], 0]
        isDay = sunrise_ts <= InterPhour[idx, 0] <= sunset_ts

        ## Icon
        if (
            InterPhour[idx, DATA_TIMEMACHINE["intensity"]]
            > HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM * prepIntensityUnit
        ):
            pIconList.append(pTypeList[idx])
            hourText = pTextList[idx]
        elif (
            InterPhour[idx, DATA_TIMEMACHINE["wind"]]
            > WIND_THRESHOLDS["light"] * windUnit
        ):
            pIconList.append("wind")
            hourText = "Windy"
        elif (
            InterPhour[idx, DATA_TIMEMACHINE["cloud"]]
            > CLOUD_COVER_THRESHOLDS["cloudy"]
        ):
            pIconList.append("cloudy")
            hourText = "Cloudy"
        elif (
            InterPhour[idx, DATA_TIMEMACHINE["cloud"]]
            > CLOUD_COVER_THRESHOLDS["partly_cloudy"]
        ):
            hourText = "Partly Cloudy"
            if isDay:
                # Before sunrise
                pIconList.append("partly-cloudy-day")
            else:  # After sunset
                pIconList.append("partly-cloudy-night")
        else:
            hourText = "Clear"
            if isDay:
                # Before sunrise
                pIconList.append("clear-day")
            else:  # After sunset
                pIconList.append("clear-night")

        hTextList.append(hourText)
        hourDict = {
            "time": int(InterPhour[idx, DATA_TIMEMACHINE["time"]]) + halfTZ,
            "summary": hourText,
            "icon": pIconList[idx],
            "precipIntensity": round(
                InterPhour[idx, DATA_TIMEMACHINE["intensity"]] * prepIntensityUnit, 4
            ),
            "precipAccumulation": round(
                InterPhour[idx, DATA_TIMEMACHINE["intensity"]] * prepAccumUnit, 4
            ),
            "precipType": pTypeList[idx],
            "temperature": round(InterPhour[idx, DATA_TIMEMACHINE["temp"]], 2),
            "apparentTemperature": round(
                InterPhour[idx, DATA_TIMEMACHINE["apparent"]], 2
            ),
            "dewPoint": round(InterPhour[idx, DATA_TIMEMACHINE["dew"]], 2),
            "pressure": round(
                InterPhour[idx, DATA_TIMEMACHINE["pressure"]] * pressUnits, 2
            ),
            "windSpeed": round(InterPhour[idx, DATA_TIMEMACHINE["wind"]] * windUnit, 2),
            "windGust": round(InterPhour[idx, DATA_TIMEMACHINE["gust"]] * windUnit, 2),
            "windBearing": int(round(InterPhour[idx, DATA_TIMEMACHINE["bearing"]], 0)),
            "cloudCover": round(InterPhour[idx, DATA_TIMEMACHINE["cloud"]], 2),
            "snowAccumulation": round(
                InterPhour[idx, DATA_TIMEMACHINE["snow"]] * prepAccumUnit, 2
            ),
        }

        try:
            precip_intensity = InterPhour[idx, 1] * prepIntensityUnit
            hourText, hourIcon = calculate_text(
                hourDict,
                prepAccumUnit,
                visUnits,
                windUnit,
                tempUnits,
                isDay,
                hourRainAccum,
                hourSnowAccum,
                ICE_ACCUMULATION,
                "hour",
                precip_intensity,
            )

            hourDict["summary"] = translation.translate(["title", hourText])
            hourDict["icon"] = hourIcon

        except Exception:
            logging.exception("HOURLY TEXT GEN ERROR:")

        hourList.append(dict(hourDict))

    # Find daily averages/max/min/times
    for j in [
        DATA_TIMEMACHINE["intensity"],
        DATA_TIMEMACHINE["temp"],
        DATA_TIMEMACHINE["apparent"],
        DATA_TIMEMACHINE["dew"],
        DATA_TIMEMACHINE["pressure"],
        DATA_TIMEMACHINE["wind"],
        DATA_TIMEMACHINE["bearing"],
        DATA_TIMEMACHINE["cloud"],
        DATA_TIMEMACHINE["snow"],
        DATA_TIMEMACHINE["gust"],
    ]:
        InterPday[j, 0] = np.average(InterPhour[:, j])
        InterPdayMax[j, 0] = np.amax(InterPhour[:, j])
        InterPdayMaxTime[j, 0] = np.argmax(InterPhour[:, j])
        InterPdayMin[j, 0] = np.amin(InterPhour[:, j])
        InterPdayMinTime[j, 0] = np.argmin(InterPhour[:, j])
        InterPdaySum[j, 0] = np.sum(InterPhour[:, j])

    # Daily precipitation type
    if InterPdaySum[DATA_TIMEMACHINE["intensity"], 0] > 0:
        if (
            InterPdaySum[DATA_TIMEMACHINE["intensity"], 0] * 0.5
            > InterPdaySum[DATA_TIMEMACHINE["snow"], 0]
        ):
            # Rain
            maxPchanceDay = 1
        else:
            # Snow
            maxPchanceDay = 2
    else:
        maxPchanceDay = 0

    # Convert to list
    dayList = []
    pTypeListDay = []
    pTextListDay = []
    dayIconList = []
    idx = 0
    if InterPdaySum[DATA_TIMEMACHINE["intensity"], 0] == 0:
        pTypeListDay.append("none")
        pTextListDay.append("None")
    elif maxPchanceDay == 1:
        pTypeListDay.append("rain")
        pTextListDay.append("Rain")
    elif maxPchanceDay == 2:
        pTypeListDay.append("snow")
        pTextListDay.append("Snow")
    else:
        pTypeListDay.append("none")
        pTextListDay.append("None")

    if (
        InterPdaySum[DATA_TIMEMACHINE["intensity"], 0]
        > DAILY_PRECIP_THRESHOLD * prepAccumUnit
    ):
        # If more than 0.5 mm of precip at any throughout the day, then the icon for whatever is happening
        pIcon = pTypeListDay[idx]
        pText = pTextListDay[idx]
    elif (
        InterPday[DATA_TIMEMACHINE["intensity"], idx]
        > WIND_THRESHOLDS["light"] * windUnit
    ):
        pIcon = "wind"
        pText = "Windy"
    elif InterPday[DATA_TIMEMACHINE["cloud"], idx] > CLOUD_COVER_THRESHOLDS["cloudy"]:
        pIcon = "cloudy"
        pText = "Cloudy"
    elif (
        InterPday[DATA_TIMEMACHINE["cloud"], idx]
        > CLOUD_COVER_THRESHOLDS["partly_cloudy"]
    ):
        pIcon = "partly-cloudy-day"
        pText = "Partly Cloudy"
    else:
        pIcon = "clear-day"
        pText = "Clear"

    dayIconList.append(pIcon)

    dayDict = {
        "time": int(InterPhour[DATA_TIMEMACHINE["time"], 0]) + halfTZ,
        "summary": pText,
        "icon": pIcon,
        "sunriseTime": int(InterPday[DATA_TIMEMACHINE["sunrise"], 0]),
        "sunsetTime": int(InterPday[DATA_TIMEMACHINE["sunset"], 0]),
        "moonPhase": round(InterPday[DATA_TIMEMACHINE["moon_phase"], 0], 2),
        "precipIntensity": round(
            InterPday[DATA_TIMEMACHINE["intensity"], idx] * prepIntensityUnit, 4
        ),
        "precipIntensityMax": round(
            InterPdayMax[DATA_TIMEMACHINE["intensity"], idx] * prepIntensityUnit, 4
        ),
        "precipIntensityMaxTime": int(
            InterPhour[int(InterPdayMaxTime[DATA_TIMEMACHINE["intensity"], idx]), 0]
        ),
        "precipAccumulation": round(
            InterPdaySum[DATA_TIMEMACHINE["intensity"], idx] * prepAccumUnit, 4
        ),
        "precipType": pTypeListDay[idx],
        "temperatureHigh": round(InterPdayMax[4, idx], 2),
        "temperatureHighTime": int(
            InterPhour[int(InterPdayMaxTime[DATA_TIMEMACHINE["temp"], idx]), 0]
        ),
        "temperatureLow": round(InterPdayMin[DATA_TIMEMACHINE["temp"], idx], 2),
        "temperatureLowTime": int(
            InterPhour[int(InterPdayMinTime[DATA_TIMEMACHINE["temp"], idx]), 0]
        ),
        "apparentTemperatureHigh": round(
            InterPdayMax[DATA_TIMEMACHINE["apparent"], idx], 2
        ),
        "apparentTemperatureHighTime": int(
            InterPhour[int(InterPdayMaxTime[DATA_TIMEMACHINE["apparent"], idx]), 0]
        ),
        "apparentTemperatureLow": round(
            InterPdayMin[DATA_TIMEMACHINE["apparent"], idx], 2
        ),
        "apparentTemperatureLowTime": int(
            InterPhour[int(InterPdayMinTime[DATA_TIMEMACHINE["apparent"], idx]), 0]
        ),
        "dewPoint": round(InterPday[DATA_TIMEMACHINE["dew"], idx], 2),
        "pressure": round(InterPday[DATA_TIMEMACHINE["pressure"], idx] * pressUnits, 2),
        "windSpeed": round(InterPday[DATA_TIMEMACHINE["wind"], idx] * windUnit, 2),
        "windGust": round(InterPday[DATA_TIMEMACHINE["gust"], idx] * windUnit, 2),
        "windGustTime": int(
            InterPhour[int(InterPdayMaxTime[DATA_TIMEMACHINE["gust"], idx]), 0]
        ),
        "windBearing": int(round(InterPday[DATA_TIMEMACHINE["bearing"], idx], 0)),
        "cloudCover": round(InterPday[DATA_TIMEMACHINE["cloud"], idx], 2),
        "temperatureMin": round(InterPdayMin[DATA_TIMEMACHINE["temp"], idx], 2),
        "temperatureMinTime": int(
            InterPhour[int(InterPdayMinTime[DATA_TIMEMACHINE["temp"], idx]), 0]
        ),
        "temperatureMax": round(InterPdayMax[DATA_TIMEMACHINE["temp"], idx], 2),
        "temperatureMaxTime": int(
            InterPhour[int(InterPdayMaxTime[DATA_TIMEMACHINE["temp"], idx]), 0]
        ),
        "apparentTemperatureMin": round(
            InterPdayMin[DATA_TIMEMACHINE["apparent"], idx], 2
        ),
        "apparentTemperatureMinTime": int(
            InterPhour[int(InterPdayMinTime[DATA_TIMEMACHINE["apparent"], idx]), 0]
        ),
        "apparentTemperatureMax": round(
            InterPdayMax[DATA_TIMEMACHINE["apparent"], idx], 2
        ),
        "apparentTemperatureMaxTime": int(
            InterPhour[int(InterPdayMaxTime[DATA_TIMEMACHINE["apparent"], idx]), 0]
        ),
        "snowAccumulation": round(
            InterPdaySum[DATA_TIMEMACHINE["snow"], idx] * prepAccumUnit, 4
        ),
    }

    try:
        dayText, dayIcon = calculate_simple_day_text(
            dayDict,
            prepAccumUnit,
            visUnits=visUnits,
            windUnit=windUnit,
            tempUnits=tempUnits,
            isDayTime=True,
            rainPrep=dayRainAccum,
            snowPrep=daySnowAccum,
            icePrep=ICE_ACCUMULATION,
        )

        dayDict["summary"] = translation.translate(["sentence", dayText])
        dayDict["icon"] = dayIcon

    except (KeyError, TypeError, ValueError, IndexError):
        logging.exception("DAILY TEXT GEN ERROR:")

    dayList.append(dict(dayDict))

    ### Currently
    baseTime_array = np.arange(
        baseTime,
        baseTime + datetime.timedelta(minutes=1),
        datetime.timedelta(minutes=1),
    )
    baseTime_grib = (
        (baseTime_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )
    # print('###InterPhour[:,0]###')
    # currentIDX       = find_nearest(InterPhour[:,0], minute_array_grib[0])
    currentIDX = find_nearest(InterPhour[:, 0], baseTime_grib[0])

    InterPcurrent = np.zeros(shape=(15))  # Time, Intensity,Probability

    InterPcurrent[DATA_TIMEMACHINE["time"]] = baseTime_grib
    InterPcurrent[DATA_TIMEMACHINE["intensity"]] = (
        InterPhour[currentIDX, DATA_TIMEMACHINE["intensity"]] / 3600
    )
    InterPcurrent[DATA_TIMEMACHINE["temp"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["temp"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["apparent"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["apparent"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["dew"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["dew"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["pressure"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["pressure"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["wind"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["wind"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["bearing"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["bearing"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["cloud"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["cloud"]
    ]
    InterPcurrent[DATA_TIMEMACHINE["gust"]] = InterPhour[
        currentIDX, DATA_TIMEMACHINE["gust"]
    ]

    cIcon = pIconList[currentIDX]
    cText = hTextList[currentIDX]
    pTypeCurrent = pTypeList[currentIDX]

    # Calculate type-based accumulation for text summaries
    currentRainAccum = 0
    currentSnowAccum = 0
    if pTypeCurrent == "snow":
        currentSnowAccum += InterPcurrent[1] * prepAccumUnit
    else:
        currentRainAccum += InterPcurrent[1] * prepAccumUnit

    returnOBJ = dict()
    returnOBJ["latitude"] = round(lat, 4)
    returnOBJ["longitude"] = round(az_Lon, 4)
    returnOBJ["timezone"] = str(tz_name)
    returnOBJ["offset"] = tz_offset / 60

    if exCurrently == 0:
        returnOBJ["currently"] = dict()
        returnOBJ["currently"]["time"] = int(InterPcurrent[DATA_TIMEMACHINE["time"]])
        returnOBJ["currently"]["summary"] = cText
        returnOBJ["currently"]["icon"] = cIcon
        returnOBJ["currently"]["precipIntensity"] = round(
            InterPcurrent[DATA_TIMEMACHINE["intensity"]] * prepIntensityUnit, 4
        )
        returnOBJ["currently"]["precipType"] = pTypeCurrent
        returnOBJ["currently"]["temperature"] = round(
            InterPcurrent[DATA_TIMEMACHINE["temp"]], 2
        )
        returnOBJ["currently"]["apparentTemperature"] = round(
            InterPcurrent[DATA_TIMEMACHINE["apparent"]], 2
        )
        returnOBJ["currently"]["dewPoint"] = round(InterPcurrent[6], 2)
        returnOBJ["currently"]["pressure"] = round(
            InterPcurrent[DATA_TIMEMACHINE["pressure"]] * pressUnits, 2
        )
        returnOBJ["currently"]["windSpeed"] = round(
            InterPcurrent[DATA_TIMEMACHINE["wind"]] * windUnit, 2
        )
        returnOBJ["currently"]["windGust"] = round(
            InterPcurrent[DATA_TIMEMACHINE["gust"]] * windUnit, 2
        )
        returnOBJ["currently"]["windBearing"] = int(
            round(InterPcurrent[DATA_TIMEMACHINE["bearing"]], 0)
        )
        returnOBJ["currently"]["cloudCover"] = round(
            InterPcurrent[DATA_TIMEMACHINE["cloud"]], 2
        )

        # Update the text
        currentDay = (
            InterPday[DATA_TIMEMACHINE["sunrise"], 0]
            <= InterPcurrent[DATA_TIMEMACHINE["time"]]
            <= InterPday[DATA_TIMEMACHINE["sunset"], 0]
        )

        try:
            currentText, currentIcon = calculate_text(
                returnOBJ["currently"],
                prepAccumUnit,
                1,
                windUnit,
                tempUnits,
                currentDay,
                currentRainAccum,
                currentSnowAccum,
                ICE_ACCUMULATION,
                "current",
                InterPcurrent[DATA_TIMEMACHINE["intensity"]] * prepIntensityUnit,
            )
            returnOBJ["currently"]["summary"] = translation.translate(
                ["title", currentText]
            )
            returnOBJ["currently"]["icon"] = currentIcon
        except Exception:
            logging.exception("CURRENTLY TEXT GEN ERROR:")

    if exHourly == 0:
        returnOBJ["hourly"] = dict()
        returnOBJ["hourly"]["data"] = hourList

    if exDaily == 0:
        returnOBJ["daily"] = dict()
        returnOBJ["daily"]["data"] = dayList

    if exFlags != 1:
        returnOBJ["flags"] = dict()
        returnOBJ["flags"]["sources"] = "ERA5"
        returnOBJ["flags"]["nearest-station"] = int(0)
        returnOBJ["flags"]["units"] = unitSystem
        returnOBJ["flags"]["version"] = apiVersion
        returnOBJ["flags"]["sourceIDX"] = {"x": y, "y": x}
        returnOBJ["flags"]["processTime"] = (
            datetime.datetime.utcnow() - T_Start
        ).microseconds

    logging.info("Complete ERA5 Request")

    return ORJSONResponse(content=returnOBJ, headers={"X-Node-ID": platform.node()})
