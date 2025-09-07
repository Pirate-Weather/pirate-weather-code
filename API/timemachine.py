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

Translations = load_all_translations()
ICE_ACCUMULATION = 0


def solar_rad(D_t, lat, t_t):
    """
    returns The theortical clear sky short wave radiation
    https://www.mdpi.com/2072-4292/5/10/4735/htm
    """

    d = 1 + 0.0167 * math.sin((2 * math.pi * (D_t - 93.5365)) / 365)
    r = 0.75
    S_0 = 1367
    delta = 0.4096 * math.sin((2 * math.pi * (D_t + 284)) / 365)
    radLat = np.deg2rad(lat)
    solarHour = math.pi * ((t_t - 12) / 12)
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
            tempUnits = 273.15  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km
            # humidUnit = 0.01  # %
            # elevUnit = 1  # m
        elif unitSystem == "uk":
            windUnit = 2.234  # mph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = 273.15  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.00062137  # miles
            # humidUnit = 0.01  # %
            # elevUnit = 1  # m
        elif unitSystem == "si":
            windUnit = 1  # m/s
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = 273.15  # Celsius
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
        InterPhour[idx, 4] = dataDict["VAR_2t"][idx]
        ## Add Precip
        InterPhour[idx, 1] = (dataDict["VAR_lsp"][idx] + dataDict["VAR_cp"][idx]) * pFac
        ## Add Dew Point
        InterPhour[idx, 6] = dataDict["VAR_2d"][idx]
        # Pressure
        InterPhour[idx, 8] = dataDict["VAR_msl"][idx]
        ## Add wind speed
        InterPhour[idx, 9] = np.sqrt(
            dataDict["VAR_10u"][idx] ** 2 + dataDict["VAR_10v"][idx] ** 2
        )
        # Add Wind Bearing
        InterPhour[idx, 11] = np.rad2deg(
            np.mod(
                np.arctan2(dataDict["VAR_10u"][idx], dataDict["VAR_10v"][idx]) + np.pi,
                2 * np.pi,
            )
        )
        # Add Cloud Cover
        InterPhour[idx, 12] = dataDict["VAR_tcc"][idx]
        # Add Snow
        InterPhour[idx, 13] = dataDict["VAR_sf"][idx] * 10000
        # Add Wind Gust
        InterPhour[idx, 14] = dataDict["VAR_i10fg"][idx]

        # Add Apparent Temperatire based on https://en.wikipedia.org/wiki/Wind_chill
        if dataDict["VAR_2t"][idx] < 283.15:  # 10C in K
            # Convert to C, then back to K
            InterPhour[idx, 5] = (
                13.12
                + 0.6215 * (dataDict["VAR_2t"][idx] - 273.15)
                - 11.37 * (InterPhour[idx, 9] * 3.6) ** 0.16
                + 0.3965
                * (dataDict["VAR_2t"][idx] - 273.15)
                * (InterPhour[idx, 9] * 3.6) ** 0.16
            )
        else:
            InterPhour[idx, 5] = (dataDict["VAR_2t"][idx] - 273.15) + (5 / 9) * (
                6.11
                * math.exp(
                    5417.7530
                    * (
                        (1 / 273.16)
                        - (1 / (273.15 + (dataDict["VAR_2d"][idx] - 273.15)))
                    )
                )
                - 10
            )

        InterPhour[idx, 5] = InterPhour[idx, 5] + 273.15

    # Put temperature into units
    if tempUnits == 0:
        for k in [4, 5, 6]:
            InterPhour[:, k] = (InterPhour[:, k] - 273.15) * 9 / 5 + 32
    else:
        for k in [4, 5, 6]:
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
            hourSnowAccum += InterPhour[idx, 1] * prepAccumUnit
        else:
            hourRainAccum += InterPhour[idx, 1] * prepAccumUnit

        dayRainAccum += hourRainAccum
        daySnowAccum += hourSnowAccum

        # Check if day or night
        sunrise_ts = InterPday[16, 0]
        sunset_ts = InterPday[17, 0]
        isDay = sunrise_ts <= InterPhour[idx, 0] <= sunset_ts

        ## Icon
        if InterPhour[idx, 1] > 0.2:
            pIconList.append(pTypeList[idx])
            hourText = pTextList[idx]
        elif InterPhour[idx, 12] > 0.75:
            pIconList.append("cloudy")
            hourText = "Cloudy"
        elif InterPhour[idx, 12] > 0.375:
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
            "time": int(InterPhour[idx, 0]) + halfTZ,
            "summary": hourText,
            "icon": pIconList[idx],
            "precipIntensity": round(InterPhour[idx, 1] * prepIntensityUnit, 4),
            "precipAccumulation": round(InterPhour[idx, 1] * prepAccumUnit, 4),
            "precipType": pTypeList[idx],
            "temperature": round(InterPhour[idx, 4], 2),
            "apparentTemperature": round(InterPhour[idx, 5], 2),
            "dewPoint": round(InterPhour[idx, 6], 2),
            "pressure": round(InterPhour[idx, 8] * pressUnits, 2),
            "windSpeed": round(InterPhour[idx, 9] * windUnit, 2),
            "windGust": round(InterPhour[idx, 14] * windUnit, 2),
            "windBearing": int(round(InterPhour[idx, 11], 0)),
            "cloudCover": round(InterPhour[idx, 12], 2),
            "snowAccumulation": round(InterPhour[idx, 13] * prepAccumUnit, 2),
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
    for j in [1, 4, 5, 6, 8, 9, 11, 12, 13, 14]:
        InterPday[j, 0] = np.average(InterPhour[:, j])
        InterPdayMax[j, 0] = np.amax(InterPhour[:, j])
        InterPdayMaxTime[j, 0] = np.argmax(InterPhour[:, j])
        InterPdayMin[j, 0] = np.amin(InterPhour[:, j])
        InterPdayMinTime[j, 0] = np.argmin(InterPhour[:, j])
        InterPdaySum[j, 0] = np.sum(InterPhour[:, j])

    # Daily precipitation type
    if InterPdaySum[1, 0] > 0:
        if InterPdaySum[1, 0] * 0.5 > InterPdaySum[13, 0]:
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
    if InterPdaySum[1, 0] == 0:
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

    if InterPdaySum[1, 0] > 0.5:
        # If more than 0.5 mm of precip at any throughout the day, then the icon for whatever is happening
        pIcon = pTypeListDay[idx]
        pText = pTextListDay[idx]
    elif InterPday[12, idx] > 0.75:
        pIcon = "cloudy"
        pText = "Cloudy"
    elif InterPday[12, idx] > 0.375:
        pIcon = "partly-cloudy-day"
        pText = "Partly Cloudy"
    else:
        pIcon = "clear-day"
        pText = "Clear"

    dayIconList.append(pIcon)

    dayDict = {
        "time": int(InterPhour[0, 0]) + halfTZ,
        "summary": pText,
        "icon": pIcon,
        "sunriseTime": int(InterPday[16, 0]),
        "sunsetTime": int(InterPday[17, 0]),
        "moonPhase": round(InterPday[18, 0], 2),
        "precipIntensity": round(InterPday[1, idx] * prepIntensityUnit, 4),
        "precipIntensityMax": round(InterPdayMax[1, idx] * prepIntensityUnit, 4),
        "precipIntensityMaxTime": int(InterPhour[int(InterPdayMaxTime[1, idx]), 0]),
        "precipAccumulation": round(InterPdaySum[1, idx] * prepAccumUnit, 4),
        "precipType": pTypeListDay[idx],
        "temperatureHigh": round(InterPdayMax[4, idx], 2),
        "temperatureHighTime": int(InterPhour[int(InterPdayMaxTime[4, idx]), 0]),
        "temperatureLow": round(InterPdayMin[4, idx], 2),
        "temperatureLowTime": int(InterPhour[int(InterPdayMinTime[4, idx]), 0]),
        "apparentTemperatureHigh": round(InterPdayMax[5, idx], 2),
        "apparentTemperatureHighTime": int(
            InterPhour[int(InterPdayMaxTime[5, idx]), 0]
        ),
        "apparentTemperatureLow": round(InterPdayMin[5, idx], 2),
        "apparentTemperatureLowTime": int(InterPhour[int(InterPdayMinTime[5, idx]), 0]),
        "dewPoint": round(InterPday[6, idx], 2),
        "pressure": round(InterPday[8, idx] * pressUnits, 2),
        "windSpeed": round(InterPday[9, idx] * windUnit, 2),
        "windGust": round(InterPday[14, idx] * windUnit, 2),
        "windGustTime": int(InterPhour[int(InterPdayMaxTime[14, idx]), 0]),
        "windBearing": int(round(InterPday[11, idx], 0)),
        "cloudCover": round(InterPday[12, idx], 2),
        "temperatureMin": round(InterPdayMin[4, idx], 2),
        "temperatureMinTime": int(InterPhour[int(InterPdayMinTime[4, idx]), 0]),
        "temperatureMax": round(InterPdayMax[4, idx], 2),
        "temperatureMaxTime": int(InterPhour[int(InterPdayMaxTime[4, idx]), 0]),
        "apparentTemperatureMin": round(InterPdayMin[5, idx], 2),
        "apparentTemperatureMinTime": int(InterPhour[int(InterPdayMinTime[5, idx]), 0]),
        "apparentTemperatureMax": round(InterPdayMax[5, idx], 2),
        "apparentTemperatureMaxTime": int(InterPhour[int(InterPdayMaxTime[5, idx]), 0]),
        "snowAccumulation": round(InterPdaySum[13, idx] * prepAccumUnit, 4),
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

    InterPcurrent = np.zeros(shape=(14))  # Time, Intensity,Probability

    InterPcurrent[0] = baseTime_grib
    InterPcurrent[1] = InterPhour[currentIDX, 1] / 3600  # "precipIntensity"
    InterPcurrent[4] = InterPhour[currentIDX, 4]  # "temperature"
    InterPcurrent[5] = InterPhour[currentIDX, 5]  # "apparentTemperature"
    InterPcurrent[6] = InterPhour[currentIDX, 6]  # "dewPoint"
    InterPcurrent[8] = InterPhour[currentIDX, 8]  # "pressure"
    InterPcurrent[9] = InterPhour[currentIDX, 9]  #
    InterPcurrent[11] = InterPhour[currentIDX, 11]  #
    InterPcurrent[12] = InterPhour[currentIDX, 12]  #
    InterPcurrent[13] = InterPhour[currentIDX, 15]  # Wind Gust

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
        returnOBJ["currently"]["time"] = int(InterPcurrent[0])
        returnOBJ["currently"]["summary"] = cText
        returnOBJ["currently"]["icon"] = cIcon
        returnOBJ["currently"]["precipIntensity"] = round(
            InterPcurrent[1] * prepIntensityUnit, 4
        )
        returnOBJ["currently"]["precipType"] = pTypeCurrent
        returnOBJ["currently"]["temperature"] = round(InterPcurrent[4], 2)
        returnOBJ["currently"]["apparentTemperature"] = round(InterPcurrent[5], 2)
        returnOBJ["currently"]["dewPoint"] = round(InterPcurrent[6], 2)
        returnOBJ["currently"]["pressure"] = round(InterPcurrent[8] * pressUnits, 2)
        returnOBJ["currently"]["windSpeed"] = round(InterPcurrent[9] * windUnit, 2)
        returnOBJ["currently"]["windGust"] = round(InterPcurrent[9] * windUnit, 2)
        returnOBJ["currently"]["windBearing"] = int(round(InterPcurrent[11], 0))
        returnOBJ["currently"]["cloudCover"] = round(InterPcurrent[12], 2)

        # Update the text
        currentDay = InterPday[16, 0] <= InterPcurrent[0] <= InterPday[17, 0]

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
                InterPcurrent[1] * prepIntensityUnit,
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
