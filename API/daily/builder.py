"""Daily calculation and object construction helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from API.api_utils import select_daily_precip_type
from API.constants.api_const import (
    PRECIP_IDX,
    ROUNDING_RULES,
    TEMPERATURE_UNITS_THRESH,
)
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.shared_const import MISSING_DATA
from API.legacy.daily import (
    apply_legacy_half_day_text,
    pick_day_icon_and_summary,
)
from API.PirateDailyText import calculate_day_text


@dataclass
class DailySection:
    day_list: list
    day_list_si: list
    day_icon_list: list
    day_text_list: list
    day_night_list: list


def build_daily_section(
    *,
    InterPhour: np.ndarray,
    hourlyDayIndex: np.ndarray,
    hourlyDay4amIndex: np.ndarray,
    hourlyDay4pmIndex: np.ndarray,
    hourlyNight4amIndex: np.ndarray,
    hourlyHighIndex: np.ndarray,
    hourlyLowIndex: np.ndarray,
    daily_days: int,
    prepAccumUnit: float,
    prepIntensityUnit: float,
    windUnit: float,
    visUnits: float,
    tempUnits: float,
    extraVars,
    summaryText: bool,
    translation,
    is_all_night: bool,
    is_all_day: bool,
    tz_name,
    icon: str,
    unitSystem: str,
    version: int,
    timeMachine: bool,
    tmExtra: bool,
    day_array_grib: np.ndarray,
    day_array_4am_grib: np.ndarray,
    day_array_5pm_grib: np.ndarray,
    InterSday: np.ndarray,
    hourList_si: list,
    pTypeMap: np.ndarray,
    pTextMap: np.ndarray,
    logger,
    loc_tag: str,
    log_timing: Optional[Callable[[str], None]] = None,
) -> DailySection:
    """Build all daily- and half-day-level objects."""
    if log_timing:
        log_timing("Daily start")

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

    masks = [hourlyDayIndex == day_index for day_index in range(daily_days)]
    for mask in masks:
        filtered_data = InterPhour[mask]

        mean_results.append(np.mean(filtered_data, axis=0))
        sum_results.append(np.sum(filtered_data, axis=0))
        max_results.append(np.max(filtered_data, axis=0))
        min_results.append(np.min(filtered_data, axis=0))
        maxTime = np.argmax(filtered_data, axis=0)
        minTime = np.argmin(filtered_data, axis=0)
        argmax_results.append(filtered_data[maxTime, 0])
        argmin_results.append(filtered_data[minTime, 0])

    masks = [hourlyDay4amIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        mean_4am_results.append(np.mean(filtered_data, axis=0))
        sum_4am_results.append(np.sum(filtered_data, axis=0))
        max_4am_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                maxPchanceDay[mIDX] = dailyTypeCount[1][0]
            else:
                maxPchanceDay[mIDX] = dailyTypeCount[0][0]

        else:
            maxPchanceDay[mIDX] = dailyTypeCount[0][0]

    masks = [hourlyDay4pmIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        mean_day_results.append(np.mean(filtered_data, axis=0))
        sum_day_results.append(np.sum(filtered_data, axis=0))
        max_day_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                max_precip_chance_day[mIDX] = dailyTypeCount[1][0]
            else:
                max_precip_chance_day[mIDX] = dailyTypeCount[0][0]

        else:
            max_precip_chance_day[mIDX] = dailyTypeCount[0][0]

    masks = [hourlyNight4amIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        mean_night_results.append(np.mean(filtered_data, axis=0))
        sum_night_results.append(np.sum(filtered_data, axis=0))
        max_night_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                max_precip_chance_night[mIDX] = dailyTypeCount[1][0]
            else:
                max_precip_chance_night[mIDX] = dailyTypeCount[0][0]

        else:
            max_precip_chance_night[mIDX] = dailyTypeCount[0][0]

    masks = [hourlyHighIndex == day_index for day_index in range(daily_days)]

    for mask in masks:
        filtered_data = InterPhour[mask]

        high_results.append(np.max(filtered_data, axis=0))
        maxTime = np.argmax(filtered_data, axis=0)
        arghigh_results.append(filtered_data[maxTime, 0])

    masks = [hourlyLowIndex == day_index for day_index in range(daily_days)]

    for mask in masks:
        filtered_data = InterPhour[mask]

        low_results.append(np.min(filtered_data, axis=0))
        minTime = np.argmin(filtered_data, axis=0)
        arglow_results.append(filtered_data[minTime, 0])

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
        logger.exception("select_daily_precip_type error %s", loc_tag)

    day_night_list = []
    max_precip_chance_day = np.array(max_precip_chance_day).astype(int)
    precip_type_half_day = pTypeMap[max_precip_chance_day]
    precip_text_half_day = pTextMap[max_precip_chance_day]
    max_precip_chance_night = np.array(max_precip_chance_night).astype(int)
    precip_type_half_night = pTypeMap[max_precip_chance_night]
    precip_text_half_night = pTextMap[max_precip_chance_night]

    dayList = []
    dayList_si = []
    dayIconList = []
    dayTextList = []

    maxPchanceDay = np.array(maxPchanceDay).astype(int)
    PTypeDay = pTypeMap[maxPchanceDay]
    PTextDay = pTextMap[maxPchanceDay]

    if log_timing:
        log_timing("Daily Loop start")

    def _conv_temp(arr):
        return arr * 9 / 5 + 32 if tempUnits == 0 else arr

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
        DATA_DAY["bearing"]: ROUNDING_RULES.get("windBearing", 0),
        DATA_DAY["moon_phase"]: ROUNDING_RULES.get("moonPhase", 2),
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

    accum_dec = ROUNDING_RULES.get("precipAccumulation", 2)
    for idx_field in (DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]):
        daily_display_sum[:, idx_field] = np.round(
            daily_display_sum[:, idx_field], accum_dec
        )

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
        DATA_HOURLY["bearing"]: ROUNDING_RULES.get("windBearing", 0),
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
            liquid_accum = display_sum[idx, DATA_HOURLY["rain"]]
            snow_accum = display_sum[idx, DATA_HOURLY["snow"]]
            ice_accum = display_sum[idx, DATA_HOURLY["ice"]]
            precip_accum = liquid_accum + snow_accum + ice_accum

            wind_bearing_val = interp_mean[idx, DATA_HOURLY["bearing"]]
            wind_bearing = (
                int(wind_bearing_val) if not np.isnan(wind_bearing_val) else 0
            )
            cape_val = interp_mean[idx, DATA_HOURLY["cape"]]
            cape_int = int(cape_val) if not np.isnan(cape_val) else 0

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
                "windBearing": wind_bearing,
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
                "cape": cape_int,
            }

            if "stationPressure" in extraVars:
                item["stationPressure"] = display_mean[
                    idx, DATA_HOURLY["station_pressure"]
                ]

            return item

        day_icon, day_text = pick_day_icon_and_summary(
            max_arr=interp_half_day_max,
            mean_arr=interp_half_day_mean,
            sum_arr=interp_half_day_sum,
            precip_type_arr=precip_type_half_day,
            precip_text_arr=precip_text_half_day,
            idx=idx,
            is_night=is_all_night,
            mode="hourly",
            prep_accum_unit=prepAccumUnit,
            vis_units=visUnits,
            wind_unit=windUnit,
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

        if idx < 8:
            day_text, day_icon = apply_legacy_half_day_text(
                summary_text=summaryText,
                translation=translation,
                hour_list_slice=hourList_si[(idx * 24) + 4 : (idx * 24) + 17],
                is_day=not is_all_night,
                tz_name=tz_name,
                icon_set=icon,
                unit_system=unitSystem,
                fallback_text=day_item["summary"],
                fallback_icon=day_item["icon"],
                logger=logger,
                loc_tag=loc_tag,
                phase="DAY HALF DAY",
            )
            day_item["summary"] = day_text
            day_item["icon"] = day_icon

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

        day_icon, day_text = pick_day_icon_and_summary(
            max_arr=interp_half_night_max,
            mean_arr=interp_half_night_mean,
            sum_arr=interp_half_night_sum,
            precip_type_arr=precip_type_half_night,
            precip_text_arr=precip_text_half_night,
            idx=idx,
            is_night=not is_all_day,
            mode="hourly",
            prep_accum_unit=prepAccumUnit,
            vis_units=visUnits,
            wind_unit=windUnit,
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

        if idx < 8:
            day_text, day_icon = apply_legacy_half_day_text(
                summary_text=summaryText,
                translation=translation,
                hour_list_slice=hourList_si[(idx * 24) + 17 : ((idx + 1) * 24) + 4],
                is_day=is_all_day,
                tz_name=tz_name,
                icon_set=icon,
                unit_system=unitSystem,
                fallback_text=day_item["summary"],
                fallback_icon=day_item["icon"],
                logger=logger,
                loc_tag=loc_tag,
                phase="NIGHT HALF DAY",
            )
            day_item["summary"] = day_text
            day_item["icon"] = day_icon

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

        dayIcon, dayText = pick_day_icon_and_summary(
            max_arr=InterPdayMax4am,
            mean_arr=InterPday4am,
            sum_arr=InterPdaySum4am,
            precip_type_arr=PTypeDay,
            precip_text_arr=PTextDay,
            idx=idx,
            is_night=is_all_night,
            mode="daily",
            prep_accum_unit=prepAccumUnit,
            vis_units=visUnits,
            wind_unit=windUnit,
        )

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
        wind_bearing_day = InterPday[idx, DATA_DAY["bearing"]]
        wind_bearing_day = (
            int(wind_bearing_day) if not np.isnan(wind_bearing_day) else 0
        )

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
            "windBearing": wind_bearing_day,
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

        if "stationPressure" in extraVars:
            dayObject["stationPressure"] = daily_display_mean[
                idx, DATA_DAY["station_pressure"]
            ]

        try:
            if idx < 8 and summaryText:
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

    return DailySection(
        day_list=dayList,
        day_list_si=dayList_si,
        day_icon_list=dayIconList,
        day_text_list=dayTextList,
        day_night_list=day_night_list,
    )
