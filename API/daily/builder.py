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


def _aggregate_stats(
    InterPhour,
    index_array,
    daily_days,
    calc_mean=False,
    calc_sum=False,
    calc_max=False,
    calc_min=False,
    calc_argmax=False,
    calc_argmin=False,
    calc_precip=False,
):
    """
    Aggregate statistics for daily data.

    Args:
        InterPhour: Hourly interpolated data.
        index_array: Array of indices mapping hours to days.
        daily_days: Number of days to process.
        calc_mean: Whether to calculate the mean.
        calc_sum: Whether to calculate the sum.
        calc_max: Whether to calculate the maximum.
        calc_min: Whether to calculate the minimum.
        calc_argmax: Whether to calculate the index of the maximum.
        calc_argmin: Whether to calculate the index of the minimum.
        calc_precip: Whether to calculate precipitation type.

    Returns:
        Tuple containing lists of aggregated statistics:
        (mean, sum, max, min, argmax, argmin, precip_type)
    """
    res_mean = []
    res_sum = []
    res_max = []
    res_min = []
    res_argmax = []
    res_argmin = []
    res_precip = np.zeros((daily_days))

    masks = [index_array == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        if calc_mean:
            res_mean.append(np.mean(filtered_data, axis=0))
        if calc_sum:
            res_sum.append(np.sum(filtered_data, axis=0))
        if calc_max:
            res_max.append(np.max(filtered_data, axis=0))
        if calc_min:
            res_min.append(np.min(filtered_data, axis=0))
        if calc_argmax:
            maxTime = np.argmax(filtered_data, axis=0)
            res_argmax.append(filtered_data[maxTime, 0])
        if calc_argmin:
            minTime = np.argmin(filtered_data, axis=0)
            res_argmin.append(filtered_data[minTime, 0])
        if calc_precip:
            dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)
            if dailyTypeCount[0][0] == 0:
                if len(dailyTypeCount) == 2:
                    res_precip[mIDX] = dailyTypeCount[1][0]
                else:
                    res_precip[mIDX] = dailyTypeCount[0][0]
            else:
                res_precip[mIDX] = dailyTypeCount[0][0]

    return (
        res_mean,
        res_sum,
        res_max,
        res_min,
        res_argmax,
        res_argmin,
        res_precip,
    )


def _calculate_precip_chance(
    InterPdaySum,
    interp_half_day_sum,
    interp_half_night_sum,
    maxPchanceDay,
    max_precip_chance_day,
    max_precip_chance_night,
    prepAccumUnit,
    logger,
    loc_tag,
):
    """
    Calculate precipitation chance for day and night periods.

    Args:
        InterPdaySum: Daily sum of precipitation.
        interp_half_day_sum: Half-day sum of precipitation.
        interp_half_night_sum: Half-night sum of precipitation.
        maxPchanceDay: Maximum precipitation chance for the day.
        max_precip_chance_day: Maximum precipitation chance for the half-day.
        max_precip_chance_night: Maximum precipitation chance for the half-night.
        prepAccumUnit: Precipitation accumulation unit.
        logger: Logger instance.
        loc_tag: Location tag for logging.

    Returns:
        Tuple containing updated precipitation chances:
        (maxPchanceDay, max_precip_chance_day, max_precip_chance_night)
    """
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

    return maxPchanceDay, max_precip_chance_day, max_precip_chance_night


def _conv_temp(arr, tempUnits):
    """
    Convert temperature from Celsius to Fahrenheit if needed.

    Args:
        arr: Temperature array.
        tempUnits: Temperature unit (0 for Fahrenheit, 1 for Celsius).

    Returns:
        Converted temperature array.
    """
    return arr * 9 / 5 + 32 if tempUnits == 0 else arr


def _build_display_data(
    InterPday,
    InterPdayHigh,
    InterPdayLow,
    InterPdayMin,
    InterPdayMax,
    InterPdaySum,
    interp_half_day_mean,
    interp_half_day_max,
    interp_half_day_sum,
    interp_half_night_mean,
    interp_half_night_max,
    interp_half_night_sum,
    tempUnits,
    windUnit,
    visUnits,
    prepIntensityUnit,
    prepAccumUnit,
    extraVars,
):
    """
    Build display data for daily and half-day periods.

    Args:
        InterPday: Daily mean data.
        InterPdayHigh: Daily high data.
        InterPdayLow: Daily low data.
        InterPdayMin: Daily minimum data.
        InterPdayMax: Daily maximum data.
        InterPdaySum: Daily sum data.
        interp_half_day_mean: Half-day mean data.
        interp_half_day_max: Half-day maximum data.
        interp_half_day_sum: Half-day sum data.
        interp_half_night_mean: Half-night mean data.
        interp_half_night_max: Half-night maximum data.
        interp_half_night_sum: Half-night sum data.
        tempUnits: Temperature unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        prepIntensityUnit: Precipitation intensity unit.
        prepAccumUnit: Precipitation accumulation unit.
        extraVars: Extra variables to include.

    Returns:
        Tuple containing processed display data arrays.
    """
    daily_display_mean = InterPday.copy()
    daily_display_mean[:, DATA_DAY["dew"]] = _conv_temp(
        InterPday[:, DATA_DAY["dew"]], tempUnits
    )
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
        InterPdayHigh[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_high[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayHigh[:, DATA_DAY["apparent"]], tempUnits
    )

    daily_display_low = InterPdayLow.copy()
    daily_display_low[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayLow[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_low[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayLow[:, DATA_DAY["apparent"]], tempUnits
    )

    daily_display_min = InterPdayMin.copy()
    daily_display_min[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayMin[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_min[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayMin[:, DATA_DAY["apparent"]], tempUnits
    )

    daily_display_max = InterPdayMax.copy()
    daily_display_max[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayMax[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_max[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayMax[:, DATA_DAY["apparent"]], tempUnits
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
        interp_half_day_mean[:, DATA_HOURLY["dew"]], tempUnits
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
        interp_half_night_mean[:, DATA_HOURLY["dew"]], tempUnits
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

    return (
        daily_display_mean,
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
        daily_display_sum,
        half_day_display_mean,
        half_day_display_max,
        half_day_display_sum,
        half_night_display_mean,
        half_night_display_max,
        half_night_display_sum,
    )


def _apply_rounding(
    daily_display_mean,
    daily_display_high,
    daily_display_low,
    daily_display_min,
    daily_display_max,
    daily_display_sum,
    half_day_display_mean,
    half_day_display_max,
    half_day_display_sum,
    half_night_display_mean,
    half_night_display_max,
    half_night_display_sum,
):
    """
    Apply rounding rules to display data.

    Args:
        daily_display_mean: Daily mean display data.
        daily_display_high: Daily high display data.
        daily_display_low: Daily low display data.
        daily_display_min: Daily minimum display data.
        daily_display_max: Daily maximum display data.
        daily_display_sum: Daily sum display data.
        half_day_display_mean: Half-day mean display data.
        half_day_display_max: Half-day maximum display data.
        half_day_display_sum: Half-day sum display data.
        half_night_display_mean: Half-night mean display data.
        half_night_display_max: Half-night maximum display data.
        half_night_display_sum: Half-night sum display data.
    """
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
    extraVars,
):
    """
    Build a single half-day item dictionary.

    Args:
        idx: Index of the item.
        time_val: Time value.
        icon: Icon string.
        text: Summary text.
        precip_type_val: Precipitation type.
        temp_val: Temperature value.
        apparent_val: Apparent temperature value.
        display_mean: Mean display data array.
        display_max: Maximum display data array.
        display_sum: Sum display data array.
        interp_mean: Interpolated mean data array.
        extraVars: Extra variables to include.

    Returns:
        Dictionary representing the half-day item.
    """
    liquid_accum = display_sum[idx, DATA_HOURLY["rain"]]
    snow_accum = display_sum[idx, DATA_HOURLY["snow"]]
    ice_accum = display_sum[idx, DATA_HOURLY["ice"]]
    precip_accum = liquid_accum + snow_accum + ice_accum

    wind_bearing_val = interp_mean[idx, DATA_HOURLY["bearing"]]
    wind_bearing = int(wind_bearing_val) if not np.isnan(wind_bearing_val) else 0
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
        item["stationPressure"] = display_mean[idx, DATA_HOURLY["station_pressure"]]

    return item


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
    """
    Build all daily- and half-day-level objects.

    This function aggregates hourly data into daily and half-day statistics,
    applies unit conversions and rounding, and constructs the final
    DailySection object containing the forecast data.

    Args:
        InterPhour: Hourly interpolated data.
        hourlyDayIndex: Indices for daily aggregation.
        hourlyDay4amIndex: Indices for 4am-to-4am aggregation.
        hourlyDay4pmIndex: Indices for day aggregation.
        hourlyNight4amIndex: Indices for night aggregation.
        hourlyHighIndex: Indices for daily high calculation.
        hourlyLowIndex: Indices for daily low calculation.
        daily_days: Number of days.
        prepAccumUnit: Precipitation accumulation unit.
        prepIntensityUnit: Precipitation intensity unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        tempUnits: Temperature unit.
        extraVars: Extra variables.
        summaryText: Whether to generate summary text.
        translation: Translation function.
        is_all_night: Whether the forecast is for all night.
        is_all_day: Whether the forecast is for all day.
        tz_name: Timezone name.
        icon: Icon set.
        unitSystem: Unit system.
        version: API version.
        timeMachine: Whether this is a time machine request.
        tmExtra: Extra time machine parameters.
        day_array_grib: Daily GRIB array.
        day_array_4am_grib: Daily 4am GRIB array.
        day_array_5pm_grib: Daily 5pm GRIB array.
        InterSday: Daily source data.
        hourList_si: Hourly list for SI units.
        pTypeMap: Precipitation type map.
        pTextMap: Precipitation text map.
        logger: Logger instance.
        loc_tag: Location tag.
        log_timing: Optional timing logger.

    Returns:
        DailySection object containing the daily forecast.
    """
    if log_timing:
        log_timing("Daily start")

    (
        mean_results,
        sum_results,
        max_results,
        min_results,
        argmax_results,
        argmin_results,
        _,
    ) = _aggregate_stats(
        InterPhour,
        hourlyDayIndex,
        daily_days,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_min=True,
        calc_argmax=True,
        calc_argmin=True,
    )

    (
        mean_4am_results,
        sum_4am_results,
        max_4am_results,
        _,
        _,
        _,
        maxPchanceDay,
    ) = _aggregate_stats(
        InterPhour,
        hourlyDay4amIndex,
        daily_days,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_precip=True,
    )

    (
        mean_day_results,
        sum_day_results,
        max_day_results,
        _,
        _,
        _,
        max_precip_chance_day,
    ) = _aggregate_stats(
        InterPhour,
        hourlyDay4pmIndex,
        daily_days,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_precip=True,
    )

    (
        mean_night_results,
        sum_night_results,
        max_night_results,
        _,
        _,
        _,
        max_precip_chance_night,
    ) = _aggregate_stats(
        InterPhour,
        hourlyNight4amIndex,
        daily_days,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_precip=True,
    )

    (
        _,
        _,
        high_results,
        _,
        arghigh_results,
        _,
        _,
    ) = _aggregate_stats(
        InterPhour,
        hourlyHighIndex,
        daily_days,
        calc_max=True,
        calc_argmax=True,
    )

    (
        _,
        _,
        _,
        low_results,
        _,
        arglow_results,
        _,
    ) = _aggregate_stats(
        InterPhour,
        hourlyLowIndex,
        daily_days,
        calc_min=True,
        calc_argmin=True,
    )

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

    maxPchanceDay, max_precip_chance_day, max_precip_chance_night = (
        _calculate_precip_chance(
            InterPdaySum,
            interp_half_day_sum,
            interp_half_night_sum,
            maxPchanceDay,
            max_precip_chance_day,
            max_precip_chance_night,
            prepAccumUnit,
            logger,
            loc_tag,
        )
    )

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

    (
        daily_display_mean,
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
        daily_display_sum,
        half_day_display_mean,
        half_day_display_max,
        half_day_display_sum,
        half_night_display_mean,
        half_night_display_max,
        half_night_display_sum,
    ) = _build_display_data(
        InterPday,
        InterPdayHigh,
        InterPdayLow,
        InterPdayMin,
        InterPdayMax,
        InterPdaySum,
        interp_half_day_mean,
        interp_half_day_max,
        interp_half_day_sum,
        interp_half_night_mean,
        interp_half_night_max,
        interp_half_night_sum,
        tempUnits,
        windUnit,
        visUnits,
        prepIntensityUnit,
        prepAccumUnit,
        extraVars,
    )

    _apply_rounding(
        daily_display_mean,
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
        daily_display_sum,
        half_day_display_mean,
        half_day_display_max,
        half_day_display_sum,
        half_night_display_mean,
        half_night_display_max,
        half_night_display_sum,
    )

    for idx in range(0, daily_days):
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
            extraVars,
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
            extraVars,
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
