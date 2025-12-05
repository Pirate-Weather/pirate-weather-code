"""Hourly computation and object construction helpers."""

from __future__ import annotations

import numpy as np

from API.api_utils import calculate_apparent_temperature, clipLog, zero_small_values
from API.constants.api_const import (
    PRECIP_ACCUM_NOISE_THRESHOLD,
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    PRECIP_PROB_NOISE_THRESHOLD,
    ROUNDING_RULES,
)
from API.constants.clip_const import (
    CLIP_CAPE,
    CLIP_CLOUD,
    CLIP_FEELS_LIKE,
    CLIP_FIRE,
    CLIP_HUMIDITY,
    CLIP_OZONE,
    CLIP_PRESSURE,
    CLIP_SMOKE,
    CLIP_SOLAR,
    CLIP_TEMP,
    CLIP_UV,
    CLIP_VIS,
    CLIP_WIND,
)
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.shared_const import MISSING_DATA
from API.legacy.hourly import apply_legacy_hourly_text
from API.PirateText import calculate_text
from API.PirateTextHelper import estimate_snow_height


def _populate_max_pchance(
    hour_array_grib,
    hour_array,
    source_list,
    InterThour_inputs,
):
    """
    Populate maximum precipitation chance for each hour.

    Args:
        hour_array_grib: GRIB hour array.
        hour_array: Hour array.
        source_list: List of data sources.
        InterThour_inputs: Inputs for hourly interpolation.

    Returns:
        Array of maximum precipitation chances.
    """
    maxPchanceHour = np.full((len(hour_array_grib), 6), MISSING_DATA)

    def populate_component_ptype(condition, target_idx, prefix):
        if not condition():
            return
        inter_thour = np.zeros(shape=(len(hour_array), 5))
        inter_thour[:, 1] = InterThour_inputs[f"{prefix}_snow"]
        inter_thour[:, 2] = InterThour_inputs[f"{prefix}_ice"]
        inter_thour[:, 3] = InterThour_inputs[f"{prefix}_freezing_rain"]
        inter_thour[:, 4] = InterThour_inputs[f"{prefix}_rain"]
        inter_thour[inter_thour < 0.01] = 0
        maxPchanceHour[:, target_idx] = np.argmax(inter_thour, axis=1)
        maxPchanceHour[np.isnan(inter_thour[:, 1]), target_idx] = MISSING_DATA

    def populate_mapped_ptype(condition, target_idx, key):
        if not condition():
            return
        ptype_hour = np.round(InterThour_inputs[key]).astype(int)
        conditions = [
            np.isin(ptype_hour, [5, 6, 9]),
            np.isin(ptype_hour, [4, 8, 10]),
            np.isin(ptype_hour, [3, 12]),
            np.isin(ptype_hour, [1, 2, 7, 11]),
        ]
        choices = [1, 2, 3, 4]
        mapped_ptype = np.select(conditions, choices, default=0)
        maxPchanceHour[:, target_idx] = mapped_ptype
        maxPchanceHour[np.isnan(ptype_hour), target_idx] = MISSING_DATA

    def populate_wmo4677_ptype(condition, target_idx, key):
        """Map WMO code 4677 (present weather) to precipitation type categories.

        WMO 4677 codes mapping:
        - 50-59: Drizzle → rain (4)
        - 60-65: Rain → rain (4)
        - 66-67: Freezing rain/drizzle → freezing rain (3)
        - 68-69: Rain/snow mix → rain (4)
        - 70-75: Snow → snow (1)
        - 76-79: Ice pellets/graupel → ice (2)
        - 80-84: Rain showers → rain (4)
        - 85-90: Snow showers → snow (1)
        - 91-99: Thunderstorms → rain (4)
        """
        if not condition():
            return
        ptype_hour = np.round(InterThour_inputs[key]).astype(int)
        conditions = [
            # Snow: 70-75, 85-90
            np.isin(ptype_hour, list(range(70, 76)) + list(range(85, 91))),
            # Ice: 76-79
            np.isin(ptype_hour, list(range(76, 80))),
            # Freezing rain: 66-67
            np.isin(ptype_hour, [66, 67]),
            # Rain: 50-65, 68-69, 80-84, 91-99
            np.isin(
                ptype_hour,
                list(range(50, 66))
                + [68, 69]
                + list(range(80, 85))
                + list(range(91, 100)),
            ),
        ]
        choices = [1, 2, 3, 4]
        mapped_ptype = np.select(conditions, choices, default=0)
        maxPchanceHour[:, target_idx] = mapped_ptype
        maxPchanceHour[np.isnan(ptype_hour), target_idx] = MISSING_DATA

    populate_component_ptype(lambda: "nbm" in source_list, 0, "nbm")
    populate_component_ptype(
        lambda: ("hrrr_0-18" in source_list) and ("hrrr_18-48" in source_list),
        1,
        "hrrr",
    )
    populate_mapped_ptype(lambda: "ecmwf_ifs" in source_list, 2, "ecmwf_ptype")
    populate_wmo4677_ptype(lambda: "dwd_mosmix" in source_list, 3, "dwd_mosmix_ptype")
    populate_component_ptype(lambda: "gefs" in source_list, 4, "gefs")
    populate_mapped_ptype(lambda: "era5" in source_list, 5, "era5_ptype")

    return maxPchanceHour


def _calculate_intensity_prob(
    hour_array_grib,
    InterPhour,
    maxPchanceHour,
    prcipIntensity_inputs,
    prcipProbability_inputs,
):
    """
    Calculate precipitation intensity and probability.

    Args:
        hour_array_grib: GRIB hour array.
        InterPhour: Hourly interpolated data.
        maxPchanceHour: Maximum precipitation chance for each hour.
        prcipIntensity_inputs: Precipitation intensity inputs.
        prcipProbability_inputs: Precipitation probability inputs.
    """
    prcipIntensityHour = np.full((len(hour_array_grib), 6), MISSING_DATA)
    intensity_sources = [
        ("nbm", 0),
        ("hrrr", 1),
        ("ecmwf", 2),
        ("dwd_mosmix", 3),
        ("gfs_gefs", 4),
        ("era5", 5),
    ]
    for source_key, idx in intensity_sources:
        val = prcipIntensity_inputs.get(source_key)
        if val is not None:
            prcipIntensityHour[:, idx] = val

    InterPhour[:, DATA_HOURLY["intensity"]] = np.choose(
        np.argmin(np.isnan(prcipIntensityHour), axis=1), prcipIntensityHour.T
    )
    InterPhour[:, DATA_HOURLY["intensity"]] = np.maximum(
        InterPhour[:, DATA_HOURLY["intensity"]], 0
    )
    InterPhour[:, DATA_HOURLY["intensity"]] = zero_small_values(
        InterPhour[:, DATA_HOURLY["intensity"]], threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    InterPhour[:, DATA_HOURLY["type"]] = np.choose(
        np.argmin(np.isnan(prcipIntensityHour), axis=1), maxPchanceHour.T
    )

    prcipProbabilityHour = np.full((len(hour_array_grib), 3), MISSING_DATA)
    prob_sources = [("nbm", 0), ("ecmwf", 1), ("gefs", 2)]
    for source_key, idx in prob_sources:
        val = prcipProbability_inputs.get(source_key)
        if val is not None:
            prcipProbabilityHour[:, idx] = val

    InterPhour[:, DATA_HOURLY["prob"]] = np.choose(
        np.argmin(np.isnan(prcipProbabilityHour), axis=1), prcipProbabilityHour.T
    )
    InterPhour[:, DATA_HOURLY["prob"]] = np.clip(
        InterPhour[:, DATA_HOURLY["prob"]], 0, 1
    )
    InterPhour[:, DATA_HOURLY["prob"]] = zero_small_values(
        InterPhour[:, DATA_HOURLY["prob"]], threshold=PRECIP_PROB_NOISE_THRESHOLD
    )
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] == 0, 2] = 0


def _process_input_vars(
    InterPhour,
    error_inputs,
    temperature_inputs,
    dew_inputs,
    humidity_inputs,
    pressure_inputs,
    wind_inputs,
    gust_inputs,
    bearing_inputs,
    cloud_inputs,
    uv_inputs,
    vis_inputs,
    ozone_inputs,
    smoke_inputs,
    accum_inputs,
    nearstorm_inputs,
    fire_inputs,
    solar_inputs,
    cape_inputs,
    feels_like_inputs,
    station_pressure_inputs,
    humidUnit,
):
    """
    Process input variables and populate InterPhour.

    Args:
        InterPhour: Hourly interpolated data.
        error_inputs: Error inputs.
        temperature_inputs: Temperature inputs.
        dew_inputs: Dew point inputs.
        humidity_inputs: Humidity inputs.
        pressure_inputs: Pressure inputs.
        wind_inputs: Wind speed inputs.
        gust_inputs: Wind gust inputs.
        bearing_inputs: Wind bearing inputs.
        cloud_inputs: Cloud cover inputs.
        uv_inputs: UV index inputs.
        vis_inputs: Visibility inputs.
        ozone_inputs: Ozone inputs.
        smoke_inputs: Smoke inputs.
        accum_inputs: Accumulation inputs.
        nearstorm_inputs: Near storm inputs.
        fire_inputs: Fire index inputs.
        solar_inputs: Solar inputs.
        cape_inputs: CAPE inputs.
        feels_like_inputs: Feels like temperature inputs.
        station_pressure_inputs: Station pressure inputs.
        humidUnit: Humidity unit.
    """
    InterPhour[:, DATA_HOURLY["error"]] = np.choose(
        np.argmin(np.isnan(error_inputs), axis=1), error_inputs.T
    )

    InterPhour[:, DATA_HOURLY["temp"]] = np.choose(
        np.argmin(np.isnan(temperature_inputs), axis=1), temperature_inputs.T
    )
    InterPhour[:, DATA_HOURLY["temp"]] = clipLog(
        InterPhour[:, DATA_HOURLY["temp"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Temperature Hour",
    )

    InterPhour[:, DATA_HOURLY["dew"]] = np.choose(
        np.argmin(np.isnan(dew_inputs), axis=1), dew_inputs.T
    )
    InterPhour[:, DATA_HOURLY["dew"]] = clipLog(
        InterPhour[:, DATA_HOURLY["dew"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Dew Point Hour",
    )

    InterPhour[:, DATA_HOURLY["humidity"]] = np.choose(
        np.argmin(np.isnan(humidity_inputs), axis=1), humidity_inputs.T
    )
    InterPhour[:, DATA_HOURLY["humidity"]] = (
        InterPhour[:, DATA_HOURLY["humidity"]] * humidUnit
    )
    InterPhour[:, DATA_HOURLY["humidity"]] = clipLog(
        InterPhour[:, DATA_HOURLY["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
        "Humidity Hour",
    )

    InterPhour[:, DATA_HOURLY["pressure"]] = np.choose(
        np.argmin(np.isnan(pressure_inputs), axis=1), pressure_inputs.T
    )
    InterPhour[:, DATA_HOURLY["pressure"]] = clipLog(
        InterPhour[:, DATA_HOURLY["pressure"]],
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Pressure Hour",
    )

    InterPhour[:, DATA_HOURLY["wind"]] = np.choose(
        np.argmin(np.isnan(wind_inputs), axis=1), wind_inputs.T
    )
    InterPhour[:, DATA_HOURLY["wind"]] = clipLog(
        InterPhour[:, DATA_HOURLY["wind"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "Wind Speed Hour",
    )

    InterPhour[:, DATA_HOURLY["gust"]] = np.choose(
        np.argmin(np.isnan(gust_inputs), axis=1), gust_inputs.T
    )
    InterPhour[:, DATA_HOURLY["gust"]] = clipLog(
        InterPhour[:, DATA_HOURLY["gust"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "Wind Gust Hour",
    )

    InterPhour[:, DATA_HOURLY["bearing"]] = np.choose(
        np.argmin(np.isnan(bearing_inputs), axis=1), bearing_inputs.T
    )
    InterPhour[:, DATA_HOURLY["bearing"]] = np.mod(
        InterPhour[:, DATA_HOURLY["bearing"]], 360
    )

    InterPhour[:, DATA_HOURLY["cloud"]] = np.choose(
        np.argmin(np.isnan(cloud_inputs), axis=1), cloud_inputs.T
    )
    InterPhour[:, DATA_HOURLY["cloud"]] = clipLog(
        InterPhour[:, DATA_HOURLY["cloud"]],
        CLIP_CLOUD["min"],
        CLIP_CLOUD["max"],
        "Cloud Cover Hour",
    )

    InterPhour[:, DATA_HOURLY["uv"]] = np.choose(
        np.argmin(np.isnan(uv_inputs), axis=1), uv_inputs.T
    )
    InterPhour[:, DATA_HOURLY["uv"]] = clipLog(
        InterPhour[:, DATA_HOURLY["uv"]],
        CLIP_UV["min"],
        CLIP_UV["max"],
        "UV Index Hour",
    )

    InterPhour[:, DATA_HOURLY["vis"]] = np.clip(
        np.choose(np.argmin(np.isnan(vis_inputs), axis=1), vis_inputs.T),
        CLIP_VIS["min"],
        CLIP_VIS["max"],
    )

    InterPhour[:, DATA_HOURLY["ozone"]] = np.choose(
        np.argmin(np.isnan(ozone_inputs), axis=1), ozone_inputs.T
    )
    InterPhour[:, DATA_HOURLY["ozone"]] = clipLog(
        InterPhour[:, DATA_HOURLY["ozone"]],
        CLIP_OZONE["min"],
        CLIP_OZONE["max"],
        "Ozone Hour",
    )

    InterPhour[:, DATA_HOURLY["smoke"]] = np.choose(
        np.argmin(np.isnan(smoke_inputs), axis=1), smoke_inputs.T
    )
    InterPhour[:, DATA_HOURLY["smoke"]] = clipLog(
        InterPhour[:, DATA_HOURLY["smoke"]],
        CLIP_SMOKE["min"],
        CLIP_SMOKE["max"],
        "Air quality Hour",
    )

    InterPhour[:, DATA_HOURLY["accum"]] = np.choose(
        np.argmin(np.isnan(accum_inputs), axis=1), accum_inputs.T
    )
    InterPhour[:, DATA_HOURLY["accum"]] = zero_small_values(
        InterPhour[:, DATA_HOURLY["accum"]], threshold=PRECIP_ACCUM_NOISE_THRESHOLD
    )
    InterPhour[:, DATA_HOURLY["storm_dist"]] = np.choose(
        np.argmin(np.isnan(nearstorm_inputs["dist"]), axis=1),
        nearstorm_inputs["dist"].T,
    )
    InterPhour[:, DATA_HOURLY["storm_dir"]] = np.choose(
        np.argmin(np.isnan(nearstorm_inputs["dir"]), axis=1),
        nearstorm_inputs["dir"].T,
    )
    InterPhour[:, DATA_HOURLY["fire"]] = np.choose(
        np.argmin(np.isnan(fire_inputs), axis=1), fire_inputs.T
    )
    InterPhour[:, DATA_HOURLY["fire"]] = np.clip(
        InterPhour[:, DATA_HOURLY["fire"]], CLIP_FIRE["min"], CLIP_FIRE["max"]
    )
    InterPhour[:, DATA_HOURLY["solar"]] = np.choose(
        np.argmin(np.isnan(solar_inputs), axis=1), solar_inputs.T
    )
    # Clip solar to var range
    InterPhour[:, DATA_HOURLY["solar"]] = np.clip(
        InterPhour[:, DATA_HOURLY["solar"]], CLIP_SOLAR["min"], CLIP_SOLAR["max"]
    )
    InterPhour[:, DATA_HOURLY["cape"]] = np.choose(
        np.argmin(np.isnan(cape_inputs), axis=1), cape_inputs.T
    )
    InterPhour[:, DATA_HOURLY["cape"]] = np.clip(
        InterPhour[:, DATA_HOURLY["cape"]], CLIP_CAPE["min"], CLIP_CAPE["max"]
    )
    InterPhour[:, DATA_HOURLY["feels_like"]] = np.choose(
        np.argmin(np.isnan(feels_like_inputs), axis=1), feels_like_inputs.T
    )
    InterPhour[:, DATA_HOURLY["feels_like"]] = np.clip(
        InterPhour[:, DATA_HOURLY["feels_like"]],
        CLIP_FEELS_LIKE["min"],
        CLIP_FEELS_LIKE["max"],
    )

    if station_pressure_inputs is not None:
        InterPhour[:, DATA_HOURLY["station_pressure"]] = np.choose(
            np.argmin(np.isnan(station_pressure_inputs), axis=1),
            station_pressure_inputs.T,
        )


def _calculate_derived_metrics(
    InterPhour,
    hourlyDayIndex,
    baseTimeOffset,
    timeMachine,
):
    """
    Calculate derived metrics like apparent temperature and accumulation.

    Args:
        InterPhour: Hourly interpolated data.
        hourlyDayIndex: Hourly day index.
        baseTimeOffset: Base time offset.
        timeMachine: Whether this is a time machine request.

    Returns:
        Tuple containing day zero rain, snow, and ice accumulation.
    """
    InterPhour[:, DATA_HOURLY["apparent"]] = calculate_apparent_temperature(
        InterPhour[:, DATA_HOURLY["temp"]],
        InterPhour[:, DATA_HOURLY["humidity"]],
        InterPhour[:, DATA_HOURLY["wind"]],
        solar=InterPhour[:, DATA_HOURLY["solar"]],
    )

    InterPhour[:, DATA_HOURLY["rain"]] = 0
    InterPhour[:, DATA_HOURLY["snow"]] = 0
    InterPhour[:, DATA_HOURLY["ice"]] = 0

    InterPhour[InterPhour[:, DATA_HOURLY["type"]] == 4, DATA_HOURLY["rain"]] = (
        InterPhour[InterPhour[:, DATA_HOURLY["type"]] == 4, DATA_HOURLY["accum"]]
    )

    snow_indices = np.where(InterPhour[:, DATA_HOURLY["type"]] == 1)[0]
    if snow_indices.size > 0:
        liquid_mm = InterPhour[snow_indices, DATA_HOURLY["accum"]]
        temp_c = InterPhour[snow_indices, DATA_HOURLY["temp"]]
        wind_mps = InterPhour[snow_indices, DATA_HOURLY["wind"]]
        snow_mm_values = estimate_snow_height(liquid_mm, temp_c, wind_mps)
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
    )

    InterPhour[:, DATA_HOURLY["intensity"]] = np.maximum(
        InterPhour[:, DATA_HOURLY["intensity"]], 0
    )

    dayZeroPrepRain = InterPhour[:, DATA_HOURLY["rain"]].copy()
    dayZeroPrepRain[hourlyDayIndex != 0] = 0
    if not timeMachine:
        dayZeroPrepRain[int(baseTimeOffset) :] = 0

    dayZeroPrepSnow = InterPhour[:, DATA_HOURLY["snow"]].copy()
    dayZeroPrepSnow[hourlyDayIndex != 0] = 0
    if not timeMachine:
        dayZeroPrepSnow[int(baseTimeOffset) :] = 0

    dayZeroPrepSleet = InterPhour[:, DATA_HOURLY["ice"]].copy()
    dayZeroPrepSleet[hourlyDayIndex != 0] = 0
    if not timeMachine:
        dayZeroPrepSleet[int(baseTimeOffset) :] = 0

    dayZeroRain = dayZeroPrepRain.sum()
    dayZeroSnow = dayZeroPrepSnow.sum()
    dayZeroIce = dayZeroPrepSleet.sum()

    if not timeMachine:
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["intensity"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["accum"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["rain"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["snow"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["ice"]] = 0
        InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["prob"]] = 0

    InterPhour[:, DATA_HOURLY["rain_intensity"]] = 0
    InterPhour[:, DATA_HOURLY["snow_intensity"]] = 0
    InterPhour[:, DATA_HOURLY["ice_intensity"]] = 0

    rain_mask = InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["rain"]
    InterPhour[rain_mask, DATA_HOURLY["rain_intensity"]] = InterPhour[
        rain_mask, DATA_HOURLY["intensity"]
    ]

    snow_intensity_indices = np.where(
        InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["snow"]
    )[0]
    if snow_intensity_indices.size > 0:
        snow_intensity_si = estimate_snow_height(
            InterPhour[snow_intensity_indices, DATA_HOURLY["intensity"]],
            InterPhour[snow_intensity_indices, DATA_HOURLY["temp"]],
            InterPhour[snow_intensity_indices, DATA_HOURLY["wind"]],
        )
        InterPhour[snow_intensity_indices, DATA_HOURLY["snow_intensity"]] = (
            snow_intensity_si
        )

    sleet_mask = (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["ice"]) | (
        InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["sleet"]
    )
    InterPhour[sleet_mask, DATA_HOURLY["ice_intensity"]] = InterPhour[
        sleet_mask, DATA_HOURLY["intensity"]
    ]

    return dayZeroRain, dayZeroSnow, dayZeroIce


def _build_hourly_display(
    hour_array,
    InterPhour,
    tempUnits,
    windUnit,
    visUnits,
    prepIntensityUnit,
    prepAccumUnit,
    station_pressure_inputs,
):
    """
    Build hourly display data.

    Args:
        hour_array: Hour array.
        InterPhour: Hourly interpolated data.
        tempUnits: Temperature unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        prepIntensityUnit: Precipitation intensity unit.
        prepAccumUnit: Precipitation accumulation unit.
        station_pressure_inputs: Station pressure inputs.

    Returns:
        Hourly display data array.
    """
    hourly_display = np.zeros((len(hour_array), max(DATA_HOURLY.values()) + 1))

    if tempUnits == 0:
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
    else:
        hourly_display[:, DATA_HOURLY["temp"]] = InterPhour[:, DATA_HOURLY["temp"]]
        hourly_display[:, DATA_HOURLY["apparent"]] = InterPhour[
            :, DATA_HOURLY["apparent"]
        ]
        hourly_display[:, DATA_HOURLY["dew"]] = InterPhour[:, DATA_HOURLY["dew"]]
        hourly_display[:, DATA_HOURLY["feels_like"]] = InterPhour[
            :, DATA_HOURLY["feels_like"]
        ]

    hourly_display[:, DATA_HOURLY["wind"]] = (
        InterPhour[:, DATA_HOURLY["wind"]] * windUnit
    )
    hourly_display[:, DATA_HOURLY["gust"]] = (
        InterPhour[:, DATA_HOURLY["gust"]] * windUnit
    )
    hourly_display[:, DATA_HOURLY["vis"]] = InterPhour[:, DATA_HOURLY["vis"]] * visUnits
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
    hourly_display[:, DATA_HOURLY["pressure"]] = (
        InterPhour[:, DATA_HOURLY["pressure"]] / 100
    )
    hourly_display[:, DATA_HOURLY["storm_dist"]] = (
        InterPhour[:, DATA_HOURLY["storm_dist"]] * visUnits
    )
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
    if station_pressure_inputs is not None:
        hourly_display[:, DATA_HOURLY["station_pressure"]] = (
            InterPhour[:, DATA_HOURLY["station_pressure"]] / 100
        )

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
        DATA_HOURLY["bearing"]: ROUNDING_RULES.get("windBearing", 0),
    }

    for idx_field, decimals in hourly_rounding_map.items():
        if decimals == 0:
            hourly_display[:, idx_field] = np.round(
                hourly_display[:, idx_field]
            ).astype(int)
        else:
            hourly_display[:, idx_field] = np.round(
                hourly_display[:, idx_field], decimals
            )

    return hourly_display


def build_hourly_block(
    *,
    source_list,
    InterPhour: np.ndarray,
    hour_array_grib: np.ndarray,
    hour_array: np.ndarray,
    InterSday: np.ndarray,
    hourlyDayIndex: np.ndarray,
    baseTimeOffset: float,
    timeMachine: bool,
    tmExtra: bool,
    prepIntensityUnit: float,
    prepAccumUnit: float,
    windUnit: float,
    visUnits: float,
    tempUnits: int,
    humidUnit: float,
    extraVars,
    summaryText: bool,
    icon: str,
    translation,
    unitSystem: str,
    is_all_night: bool,
    tz_name,
    InterThour_inputs,
    prcipIntensity_inputs,
    prcipProbability_inputs,
    temperature_inputs,
    dew_inputs,
    humidity_inputs,
    pressure_inputs,
    wind_inputs,
    gust_inputs,
    bearing_inputs,
    cloud_inputs,
    uv_inputs,
    vis_inputs,
    ozone_inputs,
    smoke_inputs,
    accum_inputs,
    nearstorm_inputs,
    station_pressure_inputs,
    era5_rain_intensity,
    era5_snow_water_equivalent,
    fire_inputs,
    feels_like_inputs,
    solar_inputs,
    cape_inputs,
    error_inputs,
    version,
):
    """
    Build hourly output objects and summary text/icon lists.

    This function coordinates the calculation of hourly weather data,
    including precipitation chances, intensity, input variable processing,
    derived metrics, and final display object construction.

    Args:
        source_list: List of data sources.
        InterPhour: Hourly interpolated data.
        hour_array_grib: GRIB hour array.
        hour_array: Hour array.
        InterSday: Daily source data.
        hourlyDayIndex: Hourly day index.
        baseTimeOffset: Base time offset.
        timeMachine: Whether this is a time machine request.
        tmExtra: Extra time machine parameters.
        prepIntensityUnit: Precipitation intensity unit.
        prepAccumUnit: Precipitation accumulation unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        tempUnits: Temperature unit.
        humidUnit: Humidity unit.
        extraVars: Extra variables.
        summaryText: Whether to generate summary text.
        icon: Icon set.
        translation: Translation function.
        unitSystem: Unit system.
        is_all_night: Whether the forecast is for all night.
        tz_name: Timezone name.
        InterThour_inputs: Inputs for hourly interpolation.
        prcipIntensity_inputs: Precipitation intensity inputs.
        prcipProbability_inputs: Precipitation probability inputs.
        temperature_inputs: Temperature inputs.
        dew_inputs: Dew point inputs.
        humidity_inputs: Humidity inputs.
        pressure_inputs: Pressure inputs.
        wind_inputs: Wind speed inputs.
        gust_inputs: Wind gust inputs.
        bearing_inputs: Wind bearing inputs.
        cloud_inputs: Cloud cover inputs.
        uv_inputs: UV index inputs.
        vis_inputs: Visibility inputs.
        ozone_inputs: Ozone inputs.
        smoke_inputs: Smoke inputs.
        accum_inputs: Accumulation inputs.
        nearstorm_inputs: Near storm inputs.
        station_pressure_inputs: Station pressure inputs.
        era5_rain_intensity: ERA5 rain intensity.
        era5_snow_water_equivalent: ERA5 snow water equivalent.
        fire_inputs: Fire index inputs.
        feels_like_inputs: Feels like temperature inputs.
        solar_inputs: Solar inputs.
        cape_inputs: CAPE inputs.
        error_inputs: Error inputs.
        version: API version.

    Returns:
        Tuple containing hourly lists and arrays.
    """

    maxPchanceHour = _populate_max_pchance(
        hour_array_grib,
        hour_array,
        source_list,
        InterThour_inputs,
    )

    _calculate_intensity_prob(
        hour_array_grib,
        InterPhour,
        maxPchanceHour,
        prcipIntensity_inputs,
        prcipProbability_inputs,
    )

    _process_input_vars(
        InterPhour,
        error_inputs,
        temperature_inputs,
        dew_inputs,
        humidity_inputs,
        pressure_inputs,
        wind_inputs,
        gust_inputs,
        bearing_inputs,
        cloud_inputs,
        uv_inputs,
        vis_inputs,
        ozone_inputs,
        smoke_inputs,
        accum_inputs,
        nearstorm_inputs,
        fire_inputs,
        solar_inputs,
        cape_inputs,
        feels_like_inputs,
        station_pressure_inputs,
        humidUnit,
    )

    dayZeroRain, dayZeroSnow, dayZeroIce = _calculate_derived_metrics(
        InterPhour,
        hourlyDayIndex,
        baseTimeOffset,
        timeMachine,
    )

    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])
    PTypeHour = pTypeMap[
        np.nan_to_num(InterPhour[:, DATA_HOURLY["type"]], 0).astype(int)
    ]
    PTextHour = pTextMap[
        np.nan_to_num(InterPhour[:, DATA_HOURLY["type"]], 0).astype(int)
    ]

    # Global zeroing
    InterPhour[((InterPhour >= -0.01) & (InterPhour <= 0.01))] = 0

    hourly_display = _build_hourly_display(
        hour_array,
        InterPhour,
        tempUnits,
        windUnit,
        visUnits,
        prepIntensityUnit,
        prepAccumUnit,
        station_pressure_inputs,
    )

    (
        hourList,
        hourList_si,
        hourIconList,
        hourTextList,
    ) = build_hourly_objects(
        hour_array_grib=hour_array_grib,
        InterSday=InterSday,
        hourlyDayIndex=hourlyDayIndex,
        InterPhour=InterPhour,
        hourly_display=hourly_display,
        PTypeHour=PTypeHour,
        PTextHour=PTextHour,
        summaryText=summaryText,
        icon=icon,
        translation=translation,
        tempUnits=tempUnits,
        timeMachine=timeMachine,
        tmExtra=tmExtra,
        extraVars=extraVars,
        version=version,
    )

    return (
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
    )


def build_hourly_objects(
    *,
    hour_array_grib: np.ndarray,
    InterSday: np.ndarray,
    hourlyDayIndex: np.ndarray,
    InterPhour: np.ndarray,
    hourly_display: np.ndarray,
    PTypeHour: np.ndarray,
    PTextHour: np.ndarray,
    summaryText: bool,
    icon: str,
    translation,
    tempUnits: int,
    timeMachine: bool,
    tmExtra: bool,
    extraVars,
    version,
):
    """
    Create hourly response objects and associated summary/icon lists.

    Args:
        hour_array_grib: GRIB hour array.
        InterSday: Daily source data.
        hourlyDayIndex: Hourly day index.
        InterPhour: Hourly interpolated data.
        hourly_display: Hourly display data.
        PTypeHour: Precipitation type for each hour.
        PTextHour: Precipitation text for each hour.
        summaryText: Whether to generate summary text.
        icon: Icon set.
        translation: Translation function.
        tempUnits: Temperature unit.
        timeMachine: Whether this is a time machine request.
        tmExtra: Extra time machine parameters.
        extraVars: Extra variables.
        version: API version.

    Returns:
        Tuple containing hour lists and icon/text lists.
    """
    hourList = []
    hourList_si = []
    hourIconList = []
    hourTextList = []

    for idx in range(0, len(hour_array_grib)):
        if hour_array_grib[idx] < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]:
            isDay = False
        elif (
            hour_array_grib[idx] >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            and hour_array_grib[idx]
            <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
        ):
            isDay = True
        else:
            isDay = False

        accum_display = (
            hourly_display[idx, DATA_HOURLY["rain"]]
            + hourly_display[idx, DATA_HOURLY["snow"]]
            + hourly_display[idx, DATA_HOURLY["ice"]]
        )

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
            "precipIntensity": InterPhour[idx, DATA_HOURLY["intensity"]],
            "precipIntensityError": InterPhour[idx, DATA_HOURLY["error"]],
        }

        if summaryText:
            hourText, hourIcon = calculate_text(hourItem_si, isDay, "hour", icon)
            hourText = translation.translate(["title", hourText])
        else:
            hourText, hourIcon = apply_legacy_hourly_text(
                hour_item_si=hourItem_si,
                is_day=isDay,
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

        if "stationPressure" in extraVars:
            hourItem["stationPressure"] = hourly_display[
                idx, DATA_HOURLY["station_pressure"]
            ]

        if version < 2:
            # Before version 2, apparentTemperature and feelsLike were the same
            hourItem["apparentTemperature"] = hourItem["feelsLike"]

        if timeMachine and not summaryText:
            hourItem["summary"] = hourText
            hourItem["icon"] = hourIcon

        if timeMachine and not tmExtra:
            hourItem.pop("uvIndex", None)
            hourItem.pop("ozone", None)

        hourList.append(hourItem)
        hourList_si.append(hourItem_si)
        hourIconList.append(hourIcon)
        hourTextList.append(hourItem["summary"])

    return hourList, hourList_si, hourIconList, hourTextList
