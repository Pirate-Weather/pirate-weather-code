"""Hourly computation and object construction helpers."""

from __future__ import annotations

import numpy as np

from API.api_utils import (
    calculate_apparent_temperature,
    clipLog,
    zero_small_values,
)
from API.constants.api_const import (
    PRECIP_ACCUM_NOISE_THRESHOLD,
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    PRECIP_PROB_NOISE_THRESHOLD,
    PRECIP_TYPE_DISPLAY,
    PRECIP_TYPES,
    ROUNDING_RULES,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
)
from API.constants.aqi_const import compute_aqi_array
from API.constants.clip_const import (
    CLIP_AQI,
    CLIP_CAPE,
    CLIP_CLOUD,
    CLIP_CO_PPB,
    CLIP_FEELS_LIKE,
    CLIP_FIRE,
    CLIP_HUMIDITY,
    CLIP_NO2_PPB,
    CLIP_O3_PPB,
    CLIP_OZONE,
    CLIP_PM10,
    CLIP_PM25,
    CLIP_PRESSURE,
    CLIP_SMOKE,
    CLIP_SO2_PPB,
    CLIP_SOLAR,
    CLIP_TEMP,
    CLIP_UV,
    CLIP_VIS,
    CLIP_WIND,
)
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.legacy.hourly import apply_legacy_hourly_text
from API.PirateText import calculate_text
from API.PirateTextHelper import estimate_snow_height
from API.utils.fire import calculate_fosberg_fire_index


def _calculate_intensity_prob(
    hour_array_grib,
    InterPhour,
    prcipType_inputs,
    prcipIntensity_inputs,
    prcipProbability_inputs,
):
    """
    Calculate precipitation intensity and probability.

    Args:
        hour_array_grib: GRIB hour array.
        InterPhour: Hourly interpolated data.
        prcipType_inputs: Precipitation type inputs.
        prcipIntensity_inputs: Precipitation intensity inputs.
        prcipProbability_inputs: Precipitation probability inputs.
    """
    InterPhour[:, DATA_HOURLY["intensity"]] = np.choose(
        np.argmin(np.isnan(prcipIntensity_inputs), axis=1), prcipIntensity_inputs.T
    )
    InterPhour[:, DATA_HOURLY["intensity"]] = np.maximum(
        InterPhour[:, DATA_HOURLY["intensity"]], 0
    )
    InterPhour[:, DATA_HOURLY["intensity"]] = zero_small_values(
        InterPhour[:, DATA_HOURLY["intensity"]], threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    InterPhour[:, DATA_HOURLY["type"]] = np.choose(
        np.argmin(np.isnan(prcipIntensity_inputs), axis=1), prcipType_inputs.T
    )

    InterPhour[:, DATA_HOURLY["prob"]] = np.choose(
        np.argmin(np.isnan(prcipProbability_inputs), axis=1),
        prcipProbability_inputs.T,
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
    # If gust is still NaN, fall back to wind speed
    gust_nan_mask = np.isnan(InterPhour[:, DATA_HOURLY["gust"]])
    InterPhour[gust_nan_mask, DATA_HOURLY["gust"]] = InterPhour[
        gust_nan_mask, DATA_HOURLY["wind"]
    ]
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
    InterPhour[:, DATA_HOURLY["fire"]] = calculate_fosberg_fire_index(
        InterPhour[:, DATA_HOURLY["temp"]],
        InterPhour[:, DATA_HOURLY["humidity"]],
        InterPhour[:, DATA_HOURLY["wind"]],
    )

    # Apply temperature-based fallback for precipitation type when type is "none" but intensity exists
    # This handles cases where WMO codes are unmapped or missing
    mask = (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["none"]) & (
        InterPhour[:, DATA_HOURLY["intensity"]] > 0
    )
    InterPhour[:, DATA_HOURLY["type"]][mask] = np.where(
        InterPhour[:, DATA_HOURLY["temp"]][mask] >= TEMP_THRESHOLD_RAIN_C,
        PRECIP_IDX["rain"],
        np.where(
            InterPhour[:, DATA_HOURLY["temp"]][mask] <= TEMP_THRESHOLD_SNOW_C,
            PRECIP_IDX["snow"],
            PRECIP_IDX["sleet"],
        ),
    )

    # Convert rain to ice (freezing rain) when temperature is at or below -1°C
    # This handles cases where models explicitly report liquid precipitation (rain)
    # but temperature indicates it should be frozen. This is a correction for model
    # output, not a fallback like the above logic for "none" types.
    # Meteorologically, rain at temps <= -1°C is freezing rain (supercooled liquid).
    freezing_rain_mask = (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["rain"]) & (
        InterPhour[:, DATA_HOURLY["temp"]] <= TEMP_THRESHOLD_SNOW_C
    )
    InterPhour[freezing_rain_mask, DATA_HOURLY["type"]] = PRECIP_IDX["ice"]

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

    # When accumulation exists but intensity is very small (especially for ECMWF),
    # derive intensity from accumulation. For hourly data, accumulation represents
    # the total precipitation over the hour, which can be used as a rate estimate (mm/h).
    # This handles cases where models provide accumulation but not intensity.
    missing_intensity_mask = (
        InterPhour[:, DATA_HOURLY["intensity"]] < PRECIP_NOISE_THRESHOLD_MMH
    ) & (InterPhour[:, DATA_HOURLY["accum"]] > PRECIP_ACCUM_NOISE_THRESHOLD)
    InterPhour[missing_intensity_mask, DATA_HOURLY["intensity"]] = InterPhour[
        missing_intensity_mask, DATA_HOURLY["accum"]
    ]

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

    snow_mask = InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["snow"]
    InterPhour[snow_mask, DATA_HOURLY["snow_intensity"]] = (
        InterPhour[snow_mask, DATA_HOURLY["intensity"]] * 10
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
            hourly_display[:, idx_field] = np.round(hourly_display[:, idx_field])
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
    prcipType_inputs,
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
    aq_inputs=None,
    inc_airqualitydetails: int = 0,
    minute_presence: dict | None = None,
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

    _calculate_intensity_prob(
        hour_array_grib,
        InterPhour,
        prcipType_inputs,
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

    # Populate AQ columns from aq_inputs (version >= 2 only)
    if aq_inputs is not None:
        try:
            num_rows = InterPhour.shape[0]

            def _fill_aq_col(col_key, src_key, clip_min, clip_max):
                vals = aq_inputs.get(src_key)
                if vals is not None:
                    arr = np.asarray(vals, dtype=float)
                    n = min(len(arr), num_rows)
                    InterPhour[:n, DATA_HOURLY[col_key]] = np.clip(
                        arr[:n], clip_min, clip_max
                    )

            _fill_aq_col("pm25", "pm25", CLIP_PM25["min"], CLIP_PM25["max"])
            _fill_aq_col("pm10", "pm10", CLIP_PM10["min"], CLIP_PM10["max"])
            _fill_aq_col("o3", "o3", CLIP_O3_PPB["min"], CLIP_O3_PPB["max"])
            _fill_aq_col("no2", "no2", CLIP_NO2_PPB["min"], CLIP_NO2_PPB["max"])
            _fill_aq_col("so2", "so2", CLIP_SO2_PPB["min"], CLIP_SO2_PPB["max"])
            _fill_aq_col("co", "co", CLIP_CO_PPB["min"], CLIP_CO_PPB["max"])

            # Compute AQI from pollutant concentrations
            aqi_arr = compute_aqi_array(
                unit_system=unitSystem,
                pm25=InterPhour[:, DATA_HOURLY["pm25"]],
                pm10=InterPhour[:, DATA_HOURLY["pm10"]],
                o3=InterPhour[:, DATA_HOURLY["o3"]],
                no2=InterPhour[:, DATA_HOURLY["no2"]],
                so2=InterPhour[:, DATA_HOURLY["so2"]],
                co=InterPhour[:, DATA_HOURLY["co"]],
            )
            InterPhour[:, DATA_HOURLY["aqi"]] = np.clip(
                aqi_arr, CLIP_AQI["min"], CLIP_AQI["max"]
            )
        except Exception:
            pass  # AQ computation is non-fatal; leave columns as MISSING_DATA

    dayZeroRain, dayZeroSnow, dayZeroIce = _calculate_derived_metrics(
        InterPhour,
        hourlyDayIndex,
        baseTimeOffset,
        timeMachine,
    )

    # Mark hours as mixed when all component types are present for that hour.
    # Use accumulation thresholds to avoid noise-driven mixed flags.
    try:
        accum_thresh = PRECIP_ACCUM_NOISE_THRESHOLD
        mixed_mask = (
            (InterPhour[:, DATA_HOURLY["rain"]] > accum_thresh)
            & (InterPhour[:, DATA_HOURLY["snow"]] > accum_thresh)
            & (InterPhour[:, DATA_HOURLY["ice"]] > accum_thresh)
        )
        InterPhour[mixed_mask, DATA_HOURLY["type"]] = PRECIP_IDX["mixed"]
    except (IndexError, TypeError, ValueError):
        # Safe no-op on any unexpected shapes/types
        pass

    # If minute-level presence flags are provided for the current hour, prefer
    # minute-derived detection for that hour (more accurate for short bursts).
    if minute_presence is not None:
        try:
            idx = int(baseTimeOffset)
            if (
                0 <= idx < InterPhour.shape[0]
                and minute_presence.get("has_rain", False)
                and minute_presence.get("has_snow", False)
                and minute_presence.get("has_ice", False)
            ):
                InterPhour[idx, DATA_HOURLY["type"]] = PRECIP_IDX["mixed"]
        except (ValueError, IndexError):
            pass

    pTypeMap = np.array(
        [
            PRECIP_TYPES["none"],
            PRECIP_TYPES["snow"],
            PRECIP_TYPES["ice"],
            PRECIP_TYPES["sleet"],
            PRECIP_TYPES["rain"],
            PRECIP_TYPES["mixed"],
        ]
    )
    pTextMap = np.array(
        [
            PRECIP_TYPE_DISPLAY["none"],
            PRECIP_TYPE_DISPLAY["snow"],
            PRECIP_TYPE_DISPLAY["ice"],
            PRECIP_TYPE_DISPLAY["sleet"],
            PRECIP_TYPE_DISPLAY["rain"],
            PRECIP_TYPE_DISPLAY["mixed"],
        ]
    )
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
        inc_airqualitydetails=inc_airqualitydetails,
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
    inc_airqualitydetails: int = 0,
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

    def _nan_to_int_or_nan(value):
        return int(value) if not np.isnan(value) else np.nan

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
            "windBearing": _nan_to_int_or_nan(
                hourly_display[idx, DATA_HOURLY["bearing"]]
            ),
            "cloudCover": hourly_display[idx, DATA_HOURLY["cloud"]],
            "uvIndex": _nan_to_int_or_nan(hourly_display[idx, DATA_HOURLY["uv"]]),
            "visibility": hourly_display[idx, DATA_HOURLY["vis"]],
            "ozone": hourly_display[idx, DATA_HOURLY["ozone"]],
            "smoke": hourly_display[idx, DATA_HOURLY["smoke"]],
            "liquidAccumulation": hourly_display[idx, DATA_HOURLY["rain"]],
            "snowAccumulation": hourly_display[idx, DATA_HOURLY["snow"]],
            "iceAccumulation": hourly_display[idx, DATA_HOURLY["ice"]],
            "nearestStormDistance": hourly_display[idx, DATA_HOURLY["storm_dist"]],
            "nearestStormBearing": _nan_to_int_or_nan(
                hourly_display[idx, DATA_HOURLY["storm_dir"]]
            ),
            "fireIndex": hourly_display[idx, DATA_HOURLY["fire"]],
            "feelsLike": hourly_display[idx, DATA_HOURLY["feels_like"]],
            "solar": hourly_display[idx, DATA_HOURLY["solar"]],
            "cape": _nan_to_int_or_nan(hourly_display[idx, DATA_HOURLY["cape"]]),
        }

        if "stationPressure" in extraVars:
            hourItem["stationPressure"] = hourly_display[
                idx, DATA_HOURLY["station_pressure"]
            ]

        if version >= 2:
            # AQI is always included for v2+; detail pollutants gated on inc_airqualitydetails
            aqi_val = InterPhour[idx, DATA_HOURLY["aqi"]]
            hourItem["airQualityIndex"] = (
                int(round(float(aqi_val))) if not np.isnan(aqi_val) else np.nan
            )
            if inc_airqualitydetails:
                pm25_val = InterPhour[idx, DATA_HOURLY["pm25"]]
                pm10_val = InterPhour[idx, DATA_HOURLY["pm10"]]
                o3_val = InterPhour[idx, DATA_HOURLY["o3"]]
                no2_val = InterPhour[idx, DATA_HOURLY["no2"]]
                so2_val = InterPhour[idx, DATA_HOURLY["so2"]]
                co_val = InterPhour[idx, DATA_HOURLY["co"]]
                hourItem["pm25"] = (
                    float(round(pm25_val, 1)) if not np.isnan(pm25_val) else np.nan
                )
                hourItem["pm10"] = (
                    float(round(pm10_val, 1)) if not np.isnan(pm10_val) else np.nan
                )
                hourItem["ozoneConcentration"] = (
                    float(round(o3_val, 1)) if not np.isnan(o3_val) else np.nan
                )
                hourItem["no2Concentration"] = (
                    float(round(no2_val, 1)) if not np.isnan(no2_val) else np.nan
                )
                hourItem["so2Concentration"] = (
                    float(round(so2_val, 1)) if not np.isnan(so2_val) else np.nan
                )
                hourItem["coConcentration"] = (
                    float(round(co_val, 1)) if not np.isnan(co_val) else np.nan
                )

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
