"""Hourly computation and object construction helpers."""

from __future__ import annotations

import numpy as np

from API.constants.api_const import PRECIP_IDX, ROUNDING_RULES
from API.constants.clip_const import (
    CLIP_CLOUD,
    CLIP_HUMIDITY,
    CLIP_PRESSURE,
    CLIP_TEMP,
    CLIP_UV,
    CLIP_VIS,
    CLIP_WIND,
)
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.shared_const import MISSING_DATA, REFC_THRESHOLD
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    PRECIP_PROB_THRESHOLD,
    WIND_THRESHOLDS,
)
from API.legacy.hourly import apply_legacy_hourly_text
from API.utils.precip import dbz_to_rate


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
    apparent_inputs,
    station_pressure_inputs,
    era5_rain_intensity,
    era5_snow_water_equivalent,
):
    """Build hourly output objects and summary text/icon lists."""

    maxPchanceHour = np.full((len(hour_array_grib), 5), MISSING_DATA)

    def populate_component_ptype(condition, target_idx, prefix):
        if not condition():
            return
        inter_thour = np.zeros(shape=(len(hour_array), 5))  # Type columns
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

    populate_component_ptype(lambda: "nbm" in source_list, 0, "nbm")
    populate_component_ptype(
        lambda: ("hrrr_0-18" in source_list) and ("hrrr_18-48" in source_list),
        1,
        "hrrr",
    )
    populate_mapped_ptype(lambda: "ecmwf_ifs" in source_list, 2, "ecmwf_ptype")
    populate_component_ptype(lambda: "gefs" in source_list, 3, "gefs")
    populate_mapped_ptype(lambda: "era5" in source_list, 4, "era5_ptype")

    prcipIntensityHour = np.full((len(hour_array_grib), 5), MISSING_DATA)
    nbm_intensity = prcipIntensity_inputs.get("nbm")
    if nbm_intensity is not None:
        prcipIntensityHour[:, 0] = nbm_intensity
    hrrr_intensity = prcipIntensity_inputs.get("hrrr")
    if hrrr_intensity is not None:
        prcipIntensityHour[:, 1] = hrrr_intensity
    ecmwf_intensity = prcipIntensity_inputs.get("ecmwf")
    if ecmwf_intensity is not None:
        prcipIntensityHour[:, 2] = ecmwf_intensity
    gfs_gefs_intensity = prcipIntensity_inputs.get("gfs_gefs")
    if gfs_gefs_intensity is not None:
        prcipIntensityHour[:, 3] = gfs_gefs_intensity
    era5_intensity = prcipIntensity_inputs.get("era5")
    if era5_intensity is not None:
        prcipIntensityHour[:, 4] = era5_intensity

    InterPhour[:, DATA_HOURLY["intensity"]] = (
        np.choose(np.argmin(np.isnan(prcipIntensityHour), axis=1), prcipIntensityHour.T)
        * prepIntensityUnit
    )
    InterPhour[:, DATA_HOURLY["intensity"]] = np.maximum(
        InterPhour[:, DATA_HOURLY["intensity"]], 0
    )
    InterPhour[
        InterPhour[:, DATA_HOURLY["intensity"]] < (0.0005 * prepIntensityUnit),
        DATA_HOURLY["intensity"],
    ] = 0
    InterPhour[:, DATA_HOURLY["type"]] = np.choose(
        np.argmin(np.isnan(prcipIntensityHour), axis=1), maxPchanceHour.T
    )

    prcipProbabilityHour = np.full((len(hour_array_grib), 3), MISSING_DATA)
    nbm_prob = prcipProbability_inputs.get("nbm")
    if nbm_prob is not None:
        prcipProbabilityHour[:, 0] = nbm_prob
    ecmwf_prob = prcipProbability_inputs.get("ecmwf")
    if ecmwf_prob is not None:
        prcipProbabilityHour[:, 1] = ecmwf_prob
    gefs_prob = prcipProbability_inputs.get("gefs")
    if gefs_prob is not None:
        prcipProbabilityHour[:, 2] = gefs_prob

    InterPhour[:, DATA_HOURLY["prob"]] = np.choose(
        np.argmin(np.isnan(prcipProbabilityHour), axis=1), prcipProbabilityHour.T
    )
    InterPhour[:, DATA_HOURLY["prob"]] = np.clip(
        InterPhour[:, DATA_HOURLY["prob"]], 0, 1
    )
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] < 0.05, DATA_HOURLY["prob"]] = 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] == 0, 2] = 0

    ecmwf_prob = prcipProbability_inputs.get("ecmwf")
    gefs_prob = prcipProbability_inputs.get("gefs")
    empty_prob = np.full(len(hour_array_grib), np.nan)
    ecmwf_prob_arr = ecmwf_prob if ecmwf_prob is not None else empty_prob
    gefs_prob_arr = gefs_prob if gefs_prob is not None else empty_prob
    InterPhour[:, DATA_HOURLY["error"]] = np.where(
        ~np.isnan(ecmwf_prob_arr),
        ecmwf_prob_arr * 1000,
        np.where(
            ~np.isnan(gefs_prob_arr),
            gefs_prob_arr,
            InterPhour[:, DATA_HOURLY["error"]],
        ),
    )

    InterPhour[:, DATA_HOURLY["temp"]] = np.choose(
        np.argmin(np.isnan(temperature_inputs), axis=1), temperature_inputs.T
    )
    InterPhour[:, DATA_HOURLY["temp"]] = np.clip(
        InterPhour[:, DATA_HOURLY["temp"]], CLIP_TEMP["min"], CLIP_TEMP["max"]
    )

    InterPhour[:, DATA_HOURLY["dew"]] = np.choose(
        np.argmin(np.isnan(dew_inputs), axis=1), dew_inputs.T
    )
    InterPhour[:, DATA_HOURLY["dew"]] = np.clip(
        InterPhour[:, DATA_HOURLY["dew"]], CLIP_TEMP["min"], CLIP_TEMP["max"]
    )

    InterPhour[:, DATA_HOURLY["humidity"]] = np.choose(
        np.argmin(np.isnan(humidity_inputs), axis=1), humidity_inputs.T
    )
    InterPhour[:, DATA_HOURLY["humidity"]] = np.clip(
        InterPhour[:, DATA_HOURLY["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
    )

    InterPhour[:, DATA_HOURLY["pressure"]] = np.choose(
        np.argmin(np.isnan(pressure_inputs), axis=1), pressure_inputs.T
    )
    InterPhour[:, DATA_HOURLY["pressure"]] = np.clip(
        InterPhour[:, DATA_HOURLY["pressure"]],
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
    )

    InterPhour[:, DATA_HOURLY["wind"]] = np.choose(
        np.argmin(np.isnan(wind_inputs), axis=1), wind_inputs.T
    )
    InterPhour[:, DATA_HOURLY["wind"]] = np.clip(
        InterPhour[:, DATA_HOURLY["wind"]], CLIP_WIND["min"], CLIP_WIND["max"]
    )

    InterPhour[:, DATA_HOURLY["gust"]] = np.choose(
        np.argmin(np.isnan(gust_inputs), axis=1), gust_inputs.T
    )
    InterPhour[:, DATA_HOURLY["gust"]] = np.clip(
        InterPhour[:, DATA_HOURLY["gust"]], CLIP_WIND["min"], CLIP_WIND["max"]
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
    InterPhour[:, DATA_HOURLY["cloud"]] = np.clip(
        InterPhour[:, DATA_HOURLY["cloud"]], CLIP_CLOUD["min"], CLIP_CLOUD["max"]
    )

    InterPhour[:, DATA_HOURLY["uv"]] = np.choose(
        np.argmin(np.isnan(uv_inputs), axis=1), uv_inputs.T
    )
    InterPhour[:, DATA_HOURLY["uv"]] = np.clip(
        InterPhour[:, DATA_HOURLY["uv"]], CLIP_UV["min"], CLIP_UV["max"]
    )

    InterPhour[:, DATA_HOURLY["vis"]] = np.choose(
        np.argmin(np.isnan(vis_inputs), axis=1), vis_inputs.T
    )
    InterPhour[:, DATA_HOURLY["vis"]] = np.clip(
        InterPhour[:, DATA_HOURLY["vis"]], CLIP_VIS["min"], CLIP_VIS["max"]
    )

    InterPhour[:, DATA_HOURLY["ozone"]] = np.choose(
        np.argmin(np.isnan(ozone_inputs), axis=1), ozone_inputs.T
    )
    InterPhour[:, DATA_HOURLY["smoke"]] = np.choose(
        np.argmin(np.isnan(smoke_inputs), axis=1), smoke_inputs.T
    )

    InterPhour[:, DATA_HOURLY["accum"]] = np.choose(
        np.argmin(np.isnan(accum_inputs), axis=1), accum_inputs.T
    )
    InterPhour[:, DATA_HOURLY["storm_dist"]] = np.choose(
        np.argmin(np.isnan(nearstorm_inputs["dist"]), axis=1),
        nearstorm_inputs["dist"].T,
    )
    InterPhour[:, DATA_HOURLY["storm_dir"]] = np.choose(
        np.argmin(np.isnan(nearstorm_inputs["dir"]), axis=1),
        nearstorm_inputs["dir"].T,
    )
    InterPhour[:, DATA_HOURLY["apparent"]] = np.choose(
        np.argmin(np.isnan(apparent_inputs), axis=1), apparent_inputs.T
    )
    InterPhour[:, DATA_HOURLY["fire"]] = InterPhour[:, DATA_HOURLY["fire"]]
    InterPhour[:, DATA_HOURLY["solar"]] = InterPhour[:, DATA_HOURLY["solar"]]
    InterPhour[:, DATA_HOURLY["cape"]] = InterPhour[:, DATA_HOURLY["cape"]]

    if station_pressure_inputs is not None:
        InterPhour[:, DATA_HOURLY["station_pressure"]] = np.choose(
            np.argmin(np.isnan(station_pressure_inputs), axis=1),
            station_pressure_inputs.T,
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

    if "era5" in source_list and era5_rain_intensity is not None:
        InterPhour[:, DATA_HOURLY["rain_intensity"]] = era5_rain_intensity
        era5_snow_intensity_si = era5_snow_water_equivalent
        InterPhour[:, DATA_HOURLY["snow_intensity"]] = era5_snow_intensity_si
    else:
        rain_mask = InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["rain"]
        InterPhour[rain_mask, DATA_HOURLY["rain_intensity"]] = InterPhour[
            rain_mask, DATA_HOURLY["intensity"]
        ]

        snow_mask = InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["snow"]
        snow_indices = np.where(snow_mask)[0]
        if snow_indices.size > 0:
            snow_intensity_si = dbz_to_rate(
                InterPhour[snow_indices, DATA_HOURLY["intensity"]],
                np.full(snow_indices.size, "snow", dtype=object),
                min_dbz=REFC_THRESHOLD,
            )
            InterPhour[snow_indices, DATA_HOURLY["snow_intensity"]] = snow_intensity_si

        sleet_mask = (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["ice"]) | (
            InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["sleet"]
        )
        InterPhour[sleet_mask, DATA_HOURLY["ice_intensity"]] = InterPhour[
            sleet_mask, DATA_HOURLY["intensity"]
        ]

    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])
    PTypeHour = pTypeMap[
        np.nan_to_num(InterPhour[:, DATA_HOURLY["type"]], 0).astype(int)
    ]
    PTextHour = pTextMap[
        np.nan_to_num(InterPhour[:, DATA_HOURLY["type"]], 0).astype(int)
    ]

    InterPhour[((InterPhour >= -0.01) & (InterPhour <= 0.01))] = 0

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
):
    """Create hourly response objects and associated summary/icon lists."""
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
            hourIcon = PTypeHour[idx]
            hourText = PTextHour[idx]
        elif InterPhour[idx, DATA_HOURLY["vis"]] < FOG_THRESHOLD_METERS:
            hourIcon = "fog"
            hourText = "Fog"
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
                hourIcon = "partly-cloudy-night"
            elif (
                hour_array_grib[idx]
                >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
                and hour_array_grib[idx]
                <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                hourIcon = "partly-cloudy-day"
            else:
                hourIcon = "partly-cloudy-night"
        else:
            hourText = "Clear"

            if (
                hour_array_grib[idx]
                < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            ):
                hourIcon = "clear-night"
            elif (
                hour_array_grib[idx]
                <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                hourIcon = "clear-day"
            else:
                hourIcon = "clear-night"

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

        if "stationPressure" in extraVars:
            hourItem["stationPressure"] = hourly_display[
                idx, DATA_HOURLY["station_pressure"]
            ]

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

        hour_summary, hour_icon = apply_legacy_hourly_text(
            summary_text=summaryText,
            translation=translation,
            hour_item_si=hourItem_si,
            is_day=isDay,
            icon=icon,
            fallback_text=hourText,
            fallback_icon=hourIcon,
        )
        hourItem["summary"] = hour_summary
        hourItem["icon"] = hour_icon

        if tempUnits < 2:
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
