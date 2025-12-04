"""Currently block helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import metpy as mp
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint

from API.api_utils import (
    calculate_apparent_temperature,
    clipLog,
    estimate_visibility_gultepe_rh_pr_numpy,
)
from API.constants.clip_const import (
    CLIP_CAPE,
    CLIP_CLOUD,
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
from API.constants.forecast_const import (
    DATA_CURRENT,
    DATA_DAY,
    DATA_MINUTELY,
)
from API.constants.model_const import (
    ECMWF,
    ERA5,
    GFS,
    HRRR,
    HRRR_SUBH,
    NBM,
    NBM_FIRE_INDEX,
    RTMA_RU,
)
from API.constants.shared_const import MISSING_DATA
from API.legacy.current import get_legacy_current_summary
from API.PirateText import calculate_text
from API.PirateTextHelper import estimate_snow_height


@dataclass
class CurrentSection:
    """Container for currently results."""

    currently: dict
    interp_current: np.ndarray
    summary_key: Optional[str] = None


@dataclass
class InterpolationState:
    """Container for interpolation state."""

    idx1: int
    idx2: int
    fac1: float
    fac2: float


def _select_value(strategies, default=MISSING_DATA):
    for predicate, getter in strategies:
        if predicate():
            return getter()
    return default


def _interp_scalar(merged, key, state: InterpolationState):
    return merged[state.idx1, key] * state.fac1 + merged[state.idx2, key] * state.fac2


def _interp_uv_magnitude(merged, u_key, v_key, state: InterpolationState):
    return math.sqrt(
        (
            merged[state.idx1, u_key] * state.fac1
            + merged[state.idx2, u_key] * state.fac2
        )
        ** 2
        + (
            merged[state.idx1, v_key] * state.fac1
            + merged[state.idx2, v_key] * state.fac2
        )
        ** 2
    )


def _bearing_from_components(u_val, v_val):
    return np.rad2deg(np.mod(np.arctan2(u_val, v_val) + np.pi, 2 * np.pi))


def _calculate_ecmwf_relative_humidity(
    ECMWF_Merged, state: InterpolationState, humidUnit
):
    humid_fac1 = relative_humidity_from_dewpoint(
        ECMWF_Merged[state.idx1, ECMWF["temp"]] * mp.units.units.degC,
        ECMWF_Merged[state.idx1, ECMWF["dew"]] * mp.units.units.degC,
        phase="auto",
    ).magnitude
    humid_fac2 = relative_humidity_from_dewpoint(
        ECMWF_Merged[state.idx2, ECMWF["temp"]] * mp.units.units.degC,
        ECMWF_Merged[state.idx2, ECMWF["dew"]] * mp.units.units.degC,
        phase="auto",
    ).magnitude

    return (humid_fac1 * state.fac1 + humid_fac2 * state.fac2) * 100 * humidUnit


def _calculate_era5_relative_humidity(
    ERA5_MERGED, state: InterpolationState, humidUnit
):
    humid_fac1 = relative_humidity_from_dewpoint(
        ERA5_MERGED[state.idx1, ERA5["2m_temperature"]] * mp.units.units.degC,
        ERA5_MERGED[state.idx1, ERA5["2m_dewpoint_temperature"]] * mp.units.units.degC,
        phase="auto",
    ).magnitude
    humid_fac2 = relative_humidity_from_dewpoint(
        ERA5_MERGED[state.idx2, ERA5["2m_temperature"]] * mp.units.units.degC,
        ERA5_MERGED[state.idx2, ERA5["2m_dewpoint_temperature"]] * mp.units.units.degC,
        phase="auto",
    ).magnitude

    return (humid_fac1 * state.fac1 + humid_fac2 * state.fac2) * 100 * humidUnit


def _get_temp(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: model_data["dataOut_rtma_ru"][0, RTMA_RU["temp"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["temp"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["temp"], state),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _interp_scalar(
                    model_data["ECMWF_Merged"], ECMWF["temp"], state
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["temp"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"], ERA5["2m_temperature"], state
                ),
            ),
        ]
    )
    return clipLog(val, CLIP_TEMP["min"], CLIP_TEMP["max"], "Temperature Current")


def _get_dew(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: model_data["dataOut_rtma_ru"][0, RTMA_RU["dew"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["dew"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["dew"], state),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _interp_scalar(model_data["ECMWF_Merged"], ECMWF["dew"], state),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["dew"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"], ERA5["2m_dewpoint_temperature"], state
                ),
            ),
        ]
    )
    return clipLog(val, CLIP_TEMP["min"], CLIP_TEMP["max"], "Dewpoint Current")


def _get_humidity(sourceList, model_data, state: InterpolationState, humidUnit):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: model_data["dataOut_rtma_ru"][0, RTMA_RU["humidity"]],
            ),
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(
                    model_data["HRRR_Merged"], HRRR["humidity"], state
                )
                * humidUnit,
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["humidity"], state)
                * humidUnit,
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["humidity"], state)
                * humidUnit,
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _calculate_ecmwf_relative_humidity(
                    model_data["ECMWF_Merged"], state, humidUnit
                ),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _calculate_era5_relative_humidity(
                    model_data["ERA5_MERGED"], state, humidUnit
                ),
            ),
        ]
    )
    return clipLog(val, CLIP_HUMIDITY["min"], CLIP_HUMIDITY["max"], "Humidity Current")


def _get_pressure(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(
                    model_data["HRRR_Merged"], HRRR["pressure"], state
                ),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _interp_scalar(
                    model_data["ECMWF_Merged"], ECMWF["pressure"], state
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(
                    model_data["GFS_Merged"], GFS["pressure"], state
                ),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"], ERA5["mean_sea_level_pressure"], state
                ),
            ),
        ]
    )
    return clipLog(val, CLIP_PRESSURE["min"], CLIP_PRESSURE["max"], "Pressure Current")


def _get_wind(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: math.sqrt(
                    model_data["dataOut_rtma_ru"][0, RTMA_RU["wind_u"]] ** 2
                    + model_data["dataOut_rtma_ru"][0, RTMA_RU["wind_v"]] ** 2
                ),
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: math.sqrt(
                    model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["wind_u"]] ** 2
                    + model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["wind_v"]] ** 2
                ),
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["wind"], state),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _interp_uv_magnitude(
                    model_data["ECMWF_Merged"],
                    ECMWF["wind_u"],
                    ECMWF["wind_v"],
                    state,
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_uv_magnitude(
                    model_data["GFS_Merged"], GFS["wind_u"], GFS["wind_v"], state
                ),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_uv_magnitude(
                    model_data["ERA5_MERGED"],
                    ERA5["10m_u_component_of_wind"],
                    ERA5["10m_v_component_of_wind"],
                    state,
                ),
            ),
        ]
    )
    return clipLog(val, CLIP_WIND["min"], CLIP_WIND["max"], "WindSpeed Current")


def _get_gust(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: model_data["dataOut_rtma_ru"][0, RTMA_RU["gust"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["gust"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["gust"], state),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["gust"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"],
                    ERA5["instantaneous_10m_wind_gust"],
                    state,
                ),
            ),
        ],
        default=MISSING_DATA,
    )
    return clipLog(val, CLIP_WIND["min"], CLIP_WIND["max"], "Gust Current")


def _get_intensity(
    sourceList,
    model_data,
    state: InterpolationState,
    InterPminute,
    InterPcurrent,
):
    if "era5" in sourceList:
        era5 = model_data["ERA5_MERGED"]
        intensity = (
            (
                era5[state.idx1, ERA5["large_scale_rain_rate"]]
                + era5[state.idx1, ERA5["convective_rain_rate"]]
                + era5[state.idx1, ERA5["large_scale_snowfall_rate_water_equivalent"]]
                + era5[state.idx1, ERA5["convective_snowfall_rate_water_equivalent"]]
            )
            * state.fac1
            + (
                era5[state.idx2, ERA5["large_scale_rain_rate"]]
                + era5[state.idx2, ERA5["convective_rain_rate"]]
                + era5[state.idx2, ERA5["large_scale_snowfall_rate_water_equivalent"]]
                + era5[state.idx2, ERA5["convective_snowfall_rate_water_equivalent"]]
            )
            * state.fac2
        ) * 3600

        rain_intensity = (
            (
                era5[state.idx1, ERA5["large_scale_rain_rate"]]
                + era5[state.idx1, ERA5["convective_rain_rate"]]
            )
            * state.fac1
            + (
                era5[state.idx2, ERA5["large_scale_rain_rate"]]
                + era5[state.idx2, ERA5["convective_rain_rate"]]
            )
            * state.fac2
        ) * 3600

        era5_current_snow_we = (
            (
                era5[state.idx1, ERA5["large_scale_snowfall_rate_water_equivalent"]]
                + era5[state.idx1, ERA5["convective_snowfall_rate_water_equivalent"]]
            )
            * state.fac1
            + (
                era5[state.idx2, ERA5["large_scale_snowfall_rate_water_equivalent"]]
                + era5[state.idx2, ERA5["convective_snowfall_rate_water_equivalent"]]
            )
            * state.fac2
        ) * 3600

        snow_intensity = estimate_snow_height(
            np.array([era5_current_snow_we]),
            np.array([InterPcurrent[DATA_CURRENT["temp"]]]),
            np.array([InterPcurrent[DATA_CURRENT["wind"]]]),
        )[0]

        return intensity, rain_intensity, snow_intensity, 0, MISSING_DATA, MISSING_DATA
    else:
        return (
            InterPminute[0, DATA_MINUTELY["intensity"]],
            MISSING_DATA,
            MISSING_DATA,
            MISSING_DATA,
            InterPminute[0, DATA_MINUTELY["prob"]],
            InterPminute[0, DATA_MINUTELY["error"]],
        )


def _get_bearing(sourceList, model_data, state: InterpolationState):
    return _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: _bearing_from_components(
                    model_data["dataOut_rtma_ru"][0, RTMA_RU["wind_u"]],
                    model_data["dataOut_rtma_ru"][0, RTMA_RU["wind_v"]],
                ),
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: _bearing_from_components(
                    model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["wind_u"]],
                    model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["wind_v"]],
                ),
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: model_data["NBM_Merged"][state.idx2, NBM["bearing"]],
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _bearing_from_components(
                    model_data["ECMWF_Merged"][state.idx2, ECMWF["wind_u"]],
                    model_data["ECMWF_Merged"][state.idx2, ECMWF["wind_v"]],
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _bearing_from_components(
                    model_data["GFS_Merged"][state.idx2, GFS["wind_u"]],
                    model_data["GFS_Merged"][state.idx2, GFS["wind_v"]],
                ),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _bearing_from_components(
                    model_data["ERA5_MERGED"][
                        state.idx2, ERA5["10m_u_component_of_wind"]
                    ],
                    model_data["ERA5_MERGED"][
                        state.idx2, ERA5["10m_v_component_of_wind"]
                    ],
                ),
            ),
        ],
        default=MISSING_DATA,
    )


def _get_cloud(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: model_data["dataOut_rtma_ru"][0, RTMA_RU["cloud"]] * 0.01,
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["cloud"], state)
                * 0.01,
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: _interp_scalar(
                    model_data["ECMWF_Merged"], ECMWF["cloud"], state
                )
                * 0.01,
            ),
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(model_data["HRRR_Merged"], HRRR["cloud"], state)
                * 0.01,
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["cloud"], state)
                * 0.01,
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"],
                    ERA5["total_cloud_cover"],
                    state,
                ),
            ),
        ],
        default=MISSING_DATA,
    )
    return clipLog(val, CLIP_CLOUD["min"], CLIP_CLOUD["max"], "Cloud Current")


def _get_uv(sourceList, model_data, state: InterpolationState):
    if "gfs" in sourceList:
        return clipLog(
            (
                model_data["GFS_Merged"][state.idx1, GFS["uv"]] * state.fac1
                + model_data["GFS_Merged"][state.idx2, GFS["uv"]] * state.fac2
            )
            * 18.9
            * 0.025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )
    elif "era5" in sourceList:
        return clipLog(
            (
                model_data["ERA5_MERGED"][
                    state.idx1, ERA5["downward_uv_radiation_at_the_surface"]
                ]
                * state.fac1
                + model_data["ERA5_MERGED"][
                    state.idx2, ERA5["downward_uv_radiation_at_the_surface"]
                ]
                * state.fac2
            )
            / 3600
            * 40
            * 0.0025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )
    else:
        return MISSING_DATA


def _get_station_pressure(sourceList, model_data, state: InterpolationState):
    val = MISSING_DATA
    if "rtma_ru" in sourceList:
        val = model_data["dataOut_rtma_ru"][0, RTMA_RU["pressure"]]
    elif "gfs" in sourceList:
        val = (
            model_data["GFS_Merged"][state.idx1, GFS["station_pressure"]] * state.fac1
            + model_data["GFS_Merged"][state.idx2, GFS["station_pressure"]] * state.fac2
        )
    elif "era5" in sourceList:
        val = (
            model_data["ERA5_MERGED"][state.idx1, ERA5["surface_pressure"]] * state.fac1
            + model_data["ERA5_MERGED"][state.idx2, ERA5["surface_pressure"]]
            * state.fac2
        )

    return clipLog(
        val,
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Station Pressure Current",
    )


def _get_vis(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: 16090
                if model_data["dataOut_rtma_ru"][0, RTMA_RU["vis"]] >= 15999
                else model_data["dataOut_rtma_ru"][0, RTMA_RU["vis"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["vis"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["vis"], state),
            ),
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(model_data["HRRR_Merged"], HRRR["vis"], state),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["vis"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: estimate_visibility_gultepe_rh_pr_numpy(
                    model_data["ERA5_MERGED"][state.idx1, :] * state.fac1
                    + model_data["ERA5_MERGED"][state.idx2, :] * state.fac2,
                    var_index=ERA5,
                    var_axis=1,
                ),
            ),
        ],
        default=MISSING_DATA,
    )
    return np.clip(val, CLIP_VIS["min"], CLIP_VIS["max"])


def _get_ozone(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["ozone"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"],
                    ERA5["total_column_ozone"],
                    state,
                )
                * 46696,
            ),
        ],
        default=MISSING_DATA,
    )
    return clipLog(val, CLIP_OZONE["min"], CLIP_OZONE["max"], "Ozone Current")


def _get_storm(sourceList, model_data, state: InterpolationState):
    if "gfs" in sourceList:
        dist = np.maximum(
            (
                model_data["GFS_Merged"][state.idx1, GFS["storm_dist"]] * state.fac1
                + model_data["GFS_Merged"][state.idx2, GFS["storm_dist"]] * state.fac2
            ),
            0,
        )
        bearing = model_data["GFS_Merged"][state.idx2, GFS["storm_dir"]]
        return dist, bearing
    else:
        return MISSING_DATA, MISSING_DATA


def _get_smoke(sourceList, model_data, state: InterpolationState):
    if model_data["has_hrrr_merged"]:
        val = (
            model_data["HRRR_Merged"][state.idx1, HRRR["smoke"]] * state.fac1
            + model_data["HRRR_Merged"][state.idx2, HRRR["smoke"]] * state.fac2
        )
        return clipLog(val, CLIP_SMOKE["min"], CLIP_SMOKE["max"], "Smoke Current")
    else:
        return MISSING_DATA


def _get_solar(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["solar"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["solar"], state),
            ),
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(model_data["HRRR_Merged"], HRRR["solar"], state),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["solar"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"],
                    ERA5["surface_solar_radiation_downwards"],
                    state,
                )
                / 3600,
            ),
        ],
        default=MISSING_DATA,
    )
    return clipLog(val, CLIP_SOLAR["min"], CLIP_SOLAR["max"], "Solar Current")


def _get_cape(sourceList, model_data, state: InterpolationState):
    val = _select_value(
        [
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(model_data["NBM_Merged"], NBM["cape"], state),
            ),
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(model_data["HRRR_Merged"], HRRR["cape"], state),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(model_data["GFS_Merged"], GFS["cape"], state),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: _interp_scalar(
                    model_data["ERA5_MERGED"],
                    ERA5["convective_available_potential_energy"],
                    state,
                ),
            ),
        ]
    )
    return clipLog(val, CLIP_CAPE["min"], CLIP_CAPE["max"], "CAPE Current")


def _get_feels_like(
    sourceList, model_data, state: InterpolationState, timeMachine, apparent
):
    val = _select_value(
        [
            (
                lambda: "nbm" in sourceList,
                lambda: _interp_scalar(
                    model_data["NBM_Merged"], NBM["apparent"], state
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: _interp_scalar(
                    model_data["GFS_Merged"], GFS["apparent"], state
                ),
            ),
            (lambda: timeMachine, lambda: apparent),
        ],
        default=MISSING_DATA,
    )
    return clipLog(
        val, CLIP_TEMP["min"], CLIP_TEMP["max"], "Apparent Temperature Current"
    )


def _get_fire(sourceList, model_data, state: InterpolationState):
    if "nbm_fire" in sourceList:
        val = (
            model_data["NBM_Fire_Merged"][state.idx1, NBM_FIRE_INDEX] * state.fac1
            + model_data["NBM_Fire_Merged"][state.idx2, NBM_FIRE_INDEX] * state.fac2
        )
        return clipLog(val, CLIP_FIRE["min"], CLIP_FIRE["max"], "Fire index Current")
    else:
        return MISSING_DATA


def build_current_section(
    *,
    sourceList,
    hour_array_grib: np.ndarray,
    minute_array_grib: np.ndarray,
    InterPminute: np.ndarray,
    minuteItems: list,
    minuteRainIntensity: np.ndarray,
    minuteSnowIntensity: np.ndarray,
    minuteSleetIntensity: np.ndarray,
    InterSday: np.ndarray,
    dayZeroRain: float,
    dayZeroSnow: float,
    dayZeroIce: float,
    prepAccumUnit: float,
    prepIntensityUnit: float,
    windUnit: float,
    visUnits: float,
    tempUnits: float,
    humidUnit: float,
    extraVars,
    summaryText: bool,
    translation,
    icon: str,
    unitSystem: str,
    version: int,
    timeMachine: bool,
    tmExtra: bool,
    lat: float,
    lon_IN: float,
    tz_name,
    tz_offset: float,
    ETOPO: float,
    elevUnit: float,
    dataOut_rtma_ru,
    hrrrSubHInterpolation,
    HRRR_Merged,
    NBM_Merged,
    ECMWF_Merged,
    GFS_Merged,
    ERA5_MERGED,
    NBM_Fire_Merged,
    logger,
    loc_tag: str,
    log_timing: Optional[Callable[[str], None]] = None,
    include_currently: bool = True,
) -> CurrentSection:
    """Calculate the currently block and return it alongside the raw array."""
    if log_timing:
        log_timing("Current Start")

    # Calculate interpolation state
    if np.min(np.abs(hour_array_grib - minute_array_grib[0])) < 120:
        idx2 = np.argmin(np.abs(hour_array_grib - minute_array_grib[0]))
        fac1 = 0
        fac2 = 1
    else:
        idx2 = np.searchsorted(hour_array_grib, minute_array_grib[0], side="left")
        fac1 = 1 - (
            abs(minute_array_grib[0] - hour_array_grib[idx2 - 1])
            / (hour_array_grib[idx2] - hour_array_grib[idx2 - 1])
        )
        fac2 = 1 - (
            abs(minute_array_grib[0] - hour_array_grib[idx2])
            / (hour_array_grib[idx2] - hour_array_grib[idx2 - 1])
        )

    idx1 = np.max((idx2 - 1, 0))
    state = InterpolationState(idx1=idx1, idx2=idx2, fac1=fac1, fac2=fac2)

    InterPcurrent = np.zeros(shape=max(DATA_CURRENT.values()) + 1)
    InterPcurrent[DATA_CURRENT["time"]] = int(minute_array_grib[0])

    model_data = {
        "dataOut_rtma_ru": dataOut_rtma_ru,
        "hrrrSubHInterpolation": hrrrSubHInterpolation,
        "HRRR_Merged": HRRR_Merged,
        "NBM_Merged": NBM_Merged,
        "ECMWF_Merged": ECMWF_Merged,
        "GFS_Merged": GFS_Merged,
        "ERA5_MERGED": ERA5_MERGED,
        "NBM_Fire_Merged": NBM_Fire_Merged,
        "has_hrrr_merged": (
            HRRR_Merged is not None
            and ("hrrr_0-18" in sourceList)
            and ("hrrr_18-48" in sourceList)
        ),
    }

    InterPcurrent[DATA_CURRENT["temp"]] = _get_temp(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["dew"]] = _get_dew(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["humidity"]] = _get_humidity(
        sourceList, model_data, state, humidUnit
    )
    InterPcurrent[DATA_CURRENT["pressure"]] = _get_pressure(
        sourceList, model_data, state
    )
    InterPcurrent[DATA_CURRENT["wind"]] = _get_wind(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["gust"]] = _get_gust(sourceList, model_data, state)

    (
        InterPcurrent[DATA_CURRENT["intensity"]],
        InterPcurrent[DATA_CURRENT["rain_intensity"]],
        InterPcurrent[DATA_CURRENT["snow_intensity"]],
        InterPcurrent[DATA_CURRENT["ice_intensity"]],
        InterPcurrent[DATA_CURRENT["prob"]],
        InterPcurrent[DATA_CURRENT["error"]],
    ) = _get_intensity(sourceList, model_data, state, InterPminute, InterPcurrent)

    InterPcurrent[DATA_CURRENT["bearing"]] = _get_bearing(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["cloud"]] = _get_cloud(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["uv"]] = _get_uv(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["station_pressure"]] = _get_station_pressure(
        sourceList, model_data, state
    )
    InterPcurrent[DATA_CURRENT["vis"]] = _get_vis(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["ozone"]] = _get_ozone(sourceList, model_data, state)

    (
        InterPcurrent[DATA_CURRENT["storm_dist"]],
        InterPcurrent[DATA_CURRENT["storm_dir"]],
    ) = _get_storm(sourceList, model_data, state)

    InterPcurrent[DATA_CURRENT["smoke"]] = _get_smoke(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["solar"]] = _get_solar(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["cape"]] = _get_cape(sourceList, model_data, state)

    InterPcurrent[DATA_CURRENT["apparent"]] = calculate_apparent_temperature(
        InterPcurrent[DATA_CURRENT["temp"]],
        InterPcurrent[DATA_CURRENT["humidity"]],
        InterPcurrent[DATA_CURRENT["wind"]],
        InterPcurrent[DATA_CURRENT["solar"]],
    )

    InterPcurrent[DATA_CURRENT["feels_like"]] = _get_feels_like(
        sourceList,
        model_data,
        state,
        timeMachine,
        InterPcurrent[DATA_CURRENT["apparent"]],
    )

    InterPcurrent[DATA_CURRENT["fire"]] = _get_fire(sourceList, model_data, state)

    curr_temp_si = InterPcurrent[DATA_CURRENT["temp"]]
    curr_dew_si = InterPcurrent[DATA_CURRENT["dew"]]
    curr_wind_si = InterPcurrent[DATA_CURRENT["wind"]]
    curr_vis_si = InterPcurrent[DATA_CURRENT["vis"]]

    if tempUnits == 0:
        curr_temp_display = np.round(
            (InterPcurrent[DATA_CURRENT["temp"]]) * 9 / 5 + 32, 2
        )
        curr_apparent_display = np.round(
            (InterPcurrent[DATA_CURRENT["apparent"]]) * 9 / 5 + 32,
            2,
        )
        curr_dew_display = np.round(
            (InterPcurrent[DATA_CURRENT["dew"]]) * 9 / 5 + 32, 2
        )
        curr_feels_like_display = np.round(
            (InterPcurrent[DATA_CURRENT["feels_like"]]) * 9 / 5 + 32,
            2,
        )
    else:
        curr_temp_display = np.round(InterPcurrent[DATA_CURRENT["temp"]], 2)
        curr_apparent_display = np.round(InterPcurrent[DATA_CURRENT["apparent"]], 2)
        curr_dew_display = np.round(InterPcurrent[DATA_CURRENT["dew"]], 2)
        curr_feels_like_display = np.round(InterPcurrent[DATA_CURRENT["feels_like"]], 2)

    curr_storm_dist_display = np.round(
        InterPcurrent[DATA_CURRENT["storm_dist"]] * visUnits, 2
    )
    curr_rain_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["rain_intensity"]] * prepIntensityUnit, 4
    )
    curr_snow_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["snow_intensity"]] * prepIntensityUnit, 4
    )
    curr_ice_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["ice_intensity"]] * prepIntensityUnit, 4
    )
    curr_pressure_display = np.round(InterPcurrent[DATA_CURRENT["pressure"]] / 100, 2)
    curr_wind_display = np.round(InterPcurrent[DATA_CURRENT["wind"]] * windUnit, 2)
    curr_gust_display = np.round(InterPcurrent[DATA_CURRENT["gust"]] * windUnit, 2)
    curr_vis_display = np.round(InterPcurrent[DATA_CURRENT["vis"]] * visUnits, 2)
    curr_station_pressure_display = np.round(
        InterPcurrent[DATA_CURRENT["station_pressure"]] / 100, 2
    )

    curr_humidity_display = np.round(InterPcurrent[DATA_CURRENT["humidity"]], 2)
    curr_cloud_display = np.round(InterPcurrent[DATA_CURRENT["cloud"]], 2)
    curr_uv_display = np.round(InterPcurrent[DATA_CURRENT["uv"]], 2)
    curr_ozone_display = np.round(InterPcurrent[DATA_CURRENT["ozone"]], 2)
    curr_smoke_display = np.round(InterPcurrent[DATA_CURRENT["smoke"]], 2)
    curr_fire_display = np.round(InterPcurrent[DATA_CURRENT["fire"]], 2)
    curr_solar_display = np.round(InterPcurrent[DATA_CURRENT["solar"]], 2)
    bearing_val = np.mod(InterPcurrent[DATA_CURRENT["bearing"]], 360)
    curr_bearing_display = (
        int(np.round(bearing_val, 0)) if not np.isnan(bearing_val) else np.nan
    )
    curr_cape_display = (
        int(np.round(InterPcurrent[DATA_CURRENT["cape"]], 0))
        if not np.isnan(InterPcurrent[DATA_CURRENT["cape"]])
        else 0
    )

    dayZeroIce = float(np.round(dayZeroIce * prepAccumUnit, 4))
    dayZeroRain = float(np.round(dayZeroRain * prepAccumUnit, 4))
    dayZeroSnow = float(np.round(dayZeroSnow * prepAccumUnit, 4))

    cText, cIcon = get_legacy_current_summary(
        minuteItems,
        prepIntensityUnit,
        InterPcurrent,
        InterSday,
    )

    if log_timing:
        log_timing("Object Start")

    InterPcurrent[DATA_CURRENT["rain_intensity"]] = minuteRainIntensity[0]
    InterPcurrent[DATA_CURRENT["snow_intensity"]] = minuteSnowIntensity[0]
    InterPcurrent[DATA_CURRENT["ice_intensity"]] = minuteSleetIntensity[0]

    InterPcurrent[((InterPcurrent > -0.01) & (InterPcurrent < 0.01))] = 0

    currently = dict()
    currently["time"] = int(minute_array_grib[0])
    currently["summary"] = cText
    currently["icon"] = cIcon
    currently["nearestStormDistance"] = curr_storm_dist_display
    currently["nearestStormBearing"] = (
        int(InterPcurrent[DATA_CURRENT["storm_dir"]])
        if not np.isnan(InterPcurrent[DATA_CURRENT["storm_dir"]])
        else np.nan
    )
    currently["precipIntensity"] = minuteItems[0]["precipIntensity"]
    currently["precipProbability"] = minuteItems[0]["precipProbability"]
    currently["precipIntensityError"] = minuteItems[0]["precipIntensityError"]
    currently["precipType"] = minuteItems[0]["precipType"]
    currently["rainIntensity"] = curr_rain_intensity_display
    currently["snowIntensity"] = curr_snow_intensity_display
    currently["iceIntensity"] = curr_ice_intensity_display
    currently["temperature"] = curr_temp_display
    currently["apparentTemperature"] = curr_apparent_display
    currently["dewPoint"] = curr_dew_display
    currently["humidity"] = curr_humidity_display
    currently["pressure"] = curr_pressure_display
    currently["windSpeed"] = curr_wind_display
    currently["windGust"] = curr_gust_display
    currently["windBearing"] = curr_bearing_display
    currently["cloudCover"] = curr_cloud_display
    currently["uvIndex"] = curr_uv_display
    currently["visibility"] = curr_vis_display
    currently["ozone"] = curr_ozone_display
    currently["smoke"] = curr_smoke_display
    currently["fireIndex"] = curr_fire_display
    currently["feelsLike"] = curr_feels_like_display
    currently["currentDayIce"] = dayZeroIce
    currently["currentDayLiquid"] = dayZeroRain
    currently["currentDaySnow"] = dayZeroSnow
    currently["solar"] = curr_solar_display
    currently["cape"] = curr_cape_display

    if "stationPressure" in extraVars:
        currently["stationPressure"] = curr_station_pressure_display

    if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
        currentDay = False
    elif (
        InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
        and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
    ):
        currentDay = True
    elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
        currentDay = False
    else:
        currentDay = True

    currently_si = dict(currently)
    currently_si["icon"] = currently["icon"]
    currently_si["precipType"] = currently["precipType"]
    currently_si["windSpeed"] = curr_wind_si
    currently_si["visibility"] = curr_vis_si
    currently_si["temperature"] = curr_temp_si
    currently_si["dewPoint"] = curr_dew_si
    currently_si["cloudCover"] = InterPcurrent[DATA_CURRENT["cloud"]]
    currently_si["humidity"] = InterPcurrent[DATA_CURRENT["humidity"]]
    currently_si["smoke"] = InterPcurrent[DATA_CURRENT["smoke"]]
    currently_si["cape"] = InterPcurrent[DATA_CURRENT["cape"]]
    currently_si["rainIntensity"] = InterPcurrent[DATA_CURRENT["rain_intensity"]]
    currently_si["snowIntensity"] = InterPcurrent[DATA_CURRENT["snow_intensity"]]
    currently_si["iceIntensity"] = InterPcurrent[DATA_CURRENT["ice_intensity"]]
    currently_si["liquidAccumulation"] = 0
    currently_si["snowAccumulation"] = 0
    currently_si["iceAccumulation"] = 0

    current_summary_key = None
    if include_currently:
        try:
            if summaryText:
                currentText, currentIcon = calculate_text(
                    currently_si,
                    currentDay,
                    "current",
                    icon,
                )
                current_summary_key = currentText
                currently["summary"] = translation.translate(["title", currentText])
                currently["icon"] = currentIcon
        except Exception:
            logger.exception("CURRENTLY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            currently.pop("smoke", None)
            currently.pop("currentDayIce", None)
            currently.pop("currentDayLiquid", None)
            currently.pop("currentDaySnow", None)
            currently.pop("fireIndex", None)
            currently.pop("feelsLike", None)
            currently.pop("solar", None)
            currently.pop("cape", None)
            currently.pop("rainIntensity", None)
            currently.pop("snowIntensity", None)
            currently.pop("iceIntensity", None)

        if timeMachine and not tmExtra:
            currently.pop("nearestStormDistance", None)
            currently.pop("nearestStormBearing", None)
            currently.pop("precipProbability", None)
            currently.pop("precipIntensityError", None)
            currently.pop("humidity", None)
            currently.pop("uvIndex", None)
            currently.pop("visibility", None)
            currently.pop("ozone", None)

    return CurrentSection(
        currently=currently,
        interp_current=InterPcurrent,
        summary_key=current_summary_key,
    )
