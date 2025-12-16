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
    DWD_MOSMIX,
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
from API.utils.source_priority import should_gfs_precede_dwd


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
    """
    Select the first valid (non-NaN) value from a list of strategies.

    Args:
        strategies: List of (predicate, getter) tuples.
        default: Default value if no strategy matches or all return NaN.

    Returns:
        Selected value or default.
    """
    for predicate, getter in strategies:
        if predicate():
            val = getter()
            # Check if value is valid (not NaN) before returning
            if val is not None and not np.isnan(val):
                return val
    return default


def _build_source_strategies(source_map, lat, lon, has_ecmwf=True):
    """
    Build source strategies in priority order based on location.

    Args:
        source_map: Dictionary mapping source names to (predicate, getter) tuples.
        lat: Latitude.
        lon: Longitude.
        has_ecmwf: Whether ECMWF has data for this variable.

    Returns:
        List of (predicate, getter) tuples in priority order.
    """
    gfs_before_dwd = should_gfs_precede_dwd(lat, lon)

    # Define the priority order
    if gfs_before_dwd:
        # North America
        if has_ecmwf:
            order = ["rtma_ru", "hrrrsubh", "nbm", "hrrr", "ecmwf_ifs", "gfs", "dwd_mosmix", "era5"]
        else:
            order = ["rtma_ru", "hrrrsubh", "nbm", "hrrr", "gfs", "dwd_mosmix", "era5"]
    else:
        # Rest of world
        if has_ecmwf:
            order = ["rtma_ru", "hrrrsubh", "nbm", "hrrr", "dwd_mosmix", "ecmwf_ifs", "gfs", "era5"]
        else:
            order = ["rtma_ru", "hrrrsubh", "nbm", "hrrr", "dwd_mosmix", "gfs", "era5"]

    # Build strategies in priority order
    strategies = []
    for source in order:
        if source in source_map:
            strategies.append(source_map[source])

    return strategies


def _interp_scalar(merged, key, state: InterpolationState):
    """
    Interpolate scalar value between two time points.

    Args:
        merged: Merged data array.
        key: Key for the value in the array.
        state: Interpolation state.

    Returns:
        Interpolated value.
    """
    return merged[state.idx1, key] * state.fac1 + merged[state.idx2, key] * state.fac2


def _interp_uv_magnitude(merged, u_key, v_key, state: InterpolationState):
    """
    Interpolate UV vector magnitude between two time points.

    Args:
        merged: Merged data array.
        u_key: Key for U component.
        v_key: Key for V component.
        state: Interpolation state.

    Returns:
        Interpolated magnitude.
    """
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
    """
    Calculate wind bearing from U and V components.

    Args:
        u_val: U component of wind.
        v_val: V component of wind.

    Returns:
        Wind bearing in degrees.
    """
    return np.rad2deg(np.mod(np.arctan2(u_val, v_val) + np.pi, 2 * np.pi))


def _calculate_ecmwf_relative_humidity(
    ECMWF_Merged, state: InterpolationState, humidUnit
):
    """
    Calculate relative humidity from ECMWF data.

    Args:
        ECMWF_Merged: ECMWF merged data.
        state: Interpolation state.
        humidUnit: Humidity unit.

    Returns:
        Interpolated relative humidity.
    """
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
    """
    Calculate relative humidity from ERA5 data.

    Args:
        ERA5_MERGED: ERA5 merged data.
        state: Interpolation state.
        humidUnit: Humidity unit.

    Returns:
        Interpolated relative humidity.
    """
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


def _get_temp(sourceList, model_data, state: InterpolationState, lat, lon):
    """
    Get current temperature from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.
        lat: Latitude.
        lon: Longitude.

    Returns:
        Current temperature.
    """
    source_map = {
        "rtma_ru": (
            lambda: "rtma_ru" in sourceList,
            lambda: model_data["dataOut_rtma_ru"][0, RTMA_RU["temp"]],
        ),
        "hrrrsubh": (
            lambda: "hrrrsubh" in sourceList,
            lambda: model_data["hrrrSubHInterpolation"][0, HRRR_SUBH["temp"]],
        ),
        "nbm": (
            lambda: "nbm" in sourceList,
            lambda: _interp_scalar(model_data["NBM_Merged"], NBM["temp"], state),
        ),
        "dwd_mosmix": (
            lambda: "dwd_mosmix" in sourceList,
            lambda: _interp_scalar(
                model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["temp"], state
            ),
        ),
        "ecmwf_ifs": (
            lambda: "ecmwf_ifs" in sourceList,
            lambda: _interp_scalar(
                model_data["ECMWF_Merged"], ECMWF["temp"], state
            ),
        ),
        "gfs": (
            lambda: "gfs" in sourceList,
            lambda: _interp_scalar(model_data["GFS_Merged"], GFS["temp"], state),
        ),
        "era5": (
            lambda: "era5" in sourceList,
            lambda: _interp_scalar(
                model_data["ERA5_MERGED"], ERA5["2m_temperature"], state
            ),
        ),
    }

    strategies = _build_source_strategies(source_map, lat, lon, has_ecmwf=True)
    val = _select_value(strategies)
    return clipLog(val, CLIP_TEMP["min"], CLIP_TEMP["max"], "Temperature Current")


def _get_dew(sourceList, model_data, state: InterpolationState):
    """
    Get current dew point from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current dew point.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["dew"], state
                ),
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
    """
    Get current humidity from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.
        humidUnit: Humidity unit.

    Returns:
        Current humidity.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["humidity"], state
                )
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
    """
    Get current pressure from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current pressure.
    """
    val = _select_value(
        [
            (
                lambda: model_data["has_hrrr_merged"],
                lambda: _interp_scalar(
                    model_data["HRRR_Merged"], HRRR["pressure"], state
                ),
            ),
            (
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["pressure"], state
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
    """
    Get current wind speed from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current wind speed.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_uv_magnitude(
                    model_data["DWD_MOSMIX_Merged"],
                    DWD_MOSMIX["wind_u"],
                    DWD_MOSMIX["wind_v"],
                    state,
                ),
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
    """
    Get current wind gust from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current wind gust.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["gust"], state
                ),
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
    """
    Calculates the intensity (e.g., precipitation rate) based on available data sources.

    Args:
        sourceList (list): A list of strings indicating the data sources to consider (e.g., "era5").
        model_data (dict): A dictionary containing various model data arrays.
        state (InterpolationState): An object holding interpolation state information.
        InterPminute: Placeholder for minute-level interpolation data or parameters.
        InterPcurrent: Placeholder for current-level interpolation data or parameters.

    Returns:
        float: The calculated intensity values:
        - intensity: The total intensity (e.g., precipitation rate).
        - rain_intensity: The rain intensity (e.g., precipitation rate).
        - snow_intensity: The snow intensity (e.g., precipitation rate).
        - ice_intensity: The ice intensity (e.g., precipitation rate).
        - prob: The probability of precipitation
        - error: The error in the intensity calculation.
    """

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
            InterPminute[0, DATA_MINUTELY["rain_intensity"]],
            InterPminute[0, DATA_MINUTELY["snow_intensity"]],
            InterPminute[0, DATA_MINUTELY["ice_intensity"]],
            InterPminute[0, DATA_MINUTELY["prob"]],
            InterPminute[0, DATA_MINUTELY["error"]],
        )


def _get_bearing(sourceList, model_data, state: InterpolationState):
    """
    Get current wind bearing from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current wind bearing.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _bearing_from_components(
                    model_data["DWD_MOSMIX_Merged"][state.idx2, DWD_MOSMIX["wind_u"]],
                    model_data["DWD_MOSMIX_Merged"][state.idx2, DWD_MOSMIX["wind_v"]],
                ),
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
    """
    Get current cloud cover from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current cloud cover.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["cloud"], state
                )
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
    """
    Get current UV index from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current UV index.
    """
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
    """
    Get current station pressure from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current station pressure.
    """
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
    """
    Get current visibility from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current visibility.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["vis"], state
                ),
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
    """
    Get current ozone from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current ozone.
    """
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
    """
    Get current storm distance and bearing from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Tuple of (storm distance, storm bearing).
    """
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
    """
    Get current smoke from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current smoke.
    """
    if model_data["has_hrrr_merged"]:
        val = (
            model_data["HRRR_Merged"][state.idx1, HRRR["smoke"]] * state.fac1
            + model_data["HRRR_Merged"][state.idx2, HRRR["smoke"]] * state.fac2
        )
        return clipLog(val, CLIP_SMOKE["min"], CLIP_SMOKE["max"], "Smoke Current")
    else:
        return MISSING_DATA


def _get_solar(sourceList, model_data, state: InterpolationState):
    """
    Get current solar radiation from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current solar radiation.
    """
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
                lambda: "dwd_mosmix" in sourceList,
                lambda: _interp_scalar(
                    model_data["DWD_MOSMIX_Merged"], DWD_MOSMIX["solar"], state
                ),
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
    """
    Get current CAPE from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current CAPE.
    """
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
    """
    Get current feels-like temperature from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.
        timeMachine: Whether this is a time machine request.
        apparent: Apparent temperature.

    Returns:
        Current feels-like temperature.
    """
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
    """
    Get current fire index from available sources.

    Args:
        sourceList: List of available sources.
        model_data: Dictionary of model data.
        state: Interpolation state.

    Returns:
        Current fire index.
    """
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
    DWD_MOSMIX_Merged,
    ECMWF_Merged,
    GFS_Merged,
    ERA5_MERGED,
    NBM_Fire_Merged,
    logger,
    loc_tag: str,
    log_timing: Optional[Callable[[str], None]] = None,
    include_currently: bool = True,
) -> CurrentSection:
    """
    Calculate the currently block and return it alongside the raw array.

    This function coordinates the retrieval of current weather conditions
    from various sources, interpolates them to the current time, and
    constructs the CurrentSection object.

    Args:
        sourceList: List of available sources.
        hour_array_grib: Hourly time array.
        minute_array_grib: Minutely time array.
        InterPminute: Minutely interpolated data.
        minuteItems: List of minute items.
        minuteRainIntensity: Minute rain intensity array.
        minuteSnowIntensity: Minute snow intensity array.
        minuteSleetIntensity: Minute sleet intensity array.
        minuteIntensity: Minute intensity array.
        minuteProbability: Minute probability array.
        minuteIntensityError: Minute intensity error array.
        now_time: Current time.
        utc_time: UTC time.
        tz_name: Timezone name.
        lat: Latitude.
        lon: Longitude.
        tempUnits: Temperature unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        prepIntensityUnit: Precipitation intensity unit.
        prepAccumUnit: Precipitation accumulation unit.
        humidUnit: Humidity unit.
        extraVars: Extra variables.
        summaryText: Whether to generate summary text.
        icon: Icon set.
        translation: Translation function.
        unitSystem: Unit system.
        version: API version.
        timeMachine: Whether this is a time machine request.
        tmExtra: Extra time machine parameters.
        hrrrSubHInterpolation: HRRR sub-hourly interpolated data.
        HRRR_Merged: HRRR merged data.
        NBM_Merged: NBM merged data.
        ECMWF_Merged: ECMWF merged data.
        GFS_Merged: GFS merged data.
        ERA5_MERGED: ERA5 merged data.
        NBM_Fire_Merged: NBM fire merged data.
        logger: Logger instance.
        loc_tag: Location tag.
        log_timing: Optional timing logger.
        include_currently: Whether to include the currently block.

    Returns:
        CurrentSection object containing the current forecast.
    """
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
        "DWD_MOSMIX_Merged": DWD_MOSMIX_Merged,
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

    InterPcurrent[DATA_CURRENT["temp"]] = _get_temp(sourceList, model_data, state, lat, lon_IN)
    InterPcurrent[DATA_CURRENT["dew"]] = _get_dew(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["humidity"]] = _get_humidity(
        sourceList, model_data, state, humidUnit
    )
    InterPcurrent[DATA_CURRENT["pressure"]] = _get_pressure(
        sourceList, model_data, state
    )
    InterPcurrent[DATA_CURRENT["wind"]] = _get_wind(sourceList, model_data, state)
    InterPcurrent[DATA_CURRENT["gust"]] = _get_gust(sourceList, model_data, state)

    # If gust is missing/invalid, fall back to wind speed
    if np.isnan(InterPcurrent[DATA_CURRENT["gust"]]):
        InterPcurrent[DATA_CURRENT["gust"]] = InterPcurrent[DATA_CURRENT["wind"]]

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

    return CurrentSection(
        currently=currently,
        interp_current=InterPcurrent,
        summary_key=current_summary_key,
    )
