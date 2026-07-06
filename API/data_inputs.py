import logging

import metpy as mp
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint

from API.api_utils import estimate_visibility_gultepe_rh_pr_numpy, map_wmo4677_to_ptype
from API.constants.model_const import (
    DWD_MOSMIX,
    ECMWF,
    ERA5,
    GEFS,
    GFS,
    HRRR,
    NBM,
)
from API.utils.source_priority import should_gfs_precede_dwd

logger = logging.getLogger(__name__)


def _stack_fields(num_hours, *arrays):
    """
    Stack valid arrays column-wise.
    If no valid arrays are provided, return a column of NaNs of length num_hours.
    """
    valid = [
        _normalize_length(num_hours, arr, label=f"stack_fields[{idx}]")
        for idx, arr in enumerate(arrays)
        if arr is not None
    ]
    if not valid:
        return np.full((num_hours, 1), np.nan)
    return np.column_stack(valid)


def _normalize_length(num_hours, values, *, label: str | None = None):
    """Pad or truncate an array-like value to the requested hour count."""
    array = np.asarray(values)

    if array.ndim == 0:
        return np.full(num_hours, array, dtype=np.result_type(array.dtype, np.float64))

    if array.shape[0] == num_hours:
        return array

    logger.warning(
        "Normalizing data input length for %s from %s to %s hours; upstream fetch should match request length.",
        label or "unnamed input",
        array.shape[0],
        num_hours,
    )

    target_shape = (num_hours, *array.shape[1:])
    result = np.full(
        target_shape, np.nan, dtype=np.result_type(array.dtype, np.float64)
    )
    copy_len = min(num_hours, array.shape[0])
    result[:copy_len] = array[:copy_len]
    return result


def _normalize_mapping_lengths(num_hours, values_by_key):
    """Normalize all non-null arrays in a mapping to the requested hour count."""
    for key, values in values_by_key.items():
        if values is not None:
            values_by_key[key] = _normalize_length(num_hours, values, label=key)


def _wind_speed(u, v):
    if u is None or v is None:
        return None
    return np.sqrt(u**2 + v**2)


def _bearing(u, v):
    if u is None or v is None:
        return None
    return np.rad2deg(np.mod(np.arctan2(u, v) + np.pi, 2 * np.pi))


# Pre-define priority orders to avoid recreating lists.
# Variables without ECMWF data reuse the same order and skip missing entries.
_PRIORITY_ORDER_NA = ["nbm", "hrrr", "ecmwf", "gfs", "dwd_mosmix", "era5"]
_PRIORITY_ORDER_ROW = ["nbm", "hrrr", "dwd_mosmix", "ecmwf", "gfs", "era5"]
_PRIORITY_ORDER_AI_NA = [
    "gefs",
    "gfs",
    "ecmwf",
    "nbm",
    "hrrr",
    "dwd_mosmix",
    "era5",
]
_PRIORITY_ORDER_AI_ROW = [
    "ecmwf",
    "dwd_mosmix",
    "gfs",
    "nbm",
    "hrrr",
    "era5",
]
_PRECIP_PRIORITY_ORDER_NA = [
    "nbm",
    "hrrr",
    "ecmwf",
    "gefs",
    "gfs",
    "dwd_mosmix",
    "era5",
]
_PRECIP_PRIORITY_ORDER_ROW = [
    "nbm",
    "hrrr",
    "dwd_mosmix",
    "ecmwf",
    "gefs",
    "gfs",
    "era5",
]
_PRECIP_PRIORITY_ORDER_AI_NA = [
    "gefs",
    "gfs",
    "ecmwf",
    "nbm",
    "hrrr",
    "dwd_mosmix",
    "era5",
]
_PRECIP_PRIORITY_ORDER_AI_ROW = [
    "ecmwf",
    "dwd_mosmix",
    "gefs",
    "gfs",
    "nbm",
    "hrrr",
    "era5",
]


def _stack_in_order(num_hours, order, source_data):
    arrays = []
    for source in order:
        data = source_data.get(source)
        if data is not None:
            arrays.append(_normalize_length(num_hours, data, label=source))

    if not arrays:
        return np.full((num_hours, 1), np.nan)

    return np.column_stack(arrays)


def _component_precip_type(snow, ice, freezing_rain, rain):
    if snow is None or ice is None or freezing_rain is None or rain is None:
        return None

    inter_thour = np.zeros((len(snow), 5))
    inter_thour[:, 1] = snow
    inter_thour[:, 2] = ice
    inter_thour[:, 3] = freezing_rain
    inter_thour[:, 4] = rain
    inter_thour[inter_thour < 0.01] = 0

    component_ptype = np.argmax(inter_thour, axis=1).astype(float)
    component_ptype[np.isnan(snow)] = np.nan
    return component_ptype


def _map_grib_precip_type(ptype_values):
    if ptype_values is None:
        return None

    ptype_vals = np.round(np.asarray(ptype_values))
    ptype_nan_mask = np.isnan(ptype_vals)
    ptype_hour = np.zeros_like(ptype_vals, dtype=int)
    ptype_hour[~ptype_nan_mask] = ptype_vals[~ptype_nan_mask].astype(int)
    conditions = [
        np.isin(ptype_hour, [5, 6, 9]),
        np.isin(ptype_hour, [4, 8, 10]),
        np.isin(ptype_hour, [3, 12]),
        np.isin(ptype_hour, [1, 2, 7, 11]),
    ]
    choices = [1, 2, 3, 4]
    mapped_ptype = np.select(conditions, choices, default=0).astype(float)
    mapped_ptype[ptype_nan_mask] = np.nan
    return mapped_ptype


def _stack_precip_with_priority(
    num_hours, lat, lon, source_data, *, prioritize_ai_models=False
):
    gfs_before_dwd = should_gfs_precede_dwd(lat, lon)

    if prioritize_ai_models and gfs_before_dwd:
        order = _PRECIP_PRIORITY_ORDER_AI_NA
    elif prioritize_ai_models and not gfs_before_dwd:
        order = _PRECIP_PRIORITY_ORDER_AI_ROW
    elif gfs_before_dwd:
        order = _PRECIP_PRIORITY_ORDER_NA
    else:
        order = _PRECIP_PRIORITY_ORDER_ROW

    return _stack_in_order(num_hours, order, source_data)


def _stack_with_priority(
    num_hours, lat, lon, source_data, *, prioritize_ai_models=False
):
    """
    Stack fields with priority based on location.

    Args:
        num_hours: Number of hours.
        lat: Latitude.
        lon: Longitude.
        source_data: Dict mapping source names to data arrays.
        prioritize_ai_models: Whether to prioritize AI model sources in the stacking order.

    Returns:
        Stacked array with sources ordered by priority.
    """
    gfs_before_dwd = should_gfs_precede_dwd(lat, lon)

    # Select pre-defined order based on priority rules
    if prioritize_ai_models and gfs_before_dwd:
        order = _PRIORITY_ORDER_AI_NA
    elif prioritize_ai_models and not gfs_before_dwd:
        order = _PRIORITY_ORDER_AI_ROW
    elif gfs_before_dwd:
        # North America: ... > ECMWF > GFS > DWD > ERA5
        order = _PRIORITY_ORDER_NA
    else:
        # Rest of world: ... > DWD > ECMWF > GFS > ERA5
        order = _PRIORITY_ORDER_ROW

    return _stack_in_order(num_hours, order, source_data)


def prepare_data_inputs(
    source_list,
    nbm_merged,
    nbm_fire_merged,
    hrrr_merged,
    dwd_mosmix_merged,
    ecmwf_merged,
    gefs_merged,
    gfs_merged,
    era5_merged,
    extra_vars,
    num_hours,
    lat,
    lon,
    prioritize_ai_models=False,
    is4fires_merged=None,
):
    """
    Prepare data inputs for the hourly block.

    Args:
        source_list: List of available data sources.
        nbm_merged: NBM merged data array.
        nbm_fire_merged: NBM fire merged data array.
        hrrr_merged: HRRR merged data array.
        dwd_mosmix_merged: DWD MOSMIX merged data array.
        ecmwf_merged: ECMWF merged data array.
        gefs_merged: GEFS merged data array.
        gfs_merged: GFS merged data array.
        era5_merged: ERA5 merged data array.
        is4fires_merged: IS4FIRES merged smoke data array.
        extra_vars: List of extra variables to include.
        num_hours: Number of forecast hours.
        lat: Latitude of the forecast location.
        lon: Longitude of the forecast location.
        prioritize_ai_models: Whether to prioritize AI model sources in the stacking order.

    Returns:
        Dictionary containing prepared data inputs for hourly processing.
    """
    # Helper to check if ERA5 is valid (it uses isinstance check in original code)
    era5_valid = isinstance(era5_merged, np.ndarray)
    dwd_valid = isinstance(dwd_mosmix_merged, np.ndarray)

    # --- InterThour_inputs ---
    inter_thour_inputs = {}
    if "nbm" in source_list and nbm_merged is not None:
        inter_thour_inputs["nbm_snow"] = nbm_merged[:, NBM["snow"]]
        inter_thour_inputs["nbm_ice"] = nbm_merged[:, NBM["ice"]]
        inter_thour_inputs["nbm_freezing_rain"] = nbm_merged[:, NBM["freezing_rain"]]
        inter_thour_inputs["nbm_rain"] = nbm_merged[:, NBM["rain"]]

    if (
        ("hrrr_0-18" in source_list)
        and ("hrrr_18-48" in source_list)
        and (hrrr_merged is not None)
    ):
        inter_thour_inputs["hrrr_snow"] = hrrr_merged[:, HRRR["snow"]]
        inter_thour_inputs["hrrr_ice"] = hrrr_merged[:, HRRR["ice"]]
        inter_thour_inputs["hrrr_freezing_rain"] = hrrr_merged[:, HRRR["freezing_rain"]]
        inter_thour_inputs["hrrr_rain"] = hrrr_merged[:, HRRR["rain"]]

    if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
        inter_thour_inputs["ecmwf_ptype"] = ecmwf_merged[:, ECMWF["ptype"]]

    # DWD MOSMIX precipitation type (WMO code)
    if "dwd_mosmix" in source_list and dwd_valid:
        inter_thour_inputs["dwd_mosmix_ptype"] = dwd_mosmix_merged[
            :, DWD_MOSMIX["ptype"]
        ]

    if "gefs" in source_list and gefs_merged is not None:
        inter_thour_inputs["gefs_snow"] = gefs_merged[:, GEFS["snow"]]
        inter_thour_inputs["gefs_ice"] = gefs_merged[:, GEFS["ice"]]
        inter_thour_inputs["gefs_freezing_rain"] = gefs_merged[:, GEFS["freezing_rain"]]
        inter_thour_inputs["gefs_rain"] = gefs_merged[:, GEFS["rain"]]
    elif "gfs" in source_list and gfs_merged is not None:
        inter_thour_inputs["gefs_snow"] = gfs_merged[:, GFS["snow"]]
        inter_thour_inputs["gefs_ice"] = gfs_merged[:, GFS["ice"]]
        inter_thour_inputs["gefs_freezing_rain"] = gfs_merged[:, GFS["freezing_rain"]]
        inter_thour_inputs["gefs_rain"] = gfs_merged[:, GFS["rain"]]

    if "era5" in source_list and era5_valid:
        inter_thour_inputs["era5_ptype"] = era5_merged[:, ERA5["precipitation_type"]]

    _normalize_mapping_lengths(num_hours, inter_thour_inputs)

    # --- prcipIntensity_inputs ---
    prcip_intensity_source_data = {
        "nbm": nbm_merged[:, NBM["intensity"]] if nbm_merged is not None else None,
        "hrrr": hrrr_merged[:, HRRR["intensity"]] * 3600
        if (
            ("hrrr_0-18" in source_list)
            and ("hrrr_18-48" in source_list)
            and (hrrr_merged is not None)
        )
        else None,
        "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["accum"]]
        if "dwd_mosmix" in source_list and dwd_valid
        else None,
        "ecmwf": ecmwf_merged[:, ECMWF["accum_mean"]] * 1000
        if "ecmwf_ifs" in source_list and ecmwf_merged is not None
        else None,
        "gefs": gefs_merged[:, GEFS["accum"]]
        if "gefs" in source_list and gefs_merged is not None
        else None,
        "gfs": gfs_merged[:, GFS["intensity"]] * 3600
        if "gfs" in source_list and gfs_merged is not None
        else None,
    }

    era5_rain_intensity = None
    era5_snow_water_equivalent = None

    if "era5" in source_list and era5_valid:
        prcip_intensity_source_data["era5"] = (
            era5_merged[:, ERA5["large_scale_rain_rate"]]
            + era5_merged[:, ERA5["convective_rain_rate"]]
            + era5_merged[:, ERA5["large_scale_snowfall_rate_water_equivalent"]]
            + era5_merged[:, ERA5["convective_snowfall_rate_water_equivalent"]]
        ) * 3600
        era5_rain_intensity = (
            era5_merged[:, ERA5["large_scale_rain_rate"]]
            + era5_merged[:, ERA5["convective_rain_rate"]]
        ) * 3600
        era5_snow_water_equivalent = (
            era5_merged[:, ERA5["large_scale_snowfall_rate_water_equivalent"]]
            + era5_merged[:, ERA5["convective_snowfall_rate_water_equivalent"]]
        ) * 3600

    prcip_intensity_inputs = _stack_precip_with_priority(
        num_hours,
        lat,
        lon,
        prcip_intensity_source_data,
        prioritize_ai_models=prioritize_ai_models,
    )
    era5_rain_intensity = (
        _normalize_length(num_hours, era5_rain_intensity)
        if era5_rain_intensity is not None
        else None
    )
    era5_snow_water_equivalent = (
        _normalize_length(num_hours, era5_snow_water_equivalent)
        if era5_snow_water_equivalent is not None
        else None
    )

    # --- prcipProbability_inputs ---
    prcip_probability_inputs = _stack_precip_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["prob"]] * 0.01
            if nbm_merged is not None
            else None,
            "ecmwf": ecmwf_merged[:, ECMWF["prob"]]
            if "ecmwf_ifs" in source_list and ecmwf_merged is not None
            else None,
            "gefs": gefs_merged[:, GEFS["prob"]]
            if "gefs" in source_list and gefs_merged is not None
            else None,
            "era5": era5_merged[:, ERA5["prob"]] * 0.01
            if "era5" in source_list and era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- prcipType_inputs ---
    prcip_type_inputs = _stack_precip_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": _component_precip_type(
                nbm_merged[:, NBM["snow"]] if nbm_merged is not None else None,
                nbm_merged[:, NBM["ice"]] if nbm_merged is not None else None,
                nbm_merged[:, NBM["freezing_rain"]] if nbm_merged is not None else None,
                nbm_merged[:, NBM["rain"]] if nbm_merged is not None else None,
            )
            if "nbm" in source_list
            else None,
            "hrrr": _component_precip_type(
                hrrr_merged[:, HRRR["snow"]] if hrrr_merged is not None else None,
                hrrr_merged[:, HRRR["ice"]] if hrrr_merged is not None else None,
                hrrr_merged[:, HRRR["freezing_rain"]]
                if hrrr_merged is not None
                else None,
                hrrr_merged[:, HRRR["rain"]] if hrrr_merged is not None else None,
            )
            if (
                ("hrrr_0-18" in source_list)
                and ("hrrr_18-48" in source_list)
                and (hrrr_merged is not None)
            )
            else None,
            "dwd_mosmix": map_wmo4677_to_ptype(
                np.round(dwd_mosmix_merged[:, DWD_MOSMIX["ptype"]]),
                temperature_c=dwd_mosmix_merged[:, DWD_MOSMIX["temp"]],
            )
            if "dwd_mosmix" in source_list and dwd_valid
            else None,
            "ecmwf": _map_grib_precip_type(
                ecmwf_merged[:, ECMWF["ptype"]] if ecmwf_merged is not None else None
            )
            if "ecmwf_ifs" in source_list
            else None,
            "gefs": _component_precip_type(
                gefs_merged[:, GEFS["snow"]] if gefs_merged is not None else None,
                gefs_merged[:, GEFS["ice"]] if gefs_merged is not None else None,
                gefs_merged[:, GEFS["freezing_rain"]]
                if gefs_merged is not None
                else None,
                gefs_merged[:, GEFS["rain"]] if gefs_merged is not None else None,
            )
            if "gefs" in source_list and gefs_merged is not None
            else None,
            "gfs": _component_precip_type(
                gfs_merged[:, GFS["snow"]] if gfs_merged is not None else None,
                gfs_merged[:, GFS["ice"]] if gfs_merged is not None else None,
                gfs_merged[:, GFS["freezing_rain"]] if gfs_merged is not None else None,
                gfs_merged[:, GFS["rain"]] if gfs_merged is not None else None,
            )
            if "gfs" in source_list and gfs_merged is not None
            else None,
            "era5": _map_grib_precip_type(
                era5_merged[:, ERA5["precipitation_type"]] if era5_valid else None
            )
            if "era5" in source_list and era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- temperature_inputs ---
    temperature_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["temp"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["temp"]] if hrrr_merged is not None else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["temp"]]
            if dwd_valid
            else None,
            "ecmwf": ecmwf_merged[:, ECMWF["temp"]]
            if ecmwf_merged is not None
            else None,
            "gfs": gfs_merged[:, GFS["temp"]] if gfs_merged is not None else None,
            "era5": era5_merged[:, ERA5["2m_temperature"]] if era5_valid else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- dew_inputs ---
    dew_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["dew"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["dew"]] if hrrr_merged is not None else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["dew"]]
            if dwd_valid
            else None,
            "ecmwf": ecmwf_merged[:, ECMWF["dew"]]
            if ecmwf_merged is not None
            else None,
            "gfs": gfs_merged[:, GFS["dew"]] if gfs_merged is not None else None,
            "era5": era5_merged[:, ERA5["2m_dewpoint_temperature"]]
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- humidity_inputs ---
    # Note: ECMWF doesn't provide humidity directly
    # In North America, DWD MOSMIX should be below GFS for this variable
    era5_humidity = None
    if era5_valid:
        era5_humidity = (
            relative_humidity_from_dewpoint(
                era5_merged[:, ERA5["2m_temperature"]] * mp.units.units.degC,
                era5_merged[:, ERA5["2m_dewpoint_temperature"]] * mp.units.units.degC,
                phase="auto",
            ).magnitude
            * 100
        )

    humidity_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["humidity"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["humidity"]]
            if hrrr_merged is not None
            else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["humidity"]]
            if dwd_valid
            else None,
            "gfs": gfs_merged[:, GFS["humidity"]] if gfs_merged is not None else None,
            "era5": era5_humidity,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- pressure_inputs ---
    pressure_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "hrrr": hrrr_merged[:, HRRR["pressure"]]
            if hrrr_merged is not None
            else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["pressure"]]
            if dwd_valid
            else None,
            "ecmwf": ecmwf_merged[:, ECMWF["pressure"]]
            if ecmwf_merged is not None
            else None,
            "gfs": gfs_merged[:, GFS["pressure"]] if gfs_merged is not None else None,
            "era5": era5_merged[:, ERA5["mean_sea_level_pressure"]]
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- wind inputs ---
    wind_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["wind"]] if nbm_merged is not None else None,
            "hrrr": _wind_speed(
                hrrr_merged[:, HRRR["wind_u"]], hrrr_merged[:, HRRR["wind_v"]]
            )
            if hrrr_merged is not None
            else None,
            "dwd_mosmix": _wind_speed(
                dwd_mosmix_merged[:, DWD_MOSMIX["wind_u"]],
                dwd_mosmix_merged[:, DWD_MOSMIX["wind_v"]],
            )
            if dwd_valid
            else None,
            "ecmwf": _wind_speed(
                ecmwf_merged[:, ECMWF["wind_u"]], ecmwf_merged[:, ECMWF["wind_v"]]
            )
            if ecmwf_merged is not None
            else None,
            "gfs": _wind_speed(
                gfs_merged[:, GFS["wind_u"]], gfs_merged[:, GFS["wind_v"]]
            )
            if gfs_merged is not None
            else None,
            "era5": _wind_speed(
                era5_merged[:, ERA5["10m_u_component_of_wind"]],
                era5_merged[:, ERA5["10m_v_component_of_wind"]],
            )
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- gust_inputs ---
    gust_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["gust"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["gust"]] if hrrr_merged is not None else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["gust"]]
            if dwd_valid
            else None,
            "gfs": gfs_merged[:, GFS["gust"]] if gfs_merged is not None else None,
            "era5": era5_merged[:, ERA5["instantaneous_10m_wind_gust"]]
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- bearing_inputs ---
    bearing_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["bearing"]] if nbm_merged is not None else None,
            "hrrr": _bearing(
                hrrr_merged[:, HRRR["wind_u"]], hrrr_merged[:, HRRR["wind_v"]]
            )
            if hrrr_merged is not None
            else None,
            "dwd_mosmix": _bearing(
                dwd_mosmix_merged[:, DWD_MOSMIX["wind_u"]],
                dwd_mosmix_merged[:, DWD_MOSMIX["wind_v"]],
            )
            if dwd_valid
            else None,
            "ecmwf": _bearing(
                ecmwf_merged[:, ECMWF["wind_u"]], ecmwf_merged[:, ECMWF["wind_v"]]
            )
            if ecmwf_merged is not None
            else None,
            "gfs": _bearing(gfs_merged[:, GFS["wind_u"]], gfs_merged[:, GFS["wind_v"]])
            if gfs_merged is not None
            else None,
            "era5": _bearing(
                era5_merged[:, ERA5["10m_u_component_of_wind"]],
                era5_merged[:, ERA5["10m_v_component_of_wind"]],
            )
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- cloud_inputs ---
    cloud_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["cloud"]] * 0.01
            if nbm_merged is not None
            else None,
            "hrrr": hrrr_merged[:, HRRR["cloud"]] * 0.01
            if hrrr_merged is not None
            else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["cloud"]] * 0.01
            if dwd_valid
            else None,
            "ecmwf": ecmwf_merged[:, ECMWF["cloud"]] * 0.01
            if ecmwf_merged is not None
            else None,
            "gfs": gfs_merged[:, GFS["cloud"]] * 0.01
            if gfs_merged is not None
            else None,
            "era5": era5_merged[:, ERA5["total_cloud_cover"]] if era5_valid else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- uv_inputs ---
    uv_inputs = _stack_fields(
        num_hours,
        (gfs_merged[:, GFS["uv"]] * 18.9 * 0.025) if gfs_merged is not None else None,
        (
            era5_merged[:, ERA5["downward_uv_radiation_at_the_surface"]]
            / 3600
            * 40
            * 0.0025
        )
        if era5_valid
        else None,
    )

    # --- vis_inputs ---
    vis_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["vis"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["vis"]] if hrrr_merged is not None else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["vis"]]
            if dwd_valid
            else None,
            "gfs": gfs_merged[:, GFS["vis"]] if gfs_merged is not None else None,
            "era5": estimate_visibility_gultepe_rh_pr_numpy(
                era5_merged, var_index=ERA5, var_axis=1
            )
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- ozone_inputs ---
    ozone_inputs = _stack_fields(
        num_hours,
        gfs_merged[:, GFS["ozone"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["total_column_ozone"]] * 46696 if era5_valid else None,
    )

    # --- smoke_inputs ---
    smoke_inputs = _stack_fields(
        num_hours,
        hrrr_merged[:, HRRR["smoke"]] if hrrr_merged is not None else None,
        is4fires_merged[:, 1] if is4fires_merged is not None else None,
    )

    # --- accum_inputs ---
    accum_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["intensity"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["accum"]] if hrrr_merged is not None else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["accum"]]
            if dwd_valid
            else None,  # kg/m^2 = mm
            "ecmwf": ecmwf_merged[:, ECMWF["accum_mean"]] * 1000
            if ecmwf_merged is not None
            else None,
            "gefs": gefs_merged[:, GEFS["accum"]] if gefs_merged is not None else None,
            "gfs": gfs_merged[:, GFS["accum"]] if gfs_merged is not None else None,
            "era5": era5_merged[:, ERA5["total_precipitation"]] * 1000
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- nearstorm_inputs ---
    nearstorm_inputs = {
        "dist": _stack_fields(
            num_hours,
            np.maximum(gfs_merged[:, GFS["storm_dist"]], 0)
            if gfs_merged is not None
            else None,
        ),
        "dir": _stack_fields(
            num_hours,
            gfs_merged[:, GFS["storm_dir"]] if gfs_merged is not None else None,
        ),
    }

    # --- station_pressure_inputs ---
    station_pressure_inputs = None
    if "stationPressure" in extra_vars:
        station_pressure_inputs = _stack_fields(
            num_hours,
            gfs_merged[:, GFS["station_pressure"]] if gfs_merged is not None else None,
            era5_merged[:, ERA5["surface_pressure"]] if era5_valid else None,
        )

    # --- fire_inputs ---
    fire_inputs = np.full((num_hours, 1), np.nan)

    # --- feels_like_inputs ---
    feels_like_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["apparent"]] if nbm_merged is not None else None,
        gfs_merged[:, GFS["apparent"]] if gfs_merged is not None else None,
    )

    # --- solar_inputs ---
    solar_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        source_data={
            "nbm": nbm_merged[:, NBM["solar"]] if nbm_merged is not None else None,
            "hrrr": hrrr_merged[:, HRRR["solar"]] if hrrr_merged is not None else None,
            "dwd_mosmix": dwd_mosmix_merged[:, DWD_MOSMIX["solar"]]
            if dwd_valid
            else None,
            "gfs": gfs_merged[:, GFS["solar"]] if gfs_merged is not None else None,
            "era5": era5_merged[:, ERA5["surface_solar_radiation_downwards"]] / 3600
            if era5_valid
            else None,
        },
        prioritize_ai_models=prioritize_ai_models,
    )

    # --- cape_inputs ---
    cape_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["cape"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["cape"]] if hrrr_merged is not None else None,
        gfs_merged[:, GFS["cape"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["convective_available_potential_energy"]]
        if era5_valid
        else None,
    )

    # --- error_inputs ---
    error_inputs = _stack_fields(
        num_hours,
        ecmwf_merged[:, ECMWF["accum_stddev"]] * 1000
        if ecmwf_merged is not None
        else None,
        gefs_merged[:, GEFS["error"]] if gefs_merged is not None else None,
    )

    return {
        "InterThour_inputs": inter_thour_inputs,
        "prcipIntensity_inputs": prcip_intensity_inputs,
        "prcipProbability_inputs": prcip_probability_inputs,
        "prcipType_inputs": prcip_type_inputs,
        "temperature_inputs": temperature_inputs,
        "dew_inputs": dew_inputs,
        "humidity_inputs": humidity_inputs,
        "pressure_inputs": pressure_inputs,
        "wind_inputs": wind_inputs,
        "gust_inputs": gust_inputs,
        "bearing_inputs": bearing_inputs,
        "cloud_inputs": cloud_inputs,
        "uv_inputs": uv_inputs,
        "vis_inputs": vis_inputs,
        "ozone_inputs": ozone_inputs,
        "smoke_inputs": smoke_inputs,
        "accum_inputs": accum_inputs,
        "nearstorm_inputs": nearstorm_inputs,
        "station_pressure_inputs": station_pressure_inputs,
        "era5_rain_intensity": era5_rain_intensity,
        "era5_snow_water_equivalent": era5_snow_water_equivalent,
        "fire_inputs": fire_inputs,
        "feels_like_inputs": feels_like_inputs,
        "solar_inputs": solar_inputs,
        "cape_inputs": cape_inputs,
        "error_inputs": error_inputs,
    }
