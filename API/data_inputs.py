import logging

import metpy as mp
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint

from API.api_utils import estimate_visibility_gultepe_rh_pr_numpy
from API.constants.model_const import (
    DWD_MOSMIX,
    ECMWF,
    ERA5,
    GEFS,
    GFS,
    HRRR,
    NBM,
    NBM_FIRE_INDEX,
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
    result = np.full(target_shape, np.nan, dtype=np.result_type(array.dtype, np.float64))
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


# Pre-define priority orders to avoid recreating lists
_PRIORITY_ORDER_NA_WITH_ECMWF = ["nbm", "hrrr", "ecmwf", "gfs", "dwd_mosmix", "era5"]
_PRIORITY_ORDER_NA_NO_ECMWF = ["nbm", "hrrr", "gfs", "dwd_mosmix", "era5"]
_PRIORITY_ORDER_ROW_WITH_ECMWF = ["nbm", "hrrr", "dwd_mosmix", "ecmwf", "gfs", "era5"]
_PRIORITY_ORDER_ROW_NO_ECMWF = ["nbm", "hrrr", "dwd_mosmix", "gfs", "era5"]
_PRIORITY_ORDER_AI_NA_WITH_ECMWF = [
    "gefs",
    "gfs",
    "ecmwf",
    "nbm",
    "hrrr",
    "dwd_mosmix",
    "era5",
]
_PRIORITY_ORDER_AI_NA_NO_ECMWF = ["gefs", "gfs", "nbm", "hrrr", "dwd_mosmix", "era5"]
_PRIORITY_ORDER_AI_ROW_WITH_ECMWF = [
    "ecmwf",
    "dwd_mosmix",
    "gfs",
    "nbm",
    "hrrr",
    "era5",
]
_PRIORITY_ORDER_AI_ROW_NO_ECMWF = ["gfs", "dwd_mosmix", "nbm", "hrrr", "era5"]


def _stack_with_priority(
    num_hours, lat, lon, has_ecmwf, source_data, *, prioritize_ai_models=False
):
    """
    Stack fields with priority based on location.

    Args:
        num_hours: Number of hours.
        lat: Latitude.
        lon: Longitude.
        has_ecmwf: Whether ECMWF has data for this variable.
        source_data: Dict mapping source names to data arrays.
        prioritize_ai_models: Whether to prioritize AI model sources in the stacking order.

    Returns:
        Stacked array with sources ordered by priority.
    """
    gfs_before_dwd = should_gfs_precede_dwd(lat, lon)

    # Select pre-defined order based on priority rules
    if prioritize_ai_models and gfs_before_dwd:
        order = (
            _PRIORITY_ORDER_AI_NA_WITH_ECMWF
            if has_ecmwf
            else _PRIORITY_ORDER_AI_NA_NO_ECMWF
        )
    elif prioritize_ai_models and not gfs_before_dwd:
        order = (
            _PRIORITY_ORDER_AI_ROW_WITH_ECMWF
            if has_ecmwf
            else _PRIORITY_ORDER_AI_ROW_NO_ECMWF
        )
    elif gfs_before_dwd:
        # North America: ... > ECMWF > GFS > DWD > ERA5
        order = (
            _PRIORITY_ORDER_NA_WITH_ECMWF if has_ecmwf else _PRIORITY_ORDER_NA_NO_ECMWF
        )
    else:
        # Rest of world: ... > DWD > ECMWF > GFS > ERA5
        order = (
            _PRIORITY_ORDER_ROW_WITH_ECMWF
            if has_ecmwf
            else _PRIORITY_ORDER_ROW_NO_ECMWF
        )

    # Collect arrays in priority order
    arrays = []
    for source in order:
        data = source_data.get(source)
        if data is not None:
            arrays.append(data)

    return _stack_fields(num_hours, *arrays)


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
    prcip_intensity_inputs = {}
    if prioritize_ai_models:
        if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
            prcip_intensity_inputs["ecmwf"] = (
                ecmwf_merged[:, ECMWF["accum_mean"]] * 1000
            )
        # Preserve the legacy key name expected by downstream hourly logic;
        # the value is whichever of GEFS or GFS is available/preferred.
        if "gefs" in source_list and gefs_merged is not None:
            prcip_intensity_inputs["gfs_gefs"] = gefs_merged[:, GEFS["accum"]]
        elif "gfs" in source_list and gfs_merged is not None:
            prcip_intensity_inputs["gfs_gefs"] = gfs_merged[:, GFS["intensity"]] * 3600
        if "nbm" in source_list and nbm_merged is not None:
            prcip_intensity_inputs["nbm"] = nbm_merged[:, NBM["intensity"]]
        if (
            ("hrrr_0-18" in source_list)
            and ("hrrr_18-48" in source_list)
            and (hrrr_merged is not None)
        ):
            prcip_intensity_inputs["hrrr"] = hrrr_merged[:, HRRR["intensity"]] * 3600
    else:
        if "nbm" in source_list and nbm_merged is not None:
            prcip_intensity_inputs["nbm"] = nbm_merged[:, NBM["intensity"]]
        if (
            ("hrrr_0-18" in source_list)
            and ("hrrr_18-48" in source_list)
            and (hrrr_merged is not None)
        ):
            prcip_intensity_inputs["hrrr"] = hrrr_merged[:, HRRR["intensity"]] * 3600
        if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
            # Use the ensemble mean (APCP_Mean) for intensity rather than the
            # deterministic IFS tprate. The deterministic tprate can be zero or
            # near-zero even when the ensemble shows significant precipitation,
            # causing snow accumulation (derived from accum_mean) to appear
            # without a matching precipIntensity. Using accum_mean * 1000 here
            # keeps intensity consistent with the accum field (also accum_mean * 1000).
            prcip_intensity_inputs["ecmwf"] = (
                ecmwf_merged[:, ECMWF["accum_mean"]] * 1000
            )
        if "gefs" in source_list and gefs_merged is not None:
            prcip_intensity_inputs["gfs_gefs"] = gefs_merged[:, GEFS["accum"]]
        elif "gfs" in source_list and gfs_merged is not None:
            prcip_intensity_inputs["gfs_gefs"] = gfs_merged[:, GFS["intensity"]] * 3600

    # DWD MOSMIX: RR1c is in kg/m^2 = mm (hourly total)
    if "dwd_mosmix" in source_list and dwd_valid:
        prcip_intensity_inputs["dwd_mosmix"] = dwd_mosmix_merged[:, DWD_MOSMIX["accum"]]

    era5_rain_intensity = None
    era5_snow_water_equivalent = None

    if "era5" in source_list and era5_valid:
        prcip_intensity_inputs["era5"] = (
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

    _normalize_mapping_lengths(num_hours, prcip_intensity_inputs)
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
    prcip_probability_inputs = {}
    if prioritize_ai_models:
        if "gefs" in source_list and gefs_merged is not None:
            prcip_probability_inputs["gefs"] = gefs_merged[:, GEFS["prob"]]
        if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
            prcip_probability_inputs["ecmwf"] = ecmwf_merged[:, ECMWF["prob"]]
        if "nbm" in source_list and nbm_merged is not None:
            prcip_probability_inputs["nbm"] = nbm_merged[:, NBM["prob"]] * 0.01
    else:
        if "nbm" in source_list and nbm_merged is not None:
            prcip_probability_inputs["nbm"] = nbm_merged[:, NBM["prob"]] * 0.01
        if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
            prcip_probability_inputs["ecmwf"] = ecmwf_merged[:, ECMWF["prob"]]
        if "gefs" in source_list and gefs_merged is not None:
            prcip_probability_inputs["gefs"] = gefs_merged[:, GEFS["prob"]]

    _normalize_mapping_lengths(num_hours, prcip_probability_inputs)

    # --- temperature_inputs ---
    temperature_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        has_ecmwf=True,  # ECMWF has temperature data
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
        has_ecmwf=True,  # ECMWF has dew point data
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
        has_ecmwf=False,  # ECMWF does not provide humidity directly
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
        has_ecmwf=True,  # ECMWF has this data
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
        has_ecmwf=True,  # ECMWF has this data
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
        has_ecmwf=False,  # ECMWF doesn't provide gust data
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
        has_ecmwf=True,  # ECMWF has data
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
        has_ecmwf=True,  # ECMWF has data
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
        has_ecmwf=False,  # ECMWF doesn't provide visibility data
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
    )

    # --- accum_inputs ---
    accum_inputs = _stack_with_priority(
        num_hours,
        lat,
        lon,
        has_ecmwf=True,  # ECMWF has data
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
    fire_inputs = _stack_fields(
        num_hours,
        nbm_fire_merged[:, NBM_FIRE_INDEX] if nbm_fire_merged is not None else None,
    )

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
        has_ecmwf=False,  # ECMWF doesn't provide solar data
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
