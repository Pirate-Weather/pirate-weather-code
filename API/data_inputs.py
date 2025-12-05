import metpy as mp
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint

from API.api_utils import estimate_visibility_gultepe_rh_pr_numpy
from API.constants.model_const import (
    ECMWF,
    ERA5,
    GEFS,
    GFS,
    HRRR,
    NBM,
    NBM_FIRE_INDEX,
)


def _stack_fields(num_hours, *arrays):
    """
    Stack valid arrays column-wise.
    If no valid arrays are provided, return a column of NaNs of length num_hours.
    """
    valid = [np.asarray(arr) for arr in arrays if arr is not None]
    if not valid:
        return np.full((num_hours, 1), np.nan)
    return np.column_stack(valid)


def _wind_speed(u, v):
    if u is None or v is None:
        return None
    return np.sqrt(u**2 + v**2)


def _bearing(u, v):
    if u is None or v is None:
        return None
    return np.rad2deg(np.mod(np.arctan2(u, v) + np.pi, 2 * np.pi))


def prepare_data_inputs(
    source_list,
    nbm_merged,
    nbm_fire_merged,
    hrrr_merged,
    ecmwf_merged,
    gefs_merged,
    gfs_merged,
    era5_merged,
    extra_vars,
    num_hours,
):
    """
    Prepare data inputs for the hourly block.
    """
    # Helper to check if ERA5 is valid (it uses isinstance check in original code)
    era5_valid = isinstance(era5_merged, np.ndarray)

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

    # --- prcipIntensity_inputs ---
    prcip_intensity_inputs = {}
    if "nbm" in source_list and nbm_merged is not None:
        prcip_intensity_inputs["nbm"] = nbm_merged[:, NBM["intensity"]]

    if (
        ("hrrr_0-18" in source_list)
        and ("hrrr_18-48" in source_list)
        and (hrrr_merged is not None)
    ):
        prcip_intensity_inputs["hrrr"] = hrrr_merged[:, HRRR["intensity"]] * 3600

    if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
        prcip_intensity_inputs["ecmwf"] = ecmwf_merged[:, ECMWF["intensity"]] * 3600

    if "gefs" in source_list and gefs_merged is not None:
        prcip_intensity_inputs["gfs_gefs"] = gefs_merged[:, GEFS["accum"]]
    elif "gfs" in source_list and gfs_merged is not None:
        prcip_intensity_inputs["gfs_gefs"] = gfs_merged[:, GFS["intensity"]] * 3600

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

    # --- prcipProbability_inputs ---
    prcip_probability_inputs = {}
    if "nbm" in source_list and nbm_merged is not None:
        prcip_probability_inputs["nbm"] = nbm_merged[:, NBM["prob"]] * 0.01
    if "ecmwf_ifs" in source_list and ecmwf_merged is not None:
        prcip_probability_inputs["ecmwf"] = ecmwf_merged[:, ECMWF["prob"]]
    if "gefs" in source_list and gefs_merged is not None:
        prcip_probability_inputs["gefs"] = gefs_merged[:, GEFS["prob"]]

    # --- temperature_inputs ---
    temperature_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["temp"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["temp"]] if hrrr_merged is not None else None,
        ecmwf_merged[:, ECMWF["temp"]] if ecmwf_merged is not None else None,
        gfs_merged[:, GFS["temp"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["2m_temperature"]] if era5_valid else None,
    )

    # --- dew_inputs ---
    dew_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["dew"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["dew"]] if hrrr_merged is not None else None,
        ecmwf_merged[:, ECMWF["dew"]] if ecmwf_merged is not None else None,
        gfs_merged[:, GFS["dew"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["2m_dewpoint_temperature"]] if era5_valid else None,
    )

    # --- humidity_inputs ---
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

    humidity_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["humidity"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["humidity"]] if hrrr_merged is not None else None,
        gfs_merged[:, GFS["humidity"]] if gfs_merged is not None else None,
        era5_humidity,
    )

    # --- pressure_inputs ---
    pressure_inputs = _stack_fields(
        num_hours,
        hrrr_merged[:, HRRR["pressure"]] if hrrr_merged is not None else None,
        ecmwf_merged[:, ECMWF["pressure"]] if ecmwf_merged is not None else None,
        gfs_merged[:, GFS["pressure"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["mean_sea_level_pressure"]] if era5_valid else None,
    )

    # --- wind_inputs ---
    wind_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["wind"]] if nbm_merged is not None else None,
        _wind_speed(hrrr_merged[:, HRRR["wind_u"]], hrrr_merged[:, HRRR["wind_v"]])
        if hrrr_merged is not None
        else None,
        _wind_speed(ecmwf_merged[:, ECMWF["wind_u"]], ecmwf_merged[:, ECMWF["wind_v"]])
        if ecmwf_merged is not None
        else None,
        _wind_speed(gfs_merged[:, GFS["wind_u"]], gfs_merged[:, GFS["wind_v"]])
        if gfs_merged is not None
        else None,
        _wind_speed(
            era5_merged[:, ERA5["10m_u_component_of_wind"]],
            era5_merged[:, ERA5["10m_v_component_of_wind"]],
        )
        if era5_valid
        else None,
    )

    # --- gust_inputs ---
    gust_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["gust"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["gust"]] if hrrr_merged is not None else None,
        gfs_merged[:, GFS["gust"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["instantaneous_10m_wind_gust"]] if era5_valid else None,
    )

    # --- bearing_inputs ---
    bearing_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["bearing"]] if nbm_merged is not None else None,
        _bearing(hrrr_merged[:, HRRR["wind_u"]], hrrr_merged[:, HRRR["wind_v"]])
        if hrrr_merged is not None
        else None,
        _bearing(ecmwf_merged[:, ECMWF["wind_u"]], ecmwf_merged[:, ECMWF["wind_v"]])
        if ecmwf_merged is not None
        else None,
        _bearing(gfs_merged[:, GFS["wind_u"]], gfs_merged[:, GFS["wind_v"]])
        if gfs_merged is not None
        else None,
        _bearing(
            era5_merged[:, ERA5["10m_u_component_of_wind"]],
            era5_merged[:, ERA5["10m_v_component_of_wind"]],
        )
        if era5_valid
        else None,
    )

    # --- cloud_inputs ---
    cloud_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["cloud"]] * 0.01 if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["cloud"]] * 0.01 if hrrr_merged is not None else None,
        ecmwf_merged[:, ECMWF["cloud"]] * 0.01 if ecmwf_merged is not None else None,
        gfs_merged[:, GFS["cloud"]] * 0.01 if gfs_merged is not None else None,
        era5_merged[:, ERA5["total_cloud_cover"]] if era5_valid else None,
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
    vis_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["vis"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["vis"]] if hrrr_merged is not None else None,
        gfs_merged[:, GFS["vis"]] if gfs_merged is not None else None,
        estimate_visibility_gultepe_rh_pr_numpy(era5_merged, var_index=ERA5, var_axis=1)
        if era5_valid
        else None,
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
    accum_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["intensity"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["accum"]] if hrrr_merged is not None else None,
        ecmwf_merged[:, ECMWF["accum_mean"]] * 1000
        if ecmwf_merged is not None
        else None,
        gefs_merged[:, GEFS["accum"]] if gefs_merged is not None else None,
        gfs_merged[:, GFS["accum"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["total_precipitation"]] * 1000 if era5_valid else None,
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
    solar_inputs = _stack_fields(
        num_hours,
        nbm_merged[:, NBM["solar"]] if nbm_merged is not None else None,
        hrrr_merged[:, HRRR["solar"]] if hrrr_merged is not None else None,
        gfs_merged[:, GFS["solar"]] if gfs_merged is not None else None,
        era5_merged[:, ERA5["surface_solar_radiation_downwards"]] / 3600
        if era5_valid
        else None,
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
