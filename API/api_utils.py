# %% Script to contain the helper functions as part of the API for Pirate Weather
# Alexander Rey. October 2025
import logging

import numpy as np
import xarray as xr

from API.constants.api_const import APPARENT_TEMP_CONSTS, APPARENT_TEMP_SOLAR_CONSTS
from API.constants.clip_const import CLIP_TEMP
from API.constants.shared_const import KELVIN_TO_CELSIUS

logger = logging.getLogger(__name__)


def calculate_apparent_temperature(air_temp, humidity, wind, solar=None):
    """
    Calculates the apparent temperature based on air temperature, wind speed, humidity and solar radiation if provided.

    Parameters:
    - air_temp (float): Air temperature in Celsuis
    - humidity (float): Relative humidity in %
    - wind (float): Wind speed in meters per second

    Returns:
    - float: Apparent temperature in Kelvin
    """

    # Convert air_temp from Kelvin to Celsius for the formula parts that use Celsius
    air_temp_c = air_temp - KELVIN_TO_CELSIUS

    # Calculate water vapor pressure 'e'
    # Ensure humidity is not 0 for calculation, replace with a small non-zero value if needed
    # The original equation does not guard for zero humidity. If relative_humidity_0_1 is 0, e will be 0.
    e = (
        humidity
        * APPARENT_TEMP_CONSTS["e_const"]
        * np.exp(
            APPARENT_TEMP_CONSTS["exp_a"]
            * air_temp_c
            / (APPARENT_TEMP_CONSTS["exp_b"] + air_temp_c)
        )
    )

    if solar is None:
        # Calculate apparent temperature in Celsius
        apparent_temp_c = (
            air_temp_c
            + APPARENT_TEMP_CONSTS["humidity_factor"] * e
            - APPARENT_TEMP_CONSTS["wind_factor"] * wind
            + APPARENT_TEMP_CONSTS["const"]
        )
    else:
        # Calculate the effective solar term 'q' used in the apparent temperature formula.
        # The model's `solar` value is Downward Short-Wave Radiation Flux in W/m^2.
        # `q_factor` scales that irradiance to the empirical Q used in the formula
        # (for example q_factor=0.1 reduces the raw W/m^2 to a smaller effective value).
        # Tuning `q_factor` controls how strongly solar irradiance influences apparent temp.
        q = solar * APPARENT_TEMP_SOLAR_CONSTS["q_factor"]

        # Calculate apparent temperature in Celsius using solar radiation
        apparent_temp_c = (
            air_temp_c
            + APPARENT_TEMP_SOLAR_CONSTS["humidity_factor"] * e
            - APPARENT_TEMP_SOLAR_CONSTS["wind_factor"] * wind
            + (APPARENT_TEMP_SOLAR_CONSTS["solar_factor"] * q) / (wind + 10)
            + APPARENT_TEMP_SOLAR_CONSTS["const"]
        )

    # Convert back to Kelvin
    apparent_temp_k = apparent_temp_c + KELVIN_TO_CELSIUS

    # Clip between -90 and 60
    return clipLog(
        apparent_temp_k,
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )


def clipLog(data, min_val, max_val, name):
    """
    Clip the data between min and max. Log if there is an error
    """

    # Print if the clipping is larger than 25 of the min
    if np.min(data) < (min_val * 0.75):
        # Print the data and the index it occurs
        logger.error("Min clipping required for " + name)
        logger.error("Min Value: " + str(np.min(data)))
        if isinstance(data, np.ndarray):
            logger.error("Min Index: " + str(np.where(data == data.min())))

        # Replace values below the threshold with np.nan
        if np.isscalar(data):
            if data < min_val:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data < min_val] = np.nan

    else:
        data = np.clip(data, a_min=min_val, a_max=None)

    # Same for max
    if np.max(data) > (max_val * 1.25):
        logger.error("Max clipping required for " + name)
        logger.error("Max Value: " + str(np.max(data)))

        # Print the data and the index it occurs
        if isinstance(data, np.ndarray):
            logger.error("Max Index: " + str(np.where(data == data.max())))

        # Replace values above the threshold with np.nan
        if np.isscalar(data):
            if data > max_val:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data > max_val] = np.nan

    else:
        data = np.clip(data, a_min=None, a_max=max_val)

    return data



## Estimate Visibility for ERA5
# https://chatgpt.com/share/68fa2b7f-0840-800d-b90e-81f5edb4bc90
def estimate_visibility_from_numpy(
    arr: np.ndarray,
    var_index: dict,
    var_axis: int = 0,   # 0 => shape (n_vars, n_time); set to 1 if (n_time, n_vars)
    params: dict | None = None,
) -> np.ndarray:
    """
    Heuristic visibility proxy (km) from a NumPy array built from ERA5 variables.

    Parameters
    ----------
    arr : np.ndarray
        Array with variables stacked along `var_axis` and time along the other axis.
        Typically shape = (n_vars, n_time).
    var_index : dict[str,int]
        Mapping from variable name -> 1-based index in `arr` (as in your ERA5 dict).
    var_axis : int
        Axis along which variables are stacked (0 or 1).
    params : dict
        Optional thresholds/parameters override.

    Returns
    -------
    np.ndarray
        Visibility in km, shape = (n_time,)
    """
    # Thresholds/parameters (same logic as xarray version)
    p = {
        # Fog/haze
        "dpd_fog_K": 1.0,
        "dpd_haze_K": 2.0,
        "lcc_fog_min": 0.6,
        "no_precip_mm_h": 0.1,
        # Precip→vis bins (mm/h)
        "rain_bins_mm_h":  np.array([0.5, 5.0, 20.0]),
        "rain_vis_km":     np.array([10.0, 8.0, 3.0, 1.0]),
        "snow_bins_mm_h":  np.array([0.1, 1.0, 5.0]),
        "snow_vis_km":     np.array([10.0, 5.0, 2.0, 0.8]),
        # Clear sky vis (km)
        "vis_fog_km": 0.5,
        "vis_haze_km": 5.0,
        "vis_clear_km": 10.0,
        # Ceiling caps
        "cap1_cbase_m": 50,   "cap1_vis_km": 1.0,
        "cap2_cbase_m": 100,  "cap2_vis_km": 3.0,
        # Wind relaxation
        "wind_relax_ms": 8.0,
        "wind_relax_min_vis_km": 2.0,
        # Output clamp
        "min_vis_km": 0.2,
        "max_vis_km": 16.0,
    }
    if params:
        p.update(params)

    # Helper to get a variable vector (time,) from index mapping
    def get(name):
        idx1 = var_index.get(name)
        if idx1 is None:
            return None
        if var_axis == 0:
            out = arr[idx1, ...]
        else:
            out = arr[..., idx1]
        # Ensure shape (time,)
        return np.asarray(out)

    # Pull what we need (some optional)
    t2m  = get("2m_temperature")
    td2m = get("2m_dewpoint_temperature")
    if t2m is None or td2m is None:
        raise ValueError("Need both '2m_temperature' and '2m_dewpoint_temperature' in var_index.")

    lcc   = get("low_cloud_cover")
    cbase = get("cloud_base_height")

    u10   = get("10m_u_component_of_wind")
    v10   = get("10m_v_component_of_wind")

    # Rates in kg/m2 s-1 → mm h-1
    def mmh(x):
        return x * 3600.0

    ls_rain = get("large_scale_rain_rate")
    cv_rain = get("convective_rain_rate")
    ls_snow = get("large_scale_snowfall_rate_water_equivalent")
    cv_snow = get("convective_snowfall_rate_water_equivalent")

    # Determine time length and make safe defaults
    n_time = t2m.shape[0]
    def zeros(): return np.zeros(n_time, dtype=float)
    def full(v): return np.full(n_time, v, dtype=float)

    if lcc   is None: lcc   = zeros()
    if cbase is None: cbase = full(1e9)
    # Replace nan in cbase with large number
    cbase = np.where(np.isnan(cbase), 1e9, cbase)
    # print('cbase:', cbase)
    if u10   is None: u10   = zeros()
    if v10   is None: v10   = zeros()
    if ls_rain is None: ls_rain = zeros()
    if cv_rain is None: cv_rain = zeros()
    if ls_snow is None: ls_snow = zeros()
    if cv_snow is None: cv_snow = zeros()

    rain_rate_mm_h = mmh(ls_rain + cv_rain)
    snow_rate_mm_h = mmh(ls_snow + cv_snow)

    # --- Precip → visibility via piecewise-constant bins ---
    def binned_vis(rate, edges, values):
        # values length = len(edges)+1
        vis = np.full_like(rate, values[-1], dtype=float)
        # fill from highest edge downward so lower bins overwrite
        for i in range(len(edges)-1, -1, -1):
            vis = np.where(rate <= edges[i], values[i], vis)
        return vis

    vis_rain = binned_vis(rain_rate_mm_h, p["rain_bins_mm_h"], p["rain_vis_km"])
    vis_snow = binned_vis(snow_rate_mm_h, p["snow_bins_mm_h"], p["snow_vis_km"])
    vis_precip = np.minimum(vis_rain, vis_snow)

    # print('vis_rain:', vis_rain)
    # print('vis_snow:', vis_snow)

    # --- Clear-sky component from dewpoint depression + low cloud ---
    dpd = t2m - td2m
    no_precip = (rain_rate_mm_h <= p["no_precip_mm_h"]) & (snow_rate_mm_h <= p["no_precip_mm_h"])
    fog_flag  = (dpd <= p["dpd_fog_K"])  & (lcc >= p["lcc_fog_min"]) & no_precip
    haze_flag = (dpd <= p["dpd_haze_K"]) & no_precip

    vis_clear = np.where(
        fog_flag, p["vis_fog_km"],
        np.where(haze_flag, p["vis_haze_km"], p["vis_clear_km"])
    )

    # print('vis_clear:', vis_clear)

    # --- Combine (more limiting wins) ---
    vis = np.minimum(vis_clear, vis_precip)
    # print("combined vis:", vis)

    # --- Ceiling penalties ---
    vis = np.where(cbase < p["cap1_cbase_m"], np.minimum(vis, p["cap1_vis_km"]), vis)
    # print("after cap1 vis:", vis)
    mid_cap = (cbase >= p["cap1_cbase_m"]) & (cbase < p["cap2_cbase_m"])
    vis = np.where(mid_cap, np.minimum(vis, p["cap2_vis_km"]), vis)
    # print("after cap2 vis:", vis)

    # --- Wind mixing relaxation for dense fog ---
    wind10 = np.hypot(u10, v10)
    relax = (wind10 >= p["wind_relax_ms"]) & fog_flag
    vis = np.where(relax, np.maximum(vis, p["wind_relax_min_vis_km"]), vis)
    # print("after wind relax vis:", vis)

    # --- Clamp & return ---
    np.clip(vis, p["min_vis_km"], p["max_vis_km"], out=vis)
    vis = vis * 1000 # Return in m

    return vis
