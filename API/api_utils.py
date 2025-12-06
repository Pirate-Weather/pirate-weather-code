# %% Script to contain the helper functions as part of the API for Pirate Weather
# Alexander Rey. October 2025
import logging
from typing import List, MutableMapping, Union

import metpy as mp
import numpy as np

from API.constants.api_const import (
    APPARENT_TEMP_CONSTS,
    APPARENT_TEMP_SOLAR_CONSTS,
    PRECIP_NOISE_THRESHOLD_MMH,
)
from API.constants.shared_const import MISSING_DATA

logger = logging.getLogger(__name__)


def fast_nearest_interp(xi, x, y):
    """Performs a fast nearest-neighbor interpolation.

    This function assumes that the input array `x` is monotonically increasing.
    It is sourced from a Stack Overflow answer by Joe Kington.

    Source: https://stackoverflow.com/a/28677914
    License: CC BY-SA 3.0

    Args:
        xi (np.ndarray): The coordinates to evaluate the interpolated values at.
        x (np.ndarray): The data point coordinates, must be monotonically increasing.
        y (np.ndarray): The data point values.

    Returns:
        np.ndarray: The interpolated values, same shape as `xi`.
    """
    # Source - https://stackoverflow.com/a/28677914
    # Posted by Joe Kington
    # Retrieved 2025-11-18, License - CC BY-SA 3.0
    #

    spacing = np.diff(x) / 2
    x = x + np.hstack([spacing, spacing[-1]])
    # Append the last point in y twice for ease of use
    y = np.hstack([y, y[-1]])
    return y[np.searchsorted(x, xi)]


def replace_nan(obj, replacement=MISSING_DATA):
    """Recursively replace np.nan with a given value in a dict/list."""
    if isinstance(obj, dict):
        return {k: replace_nan(v, replacement) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(v, replacement) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return replacement
    else:
        return obj


def calculate_apparent_temperature(air_temp_c, humidity, wind, solar=None):
    """
    Calculates the apparent temperature based on air temperature, wind speed, humidity and solar radiation if provided.

    Parameters:
    - air_temp (float): Air temperature in Celsuis
    - humidity (float): Relative humidity in %
    - wind (float): Wind speed in meters per second

    Returns:
    - float: Apparent temperature in Celsuis
    """

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

    # Return apparent temperature in Celsius
    return apparent_temp_c


def clipLog(data, min_val, max_val, name):
    """
    Clip the data between min and max. Log if there is an error
    """

    # Print if the clipping is larger than 25 of the min
    if np.min(data) < (min_val * 0.75) - 2:
        # Print the data and the index it occurs
        logger.error("Min clipping required for " + name)
        logger.error("Min Value: " + str(np.min(data)))
        if isinstance(data, np.ndarray):
            logger.error("Min Index: " + str(np.where(data == data.min())))

        # Replace values below the threshold with MISSING_DATA
        if np.isscalar(data):
            if data < min_val:
                data = MISSING_DATA
        else:
            data = np.array(data, dtype=float)
            data[data < min_val] = MISSING_DATA

    else:
        data = np.clip(data, a_min=min_val, a_max=None)

    # Same for max
    if np.max(data) > (max_val * 1.25):
        logger.error("Max clipping required for " + name)
        logger.error("Max Value: " + str(np.max(data)))

        # Print the data and the index it occurs
        if isinstance(data, np.ndarray):
            logger.error("Max Index: " + str(np.where(data == data.max())))

        # Replace values above the threshold with MISSING_DATA
        if np.isscalar(data):
            if data > max_val:
                data = MISSING_DATA
        else:
            data = np.array(data, dtype=float)
            data[data > max_val] = MISSING_DATA

    else:
        data = np.clip(data, a_min=None, a_max=max_val)

    return data


## Estimate Visibility for ERA5
# https://journals.ametsoc.org/view/journals/apme/49/1/2009jamc1927.1.xml
# https://ams.confex.com/ams/Madison2006/techprogram/paper_113177.htm
def estimate_visibility_gultepe_rh_pr_numpy(
    arr: np.ndarray,
    var_index: dict,
    var_axis: int = 0,  # 0 => (n_vars, n_time); 1 => (n_time, n_vars)
    use_precip: bool = True,  # set False to ignore PR contribution entirely
    which_rh_fit: str = "FRAM",  # "FRAM" (Eq. 2) or "AIRS2" (Eq. 3)
    params: dict | None = None,
) -> np.ndarray:
    """
    Visibility (km) using Gültepe & Milbrandt (2010):
      - RH→VIS via FRAM/AIRS2 fit
      - PRR→VIS via Table 2 rain-type parameterizations (heavy/moderate/light)

    Requires metpy for RH computation.
    Inputs (1-based indices in var_index):
      - "2m_temperature" (K), "2m_dewpoint_temperature" (K)
    Optional (for PRR in m s^-1 water equivalent):
      - "large_scale_rain_rate", "convective_rain_rate"
      - (Snow not parameterized in Table 2; ignored here by default.)
    """
    try:
        from metpy.calc import relative_humidity_from_dewpoint as _rh_from_td
    except Exception as e:
        raise ImportError("Requires MetPy (`pip install metpy`).") from e

    # ------------------- parameters -------------------
    p = {
        "rh_fit": which_rh_fit,  # "FRAM" or "AIRS2"
        "rh_min": 30.0,  # % clamp lower bound
        "rh_max": 100.0,  # % clamp upper bound
        "vis_min_km": 0.05,
        "vis_max_km": 16.0,
        # Rain-type thresholds per Glickman (2000) used by Gültepe & Milbrandt (2010):
        "pr_light_max": 2.6,  # mm/h
        "pr_moderate_max": 7.6,  # mm/h (heavy > 7.6)
    }
    if params:
        p.update(params)

    # ------------------- helpers -------------------
    def pick(name):
        idx1 = var_index.get(name)
        if idx1 is None:
            return None

        return np.asarray(arr[idx1, ...] if var_axis == 0 else arr[..., idx1])

    def mmh(x):  # kg/m2 s^-1 -> mm h^-1
        return x * 3600.0

    # ------------------- inputs -------------------
    T2m = pick("2m_temperature")  # K
    Td2m = pick("2m_dewpoint_temperature")  # K
    if T2m is None or Td2m is None:
        raise ValueError(
            "Need '2m_temperature' and '2m_dewpoint_temperature' in the array/map."
        )

    # Check if multiple times or 1d
    if T2m.ndim == 0:
        n_time = 1
    else:
        n_time = T2m.shape[0]

    # ------------------- RH via MetPy -------------------
    Rh_frac = _rh_from_td(
        (T2m * mp.units.units.degC), (Td2m * mp.units.units.degC)
    ).magnitude
    RH = np.clip(Rh_frac * 100.0, p["rh_min"], p["rh_max"])  # percent

    # RH → VIS (Gültepe RH fits)
    fit = (p["rh_fit"] or "FRAM").upper()
    if fit == "AIRS2":
        vis_rh = -0.0177 * (RH**2) + 1.462 * RH + 30.8
    else:  # FRAM
        vis_rh = -41.5 * np.log(RH) + 192.3

    beta_rh = 3 / vis_rh

    # ------------------- PRR → VIS (Gültepe Table 2) -------------------
    def _vis_from_pr_gultepe(prr_mm_h: np.ndarray) -> np.ndarray:
        """Apply rain-type thresholds and Table 2 percentile fits."""
        pr = np.clip(prr_mm_h, 0.0, np.inf)
        out = np.full(pr.shape, MISSING_DATA, dtype=float)

        heavy = pr > p["pr_moderate_max"]  # > 7.6 mm/h
        moderate = (pr >= p["pr_light_max"]) & (pr <= p["pr_moderate_max"])  # 2.6–7.6
        light = pr < p["pr_light_max"]  # < 2.6

        # Heavy rain → 5th percentile fit: 0.45*PR^0.394 + 2.28
        out[heavy] = -0.45 * np.power(pr[heavy], 0.394) + 2.28

        # Moderate rain → 50th percentile fit: 2.65*PR^0.256 + 7.65
        out[moderate] = -2.65 * np.power(pr[moderate], 0.256) + 7.65

        # Light rain → 95th percentile fit: 863.26*PR^0.003 + 874.19
        out[light] = -863.26 * np.power(pr[light], 0.003) + 874.19

        return out

    if use_precip:
        ls_rain = pick("large_scale_rain_rate")
        cv_rain = pick("convective_rain_rate")
        if ls_rain is None:
            ls_rain = np.zeros(n_time)
        if cv_rain is None:
            cv_rain = np.zeros(n_time)

        prr_mm_h = mmh(ls_rain + cv_rain)
        vis_pr = _vis_from_pr_gultepe(prr_mm_h)

        beta_pr = 3 / vis_pr
    else:
        beta_pr = np.zeros(n_time)
    # Add the beta factors and convert back to vis
    beta = beta_rh + beta_pr
    vis = 3 / beta

    # Clamp & return
    vis = np.atleast_1d(np.array(vis, dtype=float))
    np.clip(vis, p["vis_min_km"], p["vis_max_km"], out=vis)
    vis = vis * 1000  # Return in m

    return vis[0] if vis.size == 1 else vis


def select_daily_precip_type(
    InterPdaySum: np.ndarray,
    DATA_DAY: dict,
    maxPchanceDay: np.ndarray,
    PRECIP_IDX: dict,
    prepAccumUnit: float,
) -> np.ndarray:
    """Determines the dominant precipitation type for each day.

    This encapsulates the logic used by the API to pick a single "precip type" icon
    for a day based on accumulated rain/snow/ice. It updates and returns
    `maxPchanceDay` (modified in-place) for convenience.

    Args:
        InterPdaySum: 2D array-like (n_days, vars) with accumulated precip amounts.
        DATA_DAY: Mapping of variable names to column indices.
        maxPchanceDay: Array-like of ints that will be updated with precip type codes.
        PRECIP_IDX: Mapping of precipitation type names to integer codes.
        prepAccumUnit: Unit conversion multiplier for accumulation thresholds.

    Returns:
        The updated `maxPchanceDay` numpy array.
    """
    # If rain, snow and ice are all present, choose sleet (code 3)
    all_types = (
        (InterPdaySum[:, DATA_DAY["rain"]] > 0)
        & (InterPdaySum[:, DATA_DAY["snow"]] > 0)
        & (InterPdaySum[:, DATA_DAY["ice"]] > 0)
    )
    maxPchanceDay[all_types] = 3

    # Use the type with the greatest accumulation as baseline
    precip_accum = np.stack(
        [
            InterPdaySum[:, DATA_DAY["rain"]],
            InterPdaySum[:, DATA_DAY["snow"]],
            InterPdaySum[:, DATA_DAY["ice"]],
        ],
        axis=1,
    )

    type_map = np.array([PRECIP_IDX["rain"], PRECIP_IDX["snow"], PRECIP_IDX["ice"]])
    dominant_type = type_map[np.argmax(precip_accum, axis=1)]

    # Only update where not all types are present and some precip exists
    not_all_types = ~all_types
    has_precip = np.max(precip_accum, axis=1) > 0
    update_mask = not_all_types & has_precip
    maxPchanceDay[update_mask] = dominant_type[update_mask]

    # Threshold overrides: large accumulations force their respective type
    maxPchanceDay[InterPdaySum[:, DATA_DAY["rain"]] > (10 * prepAccumUnit)] = (
        PRECIP_IDX["rain"]
    )
    maxPchanceDay[InterPdaySum[:, DATA_DAY["snow"]] > (5 * prepAccumUnit)] = PRECIP_IDX[
        "snow"
    ]
    maxPchanceDay[InterPdaySum[:, DATA_DAY["ice"]] > (1 * prepAccumUnit)] = PRECIP_IDX[
        "ice"
    ]

    return maxPchanceDay


def map_wmo4677_to_ptype(ptype_codes: np.ndarray) -> np.ndarray:
    """
    Map WMO 4677 present-weather codes (50-99) to internal precip type categories.

    Returns an integer array with the following mapping:
        0 -> none/other
        1 -> snow
        2 -> ice (pellets, hail)
        3 -> freezing rain/drizzle
        4 -> rain

    The mapping follows the WMO code ranges and uses conservative grouping:
        - Freezing drizzle/rain codes (56,57,66,67) -> freezing (3)
        - Ice pellets / snow grains / hail-related codes (76-79, 87-90, 96-99) -> ice (2)
        - Snow and snow showers (70-75, 83-86, 93-94) -> snow (1)
        - Rain and drizzle ranges, plus mixed codes -> rain (4)

    Notes:
        - Some WMO codes represent mixed precipitation (e.g., rain+snow). Those are
            mapped to the category most representative for display (rain or snow) depending
            on the code. This function documents the chosen grouping.

    Args:
            ptype_codes: array-like of numeric WMO 4677 codes (may contain NaN)

        Returns:
            np.ndarray of floats (integer category codes as floats) same shape as input
            with MISSING_DATA where input was NaN.
    """
    codes = np.asarray(ptype_codes)
    # Use float dtype so we can store MISSING_DATA (NaN) without conversion errors
    out = np.zeros_like(codes, dtype=float)

    nan_mask = np.isnan(codes)

    # Define code groups (inclusive ranges/lists)
    freezing_codes = [56, 57, 66, 67]
    ice_codes = list(range(76, 80)) + [87, 88, 89, 90, 96, 97, 98, 99]
    snow_codes = list(range(70, 76)) + [83, 84, 85, 86, 93, 94]
    rain_codes = (
        list(range(50, 66)) + [68, 69] + list(range(80, 85)) + list(range(91, 96))
    )

    # Assign categories; order does not matter because groups are disjoint in our choice
    if codes.size > 0:
        vals = codes.copy()
        vals[nan_mask] = -999
        vals = vals.astype(int)

        out[np.isin(vals, snow_codes)] = 1
        out[np.isin(vals, ice_codes)] = 2
        out[np.isin(vals, freezing_codes)] = 3
        out[np.isin(vals, rain_codes)] = 4

    # Use MISSING_DATA for NaNs
    out[nan_mask] = MISSING_DATA

    return out


def zero_small_values(
    array: np.ndarray, threshold: float = PRECIP_NOISE_THRESHOLD_MMH
) -> np.ndarray:
    """Clamp near-zero values to zero to reduce floating noise."""
    array[np.abs(array) < threshold] = 0.0
    return array


# Precomputed constants – built once at import time, not every call
_FIELDS_V_LT_2 = (
    "cape",
    "capeMax",
    "capeMaxTime",
    "currentDayIce",
    "currentDayLiquid",
    "currentDaySnow",
    "dawnTime",
    "duskTime",
    "feelsLike",
    "fireIndex",
    "fireIndexMax",
    "fireIndexMaxTime",
    "iceAccumulation",
    "iceIntensity",
    "iceIntensityMax",
    "liquidAccumulation",
    "liquidIntensityMax",
    "rainIntensity",
    "rainIntensityMax",
    "sleetIntensity",
    "smoke",
    "smokeMax",
    "smokeMaxTime",
    "snowAccumulation",
    "snowIntensity",
    "snowIntensityMax",
    "solar",
    "solarMax",
    "solarMaxTime",
)

_FIELDS_TM_BASIC = (
    "nearestStormDistance",
    "nearestStormBearing",
    "precipProbability",
    "humidity",
    "visibility",
    "uvIndex",
    "uvIndexTime",
    "precipIntensityError",
    "cape",  # overlaps with _FIELDS_V_LT_2
    "solar",  # overlaps with _FIELDS_V_LT_2
)


DictOrList = Union[MutableMapping, List[MutableMapping]]


def remove_conditional_fields(
    data: DictOrList,
    version: float,
    time_machine: bool,
    tm_extra: bool,
) -> DictOrList:
    """Removes output fields based on version and request type.

    This function modifies the input data. It works with either a
    single dictionary or a list of dictionaries.

    Args:
        data (DictOrList): The data to filter, a dict or list of dicts.
        version (float): The API version.
        time_machine (bool): Whether it's a Time Machine request.
        tm_extra (bool): Whether extra Time Machine fields are requested.

    Returns:
        DictOrList: The modified data.
    """
    # Build a *set* to avoid duplicate pops ("cape", "solar" overlap)
    fields_to_remove = set()

    if version < 2:
        fields_to_remove.update(_FIELDS_V_LT_2)

    if time_machine and not tm_extra:
        fields_to_remove.update(_FIELDS_TM_BASIC)

    if not fields_to_remove:
        return data

    # Fast path: list of dicts
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue  # skip non-dicts defensively
            pop = item.pop  # cache bound method to avoid repeated lookups
            for field in fields_to_remove:
                pop(field, None)
    else:
        # Single dict
        if isinstance(data, dict):
            pop = data.pop
            for field in fields_to_remove:
                pop(field, None)

    return data
