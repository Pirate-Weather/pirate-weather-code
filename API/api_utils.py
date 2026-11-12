# %% Script to contain the helper functions as part of the API for Pirate Weather
# Alexander Rey. October 2025
import logging
import math
from collections.abc import MutableMapping

import metpy as mp
import numpy as np
from metpy.calc.thermo import (
    _saturation_vapor_pressure_liquid,
    _saturation_vapor_pressure_solid,
)
from metpy.constants import nounit as metpy_nounit

from API.constants.api_const import (
    APPARENT_TEMP_CONSTS,
    APPARENT_TEMP_SOLAR_CONSTS,
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    PRECIP_TYPES,
    TEMP_THRESHOLD_WMO_FROZEN_C,
    WET_BULB_RAIN_SNOW_THRESHOLD_C,
)
from API.constants.shared_const import MISSING_DATA

logger = logging.getLogger(__name__)

# metpy's degC -> kelvin conversion, exactly (see relative_humidity_from_dewpoint_c)
KELVIN_OFFSET = 273.15

# Visibility estimate cap: 10 statute miles, the top of the observed and
# modelled visibility other sources report (CLIP_VIS clips to 16,090 m)
VIS_ESTIMATE_MAX_KM = 16.09344
# Gültepe & Milbrandt (2010) Table 2 rain intensity classes (mm/h): light below
# 2.6, moderate to 7.6, heavy above
GULTEPE_LIGHT_RAIN_MAX_MMH = 2.6
GULTEPE_MODERATE_RAIN_MAX_MMH = 7.6
# Snow extinction of Stoelinga & Warner (2003), as in the NCEP UPP: beta (km-1)
# = 10.36 * C**0.7776, C the snow mass concentration (g m-3). C comes from the
# liquid-equivalent rate over a typical aggregate fall speed (Locatelli & Hobbs
# 1974: 0.5-1.5 m/s). Their beta is for a 2% contrast threshold (ln 50 = 3.912).
SNOW_EXTINCTION_COEFFICIENT = 10.36
SNOW_EXTINCTION_EXPONENT = 0.7776
SNOW_FALL_SPEED_M_S = 1.0
STOELINGA_WARNER_CONTRAST_CONSTANT = 3.912
# Koschmieder constant of the Gultepe fits used here (5% contrast threshold)
VIS_KOSCHMIEDER_CONSTANT = 3.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
GRAMS_PER_KG = 1000.0

# Clear-sky UV index of Allaart et al. (2004, Meteorol. Appl. 11, 59-65):
# 12.5 with the sun overhead and 300 DU of ozone
UV_CLEAR_SKY_OVERHEAD_INDEX = 12.5
UV_CLEAR_SKY_ZENITH_EXPONENT = 2.42
UV_CLEAR_SKY_OZONE_EXPONENT = -1.23
UV_REFERENCE_OZONE_DU = 300.0
# Total ozone for models without it: a typical global value (~260 DU in the
# tropics, 300-450 DU at higher latitudes)
UV_TYPICAL_OZONE_DU = 300.0
# NWS UV Index cloud attenuation (Long et al. 1996): clear 1.0, scattered
# 0.89, broken 0.73, overcast 0.31, placed at the middle of each class's cloud
# fraction (scattered 3-4 oktas, broken 5-7) and interpolated between
UV_CLOUD_COVER_POINTS = (0.0, 0.4375, 0.75, 1.0)
UV_CLOUD_FACTORS = (1.0, 0.89, 0.73, 0.31)

# Gust factor (3 s gust over the mean 10 m wind) for open, flat country
# (WMO guidelines, Harper et al. 2010: about 1.2-1.3 at sea, 1.4-1.5 over
# open land, higher over rough terrain). CFS's four-member mean wind is lower
# than any member's, which this partly offsets.
CFS_GUST_FACTOR = 1.4


def relative_humidity_from_dewpoint_c(temperature_c, dewpoint_c):
    """Relative humidity (0-1) from air and dew point temperatures in Celsius.

    This is metpy's ``relative_humidity_from_dewpoint(..., phase="auto")``
    (Ambaum 2020 saturation vapour pressure over water above the triple point,
    over ice below) called through its unit-free internals: wrapping two
    scalars in pint quantities costs ~0.3 ms per call on the current-conditions
    path, the arithmetic ~0.03 ms, and the results are bit-identical.
    """
    temperature_k = np.asarray(temperature_c, dtype=float) + KELVIN_OFFSET
    dewpoint_k = np.asarray(dewpoint_c, dtype=float) + KELVIN_OFFSET

    def saturation_vapor_pressure(kelvin):
        return np.where(
            kelvin > metpy_nounit.T0,
            _saturation_vapor_pressure_liquid._nounit(kelvin),
            _saturation_vapor_pressure_solid._nounit(kelvin),
        )

    return saturation_vapor_pressure(dewpoint_k) / saturation_vapor_pressure(
        temperature_k
    )


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
    """Recursively replace np.nan with a given value in a dict/list.

    Returns new containers; the inputs are not modified (the response holds
    shared objects such as the DWD station map entries).
    """
    # Leaf values are checked inline rather than by a call each: a response
    # holds ~5,000 of them.
    if isinstance(obj, dict):
        return {
            k: (
                (replacement if math.isnan(v) else v)
                if isinstance(v, float)
                else replace_nan(v, replacement)
                if isinstance(v, (dict, list))
                else v
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [
            (replacement if math.isnan(v) else v)
            if isinstance(v, float)
            else replace_nan(v, replacement)
            if isinstance(v, (dict, list))
            else v
            for v in obj
        ]
    elif isinstance(obj, float) and math.isnan(obj):
        return replacement
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

    # Non-solar formula (used for hours/elements where solar is unavailable)
    apparent_no_solar = (
        air_temp_c
        + APPARENT_TEMP_CONSTS["humidity_factor"] * e
        - APPARENT_TEMP_CONSTS["wind_factor"] * wind
        + APPARENT_TEMP_CONSTS["const"]
    )

    if solar is None:
        return apparent_no_solar

    # Solar formula (used where solar is a valid, non-NaN value)
    # The model's `solar` value is Downward Short-Wave Radiation Flux in W/m^2.
    # `q_factor` scales that irradiance to the empirical Q used in the formula.
    q = solar * APPARENT_TEMP_SOLAR_CONSTS["q_factor"]
    apparent_solar = (
        air_temp_c
        + APPARENT_TEMP_SOLAR_CONSTS["humidity_factor"] * e
        - APPARENT_TEMP_SOLAR_CONSTS["wind_factor"] * wind
        + (APPARENT_TEMP_SOLAR_CONSTS["solar_factor"] * q) / (wind + 10)
        + APPARENT_TEMP_SOLAR_CONSTS["const"]
    )

    # Select per-element: use solar formula where solar is valid, non-solar where NaN.
    # np.where handles both scalar and array inputs uniformly.
    return np.where(np.isnan(solar), apparent_no_solar, apparent_solar)


def clipLog(data, min_val, max_val, name):
    """Clip data to min_val and max_val, replacing implausible values as missing.

    A value a little out of range is model noise and is clamped to the limit:
    up to 25 % above a positive maximum, and below a non-negative minimum down
    to 0.75 * min_val - 2 (so tiny negatives from interpolation clamp to 0).
    A negative minimum or a maximum of 0 or below has no margin: anything past
    it becomes MISSING_DATA rather than being clamped (e.g. a feels-like below
    -60 C is missing, not -60). Out-of-range values beyond the margin are bad
    data: only that value becomes MISSING_DATA and is logged, so one bad hour
    doesn't blank plausible values beside it.
    """
    lowest_plausible = min(min_val, min_val * 0.75 - 2)
    highest_plausible = max(max_val, max_val * 1.25)

    # The current section clips about 18 scalars per request, and the array path
    # below is five numpy calls each. NaN passes through, as it does in np.clip.
    if isinstance(data, float):
        if math.isnan(data):
            return np.float64(data)
        if lowest_plausible <= data <= highest_plausible:
            if data < min_val:
                return np.float64(min_val)
            if data > max_val:
                return np.float64(max_val)
            return np.float64(data)

    clipped = np.clip(data, min_val, max_val)
    too_low = np.less(data, lowest_plausible)
    too_high = np.greater(data, highest_plausible)

    if not (np.any(too_low) or np.any(too_high)):
        return clipped

    if np.any(too_low):
        logger.error("Min clipping required for %s", name)
        logger.error("Min Value: %s", np.nanmin(data))
        logger.error("Min Index: %s", np.where(np.atleast_1d(too_low)))
    if np.any(too_high):
        logger.error("Max clipping required for %s", name)
        logger.error("Max Value: %s", np.nanmax(data))
        logger.error("Max Index: %s", np.where(np.atleast_1d(too_high)))

    if np.isscalar(data):
        return MISSING_DATA
    clipped = np.array(clipped, dtype=float)
    clipped[too_low | too_high] = MISSING_DATA
    return clipped


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
    Visibility (m) from ERA5 rows using Gültepe & Milbrandt (2010):
      - RH→VIS via FRAM/AIRS2 fit
      - PRR→VIS via Table 2 rain-type parameterizations (heavy/moderate/light)

    Inputs (indices in var_index):
      - "2m_temperature" (°C), "2m_dewpoint_temperature" (°C): the API has
        already converted ERA5's kelvin values (convert_data_to_celsius)
    Optional (rain rates in kg m-2 s-1, i.e. mm/s of water):
      - "large_scale_rain_rate", "convective_rain_rate"
    Optional (m/s): "10m_u_component_of_wind", "10m_v_component_of_wind"
      - (Snow not parameterized in Table 2; ignored here by default.)
    """
    # ------------------- parameters -------------------
    p = {
        "rh_fit": which_rh_fit,  # "FRAM" or "AIRS2"
        "rh_min": 30.0,  # % clamp lower bound
        "rh_max": 100.0,  # % clamp upper bound
        "vis_min_km": 0.05,
        "vis_max_km": VIS_ESTIMATE_MAX_KM,
    }
    if params:
        p.update(params)

    # ------------------- helpers -------------------
    def pick(name):
        idx1 = var_index.get(name)
        if idx1 is None:
            return None

        return np.asarray(arr[idx1, ...] if var_axis == 0 else arr[..., idx1])

    # ------------------- inputs -------------------
    T2m = pick("2m_temperature")  # °C
    Td2m = pick("2m_dewpoint_temperature")  # °C
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
    Rh_frac = mp.calc.relative_humidity_from_dewpoint(
        (T2m * mp.units.units.degC), (Td2m * mp.units.units.degC)
    ).magnitude
    RH = np.clip(Rh_frac * 100.0, p["rh_min"], p["rh_max"])

    if use_precip:
        ls_rain = pick("large_scale_rain_rate")
        cv_rain = pick("convective_rain_rate")
        if ls_rain is None:
            ls_rain = np.zeros(n_time)
        if cv_rain is None:
            cv_rain = np.zeros(n_time)
        # kg m-2 s-1 is mm/s of water, so seconds per hour gives mm/h
        pr_mm_hr: np.ndarray | None = (ls_rain + cv_rain) * 3600.0
    else:
        pr_mm_hr = None

    # Wind scales the humidity extinction (see estimate_visibility_from_rh_pr):
    # without it the humidity fit alone gives mist at 96% RH in a strong wind
    u10 = pick("10m_u_component_of_wind")  # m/s
    v10 = pick("10m_v_component_of_wind")  # m/s
    wind_speed_ms = None if u10 is None or v10 is None else np.hypot(u10, v10)

    vis = estimate_visibility_from_rh_pr(
        RH,
        pr_mm_hr,
        wind_speed_ms=wind_speed_ms,
        which_rh_fit=p["rh_fit"],
        rh_min=p["rh_min"],
        rh_max=p["rh_max"],
        vis_min_km=p["vis_min_km"],
        vis_max_km=p["vis_max_km"],
    )

    vis = np.atleast_1d(vis.astype(float))
    return vis[0] if vis.size == 1 else vis


def wet_bulb_temperature_stull_c(temperature_c, rh_percent):
    """Wet-bulb temperature (°C) from air temperature (°C) and RH (%), Stull (2011).

    A closed-form fit, used instead of metpy's iterative ``wet_bulb_temperature``
    because it only has to place hours either side of a rain/snow threshold.
    Within about 1 °C for RH 5-99% and -20 to 50 °C; RH is clipped to that range
    and the result capped at the air temperature, so it stays below any
    threshold above the air temperature.
    """
    temperature_c = np.asarray(temperature_c, dtype=np.float64)
    rh = np.clip(np.asarray(rh_percent, dtype=np.float64), 5.0, 99.0)
    wet_bulb = (
        temperature_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temperature_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return np.minimum(wet_bulb, temperature_c)


def estimate_visibility_from_rh_pr(
    rh_percent: np.ndarray,
    pr_mm_hr: np.ndarray | None = None,
    wind_speed_ms: np.ndarray | None = None,
    snow_mm_hr: np.ndarray | None = None,
    which_rh_fit: str = "FRAM",
    rh_min: float = 30.0,
    rh_max: float = 100.0,
    vis_min_km: float = 0.05,
    vis_max_km: float = VIS_ESTIMATE_MAX_KM,
    u_ref: float = 2.0,
) -> np.ndarray:
    """Estimate visibility (m) from RH, precipitation, and optional wind speed.

    Args:
        rh_percent: Relative humidity in % (clamped to rh_min-rh_max).
        pr_mm_hr: Precipitation rate in mm/hr. None omits precipitation, and
          rates below PRECIP_NOISE_THRESHOLD_MMH count as dry. The rate is
          treated as rain whatever its type: Table 2 has no snow fit, so pass
          snow separately as snow_mm_hr.
        wind_speed_ms: Wind speed in m/s. None omits wind adjustment.
        snow_mm_hr: Liquid-equivalent snowfall rate in mm/hr, extinguishing
          with the Stoelinga & Warner (2003) snow relation. None omits snow,
          and rates below PRECIP_NOISE_THRESHOLD_MMH count as no snow.
        which_rh_fit: "FRAM" (default), "AIRS2", or "RUC".
        rh_min, rh_max: Clamping limits for RH.
        vis_min_km, vis_max_km: Minimum/Maximum visibility output in km.
        u_ref: Reference wind speed (m/s) at which the RH-driven extinction
          begins to significantly clear due to mixing.
    """
    RH = np.clip(np.asarray(rh_percent, dtype=np.float64), rh_min, rh_max)

    # 1. Compute baseline visibility from RH
    fit = (which_rh_fit or "FRAM").upper()
    if fit == "AIRS2":
        vis_rh = -0.0177 * (RH**2) + 1.462 * RH + 30.8
    elif fit == "RUC":
        vis_rh = 60.0 * np.exp(-0.025 * (RH - 30.0))
    else:  # FRAM (default)
        vis_rh = -41.5 * np.log(RH) + 192.3

    beta_rh = 3.0 / np.maximum(vis_rh, 0.05)

    # 2. Apply Wind Speed Scaling to the RH extinction
    # If wind is high, we decrease the extinction (beta), improving visibility.
    # If wind is near 0, beta remains high.
    if wind_speed_ms is not None:
        u = np.clip(np.asarray(wind_speed_ms, dtype=np.float64), 0.0, None)
        # Scaling factor: approaches 0.1 (high visibility) at strong winds,
        # and 1.0 (baseline low visibility) at zero wind.
        wind_factor = 0.1 + 0.9 * np.exp(-u / u_ref)
        # A missing wind leaves the humidity extinction unscaled, not NaN
        beta_rh = beta_rh * np.where(np.isfinite(u), wind_factor, 1.0)

    # 3. Compute baseline visibility from precipitation, with the rain-type
    # thresholds of Glickman (2000) used by Gültepe & Milbrandt (2010)
    if pr_mm_hr is not None:
        pr = np.clip(np.asarray(pr_mm_hr, dtype=np.float64), 0.0, None)
        vis_pr = np.where(
            pr > GULTEPE_MODERATE_RAIN_MAX_MMH,
            np.maximum(-0.45 * np.power(pr, 0.394) + 2.28, 0.05),
            np.where(
                pr >= GULTEPE_LIGHT_RAIN_MAX_MMH,
                np.maximum(-2.65 * np.power(pr, 0.256) + 7.65, 0.05),
                np.maximum(-863.26 * np.power(pr, 0.003) + 874.19, 0.05),
            ),
        )
        beta_pr = 3.0 / np.maximum(vis_pr, 0.05)
        # The light-rain fit's tiny exponent makes it jump from 874 km at 0 to
        # 46 km at 1e-6 mm/h, so a trace rate (the norm in model means, e.g. a
        # CFS ensemble mean) would cut visibility by a third. Below the noise
        # threshold the hour is dry and precipitation adds no extinction.
        beta_pr = np.where(pr < PRECIP_NOISE_THRESHOLD_MMH, 0.0, beta_pr)
    else:
        beta_pr = 0.0

    # Snow lowers visibility far more than the same liquid rate of rain: 1 mm/h
    # of snow water gives about 1 km, the same rain about 11 km. The snow
    # concentration (g m-3) is the mass flux (kg m-2 s-1) over the fall speed,
    # and its extinction is rescaled to the 5% contrast used above.
    if snow_mm_hr is not None:
        snow = np.clip(np.asarray(snow_mm_hr, dtype=np.float64), 0.0, None)
        snow_concentration_g_m3 = (
            snow / SECONDS_PER_HOUR / SNOW_FALL_SPEED_M_S * GRAMS_PER_KG
        )
        beta_snow_2pct = SNOW_EXTINCTION_COEFFICIENT * np.power(
            snow_concentration_g_m3, SNOW_EXTINCTION_EXPONENT
        )
        beta_snow = (
            beta_snow_2pct
            * VIS_KOSCHMIEDER_CONSTANT
            / STOELINGA_WARNER_CONTRAST_CONSTANT
        )
        beta_pr = beta_pr + np.where(snow < PRECIP_NOISE_THRESHOLD_MMH, 0.0, beta_snow)

    # 4. Combine extinctions using the Koschmieder equation
    vis_km = np.clip(3.0 / (beta_rh + beta_pr), vis_min_km, vis_max_km)
    return (vis_km * 1000.0).astype(np.float32)


def estimate_cfs_visibility(temperature_c, rh_percent, intensity_mm_hr, wind_speed_ms):
    """Visibility (m) for CFS, which has no visibility field, for the hourly
    and current blocks alike.

    The precipitation rate is rain or snow by the same 2 m wet-bulb split as
    CFS's precipitation type. Wind scales the humidity extinction: the
    humidity fit alone gives 2.9 km at 96% RH whatever the wind, but mist and
    fog need near-calm air, and a 9 m/s wind at that humidity is clear. CFS
    winds are a four-member mean, lower than any member's, so the scaling
    errs towards lower visibility.
    """
    intensity_mm_hr = np.asarray(intensity_mm_hr, dtype=np.float64)
    snow = (
        wet_bulb_temperature_stull_c(temperature_c, rh_percent)
        < WET_BULB_RAIN_SNOW_THRESHOLD_C
    )
    return estimate_visibility_from_rh_pr(
        rh_percent,
        np.where(snow, 0.0, intensity_mm_hr),
        wind_speed_ms=wind_speed_ms,
        snow_mm_hr=np.where(snow, intensity_mm_hr, 0.0),
    )


def cos_solar_zenith(lat, lon, unix_times):
    """Cosine of the solar zenith angle at each Unix time, from NOAA's general
    solar position formulas (Spencer 1971 declination and equation of time;
    about 0.1 degree). Negative while the sun is below the horizon.
    """
    times = np.asarray(unix_times, dtype=np.float64)
    days_since_epoch = times / SECONDS_PER_DAY
    utc_hours = (days_since_epoch % 1.0) * 24.0
    # Fractional year (radians): its period is the 365.2422-day tropical year,
    # counted from 2000-01-01 (day 10957 of the epoch), close to the perihelion
    fractional_year = 2.0 * np.pi * ((days_since_epoch - 10957.0) % 365.2422) / 365.2422
    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(fractional_year)
        - 0.032077 * np.sin(fractional_year)
        - 0.014615 * np.cos(2 * fractional_year)
        - 0.040849 * np.sin(2 * fractional_year)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(fractional_year)
        + 0.070257 * np.sin(fractional_year)
        - 0.006758 * np.cos(2 * fractional_year)
        + 0.000907 * np.sin(2 * fractional_year)
        - 0.002697 * np.cos(3 * fractional_year)
        + 0.00148 * np.sin(3 * fractional_year)
    )
    # True solar time in minutes; the hour angle is zero at solar noon
    true_solar_time_min = utc_hours * 60.0 + equation_of_time_min + 4.0 * lon
    hour_angle = np.radians(true_solar_time_min / 4.0 - 180.0)
    latitude = np.radians(lat)
    return np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)


def estimate_cfs_uv_index(lat, lon, unix_times, cloud_fraction):
    """UV index for CFS, which has no UV field, for the hourly and current
    blocks alike.

    Clear-sky UV index from the sun's height (Allaart et al. 2004:
    12.5 * cos(zenith)**2.42 * (ozone / 300 DU)**-1.23) at a typical total
    ozone, as CFS has none, reduced for cloud with the NWS UV Index cloud
    factors. Neither elevation nor surface reflection (snow) is included, and
    CFS cloud is a four-member mean, so it is a typical value for the hour.
    """
    cos_zenith = np.clip(cos_solar_zenith(lat, lon, unix_times), 0.0, None)
    clear_sky_uv = (
        UV_CLEAR_SKY_OVERHEAD_INDEX
        * cos_zenith**UV_CLEAR_SKY_ZENITH_EXPONENT
        * (UV_TYPICAL_OZONE_DU / UV_REFERENCE_OZONE_DU) ** UV_CLEAR_SKY_OZONE_EXPONENT
    )
    cloud_factor = np.interp(
        np.clip(np.asarray(cloud_fraction, dtype=np.float64), 0.0, 1.0),
        UV_CLOUD_COVER_POINTS,
        UV_CLOUD_FACTORS,
    )
    return clear_sky_uv * cloud_factor


def estimate_cfs_gust(wind_speed_ms):
    """Wind gust (m/s) for CFS, which has no gust field: the 10 m wind times
    a typical gust factor (see CFS_GUST_FACTOR)."""
    return np.asarray(wind_speed_ms, dtype=np.float64) * CFS_GUST_FACTOR


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
    # If rain, snow and ice are all present, choose mixed (code 5)
    all_types = (
        (InterPdaySum[:, DATA_DAY["rain"]] > 0)
        & (InterPdaySum[:, DATA_DAY["snow"]] > 0)
        & (InterPdaySum[:, DATA_DAY["ice"]] > 0)
    )

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

    # Accumulation only has one ice column, shared by sleet and freezing rain, so the
    # hourly type (most common on the day) is what tells them apart.
    hourly_type = maxPchanceDay.copy()

    # Only update where not all types are present and some precip exists
    not_all_types = ~all_types
    has_precip = np.max(precip_accum, axis=1) > 0
    # A mixed hour's accumulation is stored as ice, so the dominant accumulation
    # can't overrule a day that is already mixed
    already_mixed = maxPchanceDay == PRECIP_IDX["mixed"]
    update_mask = not_all_types & has_precip & ~already_mixed
    maxPchanceDay[update_mask] = dominant_type[update_mask]
    maxPchanceDay[
        (maxPchanceDay == PRECIP_IDX["ice"]) & (hourly_type == PRECIP_IDX["sleet"])
    ] = PRECIP_IDX["sleet"]

    # If significant ice accumulation exists and type is not already ice/sleet,
    # default to ice (freezing rain), which is more common than sleet globally.
    maxPchanceDay[
        (InterPdaySum[:, DATA_DAY["ice"]] > (1 * prepAccumUnit))
        & (maxPchanceDay != PRECIP_IDX["ice"])
        & (maxPchanceDay != PRECIP_IDX["sleet"])
        & ~already_mixed
    ] = PRECIP_IDX["ice"]

    # If we have all types map the type to mixed
    maxPchanceDay[all_types] = PRECIP_IDX["mixed"]

    return maxPchanceDay


def _precip_code_table(groups: dict[float, list[int]], size: int) -> np.ndarray:
    """A lookup table from a source's precipitation code to the internal
    precipitation category (0 for codes in no group)."""
    table = np.zeros(size, dtype=float)
    for category, codes in groups.items():
        table[codes] = category
    return table


def _lookup_precip_codes(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Categories for integer codes through a lookup table; codes outside the
    table are category 0. One array index instead of an np.isin per group,
    which cost ~0.1 ms per source and request."""
    in_table = (codes >= 0) & (codes < table.size)
    return np.where(in_table, table[np.clip(codes, 0, table.size - 1)], 0.0)


# WMO 4677 present-weather codes (50-99) grouped as documented in
# map_wmo4677_to_ptype: 1 snow, 2 ice, 3 freezing, 4 rain.
WMO4677_PTYPE_TABLE = _precip_code_table(
    {
        1: list(range(70, 76)) + [83, 84, 85, 86, 93, 94],
        2: list(range(76, 80)) + [87, 88, 89, 90, 96, 99],
        3: [56, 57, 66, 67],
        4: (
            list(range(50, 56))
            + list(range(58, 65))
            + [68, 69]
            + list(range(80, 83))
            + [91, 92, 95, 97, 98]
        ),
    },
    100,
)
# Canadian (HRDPS/GDPS) precipitation type codes, see map_canadian_precip_type_to_ptype
CANADIAN_PTYPE_TABLE = _precip_code_table(
    {
        PRECIP_IDX["snow"]: [5],
        PRECIP_IDX["sleet"]: [4, 9],
        PRECIP_IDX["ice"]: [3, 8],
        PRECIP_IDX["rain"]: [1, 2, 7],
    },
    10,
)


def map_wmo4677_to_ptype(
    ptype_codes: np.ndarray, temperature_c: np.ndarray | None = None
) -> np.ndarray:
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
        - If temperature_c is provided, snow/ice codes are overridden to rain when
            temperature is above TEMP_THRESHOLD_WMO_FROZEN_C (unrealistic for frozen precipitation).

    Args:
        ptype_codes: array-like of numeric WMO 4677 codes (may contain NaN)
        temperature_c: Optional array of temperatures in Celsius for validation

    Returns:
        np.ndarray of floats (integer category codes as floats) same shape as input
        with MISSING_DATA where input was NaN.
    """
    codes = np.asarray(ptype_codes)
    # Use float dtype so we can store MISSING_DATA (NaN) without conversion errors
    out = np.zeros_like(codes, dtype=float)

    nan_mask = np.isnan(codes)

    if codes.size > 0:
        vals = codes.copy()
        vals[nan_mask] = -999
        out[:] = _lookup_precip_codes(vals.astype(int), WMO4677_PTYPE_TABLE)

        # Temperature validation: Override snow/ice/freezing codes to rain when too warm
        # Use TEMP_THRESHOLD_WMO_FROZEN_C as threshold - well above freezing to account for observation errors
        if temperature_c is not None:
            temp_arr = np.asarray(temperature_c)
            try:
                # This will raise ValueError if shapes are incompatible for broadcasting
                warm_mask = temp_arr > TEMP_THRESHOLD_WMO_FROZEN_C
                # Override snow (1), ice (2), and freezing (3) to rain (4) when warm
                frozen_precip_mask = np.isin(out, [1, 2, 3])
                override_mask = warm_mask & frozen_precip_mask
                out[override_mask] = 4
            except ValueError as e:
                # Shapes incompatible for broadcasting - skip temperature validation
                # This is expected when temperature data is unavailable or mismatched
                logger.debug(
                    "Temperature validation skipped due to shape mismatch: %s (codes: %s, temp: %s)",
                    e,
                    codes.shape,
                    temp_arr.shape,
                )

    # Use MISSING_DATA for NaNs
    out[nan_mask] = MISSING_DATA

    return out


def map_canadian_precip_type_to_ptype(
    ptype_codes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map Canadian precipitation type codes to internal precip type categories.

    Returns an integer array with the following mapping:
        0 -> none/other
        1 -> snow
        3 -> ice (pellets, hail)
        2 -> freezing rain/drizzle
        4 -> rain

    The mapping follows the Canadian code ranges and uses conservative grouping:
        - Freezing drizzle/rain codes (3, 8) -> freezing (2)
        - Ice pellets / hail-related codes (4, 9) -> ice (3)
        - Snow and snow showers (5) -> snow (1)
        - Rain and drizzle ranges (1, 2, 7) -> rain (4)

    Args:
        ptype_codes: array-like of numeric Canadian precipitation codes (may contain NaN)

    Returns:
        np.ndarray: Array of mapped precipitation type categories.
    """
    codes = np.asarray(ptype_codes)
    # Use float dtype so we can store MISSING_DATA (NaN) without conversion errors
    out = np.zeros_like(codes, dtype=float)

    nan_mask = np.isnan(codes)

    if codes.size > 0:
        vals = codes.copy()
        vals[nan_mask] = -999
        out[:] = _lookup_precip_codes(vals.astype(int), CANADIAN_PTYPE_TABLE)

    # Use MISSING_DATA for NaNs
    out[nan_mask] = MISSING_DATA

    return out


def map_ensemble_precip_rates_to_ptype(
    rain: np.ndarray | None = None,
    ice: np.ndarray | None = None,
    freezing_rain: np.ndarray | None = None,
    snow: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Map ensemble precipitation component rates to the internal precipitation type.

    GEPS and REPS do not provide a categorical type field; they provide rate fields for
    rain, ice pellets, freezing rain, and snow. We classify by the strongest rate, but
    if more than one channel exceeds the threshold we fall back to mixed precipitation.

    The companion mapping is:
        rain -> PRECIP_IDX["rain"]
        snow -> PRECIP_IDX["snow"]
        freezing_rain -> PRECIP_IDX["ice"]
        ice -> PRECIP_IDX["sleet"]
    """
    arrays = {
        "rain": np.asarray(rain, dtype=float) if rain is not None else np.zeros(0),
        "ice": np.asarray(ice, dtype=float) if ice is not None else np.zeros(0),
        "freezing_rain": (
            np.asarray(freezing_rain, dtype=float)
            if freezing_rain is not None
            else np.zeros(0)
        ),
        "snow": np.asarray(snow, dtype=float) if snow is not None else np.zeros(0),
    }

    if (
        not arrays["rain"].size
        and not arrays["ice"].size
        and not arrays["freezing_rain"].size
        and not arrays["snow"].size
    ):
        return np.array([], dtype=float)

    n_hours = max(len(v) for v in arrays.values() if len(v) > 0)
    component_matrix = np.zeros((n_hours, 4), dtype=float)
    component_matrix[:, 0] = (
        np.zeros(n_hours)
        if not arrays["snow"].size
        else np.resize(arrays["snow"], n_hours)
    )
    component_matrix[:, 1] = (
        np.zeros(n_hours)
        if not arrays["ice"].size
        else np.resize(arrays["ice"], n_hours)
    )
    component_matrix[:, 2] = (
        np.zeros(n_hours)
        if not arrays["freezing_rain"].size
        else np.resize(arrays["freezing_rain"], n_hours)
    )
    component_matrix[:, 3] = (
        np.zeros(n_hours)
        if not arrays["rain"].size
        else np.resize(arrays["rain"], n_hours)
    )

    strong_count = (component_matrix > threshold).sum(axis=1)
    out = np.full(n_hours, np.nan, dtype=float)

    mixed_mask = strong_count > 1
    out[mixed_mask] = PRECIP_IDX["mixed"]

    strong_mask = strong_count == 1
    if np.any(strong_mask):
        strongest_idx = np.argmax(component_matrix, axis=1)
        strongest_idx = np.where(strong_mask, strongest_idx, 0)
        component_map = {
            0: PRECIP_IDX["snow"],
            1: PRECIP_IDX["sleet"],
            2: PRECIP_IDX["ice"],
            3: PRECIP_IDX["rain"],
        }
        out[strong_mask] = np.array(
            [component_map[int(i)] for i in strongest_idx[strong_mask]]
        )

    no_precip_mask = strong_count == 0
    if np.any(no_precip_mask):
        out[no_precip_mask] = PRECIP_IDX["none"]

    # Rows with no valid component data are missing, not precipitation-free.
    all_nan_mask = np.all(np.isnan(component_matrix), axis=1)
    out[all_nan_mask] = MISSING_DATA

    return out


def zero_small_values(
    array: np.ndarray, threshold: float = PRECIP_NOISE_THRESHOLD_MMH
) -> np.ndarray:
    """Clamp near-zero values to zero to reduce floating noise."""
    array[np.abs(array) < threshold] = 0.0
    return array


# Precomputed constants – built once at import time, not every call
_FIELDS_V_LT_2 = (
    "airQualityIndex",
    "airQualityIndexMax",
    "airQualityIndexMaxTime",
    "airQualityIndexMin",
    "airQualityIndexMinTime",
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
    "liftedIndex",
    "liftedIndexMin",
    "liftedIndexMinTime",
    "convectiveInhibition",
    "convectiveInhibitionMax",
    "convectiveInhibitionMaxTime",
    "verticalVelocity",
    "verticalVelocityMin",
    "verticalVelocityMinTime",
    "kIndex",
    "kIndexMax",
    "kIndexMaxTime",
)

# AQ detail fields are only included when inc_airqualitydetails=1
_FIELDS_AQ_DETAILS = (
    "pm25",
    "pm25Max",
    "pm25MaxTime",
    "pm10",
    "pm10Max",
    "pm10MaxTime",
    "ozoneConcentration",
    "ozoneConcentrationMax",
    "ozoneConcentrationMaxTime",
    "no2Concentration",
    "no2ConcentrationMax",
    "no2ConcentrationMaxTime",
    "so2Concentration",
    "so2ConcentrationMax",
    "so2ConcentrationMaxTime",
    "coConcentration",
    "coConcentrationMax",
    "coConcentrationMaxTime",
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


DictOrList = MutableMapping | list[MutableMapping]


def remove_conditional_fields(
    data: DictOrList,
    version: float,
    time_machine: bool,
    tm_extra: bool,
    inc_airqualitydetails: int = 0,
) -> DictOrList:
    """Removes output fields based on version and request type.

    This function modifies the input data. It works with either a
    single dictionary or a list of dictionaries.

    Args:
        data (DictOrList): The data to filter, a dict or list of dicts.
        version (float): The API version.
        time_machine (bool): Whether it's a Time Machine request.
        tm_extra (bool): Whether extra Time Machine fields are requested.
        inc_airqualitydetails (int): Whether to include AQ detail fields (0 or 1).

    Returns:
        DictOrList: The modified data.
    """
    # Build a *set* to avoid duplicate pops ("cape", "solar" overlap)
    fields_to_remove = set()

    if version < 2:
        fields_to_remove.update(_FIELDS_V_LT_2)

    if time_machine and not tm_extra:
        fields_to_remove.update(_FIELDS_TM_BASIC)

    if not inc_airqualitydetails:
        fields_to_remove.update(_FIELDS_AQ_DETAILS)

    if not fields_to_remove:
        return data

    # Each response block is a dict, or a list of dicts, of scalars. Before version
    # 2, precip types that didn't exist yet are shown as sleet; internal calculations
    # keep 'ice' and 'mixed'.
    downgrade_precip = version < 2
    for item in data if isinstance(data, list) else [data]:
        if not isinstance(item, dict):
            continue  # skip non-dicts defensively
        pop = item.pop  # cache bound method to avoid repeated lookups
        for field in fields_to_remove:
            pop(field, None)
        if downgrade_precip and item.get("precipType") in ("ice", "mixed"):
            item["precipType"] = PRECIP_TYPES["sleet"]

    return data


def _safe_int(val):
    """Safely convert a value to int, returning np.nan for NaN."""
    # math.isnan is several times cheaper than np.isnan on the per-hour path
    return int(val) if not math.isnan(val) else np.nan
