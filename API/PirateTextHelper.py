# %% Script to contain the helper functions that can be used to generate the text summary of the forecast data for Pirate Weather
import math
from collections import Counter

import numpy as np

from API.constants.api_const import PRECIP_TYPES
from API.constants.shared_const import KELVIN_TO_CELSIUS
from API.constants.text_const import (
    CAPE_THRESHOLDS,
    CIN_THRESHOLDS,
    CLOUD_COVER_THRESHOLDS,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    DEWPOINT_DEPRESSION_FOR_STORM,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    KI_THRESHOLDS,
    LI_THRESHOLDS,
    LIQUID_DENSITY_CONVERSION,
    MIST_THRESHOLD_METERS,
    PRECIP_INTENSITY_THRESHOLDS,
    PRECIP_PROB_THRESHOLD,
    SMOKE_CONCENTRATION_THRESHOLD_UGM3,
    SNOW_DENSITY_CONST,
    SNOW_INTENSITY_THRESHOLDS,
    TEMP_DEWPOINT_SPREAD_FOR_FOG,
    TEMP_DEWPOINT_SPREAD_FOR_MIST,
    VV_THRESHOLDS,
    WIND_THRESHOLDS,
)

# Summary types that use hourly (short-period) icon accumulation thresholds,
# and those that use the daily thresholds.
HOURLY_SUMMARY_TYPES = ("hour", "current", "minute", "hourly")
DAILY_SUMMARY_TYPES = ("day", "week")

# Intensity wording, lightest first, used for every precipitation type.
INTENSITY_LEVELS = ("very-light", "light", "medium", "heavy")

# Per-type wording and Pirate icon set icons for each entry in INTENSITY_LEVELS.
# Medium precipitation keeps the generic type icon chosen by the accumulation
# check, which is why it has no dedicated icon.
PRECIP_LEVEL_TEXT = {
    PRECIP_TYPES["rain"]: (
        "rain",
        ("drizzle", "light-rain", None, "heavy-rain"),
    ),
    PRECIP_TYPES["snow"]: (
        "snow",
        ("flurries", "light-snow", None, "heavy-snow"),
    ),
    PRECIP_TYPES["sleet"]: (
        "sleet",
        ("very-light-sleet", "light-sleet", None, "heavy-sleet"),
    ),
    PRECIP_TYPES["ice"]: (
        "freezing-rain",
        ("freezing-drizzle", "light-freezing-rain", None, "heavy-freezing-rain"),
    ),
}

# Thunderstorm scoring: each available index contributes up to its weight
# (total 100) and the score is normalised by the weight actually available.
THUNDERSTORM_WEIGHTS = {
    "cape": 35.0,
    "li": 25.0,
    "ki": 20.0,
    "vv": 20.0,
}
THUNDERSTORM_SCORE = {
    "thunderstorm": 60.0,
    "possible": 30.0,
}
# Without at least CAPE's worth of weight there is too little data to score.
THUNDERSTORM_MIN_WEIGHT = THUNDERSTORM_WEIGHTS["cape"]
HIGH_PRECIP_PROB_THRESHOLD = 0.7
CAPE_SPAN = CAPE_THRESHOLDS["high"] - CAPE_THRESHOLDS["low"]
LI_SPAN = abs(LI_THRESHOLDS["high"] - LI_THRESHOLDS["low"])
KI_SPAN = KI_THRESHOLDS["high"] - KI_THRESHOLDS["low"]
POP_SPAN = HIGH_PRECIP_PROB_THRESHOLD - PRECIP_PROB_THRESHOLD


def most_common(lst):
    """
    Finds the most common icon to use as the icon

    Parameters:
    - lst (arr): An array of weekly icons

    Returns:
    - str: The most common icon in the lst.
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]


def calculate_sky_icon(cloudCover, isDayTime, icon="darksky"):
    """
    Calculates the sky cover icon

    Parameters:
    - cloudCover (int): The cloud cover for the period
    - isDayTime (bool): Whether its currently day or night
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - str: The icon representing the current cloud cover
    """
    day_night = "day" if isDayTime else "night"

    if cloudCover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        return "cloudy"
    if cloudCover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"] and icon == "pirate":
        return "mostly-cloudy-" + day_night
    if cloudCover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        return "partly-cloudy-" + day_night
    if cloudCover > CLOUD_COVER_THRESHOLDS["mostly_clear"] and icon == "pirate":
        return "mostly-clear-" + day_night
    return "clear-" + day_night


def matches_precip(
    prep_type, valid_types, amount, threshold, intensity, intensity_threshold
):
    """
    Determines if precipitation conditions match for a given type.

    Parameters:
    - prep_type (str): The current precipitation type (e.g., "rain", "snow").
    - valid_types (tuple of str): Precipitation types to check against.
    - amount (float): The measured precipitation amount for this type.
    - threshold (float): The threshold amount for displaying an icon.
    - intensity (float): The overall precipitation intensity.
    - intensity_threshold (float): The threshold intensity for displaying an icon.

    Returns:
    - bool: True if the precipitation type matches and either the amount or intensity is within thresholds.
    """
    return prep_type in valid_types and (
        (0 < amount < threshold) or (0 < intensity < intensity_threshold)
    )


def _intensity_level(intensity, light_thresh, mid_thresh, heavy_thresh):
    """Return the INTENSITY_LEVELS index for an intensity (mm/h).

    A NaN intensity fails every comparison and so lands in "heavy", which is
    what the original if/elif ladders did.
    """
    if intensity < light_thresh:
        return 0
    if intensity < mid_thresh:
        return 1
    if intensity < heavy_thresh:
        return 2
    return 3


def calculate_precip_text(
    precipType,
    type,
    rainAccum,
    snowAccum,
    sleetAccum,
    pop=1,
    icon="darksky",
    mode="both",
    isDayTime=True,
    # Type-specific peak or instant intensities (mm/h)
    eff_rain_intensity=None,
    eff_snow_intensity=None,
    eff_ice_intensity=None,
    num_precip_days=1,
):
    """
    Calculates the precipitation text and icon.
    All inputs are expected in SI units (mm/h for intensity, mm for accumulation).

    Parameters:
    - precipType (str): The type of precipitation
    - type (str): What type of summary is being generated.
    - rainAccum (float): The rain accumulation during a period in mm
    - snowAccum (float): The snow accumulation during a period in mm
    - sleetAccum (float): The ice/sleet accumulation during a period in mm
    - pop (float): The current probability of precipitation defaulting to 1
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the precipitation will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.
    - isDayTime (bool): Whether its currently day or night
    - eff_rain_intensity (float | None): The effective rain intensity in mm/h
    - eff_snow_intensity (float | None): The effective snow intensity in mm/h
    - eff_ice_intensity (float | None): The effective ice intensity in mm/h
    - num_precip_days (int): The number of days with precipitation (used for weekly summaries)

    Returns:
    - str | None: The summary text representing the current precipitation
    - str | None: The icon representing the current precipitation
    """

    # If any precipitation is missing, return None appropriately for the mode.
    if math.isnan(rainAccum) or math.isnan(snowAccum) or math.isnan(sleetAccum):
        return (None, None) if mode == "both" else None

    # If any effective intensity is missing, set it to 0.
    if eff_rain_intensity is None:
        eff_rain_intensity = 0
    if eff_snow_intensity is None:
        eff_snow_intensity = 0
    if eff_ice_intensity is None:
        eff_ice_intensity = 0

    # If pop is missing set it to 1 so we can calculate the precipitation text
    if math.isnan(pop):
        pop = 1

    # Use daily or hourly accumulation thresholds (mm) depending on the summary
    if type in HOURLY_SUMMARY_TYPES:
        snow_icon_threshold = HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM
        precip_icon_threshold = HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM
    elif type in DAILY_SUMMARY_TYPES:
        snow_icon_threshold = DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM
        precip_icon_threshold = DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM
    else:
        raise ValueError(f"Unknown precipitation summary type: {type}")

    # Intensity checks always use the hourly thresholds (mm/h)
    snow_intensity_icon_threshold = HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM
    precip_intensity_icon_threshold = HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM

    is_pirate_icon = icon == "pirate"
    c_icon = None
    c_text = None
    # Snow is compared as a liquid equivalent using a 10:1 ratio
    total_prep = rainAccum + (snowAccum / 10) + sleetAccum

    # Add the possible precipitation text if pop is less than 25% or if pop is
    # greater than 0 but the accumulation or intensity is too small for an icon
    is_possible_precip = (
        pop < PRECIP_PROB_THRESHOLD
        or matches_precip(
            precipType,
            ("rain", "none"),
            rainAccum,
            precip_icon_threshold,
            eff_rain_intensity,
            precip_intensity_icon_threshold,
        )
        or matches_precip(
            precipType,
            ("snow",),
            snowAccum,
            snow_icon_threshold,
            eff_snow_intensity,
            snow_intensity_icon_threshold,
        )
        or matches_precip(
            precipType,
            ("sleet", "ice"),
            sleetAccum,
            precip_icon_threshold,
            eff_ice_intensity,
            precip_intensity_icon_threshold,
        )
    )

    # Determine the number of precipitation types for the period
    num_types = (snowAccum > 0) + (rainAccum > 0) + (sleetAccum > 0)

    # Several small accumulations of different types can add up to a definite
    # (not "possible") precipitation event.
    if (
        total_prep >= precip_icon_threshold
        and is_possible_precip
        and pop >= PRECIP_PROB_THRESHOLD
        and num_types > 1
    ):
        is_possible_precip = False

    possible_precip = "possible-" if is_possible_precip else ""

    # Decide on an icon if either accumulation or intensity thresholds are met
    if pop >= PRECIP_PROB_THRESHOLD and (
        (
            rainAccum > precip_icon_threshold
            or eff_rain_intensity > precip_intensity_icon_threshold
        )
        or (
            snowAccum >= snow_icon_threshold
            or eff_snow_intensity > snow_intensity_icon_threshold
        )
        or (
            sleetAccum >= precip_icon_threshold
            or eff_ice_intensity > precip_intensity_icon_threshold
        )
        or (total_prep >= precip_icon_threshold and num_types > 1)
    ):
        if precipType == PRECIP_TYPES["none"]:
            c_icon = PRECIP_TYPES["rain"]  # Fallback icon
        elif precipType == PRECIP_TYPES["mixed"]:
            c_icon = PRECIP_TYPES["mixed"] if is_pirate_icon else PRECIP_TYPES["sleet"]
        elif precipType == PRECIP_TYPES["ice"]:
            c_icon = PRECIP_TYPES["sleet"]
        else:
            c_icon = precipType

    day_night = "day" if isDayTime else "night"

    if (num_types > 2 and total_prep > 0) or precipType == PRECIP_TYPES["mixed"]:
        c_text = "mixed-precipitation"
    elif precipType in PRECIP_LEVEL_TEXT:
        if precipType == PRECIP_TYPES["rain"]:
            accum = rainAccum
            intensity = eff_rain_intensity
            thresholds = PRECIP_INTENSITY_THRESHOLDS
        elif precipType == PRECIP_TYPES["snow"]:
            accum = snowAccum
            intensity = eff_snow_intensity
            thresholds = SNOW_INTENSITY_THRESHOLDS
        else:
            # Sleet and freezing rain share the ice accumulation and intensity
            accum = sleetAccum
            intensity = eff_ice_intensity
            thresholds = PRECIP_INTENSITY_THRESHOLDS

        if accum > 0 or intensity > 0:
            type_text, level_icons = PRECIP_LEVEL_TEXT[precipType]
            level = _intensity_level(
                intensity, thresholds["light"], thresholds["mid"], thresholds["heavy"]
            )
            c_text = possible_precip + INTENSITY_LEVELS[level] + "-" + type_text
            if is_pirate_icon and is_possible_precip:
                c_icon = "possible-" + type_text + "-" + day_night
            elif is_pirate_icon and level_icons[level] is not None:
                c_icon = level_icons[level]
    elif precipType == PRECIP_TYPES["none"] and (
        # Treat any accumulation or per-type intensity as a signal of precipitation
        rainAccum > 0
        or snowAccum > 0
        or sleetAccum > 0
        or eff_rain_intensity > 0
        or eff_snow_intensity > 0
        or eff_ice_intensity > 0
    ):
        # For an unknown precipitation type use the largest per-type intensity
        none_intensity = max(eff_rain_intensity, eff_snow_intensity, eff_ice_intensity)
        level = _intensity_level(
            none_intensity,
            PRECIP_INTENSITY_THRESHOLDS["light"],
            PRECIP_INTENSITY_THRESHOLDS["mid"],
            PRECIP_INTENSITY_THRESHOLDS["heavy"],
        )
        c_text = possible_precip + INTENSITY_LEVELS[level] + "-precipitation"

        if is_pirate_icon and is_possible_precip:
            c_icon = "possible-precipitation-" + day_night
        elif is_pirate_icon:
            c_icon = "precipitation"

    if mode == "summary":
        return c_text
    elif mode == "icon":
        return c_icon
    else:
        return c_text, c_icon


def calculate_wind_text(wind, icon="darksky", mode="both"):
    """
    Calculates the wind text.
    Wind speed is expected in SI units (m/s).

    Parameters:
    - wind (float) - The wind speed in m/s
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the wind will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - str | None: The textual representation of the wind
    - str | None: The icon representation of the wind
    """
    windText = None
    windIcon = None

    # If wind is missing, return None appropriately for the mode.
    if math.isnan(wind):
        return (None, None) if mode == "both" else None

    if wind >= WIND_THRESHOLDS["heavy"]:
        windText = "heavy-wind"
        windIcon = "dangerous-wind" if icon == "pirate" else "wind"
    elif wind >= WIND_THRESHOLDS["mid"]:
        windText = "medium-wind"
        windIcon = "wind"
    elif wind >= WIND_THRESHOLDS["light"]:
        windText = "light-wind"
        windIcon = "breezy" if icon == "pirate" else "wind"

    if mode == "summary":
        return windText
    elif mode == "icon":
        return windIcon
    else:
        return windText, windIcon


def calculate_vis_text(
    vis, temp, dewPoint, wind_speed, smoke=0, icon="darksky", mode="both"
):
    """
    Calculates the visibility text.
    All inputs are expected in SI units (meters for visibility, Celsius for temperature).

    Parameters:
    - vis (float) - The visibility in meters
    - temp (float) - The ambient temperature in Celsius
    - dewPoint (float) - The dew point temperature in Celsius
    - smoke (float) - Surface smoke concentration in ug/m3
    - wind_speed (float) - Wind speed in m/s
    - icon (str) - Which icon set to use - Dark Sky or Pirate Weather
    - mode (str) - Determines what gets returned by the function. If set to both the summary and icon for the visibility will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.
    Returns:
    - str | None: The textual representation of the visibility
    - str | None: The icon representation of the visibility
    """
    visText = None
    visIcon = None

    # If temp, dewPoint or vis are missing, return None appropriately for the mode.
    if math.isnan(temp) or math.isnan(dewPoint) or math.isnan(vis):
        return (None, None) if mode == "both" else None

    # Treat missing smoke as no smoke so the smoke comparisons stay valid
    if math.isnan(smoke):
        smoke = 0

    # Strong winds mix out fog and mist
    wind_too_strong = (
        not math.isnan(wind_speed) and wind_speed >= WIND_THRESHOLDS["light"]
    )

    # Calculate the temperature dew point spread (already in Celsius)
    tempDewSpread = temp - dewPoint

    # Fog
    if (
        not wind_too_strong
        and vis < FOG_THRESHOLD_METERS
        and tempDewSpread <= TEMP_DEWPOINT_SPREAD_FOR_FOG
    ):
        visText = "fog"
        visIcon = "fog"
    # Smoke
    elif smoke >= SMOKE_CONCENTRATION_THRESHOLD_UGM3 and vis < MIST_THRESHOLD_METERS:
        visText = "smoke"
        visIcon = "smoke" if icon == "pirate" else "fog"
    # Mist
    elif (
        not wind_too_strong
        and vis < MIST_THRESHOLD_METERS
        and tempDewSpread <= TEMP_DEWPOINT_SPREAD_FOR_MIST
    ):
        visText = "mist"
        visIcon = "mist" if icon == "pirate" else "fog"
    # Haze
    elif (
        smoke < SMOKE_CONCENTRATION_THRESHOLD_UGM3
        and vis < MIST_THRESHOLD_METERS
        and tempDewSpread > TEMP_DEWPOINT_SPREAD_FOR_MIST
    ):
        visText = "haze"
        visIcon = "haze" if icon == "pirate" else "fog"

    if mode == "summary":
        return visText
    elif mode == "icon":
        return visIcon
    else:
        return visText, visIcon


def calculate_sky_text(cloudCover, isDayTime, icon="darksky", mode="both"):
    """
    Calculates the sky cover text

    Parameters:
    - cloudCover (int): The cloud cover for the period
    - isDayTime (bool): Whether its currently day or night
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the cloud cover will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - str | None: The text representing the current cloud cover
    - str | None: The icon representing the current cloud cover
    """
    # If cloud cover is missing, return None appropriately for the mode.
    if math.isnan(cloudCover):
        return (None, None) if mode == "both" else None

    if cloudCover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        skyText = "heavy-clouds"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"]:
        skyText = "medium-clouds"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        skyText = "light-clouds"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["mostly_clear"]:
        skyText = "very-light-clouds"
    else:
        skyText = "clear"

    if mode == "summary":
        return skyText

    skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)
    if mode == "icon":
        return skyIcon
    return skyText, skyIcon


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(val, max_val))


def _valid(v: float | None) -> bool:
    return v is not None and not math.isnan(v)


def calculate_thunderstorm_text(
    cape: float,
    pop: float | None = None,
    mode: str = "both",
    icon: str = "darksky",
    is_day: bool = True,
    lifted_index: float | None = None,
    cin: float | None = None,
    vertical_velocity: float | None = None,
    k_index: float | None = None,
    dewpoint: float | None = None,
    temperature: float | None = None,
) -> tuple[str | None, str | None] | str | None:
    """Calculates thunderstorm text and icon using atmospheric stability indices and PoP.

    Uses dynamic parameter weighting, continuous feature scaling, and physical
    suppressors to reduce false positives.

    Args:
        cape (float): Convective Available Potential Energy in J/kg.
        pop (float | None): Probability of Precipitation (0.0 to 1.0). Defaults to 1.0 if None.
        mode (str): Return mode. "both" returns (text, icon); "summary" returns text; "icon" returns icon.
        icon (str): Icon set to use — "darksky" or "pirate".
        is_day (bool): Whether it is currently daytime (controls pirate icon variants).
        lifted_index (float | None): Lifted Index in K. Pass None when unavailable.
        cin (float | None): Convective Inhibition in J/kg (negative by convention). Pass None when unavailable.
        vertical_velocity (float | None): Vertical velocity (omega) in Pa/s. Pass None when unavailable.
        k_index (float | None): K Index in K (°C equivalent). Pass None when unavailable.
        dewpoint (float | None): 2 m dewpoint temperature in °C. Pass None when unavailable.
        temperature (float | None): 2 m air temperature in °C. Pass None when unavailable.

    Returns:
        tuple[str | None, str | None] | str | None: Thunderstorm text, icon, or both depending on mode.
    """
    earned_score = 0.0
    total_possible_weight = 0.0

    # 1. Continuous feature scaling of each available index
    if _valid(cape):
        total_possible_weight += THUNDERSTORM_WEIGHTS["cape"]
        earned_score += THUNDERSTORM_WEIGHTS["cape"] * clamp(
            (cape - CAPE_THRESHOLDS["low"]) / CAPE_SPAN
        )

    if _valid(lifted_index):
        total_possible_weight += THUNDERSTORM_WEIGHTS["li"]
        earned_score += THUNDERSTORM_WEIGHTS["li"] * clamp(
            (LI_THRESHOLDS["low"] - lifted_index) / LI_SPAN
        )

    if _valid(k_index):
        total_possible_weight += THUNDERSTORM_WEIGHTS["ki"]
        earned_score += THUNDERSTORM_WEIGHTS["ki"] * clamp(
            (k_index - KI_THRESHOLDS["low"]) / KI_SPAN
        )

    if _valid(vertical_velocity):
        total_possible_weight += THUNDERSTORM_WEIGHTS["vv"]
        earned_score += THUNDERSTORM_WEIGHTS["vv"] * clamp(
            -vertical_velocity / VV_THRESHOLDS["strong_upward"]
        )

    # Prevent false positives if payload lacks enough core parameters
    if total_possible_weight < THUNDERSTORM_MIN_WEIGHT:
        base_confidence = 0.0
    else:
        base_confidence = (earned_score / total_possible_weight) * 100.0

    # 2. Suppressors & Modifiers
    suppressor = 1.0

    # CIN Penalty
    if _valid(cin):
        if cin <= CIN_THRESHOLDS["high"]:
            suppressor *= 0.10
        elif cin <= CIN_THRESHOLDS["low"]:
            cin_penalty = (cin - CIN_THRESHOLDS["low"]) / (
                CIN_THRESHOLDS["high"] - CIN_THRESHOLDS["low"]
            )
            suppressor *= 1.0 - (0.70 * cin_penalty)

    # Dewpoint Depression Penalty (Applied globally)
    if _valid(temperature) and _valid(dewpoint):
        depression = temperature - dewpoint
        if depression >= DEWPOINT_DEPRESSION_FOR_STORM["high"]:
            suppressor *= 0.0
        elif depression > DEWPOINT_DEPRESSION_FOR_STORM["low"]:
            dep_penalty = (depression - DEWPOINT_DEPRESSION_FOR_STORM["low"]) / (
                DEWPOINT_DEPRESSION_FOR_STORM["high"]
                - DEWPOINT_DEPRESSION_FOR_STORM["low"]
            )
            suppressor *= 1.0 - (0.80 * dep_penalty)

    # PoP Threshold Scaling
    if _valid(pop):
        if pop <= PRECIP_PROB_THRESHOLD:
            suppressor *= 0.2  # Heavy penalty for low PoP environments
        elif pop < HIGH_PRECIP_PROB_THRESHOLD:
            pop_penalty = (pop - PRECIP_PROB_THRESHOLD) / POP_SPAN
            suppressor *= 0.2 + (0.8 * pop_penalty)  # Smooth ramp-up from 0.2 to 1.0

    final_score = base_confidence * suppressor

    # 3. Output State Decision
    thu_text = None
    thu_icon = None

    if final_score >= THUNDERSTORM_SCORE["thunderstorm"]:
        thu_text = "thunderstorm"
        thu_icon = "thunderstorm"
    elif final_score >= THUNDERSTORM_SCORE["possible"]:
        thu_text = "possible-thunderstorm"
        if icon == "pirate":
            thu_icon = (
                "possible-thunderstorm-day" if is_day else "possible-thunderstorm-night"
            )

    if mode == "summary":
        return thu_text
    elif mode == "icon":
        return thu_icon
    return thu_text, thu_icon


def kelvin_from_celsius(celsius):
    """
    Converts Celsius to Kelvin.

    Parameters:
    - celsius (float): Temperature in Celsius

    Returns:
    - float: Temperature in Kelvin
    """
    return celsius + KELVIN_TO_CELSIUS


def estimate_snow_height(precipitation_mm, temperature_c, wind_speed_mps):
    """
    Estimates the depth of snow (in mm) from liquid precipitation, temperature, and wind speed.

    Parameters:
    - precipitation_mm (float): Liquid precipitation in millimeters
    - temperature_c (float): Air temperature in Celsius
    - wind_speed_mps (float): Wind speed in meters per second

    Returns:
    - float: Estimated snow depth in millimeters
    """
    snow_density_kg_m3 = estimate_snow_density(temperature_c, wind_speed_mps)
    # 1000 is a conversion factor from mm to grams (for density in kg/m^3)
    return precipitation_mm * LIQUID_DENSITY_CONVERSION / snow_density_kg_m3


def estimate_snow_density(temperature_c, wind_speed_mps):
    """
    Estimates the density of newly fallen snow (in kg/m^3) based on temperature and wind speed.
    This function is vectorized to handle numpy arrays.

    Args:
        temperature_c (float | np.ndarray): Air temperature in Celsius.
        wind_speed_mps (float | np.ndarray): Wind speed in meters per second.

    Returns:
        float | np.ndarray: Estimated snow density in kg/m^3.
    """
    c = SNOW_DENSITY_CONST
    kelvins = kelvin_from_celsius(temperature_c)
    kelvins = np.minimum(kelvins, c["max_kelvin"])

    wind_speed_exp = np.power(wind_speed_mps, c["wind_exp"])

    density_low_temp = c["density_base"] * (
        1 - c["low_temp_exp_coeff"] * np.exp(-c["low_temp_exp_factor"] * wind_speed_exp)
    )

    power_term = np.power(c["high_temp_power_base"] - kelvins, c["high_temp_power_exp"])

    density_high_temp = c["density_base"] * (
        1
        - c["high_temp_exp_coeff"]
        * np.exp(
            -c["high_temp_exp_factor2"] * power_term
            - c["high_temp_exp_factor"] * wind_speed_exp
        )
    )

    snow_density_kg_m3 = np.where(
        kelvins <= c["low_temp_threshold"], density_low_temp, density_high_temp
    )

    return np.maximum(snow_density_kg_m3, c["min_density"])
