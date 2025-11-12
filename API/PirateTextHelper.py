# %% Script to contain the helper functions that can be used to generate the text summary of the forecast data for Pirate Weather
import math
from collections import Counter

import numpy as np

from API.constants.shared_const import KELVIN_TO_CELSIUS
from API.constants.text_const import (
    CAPE_THRESHOLDS,
    CLOUD_COVER_THRESHOLDS,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    LIQUID_DENSITY_CONVERSION,
    MIST_THRESHOLD_METERS,
    PRECIP_INTENSITY_THRESHOLDS,
    PRECIP_PROB_THRESHOLD,
    SMOKE_CONCENTRATION_THRESHOLD_UGM3,
    SNOW_DENSITY_CONST,
    SNOW_INTENSITY_THRESHOLDS,
    TEMP_DEWPOINT_SPREAD_FOR_FOG,
    TEMP_DEWPOINT_SPREAD_FOR_MIST,
    WARM_TEMPERATURE_THRESHOLD,
    WIND_THRESHOLDS,
)


def Most_Common(lst):
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
    Calculates the sky cover text

    Parameters:
    - cloudCover (int): The cloud cover for the period
    - isDayTime (bool): Whether its currently day or night
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - str: The icon representing the current cloud cover
    """
    sky_icon = None

    if cloudCover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        sky_icon = "cloudy"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"] and icon == "pirate":
        if isDayTime:
            sky_icon = "mostly-cloudy-day"
        else:
            sky_icon = "mostly-cloudy-night"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        if isDayTime:
            sky_icon = "partly-cloudy-day"
        else:
            sky_icon = "partly-cloudy-night"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["mostly_clear"] and icon == "pirate":
        if isDayTime:
            sky_icon = "mostly-clear-day"
        else:
            sky_icon = "mostly-clear-night"
    else:
        if isDayTime:
            sky_icon = "clear-day"
        else:
            sky_icon = "clear-night"

    return sky_icon


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
    if any(np.isnan(x) for x in (rainAccum, snowAccum, sleetAccum)):
        return (None, None) if mode == "both" else None

    # If any effective intensity is missing, set it to 0.
    if eff_rain_intensity is None:
        eff_rain_intensity = 0
    if eff_snow_intensity is None:
        eff_snow_intensity = 0
    if eff_ice_intensity is None:
        eff_ice_intensity = 0

    # If pop is missing set it to 1 so we can calculate the precipitation text
    if np.isnan(pop):
        pop = 1

    # Thresholds in mm/h for intensity
    lightPrecipThresh = PRECIP_INTENSITY_THRESHOLDS["light"]
    midPrecipThresh = PRECIP_INTENSITY_THRESHOLDS["mid"]
    heavyPrecipThresh = PRECIP_INTENSITY_THRESHOLDS["heavy"]
    lightSnowThresh = SNOW_INTENSITY_THRESHOLDS["light"]
    midSnowThresh = SNOW_INTENSITY_THRESHOLDS["mid"]
    heavySnowThresh = SNOW_INTENSITY_THRESHOLDS["heavy"]

    # Thresholds in mm for accumulation
    snowIconThresholdHour = HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM
    precipIconThresholdHour = HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM

    snowIconThresholdDay = DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM
    precipIconThresholdDay = DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM
    numTypes = 0

    # Use daily or hourly thresholds depending on the situation
    if type == "hour" or type == "current" or type == "minute" or type == "hourly":
        snowIconThreshold = snowIconThresholdHour
        precipIconThreshold = precipIconThresholdHour
    elif type == "day" or type == "week":
        snowIconThreshold = snowIconThresholdDay
        precipIconThreshold = precipIconThresholdDay
    possiblePrecip = ""
    cIcon = None
    cText = None
    totalPrep = rainAccum + (snowAccum / 10) + sleetAccum

    rain_condition = matches_precip(
        precipType,
        ("rain", "none"),
        rainAccum,
        precipIconThreshold,
        eff_rain_intensity,
        precipIconThresholdHour,
    )

    snow_condition = matches_precip(
        precipType,
        ("snow",),
        snowAccum,
        snowIconThreshold,
        eff_snow_intensity,
        snowIconThresholdHour,
    )

    ice_condition = matches_precip(
        precipType,
        ("sleet", "ice", "hail"),
        sleetAccum,
        precipIconThreshold,
        eff_ice_intensity,
        precipIconThresholdHour,
    )

    # Add the possible precipitation text if pop is less than 25% or if pop is greater than 0 but precipIntensity is between 0-0.02 mm/h
    if pop < PRECIP_PROB_THRESHOLD or rain_condition or snow_condition or ice_condition:
        possiblePrecip = "possible-"

    # Determine the number of precipitation types for the day
    if snowAccum > 0:
        numTypes += 1
    if rainAccum > 0:
        numTypes += 1
    if sleetAccum > 0:
        numTypes += 1

    if (
        totalPrep >= precipIconThreshold
        and possiblePrecip == "possible-"
        and pop >= PRECIP_PROB_THRESHOLD
        and numTypes > 1
    ):
        possiblePrecip = ""

    # Decide on an icon if either accumulation or intensity thresholds are met
    if pop >= PRECIP_PROB_THRESHOLD and (
        (
            rainAccum > precipIconThreshold
            or eff_rain_intensity > precipIconThresholdHour
        )
        or (
            snowAccum >= snowIconThreshold or eff_snow_intensity > snowIconThresholdHour
        )
        or (
            sleetAccum >= precipIconThreshold
            or eff_ice_intensity > precipIconThresholdHour
        )
        or (totalPrep >= precipIconThreshold and numTypes > 1)
    ):
        if precipType == "none":
            cIcon = "rain"  # Fallback icon
        elif precipType == "ice":
            cIcon = "freezing-rain"
        else:
            cIcon = precipType

    if (rainAccum > 0 or eff_rain_intensity > 0) and precipType == "rain":
        if eff_rain_intensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-rain-night"
            elif icon == "pirate":
                cIcon = "drizzle"
        elif (
            eff_rain_intensity >= lightPrecipThresh
            and eff_rain_intensity < midPrecipThresh
        ):
            cText = possiblePrecip + "light-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-rain-night"
            elif icon == "pirate":
                cIcon = "light-rain"
        elif (
            eff_rain_intensity >= midPrecipThresh
            and eff_rain_intensity < heavyPrecipThresh
        ):
            cText = possiblePrecip + "medium-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-rain-night"
        else:
            cText = possiblePrecip + "heavy-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-rain-night"
            elif icon == "pirate":
                cIcon = "heavy-rain"
        if (  # This handles the case where over a week or multiple days there is heavy rain but each day individually does not meet the heavy threshold
            # Some additional tweaking of this logic may be needed based on testing
            (type == "minute" or type == "week")
            and eff_rain_intensity < heavyPrecipThresh
            and rainAccum >= heavyPrecipThresh * num_precip_days * 2
        ):
            cText = ["and", "medium-rain", "possible-heavy-rain"]
    elif (snowAccum > 0 or eff_snow_intensity > 0) and precipType == "snow":
        if eff_snow_intensity < lightSnowThresh:
            cText = possiblePrecip + "very-light-snow"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-snow-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-snow-night"
            elif icon == "pirate":
                cIcon = "flurries"
        elif (
            eff_snow_intensity >= lightSnowThresh and eff_snow_intensity < midSnowThresh
        ):
            cText = possiblePrecip + "light-snow"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-snow-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-snow-night"
            elif icon == "pirate":
                cIcon = "light-snow"
        elif (
            eff_snow_intensity >= midSnowThresh and eff_snow_intensity < heavySnowThresh
        ):
            cText = possiblePrecip + "medium-snow"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-snow-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-snow-night"
        else:
            cText = possiblePrecip + "heavy-snow"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-snow-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-snow-night"
            elif icon == "pirate":
                cIcon = "heavy-snow"
        if (
            (type == "week" or type == "hourly")
            and snowAccum < (snowIconThreshold * num_precip_days * 2)
            and eff_snow_intensity >= heavySnowThresh
        ):
            cText = ["and", "medium-snow", "possible-heavy-snow"]
    elif (sleetAccum > 0 or eff_ice_intensity > 0) and precipType == "sleet":
        if eff_ice_intensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-sleet"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-sleet-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-sleet-night"
            elif icon == "pirate":
                cIcon = "very-light-sleet"
        elif (
            eff_ice_intensity >= lightPrecipThresh
            and eff_ice_intensity < midPrecipThresh
        ):
            cText = possiblePrecip + "light-sleet"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-sleet-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-sleet-night"
            elif icon == "pirate":
                cIcon = "light-sleet"
        elif (
            eff_ice_intensity >= midPrecipThresh
            and eff_ice_intensity < heavyPrecipThresh
        ):
            cText = possiblePrecip + "medium-sleet"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-sleet-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-sleet-night"
        else:
            cText = possiblePrecip + "heavy-sleet"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-sleet-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-sleet-night"
            elif icon == "pirate":
                cIcon = "heavy-sleet"
        if (
            (type == "week" or type == "hourly")
            and sleetAccum < (precipIconThreshold * num_precip_days * 2)
            and eff_ice_intensity >= heavyPrecipThresh
        ):
            cText = ["and", "medium-sleet", "possible-heavy-sleet"]

    elif (sleetAccum > 0 or eff_ice_intensity > 0) and precipType == "ice":
        if eff_ice_intensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
            elif icon == "pirate":
                cIcon = "freezing-drizzle"
        elif (
            eff_ice_intensity >= lightPrecipThresh
            and eff_ice_intensity < midPrecipThresh
        ):
            cText = possiblePrecip + "light-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
            elif icon == "pirate":
                cIcon = "light-freezing-rain"
        elif (
            eff_ice_intensity >= midPrecipThresh
            and eff_ice_intensity < heavyPrecipThresh
        ):
            cText = possiblePrecip + "medium-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
        else:
            cText = possiblePrecip + "heavy-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
            elif icon == "pirate":
                cIcon = "heavy-freezing-rain"
        if (
            (type == "week" or type == "hourly")
            and sleetAccum < (precipIconThreshold * num_precip_days * 2)
            and eff_ice_intensity >= heavyPrecipThresh
        ):
            cText = ["and", "medium-freezing-rain", "possible-heavy-freezing-rain"]
    elif (sleetAccum > 0 or eff_ice_intensity > 0) and precipType == "hail":
        cText = possiblePrecip + "hail"
    elif (
        rainAccum > 0
        or snowAccum > 0
        or sleetAccum > 0
        or (
            # Treat any available per-type intensity as a signal of precip when type is none
            (eff_rain_intensity is not None and eff_rain_intensity > 0)
            or (eff_snow_intensity is not None and eff_snow_intensity > 0)
            or (eff_ice_intensity is not None and eff_ice_intensity > 0)
        )
    ) and precipType == "none":
        # For unknown precip type, use the maximum of provided per-type intensities if available
        _none_intensity = max(
            [
                v
                for v in (
                    eff_rain_intensity,
                    eff_snow_intensity,
                    eff_ice_intensity,
                )
                if v is not None
            ]
        )
        if _none_intensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-precipitation"
        elif _none_intensity >= lightPrecipThresh and _none_intensity < midPrecipThresh:
            cText = possiblePrecip + "light-precipitation"
        elif _none_intensity >= midPrecipThresh and _none_intensity < heavyPrecipThresh:
            cText = possiblePrecip + "medium-precipitation"
        else:
            cText = possiblePrecip + "heavy-precipitation"
        if (
            (type == "week" or type == "hourly")
            and (
                (rainAccum + sleetAccum) < (precipIconThreshold * 2)
                or snowAccum < (snowIconThreshold * 2)
            )
            and _none_intensity >= heavyPrecipThresh
        ):
            cText = ["and", "medium-precipitation", "possible-heavy-precipitation"]

        if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
            cIcon = "possible-precipitation-day"
        elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
            cIcon = "possible-precipitation-night"
        elif icon == "pirate":
            cIcon = "precipitation"

    if mode == "summary":
        return cText
    elif mode == "icon":
        return cIcon
    else:
        return cText, cIcon


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
    if np.isnan(wind):
        return (None, None) if mode == "both" else None

    # Thresholds in m/s
    lightWindThresh = WIND_THRESHOLDS["light"]
    midWindThresh = WIND_THRESHOLDS["mid"]
    heavyWindThresh = WIND_THRESHOLDS["heavy"]

    if wind >= lightWindThresh and wind < midWindThresh:
        windText = "light-wind"
        if icon == "pirate":
            windIcon = "breezy"
        else:
            windIcon = "wind"
    elif wind >= midWindThresh and wind < heavyWindThresh:
        windText = "medium-wind"
        windIcon = "wind"
    elif wind >= heavyWindThresh:
        windText = "heavy-wind"
        if icon == "pirate":
            windIcon = "dangerous-wind"
        else:
            windIcon = "wind"

    if mode == "summary":
        return windText
    elif mode == "icon":
        return windIcon
    else:
        return windText, windIcon


def calculate_vis_text(vis, temp, dewPoint, smoke=0, icon="darksky", mode="both"):
    """
    Calculates the visibility text.
    All inputs are expected in SI units (meters for visibility, Celsius for temperature).

    Parameters:
    - vis (float) - The visibility in meters
    - temp (float) - The ambient temperature in Celsius
    - dewPoint (float) - The dew point temperature in Celsius
    - smoke (float) - Surface smoke concentration in ug/m3
    - icon (str) - Which icon set to use - Dark Sky or Pirate Weather
    - mode (str) - Determines what gets returned by the function. If set to both the summary and icon for the visibility will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.
    Returns:
    - str | None: The textual representation of the visibility
    - str | None: The icon representation of the visibility
    """
    visText = None
    visIcon = None
    # Thresholds in meters
    fogThresh = FOG_THRESHOLD_METERS
    mistThresh = MIST_THRESHOLD_METERS

    # If temp, dewPoint or vis are missing, return None appropriately for the mode.
    if any(np.isnan(x) for x in (temp, dewPoint, vis)):
        return (None, None) if mode == "both" else None

    # Calculate the temperature dew point spread (already in Celsius)
    tempDewSpread = temp - dewPoint

    # Fog
    if vis < fogThresh and tempDewSpread <= TEMP_DEWPOINT_SPREAD_FOR_FOG:
        visText = "fog"
        visIcon = "fog"
    # Smoke
    elif smoke >= SMOKE_CONCENTRATION_THRESHOLD_UGM3 and vis < mistThresh:
        visText = "smoke"
        visIcon = "smoke" if icon == "pirate" else "fog"
    # Mist
    elif vis < mistThresh and tempDewSpread <= TEMP_DEWPOINT_SPREAD_FOR_MIST:
        visText = "mist"
        visIcon = "mist" if icon == "pirate" else "fog"
    # Haze
    elif (
        smoke < SMOKE_CONCENTRATION_THRESHOLD_UGM3
        and vis < mistThresh
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
    skyText = None
    skyIcon = None

    # If cloud cover is missing, return None appropriately for the mode.
    if np.isnan(cloudCover):
        return (None, None) if mode == "both" else None

    if cloudCover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        skyText = "heavy-clouds"
        skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)

    elif cloudCover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)
        if cloudCover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"]:
            skyText = "medium-clouds"

        else:
            skyText = "light-clouds"
    else:
        skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)
        if cloudCover > CLOUD_COVER_THRESHOLDS["mostly_clear"]:
            skyText = "very-light-clouds"
        else:
            skyText = "clear"

    if mode == "summary":
        return skyText
    elif mode == "icon":
        return skyIcon
    else:
        return skyText, skyIcon


def humidity_sky_text(temp, humidity):
    """
    Calculates the humidity text.
    Temperature is expected in SI units (Celsius).

    Parameters:
    - temp (float): The temperature in Celsius
    - humidity (float): The humidity as a fraction (0.0 to 1.0)

    Returns:
    - str | None: The text representing the humidity
    """

    # Return None if humidity or temperature data is missing.
    if humidity is None or math.isnan(humidity) or np.isnan(humidity) or np.isnan(temp):
        return None

    # Only use humid if also warm (>20C)
    tempThresh = WARM_TEMPERATURE_THRESHOLD["c"]
    humidityText = None
    lowHumidityThresh = 0.15
    highHumidityThresh = 0.95

    if humidity <= lowHumidityThresh:
        humidityText = "low-humidity"
    elif humidity >= highHumidityThresh:
        if temp > tempThresh:
            humidityText = "high-humidity"

    return humidityText


def calculate_thunderstorm_text(cape, mode="both", icon="darksky", is_day=True):
    """
    Calculates the thunderstorm text based on CAPE values.

    Parameters:
    - cape (float) -  The CAPE (Convective available potential energy)
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the thunderstorm will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - is_day (bool): Whether it is day or night time

    Returns:
    - str | None: The textual representation of the thunderstorm
    - str | None: The icon representation of the thunderstorm
    """
    thuText = None
    thuIcon = None

    if CAPE_THRESHOLDS["low"] <= cape < CAPE_THRESHOLDS["high"]:
        thuText = "possible-thunderstorm"
    elif cape >= CAPE_THRESHOLDS["high"]:
        thuText = "thunderstorm"

    if thuText == "thunderstorm":
        thuIcon = "thunderstorm"
    elif thuText == "possible-thunderstorm" and icon == "pirate":
        thuIcon = (
            "possible-thunderstorm-day" if is_day else "possible-thunderstorm-night"
        )

    if mode == "summary":
        return thuText
    elif mode == "icon":
        return thuIcon
    else:
        return thuText, thuIcon


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
