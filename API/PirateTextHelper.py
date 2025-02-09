# %% Script to contain the helper functions that can be used to generate the text summary of the forecast data for Pirate Weather
from collections import Counter


cloudyThreshold = 0.875
mostlyCloudyThreshold = 0.625
partlyCloudyThreshold = 0.375
mostlyClearThreshold = 0.125


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

    if cloudCover > cloudyThreshold:
        sky_icon = "cloudy"
    elif cloudCover > mostlyCloudyThreshold and icon == "pirate":
        if isDayTime:
            sky_icon = "mostly-cloudy-day"
        else:
            sky_icon = "mostly-cloudy-night"
    elif cloudCover > partlyCloudyThreshold:
        if isDayTime:
            sky_icon = "partly-cloudy-day"
        else:
            sky_icon = "partly-cloudy-night"
    elif cloudCover > mostlyClearThreshold and icon == "pirate":
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


def calculate_precip_text(
    prepIntensity,
    prepAccumUnit,
    prepType,
    type,
    rainPrep,
    snowPrep,
    icePrep,
    pop=1,
    icon="darksky",
    mode="both",
    isDayTime=True,
):
    """
    Calculates the precipitation

    Parameters:
    - prepIntensity (float): The precipitation intensity
    - prepAccumUnit (float): The precipitation accumulation/intensity unit
    - prepType (str): The type of precipitation
    - type (str): What type of summary is being generated.
    - rainPrep (float): The rain accumulation
    - snowPrep (float): The snow accumulation
    - icePrep (float): The ice accumulation
    - pop (float): The current probability of precipitation defaulting to 1
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the precipitation will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - str: The icon representing the current precipitation
    - str: The summary text representing the current precipitation
    """
    # In mm/h
    lightPrecipThresh = 0.4 * prepAccumUnit
    midPrecipThresh = 2.5 * prepAccumUnit
    heavyPrecipThresh = 10 * prepAccumUnit
    lightSnowThresh = 1.33 * prepAccumUnit
    midSnowThresh = 8.33 * prepAccumUnit
    heavySnowThresh = 33.33 * prepAccumUnit

    snowIconThresholdHour = 0.20 * prepAccumUnit
    precipIconThresholdHour = 0.02 * prepAccumUnit

    snowIconThresholdDay = 10.0 * prepAccumUnit
    precipIconThresholdDay = 1.0 * prepAccumUnit

    # Use daily or hourly thresholds depending on the situation
    if (type == "hour") or (type == "current"):
        snowIconThreshold = snowIconThresholdHour
        precipIconThreshold = precipIconThresholdHour
    elif type == "day":
        snowIconThreshold = snowIconThresholdDay
        precipIconThreshold = precipIconThresholdDay
        lightPrecipThresh = lightPrecipThresh * 24
        midPrecipThresh = midPrecipThresh * 24
        heavyPrecipThresh = heavyPrecipThresh * 24
        lightSnowThresh = lightSnowThresh * 24
        midSnowThresh = midSnowThresh * 24
        heavySnowThresh = heavySnowThresh * 24

    possiblePrecip = ""
    cIcon = None
    cText = None
    # Add the possible precipitation text if pop is less than 25% or if pop is greater than 0 but precipIntensity is between 0-0.02 mm/h
    if (pop < 0.25) or (
        (
            (rainPrep > 0)
            and (rainPrep < precipIconThreshold)
            and ((prepIntensity > 0) and (prepIntensity < precipIconThreshold))
        )
        or (
            (snowPrep > 0)
            and (snowPrep < snowIconThreshold)
            and ((prepIntensity > 0) and (prepIntensity < snowIconThreshold))
        )
        or (
            (icePrep > 0)
            and (icePrep < precipIconThreshold)
            and ((prepIntensity > 0) and (prepIntensity < precipIconThreshold))
        )
    ):
        possiblePrecip = "possible-"

    # Find the largest percentage difference compared to the thresholds
    # rainPrepPercent = rainPrep / rainIconThreshold
    # snowPrepPercent = snowPrep / snowIconThreshold
    # icePrepPercent = icePrep / iceIconThreshold

    # Find the largest percentage difference to determine the icon
    if pop >= 0.25 and (
        (rainPrep > precipIconThreshold and prepIntensity > precipIconThreshold)
        or (snowPrep > snowIconThreshold and prepIntensity > snowIconThreshold)
        or (icePrep > precipIconThreshold and prepIntensity > precipIconThreshold)
    ):
        if prepType == "none":
            cIcon = "rain"  # Fallback icon
        else:
            cIcon = prepType

    if rainPrep > 0 and prepIntensity > 0 and prepType == "rain":
        if prepIntensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-rain-night"
            elif icon == "pirate":
                cIcon = "drizzle"
        elif prepIntensity >= lightPrecipThresh and prepIntensity < midPrecipThresh:
            cText = possiblePrecip + "light-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-rain-night"
            elif icon == "pirate":
                cIcon = "light-rain"
        elif prepIntensity >= midPrecipThresh and prepIntensity < heavyPrecipThresh:
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
    elif snowPrep > 0 and prepIntensity > 0 and prepType == "snow":
        if prepIntensity < lightSnowThresh:
            cText = possiblePrecip + "very-light-snow"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-snow-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-snow-night"
            elif icon == "pirate":
                cIcon = "flurries"
        elif prepIntensity >= lightSnowThresh and prepIntensity < midSnowThresh:
            cText = possiblePrecip + "light-snow"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-snow-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-snow-night"
            elif icon == "pirate":
                cIcon = "light-snow"
        elif prepIntensity >= midSnowThresh and prepIntensity < heavySnowThresh:
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
    elif icePrep > 0 and prepIntensity > 0 and prepType == "sleet":
        if prepIntensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-sleet"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-sleet-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-sleet-night"
            elif icon == "pirate":
                cIcon = "very-light-sleet"
        elif prepIntensity >= lightPrecipThresh and prepIntensity < midPrecipThresh:
            cText = possiblePrecip + "light-sleet"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-sleet-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-sleet-night"
            elif icon == "pirate":
                cIcon = "light-sleet"
        elif prepIntensity >= midPrecipThresh and prepIntensity < heavyPrecipThresh:
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
    elif (
        rainPrep > 0 or snowPrep > 0 or icePrep > 0 or prepIntensity > 0
    ) and prepType == "none":
        if prepIntensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-precipitation"
        elif prepIntensity >= lightPrecipThresh and prepIntensity < midPrecipThresh:
            cText = possiblePrecip + "light-precipitation"
        elif prepIntensity >= midPrecipThresh and prepIntensity < heavyPrecipThresh:
            cText = possiblePrecip + "medium-precipitation"
        else:
            cText = possiblePrecip + "heavy-precipitation"

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


def calculate_wind_text(wind, windUnit, icon="darksky", mode="both"):
    """
    Calculates the wind text

    Parameters:
    - wind (float) -  The wind speed
    - windUnit (float) -  The unit of the wind speed

    Returns:
    - windText (str) - The textual representation of the wind
    - windIcon (str) - The icon representation of the wind
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the precipitation will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.
    """
    windText = None
    windIcon = None

    lightWindThresh = 6.7056 * windUnit
    midWindThresh = 10 * windUnit
    heavyWindThresh = 17.8816 * windUnit

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
            windIcon = "dangerously-windy"
        else:
            windIcon = "wind"

    if mode == "summary":
        return windText
    elif mode == "icon":
        return windIcon
    else:
        return windText, windIcon


def calculate_vis_text(vis, visUnits, mode="both"):
    """
    Calculates the visibility text

    Parameters:
    - vis (float) -  The visibility
    - visUnit (float) -  The unit of the visibility

    Returns:
    - visText (str) - The textual representation of the visibility
    - visIcon (str) - The icon representation of the visibility
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the precipitation will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.
    """
    visText = None
    visIcon = None
    visThresh = 1000 * visUnits

    if vis < visThresh:
        visText = "fog"
        visIcon = "fog"

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
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the precipitation will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - str: The icon representing the current cloud cover
    - str: The text representing the current cloud cover
    """
    skyText = None
    skyIcon = None

    if cloudCover > cloudyThreshold:
        skyText = "heavy-clouds"
        skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)

    elif cloudCover > partlyCloudyThreshold:
        skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)
        if cloudCover > mostlyCloudyThreshold:
            skyText = "medium-clouds"

        else:
            skyText = "light-clouds"
    else:
        skyIcon = calculate_sky_icon(cloudCover, isDayTime, icon)
        if cloudCover > mostlyClearThreshold:
            skyText = "very-light-clouds"
        else:
            skyText = "clear"

    if mode == "summary":
        return skyText
    elif mode == "icon":
        return skyIcon
    else:
        return skyText, skyIcon


def humidity_sky_text(temp, tempUnits, humidity):
    """
    Calculates the sky cover text

    Parameters:
    - temp (string): The temperature for the period
    - tempUnits (int): The temperature units
    - humidity (str): The humidity for the period

    Returns:
    - str: The text representing the humidity
    """

    # Only use humid if also warm (>20C)
    if tempUnits == 0:
        tempThresh = 68
    else:
        tempThresh = 20

    humidityText = None
    lowHumidityThresh = 0.15
    highHumidityThresh = 0.95

    if humidity <= lowHumidityThresh:
        humidityText = "low-humidity"
    elif humidity >= highHumidityThresh:
        if temp > tempThresh:
            humidityText = "high-humidity"

    return humidityText
