# %% Script to contain the helper functions that can be used to generate the text summary of the forecast data for Pirate Weather
from collections import Counter
import math


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

    if prepAccumUnit == 0.1:
        prepIntensityUnit = 1
    else:
        prepIntensityUnit = prepAccumUnit

    # If pop is -999 set it to 1 so we can calculate the precipitation text
    if pop == -999:
        pop = 1

    # In mm/h
    lightPrecipThresh = 0.4 * prepIntensityUnit
    midPrecipThresh = 2.5 * prepIntensityUnit
    heavyPrecipThresh = 10 * prepIntensityUnit
    lightSnowThresh = 0.13 * prepIntensityUnit
    midSnowThresh = 0.83 * prepIntensityUnit
    heavySnowThresh = 3.33 * prepIntensityUnit

    snowIconThresholdHour = 0.20 * prepAccumUnit
    precipIconThresholdHour = 0.02 * prepAccumUnit

    snowIconThresholdDay = 10.0 * prepAccumUnit
    precipIconThresholdDay = 1.0 * prepAccumUnit
    numTypes = 0

    # Use daily or hourly thresholds depending on the situation
    if type == "hour" or type == "current" or type == "minute":
        snowIconThreshold = snowIconThresholdHour
        precipIconThreshold = precipIconThresholdHour
    elif type == "day" or type == "week":
        snowIconThreshold = snowIconThresholdDay
        precipIconThreshold = precipIconThresholdDay

    possiblePrecip = ""
    cIcon = None
    cText = None
    totalPrep = rainPrep + snowPrep + icePrep
    # Add the possible precipitation text if pop is less than 25% or if pop is greater than 0 but precipIntensity is between 0-0.02 mm/h
    if (pop < 0.25) or (
        (
            (prepType == "rain" or prepType == "none")
            and (rainPrep > 0 and rainPrep < precipIconThreshold)
            or (prepIntensity > 0 and prepIntensity < precipIconThresholdHour)
        )
        or (
            prepType == "snow"
            and snowPrep > 0
            and snowPrep < snowIconThreshold
            or (prepIntensity > 0 and prepIntensity < snowIconThresholdHour)
        )
        or (
            (prepType == "sleet" or prepType == "ice" or prepType == "hail")
            and icePrep > 0
            and icePrep < precipIconThreshold
            or (prepIntensity > 0 and prepIntensity < precipIconThresholdHour)
        )
    ):
        possiblePrecip = "possible-"

    # Determine the number of precipitation types for the day
    if snowPrep > 0:
        numTypes += 1
    if rainPrep > 0:
        numTypes += 1
    if icePrep > 0:
        numTypes += 1

    if (
        totalPrep >= precipIconThreshold
        and possiblePrecip == "possible-"
        and pop >= 0.25
        and numTypes > 1
    ):
        possiblePrecip = ""

    # Find the largest percentage difference compared to the thresholds
    # rainPrepPercent = rainPrep / rainIconThreshold
    # snowPrepPercent = snowPrep / snowIconThreshold
    # icePrepPercent = icePrep / iceIconThreshold

    # Find the largest percentage difference to determine the icon
    if pop >= 0.25 and (
        (rainPrep > precipIconThreshold and prepIntensity > precipIconThresholdHour)
        or (snowPrep >= snowIconThreshold and prepIntensity > snowIconThresholdHour)
        or (icePrep >= precipIconThreshold and prepIntensity > precipIconThresholdHour)
        or (totalPrep >= precipIconThreshold and numTypes > 1)
    ):
        if prepType == "none":
            cIcon = "rain"  # Fallback icon
        elif prepType == "ice":
            cIcon = "freezing-rain"
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
        if (
            (type == "minute" or type == "week")
            and prepIntensity < heavyPrecipThresh
            and rainPrep >= heavyPrecipThresh
        ):
            cText = ["and", "medium-rain", "possible-heavy-rain"]
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
        if (
            (type == "minute" or type == "week")
            and prepIntensity < heavySnowThresh
            and snowPrep >= heavySnowThresh
        ):
            cText = ["and", "medium-snow", "possible-heavy-snow"]
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
        if (
            (type == "minute" or type == "week")
            and prepIntensity < heavyPrecipThresh
            and icePrep >= heavyPrecipThresh
        ):
            cText = ["and", "medium-sleet", "possible-heavy-sleet"]

    elif icePrep > 0 and prepIntensity > 0 and prepType == "ice":
        if prepIntensity < lightPrecipThresh:
            cText = possiblePrecip + "very-light-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
            elif icon == "pirate":
                cIcon = "freezing-drizzle"
        elif prepIntensity >= lightPrecipThresh and prepIntensity < midPrecipThresh:
            cText = possiblePrecip + "light-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
            elif icon == "pirate":
                cIcon = "light-freezing-rain"
        elif prepIntensity >= midPrecipThresh and prepIntensity < heavyPrecipThresh:
            cText = possiblePrecip + "medium-freezing-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
        else:
            cText = possiblePrecip + "heavy-rain"
            if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                cIcon = "possible-freezing-rain-day"
            elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
                cIcon = "possible-freezing-rain-night"
            elif icon == "pirate":
                cIcon = "heavy-freezing-rain"
        if (
            (type == "minute" or type == "week")
            and prepIntensity < heavyPrecipThresh
            and rainPrep >= heavyPrecipThresh
        ):
            cText = ["and", "medium-freezing-rain", "possible-heavy-freezing-rain"]
    elif icePrep > 0 and prepIntensity > 0 and prepType == "hail":
        cText = possiblePrecip + "hail"
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
        if (
            (type == "minute" or type == "week")
            and prepIntensity < heavyPrecipThresh
            and rainPrep >= heavyPrecipThresh
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


def calculate_wind_text(wind, windUnit, icon="darksky", mode="both"):
    """
    Calculates the wind text

    Parameters:
    - wind (float) -  The wind speed
    - windUnit (float) -  The unit of the wind speed
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the wind will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - windText (str) - The textual representation of the wind
    - windIcon (str) - The icon representation of the wind
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
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
            windIcon = "dangerous-wind"
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
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the visibility will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - visText (str) - The textual representation of the visibility
    - visIcon (str) - The icon representation of the visibility
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
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the cloud cover will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

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


def calculate_thunderstorm_text(liftedIndex, cape, mode="both"):
    """
    Calculates the thunderstorm text

    Parameters:
    - liftedIndex (float) -  The lifted index
    - cape (float) -  The CAPE (Convective available potential energy)
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the thunderstorm will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - thuText (str) - The textual representation of the thunderstorm
    - thuIcon (str) - The icon representation of the thunderstorm
    """
    thuText = None
    thuIcon = None

    if 1000 <= cape < 2500:
        thuText = "possible-thunderstorm"
    elif cape >= 2500:
        thuText = "thunderstorm"
        thuIcon = "thunderstorm"

    if liftedIndex != -999 and thuText is None:
        if 0 > liftedIndex > -4:
            thuText = "possible-thunderstorm"
        elif liftedIndex <= -4:
            thuText = "thunderstorm"
            thuIcon = "thunderstorm"

    if mode == "summary":
        return thuText
    elif mode == "icon":
        return thuIcon
    else:
        return thuText, thuIcon


def kelvinFromCelsius(celsius):
    return celsius + 273.15


def estimateSnowHeight(precipitationMm, temperatureC, windSpeedMps):
    snowDensityKgM3 = estimateSnowDensity(temperatureC, windSpeedMps)
    return precipitationMm * 10 / snowDensityKgM3

    # This one is too much, with its 10-100x range
    #
    # formula from https://www.omnicalculator.com/other/rain-to-snow#how-many-inches-of-snow-is-equal-to-one-inch-of-rain
    # ratioBase = 10.3 + (-1.21 * temperatureC) + (0.0389 * temperatureC * temperatureC)
    # print(ratioBase)
    # ratio = min(max(ratioBase, 1), 100)
    # snowMm = precipitationMm / ratio
    # return snowMm


# - Returns: kg/m3
def estimateSnowDensity(temperatureC, windSpeedMps):
    # interpolation at  https://docs.google.com/spreadsheets/d/1nrCN37VpoeDgAQHr70HcLDyyt-_dQdsRJMerpKMW0ho/edit?usp=sharing
    # Ratio ranges:
    # 3-30x: https://www.eoas.ubc.ca/courses/atsc113/snow/met_concepts/07-met_concepts/07b-newly-fallen-snow-density/
    # 3-20x: https://www.researchgate.net/figure/Common-densities-of-snow_tbl1_258653078
    # 4-20x: https://www.researchgate.net/figure/Fresh-snow-density-as-a-function-of-air-temperature-and-wind-for-the-3-options-included_fig2_316868161

    # Equations: from ESOLIP, https://www.tandfonline.com/eprint/Qf3k4JEPg3xXRmzp7gQQ/full (https://www.tandfonline.com/doi/pdf/10.1080/02626667.2015.1081203?needAccess=true)
    # Originally from https://sci-hub.hkvisa.net/10.1029/1999jc900011 (Jordan, R.E., Andreas, E.L., and Makshtas, A.P., 1999. Heat budget of snow-covered sea ice at North Pole 4. Journal of Geophysical Research)
    # Problem: These seem to be considering wind speed and it's factor on compacting the snow? Is that okay to use? According to ESOLIP paper probably yes.
    kelvins = kelvinFromCelsius(temperatureC)

    # above 2.5? bring it down, it shouldn't happen, but if it does, let's just assume it's 2.5 deg
    kelvins = min(kelvins, 275.65)

    windSpeedExp17 = pow(windSpeedMps, 1.7)

    snowDensityKgM3 = 1000
    if kelvins <= 260.15:
        snowDensityKgM3 = 500 * (1 - 0.904 * math.exp(-0.008 * windSpeedExp17))
    elif kelvins <= 275.65:
        snowDensityKgM3 = 500 * (
            1
            - 0.951
            * math.exp(-1.4 * pow(278.15 - kelvins, -1.15) - 0.008 * windSpeedExp17)
        )
    else:
        # above 2.5 degrees -> fallback, return precip mm (-> ratio = 1)
        # should not happen - see above
        snowDensityKgM3 = 1000

    # ensure we don't divide by zero - ensure minimum
    snowDensityKgM3 = max(snowDensityKgM3, 50)

    return snowDensityKgM3
