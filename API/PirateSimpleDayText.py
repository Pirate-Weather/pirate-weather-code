# %% Script to contain the functions that can be used to generate a simple daily text summary of the forecast data for Pirate Weather
import math
import numpy as np

from PirateTextHelper import (
    calculate_precipitation,
    calculate_wind_text,
    calculate_visibility_text,
    calculate_sky_text,
    humidity_sky_text,
    DEFAULT_VALUES,
)

# Constants for thresholds
RAIN_THRESHOLD_MM = 10
SNOW_ACCUMULATION_THRESHOLD = 10
SNOW_THRESHOLD_MM = 5
ICE_THRESHOLD_MM = 1
PRECIP_ACCUMULATION_THRESHOLD = 0.01
LIGHT_ACCUMULATION_THRESHOLD = 0
SNOW_ERROR_DIVISOR = 2
DAY_LENGTH = 24
MEASUREMENT_CENTIMETERS = 0.1
MIN_POP_THRESHOLD = 0
MAX_PRECIP_TYPES = 3
SNOW_LESS_THAN_ACCUM_MM = 1


def calculate_simple_day_text(
    hourObject,
    prepAccumUnit,
    visUnits,
    windUnit,
    tempUnits,
    isDayTime,
    rainPrep,
    snowPrep,
    icePrep,
    icon="darksky",
):
    """
    Calculates the textual summary and icon with the given parameters

    Parameters:
    - hourObject (dict): A dictionary of the object used to generate the summary
    - prepAccumUnit (float): The precipitation unit used
    - visUnits (float): The visibility unit used
    - tempUnits (float): The temperature unit used
    - isDayTime (bool): Whether its currently day or night
    - rainPrep (float): The rain accumulation
    - snowPrep (float): The snow accumulation
    - icePrep (float): The ice accumulation
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - cText (str): A summary representing the conditions for the period.
    - cIcon (str): The icon representing the conditions for the period.
    """
    cText = cIcon = precipText = precipIcon = windText = windIcon = skyText = (
        skyIcon
    ) = visText = visIcon = secondary = snowText = snowSentence = None
    totalPrep = rainPrep + snowPrep + icePrep

    # Get key values from the hourObject
    precipType = hourObject["precipType"]
    cloudCover = hourObject["cloudCover"]
    wind = hourObject["windSpeed"]

    pop = hourObject.get("precipProbability", DEFAULT_VALUES["pop"])
    temp = hourObject.get("temperature", hourObject.get("temperatureHigh"))
    vis = hourObject.get("visibility", DEFAULT_VALUES["visibility"] * visUnits)
    # If time machine, no humidity data, so set to nan
    humidity = hourObject.get("humidity", np.nan)
    prepIntensityMax = hourObject.get("precipIntensityMax", totalPrep / DAY_LENGTH)

    if pop > MIN_POP_THRESHOLD and totalPrep >= (
        PRECIP_ACCUMULATION_THRESHOLD * prepAccumUnit
    ):
        # Determine which precipitation types are present
        presentPrecipTypes = {}
        if rainPrep > LIGHT_ACCUMULATION_THRESHOLD:
            presentPrecipTypes["rain"] = rainPrep
        if snowPrep > LIGHT_ACCUMULATION_THRESHOLD:
            presentPrecipTypes["snow"] = snowPrep
        if icePrep > LIGHT_ACCUMULATION_THRESHOLD:
            presentPrecipTypes["sleet"] = icePrep

        # If all three major types are present, it's mixed precipitation
        if len(presentPrecipTypes) == MAX_PRECIP_TYPES:
            precipText = "mixed-precipitation"
            precipType = "sleet"
            secondary = "medium-snow"
        else:
            # Determine the primary precipitation type based on accumulation
            # and set a secondary type if another is present
            sortedPrecip = sorted(
                presentPrecipTypes.items(), key=lambda item: item[1], reverse=True
            )

            if sortedPrecip:
                precipType = sortedPrecip[0][0]
                if len(sortedPrecip) > 1:
                    secondaryType = sortedPrecip[1][0]
                    secondary = f"medium-{secondaryType}"

            # Override primary precipType based on specific accumulation thresholds
            if rainPrep > (RAIN_THRESHOLD_MM * prepAccumUnit) and precipType != "rain":
                secondary = f"medium-{precipType}" if precipType else None
                precipType = "rain"
            elif (
                snowPrep > (SNOW_THRESHOLD_MM * prepAccumUnit) and precipType != "snow"
            ):
                secondary = f"medium-{precipType}" if precipType else None
                precipType = "snow"
            elif icePrep > (ICE_THRESHOLD_MM * prepAccumUnit) and precipType != "sleet":
                secondary = f"medium-{precipType}" if precipType else None
                precipType = "sleet"

        # Calculate the final precipitation text and summary
        if precipType:
            precipText, precipIcon = calculate_precipitation(
                prepIntensityMax,
                prepAccumUnit,
                precipType,
                "day",
                rainPrep,
                snowPrep,
                icePrep,
                pop,
                icon,
                "both",
            )

    if secondary == "medium-none":
        secondary = "medium-precipitation"

    # Check if snow accumulation is significant or if it's the secondary condition
    snowSentence = None
    if (
        snowPrep > (SNOW_ACCUMULATION_THRESHOLD * prepAccumUnit)
        or secondary == "medium-snow"
    ):
        # Calculate the accumulation range
        snowLowAccum = math.floor(snowPrep - (snowPrep / SNOW_ERROR_DIVISOR))
        snowMaxAccum = math.ceil(snowPrep + (snowPrep / SNOW_ERROR_DIVISOR))

        # Ensure the lower bound is not negative
        snowLowAccum = max(LIGHT_ACCUMULATION_THRESHOLD, snowLowAccum)

        # Determine the snow accumulation sentence based on the calculated range
        if snowMaxAccum > LIGHT_ACCUMULATION_THRESHOLD:
            if snowPrep == LIGHT_ACCUMULATION_THRESHOLD:
                # If no accumulation, show as < 1
                snowSentence = [
                    "less-than",
                    [
                        "centimeters"
                        if prepAccumUnit == MEASUREMENT_CENTIMETERS
                        else "inches",
                        SNOW_LESS_THAN_ACCUM_MM,
                    ],
                ]
            elif snowLowAccum == 0:
                # If lower range is 0, show as < max
                snowSentence = [
                    "less-than",
                    [
                        "centimeters"
                        if prepAccumUnit == MEASUREMENT_CENTIMETERS
                        else "inches",
                        snowMaxAccum,
                    ],
                ]
            else:
                # Otherwise, show the full range
                snowSentence = [
                    "centimeters"
                    if prepAccumUnit == MEASUREMENT_CENTIMETERS
                    else "inches",
                    ["range", snowLowAccum, snowMaxAccum],
                ]

    # If a snow sentence was generated, apply it to the summary text
    if snowSentence is not None:
        if precipType == "snow":
            # If snow is the primary precipitation type
            precipText = ["parenthetical", precipText, snowSentence]
        elif secondary == "medium-snow":
            # If snow is a secondary condition, create a separate text for it
            snowText = ["parenthetical", precipText, snowSentence]

    # Final check for a secondary condition to construct the final text
    if secondary is not None:
        if secondary != "medium-snow":
            precipText = ["and", precipText, secondary]
        else:
            precipText = snowText

    windText, windIcon = calculate_wind_text(wind, windUnit, icon, "both")
    visText, visIcon = calculate_visibility_text(vis, visUnits, "both")
    skyText, skyIcon = calculate_sky_text(cloudCover, isDayTime, icon, "both")
    humidityText = humidity_sky_text(temp, tempUnits, humidity)

    # Determine the primary and secondary text based on weather condition priority
    primaryText = None
    secondaryText = None

    if precipText:
        primaryText = precipText
        # Wind text takes priority over humidity as a secondary condition
        secondaryText = windText or humidityText
    elif visText:
        primaryText = visText
        secondaryText = humidityText
    elif windText:
        primaryText = windText
        # Combine with sky text unless it's clear
        if skyText != "clear":
            secondaryText = skyText
    elif humidityText:
        primaryText = humidityText
        secondaryText = skyText
    else:
        primaryText = skyText

    # Construct the final summary text
    cText = ["and", primaryText, secondaryText] if secondaryText else primaryText

    # Determine the icon based on priority
    cIcon = precipIcon or visIcon or windIcon or skyIcon

    return cText, cIcon
