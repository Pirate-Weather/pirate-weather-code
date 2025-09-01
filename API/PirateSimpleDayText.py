# %% Script to contain the functions that can be used to generate a simple daily text summary of the forecast data for Pirate Weather
import math
import numpy as np

from API.text_const import (
    DEFAULT_VISIBILITY,
    DEFAULT_POP,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
)

from PirateTextHelper import (
    calculate_precip_text,
    calculate_wind_text,
    calculate_vis_text,
    calculate_sky_text,
    humidity_sky_text,
)
from API.shared_const import MISSING_DATA

DAILY_RAIN_THRESHOLD = 10.0
DAILY_SNOW_THRESHOLD = 5.0


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
    # Get key values from the hourObject
    precipType = hourObject.get("precipType", "none")
    cloudCover = hourObject.get("cloudCover", 0)
    wind = hourObject.get("windSpeed", 0)
    pop = hourObject.get("pop", DEFAULT_POP)
    if pop == MISSING_DATA:
        pop = DEFAULT_POP
    temp = hourObject.get(
        "temperature", hourObject.get("temperatureHigh", MISSING_DATA)
    )
    vis = hourObject.get("visibility", DEFAULT_VISIBILITY * visUnits)
    humidity = hourObject.get("humidity", np.nan)
    prepIntensityMax = hourObject.get(
        "precipIntensityMax", (rainPrep + snowPrep + icePrep) / 24
    )
    dewPoint = hourObject.get("dewPoint", temp)
    smoke = hourObject.get("smoke", 0)

    cText = cIcon = precipText = precipIcon = windText = windIcon = skyText = (
        skyIcon
    ) = visText = visIcon = secondary = snowText = snowSentence = None
    totalPrep = rainPrep + snowPrep + icePrep

    # Only calculate the precipitation text if there is any possibility of precipitation > 0
    if pop > 0 and totalPrep >= (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit):
        # Check if there is rain, snow and ice accumulation for the day
        if snowPrep > 0 and rainPrep > 0 and icePrep > 0:
            # If there is then used the mixed precipitation text and set the icon/type to sleet. Set the secondary condition to snow so the totals can be in the summary
            precipText = "mixed-precipitation"
            precipType = "sleet"
            precipIcon = calculate_precip_text(
                prepIntensityMax,
                prepAccumUnit,
                precipType,
                "day",
                rainPrep,
                snowPrep,
                icePrep,
                pop,
                icon,
                "icon",
            )
            secondary = "medium-snow"
        else:
            # Otherwise check if we have any snow accumulation
            if snowPrep > 0:
                # If we do check if we have rain. If there is more snow than rain then set rain as the secondary condition
                if rainPrep > 0 and snowPrep > rainPrep:
                    precipType = "snow"
                    secondary = "medium-rain"
                # If we do check if we have rain. If there is more rain than snow then set snow as the secondary condition
                elif rainPrep > 0 and snowPrep < rainPrep:
                    precipType = "rain"
                    secondary = "medium-snow"
                # If we do check if we have ice. If there is more snow than ice then set ice as the secondary condition
                elif icePrep > 0 and snowPrep > icePrep:
                    precipType = "snow"
                    secondary = "medium-sleet"
                # If we do check if we have ice. If there is more ice than snow then set snow as the secondary condition
                elif icePrep > 0 and snowPrep < icePrep:
                    precipType = "sleet"
                    secondary = "medium-snow"
            # Otherwise check if we have any ice accumulation
            elif icePrep > 0:
                # If we do check if we have rain. If there is more rain than ice then set ice as the secondary condition
                if rainPrep > 0 and rainPrep > icePrep:
                    precipType = "rain"
                    secondary = "medium-sleet"
                # If we do check if we have ice. If there is more ice than rain then set rain as the secondary condition
                elif rainPrep > 0 and rainPrep < icePrep:
                    precipType = "rain"
                    secondary = "medium-sleet"

            # If the type is snow but there is no snow accumulation check if there is rain/sleet
            if snowPrep == 0 and precipType == "snow":
                if rainPrep > 0:
                    precipType = "rain"
                elif icePrep > 0:
                    precipType = "sleet"
            # If the type is rain but there is no rain accumulation check if there is snow/sleet
            elif rainPrep == 0 and precipType == "rain":
                if snowPrep > 0:
                    precipType = "snow"
                elif icePrep > 0:
                    precipType = "sleet"
            # If the type is sleet but there is no sleet accumulation check if there is rain/snow
            elif icePrep == 0 and precipType == "sleet":
                if snowPrep > 0:
                    precipType = "snow"
                elif rainPrep > 0:
                    precipType = "rain"

            # If more than 10 mm of rain is forecast, then rain
            if (
                rainPrep > (DAILY_RAIN_THRESHOLD * prepAccumUnit)
                and precipType != "rain"
            ):
                secondary = "medium-" + precipType
                precipType = "rain"
            # If more than 5 mm of snow is forecast, then snow
            if (
                snowPrep > (DAILY_SNOW_THRESHOLD * prepAccumUnit)
                and precipType != "snow"
            ):
                secondary = "medium-" + precipType
                precipType = "snow"
            # Else, if more than 1 mm of ice is forecast, then ice (use constant)
            if (
                icePrep > (DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit)
                and precipType != "sleet"
            ):
                secondary = "medium-" + precipType
                precipType = "sleet"

            # Calculate the precipitation text and summary
            precipText, precipIcon = calculate_precip_text(
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

    # If we have only snow or if snow is the secondary condition then calculate the accumulation range
    if (
        snowPrep > (DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit)
        or secondary == "medium-snow"
    ):
        # GEFS accumulation error seems to always be equal to the accumulation so use half of the accumulation as the range
        snowLowAccum = math.floor(snowPrep - (snowPrep / 2))
        snowMaxAccum = math.ceil(snowPrep + (snowPrep / 2))

        # If the snow accumulation is below 0; set it to 0
        if snowLowAccum < 0:
            snowLowAccum = 0

        # Check to see if there is any snow accumulation and if so calculate the sentence to use when creating the precipitation summaries
        if snowMaxAccum > 0:
            # If there is no accumulation then show the accumulation as < 1 cm/in
            if snowPrep == 0:
                snowSentence = [
                    "less-than",
                    ["centimeters" if prepAccumUnit == 0.1 else "inches", 1],
                ]
            # If the lower accumulation range is 0 then show accumulation as < max range cm/in
            elif snowLowAccum == 0:
                snowSentence = [
                    "less-than",
                    [
                        "centimeters" if prepAccumUnit == 0.1 else "inches",
                        snowMaxAccum,
                    ],
                ]
            # Otherwise show the range
            else:
                snowSentence = [
                    "centimeters" if prepAccumUnit == 0.1 else "inches",
                    [
                        "range",
                        snowLowAccum,
                        snowMaxAccum,
                    ],
                ]

    # If we have more than 1 cm of snow show the parenthetical or snow is the secondary condition
    if snowSentence is not None:
        # If precipitation is only show then generate the parenthetical text
        if precipType == "snow":
            precipText = [
                "parenthetical",
                precipText,
                snowSentence,
            ]
        # Otherwise if its a secondary condition then generate the text using the main condition
        elif secondary == "medium-snow":
            snowText = [
                "parenthetical",
                precipText,
                snowSentence,
            ]

    # If we have a secondary condition join them with an and if not snow otherwise use the snow text
    if secondary is not None:
        if secondary != "medium-snow":
            precipText = ["and", precipText, secondary]
        else:
            precipText = snowText

    windText, windIcon = calculate_wind_text(wind, windUnit, icon, "both")
    visText, visIcon = calculate_vis_text(
        vis, visUnits, tempUnits, temp, dewPoint, smoke, icon, "both"
    )
    skyText, skyIcon = calculate_sky_text(cloudCover, isDayTime, icon, "both")
    humidityText = humidity_sky_text(temp, tempUnits, humidity)

    if precipText is not None:
        if windText is not None:
            cText = ["and", precipText, windText]
        else:
            if humidityText is not None:
                cText = ["and", precipText, humidityText]
            else:
                cText = precipText
    elif visText is not None:
        if humidityText is not None:
            cText = ["and", visText, humidityText]
        else:
            cText = visText
    elif windText is not None:
        if skyText == "clear":
            cText = windText
        else:
            cText = ["and", windText, skyText]
    elif humidityText is not None:
        cText = ["and", humidityText, skyText]
    else:
        cText = skyText

    if precipIcon is not None:
        cIcon = precipIcon
    elif visIcon is not None:
        cIcon = visIcon
    elif windIcon is not None:
        cIcon = windIcon
    else:
        cIcon = skyIcon

    return cText, cIcon
