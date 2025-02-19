from PirateTextHelper import (
    calculate_precip_text,
    calculate_wind_text,
    calculate_vis_text,
    calculate_sky_text,
    humidity_sky_text,
)
import math


def calculate_day_text(
    hourObject,
    prepAccumUnit,
    visUnits,
    windUnit,
    tempUnits,
    isDayTime,
    rainPrep,
    snowPrep,
    icePrep,
    precipIntensity,
    precipIntensityMax,
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
    - precipIntensity (float): The precipitation intensity
    - precipIntensityMax (float): The maximum precipitation intensity
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - cText (str): A summary representing the conditions for the period.
    - cIcon (str): The icon representing the conditions for the period.
    """
    cText = cIcon = precipText = precipIcon = windText = windIcon = skyText = (
        skyIcon
    ) = visText = visIcon = secondary = snowText = snowSentence = None

    # Get key values from the hourObject
    precipType = hourObject["precipType"]
    cloudCover = hourObject["cloudCover"]
    wind = hourObject["windSpeed"]
    humidity = hourObject["humidity"]

    if "precipProbability" in hourObject:
        pop = hourObject["precipProbability"]
    else:
        pop = 1

    if "temperature" in hourObject:
        temp = hourObject["temperature"]
    else:
        temp = hourObject["temperatureHigh"]

    if "visibility" in hourObject:
        vis = hourObject["visibility"]
    else:
        vis = 10000

    # Check if there is rain, snow and ice accumulation for the day
    if snowPrep > 0 and rainPrep > 0 and icePrep > 0:
        # If there is then used the mixed precipitation text and set the icon/type to sleet. Set the secondary condition to snow so the totals can be in the summary
        precipText = "mixed-precipitation"
        precipIcon = "sleet"
        precipType = "sleet"
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
                precipIntensityMax = rainPrep / 24
                precipType = "rain"
                secondary = "medium-snow"
            # If we do check if we have ice. If there is more snow than ice then set ice as the secondary condition
            elif icePrep > 0 and snowPrep > icePrep:
                precipType = "snow"
                secondary = "medium-sleet"
            # If we do check if we have ice. If there is more ice than snow then set snow as the secondary condition
            elif icePrep > 0 and snowPrep < icePrep:
                precipIntensityMax = icePrep / 24
                precipType = "sleet"
                secondary = "medium-snow"
        # Otherwise check if we have any ice accumulation
        elif icePrep > 0:
            # If we do check if we have rain. If there is more rain than ice then set ice as the secondary condition
            if icePrep > 0 and rainPrep > icePrep:
                precipType = "rain"
                secondary = "medium-sleet"
            # If we do check if we have ice. If there is more ice than rain then set rain as the secondary condition
            elif icePrep > 0 and rainPrep < icePrep:
                precipType = "rain"
                secondary = "medium-sleet"

        # Calculate the precipitation text and summary
        precipText, precipIcon = calculate_precip_text(
            precipIntensityMax,
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

    # If we have only snow or if snow is the secondary condition then calculate the accumulation range
    if snowPrep > (5 * prepAccumUnit) or secondary == "medium-snow":
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

    # If we have more than 0.5 cm of snow show the parenthetical
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
    visText, visIcon = calculate_vis_text(vis, visUnits, "both")
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
            cText = ["and", humidityText, skyText]
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
