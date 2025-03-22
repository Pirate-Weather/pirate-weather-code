# %% Script to contain the functions that can be used to generate a simple daily text summary of the forecast data for Pirate Weather
from PirateText import calculate_text
import math


def calculate_precip_text(
    hourObject,
    prepAccumUnit,
    isDayTime,
    rainPrep,
    snowPrep,
    icePrep,
    precipType,
):
    if prepAccumUnit == 0.1:
        prepIntensityUnit = 1
    else:
        prepIntensityUnit = prepAccumUnit

    # In mm/h
    lightRainThresh = 0.4 * prepIntensityUnit
    midRainThresh = 2.5 * prepIntensityUnit
    heavyRainThresh = 10 * prepIntensityUnit
    lightSnowThresh = 0.13 * prepIntensityUnit
    midSnowThresh = 0.83 * prepIntensityUnit
    heavySnowThresh = 3.33 * prepIntensityUnit
    lightSleetThresh = 0.4 * prepIntensityUnit
    midSleetThresh = 2.5 * prepIntensityUnit
    heavySleetThresh = 10.0 * prepIntensityUnit

    snowIconThreshold = 10.0 * prepAccumUnit
    rainIconThreshold = 1.0 * prepAccumUnit
    iceIconThreshold = 1.0 * prepAccumUnit
    numTypes = 0
    totalPrep = rainPrep + snowPrep + icePrep

    if "precipProbability" in hourObject:
        pop = hourObject["precipProbability"]
    else:
        pop = 1

    if "precipIntensityMax" in hourObject:
        prepIntensity = hourObject["precipIntensityMax"]
    else:
        prepIntensity = totalPrep / 24

    possiblePrecip = ""
    cIcon = None
    cText = None

    # If the precipType is snow with no snow check if the other types have any precipitation and if they do then change the type of precipiation
    if snowPrep == 0 and precipType == "snow":
        if rainPrep > 0:
            precipType = "rain"
        elif icePrep > 0:
            precipType = "sleet"

    # If the precipType is rain with no rain check if the other types have any precipitation and if they do then change the type of precipiation
    if rainPrep == 0 and precipType == "rain":
        if snowPrep > 0:
            precipType = "snow"
        elif icePrep > 0:
            precipType = "sleet"

    # If the precipType is sleet with no sleet check if the other types have any precipitation and if they do then change the type of precipiation
    if icePrep == 0 and precipType == "sleet":
        if snowPrep > 0:
            precipType = "snow"
        elif rainPrep > 0:
            precipType = "rain"

    # Add the possible precipitation text if pop is less than 30% or if pop is greater than 0 but precipIntensity is between 0-0.02 mm/h
    if (pop < 0.25) or (
        (
            (rainPrep > 0)
            and (rainPrep < rainIconThreshold)
            and (precipType == "rain" or precipType == "none")
        )
        or ((snowPrep > 0) and (snowPrep < snowIconThreshold) and precipType == "snow")
        or ((icePrep > 0) and (icePrep < iceIconThreshold) and precipType == "sleet")
    ):
        possiblePrecip = "possible-"

    # Determine the number of precipitation types for the day
    if snowPrep > 0:
        numTypes += 1
    if rainPrep > 0:
        numTypes += 1
    if icePrep > 0:
        numTypes += 1

    # If there are more than one type of precipitation, pop is 25% and higher do not use the possible text
    if (
        totalPrep >= rainIconThreshold
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
        (rainPrep >= rainIconThreshold)
        or (snowPrep >= snowIconThreshold)
        or (icePrep >= iceIconThreshold)
        or (totalPrep >= rainIconThreshold)
    ):
        if precipType == "none":
            cIcon = "rain"  # Fallback icon
        else:
            cIcon = precipType

    if prepIntensity > 0 and precipType == "rain":
        if prepIntensity < lightRainThresh:
            cText = possiblePrecip + "very-light-rain"
        elif prepIntensity >= lightRainThresh and prepIntensity < midRainThresh:
            cText = possiblePrecip + "light-rain"
        elif prepIntensity >= midRainThresh and prepIntensity < heavyRainThresh:
            cText = "medium-rain"
        else:
            cText = "heavy-rain"
    elif prepIntensity > 0 and precipType == "snow":
        if prepIntensity < lightSnowThresh:
            cText = possiblePrecip + "very-light-snow"
        elif prepIntensity >= lightSnowThresh and prepIntensity < midSnowThresh:
            cText = possiblePrecip + "light-snow"
        elif prepIntensity >= midSnowThresh and prepIntensity < heavySnowThresh:
            cText = "medium-snow"
        else:
            cText = "heavy-snow"
    elif prepIntensity > 0 and precipType == "sleet":
        if prepIntensity < lightSleetThresh:
            cText = possiblePrecip + "very-light-sleet"
        elif prepIntensity >= lightSleetThresh and prepIntensity < midSleetThresh:
            cText = possiblePrecip + "light-sleet"
        elif prepIntensity >= midSleetThresh and prepIntensity < heavySleetThresh:
            cText = "medium-sleet"
        else:
            cText = "heavy-sleet"
    elif prepIntensity > 0 and precipType == "none":
        if prepIntensity < lightRainThresh:
            cText = possiblePrecip + "very-light-precipitation"
        elif prepIntensity >= lightRainThresh and prepIntensity < midRainThresh:
            cText = possiblePrecip + "light-precipitation"
        elif prepIntensity >= midRainThresh and prepIntensity < heavyRainThresh:
            cText = "medium-precipitation"
        else:
            cText = "heavy-precipitation"

    return cText, cIcon


def calculate_wind_text(wind, windUnit):
    """
    Calculates the wind text

    Parameters:
    - wind (float) -  The wind speed
    - windUnit (float) -  The unit of the wind speed

    Returns:
    - windText (str) - The textual representation of the wind
    - windIcon (str) - The icon representation of the wind
    """
    windText = None

    lightWindThresh = 6.7056 * windUnit
    midWindThresh = 10 * windUnit
    heavyWindThresh = 17.8816 * windUnit

    if wind >= lightWindThresh and wind < midWindThresh:
        windText = "light-wind"
    elif wind >= midWindThresh and wind < heavyWindThresh:
        windText = "medium-wind"
    elif wind >= heavyWindThresh:
        windText = "heavy-wind"

    return windText


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
    type,
):
    cText = cIcon = precipText = precipIcon = windText = secondary = snowText = (
        mixedText
    ) = snowSentence = None
    precipType = hourObject["precipType"]

    if "precipProbability" in hourObject:
        pop = hourObject["precipProbability"]
    else:
        pop = 1
    totalPrep = rainPrep + snowPrep + icePrep

    # Only calculate the precipitation text if there is any possibility of precipitation > 0
    if pop > 0 and totalPrep >= (0.01 * prepAccumUnit):
        # Check if there is rain, snow and ice accumulation for the day
        if snowPrep > 0 and rainPrep > 0 and icePrep > 0:
            # If there is then used the mixed precipitation text and set the icon/type to sleet. Set the secondary condition to snow so the totals can be in the summary
            precipText = "mixed-precipitation"
            precipType = "sleet"
            mixedText, precipIcon = calculate_precip_text(
                hourObject,
                prepAccumUnit,
                "day",
                rainPrep,
                snowPrep,
                icePrep,
                precipType,
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
                if icePrep > 0 and rainPrep > icePrep:
                    precipType = "rain"
                    secondary = "medium-sleet"
                # If we do check if we have ice. If there is more ice than rain then set rain as the secondary condition
                elif icePrep > 0 and rainPrep < icePrep:
                    precipType = "rain"
                    secondary = "medium-sleet"

            # If more than 10 mm of rain is forecast, then rain
            if rainPrep > (10 * prepAccumUnit) and precipType != "rain":
                secondary = precipType
                precipType = "rain"
            # If more than 5 mm of snow is forecast, then snow
            if snowPrep > (5 * prepAccumUnit) and precipType != "snow":
                secondary = precipType
                precipType = "snow"
            # Else, if more than 1 mm of ice is forecast, then ice
            if icePrep > (1 * prepAccumUnit) and precipType != "sleet":
                secondary = precipType
                precipType = "sleet"

            # Calculate the precipitation text and summary
            precipText, precipIcon = calculate_precip_text(
                hourObject,
                prepAccumUnit,
                "day",
                rainPrep,
                snowPrep,
                icePrep,
                precipType,
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

    # Calculate day text with no precipitation as is already calculated above
    dayText, dayIcon = calculate_text(
        hourObject,
        prepAccumUnit,
        visUnits,
        windUnit,
        tempUnits,
        isDayTime,
        0,
        0,
        0,
        "day",
        "sentence",
    )

    if precipText is not None:
        wind = hourObject["windSpeed"]
        windText = calculate_wind_text(wind, windUnit)

        if windText:
            cText = ["sentence", ["and", precipText, windText]]
        else:
            cText = ["sentence", precipText]
        cIcon = precipIcon

    if cIcon is None:
        cIcon = dayIcon
    if cText is None:
        cText = dayText

    return cText, cIcon
