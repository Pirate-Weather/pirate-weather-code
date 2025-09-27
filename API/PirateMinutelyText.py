# %% Script to contain the functions that can be used to generate the minutely text summary of the forecast data for Pirate Weather

from itertools import groupby
from operator import itemgetter

from API.constants.shared_const import MISSING_DATA
from API.PirateTextHelper import calculate_precip_text

# Number of minutes in an hour
MINUTES_IN_HOUR = 60


def minutely_summary(precipStart1, precipEnd1, precipStart2, text):
    """
    Calculates the textual minutely summary

    Parameters:
    - precipStart1 (int): The time the first precipitation starts
    - precipEnd1 (int): The time when the first precipitation ends
    - precipStart2 (int): The time when the precipitation starts again
    - text (str): The textual representation of the precipitation

    Returns:
    - cText (arr): The precipitation summary for the hour.
    """

    # If the current precipitation stops before the end of the hour
    if precipStart1 == 0 and precipEnd1 < MINUTES_IN_HOUR and precipStart2 == -1:
        cText = [
            "stopping-in",
            text,
            ["minutes", precipEnd1 + 1]
            if precipEnd1 > 0
            else ["less-than", ["minutes", 1]],
        ]
    # If the current precipitation doesn't stop before the end of the hour
    elif precipStart1 == 0 and precipEnd1 == MINUTES_IN_HOUR:
        cText = [
            "for-hour",
            text,
        ]

    # If the current precipitation stops before the hour but starts again
    elif precipStart1 == 0 and precipEnd1 < MINUTES_IN_HOUR and precipStart2 != -1:
        cText = [
            "stopping-then-starting-later",
            text,
            ["minutes", precipEnd1 + 1]
            if precipEnd1 > 0
            else ["less-than", ["minutes", 1]],
            ["minutes", precipStart2 - precipEnd1],
        ]
    # If precip starts during the hour and lasts until the end of the hour
    elif precipStart1 > 0 and precipEnd1 == MINUTES_IN_HOUR:
        cText = [
            "starting-in",
            text,
            ["minutes", precipStart1],
        ]
    # If precip starts during the hour and ends before the end of the hour
    elif precipStart1 > 0 and precipEnd1 < MINUTES_IN_HOUR:
        cText = [
            "starting-then-stopping-later",
            text,
            ["minutes", precipStart1]
            if precipEnd1 > 0
            else ["less-than", ["minutes", 1]],
            ["minutes", precipEnd1 - precipStart1],
        ]
    return cText


def calcaulate_consecutive_indexes(prepIndex):
    """
    Calculates the consecutive indexes for the minutely precipitation arrays

    Parameters:
    - prepIndex (arr): An array of the precipitaion index

    Returns:
    - consecutiveIndex (arr): The array of consecutive index arrays
    """

    consecutiveIndex = []
    for k, g in groupby(enumerate(prepIndex), lambda ix: ix[0] - ix[1]):
        consecutiveIndex.append(list(map(itemgetter(1), g)))
    return consecutiveIndex


def calculate_minutely_text(
    minuteArr, currentText, currentIcon, icon, precipIntensityUnit
):
    """
    Calculates the minutely summary given an array of minutes

    Parameters:
    - minuteArr (arr): An array of the minutes
    - currentText (str/arr): The current conditions in translations format
    - currentIcon (str): The icon representing the current conditions
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - prepAccumUnit (float): The precipitation accumulation/intensity unit

    Returns:
    - cText (arr): The precipitation summary for the hour.
    - cIcon (str): The icon representing the conditions for the hour.
    """

    # Variables to use in calculating the minutely summary
    cIcon = cText = None
    precipMinutes = rainMaxIntensity = snowMaxIntensity = sleetMaxIntensity = (
        noneMaxIntensity
    ) = hailMaxIntensity = iceMaxIntensity = 0
    first_precip = "none"
    rainIndex = []
    snowIndex = []
    sleetIndex = []
    iceIndex = []
    hailIndex = []
    noneIndex = []
    precipIndex = []

    # Loop through the minute array
    for idx, minute in enumerate(minuteArr):
        if (
            minute["precipIntensity"] == MISSING_DATA
            or minute["precipType"] == MISSING_DATA
        ):
            return [
                "next-hour-forecast-status",
                "temporarily-unavailable",
                "station-offline",
            ], "none"
        # If there is rain for the current minute in the array
        if minute["precipType"] == "rain" and minute["precipIntensity"] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            precipMinutes += 1

            # Set the maxiumum rain intensity
            if rainMaxIntensity == 0:
                rainMaxIntensity = minute["precipIntensity"]
            elif rainMaxIntensity > 0 and minute["precipIntensity"] > rainMaxIntensity:
                rainMaxIntensity = minute["precipIntensity"]

            rainIndex.append(idx)
            precipIndex.append(idx)
        # If there is snow for the current minute in the array
        elif minute["precipType"] == "snow" and minute["precipIntensity"] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            precipMinutes += 1

            # Set the maxiumum snow intensity
            if snowMaxIntensity == 0:
                snowMaxIntensity = minute["precipIntensity"]
            elif snowMaxIntensity > 0 and minute["precipIntensity"] > snowMaxIntensity:
                snowMaxIntensity = minute["precipIntensity"]

            snowIndex.append(idx)
            precipIndex.append(idx)
        # If there is sleet for the current minute in the array
        elif minute["precipType"] == "sleet" and minute["precipIntensity"] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            precipMinutes += 1

            # Set the maxiumum sleet intensity
            if sleetMaxIntensity == 0:
                sleetMaxIntensity = minute["precipIntensity"]
            elif (
                sleetMaxIntensity > 0 and minute["precipIntensity"] > sleetMaxIntensity
            ):
                sleetMaxIntensity = minute["precipIntensity"]

            sleetIndex.append(idx)
            precipIndex.append(idx)
        elif minute["precipType"] == "none" and minute["precipIntensity"] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            precipMinutes += 1

            # Set the none maxiumum precipitation intensity
            if noneMaxIntensity == 0:
                noneMaxIntensity = minute["precipIntensity"]
            elif noneMaxIntensity > 0 and minute["precipIntensity"] > noneMaxIntensity:
                noneMaxIntensity = minute["precipIntensity"]

            noneIndex.append(idx)
            precipIndex.append(idx)
        elif minute["precipType"] == "ice" and minute["precipIntensity"] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            precipMinutes += 1

            # Set the none maxiumum precipitation intensity
            if iceMaxIntensity == 0:
                iceMaxIntensity = minute["precipIntensity"]
            elif iceMaxIntensity > 0 and minute["precipIntensity"] > iceMaxIntensity:
                iceMaxIntensity = minute["precipIntensity"]

            iceIndex.append(idx)
            precipIndex.append(idx)
        elif minute["precipType"] == "hail" and minute["precipIntensity"] > 0:
            # Increase the minutes of precipitation, the precipitation unit and average intensity
            precipMinutes += 1

            # Set the none maxiumum precipitation intensity
            if hailMaxIntensity == 0:
                hailMaxIntensity = minute["precipIntensity"]
            elif hailMaxIntensity > 0 and minute["precipIntensity"] > hailMaxIntensity:
                hailMaxIntensity = minute["precipIntensity"]

            hailIndex.append(idx)
            precipIndex.append(idx)

    # Create an array of the starting times for the precipitation
    starts = []

    # Create a list of consecutive minutes for each of the types to use in the summary text
    consecutiveSleet = calcaulate_consecutive_indexes(sleetIndex)
    consecutiveRain = calcaulate_consecutive_indexes(rainIndex)
    consecutiveSnow = calcaulate_consecutive_indexes(snowIndex)
    consecutiveIce = calcaulate_consecutive_indexes(iceIndex)
    consecutiveHail = calcaulate_consecutive_indexes(hailIndex)
    consecutiveNone = calcaulate_consecutive_indexes(noneIndex)
    consecutivePrep = calcaulate_consecutive_indexes(precipIndex)

    if sleetIndex:
        starts.append(sleetIndex[0])
    if snowIndex:
        starts.append(snowIndex[0])
    if rainIndex:
        starts.append(rainIndex[0])
    if noneIndex:
        starts.append(noneIndex[0])
    if iceIndex:
        starts.append(iceIndex[0])
        sleetMaxIntensity = iceMaxIntensity
    if hailIndex:
        starts.append(hailIndex[0])
        sleetMaxIntensity = hailMaxIntensity

    # Calculate the maximum intensity
    maxIntensity = max(
        rainMaxIntensity,
        snowMaxIntensity,
        sleetMaxIntensity,
        hailMaxIntensity,
        noneMaxIntensity,
    )

    # If the array has any values check the minimum against the different precipitation start times and set that as the first precipitaion
    if starts:
        if hailIndex:
            first_precip = "hail"
        elif sleetIndex and sleetIndex[0] == min(starts):
            first_precip = "sleet"
        elif snowIndex and snowIndex[0] == min(starts):
            first_precip = "snow"
        elif rainIndex and rainIndex[0] == min(starts):
            first_precip = "rain"
        elif noneIndex and noneIndex[0] == min(starts):
            first_precip = "none"
        elif iceIndex and iceIndex[0] == min(starts):
            first_precip = "ice"

    # If there are more than two precipitation types used the mixed text
    if len(starts) > 2:
        text = "mixed-precipitation"
        cIcon = "mixed"
    # If there is two precipitation types for the hour
    elif len(starts) == 2:
        # Calculate the precipitation text and icon
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            first_precip,
            "minute",
            rainMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )

        if first_precip == "hail" and rainIndex:
            text = [
                "and",
                calculate_precip_text(
                    maxIntensity,
                    precipIntensityUnit,
                    "rain",
                    "minute",
                    rainMaxIntensity,
                    snowMaxIntensity,
                    sleetMaxIntensity,
                    1,
                    icon,
                    "summary",
                ),
                "hail",
            ]

    # If there is no precipitation then set the minutely summary/icon to the current icon/summary
    if (
        not snowIndex
        and not rainIndex
        and not sleetIndex
        and not iceIndex
        and not hailIndex
        and not noneIndex
    ):
        cText = ["for-hour", currentText]
        cIcon = currentIcon
    # If there is more than one precipitation for the hour
    elif len(starts) > 1:
        # Calculate the text using the start/end times for precipitation as a whole instead of the individual precipitation
        cText = minutely_summary(
            consecutivePrep[0][0],
            consecutivePrep[0][len(consecutivePrep[0]) - 1],
            consecutivePrep[1][0] if len(consecutivePrep) > 1 else -1,
            text,
        )
    # If there if the only one precipitation is sleet
    elif sleetIndex:
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            "sleet",
            "minute",
            rainMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )
        cText = minutely_summary(
            consecutiveSleet[0][0],
            consecutiveSleet[0][len(consecutiveSleet[0]) - 1],
            consecutiveSleet[1][0] if len(consecutiveSleet) > 1 else -1,
            text,
        )
    # If there if the only one precipitation is snow
    elif snowIndex:
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            "snow",
            "minute",
            rainMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )
        cText = minutely_summary(
            consecutiveSnow[0][0],
            consecutiveSnow[0][len(consecutiveSnow[0]) - 1],
            consecutiveSnow[1][0] if len(consecutiveSnow) > 1 else -1,
            text,
        )
    # If there if the only one precipitation is rain
    elif rainIndex:
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            "rain",
            "minute",
            rainMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )
        cText = minutely_summary(
            consecutiveRain[0][0],
            consecutiveRain[0][len(consecutiveRain[0]) - 1],
            consecutiveRain[1][0] if len(consecutiveRain) > 1 else -1,
            text,
        )
    # If there if the only one precipitation is ice
    elif iceIndex:
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            "ice",
            "minute",
            rainMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )
        cText = minutely_summary(
            consecutiveIce[0][0],
            consecutiveIce[0][len(consecutiveIce[0]) - 1],
            consecutiveIce[1][0] if len(consecutiveIce) > 1 else -1,
            text,
        )
    # If there if the only one precipitation is hail
    elif iceIndex:
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            "hail",
            "minute",
            rainMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )
        cText = minutely_summary(
            consecutiveHail[0][0],
            consecutiveHail[0][len(consecutiveHail[0]) - 1],
            consecutiveHail[1][0] if len(consecutiveHail) > 1 else -1,
            text,
        )
    # If there if the only one precipitation has any other type
    else:
        text, cIcon = calculate_precip_text(
            maxIntensity,
            precipIntensityUnit,
            "none",
            "minute",
            noneMaxIntensity,
            snowMaxIntensity,
            sleetMaxIntensity,
            1,
            icon,
            "both",
        )
        cText = minutely_summary(
            consecutiveNone[0][0],
            consecutiveNone[0][len(consecutiveNone[0]) - 1],
            consecutiveNone[1][0] if len(consecutiveNone) > 1 else -1,
            text,
        )

    # If we have no icon fallback to the current icon
    if cIcon is None:
        cIcon = currentIcon

    return cText, cIcon
