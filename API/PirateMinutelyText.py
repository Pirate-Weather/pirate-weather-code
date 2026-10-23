# %% Script to contain the functions that can be used to generate the minutely text summary of the forecast data for Pirate Weather

import math
from itertools import groupby

import numpy as np

from API.constants.api_const import PRECIP_TYPES
from API.PirateTextHelper import calculate_precip_text

# Number of minutes in an hour
MINUTES_IN_HOUR = 60

# Precipitation types that are tracked minute by minute. Minutes with any
# other type (e.g. "mixed") are not counted as precipitation.
MINUTELY_PRECIP_TYPES = (
    PRECIP_TYPES["sleet"],
    PRECIP_TYPES["snow"],
    PRECIP_TYPES["rain"],
    PRECIP_TYPES["none"],
    PRECIP_TYPES["ice"],
    PRECIP_TYPES["hail"],
)


def minutely_summary(precipStart1, precipEnd1, precipStart2, text):
    """
    Calculates the textual minutely summary

    Parameters:
    - precipStart1 (int): The time the first precipitation starts
    - precipEnd1 (int): The time when the first precipitation ends
    - precipStart2 (int): The time when the precipitation starts again
    - text (str): The textual representation of the precipitation

    Returns:
    - c_text (arr): The precipitation summary for the hour.
    """
    # A stop in the first minute is reported as "less than a minute"
    stop_minutes = (
        ["minutes", precipEnd1 + 1] if precipEnd1 > 0 else ["less-than", ["minutes", 1]]
    )

    # Precipitation is falling now
    if precipStart1 == 0:
        # It doesn't stop before the end of the hour
        if precipEnd1 >= MINUTES_IN_HOUR:
            return ["for-hour", text]
        # It stops before the end of the hour
        if precipStart2 == -1:
            return ["stopping-in", text, stop_minutes]
        # It stops before the end of the hour but starts again
        return [
            "stopping-then-starting-later",
            text,
            stop_minutes,
            ["minutes", precipStart2 - precipEnd1],
        ]

    # Precipitation starts during the hour and lasts until the end of the hour
    if precipEnd1 >= MINUTES_IN_HOUR:
        return ["starting-in", text, ["minutes", precipStart1]]
    # Precipitation starts during the hour and ends before the end of the hour
    return [
        "starting-then-stopping-later",
        text,
        ["minutes", precipStart1],
        ["minutes", precipEnd1 - precipStart1],
    ]


def calcaulate_consecutive_indexes(prepIndex):
    """
    Calculates the consecutive indexes for the minutely precipitation arrays

    Parameters:
    - prepIndex (arr): An array of the precipitaion index

    Returns:
    - consecutiveIndex (arr): The array of consecutive index arrays
    """
    return [
        [idx for _, idx in group]
        for _, group in groupby(enumerate(prepIndex), lambda ix: ix[0] - ix[1])
    ]


def calculate_minutely_text(minuteArr, currentIcon, icon):
    """
    Calculates the minutely summary given an array of minutes.
    All inputs are expected in SI units (mm/h for precipitation intensity).

    Parameters:
    - minuteArr (arr): An array of the minutely objects (with SI units)
    - currentIcon (str): The icon representing the current conditions
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - c_text (arr): The precipitation summary for the hour.
    - c_icon (str): The icon representing the conditions for the hour.
    """
    # Minutes with precipitation, grouped by type and for all types together
    type_minutes = {precip_type: [] for precip_type in MINUTELY_PRECIP_TYPES}
    precip_minutes = []

    for idx, minute in enumerate(minuteArr):
        precip_type = minute["precipType"]
        precip_intensity = minute["precipIntensity"]

        # Any missing minute means the minutely source is unavailable
        if math.isnan(precip_intensity) or (
            isinstance(precip_type, (float, np.floating)) and math.isnan(precip_type)
        ):
            return [
                "next-hour-forecast-status",
                "temporarily-unavailable",
                "station-offline",
            ], PRECIP_TYPES["none"]

        if precip_intensity > 0 and precip_type in type_minutes:
            type_minutes[precip_type].append(idx)
            precip_minutes.append(idx)

    # Without any precipitation in the hour say so rather than repeating the
    # current conditions, keeping the current icon.
    if not precip_minutes:
        return ["for-hour", "no-precipitation"], currentIcon

    # Peak intensity (mm/h) for each type with precipitation
    max_intensity = {
        precip_type: max(minuteArr[idx]["precipIntensity"] for idx in minutes)
        for precip_type, minutes in type_minutes.items()
        if minutes
    }
    rain_max_intensity = max_intensity.get(PRECIP_TYPES["rain"], 0)
    snow_max_intensity = max_intensity.get(PRECIP_TYPES["snow"], 0)
    # Hail, then freezing rain, take precedence over sleet as the ice intensity
    ice_max_intensity = max_intensity.get(
        PRECIP_TYPES["hail"],
        max_intensity.get(
            PRECIP_TYPES["ice"], max_intensity.get(PRECIP_TYPES["sleet"], 0)
        ),
    )

    # Hail always leads the summary, otherwise the type that starts first
    if PRECIP_TYPES["hail"] in max_intensity:
        first_precip = PRECIP_TYPES["hail"]
    else:
        first_precip = min(max_intensity, key=lambda t: type_minutes[t][0])

    c_icon = None
    num_types = len(max_intensity)

    if num_types > 2:
        # More than two precipitation types uses the mixed text
        text = "mixed-precipitation"
        c_icon = PRECIP_TYPES["mixed"]
    elif num_types == 2:
        # Two types are summarised by the type that starts first
        text, c_icon = calculate_precip_text(
            first_precip,
            "minute",
            0,
            0,
            0,
            1,
            icon=icon,
            mode="both",
            eff_rain_intensity=rain_max_intensity,
            eff_snow_intensity=snow_max_intensity,
            eff_ice_intensity=ice_max_intensity,
        )
        if first_precip == PRECIP_TYPES["hail"] and type_minutes[PRECIP_TYPES["rain"]]:
            text = [
                "and",
                calculate_precip_text(
                    PRECIP_TYPES["rain"],
                    "minute",
                    0,
                    0,
                    0,
                    1,
                    icon=icon,
                    mode="summary",
                    eff_rain_intensity=rain_max_intensity,
                    eff_snow_intensity=snow_max_intensity,
                    eff_ice_intensity=ice_max_intensity,
                ),
                "hail",
            ]
    else:
        # A single type. Precipitation with an unknown type is described by
        # its own intensity, as there is no rain intensity for it.
        if first_precip == PRECIP_TYPES["none"]:
            rain_max_intensity = max_intensity[PRECIP_TYPES["none"]]
        text, c_icon = calculate_precip_text(
            first_precip,
            "minute",
            0,
            0,
            0,
            1,
            icon=icon,
            mode="both",
            eff_rain_intensity=rain_max_intensity,
            eff_snow_intensity=snow_max_intensity,
            eff_ice_intensity=ice_max_intensity,
        )

    # Timing uses the start/end of precipitation as a whole, not each type
    precip_periods = calcaulate_consecutive_indexes(precip_minutes)
    c_text = minutely_summary(
        precip_periods[0][0],
        precip_periods[0][-1],
        precip_periods[1][0] if len(precip_periods) > 1 else -1,
        text,
    )

    # If we have no icon fallback to the current icon
    if c_icon is None:
        c_icon = currentIcon

    return c_text, c_icon
