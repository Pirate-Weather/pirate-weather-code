# %% Script to contain the helper functions that can be used to generate the text summary of the forecast data for Pirate Weather

from collections import Counter
from itertools import groupby
from operator import itemgetter
import math

# Constants
CLOUD_COVER_THRESHOLDS = {
    "cloudy": 0.875,
    "mostly_cloudy": 0.625,
    "partly_cloudy": 0.375,
    "mostly_clear": 0.125,
}

PRECIPITATION_INTENSITY_THRESHOLDS = {
    "light": 0.4,
    "mid": 2.5,
    "heavy": 10.0,
}

SNOW_INTENSITY_THRESHOLDS = {
    "light": 0.13,
    "mid": 0.83,
    "heavy": 3.33,
}

ICON_THRESHOLDS = {
    "hourly_snow_accumulation": 0.2,
    "hourly_precipitation_accumulation": 0.02,
    "daily_snow_accumulation": 10.0,
    "daily_precipitation_accumulation": 1.0,
    "fog_visibility": 1000,
    "mist_visibility": 10000,
    "smoke_concentration": 25,
    "temp_dewpoint_spread_fog": 2,
    "temp_dewpoint_spread_mist": 3,
}

WIND_THRESHOLDS = {
    "light": 6.7056,
    "mid": 10,
    "heavy": 17.8816,
}

DEFAULT_VALUES = {
    "visibility": 10000,
    "pop": 1,
}

MISSING_DATA = -999
PRECIP_PROBABILITY_THRESHOLD = 0.25

# New constants for refactoring
HEAVY_SNOW_AVG_FACTOR = 0.66
HEAVY_OTHER_PRECIP_AVG_FACTOR = 2
SUMMARY_TYPE_MINUTE = "minute"
SUMMARY_TYPE_WEEK = "week"
SUMMARY_TYPE_HOURLY = "hourly"

# Icon mapping for Pirate Weather specific icons
PIRATE_ICON_MAP = {
    "rain": {
        "very-light": "drizzle",
        "light": "light-rain",
        "medium": "rain",
        "heavy": "heavy-rain",
    },
    "snow": {
        "very-light": "flurries",
        "light": "light-snow",
        "medium": "snow",
        "heavy": "heavy-snow",
    },
    "sleet": {
        "very-light": "very-light-sleet",
        "light": "light-sleet",
        "medium": "medium-sleet",
        "heavy": "heavy-sleet",
    },
    "ice": {  # Corresponds to freezing-rain in text
        "very-light": "freezing-drizzle",
        "light": "light-freezing-rain",
        "medium": "freezing-rain",
        "heavy": "heavy-freezing-rain",
    },
    "precipitation": {  # For generic "none" type precipitation
        "default": "precipitation"
    },
}


def most_common(lst):
    """
    Finds the most common icon to use as the icon.

    Parameters:
    - lst (arr): An array of weekly icons.

    Returns:
    - str: The most common icon in the list.
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]


def calculate_sky_icon(cloudCover, isDayTime, iconSet="darksky"):
    """
    Calculates the sky cover icon.

    Parameters:
    - cloudCover (int): The cloud cover for the period.
    - isDayTime (bool): Whether it's currently day or night.
    - iconSet (str): Which icon set to use - Dark Sky or Pirate Weather.

    Returns:
    - str: The icon representing the current cloud cover.
    """
    if cloudCover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        return "cloudy"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"] and iconSet == "pirate":
        return "mostly-cloudy-day" if isDayTime else "mostly-cloudy-night"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        return "partly-cloudy-day" if isDayTime else "partly-cloudy-night"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["mostly_clear"] and iconSet == "pirate":
        return "mostly-clear-day" if isDayTime else "mostly-clear-night"
    else:
        return "clear-day" if isDayTime else "clear-night"


def _get_intensity_level(intensity, lightThresh, midThresh, heavyThresh):
    """
    Determines the intensity level (e.g., 'very-light', 'light') based on thresholds.

    Parameters:
    - intensity (float): The precipitation intensity value.
    - lightThresh (float): The threshold for light intensity.
    - midThresh (float): The threshold for medium intensity.
    - heavyThresh (float): The threshold for heavy intensity.

    Returns:
    - str: The string representing the intensity level.
    """
    if intensity < lightThresh:
        return "very-light"
    elif intensity < midThresh:
        return "light"
    elif intensity < heavyThresh:
        return "medium"
    else:
        return "heavy"


def _get_precip_text_and_icon_for_type(
    precipType,
    precipitationIntensity,
    intensityThresholds,
    possiblePrecipitation,
    iconSet,
    isDayTime,
    summaryType,
    avgPrecipitation,
    precipitationAccumUnit,
    accumulation,
):
    """
    Helper function to determine precipitation text and icon for a specific type.

    Parameters:
    - precipType (str): The type of precipitation (e.g., "rain", "snow", "sleet", "ice", "none").
    - precipitationIntensity (float): The current precipitation intensity.
    - intensityThresholds (dict): Dictionary containing light, mid, and heavy thresholds for the precipitation type.
    - possiblePrecipitation (str): Prefix for "possible-" precipitation.
    - iconSet (str): Which icon set to use - Dark Sky or Pirate Weather.
    - isDayTime (bool): Whether it's currently day or night.
    - summaryType (str): What type of summary is being generated (e.g., "minute", "week", "hourly").
    - avgPrecipitation (float): The average precipitation intensity for the period.
    - precipitationAccumUnit (float): The precipitation accumulation/intensity unit.
    - accumulation (float): The total accumulation for the specific precipitation type.

    Returns:
    - tuple[str | None, str | None]: The textual representation and/or icon of the precipitation.
    """
    currentText = None
    currentIcon = None

    intensityLevel = _get_intensity_level(
        precipitationIntensity,
        intensityThresholds["light"],
        intensityThresholds["mid"],
        intensityThresholds["heavy"],
    )

    # Determine base text
    if precipType == "ice":  # Special case for "freezing-rain" text
        currentText = f"{possiblePrecipitation}{intensityLevel}-freezing-rain"
    elif precipType == "none":
        currentText = f"{possiblePrecipitation}{intensityLevel}-precipitation"
    else:
        currentText = f"{possiblePrecipitation}{intensityLevel}-{precipType}"

    # Determine icon for Pirate Weather
    if iconSet == "pirate":
        if possiblePrecipitation == "":
            if precipType == "none":
                currentIcon = PIRATE_ICON_MAP["precipitation"]["default"]
            else:
                currentIcon = PIRATE_ICON_MAP[precipType][intensityLevel]
        else:
            if precipType == "none":
                currentIcon = (
                    f"possible-precipitation-{'day' if isDayTime else 'night'}"
                )
            else:
                currentIcon = f"possible-{precipType}-{'day' if isDayTime else 'night'}"
    else:
        # Logic for non-pirate icon sets (e.g., Dark Sky)
        if precipType == "none":
            currentIcon = "rain"
        elif precipType == "ice":
            currentIcon = "freezing-rain"
        else:
            currentIcon = precipType

    # Handle specific text adjustments for heavy precipitation in certain summary types
    if summaryType in [SUMMARY_TYPE_MINUTE, SUMMARY_TYPE_WEEK, SUMMARY_TYPE_HOURLY]:
        if precipType == "snow":
            if (
                avgPrecipitation
                < (
                    intensityThresholds["heavy"]
                    - (HEAVY_SNOW_AVG_FACTOR * precipitationAccumUnit)
                )
                and precipitationIntensity >= intensityThresholds["heavy"]
            ):
                currentText = [
                    "and",
                    f"medium-{precipType}",
                    f"possible-heavy-{precipType}",
                ]
        elif precipType in ["rain", "sleet", "ice", "none"]:
            if (
                avgPrecipitation
                < (
                    intensityThresholds["heavy"]
                    - (HEAVY_OTHER_PRECIP_AVG_FACTOR * precipitationAccumUnit)
                )
                and precipitationIntensity >= intensityThresholds["heavy"]
            ):
                currentText = [
                    "and",
                    f"medium-{precipType}",
                    f"possible-heavy-{precipType}",
                ]

        # Specific check for rain accumulation in minute/week summary
        if (
            precipType == "rain"
            and summaryType in [SUMMARY_TYPE_MINUTE, SUMMARY_TYPE_WEEK]
            and precipitationIntensity < intensityThresholds["heavy"]
            and accumulation >= intensityThresholds["heavy"]
        ):
            currentText = ["and", "medium-rain", "possible-heavy-rain"]

    return currentText, currentIcon


def calculate_precipitation(
    precipitationIntensity,
    precipitationAccumUnit,
    precipitationType,
    summaryType,
    rainAccumulation,
    snowAccumulation,
    iceAccumulation,
    pop=DEFAULT_VALUES["pop"],
    iconSet="darksky",
    mode="both",
    isDayTime=True,
    avgPrecipitation=0,
):
    """
    Calculates the precipitation summary and icon.

    Parameters:
    - precipitationIntensity (float): The precipitation intensity.
    - precipitationAccumUnit (float): The precipitation accumulation/intensity unit.
    - precipitationType (str): The type of precipitation.
    - summaryType (str): What type of summary is being generated.
    - rainAccumulation (float): The rain accumulation.
    - snowAccumulation (float): The snow accumulation.
    - iceAccumulation (float): The ice accumulation.
    - pop (float): The current probability of precipitation.
    - iconSet (str): Which icon set to use - Dark Sky or Pirate Weather.
    - mode (str): Determines what gets returned. "both" returns summary and icon, "icon" returns only the icon, "summary" returns only the summary.
    - avgPrecipitation (float): The average precipitation intensity.

    Returns:
    - tuple[str | None, str | None] or str | None: The summary text and/or icon.
    """
    if any(
        x == MISSING_DATA
        for x in (
            rainAccumulation,
            snowAccumulation,
            iceAccumulation,
            precipitationIntensity,
        )
    ):
        return (None, None) if mode == "both" else None

    if pop == MISSING_DATA:
        pop = 1

    if summaryType in ["hour", "current", SUMMARY_TYPE_MINUTE, SUMMARY_TYPE_HOURLY]:
        snowIconThreshold = (
            ICON_THRESHOLDS["hourly_snow_accumulation"] * precipitationAccumUnit
        )
        precipitationIconThreshold = (
            ICON_THRESHOLDS["hourly_precipitation_accumulation"]
            * precipitationAccumUnit
        )
    else:  # "day", "week"
        snowIconThreshold = (
            ICON_THRESHOLDS["daily_snow_accumulation"] * precipitationAccumUnit
        )
        precipitationIconThreshold = (
            ICON_THRESHOLDS["daily_precipitation_accumulation"] * precipitationAccumUnit
        )

    possiblePrecipitation = ""
    currentIcon = None
    currentText = None
    totalPrecipitation = rainAccumulation + snowAccumulation + iceAccumulation

    # A flag to check for very light precipitation, which would be described as "possible"
    isPossible = False
    if precipitationType in ["rain", "none"] and (
        (0 < rainAccumulation < precipitationIconThreshold)
        or (
            0
            < precipitationIntensity
            < ICON_THRESHOLDS["hourly_precipitation_accumulation"]
        )
    ):
        isPossible = True
    elif precipitationType == "snow" and (
        (0 < snowAccumulation < snowIconThreshold)
        or (0 < precipitationIntensity < ICON_THRESHOLDS["hourly_snow_accumulation"])
    ):
        isPossible = True
    elif precipitationType in ["sleet", "ice", "hail"] and (
        (0 < iceAccumulation < precipitationIconThreshold)
        or (
            0
            < precipitationIntensity
            < ICON_THRESHOLDS["hourly_precipitation_accumulation"]
        )
    ):
        isPossible = True

    # Determine if "possible" prefix should be used based on low probability or very light precip
    if pop < PRECIP_PROBABILITY_THRESHOLD or isPossible:
        possiblePrecipitation = "possible-"

    # Check for multiple precipitation types

    numTypes = sum(
        [
            1
            for accum in [snowAccumulation, rainAccumulation, iceAccumulation]
            if accum > 0
        ]
    )
    if (
        totalPrecipitation >= precipitationIconThreshold
        and possiblePrecipitation == "possible-"
        and pop >= PRECIP_PROBABILITY_THRESHOLD
        and numTypes > 1
    ):
        possiblePrecipitation = ""

    # Set icon based on precipitation type and thresholds
    isSignificantRain = (
        rainAccumulation > precipitationIconThreshold
        and precipitationIntensity
        > ICON_THRESHOLDS["hourly_precipitation_accumulation"]
    )
    isSignificantSnow = (
        snowAccumulation >= snowIconThreshold
        and precipitationIntensity > ICON_THRESHOLDS["hourly_snow_accumulation"]
    )
    isSignificantIce = (
        iceAccumulation >= precipitationIconThreshold
        and precipitationIntensity
        > ICON_THRESHOLDS["hourly_precipitation_accumulation"]
    )
    isMixedPrecipitation = (
        totalPrecipitation >= precipitationIconThreshold and numTypes > 1
    )

    if pop >= PRECIP_PROBABILITY_THRESHOLD and (
        isSignificantRain
        or isSignificantSnow
        or isSignificantIce
        or isMixedPrecipitation
    ):
        if precipitationType == "none":
            currentIcon = "rain"
        elif precipitationType == "ice":
            currentIcon = "freezing-rain"
        else:
            currentIcon = precipitationType

    # Determine summary text and specific icons
    if (
        rainAccumulation > 0
        and precipitationIntensity > 0
        and precipitationType == "rain"
    ):
        currentText, currentIcon = _get_precip_text_and_icon_for_type(
            "rain",
            precipitationIntensity,
            PRECIPITATION_INTENSITY_THRESHOLDS,
            possiblePrecipitation,
            iconSet,
            isDayTime,
            summaryType,
            avgPrecipitation,
            precipitationAccumUnit,
            rainAccumulation,
        )
    elif (
        snowAccumulation > 0
        and precipitationIntensity > 0
        and precipitationType == "snow"
    ):
        currentText, currentIcon = _get_precip_text_and_icon_for_type(
            "snow",
            precipitationIntensity,
            SNOW_INTENSITY_THRESHOLDS,
            possiblePrecipitation,
            iconSet,
            isDayTime,
            summaryType,
            avgPrecipitation,
            precipitationAccumUnit,
            snowAccumulation,
        )
    elif (
        iceAccumulation > 0
        and precipitationIntensity > 0
        and precipitationType == "sleet"
    ):
        currentText, currentIcon = _get_precip_text_and_icon_for_type(
            "sleet",
            precipitationIntensity,
            PRECIPITATION_INTENSITY_THRESHOLDS,
            possiblePrecipitation,
            iconSet,
            isDayTime,
            summaryType,
            avgPrecipitation,
            precipitationAccumUnit,
            iceAccumulation,
        )
    elif (
        iceAccumulation > 0
        and precipitationIntensity > 0
        and precipitationType == "ice"
    ):
        currentText, currentIcon = _get_precip_text_and_icon_for_type(
            "ice",  # Use "ice" as the internal type for freezing-rain
            precipitationIntensity,
            PRECIPITATION_INTENSITY_THRESHOLDS,
            possiblePrecipitation,
            iconSet,
            isDayTime,
            summaryType,
            avgPrecipitation,
            precipitationAccumUnit,
            iceAccumulation,
        )
    elif (
        iceAccumulation > 0
        and precipitationIntensity > 0
        and precipitationType == "hail"
    ):
        currentText = possiblePrecipitation + "hail"
        # Original code did not set currentIcon for hail here, so leaving it as is.
        # If a default icon is desired for hail, it should be set here.
    elif (
        rainAccumulation > 0
        or snowAccumulation > 0
        or iceAccumulation > 0
        or precipitationIntensity > 0
    ) and precipitationType == "none":
        currentText, currentIcon = _get_precip_text_and_icon_for_type(
            "none",
            precipitationIntensity,
            PRECIPITATION_INTENSITY_THRESHOLDS,
            possiblePrecipitation,
            iconSet,
            isDayTime,
            summaryType,
            avgPrecipitation,
            precipitationAccumUnit,
            0,  # Accumulation is not directly used for 'none' type in this helper
        )

    if mode == "summary":
        return currentText
    elif mode == "icon":
        return currentIcon
    else:
        return currentText, currentIcon


def calculate_wind_text(windSpeed, windUnits, iconSet="darksky", mode="both"):
    """
    Calculates the wind summary and icon.

    Parameters:
    - windSpeed (float) - The wind speed.
    - windUnits (float) - The unit of the wind speed.
    - iconSet (str): Which icon set to use - Dark Sky or Pirate Weather.
    - mode (str): Determines what gets returned. "both" returns summary and icon, "icon" returns only the icon, "summary" returns only the summary.

    Returns:
    - tuple[str | None, str | None] or str | None: The textual representation and/or icon of the wind.
    """
    windText = None
    windIcon = None

    if windSpeed == MISSING_DATA:
        return (None, None) if mode == "both" else None

    lightWindThreshold = WIND_THRESHOLDS["light"] * windUnits
    midWindThreshold = WIND_THRESHOLDS["mid"] * windUnits
    heavyWindThreshold = WIND_THRESHOLDS["heavy"] * windUnits

    if lightWindThreshold <= windSpeed < midWindThreshold:
        windText = "light-wind"
        windIcon = "breezy" if iconSet == "pirate" else "wind"
    elif midWindThreshold <= windSpeed < heavyWindThreshold:
        windText = "medium-wind"
        windIcon = "wind"
    elif windSpeed >= heavyWindThreshold:
        windText = "heavy-wind"
        windIcon = "dangerous-wind" if iconSet == "pirate" else "wind"

    if mode == "summary":
        return windText
    elif mode == "icon":
        return windIcon
    else:
        return windText, windIcon


def calculate_visibility_text(
    visibility,
    visibilityUnits,
    tempUnits,
    temperature,
    dewPoint,
    smoke=0,
    iconSet="darksky",
    mode="both",
):
    """
    Calculates the visibility summary and icon.

    Parameters:
    - visibility (float) - The visibility.
    - visibilityUnits (float) - The unit of the visibility.
    - tempUnits (float) - The unit of the temperature.
    - temperature (float) - The ambient temperature.
    - dewPoint (float) - The dew point temperature.
    - smoke (float) - Surface smoke concentration in ug/m3.
    - iconSet (str) - Which icon set to use - Dark Sky or Pirate Weather.
    - mode (str) - Determines what gets returned. "both" returns summary and icon, "icon" returns only the icon, "summary" returns only the summary.

    Returns:
    - tuple[str | None, str | None] or str | None: The textual representation and/or icon of the visibility.
    """
    visibilityText = None
    visibilityIcon = None

    if any(x == MISSING_DATA for x in (temperature, dewPoint, visibility)):
        return (None, None) if mode == "both" else None

    fogThreshold = ICON_THRESHOLDS["fog_visibility"] * visibilityUnits
    mistThreshold = ICON_THRESHOLDS["mist_visibility"] * visibilityUnits

    # Convert Fahrenheit to Celsius for temperature spread comparisons
    if tempUnits == 0:
        temperature = (temperature - 32) * 5 / 9
        dewPoint = (dewPoint - 32) * 5 / 9

    tempDewSpread = temperature - dewPoint

    if (
        visibility < fogThreshold
        and tempDewSpread <= ICON_THRESHOLDS["temp_dewpoint_spread_fog"]
    ):
        visibilityText = "fog"
        visibilityIcon = "fog"
    elif (
        smoke >= ICON_THRESHOLDS["smoke_concentration"] and visibility <= mistThreshold
    ):
        visibilityText = "smoke"
        visibilityIcon = "smoke" if iconSet == "pirate" else "fog"
    elif (
        visibility < mistThreshold
        and tempDewSpread <= ICON_THRESHOLDS["temp_dewpoint_spread_mist"]
    ):
        visibilityText = "mist"
        visibilityIcon = "mist" if iconSet == "pirate" else "fog"
    elif (
        smoke < ICON_THRESHOLDS["smoke_concentration"]
        and visibility <= mistThreshold
        and tempDewSpread > ICON_THRESHOLDS["temp_dewpoint_spread_mist"]
    ):
        visibilityText = "haze"
        visibilityIcon = "haze" if iconSet == "pirate" else "fog"

    if mode == "summary":
        return visibilityText
    elif mode == "icon":
        return visibilityIcon
    else:
        return visibilityText, visibilityIcon


def calculate_sky_text(cloudCover, isDayTime, iconSet="darksky", mode="both"):
    """
    Calculates the sky cover summary and icon.

    Parameters:
    - cloudCover (int): The cloud cover for the period.
    - isDayTime (bool): Whether it's currently day or night.
    - iconSet (str): Which icon set to use - Dark Sky or Pirate Weather.
    - mode (str): Determines what gets returned. "both" returns summary and icon, "icon" returns only the icon, "summary" returns only the summary.

    Returns:
    - tuple[str | None, str | None] or str | None: The text and/or icon representing the current cloud cover.
    """
    skyText = None
    skyIcon = None

    if cloudCover == MISSING_DATA:
        return (None, None) if mode == "both" else None

    if cloudCover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        skyText = "heavy-clouds"
    elif cloudCover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        skyText = (
            "medium-clouds"
            if cloudCover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"]
            else "light-clouds"
        )
    else:
        skyText = (
            "very-light-clouds"
            if cloudCover > CLOUD_COVER_THRESHOLDS["mostly_clear"]
            else "clear"
        )

    skyIcon = calculate_sky_icon(cloudCover, isDayTime, iconSet)

    if mode == "summary":
        return skyText
    elif mode == "icon":
        return skyIcon
    else:
        return skyText, skyIcon


def humidity_sky_text(temperature, tempUnits, humidity):
    """
    Calculates the humidity text.

    Parameters:
    - temperature (float): The temperature for the period.
    - tempUnits (int): The temperature units.
    - humidity (float): The humidity for the period.

    Returns:
    - str | None: The text representing the humidity.
    """
    if (
        humidity is None
        or math.isnan(humidity)
        or humidity == MISSING_DATA
        or temperature == MISSING_DATA
    ):
        return None

    tempThreshold = 68 if tempUnits == 0 else 20
    lowHumidityThreshold = 0.15
    highHumidityThreshold = 0.95

    if humidity <= lowHumidityThreshold:
        return "low-humidity"
    elif humidity >= highHumidityThreshold and temperature > tempThreshold:
        return "high-humidity"

    return None


def calculate_thunderstorm_text(liftedIndex, cape, mode="both"):
    """
    Calculates the thunderstorm summary and icon.

    Parameters:
    - liftedIndex (float) - The lifted index.
    - cape (float) - The CAPE (Convective available potential energy).
    - mode (str): Determines what gets returned. "both" returns summary and icon, "icon" returns only the icon, "summary" returns only the summary.

    Returns:
    - tuple[str | None, str | None] or str | None: The textual representation and/or icon of the thunderstorm.
    """
    thunderstormText = None
    thunderstormIcon = None

    if 1000 <= cape < 2500:
        thunderstormText = "possible-thunderstorm"
    elif cape >= 2500:
        thunderstormText = "thunderstorm"
        thunderstormIcon = "thunderstorm"

    if liftedIndex != MISSING_DATA and thunderstormText is None:
        if 0 > liftedIndex > -4:
            thunderstormText = "possible-thunderstorm"
        elif liftedIndex <= -4:
            thunderstormText = "thunderstorm"
            thunderstormIcon = "thunderstorm"

    if mode == "summary":
        return thunderstormText
    elif mode == "icon":
        return thunderstormIcon
    else:
        return thunderstormText, thunderstormIcon


def kelvin_from_celsius(celsius):
    """Converts Celsius to Kelvin."""
    return celsius + 273.15


def estimate_snow_height(precipitationMm, temperatureC, windSpeedMps):
    """
    Estimates the snow height based on precipitation, temperature, and wind speed.

    Parameters:
    - precipitationMm (float): Liquid equivalent precipitation in mm.
    - temperatureC (float): Temperature in Celsius.
    - windSpeedMps (float): Wind speed in meters per second.

    Returns:
    - float: Estimated snow height.
    """
    snowDensityKgm3 = estimate_snow_density(temperatureC, windSpeedMps)
    return precipitationMm * 10 / snowDensityKgm3


def estimate_snow_density(temperatureC, windSpeedMps):
    """
    Estimates the snow density in kg/m3.

    Parameters:
    - temperatureC (float): Temperature in Celsius.
    - windSpeedMps (float): Wind speed in meters per second.

    Returns:
    - float: Estimated snow density in kg/m3.
    """
    kelvins = kelvin_from_celsius(temperatureC)

    kelvins = min(kelvins, 275.65)

    windSpeedExp = pow(windSpeedMps, 1.7)

    snowDensityKgm3 = 1000
    if kelvins <= 260.15:
        snowDensityKgm3 = 500 * (1 - 0.904 * math.exp(-0.008 * windSpeedExp))
    elif kelvins <= 275.65:
        snowDensityKgm3 = 500 * (
            1
            - 0.951
            * math.exp(-1.4 * pow(278.15 - kelvins, -1.15) - 0.008 * windSpeedExp)
        )
    else:
        # Fallback for temperatures above 2.5 degrees Celsius
        snowDensityKgm3 = 1000

    snowDensityKgm3 = max(snowDensityKgm3, 50)

    return snowDensityKgm3


def calculate_consecutive_indexes(indexes):
    """
    Groups a list of indexes into sub-lists of consecutive numbers.

    Parameters:
    - indexes (arr): A list of integers representing the indexes to be grouped.

    Returns:
    - arr: A list of lists, where each inner list contains consecutive indexes.
    """
    consecutiveIndexes = []
    for k, g in groupby(enumerate(indexes), lambda ix: ix[0] - ix[1]):
        consecutiveIndexes.append(list(map(itemgetter(1), g)))
    return consecutiveIndexes
