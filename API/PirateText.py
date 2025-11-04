# %% Script to contain the functions that can be used to generate the text summary of the forecast data for Pirate Weather
import numpy as np

from API.constants.shared_const import MISSING_DATA
from API.constants.text_const import (
    DEFAULT_POP,
    DEFAULT_VISIBILITY,
)
from API.PirateTextHelper import (
    calculate_precip_text,
    calculate_sky_text,
    calculate_thunderstorm_text,
    calculate_vis_text,
    calculate_wind_text,
    humidity_sky_text,
)


def calculate_text(
    hourObject,
    isDayTime,
    type,
    icon="darksky",
):
    """
    Calculates the textual summary and icon for an hourly forecast period.
    All inputs are expected in SI units:
    - Temperature in Celsius
    - Wind speed in m/s
    - Visibility in meters
    - Precipitation intensity in mm/h
    - Precipitation accumulation in mm

    Parameters:
    - hourObject (dict): A dictionary containing hourly forecast data (in SI units).
        Expected keys include: precipType, cloudCover, windSpeed, temperature/temperatureHigh,
        humidity, visibility, cape, smoke, dewPoint, precipProbability,
        precipAccumulation, snowAccumulation, sleetAccumulation,
        liquidIntensity, snowIntensity, iceIntensity
    - isDayTime (bool): Whether it's currently daytime
    - type (str): The type of summary being generated ("current", "hour", "hourly")
    - icon (str): Which icon set to use - "darksky" or "pirate" (default: "darksky")

    Returns:
    - tuple: A tuple containing:
        - cText (str or list): A textual summary representing the conditions for the period
        - cIcon (str): The icon representing the conditions for the period
    """

    cText = cIcon = precipText = precipIcon = windText = windIcon = skyText = (
        skyIcon
    ) = visText = visIcon = None

    # Get key values from the hourObject
    precipType = hourObject["precipType"]
    cloudCover = hourObject["cloudCover"]
    wind = hourObject["windSpeed"]

    # If time machine, no humidity data, so set to 0
    if "humidity" not in hourObject:
        humidity = MISSING_DATA
    else:
        humidity = hourObject["humidity"]

    # If type is current precipitation probability should always be 1 otherwise if it exists in the hourObject use it otherwise use 1
    if type == "current":
        pop = DEFAULT_POP
    else:
        pop = hourObject.get("precipProbability", DEFAULT_POP)

    # If temperature exists in the hourObject then use it otherwise use the high temperature
    if "temperature" in hourObject:
        temp = hourObject["temperature"]
    else:
        temp = hourObject["temperatureHigh"]

    # If visibility exists in the hourObject then use it otherwise use the default
    vis = hourObject.get("visibility", DEFAULT_VISIBILITY)

    # If cape exists in the hourObject then use it otherwise -999
    cape = hourObject.get("cape", MISSING_DATA)

    # If smoke exists in the hour object then use it otherwise -999
    smoke = hourObject.get("smoke", MISSING_DATA)

    # If dewPoint exists in the hour object then use it otherwise -999
    dewPoint = hourObject.get("dewPoint", MISSING_DATA)

    # If we missing or incomplete data then return clear icon/text instead of calculating
    if all(np.isnan(v) for v in (temp, wind, vis, cloudCover, humidity, dewPoint)):
        return "unavailable", "none"

    # Extract liquidAccumulation, snowAccum, sleetAccum from hourObject
    rainAccum = hourObject.get("liquidAccumulation", 0.0)
    snowAccum = hourObject.get("snowAccumulation", 0.0)
    sleetAccum = hourObject.get("sleetAccumulation", 0.0)

    # Calculate the text/icon for precipitation, wind, visibility, sky cover, humidity and thunderstorms
    precipText, precipIcon = calculate_precip_text(
        precipType,
        type,
        rainAccum,
        snowAccum,
        sleetAccum,
        pop,
        icon,
        "both",
        # provide per-type intensities when available (mm/h)
        eff_rain_intensity=hourObject.get("rainIntensity"),
        eff_snow_intensity=hourObject.get("snowIntensity"),
        eff_ice_intensity=hourObject.get("iceIntensity"),
    )

    windText, windIcon = calculate_wind_text(wind, icon, "both")
    visText, visIcon = calculate_vis_text(vis, temp, dewPoint, smoke, icon, "both")
    thuText, thuIcon = calculate_thunderstorm_text(cape, "both", icon, isDayTime)
    skyText, skyIcon = calculate_sky_text(cloudCover, isDayTime, icon, "both")
    humidityText = humidity_sky_text(temp, humidity)

    # If there is precipitation text use that and join with thunderstorm or humidity or wind texts if they exist
    if precipText is not None:
        if thuText is not None and not (type == "current" and "possible" in thuText):
            cText = thuText
        elif windText is not None:
            cText = ["and", precipText, windText]
        else:
            if humidityText is not None:
                cText = ["and", precipText, humidityText]
            else:
                cText = precipText
    # If there is visibility text then use that and join with humidity if it exists
    elif visText is not None:
        if humidityText is not None:
            cText = ["and", visText, humidityText]
        else:
            cText = visText
    # If there is wind text use that. If the skies are clear then join with humidity text if it exists otherwise just use the wind text
    elif windText is not None:
        if skyText == "clear":
            if humidityText is not None:
                cText = ["and", windText, humidityText]
            else:
                cText = windText
        else:
            cText = ["and", windText, skyText]
    # If there is the humidity text then join with the sky text
    elif humidityText is not None:
        cText = ["and", humidityText, skyText]
    else:
        cText = skyText

    # If precipitation icon use that unless there are thunderstorms occurring
    if precipIcon is not None:
        if thuIcon is not None:
            cIcon = thuIcon
        else:
            cIcon = precipIcon
    # If visibility icon use that
    elif visIcon is not None:
        cIcon = visIcon
    # If wind icon use that
    elif windIcon is not None:
        cIcon = windIcon
    # Otherwise use the sky icon
    else:
        cIcon = skyIcon

    # If we somehow have no text
    if cText is None:
        cText = "clear"

    # If we somehow have no icon
    if cIcon is None:
        if isDayTime:
            cIcon = "clear-day"
        else:
            cIcon = "clear-night"

    return cText, cIcon
