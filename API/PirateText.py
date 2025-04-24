# %% Script to contain the functions that can be used to generate the text summary of the forecast data for Pirate Weather

from PirateTextHelper import (
    calculate_precip_text,
    calculate_wind_text,
    calculate_vis_text,
    calculate_sky_text,
    humidity_sky_text,
    calculate_thunderstorm_text,
)


def calculate_text(
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
    precipIntensity,
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
    - type (str): What type of summary is being generated.
    - precipIntensity (float): The precipitation intensity
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - cText (str): A summary representing the conditions for the period.
    - cIcon (str): The icon representing the conditions for the period.
    """

    cText = cIcon = precipText = precipIcon = windText = windIcon = skyText = (
        skyIcon
    ) = visText = visIcon = None

    # Get key values from the hourObject
    precipType = hourObject["precipType"]
    cloudCover = hourObject["cloudCover"]
    wind = hourObject["windSpeed"]
    humidity = hourObject["humidity"]

    # If time machine, no humidity data, so set to 0
    if "humidity" not in hourObject:
        humidity = 0
    else:
        humidity = hourObject["humidity"]

    # If type is current precipitation probability should always be 1 otherwise if it exists in the hourObject use it otherwise use 1
    if type == "current":
        pop = 1
    elif "precipProbability" in hourObject or  hourObject["precipProbability"] != -999:
        pop = hourObject["precipProbability"]
    else:
        pop = 1

    # If temperature exists in the hourObject then use it otherwise use the high temperature
    if "temperature" in hourObject:
        temp = hourObject["temperature"]
    else:
        temp = hourObject["temperatureHigh"]

    # If visibility exists in the hourObject then use it otherwise 10000m (10km)
    if "visibility" in hourObject:
        vis = hourObject["visibility"]
    else:
        vis = 10000

    # If liftedIndex exists in the hourObject then use it otherwise -999
    if "liftedIndex" in hourObject:
        liftedIndex = hourObject["liftedIndex"]
    else:
        liftedIndex = -999

    # If cape exists in the hourObject then use it otherwise -999
    if "cape" in hourObject:
        cape = hourObject["cape"]
    else:
        cape = -999

    # If we missing or incomplete data then return clear icon/text instead of calculating
    if (
        temp == -999
        or wind == -999
        or vis == -999
        or cloudCover == -999
        or humidity == -999
    ):
        if isDayTime:
            return "clear", "clear-day"
        else:
            return "clear", "clear-night"

    # Calculate the text/icon for precipitation, wind, visibility, sky cover, humidity and thunderstorms
    precipText, precipIcon = calculate_precip_text(
        precipIntensity,
        prepAccumUnit,
        precipType,
        type,
        rainPrep,
        snowPrep,
        icePrep,
        pop,
        icon,
        "both",
    )
    windText, windIcon = calculate_wind_text(wind, windUnit, icon, "both")
    visText, visIcon = calculate_vis_text(vis, visUnits, "both")
    thuText, thuIcon = calculate_thunderstorm_text(liftedIndex, cape, "both")
    skyText, skyIcon = calculate_sky_text(cloudCover, isDayTime, icon, "both")
    humidityText = humidity_sky_text(temp, tempUnits, humidity)

    # If there is precipitation text use that and join with thunderstorm or humidity or wind texts if they exist
    if precipText is not None:
        if thuText is not None:
            cText = ["and", thuText, precipText]
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
