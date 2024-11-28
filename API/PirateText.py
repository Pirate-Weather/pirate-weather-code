# %% Script to contain the functions that can be used to generate the text summary of the forecast data for Pirate Weather

cloudThreshold = 0.875
mostlyCloudyThreshold = 0.625
partlyCloudyThreshold = 0.375
mostlyClearThreshold = 0.125


def calculate_sky_icon(cloudCover, isDayTime):
    sky_icon = None

    if cloudCover > cloudThreshold:
        sky_icon = "cloudy"
    elif cloudCover > partlyCloudyThreshold:
        if isDayTime:
            sky_icon = "partly-cloudy-day"
        else:
            sky_icon = "partly-cloudy-night"
    else:
        if isDayTime:
            sky_icon = "clear-day"
        else:
            sky_icon = "clear-night"

    return sky_icon


def calculate_text(
    hourObject,
    prepIntensityUnit,
    visUnits,
    windUnit,
    tempUnits,
    isDayTime,
    mode="title",
):
    visThresh = 1000 * visUnits

    lightRainThresh = 0.4 * prepIntensityUnit
    midRainThresh = 2.5 * prepIntensityUnit
    heavyRainThresh = 10 * prepIntensityUnit
    lightSnowThresh = 0.13 * prepIntensityUnit
    midSnowThresh = 0.83 * prepIntensityUnit
    heavySnowThresh = 3.33 * prepIntensityUnit
    lightSleetThresh = 0.83 * prepIntensityUnit
    midSleetThresh = 3.33 * prepIntensityUnit
    heavySleetThresh = 6.67 * prepIntensityUnit

    lightWindThresh = 6.7056 * windUnit
    midWindThresh = 10 * windUnit
    heavyWindThresh = 17.8816 * windUnit

    lowHumidityThresh = 0.1
    highHumidityThresh = 0.97

    # Get key values from the hourObject
    precipIntensity = hourObject["precipIntensity"]
    precipType = hourObject["precipType"]
    cloudCover = hourObject["cloudCover"]
    wind = hourObject["windSpeed"]
    pop = hourObject["precipProbability"]
    humidity = hourObject["humidity"]
    temp = hourObject["temperature"]

    if "visibility" in hourObject:
        vis = hourObject["visibility"]
    else:
        vis = 10000

    possiblePrecip = ""
    cIcon = None
    cText = None
    cCond = None
    # Add the possible precipitation text if pop is less than 30% or if pop is greater than 0 but precipIntensity is between 0-0.02 mm/h
    if (pop < 0.3) or (
        precipIntensity > 0 and precipIntensity < (0.02 * prepIntensityUnit)
    ):
        possiblePrecip = "possible-"

    # If precipIntensity is greater than 0.02 mm/h and no type fallback to rain icon
    if precipType == "none" and precipIntensity >= (0.02 * prepIntensityUnit):
        cIcon = "rain"
    elif precipIntensity >= (0.02 * prepIntensityUnit):
        cIcon = precipType

    if (precipIntensity > 0) & (precipType != None):
        if precipType == "rain":
            if precipIntensity < lightRainThresh:
                cText = [mode, possiblePrecip + "very-light-rain"]
                cCond = possiblePrecip + "very-light-rain"
            elif precipIntensity >= lightRainThresh and precipIntensity < midRainThresh:
                cText = [mode, possiblePrecip + "light-rain"]
                cCond = possiblePrecip + "light-rain"
            elif precipIntensity >= midRainThresh and precipIntensity < heavyRainThresh:
                cText = [mode, possiblePrecip + "medium-rain"]
                cCond = possiblePrecip + "medium-rain"
            else:
                cText = [mode, "heavy-rain"]
                cCond = "heavy-rain"
        elif precipType == "snow":
            if precipIntensity < lightSnowThresh:
                cText = [mode, possiblePrecip + "very-light-snow"]
                cCond = possiblePrecip + "very-light-snow"
            elif precipIntensity >= lightSnowThresh and precipIntensity < midSnowThresh:
                cText = [mode, possiblePrecip + "light-snow"]
                cCond = possiblePrecip + "light-snow"
            elif precipIntensity >= midSnowThresh and precipIntensity < heavySnowThresh:
                cText = [mode, possiblePrecip + "medium-snow"]
                cCond = possiblePrecip + "medium-snow"
            else:
                cText = [mode, possiblePrecip + "heavy-snow"]
                cCond = possiblePrecip + "heavy-snow"
        elif precipType == "sleet":
            if precipIntensity < lightSleetThresh:
                cText = [mode, possiblePrecip + "very-light-sleet"]
                cCond = possiblePrecip + "very-light-sleet"
            elif (
                precipIntensity >= lightSleetThresh and precipIntensity < midSleetThresh
            ):
                cText = [mode, possiblePrecip + "light-sleet"]
                cCond = possiblePrecip + "light-sleet"
            elif (
                precipIntensity >= midSleetThresh and precipIntensity < heavySleetThresh
            ):
                cText = [mode, possiblePrecip + "medium-sleet"]
                cCond = possiblePrecip + "medium-sleet"
            else:
                cText = [mode, possiblePrecip + "heavy-sleet"]
                cCond = possiblePrecip + "heavy-sleet"
    elif (precipIntensity > 0) & (precipType == None):
        # Because sometimes there's precipitation not no type use a generic precipitation summary
        # Not sure if this is possibile
        cIcon = "rain"
        if precipIntensity < lightRainThresh:
            cText = [mode, possiblePrecip + "very-light-precipitation"]
            cCond = possiblePrecip + "very-light-precipitation"
        elif precipIntensity >= lightRainThresh and precipIntensity < midRainThresh:
            cText = [mode, possiblePrecip + "light-precipitation"]
            cCond = possiblePrecip + "light-precipitation"
        elif precipIntensity >= midRainThresh and precipIntensity < heavyRainThresh:
            cText = [mode, possiblePrecip + "medium-precipitation"]
            cCond = possiblePrecip + "medium-precipitation"
        else:
            cText = [mode, possiblePrecip + "heavy-precipitation"]
            cCond = possiblePrecip + "heavy-precipitation"

    # If visibility < 1000m, show fog
    elif vis < visThresh:
        return [mode, "fog"], "fog"
    elif cloudCover > cloudThreshold:
        cText = [mode, "heavy-clouds"]
        cCond = "heavy-clouds"
        cIcon = calculate_sky_icon(cloudCover, isDayTime)
    elif cloudCover > partlyCloudyThreshold:
        cIcon = calculate_sky_icon(cloudCover, isDayTime)
        if cloudCover > mostlyCloudyThreshold:
            cText = [mode, "medium-clouds"]
            cCond = "medium-clouds"
        else:
            cText = [mode, "light-clouds"]
            cCond = "light-clouds"
    else:
        cIcon = calculate_sky_icon(cloudCover, isDayTime)
        if cloudCover > mostlyClearThreshold:
            cText = [mode, "very-light-clouds"]
            cCond = "very-light-clouds"
        else:
            cText = [mode, "clear"]

    # Add wind or humidity text
    if wind >= lightWindThresh:
        if cCond == None:
            if wind >= lightWindThresh:
                cIcon = "wind"
                cText = [mode, "light-wind"]
            elif wind >= midWindThresh:
                cIcon = "wind"
                cText = [mode, "medium-wind"]
            elif wind >= heavyWindThresh:
                cIcon = "wind"
                cText = [mode, "heavy-wind"]
        elif precipIntensity < 0.02:
            # Show the wind text before the sky text
            if wind >= lightWindThresh:
                cIcon = "wind"
                cText = [mode, ["and", "light-wind", cCond]]
            elif wind >= midWindThresh:
                cIcon = "wind"
                cText = [mode, ["and", "medium-wind", cCond]]
            elif wind >= heavyWindThresh:
                cIcon = "wind"
                cText = [mode, ["and", "heavy-wind", cCond]]
        else:
            # Show the precipitation text before the wind text
            if wind >= heavyWindThresh:
                cText = [mode, ["and", cCond, "heavy-wind"]]
            elif wind >= midWindThresh:
                cText = [mode, ["and", cCond, "medium-wind"]]
            elif wind >= lightWindThresh:
                cText = [mode, ["and", cCond, "light-wind"]]
    elif humidity < lowHumidityThresh:
        # Do not change the icon
        if cCond == None:
            cText = [mode, "low-humidity"]
        else:
            cText = [mode, ["and", "low-humidity", cCond]]
    elif humidity > highHumidityThresh:
        # Only use humid if also warm (>20C)
        if tempUnits == 0:
            tempThresh = 68
        else:
            tempThresh = 20
        if temp > tempThresh:
            # Do not change the icon
            if cCond == None:
                cText = [mode, "high-humidity"]
            else:
                cText = [mode, ["and", "high-humidity", cCond]]

    # If we have a condition text but no icon then use the sky cover or fog icon
    if cIcon is None and cText is not None:
        if vis < visThresh:
            cIcon = "fog"
        elif cloudCover > cloudThreshold:
            cIcon = calculate_sky_icon(cloudCover, isDayTime)
        elif cloudCover > partlyCloudyThreshold:
            cIcon = calculate_sky_icon(cloudCover, isDayTime)
        else:
            cIcon = calculate_sky_icon(cloudCover, isDayTime)
    return cText, cIcon
