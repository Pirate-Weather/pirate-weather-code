# %% Script to contain the functions that can be used to generate the text summary of the forecast data for Pirate Weather

cloudThreshold = 0.875
mostlyCloudyThreshold = 0.625
partlyCloudyThreshold = 0.375
mostlyClearThreshold = 0.125


def calculate_sky_icon(cloudCover, isDayTime, icon="darksky"):
    sky_icon = None

    if cloudCover > cloudy:
        sky_icon = "cloudy"
    elif cloudCover > mostly_cloudy and icon == "pirate":
        if isDayTime:
            sky_icon = "mostly-cloudy-day"
        else:
            sky_icon = "mostly-cloudy-night"
    elif cloudCover > partly_cloudy:
        if isDayTime:
            sky_icon = "partly-cloudy-day"
        else:
            sky_icon = "partly-cloudy-night"
    elif cloudCover > mostly_clear and icon == "pirate":
        if isDayTime:
            sky_icon = "mostly-clear-day"
        else:
            sky_icon = "mostly-clear-night"
    else:
        if isDayTime:
            sky_icon = "clear-day"
        else:
            sky_icon = "clear-night"

    return sky_icon


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
    mode="title",
    prepInensity,
    icon="darksky"
):
    visThresh = 1000 * visUnits

    # In mm/h
    lightRainThresh = 0.4 * prepAccumUnit
    midRainThresh = 2.5 * prepAccumUnit
    heavyRainThresh = 10 * prepAccumUnit
    lightSnowThresh = 1.33 * prepAccumUnit
    midSnowThresh = 8.33 * prepAccumUnit
    heavySnowThresh = 33.33 * prepAccumUnit
    lightSleetThresh = 0.4 * prepAccumUnit
    midSleetThresh = 2.5 * prepAccumUnit
    heavySleetThresh = 10.0 * prepAccumUnit

    lightWindThresh = 6.7056 * windUnit
    midWindThresh = 10 * windUnit
    heavyWindThresh = 17.8816 * windUnit

    lowHumidityThresh = 0.15
    highHumidityThresh = 0.95

    snowIconThresholdHour = 0.20 * prepAccumUnit
    rainIconThresholdHour = 0.02 * prepAccumUnit
    iceIconThresholdHour = 0.02 * prepAccumUnit

    snowIconThresholdDay = 10.0 * prepAccumUnit
    rainIconThresholdDay = 1.0 * prepAccumUnit
    iceIconThresholdDay = 1.0 * prepAccumUnit

    # Use daily or hourly thresholds depending on the situation
    if (type == "hour") or (type == "current"):
        snowIconThreshold = snowIconThresholdHour
        rainIconThreshold = rainIconThresholdHour
        iceIconThreshold = iceIconThresholdHour
    elif type == "day":
        snowIconThreshold = snowIconThresholdDay
        rainIconThreshold = rainIconThresholdDay
        iceIconThreshold = iceIconThresholdDay
        lightRainThresh = lightRainThresh * 24
        midRainThresh = midRainThresh * 24
        heavyRainThresh = heavyRainThresh * 24
        lightSnowThresh = lightSnowThresh * 24
        midSnowThresh = midSnowThresh * 24
        heavySnowThresh = heavySnowThresh * 24
        lightSleetThresh = lightSleetThresh * 24
        midSleetThresh = midSleetThresh * 24
        heavySleetThresh = heavySleetThresh * 24

        # Get key values from the hourObject
    precipType = hourObject["precipType"]
    cloudCover = hourObject["cloudCover"]
    wind = hourObject["windSpeed"]
    humidity = hourObject["humidity"]

    if type == "current":
        pop = 1
    elif "precipProbability" in hourObject:
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

    possiblePrecip = ""
    cIcon = None
    cText = None
    cCond = None
    # Add the possible precipitation text if pop is less than 30% or if pop is greater than 0 but precipIntensity is between 0-0.02 mm/h
    if (pop < 0.25) or (
        ((rainPrep > 0) and (rainPrep < rainIconThreshold))
        or ((snowPrep > 0) and (snowPrep < snowIconThreshold))
        or ((icePrep > 0) and (icePrep < iceIconThreshold))
        or ((prepInensity > 0) and (prepInensity < rainIconThreshold))
    ):
        possiblePrecip = "possible-"

    # Find the largest percentage difference compared to the thresholds
    # rainPrepPercent = rainPrep / rainIconThreshold
    # snowPrepPercent = snowPrep / snowIconThreshold
    # icePrepPercent = icePrep / iceIconThreshold

    # Find the largest percentage difference to determine the icon
    if pop >= 0.25 and (
        (rainPrep > rainIconThreshold)
        or (snowPrep > snowIconThreshold)
        or (icePrep > iceIconThreshold)
    ):
        if precipType == "none":
            cIcon = "rain"  # Fallback icon
        else:
            cIcon = precipType


    if mode == "daily" and prepInensity > rainIconThreshold:
        if rainPrep > 0 and snowPrep > 0:
            cText = [mode, ["and", "medium-rain", "medium-snow"]]
        elif rainPrep > 0 and icePrep > 0:
            cText = [mode, ["and", "medium-rain", "medium-sleet"]]
        elif snowPrep > 0 and icePrep > 0:
            cText = [mode, ["and", "medium-snow", "medium-sleet"]]
        elif snowPrep > 0 and icePrep > 0 and rainPrep > 0:
            cText = [mode, ["mixed-precipitation"]]
            cIcon = "mixed"
    elif rainPrep > 0 and prepInensity > 0 and precipType == "rain":
        if prepInensity < lightRainThresh:
            cText = [mode, possiblePrecip + "very-light-rain"]
            cCond = possiblePrecip + "very-light-rain"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-rain-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-rain-night"
                elif icon == "pirate":
                    cIcon = "drizzle"
        elif prepInensity >= lightRainThresh and prepInensity < midRainThresh:
            cText = [mode, possiblePrecip + "light-rain"]
            cCond = possiblePrecip + "light-rain"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-rain-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-rain-night"
                elif icon == "pirate":
                    cIcon = "light-rain"
        elif prepInensity >= midRainThresh and prepInensity < heavyRainThresh:
            cText = [mode, "medium-rain"]
            cCond = possiblePrecip + "medium-rain"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-rain-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-rain-night"
        else:
            cText = [mode, "heavy-rain"]
            cCond = possiblePrecip + "heavy-rain"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-rain-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-rain-night"
                elif icon == "pirate":
                    cIcon = "heavy-rain"
    elif snowPrep > 0 and prepInensity > 0 and precipType == "snow":
        if prepInensity < lightSnowThresh:
            cText = [mode, possiblePrecip + "very-light-snow"]
            cCond = possiblePrecip + "very-light-snow"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-snow-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-snow-night"
                elif icon == "pirate":
                    cIcon = "flurries"
        elif prepInensity >= lightSnowThresh and prepInensity < midSnowThresh:
            cText = [mode, possiblePrecip + "light-snow"]
            cCond = possiblePrecip + "light-snow"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-snow-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-snow-night"
                elif icon == "pirate":
                    cIcon = "light-snow"
        elif prepInensity >= midSnowThresh and prepInensity < heavySnowThresh:
            cText = [mode, "medium-snow"]
            cCond = possiblePrecip + "medium-snow"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-snow-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-snow-night"
        else:
            cText = [mode, "heavy-snow"]
            cCond = possiblePrecip + "heavy-snow"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-snow-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-snow-night"
                elif icon == "pirate":
                    cIcon = "heavy-snow"
    elif icePrep > 0 and prepInensity > 0 and precipType == "sleet":
        if prepInensity < lightSleetThresh:
            cText = [mode, possiblePrecip + "very-light-sleet"]
            cCond = possiblePrecip + "very-light-sleet"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-sleet-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-sleet-night"
                elif icon == "pirate":
                    cIcon = "very-light-sleet"
        elif prepInensity >= lightSleetThresh and prepInensity < midSleetThresh:
            cText = [mode, possiblePrecip + "light-sleet"]
            cCond = possiblePrecip + "light-sleet"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-sleet-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-sleet-night"
                elif icon == "pirate":
                    cIcon = "light-sleet"
        elif prepInensity >= midSleetThresh and prepInensity < heavySleetThresh:
            cText = [mode, "medium-sleet"]
            cCond = possiblePrecip + "medium-sleet"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-sleet-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-sleet-night"
        else:
            cText = [mode, "heavy-sleet"]
            cCond = possiblePrecip + "heavy-sleet"
                if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
                    cIcon = "possible-sleet-day"
                elif (
                    icon == "pirate" and possiblePrecip == "possible-" and not isDayTime
                ):
                    cIcon = "possible-sleet-night"
                elif icon == "pirate":
                    cIcon = "heavy-sleet"
    elif (rainPrep > 0 or snowPrep > 0 or icePrep > 0 or prepInensity > 0) and precipType == "none":
        if prepInensity < lightRainThresh:
            cText = [mode, possiblePrecip + "very-light-precipitation"]
            cCond = possiblePrecip + "very-light-precipitation"
        elif prepInensity >= lightRainThresh and prepInensity < midRainThresh:
            cText = [mode, possiblePrecip + "light-precipitation"]
            cCond = possiblePrecip + "light-precipitation"
        elif prepInensity >= midRainThresh and prepInensity < heavyRainThresh:
            cText = [mode, "medium-precipitation"]
            cCond = possiblePrecip + "medium-precipitation"
        else:
            cText = [mode, "heavy-precipitation"]
            cCond = possiblePrecip + "heavy-precipitation"

        if icon == "pirate" and possiblePrecip == "possible-" and isDayTime:
            cIcon = "possible-precipitation-day"
        elif icon == "pirate" and possiblePrecip == "possible-" and not isDayTime:
            cIcon = "possible-precipitation-night"
        elif icon == "pirate":
            cIcon = "precipitation"

    # If visibility < 1000m, show fog
    elif vis < visThresh and wind < lightWindThresh:
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
        if cIcon not in ["rain", "snow", "sleet", "fog"]:
            cIcon = "wind"

        if cCond == None:
            if wind >= lightWindThresh and wind < midWindThresh:
                cText = [mode, "light-wind"]
            elif wind >= midWindThresh and wind < heavyWindThresh:
                cText = [mode, "medium-wind"]
            elif wind >= heavyWindThresh:
                cText = [mode, "heavy-wind"]
        else:
            # If precipitation intensity is below 0.02 mm/h then set the icon to be the wind icon otherwise use the already set icon
            if (
                (rainPrep < rainIconThreshold)
                and (snowPrep < snowIconThreshold)
                and (icePrep < iceIconThreshold)
                and (prepInensity < rainIconThreshold)
            ):
                cIcon = "wind"

            if (rainPrep + snowPrep + icePrep) == 0:
                # Show the wind text before the sky text
                if wind >= lightWindThresh and wind < midWindThresh:
                    cText = [mode, ["and", "light-wind", cCond]]
                    if icon == "pirate":
                        cIcon = "breezy"
                elif wind >= midWindThresh and wind < heavyWindThresh:
                    cText = [mode, ["and", "medium-wind", cCond]]
                elif wind >= heavyWindThresh:
                    cText = [mode, ["and", "heavy-wind", cCond]]
                    if icon == "pirate":
                        cIcon = "dangerously-windy"
            else:
                # Show the wind text after the precipitation text
                if wind >= lightWindThresh and wind < midWindThresh:
                    cText = [mode, ["and", cCond, "light-wind"]]
                elif wind >= midWindThresh and wind < heavyWindThresh:
                    cText = [mode, ["and", cCond, "medium-wind"]]
                elif wind >= heavyWindThresh:
                    cText = [mode, ["and", cCond, "heavy-wind"]]

    elif humidity <= lowHumidityThresh:
        # Do not change the icon
        if cCond is None:
            cText = [mode, "low-humidity"]
        else:
            cText = [mode, ["and", cCond, "low-humidity"]]
    elif humidity >= highHumidityThresh:
        # Only use humid if also warm (>20C)
        if tempUnits == 0:
            tempThresh = 68
        else:
            tempThresh = 20
        if temp > tempThresh:
            # Do not change the icon
            if cCond is None:
                cText = [mode, "high-humidity"]
            else:
                cText = [mode, ["and", cCond, "high-humidity"]]

    # If we have a condition text but no icon then use the sky cover or fog icon
    if cIcon is None and cText is not None:
        if vis < visThresh:
            cIcon = "fog"
        else:
            cIcon = calculate_sky_icon(cloudCover, isDayTime)

    return cText, cIcon
