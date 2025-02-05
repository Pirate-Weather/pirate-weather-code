# %% Script to contain the functions that can be used to generate the daily text summary of the forecast data for Pirate Weather
from collections import Counter
import math

cloudy = 0.875
mostly_cloudy = 0.625
partly_cloudy = 0.375
mostly_clear = 0.125
visibility = 1000


def Most_Common(lst):
    """
    Finds the most common icon to use as the weekly icon

    Parameters:
    - lst (arr): An array of weekly icons

    Returns:
    - str: The most common icon in the lst.
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]


def calculate_sky_icon(cloudCover, isDayTime):
    """
    Calculates the sky icon

    Parameters:
    - cloudCover (float) -  The cloudCover in percentage
    - isDayTime (bool) - Whether its daytime or nighttime

    Returns:
    - cIcon (float) - The icon representing the sky cover
    """
    sky_icon = None

    if cloudCover > cloudy:
        sky_icon = "cloudy"
    elif cloudCover > partly_cloudy:
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


def calculate_precip_icon(
    precipIntensity,
    prepIntensityUnit,
    precipType,
    pop=1,
):
    """
    Calculates the precipitation icon

    Parameters:
    - precipIntensity (float) -  The precipitation intensity
    - precipIntensityUnit (float) -  The unit of precipitation intensity
    - precipType (str) - The type of precipitation
    - pop (float) - The probablility of precipitation

    Returns:
    - cIcon (float) - The icon representing the precipitation
    """

    cIcon = None

    if (
        precipIntensity > (0.02 * prepIntensityUnit)
        and pop >= 0.3
        and precipType is not None
    ):
        if precipType != "none":
            cIcon = precipType
        else:
            cIcon = "rain"
    return cIcon


def calculate_precip_text(precipIntensity, prepIntensityUnit, precipType, pop):
    """
    Calculates the precipitation text

    Parameters:
    - precipIntensity (float) -  The precipitation intensity
    - precipIntensityUnit (float) -  The unit of precipitation intensity
    - precipType (str) - The type of precipitation
    - pop (float) - The probablility of precipitation

    Returns:
    - precipText (str) - The textual representation of the precipitation
    """
    precipText = None
    possiblePrecip = ""

    # Add the possible precipitation text if pop is between 10-30% or if pop is greater than 10% but precipIntensity is between 0-0.02 mm/h
    if pop < 0.3 or (
        precipIntensity > 0 and precipIntensity < (0.02 * prepIntensityUnit)
    ):
        possiblePrecip = "possible-"

    if (precipIntensity > 0) and precipType is not None:
        if precipType == "rain":
            if precipIntensity < (0.4 * prepIntensityUnit):
                precipText = possiblePrecip + "very-light-rain"
            elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
                2.5 * prepIntensityUnit
            ):
                precipText = possiblePrecip + "light-rain"
            elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
                10 * prepIntensityUnit
            ):
                precipText = "medium-rain"
            else:
                precipText = "heavy-rain"
        elif precipType == "snow":
            if precipIntensity < (0.13 * prepIntensityUnit):
                precipText = possiblePrecip + "very-light-snow"
            elif precipIntensity >= (0.13 * prepIntensityUnit) and precipIntensity < (
                0.83 * prepIntensityUnit
            ):
                precipText = possiblePrecip + "light-snow"
            elif precipIntensity >= (0.83 * prepIntensityUnit) and precipIntensity < (
                3.33 * prepIntensityUnit
            ):
                precipText = "medium-snow"
            else:
                precipText = "heavy-snow"
        elif precipType == "sleet":
            if precipIntensity < (0.4 * prepIntensityUnit):
                precipText = possiblePrecip + "very-light-sleet"
            elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
                2.5 * prepIntensityUnit
            ):
                precipText = possiblePrecip + "light-sleet"
            elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
                10 * prepIntensityUnit
            ):
                precipText = "medium-sleet"
            else:
                precipText = "heavy-sleet"
        else:
            # Because soemtimes there's precipitation not no type use a generic precipitation summary
            if precipIntensity < (0.4 * prepIntensityUnit):
                precipText = possiblePrecip + "very-light-precipitation"
            elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
                2.5 * prepIntensityUnit
            ):
                precipText = possiblePrecip + "light-precipitation"
            elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
                10 * prepIntensityUnit
            ):
                precipText = "medium-precipitation"
            else:
                precipText = "heavy-precipitation"

    return precipText


def calculate_wind_text(wind, windUnit):
    """
    Calculates the wind text

    Parameters:
    - wind (float) -  The wind speed
    - windUnit (float) -  The unit of the wind speed

    Returns:
    - windText (str) - The textual representation of the wind
    """
    windText = None
    if wind >= (6.7056 * windUnit) and wind < (10 * windUnit):
        windText = "light-wind"
    elif wind >= (10 * windUnit) and wind < (17.8816 * windUnit):
        windText = "medium-wind"
    elif wind >= (17.8816 * windUnit):
        windText = "heavy-wind"

    return windText


def calculate_vis_text(vis, visUnits):
    """
    Calculates the visibility text

    Parameters:
    - vis (float) -  The visibility
    - visUnit (float) -  The unit of the visibility

    Returns:
    - visText (str) - The textual representation of the visibility
    """
    visText = None

    if vis < (visibility * visUnits):
        visText = "fog"

    return visText


def calculate_cloud_text(cloudCover):
    """
    Calculates the visibility text

    Parameters:
    - cloudCover (float) -  The cloud cover for the period

    Returns:
    - cloudText (str) - The textual representation of the cloud cover
    - cloudLevel (int) - The level of the cloud cover
    """
    cloudText = None
    cloudLevel = None

    if cloudCover > cloudy:
        cloudText = "heavy-clouds"
        cloudLevel = 4
    elif cloudCover > partly_cloudy:
        if cloudCover > mostly_cloudy:
            cloudText = "medium-clouds"
            cloudLevel = 3
        else:
            cloudText = "light-clouds"
            cloudLevel = 2
    else:
        if cloudCover > mostly_clear:
            cloudText = "very-light-clouds"
            cloudLevel = 1
        else:
            cloudText = "clear"
            cloudLevel = 0

    return cloudText, cloudLevel


def calculate_period_text(
    periods, typePeriods, text, type, wind, morn, maxWind, windPrecip, checkPeriod, mode
):
    """
    Calculates the period text

    Parameters:
    - periods (arr) -  An array representing all the periods in the day/next 24h
    - typePeriods (arr) - An array representing the type (wind/cloud/precip/vis)
    - text (str) - A string representing the text for the type (light-rain, fog, etc.)
    - type (str) - The current type we are checking (precip, cloud, etc.)
    - wind (arr) - An array of the wind times
    - morn (arr) - The morning period. Used to determing the windSpeedUnit/precipIntensityUnit
    - maxWind (float) - The maximum wind for all the periods
    - windPrecip (bool) - Whether wind is occuring with cloud cover or precipitation.
    - checkPeriod (float) - The current period
    - mode (str) - Whether the summary is for the day or the next 24h

    Returns:
    - summary_text (str) - The textual representation of the type for the current day/next 24 hours
    - windPrecip (bool) - Returns if wind is occuring with precipitation or cloud cover
    """

    # Set the period text to the current text
    periodText = text
    summary_text = None
    # If there is only one period then just use that period
    if len(typePeriods) == 1:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in one period
        if len(wind) == 1 and (type == "precip" or type == "cloud"):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if wind[0] == typePeriods[0]:
                windPrecip = True
                periodText = ["and", periodText, calculate_wind_text(maxWind, morn[3])]
        summary_text = ["during", periodText, periods[typePeriods[0]]]
    # If the type has two periods
    elif len(typePeriods) == 2:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in two periods
        if len(wind) == 2 and (type == "precip" or type == "cloud"):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if wind[0] == typePeriods[0] and wind[1] == typePeriods[1]:
                windPrecip = True
                periodText = ["and", periodText, calculate_wind_text(maxWind, morn[3])]
        # If the type starts in the third period
        if typePeriods[0] == checkPeriod + 2 and typePeriods[1] == 3:
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        # If the type starts at the start but does not continue to the end
        elif typePeriods[0] == checkPeriod and (typePeriods[1] - typePeriods[0]) == 1:
            # We need to set the ending period to the next period. If the end occurs in the fourth period we set the period to 4 to prevent array out of bounds
            summary_text = [
                "until",
                periodText,
                periods[3 if typePeriods[1] + 1 > 3 else typePeriods[1] + 1],
            ]
        # If the type starts after the first period but doesn't continue to the end
        elif typePeriods[0] > checkPeriod and (typePeriods[1] - typePeriods[0]) == 1:
            summary_text = [
                "starting-continuing-until",
                periodText,
                periods[typePeriods[0]],
                periods[typePeriods[1] + 1],
            ]
        # If the type occurs at the start but then starts again in the last period
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[1] - typePeriods[0]) != 1
            and typePeriods[1] == 3
        ):
            summary_text = [
                "until-starting-again",
                periodText,
                periods[typePeriods[0] + 1],
                periods[typePeriods[1]],
            ]
        # If the two types are not joined and do not occur during the first or last period
        elif typePeriods[1] - typePeriods[0] != 1:
            summary_text = [
                "during",
                periodText,
                ["and", periods[typePeriods[0]], periods[typePeriods[1]]],
            ]
    # If the type occurs during three perionds
    elif len(typePeriods) == 3:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in three periods
        if len(wind) == 3 and (type != "precip" and type != "cloud"):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if (
                wind[0] == typePeriods[0]
                and wind[1] == typePeriods[1]
                and wind[2] == typePeriods[2]
            ):
                windPrecip = True
                periodText = ["and", periodText, calculate_wind_text(maxWind, morn[3])]
        # If the type starts in the second period
        if typePeriods[0] == checkPeriod + 1 and typePeriods[2] == 3:
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[1] - typePeriods[0]) == 1
            and (typePeriods[2] - typePeriods[1]) == 1
        ):
            # We need to set the ending period to the next period. If the end occurs in the fourth period we set the period to 4 to prevent array out of bounds
            summary_text = [
                "until",
                periodText,
                periods[3 if typePeriods[2] + 1 > 3 else typePeriods[2] + 1],
            ]
        # If the type starts after the first period but doesn't continue to the end and the first and second periods are connected
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[1] - typePeriods[0]) == 1
            and typePeriods[2] == 3
        ):
            summary_text = [
                "until-starting-again",
                periodText,
                periods[typePeriods[1] + 1],
                periods[typePeriods[2]],
            ]
        # If the type starts after the first period but doesn't continue to the end and the second and third periods are connected
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[1] - typePeriods[0]) != 1
            and typePeriods[1] == 2
        ):
            summary_text = [
                "until-starting-again",
                periodText,
                periods[typePeriods[0] + 1],
                periods[typePeriods[1]],
            ]
    # If the type occurs all day then use the for-day text
    elif len(typePeriods) == 4:
        # If they both occur at the same time and the type is precip/cloud then join with an and. Set windPrecip to true to skip checking the wind
        if len(wind) == 4 and (type == "precip" or type == "cloud"):
            windPrecip = True
            # Use for day text only on the daily block
            if mode == "daily":
                summary_text = [
                    "for-day",
                    ["and", periodText, calculate_wind_text(maxWind, morn[3])],
                ]
            else:
                summary_text = [
                    "starting",
                    ["and", periodText, calculate_wind_text(maxWind, morn[3])],
                    periods[typePeriods[0]],
                ]
        else:
            if mode == "daily":
                summary_text = ["for-day", periodText]
            else:
                summary_text = [
                    "starting",
                    periodText,
                    periods[typePeriods[0]],
                ]

    return summary_text, windPrecip


def calculate_day_text(morn, aft, eve, night, currPeriod, mode="daily"):
    """
    Calculates the current day/next 24h text

    Parameters:
    - morn (arr) - The first period.
    - aft (arr) - The second period.
    - eve (arr) - The third period.
    - night (arr) - The fourth period.
    - checkPeriod (float) - The current period

    Returns:
    - summary_text (str) - The textual representation of the current day/next 24 hours
    - cIcon (str) - The icon representing the current day/next 24 hours
    """

    # Variables used to calculate the summary
    summary_text = None
    period1 = []
    period2 = []
    period3 = []
    period4 = []
    precip = []
    vis = []
    wind = []
    snowAccum = 0
    snowLowAccum = 0
    snowMaxAccum = 0
    snowError = 0
    cloudLevels = []
    avgPop = 0
    # Check if the first periods later is true; and add the later- text if so
    if morn[14]:
        morn[13] = "later-" + morn[13]
    periods = [morn[13], aft[13], eve[13], night[13]]
    winds = []
    precipIntensity = []
    precipTypes = []
    maxIntensity = 0
    maxWind = 0
    numItems = 0
    starts = []
    checkPeriod = math.floor(currPeriod) - 1
    period1Level = -1
    period2Level = -1
    period3Level = -1
    period4Level = -1
    cIcon = None
    avgCloud = -1
    mostCommonPrecip = []
    snowSentence = None

    # If the current period is 3/4 the way through the first period then exclude it.
    if currPeriod < 1.75:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (morn[4] * morn[5]) > (0.02 * morn[5]):
            period1.append(morn[6])
            avgPop += morn[9]
        else:
            period1.append(None)
        # Calculate the wind text
        period1.append(calculate_wind_text(morn[2], morn[3]))
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if morn[4] * morn[5] < 0.02 and morn[2] / morn[3] < 6.7056:
            period1.append(calculate_vis_text(morn[0], morn[1]))
        else:
            period1.append(None)
        # Add the current period cloud cover
        period1.append(morn[7])
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period1Text, period1Level = calculate_cloud_text(period1[3])
        cloudLevels.append(period1Level)
    # If the current period is 3/4 the way through the second period then exclude it.
    if currPeriod < 2.75:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (aft[4] * aft[5]) > (0.02 * aft[5]):
            period2.append(aft[6])
            avgPop += aft[9]
        else:
            period2.append(None)
        # Calculate the wind text
        period2.append(calculate_wind_text(aft[2], aft[3]))
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if aft[4] * aft[5] < 0.02 and aft[2] / aft[3] < 6.7056:
            period2.append(calculate_vis_text(aft[0], aft[1]))
        else:
            period2.append(None)
        # Add the current period cloud cover
        period2.append(aft[7])
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period2Text, period2Level = calculate_cloud_text(period2[3])
        cloudLevels.append(period2Level)
    # If the current period is 3/4 the way through the third period then exclude it.
    if currPeriod < 3.75:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (eve[4] * eve[5]) > (0.02 * eve[5]):
            period3.append(eve[6])
            avgPop += eve[9]
        else:
            period3.append(None)
        period3.append(calculate_wind_text(eve[2], eve[3]))
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if eve[4] * eve[5] < 0.02 and eve[2] / eve[3] < 6.7056:
            period3.append(calculate_vis_text(eve[0], eve[1]))
        else:
            period3.append(None)
        # Add the current period cloud cover
        period3.append(eve[7])
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period3Text, period3Level = calculate_cloud_text(period3[3])
        cloudLevels.append(period3Level)

    # Check if there is enough precipitation to trigger the precipitation icon
    if (night[4] * night[5]) > (0.02 * night[5]):
        period4.append(night[6])
        avgPop += night[9]
    else:
        period4.append(None)
    # Check if there is no precipitation and the wind is less than the light wind threshold
    period4.append(calculate_wind_text(night[2], night[3]))
    if night[4] * night[5] < 0.02 and night[2] / night[3] < 6.7056:
        period4.append(calculate_vis_text(night[0], night[1]))
    else:
        period4.append(None)
    # Add the current period cloud cover
    period4.append(night[7])
    # Calculate the periods cloud text and level and add it to the cloud levels array
    period4Text, period4Level = calculate_cloud_text(period4[3])
    cloudLevels.append(period4Level)

    # If there is any periods with cloud cover then calculate the average precipitation probability
    if len(precip) > 0:
        avgPop = avgPop / len(precip)

    # If period1 exists
    if period1:
        # Add the wind speed to the wind array
        winds.append(morn[2])
        # If there is any precipitation
        if period1[0] is not None:
            # Calcaulte the intensity and add it to the precipitation array
            precipIntensity.append(morn[4] * morn[5])
            # Check if the type of precipitation is in an array of the precipitation types or if it doesn't exist add it
            if morn[6] not in precipTypes or not precipTypes:
                precipTypes.append(morn[6])
            # Add it to the list of all the precipitation types
            mostCommonPrecip.append(morn[6])
            precip.append(0)
            # If the precipitation is snow then add the accumulation and error
            if morn[11] > 0:
                snowAccum += morn[11]
                snowError += morn[12]
        # Add the wind to the wind array if the wind text exists
        if period1[1] is not None:
            wind.append(0)
        # Add the wind to the visibility array if the fog text exists
        if period1[2] is not None:
            vis.append(0)
    # If period2 exists
    if period2:
        # Add the wind speed to the wind array
        winds.append(aft[2])
        # If there is any precipitation
        if period2[0] is not None:
            precipIntensity.append(aft[4] * aft[5])
            # Check if the type of precipitation is in an array of the precipitation types or if it doesn't exist add it
            if aft[6] not in precipTypes or not precipTypes:
                precipTypes.append(aft[6])
            # Add it to the list of all the precipitation types
            mostCommonPrecip.append(aft[6])
            precip.append(1)
            # If the precipitation is snow then add the accumulation and error
            if aft[11] > 0:
                snowAccum += aft[11]
                snowError += aft[12]
        # Add the wind to the wind array if the wind text exists
        if period2[1] is not None:
            wind.append(1)
        # Add the wind to the visibility array if the fog text exists
        if period2[2] is not None:
            vis.append(1)
    # If period3 exists
    if period3:
        # Add the wind speed to the wind array
        winds.append(eve[2])
        # If there is any precipitation
        if period3[0] is not None:
            precipIntensity.append(eve[4] * eve[5])
            # Check if the type of precipitation is in an array of the precipitation types or if it doesn't exist add it
            if eve[6] not in precipTypes or not precipTypes:
                precipTypes.append(eve[6])
            # Add it to the list of all the precipitation types
            mostCommonPrecip.append(eve[6])
            precip.append(2)
            # If the precipitation is snow then add the accumulation and error
            if eve[11] > 0:
                snowAccum += eve[11]
                snowError += eve[12]
        # Add the wind to the wind array if the wind text exists
        if period3[1] is not None:
            wind.append(2)
        # Add the wind to the visibility array if the fog text exists
        if period3[2] is not None:
            vis.append(2)

    # Add the wind speed to the wind array
    winds.append(night[2])
    # If there is any precipitation
    if period4[0] is not None:
        precipIntensity.append(night[4] * night[5])
        # Check if the type of precipitation is in an array of the precipitation types or if it doesn't exist add it
        if night[6] not in precipTypes or not precipTypes:
            precipTypes.append(night[6])
        # Add it to the list of all the precipitation types
        mostCommonPrecip.append(night[6])
        precip.append(3)
        if night[11] > 0:
            snowAccum += night[11]
            snowError += night[12]
    # Add the wind to the wind array if the wind text exists
    if period4[1] is not None:
        wind.append(3)
    # Add the wind to the visibility array if the fog text exists
    if period4[2] is not None:
        vis.append(3)

    # Add the wind, wind and visibility starts to the starts array if they exist
    if precip:
        starts.append(precip[0])
    if wind:
        starts.append(wind[0])
    if vis:
        starts.append(vis[0])

    # If the precipIntensity array has any values then get the maxiumum
    if precipIntensity:
        maxIntensity = max(precipIntensity)
    # Calculate the snow precipitation range
    snowLowAccum = math.floor(snowAccum - (snowError / 2))
    snowMaxAccum = math.ceil(snowAccum + (snowError / 2))

    # If the snow accumulation is below 0; set it to 0
    if snowLowAccum < 0:
        snowLowAccum = 0

    # Check to see if there is any snow accumulation and if so calculate the sentence to use when creating the precipitation summaries
    if snowMaxAccum > 0:
        # If there is no accumulation then show the accumulation as < 1 cm/in
        if snowAccum == 0:
            snowSentence = [
                "less-than",
                ["centimeters" if morn[5] == 1 else "inches", 1],
            ]
        # If the lower accumulation range is 0 then show accumulation as < max range cm/in
        elif snowLowAccum == 0:
            snowSentence = [
                "less-than",
                ["centimeters" if morn[5] == 1 else "inches", snowMaxAccum],
            ]
        # Otherwise show the range
        else:
            snowSentence = [
                "centimeters" if morn[5] == 1 else "inches",
                [
                    "range",
                    snowLowAccum,
                    snowMaxAccum,
                ],
            ]

    # If there is more than one precipitation type
    if len(precipTypes) > 2:
        # Set the icon to sleet
        cIcon = "sleet"
        # If there is any snow precipitation
        if "snow" in precipTypes or snowAccum > 0:
            precipType = [
                "parenthetical",
                "mixed-precipitation",
                snowSentence,
            ]
        else:
            # Otherwise just use mixed precipitation
            precipType = "mixed-precipitation"
    # If there are two types of precipitation
    elif len(precipTypes) == 2:
        cIcon = calculate_precip_icon(
            maxIntensity, morn[5], Most_Common(mostCommonPrecip), avgPop
        )
        # If there is any rain precipitation
        if "rain" in precipTypes:
            # If there is any snow precipitation; set the icon to snow
            if "snow" in precipTypes:
                cIcon = "snow"
                precipType = [
                    "parenthetical",
                    calculate_precip_text(maxIntensity, night[5], "rain", avgPop),
                    snowSentence,
                ]
            # If there is any snow accumulation but not enough to show the snow icon in a block
            elif snowAccum > 0:
                precipType = [
                    "parenthetical",
                    [
                        "and",
                        calculate_precip_text(
                            maxIntensity, night[5], precipTypes[0], avgPop
                        ),
                        "medium-" + precipTypes[1],
                    ],
                    snowSentence,
                ]
            # Otherwise join the first and second type with an and
            else:
                precipType = [
                    "and",
                    calculate_precip_text(
                        maxIntensity, night[5], precipTypes[0], avgPop
                    ),
                    "medium-" + precipTypes[1],
                ]
        # If there is any sleet precipitation
        elif "sleet" in precipTypes:
            # If there is any snow precipitation; set the icon to snow
            if "snow" in precipTypes:
                cIcon = "snow"
                precipType = [
                    "parenthetical",
                    calculate_precip_text(maxIntensity, night[5], "sleet", avgPop),
                    snowSentence,
                ]
            # If there is any snow accumulation but not enough to show the snow icon in a block
            elif snowAccum > 0:
                precipType = [
                    "parenthetical",
                    [
                        "and",
                        calculate_precip_text(
                            maxIntensity, night[5], precipTypes[0], avgPop
                        ),
                        "medium-" + precipTypes[1],
                    ],
                    snowSentence,
                ]
            # Otherwise join the first and second type with an and
            else:
                precipType = [
                    "and",
                    calculate_precip_text(
                        maxIntensity, night[5], precipTypes[0], avgPop
                    ),
                    "medium-" + precipTypes[1],
                ]
        # If we have precipitation with no type
        else:
            # If there is any snow precipitation; set the icon to snow
            if "snow" in precipTypes:
                cIcon = "snow"
                # If there is no accumulation then show the accumulation as < 1 cm/in
                precipType = [
                    "parenthetical",
                    calculate_precip_text(maxIntensity, night[5], "none", avgPop),
                    snowSentence,
                ]
            # If there is any snow accumulation but not enough to show the snow icon in a block
            elif snowAccum > 0:
                precipType = [
                    "parenthetical",
                    [
                        "and",
                        calculate_precip_text(
                            maxIntensity, night[5], precipTypes[0], avgPop
                        ),
                        "medium-" + precipTypes[1],
                    ],
                    snowSentence,
                ]
            # Otherwise join the first and second type with an and
            else:
                precipType = [
                    "and",
                    calculate_precip_text(
                        maxIntensity, night[5], precipTypes[0], avgPop
                    ),
                    "medium-" + precipTypes[1],
                ]
    # If there is just one type of precipitation
    elif len(precipTypes) == 1:
        # Set the icon to the type
        cIcon = calculate_precip_icon(
            maxIntensity, morn[5], Most_Common(mostCommonPrecip), avgPop
        )
        if "snow" in precipTypes:
            precipType = [
                "parenthetical",
                calculate_precip_text(maxIntensity, night[5], "snow", avgPop),
                snowSentence,
            ]
        # If there is any snow accumulation but not enough to show the snow icon in a block
        elif snowAccum > 0:
            precipType = [
                "parenthetical",
                calculate_precip_text(
                    maxIntensity, night[5], Most_Common(mostCommonPrecip), avgPop
                ),
                snowSentence,
            ]

        # Otherwise just calculate the text normally
        else:
            precipType = calculate_precip_text(
                maxIntensity, night[5], precipTypes[0], avgPop
            )

    # Check the cloud levels and determine the most common one
    mostCommonCloud = Most_Common(cloudLevels)
    mostCommonLevels = []

    # Check the individual period levels to see if they match the most common. If so then add them to the list of periods
    if period1Level == mostCommonCloud:
        mostCommonLevels.append(0)
    if period2Level == mostCommonCloud:
        mostCommonLevels.append(1)
    if period3Level == mostCommonCloud:
        mostCommonLevels.append(2)
    if period4Level == mostCommonCloud:
        mostCommonLevels.append(3)

    # If all the periods are different
    if len(mostCommonLevels) == 1:
        avgCloud = 0
        # Determine the max/min level
        maxCloudLevel = max(cloudLevels)
        minCloudLevel = min(cloudLevels)

        if period1:
            avgCloud += morn[7]
        if period2:
            avgCloud += aft[7]
        if period3:
            avgCloud += eve[7]
        if period4:
            avgCloud += night[7]

        # Determine the average cloud for the icon
        avgCloud = avgCloud / len(cloudLevels)

        # If the first period has the maxiumum cloud level and the end has the lowest then use the lowest
        if period1Level == maxCloudLevel and period4Level == minCloudLevel:
            cloudLevels[0] = cloudLevels[3]
            mostCommonLevels.pop()
            mostCommonLevels.append(3)
        # If the second period has the maxiumum cloud level and the end has the lowest then use the lowest
        elif period2Level == maxCloudLevel and period4Level == minCloudLevel:
            cloudLevels[0] = cloudLevels[3]
            mostCommonLevels.pop()
            mostCommonLevels.append(3)
        # If the third period has the maxiumum cloud level and the end has the lowest then use the lowest
        elif period3Level == maxCloudLevel and period4Level == minCloudLevel:
            cloudLevels[0] = cloudLevels[3]
            mostCommonLevels.pop()
            mostCommonLevels.append(3)
        # If the second period has the maxiumum cloud level then use that period
        elif period2Level == maxCloudLevel:
            cloudLevels[0] = cloudLevels[1]
            mostCommonLevels.pop()
            mostCommonLevels.append(1)
        # If the third period has the maxiumum cloud level then use that period
        elif period3Level == maxCloudLevel:
            cloudLevels[0] = cloudLevels[2]
            mostCommonLevels.pop()
            mostCommonLevels.append(2)
        # If the fourth period has the maxiumum cloud level then use that period
        elif period4Level == maxCloudLevel:
            cloudLevels[0] = cloudLevels[3]
            mostCommonLevels.pop()
            mostCommonLevels.append(3)

    maxWind = max(winds)
    cloudConvertedText = None
    windPrecip = False
    cloudLevel = -1

    if(cloudLevels[mostCommonLevels[0]] != max(cloudLevels)):
        cloudLevel = cloudLevels[mostCommonLevels[0]]
    else:
        cloudLevel = max(cloudLevels)

    # Convert the level back into the textual representation to use in the summaries.
    if cloudLevel == 0:
        cloudConvertedText = "clear"
        # If the average cloud cover doesn't exist set it to 0 so the icon matches the text
        if avgCloud == -1:
            avgCloud = 0
    elif cloudLevel == 1:
        cloudConvertedText = "very-light-clouds"
        # If the average cloud cover doesn't exist set it to 0.25 so the icon matches the text
        if avgCloud == -1:
            avgCloud = 0.25
    elif cloudLevel == 2:
        cloudConvertedText = "light-clouds"
        # If the average cloud cover doesn't exist set it to 0.5 so the icon matches the text
        if avgCloud == -1:
            avgCloud = 0.50
    elif cloudLevel == 3:
        # If the average cloud cover doesn't exist set it to 0.75 so the icon matches the text
        cloudConvertedText = "medium-clouds"
        if avgCloud == -1:
            avgCloud = 0.75
    else:
        # If the average cloud cover doesn't exist set it to 1 so the icon matches the text
        if avgCloud == -1:
            avgCloud = 1
        cloudConvertedText = "heavy-clouds"

    # Calculate the cloud period text
    cloudText, windPrecip = calculate_period_text(
        periods,
        mostCommonLevels,
        cloudConvertedText,
        "cloud",
        wind,
        morn,
        maxWind,
        windPrecip,
        checkPeriod,
        mode,
    )

    # If there is only one period
    if not period1 and not period2 and not period3:
        # If there is precipitation and wind then join with an and
        if period4[0] is not None:
            if period4[1] is not None:
                summary_text = [
                    "sentence",
                    ["during", ["and", precipType, period4[1]], "night"],
                ]
            # Otherwise just use the precipitation
            else:
                return ["sentence", ["during", precipType, "night"]]
        # If there is fog then show that text
        elif period4[2] is not None:
            cIcon = "fog"
            summary_text = ["sentence", ["during", period4[2], "night"]]
        else:
            # If there is wind during the last period then join the wind with the cloud text
            if period4[1] is not None:
                cIcon = "wind"
                summary_text = [
                    "sentence",
                    ["during", ["and", period4[1], period4[3]], "night"],
                ]
            # Otherwise just show the cloud text
            else:
                summary_text = ["sentence", ["during", period4[3], "night"]]
    else:
        windText = None
        visText = None
        precipText = None
        windText = None
        visText = None

        # If there is any precipitation then calcaulate the text
        if len(precip) > 0:
            numItems += 1
            precipText, windPrecip = calculate_period_text(
                periods,
                precip,
                precipType,
                "precip",
                wind,
                morn,
                maxWind,
                windPrecip,
                checkPeriod,
                mode,
            )

        # If there is any visibility then calcaulate the text
        if len(vis) > 0 and numItems <= 1:
            numItems += 1
            visText, windPrecip = calculate_period_text(
                periods,
                vis,
                "fog",
                "vis",
                wind,
                morn,
                maxWind,
                windPrecip,
                checkPeriod,
                mode,
            )

        # If there is any wind then calcaulate the text if its not joined with precip/cloud
        if not windPrecip and numItems <= 1 and len(wind) > 0:
            numItems += 1
            windText, windPrecip = calculate_period_text(
                periods,
                wind,
                calculate_wind_text(maxWind, morn[3]),
                "vis",
                wind,
                morn,
                maxWind,
                windPrecip,
                checkPeriod,
                mode,
            )

        # If the summary text is not already set
        if summary_text is None:
            # If there is no precipitation
            if precipText is None:
                # If there is no wind
                if windText is None:
                    # If there is visbility
                    if visText is not None:
                        cIcon = "fog"
                        summary_text = ["sentence", visText]
                    # Otherwise use the cloud text
                    else:
                        summary_text = ["sentence", cloudText]
                # If there is wind text and visbility text then join with an and and show whichever one comes first at the start
                elif visText is not None:
                    if vis[0] == min(starts):
                        cIcon = "fog"
                        summary_text = ["sentence", ["and", visText, windText]]
                    else:
                        cIcon = "wind"
                        summary_text = ["sentence", ["and", windText, visText]]
                # If there is wind text
                else:
                    cIcon = "wind"
                    summary_text = ["sentence", windText]
            # If there is precipitation
            else:
                # If there is any visibility text then join with an and and show whichever one comes fist at the start
                if visText is not None:
                    if vis[0] == min(starts):
                        cIcon = "fog"
                        summary_text = ["sentence", ["and", visText, precipText]]
                    else:
                        summary_text = ["sentence", ["and", precipText, visText]]
                # If there is any wind text then join with an and and show whichever one comes fist at the start
                elif windText is not None:
                    if wind[0] == min(starts):
                        cIcon = "wind"
                        summary_text = ["sentence", ["and", windText, precipText]]
                    else:
                        summary_text = ["sentence", ["and", precipText, windText]]
                else:
                    summary_text = ["sentence", precipText]

    # If there is no icon then calculate it based on the average cloud cover for the periods
    if cIcon is None:
        cIcon = calculate_sky_icon(avgCloud, True)

    return summary_text, cIcon
