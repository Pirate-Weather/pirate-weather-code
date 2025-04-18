# %% Script to contain the functions that can be used to generate the daily text summary of the forecast data for Pirate Weather
from PirateTextHelper import *
import datetime
from dateutil import tz

cloudy = 0.875
mostly_cloudy = 0.625
partly_cloudy = 0.375
mostly_clear = 0.125
visibility = 1000


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
    periods,
    typePeriods,
    text,
    type,
    wind,
    prepAccumUnit,
    visUnits,
    windUnit,
    maxWind,
    windPrecip,
    checkPeriod,
    mode,
    icon,
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
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
        summary_text = ["during", periodText, periods[typePeriods[0]]]
    # If the type has two periods
    elif len(typePeriods) == 2:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in two periods
        if len(wind) == 2 and (type == "precip" or type == "cloud"):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if wind[0] == typePeriods[0] and wind[1] == typePeriods[1]:
                windPrecip = True
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
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
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
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
                    [
                        "and",
                        periodText,
                        calculate_wind_text(maxWind, windUnit, icon, "summary"),
                    ],
                ]
            else:
                summary_text = [
                    "starting",
                    [
                        "and",
                        periodText,
                        calculate_wind_text(maxWind, windUnit, icon, "summary"),
                    ],
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


def calculate_day_text(
    hours,
    prepAccumUnit,
    visUnits,
    windUnit,
    tempUnits,
    isDayTime,
    timeZone,
    mode="daily",
    icon="darksky",
):
    """
    Calculates the current day/next 24h text

    Parameters:
    - hours (arr) - The array of hours for the day or next 24 hours.
    - currPeriod (float) - The current period
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - summary_text (str) - The textual representation of the current day/next 24 hours
    - cIcon (str) - The icon representing the current day/next 24 hours
    """

    period1 = []
    period2 = []
    period3 = []
    period4 = []
    prepTypes = []
    mostCommonPrecip = []
    periodStats = [[], [], [], []]
    currPeriod = periodCutOff = periodEnd = None
    periodIndex = numHoursFog = numHoursWind = numHoursDry = numHoursHumid = (
        rainPrep
    ) = snowPrep = sleetPrep = snowError = cloudCover = pop = maxIntensity = maxWind = (
        length
    ) = 0
    periodIncrease = False
    today = ""

    zone = tz.gettz(timeZone)
    currDate = datetime.datetime.fromtimestamp(hours[0]["time"], zone)
    currHour = int(currDate.strftime("%H"))
    currWeekday = currDate.strftime("%A").lower()

    # Time periods are as follows:
    # morning 4:00 to 12:00
    # afternoon 12:00 to 17:00
    # evening 17:00 to 22:00
    # night: 22:00 to 4:00

    if 4 <= currHour < 12:
        currPeriod = "morning"
        periodCutOff = 10
        periodEnd = 11
    elif 12 <= currHour < 17:
        currPeriod = "afternoon"
        periodCutOff = 14
        periodEnd = 16
    elif 17 <= currHour < 22:
        currPeriod = "evening"
        periodCutOff = 20
        periodEnd = 21
    else:
        currPeriod = "night"
        periodCutOff = 26
        periodEnd = 27

    hourPeriod = currPeriod

    if mode == "hour":
        today = "today-"

    for idx, hour in enumerate(hours):
        if idx == 0:
            continue

        hourDate = datetime.datetime.fromtimestamp(hour["time"], zone)
        hourHour = int(hourDate.strftime("%H"))
        hourWeekday = hourDate.strftime("%A").lower()

        if 0 <= hourHour < 4:
            hourHour = hourHour + 24

        if (
            hourHour >= periodCutOff
            and hourHour <= periodEnd
            and currWeekday == hourWeekday
        ):
            continue
        elif (
            hourHour < periodCutOff
            and hourHour <= periodEnd
            and currWeekday == hourWeekday
        ):
            if periodIndex == 0:
                periodIndex = 1

        if not period1 and periodIndex == 0:
            periodIndex = periodIndex + 1
            hourPeriod = nextPeriod(hourPeriod)
        if hourHour == 12 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True
        if hourHour == 17 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True
        if hourHour == 22 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True
        if hourHour == 4 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True

        if hour["precipType"] == "rain" or hour["precipType"] == "none":
            rainPrep = rainPrep + hour["precipAccumulation"]
        elif hour["precipType"] == "snow":
            snowPrep = snowPrep + hour["precipAccumulation"]
            snowError = snowError + hour["precipIntensityError"]
        elif hour["precipType"] == "sleet":
            sleetPrep = sleetPrep + hour["precipAccumulation"]

        if (
            humidity_sky_text(hour["temperature"], tempUnits, hour["humidity"])
            == "high-humidity"
        ):
            numHoursHumid += 1
        if (
            humidity_sky_text(hour["temperature"], tempUnits, hour["humidity"])
            == "low-humidity"
        ):
            numHoursDry += 1
        if calculate_vis_text(hour["visibility"], visUnits, "icon") == "fog":
            numHoursFog += 1
        if (
            calculate_wind_text(hour["windSpeed"], windUnit, "darksky", "icon")
            == "wind"
        ):
            numHoursWind += 1

        cloudCover += hour["cloudCover"]
        pop += hour["precipProbability"]

        if maxIntensity == 0:
            maxIntensity = hour["precipIntensity"]
        elif maxIntensity > 0 and hour["precipIntensity"] > maxIntensity:
            maxIntensity = hour["precipIntensity"]

        if maxWind == 0:
            maxWind = hour["windSpeed"]
        elif maxWind > 0 and hour["windSpeed"] > maxWind:
            maxWind = hour["windSpeed"]

        if hour["precipIntensity"] > 0.02 * prepAccumUnit:
            mostCommonPrecip.append(hour["precipType"])

        if not prepTypes and hour["precipIntensity"] > 0.02 * prepAccumUnit:
            prepTypes.append(hour["precipType"])
        elif (
            hour["precipType"] not in prepTypes
            and hour["precipIntensity"] > 0.02 * prepAccumUnit
        ):
            prepTypes.append(hour["precipType"])

        if periodIndex == 1:
            period1.append(hour)
        elif periodIndex == 2:
            period2.append(hour)
        elif periodIndex == 3:
            period3.append(hour)
        elif periodIndex == 4:
            period4.append(hour)

        if (periodIncrease and periodIndex <= 5) or (idx == 25 and periodIndex <= 5):
            if periodIndex - 1 == 1:
                cloudCover = cloudCover / len(period1)
                pop = pop / len(period1)
                length = len(period1)
            elif periodIndex - 1 == 2:
                cloudCover = cloudCover / len(period2)
                pop = pop / len(period2)
                length = len(period2)
            elif periodIndex - 1 == 3:
                cloudCover = cloudCover / len(period3)
                pop = pop / len(period3)
                length = len(period3)
            elif periodIndex - 1 == 4:
                cloudCover = cloudCover / len(period4)
                pop = pop / len(period4)
                length = len(period4)

            if idx == 25 and periodIndex < 5:
                periodIndex += 1
                length = len(period4)

            periodStats[periodIndex - 2].append(numHoursFog)
            periodStats[periodIndex - 2].append(numHoursDry)
            periodStats[periodIndex - 2].append(numHoursWind)
            periodStats[periodIndex - 2].append(round(rainPrep, 4))
            periodStats[periodIndex - 2].append(round(snowPrep, 4))
            periodStats[periodIndex - 2].append(round(snowError, 4))
            periodStats[periodIndex - 2].append(round(sleetPrep, 4))
            periodStats[periodIndex - 2].append(round(pop, 2))
            periodStats[periodIndex - 2].append(round(maxIntensity, 4))
            periodStats[periodIndex - 2].append(round(cloudCover, 2))
            periodStats[periodIndex - 2].append(maxWind)
            periodStats[periodIndex - 2].append(length)
            periodStats[periodIndex - 2].append(today + hourPeriod)
            periodStats[periodIndex - 2].append(numHoursHumid)
            numHoursFog = numHoursWind = numHoursDry = numHoursHumid = rainPrep = (
                snowPrep
            ) = sleetPrep = snowError = cloudCover = pop = maxIntensity = maxWind = 0
            periodIncrease = False

            hourPeriod = nextPeriod(hourPeriod)

            if hourHour == 4 and period1:
                today = "tomorrow-"

    # Variables used to calculate the summary
    precip = []
    vis = []
    winds = []
    wind = []
    cloudLevels = []
    precipIntensity = []
    periods = [
        periodStats[0][12],
        periodStats[1][12],
        periodStats[2][12],
        periodStats[3][12],
    ]
    summary_text = cIcon = snowSentence = prepText = None
    period1Calc = []
    period2Calc = []
    period3Calc = []
    period4Calc = []
    snowAccum = snowLowAccum = snowMaxAccum = snowError = avgPop = maxIntensity = (
        maxWind
    ) = numItems = 0
    starts = []
    precipIntensity = []
    period1Level = period2Level = period3Level = period4Level = avgCloud = -1
    secondary = snowText = snowSentence = None
    icePrep = (
        periodStats[0][6] + periodStats[1][6] + periodStats[2][6] + periodStats[3][6]
    )
    rainPrep = (
        periodStats[0][3] / 10
        + periodStats[1][3] / 10
        + periodStats[2][3] / 10
        + periodStats[3][3] / 10
    )
    snowPrep = (
        periodStats[0][4] + periodStats[1][4] + periodStats[2][4] + periodStats[3][4]
    )
    totalPrep = rainPrep + snowPrep + sleetPrep
    if mode == "day":
        if currPeriod == "morning":
            currPeriodNum = 1 + ((8 - (12 - currHour)) / 8)
        elif currPeriod == "afternoon":
            currPeriodNum = 1 + ((5 - (17 - currHour)) / 5)
        elif currPeriod == "evening":
            currPeriodNum = 1 + ((5 - (22 - currHour)) / 5)
        elif currPeriod == "night":
            currPeriodNum = 1 + ((6 - (28 - currHour)) / 6)
    else:
        currPeriodNum = 1

    checkPeriod = math.floor(currPeriodNum) - 1

    # If the current period is 3/4 the way through the first period then exclude it.
    if currPeriodNum < 1.75:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (periodStats[0][8] * prepAccumUnit) > (0.02 * prepAccumUnit):
            period1Calc.append(True)
            avgPop += periodStats[0][7]
        else:
            period1Calc.append(None)
        # Calculate the wind text
        if periodStats[0][2] >= (periodStats[0][11] / 2):
            period1Calc.append(
                calculate_wind_text(periodStats[0][10], windUnit, icon, "summary")
            )
        else:
            period1Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[0][8] * prepAccumUnit < 0.02
            and periodStats[0][10] / windUnit < 6.7056
            and periodStats[0][0] >= (periodStats[0][11] / 2)
        ):
            period1Calc.append(calculate_vis_text(0, visUnits, "summary"))
        else:
            period1Calc.append(None)
        # Add the current period cloud cover
        period1Calc.append(periodStats[0][9])
        avgCloud += periodStats[0][9]
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period1Text, period1Level = calculate_cloud_text(periodStats[0][9])
        cloudLevels.append(period1Level)
    # If the current period is 3/4 the way through the second period then exclude it.
    if currPeriodNum < 2.75:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (periodStats[1][8] * prepAccumUnit) > (0.02 * prepAccumUnit):
            period2Calc.append(True)
            avgPop += periodStats[1][7]
        else:
            period2Calc.append(None)
        # Calculate the wind text
        if periodStats[1][2] >= (periodStats[1][11] / 2):
            period2Calc.append(
                calculate_wind_text(periodStats[1][10], windUnit, icon, "summary")
            )
        else:
            period2Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[1][8] * prepAccumUnit < 0.02
            and periodStats[1][10] / windUnit < 6.7056
            and periodStats[1][0] >= (periodStats[1][11] / 2)
        ):
            period2Calc.append(calculate_vis_text(0, visUnits, "summary"))
        else:
            period2Calc.append(None)
        # Add the current period cloud cover
        period2.append(periodStats[1][9])
        avgCloud += periodStats[1][9]
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period2Text, period2Level = calculate_cloud_text(periodStats[1][9])
        cloudLevels.append(period2Level)
    # If the current period is 3/4 the way through the third period then exclude it.
    if currPeriodNum < 3.75:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (periodStats[2][8] * prepAccumUnit) > (0.02 * prepAccumUnit):
            period3Calc.append(True)
            avgPop += periodStats[2][7]
        else:
            period3Calc.append(None)
        # Calculate the wind text
        if periodStats[2][2] >= (periodStats[2][11] / 2):
            period3Calc.append(
                calculate_wind_text(periodStats[2][10], windUnit, icon, "summary")
            )
        else:
            period3Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[2][8] * prepAccumUnit < 0.02
            and periodStats[2][10] / windUnit < 6.7056
            and periodStats[2][0] >= (periodStats[2][11] / 2)
        ):
            period3Calc.append(calculate_vis_text(0, visUnits, "summary"))
        else:
            period3Calc.append(None)
        # Add the current period cloud cover
        period3Calc.append(periodStats[2][9])
        avgCloud += periodStats[2][9]
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period3Text, period3Level = calculate_cloud_text(periodStats[2][9])
        cloudLevels.append(period3Level)

    # Check if there is enough precipitation to trigger the precipitation icon
    if (periodStats[3][8] * prepAccumUnit) > (0.02 * prepAccumUnit):
        period4Calc.append(True)
        avgPop += periodStats[3][7]
    else:
        period4Calc.append(None)
    # Calculate the wind text
    if periodStats[3][2] >= (periodStats[3][11] / 2):
        period4Calc.append(
            calculate_wind_text(periodStats[3][10], windUnit, icon, "summary")
        )
    else:
        period4Calc.append(None)
    # Check if there is no precipitation and the wind is less than the light wind threshold
    if (
        periodStats[3][8] * prepAccumUnit < 0.02
        and periodStats[3][10] / windUnit < 6.7056
        and periodStats[3][0] >= (periodStats[3][11] / 2)
    ):
        period4Calc.append(calculate_vis_text(0, visUnits, "summary"))
    else:
        period4Calc.append(None)
    # Add the current period cloud cover
    period4Calc.append(periodStats[3][9])
    avgCloud += periodStats[3][9]
    # Calculate the periods cloud text and level and add it to the cloud levels array
    period4Text, period4Level = calculate_cloud_text(periodStats[3][9])
    cloudLevels.append(period4Level)

    # If period1 exists
    if period1Calc:
        # Add the wind speed to the wind array
        winds.append(periodStats[0][10])
        # If there is any precipitation
        if period1Calc[0] is not None:
            # Calcaulte the intensity and add it to the precipitation array
            precipIntensity.append(periodStats[0][8] * prepAccumUnit)
            precip.append(0)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[0][4] > 0:
                snowAccum += periodStats[0][4]
                snowError += periodStats[0][5]
        # Add the wind to the wind array if the wind text exists
        if period1Calc[1] is not None:
            wind.append(0)
        # Add the wind to the visibility array if the fog text exists
        if period1Calc[2] is not None:
            vis.append(0)
    # If period2 exists
    if period2Calc:
        # Add the wind speed to the wind array
        winds.append(periodStats[1][10])
        # If there is any precipitation
        if period2Calc[0] is not None:
            precipIntensity.append(periodStats[1][8] * prepAccumUnit)
            precip.append(1)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[1][4] > 0:
                snowAccum += periodStats[1][4]
                snowError += periodStats[1][5]
        # Add the wind to the wind array if the wind text exists
        if period2Calc[1] is not None:
            wind.append(1)
        # Add the wind to the visibility array if the fog text exists
        if period2Calc[2] is not None:
            vis.append(1)
    # If period3 exists
    if period3Calc:
        # Add the wind speed to the wind array
        winds.append(periodStats[2][10])
        # If there is any precipitation
        if period3Calc[0] is not None:
            precipIntensity.append(periodStats[2][8] * prepAccumUnit)
            precip.append(2)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[2][4] > 0:
                snowAccum += periodStats[2][4]
                snowError += periodStats[2][5]
        # Add the wind to the wind array if the wind text exists
        if period3Calc[1] is not None:
            wind.append(2)
        # Add the wind to the visibility array if the fog text exists
        if period3Calc[2] is not None:
            vis.append(2)

    # Add the wind speed to the wind array
    winds.append(periodStats[2][10])
    # If there is any precipitation
    if period4Calc[0] is not None:
        precipIntensity.append(periodStats[3][8] * prepAccumUnit)
        precip.append(3)
        if periodStats[3][4] > 0:
            snowAccum += periodStats[3][4]
            snowError += periodStats[3][5]
    # Add the wind to the wind array if the wind text exists
    if period4Calc[1] is not None:
        wind.append(3)
    # Add the wind to the visibility array if the fog text exists
    if period4Calc[2] is not None:
        vis.append(3)

    # Add the wind, wind and visibility starts to the starts array if they exist
    if precip:
        starts.append(precip[0])
    if wind:
        starts.append(wind[0])
    if vis:
        starts.append(vis[0])

    # If there is any periods with cloud cover then calculate the average precipitation probability
    if len(precip) > 0:
        avgPop = avgPop / len(precip)

    # If the precipIntensity array has any values then get the maxiumum
    if precipIntensity:
        maxIntensity = max(precipIntensity)
        precipType = Most_Common(mostCommonPrecip)

    # Only calculate the precipitation text if there is any possibility of precipitation > 0
    if avgPop > 0 and totalPrep >= (0.01 * prepAccumUnit):
        # Check if there is rain, snow and ice accumulation for the day
        if snowPrep > 0 and rainPrep > 0 and icePrep > 0:
            # If there is then used the mixed precipitation text and set the icon/type to sleet. Set the secondary condition to snow so the totals can be in the summary
            prepText = "mixed-precipitation"
            precipType = "sleet"
            precipIcon = calculate_precip_text(
                maxIntensity,
                prepAccumUnit,
                precipType,
                "hour",
                rainPrep,
                snowPrep,
                icePrep,
                avgPop,
                icon,
                "icon",
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
                if rainPrep > 0 and rainPrep > icePrep:
                    precipType = "rain"
                    secondary = "medium-sleet"
                # If we do check if we have ice. If there is more ice than rain then set rain as the secondary condition
                elif rainPrep > 0 and rainPrep < icePrep:
                    precipType = "rain"
                    secondary = "medium-sleet"

            # If the type is snow but there is no snow accumulation check if there is rain/sleet
            if snowPrep == 0 and precipType == "snow":
                if rainPrep > 0:
                    precipType = "rain"
                elif icePrep > 0:
                    precipType = "sleet"
            # If the type is rain but there is no rain accumulation check if there is snow/sleet
            elif rainPrep == 0 and precipType == "rain":
                if snowPrep > 0:
                    precipType = "snow"
                elif icePrep > 0:
                    precipType = "sleet"
            # If the type is sleet but there is no sleet accumulation check if there is rain/snow
            elif icePrep == 0 and precipType == "sleet":
                if snowPrep > 0:
                    precipType = "snow"
                elif rainPrep > 0:
                    precipType = "rain"

            # If more than 10 mm of rain is forecast, then rain
            if rainPrep > (10 * prepAccumUnit) and precipType != "rain":
                secondary = "medium-" + precipType
                precipType = "rain"
            # If more than 5 mm of snow is forecast, then snow
            if snowPrep > (5 * prepAccumUnit) and precipType != "snow":
                secondary = "medium-" + precipType
                precipType = "snow"
            # Else, if more than 1 mm of ice is forecast, then ice
            if icePrep > (1 * prepAccumUnit) and precipType != "sleet":
                secondary = "medium-" + precipType
                precipType = "sleet"

            # Calculate the precipitation text and summary
            prepText, precipIcon = calculate_precip_text(
                maxIntensity,
                prepAccumUnit,
                precipType,
                "hour",
                rainPrep,
                snowPrep,
                icePrep,
                avgPop,
                icon,
                "both",
            )

    if secondary == "medium-none":
        secondary = "medium-precipitation"

    # If we have only snow or if snow is the secondary condition then calculate the accumulation range
    if snowPrep > (10 * prepAccumUnit) or secondary == "medium-snow":
        # GEFS accumulation error seems to always be equal to the accumulation so use half of the accumulation as the range
        snowLowAccum = math.floor(snowPrep - (snowError / 2))
        snowMaxAccum = math.ceil(snowPrep + (snowError / 2))

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

    # If we have more than 1 cm of snow show the parenthetical or snow is the secondary condition
    if snowSentence is not None:
        # If precipitation is only show then generate the parenthetical text
        if precipType == "snow":
            prepText = [
                "parenthetical",
                prepText,
                snowSentence,
            ]
        # Otherwise if its a secondary condition then generate the text using the main condition
        elif secondary == "medium-snow":
            snowText = [
                "parenthetical",
                prepText,
                snowSentence,
            ]

    # If we have a secondary condition join them with an and if not snow otherwise use the snow text
    if secondary is not None:
        if secondary != "medium-snow":
            prepText = ["and", prepText, secondary]
        else:
            prepText = snowText

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

    # Determine the average cloud for the icon
    avgCloud = avgCloud / len(cloudLevels)

    # If all the periods are different
    if len(mostCommonLevels) == 1:
        avgCloud = 0
        # Determine the max/min level
        maxCloudLevel = max(cloudLevels)
        minCloudLevel = min(cloudLevels)

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

    if cloudLevels[mostCommonLevels[0]] != max(cloudLevels):
        cloudLevel = cloudLevels[mostCommonLevels[0]]
    else:
        cloudLevel = max(cloudLevels)

    # Convert the level back into the textual representation to use in the summaries.
    if cloudLevel == 0:
        cloudConvertedText = "clear"
        # If the average cloud cover doesn't exist set it to 0 so the icon matches the text
        if len(cloudLevels) > 1:
            avgCloud = 0
    elif cloudLevel == 1:
        cloudConvertedText = "very-light-clouds"
        # If the average cloud cover doesn't exist set it to 0.25 so the icon matches the text
        if len(cloudLevels) > 1:
            avgCloud = 0.25
    elif cloudLevel == 2:
        cloudConvertedText = "light-clouds"
        # If the average cloud cover doesn't exist set it to 0.5 so the icon matches the text
        if len(cloudLevels) > 1:
            avgCloud = 0.50
    elif cloudLevel == 3:
        # If the average cloud cover doesn't exist set it to 0.75 so the icon matches the text
        cloudConvertedText = "medium-clouds"
        if len(cloudLevels) > 1:
            avgCloud = 0.75
    else:
        # If the average cloud cover doesn't exist set it to 1 so the icon matches the text
        if len(cloudLevels) > 1:
            avgCloud = 1
        cloudConvertedText = "heavy-clouds"

    # Calculate the cloud period text
    cloudText, windPrecip = calculate_period_text(
        periods,
        mostCommonLevels,
        cloudConvertedText,
        "cloud",
        wind,
        prepAccumUnit,
        visUnits,
        windUnit,
        maxWind,
        windPrecip,
        checkPeriod,
        mode,
        icon,
    )

    # If there is only one period
    if not period1Calc and not period2Calc and not period3Calc:
        # If there is precipitation and wind then join with an and
        if period4Calc[0] is not None:
            if period4Calc[1] is not None:
                summary_text = [
                    "sentence",
                    ["during", ["and", precipText, period4Calc[1]], "night"],
                ]
            # Otherwise just use the precipitation
            else:
                return ["sentence", ["during", precipText, "night"]]
        # If there is fog then show that text
        elif period4Calc[2] is not None:
            cIcon = "fog"
            summary_text = ["sentence", ["during", period4Calc[2], "night"]]
        else:
            # If there is wind during the last period then join the wind with the cloud text
            if period4Calc[1] is not None:
                cIcon = "wind"
                summary_text = [
                    "sentence",
                    ["during", ["and", period4Calc[1], period4Calc[3]], "night"],
                ]
            # Otherwise just show the cloud text
            else:
                summary_text = ["sentence", ["during", period4Calc[3], "night"]]
    else:
        windText = None
        visText = None
        precipText = None

        # If there is any precipitation then calcaulate the text
        if len(precip) > 0:
            numItems += 1
            precipText, windPrecip = calculate_period_text(
                periods,
                precip,
                prepText,
                "precip",
                wind,
                prepAccumUnit,
                visUnits,
                windUnit,
                maxWind,
                windPrecip,
                checkPeriod,
                mode,
                icon,
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
                prepAccumUnit,
                visUnits,
                windUnit,
                maxWind,
                windPrecip,
                checkPeriod,
                mode,
                icon,
            )

        # If there is any wind then calcaulate the text if its not joined with precip/cloud
        if not windPrecip and numItems <= 1 and len(wind) > 0:
            numItems += 1
            windText, windPrecip = calculate_period_text(
                periods,
                wind,
                calculate_wind_text(maxWind, windUnit, icon, "summary"),
                "wind",
                wind,
                prepAccumUnit,
                visUnits,
                windUnit,
                maxWind,
                windPrecip,
                checkPeriod,
                mode,
                icon,
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


    # If there is no icon then calculate it based on the average cloud cover for the periods if we don't have any precipitation
    if cIcon is None:
        if precipIcon:
            cIcon = precipIcon
        else:
            cIcon = calculate_sky_icon(avgCloud, True, icon)

    return cIcon, summary_text


def nextPeriod(currPeriod):
    if currPeriod == "morning":
        currPeriod = "afternoon"
    elif currPeriod == "afternoon":
        currPeriod = "evening"
    elif currPeriod == "evening":
        currPeriod = "night"
    elif currPeriod == "night":
        currPeriod = "morning"

    return currPeriod
