# %% Script to contain the functions that can be used to generate the weekly text summary of the forecast data for Pirate Weather

import datetime
from dateutil import tz
from PirateTextHelper import (
    Most_Common,
    calculate_precip_text,
    calculate_thunderstorm_text,
)
from itertools import groupby
from operator import itemgetter
from text_const import (
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    PRECIP_PROB_THRESHOLD,
)
from shared_const import MISSING_DATA

WEEK_DAYS = 7
WEEK_DAYS_PLUS_ONE = 8


def calculate_summary_text(
    precipitation, avgIntensity, intensityUnit, icon, maxIntensity
):
    """
    Calculates the precipitation summary if there are between 1 and 8 days of precipitation

    Parameters:
    - precipitation (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - avgIntensity (float): The average precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - maxIntensity (float): The maximum precipitation intensity for the week

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - wIcon (str): The textual summary of conditions (Drizzle, Sleet, etc.).
    - wWeekend (bool): If the summary includes over-weekend or not
    - cIcon (str): The icon representing the conditions for the week.
    """

    wIcon = wText = cIcon = thuText = None
    wWeekend = False
    dayIndexes = []
    days = []
    maxCape = maxLiftedIndex = -999  # Could use a constant for missing data if desired
    numThunderstormDays = 0

    # Loop through each index in the precipitation array
    for day in precipitation:
        # Create an array of day indexes
        dayIndexes.append(day[0])

        # If an icon does not exist then set it. Otherwise if already set, check if the icons are the same and if not use the mixed-precipitation text
        if not wIcon:
            wIcon = day[2]["precipType"]
        elif wIcon != day[2]["precipType"] and wIcon != "mixed-precipitation":
            wIcon = "mixed-precipitation"

        if "cape" in day[2]:
            # Calculate the maximum cape for the week
            if maxCape == -999 and day[2]["cape"] != -999:
                maxCape = day[2]["cape"]
            elif maxCape != -999 and day[2]["cape"] > maxCape:
                maxCape = day[2]["cape"]

        if "liftedIndex" in day[2]:
            # Calculate the maximum lifted index for the week
            if maxLiftedIndex == -999 and day[2]["liftedIndex"] != -999:
                maxLiftedIndex = day[2]["liftedIndex"]
            elif maxLiftedIndex != -999 and day[2]["liftedIndex"] > maxLiftedIndex:
                maxLiftedIndex = day[2]["liftedIndex"]

        # Calculate the number of days with thunderstorms forecasted
        if day[2]["icon"] == "thunderstorm":
            numThunderstormDays += 1

    # Create a list of consecutive days so we can use the through text instead of multiple ands
    for k, g in groupby(enumerate(dayIndexes), lambda ix: ix[0] - ix[1]):
        days.append(list(map(itemgetter(1), g)))

    # If the icon is not mixed precipitation change it to translations format
    if wIcon != "mixed-precipitation":
        wIcon, cIcon = calculate_precip_text(
            maxIntensity,
            intensityUnit,
            wIcon,
            "week",
            maxIntensity,
            maxIntensity,
            maxIntensity,
            1,
            icon,
            "both",
            avgIntensity,
        )
    else:
        cIcon = "sleet"

    # If there are any days with thunderstorms occurring then calculate the text
    if numThunderstormDays > 0:
        thuText = calculate_thunderstorm_text(maxLiftedIndex, maxCape, "summary")

    # If more than half the days with precipitation show thurnderstorms then set the icon to thunderstorm and add it in front of the precipitation text
    if thuText is not None and numThunderstormDays >= (
        len(precipitation) / 2
    ):  # 2 is half the week, could use a constant if desired
        cIcon = "thunderstorm"
        wIcon = ["and", thuText, wIcon]
    # Otherwise show it after the text and use the possible text instead
    elif thuText is not None:
        wIcon = ["and", wIcon, "possible-thunderstorm"]

    if len(precipitation) == 2:
        if (precipitation[0][1] == "saturday" and precipitation[1][1] == "sunday") or (
            precipitation[0][1] == "tomorrow" and precipitation[1][1] == "sunday"
        ):
            # If the precipitation occurs on the weekend then use the over weekend text
            wText = "over-weekend"
            wWeekend = True
        else:
            # Join the days together with an and
            wText = [
                "and",
                precipitation[0][1],
                precipitation[len(precipitation) - 1][1],
            ]
    elif (
        precipitation[len(precipitation) - 1][0] - precipitation[0][0]
        == len(precipitation) - 1
    ):
        wText = [
            "through",
            precipitation[0][1],
            precipitation[len(precipitation) - 1][1],
        ]
    else:
        # Calcuate the start/end of the second day array
        arrTwoStart = len(days[0])
        arrTwoEnd = len(days[0]) + len(days[1])
        arr0 = ""
        arr1 = ""
        arr2 = ""
        arr3 = ""

        if len(days[0]) > 2:
            # If there are more than two indexes in the array use the through text
            arr0 = ["through", precipitation[0][1], precipitation[arrTwoStart - 1][1]]
        elif len(days[0]) == 2:
            # Join the days together with an and
            arr0 = ["and", precipitation[0][1], precipitation[arrTwoStart - 1][1]]
        else:
            # Otherwise just return the day
            arr0 = precipitation[0][1]

        if len(days[1]) > 2:
            # If there are more than two indexes in the array use the through text
            arr1 = [
                "through",
                precipitation[arrTwoStart][1],
                precipitation[arrTwoEnd - 1][1],
            ]
        elif len(days[1]) == 2:
            # Join the days together with an and
            arr1 = [
                "and",
                precipitation[arrTwoStart][1],
                precipitation[arrTwoEnd - 1][1],
            ]
        else:
            # Otherwise just return the day
            arr1 = precipitation[arrTwoStart][1]

        # If there are more than two day indexes then calculate the text for the third index
        if len(days) >= 3:
            # Calcuate the start/end of the third day array
            arrThreeStart = len(days[0]) + len(days[1])
            arrThreeEnd = len(days[0]) + len(days[1]) + len(days[2])

            if len(days[2]) > 2:
                # If there are more than two indexes in the array use the through text
                arr2 = [
                    "through",
                    precipitation[arrThreeStart][1],
                    precipitation[arrThreeEnd - 1][1],
                ]
            elif len(days[2]) == 2:
                # Join the days together with an and
                arr2 = [
                    "and",
                    precipitation[arrThreeStart][1],
                    precipitation[arrThreeEnd - 1][1],
                ]
            else:
                # Otherwise just return the day
                arr2 = precipitation[arrThreeStart][1]

        # If there are four day indexes then calculate the text for the fourth index
        if len(days) == 4:
            # Calcuate the start/end of the fourth day array
            arrFourStart = len(days[0]) + len(days[1]) + len(days[2])
            arrFourEnd = len(days[0]) + len(days[1]) + len(days[2]) + len(days[3])

            if len(days[3]) > 2:
                # If there are more than two indexes in the array use the through text
                arr3 = [
                    "through",
                    precipitation[arrFourStart][1],
                    precipitation[arrFourEnd - 1][1],
                ]
            elif len(days[3]) == 2:
                # Join the days together with an and
                arr3 = [
                    "and",
                    precipitation[arrFourStart][1],
                    precipitation[arrFourEnd - 1][1],
                ]
            else:
                # Otherwise just return the day
                arr3 = precipitation[arrFourStart][1]

        # Join the day indexes together with an and
        if len(days) == 2:
            wText = ["and", arr0, arr1]
        if len(days) == 3:
            wText = ["and", arr0, ["and", arr1, arr2]]
        if len(days) == 4:
            wText = ["and", arr0, ["and", arr1, ["and", arr2, arr3]]]

    return wIcon, wText, wWeekend, cIcon


def calculate_precip_summary(
    precipitation,
    precipitationDays,
    icons,
    avgIntensity,
    intensityUnit,
    avgPop,
    maxIntensity,
    icon="darksky",
):
    """
    Calculates the weekly precipitation summary.

    Parameters:
    - precipitation (bool): If precipitation is occuring during the week or not
    - precipitationDays (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - icons (arr): An array of the daily icons
    - avgIntensity (float): The average precipitation intensity for the week
    - maxIntensity (float): The maximum precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - cIcon (str): An string representing the conditions for the week.
    """

    cIcon = None

    if not precipitation:
        # If there has been no precipitation use the no precipitation text.
        precipSummary = ["for-week", "no-precipitation"]
        # If there has been any fog forecast during the week show that icon, then the wind icon otherwise use the most common icon during the week

        if "fog" in icons:
            cIcon = "fog"
        elif "dangerous-wind" in icons:
            cIcon = "dangerous-windy"
        elif "wind" in icons:
            cIcon = "wind"
        elif "breezy" in icons:
            cIcon = "breezy"
        else:
            cIcon = Most_Common(icons)
    elif len(precipitationDays) == 1:
        # If one day has any precipitation then set the icon to the precipitation type and use the medium precipitation text in the precipitation summary
        text, cIcon = calculate_precip_text(
            maxIntensity,
            intensityUnit,
            precipitationDays[0][2]["precipType"],
            "week",
            precipitationDays[0][2]["precipAccumulation"],
            precipitationDays[0][2]["precipAccumulation"],
            precipitationDays[0][2]["precipAccumulation"],
            avgPop,
            icon,
            "both",
        )
        precipSummary = [
            "during",
            text,
            precipitationDays[0][1],
        ]
    elif 1 < len(precipitationDays) < 8:
        # If between 1 and 8 days have precipitation call the function to calculate the summary text using the precipitation array.
        wIcon, wSummary, wWeekend, cIcon = calculate_summary_text(
            precipitationDays, avgIntensity, intensityUnit, icon, maxIntensity
        )

        # Check if the summary has the over-weekend text
        if not wWeekend:
            precipSummary = ["during", wIcon, wSummary]
        else:
            precipSummary = [wSummary, wIcon]
    elif len(precipitationDays) == 8:
        if (
            precipitationDays[0][2]["precipType"]
            == precipitationDays[1][2]["precipType"]
            == precipitationDays[2][2]["precipType"]
            == precipitationDays[3][2]["precipType"]
            == precipitationDays[4][2]["precipType"]
            == precipitationDays[5][2]["precipType"]
            == precipitationDays[6][2]["precipType"]
            == precipitationDays[7][2]["precipType"]
        ):
            # If all days have precipitation then if they all have the same type then use that icon
            text, cIcon = calculate_precip_text(
                avgIntensity,
                intensityUnit,
                precipitationDays[0][2]["precipType"],
                "week",
                maxIntensity,
                maxIntensity,
                maxIntensity,
                avgPop,
                icon,
                "both",
            )
            # Since precipitation is occuring everyday use the for week text instead of through
            precipSummary = [
                "for-week",
                text,
            ]
        else:
            # If the types are not the same then set the icon to sleet and use the mixed precipitation text (as is how Dark Sky did it)
            cIcon = "sleet"
            precipSummary = ["for-week", "mixed-precipitation"]

    return precipSummary, cIcon


def calculate_temp_summary(highTemp, lowTemp, weekArr):
    """
    Calculates the temperature summary for the week

    Parameters:
    - highTemp (arr): An array that contains the index in the week array, the high temperture of the week and the temperature units.
    - lowTemp (arr): An array that contains the index in the week array, the low temperture of the week and the temperature units.

    Returns:
    - arr: A summary of the temperature for the week.
    """

    # Change C to celsius otherwise change it to fahrenheit
    if highTemp[3] != 0:
        highTemp[3] = "celsius"
    else:
        highTemp[3] = "fahrenheit"

    if lowTemp[3] != 0:
        lowTemp[3] = "celsius"
    else:
        lowTemp[3] = "fahrenheit"

    # If the temperature is increasing everyday or if the lowest temperatue is at the start of the week and the highest temperature is at the end of the week use the rising text
    if all(
        weekArr[i]["temperatureHigh"] < weekArr[i + 1]["temperatureHigh"]
        for i in range(WEEK_DAYS)
    ) or (highTemp[0] >= WEEK_DAYS and lowTemp[0] <= 1):
        # Set the temperature summary
        return [
            "temperatures-rising",
            [highTemp[3], int(round(highTemp[2], 0))],
            highTemp[1],
        ]
    # If the temperature is decreasing everyday or if the lowest temperatue is at the end of the week and the highest temperature is at the start of the week use the rising text
    elif all(
        weekArr[i]["temperatureHigh"] > weekArr[i + 1]["temperatureHigh"]
        for i in range(WEEK_DAYS)
    ) or (highTemp[0] <= 1 and lowTemp[0] >= WEEK_DAYS):
        return [
            "temperatures-falling",
            [lowTemp[3], int(round(lowTemp[2], 0))],
            lowTemp[1],
        ]
    # If the lowest temperatue is in the middle of the week and the highest temperature is at the start or end of the week use the valleying text
    elif (highTemp[0] <= 1 or highTemp[0] >= WEEK_DAYS) and 0 < lowTemp[
        0
    ] < WEEK_DAYS_PLUS_ONE:
        return [
            "temperatures-valleying",
            [lowTemp[3], int(round(lowTemp[2], 0))],
            lowTemp[1],
        ]
    else:
        # Otherwise use the peaking text
        return [
            "temperatures-peaking",
            [highTemp[3], int(round(highTemp[2], 0))],
            highTemp[1],
        ]


def calculate_weekly_text(weekArr, intensityUnit, tempUnit, timeZone, icon="darksky"):
    """
    Calculates the weekly summary given an array of weekdays

    Parameters:
    - weekArr (arr): An array of the weekdays
    - intensityUnit (float): The conversion factor for the precipitation intensity
    - tempUnit (float): The conversion factor for the temperature
    - timeZone (string): The timezone for the current location
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - cText (arr): The precipitation and temperature summary for the week.
    - cIcon (str): The icon representing the conditions for the week.
    """

    # Variables to use in calculating the weekly summary
    cIcon = None
    cText = None
    precipitation = False
    precipitationDays = []
    precipSummary = ""
    highTemp = []
    lowTemp = []
    icons = []
    tempSummary = ""
    avgIntensity = avgPop = maxIntensity = 0
    zone = tz.gettz(timeZone)

    # Loop through the week array
    for idx, day in enumerate(weekArr):
        # Add the daily icon to the list of weekly icons
        icons.append(day["icon"])
        # Determine the day of the week based on the epoch timestamp and the locations timezone
        dayDate = datetime.datetime.fromtimestamp(day["time"], zone)
        weekday = dayDate.strftime("%A").lower()

        # First index is always today, second index is always tomorrow and the last index has the next- text at the start
        if idx == 0:
            weekday = "today"
        elif idx == 1:
            weekday = "tomorrow"
        elif idx == len(weekArr) - 1:
            weekday = "next-" + weekday

        # Check if the day has enough precipitation to reach the threshold and record the index in the array, the day it occured on and the type
        if (
            day["precipType"] == "snow"
            and day["precipAccumulation"]
            >= (DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM * intensityUnit)
            and (
                day["precipProbability"] >= PRECIP_PROB_THRESHOLD
                or day["precipProbability"] == MISSING_DATA
            )
        ):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, weekday, day])
            avgIntensity += intensityUnit * day["precipIntensityMax"]
            avgPop += day["precipProbability"]
            if maxIntensity == 0:
                maxIntensity = day["precipIntensityMax"]
            elif day["precipIntensityMax"] > maxIntensity:
                maxIntensity = day["precipIntensityMax"]
        elif (
            day["precipType"] != "snow"
            and day["precipAccumulation"]
            >= (DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM * intensityUnit)
            and day["precipProbability"] >= PRECIP_PROB_THRESHOLD
        ):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, weekday, day])
            avgIntensity += intensityUnit * day["precipIntensityMax"]
            avgPop += day["precipProbability"]
            if maxIntensity == 0:
                maxIntensity = day["precipIntensityMax"]
            elif day["precipIntensityMax"] > maxIntensity:
                maxIntensity = day["precipIntensityMax"]

        # Determine the highest temperature of the week and record the index in the array, the day it occured on, the temperature and the temperature units
        if not highTemp:
            highTemp = [idx, weekday, day["temperatureHigh"], tempUnit]
        elif day["temperatureHigh"] > highTemp[2]:
            highTemp = [idx, weekday, day["temperatureHigh"], tempUnit]

        # Determine the lowest temperature of the week and record the index in the array, the day it occured on, the temperature and the temperature units
        if not lowTemp:
            lowTemp = [idx, weekday, day["temperatureHigh"], tempUnit]
        elif day["temperatureHigh"] < lowTemp[2]:
            lowTemp = [idx, weekday, day["temperatureHigh"], tempUnit]

    if len(precipitationDays) > 0:
        avgIntensity = avgIntensity / len(precipitationDays)
        avgPop = avgPop / len(precipitationDays)

    precipSummary, cIcon = calculate_precip_summary(
        precipitation,
        precipitationDays,
        icons,
        avgIntensity,
        intensityUnit,
        avgPop,
        maxIntensity,
        icon,
    )

    # If the icon is None then set the icon
    if cIcon is None:
        if "fog" in icons:
            cIcon = "fog"
        elif "smoke" in icons:
            cIcon = "smoke"
        elif "mist" in icons:
            cIcon = "mist"
        elif "haze" in icons:
            cIcon = "haze"
        elif "dangerous-wind" in icons:
            cIcon = "dangerous-windy"
        elif "wind" in icons:
            cIcon = "wind"
        elif "breezy" in icons:
            cIcon = "breezy"
        else:
            cIcon = Most_Common(icons)

    # If the none text exists in the precipitation summary then change it to not available
    if None in precipSummary:
        precipSummary = ["for-week", "unavailable"]

    # Only calcaulte the temperature summary if we have eight days to prevent issues with the time machine
    if len(weekArr) == WEEK_DAYS_PLUS_ONE or (
        highTemp != MISSING_DATA and lowTemp != MISSING_DATA
    ):
        tempSummary = calculate_temp_summary(highTemp, lowTemp, weekArr)
        # Combine the two texts together using with
        cText = ["with", precipSummary, tempSummary]
    else:
        # If there is no precipitation show the no precipitation text instead of no precipitation for the week.
        if len(precipitationDays) == 0:
            precipSummary = ["no-precipitation"]
        cText = precipSummary

    # If we somehow have a generic precipitation icon we use rain instead
    if cIcon == "precipitation":
        cIcon = "rain"

    return cText, cIcon
