# %% Script to contain the functions that can be used to generate the weekly text summary of the forecast data for Pirate Weather

import datetime
from dateutil import tz
from PirateTextHelper import (
    most_common,
    calculate_precipitation,
    calculate_thunderstorm_text,
    ICON_THRESHOLDS,
    MISSING_DATA,
    calculate_consecutive_indexes,
    PRECIP_PROBABILITY_THRESHOLD,
)

# Constants
FAHRENHEIT_UNIT_VALUE = 0
ONE_PRECIPITATION_DAY = 1
TWO_PRECIPITATION_DAYS = 2
THREE_PRECIPITATION_DAYS = 3
FOUR_PRECIPITATION_DAYS = 4
ALL_WEEK_PRECIPITATION_DAYS = 8
HALF_OF_PRECIPITATION_DAYS = 2
HIGH_TEMP_FALLING_THRESHOLD_INDEX = 1
LOW_TEMP_FALLING_THRESHOLD_INDEX = 6
HIGH_TEMP_RISING_THRESHOLD_INDEX = 6
LOW_TEMP_RISING_THRESHOLD_INDEX = 1
HIGH_TEMP_VALLEYING_THRESHOLD_INDEX = 0
LOW_TEMP_VALLEYING_THRESHOLD_INDEX = 7


def calculate_summary_text(
    precipitationDays, averageIntensity, intensityUnit, icon, maxIntensity
):
    """
    Calculates the precipitation summary if there are between 1 and 8 days of precipitation

    Parameters:
    - precipitationDays (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - averageIntensity (float): The average precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - maxIntensity (float): The maximum precipitation intensity for the week

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - weeklyIcon (str): The textual summary of conditions (Drizzle, Sleet, etc.).
    - isWeekend (bool): If the summary includes over-weekend or not
    - currentIcon (str): The icon representing the conditions for the week.
    """

    weeklyIcon = weeklyText = currentIcon = thunderstormText = None
    isWeekend = False
    dayIndexes = []
    days = []
    maxCape = maxLiftedIndex = MISSING_DATA
    numThunderstormDays = 0

    # Loop through each index in the precipitation array
    for day in precipitationDays:
        # Create an array of day indexes
        dayIndexes.append(day[0])

        # If an icon does not exist then set it. Otherwise if already set, check if the icons are the same and if not use the mixed-precipitation text
        if not weeklyIcon:
            weeklyIcon = day[2]["precipType"]
        elif weeklyIcon != day[2]["precipType"] and weeklyIcon != "mixed-precipitation":
            weeklyIcon = "mixed-precipitation"

        if "cape" in day[2]:
            # Calculate the maximum cape for the week
            if maxCape == MISSING_DATA and day[2]["cape"] != MISSING_DATA:
                maxCape = day[2]["cape"]
            elif maxCape != MISSING_DATA and day[2]["cape"] > maxCape:
                maxCape = day[2]["cape"]

        if "liftedIndex" in day[2]:
            # Calculate the maximum lifted index for the week
            if maxLiftedIndex == MISSING_DATA and day[2]["liftedIndex"] != MISSING_DATA:
                maxLiftedIndex = day[2]["liftedIndex"]
            elif (
                maxLiftedIndex != MISSING_DATA
                and day[2]["liftedIndex"] > maxLiftedIndex
            ):
                maxLiftedIndex = day[2]["liftedIndex"]

        # Calculate the number of days with thunderstorms forecasted
        if day[2]["icon"] == "thunderstorm":
            numThunderstormDays += 1

    # Create a list of consecutive days so we can use the through text instead of multiple ands
    days = calculate_consecutive_indexes(dayIndexes)

    # If the icon is not mixed precipitation change it to translations format
    if weeklyIcon != "mixed-precipitation":
        weeklyIcon, currentIcon = calculate_precipitation(
            maxIntensity,
            intensityUnit,
            weeklyIcon,
            "week",
            maxIntensity,
            maxIntensity,
            maxIntensity,
            1,
            icon,
            "both",
            averageIntensity,
        )
    else:
        currentIcon = "sleet"

    # If there are any days with thunderstorms occurring then calculate the text
    if numThunderstormDays > 0:
        thunderstormText = calculate_thunderstorm_text(
            maxLiftedIndex, maxCape, "summary"
        )

    # If more than half the days with precipitation show thurnderstorms then set the icon to thunderstorm and add it in front of the precipitation text
    if thunderstormText is not None and numThunderstormDays >= (
        len(precipitationDays) / HALF_OF_PRECIPITATION_DAYS
    ):
        currentIcon = "thunderstorm"
        weeklyIcon = ["and", thunderstormText, weeklyIcon]
    # Otherwise show it after the text and use the possible text instead
    elif thunderstormText is not None:
        weeklyIcon = ["and", weeklyIcon, "possible-thunderstorm"]

    if len(precipitationDays) == TWO_PRECIPITATION_DAYS:
        if (
            precipitationDays[0][1] == "saturday"
            and precipitationDays[1][1] == "sunday"
        ) or (
            precipitationDays[0][1] == "tomorrow"
            and precipitationDays[1][1] == "sunday"
        ):
            # If the precipitation occurs on the weekend then use the over weekend text
            weeklyText = "over-weekend"
            isWeekend = True
        else:
            # Join the days together with an and
            weeklyText = [
                "and",
                precipitationDays[0][1],
                precipitationDays[len(precipitationDays) - 1][1],
            ]
    elif (
        precipitationDays[len(precipitationDays) - 1][0] - precipitationDays[0][0]
        == len(precipitationDays) - 1
    ):
        weeklyText = [
            "through",
            precipitationDays[0][1],
            precipitationDays[len(precipitationDays) - 1][1],
        ]
    else:
        # Calcuate the start/end of the second day array
        arrayTwoStart = len(days[0])
        arrayTwoEnd = len(days[0]) + len(days[1])
        array0 = ""
        array1 = ""
        array2 = ""
        array3 = ""

        if len(days[0]) > TWO_PRECIPITATION_DAYS:
            # If there are more than two indexes in the array use the through text
            array0 = [
                "through",
                precipitationDays[0][1],
                precipitationDays[arrayTwoStart - 1][1],
            ]
        elif len(days[0]) == TWO_PRECIPITATION_DAYS:
            # Join the days together with an and
            array0 = [
                "and",
                precipitationDays[0][1],
                precipitationDays[arrayTwoStart - 1][1],
            ]
        else:
            # Otherwise just return the day
            array0 = precipitationDays[0][1]

        if len(days[1]) > TWO_PRECIPITATION_DAYS:
            # If there are more than two indexes in the array use the through text
            array1 = [
                "through",
                precipitationDays[arrayTwoStart][1],
                precipitationDays[arrayTwoEnd - 1][1],
            ]
        elif len(days[1]) == 2:
            # Join the days together with an and
            array1 = [
                "and",
                precipitationDays[arrayTwoStart][1],
                precipitationDays[arrayTwoEnd - 1][1],
            ]
        else:
            # Otherwise just return the day
            array1 = precipitationDays[arrayTwoStart][1]

        # If there are more than two day indexes then calculate the text for the third index
        if len(days) >= THREE_PRECIPITATION_DAYS:
            # Calcuate the start/end of the third day array
            arrayThreeStart = len(days[0]) + len(days[1])
            arrayThreeEnd = len(days[0]) + len(days[1]) + len(days[2])

            if len(days[2]) > TWO_PRECIPITATION_DAYS:
                # If there are more than two indexes in the array use the through text
                array2 = [
                    "through",
                    precipitationDays[arrayThreeStart][1],
                    precipitationDays[arrayThreeEnd - 1][1],
                ]
            elif len(days[2]) == TWO_PRECIPITATION_DAYS:
                # Join the days together with an and
                array2 = [
                    "and",
                    precipitationDays[arrayThreeStart][1],
                    precipitationDays[arrayThreeEnd - 1][1],
                ]
            else:
                # Otherwise just return the day
                array2 = precipitationDays[arrayThreeStart][1]

        # If there are four day indexes then calculate the text for the fourth index
        if len(days) == FOUR_PRECIPITATION_DAYS:
            # Calcuate the start/end of the fourth day array
            arrayFourStart = len(days[0]) + len(days[1]) + len(days[2])
            arrayFourEnd = len(days[0]) + len(days[1]) + len(days[2]) + len(days[3])

            if len(days[3]) > TWO_PRECIPITATION_DAYS:
                # If there are more than two indexes in the array use the through text
                array3 = [
                    "through",
                    precipitationDays[arrayFourStart][1],
                    precipitationDays[arrayFourEnd - 1][1],
                ]
            elif len(days[3]) == TWO_PRECIPITATION_DAYS:
                # Join the days together with an and
                array3 = [
                    "and",
                    precipitationDays[arrayFourStart][1],
                    precipitationDays[arrayFourEnd - 1][1],
                ]
            else:
                # Otherwise just return the day
                array3 = precipitationDays[arrayFourStart][1]

        # Join the day indexes together with an and
        if len(days) == TWO_PRECIPITATION_DAYS:
            weeklyText = ["and", array0, array1]
        if len(days) == THREE_PRECIPITATION_DAYS:
            weeklyText = ["and", array0, ["and", array1, array2]]
        if len(days) == FOUR_PRECIPITATION_DAYS:
            weeklyText = ["and", array0, ["and", array1, ["and", array2, array3]]]

    return weeklyIcon, weeklyText, isWeekend, currentIcon


def calculate_precip_summary(
    precipitation,
    precipitationDays,
    icons,
    averageIntensity,
    intensityUnit,
    averagePop,
    maxIntensity,
    icon="darksky",
):
    """
    Calculates the weekly precipitation summary.

    Parameters:
    - precipitation (bool): If precipitation is occuring during the week or not
    - precipitationDays (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - icons (arr): An array of the daily icons
    - averageIntensity (float): The average precipitation intensity for the week
    - maxIntensity (float): The maximum precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - currentIcon (str): An string representing the conditions for the week.
    """

    currentIcon = None
    precipSummary = None

    if not precipitation:
        # If there has been no precipitation use the no precipitation text.
        precipSummary = ["for-week", "no-precipitation"]
        # If there has been any fog forecast during the week show that icon, then the wind icon otherwise use the most common icon during the week

        if "fog" in icons:
            currentIcon = "fog"
        elif "dangerous-wind" in icons:
            currentIcon = "dangerous-windy"
        elif "wind" in icons:
            currentIcon = "wind"
        elif "breezy" in icons:
            currentIcon = "breezy"
        else:
            currentIcon = most_common(icons)
    elif len(precipitationDays) == ONE_PRECIPITATION_DAY:
        # If one day has any precipitation then set the icon to the precipitation type and use the medium precipitation text in the precipitation summary
        text, currentIcon = calculate_precipitation(
            maxIntensity,
            intensityUnit,
            precipitationDays[0][2]["precipType"],
            "week",
            precipitationDays[0][2]["precipAccumulation"],
            precipitationDays[0][2]["precipAccumulation"],
            precipitationDays[0][2]["precipAccumulation"],
            averagePop,
            icon,
            "both",
        )
        precipSummary = [
            "during",
            text,
            precipitationDays[0][1],
        ]
    elif ONE_PRECIPITATION_DAY < len(precipitationDays) < ALL_WEEK_PRECIPITATION_DAYS:
        # If between 1 and 8 days have precipitation call the function to calculate the summary text using the precipitation array.
        weeklyIcon, weeklySummary, isWeekend, currentIcon = calculate_summary_text(
            precipitationDays, averageIntensity, intensityUnit, icon, maxIntensity
        )

        # Check if the summary has the over-weekend text
        if not isWeekend:
            precipSummary = ["during", weeklyIcon, weeklySummary]
        else:
            precipSummary = [weeklySummary, weeklyIcon]
    elif len(precipitationDays) == ALL_WEEK_PRECIPITATION_DAYS:
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
            text, currentIcon = calculate_precipitation(
                averageIntensity,
                intensityUnit,
                precipitationDays[0][2]["precipType"],
                "week",
                maxIntensity,
                maxIntensity,
                maxIntensity,
                averagePop,
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
            currentIcon = "sleet"
            precipSummary = ["for-week", "mixed-precipitation"]

    return precipSummary, currentIcon


def calculate_temp_summary(highTemp, lowTemp, weekArray):
    """
    Calculates the temperature summary for the week

    Parameters:
    - highTemp (arr): An array that contains the index in the week array, the high temperture of the week and the temperature units.
    - lowTemp (arr): An array that contains the index in the week array, the low temperture of the week and the temperature units.

    Returns:
    - arr: A summary of the temperature for the week.
    """

    # Change C to celsius otherwise change it to fahrenheit
    if highTemp[3] != FAHRENHEIT_UNIT_VALUE:
        highTemp[3] = "celsius"
    else:
        highTemp[3] = "fahrenheit"

    if lowTemp[3] != FAHRENHEIT_UNIT_VALUE:
        lowTemp[3] = "celsius"
    else:
        lowTemp[3] = "fahrenheit"

    # If the temperature is increasing everyday or if the lowest temperatue is at the start of the week and the highest temperature is at the end of the week use the rising text
    if weekArray[0]["temperatureHigh"] < weekArray[1]["temperatureHigh"] < weekArray[2][
        "temperatureHigh"
    ] < weekArray[3]["temperatureHigh"] < weekArray[4]["temperatureHigh"] < weekArray[
        5
    ]["temperatureHigh"] < weekArray[6]["temperatureHigh"] < weekArray[7][
        "temperatureHigh"
    ] or (
        highTemp[0] >= HIGH_TEMP_RISING_THRESHOLD_INDEX
        and lowTemp[0] <= LOW_TEMP_RISING_THRESHOLD_INDEX
    ):
        # Set the temperature summary
        return [
            "temperatures-rising",
            [highTemp[3], int(round(highTemp[2], 0))],
            highTemp[1],
        ]
    # If the temperature is decreasing everyday or if the lowest temperatue is at the end of the week and the highest temperature is at the start of the week use the rising text
    elif weekArray[0]["temperatureHigh"] > weekArray[1]["temperatureHigh"] > weekArray[
        2
    ]["temperatureHigh"] > weekArray[3]["temperatureHigh"] > weekArray[4][
        "temperatureHigh"
    ] > weekArray[5]["temperatureHigh"] > weekArray[6]["temperatureHigh"] > weekArray[
        7
    ]["temperatureHigh"] or (
        highTemp[0] <= HIGH_TEMP_FALLING_THRESHOLD_INDEX
        and lowTemp[0] >= LOW_TEMP_FALLING_THRESHOLD_INDEX
    ):
        return [
            "temperatures-falling",
            [lowTemp[3], int(round(lowTemp[2], 0))],
            lowTemp[1],
        ]
    # If the lowest temperatue is in the middle of the week and the highest temperature is at the start or end of the week use the valleying text
    elif (
        highTemp[0] <= HIGH_TEMP_FALLING_THRESHOLD_INDEX
        or highTemp[0] >= LOW_TEMP_FALLING_THRESHOLD_INDEX
    ) and HIGH_TEMP_VALLEYING_THRESHOLD_INDEX < lowTemp[
        0
    ] < LOW_TEMP_VALLEYING_THRESHOLD_INDEX:
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


def calculate_weekly_text(weekArray, intensityUnit, tempUnit, timeZone, icon="darksky"):
    """
    Calculates the weekly summary given an array of weekdays

    Parameters:
    - weekArray (arr): An array of the weekdays
    - intensityUnit (float): The conversion factor for the precipitation intensity
    - tempUnit (float): The conversion factor for the temperature
    - timeZone (string): The timezone for the current location
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - currentText (arr): The precipitation and temperature summary for the week.
    - currentIcon (str): The icon representing the conditions for the week.
    """

    # Variables to use in calculating the weekly summary
    currentIcon = None
    currentText = None
    precipitation = False
    precipitationDays = []
    precipSummary = ""
    highTemp = []
    lowTemp = []
    icons = []
    tempSummary = ""
    averageIntensity = averagePop = maxIntensity = 0
    zone = tz.gettz(timeZone)

    # Loop through the week array
    for idx, day in enumerate(weekArray):
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
        elif idx == 7:
            weekday = "next-" + weekday

        # Check if the day has enough precipitation to reach the threshold and record the index in the array, the day it occured on and the type
        if (
            day["precipType"] == "snow"
            and day["precipAccumulation"]
            >= (ICON_THRESHOLDS["daily_snow_accumulation"] * intensityUnit)
            and (
                day["precipProbability"] >= PRECIP_PROBABILITY_THRESHOLD
                or day["precipProbability"] == MISSING_DATA
            )
        ):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, weekday, day])
            averageIntensity += intensityUnit * day["precipIntensityMax"]
            averagePop += day["precipProbability"]
            if maxIntensity == 0:
                maxIntensity = day["precipIntensityMax"]
            elif day["precipIntensityMax"] > maxIntensity:
                maxIntensity = day["precipIntensityMax"]
        elif (
            day["precipType"] != "snow"
            and day["precipAccumulation"]
            >= (ICON_THRESHOLDS["daily_precipitation_accumulation"] * intensityUnit)
            and day["precipProbability"] >= PRECIP_PROBABILITY_THRESHOLD
        ):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, weekday, day])
            averageIntensity += intensityUnit * day["precipIntensityMax"]
            averagePop += day["precipProbability"]
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
        averageIntensity = averageIntensity / len(precipitationDays)
        averagePop = averagePop / len(precipitationDays)

    precipSummary, currentIcon = calculate_precip_summary(
        precipitation,
        precipitationDays,
        icons,
        averageIntensity,
        intensityUnit,
        averagePop,
        maxIntensity,
        icon,
    )

    # If the icon is None then set the icon
    if currentIcon is None:
        if "fog" in icons:
            currentIcon = "fog"
        elif "smoke" in icons:
            currentIcon = "smoke"
        elif "mist" in icons:
            currentIcon = "mist"
        elif "haze" in icons:
            currentIcon = "haze"
        elif "dangerous-wind" in icons:
            currentIcon = "dangerous-windy"
        elif "wind" in icons:
            currentIcon = "wind"
        elif "breezy" in icons:
            currentIcon = "breezy"
        else:
            currentIcon = most_common(icons)

    # If the none text exists in the precipitation summary then change it to not available
    if None in precipSummary:
        precipSummary = ["for-week", "unavailable"]

    # Only calcaulte the temperature summary if we have eight days to prevent issues with the time machine
    if len(weekArray) == 8 or (highTemp != MISSING_DATA and lowTemp != MISSING_DATA):
        tempSummary = calculate_temp_summary(highTemp, lowTemp, weekArray)
        # Combine the two texts together using with
        currentText = ["with", precipSummary, tempSummary]
    else:
        # If there is no precipitation show the no precipitation text instead of no precipitation for the week.
        if len(precipitationDays) == 0:
            precipSummary = ["no-precipitation"]
        currentText = precipSummary

    # If we somehow have a generic precipitation icon we use rain instead
    if currentIcon == "precipitation":
        currentIcon = "rain"

    return currentText, currentIcon
