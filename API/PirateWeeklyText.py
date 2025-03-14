# %% Script to contain the functions that can be used to generate the weekly text summary of the forecast data for Pirate Weather

from collections import Counter
from itertools import groupby
from operator import itemgetter


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


def calculate_text(precipIntensity, prepIntensityUnit, precipType):
    """
    Calculates the textual precipitation text for the week.

    Parameters:
    - avgIntensity (float): The average precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity
    - precipType (str): The type of precipitation for the week

    Returns:
    - cSummary (str): The textual summary of conditions (very-light-rain, medium-sleet, etc.).
    """

    if precipType == "rain":
        if precipIntensity < (0.4 * prepIntensityUnit):
            cSummary = "very-light-rain"
        elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
            2.5 * prepIntensityUnit
        ):
            cSummary = "light-rain"
        elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
            10 * prepIntensityUnit
        ):
            cSummary = "medium-rain"
        else:
            cSummary = "heavy-rain"
    elif precipType == "snow":
        if precipIntensity < (0.13 * prepIntensityUnit):
            cSummary = "very-light-snow"
        elif precipIntensity >= (0.13 * prepIntensityUnit) and precipIntensity < (
            0.83 * prepIntensityUnit
        ):
            cSummary = "light-snow"
        elif precipIntensity >= (0.83 * prepIntensityUnit) and precipIntensity < (
            3.33 * prepIntensityUnit
        ):
            cSummary = "medium-snow"
        else:
            cSummary = "heavy-snow"
    elif precipType == "sleet":
        if precipIntensity < (0.4 * prepIntensityUnit):
            cSummary = "very-light-sleet"
        elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
            2.5 * prepIntensityUnit
        ):
            cSummary = "light-sleet"
        elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
            10 * prepIntensityUnit
        ):
            cSummary = "medium-sleet"
        else:
            cSummary = "heavy-sleet"
    else:
        # Because soemtimes there's precipitation not no type use a generic precipitation summary
        if precipIntensity < (0.4 * prepIntensityUnit):
            cSummary = "very-light-precipitation"
        elif precipIntensity >= (0.4 * prepIntensityUnit) and precipIntensity < (
            2.5 * prepIntensityUnit
        ):
            cSummary = "light-precipitation"
        elif precipIntensity >= (2.5 * prepIntensityUnit) and precipIntensity < (
            10 * prepIntensityUnit
        ):
            cSummary = "medium-precipitation"
        else:
            cSummary = "heavy-precipitation"

    return cSummary


def calculate_summary_text(precipitation, avgIntensity, intensityUnit):
    """
    Calculates the precipitation summary if there are between 1 and 8 days of precipitation

    Parameters:
    - precipitation (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - avgIntensity (float): The average precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - wIcon (str): The textual summary of conditions (Drizzle, Sleet, etc.).
    - wWeekend (bool): If the summary includes over-weekend or not
    - cIcon (str): The icon representing the conditions for the week.
    """

    wIcon = None
    wText = None
    wWeekend = False
    dayIndexes = []
    days = []
    cIcon = None

    # Loop through each index in the precipitation array
    for day in precipitation:
        # Create an array of day indexes
        dayIndexes.append(day[0])

        # If an icon does not exist then set it. Otherwise if already set, check if the icons are the same and if not use the mixed-precipitation text
        if not wIcon:
            wIcon = day[2]
        elif wIcon != day[2] and wIcon != "mixed-precipitation":
            wIcon = "mixed-precipitation"

    # Create a list of consecutive days so we can use the through text instead of multiple ands
    for k, g in groupby(enumerate(dayIndexes), lambda ix: ix[0] - ix[1]):
        days.append(list(map(itemgetter(1), g)))

    # If the icon is not mixed precipitation change it to translations format
    if wIcon != "mixed-precipitation":
        cIcon = wIcon
        wIcon = calculate_text(avgIntensity, intensityUnit, wIcon)
    else:
        cIcon = "sleet"

    if len(precipitation) == 2:
        if precipitation[0][1] == "saturday" and precipitation[1][1] == "sunday":
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
    precipitation, precipitationDays, icons, avgIntensity, intensityUnit
):
    """
    Calculates the weekly precipitation summary.

    Parameters:
    - precipitation (bool): If precipitation is occuring during the week or not
    - precipitationDays (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - icons (arr): An array of the daily icons
    - avgIntensity (float): The average precipitation intensity for the week
    - intensityUnit (int): The conversion factor for the precipitation intensity

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
        elif "wind" in icons:
            cIcon = "wind"
        else:
            cIcon = Most_Common(icons)
    elif len(precipitationDays) == 1:
        # If one day has any precipitation then set the icon to the precipitation type and use the medium precipitation text in the precipitation summary
        cIcon = precipitationDays[0][2]
        precipSummary = [
            "during",
            calculate_text(avgIntensity, intensityUnit, precipitationDays[0][2]),
            precipitationDays[0][1],
        ]
    elif 1 < len(precipitationDays) < 8:
        # If between 1 and 8 days have precipitation call the function to calculate the summary text using the precipitation array.
        wIcon, wSummary, wWeekend, cIcon = calculate_summary_text(
            precipitationDays, avgIntensity, intensityUnit
        )

        # Check if the summary has the over-weekend text
        if not wWeekend:
            precipSummary = ["during", wIcon, wSummary]
        else:
            precipSummary = [wSummary, wIcon]
    elif len(precipitationDays) == 8:
        if (
            precipitationDays[0][2]
            == precipitationDays[1][2]
            == precipitationDays[2][2]
            == precipitationDays[3][2]
            == precipitationDays[4][2]
            == precipitationDays[5][2]
            == precipitationDays[6][2]
            == precipitationDays[7][2]
        ):
            # If all days have precipitation then if they all have the same type then use that icon
            cIcon = precipitationDays[0][2]
            # Since precipitation is occuring everyday use the for week text instead of through
            precipSummary = [
                "for-week",
                calculate_text(avgIntensity, intensityUnit, precipitationDays[0][2]),
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
    if highTemp[3] == "C":
        highTemp[3] = "celsius"
    else:
        highTemp[3] = "fahrenheit"

    if lowTemp[3] == "C":
        lowTemp[3] = "celsius"
    else:
        lowTemp[3] = "fahrenheit"

    # If the temperature is increasing everyday or if the lowest temperatue is at the start of the week and the highest temperature is at the end of the week use the rising text
    if weekArr[0][5] < weekArr[1][5] < weekArr[2][5] < weekArr[3][5] < weekArr[4][
        5
    ] < weekArr[5][5] < weekArr[6][5] < weekArr[7][5] or (
        highTemp[0] >= 6 and lowTemp[0] <= 1
    ):
        # Set the temperature summary
        return [
            "temperatures-rising",
            [highTemp[3], round(highTemp[2], 0)],
            highTemp[1],
        ]
    # If the temperature is decreasing everyday or if the lowest temperatue is at the end of the week and the highest temperature is at the start of the week use the rising text
    elif weekArr[0][5] > weekArr[1][5] > weekArr[2][5] > weekArr[3][5] > weekArr[4][
        5
    ] > weekArr[5][5] > weekArr[6][5] > weekArr[7][5] or (
        highTemp[0] <= 1 and lowTemp[0] >= 6
    ):
        return ["temperatures-falling", [lowTemp[3], round(lowTemp[2], 0)], lowTemp[1]]
    # If the lowest temperatue is in the middle of the week and the highest temperature is at the start or end of the week use the valleying text
    elif (highTemp[0] <= 1 or highTemp[0] >= 6) and 0 < lowTemp[0] < 7:
        return [
            "temperatures-valleying",
            [lowTemp[3], round(lowTemp[2], 0)],
            lowTemp[1],
        ]
    else:
        # Otherwise use the peaking text
        return [
            "temperatures-peaking",
            [highTemp[3], round(highTemp[2], 0)],
            highTemp[1],
        ]


def calculate_weeky_text(weekArr):
    """
    Calculates the weekly summary given an array of weekdays

    Parameters:
    - weekArr (arr): An array of the weekdays

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
    avgIntensity = 0
    intensityUnit = 0

    # Loop through the week array
    for idx, day in enumerate(weekArr):
        # Add the daily icon to the list of weekly icons
        icons.append(day[6])

        # Check if the day has enough precipitation to reach the threshold and record the index in the array, the day it occured on and the type
        if day[1] == "snow" and (day[2] * day[3]) >= (1.0 * day[3]):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, day[0], day[1]])
            avgIntensity += day[7] * day[3]
            intensityUnit = day[3]
        elif day[1] == "rain" and (day[2] * day[3]) >= (0.01 * day[3]):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, day[0], day[1]])
            avgIntensity += day[7] * day[3]
            intensityUnit = day[3]
        elif day[1] == "sleet" and (day[2] * day[3]) >= (0.01 * day[3]):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, day[0], day[1]])
            avgIntensity += day[7] * day[3]
            intensityUnit = day[3]
        elif day[1] == "none" and (day[2] * day[3]) >= (0.01 * day[3]):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, day[0], "precipitation"])
            avgIntensity += day[7] * day[3]
            intensityUnit = day[3]

        # Determine the highest temperature of the week and record the index in the array, the day it occured on, the temperature and the temperature units
        if not highTemp:
            highTemp = [idx, day[0], day[4], day[5]]
        elif day[4] > highTemp[2]:
            highTemp = [idx, day[0], day[4], day[5]]

        # Determine the lowest temperature of the week and record the index in the array, the day it occured on, the temperature and the temperature units
        if not lowTemp:
            lowTemp = [idx, day[0], day[4], day[5]]
        elif day[4] < lowTemp[2]:
            lowTemp = [idx, day[0], day[4], day[5]]

    if len(precipitationDays) > 0:
        avgIntensity = avgIntensity / len(precipitationDays)
    precipSummary, cIcon = calculate_precip_summary(
        precipitation, precipitationDays, icons, avgIntensity, intensityUnit
    )

    tempSummary = calculate_temp_summary(highTemp, lowTemp, weekArr)

    # Combine the two texts together using with
    cText = ["sentence", ["with", precipSummary, tempSummary]]

    # If we somehow have a generic precipitation icon we use rain instead
    if cIcon == "precipitation":
        cIcon = "rain"

    return cText, cIcon
