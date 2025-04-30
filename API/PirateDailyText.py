# %% Script to contain the functions that can be used to generate the daily text summary of the forecast data for Pirate Weather
from PirateTextHelper import (
    calculate_precip_text,
    calculate_wind_text,
    calculate_vis_text,
    calculate_sky_icon,
    humidity_sky_text,
    Most_Common,
)
import datetime
import math
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
    dry,
    humid,
    tempUnits,
    dryPrecip,
    humidPrecip,
    visPrecip,
    vis,
    later,
):
    """
    Calculates the period text

    Parameters:
    - periods (arr) -  An array representing all the periods in the day/next 24h
    - typePeriods (arr) - An array representing the type (wind/cloud/precip/vis)
    - text (str) - A string representing the text for the type (light-rain, fog, etc.)
    - type (str) - The current type we are checking (precip, cloud, etc.)
    - wind (arr) - An array of the wind times
    - prepAccumUnit (float): The precipitation unit used
    - visUnits (float): The visibility unit used
    - maxWind (float) - The maximum wind for all the periods
    - windPrecip (bool) - Whether wind is occuring with cloud cover or precipitation.
    - checkPeriod (float) - The current period
    - mode (str) - Whether the summary is for the day or the next 24h
    - dry (arr) - An array of the low humidity times
    - humid (arr) - An array of the high humidity times
    - tempUnits (float): The temperature unit used
    - dryPrecip (bool) - Whether low humidity is occuring with other conditions.
    - humidPrecip (bool) - Whether high humidity is occuring with other conditions.
    - later (arr) - List of conditions that start later

    Returns:
    - summary_text (str) - The textual representation of the type for the current day/next 24 hours
    - windPrecip (bool) - Returns if wind is occuring with precipitation or cloud cover
    - dryPrecip (bool) - Whether low humidity is occuring with other conditions.
    - humidPrecip (bool) - Whether high humidity is occuring with other conditions.
    """
    # Set the period text to the current text
    periodText = text
    summary_text = None
    # If there is only one period then just use that period
    if len(typePeriods) == 1:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in one period
        if (
            len(wind) == 1
            and (type == "precip" or type == "cloud")
            and (later and "wind" in later)
        ):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if wind[0] == typePeriods[0]:
                windPrecip = True
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
        # If the type is precipitation/cloud cover then check the vis if the vis is occuring in one period
        if (
            len(vis) == 1
            and type != "vis"
            and type != "wind"
            and (later and "vis" in later)
        ):
            # If they both occur at the same time then join with an and. Set visPrecip to true to skip checking the wind
            if vis[0] == typePeriods[0]:
                visPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "fog",
                ]
        if (
            len(dry) == 1
            and (type != "precip" and type != "dry" and type != "humid")
            and (later and "humid" in later)
        ):
            if dry[0] == typePeriods[0]:
                dryPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "low-humidity",
                ]
        if (
            len(humid) == 1
            and (type != "dry" and type != "humid")
            and (later and "dry" in later)
        ):
            if humid[0] == typePeriods[0]:
                dryPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "high-humidity",
                ]
        summary_text = ["during", periodText, periods[typePeriods[0]]]
    # If the type has two periods
    elif len(typePeriods) == 2:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in two periods
        if (
            len(wind) == 2
            and (type == "precip" or type == "cloud")
            and (later and "wind" in later)
        ):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if wind[0] == typePeriods[0] and wind[1] == typePeriods[1]:
                windPrecip = True
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
        # If the type is precipitation/cloud cover then check the vis if the vis is occuring in two periods
        if (
            len(vis) == 2
            and type != "vis"
            and type != "wind"
            and (later and "vis" in later)
        ):
            # If they both occur at the same time then join with an and. Set visPrecip to true to skip checking the vis
            if vis[0] == typePeriods[0] and vis[1] == typePeriods[1]:
                visPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "fog",
                ]
        if (
            len(dry) == 2
            and (type != "precip" and type != "dry" and type != "humid")
            and (later and "dry" in later)
        ):
            if dry[0] == typePeriods[0] and dry[1] == typePeriods[1]:
                dryPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "low-humidity",
                ]
        if (
            len(humid) == 2
            and (type != "dry" and type != "humid")
            and (later and "humid" in later)
        ):
            if humid[0] == typePeriods[0] and humid[1] == typePeriods[1]:
                humidPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "high-humidity",
                ]
        # If the type starts in the third period
        if (
            typePeriods[0] == checkPeriod + 2
            and typePeriods[1] == 3
            and len(periods) == 4
        ):
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        # If the type starts at the start but does not continue to the end
        elif typePeriods[0] == checkPeriod and (typePeriods[1] - typePeriods[0]) == 1:
            # We need to set the ending period to the next period. If the end occurs in the fourth period we set the period to 4 to prevent array out of bounds
            summary_text = [
                "until",
                periodText,
                periods[
                    len(periods) - 1
                    if typePeriods[1] + 1 > len(periods) - 1
                    else typePeriods[1] + 1
                ],
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
        else:
            summary_text = [
                "during",
                periodText,
                ["and", periods[typePeriods[0]], periods[typePeriods[1]]],
            ]
        # If the first period has the later text and the summary is until change it to starting
        if (
            "later" in periods[0]
            and "until" in summary_text
            and "starting" not in summary_text
        ):
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        if "later" in periods[0] and "until-starting-again" in summary_text:
            summary_text = [
                "and",
                [
                    "during",
                    periodText,
                    periods[typePeriods[0]],
                ],
                [
                    "during",
                    periodText,
                    periods[typePeriods[1]],
                ],
            ]
    # If the type occurs during three perionds
    elif len(typePeriods) == 3:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in three periods
        if (
            len(wind) == 3
            and (type == "precip" or type == "cloud")
            and (later and "wind" in later)
        ):
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
        # If the type is precipitation/cloud cover then check the vis if the vis is occuring in three periods
        if (
            len(vis) == 3
            and type != "vis"
            and type != "wind"
            and (later and "vis" in later)
        ):
            # If they both occur at the same time then join with an and. Set visPrecip to true to skip checking the vis
            if (
                vis[0] == typePeriods[0]
                and vis[1] == typePeriods[1]
                and vis[2] == typePeriods[2]
            ):
                visPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "fog",
                ]
        # If the type is not precip/dry/humid then check if low humidity is occuring in three periods
        if (
            len(dry) == 3
            and (type != "precip" and type != "dry" and type != "humid")
            and (later and "dry" in later)
        ):
            # If they both occur at the same time then join with an and. Set dryPrecip to true to skip checking the low humidity
            if (
                dry[0] == typePeriods[0]
                and dry[1] == typePeriods[1]
                and dry[2] == typePeriods[2]
            ):
                dryPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "low-humidity",
                ]
        # If the type is not dry/humid then check if high humidity is occuring in three periods
        if (
            len(humid) == 3
            and (type != "dry" and type != "humid")
            and (later and "humid" in later)
        ):
            # If they both occur at the same time then join with an and. Set humidPrecip to true to skip checking the high humidity
            if (
                humid[0] == typePeriods[0]
                and humid[1] == typePeriods[1]
                and humid[2] == typePeriods[2]
            ):
                humidPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "high-humidity",
                ]
        # If the type starts in the second period
        if (
            typePeriods[0] == checkPeriod + 1
            and typePeriods[2] == 3
            and len(periods) == 4
        ):
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        # If the type starts in the second period
        elif (
            typePeriods[0] == checkPeriod + 2
            and typePeriods[2] == 4
            and len(periods) == 5
        ):
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
                periods[
                    len(periods) - 1
                    if typePeriods[2] + 1 > len(periods) - 1
                    else typePeriods[2] + 1
                ],
            ]
        # If the two types are not joined and do not occur during the first or last period
        elif (
            typePeriods[1] - typePeriods[0] != 1
            and typePeriods[2] - typePeriods[1] != 1
        ):
            summary_text = [
                "during",
                periodText,
                [
                    "and",
                    periods[typePeriods[0]],
                    ["and", periods[typePeriods[1]], periods[typePeriods[2]]],
                ],
            ]
        # If the type starts after the first period but doesn't continue to the end and the first and second periods are connected
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[1] - typePeriods[0]) == 1
            and typePeriods[2] >= 3
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
            and typePeriods[1] >= 2
        ):
            summary_text = [
                "until-starting-again",
                periodText,
                periods[typePeriods[0] + 1],
                periods[typePeriods[1]],
            ]
        # If the type starts after the first period but doesn't continue to the end
        elif (
            typePeriods[0] > checkPeriod
            and (typePeriods[1] - typePeriods[0]) == 1
            and (typePeriods[2] - typePeriods[1]) == 1
            and len(periods) == 5
        ):
            summary_text = [
                "starting-continuing-until",
                periodText,
                periods[typePeriods[0]],
                periods[
                    len(periods) - 1
                    if typePeriods[2] + 1 > len(periods) - 1
                    else typePeriods[2] + 1
                ],
            ]
        # If the first period has the later text and the summary is until change it to starting
        if (
            "later" in periods[0]
            and "until" in summary_text
            and "starting" not in summary_text
            and (later and type in later)
        ):
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        # If the first period has the later text and the summary is until starting again change it to be during and add the second period
        if (
            "later" in periods[0]
            and "until-starting-again" in summary_text
            and (typePeriods[1] - typePeriods[0]) == 1
            and typePeriods[2] >= 3
        ):
            summary_text = [
                "and",
                [
                    "starting-continuing-until",
                    periodText,
                    periods[typePeriods[0]],
                    periods[typePeriods[1] + 1],
                ],
                [
                    "during",
                    periodText,
                    periods[typePeriods[2]],
                ],
            ]
        # If the first period has the later text and the summary is until starting again change it to be during and add the second period
        if (
            "later" in periods[0]
            and "until-starting-again" in summary_text
            and (typePeriods[1] - typePeriods[0]) != 1
            and typePeriods[1] >= 2
        ):
            summary_text = [
                "and",
                [
                    "during",
                    periodText,
                    periods[typePeriods[0]],
                ],
                [
                    "starting-continuing-until",
                    periodText,
                    periods[typePeriods[1]],
                    periods[
                        len(periods) - 1
                        if typePeriods[2] + 1 > len(periods) - 1
                        else typePeriods[2] + 1
                    ],
                ],
            ]
    # If the type occurs during four perionds and we have five periods
    elif len(typePeriods) == 4 and len(periods) == 5:
        # If the type is precipitation/cloud cover then check the wind if the wind is occuring in four periods
        if (
            len(wind) == 4
            and (type == "precip" or type == "cloud")
            and (later and "wind" in later)
        ):
            # If they both occur at the same time then join with an and. Set windPrecip to true to skip checking the wind
            if (
                wind[0] == typePeriods[0]
                and wind[1] == typePeriods[1]
                and wind[2] == typePeriods[2]
                and wind[3] == typePeriods[3]
                and (later and "wind" in later)
            ):
                windPrecip = True
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
        # If the type is precipitation/cloud cover then check the wind if the vis is occuring in four periods
        if (
            len(vis) == 4
            and type != "vis"
            and type != "wind"
            and (later and "vis" in later)
        ):
            # If they both occur at the same time then join with an and. Set visPrecip to true to skip checking the vis
            if (
                vis[0] == typePeriods[0]
                and vis[1] == typePeriods[1]
                and vis[2] == typePeriods[2]
                and vis[3] == typePeriods[3]
                and (later and "vis" in later)
            ):
                visPrecip = True
                periodText = [
                    "and",
                    periodText,
                    calculate_wind_text(maxWind, windUnit, icon, "summary"),
                ]
        # If the type is not precip/dry/humid then check if low humidity is occuring in four periods
        if (
            len(dry) == 4
            and (type != "precip" and type != "dry" and type != "humid")
            and (later and "dry" in later)
        ):
            # If they both occur at the same time then join with an and. Set dryPrecip to true to skip checking the low humidity
            if (
                dry[0] == typePeriods[0]
                and dry[1] == typePeriods[1]
                and dry[2] == typePeriods[2]
                and dry[3] == typePeriods[3]
                and (later and "dry" in later)
            ):
                dryPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "low-humidity",
                ]
        # If the type is not /ry/humid then check the dry if high humidity is occuring in four periods
        if (
            len(humid) == 4
            and (type != "dry" and type != "humid")
            and (later and "humid" in later)
        ):
            # If they both occur at the same time then join with an and. Set humidPrecip to true to skip checking the high humidity
            if (
                humid[0] == typePeriods[0]
                and humid[1] == typePeriods[1]
                and humid[2] == typePeriods[2]
                and humid[3] == typePeriods[3]
                and (later and "humid" in later)
            ):
                humidPrecip = True
                periodText = [
                    "and",
                    periodText,
                    "high-humidity",
                ]
        # If the type starts in the second period
        if typePeriods[0] == checkPeriod + 1 and typePeriods[3] == 4:
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[1] - typePeriods[0]) == 1
            and (typePeriods[2] - typePeriods[1]) == 1
            and (typePeriods[3] - typePeriods[2]) == 1
        ):
            # We need to set the ending period to the next period. If the end occurs in the fourth period we set the period to 4 to prevent array out of bounds
            summary_text = [
                "until",
                periodText,
                periods[typePeriods[3]],
            ]
        # If the type starts after the first period but doesn't continue to the end
        elif typePeriods[0] > checkPeriod and (typePeriods[1] - typePeriods[0]) == 1:
            summary_text = [
                "starting-continuing-until",
                periodText,
                periods[typePeriods[0]],
                periods[typePeriods[3]],
            ]
        # If the type occurs in the first period but starts again after the first period
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[2] - typePeriods[1]) == 1
            and (typePeriods[3] - typePeriods[2]) == 1
            and typePeriods[3] == 4
        ):
            summary_text = [
                "until-starting-again",
                periodText,
                periods[typePeriods[0] + 1],
                periods[typePeriods[1]],
            ]
        # If the type in the first period but doesn't continue to the end and the last period isn't connected to the others
        elif (
            typePeriods[0] == checkPeriod
            and (typePeriods[3] - typePeriods[2]) != 1
            and typePeriods[3] == 4
        ):
            summary_text = [
                "until-starting-again",
                periodText,
                periods[typePeriods[2] + 1],
                periods[typePeriods[3]],
            ]
        # If the first period has the later text and the summary is until change it to starting
        if (
            "later" in periods[0]
            and "until" in summary_text
            and "starting" not in summary_text
            and (later and type in later)
        ):
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        # If the first period has the later text and the summary is until starting again change it to be during and add the second period
        if (
            "later" in periods[0]
            and "until-starting-again" in summary_text
            and (typePeriods[2] - typePeriods[1]) == 1
            and (typePeriods[3] - typePeriods[2]) == 1
            and typePeriods[3] == 4
        ):
            summary_text = [
                "and",
                [
                    "during",
                    periodText,
                    periods[typePeriods[0]],
                ],
                [
                    "starting",
                    periodText,
                    periods[typePeriods[1]],
                ],
            ]
        # If the first period has the later text and the summary is until starting again change it to be during and add the second period
        if (
            "later" in periods[0]
            and "until-starting-again" in summary_text
            and (typePeriods[3] - typePeriods[2]) != 1
            and typePeriods[3] == 4
        ):
            summary_text = [
                "and",
                [
                    "starting-continuing-until",
                    periodText,
                    periods[typePeriods[0]],
                    periods[typePeriods[2]],
                ],
                [
                    "starting",
                    periodText,
                    periods[typePeriods[3]],
                ],
            ]
    # If the type occurs all day then use the for-day text and we have four periods
    elif len(typePeriods) == 4 and len(periods) == 4:
        # If they both occur at the same time and the type is precip/cloud then join with an and. Set windPrecip to true to skip checking the wind
        if (
            len(wind) == 4
            and (type == "precip" or type == "cloud")
            and (later and "wind" in later)
        ):
            windPrecip = True
            summary_text = [
                "and",
                periodText,
                calculate_wind_text(maxWind, windUnit, icon, "summary"),
            ]
        # If they both occur at the same time and the type is precip/cloud then join with an and. Set visPrecip to true to skip checking the vis
        if (
            len(vis) == 4
            and type != "vis"
            and type != "wind"
            and (later and "vis" in later)
        ):
            visPrecip = True
            summary_text = [
                "and",
                periodText,
                "fog",
            ]
        if (
            len(dry) == 4
            and (type != "precip" and type != "dry" and type != "humid")
            and (later and "dry" in later)
        ):
            dryPrecip = True
            periodText = [
                "and",
                periodText,
                "low-humidity",
            ]
        if (
            len(humid) == 4
            and (type != "dry" and type != "humid")
            and (later and "humid" in later)
        ):
            humidPrecip = True
            periodText = [
                "and",
                periodText,
                "high-humidity",
            ]

        # If we're in hourly mode and the first period has the later text use the starting text instead of for day text
        if mode == "hour" and "later" in periods[0]:
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        else:
            summary_text = ["for-day", periodText]
    # If the type occurs all day then use the for-day text and we have five periods
    elif len(typePeriods) == 5 and len(periods) == 5:
        # If they both occur at the same time and the type is precip/cloud then join with an and. Set windPrecip to true to skip checking the wind
        if (
            len(wind) == 5
            and (type == "precip" or type == "cloud")
            and (later and "wind" in later)
        ):
            windPrecip = True
            summary_text = [
                "and",
                periodText,
                calculate_wind_text(maxWind, windUnit, icon, "summary"),
            ]
        # If they both occur at the same time and the type is precip/cloud then join with an and. Set visPrecip to true to skip checking the vis
        if (
            len(vis) == 5
            and type != "vis"
            and type != "wind"
            and (later and "vis" in later)
        ):
            visPrecip = True
            summary_text = [
                "and",
                periodText,
                "fog",
            ]
        # If they both occur at the same time and the type is not precip/dry/humid then join with an and. Set dryPrecip to true to skip checking the low humidity
        if (
            len(dry) == 5
            and (type != "precip" and type != "dry" and type != "humid")
            and (later and "dry" in later)
        ):
            dryPrecip = True
            periodText = [
                "and",
                periodText,
                "low-humidity",
            ]
        # If they both occur at the same time and the type is not dry/humid then join with an and. Set humidPrecip to true to skip checking the high humidity
        if (
            len(humid) == 5
            and (type != "dry" and type != "humid")
            and (later and "humid" in later)
        ):
            humidPrecip = True
            periodText = [
                "and",
                periodText,
                "high-humidity",
            ]

        # If the first period has the later text use the starting text instead of for day text
        if "later" in periods[0] and (later and type in later):
            summary_text = ["starting", periodText, periods[typePeriods[0]]]
        else:
            summary_text = ["for-day", periodText]
    return summary_text, windPrecip, dryPrecip, humidPrecip, visPrecip


def calculate_day_text(
    hours,
    prepAccumUnit,
    visUnits,
    windUnit,
    tempUnits,
    isDayTime,
    timeZone,
    currTime,
    mode="daily",
    icon="darksky",
):
    """
    Calculates the current day/next 24h text

    Parameters:
    - hours (arr) - The array of hours for the day or next 24 hours.
    - prepAccumUnit (float): The precipitation unit used
    - visUnits (float): The visibility unit used
    - tempUnits (float): The temperature unit used
    - isDayTime (bool): Whether its currently day or night
    - timeZone (string): The timezone for the current location
    - currTime (int): The current epoch time
    - mode (str): Which mode to run the function in
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - summary_text (str) - The textual representation of the current day/next 24 hours
    - cIcon (str) - The icon representing the current day/next 24 hours
    """

    # If we don't have 24 hours of data bail as we need 24 hours to calculate the text
    if len(hours) != 24:
        return "clear-day", ["for-day", "clear"]

    # Variables to calculate the periods from the hours array
    period1 = []
    period2 = []
    period3 = []
    period4 = []
    period5 = []
    prepTypes = []
    mostCommonPrecip = []
    periodStats = [[], [], [], [], []]
    currPeriod = None
    numHoursFog = numHoursWind = numHoursDry = numHoursHumid = rainPrep = snowPrep = (
        sleetPrep
    ) = snowError = cloudCover = pop = maxIntensity = maxWind = length = 0
    periodIncrease = False
    periodIndex = 1
    today = ""

    # Get the current time zone from the function parameters or use the first hours time field as the current time
    zone = tz.gettz(timeZone)
    if mode == "hour":
        currDate = datetime.datetime.fromtimestamp(hours[0]["time"], zone)
        currHour = int(currDate.strftime("%H"))
        # If the first forecasted hour is midnight local change it to correct the summaries assuming the forecast is for the same day
        if currHour == 0:
            currHour = 23
    else:
        # Calculate the current hour/weekday from the first hour
        currDate = datetime.datetime.fromtimestamp(currTime, zone)
        currHour = int(currDate.strftime("%H"))

    # Time periods are as follows:
    # morning 4:00 to 12:00
    # afternoon 12:00 to 17:00
    # evening 17:00 to 22:00
    # night: 22:00 to 4:00

    # Calculate the current period and set the end and when to skip the block for the hour block summaries
    if 4 <= currHour < 12:
        currPeriod = "morning"
    elif 12 <= currHour < 17:
        currPeriod = "afternoon"
    elif 17 <= currHour < 22:
        currPeriod = "evening"
    else:
        currPeriod = "night"

    # Set the hour period to the current period
    hourPeriod = currPeriod

    # For the hour block summaries add the today text
    if mode == "hour":
        today = "today-"

    # Loop through the hours to calculate the conditions for each period
    for idx, hour in enumerate(hours):
        # Calculate the time and weekday for the current hour in the loop
        hourDate = datetime.datetime.fromtimestamp(hour["time"], zone)
        hourHour = int(hourDate.strftime("%H"))

        # Since the summaries are calculated from 4am to 4am add 24 hours to hours 0 to 3 so its seen as the current day
        if 0 <= hourHour < 4:
            hourHour = hourHour + 24
        # If we are at hour 12 and the first period has data increase the period index and set the increase flag to true
        if hourHour == 12 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True
        # If we are at hour 17 and the first period has data increase the period index and set the increase flag to true
        if hourHour == 17 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True
        # If we are at hour 22 and the first period has data increase the period index and set the increase flag to true
        if hourHour == 22 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True
        # If we are at hour 12 and the first period has data increase the period index and set the increase flag to true
        if hourHour == 4 and period1:
            periodIndex = periodIndex + 1
            periodIncrease = True

        # If the current hour has any precipitation calculate the rain, snow and sleet precipitation
        if hour["precipType"] == "rain" or hour["precipType"] == "none":
            rainPrep = rainPrep + hour["precipAccumulation"]
        elif hour["precipType"] == "snow":
            snowPrep = snowPrep + hour["precipAccumulation"]
            # Increase the snow error to show the precipitation range
            snowError = snowError + hour["precipIntensityError"]
        elif hour["precipType"] == "sleet":
            sleetPrep = sleetPrep + hour["precipAccumulation"]

        # If the hour is humid increase the number of humid hours
        if (
            humidity_sky_text(hour["temperature"], tempUnits, hour["humidity"])
            == "high-humidity"
        ):
            numHoursHumid += 1
        # If the hour is dry increase the number of dry hours
        if (
            humidity_sky_text(hour["temperature"], tempUnits, hour["humidity"])
            == "low-humidity"
        ):
            numHoursDry += 1
        # If the hour is foggy increase the number of fog hours
        if (
            calculate_vis_text(hour["visibility"], visUnits, "icon") == "fog"
            and hour["precipIntensity"] <= 0.02 * prepAccumUnit
        ):
            numHoursFog += 1
        # If the hour is windy increase the number of windy hours
        if (
            calculate_wind_text(hour["windSpeed"], windUnit, "darksky", "icon")
            == "wind"
        ):
            numHoursWind += 1

        # Add the hour cloud cover to calculate the average
        cloudCover += hour["cloudCover"]
        # Calculate the maximum pop for the period
        if pop == 0:
            pop = hour["precipProbability"]
        elif hour["precipProbability"] > pop:
            pop = hour["precipProbability"]

        # Calculate the maxiumum intensity for the period
        if maxIntensity == 0:
            maxIntensity = hour["precipIntensity"]
        elif maxIntensity > 0 and hour["precipIntensity"] > maxIntensity:
            maxIntensity = hour["precipIntensity"]

        # Calculate the maximum wind speed for the period
        if maxWind == 0:
            maxWind = hour["windSpeed"]
        elif maxWind > 0 and hour["windSpeed"] > maxWind:
            maxWind = hour["windSpeed"]

        # Add the percipitation type to an array to calculate the most common precipitation to use as a baseline
        if hour["precipIntensity"] > 0.02 * prepAccumUnit:
            mostCommonPrecip.append(hour["precipType"])

        # Add the percipitation type to an array of precipitation types if it doesn;t already exist
        if not prepTypes and hour["precipIntensity"] > 0.02 * prepAccumUnit:
            prepTypes.append(hour["precipType"])
        elif (
            hour["precipType"] not in prepTypes
            and hour["precipIntensity"] > 0.02 * prepAccumUnit
        ):
            prepTypes.append(hour["precipType"])

        # Add the hour to the period array depending on the index
        if periodIndex == 1:
            period1.append(hour)
        elif periodIndex == 2:
            period2.append(hour)
        elif periodIndex == 3:
            period3.append(hour)
        elif periodIndex == 4:
            period4.append(hour)
        elif periodIndex == 5:
            period5.append(hour)

        # If the period changed and the index is 6 or below or we are at the end of the loop
        if (periodIncrease and periodIndex <= 6) or (idx == 23 and periodIndex <= 6):
            # Calculate the average cloud cover and pop for the period and calculate the length of the period
            if periodIndex - 1 == 1:
                cloudCover = cloudCover / len(period1)
                length = len(period1)
            elif periodIndex - 1 == 2:
                cloudCover = cloudCover / len(period2)
                length = len(period2)
            elif periodIndex - 1 == 3:
                cloudCover = cloudCover / len(period3)
                length = len(period3)
            elif periodIndex - 1 == 4:
                cloudCover = cloudCover / len(period4)
                length = len(period4)
            elif periodIndex - 1 == 5:
                cloudCover = cloudCover / len(period5)
                length = len(period5)

            # If we are at the end of the loop increase the index and calculate the length of the last period
            if idx == 23 and not periodIncrease:
                periodIndex += 1
                if periodIndex == 5:
                    length = len(period4)
                else:
                    length = len(period5)

            # Add the data to an array of period arrays to use to calculate the summaries
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

            # Reset the varaibles back to zero
            numHoursFog = numHoursWind = numHoursDry = numHoursHumid = rainPrep = (
                snowPrep
            ) = sleetPrep = snowError = cloudCover = pop = maxWind = 0
            periodIncrease = False

            # Calculate the next period text
            hourPeriod = nextPeriod(hourPeriod)

            # If we are in hourly mode and hit hour 4 use tomorrow as the text unless we are in hours 0, 1, 2 or 3
            if hourHour == 4 and period1 and mode == "hour" and currHour > 3:
                today = "tomorrow-"

    # If the second to last value in hour array is an increase hour we will have data in the fifth period but no stats so calculate them
    if period5 and not periodStats[4]:
        # If the current hour has any precipitation calculate the rain, snow and sleet precipitation
        if period5[0]["precipType"] == "rain" or period5[0]["precipType"] == "none":
            rainPrep = rainPrep + period5[0]["precipAccumulation"]
        elif period5[0]["precipType"] == "snow":
            snowPrep = snowPrep + period5[0]["precipAccumulation"]
            # Increase the snow error to show the precipitation range
            snowError = snowError + period5[0]["precipIntensityError"]
        elif period5[0]["precipType"] == "sleet":
            sleetPrep = sleetPrep + period5[0]["precipAccumulation"]

        # If the hour is humid increase the number of humid hours
        if (
            humidity_sky_text(
                period5[0]["temperature"], tempUnits, period5[0]["humidity"]
            )
            == "high-humidity"
        ):
            numHoursHumid += 1
        # If the hour is dry increase the number of dry hours
        if (
            humidity_sky_text(
                period5[0]["temperature"], tempUnits, period5[0]["humidity"]
            )
            == "low-humidity"
        ):
            numHoursDry += 1
        # If the hour is foggy increase the number of fog hours
        if (
            calculate_vis_text(period5[0]["visibility"], visUnits, "icon") == "fog"
            and period5[0]["precipIntensity"] <= 0.02 * prepAccumUnit
        ):
            numHoursFog += 1
        # If the hour is windy increase the number of windy hours
        if (
            calculate_wind_text(period5[0]["windSpeed"], windUnit, "darksky", "icon")
            == "wind"
        ):
            numHoursWind += 1

        # Add the hour cloud cover and pop to the variables to calculate the average
        cloudCover += period5[0]["cloudCover"]
        pop += period5[0]["precipProbability"]

        # Calculate the maxiumum intensity for the period
        if maxIntensity == 0:
            maxIntensity = period5[0]["precipIntensity"]
        elif maxIntensity > 0 and period5[0]["precipIntensity"] > maxIntensity:
            maxIntensity = period5[0]["precipIntensity"]

        # Calculate the maximum wind speed for the period
        if maxWind == 0:
            maxWind = period5[0]["windSpeed"]
        elif maxWind > 0 and period5[0]["windSpeed"] > maxWind:
            maxWind = period5[0]["windSpeed"]

        # Add the percipitation type to an array to calculate the most common precipitation to use as a baseline
        if period5[0]["precipIntensity"] > 0.02 * prepAccumUnit:
            mostCommonPrecip.append(period5[0]["precipType"])

        # Add the percipitation type to an array of precipitation types if it doesn;t already exist
        if not prepTypes and period5[0]["precipIntensity"] > 0.02 * prepAccumUnit:
            prepTypes.append(period5[0]["precipType"])
        elif (
            period5[0]["precipType"] not in prepTypes
            and period5[0]["precipIntensity"] > 0.02 * prepAccumUnit
        ):
            prepTypes.append(period5[0]["precipType"])

        # Add the data to an array of period arrays to use to calculate the summaries
        periodStats[4].append(numHoursFog)
        periodStats[4].append(numHoursDry)
        periodStats[4].append(numHoursWind)
        periodStats[4].append(round(rainPrep, 4))
        periodStats[4].append(round(snowPrep, 4))
        periodStats[4].append(round(snowError, 4))
        periodStats[4].append(round(sleetPrep, 4))
        periodStats[4].append(round(pop, 2))
        periodStats[4].append(round(maxIntensity, 4))
        periodStats[4].append(round(cloudCover, 2))
        periodStats[4].append(maxWind)
        periodStats[4].append(len(period5))
        periodStats[4].append(today + hourPeriod)
        periodStats[4].append(numHoursHumid)

    # Variables used to calculate the summary
    precip = []
    vis = []
    winds = []
    wind = []
    humid = []
    dry = []
    cloudLevels = []
    periods = [
        periodStats[0][12],
        periodStats[1][12],
        periodStats[2][12],
        periodStats[3][12],
    ]
    # If we have 5 periods append it to the list of periods
    summary_text = cIcon = snowSentence = prepText = dryText = humidText = (
        precipIcon
    ) = None
    period1Calc = []
    period2Calc = []
    period3Calc = []
    period4Calc = []
    period5Calc = []
    snowLowAccum = snowMaxAccum = snowError = avgPop = maxWind = numItems = 0
    starts = []
    period1Level = period2Level = period3Level = period4Level = avgCloud = -1
    secondary = snowText = snowSentence = None
    # Calculate the total ice precipitation
    icePrep = (
        periodStats[0][6] + periodStats[1][6] + periodStats[2][6] + periodStats[3][6]
    )
    # Calculate the total rain precipitation
    rainPrep = (
        periodStats[0][3] + periodStats[1][3] + periodStats[2][3] + periodStats[3][3]
    )
    # Calculate the total snow precipitation
    snowPrep = (
        periodStats[0][4] + periodStats[1][4] + periodStats[2][4] + periodStats[3][4]
    )
    if periodStats[4]:
        periods.append(periodStats[4][12])
        snowPrep += periodStats[4][4]
        rainPrep += periodStats[4][3]
        sleetPrep += periodStats[4][6]
    # Calculate the total precipitaion
    totalPrep = rainPrep + snowPrep + sleetPrep

    # If we have two today-night in the periods change the last one to tomorrow-night to prevent weird summaries
    if "today-night" in periods[0] and "today-night" in periods[len(periods) - 1]:
        periods[len(periods) - 1] = "tomorrow-night"

    # If we are in day mode calculate the current period number to exclude parts of the day from being calculated
    if mode == "day":
        if currPeriod == "morning":
            currPeriodNum = 1 + ((8 - (12 - currHour)) / 8)
        elif currPeriod == "afternoon":
            currPeriodNum = 2 + ((5 - (17 - currHour)) / 5)
        elif currPeriod == "evening":
            currPeriodNum = 3 + ((5 - (22 - currHour)) / 5)
        elif currPeriod == "night":
            currPeriodNum = 4
    else:
        currPeriodNum = 1

    checkPeriod = math.floor(currPeriodNum) - 1

    # If the current period is 3/4 the way through the first period then exclude it.
    if currPeriodNum < 1.75 and period1:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (
            periodStats[0][4] > (2.5 * prepAccumUnit)
            or periodStats[0][3] > (0.25 * prepAccumUnit)
            or periodStats[0][6] > (0.25 * prepAccumUnit)
        ):
            period1Calc.append(True)
            if avgPop == 0:
                avgPop = periodStats[0][7]
            elif periodStats[0][7] > avgPop:
                avgPop = periodStats[0][7]
        else:
            period1Calc.append(None)
        # Calculate the wind text
        if periodStats[0][2] >= (min(periodStats[0][11] / 2, 3)):
            period1Calc.append(
                calculate_wind_text(periodStats[0][10], windUnit, icon, "summary")
            )
        else:
            period1Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[0][8] * prepAccumUnit < 0.02
            and periodStats[0][10] / windUnit < 6.7056
            and periodStats[0][0] >= (min(periodStats[0][11] / 2, 3))
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
        # Calculate the dry text
        if periodStats[0][1] >= (min(periodStats[0][11] / 2, 3)):
            period1Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 0))
        else:
            period1Calc.append(None)
        # Calculate the humid text
        if periodStats[0][13] >= (min(periodStats[0][11] / 2, 3)):
            period1Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 1))
        else:
            period1Calc.append(None)
        # Add the wind speed to the wind array
        winds.append(periodStats[0][10])
        # If there is any precipitation
        if period1Calc[0] is not None:
            # Calcaulte the intensity and add it to the precipitation array
            precip.append(0)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[0][4] > 0:
                snowError += periodStats[0][5]
        # Add the wind to the wind array if the wind text exists
        if period1Calc[1] is not None:
            wind.append(0)
        # Add the visibility to the visibility array if the fog text exists
        if period1Calc[2] is not None:
            vis.append(0)
        # Add to the humid array if the humid text exists
        if period1Calc[5] is not None:
            humid.append(0)
        # Add to the visibility array if the dry text exists
        if period1Calc[4] is not None:
            dry.append(0)
    # If the current period is 3/4 the way through the second period then exclude it.
    if currPeriodNum < 2.75 and period2:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (
            periodStats[1][4] > (2.5 * prepAccumUnit)
            or periodStats[1][3] > (0.25 * prepAccumUnit)
            or periodStats[1][6] > (0.25 * prepAccumUnit)
        ):
            period2Calc.append(True)
            if avgPop == 0:
                avgPop = periodStats[1][7]
            elif periodStats[1][7] > avgPop:
                avgPop = periodStats[1][7]
        else:
            period2Calc.append(None)
        # Calculate the wind text
        if periodStats[1][2] >= (min(periodStats[1][11] / 2, 3)):
            period2Calc.append(
                calculate_wind_text(periodStats[1][10], windUnit, icon, "summary")
            )
        else:
            period2Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[1][8] * prepAccumUnit < 0.02
            and periodStats[1][10] / windUnit < 6.7056
            and periodStats[1][0] >= (min(periodStats[1][11] / 2, 3))
        ):
            period2Calc.append(calculate_vis_text(0, visUnits, "summary"))
        else:
            period2Calc.append(None)
        # Add the current period cloud cover
        period2Calc.append(periodStats[1][9])
        avgCloud += periodStats[1][9]
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period2Text, period2Level = calculate_cloud_text(periodStats[1][9])
        cloudLevels.append(period2Level)
        # Calculate the dry text
        if periodStats[1][1] >= (min(periodStats[1][11] / 2, 3)):
            period2Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 0))
        else:
            period2Calc.append(None)
        # Calculate the humid text
        if periodStats[1][13] >= (min(periodStats[1][11] / 2, 3)):
            period2Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 1))
        else:
            period2Calc.append(None)
        # Add the wind speed to the wind array
        winds.append(periodStats[1][10])
        # If there is any precipitation
        if period2Calc[0] is not None:
            precip.append(1)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[1][4] > 0:
                snowError += periodStats[1][5]
        # Add the wind to the wind array if the wind text exists
        if period2Calc[1] is not None:
            wind.append(1)
        # Add the wind to the visibility array if the fog text exists
        if period2Calc[2] is not None:
            vis.append(1)
        # Add to the humid array if the humid text exists
        if period2Calc[5] is not None:
            humid.append(1)
        # Add to the visibility array if the dry text exists
        if period2Calc[4] is not None:
            dry.append(1)
    # If the current period is 3/4 the way through the third period then exclude it.
    if currPeriodNum < 3.75 and period3:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (
            periodStats[2][4] > (2.5 * prepAccumUnit)
            or periodStats[2][3] > (0.25 * prepAccumUnit)
            or periodStats[2][6] > (0.25 * prepAccumUnit)
        ):
            period3Calc.append(True)
            if avgPop == 0:
                avgPop = periodStats[2][7]
            elif periodStats[2][7] > avgPop:
                avgPop = periodStats[2][7]
        else:
            period3Calc.append(None)
        # Calculate the wind text
        if periodStats[2][2] >= (min(periodStats[2][11] / 2, 3)):
            period3Calc.append(
                calculate_wind_text(periodStats[2][10], windUnit, icon, "summary")
            )
        else:
            period3Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[2][8] * prepAccumUnit < 0.02
            and periodStats[2][10] / windUnit < 6.7056
            and periodStats[2][0] >= (min(periodStats[2][11] / 2, 3))
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
        # Calculate the dry text
        if periodStats[2][1] >= (min(periodStats[2][11] / 2, 3)):
            period3Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 0))
        else:
            period3Calc.append(None)
        # Calculate the humid text
        if periodStats[2][13] >= (min(periodStats[2][11] / 2, 3)):
            period3Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 1))
        else:
            period3Calc.append(None)
        # Add the wind speed to the wind array
        winds.append(periodStats[2][10])
        # If there is any precipitation
        if period3Calc[0] is not None:
            precip.append(2)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[2][4] > 0:
                snowError += periodStats[2][5]
        # Add the wind to the wind array if the wind text exists
        if period3Calc[1] is not None:
            wind.append(2)
        # Add the wind to the visibility array if the fog text exists
        if period3Calc[2] is not None:
            vis.append(2)
        # Add to the humid array if the humid text exists
        if period3Calc[5] is not None:
            humid.append(2)
        # Add to the visibility array if the dry text exists
        if period3Calc[4] is not None:
            dry.append(2)

    if period4:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (
            periodStats[3][4] > (2.5 * prepAccumUnit)
            or periodStats[3][3] > (0.25 * prepAccumUnit)
            or periodStats[3][6] > (0.25 * prepAccumUnit)
        ):
            period4Calc.append(True)
            if avgPop == 0:
                avgPop = periodStats[3][7]
            elif periodStats[3][7] > avgPop:
                avgPop = periodStats[3][7]
        else:
            period4Calc.append(None)
        # Calculate the wind text
        if periodStats[3][2] >= (min(periodStats[3][11] / 2, 3)):
            period4Calc.append(
                calculate_wind_text(periodStats[3][10], windUnit, icon, "summary")
            )
        else:
            period4Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[3][8] * prepAccumUnit < 0.02
            and periodStats[3][10] / windUnit < 6.7056
            and periodStats[3][0] >= (min(periodStats[3][11] / 2, 3))
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
        # Calculate the dry text
        if periodStats[3][1] >= (min(periodStats[3][11] / 2, 3)):
            period4Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 0))
        else:
            period4Calc.append(None)
        # Calculate the humid text
        if periodStats[3][13] >= (min(periodStats[3][11] / 2, 3)):
            period4Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 1))
        else:
            period4Calc.append(None)

        # Add the wind speed to the wind array
        winds.append(periodStats[2][10])
        # If there is any precipitation
        if period4Calc[0] is not None:
            precip.append(3)
            if periodStats[3][4] > 0:
                snowError += periodStats[3][5]
        # Add the wind to the wind array if the wind text exists
        if period4Calc[1] is not None:
            wind.append(3)
        # Add the wind to the visibility array if the fog text exists
        if period4Calc[2] is not None:
            vis.append(3)
        # Add to the humid array if the humid text exists
        if period4Calc[5] is not None:
            humid.append(3)
        # Add to the visibility array if the dry text exists
        if period4Calc[4] is not None:
            dry.append(3)

    if period5:
        # Check if there is enough precipitation to trigger the precipitation icon
        if (
            periodStats[4][4] > (2.5 * prepAccumUnit)
            or periodStats[4][3] > (0.25 * prepAccumUnit)
            or periodStats[4][6] > (0.25 * prepAccumUnit)
        ):
            period5Calc.append(True)
            if avgPop == 0:
                avgPop = periodStats[4][7]
            elif periodStats[4][7] > avgPop:
                avgPop = periodStats[4][7]
        else:
            period5Calc.append(None)
        # Calculate the wind text
        if periodStats[4][2] >= (min(periodStats[4][11] / 2, 3)):
            period5Calc.append(
                calculate_wind_text(periodStats[4][10], windUnit, icon, "summary")
            )
        else:
            period5Calc.append(None)
        # Check if there is no precipitation and the wind is less than the light wind threshold
        if (
            periodStats[4][8] * prepAccumUnit < 0.02
            and periodStats[4][10] / windUnit < 6.7056
            and periodStats[4][0] >= (min(periodStats[4][11] / 2, 3))
        ):
            period5Calc.append(calculate_vis_text(0, visUnits, "summary"))
        else:
            period5Calc.append(None)
        # Add the current period cloud cover
        period5Calc.append(periodStats[4][9])
        avgCloud += periodStats[4][9]
        # Calculate the periods cloud text and level and add it to the cloud levels array
        period3Text, period5Level = calculate_cloud_text(periodStats[4][9])
        cloudLevels.append(period3Level)
        # Calculate the dry text
        if periodStats[4][1] >= (min(periodStats[4][11] / 2, 3)):
            period5Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 0))
        else:
            period5Calc.append(None)
        # Calculate the humid text
        if periodStats[4][13] >= (min(periodStats[4][11] / 2, 3)):
            period5Calc.append(humidity_sky_text(20 * tempUnits, tempUnits, 1))
        else:
            period5Calc.append(None)
        # Add the wind speed to the wind array
        winds.append(periodStats[4][10])
        # If there is any precipitation
        if period5Calc[0] is not None:
            precip.append(4)
            # If the precipitation is snow then add the accumulation and error
            if periodStats[2][4] > 0:
                snowError += periodStats[4][5]
        # Add the wind to the wind array f the wind text exists
        if period5Calc[1] is not None:
            wind.append(4)
        # Add the wind to the visibility array if the fog text exists
        if period5Calc[2] is not None:
            vis.append(2)
        # Add to the humid array if the humid text exists
        if period5Calc[5] is not None:
            humid.append(4)
        # Add to the visibility array if the dry text exists
        if period5Calc[4] is not None:
            dry.append(4)

    # Add the wind, wind, visibility, dry and humid starts to the starts array if they exist
    if precip:
        starts.append(precip[0])
    if wind:
        starts.append(wind[0])
    if vis:
        starts.append(vis[0])
    if dry:
        starts.append(dry[0])
    if humid:
        starts.append(humid[0])

    # If there's any precipitation find the most common one to use as the precipitation type
    if mostCommonPrecip:
        precipType = Most_Common(mostCommonPrecip)

    # If pop is -999 set it to 1 so we can calculate the precipitation text
    if avgPop == -999:
        avgPop = 1

    # Only calculate the precipitation text if there is any possibility of precipitation > 0
    if avgPop > 0 and totalPrep >= (0.1 * prepAccumUnit):
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

    # if secondary is medium none change it to medium-precipitaiton to avoid errors
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

        # If we have 0 error or error is below 0 then use the ceiling of the current precipitation in the summary
        if snowError <= 0:
            snowSentence = [
                "centimeters" if prepAccumUnit == 0.1 else "inches",
                int(math.ceil(snowPrep)),
            ]
        # Check to see if there is any snow accumulation and if so calculate the sentence to use when creating the precipitation summaries
        elif snowMaxAccum > 0:
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
    windPrecip = dryPrecip = humidPrecip = visPrecip = False
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

    # Calculate the current precipitation for the first hour in the block
    currPrecip = calculate_precip_text(
        hours[0]["precipIntensity"],
        prepAccumUnit,
        hours[0]["precipType"],
        "hour",
        hours[0]["precipAccumulation"],
        hours[0]["precipAccumulation"],
        hours[0]["precipAccumulation"],
        hours[0]["precipProbability"],
        icon,
        "summary",
    )
    later = []

    # If we are in the current period
    if starts and periods[min(starts)] == "today-" + currPeriod:
        # If we have precipitation and it starts in the first block
        if precip and precip[0] == 0:
            # If the first hour has no precipitation add the later text
            if currPrecip is None:
                periods[0] = "later-" + periods[0]
                later.append("precip")
        if vis and vis[0] == 0:
            # If we have fog and it starts in the first block and the first hour has no fog
            if (
                calculate_vis_text(hours[0]["visibility"], visUnits) is None
                and "later" not in periods[0]
            ):
                periods[0] = "later-" + periods[0]
                later.append("vis")
        # If we have wind and it starts in the first block and the first hour is not windy
        if wind and wind[0] == 0:
            if (
                calculate_wind_text(hours[0]["windSpeed"], windUnit, icon, "summary")
                is None
                and "later" not in periods[0]
            ):
                periods[0] = "later-" + periods[0]
                later.append("wind")
        # If we have dry conditions and it starts in the first block and the first hour is not dry
        if dry and dry[0] == 0:
            if (
                humidity_sky_text(
                    hours[0]["temperature"], tempUnits, hours[0]["humidity"]
                )
                is None
                and "later" not in periods[0]
            ):
                periods[0] = "later-" + periods[0]
                later.append("dry")
        # If we have humid conditions and it starts in the first block and the first hour is not humid
        if humid and humid[0] == 0:
            if (
                humidity_sky_text(
                    hours[0]["temperature"], tempUnits, hours[0]["humidity"]
                )
                is None
                and "later" not in periods[0]
            ):
                periods[0] = "later-" + periods[0]
                later.append("humid")

    # Calculate the cloud period text
    cloudText, windPrecip, dryPrecip, humidPrecip, visPrecip = calculate_period_text(
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
        dry,
        humid,
        tempUnits,
        dryPrecip,
        humidPrecip,
        visPrecip,
        vis,
        later,
    )

    # If there is only one period
    if not period1Calc and not period2Calc and not period3Calc:
        # If there is precipitation and wind then join with an and
        if period4Calc[0] is not None:
            if period4Calc[1] is not None:
                summary_text = [
                    "sentence",
                    ["during", ["and", prepText, period4Calc[1]], "night"],
                ]
            # Otherwise just use the precipitation
            else:
                return ["sentence", ["during", prepText, "night"]]
        # If there is fog then show that text
        elif period4Calc[2] is not None:
            cIcon = "fog"
            summary_text = ["sentence", ["during", period4Calc[2], "night"]]
        else:
            # If there is wind during the last period then join the wind with the cloud text
            if period4Calc[1] is not None:
                cIcon = calculate_wind_text(maxWind, windUnit, icon, "icon")
                summary_text = [
                    "sentence",
                    ["during", ["and", period4Calc[1], period4Calc[3]], "night"],
                ]
            # If there is low humidity during the last period then join it with the cloud text
            elif period4Calc[4] is not None:
                summary_text = [
                    "sentence",
                    ["during", ["and", period4Calc[3], period4Calc[4]], "night"],
                ]
            # If there is high humidity during the last period then join it with the cloud text
            elif period4Calc[5] is not None:
                summary_text = [
                    "sentence",
                    ["during", ["and", period4Calc[3], period4Calc[5]], "night"],
                ]
            # Otherwise just show the cloud text
            else:
                summary_text = ["sentence", ["during", period4Calc[3], "night"]]
    else:
        windText = None
        visText = None
        precipText = None

        # If there is any precipitation then calcaulate the text
        if len(precip) > 0 and prepText is not None:
            numItems += 1
            precipText, windPrecip, dryPrecip, humidPrecip, visPrecip = (
                calculate_period_text(
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
                    dry,
                    humid,
                    tempUnits,
                    dryPrecip,
                    humidPrecip,
                    visPrecip,
                    vis,
                    later,
                )
            )

        # If there is any visibility then calcaulate the text
        if len(vis) > 0 and numItems <= 1 and not visPrecip:
            numItems += 1
            visText, windPrecip, dryPrecip, humidPrecip, visPrecip = (
                calculate_period_text(
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
                    dry,
                    humid,
                    tempUnits,
                    dryPrecip,
                    humidPrecip,
                    visPrecip,
                    vis,
                    later,
                )
            )

        # If there is any wind then calcaulate the text if its not joined with precip/cloud
        if not windPrecip and numItems <= 1 and len(wind) > 0:
            numItems += 1
            windText, windPrecip, dryPrecip, humidPrecip, visPrecip = (
                calculate_period_text(
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
                    dry,
                    humid,
                    tempUnits,
                    dryPrecip,
                    humidPrecip,
                    visPrecip,
                    vis,
                    later,
                )
            )
        # If there is any low humidity then calcaulate the text if its not joined with any conditions
        if not dryPrecip and len(dry) > 0:
            dryText, windPrecip, dryPrecip, humidPrecip, visPrecip = (
                calculate_period_text(
                    periods,
                    dry,
                    "low-humidity",
                    "dry",
                    wind,
                    prepAccumUnit,
                    visUnits,
                    windUnit,
                    maxWind,
                    windPrecip,
                    checkPeriod,
                    mode,
                    icon,
                    dry,
                    humid,
                    tempUnits,
                    dryPrecip,
                    humidPrecip,
                    visPrecip,
                    vis,
                    later,
                )
            )

        # If there is any low humidity then calcaulate the text if its not joined with any conditions
        if not humidPrecip and len(humid) > 0:
            humidText, windPrecip, dryPrecip, humidPrecip, visPrecip = (
                calculate_period_text(
                    periods,
                    humid,
                    "high-humidity",
                    "humid",
                    wind,
                    prepAccumUnit,
                    visUnits,
                    windUnit,
                    maxWind,
                    windPrecip,
                    checkPeriod,
                    mode,
                    icon,
                    dry,
                    humid,
                    tempUnits,
                    dryPrecip,
                    humidPrecip,
                    visPrecip,
                    vis,
                    later,
                )
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
                        # If there is any dry text then join with an and and show whichever one comes fist at the start
                        if dryText is not None:
                            if dry[0] == min(starts) or (
                                len(dry) == len(periods)
                                and len(mostCommonLevels) != len(periods)
                            ):
                                summary_text = ["sentence", ["and", dryText, cloudText]]
                            else:
                                summary_text = ["sentence", ["and", cloudText, dryText]]
                        # If there is any humid text then join with an and and show whichever one comes fist at the start
                        elif humidText is not None:
                            if humid[0] == min(starts) or (
                                len(humid) == len(periods)
                                and len(mostCommonLevels) != len(periods)
                            ):
                                summary_text = [
                                    "sentence",
                                    ["and", humidText, cloudText],
                                ]
                            else:
                                summary_text = [
                                    "sentence",
                                    ["and", cloudText, humidText],
                                ]
                        else:
                            summary_text = ["sentence", cloudText]
                # If there is wind text and visbility text then join with an and and show whichever one comes first at the start
                elif visText is not None:
                    if vis[0] == min(starts) or (
                        len(vis) == len(periods) and len(wind) != len(periods)
                    ):
                        cIcon = "fog"
                        summary_text = ["sentence", ["and", visText, windText]]
                    else:
                        cIcon = calculate_wind_text(maxWind, windUnit, icon, "icon")
                        summary_text = ["sentence", ["and", windText, visText]]
                # If there is wind text
                else:
                    cIcon = calculate_wind_text(maxWind, windUnit, icon, "icon")
                    # If there is any dry text then join with an and and show whichever one comes fist at the start
                    if dryText is not None:
                        if dry[0] == min(starts) or (
                            len(dry) == len(periods) and len(wind) != len(periods)
                        ):
                            summary_text = ["sentence", ["and", dryText, windText]]
                        else:
                            summary_text = ["sentence", ["and", windText, dryText]]
                    # If there is any humid text then join with an and and show whichever one comes fist at the start
                    elif humidText is not None:
                        if humid[0] == min(starts) or (
                            len(humid) == len(periods) and len(wind) != len(periods)
                        ):
                            summary_text = ["sentence", ["and", humidText, windText]]
                        else:
                            summary_text = ["sentence", ["and", windText, humidText]]
                    else:
                        summary_text = ["sentence", windText]
            # If there is precipitation
            else:
                # If there is any visibility text then join with an and and show whichever one comes fist at the start
                if visText is not None:
                    if (
                        vis[0] == min(starts)
                        and (
                            precip[0] != min(starts)
                            or (totalPrep * len(precip)) < 0.25 * prepAccumUnit
                        )
                    ) or (len(vis) == len(periods) and len(precip) != len(periods)):
                        cIcon = "fog"
                        summary_text = ["sentence", ["and", visText, precipText]]
                    else:
                        summary_text = ["sentence", ["and", precipText, visText]]
                # If there is any wind text then join with an and and show whichever one comes fist at the start
                elif windText is not None:
                    if (
                        wind[0] == min(starts)
                        and (
                            precip[0] != min(starts)
                            or (totalPrep * len(precip)) < 0.25 * prepAccumUnit
                        )
                    ) or (len(wind) == len(periods) and len(precip) != len(periods)):
                        cIcon = calculate_wind_text(maxWind, windUnit, icon, "icon")
                        summary_text = ["sentence", ["and", windText, precipText]]
                    else:
                        summary_text = ["sentence", ["and", precipText, windText]]
                else:
                    # If there is any humid text then join with an and and show whichever one comes fist at the start
                    if humidText is not None:
                        if humid[0] == min(starts) or (
                            len(humid) == len(periods) and len(precip) != len(periods)
                        ):
                            summary_text = ["sentence", ["and", humidText, precipText]]
                        else:
                            summary_text = ["sentence", ["and", precipText, humidText]]
                    else:
                        summary_text = ["sentence", precipText]

    # If there is no icon then calculate it based on the average cloud cover for the periods if we don't have any precipitation
    if cIcon is None:
        if precipIcon is not None:
            cIcon = precipIcon
        else:
            cIcon = calculate_sky_icon(avgCloud, True, icon)

    return cIcon, summary_text


def nextPeriod(currPeriod):
    """
    Calculates the current day/next 24h text

    Parameters:
    - currPeriod (str) - The current textual representation of the period.

    Returns:
    - nextPeriod (str) - The next textual representation of the period.
    """
    if currPeriod == "morning":
        nextPeriod = "afternoon"
    elif currPeriod == "afternoon":
        nextPeriod = "evening"
    elif currPeriod == "evening":
        nextPeriod = "night"
    elif currPeriod == "night":
        nextPeriod = "morning"

    return nextPeriod
