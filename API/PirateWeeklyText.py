# %% Script to contain the functions that can be used to generate the weekly text summary of the forecast data for Pirate Weather

import datetime
from itertools import groupby

import numpy as np
from dateutil import tz

from API.constants.api_const import PRECIP_TYPES
from API.constants.shared_const import MISSING_DATA
from API.constants.text_const import (
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    DEFAULT_POP,
    PRECIP_PROB_THRESHOLD,
)
from API.PirateTextHelper import (
    PRECIPITATION_ICONS,
    calculate_precip_text,
    calculate_thunderstorm_text,
    most_common,
)

WEEK_DAYS_MINUS_ONE = 6
WEEK_DAYS = 7
WEEK_DAYS_PLUS_ONE = 8
MIN_THUNDERSTORM_DAYS = 2
# Highs that all fall within this range (°C) are "steady around" their mean
# rather than rising, falling, peaking or valleying.
STEADY_TEMPERATURE_RANGE_C = 2


def _combined_precip_type(precipitation_days):
    """
    The precipitation type for a set of days, or "mixed-precipitation" if they differ.

    Days without a known type ("none") don't make the week mixed; they can be
    included because their daily icon shows precipitation.

    Parameters:
    - precipitation_days (arr): Entries of the precipitation days array ([index, day name, day])

    Returns:
    - str: The shared precipitation type, "none", or "mixed-precipitation".
    """
    known_types = {
        day[2]["precipType"]
        for day in precipitation_days
        if day[2]["precipType"] != PRECIP_TYPES["none"]
    }
    if not known_types:
        return PRECIP_TYPES["none"]
    if len(known_types) == 1:
        return known_types.pop()
    return "mixed-precipitation"


def _day_run_items(run):
    """
    Describes a run of consecutive precipitation days.

    Parameters:
    - run (arr): Consecutive entries of the precipitation days array ([index, day name, day])

    Returns:
    - list: ["through", first, last] for three or more days, otherwise the
      individual day names.
    """
    if len(run) > 2:
        return [["through", run[0][1], run[-1][1]]]
    return [day[1] for day in run]


def calculate_summary_text(
    precipitation,
    rain_accum,
    snow_accum,
    ice_accum,
    icon,
    max_rain_intensity,
    max_snow_intensity,
    max_ice_intensity,
    max_pop,
):
    """
    Calculates the precipitation summary if there are between 1 and 8 days of precipitation.
    Intensities are expected in SI units (mm/h).

    Parameters:
    - precipitation (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - rain_accum (float): The total rain accumulation for the week in mm
    - snow_accum (float): The total snow accumulation for the week in mm
    - ice_accum (float): The total ice accumulation for the week in mm
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    - max_rain_intensity (float): The maximum precipitation intensity for the week in mm/h
    - max_snow_intensity (float): The maximum snow intensity for the week in mm/h
    - max_ice_intensity (float): The maximum ice intensity for the week in mm/h
    - max_pop (float): The highest probability of precipitation on the precipitation days

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - wIcon (str): The textual summary of conditions (Drizzle, Sleet, etc.).
    - wWeekend (bool): If the summary includes over-weekend or not
    - c_icon (str): The icon representing the conditions for the week.
    """

    wIcon = wText = c_icon = thuText = None
    wWeekend = False
    dayIndexes = []
    days = []
    maxCape = MISSING_DATA
    minLi = None
    maxCin = None
    maxKi = None
    max_vvel = None
    dewpointAtMaxInstability = None
    tempAtMaxInstability = None
    numThunderstormDays = 0

    # Loop through each index in the precipitation array
    for day in precipitation:
        # Create an array of day indexes
        dayIndexes.append(day[0])

        if "cape" in day[2]:
            # Calculate the maximum cape for the week
            # Track maximum CAPE across precipitation days
            if np.isnan(maxCape):
                if not np.isnan(day[2]["cape"]):
                    maxCape = day[2]["cape"]
                    tempAtMaxInstability = day[2].get("temperatureHigh")
                    dewpointAtMaxInstability = day[2].get("dewPoint")
            elif not np.isnan(day[2]["cape"]) and day[2]["cape"] > maxCape:
                maxCape = day[2]["cape"]
                tempAtMaxInstability = day[2].get("temperatureHigh")
                dewpointAtMaxInstability = day[2].get("dewPoint")

        # Most-unstable (minimum) Lifted Index across precipitation days
        day_li = day[2].get("liftedIndexMin")
        if (
            day_li is not None
            and not np.isnan(day_li)
            and (minLi is None or day_li < minLi)
        ):
            minLi = day_li

        # Least-inhibiting (maximum/least-negative) CIN across precipitation days
        day_cin = day[2].get("convectiveInhibitionMax")
        if (
            day_cin is not None
            and not np.isnan(day_cin)
            and (maxCin is None or day_cin > maxCin)
        ):
            maxCin = day_cin

        # Higest vertical velocity for the week
        day_vvel = day[2].get("verticalVelocityMin")
        if (
            day_vvel is not None
            and not np.isnan(day_vvel)
            and (max_vvel is None or day_vvel < max_vvel)
        ):
            max_vvel = day_vvel

        # Maximum K Index across precipitation days
        day_ki = day[2].get("kIndexMax")
        if (
            day_ki is not None
            and not np.isnan(day_ki)
            and (maxKi is None or day_ki > maxKi)
        ):
            maxKi = day_ki

        # Calculate the number of days with thunderstorms forecasted
        if day[2]["icon"] == "thunderstorm":
            numThunderstormDays += 1

    wIcon = _combined_precip_type(precipitation)

    # Create a list of consecutive days so we can use the through text instead of multiple ands
    for _, g in groupby(enumerate(dayIndexes), lambda ix: ix[0] - ix[1]):
        days.append([day_index for _, day_index in g])

    # If the icon is not mixed precipitation change it to translations format
    if wIcon != "mixed-precipitation":
        wIcon, c_icon = calculate_precip_text(
            wIcon,
            "week",
            rain_accum,
            snow_accum,
            ice_accum,
            max_pop,
            icon,
            "both",
            eff_rain_intensity=max_rain_intensity,
            eff_snow_intensity=max_snow_intensity,
            eff_ice_intensity=max_ice_intensity,
        )
    else:
        c_icon = "mixed"

    # If there are any days with thunderstorms occurring then calculate the text
    if numThunderstormDays > 0:
        thuText = calculate_thunderstorm_text(
            maxCape,
            mode="summary",
            lifted_index=minLi,
            cin=maxCin,
            vertical_velocity=max_vvel,
            k_index=maxKi,
            dewpoint=dewpointAtMaxInstability,
            temperature=tempAtMaxInstability,
        )

    # If more than half the days with precipitation show thurnderstorms then set the icon to thunderstorm and add it in front of the precipitation text
    if thuText is not None and numThunderstormDays >= (
        len(precipitation) / MIN_THUNDERSTORM_DAYS
    ):
        c_icon = "thunderstorm"
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
        # Describe each run of consecutive precipitation days. Two items are
        # joined with "and"; three or more use "list" rather than nested "and"s
        # ("today through Sunday, Tuesday and Thursday").
        day_items = []
        run_start = 0
        for run in days:
            day_items.extend(
                _day_run_items(precipitation[run_start : run_start + len(run)])
            )
            run_start += len(run)

        if len(day_items) == 2:
            wText = ["and", day_items[0], day_items[1]]
        else:
            wText = ["list", *day_items]

    return wIcon, wText, wWeekend, c_icon


def calculate_precip_summary(
    precipitation,
    precipitationDays,
    icons,
    rain_accum,
    snow_accum,
    ice_accum,
    max_pop,
    max_rain_intensity,
    max_snow_intensity,
    max_ice_intensity,
    icon="darksky",
):
    """
    Calculates the weekly precipitation summary.
    Intensities are expected in SI units (mm/h).

    Parameters:
    - precipitation (bool): If precipitation is occuring during the week or not
    - precipitationDays (arr): An array of arrays that contain the days with precipitation if there are any. The inner array contains: The index in the week array, the day of the week and the precipitation type
    - icons (arr): An array of the daily icons
    - rain_accum (float): The total rain accumulation for the week in mm
    - snow_accum (float): The total snow accumulation for the week in mm
    - ice_accum (float): The total ice accumulation for the week in mm
    - max_pop (float): The highest probability of precipitation on the precipitation days
    - max_rain_intensity (float): The maximum rain intensity for the week in mm/h
    - max_snow_intensity (float): The maximum snow intensity for the week in mm/h
    - max_ice_intensity (float): The maximum ice intensity for the week in mm/h
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - precipSummary (arr): A summary of the precipitation for the week.
    - c_icon (str): An string representing the conditions for the week.
    """

    c_icon = None

    if not precipitation:
        # If there has been no precipitation use the no precipitation text.
        precipSummary = ["for-week", "no-precipitation"]
        # If there has been any fog forecast during the week show that icon, then the wind icon otherwise use the most common icon during the week

        if "fog" in icons:
            c_icon = "fog"
        elif "dangerous-wind" in icons:
            c_icon = "dangerous-windy"
        elif "wind" in icons:
            c_icon = "wind"
        elif "breezy" in icons:
            c_icon = "breezy"
        else:
            c_icon = most_common(icons)
    elif len(precipitationDays) == 1:
        # If one day has any precipitation then set the icon to the precipitation type and use the medium precipitation text in the precipitation summary
        text, c_icon = calculate_precip_text(
            precipitationDays[0][2]["precipType"],
            "week",
            rain_accum,
            snow_accum,
            ice_accum,
            max_pop,
            icon,
            "both",
            isDayTime=True,
            eff_rain_intensity=max_rain_intensity,
            eff_snow_intensity=max_snow_intensity,
            eff_ice_intensity=max_ice_intensity,
        )
        precipSummary = [
            "during",
            text,
            precipitationDays[0][1],
        ]
    elif 1 < len(precipitationDays) < 8:
        # If between 1 and 8 days have precipitation call the function to calculate the summary text using the precipitation array.
        wIcon, wSummary, wWeekend, c_icon = calculate_summary_text(
            precipitationDays,
            rain_accum,
            snow_accum,
            ice_accum,
            icon,
            max_rain_intensity,
            max_snow_intensity,
            max_ice_intensity,
            max_pop,
        )

        # Check if the summary has the over-weekend text
        if not wWeekend:
            precipSummary = ["during", wIcon, wSummary]
        else:
            precipSummary = [wSummary, wIcon]
    elif len(precipitationDays) == 8:
        week_precip_type = _combined_precip_type(precipitationDays)
        if week_precip_type != "mixed-precipitation":
            # If all days have precipitation then if they all have the same type then use that icon
            text, c_icon = calculate_precip_text(
                week_precip_type,
                "week",
                rain_accum,
                snow_accum,
                ice_accum,
                max_pop,
                icon,
                "both",
                isDayTime=True,
                eff_rain_intensity=max_rain_intensity,
                eff_snow_intensity=max_snow_intensity,
                eff_ice_intensity=max_ice_intensity,
            )
            # Since precipitation is occuring everyday use the for week text instead of through
            precipSummary = [
                "for-week",
                text,
            ]
        else:
            # If the types are not the same then set the icon to mixed and use the mixed precipitation text (as is how Dark Sky did it)
            c_icon = "mixed"
            precipSummary = ["for-week", "mixed-precipitation"]

    return precipSummary, c_icon


def calculate_temp_summary(highTemp, lowTemp, weekArr, unitSystem="si"):
    """
    Calculates the temperature summary for the week.
    Temperatures in highTemp and lowTemp are in Celsius (SI units).
    Display values are converted based on unitSystem.

    Parameters:
    - highTemp (arr): An array that contains [index, day_name, temperature_celsius].
    - lowTemp (arr): An array that contains [index, day_name, temperature_celsius].
    - weekArr (arr): The array of week data
    - unitSystem (str): Unit system for display ("us", "si", "ca", "uk")

    Returns:
    - arr: A summary of the temperature for the week.
    """

    # Determine unit string and convert temperature if needed
    if unitSystem == "us":
        temp_unit_str = "fahrenheit"
        # Convert Celsius to Fahrenheit for display
        high_temp_display = round(highTemp[2] * 9 / 5 + 32)
        low_temp_display = round(lowTemp[2] * 9 / 5 + 32)
    else:  # si, ca, uk all use Celsius
        temp_unit_str = "celsius"
        high_temp_display = round(highTemp[2])
        low_temp_display = round(lowTemp[2])

    # If the highs hardly change over the week, say they are steady around
    # their mean. highTemp and lowTemp are the warmest and coolest highs.
    if highTemp[2] - lowTemp[2] <= STEADY_TEMPERATURE_RANGE_C:
        mean_high = np.nanmean([day["temperatureHigh"] for day in weekArr])
        if unitSystem == "us":
            mean_high = mean_high * 9 / 5 + 32
        return ["temperatures-steady", [temp_unit_str, round(mean_high)]]

    # If the temperature is increasing everyday or if the lowest temperatue is at the start of the week and the highest temperature is at the end of the week use the rising text
    if all(
        weekArr[i]["temperatureHigh"] < weekArr[i + 1]["temperatureHigh"]
        for i in range(WEEK_DAYS)
    ) or (highTemp[0] >= WEEK_DAYS_MINUS_ONE and lowTemp[0] <= 1):
        # Set the temperature summary
        return [
            "temperatures-rising",
            [temp_unit_str, high_temp_display],
            highTemp[1],
        ]
    # If the temperature is decreasing everyday or if the lowest temperatue is at the end of the week and the highest temperature is at the start of the week use the rising text
    elif all(
        weekArr[i]["temperatureHigh"] > weekArr[i + 1]["temperatureHigh"]
        for i in range(WEEK_DAYS)
    ) or (highTemp[0] <= 1 and lowTemp[0] >= WEEK_DAYS_MINUS_ONE):
        return [
            "temperatures-falling",
            [temp_unit_str, low_temp_display],
            lowTemp[1],
        ]
    # If the lowest temperatue is in the middle of the week and the highest temperature is at the start or end of the week use the valleying text
    elif (highTemp[0] <= 1 or highTemp[0] >= WEEK_DAYS_MINUS_ONE) and 0 < lowTemp[
        0
    ] < WEEK_DAYS:
        return [
            "temperatures-valleying",
            [temp_unit_str, low_temp_display],
            lowTemp[1],
        ]
    else:
        # Otherwise use the peaking text
        return [
            "temperatures-peaking",
            [temp_unit_str, high_temp_display],
            highTemp[1],
        ]


def calculate_weekly_text(weekArr, timeZone, unitSystem="si", icon="darksky"):
    """
    Calculates the weekly summary given an array of weekdays.
    All inputs are expected in SI units (mm/h for intensity, Celsius for temperature).
    Display values are converted based on unitSystem.

    Parameters:
    - weekArr (arr): An array of the weekdays (in SI units)
    - timeZone (string): The timezone for the current location
    - unitSystem (str): Unit system for display ("us", "si", "ca", "uk")
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather

    Returns:
    - c_text (arr): The precipitation and temperature summary for the week.
    - c_icon (str): The icon representing the conditions for the week.
    """

    # Variables to use in calculating the weekly summary
    c_icon = None
    c_text = None
    precipitation = False
    precipitationDays = []
    precipSummary = ""
    highTemp = []
    lowTemp = []
    icons = []
    tempSummary = ""
    max_pop = max_rain_intensity = max_snow_intensity = max_ice_intensity = 0
    zone = tz.gettz(timeZone)

    # Loop through the week array
    for idx, day in enumerate(weekArr):
        # Add the daily icon to the list of weekly icons
        icons.append(day["icon"])
        # Determine the day of the week based on the epoch timestamp and the locations timezone
        dayDate = datetime.datetime.fromtimestamp(day["time"], zone)
        weekday = dayDate.strftime("%A").lower()

        # Correct missing data for precipProbability
        if np.isnan(day["precipProbability"]):
            day["precipProbability"] = DEFAULT_POP

        # First index is always today, second index is always tomorrow and the last index has the next- text at the start
        if idx == 0:
            weekday = "today"
        elif idx == 1:
            weekday = "tomorrow"
        elif idx == len(weekArr) - 1:
            weekday = "next-" + weekday

        # A day has precipitation if its daily summary chose a precipitation
        # icon, or if it reaches the daily accumulation thresholds. The daily
        # summary uses lower, per-period thresholds, so without the icon check a
        # day showing rain (e.g. 0.8 mm) could be left out of the week.
        # Data is already in SI units (mm for accumulation, mm/h for intensity)
        is_likely = day["precipProbability"] >= PRECIP_PROB_THRESHOLD or np.isnan(
            day["precipProbability"]
        )
        if day["icon"] in PRECIPITATION_ICONS or (
            is_likely
            and (
                (
                    day["precipType"] == PRECIP_TYPES["snow"]
                    and day["snowAccumulation"] >= DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM
                )
                or (
                    day["precipType"] == PRECIP_TYPES["rain"]
                    and day["liquidAccumulation"]
                    >= DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM
                )
                or (
                    day["precipType"] in (PRECIP_TYPES["sleet"], PRECIP_TYPES["ice"])
                    and day["iceAccumulation"] >= DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM
                )
            )
        ):
            # Sets that there has been precipitation during the week
            precipitation = True
            precipitationDays.append([idx, weekday, day])
            max_pop = max(max_pop, day["precipProbability"])
            if max_rain_intensity == 0 or (
                day.get("rainIntensityMax", day.get("liquidIntensityMax", 0))
                > max_rain_intensity
            ):
                max_rain_intensity = day.get(
                    "rainIntensityMax", day.get("liquidIntensityMax", 0)
                )
            if max_snow_intensity == 0 or day["snowIntensityMax"] > max_snow_intensity:
                max_snow_intensity = day["snowIntensityMax"]
            if max_ice_intensity == 0 or day["iceIntensityMax"] > max_ice_intensity:
                max_ice_intensity = day["iceIntensityMax"]

        # Determine the highest temperature of the week and record the index in the array, the day it occured on, and the temperature (in Celsius)
        if not highTemp or day["temperatureHigh"] > highTemp[2]:
            highTemp = [idx, weekday, day["temperatureHigh"]]

        # Determine the lowest temperature of the week and record the index in the array, the day it occured on, and the temperature (in Celsius)
        if not lowTemp or day["temperatureHigh"] < lowTemp[2]:
            lowTemp = [idx, weekday, day["temperatureHigh"]]

    precipSummary, c_icon = calculate_precip_summary(
        precipitation,
        precipitationDays,
        icons,
        sum(day[2]["liquidAccumulation"] for day in precipitationDays),
        sum(day[2]["snowAccumulation"] for day in precipitationDays),
        sum(day[2]["iceAccumulation"] for day in precipitationDays),
        max_pop,
        max_rain_intensity,
        max_snow_intensity,
        max_ice_intensity,
        icon,
    )

    # If the icon is None then set the icon
    if c_icon is None:
        if "fog" in icons:
            c_icon = "fog"
        elif "smoke" in icons:
            c_icon = "smoke"
        elif "mist" in icons:
            c_icon = "mist"
        elif "haze" in icons:
            c_icon = "haze"
        elif "dangerous-wind" in icons:
            c_icon = "dangerous-windy"
        elif "wind" in icons:
            c_icon = "wind"
        elif "breezy" in icons:
            c_icon = "breezy"
        else:
            c_icon = most_common(icons)

    # If the none text exists in the precipitation summary then change it to not available
    if None in precipSummary:
        precipSummary = ["for-week", "unavailable"]

    # Only calculate the temperature summary if we have eight days to prevent issues with the time machine
    if len(weekArr) == WEEK_DAYS_PLUS_ONE:
        tempSummary = calculate_temp_summary(highTemp, lowTemp, weekArr, unitSystem)
        # Combine the two texts together using with
        c_text = ["with", precipSummary, tempSummary]
    else:
        # If there is no precipitation show the no precipitation text instead of no precipitation for the week.
        if len(precipitationDays) == 0:
            precipSummary = ["no-precipitation"]
        c_text = precipSummary

    # If we somehow have a generic precipitation icon we use rain instead
    if c_icon == "precipitation":
        c_icon = PRECIP_TYPES["rain"]
    elif c_icon == PRECIP_TYPES["mixed"] and icon != "pirate":
        c_icon = PRECIP_TYPES["sleet"]

    return c_text, c_icon
