import datetime
import math
from dateutil import tz
from PirateTextHelper import (
    calculate_precip_text,
    calculate_wind_text,
    calculate_vis_text,
    calculate_sky_icon,
    humidity_sky_text,
    Most_Common,
)

# Constants for cloud cover thresholds
CLOUDY_THRESHOLD = 0.875
MOSTLY_CLOUDY_THRESHOLD = 0.625
PARTLY_CLOUDY_THRESHOLD = 0.375
MOSTLY_CLEAR_THRESHOLD = 0.125
DEFAULT_VISIBILITY = 1000


def calculate_cloud_text(cloud_cover):
    """
    Calculates the textual representation and level of cloud cover.

    Parameters:
    - cloud_cover (float): The cloud cover for the period (0.0 to 1.0).

    Returns:
    - tuple: A tuple containing:
        - cloud_text (str): The textual representation of the cloud cover.
        - cloud_level (int): The level of the cloud cover (0-4).
    """
    if cloud_cover > CLOUDY_THRESHOLD:
        return "heavy-clouds", 4
    elif cloud_cover > MOSTLY_CLOUDY_THRESHOLD:
        return "medium-clouds", 3
    elif cloud_cover > PARTLY_CLOUDY_THRESHOLD:
        return "light-clouds", 2
    elif cloud_cover > MOSTLY_CLEAR_THRESHOLD:
        return "very-light-clouds", 1
    else:
        return "clear", 0


def _get_period_name(hour_of_day, is_today=True):
    """
    Determines the textual name of a period based on the hour of the day.

    Parameters:
    - hour_of_day (int): The hour (0-23).
    - is_today (bool): True if the period is for the current day, False for the next.

    Returns:
    - str: The textual representation of the period (e.g., "today-morning", "tomorrow-night").
    """
    prefix = "today-" if is_today else "tomorrow-"
    if 4 <= hour_of_day < 12:
        return prefix + "morning"
    elif 12 <= hour_of_day < 17:
        return prefix + "afternoon"
    elif 17 <= hour_of_day < 22:
        return prefix + "evening"
    else:  # 22:00 to 3:59
        return prefix + "night"


def _get_time_phrase(
    period_indices,
    condition_type,
    all_periods,
    check_period,
    mode,
    later_conditions,
    today_period_for_later_check,
):
    """
    Determines the appropriate time phrase (e.g., "during", "starting", "until")
    for a given set of periods, considering continuity and specific patterns.

    Parameters:
    - period_indices (list): List of indices where the condition is present.
    - condition_type (str): The type of condition ("precip", "cloud", etc.).
    - all_periods (list): List of all period names (e.g., ["today-morning"]).
    - check_period (int): The starting index for checks (usually 0).
    - mode (str): "daily" or "hour".
    - later_conditions (list): List of conditions marked as "later".
    - today_period_for_later_check (str): The name of the current period for 'later' comparison.

    Returns:
    - list: The time phrase structure.
    """
    num_periods = len(period_indices)
    total_periods_available = len(all_periods)

    if num_periods == 0:
        return None

    if num_periods == 1:
        # Single period: "during [period_name]"
        return ["during", all_periods[period_indices[0]]]
    elif num_periods == total_periods_available:
        # Condition spans all available periods: "for-day" or "starting [first_period]" if 'later' applies
        # The 'later' text only applies if in hourly mode and the first period is marked as 'later'.
        if (
            mode == "hour"
            and "later" in all_periods[0]
            and condition_type in later_conditions
        ):
            return ["starting", all_periods[period_indices[0]]]
        else:
            return ["for-day"]

    # Logic for multiple disjoint or continuous periods
    if num_periods > 1:
        start_idx = period_indices[0]
        end_idx = period_indices[-1]

        # Check if the periods are continuous (e.g., [0, 1, 2])
        is_continuous = all(
            period_indices[i] == period_indices[i - 1] + 1
            for i in range(1, num_periods)
        )

        # Handle specific patterns for 2 periods
        if num_periods == 2:
            # Starts in the 3rd period and continues to the 4th (total 4 periods)
            if (
                start_idx == check_period + 2
                and end_idx == 3
                and total_periods_available == 4
            ):
                summary_text_temp = ["starting", all_periods[start_idx]]
            # Starts in the 4th period and continues to the 5th (total 5 periods)
            elif (
                start_idx == check_period + 3
                and end_idx == 4
                and total_periods_available == 5
            ):
                summary_text_temp = ["starting", all_periods[start_idx]]
            # Starts at 'check_period' and is continuous for 2 periods
            elif start_idx == check_period and is_continuous:
                summary_text_temp = [  # Store temp to check later condition
                    "until",
                    all_periods[min(end_idx + 1, total_periods_available - 1)],
                ]
            # Starts after 'check_period' and is continuous for 2 periods
            elif start_idx > check_period and is_continuous:
                summary_text_temp = [  # Store temp to check later condition
                    "starting-continuing-until",
                    all_periods[start_idx],
                    all_periods[min(end_idx + 1, total_periods_available - 1)],
                ]
            # Starts at 'check_period', not continuous, and ends in the 4th period (index 3)
            # This covers patterns like [0, 3] in a 4-period day
            elif start_idx == check_period and not is_continuous and end_idx == 3:
                summary_text_temp = [  # Store temp to check later condition
                    "until-starting-again",
                    all_periods[start_idx + 1],  # The period after the first occurrence
                    all_periods[end_idx],  # The last period of occurrence
                ]
            else:
                # Two disjoint periods, e.g., ["during", ["and", period1, period2]]
                summary_text_temp = [  # Store temp to check later condition
                    "during",
                    ["and", all_periods[start_idx], all_periods[end_idx]],
                ]

            # Specific "later" check for 2-period conditions
            if (
                mode == "hour"  # 'later' only applies in hourly mode
                and "later"
                in all_periods[0]  # Check if the first period is indeed "later-"
                and condition_type in later_conditions
            ):
                # If the current summary_text_temp is an "until" pattern and not explicitly "starting"
                # (which would be overridden by "later")
                if "until" in summary_text_temp and "starting" not in summary_text_temp:
                    return [
                        "starting",
                        all_periods[start_idx],
                    ]  # Simplify to "starting"
                # If it's an "until-starting-again" pattern
                elif "until-starting-again" in summary_text_temp:
                    # Original logic combined into "and" for the two parts if 'later' applied.
                    return [
                        "and",
                        ["during", all_periods[start_idx]],
                        ["during", all_periods[end_idx]],
                    ]
            return summary_text_temp  # Return the determined summary phrase if 'later' didn't apply

        # Handle specific patterns for 3 periods
        elif num_periods == 3:
            mid_idx = period_indices[1]
            # Starts in the 2nd period and continuous for 3 periods (total 4 periods)
            if (
                start_idx == check_period + 1
                and end_idx == 3
                and total_periods_available == 4
            ):
                summary_text_temp = ["starting", all_periods[start_idx]]
            # Starts in the 3rd period and continuous for 3 periods (total 5 periods)
            elif (
                start_idx == check_period + 2
                and end_idx == 4
                and total_periods_available == 5
            ):
                summary_text_temp = ["starting", all_periods[start_idx]]
            # Continuous block of 3 periods starting at 'check_period'
            elif start_idx == check_period and is_continuous:
                summary_text_temp = [
                    "until",
                    all_periods[min(end_idx + 1, total_periods_available - 1)],
                ]
            # Discontinuous 3 periods
            elif not is_continuous:
                # All three periods are disjoint (e.g., [0, 2, 4])
                if (mid_idx - start_idx) != 1 and (end_idx - mid_idx) != 1:
                    summary_text_temp = [
                        "during",
                        [
                            "and",
                            all_periods[start_idx],
                            ["and", all_periods[mid_idx], all_periods[end_idx]],
                        ],
                    ]
                # First two are continuous, third is disjoint (e.g., [0, 1, 3])
                elif (
                    start_idx == check_period
                    and (mid_idx - start_idx) == 1
                    and end_idx >= 3
                ):
                    summary_text_temp = [
                        "until-starting-again",
                        all_periods[mid_idx + 1],
                        all_periods[end_idx],
                    ]
                # First is disjoint, last two are continuous (e.g., [0, 2, 3])
                elif (
                    start_idx == check_period
                    and (mid_idx - start_idx) != 1
                    and mid_idx >= 2
                ):
                    summary_text_temp = [
                        "until-starting-again",
                        all_periods[start_idx + 1],
                        all_periods[mid_idx],
                    ]
                # Three continuous periods starting after 'check_period' (for 5 total periods)
                elif (
                    start_idx > check_period
                    and is_continuous
                    and total_periods_available == 5
                ):
                    summary_text_temp = [
                        "starting-continuing-until",
                        all_periods[start_idx],
                        all_periods[min(end_idx + 1, total_periods_available - 1)],
                    ]
                # First is disjoint, next two are continuous (for 5 total periods)
                elif (
                    start_idx > check_period
                    and (mid_idx - start_idx) != 1
                    and (end_idx - mid_idx) == 1
                    and total_periods_available == 5
                ):
                    summary_text_temp = [
                        "and",
                        ["during", all_periods[start_idx]],
                        ["starting", all_periods[mid_idx]],
                    ]
                # First two are continuous, last is disjoint (for 5 total periods)
                elif (
                    start_idx > check_period
                    and (mid_idx - start_idx) == 1
                    and (end_idx - mid_idx) != 1
                    and total_periods_available == 5
                ):
                    summary_text_temp = [
                        "and",
                        [
                            "starting-continuing-until",
                            all_periods[start_idx],
                            all_periods[min(mid_idx + 1, total_periods_available - 1)],
                        ],
                        ["during", all_periods[end_idx]],
                    ]

            # Apply "later" re-structuring specific to 3-period patterns
            if (
                mode == "hour"  # 'later' only applies in hourly mode
                and "later"
                in all_periods[0]  # Check if the first period is indeed "later-"
                and condition_type in later_conditions
            ):
                # Restructuring for "until-starting-again" (0,1 and 3) to "and" format
                if (
                    len(period_indices) == 3
                    and start_idx == check_period
                    and (mid_idx - start_idx) == 1
                    and end_idx >= 3
                ):
                    return [
                        "and",
                        [
                            "starting-continuing-until",
                            all_periods[start_idx],
                            all_periods[mid_idx + 1],
                        ],
                        ["during", all_periods[end_idx]],
                    ]
                # Restructuring for "until-starting-again" (0 and 2,3) to "and" format
                elif (
                    len(period_indices) == 3
                    and start_idx == check_period
                    and (mid_idx - start_idx) != 1
                    and mid_idx >= 2
                ):
                    return [
                        "and",
                        ["during", all_periods[start_idx]],
                        [
                            "starting-continuing-until",
                            all_periods[mid_idx],
                            all_periods[min(end_idx + 1, total_periods_available - 1)],
                        ],
                    ]
                # For a continuous block where "later" applies, just "starting"
                elif is_continuous:
                    return [
                        "starting-continuing-until",
                        all_periods[start_idx],
                        all_periods[min(end_idx + 1, total_periods_available - 1)],
                    ]
            return summary_text_temp  # Return the determined summary phrase if 'later' didn't apply

        # Handle specific patterns for 4 periods (assuming total_periods_available is 5)
        elif num_periods == 4 and total_periods_available == 5:
            # Starts in the 2nd period and continuous for 4 periods
            if start_idx == check_period + 1 and end_idx == 4:
                summary_text_temp = ["starting", all_periods[start_idx]]
            # Continuous block of 4 periods starting at 'check_period'
            elif start_idx == check_period and is_continuous:
                summary_text_temp = ["until", all_periods[end_idx]]
            # Continuous block of 4 periods starting after 'check_period'
            elif start_idx > check_period and is_continuous:
                summary_text_temp = [
                    "starting-continuing-until",
                    all_periods[start_idx],
                    all_periods[end_idx],
                ]
            # Discontinuous 4 periods starting at 'check_period'
            elif start_idx == check_period and not is_continuous:
                # E.g., [0] and [2,3,4]
                if (period_indices[2] - period_indices[1]) == 1 and (
                    end_idx - period_indices[2]
                ) == 1:
                    summary_text_temp = [
                        "until-starting-again",
                        all_periods[start_idx + 1],
                        all_periods[period_indices[1]],
                    ]
                # E.g., [0,1] and [3,4]
                elif (
                    (period_indices[1] - start_idx) == 1
                    and (period_indices[2] - period_indices[1]) != 1
                    and (end_idx - period_indices[2]) == 1
                ):
                    summary_text_temp = [
                        "until-starting-again",
                        all_periods[period_indices[1] + 1],
                        all_periods[period_indices[2]],
                    ]
                # E.g., [0,1,2] and [4]
                elif (end_idx - period_indices[2]) != 1:
                    summary_text_temp = [
                        "until-starting-again",
                        all_periods[period_indices[2] + 1],
                        all_periods[end_idx],
                    ]

            # Apply "later" re-structuring specific to 4-period patterns
            if (
                mode == "hour"  # 'later' only applies in hourly mode
                and "later"
                in all_periods[0]  # Check if the first period is indeed "later-"
                and condition_type in later_conditions
            ):
                if is_continuous:
                    return ["starting", all_periods[start_idx]]
                # Restructuring for [0] and [2,3,4] to "and" format
                elif (
                    len(period_indices) == 4
                    and (period_indices[2] - period_indices[1]) == 1
                    and (end_idx - period_indices[2]) == 1
                    and end_idx == 4
                ):
                    return [
                        "and",
                        ["during", all_periods[start_idx]],
                        ["starting", all_periods[period_indices[1]]],
                    ]
                # Restructuring for [0,1,2] and [4] to "and" format
                elif (
                    len(period_indices) == 4
                    and (end_idx - period_indices[2]) != 1
                    and end_idx == 4
                ):
                    return [
                        "and",
                        [
                            "starting-continuing-until",
                            all_periods[start_idx],
                            all_periods[period_indices[2]],
                        ],
                        ["starting", all_periods[end_idx]],
                    ]
            return summary_text_temp  # Return the determined summary phrase if 'later' didn't apply

    # Default fallback: combine all individual periods with 'during' and 'and'
    # This only triggers if `num_periods > 1` and `is_continuous` is False,
    # and none of the specific 2, 3, or 4 period patterns matched.
    if num_periods > 1 and not is_continuous:
        combined_periods_text = []
        for idx in period_indices:
            combined_periods_text.append(all_periods[idx])
        return ["during", combined_periods_text]

    return None


def calculate_period_summary_text(
    period_indices,
    condition_text,
    condition_type,
    all_periods,
    all_wind_periods,
    all_dry_periods,
    all_humid_periods,
    all_vis_periods,
    max_wind_speed,
    wind_unit,
    icon_set,
    check_period,
    mode,
    later_conditions,
    today_period_for_later_check,
):
    """
    Calculates the textual summary for a specific condition (precip, cloud, wind, vis, dry, humid)
    across a set of periods.

    Parameters:
    - period_indices (list): List of indices where the condition is present.
    - condition_text (str): The base text for the condition (e.g., "light-rain", "fog").
    - condition_type (str): The type of condition ("precip", "cloud", "wind", "vis", "dry", "humid").
    - all_periods (list): List of all period names (e.g., ["today-morning", "today-afternoon"]).
    - all_wind_periods (list): Indices of periods with significant wind.
    - all_dry_periods (list): Indices of periods with low humidity.
    - all_humid_periods (list): Indices of periods with high humidity.
    - all_vis_periods (list): Indices of periods with low visibility (fog).
    - max_wind_speed (float): Maximum wind speed across all relevant periods.
    - wind_unit (float): The unit conversion for wind speed.
    - icon_set (str): Which icon set to use - Dark Sky or Pirate Weather.
    - check_period (int): The current period index being checked (usually 0 for the start).
    - mode (str): Whether the summary is for the day or the next 24h ("daily" or "hour").
    - later_conditions (list): List of conditions that start later in the first period.
    - today_period_for_later_check (str): The name of the current period for 'later' comparison.

    Returns:
    - tuple: A tuple containing:
        - summary_text (list): The textual representation of the condition for the current day/next 24 hours.
        - wind_condition_combined (bool): True if wind was combined with this condition.
        - dry_condition_combined (bool): True if dry was combined with this condition.
        - humid_condition_combined (bool): True if humid was combined with this condition.
        - vis_condition_combined (bool): True if visibility was combined with this condition.
    """
    summary_text = None
    wind_condition_combined = False
    dry_condition_combined = False
    humid_condition_combined = False
    vis_condition_combined = False
    current_condition_text = condition_text

    # Helper to check if conditions occur in the same set of periods
    def _are_periods_matching(cond_a, cond_b):
        return sorted(cond_a) == sorted(cond_b)

    # Check for accompanying conditions that can be combined with the primary condition
    # Dry and Humid should not combine with Fog (vis)
    if condition_type == "precip" or condition_type == "cloud":
        if all_wind_periods and _are_periods_matching(period_indices, all_wind_periods):
            wind_condition_combined = True
            current_condition_text = [
                "and",
                current_condition_text,
                calculate_wind_text(max_wind_speed, wind_unit, icon_set, "summary"),
            ]
        if all_vis_periods and _are_periods_matching(period_indices, all_vis_periods):
            vis_condition_combined = True
            current_condition_text = ["and", current_condition_text, "fog"]
        if all_dry_periods and _are_periods_matching(period_indices, all_dry_periods):
            dry_condition_combined = True
            current_condition_text = ["and", current_condition_text, "low-humidity"]
        if all_humid_periods and _are_periods_matching(
            period_indices, all_humid_periods
        ):
            humid_condition_combined = True
            current_condition_text = ["and", current_condition_text, "high-humidity"]
    elif condition_type == "wind":  # Wind can combine with dry/humid but not fog
        if all_dry_periods and _are_periods_matching(period_indices, all_dry_periods):
            dry_condition_combined = True
            current_condition_text = ["and", current_condition_text, "low-humidity"]
        if all_humid_periods and _are_periods_matching(
            period_indices, all_humid_periods
        ):
            humid_condition_combined = True
            current_condition_text = ["and", current_condition_text, "high-humidity"]

    # Get the base time phrase template (e.g., "during", "starting", "for-day")
    time_phrase_structure = _get_time_phrase(
        period_indices,
        condition_type,
        all_periods,
        check_period,
        mode,
        later_conditions,
        today_period_for_later_check,
    )

    if time_phrase_structure is None:
        return None, False, False, False, False

    phrase_type = time_phrase_structure[0]
    phrase_args = time_phrase_structure[1:]

    # Construct the final summary text based on the phrase template
    if phrase_type == "for-day":
        summary_text = ["for-day", current_condition_text]
    elif phrase_type == "during":
        if len(phrase_args) == 1 and isinstance(phrase_args[0], list):
            period_combination_text = phrase_args[0]
            if len(period_combination_text) > 1:
                formatted_periods = period_combination_text[0]
                for i in range(1, len(period_combination_text)):
                    formatted_periods = [
                        "and",
                        formatted_periods,
                        period_combination_text[i],
                    ]
            else:
                formatted_periods = period_combination_text[0]
            summary_text = ["during", current_condition_text, formatted_periods]
        else:
            summary_text = ["during", current_condition_text, phrase_args[0]]
    elif phrase_type == "starting":
        summary_text = ["starting", current_condition_text, phrase_args[0]]
    elif phrase_type == "until":
        summary_text = ["until", current_condition_text, phrase_args[0]]
    elif phrase_type == "starting-continuing-until":
        summary_text = [
            "starting-continuing-until",
            current_condition_text,
            phrase_args[0],
            phrase_args[1],
        ]
    elif phrase_type == "until-starting-again":
        summary_text = [
            "until-starting-again",
            current_condition_text,
            phrase_args[0],
            phrase_args[1],
        ]
    elif phrase_type == "and":

        def _inject_condition_text(phrase_part, condition_text_to_inject):
            if isinstance(phrase_part, list):
                if phrase_part[0] in [
                    "during",
                    "starting",
                    "until",
                    "starting-continuing-until",
                    "until-starting-again",
                ]:
                    new_phrase = phrase_part[:]
                    new_phrase.insert(1, condition_text_to_inject)
                    return new_phrase
                elif phrase_part[0] == "and":
                    return [phrase_part[0]] + [
                        _inject_condition_text(sub_part, condition_text_to_inject)
                        for sub_part in phrase_part[1:]
                    ]
            return phrase_part

        summary_text = _inject_condition_text(
            time_phrase_structure, current_condition_text
        )

    return (
        summary_text,
        wind_condition_combined,
        dry_condition_combined,
        humid_condition_combined,
        vis_condition_combined,
    )


def calculate_day_text(
    hours,
    precip_accum_unit,
    vis_units,
    wind_unit,
    temp_units,
    is_day_time,
    time_zone,
    curr_time,
    mode="daily",
    icon_set="darksky",
):
    """
    Calculates the daily or next 24-hour weather summary text.

    Parameters:
    - hours (list): An array of hourly forecast data.
    - precip_accum_unit (float): The precipitation accumulation unit (e.g., 0.1 for cm, 1 for inches).
    - vis_units (float): The visibility unit used.
    - wind_unit (float): The wind speed unit used.
    - temp_units (float): The temperature unit used.
    - is_day_time (bool): Whether it's currently daytime.
    - time_zone (str): The timezone for the current location.
    - curr_time (int): The current epoch time.
    - mode (str, optional): Which mode to run the function in ("daily" or "hour"). Defaults to "daily".

    Returns:
    - tuple: A tuple containing:
        - c_icon (str): The icon representing the current day/next 24 hours.
        - summary_text (list): The textual representation of the current day/next 24 hours.
    """

    # Initialize return variables to prevent UnboundLocalError
    final_constructed_summary = None
    current_c_icon = None

    # Initialize combination flags to False at the top level
    combined_vis_flag = False
    combined_dry_flag = False
    combined_humid_flag = False
    combined_wind_flag = False  # Added this for consistency

    # Return "unavailable" if too much data is provided
    if len(hours) > 25:
        return "none", ["for-day", "unavailable"]

    # Get local time and current hour
    zone = tz.gettz(time_zone)
    curr_date = datetime.datetime.fromtimestamp(hours[0]["time"], zone)
    curr_hour_local = int(curr_date.strftime("%H"))

    # This is used for the "later" check.
    # It correctly gets the period name for `curr_hour_local - 1` without redundant prefix.
    today_period_for_later_check = _get_period_name(curr_hour_local - 1, is_today=True)

    # This dictionary will store processed data for each logical standard period (Morning, Afternoon, Evening, Night)
    # The keys will be the unique period names encountered (e.g., 'today-morning', 'tomorrow-night').
    standard_periods_data = {}

    # Generate a list of unique standard period names that will be covered by the 24-hour forecast window.
    # This ensures `all_period_names` has a consistent structure regardless of `curr_hour_local`.
    all_period_names_in_forecast_order = []

    # Start from the first hour of the forecast and determine the sequence of periods
    temp_date_for_period_scan = datetime.datetime.fromtimestamp(hours[0]["time"], zone)

    # Populate `all_period_names_in_forecast_order` and initialize `standard_periods_data`.
    # This loop ensures that all 4-5 relevant periods are set up, even if some start with zero hours.
    # Iterate for enough hours to guarantee covering at least 4 unique periods (24 hours + overlap).
    max_hours_to_scan_for_periods = (
        len(hours) + 8
    )  # Scan a bit beyond forecast end to ensure next periods are captured

    for i in range(max_hours_to_scan_for_periods):
        current_iter_hour_date = temp_date_for_period_scan + datetime.timedelta(hours=i)
        current_iter_hour_local = current_iter_hour_date.hour

        # Determine `is_today` for naming periods.
        # `_get_period_name` handles the actual period string. We just need to tell it if it's the current calendar day.
        is_today_in_iter = current_iter_hour_date.date() == curr_date.date()

        period_name_for_iter = _get_period_name(
            current_iter_hour_local, is_today=is_today_in_iter
        )

        if period_name_for_iter not in all_period_names_in_forecast_order:
            all_period_names_in_forecast_order.append(period_name_for_iter)
            # Initialize an empty data structure for this new unique period name
            standard_periods_data[period_name_for_iter] = {
                "num_hours_fog": 0,
                "num_hours_dry": 0,
                "num_hours_wind": 0,
                "rain_accum": 0.0,
                "snow_accum": 0.0,
                "snow_error": 0.0,
                "sleet_accum": 0.0,
                "max_pop": 0.0,
                "max_intensity": 0.0,
                "cloud_cover_sum": 0.0,
                "max_wind_speed": 0.0,
                "period_length": 0,
                "num_hours_humid": 0,
                "precip_types_in_period": [],
                "precip_intensity_sum": 0.0,
                "precip_hours_count": 0,
                "avg_cloud_cover": 0.0,
            }

        # Stop generating period names if we have enough for a full 24-hour cycle (e.g., 5 periods)
        # and we've processed at least 24 hours of potential period boundaries.
        if len(all_period_names_in_forecast_order) >= 5 and i >= 23:
            break

    # Now iterate through the actual hourly forecast data and aggregate into the pre-defined standard periods
    for idx, hour in enumerate(hours):
        hour_date = datetime.datetime.fromtimestamp(hour["time"], zone)
        hour_in_loop = int(hour_date.strftime("%H"))

        # Determine `is_today` for naming the current hour's period.
        is_today_for_hour_data = hour_date.date() == curr_date.date()

        current_hour_period_name = _get_period_name(
            hour_in_loop, is_today=is_today_for_hour_data
        )

        # Accumulate data for the current hour into the correct standard period's data structure
        if current_hour_period_name in standard_periods_data:
            period_data = standard_periods_data[current_hour_period_name]

            period_data["period_length"] += 1
            period_data["cloud_cover_sum"] += hour["cloudCover"]
            period_data["max_wind_speed"] = max(
                period_data["max_wind_speed"], hour["windSpeed"]
            )
            period_data["max_intensity"] = max(
                period_data["max_intensity"], hour["precipIntensity"]
            )

            if (
                humidity_sky_text(hour["temperature"], temp_units, hour["humidity"])
                == "high-humidity"
            ):
                period_data["num_hours_humid"] += 1
            if (
                humidity_sky_text(hour["temperature"], temp_units, hour["humidity"])
                == "low-humidity"
            ):
                period_data["num_hours_dry"] += 1
            if (
                calculate_vis_text(hour["visibility"], vis_units, "icon") == "fog"
                and hour["precipIntensity"] <= 0.02 * precip_accum_unit
            ):
                period_data["num_hours_fog"] += 1
            if (
                calculate_wind_text(hour["windSpeed"], wind_unit, "darksky", "icon")
                == "wind"
            ):
                period_data["num_hours_wind"] += 1

            if hour["precipType"] == "rain" or hour["precipType"] == "none":
                period_data["rain_accum"] += hour["precipAccumulation"]
            elif hour["precipType"] == "snow":
                period_data["snow_accum"] += hour["precipAccumulation"]
                period_data["snow_error"] += hour["precipIntensityError"]
            elif hour["precipType"] == "sleet":
                period_data["sleet_accum"] += hour["precipAccumulation"]

            if hour["precipIntensity"] > 0 or hour["precipAccumulation"] > 0:
                period_data["precip_types_in_period"].append(hour["precipType"])
                period_data["max_pop"] = max(
                    period_data["max_pop"], hour["precipProbability"]
                )
                period_data["precip_hours_count"] += 1
                period_data["precip_intensity_sum"] += hour["precipIntensity"]

    # Finalize `period_stats` list by only including periods that actually have data,
    # and in the correct order determined by `all_period_names_in_forecast_order`.
    period_stats_list_final = []
    final_all_period_names_list = []

    for period_name in all_period_names_in_forecast_order:
        p_data = standard_periods_data.get(period_name)
        if (
            p_data and p_data["period_length"] > 0
        ):  # Only add if the period actually received data
            # Calculate final average cloud cover for this period
            p_data["avg_cloud_cover"] = (
                p_data["cloud_cover_sum"] / p_data["period_length"]
            )
            period_stats_list_final.append(p_data)
            final_all_period_names_list.append(period_name)

    # Use these finalized lists for subsequent calculations
    period_stats = period_stats_list_final
    all_period_names = final_all_period_names_list

    # --- Rest of the function remains largely the same, but now operates on correctly segmented periods ---
    # print("Final Period Stats:", period_stats) # For debugging period stats
    # print("Final All Period Names:", all_period_names) # For debugging period names

    # Initialize lists for storing period indices of various conditions
    precip_periods = []
    vis_periods = []
    wind_periods = []
    humid_periods = []
    dry_periods = []
    cloud_levels = []

    # Initialize overall accumulation and max values for the entire forecast block
    total_rain_accum = 0.0
    total_snow_accum = 0.0
    total_sleet_accum = 0.0
    total_snow_error = 0.0
    overall_max_intensity = 0.0
    overall_max_wind = 0.0
    overall_avg_cloud_cover_sum = 0.0  # Sum of average cloud cover for each period
    overall_avg_pop = 0.0
    overall_precip_hours_count = 0
    overall_precip_intensity_sum = 0.0

    overall_most_common_precip = []

    # Process collected period statistics to determine condition presence and overall totals
    for i, p_data in enumerate(period_stats):
        # Accumulate overall totals
        total_rain_accum += p_data["rain_accum"]
        total_snow_accum += p_data["snow_accum"]
        total_sleet_accum += p_data["sleet_accum"]
        total_snow_error += p_data["snow_error"]
        overall_max_intensity = max(overall_max_intensity, p_data["max_intensity"])
        overall_max_wind = max(overall_max_wind, p_data["max_wind_speed"])
        overall_avg_cloud_cover_sum += p_data[
            "avg_cloud_cover"
        ]  # Sum for calculating overall average

        # Check if precipitation is significant enough in this period
        is_precip_in_period = (
            p_data["snow_accum"] > (2.5 * precip_accum_unit)
            or p_data["rain_accum"] > (0.25 * precip_accum_unit)
            or p_data["sleet_accum"] > (0.25 * precip_accum_unit)
        )
        if is_precip_in_period:
            precip_periods.append(i)
            overall_most_common_precip.extend(p_data["precip_types_in_period"])
            overall_avg_pop = max(overall_avg_pop, p_data["max_pop"])
            overall_precip_hours_count += p_data["precip_hours_count"]
            overall_precip_intensity_sum += p_data["precip_intensity_sum"]

        # Determine if other conditions are significant in this period
        # Note: These thresholds depend on `period_length` being correct now.
        if p_data["num_hours_wind"] >= (min(p_data["period_length"] / 2, 3)):
            wind_periods.append(i)
        if (
            p_data["max_intensity"] < 0.02 * precip_accum_unit
            and p_data["max_wind_speed"] / wind_unit < 6.7056
            and p_data["num_hours_fog"] >= (min(p_data["period_length"] / 2, 3))
        ):
            vis_periods.append(i)
        if p_data["num_hours_dry"] >= (min(p_data["period_length"] / 2, 3)):
            dry_periods.append(i)
        if p_data["num_hours_humid"] >= (min(p_data["period_length"] / 2, 3)):
            humid_periods.append(i)

        # Get cloud level for this period
        _, cloud_level = calculate_cloud_text(p_data["avg_cloud_cover"])
        cloud_levels.append(cloud_level)

    # Convert total snow error if unit is cm (original had this conversion)
    if precip_accum_unit == 0.1:
        total_snow_error /= 10

    # Calculate overall average cloud cover for the entire forecast block
    overall_avg_cloud_cover = (
        overall_avg_cloud_cover_sum / len(period_stats) if period_stats else 0
    )

    # Determine the most common cloud level across all periods for representative text
    most_common_cloud_level = Most_Common(cloud_levels) if cloud_levels else 0

    # Select a representative cloud level for final cloud text and icon calculation
    representative_cloud_level = most_common_cloud_level
    if most_common_cloud_level != max(cloud_levels):
        representative_cloud_level = most_common_cloud_level
    else:
        representative_cloud_level = max(cloud_levels)

    # Convert the representative cloud level back to textual form and a derived average for icon
    if representative_cloud_level == 0:
        final_cloud_text = "clear"
        derived_avg_cloud_for_icon = 0.0
    elif representative_cloud_level == 1:
        final_cloud_text = "very-light-clouds"
        derived_avg_cloud_for_icon = 0.25
    elif representative_cloud_level == 2:
        final_cloud_text = "light-clouds"
        derived_avg_cloud_for_icon = 0.50
    elif representative_cloud_level == 3:
        final_cloud_text = "medium-clouds"
        derived_avg_cloud_for_icon = 0.75
    else:  # representative_cloud_level == 4
        final_cloud_text = "heavy-clouds"
        derived_avg_cloud_for_icon = 1.0

    # If there's only one period, use its actual average cloud cover for the icon directly
    if len(period_stats) == 1:
        derived_avg_cloud_for_icon = period_stats[0]["avg_cloud_cover"]

    # Determine the most common precipitation type for overall summary
    most_common_overall_precip_type = (
        Most_Common(overall_most_common_precip)
        if overall_most_common_precip
        else "none"
    )

    precip_summary_text = None
    precip_icon = None
    secondary_precip_condition = None

    total_precip_accum = total_rain_accum + total_snow_accum + total_sleet_accum

    # Calculate overall precipitation text and icon if significant precipitation occurs
    if overall_avg_pop > 0 and total_precip_accum >= (0.1 * precip_accum_unit):
        if total_snow_accum > 0 and total_rain_accum > 0 and total_sleet_accum > 0:
            precip_summary_text = "mixed-precipitation"
            most_common_overall_precip_type = "sleet"
            secondary_precip_condition = (
                "medium-snow"  # Indicate snow totals are relevant
            )
        else:
            # Determine primary and secondary precipitation types based on accumulation
            if total_snow_accum > 0:
                if total_rain_accum > 0 and total_snow_accum > total_rain_accum:
                    most_common_overall_precip_type = "snow"
                    secondary_precip_condition = "medium-rain"
                elif total_rain_accum > 0 and total_snow_accum < total_rain_accum:
                    most_common_overall_precip_type = "rain"
                    secondary_precip_condition = "medium-snow"
                elif total_sleet_accum > 0 and total_snow_accum > total_sleet_accum:
                    most_common_overall_precip_type = "snow"
                    secondary_precip_condition = "medium-sleet"
                elif total_sleet_accum > 0 and total_snow_accum < total_sleet_accum:
                    most_common_overall_precip_type = "sleet"
                    secondary_precip_condition = "medium-snow"
            elif total_sleet_accum > 0:
                if total_rain_accum > 0 and total_rain_accum > total_sleet_accum:
                    most_common_overall_precip_type = "rain"
                    secondary_precip_condition = "medium-sleet"
                elif total_rain_accum > 0 and total_rain_accum < total_sleet_accum:
                    most_common_overall_precip_type = "sleet"
                    secondary_precip_condition = "medium-rain"

            # Re-evaluate primary precipType if calculated type has zero accumulation
            if total_snow_accum == 0 and most_common_overall_precip_type == "snow":
                if total_rain_accum > 0:
                    most_common_overall_precip_type = "rain"
                elif total_sleet_accum > 0:
                    most_common_overall_precip_type = "sleet"
            elif total_rain_accum == 0 and most_common_overall_precip_type == "rain":
                if total_snow_accum > 0:
                    most_common_overall_precip_type = "snow"
                elif total_sleet_accum > 0:
                    most_common_overall_precip_type = "sleet"
            elif total_sleet_accum == 0 and most_common_overall_precip_type == "sleet":
                if total_snow_accum > 0:
                    most_common_overall_precip_type = "snow"
                elif total_rain_accum > 0:
                    most_common_overall_precip_type = "rain"

            # Promote to stronger precip if significant accumulation is forecast
            if (
                total_rain_accum > (10 * precip_accum_unit)
                and most_common_overall_precip_type != "rain"
            ):
                secondary_precip_condition = "medium-" + most_common_overall_precip_type
                most_common_overall_precip_type = "rain"
            if (
                total_snow_accum > (5 * precip_accum_unit)
                and most_common_overall_precip_type != "snow"
            ):
                secondary_precip_condition = "medium-" + most_common_overall_precip_type
                most_common_overall_precip_type = "snow"
            if (
                total_sleet_accum > (1 * precip_accum_unit)
                and most_common_overall_precip_type != "sleet"
            ):
                secondary_precip_condition = "medium-" + most_common_overall_precip_type
                most_common_overall_precip_type = "sleet"

        # Calculate final precipitation text and icon
        precip_summary_text, precip_icon = calculate_precip_text(
            overall_max_intensity,
            precip_accum_unit,
            most_common_overall_precip_type,
            "hourly",  # This is a fixed parameter as per original code
            total_rain_accum,
            total_snow_accum,
            total_sleet_accum,
            overall_avg_pop if overall_avg_pop != -999 else 1,
            icon_set,
            "both",
            overall_precip_intensity_sum / overall_precip_hours_count
            if overall_precip_hours_count > 0
            else 0,
        )

    # Correct "medium-none" secondary condition to "medium-precipitation"
    if secondary_precip_condition == "medium-none":
        secondary_precip_condition = "medium-precipitation"

    # Add snow accumulation range to precip text if applicable
    snow_sentence = None
    if (
        total_snow_accum > (10 * precip_accum_unit)
        or secondary_precip_condition == "medium-snow"
    ):
        snow_low_accum = math.floor(total_snow_accum - (total_snow_error / 2))
        snow_max_accum = math.ceil(
            total_snow_accum + (total_snow_error / 2)
        )  # Bug fix: used total_snow_accum instead of snow_accum
        snow_low_accum = max(0, snow_low_accum)  # Snow accumulation cannot be negative

        if total_snow_error <= 0:
            snow_sentence = [
                "centimeters" if precip_accum_unit == 0.1 else "inches",
                int(math.ceil(total_snow_accum)),
            ]
        elif snow_max_accum > 0:
            if total_snow_accum == 0:
                snow_sentence = [
                    "less-than",
                    ["centimeters" if precip_accum_unit == 0.1 else "inches", 1],
                ]
            elif snow_low_accum == 0:
                snow_sentence = [
                    "less-than",
                    [
                        "centimeters" if precip_accum_unit == 0.1 else "inches",
                        snow_max_accum,
                    ],
                ]
            else:
                snow_sentence = [
                    "centimeters" if precip_accum_unit == 0.1 else "inches",
                    ["range", snow_low_accum, snow_max_accum],
                ]

    if snow_sentence is not None:
        if most_common_overall_precip_type == "snow":
            precip_summary_text = ["parenthetical", precip_summary_text, snow_sentence]
        elif secondary_precip_condition == "medium-snow":
            precip_summary_text = ["parenthetical", precip_summary_text, snow_sentence]

    # Combine primary and secondary precipitation conditions with "and"
    if (
        secondary_precip_condition is not None
        and secondary_precip_condition != "medium-snow"
    ):
        precip_summary_text = ["and", precip_summary_text, secondary_precip_condition]

    # List to track conditions that start "later" in the first period (only for hourly mode)
    later_conditions_list = []

    # Apply "later" text only when in hourly mode and condition starts later in the first period
    # The condition `all_period_names[0] == today_period_for_later_check` is the critical check
    # to see if the forecast starts mid-period and the 'later' logic should apply.
    if (
        mode == "hour"
        and all_period_names
        and all_period_names[0] == today_period_for_later_check
    ):
        # Check precip
        if (
            precip_periods and precip_periods[0] == 0
        ):  # If precip occurs in the first period
            # But the very first hour forecast doesn't have it
            curr_precip_text_for_first_hour = calculate_precip_text(
                hours[0]["precipIntensity"],
                precip_accum_unit,
                hours[0]["precipType"],
                "hourly",
                hours[0]["precipAccumulation"],
                hours[0]["precipAccumulation"],
                hours[0]["precipAccumulation"],
                hours[0]["precipProbability"],
                icon_set,
                "summary",
                hours[0]["precipIntensity"],
            )
            if curr_precip_text_for_first_hour is None:
                all_period_names[0] = "later-" + all_period_names[0]
                later_conditions_list.append("precip")

        # Check visibility (fog)
        if vis_periods and vis_periods[0] == 0:
            if (
                calculate_vis_text(hours[0]["visibility"], vis_units, "summary") is None
                and "later" not in all_period_names[0]
            ):  # Avoid double "later"
                all_period_names[0] = "later-" + all_period_names[0]
                later_conditions_list.append("vis")

        # Check wind
        if wind_periods and wind_periods[0] == 0:
            if (
                calculate_wind_text(
                    hours[0]["windSpeed"], wind_unit, icon_set, "summary"
                )
                is None
                and "later" not in all_period_names[0]
            ):
                all_period_names[0] = "later-" + all_period_names[0]
                later_conditions_list.append("wind")

        # Check dry humidity
        if dry_periods and dry_periods[0] == 0:
            if (
                humidity_sky_text(
                    hours[0]["temperature"], temp_units, hours[0]["humidity"]
                )
                is None
                and "later" not in all_period_names[0]
            ):
                all_period_names[0] = "later-" + all_period_names[0]
                later_conditions_list.append("dry")

        # Check humid humidity
        if humid_periods and humid_periods[0] == 0:
            if (
                humidity_sky_text(
                    hours[0]["temperature"], temp_units, hours[0]["humidity"]
                )
                is None
                and "later" not in all_period_names[0]
            ):
                all_period_names[0] = "later-" + all_period_names[0]
                later_conditions_list.append("humid")

    # Flags to indicate if a condition is present at all in the forecast block
    has_precip = bool(precip_periods) and precip_summary_text is not None
    has_wind = bool(wind_periods)
    has_vis = bool(vis_periods)
    has_dry = bool(dry_periods)
    has_humid = bool(humid_periods)

    # Initialize variables for condition-specific summary texts
    precip_only_summary = None
    wind_only_summary = None
    vis_only_summary = None
    dry_only_summary = None
    humid_only_summary = None
    cloud_full_summary = None

    # Calculate summary text for each condition type, passing all relevant periods for combination logic
    if has_precip:
        # Pass all period data lists to allow calculate_period_summary_text to determine internal combinations
        (
            precip_only_summary,
            temp_wind_combined,
            temp_dry_combined,
            temp_humid_combined,
            temp_vis_combined,
        ) = calculate_period_summary_text(
            precip_periods,
            precip_summary_text,
            "precip",
            all_period_names,
            wind_periods,
            dry_periods,
            humid_periods,
            vis_periods,
            overall_max_wind,
            wind_unit,
            icon_set,
            0,
            mode,
            later_conditions_list,
            today_period_for_later_check,
        )
        # Update combination flags based on whether wind/dry/humid/vis were actually combined *into* precipitation's summary
        # This requires inspecting the structure of precip_only_summary
        if (
            isinstance(precip_only_summary, list)
            and precip_only_summary
            and precip_only_summary[0] == "and"
        ):
            combined_wind_flag = temp_wind_combined
            combined_dry_flag = temp_dry_combined
            combined_humid_flag = temp_humid_combined
            combined_vis_flag = temp_vis_combined
        else:  # If precip_only_summary is not an "and" statement, then no combination happened for these flags
            combined_wind_flag = False
            combined_dry_flag = False
            combined_humid_flag = False
            combined_vis_flag = False

    # Calculate summaries for other conditions. The combination flags for them are local to their calls.
    if has_wind:
        wind_only_summary, _, _, _, _ = calculate_period_summary_text(
            wind_periods,
            calculate_wind_text(overall_max_wind, wind_unit, icon_set, "summary"),
            "wind",
            all_period_names,
            [],
            dry_periods,
            humid_periods,
            [],  # Wind can combine with dry/humid
            overall_max_wind,
            wind_unit,
            icon_set,
            0,
            mode,
            later_conditions_list,
            today_period_for_later_check,
        )
    if has_vis:
        vis_only_summary, _, _, _, _ = calculate_period_summary_text(
            vis_periods,
            "fog",
            "vis",
            all_period_names,
            [],
            [],
            [],
            [],
            overall_max_wind,
            wind_unit,
            icon_set,
            0,
            mode,
            later_conditions_list,
            today_period_for_later_check,
        )
    if has_dry:
        dry_only_summary, _, _, _, _ = calculate_period_summary_text(
            dry_periods,
            "low-humidity",
            "dry",
            all_period_names,
            [],
            [],
            [],
            [],
            overall_max_wind,
            wind_unit,
            icon_set,
            0,
            mode,
            later_conditions_list,
            today_period_for_later_check,
        )
    if has_humid:
        humid_only_summary, _, _, _, _ = calculate_period_summary_text(
            humid_periods,
            "high-humidity",
            "humid",
            all_period_names,
            [],
            [],
            [],
            [],
            overall_max_wind,
            wind_unit,
            icon_set,
            0,
            mode,
            later_conditions_list,
            today_period_for_later_check,
        )

    # Cloud full summary, including potential combinations with wind/dry/humid/vis
    (
        cloud_full_summary,
        cloud_wind_combined_flag,
        cloud_dry_combined_flag,
        cloud_humid_combined_flag,
        cloud_vis_combined_flag,
    ) = calculate_period_summary_text(
        [
            i for i, _ in enumerate(period_stats)
        ],  # Pass all period indices for cloud to find its pattern
        final_cloud_text,
        "cloud",
        all_period_names,
        wind_periods,
        dry_periods,
        humid_periods,
        vis_periods,
        overall_max_wind,
        wind_unit,
        icon_set,
        0,
        mode,
        later_conditions_list,
        today_period_for_later_check,
    )

    # --- Final Summary Construction Logic: Select top 2 conditions based on priority ---

    # Candidate summaries: list of dictionaries, each describing a potential main summary.
    # We will prioritize and select from these.
    # Properties: 'type', 'priority', 'all_day', 'start_idx', 'text', 'icon'

    # Priority order: lower number is higher priority.
    # 0: Precipitation
    # 1: Visibility (Fog)
    # 2: Wind
    # 3: Dry/Humid (if combined, or if primary cloud is clear)
    # 4: Cloud (fallback)

    candidate_summaries_for_final_assembly = []
    total_periods_available = []

    # 1. Precipitation
    if has_precip:
        is_precip_all_day = len(precip_periods) == total_periods_available
        candidate_summaries_for_final_assembly.append(
            {
                "type": "precip",
                "priority": 0,
                "all_day": is_precip_all_day,
                "start_idx": precip_periods[0] if precip_periods else -1,
                "text": precip_only_summary,
                "icon": precip_icon,
            }
        )

    # 2. Visibility (Fog) - only if not already covered by precipitation
    if has_vis and not combined_vis_flag:
        is_vis_all_day = len(vis_periods) == total_periods_available
        candidate_summaries_for_final_assembly.append(
            {
                "type": "vis",
                "priority": 1,
                "all_day": is_vis_all_day,
                "start_idx": vis_periods[0] if vis_periods else -1,
                "text": vis_only_summary,
                "icon": "fog",
            }
        )

    # 3. Wind - only if not already covered by precipitation or visibility
    if has_wind and not combined_wind_flag:
        is_wind_all_day = len(wind_periods) == total_periods_available
        candidate_summaries_for_final_assembly.append(
            {
                "type": "wind",
                "priority": 2,
                "all_day": is_wind_all_day,
                "start_idx": wind_periods[0] if wind_periods else -1,
                "text": wind_only_summary,
                "icon": calculate_wind_text(
                    overall_max_wind, wind_unit, icon_set, "icon"
                ),
            }
        )

    # 4. Dry Humidity - only if not already covered AND (combined OR cloud is clear)
    if (
        has_dry
        and not combined_dry_flag
        and (cloud_dry_combined_flag or final_cloud_text == "clear")
    ):
        is_dry_all_day = len(dry_periods) == total_periods_available
        candidate_summaries_for_final_assembly.append(
            {
                "type": "dry",
                "priority": 3,
                "all_day": is_dry_all_day,
                "start_idx": dry_periods[0] if dry_periods else -1,
                "text": dry_only_summary,
                "icon": None,  # Dry/humid don't have dedicated icons, fallback to cloud
            }
        )

    # 5. Humid Humidity - only if not already covered AND (combined OR cloud is clear)
    if (
        has_humid
        and not combined_humid_flag
        and (cloud_humid_combined_flag or final_cloud_text == "clear")
    ):
        is_humid_all_day = len(humid_periods) == total_periods_available
        candidate_summaries_for_final_assembly.append(
            {
                "type": "humid",
                "priority": 4,
                "all_day": is_humid_all_day,
                "start_idx": humid_periods[0] if humid_periods else -1,
                "text": humid_only_summary,
                "icon": None,  # Dry/humid don't have dedicated icons, fallback to cloud
            }
        )

    # 6. Cloud Cover - as a fallback if no other primary condition is present
    if (
        not candidate_summaries_for_final_assembly
    ):  # If no higher-priority conditions are present
        is_cloud_all_day = len(period_stats) == total_periods_available
        candidate_summaries_for_final_assembly.append(
            {
                "type": "cloud",
                "priority": 5,
                "all_day": is_cloud_all_day,
                "start_idx": 0,  # Cloud is always "present" from the start of the forecast
                "text": cloud_full_summary,
                "icon": calculate_sky_icon(derived_avg_cloud_for_icon, True, icon_set),
            }
        )

    # Sort candidates:
    # 1. By 'all_day' (True comes before False, so `not x["all_day"]` makes all-day items sort first)
    # 2. By 'priority' (lower number is higher priority)
    # 3. By 'start_idx' (earliest start comes first)
    sorted_summaries_candidates = sorted(
        candidate_summaries_for_final_assembly,
        key=lambda x: (not x["all_day"], x["priority"], x["start_idx"]),
    )

    # Select the top 1 or 2 summaries
    selected_final_summary_texts = []

    for summary_data in sorted_summaries_candidates:
        # The icon logic needs to pick the icon of the highest priority summary *before* combining.
        # If no icon has been set yet, and this summary has an icon, use it.
        if summary_data["icon"] and current_c_icon is None:
            current_c_icon = summary_data["icon"]

        selected_final_summary_texts.append(summary_data["text"])

        if len(selected_final_summary_texts) >= 2:
            break  # Limit to top 2 conditions

    # Final summary text construction
    if len(selected_final_summary_texts) == 1:
        final_constructed_summary = ["sentence", selected_final_summary_texts[0]]
    elif len(selected_final_summary_texts) == 2:
        final_constructed_summary = [
            "sentence",
            ["and", selected_final_summary_texts[0], selected_final_summary_texts[1]],
        ]
    else:  # Fallback if no summaries generated (shouldn't happen with cloud fallback)
        final_constructed_summary = ["for-day", "unavailable"]

    # Ensure an icon is always returned, defaulting to overall average cloud cover if none set.
    if current_c_icon is None:
        current_c_icon = calculate_sky_icon(overall_avg_cloud_cover, True, icon_set)

    return current_c_icon, final_constructed_summary


def nextPeriod(curr_period):
    """
    Calculates the next textual representation of a period.

    Parameters:
    - curr_period (str): The current textual representation of the period.

    Returns:
    - str: The next textual representation of the period.
    """
    if "morning" in curr_period:
        return curr_period.replace("morning", "afternoon")
    elif "afternoon" in curr_period:
        return curr_period.replace("afternoon", "evening")
    elif "evening" in curr_period:
        return curr_period.replace("evening", "night")
    elif "night" in curr_period:
        return curr_period.replace("night", "morning").replace("today-", "tomorrow-")
    return curr_period
