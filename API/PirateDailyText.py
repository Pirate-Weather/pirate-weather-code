# %% Script to contain the functions that generate the daily, next 24 hour and
# day/night (half-day) text summaries of the forecast data for Pirate Weather.
#
# How a summary is built (calculate_day_text):
#   1. Split the hours into periods:
#        morning    06:00-12:00
#        afternoon  12:00-18:00
#        evening    18:00-24:00
#        night      00:00-06:00 (translated as "overnight")
#      Every summary window starts at 6am or 6pm, so it covers whole periods:
#        - daily (mode "day"/"daily"): 6am to 6am, morning to night
#        - day/night (half_day=True): 6am-6pm (morning, afternoon) or
#          6pm-6am (evening, night)
#        - next 24 hours (mode "hour"): from the current hour, with today-/
#          tomorrow- prefixes, so the first and last periods can be partial
#   2. Aggregate each period (accumulations, maxima, hours of fog/wind/storms).
#   3. Decide which periods have precipitation, thunderstorms, fog and wind,
#      and the most common cloud cover.
#   4. Build the precipitation/thunderstorm text for the whole window.
#   5. Describe when each condition happens with a time phrase
#      (_get_time_phrase), e.g. ["until", "evening"].
#   6. Keep the one or two most important conditions and join them.
import datetime
import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from dateutil import tz

from API.constants.api_const import PRECIP_TYPES
from API.constants.shared_const import MISSING_DATA
from API.constants.text_const import (
    CLOUD_COVER_DAILY_THRESHOLDS,
    CLOUD_COVER_THRESHOLDS,
    DAILY_PRECIP_ACCUM_TEXT_THRESHOLD_MM,
    DEFAULT_HUMIDITY,
    DEFAULT_POP,
    DEFAULT_VISIBILITY,
    HALF_DAY_PRECIP_ACCUM_TEXT_THRESHOLD_MM,
    LESS_THAN_TOLERANCE,
    PRECIP_INTENSITY_THRESHOLDS,
    PRECIP_PROB_THRESHOLD,
    WIND_THRESHOLDS,
)
from API.PirateTextHelper import (
    calculate_precip_text,
    calculate_sky_icon,
    calculate_thunderstorm_text,
    calculate_vis_text,
    calculate_wind_text,
    estimate_snow_height,
    most_common,
)

# Local hours at which each period starts. The night runs from midnight until
# the morning, so no period crosses midnight.
MORNING_START = 6
AFTERNOON_START = 12
EVENING_START = 18

# Longest windows that are summarised; anything longer is "unavailable".
# The extra hour allows for the day clocks fall back.
MAX_HOURS = 25
MAX_HALF_DAY_HOURS = 13

# Fog is only reported when any precipitation is too light to reduce
# visibility itself (mm/h, snow using a 10:1 ratio).
FOG_MAX_LIQUID_INTENSITY = 0.02
FOG_MAX_SNOW_INTENSITY = 0.2

# A condition is significant in a period when it lasts for half of the period,
# capped at this many hours.
SIGNIFICANT_CONDITION_MAX_HOURS = 3
SIGNIFICANT_THUNDERSTORM_MAX_HOURS = 1

# Minimum total accumulation (mm) before a precipitation summary is generated.
PRECIP_SUMMARY_MIN_ACCUM_MM = 0.1
# Snow accumulation (mm) above which the snow total is added to the summary.
SNOW_SENTENCE_MIN_ACCUM_MM = 10

MM_PER_INCH = 25.4
MM_PER_CENTIMETER = 10

# Representative cloud cover for each cloud level, used for the daily icon.
CLOUD_LEVEL_TEXT_AND_COVER = (
    ("clear", CLOUD_COVER_DAILY_THRESHOLDS["clear"]),
    ("very-light-clouds", CLOUD_COVER_DAILY_THRESHOLDS["mostly_clear"]),
    ("light-clouds", CLOUD_COVER_DAILY_THRESHOLDS["partly_cloudy"]),
    ("medium-clouds", CLOUD_COVER_DAILY_THRESHOLDS["mostly_cloudy"]),
    ("heavy-clouds", CLOUD_COVER_DAILY_THRESHOLDS["cloudy"]),
)


def _value_or_default(value, default):
    """Return ``default`` when ``value`` is None or NaN, otherwise ``value``."""
    if value is None:
        return default
    try:
        if math.isnan(value):
            return default
    except TypeError:
        pass
    return value


def _get_zone(time_zone):
    """
    Returns the tzinfo for a time zone name.

    zoneinfo converts timestamps several times faster than dateutil, which
    matters as every hour of every summary is converted. Names zoneinfo does
    not recognise fall back to dateutil, as before.
    """
    try:
        return ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError):
        return tz.gettz(time_zone)


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
    if cloud_cover > CLOUD_COVER_THRESHOLDS["cloudy"]:
        return "heavy-clouds", 4
    elif cloud_cover > CLOUD_COVER_THRESHOLDS["mostly_cloudy"]:
        return "medium-clouds", 3
    elif cloud_cover > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        return "light-clouds", 2
    elif cloud_cover > CLOUD_COVER_THRESHOLDS["mostly_clear"]:
        return "very-light-clouds", 1
    else:
        return "clear", 0


def _get_period_name(local_time, start_time, mode):
    """
    Names the period an hour belongs to.

    In 'hour' mode the name has a today-/tomorrow- prefix. The night after
    today's evening (midnight to 6am on the next calendar day) is still
    "today-night" ("tonight"); a forecast starting during the night calls the
    following night "tomorrow-night".

    Parameters:
    - local_time (datetime.datetime): The local time of the hour.
    - start_time (datetime.datetime): The local time of the first forecast hour.
    - mode (str): "daily"/"day" or "hour".

    Returns:
    - str: The period name (e.g. "morning", "tomorrow-night").
    """
    hour = local_time.hour
    if hour < MORNING_START:
        period = "night"
    elif hour < AFTERNOON_START:
        period = "morning"
    elif hour < EVENING_START:
        period = "afternoon"
    else:
        period = "evening"

    if mode != "hour":
        return period

    is_today = local_time.date() == start_time.date() or (
        period == "night" and start_time.hour >= MORNING_START
    )
    return ("today-" if is_today else "tomorrow-") + period


def _join_with_and(items):
    """Join items as nested "and" phrases: [a, b, c] -> ["and", a, ["and", b, c]]."""
    joined = items[-1]
    for item in reversed(items[:-1]):
        joined = ["and", item, joined]
    return joined


def _get_time_phrase(period_indices, all_periods):
    """
    Describes when a condition happens, given the periods it occurs in.

    The periods are split into runs of consecutive periods. Each run is then
    described relative to the start and end of the window:

        whole window                       ["for-day"]
        run from the start and a run to    ["until-starting-again", first dry, first wet again]
        the end (dry in between)
        only single, separate periods      ["during", P] / ["during", ["and", P1, P2]]
        otherwise each run is described and the runs are joined with "and":
            run of one period              ["during", P]
            run from the start             ["until", first dry period]
            run to the end                 ["starting", first wet period]
            run in the middle              ["starting-continuing-until", first wet, first dry]

    For example, rain in the morning and afternoon of a daily summary is
    ["until", "evening"]: "rain until evening".

    Parameters:
    - period_indices (list): Sorted indices of the periods with the condition.
    - all_periods (list): The period names of the window, in order.

    Returns:
    - list | None: The time phrase, or None if the condition never happens.
    """
    if not period_indices:
        return None

    num_periods = len(all_periods)
    if len(period_indices) == num_periods:
        return ["for-day"]

    # Runs of consecutive periods, e.g. [0, 1, 3] -> [[0, 1], [3]]
    runs = [[period_indices[0]]]
    for idx in period_indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])

    if len(runs) == 2 and runs[0][0] == 0 and runs[1][-1] == num_periods - 1:
        return [
            "until-starting-again",
            all_periods[runs[0][-1] + 1],
            all_periods[runs[1][0]],
        ]

    if all(len(run) == 1 for run in runs):
        return ["during", _join_with_and([all_periods[run[0]] for run in runs])]

    run_phrases = []
    for run in runs:
        first, last = run[0], run[-1]
        if first == last:
            run_phrases.append(["during", all_periods[first]])
        elif first == 0:
            run_phrases.append(["until", all_periods[last + 1]])
        elif last == num_periods - 1:
            run_phrases.append(["starting", all_periods[first]])
        else:
            run_phrases.append(
                ["starting-continuing-until", all_periods[first], all_periods[last + 1]]
            )
    return _join_with_and(run_phrases)


def _insert_condition_text(time_phrase, condition_text):
    """
    Puts the condition text into a time phrase, after the phrase type.

    ["until", "evening"] -> ["until", "light-rain", "evening"]. Phrases joined
    with "and" each get the condition text.
    """
    if time_phrase[0] == "and":
        return [
            "and",
            _insert_condition_text(time_phrase[1], condition_text),
            _insert_condition_text(time_phrase[2], condition_text),
        ]
    return [time_phrase[0], condition_text] + time_phrase[1:]


def calculate_period_summary_text(
    period_indices,
    condition_text,
    condition_type,
    all_periods,
    wind_periods,
    vis_periods,
    max_wind_speed,
    icon_set,
    half_day=False,
    overall_cloud_text=None,
    overall_cloud_idx=None,
):
    """
    Calculates the textual summary for a specific condition (precip, cloud, wind, vis)
    across a set of periods.
    Wind speed is expected in SI units (m/s).

    Wind and fog that happen in exactly the same periods as precipitation or
    cloud are combined into that summary ("rain and windy"), and wind that
    matches the cloud cover is combined with it.

    Parameters:
    - period_indices (list): Indices of the periods with the condition.
    - condition_text (str | list): The text for the condition (e.g., "light-rain", "fog").
    - condition_type (str): The type of condition ("precip", "cloud", "wind", "vis").
    - all_periods (list): The period names of the window.
    - wind_periods (list): Indices of periods with significant wind.
    - vis_periods (list): Indices of periods with fog.
    - max_wind_speed (float): Maximum wind speed across all periods (m/s).
    - icon_set (str): Which icon set to use - Dark Sky or Pirate Weather.
    - half_day (bool): Whether this is a day/night (half-day) summary.
    - overall_cloud_text (str, optional): The overall cloud text, for wind to combine with.
    - overall_cloud_idx (list, optional): The indices of periods with the overall cloud cover.

    Returns:
    - tuple: A tuple containing:
        - summary_text (list | str | None): The summary of the condition.
        - wind_condition_combined (bool): True if wind was combined with this condition.
        - vis_condition_combined (bool): True if visibility was combined with this condition.
    """
    wind_condition_combined = False
    vis_condition_combined = False

    if condition_type in ("precip", "cloud"):
        if wind_periods and period_indices == wind_periods:
            wind_condition_combined = True
            condition_text = [
                "and",
                condition_text,
                calculate_wind_text(max_wind_speed, icon_set, "summary"),
            ]
        if vis_periods and period_indices == vis_periods:
            vis_condition_combined = True
            condition_text = ["and", condition_text, "fog"]
    elif condition_type == "wind":
        if overall_cloud_idx and period_indices == overall_cloud_idx:
            wind_condition_combined = True
            condition_text = ["and", overall_cloud_text, condition_text]

    time_phrase = _get_time_phrase(period_indices, all_periods)

    if time_phrase is None:
        summary_text = None
    elif time_phrase == ["for-day"]:
        # Half-day summaries describe the whole half-day with the bare condition
        summary_text = condition_text if half_day else ["for-day", condition_text]
    else:
        summary_text = _insert_condition_text(time_phrase, condition_text)

    return summary_text, wind_condition_combined, vis_condition_combined


def _new_period_data():
    """Return the empty aggregation for one standard period."""
    return {
        "num_hours_fog": 0,
        "num_hours_wind": 0,
        "num_hours_thunderstorm": 0,
        "rain_accum": 0.0,
        "snow_accum": 0.0,
        "snow_error": 0.0,
        "has_snow_error_data": False,  # Track if any error data exists
        "sleet_accum": 0.0,
        "frzrain_accum": 0.0,
        "max_pop": 0.0,
        "max_rain_intensity": 0.0,
        "max_snow_intensity": 0.0,
        "max_ice_intensity": 0.0,
        "cloud_cover_sum": 0.0,
        "max_wind_speed": 0.0,
        "period_length": 0,
        "precip_types_in_period": [],
        "avg_cloud_cover": 0.0,
        "min_visibility": float("inf"),
        "max_smoke": 0.0,
        "max_cape_with_precip": 0.0,
        "min_li_with_precip": None,  # Most-unstable (minimum) Lifted Index during precip
        "max_cin_with_precip": None,  # Least-inhibiting (maximum) CIN during precip
        "max_ki_with_precip": None,  # Maximum K Index during precip
        "max_vert_vel_with_precip": None,  # Strongest upward (minimum) vertical velocity during precip
        "max_dewpoint_depression": None,  # Maximum dewpoint depression
        "temp_at_max_depression": None,  # Temperature at max depression
        "dewpoint_at_max_depression": None,  # Dewpoint at max depression
    }


def _sanitize_hour(hour):
    """Copy an hour, replacing missing optional values with their defaults.

    The copy keeps the defaults from leaking back into the caller's data.
    """
    sanitized_hour = dict(hour)
    sanitized_hour["humidity"] = _value_or_default(
        hour.get("humidity", DEFAULT_HUMIDITY), DEFAULT_HUMIDITY
    )
    sanitized_hour["visibility"] = _value_or_default(
        hour.get("visibility", DEFAULT_VISIBILITY), DEFAULT_VISIBILITY
    )
    sanitized_hour["precipIntensityError"] = _value_or_default(
        hour.get("precipIntensityError", np.nan), np.nan
    )
    sanitized_hour["precipProbability"] = _value_or_default(
        hour.get("precipProbability", DEFAULT_POP), DEFAULT_POP
    )
    sanitized_hour["smoke"] = _value_or_default(hour.get("smoke", 0.0), 0.0)

    # Without a dew point assume saturation (dew point equal to the temperature)
    dew_point_default = hour.get("temperature")
    sanitized_hour["dewPoint"] = _value_or_default(
        hour.get("dewPoint", dew_point_default), dew_point_default
    )
    return sanitized_hour


def _add_hour_to_period(period_data, hour, icon_set):
    """
    Accumulates one hour of forecast data into its period.

    Parameters:
    - period_data (dict): The period aggregation from _new_period_data.
    - hour (dict): A sanitized hourly forecast (SI units).
    - icon_set (str): Which icon set to use - Dark Sky or Pirate Weather.

    Returns:
    - float | None: The temperature-dewpoint spread if the hour is foggy, otherwise None.
    """
    period_data["period_length"] += 1
    period_data["cloud_cover_sum"] += hour["cloudCover"]
    period_data["max_wind_speed"] = max(
        period_data["max_wind_speed"], hour["windSpeed"]
    )
    period_data["max_rain_intensity"] = max(
        period_data["max_rain_intensity"], hour["rainIntensity"]
    )
    period_data["max_snow_intensity"] = max(
        period_data["max_snow_intensity"], hour["snowIntensity"]
    )
    period_data["max_ice_intensity"] = max(
        period_data["max_ice_intensity"], hour["iceIntensity"]
    )
    period_data["min_visibility"] = min(
        period_data["min_visibility"], hour["visibility"]
    )

    fog_spread = None
    if (
        calculate_vis_text(
            hour["visibility"],
            hour["temperature"],
            hour["dewPoint"],
            hour["windSpeed"],
            hour["smoke"],
            icon_set,
            "icon",
        )
        is not None
        and hour["rainIntensity"] <= FOG_MAX_LIQUID_INTENSITY
        and hour["snowIntensity"] <= FOG_MAX_SNOW_INTENSITY
        and hour["iceIntensity"] <= FOG_MAX_LIQUID_INTENSITY
    ):
        period_data["max_smoke"] = max(period_data["max_smoke"], hour["smoke"])
        period_data["num_hours_fog"] += 1
        fog_spread = abs(hour["temperature"] - hour["dewPoint"])

    # Count hours at or above the light wind threshold (NaN wind never counts)
    if hour["windSpeed"] >= WIND_THRESHOLDS["light"]:
        period_data["num_hours_wind"] += 1

    period_data["rain_accum"] += hour["liquidAccumulation"]
    period_data["snow_accum"] += hour["snowAccumulation"]
    # Only accumulate error if it's not nan (missing data), the precipitation
    # type is snow and there is any accumulation.
    # precipIntensityError is an intensity error (mm/h). Convert that
    # to a liquid accumulation error for the hour (mm) then convert
    # to snow depth (mm) via estimate_snow_height. If temperature is
    # missing, fall back to using error directly.
    if (
        not math.isnan(hour["precipIntensityError"])
        and hour.get("precipType") == PRECIP_TYPES["snow"]
        and hour.get("snowAccumulation", 0.0) > 0.0
    ):
        liquid_error_mm = hour["precipIntensityError"] * 1.0
        temp = hour.get("temperature", MISSING_DATA)
        if temp is None or (isinstance(temp, float) and math.isnan(temp)):
            estimated_snow_error_mm = liquid_error_mm
        else:
            estimated_snow_error_mm = estimate_snow_height(
                liquid_error_mm, temp, hour.get("windSpeed", 0.0)
            )
        period_data["snow_error"] += estimated_snow_error_mm
        period_data["has_snow_error_data"] = True

    if hour["precipType"] == PRECIP_TYPES["ice"]:
        period_data["frzrain_accum"] += hour["iceAccumulation"]
    elif hour["precipType"] == PRECIP_TYPES["sleet"]:
        period_data["sleet_accum"] += hour["iceAccumulation"]

    if not (
        hour["liquidAccumulation"] > 0
        or hour["snowAccumulation"] > 0
        or hour["iceAccumulation"] > 0
    ):
        return fog_spread

    period_data["precip_types_in_period"].append(hour["precipType"])
    period_data["max_pop"] = max(period_data["max_pop"], hour["precipProbability"])

    # Track stability indices for hours with likely precipitation. Missing CAPE
    # is left in: it never raises the max() and the thunderstorm score can
    # still use the other indices.
    hour_pop = hour.get("precipProbability", DEFAULT_POP)
    if hour_pop < PRECIP_PROB_THRESHOLD:
        return fog_spread

    hour_cape = hour.get("cape", MISSING_DATA)
    hour_li = hour.get("liftedIndex", None)
    hour_cin = hour.get("convectiveInhibition", None)
    hour_vert_vel = hour.get("verticalVelocity", None)
    hour_ki = hour.get("kIndex", None)
    hour_temp = hour.get("temperature", None)
    hour_dew = hour.get("dewPoint", None)

    period_data["max_cape_with_precip"] = max(
        period_data["max_cape_with_precip"], hour_cape
    )

    # Record highest dewpoint depression
    if (
        hour_temp is not None
        and not math.isnan(hour_temp)
        and hour_dew is not None
        and not math.isnan(hour_dew)
    ):
        hour_dep = hour_temp - hour_dew
        if (
            period_data["max_dewpoint_depression"] is None
            or hour_dep > period_data["max_dewpoint_depression"]
        ):
            period_data["max_dewpoint_depression"] = hour_dep
            period_data["temp_at_max_depression"] = hour_temp
            period_data["dewpoint_at_max_depression"] = hour_dew

    # Most-unstable (minimum) Lifted Index
    if (
        hour_li is not None
        and not math.isnan(hour_li)
        and (
            period_data["min_li_with_precip"] is None
            or hour_li < period_data["min_li_with_precip"]
        )
    ):
        period_data["min_li_with_precip"] = hour_li

    # Strongest upward (minimum) vertical velocity
    if (
        hour_vert_vel is not None
        and not math.isnan(hour_vert_vel)
        and (
            period_data["max_vert_vel_with_precip"] is None
            or hour_vert_vel < period_data["max_vert_vel_with_precip"]
        )
    ):
        period_data["max_vert_vel_with_precip"] = hour_vert_vel

    # Least-inhibiting (maximum/least-negative) CIN
    if (
        hour_cin is not None
        and not math.isnan(hour_cin)
        and (
            period_data["max_cin_with_precip"] is None
            or hour_cin > period_data["max_cin_with_precip"]
        )
    ):
        period_data["max_cin_with_precip"] = hour_cin

    # Maximum K Index
    if (
        hour_ki is not None
        and not math.isnan(hour_ki)
        and (
            period_data["max_ki_with_precip"] is None
            or hour_ki > period_data["max_ki_with_precip"]
        )
    ):
        period_data["max_ki_with_precip"] = hour_ki

    # Count hours with thunderstorms using all available stability indices
    if (
        calculate_thunderstorm_text(
            hour_cape,
            mode="summary",
            pop=hour_pop,
            lifted_index=hour_li,
            cin=hour_cin,
            vertical_velocity=hour_vert_vel,
            k_index=hour_ki,
            dewpoint=hour_dew,
            temperature=hour_temp,
        )
        is not None
    ):
        period_data["num_hours_thunderstorm"] += 1

    return fog_spread


def _snow_accumulation_sentence(
    total_snow_accum, total_snow_error, has_any_snow_error_data, unit_system
):
    """
    Builds the snow accumulation (or accumulation range) for the summary.

    Parameters:
    - total_snow_accum (float): Total snow accumulation in mm.
    - total_snow_error (float): Total snow accumulation error in mm.
    - has_any_snow_error_data (bool): Whether any hour had error data.
    - unit_system (str): Unit system for display ("us", "si", "ca", "uk").

    Returns:
    - list | None: The snow accumulation text, or None if there is nothing to show.
    """
    if unit_system == "us":
        snow_unit_str = "inches"
        mm_per_unit = MM_PER_INCH
    else:  # si, ca, uk use centimeters
        snow_unit_str = "centimeters"
        mm_per_unit = MM_PER_CENTIMETER
    snow_accum_display = total_snow_accum / mm_per_unit
    snow_error_display = total_snow_error / mm_per_unit
    less_than_tolerance = LESS_THAN_TOLERANCE / mm_per_unit

    # If error data is missing (ERA5, etc.), show exact value without range or "<"
    if not has_any_snow_error_data and snow_accum_display > 0:
        return [snow_unit_str, math.ceil(snow_accum_display)]

    # Error data exists - calculate range. Snow accumulation cannot be negative.
    snow_low_accum = max(0, math.floor(snow_accum_display - (snow_error_display / 2)))
    snow_max_accum = math.ceil(snow_accum_display + (snow_error_display / 2))

    if snow_max_accum >= less_than_tolerance and snow_low_accum == 0:
        snow_low_accum = 1

    if snow_max_accum <= 0:
        return None
    if snow_accum_display == 0:
        return ["less-than", [snow_unit_str, 1]]
    # If error is very small or lower range is 0, use less-than format
    if snow_error_display <= 0.01 or snow_low_accum == 0:
        return ["less-than", [snow_unit_str, snow_max_accum]]
    return [snow_unit_str, ["range", snow_low_accum, snow_max_accum]]


def calculate_day_text(
    hours,
    is_day_time,
    time_zone,
    mode="daily",
    icon_set="darksky",
    unit_system="si",
    half_day=False,
):
    """
    Calculates the daily, next 24-hour or half-day weather summary text.
    All inputs are expected in SI units:
    - Temperature in Celsius
    - Wind speed in m/s
    - Visibility in meters
    - Precipitation in mm/h (intensity) and mm (accumulation)
    Display values are converted based on unit_system.

    Parameters:
    - hours (list): An array of hourly forecast data (in SI units).
    - is_day_time (bool): Whether it's currently daytime.
    - time_zone (str): The timezone for the current location.
    - mode (str, optional): Which mode to run the function in ("daily"/"day" or "hour"). Defaults to "daily".
    - icon_set (str): Which icon set to use - Dark Sky or Pirate Weather
    - unit_system (str): Unit system for display ("us", "si", "ca", "uk")
    - half_day (bool): Summarise a day/night (half-day) window instead of a full day.

    Returns:
    - tuple: A tuple containing:
        - c_icon (str): The icon representing the period.
        - summary_text (list | str): The textual representation of the period.
    """
    # Return "unavailable" if too much data is provided
    if len(hours) > (MAX_HALF_DAY_HOURS if half_day else MAX_HOURS):
        return "none", ("unavailable" if half_day else ["for-day", "unavailable"])

    # Half-day summaries treat light rain/sleet/freezing rain a little more strictly
    precip_text_threshold = (
        HALF_DAY_PRECIP_ACCUM_TEXT_THRESHOLD_MM
        if half_day
        else DAILY_PRECIP_ACCUM_TEXT_THRESHOLD_MM
    )

    hours = [_sanitize_hour(hour) for hour in hours]

    # --- 1. Split the hours into periods and 2. aggregate each period ---
    zone = _get_zone(time_zone)
    start_time = datetime.datetime.fromtimestamp(hours[0]["time"], zone)

    # Periods in forecast order (dicts keep insertion order)
    periods_data = {}
    overall_min_temp_dewpoint_spread = float("inf")
    overall_temp_at_min_spread = 0.0
    overall_dewpoint_at_min_spread = 0.0

    for hour in hours:
        local_time = datetime.datetime.fromtimestamp(hour["time"], zone)
        period_name = _get_period_name(local_time, start_time, mode)
        if period_name not in periods_data:
            periods_data[period_name] = _new_period_data()

        fog_spread = _add_hour_to_period(periods_data[period_name], hour, icon_set)
        # Keep the temperature and dew point of the foggiest hour for the fog text
        if fog_spread is not None and fog_spread < overall_min_temp_dewpoint_spread:
            overall_min_temp_dewpoint_spread = fog_spread
            overall_temp_at_min_spread = hour["temperature"]
            overall_dewpoint_at_min_spread = hour["dewPoint"]

    all_period_names = list(periods_data)
    period_stats = list(periods_data.values())
    for p_data in period_stats:
        p_data["avg_cloud_cover"] = p_data["cloud_cover_sum"] / p_data["period_length"]

    # --- 3. Which periods have each condition, and totals for the whole window ---
    precip_periods = []
    thunderstorm_periods = []
    vis_periods = []
    wind_periods = []
    cloud_levels = []

    total_rain_accum = 0.0
    total_snow_accum = 0.0
    total_sleet_accum = 0.0
    total_frzrain_accum = 0.0
    total_snow_error = 0.0
    has_any_snow_error_data = False
    overall_max_rain_intensity = 0.0
    overall_max_snow_intensity = 0.0
    overall_max_ice_intensity = 0.0
    overall_max_wind = 0.0
    overall_avg_cloud_cover_sum = 0.0  # Sum of average cloud cover for each period
    overall_max_pop = 0.0
    overall_min_visibility = float("inf")
    overall_max_smoke = 0.0
    # Peak instability that occurs with precipitation, and the temperature and
    # dew point from the period with the highest CAPE
    overall_max_cape_with_precip = 0.0
    overall_min_li_with_precip = None
    overall_max_cin_with_precip = None
    overall_max_ki_with_precip = None
    overall_max_vert_vel_with_precip = None
    overall_temp_at_max_instability = None
    overall_dewpoint_at_max_instability = None
    overall_precip_types = []

    for i, p_data in enumerate(period_stats):
        total_rain_accum += p_data["rain_accum"]
        total_snow_accum += p_data["snow_accum"]
        total_sleet_accum += p_data["sleet_accum"]
        total_frzrain_accum += p_data["frzrain_accum"]
        total_snow_error += p_data["snow_error"]
        if p_data["has_snow_error_data"]:
            has_any_snow_error_data = True
        overall_max_rain_intensity = max(
            overall_max_rain_intensity, p_data["max_rain_intensity"]
        )
        overall_max_snow_intensity = max(
            overall_max_snow_intensity, p_data["max_snow_intensity"]
        )
        overall_max_ice_intensity = max(
            overall_max_ice_intensity, p_data["max_ice_intensity"]
        )
        overall_max_wind = max(overall_max_wind, p_data["max_wind_speed"])
        overall_avg_cloud_cover_sum += p_data["avg_cloud_cover"]
        overall_min_visibility = min(overall_min_visibility, p_data["min_visibility"])
        overall_max_smoke = max(overall_max_smoke, p_data["max_smoke"])

        # Check if precipitation is significant enough in this period (thresholds in mm)
        is_precip_in_period = (
            p_data["snow_accum"] > PRECIP_INTENSITY_THRESHOLDS["mid"]
            or p_data["rain_accum"] > precip_text_threshold
            or p_data["sleet_accum"] > precip_text_threshold
            or p_data["frzrain_accum"] > precip_text_threshold
        )
        if is_precip_in_period:
            precip_periods.append(i)
            overall_precip_types.extend(p_data["precip_types_in_period"])
            overall_max_pop = max(overall_max_pop, p_data["max_pop"])

            if p_data["max_cape_with_precip"] > overall_max_cape_with_precip:
                overall_max_cape_with_precip = p_data["max_cape_with_precip"]
                overall_temp_at_max_instability = p_data["temp_at_max_depression"]
                overall_dewpoint_at_max_instability = p_data[
                    "dewpoint_at_max_depression"
                ]

            if p_data["min_li_with_precip"] is not None and (
                overall_min_li_with_precip is None
                or p_data["min_li_with_precip"] < overall_min_li_with_precip
            ):
                overall_min_li_with_precip = p_data["min_li_with_precip"]

            if p_data["max_cin_with_precip"] is not None and (
                overall_max_cin_with_precip is None
                or p_data["max_cin_with_precip"] > overall_max_cin_with_precip
            ):
                overall_max_cin_with_precip = p_data["max_cin_with_precip"]

            if p_data["max_vert_vel_with_precip"] is not None and (
                overall_max_vert_vel_with_precip is None
                or p_data["max_vert_vel_with_precip"] < overall_max_vert_vel_with_precip
            ):
                overall_max_vert_vel_with_precip = p_data["max_vert_vel_with_precip"]

            if p_data["max_ki_with_precip"] is not None and (
                overall_max_ki_with_precip is None
                or p_data["max_ki_with_precip"] > overall_max_ki_with_precip
            ):
                overall_max_ki_with_precip = p_data["max_ki_with_precip"]

        # Thunderstorms require both precipitation and sufficient atmospheric instability
        if is_precip_in_period and p_data["num_hours_thunderstorm"] >= min(
            p_data["period_length"] / 2, SIGNIFICANT_THUNDERSTORM_MAX_HOURS
        ):
            thunderstorm_periods.append(i)

        significant_hours = min(
            p_data["period_length"] / 2, SIGNIFICANT_CONDITION_MAX_HOURS
        )
        if p_data["num_hours_wind"] >= significant_hours:
            wind_periods.append(i)
        # Fog needs calm, dry hours
        if (
            p_data["max_rain_intensity"] < FOG_MAX_LIQUID_INTENSITY
            and p_data["max_snow_intensity"] < FOG_MAX_SNOW_INTENSITY
            and p_data["max_ice_intensity"] < FOG_MAX_LIQUID_INTENSITY
            and p_data["max_wind_speed"] < WIND_THRESHOLDS["light"]
            and p_data["num_hours_fog"] >= significant_hours
        ):
            vis_periods.append(i)

        _, cloud_level = calculate_cloud_text(p_data["avg_cloud_cover"])
        cloud_levels.append(cloud_level)

    overall_avg_cloud_cover = (
        overall_avg_cloud_cover_sum / len(period_stats) if period_stats else 0
    )

    # --- 3b. Overall cloud cover text and icon ---
    final_cloud_text = "clear"
    derived_avg_cloud_for_icon = 0.0
    overall_cloud_idx = []

    if cloud_levels:
        # The periods with the most common cloud level (the first level on a tie)
        most_common_cloud_level = most_common(cloud_levels)
        overall_cloud_idx = [
            idx
            for idx, level in enumerate(cloud_levels)
            if level == most_common_cloud_level
        ]

        if len(cloud_levels) > 1 and len(set(cloud_levels)) == len(cloud_levels):
            # All cloud levels are different: use the period with the highest
            # average cloud cover (the first one on a tie, skipping missing data)
            cloudiest_idx = None
            max_avg_cloud_value = -1.0
            for idx, p_data in enumerate(period_stats):
                if p_data["avg_cloud_cover"] > max_avg_cloud_value:
                    max_avg_cloud_value = p_data["avg_cloud_cover"]
                    cloudiest_idx = idx

            if cloudiest_idx is not None:
                final_cloud_text, _ = calculate_cloud_text(max_avg_cloud_value)
                derived_avg_cloud_for_icon = max_avg_cloud_value
                overall_cloud_idx = [cloudiest_idx]
            else:  # Every period is missing cloud cover
                final_cloud_text, _ = calculate_cloud_text(overall_avg_cloud_cover)
                derived_avg_cloud_for_icon = overall_avg_cloud_cover
        else:
            # Otherwise use the most common cloud level, with a representative
            # cloud cover for the icon
            final_cloud_text, derived_avg_cloud_for_icon = CLOUD_LEVEL_TEXT_AND_COVER[
                most_common_cloud_level
            ]

    # With a single period use its actual average cloud cover for the icon
    if len(period_stats) == 1:
        derived_avg_cloud_for_icon = period_stats[0]["avg_cloud_cover"]

    # --- 4. Precipitation text and icon for the whole window ---
    most_common_overall_precip_type = (
        most_common(overall_precip_types) if overall_precip_types else "none"
    )

    precip_summary_text = None
    precip_icon = None
    secondary_precip_condition = None

    total_precip_accum = (
        total_rain_accum + total_snow_accum + total_sleet_accum + total_frzrain_accum
    )
    # Combined ice accumulation (sleet + freezing rain) used for mixed/threshold checks
    total_ice_accum = total_sleet_accum + total_frzrain_accum

    if overall_max_pop > 0 and total_precip_accum >= PRECIP_SUMMARY_MIN_ACCUM_MM:
        if total_snow_accum > 0 and total_rain_accum > 0 and total_ice_accum > 0:
            precip_summary_text = "mixed-precipitation"
            most_common_overall_precip_type = (
                PRECIP_TYPES["sleet"] if icon_set != "pirate" else PRECIP_TYPES["mixed"]
            )
            secondary_precip_condition = (
                "medium-snow"  # Indicate snow totals are relevant
            )
        else:
            # Determine dominant ice type (sleet vs freezing rain) and labels for secondary conditions
            if total_frzrain_accum >= total_sleet_accum:
                dominant_ice_type = PRECIP_TYPES["ice"]
                ice_as_secondary = (
                    "medium-freezing-rain"  # dominant ice as secondary to snow/rain
                )
                other_ice_as_secondary = (
                    "medium-sleet"  # non-dominant ice alongside dominant ice
                )
            else:
                dominant_ice_type = PRECIP_TYPES["sleet"]
                ice_as_secondary = "medium-sleet"
                other_ice_as_secondary = "medium-freezing-rain"

            # Determine primary and secondary precipitation types based on accumulation
            if total_snow_accum > 0:
                if total_rain_accum > 0 and total_snow_accum > total_rain_accum:
                    most_common_overall_precip_type = PRECIP_TYPES["snow"]
                    secondary_precip_condition = "medium-rain"
                elif total_rain_accum > 0 and total_snow_accum < total_rain_accum:
                    most_common_overall_precip_type = PRECIP_TYPES["rain"]
                    secondary_precip_condition = "medium-snow"
                elif total_ice_accum > 0 and total_snow_accum > total_ice_accum:
                    most_common_overall_precip_type = PRECIP_TYPES["snow"]
                    secondary_precip_condition = ice_as_secondary
                elif total_ice_accum > 0 and total_snow_accum < total_ice_accum:
                    most_common_overall_precip_type = dominant_ice_type
                    secondary_precip_condition = "medium-snow"
            elif total_ice_accum > 0:
                if total_rain_accum > 0 and total_rain_accum > total_ice_accum:
                    most_common_overall_precip_type = PRECIP_TYPES["rain"]
                    secondary_precip_condition = ice_as_secondary
                elif total_rain_accum > 0 and total_rain_accum < total_ice_accum:
                    most_common_overall_precip_type = dominant_ice_type
                    secondary_precip_condition = "medium-rain"
                elif total_sleet_accum > 0 and total_frzrain_accum > 0:
                    # Both sleet and freezing rain on the same day (with or without rain);
                    # this branch is reached when there is no rain or rain == total_ice_accum
                    most_common_overall_precip_type = dominant_ice_type
                    secondary_precip_condition = other_ice_as_secondary

            # Re-evaluate primary precipType if calculated type has zero accumulation
            if (
                total_snow_accum == 0
                and most_common_overall_precip_type == PRECIP_TYPES["snow"]
            ):
                if total_rain_accum > 0:
                    most_common_overall_precip_type = PRECIP_TYPES["rain"]
                elif total_ice_accum > 0:
                    most_common_overall_precip_type = dominant_ice_type
            elif (
                total_rain_accum == 0
                and most_common_overall_precip_type == PRECIP_TYPES["rain"]
            ):
                if total_snow_accum > 0:
                    most_common_overall_precip_type = PRECIP_TYPES["snow"]
                elif total_ice_accum > 0:
                    most_common_overall_precip_type = dominant_ice_type
            elif total_ice_accum == 0 and most_common_overall_precip_type in (
                PRECIP_TYPES["sleet"],
                PRECIP_TYPES["ice"],
            ):
                if total_snow_accum > 0:
                    most_common_overall_precip_type = PRECIP_TYPES["snow"]
                elif total_rain_accum > 0:
                    most_common_overall_precip_type = PRECIP_TYPES["rain"]

        # Calculate final precipitation text and icon (all in mm)
        precip_summary_text, precip_icon = calculate_precip_text(
            most_common_overall_precip_type,
            "hourly",
            total_rain_accum,
            total_snow_accum,
            total_ice_accum,
            overall_max_pop,
            icon_set,
            "both",
            isDayTime=is_day_time,
            eff_rain_intensity=overall_max_rain_intensity,
            eff_snow_intensity=overall_max_snow_intensity,
            eff_ice_intensity=overall_max_ice_intensity,
        )

    # Correct "medium-none" secondary condition to "medium-precipitation"
    if secondary_precip_condition == "medium-none":
        secondary_precip_condition = "medium-precipitation"
    # Correct medium-ice secondary condition to "medium-freezing-rain"
    if secondary_precip_condition == "medium-ice":
        secondary_precip_condition = "medium-freezing-rain"

    # Add snow accumulation range to precip text if applicable
    if (
        total_snow_accum > SNOW_SENTENCE_MIN_ACCUM_MM
        or secondary_precip_condition == "medium-snow"
    ) and (
        most_common_overall_precip_type == PRECIP_TYPES["snow"]
        or secondary_precip_condition == "medium-snow"
    ):
        snow_sentence = _snow_accumulation_sentence(
            total_snow_accum, total_snow_error, has_any_snow_error_data, unit_system
        )
        if snow_sentence is not None:
            precip_summary_text = ["parenthetical", precip_summary_text, snow_sentence]

    # Combine primary and secondary precipitation conditions with "and"
    if (
        secondary_precip_condition is not None
        and secondary_precip_condition != "medium-snow"
    ):
        precip_summary_text = ["and", precip_summary_text, secondary_precip_condition]

    # Flags to indicate if a condition is present at all in the forecast block
    has_precip = bool(precip_periods) and precip_summary_text is not None
    has_thunderstorm = bool(thunderstorm_periods)
    has_wind = bool(wind_periods)
    has_vis = bool(vis_periods)

    # --- 4b. Thunderstorm text, replacing the precipitation text when they match ---
    thunderstorm_summary_text = None
    thunderstorm_icon = None
    thunderstorms_match_precip = False

    if has_thunderstorm:
        # Use peak instability values that occurred with precipitation
        thunderstorm_summary_text, thunderstorm_icon = calculate_thunderstorm_text(
            overall_max_cape_with_precip,
            mode="both",
            pop=overall_max_pop,
            lifted_index=overall_min_li_with_precip,
            cin=overall_max_cin_with_precip,
            vertical_velocity=overall_max_vert_vel_with_precip,
            k_index=overall_max_ki_with_precip,
            dewpoint=overall_dewpoint_at_max_instability,
            temperature=overall_temp_at_max_instability,
        )
        # Thunderstorms in exactly the precipitation periods replace the precipitation text
        if sorted(thunderstorm_periods) == sorted(precip_periods):
            thunderstorms_match_precip = True
            if precip_summary_text and thunderstorm_summary_text:
                precip_summary_text = thunderstorm_summary_text
                if thunderstorm_icon:
                    precip_icon = thunderstorm_icon

    # --- 5. Time phrase summary for each condition ---
    precip_only_summary = None
    thunderstorm_only_summary = None
    wind_only_summary = None
    vis_only_summary = None
    # Track which conditions have been "subsumed" by the precipitation summary
    combined_vis_flag = False
    combined_wind_flag = False

    if has_precip:
        (
            precip_only_summary,
            combined_wind_flag,
            combined_vis_flag,
        ) = calculate_period_summary_text(
            precip_periods,
            precip_summary_text,
            "precip",
            all_period_names,
            wind_periods,
            vis_periods,
            overall_max_wind,
            icon_set,
            half_day,
        )

    # Thunderstorms that don't match the precipitation periods get their own summary
    if has_thunderstorm and not thunderstorms_match_precip:
        thunderstorm_only_summary, _, _ = calculate_period_summary_text(
            thunderstorm_periods,
            thunderstorm_summary_text,
            "precip",  # Use precip type since thunderstorms have same priority
            all_period_names,
            wind_periods,
            vis_periods,
            overall_max_wind,
            icon_set,
            half_day,
        )

    if has_wind:
        wind_only_summary, _, _ = calculate_period_summary_text(
            wind_periods,
            calculate_wind_text(overall_max_wind, icon_set, "summary"),
            "wind",
            all_period_names,
            [],
            [],
            overall_max_wind,
            icon_set,
            half_day,
            overall_cloud_text=final_cloud_text,
            overall_cloud_idx=overall_cloud_idx,
        )

    if has_vis:
        vis_only_summary, _, _ = calculate_period_summary_text(
            vis_periods,
            calculate_vis_text(
                overall_min_visibility,
                overall_temp_at_min_spread,
                overall_dewpoint_at_min_spread,
                0,
                overall_max_smoke,
                icon_set,
                "summary",
            ),
            "vis",
            all_period_names,
            [],
            [],
            overall_max_wind,
            icon_set,
            half_day,
        )

    # --- 6. Keep the top 2 conditions and join them ---
    # Priority order (lower number = higher priority):
    # 0: Precipitation and Thunderstorms (same priority)
    # 1: Visibility (Fog)
    # 2: Wind
    # 3: Cloud (fallback)
    num_periods = len(period_stats)
    candidate_summaries = []

    if precip_only_summary:
        candidate_summaries.append(
            {
                "priority": 0,
                "all_day": len(precip_periods) == num_periods,
                "start_idx": precip_periods[0],
                "text": precip_only_summary,
                "icon": precip_icon,
            }
        )

    if thunderstorm_only_summary:
        candidate_summaries.append(
            {
                "priority": 0,
                "all_day": len(thunderstorm_periods) == num_periods,
                "start_idx": thunderstorm_periods[0],
                "text": thunderstorm_only_summary,
                "icon": thunderstorm_icon,
            }
        )

    # Fog, unless already covered by the precipitation summary
    if has_vis and not combined_vis_flag:
        candidate_summaries.append(
            {
                "priority": 1,
                "all_day": len(vis_periods) == num_periods,
                "start_idx": vis_periods[0],
                "text": vis_only_summary,
                # Fog periods are already calm, so ignore wind from the rest of
                # the window, as the fog text does
                "icon": calculate_vis_text(
                    overall_min_visibility,
                    overall_temp_at_min_spread,
                    overall_dewpoint_at_min_spread,
                    0,
                    overall_max_smoke,
                    icon_set,
                    "icon",
                ),
            }
        )

    # Wind, unless already covered by the precipitation summary
    if has_wind and not combined_wind_flag:
        candidate_summaries.append(
            {
                "priority": 2,
                "all_day": len(wind_periods) == num_periods,
                "start_idx": wind_periods[0],
                "text": wind_only_summary,
                "icon": calculate_wind_text(overall_max_wind, icon_set, "icon"),
            }
        )

    # Cloud cover as a fallback if no other condition is present
    if not candidate_summaries:
        # Cloud full summary, including potential combinations with wind/vis
        cloud_full_summary, _, _ = calculate_period_summary_text(
            overall_cloud_idx,
            final_cloud_text,
            "cloud",
            all_period_names,
            wind_periods,
            vis_periods,
            overall_max_wind,
            icon_set,
            half_day,
        )
        candidate_summaries.append(
            {
                "priority": 3,
                "all_day": len(overall_cloud_idx) == num_periods,
                "start_idx": 0,  # Cloud is always "present" from the start of the forecast
                "text": cloud_full_summary,
                "icon": calculate_sky_icon(
                    derived_avg_cloud_for_icon, is_day_time, icon_set
                ),
            }
        )

    # Sort candidates: all-day conditions first, then the earliest start, then priority
    candidate_summaries.sort(
        key=lambda x: (not x["all_day"], x["start_idx"], x["priority"])
    )

    # Select the top 2 summaries; the icon comes from the first one that has an icon
    selected_texts = [summary["text"] for summary in candidate_summaries[:2]]
    current_c_icon = next(
        (summary["icon"] for summary in candidate_summaries[:2] if summary["icon"]),
        None,
    )

    if len(selected_texts) == 1:
        final_constructed_summary = ["sentence", selected_texts[0]]
    else:
        final_constructed_summary = [
            "sentence",
            ["and", selected_texts[0], selected_texts[1]],
        ]

    # Ensure an icon is always returned, defaulting to overall average cloud cover if none set.
    if current_c_icon is None:
        current_c_icon = calculate_sky_icon(
            overall_avg_cloud_cover, is_day_time, icon_set
        )

    return current_c_icon, final_constructed_summary
