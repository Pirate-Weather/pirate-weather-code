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
#   6. Keep the one or two most important conditions and give each its own
#      sentence, followed by the expected precipitation amount when it is
#      significant.
import datetime
import itertools
import math
import statistics
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
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM,
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

# Precipitation that reaches every period is only described as lasting
# "throughout the day" when each period is wet for at least this fraction of
# its hours. One wet hour per period would otherwise read as all-day rain.
PERSISTENT_PRECIP_PERIOD_FRACTION = 0.5
# Below that, precipitation falling in less than this fraction of all hours is
# "occasional"; otherwise a few long spells are "periods of" precipitation and
# anything choppier is "intermittent".
INTERMITTENT_PRECIP_MIN_FRACTION = 1 / 3
PERIODS_OF_PRECIP_MAX_SPELLS = 3
PERIODS_OF_PRECIP_MIN_SPELL_HOURS = 3
# Precipitation that runs from the start of the window is "tapering off" when
# its amount doesn't rise over its last TAPERING_OFF_HOURS hours and the last
# hour has at most TAPERING_OFF_END_FRACTION of the spell's peak hourly amount.
TAPERING_OFF_HOURS = 3
TAPERING_OFF_END_FRACTION = 0.5

# The "of-expected" sentence names one precipitation type. When several fall,
# the first of these that is exceeded is shown (mm; snow as depth), otherwise
# the type with the most liquid-equivalent accumulation.
EXPECTED_ACCUM_TYPE_ICE_MIN_MM = 1
EXPECTED_ACCUM_TYPE_SNOW_MIN_MM = 50
EXPECTED_ACCUM_TYPE_RAIN_MIN_MM = 10
# The sentence is only added once the shown type reaches these amounts (mm).
EXPECTED_ACCUM_SENTENCE_MIN_MM = {
    "rain": 15,
    "ice": 1,
    "snow": 10,
}

MM_PER_INCH = 25.4
MM_PER_CENTIMETER = 10

# Cloud cover transitions (clearing / increasing / becoming).
# HRRR, HRDPS and similar models can spike for an hour or two (100 -> 0 -> 14
# -> 100), so hourly cloud cover is smoothed with a centred rolling median
# that removes spikes of up to two hours but keeps the edge of a real change.
CLOUD_SMOOTHING_HOURS = 5
# A transition needs the cloud level to change by at least this many levels
# (e.g. partly cloudy to overcast) between the start and end of the window.
CLOUD_TRANSITION_MIN_LEVELS = 2
# A change spread over at least this many hours is a trend ("increasing
# clouds", "clearing"); a quicker step change is "X becoming Y".
CLOUD_TREND_MIN_HOURS = 6
# Hour-to-hour movement against the trend (fraction of sky) that is still
# treated as part of a steady trend.
CLOUD_TREND_REVERSAL_TOLERANCE = 0.1
# Periods split between two neighbouring cloud levels are "generally" the more
# common level when it covers at least this fraction of the hours, otherwise
# "X to Y".
GENERALLY_CLOUD_MIN_FRACTION = 0.75

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


def _period_cloud_cover(cloud_covers):
    """
    Median cloud cover of a period's hours, skipping missing (NaN) hours.

    A period missing most of its hours is treated as clear (0.0), rather than
    letting one or two hours stand for it. statistics.median is several times
    faster than np.median on the handful of hourly values in a period, and
    this runs for every summary.

    Returns:
    - tuple: The cloud cover (0.0 to 1.0) and whether most hours were missing.
    """
    valid_covers = [cover for cover in cloud_covers if not math.isnan(cover)]
    if len(valid_covers) * 2 < len(cloud_covers):
        return 0.0, True
    return statistics.median(valid_covers), False


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
                                           / ["during", ["list", P1, P2, P3, ...]]
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
        single_periods = [all_periods[run[0]] for run in runs]
        if len(single_periods) > 2:
            # "list" avoids nested "and"s: "this morning, evening and tomorrow morning"
            return ["during", ["list", *single_periods]]
        return ["during", _join_with_and(single_periods)]

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
    for_night=False,
    time_phrase=None,
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
    - for_night (bool): Describe a condition lasting the whole half-day with
      "for-night" instead of the bare condition (night half-days).
    - time_phrase (list, optional): Use this time phrase instead of the one
      for period_indices (e.g. ["during", P] for precipitation tapering off).

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

    if time_phrase is None:
        time_phrase = _get_time_phrase(period_indices, all_periods)

    if time_phrase is None:
        summary_text = None
    elif time_phrase == ["for-day"]:
        # Half-day summaries describe the whole half-day with the bare condition
        # (#385), except a night half-day can say "throughout the night"
        if for_night:
            summary_text = ["for-night", condition_text]
        elif half_day:
            summary_text = condition_text
        else:
            summary_text = ["for-day", condition_text]
    else:
        summary_text = _insert_condition_text(time_phrase, condition_text)

    return summary_text, wind_condition_combined, vis_condition_combined


def _new_period_data():
    """Return the empty aggregation for one standard period."""
    return {
        "num_hours_fog": 0,
        "num_hours_wind": 0,
        "num_hours_thunderstorm": 0,
        "num_hours_precip": 0,
        "rain_accum": 0.0,
        "snow_accum": 0.0,
        # Accumulation errors (mm), and whether any hour had error data
        "rain_error": 0.0,
        "has_rain_error_data": False,
        "snow_error": 0.0,
        "has_snow_error_data": False,
        "ice_error": 0.0,
        "has_ice_error_data": False,
        "sleet_accum": 0.0,
        "frzrain_accum": 0.0,
        "max_pop": 0.0,
        "max_rain_intensity": 0.0,
        "max_snow_intensity": 0.0,
        "max_ice_intensity": 0.0,
        "cloud_covers": [],
        "max_wind_speed": 0.0,
        "period_length": 0,
        "precip_types_in_period": [],
        # Median rather than mean, so a one or two hour model spike doesn't
        # move the whole period to a different cloud level
        "median_cloud_cover": 0.0,
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


def _hour_precip_accum(hour):
    """Liquid-equivalent precipitation for an hour (mm), snow using a 10:1 ratio."""
    return (
        hour["liquidAccumulation"]
        + hour["iceAccumulation"]
        + hour["snowAccumulation"] / 10
    )


def _is_wet_hour(hour):
    """
    Whether an hour has at least enough precipitation for an hourly
    precipitation icon. Used to tell all-day precipitation from a few showers.
    """
    return (
        hour["liquidAccumulation"] + hour["iceAccumulation"]
        >= HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM
        or hour["snowAccumulation"] >= HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM
    )


def _wet_spells(hourly_wet):
    """Runs of consecutive wet hours, as lists of hour indices."""
    spells = []
    for i, is_wet in enumerate(hourly_wet):
        if not is_wet:
            continue
        if spells and spells[-1][-1] == i - 1:
            spells[-1].append(i)
        else:
            spells.append([i])
    return spells


def _is_tapering_off(spell, hourly_precip_accum):
    """
    Whether a spell of precipitation is tapering off: its hourly amount
    doesn't rise over its last TAPERING_OFF_HOURS hours and ends at no more
    than TAPERING_OFF_END_FRACTION of its peak.
    """
    if len(spell) < TAPERING_OFF_HOURS:
        return False
    amounts = [hourly_precip_accum[i] for i in spell]
    last_amounts = amounts[-TAPERING_OFF_HOURS:]
    return (
        all(later <= earlier for earlier, later in itertools.pairwise(last_amounts))
        and last_amounts[-1] <= max(amounts) * TAPERING_OFF_END_FRACTION
    )


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
    period_data["cloud_covers"].append(hour["cloudCover"])
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

    if _is_wet_hour(hour):
        period_data["num_hours_precip"] += 1

    # Only accumulate error if it's not nan (missing data) and the hour has
    # accumulation of its precipitation type. precipIntensityError is an
    # intensity error (mm/h), so over one hour it is a liquid accumulation
    # error (mm).
    liquid_error_mm = hour["precipIntensityError"] * 1.0
    has_error = not math.isnan(liquid_error_mm)
    if (
        has_error
        and hour["precipType"] == PRECIP_TYPES["snow"]
        and hour["snowAccumulation"] > 0.0
    ):
        # Convert to snow depth (mm) via estimate_snow_height. If temperature
        # is missing, fall back to using error directly.
        temp = hour.get("temperature", MISSING_DATA)
        if temp is None or (isinstance(temp, float) and math.isnan(temp)):
            estimated_snow_error_mm = liquid_error_mm
        else:
            estimated_snow_error_mm = estimate_snow_height(
                liquid_error_mm, temp, hour.get("windSpeed", 0.0)
            )
        period_data["snow_error"] += estimated_snow_error_mm
        period_data["has_snow_error_data"] = True
    elif (
        has_error
        and hour["precipType"] == PRECIP_TYPES["rain"]
        and hour["liquidAccumulation"] > 0.0
    ):
        period_data["rain_error"] += liquid_error_mm
        period_data["has_rain_error_data"] = True
    elif (
        has_error
        and hour["precipType"] in (PRECIP_TYPES["sleet"], PRECIP_TYPES["ice"])
        and hour["iceAccumulation"] > 0.0
    ):
        period_data["ice_error"] += liquid_error_mm
        period_data["has_ice_error_data"] = True

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


def _accumulation_amount(total_accum, total_error, has_error_data, unit_str):
    """
    Builds the accumulation (or accumulation range) for the summary.

    The error is the summed hourly ensemble standard deviation, and the range
    spans half of it either side. Precipitation totals are skewed (they can't
    fall below zero but can run well above the mean), so only the top of the
    range adds the spread; the bottom divides by the same factor instead:

        low = amount * amount / (amount + spread)

    For a small spread this is the usual amount - spread (19 +/- 2 -> 17-21),
    but a spread as large as the amount gives 9-39 rather than reaching 0.

    The translations only take whole numbers, so amounts are rounded out to
    whole units ("< 1 in" for a few millimetres of rain in US units).

    Parameters:
    - total_accum (float): Total accumulation in mm.
    - total_error (float): Total accumulation error in mm.
    - has_error_data (bool): Whether any hour had error data.
    - unit_str (str): "inches", "centimeters" or "millimeters".

    Returns:
    - list | None: The accumulation text, or None if there is nothing to show.
    """
    mm_per_unit = {
        "inches": MM_PER_INCH,
        "centimeters": MM_PER_CENTIMETER,
        "millimeters": 1,
    }[unit_str]
    accum_display = total_accum / mm_per_unit
    spread_display = total_error / mm_per_unit / 2
    less_than_tolerance = LESS_THAN_TOLERANCE / mm_per_unit

    if accum_display <= 0:
        return None

    # Without error data (ERA5, etc.) or with a zero spread there is no range
    if not has_error_data or spread_display <= 0.01:
        return [unit_str, math.ceil(accum_display)]

    low_accum = math.floor(accum_display**2 / (accum_display + spread_display))
    max_accum = math.ceil(accum_display + spread_display)

    # Too small to have a whole-unit lower bound: a range from 1 once the top
    # is large enough to be worth it (#510), otherwise "< N"
    if low_accum == 0:
        if max_accum < less_than_tolerance:
            return ["less-than", [unit_str, max_accum]]
        low_accum = 1
    if low_accum == max_accum:
        return [unit_str, max_accum]
    return [unit_str, ["range", low_accum, max_accum]]


def _expected_accumulation_sentence(
    totals, errors, has_error_data, dominant_ice_text, unit_system
):
    """
    Builds the "N of <type> expected" sentence for a significant amount of
    precipitation, e.g. ["of-expected", ["millimeters", ["range", 7, 16]], "medium-rain"].

    Ice is shown first because even a little freezing rain or sleet is
    hazardous, then a large amount of snow, then a large amount of rain.

    Parameters:
    - totals (dict): Accumulation (mm) for "rain", "snow" (depth) and "ice".
    - errors (dict): Accumulation error (mm) for each type.
    - has_error_data (dict): Whether each type had any error data.
    - dominant_ice_text (str): "medium-sleet" or "medium-freezing-rain".
    - unit_system (str): Unit system for display ("us", "si", "ca", "uk").

    Returns:
    - list | None: The sentence, or None if no type reaches its threshold.
    """
    if totals["ice"] > EXPECTED_ACCUM_TYPE_ICE_MIN_MM:
        accum_type = "ice"
    elif totals["snow"] > EXPECTED_ACCUM_TYPE_SNOW_MIN_MM:
        accum_type = "snow"
    elif totals["rain"] > EXPECTED_ACCUM_TYPE_RAIN_MIN_MM:
        accum_type = "rain"
    else:
        # Compare snow as a liquid equivalent using a 10:1 ratio
        accum_type = max(
            ("rain", "snow", "ice"),
            key=lambda t: totals[t] / 10 if t == "snow" else totals[t],
        )

    if totals[accum_type] < EXPECTED_ACCUM_SENTENCE_MIN_MM[accum_type]:
        return None

    if unit_system == "us":
        unit_str = "inches"
    elif accum_type == "snow":  # si, ca, uk
        unit_str = "centimeters"
    else:
        unit_str = "millimeters"

    amount = _accumulation_amount(
        totals[accum_type],
        errors[accum_type],
        has_error_data[accum_type],
        unit_str,
    )
    if amount is None:
        return None

    type_text = {
        "rain": "medium-rain",
        "snow": "medium-snow",
        "ice": dominant_ice_text,
    }[accum_type]
    return ["of-expected", amount, type_text]


def _smoothed_cloud_cover(cloud_covers, hour_idx):
    """
    Centred rolling median of the hourly cloud cover at one hour
    (CLOUD_SMOOTHING_HOURS wide, narrower at the ends of the window). Removes
    the short spikes some models produce without blurring the edge of a real
    change. The cloud cover must not be missing (NaN).
    """
    half_width = CLOUD_SMOOTHING_HOURS // 2
    return statistics.median(
        cloud_covers[max(0, hour_idx - half_width) : hour_idx + half_width + 1]
    )


def _cloud_transition(cloud_covers, hour_period_idx):
    """
    Describes cloud cover that changes over the window.

    Using the smoothed hourly cloud cover, the cloud level has to change by
    CLOUD_TRANSITION_MIN_LEVELS from the first hour to the last, staying
    within one level of the starting level until the change begins and of the
    new level once it is reached. Then:

        steady change over CLOUD_TREND_MIN_HOURS or more   "increasing" / "clearing"
        quicker step change                               ["becoming", from, to]

    A change that goes back and forth (beyond CLOUD_TREND_REVERSAL_TOLERANCE)
    is not described as a transition. For example, 75% for the morning then
    15%, 13%, 20% from midday is "mostly cloudy becoming mostly clear in the
    afternoon" rather than "clearing".

    Parameters:
    - cloud_covers (list): Hourly cloud cover (0.0 to 1.0) for the window.
    - hour_period_idx (list): The period index of each hour.

    Returns:
    - tuple | None: The cloud condition text and the indices of the periods
      the change happens in, or None if the cloud cover doesn't transition.
    """
    if len(cloud_covers) < CLOUD_TREND_MIN_HOURS or math.isnan(sum(cloud_covers)):
        return None

    # Most windows end near the level they start at, so check the ends before
    # smoothing every hour (this runs for every summary on the request path)
    num_hours = len(cloud_covers)
    _, start_level = calculate_cloud_text(_smoothed_cloud_cover(cloud_covers, 0))
    _, end_level = calculate_cloud_text(
        _smoothed_cloud_cover(cloud_covers, num_hours - 1)
    )
    if abs(end_level - start_level) < CLOUD_TRANSITION_MIN_LEVELS:
        return None

    smoothed = [_smoothed_cloud_cover(cloud_covers, i) for i in range(num_hours)]
    levels = [calculate_cloud_text(cover)[1] for cover in smoothed]
    direction = 1 if end_level > start_level else -1

    # First hour from which the cloud stays near the new level
    settle_hour = len(levels) - 1
    while settle_hour > 0 and abs(levels[settle_hour - 1] - end_level) <= 1:
        settle_hour -= 1
    while levels[settle_hour] != end_level:
        settle_hour += 1

    # Last hour still at the starting level before the change
    leave_hour = max(i for i in range(settle_hour) if levels[i] == start_level)

    if any(abs(level - start_level) > 1 for level in levels[:leave_hour]) or any(
        direction * (smoothed[i + 1] - smoothed[i]) < -CLOUD_TREND_REVERSAL_TOLERANCE
        for i in range(leave_hour, settle_hour)
    ):
        return None

    if settle_hour - leave_hour >= CLOUD_TREND_MIN_HOURS:
        trend_text = "increasing" if direction > 0 else "clearing"
        trend_periods = sorted(set(hour_period_idx[leave_hour + 1 : settle_hour + 1]))
        return trend_text, trend_periods

    becoming_text = [
        "becoming",
        CLOUD_LEVEL_TEXT_AND_COVER[start_level][0],
        CLOUD_LEVEL_TEXT_AND_COVER[end_level][0],
    ]
    return becoming_text, [hour_period_idx[settle_hour]]


def _cloud_range(cloud_levels, cloud_covers):
    """
    Describes cloud cover whose periods are split between two neighbouring
    cloud levels: ["generally", level] when one level covers at least
    GENERALLY_CLOUD_MIN_FRACTION of the hours, otherwise
    ["to", lower level, higher level] ("partly cloudy to mostly cloudy").

    Parameters:
    - cloud_levels (list): The cloud level (0-4) of each period.
    - cloud_covers (list): Hourly cloud cover (0.0 to 1.0) for the window.

    Returns:
    - list | None: The cloud text, or None if the periods aren't split
      between two neighbouring levels.
    """
    low_level = min(cloud_levels)
    high_level = max(cloud_levels)
    if high_level - low_level != 1 or math.isnan(sum(cloud_covers)):
        return None

    # Raw hours are enough here: a one or two hour spike changes the share by
    # less than a tenth, and smoothing every hour costs time on every request
    hourly_levels = [calculate_cloud_text(cover)[1] for cover in cloud_covers]
    for level in (low_level, high_level):
        if (
            hourly_levels.count(level)
            >= len(hourly_levels) * GENERALLY_CLOUD_MIN_FRACTION
        ):
            return ["generally", CLOUD_LEVEL_TEXT_AND_COVER[level][0]]
    return [
        "to",
        CLOUD_LEVEL_TEXT_AND_COVER[low_level][0],
        CLOUD_LEVEL_TEXT_AND_COVER[high_level][0],
    ]


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
    # Hourly cloud cover, precipitation and the period index of each hour, for
    # cloud transitions and how the precipitation falls
    hourly_cloud_covers = []
    hourly_precip_accum = []
    hourly_wet = []
    hour_period_idx = []
    period_indices = {}
    overall_min_temp_dewpoint_spread = float("inf")
    overall_temp_at_min_spread = 0.0
    overall_dewpoint_at_min_spread = 0.0

    for hour in hours:
        local_time = datetime.datetime.fromtimestamp(hour["time"], zone)
        period_name = _get_period_name(local_time, start_time, mode)
        if period_name not in periods_data:
            periods_data[period_name] = _new_period_data()
            period_indices[period_name] = len(period_indices)
        hourly_cloud_covers.append(hour["cloudCover"])
        hourly_precip_accum.append(_hour_precip_accum(hour))
        hourly_wet.append(_is_wet_hour(hour))
        hour_period_idx.append(period_indices[period_name])

        fog_spread = _add_hour_to_period(periods_data[period_name], hour, icon_set)
        # Keep the temperature and dew point of the foggiest hour for the fog text
        if fog_spread is not None and fog_spread < overall_min_temp_dewpoint_spread:
            overall_min_temp_dewpoint_spread = fog_spread
            overall_temp_at_min_spread = hour["temperature"]
            overall_dewpoint_at_min_spread = hour["dewPoint"]

    all_period_names = list(periods_data)
    period_stats = list(periods_data.values())
    for p_data in period_stats:
        p_data["median_cloud_cover"], p_data["cloud_cover_missing"] = (
            _period_cloud_cover(p_data["cloud_covers"])
        )

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
    total_rain_error = 0.0
    total_snow_error = 0.0
    total_ice_error = 0.0
    has_any_rain_error_data = False
    has_any_snow_error_data = False
    has_any_ice_error_data = False
    total_hours_precip = 0
    overall_max_rain_intensity = 0.0
    overall_max_snow_intensity = 0.0
    overall_max_ice_intensity = 0.0
    overall_max_wind = 0.0
    overall_cloud_cover_sum = 0.0  # Sum of the median cloud cover of each period
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
        total_rain_error += p_data["rain_error"]
        total_snow_error += p_data["snow_error"]
        total_ice_error += p_data["ice_error"]
        has_any_rain_error_data |= p_data["has_rain_error_data"]
        has_any_snow_error_data |= p_data["has_snow_error_data"]
        has_any_ice_error_data |= p_data["has_ice_error_data"]
        total_hours_precip += p_data["num_hours_precip"]
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
        overall_cloud_cover_sum += p_data["median_cloud_cover"]
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

        _, cloud_level = calculate_cloud_text(p_data["median_cloud_cover"])
        cloud_levels.append(cloud_level)

    overall_avg_cloud_cover = (
        overall_cloud_cover_sum / len(period_stats) if period_stats else 0
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
            # cloud cover (the first one on a tie, skipping missing data)
            cloudiest_idx = None
            max_avg_cloud_value = -1.0
            for idx, p_data in enumerate(period_stats):
                if p_data["median_cloud_cover"] > max_avg_cloud_value:
                    max_avg_cloud_value = p_data["median_cloud_cover"]
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

    # With a single period use its actual cloud cover for the icon
    if len(period_stats) == 1:
        derived_avg_cloud_for_icon = period_stats[0]["median_cloud_cover"]

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

    # Combine primary and secondary precipitation conditions with "and"
    if secondary_precip_condition is not None:
        precip_summary_text = ["and", precip_summary_text, secondary_precip_condition]

    # The expected amount follows the precipitation summary as its own sentence
    # ("Rain throughout the day. 7–16 mm of rain expected."). It replaces the
    # snow-only "parenthetical". A "possible" summary doesn't get one, as an
    # expected amount would contradict it.
    accumulation_sentence = None
    if precip_summary_text is not None and overall_max_pop >= PRECIP_PROB_THRESHOLD:
        accumulation_sentence = _expected_accumulation_sentence(
            {
                "rain": total_rain_accum,
                "snow": total_snow_accum,
                "ice": total_ice_accum,
            },
            {
                "rain": total_rain_error,
                "snow": total_snow_error,
                "ice": total_ice_error,
            },
            {
                "rain": has_any_rain_error_data,
                "snow": has_any_snow_error_data,
                "ice": has_any_ice_error_data,
            },
            "medium-freezing-rain"
            if total_frzrain_accum >= total_sleet_accum
            else "medium-sleet",
            unit_system,
        )

    # Flags to indicate if a condition is present at all in the forecast block
    has_precip = bool(precip_periods) and precip_summary_text is not None
    has_thunderstorm = bool(thunderstorm_periods)
    has_wind = bool(wind_periods)
    has_vis = bool(vis_periods)

    # --- 4b. Thunderstorm text, replacing the precipitation text when they match ---
    # (otherwise it accompanies the precipitation, see 5b)
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
    num_periods = len(period_stats)

    # 5a. How the precipitation falls. Precipitation reaching every period
    # only lasts "throughout the day" when each period is wet for at least half
    # its hours, with at most PERIODS_OF_PRECIP_MAX_SPELLS spells. Otherwise
    # one continuous spell that only clips the first or last period is
    # described without those periods, and showers through the day are
    # occasional, periods of, or intermittent precipitation.
    precip_time_phrase = None
    wet_spells = _wet_spells(hourly_wet)
    if has_precip and len(precip_periods) == num_periods and wet_spells:

        def is_persistent(idx):
            p_data = period_stats[idx]
            return (
                p_data["num_hours_precip"]
                >= p_data["period_length"] * PERSISTENT_PRECIP_PERIOD_FRACTION
            )

        def can_trim(idx):
            # Keep a period with thunderstorms that accompany the precipitation,
            # or they would be described in a period the precipitation isn't
            # ("rain from afternoon until night, with thunderstorms in the
            # morning"). Storms that replace the precipitation text are
            # trimmed along with it.
            return not is_persistent(idx) and (
                thunderstorms_match_precip or idx not in thunderstorm_periods
            )

        # Many short breaks also stop it being all-day precipitation, even
        # when every period is half wet (rain every other hour)
        if len(wet_spells) > PERIODS_OF_PRECIP_MAX_SPELLS or not all(
            is_persistent(idx) for idx in precip_periods
        ):
            if len(wet_spells) == 1:
                while len(precip_periods) > 1 and can_trim(precip_periods[0]):
                    precip_periods = precip_periods[1:]
                while len(precip_periods) > 1 and can_trim(precip_periods[-1]):
                    precip_periods = precip_periods[:-1]
            elif total_hours_precip < len(hours) * INTERMITTENT_PRECIP_MIN_FRACTION:
                precip_summary_text = ["occasional", precip_summary_text]
            elif len(wet_spells) <= PERIODS_OF_PRECIP_MAX_SPELLS and all(
                len(spell) >= PERIODS_OF_PRECIP_MIN_SPELL_HOURS for spell in wet_spells
            ):
                precip_summary_text = ["periods-of", precip_summary_text]
            else:
                precip_summary_text = ["intermittent", precip_summary_text]

    # Precipitation from the start of the window that ends by weakening is
    # "rain tapering off in the afternoon" rather than "rain until evening".
    # Not when wind or fog shares its periods, which would read "rain tapering
    # off and windy". The spell judged is the last one starting by the end of
    # the precipitation periods, not a later shower too light to be reported.
    ending_spell = None
    if has_precip:
        ending_spell = next(
            (
                spell
                for spell in reversed(wet_spells)
                if hour_period_idx[spell[0]] <= precip_periods[-1]
            ),
            None,
        )
    if (
        ending_spell is not None
        and precip_periods == list(range(len(precip_periods)))
        and precip_periods[-1] < num_periods - 1
        and precip_periods not in (wind_periods, vis_periods)
        and _is_tapering_off(ending_spell, hourly_precip_accum)
    ):
        precip_summary_text = ["tapering-off", precip_summary_text]
        precip_time_phrase = ["during", all_period_names[precip_periods[-1]]]

    # Precipitation lasting a whole night half-day is "throughout the night".
    # The window has to reach the overnight period: a slice of the evening
    # alone (e.g. the rest of tonight from 11pm) is not the whole night.
    night_half_day = (
        half_day
        and all_period_names[-1].endswith("night")
        and all(name.endswith(("evening", "night")) for name in all_period_names)
    )

    # 5b. Thunderstorms in only some of the precipitation periods are attached
    # to the precipitation: "Rain, with possible thunderstorms starting in the
    # afternoon." Listing both separately ("Rain throughout the day and
    # thunderstorms in the afternoon") reads awkwardly and can get very long.
    storms_accompany_precip = (
        has_precip
        and has_thunderstorm
        and not thunderstorms_match_precip
        and thunderstorm_summary_text is not None
    )

    precip_only_summary = None
    thunderstorm_only_summary = None
    wind_only_summary = None
    vis_only_summary = None
    # Track which conditions have been "subsumed" by the precipitation summary
    combined_vis_flag = False
    combined_wind_flag = False

    if has_precip:
        # Alongside thunderstorms, all-day precipitation is the bare condition
        # ("Rain, with ..."), as in half-day summaries
        bare_all_day_precip = half_day or (
            storms_accompany_precip and len(precip_periods) == num_periods
        )
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
            bare_all_day_precip,
            for_night=night_half_day and not storms_accompany_precip,
            time_phrase=precip_time_phrase,
        )

    if storms_accompany_precip:
        thunderstorm_with_precip, _, _ = calculate_period_summary_text(
            thunderstorm_periods,
            thunderstorm_summary_text,
            "precip",
            all_period_names,
            [],
            [],
            overall_max_wind,
            icon_set,
            half_day,
        )
        precip_only_summary = [
            "accompanied-by",
            precip_only_summary,
            thunderstorm_with_precip,
        ]
        # Definite thunderstorms take the icon; possible ones keep the
        # precipitation icon
        if thunderstorm_summary_text == "thunderstorm" and thunderstorm_icon:
            precip_icon = thunderstorm_icon
    elif has_thunderstorm and not has_precip and thunderstorm_summary_text:
        # Without a precipitation summary the thunderstorms stand on their own
        (
            thunderstorm_only_summary,
            combined_wind_flag,
            combined_vis_flag,
        ) = calculate_period_summary_text(
            thunderstorm_periods,
            thunderstorm_summary_text,
            "precip",  # Use precip type since thunderstorms have same priority
            all_period_names,
            wind_periods,
            vis_periods,
            overall_max_wind,
            icon_set,
            half_day,
            for_night=night_half_day,
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
    candidate_summaries = []

    if precip_only_summary:
        candidate_summaries.append(
            {
                "priority": 0,
                "all_day": len(precip_periods) == num_periods,
                "start_idx": precip_periods[0],
                "text": precip_only_summary,
                "icon": precip_icon,
                "accumulation": accumulation_sentence,
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

    # Cloud cover as a fallback if no other condition is present. Wind and fog
    # would be candidates themselves, so there is nothing to combine it with.
    if not candidate_summaries:
        cloud_text = final_cloud_text
        cloud_idx = overall_cloud_idx
        # A change over the window ("clearing", "X becoming Y"), otherwise
        # periods split between two neighbouring levels ("generally overcast",
        # "partly cloudy to mostly cloudy") throughout the window
        # Missing hours take their period's cloud cover. A period missing most
        # of its hours is only assumed clear, so it can't show a change.
        cloud_transition = None
        cloud_range = None
        if not any(p_data["cloud_cover_missing"] for p_data in period_stats):
            filled_cloud_covers = [
                period_stats[hour_period_idx[i]]["median_cloud_cover"]
                if math.isnan(cover)
                else cover
                for i, cover in enumerate(hourly_cloud_covers)
            ]
            cloud_transition = _cloud_transition(filled_cloud_covers, hour_period_idx)
            if cloud_transition is None:
                cloud_range = _cloud_range(cloud_levels, filled_cloud_covers)
        if cloud_transition is not None:
            cloud_text, cloud_idx = cloud_transition
        elif cloud_range is not None:
            cloud_text = cloud_range
            cloud_idx = list(range(num_periods))

        cloud_full_summary, _, _ = calculate_period_summary_text(
            cloud_idx,
            cloud_text,
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
                "all_day": len(cloud_idx) == num_periods,
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
    selected_summaries = candidate_summaries[:2]
    current_c_icon = next(
        (summary["icon"] for summary in selected_summaries if summary["icon"]),
        None,
    )

    # Each condition gets its own sentence ("Drizzle until tomorrow afternoon.
    # Breezy from tomorrow morning until tomorrow evening."). Conditions in the
    # same periods were already joined with "and" above ("Rain and windy until
    # afternoon."). The expected amount follows the precipitation sentence.
    sentences = []
    for summary in selected_summaries:
        sentences.append(["sentence", summary["text"]])
        if summary.get("accumulation") is not None:
            sentences.append(["sentence", summary["accumulation"]])

    final_constructed_summary = sentences[-1]
    for sentence in reversed(sentences[:-1]):
        final_constructed_summary = [
            "multiple-sentences",
            sentence,
            final_constructed_summary,
        ]

    # Ensure an icon is always returned, defaulting to overall average cloud cover if none set.
    if current_c_icon is None:
        current_c_icon = calculate_sky_icon(
            overall_avg_cloud_cover, is_day_time, icon_set
        )

    return current_c_icon, final_constructed_summary
