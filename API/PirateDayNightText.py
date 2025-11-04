# %% Script to generate simplified day/night text summaries for Pirate Weather
# This is designed for half-day forecasts (day: 4am-4pm, night: 5pm-4am)
# Unlike the full daily summary, this only uses 2 periods and simpler phrases

import numpy as np

from API.constants.shared_const import MISSING_DATA
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    DEFAULT_HUMIDITY,
    DEFAULT_POP,
    DEFAULT_VISIBILITY,
    PRECIP_INTENSITY_THRESHOLDS,
)
from API.PirateTextHelper import (
    Most_Common,
    calculate_precip_text,
    calculate_sky_icon,
    calculate_thunderstorm_text,
    calculate_vis_text,
    calculate_wind_text,
    humidity_sky_text,
)

# Threshold for precipitation to be considered significant
PRECIP_THRESH = 0.25


def _value_or_default(value, default):
    """Return value or default if value is None or NaN"""
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return value


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


def calculate_half_day_text(
    hours,
    precip_accum_unit,
    vis_units,
    wind_unit,
    temp_units,
    is_day_time,
    time_zone,
    curr_time,
    icon_set="darksky",
):
    """
    Calculates a simplified day or night weather summary text.
    Designed for half-day periods (day: 4am-4pm, night: 5pm-4am).

    This function uses simplified logic compared to calculate_day_text:
    - Only splits into at most 2 periods (early/late)
    - Only uses "for-day" and "during" time phrases
    - Suitable for 10-13 hour forecast windows

    Parameters:
    - hours (list): An array of hourly forecast data (10-13 hours).
    - precip_accum_unit (float): The precipitation accumulation unit.
    - vis_units (float): The visibility unit used.
    - wind_unit (float): The wind speed unit used.
    - temp_units (float): The temperature unit used.
    - is_day_time (bool): Whether it's daytime (True) or nighttime (False).
    - time_zone (str): The timezone for the current location.
    - curr_time (int): The current epoch time.
    - icon_set (str): Which icon set to use ("darksky" or "pirate").

    Returns:
    - tuple: A tuple containing:
        - c_icon (str): The icon representing the half-day period.
        - summary_text (list): The textual representation of the half-day period.
    """

    # Return "unavailable" if insufficient data
    if len(hours) < 8 or len(hours) > 15:
        return "none", ["unavailable"]

    # Sanitize hourly data with defaults
    sanitized_hours = []
    for hour in hours:
        sanitized_hour = dict(hour)
        sanitized_hour["humidity"] = _value_or_default(
            sanitized_hour.get("humidity", DEFAULT_HUMIDITY), DEFAULT_HUMIDITY
        )
        sanitized_hour["visibility"] = _value_or_default(
            sanitized_hour.get("visibility", DEFAULT_VISIBILITY), DEFAULT_VISIBILITY
        )
        sanitized_hour["precipIntensityError"] = _value_or_default(
            sanitized_hour.get("precipIntensityError", 0), 0
        )
        sanitized_hour["precipProbability"] = _value_or_default(
            sanitized_hour.get("precipProbability", DEFAULT_POP), DEFAULT_POP
        )
        sanitized_hour["smoke"] = _value_or_default(
            sanitized_hour.get("smoke", 0.0), 0.0
        )
        dew_point_default = sanitized_hour.get("temperature")
        sanitized_hour["dewPoint"] = _value_or_default(
            sanitized_hour.get("dewPoint", dew_point_default), dew_point_default
        )
        sanitized_hours.append(sanitized_hour)

    hours = sanitized_hours

    # Split into 2 periods: first half and second half
    mid_point = len(hours) // 2
    periods = [
        {"name": "early", "hours": hours[:mid_point]},
        {"name": "late", "hours": hours[mid_point:]},
    ]

    # Aggregate data for each period
    period_stats = []
    for period in periods:
        stats = {
            "rain_accum": 0.0,
            "snow_accum": 0.0,
            "sleet_accum": 0.0,
            "snow_error": 0.0,
            "max_pop": 0.0,
            "max_intensity": 0.0,
            "cloud_cover_sum": 0.0,
            "max_wind_speed": 0.0,
            "num_hours": len(period["hours"]),
            "num_hours_fog": 0,
            "num_hours_wind": 0,
            "num_hours_humid": 0,
            "num_hours_dry": 0,
            "num_hours_thunderstorm": 0,
            "min_visibility": float("inf"),
            "max_smoke": 0.0,
            "max_cape_with_precip": 0.0,
            "max_lifted_index_with_precip": MISSING_DATA,
            "precip_types": [],
        }

        for hour in period["hours"]:
            # Cloud cover
            stats["cloud_cover_sum"] += hour["cloudCover"]

            # Wind
            stats["max_wind_speed"] = max(stats["max_wind_speed"], hour["windSpeed"])

            # Precipitation
            stats["max_intensity"] = max(
                stats["max_intensity"], hour["precipIntensity"]
            )
            stats["max_pop"] = max(stats["max_pop"], hour["precipProbability"])

            if hour["precipType"] == "rain" or hour["precipType"] == "none":
                stats["rain_accum"] += hour["precipAccumulation"]
            elif hour["precipType"] == "snow":
                stats["snow_accum"] += hour["precipAccumulation"]
                stats["snow_error"] += hour["precipIntensityError"]
            elif hour["precipType"] == "sleet":
                stats["sleet_accum"] += hour["precipAccumulation"]

            if hour["precipIntensity"] > 0 or hour["precipAccumulation"] > 0:
                stats["precip_types"].append(hour["precipType"])

            # Visibility
            stats["min_visibility"] = min(stats["min_visibility"], hour["visibility"])
            stats["max_smoke"] = max(stats["max_smoke"], hour["smoke"])

            # Humidity
            humidity_text = humidity_sky_text(
                hour["temperature"], temp_units, hour["humidity"]
            )
            if humidity_text == "high-humidity":
                stats["num_hours_humid"] += 1
            elif humidity_text == "low-humidity":
                stats["num_hours_dry"] += 1

            # Fog
            vis_text = calculate_vis_text(
                hour["visibility"],
                vis_units,
                temp_units,
                hour["temperature"],
                hour["dewPoint"],
                hour["smoke"],
                icon_set,
                "icon",
            )
            if (
                vis_text is not None
                and hour["precipIntensity"] <= 0.02 * precip_accum_unit
            ):
                stats["num_hours_fog"] += 1

            # Wind
            wind_text = calculate_wind_text(
                hour["windSpeed"], wind_unit, "darksky", "icon"
            )
            if wind_text == "wind":
                stats["num_hours_wind"] += 1

            # Thunderstorms
            hour_cape = hour.get("cape", MISSING_DATA)
            hour_lifted_index = hour.get("liftedIndex", MISSING_DATA)

            if hour["precipIntensity"] > 0 or hour["precipAccumulation"] > 0:
                if (
                    hour_cape != MISSING_DATA
                    and hour_cape > stats["max_cape_with_precip"]
                ):
                    stats["max_cape_with_precip"] = hour_cape

                if hour_lifted_index != MISSING_DATA:
                    if stats["max_lifted_index_with_precip"] == MISSING_DATA:
                        stats["max_lifted_index_with_precip"] = hour_lifted_index
                    elif hour_lifted_index < stats["max_lifted_index_with_precip"]:
                        stats["max_lifted_index_with_precip"] = hour_lifted_index

                thu_text = calculate_thunderstorm_text(
                    hour_lifted_index, hour_cape, "summary"
                )
                if thu_text is not None:
                    stats["num_hours_thunderstorm"] += 1

        # Calculate averages
        stats["avg_cloud_cover"] = stats["cloud_cover_sum"] / stats["num_hours"]
        period_stats.append(stats)

    # Calculate overall statistics
    total_rain = sum(p["rain_accum"] for p in period_stats)
    total_snow = sum(p["snow_accum"] for p in period_stats)
    total_sleet = sum(p["sleet_accum"] for p in period_stats)
    overall_max_intensity = max(p["max_intensity"] for p in period_stats)
    overall_max_pop = max(p["max_pop"] for p in period_stats)
    overall_avg_cloud = sum(p["avg_cloud_cover"] for p in period_stats) / len(
        period_stats
    )

    all_precip_types = []
    for p in period_stats:
        all_precip_types.extend(p["precip_types"])

    overall_most_common_precip = (
        Most_Common(all_precip_types) if all_precip_types else "none"
    )

    overall_max_cape = max(p["max_cape_with_precip"] for p in period_stats)
    overall_max_lifted_index = MISSING_DATA
    for p in period_stats:
        if p["max_lifted_index_with_precip"] != MISSING_DATA:
            if overall_max_lifted_index == MISSING_DATA:
                overall_max_lifted_index = p["max_lifted_index_with_precip"]
            else:
                overall_max_lifted_index = min(
                    overall_max_lifted_index, p["max_lifted_index_with_precip"]
                )

    # Determine which periods have precipitation
    precip_periods = []
    for i, stats in enumerate(period_stats):
        has_precip = (
            stats["snow_accum"]
            > (PRECIP_INTENSITY_THRESHOLDS["mid"] * precip_accum_unit)
            or stats["rain_accum"] > (PRECIP_THRESH * precip_accum_unit)
            or stats["sleet_accum"] > (PRECIP_THRESH * precip_accum_unit)
        )
        if has_precip:
            precip_periods.append(i)

    # Determine which periods have other conditions
    fog_periods = [i for i, s in enumerate(period_stats) if s["num_hours_fog"] >= 2]
    wind_periods = [
        i for i, s in enumerate(period_stats) if s["num_hours_wind"] >= 3
    ]

    # Determine cloud levels for each period
    cloud_levels = []
    for stats in period_stats:
        _, level = calculate_cloud_text(stats["avg_cloud_cover"])
        cloud_levels.append(level)

    # Build the summary text
    summary_parts = []
    c_icon = None

    # Priority 1: Precipitation
    if precip_periods:
        # Calculate precipitation text
        precip_text, precip_icon = calculate_precip_text(
            overall_max_intensity,
            precip_accum_unit,
            overall_most_common_precip,
            "day",
            total_rain,
            total_snow,
            total_sleet,
            overall_max_pop,
            icon_set,
            "both",
        )

        # Check for thunderstorms
        thu_text = calculate_thunderstorm_text(
            overall_max_lifted_index, overall_max_cape, "summary"
        )

        if thu_text is not None:
            precip_text = thu_text

        # Determine time phrase
        if len(precip_periods) == 2:
            # Precipitation throughout the period
            summary_parts.append(["for-day", precip_text])
        else:
            # Precipitation in only one half
            period_name = "early" if precip_periods[0] == 0 else "late"
            summary_parts.append(["during", precip_text, period_name])

        c_icon = precip_icon

    # Priority 2: Fog/visibility
    elif fog_periods:
        fog_text = "Fog"

        if len(fog_periods) == 2:
            summary_parts.append(["for-day", fog_text])
        else:
            period_name = "early" if fog_periods[0] == 0 else "late"
            summary_parts.append(["during", fog_text, period_name])

        c_icon = "fog"

    # Priority 3: Wind
    elif wind_periods:
        wind_text = "Windy"

        if len(wind_periods) == 2:
            summary_parts.append(["for-day", wind_text])
        else:
            period_name = "early" if wind_periods[0] == 0 else "late"
            summary_parts.append(["during", wind_text, period_name])

        c_icon = "wind"

    # Priority 4: Clouds
    else:
        # Determine cloud description
        cloud_text, cloud_level = calculate_cloud_text(overall_avg_cloud)

        # Check if cloud cover varies between periods
        if abs(cloud_levels[0] - cloud_levels[1]) > 1:
            # Significant variation
            if cloud_levels[0] > cloud_levels[1]:
                # Cloudier in first half
                period_name = "early"
                idx = 0
            else:
                # Cloudier in second half
                period_name = "late"
                idx = 1

            period_cloud_text, _ = calculate_cloud_text(
                period_stats[idx]["avg_cloud_cover"]
            )
            summary_parts.append(["during", period_cloud_text, period_name])
        else:
            # Consistent cloud cover
            summary_parts.append(["for-day", cloud_text])

        # Set icon
        c_icon = calculate_sky_icon(overall_avg_cloud, is_day_time, icon_set)

    # Build final summary
    if summary_parts:
        summary_text = summary_parts[0]
    else:
        # Fallback
        summary_text = ["for-day", "clear"]
        c_icon = "clear-day" if is_day_time else "clear-night"

    return c_icon, summary_text
