"""Legacy daily icon/text helpers kept opt-in for backward compatibility."""

from __future__ import annotations

import logging
from typing import Tuple

from API.PirateDayNightText import calculate_half_day_text
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    PRECIP_PROB_THRESHOLD,
    WIND_THRESHOLDS,
)


def pick_day_icon_and_summary(
    *,
    max_arr,
    mean_arr,
    sum_arr,
    precip_type_arr,
    precip_text_arr,
    idx: int,
    is_night: bool,
    mode: str,
    prep_accum_unit: float,
    vis_units: float,
    wind_unit: float,
) -> Tuple[str, str]:
    """Select the icon/summary for daily or half-day entries using legacy rules."""
    if mode == "hourly":
        prob = max_arr[idx, DATA_HOURLY["prob"]]
        rain = mean_arr[idx, DATA_HOURLY["rain"]]
        ice = mean_arr[idx, DATA_HOURLY["ice"]]
        snow = mean_arr[idx, DATA_HOURLY["snow"]]
        accum_thresh = HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prep_accum_unit
        precip_type = precip_type_arr[idx]
        precip_text = precip_text_arr[idx]
    else:
        prob = max_arr[idx, DATA_DAY["prob"]]
        rain = sum_arr[idx, DATA_DAY["rain"]]
        ice = sum_arr[idx, DATA_DAY["ice"]]
        snow = sum_arr[idx, DATA_DAY["snow"]]
        accum_thresh = DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prep_accum_unit
        snow_thresh = DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM * prep_accum_unit
        precip_type = precip_type_arr[idx]
        precip_text = precip_text_arr[idx]

    if prob >= PRECIP_PROB_THRESHOLD:
        if mode == "hourly":
            if (rain + ice) > accum_thresh or snow > accum_thresh:
                return precip_type, precip_text
        else:
            if (rain + ice) > accum_thresh or snow > snow_thresh:
                return precip_type, precip_text

    vis_val = (
        mean_arr[idx, DATA_HOURLY["vis"]]
        if mode == "hourly"
        else mean_arr[idx, DATA_DAY["vis"]]
    )
    if vis_val < FOG_THRESHOLD_METERS * vis_units:
        return "fog", "Fog"

    wind_val = (
        mean_arr[idx, DATA_HOURLY["wind"]]
        if mode == "hourly"
        else mean_arr[idx, DATA_DAY["wind"]]
    )
    if wind_val > WIND_THRESHOLDS["light"] * wind_unit:
        return "wind", "Windy"

    cloud_val = (
        mean_arr[idx, DATA_HOURLY["cloud"]]
        if mode == "hourly"
        else mean_arr[idx, DATA_DAY["cloud"]]
    )
    if cloud_val > CLOUD_COVER_THRESHOLDS["cloudy"]:
        return "cloudy", "Cloudy"
    if cloud_val > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        if is_night:
            return "partly-cloudy-night", "Partly Cloudy"
        return "partly-cloudy-day", "Partly Cloudy"

    if is_night:
        return "clear-night", "Clear"
    return "clear-day", "Clear"


def apply_legacy_half_day_text(
    *,
    summary_text: bool,
    translation,
    hour_list_slice,
    is_day: bool,
    tz_name,
    icon_set: str,
    unit_system: str,
    fallback_text: str,
    fallback_icon: str,
    logger: logging.Logger,
    loc_tag: str,
    phase: str,
) -> Tuple[str, str]:
    """Apply PirateDayNightText for a half-day period when enabled."""
    if not summary_text:
        return fallback_text, fallback_icon

    try:
        day_icon, day_text = calculate_half_day_text(
            hour_list_slice,
            is_day,
            str(tz_name),
            icon_set=icon_set,
            unit_system=unit_system,
        )
        return translation.translate(["sentence", day_text]), day_icon
    except Exception:
        logger.exception("%s TEXT GEN ERROR %s", phase, loc_tag)
        return fallback_text, fallback_icon
