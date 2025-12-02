"""Legacy summary/icon helpers kept opt-in for backward compatibility."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Tuple

from API.PirateMinutelyText import calculate_minutely_text
from API.PirateWeeklyText import calculate_weekly_text
from API.PirateDailyText import calculate_day_text


def build_minutely_summary(
    *,
    summary_text: bool,
    translation,
    inter_p_current,
    inter_p_hour,
    minute_items_si,
    current_text,
    current_icon,
    icon: str,
    max_p_chance,
    p_types_text,
    p_types_icon,
    logger: logging.Logger,
    loc_tag: str,
) -> Tuple[str, str]:
    """Compute minutely summary/icon text."""
    if summary_text:
        max_cape = max(inter_p_current, inter_p_hour)
        minute_text, minute_icon = calculate_minutely_text(
            minute_items_si, current_text, current_icon, icon, max_cape
        )
        return translation.translate(["sentence", minute_text]), minute_icon

    try:
        dominant = int(Counter(max_p_chance).most_common(1)[0][0])
        return p_types_text[dominant], p_types_icon[dominant]
    except Exception:
        logger.exception("MINUTELY TEXT GEN ERROR %s", loc_tag)
        dominant = int(Counter(max_p_chance).most_common(1)[0][0])
        return p_types_text[dominant], p_types_icon[dominant]


def build_hourly_summary(
    *,
    summary_text: bool,
    translation,
    hour_list_si,
    is_all_night: bool,
    tz_name: str,
    icon: str,
    unit_system: str,
    hour_text_list,
    hour_icon_list,
    time_machine: bool,
    base_time_offset_int: int,
    logger: logging.Logger,
    loc_tag: str,
) -> Tuple[str, str]:
    """Compute hourly summary/icon text."""
    if time_machine:
        return (
            max(set(hour_text_list), key=hour_text_list.count),
            max(set(hour_icon_list), key=hour_icon_list.count),
        )

    try:
        if summary_text:
            hourIcon, hourText = calculate_day_text(
                hour_list_si[base_time_offset_int : base_time_offset_int + 24],
                not is_all_night,
                str(tz_name),
                "hour",
                icon,
                unit_system,
            )
            return translation.translate(["sentence", hourText]), hourIcon
        return (
            max(set(hour_text_list), key=hour_text_list.count),
            max(set(hour_icon_list), key=hour_icon_list.count),
        )
    except Exception:
        logger.exception("TEXT GEN ERROR %s", loc_tag)
        return (
            max(set(hour_text_list), key=hour_text_list.count),
            max(set(hour_icon_list), key=hour_icon_list.count),
        )


def build_daily_summary(
    *,
    summary_text: bool,
    translation,
    day_list_si,
    tz_name: str,
    unit_system: str,
    icon: str,
    day_text_list,
    day_icon_list,
    time_machine: bool,
    logger: logging.Logger,
    loc_tag: str,
) -> Tuple[str, str]:
    """Compute daily summary/icon text."""
    if time_machine:
        return (
            max(set(day_text_list), key=day_text_list.count),
            max(set(day_icon_list), key=day_icon_list.count),
        )

    try:
        if summary_text:
            weekText, weekIcon = calculate_weekly_text(
                day_list_si, str(tz_name), unit_system, icon
            )
            return translation.translate(["sentence", weekText]), weekIcon
        return (
            max(set(day_text_list), key=day_text_list.count),
            max(set(day_icon_list), key=day_icon_list.count),
        )
    except Exception:
        logger.exception("DAILY SUMMARY TEXT GEN ERROR %s", loc_tag)
        return (
            max(set(day_text_list), key=day_text_list.count),
            max(set(day_icon_list), key=day_icon_list.count),
        )
