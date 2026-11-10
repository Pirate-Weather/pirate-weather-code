"""Legacy summary/icon helpers kept opt-in for backward compatibility."""

from __future__ import annotations

import logging
from collections import Counter

from API.PirateDailyText import HourColumns, calculate_day_text
from API.PirateMinutelyText import calculate_minutely_text
from API.PirateTextHelper import most_common
from API.PirateWeeklyText import calculate_weekly_text


def build_minutely_summary(
    *,
    summary_text: bool,
    translation,
    minute_items_si,
    current_icon,
    icon: str,
    max_p_chance,
    p_types_text,
    p_types_icon,
    logger: logging.Logger,
    loc_tag: str,
) -> tuple[str, str]:
    """Compute minutely summary/icon text."""
    if summary_text:
        try:
            minute_text, minute_icon = calculate_minutely_text(
                minute_items_si, current_icon, icon
            )
            return translation.translate(["sentence", minute_text]), minute_icon
        except Exception:
            logger.exception("MINUTELY TEXT GEN ERROR %s", loc_tag)

    # Without summary text, or if it fails, use the most common precipitation
    # text/icon from the minute items
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
    hour_columns: HourColumns | None = None,
) -> tuple[str, str]:
    """Compute hourly summary/icon text. ``hour_columns`` (the forecast's
    HourColumns, the same hours as ``hour_list_si``) saves building them."""
    # The block text and icon are the most common hourly ones. most_common
    # counts with a Counter, so a tie goes to the value of the earliest hour
    # carrying a tied value; picking from a set made the winner depend on the
    # process's string hash seed, so pods disagreed on the same request.
    if time_machine:
        return (
            most_common(hour_text_list),
            most_common(hour_icon_list),
        )

    try:
        if summary_text:
            next_24_hours = slice(base_time_offset_int, base_time_offset_int + 24)
            if hour_columns is not None:
                next_24_hours = hour_columns.window(
                    next_24_hours.start, min(next_24_hours.stop, len(hour_columns))
                )
            else:
                next_24_hours = hour_list_si[next_24_hours]
            hourIcon, hourText = calculate_day_text(
                next_24_hours,
                not is_all_night,
                str(tz_name),
                "hour",
                icon,
                unit_system,
            )
            return translation.translate(["sentence", hourText]), hourIcon
        return (
            most_common(hour_text_list),
            most_common(hour_icon_list),
        )
    except Exception:
        logger.exception("TEXT GEN ERROR %s", loc_tag)
        return (
            most_common(hour_text_list),
            most_common(hour_icon_list),
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
) -> tuple[str, str]:
    """Compute daily summary/icon text."""
    # As for the hourly block, a tie goes to the earliest day's value.
    if time_machine:
        return (
            most_common(day_text_list),
            most_common(day_icon_list),
        )

    try:
        if summary_text:
            weekText, weekIcon = calculate_weekly_text(
                day_list_si, str(tz_name), unit_system, icon
            )
            return translation.translate(["sentence", weekText]), weekIcon
        return (
            most_common(day_text_list),
            most_common(day_icon_list),
        )
    except Exception:
        logger.exception("DAILY SUMMARY TEXT GEN ERROR %s", loc_tag)
        return (
            most_common(day_text_list),
            most_common(day_icon_list),
        )
