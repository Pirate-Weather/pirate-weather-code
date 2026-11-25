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
    HourColumns, the same hours as ``hour_list_si``) saves building them.
    ``time_machine`` is a single-day (Time Machine or future-date) request."""
    # Without summary text the block text and icon are the most common hourly
    # ones. most_common counts with a Counter, so a tie goes to the value of
    # the earliest hour carrying a tied value; picking from a set made the
    # winner depend on the process's string hash seed, so pods disagreed on
    # the same request.
    try:
        if summary_text:
            if time_machine:
                # A requested date is summarised as the forecast's next 24 hours
                # are (#711), over the first 24 hours from its local midnight.
                # Its periods are named as a daily summary's ("overnight", "in
                # the morning"): the next-24-hours names ("tonight", "this
                # morning") are relative to now, wrong for any other date.
                first_hour = 0
                period_mode = "day"
            else:
                first_hour = base_time_offset_int
                period_mode = "hour"
            next_24_hours = slice(first_hour, first_hour + 24)
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
                period_mode,
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
    """Compute daily summary/icon text. ``time_machine`` is a single-day (Time
    Machine or future-date) request."""
    # As for the hourly block, a tie goes to the earliest day's value.
    try:
        if summary_text:
            if time_machine and len(day_list_si) == 1:
                # One requested day: the block summary is that day's summary.
                return day_text_list[0], day_icon_list[0]
            # Days of a requested range are named by weekday ("on Monday"), not
            # relative to now ("today", "tomorrow") as a forecast's are.
            weekText, weekIcon = calculate_weekly_text(
                day_list_si,
                str(tz_name),
                unit_system,
                icon,
                relative_day_names=not time_machine,
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
