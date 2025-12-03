"""Legacy daily and half-day text helpers kept opt-in for backward compatibility."""

from __future__ import annotations

from typing import Tuple

from API.PirateDayNightText import calculate_half_day_text


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
    logger,
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
