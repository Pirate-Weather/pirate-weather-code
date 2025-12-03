"""Legacy hourly text helpers kept opt-in for backward compatibility."""

from __future__ import annotations

from typing import Tuple

from API.PirateText import calculate_text


def apply_legacy_hourly_text(
    *,
    summary_text: bool,
    translation,
    hour_item_si,
    is_day: bool,
    icon: str,
    fallback_text: str,
    fallback_icon: str,
) -> Tuple[str, str]:
    """Apply PirateText hourly text/icon generation when requested."""
    if not summary_text:
        return fallback_text, fallback_icon

    try:
        hour_text, hour_icon = calculate_text(hour_item_si, is_day, "hour", icon)
        return translation.translate(["title", hour_text]), hour_icon
    except Exception:
        return fallback_text, fallback_icon
