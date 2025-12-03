"""Legacy hourly text helpers kept opt-in for backward compatibility."""

from __future__ import annotations

from typing import Tuple


from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    PRECIP_PROB_THRESHOLD,
    WIND_THRESHOLDS,
)


def apply_legacy_hourly_text(
    *,
    hour_item_si,
    is_day: bool,
) -> Tuple[str, str]:
    """Apply PirateText hourly text/icon generation when requested."""
    
    # Explicit logic from block.py
    prob = hour_item_si["precipProbability"]
    rain = hour_item_si["liquidAccumulation"]
    ice = hour_item_si["iceAccumulation"]
    snow = hour_item_si["snowAccumulation"]
    vis = hour_item_si["visibility"]
    wind = hour_item_si["windSpeed"]
    cloud = hour_item_si["cloudCover"]
    ptype = hour_item_si["precipType"]

    if prob >= PRECIP_PROB_THRESHOLD and (
        ((rain + ice) > HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM)
        or (snow > HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM)
    ):
        hour_icon = ptype
        hour_text = ptype.title() if ptype != "none" else "None"
    elif vis < FOG_THRESHOLD_METERS:
        hour_icon = "fog"
        hour_text = "Fog"
    elif wind > WIND_THRESHOLDS["light"]:
        hour_icon = "wind"
        hour_text = "Windy"
    elif cloud > CLOUD_COVER_THRESHOLDS["cloudy"]:
        hour_icon = "cloudy"
        hour_text = "Cloudy"
    elif cloud > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        hour_text = "Partly Cloudy"
        hour_icon = "partly-cloudy-day" if is_day else "partly-cloudy-night"
    else:
        hour_text = "Clear"
        hour_icon = "clear-day" if is_day else "clear-night"

    return  hour_text, hour_icon

