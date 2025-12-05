"""Legacy current summary generation."""

from API.constants.forecast_const import DATA_CURRENT, DATA_DAY
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    WIND_THRESHOLDS,
)


def get_legacy_current_summary(
    minuteItems, prepIntensityUnit, InterPcurrent, InterSday
):
    """Generate legacy summary text and icon."""
    cIcon = None
    cText = None

    if (
        (minuteItems[0]["precipIntensity"])
        > (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepIntensityUnit)
    ) & (minuteItems[0]["precipType"] is not None):
        cIcon = minuteItems[0]["precipType"]
        cText = (
            minuteItems[0]["precipType"][0].upper() + minuteItems[0]["precipType"][1:]
        )

    elif InterPcurrent[DATA_CURRENT["vis"]] < FOG_THRESHOLD_METERS:
        cIcon = "fog"
        cText = "Fog"
    elif InterPcurrent[DATA_CURRENT["wind"]] > WIND_THRESHOLDS["light"]:
        cIcon = "wind"
        cText = "Windy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
        cIcon = "cloudy"
        cText = "Cloudy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        cText = "Partly Cloudy"

        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            cIcon = "partly-cloudy-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            cIcon = "partly-cloudy-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            cIcon = "partly-cloudy-night"
    else:
        cText = "Clear"
        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            cIcon = "clear-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            cIcon = "clear-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            cIcon = "clear-night"

    return cText, cIcon
