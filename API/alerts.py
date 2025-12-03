"""
Alert reading helpers for Pirate Weather API responses.
"""

import datetime
import logging
import re
from typing import Any, List

import numpy as np
from pytz import utc

NWS_ALERT_LONS = np.arange(-127, -65, 0.025)
NWS_ALERT_LATS = np.arange(24, 50, 0.025)
ALERT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def build_alerts(
    *,
    time_machine: bool,
    ex_alerts: int,
    lat: float,
    az_lon: float,
    nws_alerts_zarr: Any,
    wmo_alert_data: Any,
    read_wmo_alerts: bool,
    logger: logging.Logger,
    loc_tag: str,
) -> List[dict]:
    """
    Read any matching NWS or WMO alerts for the provided location.
    """
    if time_machine or ex_alerts != 0:
        return []

    now_utc = datetime.datetime.now(datetime.UTC).astimezone(utc)
    alerts: List[dict] = []

    try:
        alerts.extend(
            _read_nws_alerts(
                lat=lat,
                az_lon=az_lon,
                alerts_zarr=nws_alerts_zarr,
                now_utc=now_utc,
                logger=logger,
                loc_tag=loc_tag,
            )
        )
    except Exception:
        logger.exception("An Alert error occurred %s", loc_tag)

    try:
        alerts.extend(
            _read_wmo_alerts(
                wmo_alert_data=wmo_alert_data,
                read_wmo_alerts=read_wmo_alerts,
                now_utc=now_utc,
                logger=logger,
                loc_tag=loc_tag,
            )
        )
    except Exception:
        logger.exception("A WMO Alert error occurred %s", loc_tag)

    return alerts


def _read_nws_alerts(
    *,
    lat: float,
    az_lon: float,
    alerts_zarr: Any,
    now_utc: datetime.datetime,
    logger: logging.Logger,
    loc_tag: str,
) -> List[dict]:
    if alerts_zarr is None or not _in_us_bounds(lat, az_lon):
        return []

    try:
        alerts_y_p = int(np.argmin(np.abs(NWS_ALERT_LATS - lat)))
        alerts_x_p = int(np.argmin(np.abs(NWS_ALERT_LONS - az_lon)))
        alert_data = alerts_zarr[alerts_y_p, alerts_x_p]
    except Exception:
        logger.exception("Failed to read NWS alerts for %s", loc_tag)
        return []

    if alert_data in ("", None):
        return []

    alert_list: List[dict] = []
    for raw_alert in str(alert_data).split("|"):
        alert_details = raw_alert.split("}{")
        if len(alert_details) < 7:
            logger.debug("Skipping malformed NWS alert entry for %s", loc_tag)
            continue

        alert_onset = _parse_alert_time(alert_details[3])
        alert_end = _parse_alert_time(alert_details[4])
        if alert_onset is None or alert_end is None:
            logger.debug("Skipping NWS alert with invalid times for %s", loc_tag)
            continue

        formatted_text = _format_alert_description(alert_details[1])
        alert_dict = {
            "title": alert_details[0],
            "regions": [s.lstrip() for s in alert_details[2].split(";")],
            "severity": alert_details[5],
            "time": int(alert_onset.timestamp()),
            "expires": int(alert_end.timestamp()),
            "description": formatted_text,
            "uri": alert_details[6],
        }

        if alert_end > now_utc:
            alert_list.append(alert_dict)
        else:
            logger.debug("Skipping expired NWS alert: %s", alert_details[0])

    return alert_list


def _read_wmo_alerts(
    *,
    wmo_alert_data: Any,
    read_wmo_alerts: bool,
    now_utc: datetime.datetime,
    logger: logging.Logger,
    loc_tag: str,
) -> List[dict]:
    if (not read_wmo_alerts) or wmo_alert_data in ("", None):
        return []

    alert_list: List[dict] = []
    for raw_alert in str(wmo_alert_data).split("~"):
        alert_details = raw_alert.split("}{")
        if len(alert_details) < 3:
            logger.debug("Skipping malformed WMO alert entry for %s", loc_tag)
            continue

        alert_onset = (
            _parse_alert_time(alert_details[3]) if len(alert_details) > 3 else None
        )
        alert_end = (
            _parse_alert_time(alert_details[4]) if len(alert_details) > 4 else None
        )
        alert_severity = alert_details[5] if len(alert_details) > 5 else "Unknown"
        alert_uri = alert_details[6] if len(alert_details) > 6 else ""

        alert_dict = {
            "title": alert_details[0],
            "regions": [s.lstrip() for s in alert_details[2].split(";") if s.strip()],
            "severity": alert_severity,
            "time": int(alert_onset.timestamp()) if alert_onset else -999,
            "expires": int(alert_end.timestamp()) if alert_end else -999,
            "description": alert_details[1],
            "uri": alert_uri,
        }

        if (alert_end is None) or (alert_end > now_utc):
            alert_list.append(alert_dict)
        else:
            logger.debug("Skipping expired WMO alert: %s", alert_details[0])

    return alert_list


def _in_us_bounds(lat: float, az_lon: float) -> bool:
    return (-127 < az_lon < -65) and (24 < lat < 50)


def _parse_alert_time(value: str) -> datetime.datetime | None:
    if not value:
        return None
    try:
        return datetime.datetime.strptime(value, ALERT_TIME_FORMAT).astimezone(utc)
    except (TypeError, ValueError):
        return None


def _format_alert_description(alert_description: str) -> str:
    formatted_text = re.sub(r"(?<!\n)\n(?!\n)", " ", alert_description)
    return re.sub(r"\n\n", "\n", formatted_text)
