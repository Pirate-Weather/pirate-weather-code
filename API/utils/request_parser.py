from __future__ import annotations

import datetime as _dt
from typing import Tuple

from fastapi import HTTPException
from timezonefinder import TimezoneFinder

from .weather_utils import get_offset


def parse_location(
    location: str,
    now_time: _dt.datetime,
    stage: str,
    tf: TimezoneFinder,
) -> Tuple[float, float, float, float, _dt.datetime, bool]:
    """Parse a ``lat,lon[,time]`` location string.

    Returns latitude, longitude (0-360), adjusted longitude (-180-180),
    the requested UTC time, and whether the request should use timemachine.
    """

    parts = location.split(",")
    try:
        lat = float(parts[0])
        lon_in = float(parts[1])
    except Exception as exc:
        raise HTTPException(400, "Invalid Location Specification") from exc

    if lon_in < -180 or lon_in > 360:
        raise HTTPException(400, "Invalid Longitude")
    if lat < -90 or lat > 90:
        raise HTTPException(400, "Invalid Latitude")

    lon = lon_in % 360
    az_lon = ((lon + 180) % 360) - 180

    if len(parts) == 2:
        if stage == "TIMEMACHINE":
            raise HTTPException(400, "Missing Time Specification")
        utc_time = now_time
    elif len(parts) == 3:
        time_part = parts[2]
        if time_part.lstrip("-+").isnumeric():
            val = float(time_part)
            if val > 0 or val < -100000:
                utc_time = _dt.datetime.utcfromtimestamp(val)
            else:
                utc_time = now_time + _dt.timedelta(seconds=val)
        else:
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%Z", "%Y-%m-%dT%H:%M:%S"):
                try:
                    parsed = _dt.datetime.strptime(time_part, fmt)
                    if fmt == "%Y-%m-%dT%H:%M:%S":
                        tz_off, _ = get_offset(lat=lat, lng=az_lon, utcTime=parsed, tf=tf)
                        parsed -= _dt.timedelta(minutes=tz_off)
                    utc_time = parsed.replace(tzinfo=None)
                    break
                except Exception:
                    continue
            else:  # pragma: no cover - invalid format
                raise HTTPException(400, "Invalid Time Specification")
    else:
        raise HTTPException(400, "Invalid Time or Location Specification")

    time_machine = False
    if utc_time < _dt.datetime(2024, 5, 1):
        time_machine = True
    elif now_time - utc_time > _dt.timedelta(hours=25):
        time_machine = True
    elif utc_time > now_time and (utc_time - now_time) >= _dt.timedelta(hours=1):
        raise HTTPException(400, "Requested Time is in the Future")
    else:
        utc_time = min(utc_time, now_time)

    return lat, lon, az_lon, utc_time, time_machine
