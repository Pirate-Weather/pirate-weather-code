"""Shared URMA coordinates and GribStream comparison helpers."""

import datetime
import json
import math
from urllib.request import Request, urlopen

from API.constants.grid_const import (
    RTMA_RU_AXIS,
    RTMA_RU_CENTRAL_LAT,
    RTMA_RU_CENTRAL_LONG,
    RTMA_RU_DELTA,
    RTMA_RU_MIN_X,
    RTMA_RU_MIN_Y,
    RTMA_RU_PARALLEL,
)
from API.constants.model_const import URMA
from API.utils.geo import lambertGridMatch

GRIBSTREAM_URMA_RUNS_URL = "https://gribstream.com/api/v2/urma/runs"
GRIBSTREAM_COMPARISON_LOCATIONS = (
    {"name": "denver", "lat": 40.0, "lon": -105.0},
    {"name": "new_york", "lat": 40.7, "lon": -74.0},
    {"name": "seattle", "lat": 47.6, "lon": -122.3},
)
GRIBSTREAM_VARIABLES = (
    {
        "alias": "temperature",
        "zarr_index": URMA["temp"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "TMP", "level": "2 m above ground", "info": ""},
    },
    {
        "alias": "dew_point",
        "zarr_index": URMA["dew"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "DPT", "level": "2 m above ground", "info": ""},
    },
    {
        "alias": "wind_gust",
        "zarr_index": URMA["gust"],
        "absolute_tolerance": 0.1,
        "selector": {
            "name": "GUST",
            "level": "10 m above ground",
            "info": "",
        },
    },
    {
        "alias": "visibility",
        "zarr_index": URMA["vis"],
        "absolute_tolerance": 100.0,
        "selector": {"name": "VIS", "level": "surface", "info": ""},
    },
    {
        "alias": "pressure",
        "zarr_index": URMA["pressure"],
        "absolute_tolerance": 10.0,
        "selector": {"name": "PRES", "level": "surface", "info": ""},
    },
)


def _format_utc(value) -> str:
    """Format a datetime-like value as the UTC representation GribStream expects."""
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _urma_grid_point(lat: float, lon: float) -> tuple[float, float, int, int]:
    """Return the nearest URMA grid point and its array indices."""
    grid_lat, grid_lon, x_index, y_index = lambertGridMatch(
        math.radians(RTMA_RU_CENTRAL_LONG),
        math.radians(RTMA_RU_CENTRAL_LAT),
        math.radians(RTMA_RU_PARALLEL),
        RTMA_RU_AXIS,
        lat,
        lon % 360,
        RTMA_RU_MIN_X,
        RTMA_RU_MIN_Y,
        RTMA_RU_DELTA,
    )
    normalized_lon = ((grid_lon + 180) % 360) - 180
    return grid_lat, normalized_lon, x_index, y_index


def _comparison_grid_points() -> tuple[list[dict], dict[str, tuple[int, int]]]:
    """Build named GribStream coordinates and their corresponding Zarr indices."""
    coordinates = []
    indices = {}
    for location in GRIBSTREAM_COMPARISON_LOCATIONS:
        grid_lat, grid_lon, x_index, y_index = _urma_grid_point(
            location["lat"], location["lon"]
        )
        coordinates.append({"name": location["name"], "lat": grid_lat, "lon": grid_lon})
        indices[location["name"]] = (x_index, y_index)
    return coordinates, indices


def _fetch_gribstream_urma(
    api_key: str, run_times: list, coordinates: list[dict]
) -> list[dict]:
    """Fetch selected fields for one or more URMA analyses at several grid points."""
    payload = {
        "timesList": [_format_utc(run_time) for run_time in run_times],
        "minLeadTime": "0h",
        "maxLeadTime": "0h",
        "coordinates": coordinates,
        "variables": [
            {**variable["selector"], "alias": variable["alias"]}
            for variable in GRIBSTREAM_VARIABLES
        ],
    }
    request = Request(
        GRIBSTREAM_URMA_RUNS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))
