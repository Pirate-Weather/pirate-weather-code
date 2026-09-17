"""Live comparison tests for the produced URMA historical Zarr."""

import datetime
import json
import math
import os
import pickle
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import zarr

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
from API.constants.shared_const import INGEST_VERSION_STR
from API.utils.geo import lambertGridMatch

LOCAL_TEST_ENV_VAR = "PW_RUN_LOCAL_INGEST_TESTS"
GRIBSTREAM_API_KEY_ENV_VAR = "GRIBSTREAM_API_KEY"
URMA_OUTPUT_ROOT_ENV_VAR = "PW_URMA_OUTPUT_ROOT"
GRIBSTREAM_URMA_RUNS_URL = "https://gribstream.com/api/v2/urma/runs"
DEFAULT_URMA_OUTPUT_ROOT = Path("/mnt/nvme/data/Prod/URMA")
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


def _fetch_gribstream_urma_run(
    api_key: str, run_time, coordinates: list[dict]
) -> list[dict]:
    """Fetch selected fields for one URMA analysis at several grid points."""
    payload = {
        "timesList": [_format_utc(run_time)],
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
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.fixture(scope="module")
def urma_ingest_output() -> tuple[Path, Path]:
    """Locate the URMA output produced by the local ingest launch configuration."""
    if os.getenv(LOCAL_TEST_ENV_VAR) != "1":
        pytest.skip(
            "URMA live output comparison is opt-in; "
            f"set {LOCAL_TEST_ENV_VAR}=1 to run it"
        )

    output_root = Path(
        os.getenv(URMA_OUTPUT_ROOT_ENV_VAR, str(DEFAULT_URMA_OUTPUT_ROOT))
    )
    versioned_root = output_root / INGEST_VERSION_STR
    zarr_path = versioned_root / "URMA_Hist.zarr"
    time_pickle_path = versioned_root / "URMA_Hist.time.pickle"
    if not zarr_path.exists() or not time_pickle_path.exists():
        pytest.skip(
            f"URMA output is missing under {versioned_root}; run the URMA ingest first "
            f"or set {URMA_OUTPUT_ROOT_ENV_VAR}"
        )

    return zarr_path, time_pickle_path


@pytest.mark.skipif(
    not os.getenv(GRIBSTREAM_API_KEY_ENV_VAR),
    reason=f"set {GRIBSTREAM_API_KEY_ENV_VAR} to run the GribStream comparison",
)
def test_urma_zarr_matches_gribstream_at_multiple_points(urma_ingest_output):
    """Compare the newest URMA Zarr analysis with live GribStream values."""
    zarr_path, time_pickle_path = urma_ingest_output
    with time_pickle_path.open("rb") as file:
        run_time = pickle.load(file)

    coordinates, indices = _comparison_grid_points()
    try:
        rows = _fetch_gribstream_urma_run(
            os.environ[GRIBSTREAM_API_KEY_ENV_VAR],
            run_time,
            coordinates,
        )
    except HTTPError as exc:
        if exc.code in {429, 503, 504}:
            pytest.skip(f"GribStream temporarily unavailable: HTTP {exc.code}")
        pytest.fail(f"GribStream request failed: HTTP {exc.code}")
    except URLError as exc:
        pytest.skip(f"Could not reach GribStream: {exc.reason}")

    assert len(rows) == len(coordinates), (
        f"Expected {len(coordinates)} GribStream rows for URMA run "
        f"{_format_utc(run_time)}, got {len(rows)}"
    )
    rows_by_name = {row["name"]: row for row in rows}
    assert rows_by_name.keys() == indices.keys()

    zarr_array = zarr.open_array(str(zarr_path), mode="r")
    expected_timestamp = run_time.replace(tzinfo=datetime.UTC).timestamp()
    zarr_times = np.asarray(zarr_array[0, :, 0, 0], dtype=np.float64)
    time_index = int(np.argmin(np.abs(zarr_times - expected_timestamp)))
    assert zarr_times[time_index] == pytest.approx(expected_timestamp, abs=128)

    for name, (x_index, y_index) in indices.items():
        row = rows_by_name[name]
        assert row["forecasted_at"] == _format_utc(run_time)
        assert row["forecasted_time"] == _format_utc(run_time)
        for variable in GRIBSTREAM_VARIABLES:
            local_value = float(
                zarr_array[
                    variable["zarr_index"],
                    time_index,
                    y_index,
                    x_index,
                ]
            )
            assert local_value == pytest.approx(
                row[variable["alias"]],
                abs=variable["absolute_tolerance"],
                rel=1e-4,
            ), f"{name}: {variable['alias']}"
