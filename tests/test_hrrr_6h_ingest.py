"""Live ingest test for the HRRR_6H ingest script."""

import datetime
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
import zarr

from API.constants.model_const import HRRR
from API.constants.shared_const import INGEST_VERSION_STR
from API.utils.geo import lambertGridMatch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "API" / "HRRR_6H_Local_Ingest.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VAR_COUNT = 20
LOCAL_TEST_ENV_VAR = "PW_RUN_LOCAL_INGEST_TESTS"
GRIBSTREAM_API_KEY_ENV_VAR = "GRIBSTREAM_API_KEY"
GRIBSTREAM_HRRR_RUNS_URL = "https://gribstream.com/api/v2/hrrr/runs"
GRIBSTREAM_COMPARISON_LEAD_HOURS = 18
GRIBSTREAM_COMPARISON_LOCATION = (40.0, -105.0)
GRIBSTREAM_VARIABLES = (
    {
        "alias": "temperature",
        "zarr_index": HRRR["temp"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "TMP", "level": "2 m above ground", "info": ""},
    },
    {
        "alias": "dew_point",
        "zarr_index": HRRR["dew"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "DPT", "level": "2 m above ground", "info": ""},
    },
    {
        "alias": "wind_gust",
        "zarr_index": HRRR["gust"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "GUST", "level": "surface", "info": ""},
    },
    {
        "alias": "visibility",
        "zarr_index": HRRR["vis"],
        "absolute_tolerance": 100.0,
        "selector": {"name": "VIS", "level": "surface", "info": ""},
    },
)


def _build_pythonpath(env: dict[str, str]) -> str:
    existing_path = env.get("PYTHONPATH", "")
    repo_root = str(REPO_ROOT)
    return repo_root if not existing_path else repo_root + os.pathsep + existing_path


def _format_utc(value) -> str:
    """Format a datetime-like value as the UTC representation GribStream expects."""
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hrrr_grid_point(lat: float, lon: float) -> tuple[float, float, int, int]:
    """Return the nearest HRRR grid point and its array indices."""
    grid_lat, grid_lon, x_index, y_index = lambertGridMatch(
        math.radians(262.5),
        math.radians(38.5),
        math.radians(38.5),
        6371229,
        lat,
        lon % 360,
        -2697500,
        -1587300,
        3000,
    )
    normalized_lon = ((grid_lon + 180) % 360) - 180
    return grid_lat, normalized_lon, x_index, y_index


def _fetch_gribstream_hrrr_run(
    api_key: str, run_time, lat: float, lon: float
) -> list[dict]:
    """Fetch the HRRR fields used to validate one ingested forecast hour."""
    lead_time = f"{GRIBSTREAM_COMPARISON_LEAD_HOURS}h"
    payload = {
        "timesList": [_format_utc(run_time)],
        "minLeadTime": lead_time,
        "maxLeadTime": lead_time,
        "coordinates": [{"lat": lat, "lon": lon}],
        "variables": [
            {**variable["selector"], "alias": variable["alias"]}
            for variable in GRIBSTREAM_VARIABLES
        ],
    }
    request = Request(
        GRIBSTREAM_HRRR_RUNS_URL,
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


def test_gribstream_request_pins_hrrr_run_and_lead_time(monkeypatch):
    """Build the documented GribStream request without making a network call."""
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b'[{"temperature": 280.0}]'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("tests.test_hrrr_6h_ingest.urlopen", fake_urlopen)
    run_time = datetime.datetime(2026, 9, 9, 6, tzinfo=datetime.UTC)

    rows = _fetch_gribstream_hrrr_run("test-token", run_time, 40.0, -105.0)

    assert rows == [{"temperature": 280.0}]
    assert captured["timeout"] == 30
    request = captured["request"]
    assert request.full_url == GRIBSTREAM_HRRR_RUNS_URL
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert request.get_header("Accept") == "application/json"
    payload = json.loads(request.data)
    assert payload["timesList"] == ["2026-09-09T06:00:00Z"]
    assert payload["minLeadTime"] == "18h"
    assert payload["maxLeadTime"] == "18h"
    assert payload["coordinates"] == [{"lat": 40.0, "lon": -105.0}]
    assert [variable["alias"] for variable in payload["variables"]] == [
        "temperature",
        "dew_point",
        "wind_gust",
        "visibility",
    ]


@pytest.fixture(scope="module")
def hrrr_6h_ingest_output():
    """Run HRRR_6H ingest once and retain its output for module tests."""
    if os.getenv(LOCAL_TEST_ENV_VAR) != "1":
        pytest.skip(
            "HRRR_6H live ingest test requires a local wgrib2-enabled environment; "
            f"set {LOCAL_TEST_ENV_VAR}=1 to run it"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["forecast_process_dir"] = os.path.join(tmpdir, "HRRR_6H")
        env["forecast_path"] = os.path.join(tmpdir, "Prod", "HRRR_6H")
        env["save_type"] = "Download"
        env["AWS_KEY"] = ""
        env["AWS_SECRET"] = ""
        env["PYTHONPATH"] = _build_pythonpath(env)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )

        assert result.returncode == 0, (
            f"HRRR_6H ingest failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        output_root = Path(env["forecast_path"]) / INGEST_VERSION_STR
        zarr_path = output_root / "HRRR_6H.zarr"
        time_pickle_path = output_root / "HRRR_6H.time.pickle"

        yield zarr_path, time_pickle_path, result


def test_hrrr_6h_ingest_produces_zarr(hrrr_6h_ingest_output):
    """Verify the live HRRR_6H ingest writes a readable Zarr store."""
    zarr_path, time_pickle_path, result = hrrr_6h_ingest_output

    assert zarr_path.exists(), (
        f"Expected zarr output at {zarr_path}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert time_pickle_path.exists(), (
        f"Expected timestamp pickle at {time_pickle_path}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    zarr_array = zarr.open(str(zarr_path), mode="r")

    assert isinstance(zarr_array, zarr.Array), (
        f"Expected a zarr array at {zarr_path}, got {type(zarr_array).__name__}"
    )
    assert zarr_array.ndim == 4, f"Expected 4D zarr output, got {zarr_array.ndim}D"
    assert zarr_array.shape[0] == EXPECTED_VAR_COUNT, (
        f"Expected {EXPECTED_VAR_COUNT} variables, got shape {zarr_array.shape}"
    )
    assert zarr_array.shape[1] > 0, (
        f"Expected time dimension > 0, got {zarr_array.shape}"
    )
    assert zarr_array.shape[2] > 0 and zarr_array.shape[3] > 0, (
        f"Expected non-empty spatial dimensions, got {zarr_array.shape}"
    )


@pytest.mark.skipif(
    not os.getenv(GRIBSTREAM_API_KEY_ENV_VAR),
    reason=f"set {GRIBSTREAM_API_KEY_ENV_VAR} to run the GribStream comparison",
)
def test_hrrr_6h_ingest_matches_gribstream(hrrr_6h_ingest_output):
    """Compare selected ingested HRRR fields with the matching GribStream run."""
    zarr_path, time_pickle_path, _ = hrrr_6h_ingest_output
    with time_pickle_path.open("rb") as file:
        run_time = pickle.load(file)

    grid_lat, grid_lon, x_index, y_index = _hrrr_grid_point(
        *GRIBSTREAM_COMPARISON_LOCATION
    )
    try:
        rows = _fetch_gribstream_hrrr_run(
            os.environ[GRIBSTREAM_API_KEY_ENV_VAR],
            run_time,
            grid_lat,
            grid_lon,
        )
    except HTTPError as exc:
        if exc.code in {429, 503, 504}:
            pytest.skip(f"GribStream temporarily unavailable: HTTP {exc.code}")
        pytest.fail(f"GribStream request failed: HTTP {exc.code}")
    except URLError as exc:
        pytest.skip(f"Could not reach GribStream: {exc.reason}")

    assert len(rows) == 1, (
        "Expected one GribStream row for HRRR run "
        f"{_format_utc(run_time)} at lead {GRIBSTREAM_COMPARISON_LEAD_HOURS}h, "
        f"got {len(rows)}"
    )
    row = rows[0]
    expected_valid_time = run_time + datetime.timedelta(
        hours=GRIBSTREAM_COMPARISON_LEAD_HOURS
    )
    assert row["forecasted_at"] == _format_utc(run_time)
    assert row["forecasted_time"] == _format_utc(expected_valid_time)

    zarr_array = zarr.open(str(zarr_path), mode="r")
    for variable in GRIBSTREAM_VARIABLES:
        local_value = float(zarr_array[variable["zarr_index"], 0, y_index, x_index])
        assert local_value == pytest.approx(
            row[variable["alias"]],
            abs=variable["absolute_tolerance"],
            rel=1e-4,
        ), variable["alias"]
