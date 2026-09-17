"""Test for RTMA-RU ingest script functionality."""

import ast
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

from API.constants.grid_const import (
    RTMA_RU_AXIS,
    RTMA_RU_CENTRAL_LAT,
    RTMA_RU_CENTRAL_LONG,
    RTMA_RU_DELTA,
    RTMA_RU_MIN_X,
    RTMA_RU_MIN_Y,
    RTMA_RU_PARALLEL,
)
from API.constants.model_const import RTMA_RU
from API.constants.shared_const import INGEST_VERSION_STR
from API.utils.geo import lambertGridMatch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "API" / "RTMA-RU_Local_Ingest.py"
EXPECTED_VAR_COUNT = 10
GRIBSTREAM_API_KEY_ENV_VAR = "GRIBSTREAM_API_KEY"
GRIBSTREAM_RTMA_RU_RUNS_URL = "https://gribstream.com/api/v2/rtmaru/runs"
GRIBSTREAM_COMPARISON_LOCATION = (40.0, -105.0)
GRIBSTREAM_VARIABLES = (
    {
        "alias": "temperature",
        "zarr_index": RTMA_RU["temp"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "TMP", "level": "2 m above ground", "info": ""},
    },
    {
        "alias": "dew_point",
        "zarr_index": RTMA_RU["dew"],
        "absolute_tolerance": 0.1,
        "selector": {"name": "DPT", "level": "2 m above ground", "info": ""},
    },
    {
        "alias": "wind_gust",
        "zarr_index": RTMA_RU["gust"],
        "absolute_tolerance": 0.1,
        "selector": {
            "name": "GUST",
            "level": "10 m above ground",
            "info": "",
        },
    },
    {
        "alias": "visibility",
        "zarr_index": RTMA_RU["vis"],
        "absolute_tolerance": 100.0,
        "selector": {"name": "VIS", "level": "surface", "info": ""},
    },
    {
        "alias": "pressure",
        "zarr_index": RTMA_RU["pressure"],
        "absolute_tolerance": 10.0,
        "selector": {"name": "PRES", "level": "surface", "info": ""},
    },
    {
        "alias": "cloud_cover",
        "zarr_index": RTMA_RU["cloud"],
        "absolute_tolerance": 1.0,
        "selector": {
            "name": "TCDC",
            "level": "entire atmosphere (considered as a single layer)",
            "info": "",
        },
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


def _rtma_ru_grid_point(lat: float, lon: float) -> tuple[float, float, int, int]:
    """Return the nearest RTMA-RU grid point and its array indices."""
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


def _fetch_gribstream_rtma_ru_run(
    api_key: str, run_time, lat: float, lon: float
) -> list[dict]:
    """Fetch fields for one exact RTMA-RU analysis run."""
    payload = {
        "timesList": [_format_utc(run_time)],
        "coordinates": [{"lat": lat, "lon": lon}],
        "variables": [
            {**variable["selector"], "alias": variable["alias"]}
            for variable in GRIBSTREAM_VARIABLES
        ],
    }
    request = Request(
        GRIBSTREAM_RTMA_RU_RUNS_URL,
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
def rtma_ru_ingest_output():
    """Run RTMA-RU ingest once and retain its output for module tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["forecast_process_dir"] = os.path.join(tmpdir, "RTMA_RU")
        env["forecast_path"] = os.path.join(tmpdir, "Prod", "RTMA_RU")
        env["save_type"] = "Download"
        env["AWS_KEY"] = ""
        env["AWS_SECRET"] = ""
        env["PYTHONPATH"] = _build_pythonpath(env)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        assert result.returncode == 0, (
            f"RTMA-RU ingest failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        output_root = Path(env["forecast_path"]) / INGEST_VERSION_STR
        zarr_path = output_root / "RTMA_RU.zarr"
        time_pickle_path = output_root / "RTMA_RU.time.pickle"

        yield zarr_path, time_pickle_path, result


def test_rtma_ingest_produces_zarr(rtma_ru_ingest_output):
    """Verify the live RTMA-RU ingest writes a readable Zarr store."""
    zarr_path, time_pickle_path, result = rtma_ru_ingest_output

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
    assert zarr_array.shape[1] == 1, (
        f"Expected one analysis time, got shape {zarr_array.shape}"
    )
    assert zarr_array.shape[2] > 0 and zarr_array.shape[3] > 0, (
        f"Expected non-empty spatial dimensions, got {zarr_array.shape}"
    )


def test_gribstream_request_pins_rtma_ru_run(monkeypatch):
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

    monkeypatch.setattr("tests.test_rtma_ingest.urlopen", fake_urlopen)
    run_time = datetime.datetime(2026, 9, 9, 13, 30, tzinfo=datetime.UTC)

    rows = _fetch_gribstream_rtma_ru_run("test-token", run_time, 40.0, -105.0)

    assert rows == [{"temperature": 280.0}]
    assert captured["timeout"] == 30
    request = captured["request"]
    assert request.full_url == GRIBSTREAM_RTMA_RU_RUNS_URL
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer test-token"
    assert request.get_header("Accept") == "application/json"
    payload = json.loads(request.data)
    assert payload["timesList"] == ["2026-09-09T13:30:00Z"]
    assert payload["coordinates"] == [{"lat": 40.0, "lon": -105.0}]
    assert [variable["alias"] for variable in payload["variables"]] == [
        "temperature",
        "dew_point",
        "wind_gust",
        "visibility",
        "pressure",
        "cloud_cover",
    ]


def test_projection_import_does_not_require_timezone_dependencies():
    """The ingest comparison should not require API-only timezone packages."""
    source = """
import builtins

real_import = builtins.__import__

def reject_timezone_imports(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'pytz', 'timezonefinder'}:
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_timezone_imports
from API.utils.geo import lambertGridMatch
assert callable(lambertGridMatch)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not os.getenv(GRIBSTREAM_API_KEY_ENV_VAR),
    reason=f"set {GRIBSTREAM_API_KEY_ENV_VAR} to run the GribStream comparison",
)
def test_rtma_ru_ingest_matches_gribstream(rtma_ru_ingest_output):
    """Compare selected ingested RTMA-RU fields with the matching GribStream run."""
    zarr_path, time_pickle_path, _ = rtma_ru_ingest_output
    with time_pickle_path.open("rb") as file:
        run_time = pickle.load(file)

    grid_lat, grid_lon, x_index, y_index = _rtma_ru_grid_point(
        *GRIBSTREAM_COMPARISON_LOCATION
    )
    try:
        rows = _fetch_gribstream_rtma_ru_run(
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
        f"Expected one GribStream row for RTMA-RU run {_format_utc(run_time)}, "
        f"got {len(rows)}"
    )
    row = rows[0]
    assert row["forecasted_at"] == _format_utc(run_time)

    zarr_array = zarr.open(str(zarr_path), mode="r")
    for variable in GRIBSTREAM_VARIABLES:
        local_value = float(zarr_array[variable["zarr_index"], 0, y_index, x_index])
        assert local_value == pytest.approx(
            row[variable["alias"]],
            abs=variable["absolute_tolerance"],
            rel=1e-4,
        ), variable["alias"]


def test_rtma_script_exists():
    """Test that RTMA-RU_Local_Ingest.py exists."""
    assert SCRIPT_PATH.exists(), f"RTMA script not found at {SCRIPT_PATH}"
    assert SCRIPT_PATH.is_file(), "RTMA script is not a file"


def test_rtma_script_is_valid_python():
    """Test that RTMA-RU_Local_Ingest.py is valid Python syntax."""
    # Read the script content
    script_content = SCRIPT_PATH.read_text()

    # Try to parse it as Python code
    try:
        ast.parse(script_content)
    except SyntaxError as e:
        pytest.fail(f"RTMA script has invalid Python syntax: {e}")


def test_rtma_script_has_required_imports():
    """Test that the RTMA script has all required imports."""
    # Read the script content
    script_content = SCRIPT_PATH.read_text()

    # Check for required imports
    required_imports = [
        "import numpy",
        "import s3fs",
        "import xarray",
        "import zarr",
        "from herbie import Herbie",
        "from herbie.fast import Herbie_latest",
        "from metpy.calc import relative_humidity_from_specific_humidity",
    ]

    for import_stmt in required_imports:
        assert import_stmt in script_content, f"Missing required import: {import_stmt}"


def test_rtma_script_has_required_components():
    """Test that the RTMA script contains expected components."""
    # Read the script content
    script_content = SCRIPT_PATH.read_text()

    # Check for key processing steps
    assert "zarr_vars" in script_content, "Missing zarr_vars definition"
    assert "Herbie_latest" in script_content, "Missing Herbie_latest usage"
    assert "base_time" in script_content, "Missing base_time variable"
    assert "match_strings" in script_content, "Missing match_strings definition"

    # Check for RTMA-specific elements
    assert "rtma_ru" in script_content.lower(), "Script should reference rtma_ru model"

    # Check for key variables
    assert "vis" in script_content, "Missing visibility variable"
    assert "t2m" in script_content, "Missing temperature variable"
    assert "u10" in script_content, "Missing u-wind variable"
    assert "v10" in script_content, "Missing v-wind variable"


def test_rtma_script_python_check():
    """Test that the RTMA script can be checked with python -m py_compile."""
    # Use py_compile to check if the script compiles
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"Script failed to compile: {result.stderr}"
