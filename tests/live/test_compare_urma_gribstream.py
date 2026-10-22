"""Compare a local URMA Time Machine day with matching GribStream analyses.

The only test that checks response values end to end against an independent
source. Needs local URMA_Hist and GFS archives (``PW_URMA_DATA_DIR``) and a
``GRIBSTREAM_API_KEY``; skipped when either is missing.
"""

import datetime
import importlib.util
import itertools
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from API.constants.api_const import CONVERSION_FACTORS
from API.constants.shared_const import KELVIN_TO_CELSIUS
from tests.helpers.urma import _fetch_gribstream_urma, _format_utc, _urma_grid_point

GRIBSTREAM_API_KEY_ENV_VAR = "GRIBSTREAM_API_KEY"
URMA_DATA_DIR_ENV_VAR = "PW_URMA_DATA_DIR"
DEFAULT_URMA_DATA_DIR = Path("/mnt/nvme/data/ProdSync")
COMPARISON_LOCATION = (40.0, -105.0)
COMPARISON_HOURS = 24
HOURS_IN_PAST = 72
SECONDS_PER_HOUR = 3600
METERS_PER_KILOMETER = 1000


def _fetch_local_day(
    monkeypatch, data_dir: Path, lat: float, lon: float, timestamp: int
) -> dict:
    """Request a full Time Machine day with real local ZIP stores loaded by the app."""
    monkeypatch.setenv("STAGE", "TM_TESTING")
    monkeypatch.setenv("save_type", "Download")
    monkeypatch.setenv("save_dir", str(data_dir))
    monkeypatch.setenv("s3_bucket", str(data_dir) + os.sep)
    monkeypatch.setenv("SKIP_ERA5", "1")
    monkeypatch.setenv("use_etopo", "False")

    api_path = Path(__file__).resolve().parents[2] / "API"
    monkeypatch.syspath_prepend(str(api_path))
    spec = importlib.util.spec_from_file_location(
        "urma_comparison_response", api_path / "responseLocal.py"
    )
    assert spec is not None and spec.loader is not None
    response_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(response_module)
    assert response_module.URMA_Zarr is not None
    assert response_module.GFS_Zarr is not None

    with TestClient(response_module.app) as client:
        response = client.get(
            f"/timemachine/local-comparison/{lat},{lon},{timestamp}",
            params={
                "units": "si",
                "version": "2",
                "tmextra": "1",
                "extraVars": "stationPressure",
                "hours": COMPARISON_HOURS,
            },
        )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.skipif(
    not os.getenv(GRIBSTREAM_API_KEY_ENV_VAR),
    reason=f"set {GRIBSTREAM_API_KEY_ENV_VAR} to run the GribStream comparison",
)
def test_local_urma_day_matches_gribstream(monkeypatch):
    """A URMA-backed Time Machine day must reproduce the URMA analyses it came from.

    Protects the whole URMA history path: grid lookup, local-day windowing, unit
    conversion (K, m, Pa), and the station-pressure/sea-level-pressure distinction.
    """
    data_dir = Path(os.getenv(URMA_DATA_DIR_ENV_VAR, str(DEFAULT_URMA_DATA_DIR)))
    if not all((data_dir / name).exists() for name in ("URMA_Hist.zarr", "GFS.zarr")):
        pytest.skip(f"Local URMA and GFS stores are missing under {data_dir}")

    grid_lat, grid_lon, _, _ = _urma_grid_point(*COMPARISON_LOCATION)
    request_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
        hours=HOURS_IN_PAST
    )
    local_data = _fetch_local_day(
        monkeypatch, data_dir, grid_lat, grid_lon, int(request_time.timestamp())
    )

    assert "urma" in local_data["flags"]["sources"]
    hourly = local_data["hourly"]["data"]
    assert len(hourly) == COMPARISON_HOURS
    hours = [item["time"] for item in hourly]
    assert len(set(hours)) == COMPARISON_HOURS
    local_day = request_time.astimezone(ZoneInfo("America/Denver")).date()
    assert (
        datetime.datetime.fromtimestamp(hours[0], ZoneInfo("America/Denver")).date()
        == local_day
    )
    assert all(
        second - first == SECONDS_PER_HOUR
        for first, second in itertools.pairwise(hours)
    )

    try:
        rows = _fetch_gribstream_urma(
            os.environ[GRIBSTREAM_API_KEY_ENV_VAR],
            [datetime.datetime.fromtimestamp(hour, datetime.UTC) for hour in hours],
            [{"name": "denver", "lat": grid_lat, "lon": grid_lon}],
        )
    except HTTPError as exc:
        if exc.code in {429, 503, 504}:
            pytest.skip(f"Gribstream temporarily unavailable: HTTP {exc.code}")
        pytest.fail(f"Gribstream request failed: HTTP {exc.code}")
    except URLError as exc:
        pytest.skip(f"Could not reach Gribstream: {exc.reason}")

    assert len(rows) == COMPARISON_HOURS
    rows_by_time = {row["forecasted_at"]: row for row in rows}
    assert len(rows_by_time) == COMPARISON_HOURS
    expected_times = {
        _format_utc(datetime.datetime.fromtimestamp(hour, datetime.UTC))
        for hour in hours
    }
    assert rows_by_time.keys() == expected_times

    tolerances = {
        "temperature": 0.15,
        "dewPoint": 0.15,
        "windGust": 0.11,
        "visibility": 0.11,
        "stationPressure": 0.11,
    }
    for hour in hourly:
        timestamp = _format_utc(
            datetime.datetime.fromtimestamp(hour["time"], datetime.UTC)
        )
        gribstream = rows_by_time[timestamp]
        assert gribstream["name"] == "denver"
        assert gribstream["forecasted_time"] == timestamp
        expected = {
            "temperature": gribstream["temperature"] - KELVIN_TO_CELSIUS,
            "dewPoint": gribstream["dew_point"] - KELVIN_TO_CELSIUS,
            "windGust": gribstream["wind_gust"],
            "visibility": gribstream["visibility"] / METERS_PER_KILOMETER,
            "stationPressure": gribstream["pressure"]
            / CONVERSION_FACTORS["pressure_to_hpa"],
        }
        for field, expected_value in expected.items():
            actual = hour[field]
            assert isinstance(actual, (int, float)) and math.isfinite(actual)
            assert actual == pytest.approx(expected_value, abs=tolerances[field]), (
                f"hourly.{field} at {timestamp}: local={actual}, Gribstream={expected_value}"
            )
