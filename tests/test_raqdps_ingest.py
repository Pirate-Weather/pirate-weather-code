"""Tests for RAQDPS ingest helpers and Herbie template wiring."""

import ast
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from API.raqdps_herbie_template import raqdps
from API.raqdps_utils import (
    KG_M3_TO_UG_M3,
    MOL_MOL_TO_PPB,
    build_raqdps_filename,
    build_raqdps_url,
    candidate_raqdps_runs,
    convert_to_output_units,
    herbie_naive_utc,
    history_run_for_valid_time,
    history_valid_times,
    normalize_utc,
    output_units_for_variable,
)
from API.request.grid_indexing import _nearest_raqdps_grid_coords

REPO_ROOT = Path(__file__).resolve().parents[1]
RAQDPS_SCRIPT = REPO_ROOT / "API" / "RAQDPS_Local_Ingest.py"


def test_raqdps_script_exists_and_is_valid_python():
    """The RAQDPS ingest script should exist and be syntactically valid."""
    assert RAQDPS_SCRIPT.exists()
    tree = ast.parse(RAQDPS_SCRIPT.read_text(encoding="utf-8"))
    assert tree is not None


def test_build_raqdps_filename():
    """RAQDPS filenames should match ECCC Datamart nomenclature."""
    run_time = datetime(2026, 7, 6, 0, tzinfo=timezone.utc)
    assert (
        build_raqdps_filename(run_time, "NO2", 3)
        == "20260706T00Z_MSC_RAQDPS_NO2_Sfc_RLatLon0.09_PT003H.grib2"
    )


def test_build_raqdps_url():
    """RAQDPS URLs should include date, run hour, forecast hour, and filename."""
    run_time = datetime(2026, 7, 6, 12, tzinfo=timezone.utc)
    url = build_raqdps_url(run_time, "PM2.5", 72)
    assert url == (
        "https://dd.weather.gc.ca/20260706/WXO-DD/model_raqdps/10km/grib2/"
        "12/072/20260706T12Z_MSC_RAQDPS_PM2.5_Sfc_RLatLon0.09_PT072H.grib2"
    )


def test_candidate_raqdps_runs_are_recent_cycles():
    """Candidate run probing should walk backward through 00/12 UTC cycles."""
    now = datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc)
    assert candidate_raqdps_runs(now, count=4) == [
        datetime(2026, 7, 6, 12, tzinfo=timezone.utc),
        datetime(2026, 7, 6, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 5, 12, tzinfo=timezone.utc),
        datetime(2026, 7, 5, 0, tzinfo=timezone.utc),
    ]


def test_raqdps_time_helpers_normalize_aware_and_naive_utc():
    """Herbie gets naive UTC while ingest logic keeps aware UTC datetimes."""
    aware_time = datetime(2026, 7, 6, 8, 30, tzinfo=timezone.utc)
    naive_time = datetime(2026, 7, 6, 8, 30)

    assert normalize_utc(naive_time) == datetime(2026, 7, 6, 8, tzinfo=timezone.utc)
    assert herbie_naive_utc(aware_time) == datetime(2026, 7, 6, 8)
    assert herbie_naive_utc(naive_time) == datetime(2026, 7, 6, 8)


def test_history_run_for_valid_time_prefers_smallest_0_to_12h_lead():
    """Historic valid times should map to the current 00/12 cycle."""
    run_time, forecast_hour = history_run_for_valid_time(
        datetime(2026, 7, 6, 11, tzinfo=timezone.utc)
    )
    assert run_time == datetime(2026, 7, 6, 0, tzinfo=timezone.utc)
    assert forecast_hour == 11

    run_time, forecast_hour = history_run_for_valid_time(
        datetime(2026, 7, 6, 18, tzinfo=timezone.utc)
    )
    assert run_time == datetime(2026, 7, 6, 12, tzinfo=timezone.utc)
    assert forecast_hour == 6


def test_history_valid_times_exclude_base_time():
    """History should end one hour before the forecast base time."""
    base_time = datetime(2026, 7, 6, 12, tzinfo=timezone.utc)
    assert history_valid_times(base_time, 3) == [
        datetime(2026, 7, 6, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 6, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 6, 11, tzinfo=timezone.utc),
    ]


def test_raqdps_unit_conversions():
    """RAQDPS native units should keep gases as ppb outputs."""
    values = np.array([1.0], dtype=np.float32)
    np.testing.assert_allclose(convert_to_output_units(values, "PM2.5"), KG_M3_TO_UG_M3)
    np.testing.assert_allclose(convert_to_output_units(values, "PM10"), KG_M3_TO_UG_M3)
    np.testing.assert_allclose(convert_to_output_units(values, "O3"), MOL_MOL_TO_PPB)
    np.testing.assert_allclose(convert_to_output_units(values, "NO2"), MOL_MOL_TO_PPB)
    np.testing.assert_allclose(convert_to_output_units(values, "SO2"), MOL_MOL_TO_PPB)


def test_raqdps_output_unit_labels():
    """RAQDPS output metadata should label gases as ppb."""
    assert output_units_for_variable("PM2.5") == "µg m-3"
    assert output_units_for_variable("PM10") == "µg m-3"
    assert output_units_for_variable("O3") == "ppb"
    assert output_units_for_variable("NO2") == "ppb"
    assert output_units_for_variable("SO2") == "ppb"


def test_herbie_template_sets_raqdps_source():
    """The repo-local Herbie template should set an MSC RAQDPS source URL."""

    class DummyHerbie:
        date = datetime(2026, 7, 6, 0, tzinfo=timezone.utc)
        product = None
        variable = "O3"
        level = "Sfc"
        fxx = 12

        @property
        def get_remoteFileName(self):
            return self.SOURCES["msc"].split("/")[-1]

    dummy = DummyHerbie()
    raqdps.template(dummy)

    assert dummy.product == "10km/grib2"
    assert dummy.SOURCES["msc"].endswith(
        "20260706T00Z_MSC_RAQDPS_O3_Sfc_RLatLon0.09_PT012H.grib2"
    )
    assert dummy.LOCALFILE == dummy.SOURCES["msc"].split("/")[-1]


def test_raqdps_nearest_grid_wraps_antimeridian_and_caches_tree():
    """RAQDPS lookup should use spherical distance and cache the KD-tree."""
    lat_lon_grid = {
        "latitude": np.array([[0.0, 0.0]]),
        "longitude": np.array([[179.8, -170.0]]),
    }

    x_idx, y_idx, grid_lat, grid_lon = _nearest_raqdps_grid_coords(
        0.0,
        -179.9,
        lat_lon_grid,
    )

    assert (x_idx, y_idx) == (0, 0)
    assert grid_lat == 0.0
    assert grid_lon == 179.8
    cache = lat_lon_grid["_lookup_cache"]

    _nearest_raqdps_grid_coords(0.0, -179.9, lat_lon_grid)

    assert lat_lon_grid["_lookup_cache"] is cache


def test_raqdps_nearest_grid_uses_spherical_distance_at_high_latitudes():
    """Longitude degrees should shrink with latitude when selecting RAQDPS cells.

    A naive Euclidean distance using raw lat/lon coordinates would rank
    (79.7°N, 0°E) as closer to the query (0.3 degrees away in latitude) than
    (80°N, 0.5°E) (0.5 degrees away in longitude).  The spherical KD-tree
    correctly identifies (80°N, 0.5°E) as the nearer point because at 80°N a
    half-degree of longitude is only ~10 km while 0.3° of latitude is ~33 km.
    """
    lat_lon_grid = {
        "latitude": np.array([[80.0, 79.7]]),
        "longitude": np.array([[0.5, 0.0]]),
    }

    x_idx, y_idx, grid_lat, grid_lon = _nearest_raqdps_grid_coords(
        80.0,
        0.0,
        lat_lon_grid,
    )

    assert (x_idx, y_idx) == (0, 0)
    assert grid_lat == 80.0
    assert grid_lon == 0.5


def test_raqdps_outside_domain_raises():
    """A query far from all grid points should raise ValueError."""
    import pytest

    lat_lon_grid = {
        "latitude": np.array([[55.0, 56.0]]),
        "longitude": np.array([[-100.0, -100.0]]),
    }
    # Query in the tropics — far outside the small two-cell test domain
    with pytest.raises(ValueError, match="outside the RAQDPS domain"):
        _nearest_raqdps_grid_coords(0.0, 0.0, lat_lon_grid)
