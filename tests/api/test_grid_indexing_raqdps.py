"""RAQDPS grid lookup tests for API requests."""

import numpy as np

from API.request.grid_indexing import _nearest_2d_grid_coords


def test_raqdps_nearest_grid_wraps_antimeridian_and_caches_tree():
    """RAQDPS lookup should use spherical distance and cache the KD-tree."""
    lat_lon_grid = {
        "latitude": np.array([[0.0, 0.0]]),
        "longitude": np.array([[179.8, -170.0]]),
    }

    x_idx, y_idx, grid_lat, grid_lon = _nearest_2d_grid_coords(
        0.0, -179.9, lat_lon_grid, max_distance=0.020, model_name="RAQDPS"
    )

    assert (x_idx, y_idx) == (0, 0)
    assert grid_lat == 0.0
    assert grid_lon == 179.8
    cache = lat_lon_grid["_lookup_cache"]

    _nearest_2d_grid_coords(
        0.0, -179.9, lat_lon_grid, max_distance=0.020, model_name="RAQDPS"
    )

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

    x_idx, y_idx, grid_lat, grid_lon = _nearest_2d_grid_coords(
        80.0, 0.0, lat_lon_grid, max_distance=0.020, model_name="RAQDPS"
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
        _nearest_2d_grid_coords(
            0.0, 0.0, lat_lon_grid, max_distance=0.020, model_name="RAQDPS"
        )
