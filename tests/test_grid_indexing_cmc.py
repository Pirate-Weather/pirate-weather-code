import pytest

from API.request.grid_indexing import (
    HRDPS_ROTATED_GRID,
    REPS_ROTATED_GRID,
    _nearest_rotated_grid_coords,
)


@pytest.mark.parametrize(
    ("grid", "latitude", "longitude", "max_distance"),
    [
        (HRDPS_ROTATED_GRID, 39.626032, -133.629520, 0.005),
        (REPS_ROTATED_GRID, 2.778652, -118.381280, 0.020),
    ],
)
def test_rotated_cmc_grid_lookup_matches_grib_first_point(
    grid, latitude, longitude, max_distance
):
    x_index, y_index, grid_latitude, grid_longitude = _nearest_rotated_grid_coords(
        latitude,
        longitude,
        grid,
        max_distance=max_distance,
        model_name="CMC",
    )

    assert (x_index, y_index) == (0, 0)
    assert grid_latitude == pytest.approx(latitude, abs=1e-5)
    assert grid_longitude == pytest.approx(longitude, abs=1e-5)


def test_rotated_cmc_grid_lookup_rejects_location_outside_domain():
    with pytest.raises(ValueError, match="outside the HRDPS domain"):
        _nearest_rotated_grid_coords(
            0.0,
            0.0,
            HRDPS_ROTATED_GRID,
            max_distance=0.005,
            model_name="HRDPS",
        )
