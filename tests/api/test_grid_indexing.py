"""Point lookup on model grids that are not a simple regular lat/lon index."""

import datetime

import numpy as np
import pytest
import xarray as xr

from API.constants.model_const import ERA5
from API.request.grid_indexing import (
    HRDPS_ROTATED_GRID,
    REPS_ROTATED_GRID,
    _load_era5_slice,
    _nearest_2d_grid_coords,
    _nearest_rotated_grid_coords,
)

ERA5_BASE_DAY = datetime.datetime(2024, 1, 1)


@pytest.mark.parametrize(
    ("grid", "latitude", "longitude", "max_distance"),
    [
        pytest.param(HRDPS_ROTATED_GRID, 39.626032, -133.629520, 0.005, id="hrdps"),
        pytest.param(REPS_ROTATED_GRID, 2.778652, -118.381280, 0.020, id="reps"),
    ],
)
def test_rotated_cmc_grid_first_grib_point_maps_to_index_zero(
    grid, latitude, longitude, max_distance
):
    """The rotated-pole grid definitions must reproduce the GRIB file's first
    point at (0, 0); an off-by-one or wrong pole rotation would read every
    HRDPS/REPS forecast from the wrong cell.
    """
    x_index, y_index, grid_latitude, grid_longitude = _nearest_rotated_grid_coords(
        latitude, longitude, grid, max_distance=max_distance, model_name="CMC"
    )

    assert (x_index, y_index) == (0, 0)
    assert grid_latitude == pytest.approx(latitude, abs=1e-5)
    assert grid_longitude == pytest.approx(longitude, abs=1e-5)


def test_rotated_cmc_grid_rejects_location_outside_domain():
    """A regional grid must refuse points outside its domain instead of silently
    returning the nearest edge cell (which would serve far-away weather).
    """
    with pytest.raises(ValueError, match="outside the HRDPS domain"):
        _nearest_rotated_grid_coords(
            0.0, 0.0, HRDPS_ROTATED_GRID, max_distance=0.005, model_name="HRDPS"
        )


@pytest.mark.parametrize(
    ("query", "latitudes", "longitudes", "expected"),
    [
        # -179.9 is 0.3 deg from 179.8 across the antimeridian, 9.9 deg from -170.
        pytest.param(
            (0.0, -179.9),
            [0.0, 0.0],
            [179.8, -170.0],
            (0.0, 179.8),
            id="antimeridian",
        ),
        # At 80N, 0.5 deg of longitude (~10 km) is nearer than 0.3 deg of
        # latitude (~33 km); a raw lat/lon distance would pick the wrong cell.
        pytest.param(
            (80.0, 0.0), [80.0, 79.7], [0.5, 0.0], (80.0, 0.5), id="high-latitude"
        ),
    ],
)
def test_raqdps_nearest_grid_uses_spherical_distance(
    query, latitudes, longitudes, expected
):
    """RAQDPS is a 2D curvilinear grid; nearest-cell selection must use
    great-circle distance so lookups near the antimeridian and poles are right.
    """
    lat_lon_grid = {
        "latitude": np.array([latitudes]),
        "longitude": np.array([longitudes]),
    }

    x_idx, y_idx, grid_lat, grid_lon = _nearest_2d_grid_coords(
        *query, lat_lon_grid, max_distance=0.020, model_name="RAQDPS"
    )

    assert (x_idx, y_idx) == (0, 0)
    assert (grid_lat, grid_lon) == expected


def _era5_data(hours, latitudes, longitudes, fill):
    """Build the ``era5_data`` dict that ``init_ERA5`` returns, for a tiny grid.

    Each ERA5 variable is filled with its column index unless ``fill`` overrides it.
    """
    times = np.datetime64("2024-01-01T00:00:00") + np.arange(hours) * np.timedelta64(
        1, "h"
    )
    shape = (hours, len(latitudes), len(longitudes))
    data_vars = {}
    for name, index in ERA5.items():
        if name == "prob":  # derived by _load_era5_slice, not stored in ERA5
            continue
        values = fill[name] if name in fill else np.full(shape, float(index))
        data_vars[name] = (("time", "latitude", "longitude"), values)
    latitudes, longitudes = np.array(latitudes), np.array(longitudes)
    return {
        "ERA5_lats": latitudes,
        "ERA5_lons": longitudes,
        "ERA5_times": times,
        "dsERA5": xr.Dataset(
            data_vars=data_vars,
            coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        ),
    }


def test_era5_slice_length_timestamps_and_ptype_rounding():
    """The ERA5 fallback must return exactly num_hours rows stamped with unix
    seconds, and round interpolated precip type so it maps to a category.
    """
    era5_data = _era5_data(
        60, [40.0], [286.0], {"precipitation_type": np.full((60, 1, 1), 4.6)}
    )

    result = _load_era5_slice(
        era5_data, lat=40.0, lon=286.0, base_day_utc=ERA5_BASE_DAY, num_hours=49
    )

    start = int(ERA5_BASE_DAY.replace(tzinfo=datetime.UTC).timestamp())
    assert result.shape[0] == 49
    np.testing.assert_array_equal(result[:, 0], start + 3600 * np.arange(49))
    np.testing.assert_array_equal(result[:, ERA5["precipitation_type"]], 5)


def test_era5_precip_probability_from_neighbourhood_wraps_longitude():
    """ERA5 has no precip probability, so it is estimated from the 3x3
    neighbourhood; at lon 0 the window must wrap to lon 270 (not 180), and
    missing cells must be left out of the denominator.
    """
    lats, lons = [40.0, 39.75], [0.0, 90.0, 180.0, 270.0]
    precip = np.zeros((3, 2, 4))
    neighbourhood = [3, 0, 1]  # lon 270, 0, 90
    precip[0][:, neighbourhood] = 0.0002  # every cell wet -> 100%
    precip[1, 0, neighbourhood] = 0.0002  # 3 wet of 5 valid -> 60%
    precip[1, 1, 3] = np.nan
    precip[2][:, neighbourhood] = np.nan  # no valid cells -> 0%, not NaN
    era5_data = _era5_data(3, lats, lons, {"total_precipitation": precip})

    result = _load_era5_slice(
        era5_data, lat=40.0, lon=0.0, base_day_utc=ERA5_BASE_DAY, num_hours=3
    )

    np.testing.assert_allclose(result[:, ERA5["prob"]], [100.0, 60.0, 0.0])
