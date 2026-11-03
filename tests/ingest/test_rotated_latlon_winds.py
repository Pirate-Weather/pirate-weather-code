"""Wind rotation on the RRFS North American rotated latitude-longitude grid."""

import numpy as np

from API.ingest_utils import rotated_latlon_grid_geometry
from ingest.RRFS_NA_Local_Ingest import (
    RRFS_NA_FIRST_RLAT,
    RRFS_NA_FIRST_RLON,
    RRFS_NA_GRID_SHAPE,
    RRFS_NA_GRID_STEP_DEG,
    RRFS_NA_NORTH_POLE_LAT,
    RRFS_NA_NORTH_POLE_LON,
)


def test_rrfs_na_wind_rotation_matches_closed_form_angle():
    """wgrib2 cannot rotate winds on the NA grid, so the ingest rotates them
    itself from a computed angle; a sign or convention slip there would publish
    every NA wind in the wrong direction while all speeds still look right.
    Check the angle against the closed-form rotated-pole formula (COSMO
    documentation) across the grid, and that the rotated origin lands where the
    GRIB pole puts it (a wrong pole moves the grid and every angle). Against
    wgrib2's own earth-relative interpolation of a real NA file (2026-11-03 12z
    f018) the rotated winds agreed to 0.02 m/s at the 99th percentile, versus
    14 m/s for the unrotated ones.
    """
    rows = np.arange(0, RRFS_NA_GRID_SHAPE[0], 25)
    cols = np.arange(0, RRFS_NA_GRID_SHAPE[1], 25)
    rlat = RRFS_NA_FIRST_RLAT + RRFS_NA_GRID_STEP_DEG * rows
    rlon = RRFS_NA_FIRST_RLON + RRFS_NA_GRID_STEP_DEG * cols

    lat, lon, angle = rotated_latlon_grid_geometry(
        RRFS_NA_NORTH_POLE_LAT, RRFS_NA_NORTH_POLE_LON, rlat, rlon
    )

    # Angle from geographic north to the rotated grid's north, which equals the
    # angle from geographic east to the grid's east
    phi = np.radians(lat)
    lam = np.radians(lon)
    pole_phi = np.radians(RRFS_NA_NORTH_POLE_LAT)
    pole_lam = np.radians(RRFS_NA_NORTH_POLE_LON)
    expected = np.arctan2(
        np.cos(pole_phi) * np.sin(lam - pole_lam),
        np.cos(phi) * np.sin(pole_phi)
        - np.sin(phi) * np.cos(pole_phi) * np.cos(lam - pole_lam),
    )
    angle_error = (angle - expected + np.pi) % (2 * np.pi) - np.pi
    assert np.degrees(np.abs(angle_error)).max() < 1e-3

    # The rotated origin sits 90 deg from the rotated pole, over western Canada
    centre = rotated_latlon_grid_geometry(
        RRFS_NA_NORTH_POLE_LAT, RRFS_NA_NORTH_POLE_LON, np.zeros(1), np.zeros(1)
    )
    assert np.allclose([centre[0][0, 0], centre[1][0, 0]], [55.0, -113.0])


def test_api_rrfs_na_lookup_finds_the_cell_the_ingest_wrote():
    """The ingest places the NA grid with cartopy (checked against wgrib2's
    coordinates on every run), while the API finds a request's cell with its
    own closed-form rotated-pole lookup. A pole or longitude convention that
    differed between the two would read a neighbouring or distant cell for
    every North American RRFS request, with nothing failing. Round-trip
    sampled cell centres through the API lookup.
    """
    from API.request.grid_indexing import (
        RRFS_NA_ROTATED_GRID,
        _nearest_rotated_grid_coords,
    )

    rows = np.arange(0, RRFS_NA_GRID_SHAPE[0], 37)
    cols = np.arange(0, RRFS_NA_GRID_SHAPE[1], 41)
    rlat = RRFS_NA_FIRST_RLAT + RRFS_NA_GRID_STEP_DEG * rows
    rlon = RRFS_NA_FIRST_RLON + RRFS_NA_GRID_STEP_DEG * cols
    lat, lon, _ = rotated_latlon_grid_geometry(
        RRFS_NA_NORTH_POLE_LAT, RRFS_NA_NORTH_POLE_LON, rlat, rlon
    )

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            x, y, _, _ = _nearest_rotated_grid_coords(
                float(lat[i, j]),
                float(lon[i, j]),
                RRFS_NA_ROTATED_GRID,
                max_distance=0.005,
                model_name="RRFS_NA",
            )
            assert (y, x) == (row, col)
