"""Wind rotation on the Lambert conformal grid shared by RTMA, RTMA-RU and URMA."""

import cartopy.crs as ccrs
import metpy  # noqa: F401  (registers the .metpy accessor)
import numpy as np
import xarray as xr

from API.ingest_utils import earth_relative_wind_components

# The 2.5 km NDFD grid: LoV 265 E, tangent at 25 N
NDFD_LOV_DEG = 265.0
NDFD_LATIN_DEG = 25.0
EARTH_RADIUS_M = 6371200.0


def test_lambert_rotation_with_herbie_index_coordinates():
    """Herbie's datasets have x/y indices (0, 1, 2...) rather than projection
    metres. The first version of this rotation read them as metres, so every
    point used the convergence at the grid origin (25 N, 95 W) and published
    RTMA, RTMA-RU and URMA winds were effectively unrotated: directions up to 21
    deg off with correct speeds. Build a dataset shaped like Herbie's and compare
    with the bearing of the grid axes found by stepping along them in projection
    metres. Against wgrib2's own -new_grid_winds earth, on real files (2026-11-04
    RTMA-RU 18z, URMA 12z), the median difference is 0.03 deg.
    """
    globe = ccrs.Globe(
        ellipse="sphere", semimajor_axis=EARTH_RADIUS_M, semiminor_axis=EARTH_RADIUS_M
    )
    grid_crs = ccrs.LambertConformal(
        central_longitude=NDFD_LOV_DEG,
        central_latitude=NDFD_LATIN_DEG,
        standard_parallels=(NDFD_LATIN_DEG, NDFD_LATIN_DEG),
        globe=globe,
    )
    geographic = ccrs.PlateCarree()
    x_m, y_m = np.meshgrid(np.linspace(-2.7e6, 2.9e6, 7), np.linspace(-0.6e6, 3.2e6, 5))
    centre = geographic.transform_points(grid_crs, x_m, y_m)
    lat, lon = centre[..., 1], centre[..., 0]

    grid_u = 7.0 * np.ones_like(x_m)
    grid_v = 4.0 * np.ones_like(x_m)
    ds = xr.Dataset(
        {
            "u10": (("y", "x"), grid_u, {"grid_mapping": "gribfile_projection"}),
            "v10": (("y", "x"), grid_v, {"grid_mapping": "gribfile_projection"}),
        },
        coords={
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon % 360.0),
            "gribfile_projection": (
                (),
                0,
                {
                    "grid_mapping_name": "lambert_conformal_conic",
                    "standard_parallel": (NDFD_LATIN_DEG, NDFD_LATIN_DEG),
                    "longitude_of_central_meridian": NDFD_LOV_DEG,
                    "latitude_of_projection_origin": NDFD_LATIN_DEG,
                    "earth_radius": EARTH_RADIUS_M,
                },
            ),
        },
    )
    ds = ds.assign_coords(ds.metpy.parse_cf().coords)

    u_earth, v_earth = earth_relative_wind_components(ds["u10"], ds["v10"])

    # Move a short way along the wind in projection metres, then measure the
    # east and north distance covered on the sphere
    step = 100.0 / np.hypot(grid_u, grid_v)
    ahead = geographic.transform_points(
        grid_crs, x_m + step * grid_u, y_m + step * grid_v
    )
    east = np.radians(ahead[..., 0] - lon) * np.cos(np.radians(lat))
    north = np.radians(ahead[..., 1] - lat)

    bearing_error = np.degrees(np.arctan2(u_earth, v_earth) - np.arctan2(east, north))
    bearing_error = (bearing_error + 180.0) % 360.0 - 180.0
    assert np.abs(bearing_error).max() < 0.01
    # The grid must span enough convergence for the check to mean anything
    assert np.ptp(lon) > 40.0
    np.testing.assert_allclose(np.hypot(u_earth, v_earth), np.hypot(grid_u, grid_v))
