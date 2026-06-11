"""Storm proximity and direction processing utilities."""

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


def _haversine_distance_m(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in meters."""
    earth_radius_m = 6_371_000.0

    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_m * c


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Compute forward azimuth in degrees [0, 360)."""
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    dlon_rad = np.deg2rad(((lon2 - lon1 + 180) % 360) - 180)

    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - (
        np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
    )

    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _storm_fields_for_slice(apcp_slice, lat_values, lon_values, threshold, max_distance_m):
    """Compute nearest-storm distance and direction for one 2D precipitation slice."""
    storm_mask = np.isfinite(apcp_slice) & (apcp_slice > threshold)

    if not storm_mask.any():
        distance_fill = np.nan if max_distance_m is None else max_distance_m
        distance_out = np.full(apcp_slice.shape, distance_fill, dtype=np.float32)
        direction_out = np.full(apcp_slice.shape, np.nan, dtype=np.float32)
        return distance_out, direction_out

    _, nearest_indices = distance_transform_edt(~storm_mask, return_indices=True)
    nearest_lat = lat_values[nearest_indices[0]]
    nearest_lon = lon_values[nearest_indices[1]]

    lat_grid = lat_values[:, np.newaxis]
    lon_grid = lon_values[np.newaxis, :]

    distance_out = _haversine_distance_m(
        lat_grid,
        lon_grid,
        nearest_lat,
        nearest_lon,
    ).astype(np.float32)
    direction_out = _bearing_deg(
        lat_grid,
        lon_grid,
        nearest_lat,
        nearest_lon,
    ).astype(np.float32)

    # Grid cells containing precipitation sources have zero direction by definition.
    direction_out[storm_mask] = 0.0

    if max_distance_m is not None:
        too_far = distance_out > max_distance_m
        distance_out[too_far] = max_distance_m
        direction_out[too_far] = np.nan

    return distance_out, direction_out


def compute_storm_distance_direction(
    apcp_data,
    lat_values,
    lon_values,
    threshold,
    max_distance_m,
):
    """Compute storm distance and direction for all timesteps using dask blocks."""

    def _storm_fields_block(block, lat_values, lon_values, threshold, max_distance_m):
        distances = np.empty(block.shape, dtype=np.float32)
        directions = np.empty(block.shape, dtype=np.float32)

        for time_idx in range(block.shape[0]):
            distance_out, direction_out = _storm_fields_for_slice(
                block[time_idx],
                lat_values=lat_values,
                lon_values=lon_values,
                threshold=threshold,
                max_distance_m=max_distance_m,
            )
            distances[time_idx] = distance_out
            directions[time_idx] = direction_out

        return np.stack((distances, directions), axis=0)

    return da.map_blocks(
        _storm_fields_block,
        apcp_data,
        lat_values=np.asarray(lat_values),
        lon_values=np.asarray(lon_values),
        threshold=threshold,
        max_distance_m=max_distance_m,
        dtype=np.float32,
        chunks=((2,),) + apcp_data.chunks,
        new_axis=0,
    )


def compute_storm_fields_from_apcp_dataarray(
    apcp_dataarray,
    threshold,
    max_distance_m,
):
    """Compute storm distance/direction for an APCP DataArray and restore longitude order."""

    longitude_original = apcp_dataarray.longitude.data

    longitude_normalized = ((longitude_original + 180) % 360) - 180

    # Position indices that sort normalized longitudes from -180 to 180.
    longitude_sort_index = np.argsort(longitude_normalized)

    # Position indices that restore the original longitude order.
    longitude_restore_index = np.argsort(longitude_sort_index)

    apcp_normalized = (
        apcp_dataarray
        .assign_coords(
            longitude=("longitude", longitude_normalized)
        )
        .isel(longitude=longitude_sort_index)
    )

    storm_fields = compute_storm_distance_direction(
        apcp_data=apcp_normalized.data,
        lat_values=apcp_normalized.latitude.data,
        lon_values=apcp_normalized.longitude.data,
        threshold=threshold,
        max_distance_m=max_distance_m,
    )

    distance_sorted = xr.DataArray(
        storm_fields[0],
        coords=apcp_normalized.coords,
        dims=apcp_normalized.dims,
    )

    direction_sorted = xr.DataArray(
        storm_fields[1],
        coords=apcp_normalized.coords,
        dims=apcp_normalized.dims,
    )

    distance_original = (
        distance_sorted
        .isel(longitude=longitude_restore_index)
        .assign_coords(longitude=("longitude", longitude_original))
    )

    direction_original = (
        direction_sorted
        .isel(longitude=longitude_restore_index)
        .assign_coords(longitude=("longitude", longitude_original))
    )

    return distance_original.data, direction_original.data
