import asyncio

import numpy as np

from API.io.zarr_reader import WeatherParallel


def test_zarr_read_max_square_returns_spatial_max_with_zarr_read_shape():
    store = np.zeros((2, 3, 7, 7), dtype=float)
    store[:, :, 3:6, 2:5] = np.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
    store[:, :, 0, 0] = 9999

    weather = WeatherParallel()

    data_out = asyncio.run(weather.zarr_read_max_square("TEST", store, 3, 4))

    expected = np.nanmax(store[:, :, 3:6, 2:5], axis=(-2, -1)).T
    assert np.array_equal(data_out, expected)


def test_zarr_read_max_square_clips_to_grid_edges():
    store = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
    weather = WeatherParallel()

    data_out = asyncio.run(weather.zarr_read_max_square("TEST", store, 0, 0))

    expected = np.nanmax(store[:, :, 0:2, 0:2], axis=(-2, -1)).T
    assert np.array_equal(data_out, expected)
