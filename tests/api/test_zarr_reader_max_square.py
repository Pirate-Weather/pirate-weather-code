import asyncio

import numpy as np

from API.io.zarr_reader import WeatherParallel


def test_zarr_read_max_square_clips_to_grid_edges():
    """Air quality reads (#682) take the maximum over a 3x3 neighbourhood. At a
    grid corner the window must shrink to the cells that exist, not wrap to the
    opposite edge or fail, and the result must be (time, variable) like
    zarr_read."""
    # Store layout is (variable, time, y, x). Everything outside the 2x2 corner
    # window (rows/cols 2-3, including the opposite edge) is 99.
    store = np.full((2, 2, 4, 4), 99.0)
    store[:, :, :2, :2] = 0.0
    store[0, 0, 1, 1] = 5.0
    store[0, 1, 0, 1] = 7.0
    store[1, 0, 1, 0] = 11.0
    store[1, 1, 0, 0] = 13.0

    data_out = asyncio.run(WeatherParallel().zarr_read_max_square("TEST", store, 0, 0))

    np.testing.assert_array_equal(data_out, [[5.0, 11.0], [7.0, 13.0]])
