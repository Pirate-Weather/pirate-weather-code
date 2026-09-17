import dask.array as da
import numpy as np
import pytest

from API.ingest_utils import broadcast_times_to_grid


def test_broadcast_times_to_grid_is_lazy_and_chunked():
    times = np.arange(283, dtype=np.float32)

    result = broadcast_times_to_grid(times, ny=1600, nx=2350, spatial_chunk=200)

    assert isinstance(result, da.Array)
    assert result.shape == (283, 1600, 2350)
    assert result.chunksize == (283, 200, 200)
    assert result.npartitions == 96
    np.testing.assert_array_equal(result[:, -1, -1].compute(), times)


@pytest.mark.parametrize(
    ("times", "ny", "nx", "spatial_chunk"),
    [
        (np.array([]), 10, 10, 5),
        (np.array([[1]]), 10, 10, 5),
        (np.array([1]), 0, 10, 5),
        (np.array([1]), 10, 0, 5),
        (np.array([1]), 10, 10, 0),
    ],
)
def test_broadcast_times_to_grid_rejects_invalid_dimensions(
    times, ny, nx, spatial_chunk
):
    with pytest.raises(ValueError):
        broadcast_times_to_grid(times, ny, nx, spatial_chunk)
