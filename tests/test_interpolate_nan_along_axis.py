import dask.array as da
import numpy as np

from API.ingest_utils import interpolate_nan_along_axis


def test_interpolates_internal_gap_one_dim():
    arr = da.from_array(np.array([1.0, np.nan, 3.0], dtype=float), chunks=-1)

    result = interpolate_nan_along_axis(arr, axis=0).compute()

    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0], dtype=float))


def test_preserves_all_nan_slice():
    arr = da.from_array(np.array([np.nan, np.nan, np.nan], dtype=float), chunks=-1)

    result = interpolate_nan_along_axis(arr, axis=0).compute()

    assert np.isnan(result).all()


def test_preserves_edges_and_fills_between_points():
    arr = da.from_array(
        np.array([np.nan, 1.0, np.nan, np.nan, 3.0, np.nan], dtype=float),
        chunks=-1,
    )

    result = interpolate_nan_along_axis(arr, axis=0).compute()

    expected = np.array([1.0, 1.0, 1.6666667, 2.3333333, 3.0, 3.0])
    np.testing.assert_allclose(result, expected)


def test_multidimensional_interpolation_per_slice():
    data = np.array(
        [
            [1.0, np.nan],
            [np.nan, np.nan],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    arr = da.from_array(data, chunks=(2, 1))

    result = interpolate_nan_along_axis(arr, axis=0).compute()

    expected = np.array(
        [
            [1.0, 3.0],
            [2.0, 3.0],
            [3.0, 3.0],
        ]
    )
    np.testing.assert_allclose(result, expected)