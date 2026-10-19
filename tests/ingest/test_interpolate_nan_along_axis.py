import dask.array as da
import numpy as np
import xarray as xr

from API.ingest_utils import interpolate_temporal_gaps_efficiently


def _make_dataset_1d(values: list[float]) -> xr.Dataset:
    """Create an xr.Dataset with dims (time, y, x) where y and x are size 1.

    Args:
        values (list[float]): A list of float values for the data variable.

    Returns:
        xr.Dataset: A dataset containing one variable 'var' with the provided values.
    """
    arr = np.array(values, dtype=float).reshape((len(values), 1, 1))
    darr = da.from_array(arr, chunks=arr.shape)
    da_var = xr.DataArray(
        darr, dims=("time", "y", "x"), coords={"time": np.arange(len(values))}
    )
    return xr.Dataset({"var": da_var})


def test_interpolates_internal_gap_one_dim():
    ds = _make_dataset_1d([1.0, np.nan, 3.0])

    result = interpolate_temporal_gaps_efficiently(ds, max_gap_hours=3, time_dim="time")

    out = result["var"].data.compute().reshape(-1)

    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=float))


def test_preserves_all_nan_slice():
    ds = _make_dataset_1d([np.nan, np.nan, np.nan])

    result = interpolate_temporal_gaps_efficiently(ds, max_gap_hours=3, time_dim="time")

    out = result["var"].data.compute().reshape(-1)

    assert np.isnan(out).all()


def test_preserves_edges_and_fills_between_points():
    ds = _make_dataset_1d([np.nan, 1.0, np.nan, np.nan, 3.0, np.nan])

    result = interpolate_temporal_gaps_efficiently(ds, max_gap_hours=3, time_dim="time")

    out = result["var"].data.compute().reshape(-1)

    expected = np.array([1.0, 1.0, 1.6666667, 2.3333333, 3.0, 3.0])
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_multidimensional_interpolation_per_slice():
    # time x y data (x dimension is size 1)
    data = np.array(
        [
            [1.0, np.nan],
            [np.nan, np.nan],
            [3.0, 3.0],
        ],
        dtype=float,
    )
    # reshape to (time, y, x)
    arr = data.reshape((3, 2, 1))
    darr = da.from_array(arr, chunks=arr.shape)
    da_var = xr.DataArray(darr, dims=("time", "y", "x"), coords={"time": np.arange(3)})
    ds = xr.Dataset({"var": da_var})

    result = interpolate_temporal_gaps_efficiently(ds, max_gap_hours=3, time_dim="time")

    out = result["var"].data.compute().reshape((3, 2))

    expected = np.array(
        [
            [1.0, 3.0],
            [2.0, 3.0],
            [3.0, 3.0],
        ]
    )
    np.testing.assert_allclose(out, expected)
