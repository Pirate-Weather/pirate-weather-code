import os
import sys
import unittest

import dask.array as da
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import xarray as xr

from API.ingest_utils import interpolate_temporal_gaps_efficiently


class TestDWDInterpFlow(unittest.TestCase):
    def test_interp_time_take_blend_logic(self):
        """Verify that temporal interpolation converts irregular time steps
        to a regular hourly grid using the new xarray-based utility.
        """
        # 1. Setup Source Data (Irregular)
        # T=0 (0.0), T=3 (10.0), T=6 (20.0) -> Values
        # Times: 0, 3600*3, 3600*6
        source_times = pd.to_datetime(
            ["2025-01-01 00:00", "2025-01-01 03:00", "2025-01-01 06:00"]
        )
        # (unix conversions not needed for new xarray-based interpolator)

        # Array shape: (Var=1, Time=3, Y=1, X=1)
        # Values: 0, 30, 60
        data_np = np.array([[[[0.0]], [[30.0]], [[60.0]]]], dtype=np.float32)
        data_dask = da.from_array(data_np, chunks=(1, -1, 1, 1))

        # 2. Setup Target Grid (Hourly)
        # 00:00, 01:00, 02:00, 03:00, 04:00, 05:00, 06:00
        # target hourly times are implied by the interpolator behavior

        # 3. Convert to an xarray Dataset and call the new interpolator
        # data_dask has shape (var=1, time=3, y=1, x=1) -> extract the single
        # variable slice as a dask array with dims (time, y, x)
        var_dask = data_dask[0]

        da_xr = xr.DataArray(
            var_dask,
            dims=("time", "y", "x"),
            coords={"time": source_times, "y": [0], "x": [0]},
            name="var",
        )

        ds = xr.Dataset({"var": da_xr})

        # Reindex to the target hourly grid (creates NaNs at missing hours)
        target_times = pd.date_range("2025-01-01 00:00", "2025-01-01 06:00", freq="1h")
        ds = ds.reindex({"time": target_times})

        # Ensure time is a single chunk in the dask-backed DataArray
        ds = ds.chunk({"time": -1})

        # Interpolate internal gaps and extrapolate edges
        result_ds = interpolate_temporal_gaps_efficiently(ds)

        # Extract the computed numpy result for the variable and reshape to
        # match the original test's expected shape: (var, time, y, x)
        result = result_ds["var"].data.compute()  # shape (7,1,1)
        result = result.reshape(1, result.shape[0], 1, 1)

        # Expected hourly values from 00:00 to 06:00
        expected = np.array([0, 10, 20, 30, 40, 50, 60], dtype=np.float32).reshape(
            1, 7, 1, 1
        )

        np.testing.assert_allclose(result, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
