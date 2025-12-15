import os
import sys
import unittest

import dask.array as da
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from API.ingest_utils import interp_time_take_blend


class TestDWDInterpFlow(unittest.TestCase):
    def test_interp_time_take_blend_logic(self):
        """
        Verify that interp_time_take_blend correctly interpolates irregular time steps
        to a regular hourly grid.
        """
        # 1. Setup Source Data (Irregular)
        # T=0 (0.0), T=3 (10.0), T=6 (20.0) -> Values
        # Times: 0, 3600*3, 3600*6
        unix_epoch = np.datetime64(0, "s")
        source_times = pd.to_datetime(
            ["2025-01-01 00:00", "2025-01-01 03:00", "2025-01-01 06:00"]
        )
        source_times_unix = (source_times.values - unix_epoch) / np.timedelta64(1, "s")

        # Array shape: (Var=1, Time=3, Y=1, X=1)
        # Values: 0, 30, 60
        data_np = np.array([[[[0.0]], [[30.0]], [[60.0]]]], dtype=np.float32)
        data_dask = da.from_array(data_np, chunks=(1, -1, 1, 1))

        # 2. Setup Target Grid (Hourly)
        # 00:00, 01:00, 02:00, 03:00, 04:00, 05:00, 06:00
        target_times = pd.date_range("2025-01-01 00:00", "2025-01-01 06:00", freq="1h")
        target_times_unix = (target_times.values - unix_epoch) / np.timedelta64(1, "s")

        # 3. Interpolate
        result_dask = interp_time_take_blend(
            data_dask,
            stacked_timesUnix=source_times_unix,
            hourly_timesUnix=target_times_unix,
        )

        # 4. Compute and Verify
        result = result_dask.compute()

        # Expected shape: (1, 7, 1, 1)
        self.assertEqual(result.shape, (1, 7, 1, 1))

        # Expected values:
        # 00:00 -> 0.0
        # 01:00 -> 10.0 (1/3 of way to 30)
        # 02:00 -> 20.0 (2/3 of way to 30)
        # 03:00 -> 30.0
        # 04:00 -> 40.0 (1/3 of way to 60)
        # 05:00 -> 50.0
        # 06:00 -> 60.0
        expected = np.array([0, 10, 20, 30, 40, 50, 60], dtype=np.float32).reshape(
            1, 7, 1, 1
        )

        np.testing.assert_allclose(result, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
