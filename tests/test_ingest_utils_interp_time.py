import os
import sys
import unittest

import dask.array as da
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from API.ingest_utils import interp_time_map_blocks_nan


class TestInterpTimeMapBlocksNan(unittest.TestCase):
    def test_interpolates_nan_gaps_in_column_batches(self):
        source_times = np.array([0, 3600, 7200, 10800], dtype=np.float64)
        target_times = np.array(
            [0, 1800, 3600, 5400, 7200, 9000, 10800],
            dtype=np.float64,
        )
        source = np.array(
            [
                [
                    [[0.0, 0.0, np.nan, np.nan, 5.0]],
                    [[10.0, np.nan, 10.0, np.nan, np.nan]],
                    [[20.0, 20.0, 20.0, np.nan, np.nan]],
                    [[30.0, 30.0, np.nan, np.nan, np.nan]],
                ],
                [
                    [[0.0, np.nan, 1.0, 1.0, 1.0]],
                    [[10.0, 100.0, 2.0, 2.0, 2.0]],
                    [[20.0, np.nan, 3.0, 3.0, 3.0]],
                    [[30.0, 300.0, 4.0, 4.0, 4.0]],
                ],
            ],
            dtype=np.float32,
        )
        source_dask = da.from_array(source, chunks=(1, -1, 1, 5))

        result = interp_time_map_blocks_nan(
            source_dask,
            stacked_timesUnix=source_times,
            hourly_timesUnix=target_times,
            nearest_vars=[1],
            max_columns_per_batch=2,
        ).compute()

        expected_var_0 = np.array(
            [
                [0.0, 0.0, np.nan, np.nan, 5.0],
                [5.0, 5.0, np.nan, np.nan, np.nan],
                [10.0, 10.0, 10.0, np.nan, np.nan],
                [15.0, 15.0, 15.0, np.nan, np.nan],
                [20.0, 20.0, 20.0, np.nan, np.nan],
                [25.0, 25.0, np.nan, np.nan, np.nan],
                [30.0, 30.0, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )
        expected_var_1 = np.array(
            [
                [0.0, np.nan, 1.0, 1.0, 1.0],
                [0.0, np.nan, 1.0, 1.0, 1.0],
                [10.0, 100.0, 2.0, 2.0, 2.0],
                [10.0, 100.0, 2.0, 2.0, 2.0],
                [20.0, 100.0, 3.0, 3.0, 3.0],
                [20.0, 300.0, 3.0, 3.0, 3.0],
                [30.0, 300.0, 4.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result[0, :, 0, :], expected_var_0)
        np.testing.assert_allclose(result[1, :, 0, :], expected_var_1)


if __name__ == "__main__":
    unittest.main()
