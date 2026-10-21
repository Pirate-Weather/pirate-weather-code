import dask.array as da
import numpy as np

from API.ingest_utils import fill_isolated_time_gaps, interp_time_take_blend


def test_msc_time_interpolation_leaves_no_gaps():
    """The MSC ingests (GDPS, HRDPS, GEPS, REPS) must publish hourly stores
    without NaN where data exists: a value masked at ingest is bridged from its
    valid neighbours (precipitation type from the nearest step, never blended
    into an in-between category), a value masked at the first lead holds the
    first valid value, and fields on mixed 3- and 6-hourly leads are
    interpolated to every hour. Points with no data at all stay NaN rather than
    being invented.
    """
    hour = 3600.0
    hourly_times = np.arange(1, 7) * hour
    source = np.array(
        [
            # Linear field with a masked value at 3 h (point 0); point 1 has no data
            [
                [[10.0, np.nan]],
                [[20.0, np.nan]],
                [[np.nan, np.nan]],
                [[40.0, np.nan]],
                [[50.0, np.nan]],
                [[60.0, np.nan]],
            ],
            # Precipitation type with two masked steps between categories 1 and 3;
            # point 1 is masked at the first lead
            [
                [[1.0, np.nan]],
                [[1.0, 2.0]],
                [[np.nan, 2.0]],
                [[np.nan, 2.0]],
                [[3.0, 2.0]],
                [[3.0, 2.0]],
            ],
        ],
        dtype=np.float32,
    )  # (V, T, Y, X)
    filled = fill_isolated_time_gaps(
        da.from_array(source, chunks=(1, -1, 1, 2)), hourly_times, nearest_vars=[1]
    ).compute()

    np.testing.assert_allclose(filled[0, :, 0, 0], [10, 20, 30, 40, 50, 60])
    assert np.isnan(filled[0, :, 0, 1]).all()
    np.testing.assert_array_equal(filled[1, :, 0, 0], [1, 1, 1, 3, 3, 3])
    np.testing.assert_array_equal(filled[1, :, 0, 1], [2, 2, 2, 2, 2, 2])

    # GDPS-style lead set: 3-hourly, then 6-hourly
    lead_hours = np.array([3, 6, 9, 15, 21])
    leads = (lead_hours * 2.0).reshape(1, -1, 1, 1).astype(np.float32)
    hourly = interp_time_take_blend(
        da.from_array(leads, chunks=(1, -1, 1, 1)),
        stacked_timesUnix=lead_hours * hour,
        hourly_timesUnix=np.arange(3, 22) * hour,
        fill_value=np.nan,
    ).compute()

    np.testing.assert_allclose(hourly[0, :, 0, 0], np.arange(3, 22) * 2.0)
