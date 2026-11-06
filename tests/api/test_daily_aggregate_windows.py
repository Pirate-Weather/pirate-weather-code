"""Regression tests for the stacked daily aggregation in API.daily.builder."""

import numpy as np

from API.constants.forecast_const import DATA_HOURLY
from API.daily.builder import _aggregate_stats

HOUR_SECONDS = 3600
GAP = -999


def test_days_of_different_lengths_aggregate_like_their_own_hours():
    """Days are stacked into one array, so a DST day (23 or 25 hours) is padded.
    Padding that leaked into a mean, sum, extreme or its time would misreport
    those days with no error; a NaN must still make its whole day NaN, with the
    extreme's time at the first NaN.
    """
    rng = np.random.default_rng(0)
    hours_per_day = [24, 25, 23]
    n_rows = sum(hours_per_day) + 4
    hourly = rng.normal(size=(n_rows, 6))
    hourly[:, DATA_HOURLY["time"]] = np.arange(n_rows) * HOUR_SECONDS
    # An all -0.0 column in the shortest day must not gain a sign from padding.
    hourly[:, 4] = -0.0
    hourly[30, 5] = np.nan

    index_array = np.full(n_rows, GAP)
    index_array[2:-2] = np.repeat(np.arange(3), hours_per_day)

    mean, total, high, low, high_time, low_time, _ = _aggregate_stats(
        hourly,
        index_array,
        3,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_min=True,
        calc_argmax=True,
        calc_argmin=True,
    )

    for day in range(3):
        day_hours = hourly[index_array == day]
        day_times = day_hours[:, DATA_HOURLY["time"]]
        assert np.array_equal(mean[day], day_hours.mean(axis=0), equal_nan=True)
        assert total[day].tobytes() == np.add.reduce(day_hours, axis=0).tobytes()
        assert np.array_equal(high[day], day_hours.max(axis=0), equal_nan=True)
        assert np.array_equal(low[day], day_hours.min(axis=0), equal_nan=True)
        assert np.array_equal(high_time[day], day_times[day_hours.argmax(axis=0)])
        assert np.array_equal(low_time[day], day_times[day_hours.argmin(axis=0)])


def test_daily_precip_type_prefers_wet_hours_and_the_first_seen_type():
    """A day's icon and precipType start from this: a wet type must beat more
    numerous dry hours, and a tie must go to the type seen first. Numpy's handling
    of repeated indices makes the tie-break easy to get backwards silently.
    """
    day_types = [
        [0, 0, 0, 0, 0, 0],  # dry
        [0, 0, 0, 0, 4, 1],  # one hour each of rain and snow: first seen wins
        [0, 0, 1, 4, 4, 0],  # the more common wet type wins
        [3, 0, 0, 0, np.nan, 0],  # an hour without a type counts as dry
    ]
    n_rows = len(day_types) * len(day_types[0])
    hourly = np.zeros((n_rows, 2))
    hourly[:, DATA_HOURLY["type"]] = np.ravel(day_types)
    index_array = np.repeat(np.arange(len(day_types)), len(day_types[0]))

    precip_type = _aggregate_stats(hourly, index_array, 4, calc_precip=True)[6]

    assert precip_type.tolist() == [0.0, 4.0, 4.0, 3.0]
