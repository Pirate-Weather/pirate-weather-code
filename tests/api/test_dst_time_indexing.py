"""Daily bucketing of hourly data across daylight saving time transitions.

``calculate_time_indexing`` decides which local day every hour belongs to. On DST
transition days the local day is 23 or 25 hours long; treating every day as 24 UTC
hours shifts daily highs, lows, and precipitation totals into the wrong day.
"""

import datetime

import numpy as np
import pytest
import pytz

from API.utils.time_indexing import calculate_time_indexing

EASTERN = pytz.timezone("America/New_York")
HOUR = 3600


def _hour_array_grib(base_date: datetime.datetime, hours: int) -> np.ndarray:
    """Hourly Unix timestamps from midnight UTC on ``base_date``, like ``hour_array_grib``."""
    start = int(base_date.replace(tzinfo=datetime.UTC).timestamp())
    return np.arange(start, start + hours * HOUR, HOUR, dtype=np.int32)


@pytest.mark.parametrize(
    ("date", "first_day_hours"),
    [
        pytest.param(datetime.datetime(2024, 3, 10), 23, id="spring-forward"),
        pytest.param(datetime.datetime(2024, 11, 3), 25, id="fall-back"),
    ],
)
def test_dst_day_has_correct_local_length(date, first_day_hours):
    """Day boundaries must follow local midnight, so DST days are 23 or 25 hours long.

    Also guards against hours left unassigned (NaN) or assigned out of order around
    the skipped/repeated local hour.
    """
    result = calculate_time_indexing(
        base_time=date,
        timezone_localizer=EASTERN,
        hour_array_grib=_hour_array_grib(date, hours=200),
        time_machine=False,
    )

    day_lengths = np.diff(result.day_array_grib.astype(np.int64)) // HOUR
    assert day_lengths[0] == first_day_hours
    assert np.all(day_lengths[1:] == 24)

    day_index = result.hourly_day_index
    assert not np.any(np.isnan(day_index))
    assert np.all(np.diff(day_index) >= 0)


def test_multi_day_timemachine_request_uses_one_bucket_per_day():
    """Multi-day time machine requests must split hours into separate days (#624).

    Before #624, every time machine hour was placed in day 0, collapsing a
    three-day request into a single daily summary.
    """
    date = datetime.datetime(2024, 3, 10)
    result = calculate_time_indexing(
        base_time=date,
        timezone_localizer=EASTERN,
        hour_array_grib=_hour_array_grib(date, hours=73),
        time_machine=True,
        daily_days=3,
    )

    assert np.all(np.diff(result.hourly_day_index) >= 0)
    assert np.array_equal(np.unique(result.hourly_day_index), [0, 1, 2])
