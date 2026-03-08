"""Tests for the timemachine endpoint during daylight saving time transitions.

DST spring-forward (clocks advance 1 hour):
    US Eastern: second Sunday in March, 2:00 AM -> 3:00 AM.
    Chosen date: 2024-03-10 (Sunday).

DST fall-back (clocks retreat 1 hour):
    US Eastern: first Sunday in November, 2:00 AM -> 1:00 AM.
    Chosen date: 2024-11-03 (Sunday).
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import pytest
import pytz

from API.utils.time_indexing import calculate_time_indexing
from tests.test_s3_live import _get_client

PW_API = os.environ.get("PW_API")

# US Eastern timezone – observes DST
EASTERN = pytz.timezone("America/New_York")

# New York City used as a representative US Eastern DST location
NYC_LAT = 40.7128
NYC_LON = -74.0060

# Spring-forward: 2024-03-10, clocks go 2:00 AM -> 3:00 AM (23-hour local day)
SPRING_FORWARD_DATE = datetime.datetime(2024, 3, 10)
# Fall-back: 2024-11-03, clocks go 2:00 AM -> 1:00 AM (25-hour local day)
FALL_BACK_DATE = datetime.datetime(2024, 11, 3)

# Timemachine timestamps: noon UTC on each DST transition day (well in the past)
SPRING_FORWARD_TIMESTAMP = int(
    datetime.datetime(2024, 3, 10, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
)
FALL_BACK_TIMESTAMP = int(
    datetime.datetime(2024, 11, 3, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hour_array_grib(base_date: datetime.datetime, hours: int = 200) -> np.ndarray:
    """Return an int32 array of hourly Unix timestamps starting at midnight UTC
    on *base_date* and spanning *hours* hours.  This mirrors the structure of
    ``hour_array_grib`` produced by :func:`initialize_time_grids`.
    """
    epoch = datetime.datetime(1970, 1, 1, 0, 0, 0)
    base_utc = datetime.datetime(base_date.year, base_date.month, base_date.day, 0, 0, 0)
    return np.array(
        [
            int((base_utc + datetime.timedelta(hours=i) - epoch).total_seconds())
            for i in range(hours)
        ],
        dtype=np.int32,
    )


def _check_day_index(result, label: str) -> None:
    """Assert that ``hourly_day_index`` contains no sentinel missing-data values."""
    FINALIZED_MISSING = -999
    bad = np.sum(result.hourly_day_index == FINALIZED_MISSING)
    assert bad == 0, (
        f"{label}: hourly_day_index has {bad} missing-data sentinels (-999). "
        "DST forward-fill may not be working correctly."
    )
    assert np.all(result.hourly_day_index >= 0), (
        f"{label}: hourly_day_index contains negative values."
    )


# ---------------------------------------------------------------------------
# Unit tests – calculate_time_indexing with DST transition dates
# ---------------------------------------------------------------------------


def test_time_indexing_no_missing_on_dst_spring_forward():
    """DST spring-forward: hourly_day_index must have no missing sentinels.

    On 2024-03-10 clocks in the US Eastern timezone spring forward from
    2:00 AM to 3:00 AM, creating a 23-hour local day.  The hour that is
    skipped locally (2:00–3:00 AM EST, i.e. 07:00 UTC) could fall into a
    gap between consecutive day-boundary timestamps.  The forward-fill
    logic in ``calculate_time_indexing`` must prevent any -999 sentinels
    from remaining in the output.
    """
    hour_array_grib = _make_hour_array_grib(SPRING_FORWARD_DATE)
    result = calculate_time_indexing(
        base_time=SPRING_FORWARD_DATE,
        timezone_localizer=EASTERN,
        hour_array_grib=hour_array_grib,
        time_machine=False,
    )
    _check_day_index(result, "DST spring-forward (2024-03-10)")


def test_time_indexing_no_missing_on_dst_fall_back():
    """DST fall-back: hourly_day_index must have no missing sentinels.

    On 2024-11-03 clocks in the US Eastern timezone fall back from
    2:00 AM to 1:00 AM, creating a 25-hour local day.  The repeated hour
    (1:00–2:00 AM EST, i.e. 06:00 UTC appearing twice) could cause an
    overlap in day-boundary ranges.  The forward-fill logic in
    ``calculate_time_indexing`` must prevent any -999 sentinels.
    """
    hour_array_grib = _make_hour_array_grib(FALL_BACK_DATE)
    result = calculate_time_indexing(
        base_time=FALL_BACK_DATE,
        timezone_localizer=EASTERN,
        hour_array_grib=hour_array_grib,
        time_machine=False,
    )
    _check_day_index(result, "DST fall-back (2024-11-03)")


def test_time_indexing_dst_spring_forward_day_boundaries():
    """Day boundaries must be monotonically increasing during DST spring-forward."""
    hour_array_grib = _make_hour_array_grib(SPRING_FORWARD_DATE)
    result = calculate_time_indexing(
        base_time=SPRING_FORWARD_DATE,
        timezone_localizer=EASTERN,
        hour_array_grib=hour_array_grib,
        time_machine=False,
    )
    # day_array_grib stores midnight-UTC timestamps for each local day
    diffs = np.diff(result.day_array_grib.astype(np.int64))
    assert np.all(diffs > 0), (
        "day_array_grib is not strictly increasing during DST spring-forward. "
        f"Differences: {diffs}"
    )


def test_time_indexing_dst_fall_back_day_boundaries():
    """Day boundaries must be monotonically increasing during DST fall-back."""
    hour_array_grib = _make_hour_array_grib(FALL_BACK_DATE)
    result = calculate_time_indexing(
        base_time=FALL_BACK_DATE,
        timezone_localizer=EASTERN,
        hour_array_grib=hour_array_grib,
        time_machine=False,
    )
    diffs = np.diff(result.day_array_grib.astype(np.int64))
    assert np.all(diffs > 0), (
        "day_array_grib is not strictly increasing during DST fall-back. "
        f"Differences: {diffs}"
    )


def test_time_indexing_timemachine_mode_dst_spring_forward():
    """In time-machine mode all day indices must be zero on a DST spring-forward day."""
    hour_array_grib = _make_hour_array_grib(SPRING_FORWARD_DATE, hours=24)
    result = calculate_time_indexing(
        base_time=SPRING_FORWARD_DATE,
        timezone_localizer=EASTERN,
        hour_array_grib=hour_array_grib,
        time_machine=True,
    )
    assert np.all(result.hourly_day_index == 0), (
        "Time-machine mode must return all-zero hourly_day_index (DST spring-forward)."
    )


def test_time_indexing_timemachine_mode_dst_fall_back():
    """In time-machine mode all day indices must be zero on a DST fall-back day."""
    hour_array_grib = _make_hour_array_grib(FALL_BACK_DATE, hours=24)
    result = calculate_time_indexing(
        base_time=FALL_BACK_DATE,
        timezone_localizer=EASTERN,
        hour_array_grib=hour_array_grib,
        time_machine=True,
    )
    assert np.all(result.hourly_day_index == 0), (
        "Time-machine mode must return all-zero hourly_day_index (DST fall-back)."
    )


# ---------------------------------------------------------------------------
# Integration tests – timemachine HTTP endpoint with DST transition dates
# These require live S3 data and are skipped when PW_API is not set.
# ---------------------------------------------------------------------------


def _check_timemachine_response(data: dict, label: str) -> None:
    """Validate the key blocks of a timemachine response."""
    assert "hourly" in data, f"{label}: response missing 'hourly' block"
    hourly_data = data["hourly"]["data"]
    assert len(hourly_data) >= 1, f"{label}: 'hourly.data' is empty"

    # Timestamps must be strictly increasing
    times = [h["time"] for h in hourly_data]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], (
            f"{label}: hourly timestamps not monotonically increasing at index {i}: "
            f"{times[i - 1]} -> {times[i]}"
        )

    assert "daily" in data, f"{label}: response missing 'daily' block"
    assert len(data["daily"]["data"]) >= 1, f"{label}: 'daily.data' is empty"


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
def test_timemachine_endpoint_dst_spring_forward():
    """Timemachine endpoint returns valid data on a DST spring-forward day.

    Calls the timemachine endpoint for New York City on 2024-03-10 (the day
    US Eastern clocks spring forward) and verifies the response structure is
    well-formed with monotonically increasing hourly timestamps.
    """
    client = _get_client()
    response = client.get(
        f"/timemachine/{PW_API}/{NYC_LAT},{NYC_LON},{SPRING_FORWARD_TIMESTAMP}?version=2"
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code} for DST spring-forward timemachine request"
    )
    _check_timemachine_response(response.json(), "DST spring-forward (2024-03-10)")


@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
def test_timemachine_endpoint_dst_fall_back():
    """Timemachine endpoint returns valid data on a DST fall-back day.

    Calls the timemachine endpoint for New York City on 2024-11-03 (the day
    US Eastern clocks fall back) and verifies the response structure is
    well-formed with monotonically increasing hourly timestamps.
    """
    client = _get_client()
    response = client.get(
        f"/timemachine/{PW_API}/{NYC_LAT},{NYC_LON},{FALL_BACK_TIMESTAMP}?version=2"
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code} for DST fall-back timemachine request"
    )
    _check_timemachine_response(response.json(), "DST fall-back (2024-11-03)")
