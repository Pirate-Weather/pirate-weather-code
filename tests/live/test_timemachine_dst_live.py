"""Time machine requests on DST transition days, served from live ERA5 data."""

import datetime
import itertools

import pytest

NYC_LAT = 40.7128
NYC_LON = -74.0060


@pytest.mark.xfail(
    strict=True,
    reason="TODO.md: local midnight is off by an hour on DST transition days",
)
@pytest.mark.parametrize(
    ("date", "local_day_hours"),
    [
        pytest.param(
            datetime.datetime(2024, 3, 10, 12, tzinfo=datetime.UTC),
            23,
            id="spring-forward",
        ),
        pytest.param(
            datetime.datetime(2024, 11, 3, 12, tzinfo=datetime.UTC), 25, id="fall-back"
        ),
    ],
)
def test_timemachine_dst_day_returns_full_local_day(
    live_client, pw_api_key, date, local_day_hours
):
    """A time machine day must contain exactly the hours of the local day.

    Protects the end-to-end time machine path (ERA5 read, local-day windowing,
    daily aggregation) on the two days a year where a local day is not 24 hours.
    """
    timestamp = int(date.timestamp())
    response = live_client.get(
        f"/timemachine/{pw_api_key}/{NYC_LAT},{NYC_LON},{timestamp}?version=2"
    )
    assert response.status_code == 200

    data = response.json()
    times = [hour["time"] for hour in data["hourly"]["data"]]
    assert len(times) == local_day_hours
    assert all(later > earlier for earlier, later in itertools.pairwise(times))
    assert len(data["daily"]["data"]) == 1
