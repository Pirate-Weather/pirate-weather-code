"""Request parsing: location/time strings, model exclusion flags, and
Time Machine ranges, as produced by prepare_initial_request.
"""

import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from timezonefinder import TimezoneFinder

from API.request.preprocess import (
    _parse_location,
    _parse_timemachine_range,
    parse_request_time,
    prepare_initial_request,
)

TF = TimezoneFinder(in_memory=True)
NOW = datetime.datetime(2025, 1, 1, 12, 0, 0)
FORCE_NOW = int(NOW.replace(tzinfo=datetime.UTC).timestamp())
NEW_YORK = "40.7128,-74.0060"
HAVANA = "23.1136,-82.3666"
BEIRUT = "33.8938,35.5018"


async def _prepare(
    location, *, query_params=None, exclude=None, include=None, force_now=FORCE_NOW
):
    """Run prepare_initial_request offline with a fixed "now"."""
    request = MagicMock()
    request.url = f"http://localhost/forecast/apikey/{location}"
    request.query_params = query_params or {}
    return await prepare_initial_request(
        request=request,
        location=location,
        units="us",
        extend=None,
        exclude=exclude,
        include=include,
        lang="en",
        version=None,
        tmextra=None,
        icon="pirate",
        extraVars=None,
        tf=TF,
        translations={"en": {}},
        timing_enabled=False,
        force_now=str(force_now),
        logger=MagicMock(),
        start_time=NOW,
    )


def _parse_time(time_str):
    return parse_request_time(
        time_str=time_str, now_time=NOW, lat=40.7128, az_lon=-74.0060, tf=TF
    )


@pytest.mark.parametrize(
    ("time_str", "expected_delta"),
    [
        pytest.param("-30", datetime.timedelta(seconds=-30), id="bare-seconds"),
        pytest.param("-2h", datetime.timedelta(hours=-2), id="hours-suffix"),
    ],
)
def test_parse_request_time_relative_offsets(time_str, expected_delta):
    """Dark Sky-compatible clients send negative relative offsets; they must be
    applied to "now" rather than read as unix timestamps.
    """
    assert _parse_time(time_str) == NOW + expected_delta


@pytest.mark.parametrize(
    "time_str",
    [
        pytest.param("A" * 65, id="overlong"),
        pytest.param("+1h", id="positive-relative-offset"),
    ],
)
def test_parse_request_time_rejects_invalid_specifications(time_str):
    """Malformed or future-relative times must be a 400, not a crash or a
    silently wrong request time.
    """
    with pytest.raises(HTTPException, match="Invalid Time Specification"):
        _parse_time(time_str)


@pytest.mark.parametrize(
    ("location", "geocoded", "expected_request"),
    [
        pytest.param(
            "New York,US", (40.7128, -74.006), ["40.7128", "-74.006"], id="city"
        ),
        pytest.param(
            "Paris,France,1704067200",
            (48.8566, 2.3522),
            ["48.8566", "2.3522", "1704067200"],
            id="city-with-time",
        ),
    ],
)
def test_parse_location_city_country(location, geocoded, expected_request, monkeypatch):
    """City,Country[,time] must be geocoded and rewritten into the numeric
    lat,lon[,time] form the rest of the pipeline expects, keeping the time part.
    """
    calls = []

    def fake_geocode(city, country):
        calls.append((city, country))
        return geocoded

    monkeypatch.setattr("API.request.preprocess.geocode_city_country", fake_geocode)

    lat, lon_in, lon, az_lon, location_req = _parse_location(location)

    assert calls == [tuple(location.split(",")[:2])]
    assert (lat, lon_in) == geocoded
    assert lon == pytest.approx(geocoded[1] % 360)
    assert az_lon == pytest.approx(geocoded[1])
    assert location_req == expected_request


@pytest.mark.asyncio
async def test_ai_mode_excludes_cmc_and_honours_ai_model_exclusion():
    """include=aimodels must drop the CMC models (they would outrank AI output in
    Canada), and exclude=aigefs must drop only AIGEFS among the AI models.
    """
    result = await _prepare(NEW_YORK, exclude="aigefs", include="aimodels")

    assert result.inc_aimodels == 1
    assert (result.ex_aigefs, result.ex_aigfs, result.ex_aifs) == (1, 0, 0)
    assert (result.ex_hrdps, result.ex_gdps, result.ex_geps, result.ex_reps) == (
        1,
        1,
        1,
        1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("include", [None, "aimodels"])
@pytest.mark.parametrize(
    ("exclude", "flags"),
    [
        pytest.param("aigfs", (1, 0, 0, 0), id="aigfs"),
        pytest.param("aigefs", (0, 1, 0, 0), id="aigefs"),
        pytest.param("gfs", (0, 0, 1, 0), id="gfs"),
        pytest.param("gefs", (0, 0, 0, 1), id="gefs"),
    ],
)
async def test_excluding_a_model_keeps_its_counterpart(exclude, flags, include):
    """The names overlap ("aigfs" contains "gfs"), and a substring check made
    excluding an AI model silently drop the conventional model that the forecast
    falls back on.
    """
    result = await _prepare(NEW_YORK, exclude=exclude, include=include)

    assert (result.ex_aigfs, result.ex_aigefs, result.ex_gfs, result.ex_gefs) == flags


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_time", "midnight_utc", "day_hours"),
    [
        pytest.param(
            datetime.datetime(2024, 3, 10, 12, tzinfo=datetime.UTC),
            datetime.datetime(2024, 3, 10, 5, tzinfo=datetime.UTC),
            23,
            id="spring-forward",
        ),
        pytest.param(
            datetime.datetime(2024, 11, 3, 12, tzinfo=datetime.UTC),
            datetime.datetime(2024, 11, 3, 4, tzinfo=datetime.UTC),
            25,
            id="fall-back",
        ),
    ],
)
async def test_dst_day_starts_at_local_midnight(request_time, midnight_utc, day_hours):
    """After the clocks change, keeping the current UTC offset put midnight an hour
    off: every hourly row and the daily buckets slid against the true local day,
    and time machine returned 24 hours for a 23 or 25 hour day. Cover both the
    forecast and time machine paths, which share this setup, and check that an
    explicit hours request still returns what it asked for.
    """
    timestamp = int(request_time.timestamp())
    midnight = int(midnight_utc.timestamp())

    forecast = await _prepare(NEW_YORK, force_now=timestamp)
    assert forecast.hour_array_grib[0] == midnight
    assert forecast.day_array_grib[0] == midnight
    # The current hour is found from these two, and must be the row for "now"
    current_hour = (
        int(forecast.base_hour.timestamp() - forecast.base_day.timestamp()) // 3600
    )
    assert forecast.hour_array_grib[current_hour] == timestamp

    time_machine = await _prepare(f"{NEW_YORK},{timestamp}")
    assert time_machine.hour_array_grib[0] == midnight
    assert time_machine.output_hours == day_hours
    assert (
        time_machine.hour_array_grib[day_hours - 1] == midnight + (day_hours - 1) * 3600
    )
    assert len(time_machine.hour_array) >= day_hours

    explicit = await _prepare(f"{NEW_YORK},{timestamp}", query_params={"hours": "24"})
    assert explicit.output_hours == 24


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("location", "day_start_utc", "day_hours"),
    [
        pytest.param(
            HAVANA,
            datetime.datetime(2024, 11, 3, 4, tzinfo=datetime.UTC),
            25,
            id="midnight-happens-twice",
        ),
        pytest.param(
            BEIRUT,
            datetime.datetime(2024, 3, 30, 22, tzinfo=datetime.UTC),
            23,
            id="midnight-never-happens",
        ),
    ],
)
async def test_day_starts_correctly_where_clocks_change_at_midnight(
    location, day_start_utc, day_hours
):
    """Most zones change their clocks overnight, but some do it at midnight. Where
    midnight happens twice the day must start at the first, or its first hour is
    lost from the day and from every daily statistic. Where it never happens the
    day starts when the clocks jump. The hour grid and day buckets must agree.
    """
    noon = int((day_start_utc + datetime.timedelta(hours=14)).timestamp())
    day_start = int(day_start_utc.timestamp())

    time_machine = await _prepare(f"{location},{noon}")

    assert time_machine.hour_array_grib[0] == day_start
    assert time_machine.day_array_grib[0] == day_start
    assert time_machine.output_hours == day_hours


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("location", "request_time", "row_start"),
    [
        pytest.param(
            NEW_YORK,
            datetime.datetime(2024, 11, 3, 5, 30, tzinfo=datetime.UTC),
            datetime.datetime(2024, 11, 3, 5, tzinfo=datetime.UTC),
            id="new-york-first-0130",
        ),
        pytest.param(
            NEW_YORK,
            datetime.datetime(2024, 11, 3, 6, 30, tzinfo=datetime.UTC),
            datetime.datetime(2024, 11, 3, 6, tzinfo=datetime.UTC),
            id="new-york-second-0130",
        ),
        pytest.param(
            HAVANA,
            datetime.datetime(2024, 11, 3, 4, 30, tzinfo=datetime.UTC),
            datetime.datetime(2024, 11, 3, 4, tzinfo=datetime.UTC),
            id="havana-first-0030",
        ),
        pytest.param(
            HAVANA,
            datetime.datetime(2024, 11, 3, 5, 30, tzinfo=datetime.UTC),
            datetime.datetime(2024, 11, 3, 5, tzinfo=datetime.UTC),
            id="havana-second-0030",
        ),
    ],
)
async def test_current_hour_during_the_repeated_hour(location, request_time, row_start):
    """The same wall-clock hour happens twice when clocks fall back. Reading the
    first as the second started the forecast an hour ahead of now, and zeroed an
    hour of today's precipitation as if it had passed.
    """
    result = await _prepare(location, force_now=int(request_time.timestamp()))

    current_row = (
        int(result.base_hour.timestamp() - result.base_day.timestamp()) // 3600
    )
    assert result.hour_array_grib[current_row] == int(row_start.timestamp())


@pytest.mark.asyncio
async def test_urma_exclusion_is_independent_of_gfs():
    """URMA (#746) fills recent Time Machine hours and falls back to GFS per hour,
    so exclude=urma and exclude=gfs must each switch off only their own model.
    """
    two_days_ago = FORCE_NOW - 2 * 86400
    location = f"{NEW_YORK},{two_days_ago}"

    exclude_urma = await _prepare(location, exclude="urma")
    exclude_gfs = await _prepare(location, exclude="gfs")

    assert (exclude_urma.ex_urma, exclude_urma.ex_gfs) == (1, 0)
    assert (exclude_gfs.ex_urma, exclude_gfs.ex_gfs) == (0, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_params", "daily_days", "output_days", "output_hours", "hour_rows"),
    [
        pytest.param({"days": "3"}, 3, 3, 72, 73, id="days"),
        # hours=30 needs two days of backing data but only one full output day.
        pytest.param({"hours": "30"}, 2, 1, 30, 49, id="hours"),
    ],
)
async def test_timemachine_days_and_hours_set_output_ranges(
    query_params, daily_days, output_days, output_hours, hour_rows
):
    """Time Machine days/hours parameters size the hourly/daily output and the
    data grid; a mismatch truncates the response or reads too little history.
    """
    result = await _prepare(f"{NEW_YORK},1704067200", query_params=query_params)

    assert result.time_machine is True
    assert result.daily_days == daily_days
    assert result.output_days == output_days
    assert result.output_hours == output_hours
    assert len(result.hour_array) == hour_rows


@pytest.mark.parametrize(
    ("query_params", "message"),
    [
        pytest.param({"days": "0"}, "Invalid days parameter", id="days-zero"),
        pytest.param({"days": "9"}, "Invalid days parameter", id="days-over-max"),
        pytest.param({"days": "abc"}, "Invalid days parameter", id="days-text"),
        pytest.param({"hours": "193"}, "Invalid hours parameter", id="hours-over-max"),
    ],
)
def test_timemachine_range_rejects_invalid_values(query_params, message):
    """Out-of-range Time Machine ranges must be a 400 rather than an oversized
    (expensive) read or an empty response.
    """
    request = MagicMock()
    request.query_params = query_params

    with pytest.raises(HTTPException, match=message) as exc:
        _parse_timemachine_range(request, time_machine=True)
    assert exc.value.status_code == 400


def _utc(*args):
    return int(datetime.datetime(*args, tzinfo=datetime.UTC).timestamp())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("force_now", "request_time", "expected"),
    [
        pytest.param(
            FORCE_NOW,
            FORCE_NOW + 3 * 86400,
            {"midnight": _utc(2025, 1, 4, 5), "hours": 24},
            id="three-days-ahead",
        ),
        pytest.param(
            _utc(2025, 10, 30, 12),
            _utc(2025, 11, 2, 16),
            {"midnight": _utc(2025, 11, 2, 4), "hours": 25},
            id="fall-back-day",
        ),
        # The requested local day ends 209 h from now: inside the horizon.
        pytest.param(
            FORCE_NOW,
            FORCE_NOW + 8 * 86400,
            {"midnight": _utc(2025, 1, 9, 5), "hours": 24},
            id="eight-days-ahead",
        ),
        pytest.param(FORCE_NOW, FORCE_NOW + 30 * 60, "now", id="within-an-hour"),
        # The requested local day ends 233 h from now, past the GFS/GEFS stores.
        pytest.param(FORCE_NOW, FORCE_NOW + 9 * 86400, "rejected", id="past-horizon"),
    ],
)
async def test_future_request_returns_the_requested_local_day(
    force_now, request_time, expected
):
    """#689: a future time returns Dark Sky's single-day shape (hourly from local
    midnight to the next, one daily point, no minutely or alerts) instead of a
    400. It must not be read as Time Machine (which drops the forecast-only
    models and fields), a time within the hour must still mean now, and a day
    the longest stores don't reach must be refused rather than served empty.
    """
    location = f"{NEW_YORK},{request_time}"

    if expected == "rejected":
        with pytest.raises(HTTPException, match="beyond the forecast horizon") as exc:
            await _prepare(location, force_now=force_now)
        assert exc.value.status_code == 400
        return

    result = await _prepare(location, force_now=force_now)
    assert result.time_machine is False

    if expected == "now":
        assert result.future_day is False
        assert result.utc_time == result.now_time
        assert result.output_hours == 48
        return

    assert result.future_day is True
    assert (result.ex_minutely, result.ex_alerts, result.read_wmo_alerts) == (
        1,
        1,
        False,
    )
    assert (result.ex_rtma_ru, result.ex_librewxr) == (1, 1)
    # Forecast models stay in, unlike Time Machine
    assert (result.ex_nbm, result.ex_hrrr, result.ex_gefs, result.ex_ecmwf) == (
        0,
        0,
        0,
        0,
    )
    assert int(result.base_day_utc.timestamp()) == expected["midnight"]
    assert result.hour_array_grib[0] == expected["midnight"]
    assert result.output_days == 1
    assert result.output_hours == expected["hours"]
    assert len(result.hour_array) >= expected["hours"]
