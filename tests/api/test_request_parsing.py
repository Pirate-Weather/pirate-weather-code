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


async def _prepare(location, *, query_params=None, exclude=None, include=None):
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
        force_now=str(FORCE_NOW),
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
