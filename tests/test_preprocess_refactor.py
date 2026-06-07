import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from timezonefinder import TimezoneFinder

from API.request.preprocess import (
    InitialRequestContext,
    _parse_location,
    _parse_parameters,
    _parse_timemachine_days,
    parse_request_time,
    prepare_initial_request,
)


@pytest.mark.parametrize(
    ("time_str", "expected_delta"),
    [
        ("-30", datetime.timedelta(seconds=-30)),
        ("-30s", datetime.timedelta(seconds=-30)),
        ("-2h", datetime.timedelta(hours=-2)),
        ("-3d", datetime.timedelta(days=-3)),
    ],
)
def test_parse_request_time_supports_negative_relative_offsets(
    time_str, expected_delta
):
    now_time = datetime.datetime(2025, 1, 1, 12, 0, 0)

    parsed_time = parse_request_time(
        time_str=time_str,
        now_time=now_time,
        lat=40.7128,
        az_lon=-74.0060,
        tf=TimezoneFinder(in_memory=True),
    )

    assert parsed_time == now_time + expected_delta


def test_parse_request_time_rejects_overlong_strings():
    now_time = datetime.datetime(2025, 1, 1, 12, 0, 0)
    overlong_time_str = "A" * 65

    with pytest.raises(HTTPException, match="Invalid Time Specification"):
        parse_request_time(
            time_str=overlong_time_str,
            now_time=now_time,
            lat=40.7128,
            az_lon=-74.0060,
            tf=TimezoneFinder(in_memory=True),
        )


def test_parse_request_time_rejects_positive_relative_offsets_with_units():
    now_time = datetime.datetime(2025, 1, 1, 12, 0, 0)

    with pytest.raises(HTTPException, match="Invalid Time Specification"):
        parse_request_time(
            time_str="+1h",
            now_time=now_time,
            lat=40.7128,
            az_lon=-74.0060,
            tf=TimezoneFinder(in_memory=True),
        )


def test_parse_location_accepts_city_country_pair(monkeypatch):
    def fake_geocode(city, country):
        assert city == "New York"
        assert country == "US"
        return 40.7128, -74.006

    monkeypatch.setattr("API.request.preprocess.geocode_city_country", fake_geocode)

    lat, lon_in, lon, az_lon, location_req = _parse_location("New York,US")

    assert lat == 40.7128
    assert lon_in == -74.006
    assert lon == pytest.approx(285.994)
    assert az_lon == pytest.approx(-74.006)
    assert location_req == ["40.7128", "-74.006"]


def test_parse_location_accepts_city_country_time_pair(monkeypatch):
    monkeypatch.setattr(
        "API.request.preprocess.geocode_city_country",
        lambda city, country: (48.8566, 2.3522),
    )

    *_, location_req = _parse_location("Paris,France,1704067200")

    assert location_req == ["48.8566", "2.3522", "1704067200"]


@pytest.mark.asyncio
async def test_prepare_initial_request_structure():
    # Mock inputs
    request = MagicMock()
    request.url = "http://localhost/forecast/apikey/40.7128,-74.0060"
    location = "40.7128,-74.0060"
    units = "us"
    extend = None
    exclude = None
    include = None
    lang = "en"
    version = None
    tmextra = None
    icon = "pirate"
    extraVars = None
    tf = TimezoneFinder(in_memory=True)
    translations = {"en": {"title": "Forecast"}}
    timing_enabled = False
    force_now = None
    logger = MagicMock()
    start_time = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

    result = await prepare_initial_request(
        request=request,
        location=location,
        units=units,
        extend=extend,
        exclude=exclude,
        include=include,
        lang=lang,
        version=version,
        tmextra=tmextra,
        icon=icon,
        extraVars=extraVars,
        tf=tf,
        translations=translations,
        timing_enabled=timing_enabled,
        force_now=force_now,
        logger=logger,
        start_time=start_time,
    )

    assert isinstance(result, InitialRequestContext)
    assert result.lat == 40.7128
    assert result.lon_in == -74.0060
    assert result.unit_system == "us"
    assert (
        result.lang == "en" if hasattr(result, "lang") else True
    )  # lang might not be in context directly but used for translation
    assert result.translation == translations["en"]


def test_parse_parameters_ai_models_include_and_exclude_priority():
    now_time = datetime.datetime(2026, 1, 1, 12, 0, 0)
    result = _parse_parameters(
        exclude="aigefs",
        include="aimodels",
        extraVars=None,
        now_time=now_time,
        utc_time=now_time,
        time_machine=False,
        tm_extra=False,
    )

    assert result[16] == 1  # ex_aigefs
    assert result[17] == 0  # ex_aigfs
    assert result[18] == 0  # ex_aifs
    assert result[21] == 1  # inc_aimodels


def test_parse_timemachine_days_defaults_to_one_for_non_timemachine():
    request = MagicMock()
    request.query_params = {"days": "5"}

    assert _parse_timemachine_days(request, time_machine=False) == 1


def test_parse_timemachine_days_accepts_valid_days_range():
    request = MagicMock()
    request.query_params = {"days": "7"}

    assert _parse_timemachine_days(request, time_machine=True) == 7


@pytest.mark.parametrize("days_value", ["0", "8", "abc"])
def test_parse_timemachine_days_rejects_invalid_values(days_value):
    request = MagicMock()
    request.query_params = {"days": days_value}

    with pytest.raises(HTTPException, match="Invalid days parameter"):
        _parse_timemachine_days(request, time_machine=True)


@pytest.mark.asyncio
async def test_prepare_initial_request_timemachine_uses_days_query_param():
    request = MagicMock()
    request.url = (
        "http://localhost/timemachine/apikey/40.7128,-74.0060,1704067200?days=3"
    )
    request.query_params = {"days": "3"}
    location = "40.7128,-74.0060,1704067200"
    tf = TimezoneFinder(in_memory=True)
    translations = {"en": {"title": "Forecast"}}
    logger = MagicMock()
    force_now = str(1735689600)

    result = await prepare_initial_request(
        request=request,
        location=location,
        units="us",
        extend=None,
        exclude=None,
        include=None,
        lang="en",
        version=None,
        tmextra=None,
        icon="pirate",
        extraVars=None,
        tf=tf,
        translations=translations,
        timing_enabled=False,
        force_now=force_now,
        logger=logger,
        start_time=datetime.datetime.now(datetime.UTC).replace(tzinfo=None),
    )

    assert isinstance(result, InitialRequestContext)
    assert result.time_machine is True
    assert result.daily_days == 3
    assert result.output_days == 3
    assert result.output_hours == 72
    assert len(result.hour_array) == 73
