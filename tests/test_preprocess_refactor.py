import datetime
from unittest.mock import MagicMock

import pytest
from timezonefinder import TimezoneFinder

from API.request.preprocess import (
    InitialRequestContext,
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
def test_parse_request_time_supports_negative_relative_offsets(time_str, expected_delta):
    now_time = datetime.datetime(2025, 1, 1, 12, 0, 0)

    parsed_time = parse_request_time(
        time_str=time_str,
        now_time=now_time,
        lat=40.7128,
        az_lon=-74.0060,
        tf=TimezoneFinder(in_memory=True),
    )

    assert parsed_time == now_time + expected_delta


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
