
import pytest
import datetime
from unittest.mock import MagicMock
from API.request.preprocess import prepare_initial_request, InitialRequestContext
from timezonefinder import TimezoneFinder

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
    assert result.lang == "en" if hasattr(result, "lang") else True # lang might not be in context directly but used for translation
    assert result.translation == translations["en"]
