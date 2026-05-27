import asyncio
import importlib
import sys
import types

import pytest


class _FakeFastMCP:
    def __init__(self, *args, **kwargs):
        pass

    def http_app(self, *args, **kwargs):
        async def app(scope, receive, send):
            return None

        return app

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def tool(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


@pytest.fixture
def mcp_module(monkeypatch):
    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = _FakeFastMCP
    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)
    monkeypatch.delenv("PW_MCP_PUBLIC_URL", raising=False)
    sys.modules.pop("MCP.pirate_weather_mcp", None)
    return importlib.import_module("MCP.pirate_weather_mcp")


def test_public_url_middleware_rewrites_origin(mcp_module):
    captured_scope = {}

    async def app(scope, receive, send):
        captured_scope.update(scope)

    middleware = mcp_module.PublicUrlMiddleware(
        app,
        "https://mcp.pirateweather.net/mcp",
    )
    scope = {
        "type": "http",
        "scheme": "http",
        "server": ("127.0.0.1", 8084),
        "headers": [(b"host", b"127.0.0.1:8084"), (b"accept", b"*/*")],
    }

    asyncio.run(middleware(scope, None, None))

    assert captured_scope["scheme"] == "https"
    assert captured_scope["server"] == ("mcp.pirateweather.net", 443)
    assert (b"host", b"mcp.pirateweather.net") in captured_scope["headers"]
    assert (b"host", b"127.0.0.1:8084") not in captured_scope["headers"]


def test_public_url_middleware_ignores_unconfigured_origin(mcp_module):
    captured_scope = {}

    async def app(scope, receive, send):
        captured_scope.update(scope)

    middleware = mcp_module.PublicUrlMiddleware(app)
    scope = {
        "type": "http",
        "scheme": "http",
        "server": ("127.0.0.1", 8084),
        "headers": [(b"host", b"127.0.0.1:8084")],
    }

    asyncio.run(middleware(scope, None, None))

    assert captured_scope["scheme"] == "http"
    assert captured_scope["server"] == ("127.0.0.1", 8084)
    assert captured_scope["headers"] == [(b"host", b"127.0.0.1:8084")]


def test_public_url_requires_absolute_http_url(mcp_module):
    with pytest.raises(ValueError, match="absolute http"):
        mcp_module.PublicUrlMiddleware(
            lambda scope, receive, send: None, "mcp.pirateweather.net/mcp"
        )


def test_historical_weather_omits_false_tmextra(mcp_module, monkeypatch):
    captured = {}

    def fake_request_forecast(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(mcp_module, "_request_forecast", fake_request_forecast)

    assert mcp_module.get_historical_weather(
        latitude=45.4215,
        longitude=-75.6972,
        time="1730869200",
        tmextra=False,
    ) == {"ok": True}
    assert captured["tmextra"] is None

    mcp_module.get_historical_weather(
        latitude=45.4215,
        longitude=-75.6972,
        time="1730869200",
        tmextra=True,
    )
    assert captured["tmextra"] == 1


def test_with_iso_times_adds_time_iso_after_numeric_time(mcp_module):
    result = mcp_module._with_iso_times(
        {
            "time": 1730869200,
            "summary": "Clear",
            "nested": [
                {"time": 1730872800, "summary": "Still clear"},
                {"time": "not-a-unix-time"},
            ],
        }
    )

    assert list(result) == ["time", "timeISO", "summary", "nested"]
    assert result["timeISO"] == "2024-11-06T05:00:00+00:00"
    assert list(result["nested"][0]) == ["time", "timeISO", "summary"]
    assert result["nested"][0]["timeISO"] == "2024-11-06T06:00:00+00:00"
    assert "timeISO" not in result["nested"][1]


def test_get_forecast_returns_near_term_forecast_alerts_and_summary_text(
    mcp_module, monkeypatch
):
    captured = {}

    def fake_request_forecast(**kwargs):
        captured.update(kwargs)
        return {
            "latitude": 45.4215,
            "longitude": -75.6972,
            "timezone": "America/Toronto",
            "offset": -4,
            "currently": {
                "time": 1730869200,
                "summary": "Clear",
                "icon": "clear-day",
                "temperature": 12,
            },
            "minutely": {"summary": "No precipitation", "icon": "clear-day"},
            "hourly": {
                "summary": "Clear for the hour",
                "icon": "clear-day",
                "data": [
                    {"time": 1730869200, "summary": "Clear this hour"},
                    {"time": 1730872800, "summary": "Clear next hour"},
                ],
            },
            "daily": {
                "summary": "Clear throughout the day",
                "icon": "clear-day",
                "data": [
                    {"time": 1730851200, "summary": "Clear today"},
                    {"time": 1730937600, "summary": "Clear tomorrow"},
                ],
            },
            "alerts": [{"title": "Wind Advisory"}],
            "flags": {"units": "si"},
        }

    monkeypatch.setattr(mcp_module, "_request_forecast", fake_request_forecast)

    result = mcp_module.get_forecast(
        latitude=45.4215,
        longitude=-75.6972,
        units="si",
        lang="en",
    )

    assert captured["version"] == 2
    assert captured["blocks"] == "currently,minutely,hourly,daily,alerts,flags"
    assert captured["hourly_indices"] == "0,1"
    assert captured["daily_indices"] == "0,1"
    assert result["currently"]["summary"] == "Clear"
    assert result["currently"]["timeISO"] == "2024-11-06T05:00:00+00:00"
    assert result["this_hour"] == {
        "time": 1730869200,
        "timeISO": "2024-11-06T05:00:00+00:00",
        "summary": "Clear this hour",
    }
    assert result["next_hour"] == {
        "time": 1730872800,
        "timeISO": "2024-11-06T06:00:00+00:00",
        "summary": "Clear next hour",
    }
    assert result["today"] == {
        "time": 1730851200,
        "timeISO": "2024-11-06T00:00:00+00:00",
        "summary": "Clear today",
    }
    assert result["tomorrow"] == {
        "time": 1730937600,
        "timeISO": "2024-11-07T00:00:00+00:00",
        "summary": "Clear tomorrow",
    }
    assert result["alerts"] == [{"title": "Wind Advisory"}]
    assert result["summary_text"] == {
        "currently": {"summary": "Clear", "icon": "clear-day"},
        "minutely": {"summary": "No precipitation", "icon": "clear-day"},
        "hourly": {"summary": "Clear for the hour", "icon": "clear-day"},
        "daily": {"summary": "Clear throughout the day", "icon": "clear-day"},
    }
