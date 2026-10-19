"""MCP server: public-origin rewriting, request forwarding, and tool contract."""

import asyncio
import importlib
import inspect
import json
import sys
import types
from typing import ClassVar

import pytest


class _FakeFastMCP:
    """Stand-in for fastmcp that records the functions registered as tools."""

    tools: ClassVar[list] = []

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
            self.tools.append(func)
            return func

        return decorator


@pytest.fixture
def mcp_module(monkeypatch):
    fake_fastmcp = types.ModuleType("fastmcp")
    fake_fastmcp.FastMCP = _FakeFastMCP
    monkeypatch.setattr(_FakeFastMCP, "tools", [])
    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)
    monkeypatch.delenv("PW_MCP_PUBLIC_URL", raising=False)
    sys.modules.pop("MCP.pirate_weather_mcp", None)
    return importlib.import_module("MCP.pirate_weather_mcp")


@pytest.mark.parametrize(
    ("public_url", "expected_scheme", "expected_server", "expected_host"),
    [
        pytest.param(
            "https://mcp.pirateweather.net/mcp",
            "https",
            ("mcp.pirateweather.net", 443),
            b"mcp.pirateweather.net",
            id="configured",
        ),
        pytest.param(
            None, "http", ("127.0.0.1", 8084), b"127.0.0.1:8084", id="unconfigured"
        ),
    ],
)
def test_public_url_middleware_rewrites_origin(
    mcp_module, public_url, expected_scheme, expected_server, expected_host
):
    """Behind the reverse proxy the MCP server sees 127.0.0.1; without rewriting
    the origin, the URLs it advertises to clients point at localhost. When no
    public URL is configured the request must pass through untouched.
    """
    captured_scope = {}

    async def app(scope, receive, send):
        captured_scope.update(scope)

    args = (app,) if public_url is None else (app, public_url)
    middleware = mcp_module.PublicUrlMiddleware(*args)
    scope = {
        "type": "http",
        "scheme": "http",
        "server": ("127.0.0.1", 8084),
        "headers": [(b"host", b"127.0.0.1:8084"), (b"accept", b"*/*")],
    }

    asyncio.run(middleware(scope, None, None))

    assert captured_scope["scheme"] == expected_scheme
    assert captured_scope["server"] == expected_server
    assert set(captured_scope["headers"]) == {
        (b"host", expected_host),
        (b"accept", b"*/*"),
    }


def test_historical_weather_omits_false_tmextra(mcp_module, monkeypatch):
    """The API treats any tmextra value as true (bool("False")), so the MCP tool
    must omit the parameter for False rather than forward it.
    """
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


def test_get_forecast_returns_near_term_forecast_alerts_and_summary_text(
    mcp_module, monkeypatch
):
    """get_forecast trims the API response to what an LLM needs (current, next
    two hours/days, alerts, summaries); a change to the request filters or the
    reshaping silently changes what assistants tell users.
    """
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

    assert "version" not in captured
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


def test_city_country_location_is_forwarded_to_api(mcp_module, monkeypatch):
    """City,Country requests must reach the API URL-encoded in the location path
    segment with version=2 pinned, or the API cannot geocode them.
    """
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("PW_MCP_BASE_URL", "http://api.local")
    monkeypatch.setattr(mcp_module, "urlopen", fake_urlopen)

    assert mcp_module._request_forecast(
        city="New York",
        country="US",
    ) == {"ok": True}

    assert captured["url"] == (
        "http://api.local/forecast/mcp-proxy/New%20York,US?version=2"
    )


def test_public_tool_signatures_do_not_expose_version(mcp_module):
    """API contract (#641): the MCP server pins version=2 itself; exposing a
    version argument on any tool would let models request the v1 schema.
    """
    tools = _FakeFastMCP.tools

    assert "get_forecast" in {tool.__name__ for tool in tools}
    assert [
        tool.__name__
        for tool in tools
        if "version" in inspect.signature(tool).parameters
    ] == []
