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
