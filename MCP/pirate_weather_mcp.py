#!/usr/bin/env python3
"""FastMCP proxy for a local Pirate Weather responseLocal API server.

Run the main API separately, for example:

    responseLocal:app --host 127.0.0.1 --port 8083

Then start this MCP server:

    PW_MCP_BASE_URL=http://127.0.0.1:8083 python -m MCP.pirate_weather_mcp
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException

from API.utils.offline_geocode import geocode_city_country
from MCP import __version__
from MCP.resources import EXAMPLE_USAGE, FORECAST_BLOCK_METADATA

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - compatibility with the official MCP SDK
    from mcp.server.fastmcp import FastMCP


DEFAULT_BASE_URL = "http://127.0.0.1:8083"
ROUTE_API_KEY = "mcp-proxy"
DEFAULT_TEST_LOCATION = (45.4215, -75.6972)
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8000
DEFAULT_MCP_PATH = "/mcp"

mcp = FastMCP("Pirate Weather", version=__version__)
app = mcp.http_app(
    path=os.environ.get("PW_MCP_PATH", DEFAULT_MCP_PATH),
    transport="streamable-http",
)


def _base_url() -> str:
    return (os.environ.get("PW_MCP_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _csv_indices(count: int, start: int = 0) -> str | None:
    if count <= 0:
        return None
    return ",".join(str(i) for i in range(start, start + count))


def _resolve_location(
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
) -> tuple[float, float]:
    if latitude is not None and longitude is not None:
        return latitude, longitude
    if city and country:
        return geocode_city_country(city, country)
    raise HTTPException(
        status_code=400,
        detail="Provide either latitude and longitude, or city and country.",
    )


def _location(latitude: float, longitude: float, time: str | int | None = None) -> str:
    location = f"{latitude},{longitude}"
    if time is not None:
        location = f"{location},{time}"
    return location


def _request_forecast(
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    time: str | int | None = None,
    timeout: float = 30.0,
    **params: Any,
) -> dict[str, Any]:
    try:
        resolved_latitude, resolved_longitude = _resolve_location(
            latitude=latitude,
            longitude=longitude,
            city=city,
            country=country,
        )
    except HTTPException as exc:
        return {"ok": False, "status": exc.status_code, "error": exc.detail}

    query = urlencode(_clean_params(params), doseq=False)
    path_location = quote(
        _location(resolved_latitude, resolved_longitude, time),
        safe=",",
    )
    url = f"{_base_url()}/forecast/{ROUTE_API_KEY}/{path_location}"
    if query:
        url = f"{url}?{query}"

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail: Any = json.loads(body)
        except json.JSONDecodeError:
            detail = body
        return {"ok": False, "status": exc.code, "error": detail}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON response: {exc}"}
    except (OSError, URLError) as exc:
        return {"ok": False, "error": str(exc)}


def _forecast_block(block: str, **kwargs: Any) -> dict[str, Any]:
    forecast = _request_forecast(blocks=block, **kwargs)
    if forecast.get("ok") is False:
        return forecast
    return {
        "latitude": forecast.get("latitude"),
        "longitude": forecast.get("longitude"),
        "timezone": forecast.get("timezone"),
        "offset": forecast.get("offset"),
        block: forecast.get(block),
        "flags": forecast.get("flags"),
    }


@mcp.resource(
    "pirate-weather://metadata/forecast-blocks",
    name="forecast_blocks",
    title="Pirate Weather Forecast Block Metadata",
    description="Metadata for Pirate Weather response blocks and their matching MCP tools.",
    mime_type="application/json",
)
def forecast_block_metadata() -> str:
    """Return metadata describing forecast blocks exposed by the MCP tools."""
    return json.dumps(FORECAST_BLOCK_METADATA, indent=2)


@mcp.resource(
    "pirate-weather://examples/tool-usage",
    name="tool_usage_examples",
    title="Pirate Weather MCP Tool Usage Examples",
    description="Example MCP tool calls for common Pirate Weather workflows.",
    mime_type="application/json",
)
def tool_usage_examples() -> str:
    """Return example tool calls for common weather requests."""
    return json.dumps(EXAMPLE_USAGE, indent=2)


@mcp.tool()
def get_current_weather(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return the current weather block for coordinates or a city/country pair."""
    return _forecast_block(
        "currently",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
    )


@mcp.tool()
def get_hourly_forecast(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    hours: int = 24,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return the hourly forecast block, optionally limited to the first N hours."""
    return _forecast_block(
        "hourly",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
        extend="hourly" if hours > 48 else None,
        hourly_indices=_csv_indices(hours),
    )


@mcp.tool()
def get_minutely_forecast(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return the minute-by-minute precipitation forecast block."""
    return _forecast_block(
        "minutely",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
    )


@mcp.tool()
def get_tomorrow_forecast(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return tomorrow's daily forecast entry."""
    return _forecast_block(
        "daily",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
        daily_indices="1",
    )


@mcp.tool()
def get_daily_forecast(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    days: int = 7,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return the daily forecast block, optionally limited to the first N days."""
    return _forecast_block(
        "daily",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
        daily_indices=_csv_indices(days),
    )


@mcp.tool()
def get_alerts(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return weather alerts for a latitude and longitude."""
    return _forecast_block(
        "alerts",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
    )


@mcp.tool()
def get_historical_weather(
    latitude: float | None = None,
    longitude: float | None = None,
    time: str | None = None,
    city: str | None = None,
    country: str | None = None,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
    tmextra: int | None = None,
) -> dict[str, Any]:
    """Return a time-machine weather response for a UNIX timestamp, ISO date, or relative time."""
    if time is None:
        return {"ok": False, "status": 400, "error": "Provide a historical time."}
    return _request_forecast(
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        time=time,
        units=units,
        lang=lang,
        version=version,
        tmextra=tmextra,
    )


@mcp.tool()
def get_weather_summary(
    latitude: float | None = None,
    longitude: float | None = None,
    city: str | None = None,
    country: str | None = None,
    units: str | None = None,
    lang: str | None = None,
    version: int | None = 2,
) -> dict[str, Any]:
    """Return concise current, minutely, hourly, and daily weather summaries."""
    forecast = _request_forecast(
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        version=version,
    )
    if forecast.get("ok") is False:
        return forecast

    currently = forecast.get("currently") or {}
    minutely = forecast.get("minutely") or {}
    hourly = forecast.get("hourly") or {}
    daily = forecast.get("daily") or {}
    return {
        "latitude": forecast.get("latitude"),
        "longitude": forecast.get("longitude"),
        "timezone": forecast.get("timezone"),
        "current": {
            "time": currently.get("time"),
            "summary": currently.get("summary"),
            "icon": currently.get("icon"),
            "temperature": currently.get("temperature"),
            "apparentTemperature": currently.get("apparentTemperature"),
            "precipProbability": currently.get("precipProbability"),
        },
        "minutely": {
            "summary": minutely.get("summary"),
            "icon": minutely.get("icon"),
        },
        "hourly": {
            "summary": hourly.get("summary"),
            "icon": hourly.get("icon"),
        },
        "daily": {
            "summary": daily.get("summary"),
            "icon": daily.get("icon"),
        },
        "alerts": forecast.get("alerts", []),
    }


@mcp.tool()
def test_api_connection(
    latitude: float = DEFAULT_TEST_LOCATION[0],
    longitude: float = DEFAULT_TEST_LOCATION[1],
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check whether the configured Pirate Weather API server is reachable."""
    response = _request_forecast(
        latitude=latitude,
        longitude=longitude,
        blocks="currently",
        timeout=timeout,
    )
    if response.get("ok") is False:
        return {
            "ok": False,
            "base_url": _base_url(),
            "error": response.get("error"),
            "status": response.get("status"),
        }
    return {
        "ok": True,
        "base_url": _base_url(),
        "latitude": response.get("latitude"),
        "longitude": response.get("longitude"),
        "timezone": response.get("timezone"),
        "has_currently": "currently" in response,
    }


@mcp.tool()
def get_subscription_status() -> dict[str, Any]:
    """Report local proxy status and API reachability.

    The local responseLocal API does not expose quota or subscription metadata.
    """
    return {
        "ok": True,
        "base_url": _base_url(),
        "route_api_key": ROUTE_API_KEY,
        "subscription_metadata_available": False,
        "message": "responseLocal does not expose subscription or quota metadata.",
        "connection": test_api_connection(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Pirate Weather FastMCP proxy server with Uvicorn."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PW_MCP_HOST", DEFAULT_MCP_HOST),
        help="Host interface for the MCP HTTP server.",
    )
    parser.add_argument(
        "--port",
        default=int(os.environ.get("PW_MCP_PORT", str(DEFAULT_MCP_PORT))),
        type=int,
        help="Port for the MCP HTTP server.",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("MCP.pirate_weather_mcp:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
