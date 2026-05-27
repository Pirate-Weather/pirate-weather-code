#!/usr/bin/env python3
"""FastMCP proxy for a local Pirate Weather API server.

Run the main API separately, for example:

    responseLocal:app --host 127.0.0.1 --port 8083

Then start this MCP server:

    PW_MCP_BASE_URL=http://127.0.0.1:8083 \
    PW_MCP_PUBLIC_URL=https://mcp.pirateweather.net/mcp \
    python -m MCP.pirate_weather_mcp --port 8084

Environment variables:

    PW_MCP_BASE_URL
        Internal base URL for the Pirate Weather API that this MCP proxy calls
        when a tool needs forecast data. In production this should usually stay
        on the private loopback interface, for example http://127.0.0.1:8083.
        This is not the public MCP URL clients connect to.

    PW_MCP_HOST
        Interface the MCP HTTP server binds to when run through this module's
        main() entrypoint. The default is 127.0.0.1 so a reverse proxy can
        expose it safely. Use 0.0.0.0 only when the container or host network
        boundary is already controlling access.

    PW_MCP_PORT
        Port the MCP HTTP server listens on when run through this module's
        main() entrypoint. If nginx/Caddy forwards public MCP traffic to
        127.0.0.1:8084, set this to 8084 or pass --port 8084.

    PW_MCP_PATH
        HTTP path mounted by FastMCP for streamable HTTP traffic. The default
        is /mcp. The reverse proxy should forward the same path unless it is
        intentionally rewriting paths.

    PW_MCP_PUBLIC_URL
        Absolute external URL for the MCP endpoint, for example
        https://mcp.pirateweather.net/mcp. Set this when the MCP server is
        behind a reverse proxy so generated redirects/protocol URLs do not
        fall back to the internal upstream such as http://127.0.0.1:8084/mcp.

    PW_MCP_FORWARDED_ALLOW_IPS
        Comma-separated proxy IPs whose X-Forwarded-* headers Uvicorn should
        trust when run through this module's main() entrypoint. The default is
        127.0.0.1, matching a local reverse proxy. Use * only if the server is
        protected from direct untrusted traffic.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from typing import Annotated, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import Field

from MCP import __version__
from MCP.resources import EXAMPLE_USAGE, FORECAST_BLOCK_METADATA

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - compatibility with the official MCP SDK
    from mcp.server.fastmcp import FastMCP


# Internal Pirate Weather API target used by MCP tools.
DEFAULT_BASE_URL = "http://127.0.0.1:8083"
DEFAULT_API_VERSION = 2
ROUTE_API_KEY = "mcp-proxy"
DEFAULT_TEST_LOCATION = (45.4215, -75.6972)

# MCP server listener defaults used by main().
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8000

# Public streamable HTTP endpoint defaults used by FastMCP.
DEFAULT_MCP_PATH = "/mcp"
DEFAULT_MCP_PUBLIC_URL = ""

UnitSystem = Literal["auto", "us", "si", "ca", "uk", "uk2"]
LanguageCode = Literal[
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "fa",
    "fi",
    "fr",
    "ga",
    "gd",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kn",
    "ko",
    "kw",
    "lv",
    "ml",
    "mr",
    "nl",
    "no",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "ta",
    "te",
    "tet",
    "tr",
    "uk",
    "ur",
    "vi",
    "x-pig-latin",
    "zh",
    "zh-tw",
]

Latitude = Annotated[
    float, Field(ge=-90, le=90, description="Latitude in decimal degrees.")
]
Longitude = Annotated[
    float,
    Field(ge=-180, le=360, description="Longitude in decimal degrees."),
]
Units = Annotated[
    UnitSystem | None,
    Field(
        description=(
            "Measurement system. Allowed values: auto (choose by country), us "
            "(F, mph, inches, miles), si (C, m/s, mm, km), ca (C, km/h, mm, km), "
            "uk/uk2 (C, mph, mm, miles). Defaults to us."
        ),
    ),
]
Language = Annotated[
    LanguageCode | None,
    Field(
        description=(
            "Text summary language code. Allowed values: ar, az, be, bg, bn, bs, ca, cs, "
            "cy, da, de, el, en, eo, es, et, fa, fi, fr, ga, gd, he, hi, hr, hu, "
            "id, is, it, ja, ka, kn, ko, kw, lv, ml, mr, nl, no, pa, pl, pt, ro, "
            "ru, sk, sl, sr, sv, ta, te, tet, tr, uk, ur, vi, x-pig-latin, zh, "
            "zh-tw. Defaults to en."
        ),
    ),
]
HistoricalTime = Annotated[
    str,
    Field(
        description=(
            "Requested time for a time-machine request. Use one of: UNIX timestamp "
            "seconds as a string, e.g. '1730869200'; ISO local time "
            "'YYYY-MM-DDTHH:MM:SS', interpreted at the requested coordinates; ISO "
            "with numeric UTC offset 'YYYY-MM-DDTHH:MM:SS+0000'; or a negative "
            "relative offset from now using s, h, or d, e.g. '-6h' or '-2d'. "
            "Future times more than one hour ahead are rejected."
        ),
    ),
]
TimeMachineExtra = Annotated[
    bool,
    Field(
        description=(
            "When true, include extra time-machine fields in the response where "
            "available. This is a boolean flag, not a numeric value, and has no units."
        ),
    ),
]
City = Annotated[str | None, Field(description="City name to geocode.")]
Country = Annotated[
    str | None,
    Field(description="Country name or code used with city geocoding."),
]


class PublicUrlMiddleware:
    """Force generated absolute URLs to use the public MCP origin when configured."""

    def __init__(self, app: Any, public_url: str | None = None) -> None:
        self.app = app
        self.public_url = _split_public_url(public_url)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        public_url = self.public_url
        if public_url is None:
            await self.app(scope, receive, send)
            return

        forwarded_scope = dict(scope)
        forwarded_scope["scheme"] = public_url.scheme
        forwarded_scope["server"] = (
            public_url.hostname,
            public_url.port or _default_port(public_url.scheme),
        )
        forwarded_scope["headers"] = _replace_host_header(
            scope.get("headers", []),
            public_url.netloc,
        )

        await self.app(forwarded_scope, receive, send)


def _split_public_url(public_url: str | None) -> SplitResult | None:
    if not public_url:
        return None

    parsed = urlsplit(public_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        msg = "PW_MCP_PUBLIC_URL must be an absolute http(s) URL, such as https://mcp.pirateweather.net/mcp"
        raise ValueError(msg)
    return parsed


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _replace_host_header(
    headers: list[tuple[bytes, bytes]], host: str
) -> list[tuple[bytes, bytes]]:
    host_bytes = host.encode("latin-1")
    next_headers = [(key, value) for key, value in headers if key.lower() != b"host"]
    next_headers.append((b"host", host_bytes))
    return next_headers


mcp = FastMCP("Pirate Weather", version=__version__)
app = mcp.http_app(
    path=os.environ.get("PW_MCP_PATH", DEFAULT_MCP_PATH),
    transport="streamable-http",
)
app = PublicUrlMiddleware(
    app,
    os.environ.get("PW_MCP_PUBLIC_URL", DEFAULT_MCP_PUBLIC_URL),
)


def _base_url() -> str:
    return (os.environ.get("PW_MCP_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _csv_indices(count: int, start: int = 0) -> str | None:
    if count <= 0:
        return None
    return ",".join(str(i) for i in range(start, start + count))


def _location(
    *,
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: str | None = None,
    country: str | None = None,
    time: str | int | None = None,
) -> str | None:
    if latitude is not None and longitude is not None:
        location = f"{latitude},{longitude}"
    elif city and country:
        location = f"{city},{country}"
    else:
        return None

    if time is not None:
        location = f"{location},{time}"
    return location


def _iso_from_unix_time(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None

    try:
        return datetime.datetime.fromtimestamp(value, datetime.UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _with_iso_times(value: Any) -> Any:
    if isinstance(value, list):
        return [_with_iso_times(item) for item in value]
    if not isinstance(value, dict):
        return value

    next_value: dict[str, Any] = {}
    for key, item in value.items():
        next_value[key] = _with_iso_times(item)
        if key == "time":
            time_iso = _iso_from_unix_time(item)
            if time_iso is not None:
                next_value["timeISO"] = time_iso
    return next_value


def _request_forecast(
    *,
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: str | None = None,
    country: str | None = None,
    time: str | int | None = None,
    timeout: float = 30.0,
    **params: Any,
) -> dict[str, Any]:
    location = _location(
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        time=time,
    )
    if location is None:
        return {
            "ok": False,
            "status": 400,
            "error": "Provide either latitude and longitude, or city and country.",
        }

    query = urlencode(
        _clean_params({"version": DEFAULT_API_VERSION, **params}), doseq=False
    )
    path_location = quote(location, safe=",")
    url = f"{_base_url()}/forecast/{ROUTE_API_KEY}/{path_location}"
    if query:
        url = f"{url}?{query}"

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return _with_iso_times(json.loads(body))
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
    return _with_iso_times(
        {
            "latitude": forecast.get("latitude"),
            "longitude": forecast.get("longitude"),
            "timezone": forecast.get("timezone"),
            "offset": forecast.get("offset"),
            block: forecast.get(block),
            "flags": forecast.get("flags"),
        }
    )


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
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
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
    )


@mcp.tool()
def get_hourly_forecast(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    hours: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Number of hourly entries to return. Values above 48 request the "
                "extended hourly forecast when the upstream API has it available."
            ),
        ),
    ] = 24,
    units: Units = None,
    lang: Language = None,
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
        extend="hourly" if hours > 48 else None,
        hourly_indices=_csv_indices(hours),
    )


@mcp.tool()
def get_minutely_forecast(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
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
    )


@mcp.tool()
def get_tomorrow_forecast(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
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
        daily_indices="1",
    )


@mcp.tool()
def get_daily_forecast(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    days: Annotated[
        int,
        Field(ge=1, le=7, description="Number of daily forecast entries to return."),
    ] = 7,
    units: Units = None,
    lang: Language = None,
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
        daily_indices=_csv_indices(days),
    )


@mcp.tool()
def get_alerts(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
) -> dict[str, Any]:
    """Return weather alerts for coordinates or a city/country pair."""
    return _forecast_block(
        "alerts",
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
    )


@mcp.tool()
def get_historical_weather(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    time: HistoricalTime | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
    tmextra: TimeMachineExtra = False,
) -> dict[str, Any]:
    """Return a time-machine weather response for a specific historical time."""
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
        tmextra=1 if tmextra else None,
    )


def _summary_text(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"summary": None, "icon": None}
    return {
        "summary": block.get("summary"),
        "icon": block.get("icon"),
    }


@mcp.tool()
def get_forecast(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
) -> dict[str, Any]:
    """Return current conditions, near-term forecasts, alerts, and summary text."""
    forecast = _request_forecast(
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
        blocks="currently,minutely,hourly,daily,alerts,flags",
        hourly_indices="0,1",
        daily_indices="0,1",
    )
    if forecast.get("ok") is False:
        return forecast

    currently = forecast.get("currently") or {}
    minutely = forecast.get("minutely") or {}
    hourly = forecast.get("hourly") or {}
    daily = forecast.get("daily") or {}
    hourly_data = hourly.get("data") if isinstance(hourly, dict) else None
    daily_data = daily.get("data") if isinstance(daily, dict) else None
    this_hour = hourly_data[0] if hourly_data else None
    next_hour = hourly_data[1] if hourly_data and len(hourly_data) > 1 else None
    today = daily_data[0] if daily_data else None
    tomorrow = daily_data[1] if daily_data and len(daily_data) > 1 else None

    return _with_iso_times(
        {
            "latitude": forecast.get("latitude"),
            "longitude": forecast.get("longitude"),
            "timezone": forecast.get("timezone"),
            "offset": forecast.get("offset"),
            "currently": currently,
            "this_hour": this_hour,
            "next_hour": next_hour,
            "today": today,
            "tomorrow": tomorrow,
            "alerts": forecast.get("alerts", []),
            "summary_text": {
                "currently": _summary_text(currently),
                "minutely": _summary_text(minutely),
                "hourly": _summary_text(hourly),
                "daily": _summary_text(daily),
            },
            "flags": forecast.get("flags"),
        }
    )


@mcp.tool()
def get_weather_summary(
    latitude: Latitude | None = None,
    longitude: Longitude | None = None,
    city: City = None,
    country: Country = None,
    units: Units = None,
    lang: Language = None,
) -> dict[str, Any]:
    """Return concise current, minutely, hourly, and daily weather summaries."""
    forecast = _request_forecast(
        latitude=latitude,
        longitude=longitude,
        city=city,
        country=country,
        units=units,
        lang=lang,
    )
    if forecast.get("ok") is False:
        return forecast

    currently = forecast.get("currently") or {}
    minutely = forecast.get("minutely") or {}
    hourly = forecast.get("hourly") or {}
    daily = forecast.get("daily") or {}
    return _with_iso_times(
        {
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
    )


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
    return _with_iso_times(
        {
            "ok": True,
            "base_url": _base_url(),
            "latitude": response.get("latitude"),
            "longitude": response.get("longitude"),
            "timezone": response.get("timezone"),
            "has_currently": "currently" in response,
        }
    )


@mcp.tool()
def get_subscription_status() -> dict[str, Any]:
    """Report local proxy status and API reachability.

    The local responseLocal API does not expose quota or subscription metadata.
    """
    return _with_iso_times(
        {
            "ok": True,
            "base_url": _base_url(),
            "route_api_key": ROUTE_API_KEY,
            "subscription_metadata_available": False,
            "message": "responseLocal does not expose subscription or quota metadata.",
            "connection": test_api_connection(),
        }
    )


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
    parser.add_argument(
        "--forwarded-allow-ips",
        default=os.environ.get("PW_MCP_FORWARDED_ALLOW_IPS", "127.0.0.1"),
        help="Comma-separated proxy IPs whose forwarded headers Uvicorn should trust.",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "MCP.pirate_weather_mcp:app",
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
