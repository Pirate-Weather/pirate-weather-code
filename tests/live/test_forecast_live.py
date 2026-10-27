"""Live end-to-end forecasts served from the production S3 Zarr stores.

These are the only tests that run a real forecast request through source
selection, Zarr reads, merging, text generation, and response formatting against
current model data. Values change every model run, so assertions are limited to
structure, physical plausibility, and which models each region is served from.
"""

import time
from urllib.parse import urlencode

import pytest


def _check_forecast_structure(data: dict) -> None:
    """Validate that the key forecast blocks contain realistic data.

    The checks below use the Pirate Weather OpenAPI spec as reference for the
    expected fields in each forecast block.
    """

    assert "currently" in data
    assert "minutely" in data
    assert "hourly" in data
    assert "daily" in data

    curr = data["currently"]
    assert isinstance(curr.get("summary"), str)
    assert curr.get("summary")
    assert isinstance(curr.get("icon"), str)
    assert curr.get("icon")
    assert isinstance(curr.get("temperature"), (int, float))
    assert -100 <= curr["temperature"] <= 150
    assert isinstance(curr.get("humidity"), (int, float))
    assert 0 <= curr["humidity"] <= 1
    assert isinstance(curr.get("pressure"), (int, float))
    assert 800 <= curr["pressure"] <= 1100
    assert isinstance(curr.get("windSpeed"), (int, float))
    assert curr["windSpeed"] >= 0
    assert isinstance(curr.get("windBearing"), int)
    assert 0 <= curr["windBearing"] <= 360
    assert isinstance(curr.get("time"), int)
    # The timestamp should be within three hours of "now"
    assert abs(curr["time"] - int(time.time())) < 3 * 3600

    minute = data["minutely"]
    assert isinstance(minute.get("summary"), str)
    minute_data = minute["data"]
    assert isinstance(minute_data, list)
    assert 0 < len(minute_data) <= 61
    first_minute = minute_data[0]
    assert isinstance(first_minute.get("time"), int)
    assert isinstance(first_minute.get("precipIntensity"), (int, float))
    assert isinstance(first_minute.get("precipProbability"), (int, float))
    assert 0 <= first_minute["precipProbability"] <= 1

    hourly = data["hourly"]
    assert isinstance(hourly.get("summary"), str)
    hour_data = hourly["data"]
    assert isinstance(hour_data, list)
    assert len(hour_data) >= 24
    first_hour = hour_data[0]
    assert isinstance(first_hour.get("time"), int)
    assert isinstance(first_hour.get("temperature"), (int, float))
    assert isinstance(first_hour.get("humidity"), (int, float))
    assert 0 <= first_hour["humidity"] <= 1
    assert isinstance(first_hour.get("windSpeed"), (int, float))

    daily = data["daily"]
    assert isinstance(daily.get("summary"), str)
    day_data = daily["data"]
    assert isinstance(day_data, list)
    assert len(day_data) >= 1
    first_day = day_data[0]
    assert isinstance(first_day.get("time"), int)
    assert isinstance(first_day.get("sunriseTime"), int)
    assert isinstance(first_day.get("sunsetTime"), int)
    assert isinstance(first_day.get("temperatureHigh"), (int, float))
    assert isinstance(first_day.get("temperatureLow"), (int, float))
    assert isinstance(first_day.get("humidity"), (int, float))
    assert 0 <= first_day["humidity"] <= 1


@pytest.mark.parametrize(
    ("lat", "lon", "expected_sources", "unexpected_sources"),
    [
        pytest.param(
            32.73,
            -117.192,
            {"hrrr_0-18", "nbm", "rtma_ru", "gfs"},
            {"hrdps"},
            id="conus",
        ),
        pytest.param(
            45.4215, -75.6972, {"hrdps", "rtma_ru", "nbm", "gfs"}, set(), id="canada"
        ),
        pytest.param(
            28.64,
            77.09,
            {"ecmwf_ifs", "gfs", "gefs", "dwd_mosmix"},
            {"hrrr_0-18", "nbm", "rtma_ru", "hrdps"},
            id="rest-of-world",
        ),
    ],
)
def test_forecast_is_plausible_and_uses_regional_models(
    live_client, pw_api_key, lat, lon, expected_sources, unexpected_sources
):
    """Protect the full request path and regional model coverage.

    Regional models (HRRR/NBM/RTMA in the US, HRDPS in Canada) only exist for some
    points; a domain or grid-indexing regression silently drops them and the
    forecast falls back to coarser global models without any error.
    """
    response = live_client.get(f"/forecast/{pw_api_key}/{lat},{lon}?version=2")
    assert response.status_code == 200

    data = response.json()
    assert data["latitude"] == pytest.approx(lat, abs=0.5)
    assert data["longitude"] == pytest.approx(lon, abs=0.5)
    _check_forecast_structure(data)

    sources = set(data["flags"]["sources"])
    assert expected_sources <= sources
    assert not (unexpected_sources & sources)


def test_blocks_and_indices_filter_the_response(live_client, pw_api_key):
    """Protect the MCP-facing ``blocks``/``*_indices`` contract end to end.

    The filters are applied after the full response is built, so a regression
    in either the parsing or the post-processing would change what clients get.
    """
    params = {
        "blocks": "currently,hourly,daily,day_night,flags",
        "hourly_indices": "0,1",
        "daily_indices": "0,1",
        "day_night_indices": "0,1",
    }
    response = live_client.get(f"/forecast/{pw_api_key}/45.0,-75.0?{urlencode(params)}")
    assert response.status_code == 200

    data = response.json()
    assert {"currently", "hourly", "daily", "day_night", "flags"} <= data.keys()
    assert "minutely" not in data
    assert "alerts" not in data
    for block in ("hourly", "daily", "day_night"):
        assert len(data[block]["data"]) == 2


def test_excluding_every_model_is_rejected(live_client, pw_api_key):
    """Excluding every forecast model must return 400, not an empty forecast (#570)."""
    exclude = "gfs,gefs,ecmwf_ifs,dwd_mosmix,gdps,geps,era5"
    response = live_client.get(f"/forecast/{pw_api_key}/28.64,77.09?exclude={exclude}")
    assert response.status_code == 400
