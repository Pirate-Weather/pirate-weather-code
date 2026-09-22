"""Regression tests for air quality source availability in API responses."""

import asyncio
import datetime
import logging

import numpy as np
import pytest

from API.constants.model_const import RAQDPS, SILAM
from API.forecast_sources import build_source_metadata
from API.request.grid_indexing import (
    ZarrSources,
    _aq_source_covers_request,
    _mask_old_aq_values,
    calculate_grid_indexing,
)

NOW = datetime.datetime(2026, 9, 22, 12)
REPORTED_TIME = datetime.datetime.fromtimestamp(1789665910, datetime.UTC).replace(
    tzinfo=None
)


class _AQWeather:
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data

    async def zarr_read_max_square(self, model, _store, _x, _y):
        return self.data[model]


@pytest.mark.parametrize(
    (
        "requested",
        "time_machine",
        "model_shift_days",
        "num_hours",
        "missing_source",
        "expected_sources",
    ),
    [
        (NOW - datetime.timedelta(days=30), True, 0, 24, None, set()),
        (REPORTED_TIME, True, -3, 24, None, set()),
        (NOW - datetime.timedelta(days=3), True, -3, 96, None, {"raqdps", "silam"}),
        (NOW - datetime.timedelta(days=1), True, 0, 24, None, {"raqdps", "silam"}),
        (NOW, False, 0, 24, None, {"raqdps", "silam"}),
        (NOW, False, 10, 24, None, set()),
        (NOW, False, 0, 24, "raqdps", {"silam"}),
        (NOW, False, 0, 24, "silam", {"raqdps"}),
    ],
)
def test_aq_sources_reported_only_when_request_has_coverage(
    requested,
    time_machine,
    model_shift_days,
    num_hours,
    missing_source,
    expected_sources,
    monkeypatch,
):
    monkeypatch.setattr(
        "API.request.grid_indexing._load_era5_slice", lambda *args, **kwargs: False
    )
    now = NOW
    model_times = (now + datetime.timedelta(days=model_shift_days)).replace(
        tzinfo=datetime.UTC
    ).timestamp() + (np.arange(120) - 48) * 3600
    raqdps = np.full((120, max(RAQDPS.values()) + 1), np.nan)
    silam = np.full((120, max(SILAM.values()) + 1), np.nan)
    raqdps[:, RAQDPS["time"]] = model_times
    silam[:, SILAM["time"]] = model_times
    if missing_source != "raqdps":
        raqdps[:, RAQDPS["pm25"]] = 10.0
    if missing_source != "silam":
        silam[:, SILAM["pm25"]] = 20.0
    else:
        silam[:, SILAM["blh"]] = 1000.0
    sources = ZarrSources(
        subh=None,
        hrrr_6h=None,
        hrrr=None,
        nbm=None,
        gfs=None,
        ecmwf=None,
        gefs=None,
        raqdps=object(),
        silam=object(),
        raqdps_lat_lon={
            "latitude": np.array([[45.0]]),
            "longitude": np.array([[-75.0]]),
        },
    )
    weather = _AQWeather({"RAQDPS": raqdps, "SILAM": silam})
    result = asyncio.run(
        calculate_grid_indexing(
            lat=45.0,
            lon=285.0,
            az_lon=-75.0,
            utc_time=requested,
            now_time=now,
            time_machine=time_machine,
            ex_hrrr=1,
            ex_nbm=1,
            ex_gfs=1,
            ex_ecmwf=1,
            ex_gefs=1,
            ex_rtma_ru=1,
            ex_dwd_mosmix=1,
            ex_aigfs=1,
            ex_aigefs=1,
            ex_aifs=1,
            read_wmo_alerts=False,
            base_day_utc=requested.replace(tzinfo=datetime.UTC),
            num_hours=num_hours,
            zarr_sources=sources,
            weather=weather,
            logger=logging.getLogger(__name__),
        )
    )
    metadata = build_source_metadata(
        grid_result=result,
        era5_merged=False,
        use_etopo=False,
        time_machine=time_machine,
    )

    assert set(metadata.source_list) == expected_sources
    assert set(metadata.source_times) == expected_sources
    assert set(metadata.source_idx) == expected_sources
    assert isinstance(result.dataOut_raqdps, np.ndarray) is (
        "raqdps" in expected_sources
    )
    assert isinstance(result.dataOut_silam, np.ndarray) is ("silam" in expected_sources)
    if num_hours > 24:
        assert np.isnan(result.dataOut_silam[0, SILAM["pm25"]])
        assert np.isfinite(result.dataOut_silam[-1, SILAM["pm25"]])


def test_mask_old_aq_values_preserves_boundary_and_input():
    now = NOW.replace(tzinfo=datetime.UTC)
    times = [
        (now - datetime.timedelta(hours=hours)).timestamp() for hours in (49, 48, 47)
    ]
    data = np.column_stack([times, [10.0, 20.0, 30.0]])

    masked = _mask_old_aq_values(data, NOW, 48)

    assert np.isnan(masked[0, 1])
    np.testing.assert_array_equal(masked[1:, 1], [20.0, 30.0])
    np.testing.assert_array_equal(data[:, 1], [10.0, 20.0, 30.0])


def test_aq_source_coverage_matches_float32_times_at_half_hour_offset():
    request_time = datetime.datetime(2026, 9, 22, 11, 30, tzinfo=datetime.UTC)
    model_time = np.float32(
        datetime.datetime(2026, 9, 22, 12, tzinfo=datetime.UTC).timestamp()
    )
    data = np.array([[model_time, 10.0]])

    assert _aq_source_covers_request(data, request_time, 1, (1,))
    data[0, 1] = np.nan
    assert not _aq_source_covers_request(data, request_time, 1, (1,))
