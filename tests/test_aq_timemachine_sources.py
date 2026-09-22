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
    calculate_grid_indexing,
)


class _AQWeather:
    def __init__(self, data: dict[str, np.ndarray]):
        self.data = data

    async def zarr_read_max_square(self, model, _store, _x, _y):
        return self.data[model]


@pytest.mark.parametrize(
    (
        "age_days",
        "time_machine",
        "model_shift_days",
        "missing_source",
        "expected_sources",
    ),
    [
        (30, True, 0, None, set()),
        (1, True, 0, None, {"raqdps", "silam"}),
        (0, False, 0, None, {"raqdps", "silam"}),
        (0, False, 10, None, set()),
        (0, False, 0, "raqdps", {"silam"}),
        (0, False, 0, "silam", {"raqdps"}),
    ],
)
def test_aq_sources_reported_only_when_request_has_coverage(
    age_days,
    time_machine,
    model_shift_days,
    missing_source,
    expected_sources,
    monkeypatch,
):
    monkeypatch.setattr(
        "API.request.grid_indexing._load_era5_slice", lambda *args, **kwargs: False
    )
    now = datetime.datetime(2026, 9, 22, 12)
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
            "latitude": np.array([[0.0]]),
            "longitude": np.array([[0.0]]),
        },
    )
    requested = now - datetime.timedelta(days=age_days)
    result = asyncio.run(
        calculate_grid_indexing(
            lat=0.0,
            lon=0.0,
            az_lon=0.0,
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
            num_hours=24,
            zarr_sources=sources,
            weather=_AQWeather({"RAQDPS": raqdps, "SILAM": silam}),
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


def test_aq_source_coverage_matches_float32_times_at_half_hour_offset():
    request_time = datetime.datetime(2026, 9, 22, 11, 30, tzinfo=datetime.UTC)
    model_time = np.float32(
        datetime.datetime(2026, 9, 22, 12, tzinfo=datetime.UTC).timestamp()
    )
    data = np.array([[model_time, 10.0]])

    assert _aq_source_covers_request(data, request_time, 1, (1,))
    data[0, 1] = np.nan
    assert not _aq_source_covers_request(data, request_time, 1, (1,))
