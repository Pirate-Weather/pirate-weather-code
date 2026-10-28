"""Regression tests for WMO alert reads when the WMO Alerts store is absent."""

import asyncio
import datetime
import logging

import pytest

from API.request.grid_indexing import ZarrSources, calculate_grid_indexing


class _Weather:
    async def zarr_read(self, model, opened_zarr, x, y):
        return False


class _AlertStore:
    def __getitem__(self, key):
        return "Test Alert"


def _grid_indexing(sources: ZarrSources, monkeypatch):
    monkeypatch.setattr(
        "API.request.grid_indexing._load_era5_slice", lambda *args, **kwargs: False
    )
    now = datetime.datetime(2026, 9, 21)
    return asyncio.run(
        calculate_grid_indexing(
            lat=51.5,
            lon=359.9,
            az_lon=-0.1,
            utc_time=now,
            now_time=now,
            time_machine=False,
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
            read_wmo_alerts=True,
            base_day_utc=now.replace(tzinfo=datetime.UTC),
            num_hours=24,
            zarr_sources=sources,
            weather=_Weather(),
            logger=logging.getLogger(__name__),
        )
    )


@pytest.mark.parametrize(
    ("wmo_alerts", "expected"),
    [(None, None), (_AlertStore(), "Test Alert")],
)
def test_wmo_alerts_read_only_when_store_loaded(wmo_alerts, expected, monkeypatch):
    sources = ZarrSources(
        subh=None,
        hrrr_6h=None,
        hrrr=None,
        nbm=None,
        gfs=None,
        ecmwf=None,
        gefs=None,
        wmo_alerts=wmo_alerts,
    )

    result = _grid_indexing(sources, monkeypatch)

    assert result.WMO_alertDat == expected
