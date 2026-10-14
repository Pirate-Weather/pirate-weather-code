"""Regression tests for URMA as a recent Time Machine source."""

import datetime
import logging

import numpy as np
import pytest

from API.constants.forecast_const import DATA_HOURLY
from API.constants.model_const import GFS, URMA
from API.current.metrics import (
    InterpolationState,
    _get_feels_like,
    _get_humidity,
    _get_pressure,
    _get_station_pressure,
    _get_temp,
    _urma_with_gfs_fallback,
)
from API.data_inputs import prepare_data_inputs
from API.forecast_sources import SourceMetadata, _merge_urma_source, merge_hourly_models
from API.hourly.block import _calculate_derived_metrics
from API.io.zarr_reader import WeatherParallel
from API.request.grid_indexing import ZarrSources, calculate_grid_indexing


def _model_data(hours: int = 3) -> tuple[np.ndarray, np.ndarray]:
    urma = np.full((hours, max(URMA.values()) + 1), np.nan)
    gfs = np.full((hours, max(GFS.values()) + 1), np.nan)
    urma[:, URMA["temp"]] = 15
    urma[:, URMA["humidity"]] = 0.6
    urma[:, URMA["wind_u"]] = 3
    urma[:, URMA["wind_v"]] = 4
    urma[:, URMA["pressure"]] = 90000
    gfs[:, GFS["temp"]] = 10
    gfs[:, GFS["apparent"]] = 9
    gfs[:, GFS["humidity"]] = 40
    gfs[:, GFS["pressure"]] = 101000
    gfs[:, GFS["station_pressure"]] = 91000
    gfs[:, GFS["wind_u"]] = 6
    gfs[:, GFS["wind_v"]] = 8
    return urma, gfs


def test_urma_merge_aligns_actual_hours_and_reports_only_contributing_data():
    base = datetime.datetime(2026, 9, 20, tzinfo=datetime.UTC).timestamp()
    raw, _ = _model_data(4)
    raw[:, 0] = [base - 3600, base + 60, base + 2 * 3600 + 64, base + 7 * 3600]
    raw[1, URMA["temp"]] = np.nan
    merged, latest = _merge_urma_source(raw, base, 4, base - 1)

    assert merged is not None
    assert latest == base + 2 * 3600
    assert np.isnan(merged[0, URMA["temp"]])
    assert np.isnan(merged[1, URMA["temp"]])
    assert merged[2, URMA["temp"]] == 15
    assert np.isnan(merged[3, URMA["temp"]])

    empty, empty_time = _merge_urma_source(raw, base, 4, base + 3 * 3600)
    assert empty is None and empty_time is None


def test_urma_merge_adds_source_flag_only_for_valid_requested_hours():
    base = datetime.datetime(2026, 9, 20, tzinfo=datetime.UTC).timestamp()
    raw, _ = _model_data(1)
    raw[0, 0] = base
    args = {
        "metadata": SourceMetadata([], {}, {}),
        "num_hours": 2,
        "base_day_utc_grib": base,
        "data_hrrrh": None,
        "data_h2": None,
        "data_nbm": None,
        "data_gfs": None,
        "data_ecmwf": None,
        "data_gefs": None,
        "data_hrdps": None,
        "data_gdps": None,
        "data_geps": None,
        "data_reps": None,
        "data_dwd_mosmix": None,
        "data_aigfs": None,
        "data_aigefs": None,
        "data_aifs": None,
        "data_urma": raw,
        "urma_min_timestamp": base - 3600,
        "logger": logging.getLogger(__name__),
        "loc_tag": "test",
    }
    result = merge_hourly_models(**args)
    assert "urma" in result.metadata.source_list
    assert result.metadata.source_times["urma"] == "2026-09-20 00Z"

    args["metadata"] = SourceMetadata([], {}, {})
    args["urma_min_timestamp"] = base + 3600
    result = merge_hourly_models(**args)
    assert result.urma is None
    assert "urma" not in result.metadata.source_list


def test_hourly_inputs_prefer_urma_and_preserve_pressure_meaning():
    urma, gfs = _model_data()
    urma[1, URMA["temp"]] = np.nan
    urma[2, URMA["pressure"]] = np.nan
    gfs[:, GFS["intensity"]] = 0.001
    inputs = prepare_data_inputs(
        source_list=["urma", "gfs"],
        nbm_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=None,
        ecmwf_merged=None,
        gefs_merged=None,
        gfs_merged=gfs,
        era5_merged=None,
        extra_vars=["stationPressure"],
        num_hours=3,
        lat=40,
        lon=-105,
        urma_merged=urma,
    )

    np.testing.assert_array_equal(inputs["temperature_inputs"][:, 0], [15, np.nan, 15])
    np.testing.assert_array_equal(inputs["temperature_inputs"][:, 1], [10, 10, 10])
    np.testing.assert_array_equal(inputs["humidity_inputs"][:, 0], [60, 60, 60])
    np.testing.assert_array_equal(inputs["pressure_inputs"][:, 0], [101000] * 3)
    np.testing.assert_array_equal(
        inputs["station_pressure_inputs"][:, 0], [90000, 90000, np.nan]
    )
    np.testing.assert_array_equal(inputs["station_pressure_inputs"][:, 1], [91000] * 3)
    np.testing.assert_array_equal(inputs["prcipIntensity_inputs"][:, 0], [3.6] * 3)


def test_current_uses_urma_with_per_hour_gfs_fallback():
    urma, gfs = _model_data(2)
    urma[1, URMA["temp"]] = np.nan
    urma[1, URMA["humidity"]] = np.nan
    combined = _urma_with_gfs_fallback(urma, gfs)
    state = InterpolationState(idx1=0, idx2=1, fac1=0.5, fac2=0.5)
    model_data = {"URMA_Merged": combined, "GFS_Merged": gfs, "has_hrrr_merged": False}
    sources = ["urma", "gfs"]

    assert _get_temp(sources, model_data, state, 40, -105) == pytest.approx(12.5)
    assert _get_humidity(sources, model_data, state, 0.01, 40, -105) == pytest.approx(
        0.5
    )
    assert _get_pressure(sources, model_data, state, 40, -105) == 101000
    assert _get_station_pressure(sources, model_data, state) == 90000

    model_data["urma_active_current"] = True
    assert _get_feels_like(sources, model_data, state, True, 13) == 13
    model_data["urma_active_current"] = False
    assert _get_feels_like(sources, model_data, state, True, 13) == 9


def test_hourly_feels_like_uses_selected_conditions_on_urma_hours():
    hourly = np.zeros((2, max(DATA_HOURLY.values()) + 1))
    hourly[:, DATA_HOURLY["temp"]] = 20
    hourly[:, DATA_HOURLY["humidity"]] = 0.5
    hourly[:, DATA_HOURLY["wind"]] = 2
    hourly[:, DATA_HOURLY["solar"]] = 100
    hourly[:, DATA_HOURLY["feels_like"]] = 25

    _calculate_derived_metrics(
        hourly, np.array([0, 0]), 0, True, urma_hour_mask=np.array([True, False])
    )

    assert hourly[0, DATA_HOURLY["feels_like"]] == pytest.approx(
        hourly[0, DATA_HOURLY["apparent"]]
    )
    assert hourly[1, DATA_HOURLY["feels_like"]] == 25


class _Weather:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def zarr_read(self, model, opened_zarr, x, y):
        self.calls.append((model, x, y))
        return self.data


@pytest.mark.asyncio
async def test_urma_reader_keeps_missing_observations_for_gfs_fallback():
    store = np.ones((max(URMA.values()) + 1, 3, 1, 1))
    store[URMA["temp"], 1, 0, 0] = np.nan
    weather = WeatherParallel()

    urma = await weather.zarr_read("URMA", store, 0, 0)
    assert np.isnan(urma[1, URMA["temp"]])

    gfs = await weather.zarr_read("GFS", store, 0, 0)
    assert np.isfinite(gfs[1, URMA["temp"]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lat", "lon", "time_machine", "excluded", "age_days", "read_urma"),
    [
        (40, -105, True, 0, 2, True),
        (40, -105, True, 1, 2, False),
        (40, -105, False, 0, 2, False),
        (0, 0, True, 0, 2, False),
        (40, -105, True, 0, 20, False),
        (40, -105, True, 0, -2, False),
    ],
)
async def test_urma_grid_read_requires_recent_history_inside_domain(
    lat, lon, time_machine, excluded, age_days, read_urma, monkeypatch
):
    monkeypatch.setattr(
        "API.request.grid_indexing._load_era5_slice", lambda *args, **kwargs: False
    )
    now = datetime.datetime(2026, 9, 21)
    raw, _ = _model_data(1)
    raw[0, 0] = (
        (now - datetime.timedelta(days=1)).replace(tzinfo=datetime.UTC).timestamp()
    )
    weather = _Weather(raw)
    sources = ZarrSources(
        subh=None,
        hrrr_6h=None,
        hrrr=None,
        nbm=None,
        gfs=None,
        ecmwf=None,
        gefs=None,
        urma=object(),
    )
    result = await calculate_grid_indexing(
        lat=lat,
        lon=lon % 360,
        az_lon=lon,
        utc_time=now - datetime.timedelta(days=age_days),
        now_time=now,
        time_machine=time_machine,
        ex_hrrr=1,
        ex_nbm=1,
        ex_gfs=1,
        ex_urma=excluded,
        ex_ecmwf=1,
        ex_gefs=1,
        ex_rtma_ru=1,
        ex_dwd_mosmix=1,
        ex_aigfs=1,
        ex_aigefs=1,
        ex_aifs=1,
        read_wmo_alerts=False,
        base_day_utc=now - datetime.timedelta(days=age_days),
        num_hours=24,
        zarr_sources=sources,
        weather=weather,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(result.dataOut_urma, np.ndarray) is read_urma
    assert bool(weather.calls) is read_urma
