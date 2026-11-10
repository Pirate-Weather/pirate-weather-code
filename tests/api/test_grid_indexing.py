"""Point lookup on model grids that are not a simple regular lat/lon index."""

import asyncio
import datetime
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts
import xarray as xr

from API.constants.model_const import ERA5
from API.request.grid_indexing import (
    HRDPS_ROTATED_GRID,
    REPS_ROTATED_GRID,
    RRFS_NA_ROTATED_GRID,
    ZarrSources,
    _load_era5_slice,
    _nearest_2d_grid_coords,
    _nearest_rotated_grid_coords,
    calculate_grid_indexing,
)

ERA5_BASE_DAY = datetime.datetime(2024, 1, 1)


@pytest.mark.parametrize(
    ("grid", "latitude", "longitude", "max_distance"),
    [
        pytest.param(HRDPS_ROTATED_GRID, 39.626032, -133.629520, 0.005, id="hrdps"),
        pytest.param(REPS_ROTATED_GRID, 2.778652, -118.381280, 0.020, id="reps"),
    ],
)
def test_rotated_cmc_grid_first_grib_point_maps_to_index_zero(
    grid, latitude, longitude, max_distance
):
    """The rotated-pole grid definitions must reproduce the GRIB file's first
    point at (0, 0); an off-by-one or wrong pole rotation would read every
    HRDPS/REPS forecast from the wrong cell.
    """
    x_index, y_index, grid_latitude, grid_longitude = _nearest_rotated_grid_coords(
        latitude, longitude, grid, max_distance=max_distance, model_name="CMC"
    )

    assert (x_index, y_index) == (0, 0)
    assert grid_latitude == pytest.approx(latitude, abs=1e-5)
    assert grid_longitude == pytest.approx(longitude, abs=1e-5)


# (wgrib2 1-based i, j) -> (longitude, latitude) that wgrib2 -ijlat reports for
# cells of the RRFS North American grid (rrfs.t12z.2dfld.13km.f001.na.grib2,
# 2026-11-03): the four corners, the rotated origin and two interior cells.
RRFS_NA_WGRIB2_CELLS = [
    ((1, 1), 202.621079, -1.557179),
    ((1127, 1), 291.348687, -1.526135),
    ((1, 683), 135.891368, 41.459025),
    ((1127, 683), 358.091675, 41.500447),
    ((564, 342), 246.952753, 54.999991),
    ((300, 500), 184.686314, 58.871458),
    ((900, 120), 282.625936, 21.638009),
]


@pytest.mark.parametrize(("ij", "longitude", "latitude"), RRFS_NA_WGRIB2_CELLS)
def test_rrfs_na_lookup_finds_the_cell_wgrib2_places_there(ij, longitude, latitude):
    """The API finds an RRFS North America cell with its own closed-form
    rotated-pole lookup, while the ingest stores the grid as wgrib2 writes it.
    A pole, longitude or row-order convention that differed would read a
    neighbouring or distant cell for every North American include=rrfs request
    with nothing failing, so pin the lookup to wgrib2's own cell coordinates
    across the grid (the ingest checks its grid against wgrib2 on every run).
    """
    longitude = (longitude + 180.0) % 360.0 - 180.0
    x_index, y_index, grid_latitude, grid_longitude = _nearest_rotated_grid_coords(
        latitude,
        longitude,
        RRFS_NA_ROTATED_GRID,
        max_distance=0.005,
        model_name="RRFS_NA",
    )

    assert (x_index + 1, y_index + 1) == ij
    assert grid_latitude == pytest.approx(latitude, abs=1e-3)
    assert grid_longitude == pytest.approx(longitude, abs=1e-3)


def test_rotated_cmc_grid_rejects_location_outside_domain():
    """A regional grid must refuse points outside its domain instead of silently
    returning the nearest edge cell (which would serve far-away weather).
    """
    with pytest.raises(ValueError, match="outside the HRDPS domain"):
        _nearest_rotated_grid_coords(
            0.0, 0.0, HRDPS_ROTATED_GRID, max_distance=0.005, model_name="HRDPS"
        )


@pytest.mark.parametrize(
    ("query", "latitudes", "longitudes", "expected"),
    [
        # -179.9 is 0.3 deg from 179.8 across the antimeridian, 9.9 deg from -170.
        pytest.param(
            (0.0, -179.9),
            [0.0, 0.0],
            [179.8, -170.0],
            (0.0, 179.8),
            id="antimeridian",
        ),
        # At 80N, 0.5 deg of longitude (~10 km) is nearer than 0.3 deg of
        # latitude (~33 km); a raw lat/lon distance would pick the wrong cell.
        pytest.param(
            (80.0, 0.0), [80.0, 79.7], [0.5, 0.0], (80.0, 0.5), id="high-latitude"
        ),
    ],
)
def test_raqdps_nearest_grid_uses_spherical_distance(
    query, latitudes, longitudes, expected
):
    """RAQDPS is a 2D curvilinear grid; nearest-cell selection must use
    great-circle distance so lookups near the antimeridian and poles are right.
    """
    lat_lon_grid = {
        "latitude": np.array([latitudes]),
        "longitude": np.array([longitudes]),
    }

    x_idx, y_idx, grid_lat, grid_lon = _nearest_2d_grid_coords(
        *query, lat_lon_grid, max_distance=0.020, model_name="RAQDPS"
    )

    assert (x_idx, y_idx) == (0, 0)
    assert (grid_lat, grid_lon) == expected


def _era5_data(hours, latitudes, longitudes, fill):
    """Build the ``era5_data`` dict that ``init_ERA5`` returns, for a tiny grid.

    Each ERA5 variable is filled with its column index unless ``fill`` overrides it.
    """
    times = np.datetime64("2024-01-01T00:00:00") + np.arange(hours) * np.timedelta64(
        1, "h"
    )
    shape = (hours, len(latitudes), len(longitudes))
    data_vars = {}
    for name, index in ERA5.items():
        if name == "prob":  # derived by _load_era5_slice, not stored in ERA5
            continue
        values = fill[name] if name in fill else np.full(shape, float(index))
        data_vars[name] = (("time", "latitude", "longitude"), values)
    latitudes, longitudes = np.array(latitudes), np.array(longitudes)
    return {
        "ERA5_lats": latitudes,
        "ERA5_lons": longitudes,
        "ERA5_times": times,
        "dsERA5": xr.Dataset(
            data_vars=data_vars,
            coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        ),
    }


def test_era5_slice_length_timestamps_and_ptype_rounding():
    """The ERA5 fallback must return exactly num_hours rows stamped with unix
    seconds, and round interpolated precip type so it maps to a category.
    """
    era5_data = _era5_data(
        60, [40.0], [286.0], {"precipitation_type": np.full((60, 1, 1), 4.6)}
    )

    result = _load_era5_slice(
        era5_data, lat=40.0, lon=286.0, base_day_utc=ERA5_BASE_DAY, num_hours=49
    )

    start = int(ERA5_BASE_DAY.replace(tzinfo=datetime.UTC).timestamp())
    assert result.shape[0] == 49
    np.testing.assert_array_equal(result[:, 0], start + 3600 * np.arange(49))
    np.testing.assert_array_equal(result[:, ERA5["precipitation_type"]], 5)


def test_era5_precip_probability_from_neighbourhood_wraps_longitude():
    """ERA5 has no precip probability, so it is estimated from the 3x3
    neighbourhood; at lon 0 the window must wrap to lon 270 (not 180), and
    missing cells must be left out of the denominator.
    """
    lats, lons = [40.0, 39.75], [0.0, 90.0, 180.0, 270.0]
    precip = np.zeros((3, 2, 4))
    neighbourhood = [3, 0, 1]  # lon 270, 0, 90
    precip[0][:, neighbourhood] = 0.0002  # every cell wet -> 100%
    precip[1, 0, neighbourhood] = 0.0002  # 3 wet of 5 valid -> 60%
    precip[1, 1, 3] = np.nan
    precip[2][:, neighbourhood] = np.nan  # no valid cells -> 0%, not NaN
    era5_data = _era5_data(3, lats, lons, {"total_precipitation": precip})

    result = _load_era5_slice(
        era5_data, lat=40.0, lon=0.0, base_day_utc=ERA5_BASE_DAY, num_hours=3
    )

    np.testing.assert_allclose(result[:, ERA5["prob"]], [100.0, 60.0, 0.0])


def test_forecast_grid_read_without_published_wmo_alert_store():
    """A new deployment serves requests before every ingest has published, and
    the alert stores are optional: a missing WMO alert store must mean no WMO
    alerts, not a TypeError that turns every forecast request into HTTP 500.
    """
    now = datetime.datetime(2026, 10, 30, 19)
    sources = ZarrSources(
        subh=None, hrrr_6h=None, hrrr=None, nbm=None, gfs=None, ecmwf=None, gefs=None
    )

    result = asyncio.run(
        calculate_grid_indexing(
            lat=52.52,
            lon=13.40,
            az_lon=13.40,
            utc_time=now,
            now_time=now,
            time_machine=False,
            ex_hrrr=1,
            ex_nbm=1,
            ex_gfs=1,
            ex_urma=1,
            ex_ecmwf=1,
            ex_gefs=1,
            ex_rtma_ru=1,
            ex_dwd_mosmix=1,
            ex_aigfs=1,
            ex_aigefs=1,
            ex_aifs=1,
            read_wmo_alerts=True,
            base_day_utc=now,
            num_hours=24,
            zarr_sources=sources,
            weather=None,
            logger=logging.getLogger(__name__),
        )
    )

    assert result.WMO_alertDat is None


@pytest.mark.parametrize(
    ("ecmwf_y_size", "expected_read"),
    [
        pytest.param(1803, {"ECMWF": (1934, 375), "ECMWF_AIFS": (774, 150)}, id="0p1"),
        pytest.param(723, {"ECMWF_AIFS": (774, 150)}, id="stale-0p25-store"),
    ],
)
def test_ecmwf_and_aifs_grids_are_looked_up_separately(ecmwf_y_size, expected_read):
    """IFS moved to a 0.1 degree grid while AIFS stays at 0.25 degrees. Sharing one
    index would read AIFS from the wrong place (or past its edge), and a 0.25
    degree IFS store still published during the switch must be skipped rather than
    read at 0.1 degree indices, which would serve weather from another location.
    """
    reads = {}

    async def zarr_read(model, _store, x, y):
        reads[model] = (int(x), int(y))
        return False

    now = datetime.datetime(2026, 11, 4, 12)
    sources = ZarrSources(
        subh=None,
        hrrr_6h=None,
        hrrr=None,
        nbm=None,
        gfs=None,
        ecmwf=SimpleNamespace(shape=(13, 289, ecmwf_y_size, 3600)),
        gefs=None,
        ecmwf_aifs=SimpleNamespace(shape=(10, 235, 723, 1440)),
    )
    asyncio.run(
        calculate_grid_indexing(
            lat=52.5,
            lon=13.4,
            az_lon=13.4,
            utc_time=now,
            now_time=now,
            time_machine=False,
            ex_hrrr=1,
            ex_nbm=1,
            ex_gfs=1,
            ex_urma=1,
            ex_ecmwf=0,
            ex_gefs=1,
            ex_rtma_ru=1,
            ex_dwd_mosmix=1,
            ex_aigfs=1,
            ex_aigefs=1,
            ex_aifs=0,
            inc_aimodels=1,
            read_wmo_alerts=False,
            base_day_utc=now,
            num_hours=24,
            zarr_sources=sources,
            weather=SimpleNamespace(zarr_read=zarr_read),
            logger=logging.getLogger(__name__),
        )
    )

    assert reads == expected_read


@pytest.mark.parametrize(
    ("age_days", "archive", "expected"),
    [
        pytest.param(3, True, {"ERA5", "GFS"}, id="recent-with-archive"),
        pytest.param(3, False, {"GFS"}, id="recent-google-only"),
        pytest.param(11, True, {"ERA5", "GFS"}, id="inside-gfs-history"),
        pytest.param(20, True, {"ERA5"}, id="older-than-gfs-history"),
        pytest.param(20, False, {"ERA5"}, id="older-google-only"),
    ],
)
def test_time_machine_reads_era5_and_gfs_where_each_can_contribute(
    monkeypatch, age_days, archive, expected
):
    """Protect time-machine source selection: with the own ERA5 archive, ERA5 is
    read for every time-machine request (it wins where it has data) together with
    GFS while the request overlaps GFS's ~12-day history, which fills the hours
    ERA5 hasn't reached. Without the archive, ERA5 would come from Google at
    seconds per read, so recent requests keep using GFS only. Reading neither, or
    only the source that lacks the hours, leaves recent days empty.
    """
    from API.request import grid_indexing

    reads = set()

    async def zarr_read(model, _store, x, y):
        reads.add(model)
        return False

    def load_era5_slice(*_args, **_kwargs):
        reads.add("ERA5")
        return np.full((24, max(ERA5.values()) + 1), np.nan)

    monkeypatch.setattr(grid_indexing, "_load_era5_slice", load_era5_slice)
    now = datetime.datetime(2026, 11, 6, 12)
    requested = now - datetime.timedelta(days=age_days)
    era5_data = {"ERA5_archive": {} if archive else None}
    sources = ZarrSources(
        subh=None,
        hrrr_6h=None,
        hrrr=None,
        nbm=None,
        gfs=SimpleNamespace(shape=(20, 500, 721, 1440)),
        ecmwf=None,
        gefs=None,
        era5_data=era5_data,
    )
    asyncio.run(
        calculate_grid_indexing(
            lat=52.5,
            lon=13.4,
            az_lon=13.4,
            utc_time=requested,
            now_time=now,
            time_machine=True,
            ex_hrrr=1,
            ex_nbm=1,
            ex_gfs=0,
            ex_urma=1,
            ex_ecmwf=1,
            ex_gefs=1,
            ex_rtma_ru=1,
            ex_dwd_mosmix=1,
            ex_aigfs=1,
            ex_aigefs=1,
            ex_aifs=1,
            read_wmo_alerts=False,
            base_day_utc=requested,
            num_hours=24,
            zarr_sources=sources,
            weather=SimpleNamespace(zarr_read=zarr_read),
            logger=logging.getLogger(__name__),
        )
    )

    assert reads == expected


def test_future_request_keeps_the_current_gfs_run():
    """#689: run ages were measured from the requested time, so for a day five
    days ahead even a 6 h old GFS run looked over five days stale and every
    forecast model was dropped (HTTP 400, no sources). For a future time the
    newest run is the forecast to serve, so its age is measured from now.
    """
    from API.constants.shared_const import HISTORY_PERIODS

    now = datetime.datetime(2026, 11, 9, 12)
    requested = now + datetime.timedelta(days=5)
    run_time = now - datetime.timedelta(hours=6)
    gfs_rows = np.full((HISTORY_PERIODS["GFS"] + 240, 20), np.nan)
    gfs_rows[HISTORY_PERIODS["GFS"] - 1, 0] = run_time.replace(
        tzinfo=datetime.UTC
    ).timestamp()

    async def zarr_read(model, _store, x, y):
        return gfs_rows if model == "GFS" else False

    sources = ZarrSources(
        subh=None,
        hrrr_6h=None,
        hrrr=None,
        nbm=None,
        gfs=SimpleNamespace(shape=(20, 528, 721, 1440)),
        ecmwf=None,
        gefs=None,
    )
    result = asyncio.run(
        calculate_grid_indexing(
            lat=52.5,
            lon=13.4,
            az_lon=13.4,
            utc_time=requested,
            now_time=now,
            time_machine=False,
            ex_hrrr=1,
            ex_nbm=1,
            ex_gfs=0,
            ex_urma=1,
            ex_ecmwf=1,
            ex_gefs=1,
            ex_rtma_ru=1,
            ex_dwd_mosmix=1,
            ex_aigfs=1,
            ex_aigefs=1,
            ex_aifs=1,
            read_wmo_alerts=False,
            base_day_utc=requested,
            num_hours=25,
            zarr_sources=sources,
            weather=SimpleNamespace(zarr_read=zarr_read),
            logger=logging.getLogger(__name__),
        )
    )

    assert result.dataOut_gfs is gfs_rows


def test_far_future_day_is_served_from_cfs_and_only_then_read():
    """Past the GFS store's last hour a future day (#689) comes from CFS: the
    day's window is read at the nearest Gaussian cell (longitudes near 360 wrap
    to column 0), its rows land on the day's hours, and ``cfs`` is listed as a
    source and fills each element last. CFS is time-chunked and months long, so
    a forecast request or a future day GFS still covers must not read it at
    all; that would add a read to every request.
    """
    from API.constants.model_const import CFS, GFS
    from API.constants.shared_const import HISTORY_PERIODS
    from API.forecast_sources import build_source_metadata, merge_hourly_models
    from API.io.store_reader import ModelStore, WeatherParallel
    from tests.helpers.model_arrays import hourly_inputs

    now = datetime.datetime(2026, 11, 10, 12)
    run_time = now.replace(hour=0, tzinfo=datetime.UTC).timestamp()
    gfs_rows = np.full((HISTORY_PERIODS["GFS"] + 240, max(GFS.values()) + 1), np.nan)
    gfs_rows[HISTORY_PERIODS["GFS"] - 1, 0] = run_time
    gfs_store = SimpleNamespace(
        shape=(20, 528, 721, 1440), times=run_time + 3600 * np.arange(241.0)
    )

    # Hourly from run + 6 h for 70 days; only the cell read below holds data.
    # Stores hold no time column: API column c is store variable c - 1.
    cfs_times = run_time + 3600 * (6 + np.arange(70 * 24.0))
    cell_y = 31  # Gaussian row nearest 60 S
    values = np.full((max(CFS.values()), len(cfs_times), cell_y + 1, 1), np.nan)
    hour_of_store = np.arange(len(cfs_times))
    values[CFS["temp"] - 1, :, cell_y, 0] = 250 + hour_of_store % 24
    values[CFS["intensity"] - 1, :, cell_y, 0] = 0.5
    values[CFS["prob"] - 1, :, cell_y, 0] = 0.25
    cfs_store = ModelStore(
        array=ts.array(values.astype(np.float32)), times=cfs_times, version="test"
    )

    class _Weather(WeatherParallel):
        def __init__(self):
            super().__init__()
            self.window_reads = []

        async def zarr_read(self, model, _store, x, y):
            return gfs_rows if model == "GFS" else False

        async def zarr_read_window(self, model, store, x, y, start, end):
            self.window_reads.append((model, x, y))
            return await super().zarr_read_window(model, store, x, y, start, end)

    def run(day, future_day):
        weather = _Weather()
        result = asyncio.run(
            calculate_grid_indexing(
                lat=-60.0,
                lon=359.9,
                az_lon=-0.1,
                utc_time=day + datetime.timedelta(hours=12),
                now_time=now,
                time_machine=False,
                ex_hrrr=1,
                ex_nbm=1,
                ex_gfs=0,
                ex_urma=1,
                ex_ecmwf=1,
                ex_gefs=1,
                ex_rtma_ru=1,
                ex_dwd_mosmix=1,
                ex_aigfs=1,
                ex_aigefs=1,
                ex_aifs=1,
                future_day=future_day,
                read_wmo_alerts=False,
                base_day_utc=day,
                num_hours=25,
                zarr_sources=ZarrSources(
                    subh=None,
                    hrrr_6h=None,
                    hrrr=None,
                    nbm=None,
                    gfs=gfs_store,
                    ecmwf=None,
                    gefs=None,
                    cfs=cfs_store,
                ),
                weather=weather,
                logger=logging.getLogger(__name__),
            )
        )
        return result, weather.window_reads

    # Today, and a day 5 days out that GFS covers: no CFS read
    for day, future_day in (
        (datetime.datetime(2026, 11, 10), False),
        (datetime.datetime(2026, 11, 15), True),
    ):
        result, window_reads = run(day, future_day)
        assert window_reads == []
        assert result.dataOut_cfs is False

    day = datetime.datetime(2027, 1, 9)
    result, window_reads = run(day, True)
    assert window_reads == [("CFS", 0, cell_y)]

    base = day.replace(tzinfo=datetime.UTC).timestamp()
    metadata = build_source_metadata(
        grid_result=result, era5_merged=False, use_etopo=False, time_machine=False
    )
    assert "cfs" in metadata.source_list
    assert metadata.source_times["cfs"] == "2026-11-10 00Z"
    merged = merge_hourly_models(
        metadata=metadata,
        num_hours=25,
        base_day_utc_grib=base,
        data_hrrrh=None,
        data_h2=None,
        data_nbm=None,
        data_gfs=result.dataOut_gfs,
        data_ecmwf=None,
        data_gefs=None,
        data_hrdps=None,
        data_gdps=None,
        data_geps=None,
        data_reps=None,
        data_dwd_mosmix=None,
        data_aigfs=None,
        data_aigefs=None,
        data_aifs=None,
        logger=logging.getLogger(__name__),
        loc_tag="test",
        data_cfs=result.dataOut_cfs,
    )
    np.testing.assert_array_equal(merged.cfs[:, 0], base + 3600 * np.arange(25))
    hours_since_first_store_time = (base - cfs_times[0]) // 3600 + np.arange(25)
    np.testing.assert_array_equal(
        merged.cfs[:, CFS["temp"]], 250 + hours_since_first_store_time % 24
    )

    # The day is past GFS, so CFS is the value each element takes
    inputs = hourly_inputs(
        lat=-60.0,
        lon=359.9,
        source_list=merged.metadata.source_list,
        gfs=merged.gfs,
        cfs=merged.cfs,
    )
    temperature = inputs["temperature_inputs"]
    np.testing.assert_array_equal(
        temperature[np.arange(25), np.argmin(np.isnan(temperature), axis=1)],
        merged.cfs[:, CFS["temp"]],
    )
    np.testing.assert_array_equal(inputs["prcipIntensity_inputs"][:, -1], 0.5)
    np.testing.assert_array_equal(inputs["prcipProbability_inputs"][:, -1], 0.25)
    # CFS has no type; "none" lets the hourly block type it from temperature
    np.testing.assert_array_equal(inputs["prcipType_inputs"][:, -1], 0)
