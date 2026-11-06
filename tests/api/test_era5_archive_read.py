"""ERA5 read path: own archive first, Google ERA5 for anything it does not have."""

import datetime
import json
import shutil

import numpy as np
import xarray as xr
import zarr

from API.constants.model_const import (
    ERA5,
    ERA5_ARCHIVE,
    ERA5_ARCHIVE_DELTA_VARS,
    ERA5_ARCHIVE_ENCODING,
    ERA5_ARCHIVE_FILL,
    ERA5_ARCHIVE_NAN_VARS,
    ERA5_ARCHIVE_SOURCES,
    ERA5_ARCHIVE_STORE_DIR,
    ERA5_SOURCE_VARS,
)
from API.io.ZarrHelpers import _era5_archive_cell, init_ERA5_archive
from API.request.grid_indexing import _load_era5_slice

WEEK_HOURS, N_LAT, N_LON, STEP = 168, 10, 10, 0.25
LATS = 90 - STEP * np.arange(N_LAT)
LONS = STEP * np.arange(N_LON)
WEEK_A = np.datetime64("2015-06-29T00", "h")  # a 1940-aligned Monday
GOOGLE_MARKER_K = 999.0  # Google's fake temperature, so the source of a row is visible


def _google_fields(rng, times):
    """Plausible fields in Google's units on the small grid, one per ERA5 source var."""
    shape = (len(times), N_LAT, N_LON)
    smooth = np.sin(np.arange(len(times)) / 12.0)[
        :, None, None
    ] + 0.01 * rng.standard_normal(shape)
    wet = rng.random(shape) < 0.3
    rate = np.where(wet, rng.gamma(0.5, 2e-4, shape), 0.0)  # kg m-2 s-1
    return {
        "instantaneous_10m_wind_gust": 8 + 6 * smooth,
        "mean_sea_level_pressure": 101200 + 1500 * smooth,
        "2m_temperature": 285 + 10 * smooth,
        "2m_dewpoint_temperature": 278 + 8 * smooth,
        "10m_u_component_of_wind": 5 * smooth,
        "10m_v_component_of_wind": -4 * smooth,
        "precipitation_type": np.where(wet, 1.0, 0.0),
        "total_precipitation": rate * 3.6,  # m over the hour
        "large_scale_rain_rate": rate,
        "convective_rain_rate": 0.5 * rate,
        "large_scale_snowfall_rate_water_equivalent": 0.1 * rate,
        "convective_snowfall_rate_water_equivalent": 0.05 * rate,
        "total_column_ozone": (300 + 40 * smooth) / 46696,
        "total_cloud_cover": rng.random(shape),
        "downward_uv_radiation_at_the_surface": np.clip(1.4e5 * smooth, 0, None),
        "surface_solar_radiation_downwards": np.clip(3.2e6 * smooth, 0, None),
        "convective_available_potential_energy": 3000 * rng.random(shape),
        "surface_pressure": 95000 + 1500 * smooth,
        "total_sky_direct_solar_radiation_at_surface": np.clip(2.5e6 * smooth, 0, None),
        "cloud_base_height": np.where(
            rng.random(shape) < 0.2, np.nan, 3000 * rng.random(shape)
        ),
    }


def _encode(name, values):
    """The archive's int16 encoding, written out from ERA5_ARCHIVE_ENCODING.

    The API test environment has no ingest dependencies (API.ingest_utils needs
    cartopy), so this encodes from the published contract rather than calling the
    builder's encoder; tests/ingest/test_era5_archive.py pins that encoder to the
    same decoder.
    """
    kind, step_or_scale, offset = ERA5_ARCHIVE_ENCODING[name]
    missing = np.isnan(values)
    if kind == "lin":
        scaled = (values - offset) / step_or_scale
    else:
        scaled = np.sqrt(np.clip(values, 0, None)) * step_or_scale
    encoded = np.rint(np.where(missing, 0, scaled)).astype(np.int16)
    encoded[missing] = ERA5_ARCHIVE_FILL
    if name in ERA5_ARCHIVE_DELTA_VARS:
        encoded[1:] = np.diff(encoded, axis=0)
    return encoded


def _write_week(root, week_start, google, hour0, store=None, hours=WEEK_HOURS):
    """Write one archive week from the Google-unit fields in the builder's layout:
    int16 (variable, time, y, x), every variable and hour per chunk, sharded,
    Blosc zstd bitshuffle, with the encoding in the attributes. ``store`` is the
    path under the archive root (default the immutable ``<week>.zarr``); a week
    with fewer ``hours`` is partial, as the daily job publishes the current week.
    """
    variables = sorted(ERA5_ARCHIVE, key=ERA5_ARCHIVE.get)
    stacked = np.full(
        (len(variables), WEEK_HOURS, N_LAT, N_LON), ERA5_ARCHIVE_FILL, np.int16
    )
    week = slice(hour0, hour0 + hours)
    for i, name in enumerate(variables):
        if name == "precipitation_probability":
            values = 100 * (google["total_precipitation"][week] > 1e-4).astype(float)
        else:
            sources, factor = ERA5_ARCHIVE_SOURCES[name]
            values = sum(google[source][week] for source in sources) * factor
        stacked[i, :hours] = _encode(name, values.astype(np.float32))
    times = (week_start + np.arange(WEEK_HOURS).astype("timedelta64[h]")).astype(
        "datetime64[s]"
    )
    week_name = str(week_start.astype("datetime64[D]"))
    path = root / (store or f"{ERA5_ARCHIVE_STORE_DIR}/{week_name}.zarr")
    array = zarr.create_array(
        store=zarr.storage.LocalStore(str(path)),
        shape=stacked.shape,
        chunks=(len(variables), WEEK_HOURS, 5, 5),
        shards=(len(variables), WEEK_HOURS, 10, 10),
        dtype="int16",
        fill_value=ERA5_ARCHIVE_FILL,
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle="bitshuffle", typesize=2
        ),
        attributes={
            "time": times.astype(np.int64).tolist(),
            "variables": variables,
            "encoding": {name: list(ERA5_ARCHIVE_ENCODING[name]) for name in variables},
            "delta_variables": list(ERA5_ARCHIVE_DELTA_VARS),
            "nan_variables": list(ERA5_ARCHIVE_NAN_VARS),
            "fill_value": ERA5_ARCHIVE_FILL,
            "complete_through": int(times[hours - 1].astype(np.int64)),
            "latitude_first": float(LATS[0]),
            "longitude_first": float(LONS[0]),
            "grid_step": STEP,
            "n_latitude": N_LAT,
            "n_longitude": N_LON,
        },
    )
    array[:] = stacked


def _publish_version(root, week_start, google, hour0, stamp, hours):
    """A new version of a changing week and the pointer to it, as publish_store
    does: the store first, then the pointer."""
    week_name = str(week_start.astype("datetime64[D]"))
    store = f"{ERA5_ARCHIVE_STORE_DIR}/{week_name}/{stamp}.zarr"
    _write_week(root, week_start, google, hour0, store=store, hours=hours)
    pointer = root / ERA5_ARCHIVE_STORE_DIR / f"{week_name}.current.json"
    pointer.write_text(json.dumps({"store": store, "run_time": None}))
    return store


def _era5_data(google, times, archive):
    """The ERA5_Data dict init_ERA5 builds, with a fake Google dataset whose
    temperature is GOOGLE_MARKER_K so a fallback is visible in the result."""
    data_vars = {
        var: (("time", "latitude", "longitude"), google[var])
        for var in ERA5_SOURCE_VARS
    }
    data_vars["2m_temperature"] = (
        ("time", "latitude", "longitude"),
        np.full_like(google["2m_temperature"], GOOGLE_MARKER_K),
    )
    ds = xr.Dataset(
        data_vars, coords={"time": times, "latitude": LATS, "longitude": LONS}
    )
    return {
        "dsERA5": ds,
        "ERA5_lats": LATS,
        "ERA5_lons": LONS,
        "ERA5_times": times,
        "ERA5_archive": archive,
    }


def test_era5_reads_archive_first_and_falls_back_to_google(tmp_path, monkeypatch):
    """Protect the time-machine read path: a request the archive covers must be
    served from it in the exact layout and units the API expects from Google
    (across a week boundary, with merged rain/snow rates and the precomputed
    probability), and a request touching a week the archive lacks must come
    entirely from Google rather than failing or mixing sources. Also pins the
    archive's cell lookup to the Google path's nearest-cell search.
    """
    rng = np.random.default_rng(5)
    times = WEEK_A + np.arange(2 * WEEK_HOURS).astype("timedelta64[h]")
    google = _google_fields(rng, times)
    _write_week(tmp_path, WEEK_A, google, 0)
    monkeypatch.setenv("ERA5_ARCHIVE", str(tmp_path))
    archive = init_ERA5_archive()

    lat, lon = 89.0, 1.3
    base = datetime.datetime(2015, 7, 5, 16, tzinfo=datetime.UTC)  # 8 h before week B
    y, x = np.argmin(np.abs(LATS - lat)), np.argmin(np.abs(LONS - lon))
    hours = slice(160, 185)

    # Week B is missing: the whole request comes from Google.
    rows = _load_era5_slice(_era5_data(google, times, archive), lat, lon, base, 25)
    assert rows.shape == (25, max(ERA5.values()) + 1)
    np.testing.assert_array_equal(rows[:, ERA5["2m_temperature"]], GOOGLE_MARKER_K)

    # Once week B is published, the same request is read from the archive.
    _write_week(tmp_path, WEEK_A + np.timedelta64(WEEK_HOURS, "h"), google, WEEK_HOURS)
    archive["weeks"].clear()  # forget the earlier "missing" lookup
    rows = _load_era5_slice(_era5_data(google, times, archive), lat, lon, base, 25)
    expected_times = (times[hours].astype("datetime64[s]")).astype(np.int64)
    np.testing.assert_array_equal(rows[:, 0], expected_times)
    point = {var: google[var][hours, y, x] for var in google}
    np.testing.assert_allclose(
        rows[:, ERA5["2m_temperature"]], point["2m_temperature"], atol=0.051
    )
    np.testing.assert_allclose(
        rows[:, ERA5["mean_sea_level_pressure"]],
        point["mean_sea_level_pressure"],
        atol=5.1,
    )
    rain = point["large_scale_rain_rate"] + point["convective_rain_rate"]
    np.testing.assert_allclose(
        rows[:, ERA5["large_scale_rain_rate"]] + rows[:, ERA5["convective_rain_rate"]],
        rain,
        rtol=0.011,
        atol=3e-6,
    )
    np.testing.assert_array_equal(rows[:, ERA5["convective_rain_rate"]], 0)
    np.testing.assert_allclose(
        rows[:, ERA5["total_cloud_cover"]], point["total_cloud_cover"], atol=0.0051
    )
    np.testing.assert_array_equal(
        rows[:, ERA5["prob"]], 100 * (point["total_precipitation"] > 1e-4)
    )

    # Same cell as Google's argmin, on the real 0.25 deg grid, with exact ties.
    attrs = {
        "latitude_first": 90.0,
        "longitude_first": 0.0,
        "grid_step": 0.25,
        "n_latitude": 721,
        "n_longitude": 1440,
    }
    lats, lons = 90 - 0.25 * np.arange(721), 0.25 * np.arange(1440)
    for p_lat, p_lon in [(45.125, 10.125), (-90, 359.99), (90, 0), (0.1, 359.875)] + [
        (float(a), float(b))
        for a, b in zip(rng.uniform(-90, 90, 200), rng.uniform(0, 360, 200))
    ]:
        assert _era5_archive_cell(attrs, p_lat, p_lon) == (
            np.argmin(np.abs(lats - p_lat)),
            np.argmin(np.abs(lons - p_lon)),
        ), (p_lat, p_lon)


def test_era5_reader_follows_daily_updates_without_restart(tmp_path, monkeypatch):
    """Protect the reader against the daily job's updates: a partial week serves
    the hours it has and leaves later hours NaN for the next source (GFS) rather
    than sending the request to Google; a republished version is
    picked up at the next re-check without a restart and without reading a store
    that changes under it; a version deleted under an open handle costs one Google
    fallback, not an error or bad data; and a week promoted to final ERA5 switches
    to the immutable store even while its pointer still exists.
    """
    from API.io import ZarrHelpers

    rng = np.random.default_rng(7)
    times = WEEK_A + np.arange(2 * WEEK_HOURS).astype("timedelta64[h]")
    google = _google_fields(rng, times)
    monkeypatch.setenv("ERA5_ARCHIVE", str(tmp_path))
    archive = init_ERA5_archive()
    data = _era5_data(google, times, archive)
    lat, lon, week = 89.0, 1.3, "2015-06-29"
    y, x = np.argmin(np.abs(LATS - lat)), np.argmin(np.abs(LONS - lon))
    monday = datetime.datetime(2015, 6, 29, 0, tzinfo=datetime.UTC)  # hours 0-24
    tuesday = datetime.datetime(2015, 6, 30, 0, tzinfo=datetime.UTC)  # hours 24-48

    def read(base):
        rows = _load_era5_slice(data, lat, lon, base, 25)
        from_archive = not np.any(rows[:, ERA5["2m_temperature"]] == GOOGLE_MARKER_K)
        return rows, from_archive

    def missing_hours(rows):
        return int(np.isnan(rows[:, ERA5["2m_temperature"]]).sum())

    def store_in_use():
        return archive["weeks"][week][0][2]

    # The first daily run published 30 hours of the week.
    v1 = _publish_version(tmp_path, WEEK_A, google, 0, "20150701T000000Z", 30)
    assert read(monday)[1]
    # Tuesday 00-05 UTC are in the archive; the 19 hours after its last one are
    # NaN (not fill decoded as data), and the request does not go to Google.
    rows, from_archive = read(tuesday)
    assert from_archive and missing_hours(rows) == 19
    np.testing.assert_allclose(
        rows[:6, ERA5["2m_temperature"]],
        google["2m_temperature"][24:30, y, x],
        atol=0.051,
    )

    # The next run publishes the whole week. Readers keep the open version until
    # their re-check, then move to the new one.
    v2 = _publish_version(tmp_path, WEEK_A, google, 0, "20150708T000000Z", WEEK_HOURS)
    assert missing_hours(read(tuesday)[0]) == 19 and store_in_use() == v1
    monkeypatch.setattr(ZarrHelpers, "ERA5_ARCHIVE_VERSIONED_RECHECK_S", 0)
    rows, from_archive = read(tuesday)
    assert from_archive and missing_hours(rows) == 0 and store_in_use() == v2
    np.testing.assert_allclose(
        rows[:, ERA5["2m_temperature"]],
        google["2m_temperature"][24:49, y, x],
        atol=0.051,
    )

    # A version deleted under an open handle: one Google fallback, then forgotten.
    monkeypatch.setattr(ZarrHelpers, "ERA5_ARCHIVE_VERSIONED_RECHECK_S", 3600)
    shutil.rmtree(tmp_path / v2)
    assert not read(tuesday)[1]
    assert week not in archive["weeks"]

    # Final ERA5 arrives: the immutable store wins while the pointer still exists,
    # and stays in use after the pointer and the old versions are deleted.
    _write_week(tmp_path, WEEK_A, google, 0)
    assert read(tuesday)[1]
    assert store_in_use() == f"{ERA5_ARCHIVE_STORE_DIR}/{week}.zarr"
    (tmp_path / ERA5_ARCHIVE_STORE_DIR / f"{week}.current.json").unlink()
    shutil.rmtree(tmp_path / v1)
    assert read(monday)[1] and read(tuesday)[1]
