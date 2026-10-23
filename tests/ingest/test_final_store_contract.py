"""Final store contract: ingest write -> S3 publish -> local sync -> API read."""

import asyncio
import json
import os
import time

import dask.array as da
import numcodecs
import numpy as np
import obstore

from API import store_sync
from API.ingest_utils import (
    KEEPBITS,
    open_remote,
    pad_to_chunk_size,
    publish_store,
    remote_object_exists,
    write_final_zarr,
)
from API.io import store_reader
from API.io.store_reader import WeatherParallel

VARIABLES = ("TMP_2maboveground", "RH_2maboveground")
TIMES = 1.79e9 + 3600 * np.arange(4)


def _write_and_publish(tmp_path, values, name="HRRR"):
    source = da.from_array(values, chunks=(2, 4, 10, 10))
    # Each ingest run writes into a fresh processing directory
    store_path = tmp_path / f"process_{time.time_ns()}" / f"{name}.zarr"
    write_final_zarr(
        pad_to_chunk_size(source, 3),
        str(store_path),
        final_chunk=3,
        shard=9,
        zarr_store_workers=2,
        variables=VARIABLES,
        times=TIMES,
    )
    remote = open_remote(str(tmp_path / "remote"), "Download")
    return remote, publish_store(str(store_path), remote, f"{name}.zarr")


def _load(api_dir):
    stores, _ = asyncio.run(
        store_reader.load_stores(
            stage="PROD",
            save_dir=str(api_dir),
            use_etopo=False,
            save_type="Download",
            s3_bucket="",
        )
    )
    return stores


def test_published_store_reads_back_with_time_column(tmp_path):
    """Protect the ingest/API contract after times moved out of the array: a
    store written by write_final_zarr, published and synced, must read back
    through the tensorstore reader as the (time, 1 + variable) matrix every
    API module indexes, with exact times in column 0 and each variable
    bitrounded to its KEEPBITS precision. Time must be NaN where every
    variable is NaN (chunk padding, cells a model doesn't cover), as it was
    when time was stored, because the merge uses finite times as coverage.
    The air-quality max-square read must clip at the grid edge (#682), not
    wrap to the opposite edge.
    """
    rng = np.random.default_rng(0)
    values = rng.random((2, 4, 23, 31), dtype="float32") * 100
    values[0] += 230
    values[:, :, 0, 0] = np.nan  # a cell the model doesn't cover
    values[0, 1, 5, 7] = np.nan
    values[:, :, 20:, 28:] = 99.0  # far corner: must not leak into the (0, 1) square
    remote, _ = _write_and_publish(tmp_path, values)

    api_dir = tmp_path / "api"
    assert store_sync.sync_store(remote, "HRRR", str(api_dir))
    stores = _load(api_dir)
    weather = WeatherParallel()

    point = asyncio.run(weather.zarr_read("HRRR", stores.HRRR_Zarr, 7, 5))
    np.testing.assert_array_equal(point[:, 0], TIMES)
    for index, name in enumerate(VARIABLES):
        bitround = numcodecs.BitRound(keepbits=KEEPBITS[name])
        expected = np.asarray(bitround.encode(values[index, :, 5, 7])).view("float32")
        # The NaN at time 1 is an interior hole the reader interpolates
        expected = expected.copy()
        if index == 0:
            expected[1] = (expected[0] + expected[2]) / 2
        np.testing.assert_allclose(point[:, 1 + index], expected, rtol=1e-6)

    uncovered = asyncio.run(weather.zarr_read("HRRR", stores.HRRR_Zarr, 0, 0))
    assert np.isnan(uncovered).all()
    padding = asyncio.run(weather.zarr_read("HRRR", stores.HRRR_Zarr, 32, 23))
    assert np.isnan(padding).all()

    square = asyncio.run(weather.zarr_read_max_square("SILAM", stores.HRRR_Zarr, 1, 0))
    np.testing.assert_array_equal(square[:, 0], TIMES)
    np.testing.assert_allclose(
        square[:, 1], np.nanmax(values[0, :, :2, :3], axis=(-2, -1)), rtol=1e-2
    )


def test_old_version_is_kept_until_readers_move_on(tmp_path, monkeypatch):
    """Deleting a store under a running reader makes its reads silently return
    NaN, so a new publish must be swapped in by the API worker and the old
    version deleted by the sync only after no live reader lists it. On S3 the
    previous version is kept so a sync copying it when the pointer moves can
    finish.
    """
    values = np.ones((2, 4, 23, 31), dtype="float32")
    remote, first = _write_and_publish(tmp_path, values)
    api_dir = tmp_path / "api"
    store_sync.sync_store(remote, "HRRR", str(api_dir))
    stores = _load(api_dir)
    assert stores.HRRR_Zarr.version == first

    time.sleep(1.1)  # versions are stamped to the second
    _, second = _write_and_publish(tmp_path, values * 2)
    store_sync.sync_store(remote, "HRRR", str(api_dir))
    later = time.time() + store_sync.SUPERSEDED_GRACE_SECONDS + 1
    heartbeat = next((api_dir / ".readers").iterdir())
    os.utime(heartbeat, (later, later))

    # The worker still holds the first version: it must survive the sync's cleanup
    store_sync.remove_old_versions(str(api_dir), now=later)
    assert (api_dir / first).is_dir()

    root = store_reader.local_store_root(str(api_dir))
    asyncio.run(store_reader.refresh_stores(stores, root, "PROD"))
    assert stores.HRRR_Zarr.version == second
    assert json.loads(heartbeat.read_text())["versions"] == [second]
    point = asyncio.run(WeatherParallel().zarr_read("HRRR", stores.HRRR_Zarr, 3, 3))
    np.testing.assert_array_equal(point[:, 1:], 2.0)

    os.utime(heartbeat, (later, later))
    store_sync.remove_old_versions(str(api_dir), now=later)
    assert not (api_dir / first).exists()
    assert (api_dir / second).is_dir()

    time.sleep(1.1)
    remote, third = _write_and_publish(tmp_path, values * 3)
    listing = obstore.list_with_delimiter(remote, prefix="HRRR/")
    published = sorted(
        prefix.rstrip("/")
        for prefix in listing["common_prefixes"]
        if remote_object_exists(remote, prefix.rstrip("/") + "/zarr.json")
    )
    assert published == [second, third]
