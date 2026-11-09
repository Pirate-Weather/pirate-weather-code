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
from API.constants.shared_const import ENSEMBLE_MEMBER_COUNTS, member_mask_var_names
from API.daily.builder import _member_precip_share
from API.ingest_utils import (
    KEEPBITS,
    interp_time_take_blend,
    mask_invalid_data,
    open_remote,
    pack_member_mask,
    pad_to_chunk_size,
    publish_store,
    remote_object_exists,
    write_final_zarr,
)
from API.io import store_reader
from API.io.store_reader import WeatherParallel

VARIABLES = ("TMP_2maboveground", "RH_2maboveground")
TIMES = 1.79e9 + 3600 * np.arange(4)


def _write_and_publish(tmp_path, values, name="HRRR", variables=VARIABLES, times=TIMES):
    if isinstance(values, da.Array):
        source = values
    else:
        source = da.from_array(values, chunks=(2, 4, 10, 10))
    # Each ingest run writes into a fresh processing directory
    store_path = tmp_path / f"process_{time.time_ns()}" / f"{name}.zarr"
    write_final_zarr(
        pad_to_chunk_size(source, 3),
        str(store_path),
        final_chunk=3,
        shard=9,
        zarr_store_workers=2,
        variables=variables,
        times=times,
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
        # The NaN at time 1 stays NaN: the reader does not fill gaps
        expected = np.asarray(bitround.encode(values[index, :, 5, 7])).view("float32")
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


def test_member_mask_words_survive_the_store_round_trip(tmp_path):
    """The daily probability popcounts member bitmask words that travel as
    float32 through the whole ingest tail: the invalid-range mask (whose ceiling
    is below 2^24), the hourly interpolation (a blend of two member sets is
    meaningless; they must take the nearest lead's), chunk padding, bitrounding
    and the sharded store. Any of those can flip a bit, which is a member gained
    or lost with no error, so the words read back must equal the packed input
    and give the brute-force "wet at any hour" share.
    """
    rng = np.random.default_rng(1)
    members = ENSEMBLE_MEMBER_COUNTS["GEFS"]
    lead_times = TIMES[0] + 3 * 3600 * np.arange(4)  # 3-hourly leads
    hourly_times = TIMES[0] + 3600 * np.arange(10)
    wet = rng.random((members, 4, 23, 31)) < 0.3

    words = pack_member_mask(da.from_array(wet, chunks=-1), members)
    prob = da.from_array(wet.mean(axis=0, dtype="float32"), chunks=-1)
    variables = ("Precipitation_Prob", *member_mask_var_names(members))
    mask_indices = [1, 2]
    stacked = mask_invalid_data(
        da.stack([prob, *words], axis=0), ignoreAxis=mask_indices
    )
    hourly = interp_time_take_blend(
        stacked, lead_times, hourly_times, nearest_vars=mask_indices
    )
    remote, _ = _write_and_publish(
        tmp_path, hourly, variables=variables, times=hourly_times
    )
    api_dir = tmp_path / "api"
    assert store_sync.sync_store(remote, "HRRR", str(api_dir))
    point = asyncio.run(
        WeatherParallel().zarr_read("HRRR", _load(api_dir).HRRR_Zarr, 7, 5)
    )

    nearest_lead = np.abs(hourly_times[:, None] - lead_times[None, :]).argmin(axis=1)
    expected_words = np.column_stack([w.compute()[:, 5, 7] for w in words])
    np.testing.assert_array_equal(point[:, 2:4], expected_words[nearest_lead])
    # The probability itself still blends linearly between leads
    np.testing.assert_allclose(
        point[:, 1],
        np.interp(hourly_times, lead_times, wet[:, :, 5, 7].mean(axis=0)),
        rtol=0.04,
    )

    share = _member_precip_share(
        [("gefs", point[:, 2:4], members)], [np.zeros(10, dtype=int)], 1, 0
    )
    assert share[0, 0] == wet[:, :, 5, 7].any(axis=1).mean()


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


def test_etopo_is_on_disk_before_the_last_store_pointer(tmp_path, monkeypatch):
    """Workers open ETOPO only at startup, and the OVH chart starts them once the
    WMO_Alerts pointer exists. If the sync copied ETOPO after that pointer, a new
    node would report elevation 0 until its workers next restarted.
    """
    remote_dir = tmp_path / "remote"
    etopo_dir = remote_dir / f"{store_reader.ETOPO_STORE_NAME}.zarr"
    etopo_dir.mkdir(parents=True)
    (etopo_dir / "zarr.json").write_text("{}")
    api_dir = tmp_path / "api"
    api_dir.mkdir()

    etopo_present_at_last_pointer = []

    def record_store(remote, name, base_dir):
        if name == store_sync.STORE_NAMES[-1]:
            etopo_present_at_last_pointer.append(
                os.path.exists(
                    os.path.join(
                        base_dir, f"{store_reader.ETOPO_STORE_NAME}.zarr", "zarr.json"
                    )
                )
            )
        return True

    monkeypatch.setattr(store_sync, "sync_store", record_store)
    remote = obstore.store.LocalStore(str(remote_dir))
    store_sync.sync_once(remote, str(api_dir))

    assert store_sync.STORE_NAMES[-1] == "WMO_Alerts"
    assert etopo_present_at_last_pointer == [True]
