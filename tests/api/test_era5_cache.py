"""ERA5 read-through disk cache (DiskCacheStore) and its startup switch."""

from unittest.mock import patch

import numpy as np
import zarr
from zarr.storage import LocalStore

from API.io import zarr_reader
from API.io.ZarrHelpers import DiskCacheStore

CHUNK = 8192  # float64 elements -> 64 KiB per uncompressed chunk
N_CHUNKS = 16


def _cached_array(tmp_path, size_limit):
    """Write a 1 MiB uncompressed source array and open it through the cache."""
    source = LocalStore(tmp_path / "source")
    values = np.random.default_rng(0).random(CHUNK * N_CHUNKS)
    array = zarr.create_array(
        store=source, shape=values.shape, chunks=(CHUNK,), dtype="f8", compressors=None
    )
    array[:] = values
    cache_store = DiskCacheStore(
        store=source.with_read_only(True),
        cache_dir=str(tmp_path / "cache"),
        size_limit=size_limit,
    )
    return values, cache_store, zarr.open_array(store=cache_store, mode="r")


def test_disk_cache_store_reuses_cached_chunks(tmp_path):
    """ERA5 is read from a remote bucket; a repeated read of the same region must
    be served from the local diskcache instead of going back to the network.
    """
    values, cache_store, array = _cached_array(tmp_path, size_limit=64 * 1024**2)

    np.testing.assert_array_equal(array[: 2 * CHUNK], values[: 2 * CHUNK])
    after_first = cache_store.cache_stats()
    np.testing.assert_array_equal(array[: 2 * CHUNK], values[: 2 * CHUNK])
    after_second = cache_store.cache_stats()

    assert after_first["misses"] > 0
    assert after_second["misses"] == after_first["misses"]
    assert after_second["hits"] - after_first["hits"] == 2


def test_disk_cache_store_stays_within_size_limit(tmp_path):
    """Regression for #696 (ERA5 cache memory/disk leak): reading more data than
    the quota must evict old chunks so the cache never grows past size_limit,
    while still returning correct data.
    """
    size_limit = 300_000  # < 1 MiB of chunks read below
    values, cache_store, array = _cached_array(tmp_path, size_limit=size_limit)

    for _ in range(2):
        np.testing.assert_array_equal(array[:], values)

    info = cache_store.cache_info()
    assert info["cache_store_type"] == "Cache"  # the real diskcache backend
    assert info["current_size"] <= size_limit
    assert info["cached_keys"] < N_CHUNKS


def test_update_zarr_store_skips_era5_when_env_enabled(tmp_path, monkeypatch):
    """SKIP_ERA5 lets dev/CI hosts start without network access to the ERA5
    bucket; it must stop init_ERA5 from being called at all.
    """
    monkeypatch.setenv("SKIP_ERA5", "true")

    with (
        patch("API.io.zarr_reader.init_ERA5") as init_era5,
        patch("API.io.zarr_reader._load_local_store"),
    ):
        stores = zarr_reader.update_zarr_store(
            False,
            stage="DEV",
            save_dir=str(tmp_path),
            use_etopo=False,
            save_type="Local",
            s3_bucket="",
            aws_access_key_id="",
            aws_secret_access_key="",
        )

    init_era5.assert_not_called()
    assert stores.ERA5_Data is None
