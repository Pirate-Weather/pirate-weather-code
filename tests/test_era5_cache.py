from unittest.mock import MagicMock, patch

import numpy as np
import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import LocalStore

from API.io import zarr_reader
from API.io.ZarrHelpers import (
    ERA5_CACHE_MAX_SIZE,
    ERA5_CACHE_VERSION,
    ERA5_DASK_CHUNKS,
    ERA5_ZARR_URL,
    init_ERA5,
)


def test_init_era5_uses_zarr_cache_store(tmp_path):
    dataset = {
        "latitude": MagicMock(values=[90.0, 89.75]),
        "longitude": MagicMock(values=[0.0, 0.25]),
        "time": MagicMock(values=["2024-01-01T00:00:00"]),
    }
    source_store = object()
    local_store = object()
    cache_store = object()
    cache_dir = tmp_path / "era5-cache"

    with (
        patch(
            "API.io.ZarrHelpers.FsspecStore.from_url",
            return_value=source_store,
        ) as from_url,
        patch("API.io.ZarrHelpers.LocalStore", return_value=local_store) as local,
        patch("API.io.ZarrHelpers.CacheStore", return_value=cache_store) as cache,
        patch("API.io.ZarrHelpers.xr.open_zarr", return_value=dataset) as open_zarr,
    ):
        result = init_ERA5(str(cache_dir))

    from_url.assert_called_once_with(
        ERA5_ZARR_URL,
        storage_options={
            "token": "anon",
            "skip_instance_cache": True,
        },
        read_only=True,
    )
    local.assert_called_once_with(str(cache_dir / ERA5_CACHE_VERSION))
    cache.assert_called_once_with(
        store=source_store,
        cache_store=local_store,
        max_age_seconds="infinity",
        max_size=ERA5_CACHE_MAX_SIZE,
        cache_set_data=False,
    )
    open_zarr.assert_called_once_with(cache_store, chunks=ERA5_DASK_CHUNKS)
    assert result["dsERA5"] is dataset
    assert result["ERA5_cache_dir"] == str(cache_dir)
    assert result["ERA5_cache_store"] is cache_store
    assert result["ERA5_source"] == ERA5_ZARR_URL


def test_zarr_cache_store_reuses_local_cached_chunks(tmp_path):
    source_store = LocalStore(tmp_path / "source")
    source_array = zarr.create_array(
        store=source_store,
        shape=(4,),
        chunks=(2,),
        dtype="i4",
    )
    source_array[:] = np.arange(4)
    cache_store = CacheStore(
        store=source_store.with_read_only(True),
        cache_store=LocalStore(tmp_path / "cache"),
        max_age_seconds="infinity",
        max_size=ERA5_CACHE_MAX_SIZE,
        cache_set_data=False,
    )
    cached_array = zarr.open_array(store=cache_store, mode="r")

    np.testing.assert_array_equal(cached_array[:2], [0, 1])
    stats_after_first_read = cache_store.cache_stats()
    np.testing.assert_array_equal(cached_array[:2], [0, 1])
    stats_after_second_read = cache_store.cache_stats()

    assert cache_store.cache_info()["max_size"] == ERA5_CACHE_MAX_SIZE
    assert stats_after_first_read["misses"] > 0
    assert stats_after_second_read["hits"] > stats_after_first_read["hits"]


def test_update_zarr_store_uses_shared_cache_under_save_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("ERA5_CACHE_DIR", raising=False)

    with (
        patch("API.io.zarr_reader.init_ERA5", return_value={}) as init_era5,
        patch("API.io.zarr_reader._load_local_store"),
    ):
        zarr_reader.update_zarr_store(
            False,
            stage="DEV",
            save_dir=str(tmp_path),
            use_etopo=False,
            save_type="Local",
            s3_bucket="",
            aws_access_key_id="",
            aws_secret_access_key="",
        )

    init_era5.assert_called_once_with(str(tmp_path / "ERA5_cache"))


def test_update_zarr_store_honors_era5_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "shared-era5"
    monkeypatch.setenv("ERA5_CACHE_DIR", str(cache_dir))

    with (
        patch("API.io.zarr_reader.init_ERA5", return_value={}) as init_era5,
        patch("API.io.zarr_reader._load_local_store"),
    ):
        zarr_reader.update_zarr_store(
            False,
            stage="DEV",
            save_dir=str(tmp_path),
            use_etopo=False,
            save_type="Local",
            s3_bucket="",
            aws_access_key_id="",
            aws_secret_access_key="",
        )

    init_era5.assert_called_once_with(str(cache_dir))
