# Helpers for working with Zarr data in the Pirate Weather API
# Alexander Rey, October 2025

import os
import random
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

import s3fs
import xarray as xr
import zarr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore

from API.constants.api_const import (
    MAX_S3_RETRIES,
    S3_BASE_DELAY,
)

pw_api_key = os.environ.get("PW_API", "")

ERA5_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_DASK_CHUNKS = {"time": 24}
ERA5_CACHE_VERSION = "cache-store-v1"
ERA5_CACHE_MAX_SIZE = 20 * 1024**3
ERA5_CACHE_DIR_DEFAULT = "ERA5_cache"


@dataclass
class _DiskCacheStats:
    hits: int = 0
    misses: int = 0


class DiskCacheStore(Store):
    """Read-through Zarr store with a persistent diskcache byte quota."""

    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False
    supports_consolidated_metadata = True

    def __init__(
        self,
        store: Store,
        *,
        cache_dir: str,
        size_limit: int,
        read_only: bool = True,
        cache: Any | None = None,
        stats: _DiskCacheStats | None = None,
    ) -> None:
        super().__init__(read_only=read_only)
        if cache is None:
            from diskcache import Cache

            cache = Cache(
                cache_dir,
                size_limit=size_limit,
                eviction_policy="least-recently-used",
            )
        self._store = store
        self._cache = cache
        self._cache_dir = cache_dir
        self._size_limit = size_limit
        self._stats = stats or _DiskCacheStats()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, type(self)) and self._store == value._store

    def with_read_only(self, read_only: bool = False):
        return type(self)(
            self._store.with_read_only(read_only),
            cache_dir=self._cache_dir,
            size_limit=self._size_limit,
            read_only=read_only,
            cache=self._cache,
            stats=self._stats,
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        cached_bytes = self._cache.get(key)
        if cached_bytes is not None:
            self._stats.hits += 1
            return prototype.buffer.from_bytes(
                _slice_cached_bytes(cached_bytes, byte_range)
            )

        self._stats.misses += 1
        result = await self._store.get(key, prototype)
        if result is None:
            self._cache.pop(key, None)
            return None

        value = result.to_bytes()
        if len(value) <= self._size_limit:
            self._cache.set(key, value)
            self._cache.cull()
        return prototype.buffer.from_bytes(_slice_cached_bytes(value, byte_range))

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        import asyncio
        tasks = [
            self.get(key, prototype, byte_range) for key, byte_range in key_ranges
        ]
        return list(await asyncio.gather(*tasks))

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("DiskCacheStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("DiskCacheStore is read-only")

    async def exists(self, key: str) -> bool:
        return await self._store.exists(key)

    def list(self) -> AsyncIterator[str]:
        return self._store.list()

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        return self._store.list_prefix(prefix)

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        return self._store.list_dir(prefix)

    async def getsize(self, key: str) -> int:
        return await self._store.getsize(key)

    async def getsize_prefix(self, prefix: str) -> int:
        return await self._store.getsize_prefix(prefix)

    def cache_info(self) -> dict[str, Any]:
        return {
            "cache_store_type": type(self._cache).__name__,
            "max_size": self._size_limit,
            "current_size": self._cache.volume(),
            "cached_keys": len(self._cache),
        }

    def cache_stats(self) -> dict[str, Any]:
        total_requests = self._stats.hits + self._stats.misses
        hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0.0
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": 0,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }

    def close(self) -> None:
        self._cache.close()
        self._store.close()
        super().close()


def _slice_cached_bytes(value: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return value
    if isinstance(byte_range, RangeByteRequest):
        return value[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return value[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return value[-byte_range.suffix :]
    raise TypeError(f"Unexpected byte_range, got {byte_range}.")


def _get_era5_cache_max_size() -> int:
    """Get ERA5 cache max size from env var (bytes), defaulting to 20 GiB."""
    env_value = os.environ.get("ERA5_CACHE_MAX_SIZE")
    if not env_value:
        return ERA5_CACHE_MAX_SIZE
    try:
        max_size = int(env_value)
    except ValueError:
        return ERA5_CACHE_MAX_SIZE
    return max_size if max_size > 0 else ERA5_CACHE_MAX_SIZE


def _add_custom_header(request, **kwargs):
    request.headers["apikey"] = pw_api_key


class S3ZipStore(zarr.storage.ZipStore):
    def __init__(self, path: s3fs.S3File) -> None:
        super().__init__(path="", mode="r")
        self.path = path


def _retry_s3_operation(
    operation, max_retries=MAX_S3_RETRIES, base_delay=S3_BASE_DELAY
):
    """Retry S3 operations with exponential backoff for rate limiting."""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            # Check if it's a rate limiting error
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"S3 rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
            # Re-raise the exception if it's not rate limiting or max retries reached
            raise e
    raise Exception(f"Failed after {max_retries} attempts")


def setup_testing_zipstore(s3, s3_bucket, ingest_version, save_type, model_name):
    """Sets up a zarr store from a zipped zarr file in S3 or locally.

    Parameters:
        - s3 (s3fs.S3FileSystem): An s3fs filesystem object.
        - s3_bucket (str): The S3 bucket name or local path.
        - ingest_version (str): The version string for the data.
        - save_type (str): The type of storage ("S3", "S3Zarr", or local path).
        - model_name (str): The name of the model.

    Returns:
        - store: A zarr store object.
    """

    if save_type == "S3":
        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
        # Try an old ingest version for testing
        except FileNotFoundError:
            ingest_version = "v28"
            print("Using old ingest version: " + ingest_version)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
        except FileNotFoundError:
            ingest_version = "v28"
            print("Using old ingest version: " + ingest_version)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)

    else:
        f = s3_bucket + model_name + ".zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    return store


# Function to initialize in ERA5 xarray dataset
def init_ERA5(cache_dir: str | None = None):
    """Open Google ERA5 through the persistent object cache."""
    cache_dir = os.environ.get("ERA5_CACHE_DIR", cache_dir or ERA5_CACHE_DIR_DEFAULT)
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    object_cache_dir = os.path.join(cache_dir, ERA5_CACHE_VERSION)
    source_store = FsspecStore.from_url(
        ERA5_ZARR_URL,
        storage_options={
            "token": "anon",
            "skip_instance_cache": True,
        },
        read_only=True,
    )
    cache_store = DiskCacheStore(
        store=source_store,
        cache_dir=object_cache_dir,
        size_limit=_get_era5_cache_max_size(),
    )

    dsERA5 = xr.open_zarr(
        cache_store,
        chunks=ERA5_DASK_CHUNKS,
    )
    source = ERA5_ZARR_URL

    ERA5_lats = dsERA5["latitude"].values
    ERA5_lons = dsERA5["longitude"].values
    ERA5_times = dsERA5["time"].values

    ERA5_Data = {
        "dsERA5": dsERA5,
        "ERA5_lats": ERA5_lats,
        "ERA5_lons": ERA5_lons,
        "ERA5_times": ERA5_times,
        "ERA5_cache_dir": cache_dir,
        "ERA5_cache_store": cache_store,
        "ERA5_source": source,
    }

    return ERA5_Data
