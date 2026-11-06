# Helpers for working with Zarr data in the Pirate Weather API
# Alexander Rey, October 2025

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import FsspecStore

logger = logging.getLogger(__name__)

pw_api_key = os.environ.get("PW_API", "")

ERA5_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_DASK_CHUNKS = {"time": 24}
ERA5_CACHE_VERSION = "cache-store-v1"
ERA5_CACHE_MAX_SIZE = 20 * 1024**3
ERA5_CACHE_DIR_DEFAULT = "ERA5_cache"
# Own ERA5 archive (scripts/build_era5_point_test_store.py). Opened weeks keep
# their tensorstore handle; shard indexes are held in a shared cache pool, so a
# repeat read of a week is one ranged GET of 110 KiB (measured 24-29 ms on OVH
# High Performance storage against 0.6-8 s for Google).
# A week is either immutable (final ERA5, cached for good) or versioned behind a
# pointer (ERA5T and the current, partial week; scripts/era5_archive_daily.py
# republishes them daily). A versioned week's pointer is re-resolved after
# ERA5_ARCHIVE_VERSIONED_RECHECK_S and a missing week is looked up again after
# ERA5_ARCHIVE_MISSING_RECHECK_S, so new data and newly final weeks are picked up
# without a restart. The daily job keeps the previous version for a day, far
# longer than a reader can hold a stale pointer.
ERA5_ARCHIVE_OPEN_WEEKS = 512
ERA5_ARCHIVE_INDEX_CACHE_BYTES = 256 * 1024**2
ERA5_ARCHIVE_VERSIONED_RECHECK_S = 300
ERA5_ARCHIVE_MISSING_RECHECK_S = 300
# A missing week starting less than this long ago is one the data has not reached
# yet (NaN rows, filled by GFS); an older missing week is a gap (read Google).
ERA5_ARCHIVE_UNPUBLISHED_DAYS = 14


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

        tasks = [self.get(key, prototype, byte_range) for key, byte_range in key_ranges]
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
        # Read first when configured; Google is the fallback (_load_era5_slice).
        "ERA5_archive": init_ERA5_archive(),
    }

    return ERA5_Data


def decode_era5_archive(columns: np.ndarray, attributes: dict) -> np.ndarray:
    """Decode int16 ``(variable, hours)`` columns of an ERA5 archive store to
    float64 values in the store's API units (``attributes["units"]``).

    ``columns`` must start at the first hour of the store's week: the
    ``delta_variables`` hold hour-to-hour differences, so their values are the
    cumulative sum from that hour. The fill value (padding rows; no cloud base)
    becomes NaN. Reads the encoding from the store's attributes, so a store keeps
    decoding correctly if the constants change for later builds.
    """
    fill = attributes["fill_value"]
    delta = set(attributes["delta_variables"])
    missing = columns == fill
    values = np.full(columns.shape, np.nan)
    for i, name in enumerate(attributes["variables"]):
        series = columns[i].astype(np.int64)
        if name in delta:
            if missing[i].any():
                continue  # padding row: no data
            series = np.cumsum(series)
        kind, step_or_scale, offset = attributes["encoding"][name]
        if kind == "lin":
            decoded = series * step_or_scale + offset
        else:
            decoded = (series / step_or_scale) ** 2
        values[i] = np.where(missing[i], np.nan, decoded)
    return values


def init_ERA5_archive() -> dict | None:
    """Locate the own ERA5 archive from ``ERA5_ARCHIVE``, or None when unset.

    ``ERA5_ARCHIVE`` is ``s3://<bucket>/<prefix>`` (weeks under
    ``<prefix>/ERA5_archive/``) or a local directory. On S3 the endpoint and region
    come from ``ERA5_ARCHIVE_S3_ENDPOINT`` / ``ERA5_ARCHIVE_S3_REGION`` (for the
    OVH private endpoint), else ``s3_endpoint`` / ``s3_region``; credentials from
    the usual AWS environment variables, as for the other S3 stores.
    """
    import tensorstore as ts

    location = os.environ.get("ERA5_ARCHIVE", "").strip()
    if not location:
        return None
    if location.startswith("s3://"):
        bucket, _, prefix = location[len("s3://") :].partition("/")
        prefix = prefix.strip("/")
        kvstore = {
            "driver": "s3",
            "bucket": bucket,
            "path": f"{prefix}/" if prefix else "",
        }
        endpoint = os.getenv("ERA5_ARCHIVE_S3_ENDPOINT") or os.getenv("s3_endpoint")
        region = os.getenv("ERA5_ARCHIVE_S3_REGION") or os.getenv("s3_region")
        if endpoint:
            kvstore["endpoint"] = endpoint
        if region:
            kvstore["aws_region"] = region
    else:
        kvstore = {"driver": "file", "path": os.path.abspath(location) + "/"}
    logger.info("ERA5 archive: %s (Google ERA5 for missing weeks)", location)
    return {
        "kvstore": kvstore,
        "context": ts.Context(
            {"cache_pool": {"total_bytes_limit": ERA5_ARCHIVE_INDEX_CACHE_BYTES}}
        ),
        "weeks": OrderedDict(),
        "lock": threading.Lock(),
    }


def _resolve_era5_archive_week(
    archive: dict, week_start: str
) -> tuple[str, dict] | None:
    """``(store path, attributes)`` of the store to read for a week, or None.

    The immutable ``<week>.zarr`` (final ERA5) wins over the pointer
    ``<week>.current.json`` (ERA5T or the current partial week), because while a
    week is being promoted to final both exist. A store counts only once its root
    ``zarr.json`` exists, which is uploaded last.
    """
    import tensorstore as ts

    from API.constants.model_const import ERA5_ARCHIVE_STORE_DIR

    root = archive.get("root_kvstore")
    if root is None:
        root = ts.KvStore.open(archive["kvstore"], context=archive["context"]).result()
        archive["root_kvstore"] = root
    immutable = f"{ERA5_ARCHIVE_STORE_DIR}/{week_start}.zarr"
    metadata = root.read(f"{immutable}/zarr.json").result()
    if metadata.state == "value":
        return immutable, json.loads(bytes(metadata.value))["attributes"]
    pointer = root.read(f"{ERA5_ARCHIVE_STORE_DIR}/{week_start}.current.json").result()
    if pointer.state != "value":
        return None
    version = json.loads(bytes(pointer.value))["store"]
    metadata = root.read(f"{version}/zarr.json").result()
    if metadata.state != "value":
        return None  # pointer written before its store: never, but read as missing
    return version, json.loads(bytes(metadata.value))["attributes"]


def _open_era5_archive_week(archive: dict, week_start: str):
    """``(tensorstore array, attributes, store path)`` for one archive week, or
    None if the week is not published or cannot be opened.

    Immutable weeks are cached for good. A versioned week is re-resolved after
    ``ERA5_ARCHIVE_VERSIONED_RECHECK_S``: the open handle is kept while its
    pointer names the same version and replaced when the pointer moves, so a
    store never changes under a reader. A failed lookup keeps a cached version
    in use rather than dropping to Google.
    """
    import tensorstore as ts

    from API.constants.model_const import ERA5_ARCHIVE_STORE_DIR

    now = time.monotonic()
    with archive["lock"]:
        cached = archive["weeks"].get(week_start)
    if cached is not None:
        opened, checked_at = cached
        if opened is None:
            fresh = now - checked_at < ERA5_ARCHIVE_MISSING_RECHECK_S
        elif opened[2] == f"{ERA5_ARCHIVE_STORE_DIR}/{week_start}.zarr":
            fresh = True  # immutable
        else:
            fresh = now - checked_at < ERA5_ARCHIVE_VERSIONED_RECHECK_S
        if fresh:
            with archive["lock"]:
                if week_start in archive["weeks"]:
                    archive["weeks"].move_to_end(week_start)
            return opened
    previous = cached[0] if cached is not None else None

    try:
        resolved = _resolve_era5_archive_week(archive, week_start)
        if resolved is None:
            opened = None
        elif previous is not None and previous[2] == resolved[0]:
            opened = previous  # pointer unchanged: keep the open handle
        else:
            store_path, attributes = resolved
            store_kvstore = {
                **archive["kvstore"],
                "path": f"{archive['kvstore']['path']}{store_path}/",
            }
            array = ts.open(
                {"driver": "zarr3", "kvstore": store_kvstore},
                read=True,
                context=archive["context"],
            ).result()
            opened = (array, attributes, store_path)
            if previous is not None:
                logger.info("ERA5 archive week %s now reads %s", week_start, store_path)
    except (
        ValueError
    ) as exc:  # tensorstore reports I/O and metadata errors as ValueError
        logger.warning(
            "ERA5 archive week %s could not be resolved: %s", week_start, exc
        )
        opened = previous

    with archive["lock"]:
        archive["weeks"][week_start] = (opened, time.monotonic())
        archive["weeks"].move_to_end(week_start)
        while len(archive["weeks"]) > ERA5_ARCHIVE_OPEN_WEEKS:
            archive["weeks"].popitem(last=False)
    return opened


def _forget_era5_archive_week(archive: dict, week_start: str) -> None:
    """Drop a week from the cache so the next request resolves it again (after a
    read failed, e.g. because the version it pointed at was deleted)."""
    with archive["lock"]:
        archive["weeks"].pop(week_start, None)


def _era5_archive_cell(attributes: dict, lat: float, lon: float) -> tuple[int, int]:
    """Archive grid cell for a point, matching the nearest-cell search the Google
    path does (np.argmin over its latitudes and 0-360 longitudes; exact ties take
    the lower index, and longitudes past the last column stay in it)."""
    step = attributes["grid_step"]
    y = int(np.ceil((attributes["latitude_first"] - lat) / step - 0.5))
    x = int(np.ceil(((lon - attributes["longitude_first"]) % 360) / step - 0.5))
    y = min(max(y, 0), attributes["n_latitude"] - 1)
    x = min(max(x, 0), attributes["n_longitude"] - 1)
    return y, x


def read_era5_archive_point(
    archive: dict, lat: float, lon: float, base_time: np.datetime64, num_hours: int
) -> np.ndarray | None:
    """ERA5 point rows from the archive, in the layout and units ``_load_era5_slice``
    returns from Google (time column 0, ``ERA5`` columns, ``prob`` last).

    Hours the archive does not have yet (past a partial week's last hour, or in a
    recent week that is not published) are NaN rows, as Google returns past its
    own end; the API fills them from the next source (GFS) hour by hour. Returns
    None, so the caller reads Google, only when an older week is missing (a gap
    that should not exist), a store lacks a variable, or a read fails.

    A delta-coded week has to be read from its first hour, so a read takes the
    week's hours up to the last one needed; that is the same chunk either way.
    """
    from API.constants.model_const import (
        ERA5,
        ERA5_ARCHIVE_SOURCES,
        ERA5_ARCHIVE_WEEK_EPOCH,
        ERA5_ARCHIVE_WEEK_HOURS,
    )

    epoch = np.datetime64(ERA5_ARCHIVE_WEEK_EPOCH, "s")
    # Nearest hour, ties to the earlier one (as argmin over Google's hourly times).
    first_hour = int(
        np.ceil((np.datetime64(base_time, "s") - epoch) / np.timedelta64(1, "h") - 0.5)
    )
    hours = first_hour + np.arange(num_hours)
    if first_hour < 0:
        return None
    hour_times = (
        (epoch + hours.astype("timedelta64[h]"))
        .astype("datetime64[s]")
        .astype(np.int64)
    )
    n_columns = max(ERA5.values()) + 1
    # Unpublished weeks: Google's ERA5T runs ~5-7 days behind and the daily job
    # publishes the current week partial, so a missing week this recent is one
    # the data has not reached yet, not a gap.
    unpublished_after = (
        np.datetime64("now", "s") - np.timedelta64(ERA5_ARCHIVE_UNPUBLISHED_DAYS, "D")
    ).astype(np.int64)
    rows = []
    for week in np.unique(hours // ERA5_ARCHIVE_WEEK_HOURS):
        in_request = hours // ERA5_ARCHIVE_WEEK_HOURS == week
        in_week = hours[in_request] - week * ERA5_ARCHIVE_WEEK_HOURS
        week_rows = np.full((len(in_week), n_columns), np.nan)
        week_rows[:, 0] = hour_times[in_request]
        week_start_time = int(hour_times[in_request][0] - in_week[0] * 3600)
        week_start = str(np.datetime64(week_start_time, "s").astype("datetime64[D]"))

        opened = _open_era5_archive_week(archive, week_start)
        if opened is None:
            if week_start_time >= unpublished_after:
                rows.append(week_rows)  # not published yet: NaN, filled by GFS
                continue
            return None  # an older week is missing: read Google
        array, attributes, _ = opened
        # A partial week (the current one) has data through complete_through.
        complete_through = attributes.get("complete_through", int(week_rows[-1, 0]))
        have = week_rows[:, 0] <= complete_through
        if have.any():
            y, x = _era5_archive_cell(attributes, lat, lon)
            last = int(in_week[have][-1])
            try:
                columns = array[:, : last + 1, y, x].read().result()
            except ValueError as exc:
                # Most likely a version deleted under a stale handle.
                logger.warning("ERA5 archive week %s read failed: %s", week_start, exc)
                _forget_era5_archive_week(archive, week_start)
                return None
            decoded = decode_era5_archive(columns, attributes)[:, in_week[have]]
            filled = {0}
            for i, name in enumerate(attributes["variables"]):
                if name == "precipitation_probability":
                    week_rows[have, ERA5["prob"]] = decoded[i]
                    filled.add(ERA5["prob"])
                    continue
                sources, factor = ERA5_ARCHIVE_SOURCES.get(name, ((), 1.0))
                if not sources or sources[0] not in ERA5:
                    continue  # stored for later use (direct solar, cloud base)
                week_rows[have, ERA5[sources[0]]] = decoded[i] / factor
                # The API only uses large-scale + convective sums: 0 in the second.
                for source in sources[1:]:
                    week_rows[have, ERA5[source]] = 0.0
                filled.update(ERA5[source] for source in sources)
            if filled != set(range(n_columns)):
                return None  # a store without a variable the API needs
            if np.isnan(week_rows[have]).any():
                # Hours the store says it has came back as fill: its shards are
                # gone (tensorstore reads a deleted shard as fill, not an error).
                _forget_era5_archive_week(archive, week_start)
                return None
        rows.append(week_rows)
    return np.concatenate(rows)
