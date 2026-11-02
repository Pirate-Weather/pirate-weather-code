"""Open the published model stores and read point forecasts from them.

Each final store is a sharded zarr v3 array of ``(variable, time, y, x)`` with
its valid times in the ``time`` attribute (see ``write_final_zarr``). The API
reads it with tensorstore: a 16-model request took ~1.3 ms against ~10 ms
through zarr-python, which spent most of each read in Python under the GIL
(Notes/STORAGE_AND_READ_BENCHMARKS.md).

Every store is published as versions: ``<MODEL>/<stamp>.zarr`` plus a pointer
``<MODEL>.current.json`` naming the current one, on S3 and, after
``API/store_sync.py`` copies it, on local disk. Each worker re-reads the
pointers every ``STORE_POLL_SECONDS``, opens a new version when one appears
and swaps it in; requests already running keep the version they started with.
Each worker also writes a heartbeat file listing the versions it has open, and
the sync deletes an old version only once no live heartbeat lists it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import socket
import struct
from dataclasses import dataclass
from typing import Any

import google_crc32c
import numpy as np
import tensorstore as ts
import zarr
from numcodecs.zstd import Zstd

from API.constants.api_const import MAX_ZARR_READ_RETRIES
from API.constants.shared_const import INGEST_VERSION_STR, MISSING_DATA
from API.io.ZarrHelpers import init_ERA5

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# How often each worker checks for newly synced store versions.
STORE_POLL_SECONDS = float(os.getenv("STORE_POLL_SECONDS", "10"))
# Heartbeats live next to the stores so the sync (another process or
# container on the same volume) can see which versions are still open.
READERS_DIR_NAME = ".readers"

# The public, S3-compatible files endpoint the TESTING stages read through.
REMOTE_FILES_URL = "https://api.pirateweather.net/files/"
pw_api_key = os.environ.get("PW_API", "")

# One tensorstore context per worker, shared by every store. The cache pool
# keeps the shard indexes (~35 MB for all stores) and hot chunks decoded;
# recheck_cached_data="open" (set at open) trusts that cache because a version
# never changes once published. Two I/O and copy threads were as fast as the
# defaults with 4 workers per host and keep each worker's thread count bounded.
TS_CONTEXT = ts.Context(
    {
        "cache_pool": {
            "total_bytes_limit": int(os.getenv("TS_CACHE_BYTES", str(256 << 20)))
        },
        "data_copy_concurrency": {"limit": 2},
        "file_io_concurrency": {"limit": 2},
    }
)

# (ZarrStores attribute, published store name). The forecast stores are read
# with tensorstore; the alert grids are variable-length string arrays, which
# tensorstore cannot read, so they stay on zarr-python.
FORECAST_STORES = (
    ("GFS_Zarr", "GFS"),
    ("URMA_Zarr", "URMA_Hist"),
    ("SubH_Zarr", "SubH"),
    ("HRRR_6H_Zarr", "HRRR_6H"),
    ("ECMWF_Zarr", "ECMWF"),
    ("NBM_Zarr", "NBM"),
    ("GEFS_Zarr", "GEFS"),
    ("HRDPS_Zarr", "HRDPS"),
    ("GDPS_Zarr", "GDPS"),
    ("GEPS_Zarr", "GEPS"),
    ("REPS_Zarr", "REPS"),
    ("HRRR_Zarr", "HRRR"),
    ("RTMA_RU_Zarr", "RTMA_RU"),
    ("DWD_MOSMIX_Zarr", "DWD_MOSMIX"),
    ("AIGFS_Zarr", "AIGFS"),
    ("AIGEFS_Zarr", "AIGEFS"),
    ("ECMWF_AIFS_Zarr", "ECMWF_AIFS"),
    ("RAQDPS_Zarr", "RAQDPS"),
    ("SILAM_Zarr", "SILAM"),
)
ALERT_STORES = (
    ("NWS_Alerts_Zarr", "NWS_Alerts"),
    ("WMO_Alerts_Zarr", "WMO_Alerts"),
)
# TimeMachine only needs the stores that hold history.
TIMEMACHINE_STORES = (("GFS_Zarr", "GFS"), ("URMA_Zarr", "URMA_Hist"))
# Static elevation grid, synced once and never versioned.
ETOPO_STORE_NAME = "ETOPO_DA_C"


def _default_logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelStore:
    """One opened version of a final store."""

    array: Any  # tensorstore.TensorStore (variable, time, y, x)
    times: np.ndarray  # Unix seconds, one per time step
    version: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)


# A shard index entry of all ones marks an inner chunk that was never written.
_SHARD_EMPTY_CHUNK = (2**64 - 1, 2**64 - 1)
_ZSTD = Zstd()


@dataclass(frozen=True)
class AlertShardReader:
    """Reads single cells of a sharded zarr v3 string array straight from its
    shard files on local disk.

    zarr-python spends ~1 ms per cell in its codec pipeline under the GIL; the
    same cell read directly (shard index, one inner chunk, zstd, vlen-utf8)
    takes ~0.03 ms. The reader handles exactly what the alert ingests write
    (``zarr.create_array`` with a string dtype, ``FINAL_CHUNK_SIZES["ALERTS"]``
    inner chunks inside ``FINAL_SHARD_SIZES["ALERTS"]`` shards): a 2-D array,
    ``sharding_indexed`` with the index at the end, ``vlen-utf8`` then ``zstd``
    inner codecs, ``bytes`` + ``crc32c`` index codecs and ``c/<i>/<j>`` chunk
    keys. ``from_metadata`` returns None for any other layout, and the store
    then stays on zarr-python.
    """

    path: str
    shard_shape: tuple[int, int]
    chunk_shape: tuple[int, int]
    fill_value: str

    @classmethod
    def from_metadata(cls, path: str, metadata: dict) -> AlertShardReader | None:
        if (
            metadata.get("zarr_format") != 3
            or metadata.get("data_type") != "string"
            or len(metadata.get("shape", ())) != 2
            or metadata.get("chunk_grid", {}).get("name") != "regular"
            or metadata.get("chunk_key_encoding", {}).get("name") != "default"
            or metadata["chunk_key_encoding"]
            .get("configuration", {})
            .get("separator", "/")
            != "/"
            or len(metadata.get("codecs", ())) != 1
            or metadata["codecs"][0].get("name") != "sharding_indexed"
        ):
            return None
        sharding = metadata["codecs"][0]["configuration"]
        inner_codecs = [codec.get("name") for codec in sharding.get("codecs", ())]
        index_codecs = [codec.get("name") for codec in sharding.get("index_codecs", ())]
        if (
            inner_codecs != ["vlen-utf8", "zstd"]
            or index_codecs != ["bytes", "crc32c"]
            or sharding["index_codecs"][0].get("configuration", {}).get("endian")
            != "little"
            or sharding.get("index_location") != "end"
        ):
            return None
        shard_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
        chunk_shape = tuple(sharding["chunk_shape"])
        if len(shard_shape) != 2 or len(chunk_shape) != 2:
            return None
        if any(s % c for s, c in zip(shard_shape, chunk_shape)):
            return None
        fill_value = metadata.get("fill_value", "")
        if not isinstance(fill_value, str):
            return None
        return cls(path, shard_shape, chunk_shape, fill_value)

    def read_cell(self, y: int, x: int) -> str:
        """The string at (y, x); the fill value where nothing was written."""
        shard_y, shard_x = self.shard_shape
        chunk_y, chunk_x = self.chunk_shape
        in_shard_y, in_shard_x = y % shard_y, x % shard_x
        chunks_per_row = shard_x // chunk_x
        chunk_index = (in_shard_y // chunk_y) * chunks_per_row + in_shard_x // chunk_x
        cell_index = (in_shard_y % chunk_y) * chunk_x + in_shard_x % chunk_x
        # The index holds (offset, nbytes) per inner chunk, then its crc32c.
        index_nbytes = (shard_y // chunk_y) * chunks_per_row * 16 + 4
        shard_file = os.path.join(self.path, "c", str(y // shard_y), str(x // shard_x))
        try:
            with open(shard_file, "rb") as handle:
                handle.seek(-index_nbytes, os.SEEK_END)
                index = handle.read(index_nbytes)
                if len(index) != index_nbytes:
                    raise ValueError(f"{shard_file}: truncated shard index")
                if google_crc32c.value(index[:-4]) != int.from_bytes(
                    index[-4:], "little"
                ):
                    raise ValueError(f"{shard_file}: shard index checksum mismatch")
                offset, nbytes = struct.unpack_from("<QQ", index, chunk_index * 16)
                if (offset, nbytes) == _SHARD_EMPTY_CHUNK:
                    return self.fill_value
                handle.seek(offset)
                compressed = handle.read(nbytes)
        except FileNotFoundError:
            return self.fill_value
        if len(compressed) != nbytes:
            raise ValueError(f"{shard_file}: truncated chunk {chunk_index}")
        # vlen-utf8: a little-endian uint32 item count, then each string as a
        # uint32 byte length followed by its UTF-8 bytes.
        data = bytes(_ZSTD.decode(compressed))
        (count,) = struct.unpack_from("<I", data, 0)
        if cell_index >= count:
            raise ValueError(f"{shard_file}: chunk {chunk_index} holds {count} items")
        position = 4
        for item in range(count):
            (length,) = struct.unpack_from("<I", data, position)
            position += 4
            if item == cell_index:
                return data[position : position + length].decode("utf-8")
            position += length
        raise ValueError(f"{shard_file}: chunk {chunk_index} ended early")


@dataclass(frozen=True)
class AlertStore:
    """One opened version of an alert grid (a zarr-python string array), read
    cell by cell through ``AlertShardReader`` when it is on local disk."""

    array: Any
    version: str
    reader: AlertShardReader | None = None

    def __getitem__(self, key):
        if self.reader is not None:
            try:
                return self.reader.read_cell(int(key[0]), int(key[1]))
            except (OSError, ValueError, struct.error) as exc:
                # A damaged shard: let zarr-python read (or report) it.
                _default_logger().warning(
                    "Direct alert read failed for %s%s, using zarr-python: %s",
                    self.version,
                    tuple(key),
                    exc,
                )
        return self.array[key]


@dataclass
class ZarrStores:
    ETOPO_f: Any | None = None
    SubH_Zarr: ModelStore | None = None
    HRRR_6H_Zarr: ModelStore | None = None
    GFS_Zarr: ModelStore | None = None
    URMA_Zarr: ModelStore | None = None
    ECMWF_Zarr: ModelStore | None = None
    NBM_Zarr: ModelStore | None = None
    GEFS_Zarr: ModelStore | None = None
    HRDPS_Zarr: ModelStore | None = None
    GDPS_Zarr: ModelStore | None = None
    GEPS_Zarr: ModelStore | None = None
    REPS_Zarr: ModelStore | None = None
    HRRR_Zarr: ModelStore | None = None
    NWS_Alerts_Zarr: AlertStore | None = None
    WMO_Alerts_Zarr: AlertStore | None = None
    RTMA_RU_Zarr: ModelStore | None = None
    ERA5_Data: Any | None = None
    DWD_MOSMIX_Zarr: ModelStore | None = None
    AIGFS_Zarr: ModelStore | None = None
    AIGEFS_Zarr: ModelStore | None = None
    ECMWF_AIFS_Zarr: ModelStore | None = None
    RAQDPS_Zarr: ModelStore | None = None
    SILAM_Zarr: ModelStore | None = None
    RAQDPS_LatLon: Any | None = None


@dataclass(frozen=True)
class StoreRoot:
    """Where a stage's stores are published: a tensorstore kvstore spec for the
    forecast stores and the matching location for zarr-python (alerts)."""

    kvstore: dict
    local_dir: str | None = None  # set for local disk; heartbeats go here
    alert_url: str | None = None  # remote base URL for the alert stores
    alert_storage_options: dict | None = None


def local_store_root(save_dir: str) -> StoreRoot:
    path = os.path.abspath(save_dir)
    return StoreRoot(kvstore={"driver": "file", "path": path + "/"}, local_dir=path)


def remote_store_root(save_type: str, s3_bucket: str) -> StoreRoot:
    """The published stores on S3: through the public files endpoint with the
    API key ("S3"), or straight from the bucket with AWS credentials from the
    environment ("S3Zarr")."""
    prefix = f"ForecastTar_v2/{INGEST_VERSION_STR}/"
    if save_type == "S3":
        headers = {"apikey": pw_api_key}
        return StoreRoot(
            kvstore={
                "driver": "http",
                "base_url": REMOTE_FILES_URL + prefix,
                "headers": [f"{name}: {value}" for name, value in headers.items()],
            },
            alert_url=REMOTE_FILES_URL + prefix,
            alert_storage_options={"client_kwargs": {"headers": headers}},
        )
    kvstore = {"driver": "s3", "bucket": s3_bucket, "path": prefix}
    if os.getenv("s3_endpoint"):
        kvstore["endpoint"] = os.environ["s3_endpoint"]
    return StoreRoot(
        kvstore=kvstore,
        alert_url=f"s3://{s3_bucket}/{prefix}",
        alert_storage_options={"anon": False},
    )


def _child_kvstore(root: StoreRoot, store_path: str) -> dict:
    return {**root.kvstore, "path": root.kvstore.get("path", "") + store_path + "/"}


async def resolve_version(root: StoreRoot, name: str) -> str | None:
    """The current version path of store ``name`` (``<MODEL>/<stamp>.zarr``),
    from its ``<MODEL>.current.json`` pointer, or None if it isn't published.

    A store without a pointer but with a plain ``<name>.zarr`` (ETOPO, or a
    store written locally by hand) is used as is.
    """
    kvstore = await ts.KvStore.open(root.kvstore)
    pointer = await kvstore.read(f"{name}.current.json")
    if pointer.state == "value":
        return json.loads(bytes(pointer.value))["store"]
    plain = await kvstore.read(f"{name}.zarr/zarr.json")
    if plain.state == "value":
        return f"{name}.zarr"
    return None


async def open_model_store(root: StoreRoot, version: str) -> ModelStore:
    """Open one version of a final store with tensorstore."""
    array = await ts.open(
        {"driver": "zarr3", "kvstore": _child_kvstore(root, version)},
        read=True,
        context=TS_CONTEXT,
        recheck_cached_data="open",
    )
    attributes = array.spec().to_json()["metadata"].get("attributes", {})
    if "time" not in attributes:
        raise ValueError(f"{version} has no time attribute")
    times = np.asarray(attributes["time"], dtype=np.float64)
    if times.shape != (array.shape[1],):
        raise ValueError(
            f"{version}: {times.shape} times for {array.shape[1]} time steps"
        )
    return ModelStore(array=array, times=times, version=version)


async def open_etopo(root: StoreRoot):
    """Open the static ETOPO grid with tensorstore (zarr v3 or v2)."""
    for driver in ("zarr3", "zarr"):
        try:
            return await ts.open(
                {
                    "driver": driver,
                    "kvstore": _child_kvstore(root, ETOPO_STORE_NAME + ".zarr"),
                },
                read=True,
                context=TS_CONTEXT,
            )
        except ValueError:
            continue
    raise ValueError(f"{ETOPO_STORE_NAME}.zarr is not a zarr array")


def open_alert_store(
    root: StoreRoot, version: str, *, logger: logging.Logger | None = None
) -> AlertStore:
    """Open one version of an alert grid with zarr-python (string data). On
    local disk the cells are then read directly from the shard files."""
    logger = logger or _default_logger()
    reader = None
    if root.local_dir is not None:
        path = os.path.join(root.local_dir, version)
        store = zarr.storage.LocalStore(path, read_only=True)
        with open(os.path.join(path, "zarr.json"), "rb") as handle:
            metadata = json.load(handle)
        reader = AlertShardReader.from_metadata(path, metadata)
        if reader is None:
            logger.warning(
                "%s is not in the sharded string layout the direct reader "
                "handles; reading it with zarr-python (~1 ms per cell)",
                version,
            )
    else:
        store = zarr.storage.FsspecStore.from_url(
            root.alert_url + version,
            storage_options=root.alert_storage_options,
            read_only=True,
        )
    return AlertStore(zarr.open_array(store, mode="r", zarr_format=3), version, reader)


def _store_names(stage: str) -> tuple[tuple, tuple]:
    if stage in ("TIMEMACHINE", "TM_TESTING"):
        return TIMEMACHINE_STORES, ()
    if stage in ("DEV", "PROD", "TESTING"):
        return FORECAST_STORES, ALERT_STORES
    return (), ()


async def refresh_stores(
    stores: ZarrStores,
    root: StoreRoot,
    stage: str,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Open any store version that is newer than the one held, then write this
    worker's heartbeat.

    A failed open keeps the version already held, so a bad publish degrades to
    serving the previous run rather than failing requests.
    """
    logger = logger or _default_logger()
    forecast_stores, alert_stores = _store_names(stage)

    async def refresh(attr: str, name: str, is_alert: bool) -> None:
        current = getattr(stores, attr)
        try:
            version = await resolve_version(root, name)
            if version is None:
                if current is None:
                    logger.info("%s not published", name)
                return
            if current is not None and current.version == version:
                return
            if is_alert:
                opened = await asyncio.to_thread(
                    open_alert_store, root, version, logger=logger
                )
            else:
                opened = await open_model_store(root, version)
        except (ValueError, OSError, KeyError, zarr.errors.BaseZarrError) as exc:
            logger.warning("Could not open %s, keeping the held version: %s", name, exc)
            return
        setattr(stores, attr, opened)
        logger.info("Opened %s", version)

    await asyncio.gather(
        *(refresh(attr, name, False) for attr, name in forecast_stores),
        *(refresh(attr, name, True) for attr, name in alert_stores),
    )
    if root.local_dir is not None:
        write_heartbeat(stores, root.local_dir)


def held_versions(stores: ZarrStores) -> list[str]:
    return [
        getattr(stores, attr).version
        for attr, _ in FORECAST_STORES + ALERT_STORES
        if getattr(stores, attr) is not None
    ]


def write_heartbeat(stores: ZarrStores, local_dir: str) -> None:
    """Record the versions this worker has open (the file's mtime is the
    heartbeat). Written via a temporary file so the sync never reads half of it."""
    readers_dir = os.path.join(local_dir, READERS_DIR_NAME)
    os.makedirs(readers_dir, exist_ok=True)
    path = os.path.join(readers_dir, f"{socket.gethostname()}-{os.getpid()}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump({"versions": held_versions(stores)}, fh)
    os.replace(tmp_path, path)


async def watch_stores(
    stores: ZarrStores,
    root: StoreRoot,
    stage: str,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Refresh the stores every ``STORE_POLL_SECONDS`` until cancelled."""
    while True:
        await asyncio.sleep(STORE_POLL_SECONDS)
        await refresh_stores(stores, root, stage, logger=logger)


async def load_stores(
    *,
    stage: str,
    save_dir: str,
    use_etopo: bool,
    save_type: str,
    s3_bucket: str,
    logger: logging.Logger | None = None,
) -> tuple[ZarrStores, StoreRoot]:
    """Open every store a stage reads; returns them with their root, which
    ``watch_stores`` keeps polling."""
    logger = logger or _default_logger()
    stores = ZarrStores()
    if stage in ("TESTING", "TM_TESTING") and save_type in ("S3", "S3Zarr"):
        root = remote_store_root(save_type, s3_bucket)
    elif stage in ("TESTING", "TM_TESTING"):
        root = local_store_root(s3_bucket)
    else:
        root = local_store_root(save_dir)

    # ETOPO is only read in PROD and DEV (responseLocal).
    if use_etopo and stage in ("DEV", "PROD"):
        try:
            stores.ETOPO_f = await open_etopo(root)
            logger.info("Loaded ETOPO")
        except ValueError as exc:
            logger.info("ETOPO not available: %s", exc)

    skip_era5 = str(os.getenv("SKIP_ERA5", "False")).lower() not in {
        "0",
        "false",
        "no",
    }
    if stage in ("DEV", "TIMEMACHINE", "TESTING", "TM_TESTING") and not skip_era5:
        era5_cache_dir = os.environ.get(
            "ERA5_CACHE_DIR", os.path.join(save_dir, "ERA5_cache")
        )
        stores.ERA5_Data = init_ERA5(era5_cache_dir)
        logger.info("ERA5 disk cache: %s", era5_cache_dir)
    elif stage in ("DEV", "TIMEMACHINE", "TESTING", "TM_TESTING"):
        logger.info("Skipping ERA5 initialization because SKIP_ERA5 is enabled")

    await refresh_stores(stores, root, stage, logger=logger)

    if stage in ("DEV", "PROD", "TESTING"):
        _load_aq_lat_lon_pickles(stores, logger=logger)

    logger.info("Zarr stores loaded")
    return stores, root


def _load_aq_lat_lon_pickles(stores: ZarrStores, *, logger: logging.Logger) -> None:
    """Load AQ model lat/lon pickle files when available."""
    for attr, fname in (("RAQDPS_LatLon", "RAQDPS.lat_lon.pickle"),):
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    setattr(stores, attr, pickle.load(fh))
                logger.info("Loaded %s from: %s", attr, path)
            except (OSError, EOFError, pickle.UnpicklingError) as exc:
                logger.warning("Could not load %s: %s", attr, exc)


def has_interior_nan_holes(arr: np.ndarray) -> tuple[bool, int | None]:
    """Detect an interior block of NaNs in a 2D array: the first row where a
    NaN run starts after the first column and one ends before the last."""
    mask = np.isnan(arr)
    if not mask.any():
        return False, None
    run_starts = mask[:, 1:] & ~mask[:, :-1]
    run_ends = ~mask[:, 1:] & mask[:, :-1]
    matching_rows = np.flatnonzero(run_starts.any(axis=1) & run_ends.any(axis=1))
    if matching_rows.size:
        return True, int(matching_rows[0])
    return False, None


def _interp_row(row: np.ndarray) -> np.ndarray:
    """Fill only strictly interior NaN-runs in a 1D array."""
    n = row.size
    x = np.arange(n)
    mask = np.isnan(row)
    if mask.any() and not mask.all():
        good = ~mask
        row[mask] = np.interp(
            x[mask], x[good], row[good], left=MISSING_DATA, right=MISSING_DATA
        )
    return row


def with_time_column(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Turn a ``(variable, time)`` read into the ``(time, 1 + variable)``
    matrix the API works with, valid times in column 0.

    Time is NaN on rows where every variable is NaN. When time was stored as a
    variable it was NaN exactly there: chunk padding, times outside an
    interpolated model's range, and DWD cells without a station. The API's
    coverage checks (``np.isfinite(data[:, 0])``) depend on that.
    """
    data = values.T
    time_column = np.where(np.isnan(data).all(axis=1), np.nan, times)
    return np.column_stack((time_column, data))


class WeatherParallel:
    """Point reads for one API request, issued concurrently."""

    def __init__(
        self,
        loc_tag: str = "",
        *,
        logger: logging.Logger | None = None,
        timing_enabled: bool = False,
    ):
        self.loc_tag = loc_tag
        self.logger = logger or _default_logger()
        self.timing_enabled = timing_enabled
        # Tensorstore futures issued since the last loop iteration, each with
        # the asyncio future its reader awaits (see _await_read)
        self._read_batch: list[tuple[Any, asyncio.Future]] | None = None

    def _await_read(self, future) -> asyncio.Future:
        """Await a tensorstore read future.

        Awaiting a tensorstore future directly costs ~120 us of event-loop
        plumbing each: a 19-store request spent 1.7 ms awaiting 0.5 ms of
        reads. The futures issued in one loop iteration (the request's gather
        issues all of its reads in one) are instead waited on together in a
        worker thread, which wakes the loop once when every read is done. The
        loop stays free for other requests meanwhile, which matters when the
        stores are remote.
        """
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        if self._read_batch is None:
            self._read_batch = []
            # Runs after the callbacks already queued for this iteration, i.e.
            # after every reader started by the same gather has issued its read
            loop.call_soon(self._wait_for_batch, loop)
        self._read_batch.append((future, waiter))
        return waiter

    def _wait_for_batch(self, loop: asyncio.AbstractEventLoop) -> None:
        batch, self._read_batch = self._read_batch, None

        def wait_all():
            # Each outcome is handed to its own reader, which handles it there;
            # nothing is swallowed here.
            outcomes = []
            for future, _ in batch:
                try:
                    outcomes.append((future.result(), None))
                except Exception as exc:  # noqa: BLE001 - forwarded to the reader
                    outcomes.append((None, exc))
            return outcomes

        def deliver(done: asyncio.Future) -> None:
            try:
                outcomes = done.result()
            except Exception as exc:  # noqa: BLE001 - the executor itself failed
                outcomes = [(None, exc)] * len(batch)
            for (_, waiter), (value, exc) in zip(batch, outcomes):
                if waiter.done():
                    continue
                if exc is None:
                    waiter.set_result(value)
                else:
                    waiter.set_exception(exc)

        loop.run_in_executor(None, wait_all).add_done_callback(deliver)

    async def zarr_read(self, model, store: ModelStore | None, x, y):
        """Every variable and time at grid cell (y, x) as a (time, 1 + variable)
        matrix, or False if the store is missing or unreadable."""
        if store is None:
            self.logger.error("### %s store not loaded! %s", model, self.loc_tag)
            return False
        if self.timing_enabled:
            self.logger.debug("### %s Reading! %s", model, self.loc_tag)
        for _ in range(MAX_ZARR_READ_RETRIES):
            try:
                values = await self._await_read(store.array[:, :, y, x].read())
            except (ValueError, OSError, IndexError):
                self.logger.exception("### %s Failure! %s", model, self.loc_tag)
                continue
            data_out = with_time_column(values, store.times)

            has_missing_data, missing_row = has_interior_nan_holes(data_out.T)
            if has_missing_data:
                if model in {"DWD_MOSMIX", "URMA"}:
                    # Preserve missing observations so downstream source selection
                    # can use another model for those hours.
                    self.logger.debug(
                        "### %s Has missing data (row %s), leaving NaN for merge fallback",
                        model,
                        missing_row,
                    )
                else:
                    self.logger.warning(
                        "### %s Interpolating missing data (row %s)!",
                        model,
                        missing_row,
                    )
                    data_out = np.apply_along_axis(_interp_row, 0, data_out)

            if self.timing_enabled:
                self.logger.debug("### %s Done! %s", model, self.loc_tag)
            return data_out

        self.logger.error("### %s Failure! %s", model, self.loc_tag)
        return False

    async def zarr_read_max_square(
        self, model, store: ModelStore | None, x, y, square_size=3
    ):
        """Like ``zarr_read``, but the maximum of each variable over a
        ``square_size`` square around (y, x) (air-quality plumes are narrow)."""
        if store is None:
            self.logger.error("### %s store not loaded! %s", model, self.loc_tag)
            return False
        if self.timing_enabled:
            self.logger.debug("### %s Reading max square! %s", model, self.loc_tag)
        radius = square_size // 2
        y_start = max(0, y - radius)
        y_end = min(store.shape[-2], y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(store.shape[-1], x + radius + 1)
        for _ in range(MAX_ZARR_READ_RETRIES):
            try:
                square = await self._await_read(
                    store.array[:, :, y_start:y_end, x_start:x_end].read()
                )
            except (ValueError, OSError, IndexError):
                self.logger.exception("### %s Failure! %s", model, self.loc_tag)
                continue
            with np.errstate(all="ignore"):
                values = np.nanmax(square, axis=(-2, -1))
            data_out = with_time_column(values, store.times)

            has_missing_data, missing_row = has_interior_nan_holes(data_out.T)
            if has_missing_data:
                self.logger.warning(
                    "### %s Interpolating missing data (row %s)!",
                    model,
                    missing_row,
                )
                data_out = np.apply_along_axis(_interp_row, 0, data_out)

            if self.timing_enabled:
                self.logger.debug("### %s Done! %s", model, self.loc_tag)
            return data_out

        self.logger.error("### %s Failure! %s", model, self.loc_tag)
        return False
