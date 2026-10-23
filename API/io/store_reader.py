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
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorstore as ts
import zarr

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


@dataclass(frozen=True)
class AlertStore:
    """One opened version of an alert grid (a zarr-python string array)."""

    array: Any
    version: str

    def __getitem__(self, key):
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


def open_alert_store(root: StoreRoot, version: str) -> AlertStore:
    """Open one version of an alert grid with zarr-python (string data)."""
    if root.local_dir is not None:
        store = zarr.storage.LocalStore(
            os.path.join(root.local_dir, version), read_only=True
        )
    else:
        store = zarr.storage.FsspecStore.from_url(
            root.alert_url + version,
            storage_options=root.alert_storage_options,
            read_only=True,
        )
    return AlertStore(zarr.open_array(store, mode="r", zarr_format=3), version)


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
                opened = await asyncio.to_thread(open_alert_store, root, version)
            else:
                opened = await open_model_store(root, version)
        except (ValueError, OSError, KeyError, zarr.errors.ZarrError) as exc:
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
    """Detect an interior block of NaNs in a 2D array."""
    mask = np.isnan(arr)
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    diff = padded[:, 1:].astype(int) - padded[:, :-1].astype(int)
    starts = diff == 1
    ends = diff == -1
    interior_starts = starts[:, 1:-1]
    interior_ends = ends[:, 1:-1]
    row_has_start = interior_starts.any(axis=1)
    row_has_end = interior_ends.any(axis=1)
    matching_rows = np.flatnonzero(row_has_start & row_has_end)
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
                values = await store.array[:, :, y, x].read()
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
                square = await store.array[:, :, y_start:y_end, x_start:x_end].read()
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
