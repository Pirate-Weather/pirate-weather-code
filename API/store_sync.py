"""Copy the published model stores from S3 to the API's local disk.

Ingests publish each run as ``<MODEL>/<stamp>.zarr`` and then rewrite the
pointer ``<MODEL>.current.json`` (``API.ingest_utils.publish_store``). Every
``SYNC_INTERVAL`` seconds this copies any version the local disk lacks into the
same layout under ``BASE_DIR`` and then replaces the local pointer, so an API
worker never sees a half-copied store. Sharded stores are a few hundred files
each, so a whole model copies in seconds.

The API workers (``API.io.store_reader``) poll the local pointers, open new
versions and write heartbeat files under ``BASE_DIR/.readers`` listing the
versions they hold. An old version is deleted only once its pointer has moved
on for ``SUPERSEDED_GRACE_SECONDS`` and no fresh heartbeat lists it: deleting a
store under a reader makes its reads silently return NaN.

Run it next to the API with the same volume: ``python -m API.store_sync``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

import obstore
import obstore.store

from API.io.store_reader import (
    ALERT_STORES,
    ETOPO_STORE_NAME,
    FORECAST_STORES,
    READERS_DIR_NAME,
)

logger = logging.getLogger(__name__)


def positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


BASE_DIR = os.getenv("BASE_DIR", "/mnt/nvme/data/ProdTest3")
# bucket/prefix the ingests publish to (their forecast_path + ingest version)
REMOTE_PATH = os.getenv("REMOTE_PATH", "piratezarr2/ForecastTar_v2/v31")
SAVE_TYPE = os.getenv("save_type", "S3")
SYNC_INTERVAL = positive_int_env("SYNC_INTERVAL", 300)
DOWNLOAD_CONCURRENCY = positive_int_env("SYNC_CONCURRENCY", 32)
# A worker polls every STORE_POLL_SECONDS (10 s by default); a heartbeat this
# old belongs to a worker that has stopped.
READER_STALE_SECONDS = positive_int_env("READER_STALE_SECONDS", 60)
# Time for every worker to pick up a new version before the old one may go,
# even if a worker started too recently to have written a heartbeat.
SUPERSEDED_GRACE_SECONDS = positive_int_env("SUPERSEDED_GRACE_SECONDS", 120)

STORE_NAMES = tuple(name for _, name in FORECAST_STORES + ALERT_STORES)
# Side files the API loads at startup
SUPPORT_FILES = ("DWD_MOSMIX_stations.pickle", "RAQDPS.lat_lon.pickle")
READY_FILE = "models_ready"


def _download_files(remote, keys: list[str], local_dir: str, prefix: str) -> None:
    """Download ``keys`` (all under ``prefix``) into ``local_dir``,
    ``DOWNLOAD_CONCURRENCY`` at a time. obstore retries each request itself."""

    async def get_all() -> None:
        semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

        async def get(key: str) -> None:
            local_file = os.path.join(local_dir, key.removeprefix(prefix))
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            async with semaphore:
                result = await obstore.get_async(remote, key)
                data = await result.bytes_async()
            await asyncio.to_thread(Path(local_file).write_bytes, data)

        await asyncio.gather(*(get(key) for key in keys))

    asyncio.run(get_all())


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp_path, path)


def sync_store(remote, name: str, base_dir: str) -> bool:
    """Bring store ``name`` up to the published version; True if it is current."""
    pointer_name = f"{name}.current.json"
    try:
        pointer = json.loads(bytes(obstore.get(remote, pointer_name).bytes()))
    except (FileNotFoundError, obstore.exceptions.NotFoundError):
        logger.warning("%s is not published", name)
        return False
    version = pointer["store"]
    local_version = os.path.join(base_dir, version)

    if not os.path.isdir(local_version):
        started = time.time()
        partial = local_version + ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        keys = [
            meta["path"]
            for batch in obstore.list(remote, prefix=f"{version}/")
            for meta in batch
        ]
        if not any(key == f"{version}/zarr.json" for key in keys):
            # publish_store uploads the metadata last, so this is a store the
            # pointer should never name; leave the local copy where it is.
            logger.error("%s has no zarr.json; not syncing it", version)
            return False
        _download_files(remote, keys, partial, f"{version}/")
        os.rename(partial, local_version)
        logger.info(
            "Copied %s (%d files) in %.1f s", version, len(keys), time.time() - started
        )

    local_pointer = os.path.join(base_dir, pointer_name)
    try:
        with open(local_pointer) as fh:
            current_local = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        current_local = None
    if current_local != pointer:
        _write_json_atomic(local_pointer, pointer)
        logger.info("%s now points at %s", name, version)
    return True


def sync_file(remote, name: str, base_dir: str) -> bool:
    """Copy a side file if its size or modification time changed."""
    try:
        meta = obstore.head(remote, name)
    except (FileNotFoundError, obstore.exceptions.NotFoundError):
        logger.warning("Support file %s is not published", name)
        return False
    local_file = os.path.join(base_dir, name)
    remote_mtime = meta["last_modified"].timestamp()
    if (
        os.path.exists(local_file)
        and os.path.getsize(local_file) == meta["size"]
        and os.path.getmtime(local_file) >= remote_mtime
    ):
        return True
    tmp_file = local_file + ".tmp"
    with open(tmp_file, "wb") as fh:
        fh.write(bytes(obstore.get(remote, name).bytes()))
    os.replace(tmp_file, local_file)
    logger.info("Updated %s", name)
    return True


def sync_static_store(remote, name: str, base_dir: str) -> bool:
    """Copy an unversioned store (ETOPO) once."""
    local_store = os.path.join(base_dir, f"{name}.zarr")
    if os.path.isdir(local_store):
        return True
    prefix = f"{name}.zarr/"
    keys = [
        meta["path"] for batch in obstore.list(remote, prefix=prefix) for meta in batch
    ]
    if not keys:
        logger.warning("Static store %s is not published", name)
        return False
    partial = local_store + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    _download_files(remote, keys, partial, prefix)
    os.rename(partial, local_store)
    logger.info("Copied static store %s", name)
    return True


def versions_in_use(base_dir: str, now: float) -> set[str]:
    """Versions listed by a live worker's heartbeat. Heartbeats from stopped
    workers are removed."""
    readers_dir = os.path.join(base_dir, READERS_DIR_NAME)
    in_use: set[str] = set()
    if not os.path.isdir(readers_dir):
        return in_use
    for entry in os.scandir(readers_dir):
        if not entry.name.endswith(".json"):
            continue
        if now - entry.stat().st_mtime > READER_STALE_SECONDS:
            os.remove(entry.path)
            continue
        try:
            with open(entry.path) as fh:
                in_use.update(json.load(fh)["versions"])
        except (OSError, json.JSONDecodeError, KeyError):
            # A half-written or vanished heartbeat: keep everything this round.
            return {"*"}
    return in_use


def remove_old_versions(base_dir: str, now: float | None = None) -> None:
    """Delete local versions that are no longer current and no reader holds."""
    now = time.time() if now is None else now
    in_use = versions_in_use(base_dir, now)
    if "*" in in_use:
        return
    for name in STORE_NAMES:
        local_pointer = os.path.join(base_dir, f"{name}.current.json")
        model_dir = os.path.join(base_dir, name)
        if not os.path.exists(local_pointer) or not os.path.isdir(model_dir):
            continue
        if now - os.path.getmtime(local_pointer) < SUPERSEDED_GRACE_SECONDS:
            continue
        with open(local_pointer) as fh:
            current = json.load(fh)["store"]
        for entry in os.scandir(model_dir):
            version = f"{name}/{entry.name}"
            if version == current or version in in_use:
                continue
            shutil.rmtree(entry.path)
            logger.info("Deleted old version %s", version)


def sync_once(remote, base_dir: str) -> bool:
    """One pass over every store and side file; True if all are current."""
    all_ok = True
    for name in STORE_NAMES:
        all_ok &= sync_store(remote, name, base_dir)
    for name in SUPPORT_FILES:
        all_ok &= sync_file(remote, name, base_dir)
    all_ok &= sync_static_store(remote, ETOPO_STORE_NAME, base_dir)
    remove_old_versions(base_dir)
    return all_ok


def open_remote(path: str, save_type: str):
    """The published stores: ``bucket/prefix`` on S3 (``s3_endpoint`` and
    ``s3_region`` select a non-AWS store such as OVH; keys from ``AWS_KEY`` and
    ``AWS_SECRET`` or obstore's ``AWS_*`` variables), or a local directory.
    Same rules as ``API.ingest_utils.open_remote``, which the API image cannot
    import (it pulls in the GRIB stack)."""
    if save_type != "S3":
        return obstore.store.LocalStore(path)
    bucket, _, prefix = path.strip("/").partition("/")
    options = {
        "endpoint": os.getenv("s3_endpoint"),
        "region": os.getenv("s3_region"),
        "access_key_id": os.getenv("AWS_KEY"),
        "secret_access_key": os.getenv("AWS_SECRET"),
    }
    return obstore.store.S3Store(
        bucket,
        prefix=prefix or None,
        **{name: value for name, value in options.items() if value},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    os.makedirs(BASE_DIR, exist_ok=True)
    ready_file = os.path.join(BASE_DIR, READY_FILE)
    remote = open_remote(REMOTE_PATH, SAVE_TYPE)
    while True:
        started = time.time()
        try:
            ready = sync_once(remote, BASE_DIR)
        except (OSError, obstore.exceptions.BaseError):
            # A network or disk failure skips this pass; the next one retries.
            logger.exception("Sync pass failed")
            ready = False
        # The ready flag marks a host whose stores are all present and current
        if ready:
            open(ready_file, "a").close()
        elif os.path.exists(ready_file):
            os.remove(ready_file)
        logger.info(
            "Sync pass finished in %.1f s (ready: %s)", time.time() - started, ready
        )
        time.sleep(max(0, SYNC_INTERVAL - (time.time() - started)))


if __name__ == "__main__":
    main()
