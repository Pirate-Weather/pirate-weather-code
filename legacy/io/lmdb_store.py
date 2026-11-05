"""Zarr v3 store backed by a single LMDB file.

A published store is tens of thousands of ~8 KiB chunk files (HRRR: 76,321),
and uploading or syncing that many objects dominates publishing. LMDB keeps
the exact Zarr layout (one key per chunk, so a point read is still one ~8 KiB
value) in one memory-mapped file: one object to upload or copy, and a read is
a B-tree lookup in a shared mmap with no per-chunk open()/close().

Port of zarr v2's LMDBStore to the zarr 3 Store API. LMDB allows one writer at
a time, so ingests write the usual directory store in parallel and then pack it
with ``pack_directory_store``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import lmdb
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

# Address space reserved for a writable environment; LMDB only uses what it
# writes, and read-only opens map just the existing file.
WRITE_MAP_SIZE = 2**40


def _slice(value: bytes | memoryview, byte_range: ByteRequest | None):
    if byte_range is None:
        return value
    if isinstance(byte_range, RangeByteRequest):
        return value[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return value[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return value[max(0, len(value) - byte_range.suffix) :]
    raise TypeError(f"Unexpected byte_range, got {byte_range}.")


class LMDBStore(Store):
    """Zarr store in one LMDB file (``subdir=False``).

    Read-only opens skip LMDB's lock file (``lock=False``): the published file
    is immutable, and it lets the store sit on a read-only path. Readahead is
    off because the API does random point reads, one chunk each.
    """

    supports_writes = True
    supports_deletes = True
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, path: Path | str, *, read_only: bool = True) -> None:
        super().__init__(read_only=read_only)
        self.path = Path(path)

    def _sync_open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")
        if self.read_only:
            self._env = lmdb.open(
                str(self.path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                max_spare_txns=os.cpu_count() or 1,
            )
        else:
            # sync=False: one fsync at close instead of one per commit.
            self._env = lmdb.open(
                str(self.path),
                subdir=False,
                map_size=WRITE_MAP_SIZE,
                readahead=False,
                sync=False,
                metasync=False,
            )
        self._write_lock = threading.Lock()
        self._is_open = True

    async def _open(self) -> None:
        self._sync_open()

    def _env_open(self) -> lmdb.Environment:
        if not self._is_open:
            self._sync_open()
        return self._env

    def close(self) -> None:
        if not self._is_open:
            return
        super().close()
        if not self.read_only:
            self._env.sync(True)
        self._env.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_env", None)
        state.pop("_write_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        self._is_open = False

    def __str__(self) -> str:
        return f"lmdb://{self.path}"

    def __repr__(self) -> str:
        return f"LMDBStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.path == other.path

    def _get(
        self,
        txn: lmdb.Transaction,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None,
    ) -> Buffer | None:
        value = txn.get(key.encode())
        if value is None:
            return None
        # Copy out of the mmap before the transaction ends.
        return prototype.buffer.from_bytes(bytes(_slice(value, byte_range)))

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        with self._env_open().begin(buffers=True) as txn:
            return self._get(txn, key, prototype, byte_range)

    # Synchronous IO surface: lets zarr's FusedCodecPipeline read chunks
    # without one coroutine per chunk (it needs all three methods).
    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        with self._env_open().begin(buffers=True) as txn:
            return self._get(
                txn, key, prototype or default_buffer_prototype(), byte_range
            )

    def set_sync(self, key: str, value: Buffer) -> None:
        self._check_writable()
        env = self._env_open()
        with self._write_lock, env.begin(write=True) as txn:
            txn.put(key.encode(), value.to_bytes())

    def delete_sync(self, key: str) -> None:
        self._check_writable()
        env = self._env_open()
        with self._write_lock, env.begin(write=True) as txn:
            txn.delete(key.encode())

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        with self._env_open().begin(buffers=True) as txn:
            return [
                self._get(txn, key, prototype, byte_range)
                for key, byte_range in key_ranges
            ]

    async def exists(self, key: str) -> bool:
        with self._env_open().begin(buffers=True) as txn:
            return txn.get(key.encode()) is not None

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        env = self._env_open()
        with self._write_lock, env.begin(write=True) as txn:
            txn.put(key.encode(), value.to_bytes())

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        env = self._env_open()
        with self._write_lock, env.begin(write=True) as txn:
            txn.put(key.encode(), value.to_bytes(), overwrite=False)

    async def delete(self, key: str) -> None:
        self._check_writable()
        env = self._env_open()
        with self._write_lock, env.begin(write=True) as txn:
            txn.delete(key.encode())

    def _keys(self, prefix: str = "") -> list[str]:
        encoded = prefix.encode()
        keys = []
        with self._env_open().begin() as txn, txn.cursor() as cursor:
            if not cursor.set_range(encoded):
                return keys
            for key in cursor.iternext(keys=True, values=False):
                if not key.startswith(encoded):
                    break
                keys.append(key.decode())
        return keys

    async def list(self) -> AsyncIterator[str]:
        for key in self._keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self._keys(prefix):
            yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        if prefix:
            prefix += "/"
        children = dict.fromkeys(
            key.removeprefix(prefix).split("/")[0] for key in self._keys(prefix)
        )
        for child in children:
            yield child


def pack_directory_store(src_dir: Path | str, dst_path: Path | str) -> None:
    """Copy a local directory Zarr store into a new LMDB file.

    Keys are written in sorted order in one transaction with ``append=True``,
    which fills B-tree pages sequentially instead of splitting them.
    """
    src_dir = Path(src_dir)
    keys = sorted(
        path.relative_to(src_dir).as_posix()
        for path in src_dir.rglob("*")
        if path.is_file()
    )
    dst_path = Path(dst_path)
    dst_path.unlink(missing_ok=True)
    env = lmdb.open(
        str(dst_path),
        subdir=False,
        map_size=WRITE_MAP_SIZE,
        readahead=False,
        sync=False,
        metasync=False,
    )
    try:
        with env.begin(write=True) as txn:
            for key in keys:
                txn.put(
                    key.encode(),
                    (src_dir / key).read_bytes(),
                    append=True,
                )
        env.sync(True)
    finally:
        env.close()
    # Drop the lock file LMDB creates next to a writable environment.
    Path(f"{dst_path}-lock").unlink(missing_ok=True)
