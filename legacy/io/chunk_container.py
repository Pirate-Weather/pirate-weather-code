"""Immutable single-file chunk container for the published Zarr stores.

A published store is tens of thousands of small chunk files (HRRR 76,321, NBM
416,807). Each generation of a store is instead written once as two files:

    <generation>.data   every chunk's bytes, concatenated in key order
    <generation>.index  key -> (offset, length, crc32), loaded whole into RAM

and a ``CURRENT`` file names the generation readers should open. A point read
is a dict lookup plus one ``os.pread``: no per-chunk open/stat/close and no
directory lookups, while the OS page cache still keeps hot chunks in memory.

Generations are never modified. A new run writes a new generation, then
replaces ``CURRENT`` with ``os.replace``; a reader object stays on the
generation it opened, so requests in flight finish on the old files while new
requests open the new ones.

Index layout (little endian)::

    header  magic b"PWCHUNK1", u32 version, u32 entries, u64 data size,
            u64 key blob length
    keys    UTF-8 keys joined with "\\n" (Zarr keys never contain newlines)
    arrays  u64 offsets[entries], u32 lengths[entries], u32 crc32[entries]
    trailer u32 crc32 of everything above
"""

from __future__ import annotations

import datetime
import logging
import os
import struct
import time
import zlib
from array import array
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

logger = logging.getLogger(__name__)

INDEX_MAGIC = b"PWCHUNK1"
INDEX_VERSION = 1
INDEX_HEADER = struct.Struct("<8sIIQQ")
INDEX_TRAILER = struct.Struct("<I")
CURRENT_FILE = "CURRENT"


class CorruptContainerError(ValueError):
    """The index or data file does not match what the index records."""


def new_generation_id() -> str:
    """UTC timestamp to the microsecond, so generations sort by age."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_generation(
    root: Path | str,
    items: Iterable[tuple[str, bytes]],
    generation: str | None = None,
) -> str:
    """Write one complete generation from ``(key, bytes)`` pairs.

    Both files are written under ``.tmp`` names, fsynced, and only then renamed
    to their final names, so a crash leaves no half-written generation behind a
    final name. The generation is not visible to readers until
    ``publish_generation`` points ``CURRENT`` at it.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation = generation or new_generation_id()
    data_path = root / f"{generation}.data"
    index_path = root / f"{generation}.index"
    data_tmp = data_path.with_name(data_path.name + ".tmp")
    index_tmp = index_path.with_name(index_path.name + ".tmp")

    keys: list[str] = []
    offsets = array("Q")
    lengths = array("I")
    crcs = array("I")
    offset = 0
    t0 = time.perf_counter()
    with open(data_tmp, "wb") as data_file:
        for key, value in items:
            if "\n" in key:
                raise ValueError(f"key contains a newline: {key!r}")
            data_file.write(value)
            keys.append(key)
            offsets.append(offset)
            lengths.append(len(value))
            crcs.append(zlib.crc32(value))
            offset += len(value)
        t_write = time.perf_counter()
        data_file.flush()
        os.fsync(data_file.fileno())
    t_data_fsync = time.perf_counter()

    key_blob = "\n".join(keys).encode()
    body = b"".join(
        (
            INDEX_HEADER.pack(
                INDEX_MAGIC, INDEX_VERSION, len(keys), offset, len(key_blob)
            ),
            key_blob,
            offsets.tobytes(),
            lengths.tobytes(),
            crcs.tobytes(),
        )
    )
    with open(index_tmp, "wb") as index_file:
        index_file.write(body)
        index_file.write(INDEX_TRAILER.pack(zlib.crc32(body)))
        index_file.flush()
        os.fsync(index_file.fileno())
    t_index = time.perf_counter()

    os.replace(data_tmp, data_path)
    os.replace(index_tmp, index_path)
    _fsync_dir(root)
    t_done = time.perf_counter()
    logger.info(
        "Wrote generation %s: %d chunks, %.1f MiB; data %.2f s + fsync %.2f s, "
        "index %.2f s, rename %.3f s",
        generation,
        len(keys),
        offset / 2**20,
        t_write - t0,
        t_data_fsync - t_write,
        t_index - t_data_fsync,
        t_done - t_index,
    )
    return generation


def publish_generation(root: Path | str, generation: str) -> None:
    """Atomically point ``CURRENT`` at a fully written generation."""
    root = Path(root)
    for suffix in (".data", ".index"):
        if not (root / f"{generation}{suffix}").exists():
            raise FileNotFoundError(root / f"{generation}{suffix}")
    current_tmp = root / f"{CURRENT_FILE}.tmp"
    with open(current_tmp, "w") as fh:
        fh.write(generation)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(current_tmp, root / CURRENT_FILE)
    _fsync_dir(root)


def read_current_generation(root: Path | str) -> str:
    return (Path(root) / CURRENT_FILE).read_text().strip()


def pack_directory_store(
    src_dir: Path | str, root: Path | str, generation: str | None = None
) -> str:
    """Write a local directory Zarr store as a new (unpublished) generation."""
    src_dir = Path(src_dir)
    keys = sorted(
        path.relative_to(src_dir).as_posix()
        for path in src_dir.rglob("*")
        if path.is_file()
    )
    return write_generation(
        root, ((key, (src_dir / key).read_bytes()) for key in keys), generation
    )


class ChunkContainer:
    """Read-only view of one generation; it never changes generation.

    ``close()`` only stops new reads. The file descriptor is closed when the
    object is garbage collected, so a read still running in another thread can
    never ``pread`` a descriptor number that was closed and reused for some
    other file. Swap generations by replacing the reference and dropping the
    old one.
    """

    def __init__(self, root: Path | str, generation: str) -> None:
        self.root = Path(root)
        self.generation = generation
        index_bytes = (self.root / f"{generation}.index").read_bytes()
        self._load_index(index_bytes)
        self._fd = os.open(self.root / f"{generation}.data", os.O_RDONLY)
        data_size = os.fstat(self._fd).st_size
        if data_size != self.data_size:
            os.close(self._fd)
            self._fd = None
            raise CorruptContainerError(
                f"{generation}.data is {data_size} bytes; index expects {self.data_size}"
            )
        self.closed = False

    def _load_index(self, raw: bytes) -> None:
        name = f"{self.generation}.index"
        if len(raw) < INDEX_HEADER.size + INDEX_TRAILER.size:
            raise CorruptContainerError(f"{name} is truncated")
        body, trailer = raw[: -INDEX_TRAILER.size], raw[-INDEX_TRAILER.size :]
        if zlib.crc32(body) != INDEX_TRAILER.unpack(trailer)[0]:
            raise CorruptContainerError(f"{name} checksum mismatch")
        magic, version, entries, data_size, key_blob_len = INDEX_HEADER.unpack_from(
            body
        )
        if magic != INDEX_MAGIC or version != INDEX_VERSION:
            raise CorruptContainerError(f"{name} is not a version-1 chunk index")
        pos = INDEX_HEADER.size
        keys = body[pos : pos + key_blob_len].decode().split("\n") if entries else []
        pos += key_blob_len
        self._offsets = array("Q", body[pos : pos + 8 * entries])
        pos += 8 * entries
        self._lengths = array("I", body[pos : pos + 4 * entries])
        pos += 4 * entries
        self._crcs = array("I", body[pos : pos + 4 * entries])
        if len(keys) != entries or len(self._crcs) != entries:
            raise CorruptContainerError(f"{name} entry count mismatch")
        # key -> position in the offset/length arrays: lighter than a tuple
        # per key, which matters with ~20 stores open in every API worker.
        self._positions = {key: i for i, key in enumerate(keys)}
        self.data_size = data_size

    def __del__(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)

    def __len__(self) -> int:
        return len(self._positions)

    def __contains__(self, key: str) -> bool:
        return key in self._positions

    def keys(self) -> Iterator[str]:
        return iter(self._positions)

    def locate(self, key: str) -> tuple[int, int] | None:
        """(offset, length) of a key in the data file, or None if absent."""
        pos = self._positions.get(key)
        if pos is None:
            return None
        return self._offsets[pos], self._lengths[pos]

    def get(self, key: str) -> bytes | None:
        if self.closed:
            raise ValueError("container is closed")
        pos = self._positions.get(key)
        if pos is None:
            return None
        return os.pread(self._fd, self._lengths[pos], self._offsets[pos])

    def get_many(self, keys: Iterable[str]) -> list[bytes | None]:
        return [self.get(key) for key in keys]

    def get_range(self, key: str, byte_range: ByteRequest | None) -> bytes | None:
        located = self.locate(key)
        if located is None:
            return None
        if self.closed:
            raise ValueError("container is closed")
        offset, length = located
        if byte_range is None:
            start, stop = 0, length
        elif isinstance(byte_range, RangeByteRequest):
            start, stop = byte_range.start, min(byte_range.end, length)
        elif isinstance(byte_range, OffsetByteRequest):
            start, stop = byte_range.offset, length
        elif isinstance(byte_range, SuffixByteRequest):
            start, stop = max(0, length - byte_range.suffix), length
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")
        return os.pread(self._fd, max(0, stop - start), offset + start)

    def verify(self) -> None:
        """Check every chunk against its stored crc32 (a full read: run it at
        publish or sync time, not on the request path)."""
        for key, pos in self._positions.items():
            value = os.pread(self._fd, self._lengths[pos], self._offsets[pos])
            if len(value) != self._lengths[pos] or zlib.crc32(value) != self._crcs[pos]:
                raise CorruptContainerError(
                    f"{self.generation}: chunk {key} is corrupt"
                )

    def close(self) -> None:
        self.closed = True


def open_current(root: Path | str) -> ChunkContainer:
    root = Path(root)
    return ChunkContainer(root, read_current_generation(root))


class ContainerStore(Store):
    """Read-only Zarr store over one ``ChunkContainer`` generation.

    Reads run inline on zarr's event loop thread rather than through
    ``asyncio.to_thread`` as LocalStore does: a warm ``pread`` takes a few
    microseconds, less than the thread hand-off.
    """

    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False

    def __init__(self, container: ChunkContainer) -> None:
        super().__init__(read_only=True)
        self.container = container

    @classmethod
    def open_current(cls, root: Path | str) -> ContainerStore:
        return cls(open_current(root))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.container.root == other.container.root
            and self.container.generation == other.container.generation
        )

    def __str__(self) -> str:
        return f"container://{self.container.root}@{self.container.generation}"

    def __repr__(self) -> str:
        return f"ContainerStore('{self}')"

    def close(self) -> None:
        super().close()
        self.container.close()

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        value = (
            self.container.get(key)
            if byte_range is None
            else self.container.get_range(key, byte_range)
        )
        return None if value is None else prototype.buffer.from_bytes(value)

    # Synchronous IO surface: lets zarr's FusedCodecPipeline read chunks
    # without one coroutine per chunk (it needs all three methods).
    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        value = (
            self.container.get(key)
            if byte_range is None
            else self.container.get_range(key, byte_range)
        )
        if value is None:
            return None
        return (prototype or default_buffer_prototype()).buffer.from_bytes(value)

    def set_sync(self, key: str, value: Buffer) -> None:
        self._check_writable()

    def delete_sync(self, key: str) -> None:
        self._check_writable()

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype, byte_range) for key, byte_range in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        return key in self.container

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()

    async def delete(self, key: str) -> None:
        self._check_writable()

    async def list(self) -> AsyncIterator[str]:
        for key in self.container:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self.container:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        if prefix:
            prefix += "/"
        children = dict.fromkeys(
            key.removeprefix(prefix).split("/")[0]
            for key in self.container
            if key.startswith(prefix)
        )
        for child in children:
            yield child
