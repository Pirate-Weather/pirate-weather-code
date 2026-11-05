# Legacy code

Kept for reference only. Nothing in `API/`, `ingest/` or `tests/` imports
these files, they are not tested, and they may not run against the current
store layout.

`io/` holds the read and sync path that the sharded-store and tensorstore
change replaced (see `Notes/STORAGE_AND_READ_BENCHMARKS.md`):

- `zarr_reader.py`: the zarr-python point reader and the import-time store
  loading (`update_zarr_store`). Replaced by `API/io/store_reader.py`.
- `zarr_handling.py`: an older, unused duplicate of the same reader.
- `zip_sync.sh`: the rclone + bsdtar sync of `<MODEL>.zarr.zip` archives into
  symlinked directories. Replaced by `API/store_sync.py`.
- `lmdb_store.py`, `chunk_container.py`: prototype single-file stores (LMDB,
  and an immutable data + index container) that were benchmarked but not
  adopted. They need `lmdb==3.0.0`, which is no longer in the requirements.
