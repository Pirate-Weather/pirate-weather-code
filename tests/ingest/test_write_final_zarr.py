"""Final store write used by every gridded ingest."""

import dask.array as da
import numpy as np
import zarr

from API.ingest_utils import pad_to_chunk_size, write_final_zarr


def test_tiled_final_write_matches_source_for_unaligned_chunks(tmp_path):
    """write_final_zarr writes source-sized dask tiles into a store with tiny
    (3x3) final chunks; here the source chunk (10) is not a multiple of the
    final chunk, so a misaligned tile would split and rewrite chunks. Values
    must be rounded to 5 decimals and the padded edge left as NaN. Every chunk
    file must exist, all-NaN ones included: publishing uploads over the
    previous run's objects, and a chunk left out would keep the old run's data.
    """
    rng = np.random.default_rng(0)
    values = rng.random((3, 4, 23, 31), dtype="float32") * 100
    values[0, 1, 5, 7] = np.nan
    values[:, :, :3, :3] = np.nan
    source = da.from_array(values, chunks=(3, 4, 10, 10))
    padded = pad_to_chunk_size(source, 3)

    store_path = tmp_path / "MODEL.zarr"
    write_final_zarr(padded, str(store_path), final_chunk=3, zarr_store_workers=2)

    written = zarr.open_array(
        zarr.storage.LocalStore(store_path, read_only=True), mode="r"
    )
    assert written.shape == (3, 4, 24, 33)
    assert written.chunks == (3, 4, 3, 3)
    np.testing.assert_array_equal(
        written[:, :, :23, :31], np.round(values, 5).astype("float32")
    )
    assert np.isnan(written[:, :, 23:, :]).all()
    assert np.isnan(written[:, :, :, 31:]).all()
    chunk_files = [p for p in (store_path / "c").rglob("*") if p.is_file()]
    assert len(chunk_files) == 8 * 11
