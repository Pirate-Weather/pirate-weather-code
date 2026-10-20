"""Final store write used by every gridded ingest."""

import zipfile

import dask.array as da
import numpy as np
import zarr

from API.ingest_utils import pad_to_chunk_size, write_final_zarr


def test_tiled_final_write_matches_source_for_unaligned_chunks(tmp_path):
    """write_final_zarr writes source-sized dask tiles into a store
    with tiny (3x3) final chunks. When the source chunk (here 10) is not a
    multiple of the final chunk, a tile edge inside a final chunk makes zarr
    rewrite that chunk; in the S3 ZipStore every rewrite appends a duplicate
    entry, bloating the uploaded file and leaving readers to pick one copy. Each
    chunk must be written exactly once, with the values rounded to 5 decimals and
    the padded edge left as NaN.
    """
    rng = np.random.default_rng(0)
    values = rng.random((3, 4, 23, 31), dtype="float32") * 100
    values[0, 1, 5, 7] = np.nan
    source = da.from_array(values, chunks=(3, 4, 10, 10))
    padded = pad_to_chunk_size(source, 3)

    store_path = str(tmp_path / "MODEL.zarr.zip")
    write_final_zarr(padded, store_path, final_chunk=3, zarr_store_workers=2)

    written = zarr.open_array(zarr.storage.ZipStore(store_path, mode="r"), mode="r")
    assert written.shape == (3, 4, 24, 33)
    assert written.chunks == (3, 4, 3, 3)
    np.testing.assert_array_equal(
        written[:, :, :23, :31], np.round(values, 5).astype("float32")
    )
    assert np.isnan(written[:, :, 23:, :]).all()
    assert np.isnan(written[:, :, :, 31:]).all()
    with zipfile.ZipFile(store_path) as archive:
        names = archive.namelist()
    assert len(names) == len(set(names))
