"""Alert grid ingest <-> API reader contract."""

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from API.ingest_utils import FINAL_CHUNK_SIZES, FINAL_SHARD_SIZES
from API.io.store_reader import local_store_root, open_alert_store

SHAPE = (230, 250)  # not a multiple of the 100-cell shard: partial edge shards
FRENCH = "Avertissement de pluie verglaçante}{Pluie verglaçante – 冻雨|https://x.ca/é"
WRITTEN = {
    (0, 0): "Flood Warning}{Heavy rain|https://alerts.weather.gov/a",
    (3, 17): FRENCH,
    (10, 10): "x" * 5000,  # long string: the chunk is far larger than the minimum
    (99, 99): "last cell of the first shard",
    (100, 100): "first cell of the second shard",
    (150, 5): "ü",
    (229, 249): "very last index",
}


def _write(path, **layout):
    data = np.full(SHAPE, "", dtype=object)
    for cell, text in WRITTEN.items():
        data[cell] = text
    arr = zarr.create_array(
        store=LocalStore(path),
        shape=SHAPE,
        dtype=zarr.dtype.VariableLengthUTF8(),
        config={"write_empty_chunks": True},
        overwrite=True,
        **layout,
    )
    arr[:] = data
    return arr


def test_direct_shard_reader_matches_zarr_python(tmp_path):
    """Protect the ingest/API layout contract: the API reads alert cells straight
    from the shard files, so if the ingest's chunk/shard/codec layout and the
    reader drift apart, alerts silently vanish or the wrong text is returned.
    Also checks unsharded stores fall back to zarr-python rather than misreading.
    """
    chunk, shard = FINAL_CHUNK_SIZES["ALERTS"], FINAL_SHARD_SIZES["ALERTS"]
    version = "NWS_Alerts/x.zarr"
    (tmp_path / "NWS_Alerts").mkdir()
    arr = _write(str(tmp_path / version), chunks=(chunk,) * 2, shards=(shard,) * 2)
    store = open_alert_store(local_store_root(str(tmp_path)), version)
    assert store.reader is not None

    # Written cells, empty cells inside written chunks, entirely empty shards
    # (rows 200+, cols 200+) and the partial last row/column.
    empty = [(1, 1), (50, 50), (199, 0), (0, 199), (215, 120), (229, 0), (0, 249)]
    for y, x in [*WRITTEN, *empty]:
        assert store[y, x] == str(arr[y, x]), (y, x)
    assert store[3, 17] == FRENCH
    assert store[229, 249] == "very last index"

    # A flipped index checksum must never return a silently wrong value: the
    # reader raises ValueError internally and AlertStore defers to zarr-python,
    # which reports the damaged shard instead of returning data.
    shard_file = tmp_path / version / "c" / "0" / "0"
    raw = bytearray(shard_file.read_bytes())
    raw[-1] ^= 0xFF
    shard_file.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        store[3, 17]

    # Plain (unsharded) chunks are not handled by the direct reader.
    plain_version = "NWS_Alerts/plain.zarr"
    _write(str(tmp_path / plain_version), chunks=(chunk,) * 2)
    plain = open_alert_store(local_store_root(str(tmp_path)), plain_version)
    assert plain.reader is None
    for (y, x), text in WRITTEN.items():
        assert plain[y, x] == text
