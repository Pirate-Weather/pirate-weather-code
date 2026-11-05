"""Short-term history stores kept on the remote between ingest runs."""

import numpy as np
import pytest
import zarr

from API.ingest_utils import (
    history_store_exists,
    open_history_arrays,
    open_remote,
    publish_history_store,
)


def _write_history(path, variables):
    root = zarr.open_group(zarr.storage.LocalStore(path), mode="w")
    for variable in variables:
        root.create_array(variable, shape=(1, 2, 2), dtype="float32")[:] = 1.0


def test_history_store_round_trip_and_stale_store_rebuild(tmp_path):
    """The history ingests (GFS, ECMWF, SILAM, RTMA, URMA) publish one store per
    past run and later merge them straight from the remote. A published store
    must be seen as complete and read back. A store missing an expected variable
    (written by an older ingest) must fail the merge and be deleted, so the next
    run rebuilds it instead of failing forever (#644). Stores are opened
    concurrently, so the results must come back in the requested order and a
    stale or missing store among them must still fail the merge. They are
    downloaded first, so the merge reads local copies.
    """
    remote = open_remote(str(tmp_path / "History"), "Download")
    _write_history(tmp_path / "tmp_full.zarr", ("time", "TMP_2maboveground"))
    _write_history(tmp_path / "tmp_stale.zarr", ("time",))

    publish_history_store(str(tmp_path / "tmp_full.zarr"), remote, "Hist_A.zarr")
    publish_history_store(str(tmp_path / "tmp_stale.zarr"), remote, "Hist_B.zarr")
    assert not (tmp_path / "tmp_full.zarr").exists()
    assert history_store_exists(remote, "Hist_A.zarr")
    assert not history_store_exists(remote, "Hist_C.zarr")

    _write_history(tmp_path / "tmp_full.zarr", ("time", "TMP_2maboveground"))
    publish_history_store(str(tmp_path / "tmp_full.zarr"), remote, "Hist_D.zarr")
    local_dir = tmp_path / "local"
    stores = open_history_arrays(
        remote,
        ["Hist_D.zarr", "Hist_A.zarr"],
        ("time", "TMP_2maboveground"),
        str(local_dir),
    )
    assert [store["TMP_2maboveground"].store.root.name for store in stores] == [
        "Hist_D.zarr",
        "Hist_A.zarr",
    ]
    np.testing.assert_array_equal(stores[1]["TMP_2maboveground"][:], np.ones((1, 2, 2)))

    with pytest.raises(ValueError, match="missing variables"):
        open_history_arrays(
            remote,
            ["Hist_A.zarr", "Hist_B.zarr"],
            ("time", "TMP_2maboveground"),
            str(local_dir),
        )
    assert not history_store_exists(remote, "Hist_B.zarr")
    with pytest.raises(RuntimeError, match="missing"):
        open_history_arrays(
            remote, ["Hist_A.zarr", "Hist_C.zarr"], ("time",), str(local_dir)
        )
