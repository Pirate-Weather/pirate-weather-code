"""Publishing a finished ingest run to the remote (a local directory here)."""

import numpy as np
import pandas as pd
import zarr

from API import ingest_utils
from API.ingest_utils import open_remote, publish_ingest_run, read_published_run_time


def _write_run(process_dir, value):
    zarr_array = zarr.create_array(
        zarr.storage.LocalStore(process_dir / "GEFS.zarr"),
        shape=(2, 4, 4),
        chunks=(2, 2, 2),
        dtype="float32",
        overwrite=True,
    )
    zarr_array[:] = value


def test_second_publish_replaces_store_and_run_time(tmp_path, monkeypatch):
    """Every ingest publishes through this helper, and the API and the next
    scheduled ingest trust the run-time pickle to describe the store next to it.
    A newer run must replace every chunk of the previous one, the pickle must
    carry the newer run time, and the store's root ``zarr.json`` must be uploaded
    after its chunks: the history ingests treat a store with root metadata as
    complete, so uploading it earlier could make a half-uploaded store look done.
    """
    remote = open_remote(str(tmp_path / "Prod" / "v31"), "Download")
    upload_groups = []
    put_files = ingest_utils._put_files

    def recording_put_files(remote, uploads):
        upload_groups.append([key for _, key in uploads])
        put_files(remote, uploads)

    monkeypatch.setattr(ingest_utils, "_put_files", recording_put_files)

    first_run = pd.Timestamp("2026-10-19 06:00")
    second_run = pd.Timestamp("2026-10-19 12:00")
    for run_time, value in ((first_run, 1.0), (second_run, 2.0)):
        process_dir = tmp_path / f"process_{run_time.hour}"
        _write_run(process_dir, value)
        publish_ingest_run(
            ["GEFS.zarr"],
            run_time,
            "GEFS.time.pickle",
            process_dir=str(process_dir),
            remote=remote,
        )

    published = zarr.open_array(
        zarr.storage.LocalStore(tmp_path / "Prod" / "v31" / "GEFS.zarr"), mode="r"
    )
    np.testing.assert_array_equal(published[:], np.full((2, 4, 4), 2.0, "float32"))
    assert read_published_run_time(remote, "GEFS.time.pickle") == second_run
    chunks, array_metadata, root_metadata = upload_groups[-3:]
    assert len(chunks) == 4 and all("/c/" in key for key in chunks)
    assert array_metadata == []
    assert root_metadata == ["GEFS.zarr/zarr.json"]
