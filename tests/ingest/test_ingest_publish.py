"""Publishing a finished ingest run to the remote (a local directory here)."""

import json
import time

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


def test_second_publish_points_at_new_version_and_run_time(tmp_path, monkeypatch):
    """Every ingest publishes through this helper. A newer run must become a
    new version that the ``<MODEL>.current.json`` pointer names, with the
    previous version still in place (a sync may be copying it), and the pickle
    must carry the newer run time. Within a version the root ``zarr.json`` must
    be uploaded after the chunks and the pointer only after the whole store:
    the sync trusts the pointer to name a complete store, so moving it earlier
    could copy a half-uploaded one.
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
    versions = []
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
        pointer = json.loads(
            (tmp_path / "Prod" / "v31" / "GEFS.current.json").read_text()
        )
        versions.append(pointer["store"])
        time.sleep(1.1)  # versions are stamped to the second

    assert versions[0] != versions[1]
    assert pointer["run_time"] == str(second_run)
    published = zarr.open_array(
        zarr.storage.LocalStore(tmp_path / "Prod" / "v31" / versions[1]), mode="r"
    )
    np.testing.assert_array_equal(published[:], np.full((2, 4, 4), 2.0, "float32"))
    assert (tmp_path / "Prod" / "v31" / versions[0] / "zarr.json").exists()
    assert read_published_run_time(remote, "GEFS.time.pickle") == second_run
    chunks, array_metadata, root_metadata = upload_groups[-3:]
    assert len(chunks) == 4 and all("/c/" in key for key in chunks)
    assert array_metadata == []
    assert root_metadata == [f"{versions[1]}/zarr.json"]
