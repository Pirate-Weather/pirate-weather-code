"""Local publishing of a finished ingest run."""

import pickle

import pandas as pd

from API.ingest_utils import publish_ingest_run, read_published_run_time


def _write_run(process_dir, chunk_names):
    store = process_dir / "GEFS.zarr"
    store.mkdir(parents=True)
    for name in chunk_names:
        (store / name).write_text(name)


def test_second_local_publish_replaces_store_and_run_time(tmp_path):
    """Every ingest publishes through this helper, and the API trusts the
    run-time pickle to describe the store next to it. A newer run must fully
    replace the previous store (copying over it used to leave the old run's
    chunks behind when the grid or time axis changed) and the pickle must carry
    the newer run time, which is also what the next ingest uses to skip a run
    that is already published.
    """
    publish_dir = tmp_path / "Prod" / "v31"
    publish_dir.mkdir(parents=True)
    first_run = pd.Timestamp("2026-10-19 06:00")
    second_run = pd.Timestamp("2026-10-19 12:00")

    for run_time, chunks in ((first_run, ["0.0.0", "0.0.1"]), (second_run, ["0.0.0"])):
        process_dir = tmp_path / f"process_{run_time.hour}"
        _write_run(process_dir, chunks)
        publish_ingest_run(
            ["GEFS.zarr"],
            run_time,
            "GEFS.time.pickle",
            process_dir=str(process_dir),
            publish_dir=str(publish_dir),
            save_type="Download",
            s3=None,
        )

    assert sorted(p.name for p in (publish_dir / "GEFS.zarr").iterdir()) == ["0.0.0"]
    assert sorted(p.name for p in publish_dir.iterdir()) == [
        "GEFS.time.pickle",
        "GEFS.zarr",
    ]
    with open(publish_dir / "GEFS.time.pickle", "rb") as f:
        assert pickle.load(f) == second_run
    published = read_published_run_time(
        str(publish_dir / "GEFS.time.pickle"), "Download", None
    )
    assert published == second_run
