"""The production ZIP sync script must download every store the API opens."""

import re
import subprocess
from pathlib import Path
from unittest.mock import patch

from API.io import zarr_reader

ZIP_SYNC_PATH = Path(__file__).resolve().parents[2] / "API" / "zip_sync.sh"


def _synced_models() -> set[str]:
    source = ZIP_SYNC_PATH.read_text(encoding="utf-8")
    match = re.search(
        r'^MODELS="\n(?P<models>.*?)\n"$', source, flags=re.MULTILINE | re.DOTALL
    )
    assert match is not None, "MODELS block not found in zip_sync.sh"
    return set(match.group("models").split())


def _local_store_names(stage: str, tmp_path) -> set[str]:
    """Store names update_zarr_store tries to open from local disk for ``stage``."""
    opened = []

    def record(stores, attr_name, save_dir, filename, *, logger):
        opened.append(filename.removesuffix(".zarr"))

    with patch.object(zarr_reader, "_load_local_store", side_effect=record):
        zarr_reader.update_zarr_store(
            False,
            stage=stage,
            save_dir=str(tmp_path),
            use_etopo=False,
            save_type="Local",
            s3_bucket="",
            aws_access_key_id="",
            aws_secret_access_key="",
        )
    return set(opened)


def test_zip_sync_covers_every_local_store(tmp_path, monkeypatch):
    """Protects against the #730/#742 bug class: a model added to the API (or a
    store renamed, e.g. URMA_Hist) but missing from zip_sync.sh is never
    downloaded, so production silently serves without it. Also checks the
    script still parses as POSIX shell.
    """
    monkeypatch.setenv("SKIP_ERA5", "1")

    store_names = _local_store_names("DEV", tmp_path) | _local_store_names(
        "PROD", tmp_path
    )

    assert "GFS" in store_names  # sanity: the recorder saw the store list
    assert store_names - _synced_models() == set()
    syntax = subprocess.run(
        ["sh", "-n", str(ZIP_SYNC_PATH)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
