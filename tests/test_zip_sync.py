"""Regression tests for the local Zarr ZIP synchronization script."""

import re
import subprocess
from pathlib import Path

ZIP_SYNC_PATH = Path(__file__).resolve().parents[1] / "API" / "zip_sync.sh"
CMC_MODELS = {"GDPS", "GEPS", "HRDPS", "RAQDPS", "REPS"}


def _configured_models() -> set[str]:
    source = ZIP_SYNC_PATH.read_text(encoding="utf-8")
    match = re.search(
        r'^MODELS="\n(?P<models>.*?)\n"$', source, flags=re.MULTILINE | re.DOTALL
    )
    assert match is not None, "MODELS block not found in zip_sync.sh"
    return set(match.group("models").split())


def test_zip_sync_includes_cmc_models():
    """All CMC forecast archives should be included in the sync allowlist."""
    assert CMC_MODELS <= _configured_models()


def test_zip_sync_has_valid_shell_syntax():
    """The synchronization script should remain valid POSIX shell syntax."""
    result = subprocess.run(
        ["sh", "-n", str(ZIP_SYNC_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
