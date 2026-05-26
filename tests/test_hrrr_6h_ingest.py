"""Live ingest test for the HRRR_6H ingest script."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import zarr

from API.constants.shared_const import INGEST_VERSION_STR

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "API" / "HRRR_6H_Local_Ingest.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VAR_COUNT = 20
LOCAL_TEST_ENV_VAR = "PW_RUN_LOCAL_INGEST_TESTS"


def _build_pythonpath(env: dict[str, str]) -> str:
    existing_path = env.get("PYTHONPATH", "")
    repo_root = str(REPO_ROOT)
    return repo_root if not existing_path else repo_root + os.pathsep + existing_path


@pytest.mark.skipif(
    os.getenv(LOCAL_TEST_ENV_VAR) != "1",
    reason=(
        "HRRR_6H live ingest test requires a local wgrib2-enabled environment; "
        f"set {LOCAL_TEST_ENV_VAR}=1 to run it"
    ),
)
def test_hrrr_6h_ingest_produces_zarr():
    """Run the live HRRR_6H ingest script and verify it writes a readable zarr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["forecast_process_dir"] = os.path.join(tmpdir, "HRRR_6H")
        env["forecast_path"] = os.path.join(tmpdir, "Prod", "HRRR_6H")
        env["save_type"] = "Download"
        env["AWS_KEY"] = ""
        env["AWS_SECRET"] = ""
        env["PYTHONPATH"] = _build_pythonpath(env)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        assert result.returncode == 0, (
            f"HRRR_6H ingest failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        output_root = Path(env["forecast_path"]) / INGEST_VERSION_STR
        zarr_path = output_root / "HRRR_6H.zarr"
        time_pickle_path = output_root / "HRRR_6H.time.pickle"

        assert zarr_path.exists(), (
            f"Expected zarr output at {zarr_path}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert time_pickle_path.exists(), (
            f"Expected timestamp pickle at {time_pickle_path}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        zarr_array = zarr.open(str(zarr_path), mode="r")

        assert isinstance(zarr_array, zarr.Array), (
            f"Expected a zarr array at {zarr_path}, got {type(zarr_array).__name__}"
        )
        assert zarr_array.ndim == 4, f"Expected 4D zarr output, got {zarr_array.ndim}D"
        assert zarr_array.shape[0] == EXPECTED_VAR_COUNT, (
            f"Expected {EXPECTED_VAR_COUNT} variables, got shape {zarr_array.shape}"
        )
        assert zarr_array.shape[1] > 0, (
            f"Expected time dimension > 0, got {zarr_array.shape}"
        )
        assert zarr_array.shape[2] > 0 and zarr_array.shape[3] > 0, (
            f"Expected non-empty spatial dimensions, got {zarr_array.shape}"
        )
