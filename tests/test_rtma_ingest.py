"""Test for RTMA-RU ingest script functionality."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def test_rtma_script_exists():
    """Test that RTMA-RU_Local_Ingest.py exists."""
    rtma_script_path = (
        Path(__file__).resolve().parents[1] / "API" / "RTMA-RU_Local_Ingest.py"
    )
    assert rtma_script_path.exists(), f"RTMA script not found at {rtma_script_path}"
    assert rtma_script_path.is_file(), "RTMA script is not a file"


def test_rtma_script_is_valid_python():
    """Test that RTMA-RU_Local_Ingest.py is valid Python syntax."""
    rtma_script_path = (
        Path(__file__).resolve().parents[1] / "API" / "RTMA-RU_Local_Ingest.py"
    )

    # Read the script content
    script_content = rtma_script_path.read_text()

    # Try to parse it as Python code
    try:
        ast.parse(script_content)
    except SyntaxError as e:
        pytest.fail(f"RTMA script has invalid Python syntax: {e}")


def test_rtma_script_has_required_imports():
    """Test that the RTMA script has all required imports."""
    rtma_script_path = (
        Path(__file__).resolve().parents[1] / "API" / "RTMA-RU_Local_Ingest.py"
    )

    # Read the script content
    script_content = rtma_script_path.read_text()

    # Check for required imports
    required_imports = [
        "import numpy",
        "import s3fs",
        "import xarray",
        "import zarr",
        "from herbie import Herbie",
        "from herbie.fast import Herbie_latest",
        "from metpy.calc import relative_humidity_from_specific_humidity",
    ]

    for import_stmt in required_imports:
        assert import_stmt in script_content, f"Missing required import: {import_stmt}"


def test_rtma_script_has_required_components():
    """Test that the RTMA script contains expected components."""
    rtma_script_path = (
        Path(__file__).resolve().parents[1] / "API" / "RTMA-RU_Local_Ingest.py"
    )

    # Read the script content
    script_content = rtma_script_path.read_text()

    # Check for key processing steps
    assert "zarr_vars" in script_content, "Missing zarr_vars definition"
    assert "Herbie_latest" in script_content, "Missing Herbie_latest usage"
    assert "base_time" in script_content, "Missing base_time variable"
    assert "match_strings" in script_content, "Missing match_strings definition"

    # Check for RTMA-specific elements
    assert "rtma_ru" in script_content.lower(), "Script should reference rtma_ru model"

    # Check for key variables
    assert "vis" in script_content, "Missing visibility variable"
    assert "t2m" in script_content, "Missing temperature variable"
    assert "u10" in script_content, "Missing u-wind variable"
    assert "v10" in script_content, "Missing v-wind variable"


def test_rtma_script_python_check():
    """Test that the RTMA script can be checked with python -m py_compile."""
    rtma_script_path = (
        Path(__file__).resolve().parents[1] / "API" / "RTMA-RU_Local_Ingest.py"
    )

    # Use py_compile to check if the script compiles
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(rtma_script_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed to compile: {result.stderr}"
