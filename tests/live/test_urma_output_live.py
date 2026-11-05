"""Compare a locally produced URMA history Zarr with independent GribStream analyses.

Needs the output of a real URMA ingest on this machine (``PW_URMA_OUTPUT_ROOT``)
and a ``GRIBSTREAM_API_KEY``; skipped when either is missing.
"""

import datetime
import os
import pickle
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
import zarr

from API.constants.shared_const import INGEST_VERSION_STR
from tests.helpers.urma import (
    GRIBSTREAM_VARIABLES,
    _comparison_grid_points,
    _fetch_gribstream_urma,
    _format_utc,
)

GRIBSTREAM_API_KEY_ENV_VAR = "GRIBSTREAM_API_KEY"
URMA_OUTPUT_ROOT_ENV_VAR = "PW_URMA_OUTPUT_ROOT"
DEFAULT_URMA_OUTPUT_ROOT = Path("/mnt/nvme/data/Prod/URMA")


@pytest.fixture(scope="module")
def urma_ingest_output() -> tuple[Path, Path]:
    """Locate the URMA output produced by the local ingest launch configuration."""
    output_root = Path(
        os.getenv(URMA_OUTPUT_ROOT_ENV_VAR, str(DEFAULT_URMA_OUTPUT_ROOT))
    )
    versioned_root = output_root / INGEST_VERSION_STR
    zarr_path = versioned_root / "URMA_Hist.zarr"
    time_pickle_path = versioned_root / "URMA_Hist.time.pickle"
    if not zarr_path.exists() or not time_pickle_path.exists():
        pytest.skip(
            f"URMA output is missing under {versioned_root}; run the URMA ingest first "
            f"or set {URMA_OUTPUT_ROOT_ENV_VAR}"
        )

    return zarr_path, time_pickle_path


@pytest.mark.skipif(
    not os.getenv(GRIBSTREAM_API_KEY_ENV_VAR),
    reason=f"set {GRIBSTREAM_API_KEY_ENV_VAR} to run the GribStream comparison",
)
def test_urma_zarr_matches_gribstream_at_multiple_points(urma_ingest_output):
    """The newest URMA analysis in our Zarr must match GribStream at several points.

    Protects the URMA ingest's grid orientation, variable order, and unit handling:
    any of these going wrong produces plausible-looking but wrong values.
    """
    zarr_path, time_pickle_path = urma_ingest_output
    with time_pickle_path.open("rb") as file:
        run_time = pickle.load(file)

    coordinates, indices = _comparison_grid_points()
    try:
        rows = _fetch_gribstream_urma(
            os.environ[GRIBSTREAM_API_KEY_ENV_VAR],
            [run_time],
            coordinates,
        )
    except HTTPError as exc:
        if exc.code in {429, 503, 504}:
            pytest.skip(f"GribStream temporarily unavailable: HTTP {exc.code}")
        pytest.fail(f"GribStream request failed: HTTP {exc.code}")
    except URLError as exc:
        pytest.skip(f"Could not reach GribStream: {exc.reason}")

    assert len(rows) == len(coordinates), (
        f"Expected {len(coordinates)} GribStream rows for URMA run "
        f"{_format_utc(run_time)}, got {len(rows)}"
    )
    rows_by_name = {row["name"]: row for row in rows}
    assert rows_by_name.keys() == indices.keys()

    zarr_array = zarr.open_array(str(zarr_path), mode="r")
    expected_timestamp = run_time.replace(tzinfo=datetime.UTC).timestamp()
    zarr_times = np.asarray(zarr_array[0, :, 0, 0], dtype=np.float64)
    time_index = int(np.argmin(np.abs(zarr_times - expected_timestamp)))
    assert zarr_times[time_index] == pytest.approx(expected_timestamp, abs=128)

    for name, (x_index, y_index) in indices.items():
        row = rows_by_name[name]
        assert row["forecasted_at"] == _format_utc(run_time)
        assert row["forecasted_time"] == _format_utc(run_time)
        for variable in GRIBSTREAM_VARIABLES:
            local_value = float(
                zarr_array[
                    variable["zarr_index"],
                    time_index,
                    y_index,
                    x_index,
                ]
            )
            assert local_value == pytest.approx(
                row[variable["alias"]],
                abs=variable["absolute_tolerance"],
                rel=1e-4,
            ), f"{name}: {variable['alias']}"
