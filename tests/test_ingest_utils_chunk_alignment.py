"""Tests for alignment between ingest processing and final Zarr chunks."""

import pytest

from API.ingest_utils import CHUNK_SIZES, FINAL_CHUNK_SIZES


@pytest.mark.parametrize("model", ("HRDPS", "RDPS", "REPS", "GDPS", "GEPS"))
def test_canadian_weather_process_tiles_align_with_final_zarr_chunks(
    model: str,
) -> None:
    process_chunk = CHUNK_SIZES[model]
    final_chunk = FINAL_CHUNK_SIZES[model]

    assert process_chunk == 200
    assert final_chunk == 5
    assert process_chunk % final_chunk == 0
