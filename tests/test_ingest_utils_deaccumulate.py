"""Tests for accumulated-energy conversion used by model ingests."""

import dask.array as da
import numpy as np
import pytest

from API.ingest_utils import deaccumulate_energy_to_flux


def test_deaccumulate_energy_to_flux_uses_interval_lengths_and_clips() -> None:
    times = np.array(
        [
            "2026-01-01T03:00:00",
            "2026-01-01T06:00:00",
            "2026-01-01T12:00:00",
        ],
        dtype="datetime64[s]",
    )
    values = da.from_array(
        np.array(
            [
                [10800.0, 21600.0],
                [32400.0, 10800.0],
                [75600.0, 54000.0],
            ]
        )
    )

    result = deaccumulate_energy_to_flux(values, times).compute()

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1.0, 2.0],
                [2.0, 0.0],
                [2.0, 2.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "times",
    [
        np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]"),
        np.array(
            ["2026-01-01T01:00:00", "2026-01-01T00:00:00"],
            dtype="datetime64[s]",
        ),
    ],
)
def test_deaccumulate_energy_to_flux_rejects_invalid_times(
    times: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        deaccumulate_energy_to_flux(np.ones((len(times), 1)), times)
