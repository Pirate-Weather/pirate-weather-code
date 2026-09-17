"""Tests for accumulated-field conversions used by model ingests."""

import dask.array as da
import numpy as np
import pytest

from API.ingest_utils import (
    FORECAST_LEAD_RANGES,
    deaccumulate_energy_to_flux,
    deaccumulate_to_hourly_rate,
)


def test_deaccumulate_to_hourly_rate_uses_geps_interval_lengths() -> None:
    forecast_hours = np.asarray(FORECAST_LEAD_RANGES["GEPS"])
    interval_hours = np.diff(np.insert(forecast_hours, 0, 0))
    expected_rates = np.stack(
        (
            np.linspace(0.5, 5.0, len(forecast_hours)),
            np.linspace(1.0, 10.0, len(forecast_hours)),
        )
    )[:, :, None, None]
    accumulations = np.cumsum(
        expected_rates * interval_hours[None, :, None, None], axis=1
    )

    result = deaccumulate_to_hourly_rate(
        da.from_array(accumulations), forecast_hours, time_axis=1
    ).compute()

    assert interval_hours[forecast_hours.tolist().index(192)] == 3
    assert interval_hours[forecast_hours.tolist().index(198)] == 6
    np.testing.assert_allclose(result, expected_rates)


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
