"""ERA5 archive contract: int16 encode -> sharded write -> tensorstore read -> decode."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from API.constants.model_const import (
    ERA5_ARCHIVE,
    ERA5_ARCHIVE_DELTA_VARS,
    ERA5_ARCHIVE_ENCODING,
    ERA5_ARCHIVE_FILL,
    ERA5_ARCHIVE_NAN_VARS,
)
from API.ingest_utils import encode_era5_archive_field, write_era5_archive_zarr
from API.io.ZarrHelpers import decode_era5_archive

HOURS, N_LAT, N_LON, CHUNK, SHARD = 168, 25, 30, 5, 10
PADDED_LAT = 30
BUILDER = (
    Path(__file__).resolve().parents[2] / "scripts" / "build_era5_point_test_store.py"
)


def _source_fields(rng):
    """Plausible API-unit week for every archive variable, with the awkward
    cases: zero-heavy precipitation, tiny negative accumulations, cloud base NaN."""
    shape = (HOURS, N_LAT, N_LON)
    hours = np.arange(HOURS)[:, None, None]
    smooth = np.sin(hours / 12.0) + 0.01 * rng.standard_normal(shape)
    wet = rng.random(shape) < 0.3
    fields = {
        "instantaneous_10m_wind_gust": 8 + 6 * smooth,
        "mean_sea_level_pressure": 1012 + 15 * smooth,
        "2m_temperature": 285 + 10 * smooth,
        "2m_dewpoint_temperature": 278 + 8 * smooth,
        "10m_u_component_of_wind": 5 * smooth,
        "10m_v_component_of_wind": -4 * smooth,
        "precipitation_type": np.where(wet, 1.0, 0.0),
        "total_precipitation": np.where(wet, rng.gamma(0.5, 2.0, shape), -1e-9),
        "rain_rate": np.where(wet, rng.gamma(0.5, 2.0, shape), 0.0),
        "snowfall_rate_water_equivalent": np.where(rng.random(shape) < 0.05, 0.3, 0.0),
        "total_column_ozone": 300 + 40 * smooth,
        "total_cloud_cover": 100 * rng.random(shape),
        "downward_uv_radiation_at_the_surface": np.clip(40 * smooth, 0, None),
        "surface_solar_radiation_downwards": np.clip(900 * smooth, 0, None),
        "convective_available_potential_energy": 3000 * rng.random(shape),
        "surface_pressure": 950 + 15 * smooth,
        "precipitation_probability": 100 * rng.integers(0, 10, shape) / 9,
        "total_sky_direct_solar_radiation_at_surface": np.clip(700 * smooth, 0, None),
        "cloud_base_height": np.where(
            rng.random(shape) < 0.2, np.nan, 3000 * rng.random(shape)
        ),
    }
    return {name: values.astype(np.float32) for name, values in fields.items()}


def test_era5_archive_round_trip_matches_source_within_precision(tmp_path):
    """Protect the archive's ingest/API contract: a week encoded and written by
    the builder must decode, through the tensorstore reader the API uses, to the
    source within each variable's quantization step. Catches a wrong delta
    (values drift by the running sum), a lost cloud-base NaN, padding read as
    data, an encoding that overflows int16 instead of failing, and a precomputed
    precipitation probability that no longer matches the API's 3x3 neighbourhood.
    """
    rng = np.random.default_rng(3)
    variables = sorted(ERA5_ARCHIVE, key=ERA5_ARCHIVE.get)
    source = _source_fields(rng)
    stacked = np.full(
        (len(variables), HOURS, PADDED_LAT, N_LON), ERA5_ARCHIVE_FILL, np.int16
    )
    for i, name in enumerate(variables):
        stacked[i, :, :N_LAT] = encode_era5_archive_field(name, source[name])

    attributes = {
        "time": [1_435_536_000 + 3600 * h for h in range(HOURS)],
        "variables": variables,
        "encoding": {name: list(ERA5_ARCHIVE_ENCODING[name]) for name in variables},
        "delta_variables": list(ERA5_ARCHIVE_DELTA_VARS),
        "nan_variables": list(ERA5_ARCHIVE_NAN_VARS),
    }
    path = tmp_path / "ERA5_archive.zarr"
    write_era5_archive_zarr(stacked, str(path), CHUNK, SHARD, 2, attributes)

    array = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}
    ).result()
    stored_attributes = {**attributes, "fill_value": ERA5_ARCHIVE_FILL}
    for y, x in [(0, 0), (N_LAT - 1, N_LON - 1), (12, 17), (7, 4)]:
        decoded = decode_era5_archive(
            array[:, :, y, x].read().result(), stored_attributes
        )
        for i, name in enumerate(variables):
            kind, step_or_scale, _ = ERA5_ARCHIVE_ENCODING[name]
            truth = source[name][:, y, x].astype(np.float64)
            finite = np.isfinite(truth)
            np.testing.assert_array_equal(np.isfinite(decoded[i]), finite, err_msg=name)
            if kind == "lin":
                allowed = 0.5 * step_or_scale * 1.001 + 1e-6 * np.abs(truth)
            else:
                truth = np.clip(truth, 0, None)
                allowed = (
                    np.sqrt(truth) / step_or_scale + 0.25 / step_or_scale**2 + 1e-6
                )
            assert np.all(np.abs(decoded[i] - truth)[finite] <= allowed[finite]), name

    # Padding rows past the last latitude decode as missing, not as data.
    padding = decode_era5_archive(
        array[:, :, N_LAT, 0].read().result(), stored_attributes
    )
    assert np.isnan(padding).all()

    # Out-of-range values and unexpected NaNs fail before anything is written.
    with pytest.raises(OverflowError):
        encode_era5_archive_field(
            "2m_temperature", np.full((2, 1, 1), 5000.0, np.float32)
        )
    with pytest.raises(ValueError):
        encode_era5_archive_field(
            "2m_temperature", np.full((2, 1, 1), np.nan, np.float32)
        )

    # Precomputed probability: one wet cell at the dateline next to the pole is
    # 1 of 6 cells for its polar neighbours, 1 of 9 below, across the wrap.
    spec = importlib.util.spec_from_file_location("era5_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    tp = np.zeros((1, 5, 8), np.float32)
    tp[0, 0, 0] = 0.001
    prob = builder.precipitation_probability(tp)[0]
    assert prob[0, 7] == pytest.approx(100 / 6) and prob[0, 1] == pytest.approx(100 / 6)
    assert prob[1, 7] == pytest.approx(100 / 9) and prob[1, 0] == pytest.approx(100 / 9)
    assert prob[2, 0] == 0 and prob[0, 3] == 0
