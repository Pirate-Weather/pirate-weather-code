#!/usr/bin/env python3
"""Compare random FMI IS4FIRES production Zarr points with OPeNDAP best data.

Default usage from this repo:

    .venvs/ingest-test/bin/python scripts/compare_is4fires_zarr_opendap_points.py

Or activate the ingest test environment first:

    source .venvs/ingest-test/bin/activate-pirate-ingest
    python scripts/compare_is4fires_zarr_opendap_points.py
"""

from __future__ import annotations

import argparse
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from xarray.coding.variables import SerializationWarning

KG_M3_TO_UG_M3 = 1e9
VALID_DATA_MIN = -100
VALID_DATA_MAX = 120000

DEFAULT_ZARR_PATH = "/home/reya/Weather/Prod/IS4FIRES/v30/IS4FIRES.zarr"
DEFAULT_OPENDAP_URL = (
    "https://thredds.silam.fmi.fi/thredds/dodsC/"
    "i4f20-fc/IS4FIRES-fc_best.ncd.html"
)

ZARR_VARS = ("time", "cnc_PM_FRP")
SOURCE_VARS = {"cnc_PM_FRP": "cnc_PM_FRP"}


@dataclass(frozen=True)
class TimeMatch:
    zarr_index: int
    source_index: int
    zarr_unix_seconds: float
    source_unix_seconds: int
    delta_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare random FMI IS4FIRES production Zarr points against the "
            "FMI THREDDS OPeNDAP best time series."
        )
    )
    parser.add_argument(
        "--zarr",
        default=DEFAULT_ZARR_PATH,
        help=f"Production IS4FIRES Zarr path (default: {DEFAULT_ZARR_PATH})",
    )
    parser.add_argument(
        "--opendap-url",
        default=DEFAULT_OPENDAP_URL,
        help="FMI IS4FIRES OPeNDAP URL. A trailing .html is accepted.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random data points to compare (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible samples (default: 42)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance after ingest conversion/rounding (default: 1e-3)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance after ingest conversion/rounding (default: 1e-6)",
    )
    parser.add_argument(
        "--max-time-delta-seconds",
        type=float,
        default=180,
        help=(
            "Maximum allowed delta when matching Zarr float32 Unix times to "
            "OPeNDAP times (default: 180)"
        ),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(SOURCE_VARS),
        default=tuple(SOURCE_VARS),
        help="Variables to sample (default: cnc_PM_FRP)",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Print matching points as well as mismatches.",
    )
    return parser.parse_args()


def normalize_opendap_url(url: str) -> str:
    return url[:-5] if url.endswith(".html") else url


def unix_seconds(values: np.ndarray) -> np.ndarray:
    return (pd.to_datetime(values).astype("int64") // 1_000_000_000).to_numpy()


def open_production_zarr(path: str) -> zarr.Array:
    zarr_path = Path(path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    if not isinstance(root, zarr.Array):
        raise TypeError("Expected IS4FIRES production Zarr to be a root zarr.Array")
    if root.ndim != 4 or root.shape[0] < len(ZARR_VARS):
        raise ValueError(
            "Expected IS4FIRES production Zarr shape "
            "(variable, time, latitude, longitude) with known variables."
        )
    return root


def open_opendap_dataset(url: str) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SerializationWarning)
        return xr.open_dataset(
            normalize_opendap_url(url),
            engine="netcdf4",
            decode_times=True,
        )


def build_time_matches(
    zarr_array: zarr.Array,
    source_times: np.ndarray,
    max_delta_seconds: float,
) -> list[TimeMatch]:
    zarr_times = np.asarray(zarr_array[0, :, 0, 0], dtype=np.float64)
    valid_zarr = np.flatnonzero(np.isfinite(zarr_times) & (zarr_times > 0))
    source_seconds = unix_seconds(source_times)
    matches = []

    for zarr_index in valid_zarr:
        zarr_second = zarr_times[zarr_index]
        insert_at = int(np.searchsorted(source_seconds, zarr_second))
        candidates = []
        if insert_at < len(source_seconds):
            candidates.append(insert_at)
        if insert_at > 0:
            candidates.append(insert_at - 1)
        if not candidates:
            continue

        source_index = min(
            candidates, key=lambda index: abs(source_seconds[index] - zarr_second)
        )
        delta = abs(float(source_seconds[source_index]) - zarr_second)
        if delta <= max_delta_seconds:
            matches.append(
                TimeMatch(
                    zarr_index=int(zarr_index),
                    source_index=int(source_index),
                    zarr_unix_seconds=float(zarr_second),
                    source_unix_seconds=int(source_seconds[source_index]),
                    delta_seconds=delta,
                )
            )

    if not matches:
        raise ValueError(
            "No overlapping times found between the production Zarr and OPeNDAP "
            f"within {max_delta_seconds} seconds."
        )
    return matches


def scalar_dataarray(data_array: xr.DataArray) -> float:
    value = data_array.squeeze(drop=True).values
    return float(np.asarray(value).item())


def source_scalar(ds: xr.Dataset, var_name: str, t: int, y: int, x: int) -> float:
    source_var = SOURCE_VARS[var_name]
    raw = scalar_dataarray(ds[source_var].isel(time=t, lat=y, lon=x))
    converted = np.float32(raw * KG_M3_TO_UG_M3)

    if (
        not np.isfinite(converted)
        or converted < VALID_DATA_MIN
        or converted > VALID_DATA_MAX
    ):
        return math.nan
    return float(np.float32(np.round(converted, 5)))


def zarr_scalar(zarr_array: zarr.Array, var_name: str, t: int, y: int, x: int) -> float:
    var_index = ZARR_VARS.index(var_name)
    return float(zarr_array[var_index, t, y, x])


def values_match(
    left: float, right: float, atol: float, rtol: float
) -> tuple[bool, float]:
    if math.isnan(left) and math.isnan(right):
        return True, 0.0
    if math.isnan(left) or math.isnan(right):
        return False, math.inf

    abs_diff = abs(left - right)
    return abs_diff <= atol + rtol * abs(right), abs_diff


def compare_random_points(args: argparse.Namespace) -> int:
    zarr_array = open_production_zarr(args.zarr)
    ds = open_opendap_dataset(args.opendap_url)

    missing_source = sorted(
        {SOURCE_VARS[var] for var in args.variables} - set(ds.data_vars)
    )
    if missing_source:
        raise KeyError(
            f"OPeNDAP dataset is missing required variables: {missing_source}"
        )

    time_matches = build_time_matches(
        zarr_array,
        ds["time"].values,
        max_delta_seconds=args.max_time_delta_seconds,
    )
    y_size = min(int(zarr_array.shape[2]), int(ds.sizes["lat"]))
    x_size = min(int(zarr_array.shape[3]), int(ds.sizes["lon"]))
    if y_size <= 0 or x_size <= 0:
        raise ValueError("No overlapping spatial grid was found.")

    rng = random.Random(args.seed)
    matches = 0
    mismatches = 0
    max_abs_diff = -math.inf
    max_abs_diff_record = ""

    source_first = pd.to_datetime(ds.time.values[0], utc=True)
    source_last = pd.to_datetime(ds.time.values[-1], utc=True)
    matched_first = pd.to_datetime(
        min(match.source_unix_seconds for match in time_matches), unit="s", utc=True
    )
    matched_last = pd.to_datetime(
        max(match.source_unix_seconds for match in time_matches), unit="s", utc=True
    )

    print("Comparing FMI IS4FIRES production Zarr to OPeNDAP best time series")
    print(f"Zarr: {args.zarr}")
    print(f"OPeNDAP: {normalize_opendap_url(args.opendap_url)}")
    print(f"Samples: {args.samples}")
    print(f"Seed: {args.seed}")
    print(f"Variables: {', '.join(args.variables)}")
    print(f"Tolerance: atol={args.atol} rtol={args.rtol}")
    print(f"OPeNDAP time coverage: {source_first} to {source_last}")
    print(f"Matched Zarr/OPeNDAP times: {len(time_matches)}")
    print(f"Matched comparison coverage: {matched_first} to {matched_last}")
    print(f"Spatial comparison shape: lat={y_size} lon={x_size}")
    print()

    for sample_index in range(1, args.samples + 1):
        var_name = rng.choice(args.variables)
        time_match = rng.choice(time_matches)
        y = rng.randrange(y_size)
        x = rng.randrange(x_size)

        expected = source_scalar(ds, var_name, time_match.source_index, y, x)
        actual = zarr_scalar(zarr_array, var_name, time_match.zarr_index, y, x)
        is_match, abs_diff = values_match(actual, expected, args.atol, args.rtol)

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_abs_diff_record = (
                f"sample={sample_index} var={var_name} zarr_t={time_match.zarr_index} "
                f"opendap_t={time_match.source_index} y={y} x={x} "
                f"actual={actual} expected={expected} abs_diff={abs_diff}"
            )

        source_time = pd.to_datetime(time_match.source_unix_seconds, unit="s", utc=True)
        line = (
            f"[{sample_index:03d}] {var_name} time={source_time} "
            f"zarr_t={time_match.zarr_index} opendap_t={time_match.source_index} "
            f"time_delta_s={time_match.delta_seconds:.0f} "
            f"lat_index={y} lon_index={x} "
            f"actual={actual} expected={expected} abs_diff={abs_diff}"
        )

        if is_match:
            matches += 1
            if args.show_matches:
                print(f"MATCH    {line}")
        else:
            mismatches += 1
            print(f"MISMATCH {line}")

    print()
    print("Summary")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Match rate: {matches / args.samples:.2%}")
    print(f"Max abs diff: {max_abs_diff}")
    print(f"Max diff record: {max_abs_diff_record}")

    return 1 if mismatches else 0


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.atol < 0:
        raise ValueError("--atol must be >= 0")
    if args.rtol < 0:
        raise ValueError("--rtol must be >= 0")
    if args.max_time_delta_seconds < 0:
        raise ValueError("--max-time-delta-seconds must be >= 0")

    raise SystemExit(compare_random_points(args))


if __name__ == "__main__":
    main()
