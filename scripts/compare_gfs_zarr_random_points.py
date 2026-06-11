#!/usr/bin/env python3
"""Compare two GFS Zarr forecasts at random points.

This utility samples random (variable, time, latitude, longitude) points from two
GFS forecast stores and reports value differences.

Supported Zarr layouts:
1. Root 4D array shaped (variable, time, latitude, longitude)
2. Root group with one 3D array per variable (time, latitude, longitude)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import zarr

DEFAULT_GFS_VAR_ORDER = [
    "time",
    "VIS_surface",
    "GUST_surface",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "APTMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "TOZNE_entireatmosphere_consideredasasinglelayer_",
    "TCDC_entireatmosphere",
    "DUVB_surface",
    "Storm_Distance",
    "Storm_Direction",
    "REFC_entireatmosphere",
    "DSWRF_surface",
    "CAPE_surface",
    "PRES_station",
]


@dataclass
class VarPair:
    """Two matching variable arrays and their overlap shape."""

    left_get: Callable[[int, int, int], float]
    right_get: Callable[[int, int, int], float]
    t_size: int
    y_size: int
    x_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two GFS Zarr forecasts at random points."
    )
    parser.add_argument("left", help="Path to first Zarr forecast store")
    parser.add_argument("right", help="Path to second Zarr forecast store")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random points to compare (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance for numeric equality (default: 0.0)",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Print matching points as well as mismatches",
    )
    return parser.parse_args()


def open_store(path: str) -> zarr.Array | zarr.Group:
    store_path = Path(path)
    if not store_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {path}")
    return zarr.open(path, mode="r")


def as_variable_arrays(
    root: zarr.Array | zarr.Group,
) -> dict[str, VarPair]:
    if isinstance(root, zarr.Array):
        if root.ndim != 4:
            raise ValueError(
                "Root array format expects shape (variable, time, latitude, longitude)."
            )
        var_count = root.shape[0]
        if var_count > len(DEFAULT_GFS_VAR_ORDER):
            raise ValueError(
                f"Root array has {var_count} variables but only "
                f"{len(DEFAULT_GFS_VAR_ORDER)} default names are known."
            )

        arrays: dict[str, VarPair] = {}
        for var_index, var_name in enumerate(DEFAULT_GFS_VAR_ORDER[:var_count]):
            arrays[var_name] = VarPair(
                left_get=lambda t, y, x, vi=var_index: scalar(root[vi, t, y, x]),
                right_get=lambda t, y, x, vi=var_index: scalar(root[vi, t, y, x]),
                t_size=root.shape[1],
                y_size=root.shape[2],
                x_size=root.shape[3],
            )
        return arrays

    arrays_raw = {
        key: root[key]
        for key in root.array_keys()
        if isinstance(root[key], zarr.Array) and root[key].ndim == 3
    }

    if not arrays_raw:
        raise ValueError(
            "No 3D arrays found in group root. Expected per-variable arrays with "
            "shape (time, latitude, longitude)."
        )

    arrays: dict[str, VarPair] = {}
    for key, arr in arrays_raw.items():
        arrays[key] = VarPair(
            left_get=lambda t, y, x, a=arr: scalar(a[t, y, x]),
            right_get=lambda t, y, x, a=arr: scalar(a[t, y, x]),
            t_size=arr.shape[0],
            y_size=arr.shape[1],
            x_size=arr.shape[2],
        )

    return arrays


def build_pairs(
    left_vars: dict[str, VarPair],
    right_vars: dict[str, VarPair],
) -> dict[str, VarPair]:
    common_vars = sorted(set(left_vars) & set(right_vars))
    if not common_vars:
        raise ValueError("No common variables were found between the two forecasts.")

    pairs: dict[str, VarPair] = {}
    for var_name in common_vars:
        left_arr = left_vars[var_name]
        right_arr = right_vars[var_name]

        t_size = min(left_arr.t_size, right_arr.t_size)
        y_size = min(left_arr.y_size, right_arr.y_size)
        x_size = min(left_arr.x_size, right_arr.x_size)

        if t_size <= 0 or y_size <= 0 or x_size <= 0:
            continue

        pairs[var_name] = VarPair(
            left_get=left_arr.left_get,
            right_get=right_arr.right_get,
            t_size=t_size,
            y_size=y_size,
            x_size=x_size,
        )

    if not pairs:
        raise ValueError("No comparable 3D variable arrays found.")

    return pairs


def scalar(value: np.ndarray | np.generic | float | int) -> float:
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def compare_random_points(
    var_pairs: dict[str, VarPair],
    samples: int,
    seed: int,
    tolerance: float,
    show_matches: bool,
) -> None:
    rng = random.Random(seed)
    var_names = list(var_pairs.keys())

    matches = 0
    mismatches = 0
    max_abs_diff = -math.inf
    max_abs_diff_record = ""

    print("Comparing random points")
    print(f"Variables in comparison set: {len(var_names)}")
    print(f"Samples: {samples}")
    print(f"Seed: {seed}")
    print(f"Tolerance: {tolerance}")
    print()

    for sample_idx in range(1, samples + 1):
        var_name = rng.choice(var_names)
        pair = var_pairs[var_name]

        t = rng.randrange(pair.t_size)
        y = rng.randrange(pair.y_size)
        x = rng.randrange(pair.x_size)

        left_value = pair.left_get(t, y, x)
        right_value = pair.right_get(t, y, x)

        if math.isnan(left_value) and math.isnan(right_value):
            abs_diff = 0.0
            is_match = True
        elif math.isnan(left_value) or math.isnan(right_value):
            abs_diff = math.inf
            is_match = False
        else:
            abs_diff = abs(left_value - right_value)
            is_match = abs_diff <= tolerance

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_abs_diff_record = (
                f"sample={sample_idx} var={var_name} t={t} y={y} x={x} "
                f"left={left_value} right={right_value} abs_diff={abs_diff}"
            )

        line = (
            f"[{sample_idx:03d}] {var_name} t={t} y={y} x={x} "
            f"left={left_value} right={right_value} abs_diff={abs_diff}"
        )

        if is_match:
            matches += 1
            if show_matches:
                print(f"MATCH    {line}")
        else:
            mismatches += 1
            print(f"MISMATCH {line}")

    print()
    print("Summary")
    print(f"Matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Match rate: {matches / samples:.2%}")
    print(f"Max abs diff: {max_abs_diff}")
    print(f"Max diff record: {max_abs_diff_record}")


def main() -> None:
    args = parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be >= 0")

    left_root = open_store(args.left)
    right_root = open_store(args.right)

    left_vars = as_variable_arrays(left_root)
    right_vars = as_variable_arrays(right_root)

    var_pairs = build_pairs(left_vars, right_vars)

    compare_random_points(
        var_pairs=var_pairs,
        samples=args.samples,
        seed=args.seed,
        tolerance=args.tolerance,
        show_matches=args.show_matches,
    )


if __name__ == "__main__":
    main()
