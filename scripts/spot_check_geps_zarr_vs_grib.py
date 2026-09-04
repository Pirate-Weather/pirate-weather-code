"""Spot-check a GEPS final Zarr against cached raw ensemble GRIB2 files.

GEPS writes ensemble precipitation statistics to a root 4D Zarr array with
dimensions (variable, time, latitude, longitude). This script reconstructs
those statistics from the control and perturbed-member GRIB messages, then
checks both raw source hours and hourly values created by interpolation.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

DEFAULT_ZARR_PATH = Path("/mnt/nvme/data/Prod/v30/GEPS.zarr")
DEFAULT_GRIB_ROOT = Path("/mnt/nvme/data/Process/GEPS/Downloads/herbie/geps")
DEFAULT_BASE_TIME_PICKLE = Path("/mnt/nvme/data/Prod/v30/GEPS.time.pickle")

ZARR_VARS = (
    "time",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
    "AFRAIN_Mean",
    "AICEP_Mean",
    "ARAIN_Mean",
    "ASNOW_Mean",
)

GEPS_SOURCE_HOURS = (*range(3, 193, 3), *range(198, 241, 6))
PRECIP_THRESHOLD = 0.1


@dataclass(frozen=True)
class VariableSpec:
    """Map a GEPS final statistic to its source accumulated GRIB field."""

    zarr_name: str
    grib_token: str
    statistic: str


VARIABLE_SPECS = {
    "Precipitation_Prob": VariableSpec("Precipitation_Prob", "APCP", "probability"),
    "APCP_Mean": VariableSpec("APCP_Mean", "APCP", "mean"),
    "APCP_StdDev": VariableSpec("APCP_StdDev", "APCP", "stddev"),
    "AFRAIN_Mean": VariableSpec("AFRAIN_Mean", "AFRAIN", "mean"),
    "AICEP_Mean": VariableSpec("AICEP_Mean", "AICEP", "mean"),
    "ARAIN_Mean": VariableSpec("ARAIN_Mean", "ARAIN", "mean"),
    "ASNOW_Mean": VariableSpec("ASNOW_Mean", "ASNOW", "mean"),
}


@dataclass(frozen=True)
class CheckResult:
    """One sampled comparison result."""

    kind: str
    variable: str
    hour: int
    y: int
    x: int
    zarr_value: float
    expected_value: float
    abs_diff: float
    passed: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spot-check GEPS final Zarr values against raw ensemble GRIB2 files."
    )
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--grib-root", type=Path, default=DEFAULT_GRIB_ROOT)
    parser.add_argument(
        "--base-time-pickle", type=Path, default=DEFAULT_BASE_TIME_PICKLE
    )
    parser.add_argument(
        "--variables",
        default="Precipitation_Prob,APCP_Mean,APCP_StdDev,AFRAIN_Mean,AICEP_Mean,ARAIN_Mean,ASNOW_Mean",
        help="Comma-separated final GEPS variables to check.",
    )
    parser.add_argument(
        "--source-hours",
        default="3,6,192,198,240",
        help="Comma-separated GEPS source forecast hours expected to match raw GRIB statistics.",
    )
    parser.add_argument(
        "--interp-hours",
        default="4,5,7,193,194,195,196,197,199,200,201,202,203,239",
        help="Comma-separated forecast hours expected to be produced by interpolation.",
    )
    parser.add_argument(
        "--detail-hours",
        default="3,4,5,6,7,8,9,192,193,194,195,196,197,198,199,200,201,202,203,204,234,235,236,237,238,239,240",
        help="Comma-separated forecast hours to print in detailed point tables.",
    )
    parser.add_argument(
        "--detail-vars",
        default="Precipitation_Prob,APCP_Mean,APCP_StdDev,AFRAIN_Mean,AICEP_Mean,ARAIN_Mean,ASNOW_Mean",
        help="Comma-separated variables to print as point detail tables.",
    )
    parser.add_argument(
        "--y", type=int, help="Latitude/y grid index for point-specific output."
    )
    parser.add_argument(
        "--x", type=int, help="Longitude/x grid index for point-specific output."
    )
    parser.add_argument(
        "--lat",
        type=float,
        help="Latitude for point-specific output; overrides --y if set.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude for point-specific output; overrides --x if set.",
    )
    parser.add_argument(
        "--points", type=int, default=1, help="Number of random grid points per check."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-3,
        help="Value tolerance; accommodates small cfgrib versus wgrib2 GRIB-decoding differences.",
    )
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=240,
        help="Allowed final time-coordinate drift from float32 epoch-second storage.",
    )
    parser.add_argument("--show-passes", action="store_true")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_variables(raw: str) -> list[str]:
    """Parse and validate comma-separated GEPS variable names."""
    variables = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(variables) - set(VARIABLE_SPECS))
    if unknown:
        raise ValueError(f"Unknown GEPS variables: {unknown}")
    return variables


def load_base_time(path: Path) -> pd.Timestamp:
    """Load the GEPS base run time from the ingest pickle."""
    with path.open("rb") as file:
        return pd.Timestamp(pickle.load(file))


def resolve_grib_run_dir(grib_root: Path, base_time: pd.Timestamp) -> Path:
    """Resolve either a parent GRIB cache path or a single GEPS run directory."""
    if any(grib_root.glob("*.grib2")):
        return grib_root

    expected_dir = grib_root / base_time.strftime("%Y%m%d")
    if any(expected_dir.glob("*.grib2")):
        return expected_dir

    candidates = sorted(
        path
        for path in grib_root.glob("*")
        if path.is_dir() and any(path.glob("*.grib2"))
    )
    if not candidates:
        raise FileNotFoundError(f"No GRIB2 files found under {grib_root}")
    return candidates[-1]


def validate_root_array(root: zarr.Array) -> None:
    """Validate the final Zarr layout expected by GEPS ingest."""
    if not isinstance(root, zarr.Array):
        raise TypeError("Expected final GEPS Zarr to be a root zarr.Array.")
    if root.ndim != 4:
        raise ValueError(
            "Expected final GEPS Zarr shape (variable, time, latitude, longitude)."
        )
    if root.shape[0] != len(ZARR_VARS):
        raise ValueError(f"Expected {len(ZARR_VARS)} variables, found {root.shape[0]}.")


def infer_forecast_offset(final_time_len: int) -> int:
    """Infer the final-array index where GEPS forecast hour zero would sit."""
    offset = final_time_len - max(GEPS_SOURCE_HOURS) - 1
    if offset < 0:
        raise ValueError(
            f"Final time length {final_time_len} is too short for GEPS hour {max(GEPS_SOURCE_HOURS)}."
        )
    return offset


def expected_unix_seconds(
    base_time: pd.Timestamp, forecast_offset: int, time_len: int
) -> np.ndarray:
    """Build the expected hourly final time axis from the GEPS run time."""
    hours = np.arange(-forecast_offset, time_len - forecast_offset, dtype=np.int64)
    return int(base_time.timestamp()) + hours * 3600


def grib_file_path(
    run_dir: Path, base_time: pd.Timestamp, grib_token: str, hour: int
) -> Path:
    """Find a GEPS all-members GRIB2 file for an accumulated field and hour."""
    run_token = base_time.strftime("%Y%m%d%H")
    matches = sorted(
        run_dir.glob(
            f"*geps-raw_{grib_token}_SFC_0_*_{run_token}_P{hour:03d}_allmbrs.grib2"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing GRIB for {grib_token} hour {hour} under {run_dir}"
        )
    return matches[0]


def grib_data_values(dataset: xr.Dataset) -> np.ndarray:
    """Return the sole precipitation field from one filtered GEPS GRIB dataset."""
    data_vars = list(dataset.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Expected one data variable, found {data_vars}")
    return np.asarray(dataset[data_vars[0]].values, dtype=np.float64)


@lru_cache(maxsize=256)
def load_grib_members_at_point(path: str, y: int, x: int) -> tuple[float, ...]:
    """Load the GEPS control and perturbed values at one raw-grid point."""
    member_values = []
    for data_type in ("cf", "pf"):
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": data_type}},
        )
        try:
            values = grib_data_values(dataset)
            if values.ndim == 2:
                member_values.extend([float(values[y, x])])
            elif values.ndim == 3:
                member_values.extend(float(value) for value in values[:, y, x])
            else:
                raise ValueError(
                    f"Unexpected GEPS member field shape {values.shape} in {path}"
                )
        finally:
            dataset.close()
    return tuple(member_values)


@lru_cache(maxsize=8)
def grib_grid_shape(path: str) -> tuple[int, int]:
    """Read the raw GEPS grid shape from the control-member GRIB message."""
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": "cf"}},
    )
    try:
        values = grib_data_values(dataset)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D GEPS control field, found {values.shape}")
        return values.shape
    finally:
        dataset.close()


@lru_cache(maxsize=8)
def grib_coordinates(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load GEPS latitude and longitude coordinates from the control GRIB message."""
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": "cf"}},
    )
    try:
        return (
            np.asarray(dataset["latitude"].values, dtype=np.float64),
            np.asarray(dataset["longitude"].values, dtype=np.float64),
        )
    finally:
        dataset.close()


def normalize_longitude(lon: float, longitudes: np.ndarray) -> float:
    """Normalize longitude to the coordinate convention used by the GEPS grid."""
    if float(np.nanmin(longitudes)) >= 0 and lon < 0:
        return lon % 360
    if float(np.nanmax(longitudes)) <= 180 and lon > 180:
        return ((lon + 180) % 360) - 180
    return lon


def nearest_point_from_lat_lon(
    path: Path, lat: float, lon: float
) -> tuple[int, int, float, float]:
    """Map latitude and longitude to the nearest raw GEPS grid point."""
    latitudes, longitudes = grib_coordinates(str(path))
    target_lon = normalize_longitude(lon, longitudes)
    y = int(np.nanargmin(np.abs(latitudes - lat)))
    x = int(np.nanargmin(np.abs(longitudes - target_lon)))
    return y, x, float(latitudes[y]), float(longitudes[x])


def stored_expected(value: float) -> float:
    """Match final ingest rounding and float32 storage."""
    if math.isnan(value):
        return value
    return float(np.float32(np.round(value, 5)))


def compare_values(
    zarr_value: float, expected_value: float, tolerance: float
) -> tuple[bool, float]:
    """Compare two scalar values with NaN handling."""
    if math.isnan(zarr_value) and math.isnan(expected_value):
        return True, 0.0
    if math.isnan(zarr_value) or math.isnan(expected_value):
        return False, math.inf
    diff = abs(zarr_value - expected_value)
    return diff <= tolerance, diff


def is_finite(value: float) -> bool:
    """Return True for finite scalar values."""
    return not math.isnan(value) and not math.isinf(value)


def previous_source_hour(hour: int) -> int | None:
    """Return the previous GEPS forecast source hour, if one exists."""
    previous = [source_hour for source_hour in GEPS_SOURCE_HOURS if source_hour < hour]
    return previous[-1] if previous else None


def source_members(
    run_dir: Path, base_time: pd.Timestamp, grib_token: str, hour: int, y: int, x: int
) -> np.ndarray:
    """Load all control and perturbed values for one GEPS source field and hour."""
    path = grib_file_path(run_dir, base_time, grib_token, hour)
    return np.asarray(load_grib_members_at_point(str(path), y, x), dtype=np.float64)


def source_statistic(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Rebuild one GEPS output statistic at a raw forecast source hour.

    Each difference is divided by its actual adjacent source interval.
    """
    current = source_members(run_dir, base_time, spec.grib_token, hour, y, x)
    previous_hour = previous_source_hour(hour)
    previous = (
        np.zeros_like(current)
        if previous_hour is None
        else source_members(run_dir, base_time, spec.grib_token, previous_hour, y, x)
    )
    if current.shape != previous.shape:
        raise ValueError(
            f"GEPS ensemble member count changed for {spec.grib_token} at hour {hour}: "
            f"{current.shape[0]} versus {previous.shape[0]}"
        )

    interval_hours = hour if previous_hour is None else hour - previous_hour
    per_hour = np.maximum(current - previous, 0) / interval_hours
    if spec.statistic == "mean":
        return float(per_hour.mean())
    if spec.statistic == "stddev":
        return float(per_hour.std())
    if spec.statistic == "probability":
        return float((per_hour > PRECIP_THRESHOLD).mean())
    raise ValueError(f"Unsupported statistic: {spec.statistic}")


def source_cumulative_mean(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Return the raw, pre-de-accumulation ensemble-mean total for diagnostics."""
    return float(source_members(run_dir, base_time, spec.grib_token, hour, y, x).mean())


def bracketing_source_hours(hour: int) -> tuple[int, int] | None:
    """Return GEPS source hours surrounding an interpolated target hour."""
    if hour in GEPS_SOURCE_HOURS:
        return None
    left = [source_hour for source_hour in GEPS_SOURCE_HOURS if source_hour < hour]
    right = [source_hour for source_hour in GEPS_SOURCE_HOURS if source_hour > hour]
    if not left or not right:
        return None
    return left[-1], right[0]


def nearest_finite_source_statistic(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    candidate_hours: list[int],
    y: int,
    x: int,
) -> tuple[int, float] | None:
    """Return the first finite statistic from ordered source-hour candidates."""
    for source_hour in candidate_hours:
        value = source_statistic(run_dir, base_time, spec, source_hour, y, x)
        if is_finite(value):
            return source_hour, value
    return None


def expected_nan_aware_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Mirror interp_time_map_blocks_nan for one GEPS statistic and grid point."""
    left = nearest_finite_source_statistic(
        run_dir,
        base_time,
        spec,
        [
            source_hour
            for source_hour in reversed(GEPS_SOURCE_HOURS)
            if source_hour <= hour
        ],
        y,
        x,
    )
    right = nearest_finite_source_statistic(
        run_dir,
        base_time,
        spec,
        [source_hour for source_hour in GEPS_SOURCE_HOURS if source_hour >= hour],
        y,
        x,
    )
    if left is None or right is None:
        return math.nan

    left_hour, left_value = left
    right_hour, right_value = right
    if left_hour == right_hour:
        return left_value

    weight = (hour - left_hour) / (right_hour - left_hour)
    return float((1 - weight) * left_value + weight * right_value)


def expected_interpolated_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Compute the expected GEPS output for a non-source forecast hour."""
    bracket = bracketing_source_hours(hour)
    if bracket is None:
        raise ValueError(f"Hour {hour} is not an interpolated GEPS source hour.")

    left_hour, right_hour = bracket
    left_value = source_statistic(run_dir, base_time, spec, left_hour, y, x)
    right_value = source_statistic(run_dir, base_time, spec, right_hour, y, x)
    if not is_finite(left_value) or not is_finite(right_value):
        return expected_nan_aware_value(run_dir, base_time, spec, hour, y, x)

    weight = (hour - left_hour) / (right_hour - left_hour)
    return float((1 - weight) * left_value + weight * right_value)


def sample_points(
    y_size: int, x_size: int, count: int, seed: int
) -> list[tuple[int, int]]:
    """Pick deterministic random raw-grid points."""
    rng = random.Random(seed)
    return [(rng.randrange(y_size), rng.randrange(x_size)) for _ in range(count)]


def point_from_args(
    args: argparse.Namespace,
    first_grib_path: Path,
    y_size: int,
    x_size: int,
) -> tuple[int, int, str]:
    """Resolve a requested point or choose one deterministic raw-grid point."""
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("--lat and --lon must be provided together.")
        y, x, grid_lat, grid_lon = nearest_point_from_lat_lon(
            first_grib_path, args.lat, args.lon
        )
        return (
            y,
            x,
            f"requested lat/lon=({args.lat}, {args.lon}) nearest_grid=({grid_lat}, {grid_lon})",
        )

    if args.y is not None or args.x is not None:
        if args.y is None or args.x is None:
            raise ValueError("--y and --x must be provided together.")
        y = int(args.y)
        x = int(args.x)
        if y < 0 or y >= y_size or x < 0 or x >= x_size:
            raise ValueError(
                f"Point y={y} x={x} is outside raw grid shape {(y_size, x_size)}."
            )
        return y, x, "requested grid indices"

    y, x = sample_points(y_size, x_size, 1, args.seed)[0]
    return y, x, f"deterministic random raw-grid point (seed={args.seed})"


def print_point_table(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    variable: str,
    y: int,
    x: int,
    hours: list[int],
    tolerance: float,
) -> tuple[int, int]:
    """Print source and interpolated GEPS values at one grid point."""
    spec = VARIABLE_SPECS[variable]
    var_index = ZARR_VARS.index(variable)
    failures = 0
    rows = 0

    print()
    print(f"{variable} point table y={y} x={x}")
    print(
        "hour valid_time           raw_total_mean source_hour source_stat "
        "left_hour left_stat right_hour right_stat zarr_value expected_value abs_diff status"
    )
    for hour in hours:
        time_index = forecast_offset + hour
        if time_index < 0 or time_index >= root.shape[1]:
            continue

        zarr_value = float(root[var_index, time_index, y, x])
        exact_source = hour if hour in GEPS_SOURCE_HOURS else None
        if exact_source is not None:
            raw_total_mean = source_cumulative_mean(
                run_dir, base_time, spec, hour, y, x
            )
            source_value = source_statistic(run_dir, base_time, spec, hour, y, x)
            left_hour = right_hour = hour
            left_value = right_value = source_value
            expected_value = stored_expected(source_value)
        else:
            raw_total_mean = math.nan
            source_value = math.nan
            bracket = bracketing_source_hours(hour)
            if bracket is None:
                left_hour = right_hour = None
                left_value = right_value = math.nan
                expected_value = math.nan
            else:
                left_hour, right_hour = bracket
                left_value = source_statistic(run_dir, base_time, spec, left_hour, y, x)
                right_value = source_statistic(
                    run_dir, base_time, spec, right_hour, y, x
                )
                expected_value = stored_expected(
                    expected_interpolated_value(run_dir, base_time, spec, hour, y, x)
                )

        if math.isnan(expected_value):
            diff = math.nan
            status = "NO_GRIB"
        else:
            passed, diff = compare_values(zarr_value, expected_value, tolerance)
            failures += 0 if passed else 1
            status = "PASS" if passed else "FAIL"

        rows += 1
        valid_time = base_time + pd.Timedelta(hours=hour)
        source_hour_text = str(exact_source) if exact_source is not None else "NA"
        left_hour_text = str(left_hour) if left_hour is not None else "NA"
        right_hour_text = str(right_hour) if right_hour is not None else "NA"
        print(
            f"{hour:>4d} {valid_time:%Y-%m-%d %H:%M} {raw_total_mean:>14.8g} "
            f"{source_hour_text:>11s} {source_value:>11.8g} {left_hour_text:>9s} "
            f"{left_value:>9.8g} {right_hour_text:>10s} {right_value:>10.8g} "
            f"{zarr_value:>10.8g} {expected_value:>14.8g} {diff:>8.3g} {status}"
        )

    return failures, rows


def check_time_axis(
    root: zarr.Array,
    base_time: pd.Timestamp,
    forecast_offset: int,
    tolerance_seconds: float,
) -> bool:
    """Check that the final time variable follows the expected hourly cadence."""
    expected = expected_unix_seconds(base_time, forecast_offset, root.shape[1])
    actual = np.asarray(root[ZARR_VARS.index("time"), :, 0, 0], dtype=np.float64)
    max_diff = float(np.nanmax(np.abs(actual - expected)))
    passed = bool(max_diff <= tolerance_seconds)
    print(
        f"{'PASS' if passed else 'FAIL'} time-axis max_abs_diff_seconds={max_diff:.1f} "
        f"tolerance={tolerance_seconds}"
    )
    return passed


def run_source_checks(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    variables: list[str],
    source_hours: list[int],
    points: list[tuple[int, int]],
    tolerance: float,
) -> list[CheckResult]:
    """Compare final GEPS Zarr values at raw source forecast hours."""
    results = []
    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        var_index = ZARR_VARS.index(variable)
        for hour in source_hours:
            if hour not in GEPS_SOURCE_HOURS:
                continue
            time_index = forecast_offset + hour
            if time_index < 0 or time_index >= root.shape[1]:
                continue
            for y, x in points:
                raw_value = source_statistic(run_dir, base_time, spec, hour, y, x)
                expected_value = stored_expected(
                    raw_value
                    if is_finite(raw_value)
                    else expected_nan_aware_value(run_dir, base_time, spec, hour, y, x)
                )
                zarr_value = float(root[var_index, time_index, y, x])
                passed, diff = compare_values(zarr_value, expected_value, tolerance)
                results.append(
                    CheckResult(
                        "source",
                        variable,
                        hour,
                        y,
                        x,
                        zarr_value,
                        expected_value,
                        diff,
                        passed,
                    )
                )
    return results


def run_interpolation_checks(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    variables: list[str],
    interp_hours: list[int],
    points: list[tuple[int, int]],
    tolerance: float,
) -> list[CheckResult]:
    """Compare final GEPS Zarr values at temporally interpolated forecast hours."""
    results = []
    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        var_index = ZARR_VARS.index(variable)
        for hour in interp_hours:
            bracket = bracketing_source_hours(hour)
            if bracket is None:
                continue
            time_index = forecast_offset + hour
            if time_index < 0 or time_index >= root.shape[1]:
                continue
            for y, x in points:
                expected_value = stored_expected(
                    expected_interpolated_value(run_dir, base_time, spec, hour, y, x)
                )
                zarr_value = float(root[var_index, time_index, y, x])
                passed, diff = compare_values(zarr_value, expected_value, tolerance)
                results.append(
                    CheckResult(
                        "interp",
                        variable,
                        hour,
                        y,
                        x,
                        zarr_value,
                        expected_value,
                        diff,
                        passed,
                        note=f"bracket={bracket[0]}-{bracket[1]} method=linear",
                    )
                )
    return results


def print_results(results: list[CheckResult], show_passes: bool) -> int:
    """Print comparison results and return the failure count."""
    failures = 0
    for result in results:
        if not result.passed:
            failures += 1
        if result.passed and not show_passes:
            continue
        print(
            f"{'PASS' if result.passed else 'FAIL'} {result.kind:6s} {result.variable:18s} "
            f"h={result.hour:03d} y={result.y:03d} x={result.x:03d} "
            f"zarr={result.zarr_value:.8g} expected={result.expected_value:.8g} "
            f"abs_diff={result.abs_diff:.8g} {result.note}"
        )
    return failures


def main() -> None:
    """Run GEPS spot checks."""
    args = parse_args()
    variables = parse_csv_variables(args.variables)
    detail_vars = parse_csv_variables(args.detail_vars)
    source_hours = parse_csv_ints(args.source_hours)
    interp_hours = parse_csv_ints(args.interp_hours)
    detail_hours = parse_csv_ints(args.detail_hours)

    base_time = load_base_time(args.base_time_pickle)
    run_dir = resolve_grib_run_dir(args.grib_root, base_time)
    root = zarr.open(str(args.zarr_path), mode="r")
    validate_root_array(root)
    forecast_offset = infer_forecast_offset(root.shape[1])

    first_path = grib_file_path(run_dir, base_time, "APCP", GEPS_SOURCE_HOURS[0])
    raw_y_size, raw_x_size = grib_grid_shape(str(first_path))
    if root.shape[2] < raw_y_size or root.shape[3] < raw_x_size:
        raise ValueError(
            f"Final GEPS spatial shape {root.shape[2:]} is smaller than raw grid "
            f"{(raw_y_size, raw_x_size)}."
        )
    y, x, point_note = point_from_args(args, first_path, raw_y_size, raw_x_size)
    random_points = sample_points(raw_y_size, raw_x_size, args.points, args.seed)
    points = [(y, x), *[point for point in random_points if point != (y, x)]]

    print(f"Final Zarr: {args.zarr_path} shape={root.shape} chunks={root.chunks}")
    print(f"GRIB run dir: {run_dir}")
    print(f"Base time: {base_time}")
    print(f"Forecast hour offset: {forecast_offset}")
    print(f"Selected point: y={y} x={x} ({point_note})")
    print(f"Check points: {points}")
    print(
        "Source statistics de-accumulate each member from the prior GEPS source hour, "
        "then divide by the adjacent interval (3 hours through lead 192 and 6 hours "
        "thereafter) before clamping negatives."
    )
    print()

    time_passed = check_time_axis(
        root, base_time, forecast_offset, args.time_tolerance_seconds
    )
    source_results = run_source_checks(
        root,
        run_dir,
        base_time,
        forecast_offset,
        variables,
        source_hours,
        points,
        args.tolerance,
    )
    interp_results = run_interpolation_checks(
        root,
        run_dir,
        base_time,
        forecast_offset,
        variables,
        interp_hours,
        points,
        args.tolerance,
    )

    print()
    failures = 0 if time_passed else 1
    failures += print_results(source_results + interp_results, args.show_passes)
    detail_rows = 0
    for variable in detail_vars:
        table_failures, table_rows = print_point_table(
            root,
            run_dir,
            base_time,
            forecast_offset,
            variable,
            y,
            x,
            detail_hours,
            args.tolerance,
        )
        failures += table_failures
        detail_rows += table_rows

    total = len(source_results) + len(interp_results) + detail_rows + 1
    print()
    print(
        f"Summary: passed={total - failures} failed={failures} total={total} tolerance={args.tolerance}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
