"""Spot-check a REPS final Zarr against cached raw ensemble GRIB2 files.

REPS writes ensemble precipitation statistics to a root 4D Zarr array with
dimensions (variable, time, y, x). This script rebuilds the per-member
three-hour accumulations, then validates the production ensemble statistics
at source and interpolated hours.
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

DEFAULT_ZARR_PATH = Path("/mnt/nvme/data/Prod/v30/REPS.zarr")
DEFAULT_GRIB_ROOT = Path("/mnt/nvme/data/Process/REPS/Downloads/herbie/reps")
DEFAULT_BASE_TIME_PICKLE = Path("/mnt/nvme/data/Prod/v30/REPS.time.pickle")

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
REPS_SOURCE_HOURS = tuple(range(3, 73, 3))
REPS_ACCUMULATION_HOURS = 3
PRECIP_THRESHOLD = 0.1


@dataclass(frozen=True)
class VariableSpec:
    """Map one REPS output statistic to its accumulated GRIB field."""

    zarr_name: str
    grib_token: str
    statistic: str


@dataclass(frozen=True)
class CheckResult:
    """One source or interpolation comparison."""

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


VARIABLE_SPECS = {
    "Precipitation_Prob": VariableSpec("Precipitation_Prob", "APCP", "probability"),
    "APCP_Mean": VariableSpec("APCP_Mean", "APCP", "mean"),
    "APCP_StdDev": VariableSpec("APCP_StdDev", "APCP", "stddev"),
    "AFRAIN_Mean": VariableSpec("AFRAIN_Mean", "AFRAIN", "mean"),
    "AICEP_Mean": VariableSpec("AICEP_Mean", "AICEP", "mean"),
    "ARAIN_Mean": VariableSpec("ARAIN_Mean", "ARAIN", "mean"),
    "ASNOW_Mean": VariableSpec("ASNOW_Mean", "ASNOW", "mean"),
}


def parse_args() -> argparse.Namespace:
    """Parse REPS spot-check options."""
    parser = argparse.ArgumentParser(
        description="Spot-check REPS final Zarr values against raw ensemble GRIB2 files."
    )
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--grib-root", type=Path, default=DEFAULT_GRIB_ROOT)
    parser.add_argument(
        "--base-time-pickle", type=Path, default=DEFAULT_BASE_TIME_PICKLE
    )
    parser.add_argument(
        "--variables",
        default=",".join(VARIABLE_SPECS),
        help="Comma-separated REPS statistics to check.",
    )
    parser.add_argument(
        "--source-hours",
        default="3,6,24,69,72",
        help="Comma-separated REPS source hours to compare directly to GRIBs.",
    )
    parser.add_argument(
        "--interp-hours",
        default="4,5,7,70,71",
        help="Comma-separated hourly values to compare to linear interpolation.",
    )
    parser.add_argument(
        "--detail-hours",
        default="3,4,5,6,7,8,9,66,67,68,69,70,71,72",
        help="Comma-separated hours to print in the detailed point tables.",
    )
    parser.add_argument(
        "--detail-vars",
        default="Precipitation_Prob,APCP_Mean,APCP_StdDev,AFRAIN_Mean,AICEP_Mean,ARAIN_Mean,ASNOW_Mean",
        help="Comma-separated REPS statistics to print in detailed point tables.",
    )
    parser.add_argument("--y", type=int, help="Raw REPS y grid index.")
    parser.add_argument("--x", type=int, help="Raw REPS x grid index.")
    parser.add_argument("--lat", type=float, help="Latitude for point output.")
    parser.add_argument("--lon", type=float, help="Longitude for point output.")
    parser.add_argument(
        "--points", type=int, default=1, help="Additional random grid points per check."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-3,
        help="Absolute tolerance for decoded and float32-stored values.",
    )
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=240,
        help="Allowed time-coordinate drift from float32 epoch-second storage.",
    )
    parser.add_argument("--show-passes", action="store_true")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_variables(raw: str) -> list[str]:
    """Parse and validate comma-separated REPS statistic names."""
    variables = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(variables) - set(VARIABLE_SPECS))
    if unknown:
        raise ValueError(f"Unknown REPS variables: {unknown}")
    return variables


def load_base_time(path: Path) -> pd.Timestamp:
    """Load the REPS base run time written by the ingest."""
    with path.open("rb") as file:
        return pd.Timestamp(pickle.load(file))


def resolve_grib_run_dir(grib_root: Path, base_time: pd.Timestamp) -> Path:
    """Resolve a REPS cache parent path to the matching run directory."""
    if any(grib_root.glob("*.grib2")):
        return grib_root

    expected_dir = grib_root / base_time.strftime("%Y%m%d")
    if any(expected_dir.glob("*.grib2")):
        return expected_dir

    raise FileNotFoundError(
        f"No GRIB directory for REPS run {base_time:%Y-%m-%d %HZ} under {grib_root}"
    )


def grib_file_path(
    run_dir: Path, base_time: pd.Timestamp, grib_token: str, hour: int
) -> Path:
    """Find one REPS all-member GRIB2 accumulation field."""
    filename = (
        f"{base_time:%Y%m%dT%HZ}_MSC_REPS_{grib_token}_SFC_"
        f"RLatLon0.09x0.09_PT{hour:03d}H.grib2"
    )
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing GRIB for {grib_token} hour {hour}: {path}")
    return path


def grib_data_values(dataset: xr.Dataset, path: str) -> np.ndarray:
    """Return the sole field from one control or perturbed-member REPS dataset."""
    data_vars = list(dataset.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Expected one data variable in {path}, found {data_vars}")
    return np.asarray(dataset[data_vars[0]].values, dtype=np.float64)


@lru_cache(maxsize=256)
def load_grib_members_at_point(path: str, y: int, x: int) -> tuple[float, ...]:
    """Read the REPS control and perturbed-member values at one raw-grid point."""
    member_values: list[float] = []
    for data_type in ("cf", "pf"):
        dataset = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": data_type}},
        )
        try:
            values = grib_data_values(dataset, path)
            if values.ndim == 2:
                member_values.append(float(values[y, x]))
            elif values.ndim == 3:
                member_values.extend(float(value) for value in values[:, y, x])
            else:
                raise ValueError(
                    f"Unexpected REPS member field shape {values.shape} in {path}"
                )
        finally:
            dataset.close()
    return tuple(member_values)


@lru_cache(maxsize=8)
def grib_grid(path: str) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Read raw REPS shape and curvilinear latitude/longitude coordinates."""
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"dataType": "cf"}},
    )
    try:
        values = grib_data_values(dataset, path)
        if values.ndim != 2:
            raise ValueError(f"Expected 2D REPS control field, found {values.shape}")
        return (
            values.shape,
            np.asarray(dataset["latitude"].values, dtype=np.float64),
            np.asarray(dataset["longitude"].values, dtype=np.float64),
        )
    finally:
        dataset.close()


def validate_root_array(root: zarr.Array) -> None:
    """Validate the final Zarr layout written by REPS ingest."""
    if not isinstance(root, zarr.Array):
        raise TypeError("Expected final REPS Zarr to be a root zarr.Array.")
    if root.ndim != 4:
        raise ValueError("Expected final REPS Zarr shape (variable, time, y, x).")
    if root.shape[0] != len(ZARR_VARS):
        raise ValueError(f"Expected {len(ZARR_VARS)} variables, found {root.shape[0]}.")


def infer_forecast_offset(final_time_len: int) -> int:
    """Infer the final-array index corresponding to forecast hour zero."""
    offset = final_time_len - max(REPS_SOURCE_HOURS) - 1
    if offset < 0:
        raise ValueError(
            f"Final time length {final_time_len} is too short for REPS hour 72."
        )
    return offset


def expected_unix_seconds(
    base_time: pd.Timestamp, forecast_offset: int, time_len: int
) -> np.ndarray:
    """Build the expected hourly production time coordinate."""
    hours = np.arange(-forecast_offset, time_len - forecast_offset, dtype=np.int64)
    return int(base_time.timestamp()) + hours * 3600


def stored_expected(value: float) -> float:
    """Match final ingest rounding and float32 storage."""
    if not math.isfinite(value):
        return math.nan
    return float(np.float32(np.round(value, 5)))


def compare_values(
    zarr_value: float, expected_value: float, tolerance: float
) -> tuple[bool, float]:
    """Compare scalars with NaN handling."""
    if math.isnan(zarr_value) and math.isnan(expected_value):
        return True, 0.0
    if math.isnan(zarr_value) or math.isnan(expected_value):
        return False, math.inf
    difference = abs(zarr_value - expected_value)
    return difference <= tolerance, difference


def source_members(
    run_dir: Path, base_time: pd.Timestamp, grib_token: str, hour: int, y: int, x: int
) -> np.ndarray:
    """Load all REPS members for one source accumulation field and point."""
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
    """Rebuild one REPS statistic exactly as the forecast ingest does."""
    current = source_members(run_dir, base_time, spec.grib_token, hour, y, x)
    previous_hours = [
        source_hour for source_hour in REPS_SOURCE_HOURS if source_hour < hour
    ]
    previous = (
        np.zeros_like(current)
        if not previous_hours
        else source_members(
            run_dir, base_time, spec.grib_token, previous_hours[-1], y, x
        )
    )
    if current.shape != previous.shape:
        raise ValueError(
            f"REPS member count changed for {spec.grib_token} at hour {hour}: "
            f"{current.shape[0]} versus {previous.shape[0]}"
        )

    per_hour = np.maximum(current - previous, 0) / REPS_ACCUMULATION_HOURS
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
    """Return raw pre-de-accumulation ensemble mean for detailed diagnostics."""
    return float(source_members(run_dir, base_time, spec.grib_token, hour, y, x).mean())


def bracketing_source_hours(hour: int) -> tuple[int, int] | None:
    """Return REPS source hours around an interpolated forecast hour."""
    if hour in REPS_SOURCE_HOURS:
        return None
    left = [source_hour for source_hour in REPS_SOURCE_HOURS if source_hour < hour]
    right = [source_hour for source_hour in REPS_SOURCE_HOURS if source_hour > hour]
    if not left or not right:
        return None
    return left[-1], right[0]


def expected_output_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> tuple[float, int | None, float, int | None, float]:
    """Mirror REPS's NaN-aware linear interpolation at one point and hour."""
    if hour in REPS_SOURCE_HOURS:
        value = source_statistic(run_dir, base_time, spec, hour, y, x)
        return value, hour, value, hour, value

    bracket = bracketing_source_hours(hour)
    if bracket is None:
        return math.nan, None, math.nan, None, math.nan
    left_hour, right_hour = bracket
    left_value = source_statistic(run_dir, base_time, spec, left_hour, y, x)
    right_value = source_statistic(run_dir, base_time, spec, right_hour, y, x)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return math.nan, left_hour, left_value, right_hour, right_value
    weight = (hour - left_hour) / (right_hour - left_hour)
    expected = (1 - weight) * left_value + weight * right_value
    return expected, left_hour, left_value, right_hour, right_value


def nearest_point_from_lat_lon(
    latitudes: np.ndarray, longitudes: np.ndarray, lat: float, lon: float
) -> tuple[int, int, float, float]:
    """Map a latitude/longitude request to the nearest curvilinear REPS cell."""
    lon_delta = ((longitudes - lon + 180) % 360) - 180
    lon_scale = math.cos(math.radians(lat))
    distance = np.square(latitudes - lat) + np.square(lon_delta * lon_scale)
    y, x = np.unravel_index(np.nanargmin(distance), distance.shape)
    return int(y), int(x), float(latitudes[y, x]), float(longitudes[y, x])


def sample_points(
    y_size: int, x_size: int, count: int, seed: int
) -> list[tuple[int, int]]:
    """Pick deterministic random raw-grid points."""
    rng = random.Random(seed)
    return [(rng.randrange(y_size), rng.randrange(x_size)) for _ in range(count)]


def point_from_args(
    args: argparse.Namespace,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[int, int, str]:
    """Resolve an explicit point, lat/lon request, or deterministic default."""
    y_size, x_size = latitudes.shape
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("--lat and --lon must be provided together.")
        y, x, grid_lat, grid_lon = nearest_point_from_lat_lon(
            latitudes, longitudes, args.lat, args.lon
        )
        return (
            y,
            x,
            f"requested lat/lon=({args.lat}, {args.lon}) nearest_grid=({grid_lat}, {grid_lon})",
        )
    if args.y is not None or args.x is not None:
        if args.y is None or args.x is None:
            raise ValueError("--y and --x must be provided together.")
        if args.y < 0 or args.y >= y_size or args.x < 0 or args.x >= x_size:
            raise ValueError(
                f"Point y={args.y} x={args.x} is outside raw grid {(y_size, x_size)}."
            )
        return args.y, args.x, "requested grid indices"
    y, x = sample_points(y_size, x_size, 1, args.seed)[0]
    return y, x, f"deterministic random raw-grid point (seed={args.seed})"


def check_time_axis(
    root: zarr.Array,
    base_time: pd.Timestamp,
    forecast_offset: int,
    tolerance_seconds: float,
) -> bool:
    """Check that production time values follow the expected hourly cadence."""
    expected = expected_unix_seconds(base_time, forecast_offset, root.shape[1])
    actual = np.asarray(root[0, :, 0, 0], dtype=np.float64)
    max_difference = float(np.nanmax(np.abs(actual - expected)))
    passed = max_difference <= tolerance_seconds
    print(
        f"{'PASS' if passed else 'FAIL'} time-axis max_abs_diff_seconds={max_difference:.1f} "
        f"tolerance={tolerance_seconds}"
    )
    return passed


def run_checks(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    variables: list[str],
    hours: list[int],
    points: list[tuple[int, int]],
    kind: str,
    tolerance: float,
) -> list[CheckResult]:
    """Compare source or interpolated REPS values at sampled points."""
    results = []
    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        var_index = ZARR_VARS.index(variable)
        for hour in hours:
            is_source = hour in REPS_SOURCE_HOURS
            if (kind == "source") != is_source:
                continue
            time_index = forecast_offset + hour
            if time_index < 0 or time_index >= root.shape[1]:
                continue
            for y, x in points:
                expected, left_hour, _, right_hour, _ = expected_output_value(
                    run_dir, base_time, spec, hour, y, x
                )
                expected = stored_expected(expected)
                zarr_value = float(root[var_index, time_index, y, x])
                passed, difference = compare_values(zarr_value, expected, tolerance)
                note = (
                    ""
                    if is_source
                    else f"bracket={left_hour}-{right_hour} method=linear"
                )
                results.append(
                    CheckResult(
                        kind,
                        variable,
                        hour,
                        y,
                        x,
                        zarr_value,
                        expected,
                        difference,
                        passed,
                        note,
                    )
                )
    return results


def print_results(results: list[CheckResult], show_passes: bool) -> int:
    """Print failed checks, or all checks when requested, and return failures."""
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
    """Print detailed raw, interpolation-source, and final values at one point."""
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
        expected, left_hour, left_value, right_hour, right_value = (
            expected_output_value(run_dir, base_time, spec, hour, y, x)
        )
        expected = stored_expected(expected)
        zarr_value = float(root[var_index, time_index, y, x])
        passed, difference = compare_values(zarr_value, expected, tolerance)
        failures += 0 if passed else 1
        rows += 1
        exact_source = hour if hour in REPS_SOURCE_HOURS else None
        raw_total_mean = (
            source_cumulative_mean(run_dir, base_time, spec, hour, y, x)
            if exact_source is not None
            else math.nan
        )
        source_stat = left_value if exact_source is not None else math.nan
        valid_time = base_time + pd.Timedelta(hours=hour)
        source_text = str(exact_source) if exact_source is not None else "NA"
        left_text = str(left_hour) if left_hour is not None else "NA"
        right_text = str(right_hour) if right_hour is not None else "NA"
        print(
            f"{hour:>4d} {valid_time:%Y-%m-%d %H:%M} {raw_total_mean:>14.8g} "
            f"{source_text:>11s} {source_stat:>11.8g} {left_text:>9s} "
            f"{left_value:>9.8g} {right_text:>10s} {right_value:>10.8g} "
            f"{zarr_value:>10.8g} {expected:>14.8g} {difference:>8.3g} "
            f"{'PASS' if passed else 'FAIL'}"
        )
    return failures, rows


def main() -> None:
    """Run REPS raw-GRIB versus final-Zarr spot checks."""
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

    first_path = grib_file_path(run_dir, base_time, "APCP", REPS_SOURCE_HOURS[0])
    raw_shape, latitudes, longitudes = grib_grid(str(first_path))
    if root.shape[2] < raw_shape[0] or root.shape[3] < raw_shape[1]:
        raise ValueError(
            f"Final REPS spatial shape {root.shape[2:]} is smaller than raw grid {raw_shape}."
        )
    y, x, point_note = point_from_args(args, latitudes, longitudes)
    random_points = sample_points(raw_shape[0], raw_shape[1], args.points, args.seed)
    points = [(y, x), *[point for point in random_points if point != (y, x)]]

    print(f"Final Zarr: {args.zarr_path} shape={root.shape} chunks={root.chunks}")
    print(f"GRIB run dir: {run_dir}")
    print(f"Base time: {base_time}")
    print(f"Forecast hour offset: {forecast_offset}")
    print(f"Selected point: y={y} x={x} ({point_note})")
    print(f"Check points: {points}")
    print(
        "Source statistics de-accumulate each member from the prior 3-hour REPS "
        "source, clamp negatives, and divide by 3 to produce mm/h."
    )
    print()

    time_passed = check_time_axis(
        root, base_time, forecast_offset, args.time_tolerance_seconds
    )
    source_results = run_checks(
        root,
        run_dir,
        base_time,
        forecast_offset,
        variables,
        source_hours,
        points,
        "source",
        args.tolerance,
    )
    interpolation_results = run_checks(
        root,
        run_dir,
        base_time,
        forecast_offset,
        variables,
        interp_hours,
        points,
        "interp",
        args.tolerance,
    )

    failures = 0 if time_passed else 1
    failures += print_results(source_results + interpolation_results, args.show_passes)
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

    total = len(source_results) + len(interpolation_results) + detail_rows + 1
    print()
    print(
        f"Summary: passed={total - failures} failed={failures} total={total} "
        f"tolerance={args.tolerance}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
