"""Spot-check an HRDPS final Zarr against cached raw GRIB2 files.

The HRDPS ingest writes a root 4D Zarr array with dimensions
(variable, time, y, x). This script rebuilds source values from the matching
Herbie GRIB cache, including APCP de-accumulation, validity masking, and the
NaN-aware temporal interpolation used by the production ingest.
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

DEFAULT_ZARR_PATH = Path("/mnt/nvme/data/Prod/v30/HRDPS.zarr")
DEFAULT_GRIB_ROOT = Path("/mnt/nvme/data/Process/HRDPS/Downloads/herbie/hrdps")
DEFAULT_BASE_TIME_PICKLE = Path("/mnt/nvme/data/Prod/v30/HRDPS.time.pickle")

ZARR_VARS = (
    "time",
    "GUST_10maboveground",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "WIND_10maboveground",
    "WDIR_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "PTYPE_surface",
    "TCDC_surface",
    "UVI_surface",
    "DSWRF_surface",
    "CAPE_surface",
    "PRES_surface",
    "LFTX_500mb",
    "VVEL_500mb",
)

HRDPS_SOURCE_HOURS = tuple(range(1, 49))
VALID_DATA_MIN = -100
VALID_DATA_MAX = 120000
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class VariableSpec:
    """Map an HRDPS final variable to its raw GRIB file."""

    zarr_name: str
    file_token: str
    nearest: bool = False
    deaccumulate: bool = False
    energy_to_flux: bool = False


VARIABLE_SPECS = {
    "GUST_10maboveground": VariableSpec("GUST_10maboveground", "GUST_AGL-10m"),
    "PRMSL_meansealevel": VariableSpec("PRMSL_meansealevel", "PRMSL_MSL"),
    "TMP_2maboveground": VariableSpec("TMP_2maboveground", "TMP_AGL-2m"),
    "DPT_2maboveground": VariableSpec("DPT_2maboveground", "DPT_AGL-2m"),
    "RH_2maboveground": VariableSpec("RH_2maboveground", "RH_AGL-2m"),
    "WIND_10maboveground": VariableSpec("WIND_10maboveground", "WIND_AGL-10m"),
    "WDIR_10maboveground": VariableSpec("WDIR_10maboveground", "WDIR_AGL-10m"),
    "PRATE_surface": VariableSpec("PRATE_surface", "PRATE_Sfc"),
    "APCP_surface": VariableSpec("APCP_surface", "APCP_Sfc", deaccumulate=True),
    "PTYPE_surface": VariableSpec("PTYPE_surface", "PTYPE_Sfc", nearest=True),
    "TCDC_surface": VariableSpec("TCDC_surface", "TCDC_Sfc"),
    "UVI_surface": VariableSpec("UVI_surface", "UVIndex_Sfc"),
    "DSWRF_surface": VariableSpec(
        "DSWRF_surface", "DSWRF_Sfc", deaccumulate=True, energy_to_flux=True
    ),
    "CAPE_surface": VariableSpec("CAPE_surface", "CAPE_Sfc"),
    "PRES_surface": VariableSpec("PRES_surface", "PRES_Sfc"),
    "LFTX_500mb": VariableSpec("LFTX_500mb", "LFTX_ISBL_0500"),
    "VVEL_500mb": VariableSpec("VVEL_500mb", "VVEL_ISBL_0500"),
}


@dataclass(frozen=True)
class CheckResult:
    """One sampled comparison result."""

    variable: str
    hour: int
    y: int
    x: int
    zarr_value: float
    expected_value: float
    abs_diff: float
    allowed_diff: float
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class SelectedPoint:
    """A point selected for validation output."""

    y: int
    x: int
    note: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spot-check HRDPS final Zarr values against raw GRIB2 files."
    )
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--grib-root", type=Path, default=DEFAULT_GRIB_ROOT)
    parser.add_argument(
        "--base-time-pickle", type=Path, default=DEFAULT_BASE_TIME_PICKLE
    )
    parser.add_argument(
        "--variables",
        default="PTYPE_surface,APCP_surface,UVI_surface,DSWRF_surface",
        help="Comma-separated final HRDPS variables to check.",
    )
    parser.add_argument(
        "--hours",
        default="1,2,3,6,12,24,36,42,47,48",
        help="Comma-separated forecast hours to compare against source GRIBs.",
    )
    parser.add_argument(
        "--detail-hours",
        default="1,2,3,4,5,6,12,24,36,37,38,39,40,41,42,43,44,45,46,47,48",
        help="Comma-separated forecast hours to print in detailed point tables.",
    )
    parser.add_argument(
        "--detail-vars",
        default="PTYPE_surface,APCP_surface,UVI_surface,DSWRF_surface",
        help="Comma-separated variables to print as point detail tables.",
    )
    parser.add_argument(
        "--find-ptype-points",
        type=int,
        default=1,
        help="Automatically select this many points where raw PTYPE changes over time.",
    )
    parser.add_argument(
        "--ptype-search-hours",
        default="1,6,12,18,24,30,36,42,48",
        help="Comma-separated source hours used to find changing-PTYPE locations.",
    )
    parser.add_argument(
        "--ptype-search-candidates",
        type=int,
        default=20000,
        help="Number of deterministic random cells sampled during the PTYPE search.",
    )
    parser.add_argument("--y", type=int, help="Raw HRDPS y grid index.")
    parser.add_argument("--x", type=int, help="Raw HRDPS x grid index.")
    parser.add_argument(
        "--lat", type=float, help="Latitude for point output; overrides --y if set."
    )
    parser.add_argument(
        "--lon", type=float, help="Longitude for point output; overrides --x if set."
    )
    parser.add_argument(
        "--points", type=int, default=1, help="Additional random points per check."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=180,
        help="Allowed time-coordinate drift from float32 epoch-second storage.",
    )
    parser.add_argument("--show-passes", action="store_true")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_variables(raw: str) -> list[str]:
    """Parse and validate comma-separated HRDPS variables."""
    variables = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(variables) - set(VARIABLE_SPECS))
    if unknown:
        raise ValueError(f"Unknown HRDPS variables: {unknown}")
    return variables


def load_base_time(path: Path) -> pd.Timestamp:
    """Load the HRDPS base run time from the ingest pickle."""
    with path.open("rb") as file:
        return pd.Timestamp(pickle.load(file))


def resolve_grib_run_dir(grib_root: Path, base_time: pd.Timestamp) -> Path:
    """Resolve either a parent cache path or the dated HRDPS run directory."""
    if any(grib_root.glob("*.grib2")):
        return grib_root

    expected_dir = grib_root / base_time.strftime("%Y%m%d")
    if any(expected_dir.glob("*.grib2")):
        return expected_dir

    raise FileNotFoundError(
        f"No GRIB directory for HRDPS run {base_time:%Y-%m-%d %HZ} under {grib_root}"
    )


def grib_file_path(
    run_dir: Path, base_time: pd.Timestamp, spec: VariableSpec, hour: int
) -> Path:
    """Find the exact HRDPS cycle, variable, and forecast-hour GRIB file."""
    filename = (
        f"{base_time:%Y%m%dT%HZ}_MSC_HRDPS_{spec.file_token}_"
        f"RLatLon0.0225_PT{hour:03d}H.grib2"
    )
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GRIB for {spec.zarr_name} hour {hour}: {path}"
        )
    return path


def grib_data_array(dataset: xr.Dataset, path: str) -> xr.DataArray:
    """Return the sole field from a single-variable HRDPS GRIB dataset."""
    data_vars = list(dataset.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Expected one data variable in {path}, found {data_vars}")
    return dataset[data_vars[0]]


@lru_cache(maxsize=512)
def load_grib_value_at_point(path: str, y: int, x: int) -> float:
    """Read one HRDPS GRIB value at a grid point."""
    dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        values = grib_data_array(dataset, path).values
        return float(values[y, x])
    finally:
        dataset.close()


def load_grib_array(path: Path) -> np.ndarray:
    """Read a complete HRDPS GRIB field for point selection."""
    dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        return np.asarray(grib_data_array(dataset, str(path)).values)
    finally:
        dataset.close()


def grib_grid(path: Path) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Read HRDPS field shape and curvilinear latitude/longitude coordinates."""
    dataset = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        shape = grib_data_array(dataset, str(path)).shape
        latitudes = np.asarray(dataset["latitude"].values, dtype=np.float64)
        longitudes = np.asarray(dataset["longitude"].values, dtype=np.float64)
        return shape, latitudes, longitudes
    finally:
        dataset.close()


def validate_root_array(root: zarr.Array) -> None:
    """Validate the final Zarr layout expected by HRDPS ingest."""
    if not isinstance(root, zarr.Array):
        raise TypeError("Expected final HRDPS Zarr to be a root zarr.Array.")
    if root.ndim != 4:
        raise ValueError("Expected final HRDPS Zarr shape (variable, time, y, x).")
    if root.shape[0] != len(ZARR_VARS):
        raise ValueError(f"Expected {len(ZARR_VARS)} variables, found {root.shape[0]}.")


def infer_forecast_offset(final_time_len: int) -> int:
    """Infer the final-array index where forecast hour zero would sit."""
    offset = final_time_len - max(HRDPS_SOURCE_HOURS) - 1
    if offset < 0:
        raise ValueError(
            f"Final time length {final_time_len} is too short for HRDPS hour 48."
        )
    return offset


def expected_unix_seconds(
    base_time: pd.Timestamp, forecast_offset: int, time_len: int
) -> np.ndarray:
    """Build the expected hourly final time axis."""
    hours = np.arange(-forecast_offset, time_len - forecast_offset, dtype=np.int64)
    return int(base_time.timestamp()) + hours * 3600


def nearest_point_from_lat_lon(
    latitudes: np.ndarray, longitudes: np.ndarray, lat: float, lon: float
) -> tuple[int, int, float, float]:
    """Map latitude/longitude to the nearest curvilinear HRDPS grid point."""
    lon_delta = ((longitudes - lon + 180) % 360) - 180
    lon_scale = math.cos(math.radians(lat))
    distance = np.square(latitudes - lat) + np.square(lon_delta * lon_scale)
    y, x = np.unravel_index(np.nanargmin(distance), distance.shape)
    return int(y), int(x), float(latitudes[y, x]), float(longitudes[y, x])


def raw_grib_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Read a raw value transformed as it is before HRDPS stacking."""
    path = grib_file_path(run_dir, base_time, spec, hour)
    current = load_grib_value_at_point(str(path), y, x)
    if not spec.deaccumulate:
        return current

    previous = 0.0
    if hour > HRDPS_SOURCE_HOURS[0]:
        previous_path = grib_file_path(run_dir, base_time, spec, hour - 1)
        previous = load_grib_value_at_point(str(previous_path), y, x)
    increment = current - previous
    if spec.energy_to_flux:
        return max(increment / SECONDS_PER_HOUR, 0)
    return increment


def mask_source_value(value: float) -> float:
    """Apply the global validity mask used before HRDPS interpolation."""
    if not math.isfinite(value) or value < VALID_DATA_MIN or value > VALID_DATA_MAX:
        return math.nan
    return value


def stored_expected(value: float) -> float:
    """Match final ingest rounding and float32 storage."""
    if math.isnan(value):
        return value
    return float(np.float32(np.round(value, 5)))


def compare_values(
    zarr_value: float, expected_value: float, atol: float, rtol: float
) -> tuple[bool, float, float]:
    """Compare scalar values with NaN, absolute, and relative tolerance handling."""
    if math.isnan(zarr_value) and math.isnan(expected_value):
        return True, 0.0, atol
    if math.isnan(zarr_value) or math.isnan(expected_value):
        return False, math.inf, atol
    diff = abs(zarr_value - expected_value)
    allowed_diff = atol + rtol * abs(expected_value)
    return diff <= allowed_diff, diff, allowed_diff


def nearest_finite_source_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    candidate_hours: list[int],
    y: int,
    x: int,
) -> tuple[int, float] | None:
    """Return the first finite, validity-masked source value from candidate hours."""
    for source_hour in candidate_hours:
        value = mask_source_value(
            raw_grib_value(run_dir, base_time, spec, source_hour, y, x)
        )
        if math.isfinite(value):
            return source_hour, value
    return None


def interpolation_sources(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
    """Find finite HRDPS forecast sources bracketing a target hour."""
    left = nearest_finite_source_value(
        run_dir,
        base_time,
        spec,
        [
            source_hour
            for source_hour in reversed(HRDPS_SOURCE_HOURS)
            if source_hour <= hour
        ],
        y,
        x,
    )
    right = nearest_finite_source_value(
        run_dir,
        base_time,
        spec,
        [source_hour for source_hour in HRDPS_SOURCE_HOURS if source_hour >= hour],
        y,
        x,
    )
    return left, right


def expected_output_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> tuple[float, int | None, float, int | None, float]:
    """Mirror HRDPS's NaN-aware interpolation for one forecast grid point."""
    left, right = interpolation_sources(run_dir, base_time, spec, hour, y, x)
    if left is None or right is None:
        return math.nan, None, math.nan, None, math.nan

    left_hour, left_value = left
    right_hour, right_value = right
    if spec.nearest:
        if hour - left_hour <= right_hour - hour:
            expected = left_value
        else:
            expected = right_value
    elif left_hour == right_hour:
        expected = left_value
    else:
        weight = (hour - left_hour) / (right_hour - left_hour)
        expected = (1 - weight) * left_value + weight * right_value
    return expected, left_hour, left_value, right_hour, right_value


def mixed_ptype_summary(values: np.ndarray) -> tuple[int, str]:
    """Summarize finite PTYPE values for one candidate point."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, "none"
    unique_values = np.unique(finite)
    return len(unique_values), ",".join(f"{value:g}" for value in unique_values)


def find_mixed_ptype_points(
    run_dir: Path,
    base_time: pd.Timestamp,
    y_size: int,
    x_size: int,
    search_hours: list[int],
    count: int,
    candidates: int,
    seed: int,
) -> list[SelectedPoint]:
    """Find grid points where raw HRDPS PTYPE changes over time."""
    if count <= 0:
        return []
    if candidates <= 0:
        raise ValueError("--ptype-search-candidates must be positive.")

    source_hours = sorted(set(search_hours) & set(HRDPS_SOURCE_HOURS))
    if len(source_hours) < 2:
        raise ValueError("At least two valid PTYPE search hours are required.")

    rng = random.Random(seed)
    candidate_count = min(candidates, y_size * x_size)
    point_set: set[tuple[int, int]] = set()
    point_list = []
    while len(point_list) < candidate_count:
        point = (rng.randrange(y_size), rng.randrange(x_size))
        if point in point_set:
            continue
        point_set.add(point)
        point_list.append(point)

    ys = np.asarray([point[0] for point in point_list], dtype=np.intp)
    xs = np.asarray([point[1] for point in point_list], dtype=np.intp)
    spec = VARIABLE_SPECS["PTYPE_surface"]
    values_by_hour = []
    for hour in source_hours:
        path = grib_file_path(run_dir, base_time, spec, hour)
        values_by_hour.append(load_grib_array(path)[ys, xs])

    point_values = np.stack(values_by_hour, axis=0)
    selected = []
    for point_index, (y, x) in enumerate(point_list):
        unique_count, unique_text = mixed_ptype_summary(point_values[:, point_index])
        if unique_count > 1:
            selected.append(
                SelectedPoint(
                    y,
                    x,
                    f"auto mixed raw PTYPE over hours={source_hours} values={unique_text}",
                )
            )
            if len(selected) == count:
                break
    return selected


def points_from_args(
    args: argparse.Namespace,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    run_dir: Path,
    base_time: pd.Timestamp,
    search_hours: list[int],
) -> list[SelectedPoint]:
    """Resolve explicit coordinates or automatically select changing-PTYPE points."""
    y_size, x_size = latitudes.shape
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("--lat and --lon must be provided together.")
        y, x, grid_lat, grid_lon = nearest_point_from_lat_lon(
            latitudes, longitudes, args.lat, args.lon
        )
        return [
            SelectedPoint(
                y,
                x,
                f"requested lat/lon=({args.lat}, {args.lon}) "
                f"nearest_grid=({grid_lat}, {grid_lon})",
            )
        ]

    if args.y is not None or args.x is not None:
        if args.y is None or args.x is None:
            raise ValueError("--y and --x must be provided together.")
        y, x = int(args.y), int(args.x)
        if y < 0 or y >= y_size or x < 0 or x >= x_size:
            raise ValueError(
                f"Point y={y} x={x} is outside raw grid shape {(y_size, x_size)}."
            )
        return [SelectedPoint(y, x, "requested grid indices")]

    selected = find_mixed_ptype_points(
        run_dir,
        base_time,
        y_size,
        x_size,
        search_hours,
        args.find_ptype_points,
        args.ptype_search_candidates,
        args.seed,
    )
    if selected:
        return selected

    y, x = (
        random.Random(args.seed).randrange(y_size),
        random.Random(args.seed + 1).randrange(x_size),
    )
    return [SelectedPoint(y, x, "fallback deterministic random grid point")]


def sample_points(
    y_size: int, x_size: int, count: int, seed: int
) -> list[tuple[int, int]]:
    """Pick deterministic additional raw-grid points."""
    rng = random.Random(seed)
    return [(rng.randrange(y_size), rng.randrange(x_size)) for _ in range(count)]


def check_time_axis(
    root: zarr.Array,
    base_time: pd.Timestamp,
    forecast_offset: int,
    tolerance_seconds: float,
) -> bool:
    """Check the final time variable against the expected hourly cadence."""
    expected = expected_unix_seconds(base_time, forecast_offset, root.shape[1])
    actual = np.asarray(root[ZARR_VARS.index("time"), :, 0, 0], dtype=np.float64)
    max_diff = float(np.nanmax(np.abs(actual - expected)))
    passed = max_diff <= tolerance_seconds
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
    hours: list[int],
    points: list[tuple[int, int]],
    atol: float,
    rtol: float,
) -> list[CheckResult]:
    """Compare final HRDPS values with transformed and interpolated GRIB values."""
    results = []
    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        var_index = ZARR_VARS.index(variable)
        for hour in hours:
            if hour not in HRDPS_SOURCE_HOURS:
                continue
            time_index = forecast_offset + hour
            for y, x in points:
                raw_value = raw_grib_value(run_dir, base_time, spec, hour, y, x)
                expected, left_hour, _, right_hour, _ = expected_output_value(
                    run_dir, base_time, spec, hour, y, x
                )
                expected = stored_expected(expected)
                zarr_value = float(root[var_index, time_index, y, x])
                passed, diff, allowed_diff = compare_values(
                    zarr_value, expected, atol, rtol
                )
                note = ""
                if not math.isfinite(mask_source_value(raw_value)):
                    note = (
                        f"masked_source interpolation_bracket={left_hour}-{right_hour}"
                    )
                results.append(
                    CheckResult(
                        variable,
                        hour,
                        y,
                        x,
                        zarr_value,
                        expected,
                        diff,
                        allowed_diff,
                        passed,
                        note,
                    )
                )
    return results


def print_results(results: list[CheckResult], show_passes: bool) -> int:
    """Print sampled comparison results and return the failure count."""
    failures = 0
    for result in results:
        if not result.passed:
            failures += 1
        if result.passed and not show_passes:
            continue
        print(
            f"{'PASS' if result.passed else 'FAIL'} source {result.variable:22s} "
            f"h={result.hour:02d} y={result.y:04d} x={result.x:04d} "
            f"zarr={result.zarr_value:.8g} expected={result.expected_value:.8g} "
            f"abs_diff={result.abs_diff:.8g} allowed={result.allowed_diff:.8g} "
            f"{result.note}"
        )
    return failures


def print_ptype_table(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    y: int,
    x: int,
    hours: list[int],
    atol: float,
    rtol: float,
) -> tuple[int, int]:
    """Print raw and production PTYPE values at one HRDPS grid point."""
    spec = VARIABLE_SPECS["PTYPE_surface"]
    var_index = ZARR_VARS.index("PTYPE_surface")
    failures = 0
    rows = 0

    print()
    print(f"PTYPE_surface point table y={y} x={x}")
    print(
        "hour valid_time           target_grib_ptype interp_source_hour "
        "interp_source_ptype zarr_ptype expected_ptype abs_diff status"
    )
    for hour in hours:
        if hour not in HRDPS_SOURCE_HOURS:
            continue
        target_value = raw_grib_value(run_dir, base_time, spec, hour, y, x)
        expected, left_hour, left_value, right_hour, right_value = (
            expected_output_value(run_dir, base_time, spec, hour, y, x)
        )
        if left_hour is None or right_hour is None:
            interp_hour = None
            interp_value = math.nan
        elif hour - left_hour <= right_hour - hour:
            interp_hour, interp_value = left_hour, left_value
        else:
            interp_hour, interp_value = right_hour, right_value

        expected = stored_expected(expected)
        zarr_value = float(root[var_index, forecast_offset + hour, y, x])
        passed, diff, _ = compare_values(zarr_value, expected, atol, rtol)
        failures += 0 if passed else 1
        rows += 1
        valid_time = base_time + pd.Timedelta(hours=hour)
        interp_hour_text = str(interp_hour) if interp_hour is not None else "NA"
        print(
            f"{hour:>4d} {valid_time:%Y-%m-%d %H:%M} {target_value:>18.8g} "
            f"{interp_hour_text:>18s} {interp_value:>19.8g} {zarr_value:>10.8g} "
            f"{expected:>14.8g} {diff:>8.3g} {'PASS' if passed else 'FAIL'}"
        )
    return failures, rows


def print_continuous_table(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    variable: str,
    y: int,
    x: int,
    hours: list[int],
    atol: float,
    rtol: float,
) -> tuple[int, int]:
    """Print raw, masked/interpolated, and production values at one point."""
    spec = VARIABLE_SPECS[variable]
    var_index = ZARR_VARS.index(variable)
    failures = 0
    rows = 0

    print()
    print(f"{variable} point table y={y} x={x}")
    print(
        "hour valid_time           target_grib_value ingest_source_value "
        "left_hour left_value right_hour right_value zarr_value expected_value abs_diff status"
    )
    for hour in hours:
        if hour not in HRDPS_SOURCE_HOURS:
            continue
        target_value = raw_grib_value(run_dir, base_time, spec, hour, y, x)
        ingest_source = mask_source_value(target_value)
        expected, left_hour, left_value, right_hour, right_value = (
            expected_output_value(run_dir, base_time, spec, hour, y, x)
        )
        expected = stored_expected(expected)
        zarr_value = float(root[var_index, forecast_offset + hour, y, x])
        passed, diff, _ = compare_values(zarr_value, expected, atol, rtol)
        failures += 0 if passed else 1
        rows += 1
        valid_time = base_time + pd.Timedelta(hours=hour)
        left_text = str(left_hour) if left_hour is not None else "NA"
        right_text = str(right_hour) if right_hour is not None else "NA"
        print(
            f"{hour:>4d} {valid_time:%Y-%m-%d %H:%M} {target_value:>17.8g} "
            f"{ingest_source:>19.8g} {left_text:>9s} {left_value:>10.8g} "
            f"{right_text:>10s} {right_value:>11.8g} {zarr_value:>10.8g} "
            f"{expected:>14.8g} {diff:>8.3g} {'PASS' if passed else 'FAIL'}"
        )
    return failures, rows


def main() -> None:
    """Run HRDPS production spot checks."""
    args = parse_args()
    variables = parse_csv_variables(args.variables)
    detail_vars = parse_csv_variables(args.detail_vars)
    hours = parse_csv_ints(args.hours)
    detail_hours = parse_csv_ints(args.detail_hours)
    search_hours = parse_csv_ints(args.ptype_search_hours)

    base_time = load_base_time(args.base_time_pickle)
    run_dir = resolve_grib_run_dir(args.grib_root, base_time)
    root = zarr.open(str(args.zarr_path), mode="r")
    validate_root_array(root)
    forecast_offset = infer_forecast_offset(root.shape[1])

    first_spec = VARIABLE_SPECS[variables[0]]
    first_path = grib_file_path(run_dir, base_time, first_spec, HRDPS_SOURCE_HOURS[0])
    raw_shape, latitudes, longitudes = grib_grid(first_path)
    if root.shape[2] < raw_shape[0] or root.shape[3] < raw_shape[1]:
        raise ValueError(
            f"Final HRDPS spatial shape {root.shape[2:]} is smaller than raw grid {raw_shape}."
        )

    selected_points = points_from_args(
        args, latitudes, longitudes, run_dir, base_time, search_hours
    )
    random_points = sample_points(raw_shape[0], raw_shape[1], args.points, args.seed)
    selected_tuples = [(point.y, point.x) for point in selected_points]
    points = [
        *selected_tuples,
        *[point for point in random_points if point not in selected_tuples],
    ]

    print(f"Final Zarr: {args.zarr_path} shape={root.shape} chunks={root.chunks}")
    print(f"GRIB run dir: {run_dir}")
    print(f"Base time: {base_time}")
    print(f"Forecast hour offset: {forecast_offset}")
    print(f"Raw grid shape: {raw_shape}; final padding: {root.shape[2:]}")
    for point in selected_points:
        print(f"Selected point: y={point.y} x={point.x} ({point.note})")
    print(f"Check points: {points}")
    print(
        "APCP target values are de-accumulated; DSWRF target values are "
        "de-accumulated and converted from J/m^2 to W/m^2."
    )
    print(
        f"Values outside [{VALID_DATA_MIN}, {VALID_DATA_MAX}] become NaN before interpolation."
    )
    print()

    time_passed = check_time_axis(
        root, base_time, forecast_offset, args.time_tolerance_seconds
    )
    results = run_source_checks(
        root,
        run_dir,
        base_time,
        forecast_offset,
        variables,
        hours,
        points,
        args.atol,
        args.rtol,
    )

    print()
    failures = 0 if time_passed else 1
    failures += print_results(results, args.show_passes)
    detail_rows = 0
    for point in selected_points:
        for variable in detail_vars:
            if variable == "PTYPE_surface":
                table_failures, table_rows = print_ptype_table(
                    root,
                    run_dir,
                    base_time,
                    forecast_offset,
                    point.y,
                    point.x,
                    detail_hours,
                    args.atol,
                    args.rtol,
                )
            else:
                table_failures, table_rows = print_continuous_table(
                    root,
                    run_dir,
                    base_time,
                    forecast_offset,
                    variable,
                    point.y,
                    point.x,
                    detail_hours,
                    args.atol,
                    args.rtol,
                )
            failures += table_failures
            detail_rows += table_rows

    total = len(results) + detail_rows + 1
    print()
    print(
        f"Summary: passed={total - failures} failed={failures} total={total} "
        f"atol={args.atol} rtol={args.rtol}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
