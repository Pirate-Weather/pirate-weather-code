"""Spot-check a GDPS final Zarr against cached raw GRIB2 files.

The GDPS ingest writes a root 4D Zarr array with dimensions
(variable, time, latitude, longitude). This script compares selected final
values to cached Herbie GRIB2 files and focuses on the temporal interpolation
step used to create hourly output.
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

DEFAULT_PROCESS_DIR = Path("/mnt/nvme/data/Process/GDPS")
DEFAULT_ZARR_PATH = DEFAULT_PROCESS_DIR / "GDPS.zarr"
DEFAULT_GRIB_ROOT = DEFAULT_PROCESS_DIR / "Downloads" / "herbie" / "gdps"
DEFAULT_BASE_TIME_PICKLE = DEFAULT_PROCESS_DIR / "GDPS.time.pickle"
DEFAULT_COORD_ZARR_PATH = DEFAULT_PROCESS_DIR / "GDPS_Process_.zarr"

ZARR_VARS = (
    "time",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "WIND_10maboveground",
    "WDIR_10maboveground",
    "GUST_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "UVI_surface",
    "DSWRF_surface",
    "PRES_surface",
    "TCDC_surface",
    "PRMSL_meansealevel",
    "TCIOZ_surface",
    "PTYPE_surface",
    "CAPE_surface",
    "CIN_surface",
    "VVEL_500mb",
    "KX_surface",
)

GDPS_FILE_HOURS = [*range(1, 84), *range(84, 241, 3)]
GDPS_REDUCED_FILE_HOURS = [
    hour for hour in GDPS_FILE_HOURS if hour % 3 == 0 and (hour <= 168 or hour % 6 == 0)
]
GDPS_3_HOUR_FILE_HOURS = [hour for hour in GDPS_FILE_HOURS if hour % 3 == 0]


@dataclass(frozen=True)
class VariableSpec:
    """GDPS final variable to raw GRIB file mapping."""

    zarr_name: str
    grib_token: str
    source_hours: tuple[int, ...]
    nearest: bool = False


VARIABLE_SPECS = {
    "TMP_2maboveground": VariableSpec(
        "TMP_2maboveground", "AirTemp_AGL-2m", tuple(GDPS_FILE_HOURS)
    ),
    "DPT_2maboveground": VariableSpec(
        "DPT_2maboveground", "DewPoint_AGL-2m", tuple(GDPS_FILE_HOURS)
    ),
    "RH_2maboveground": VariableSpec(
        "RH_2maboveground", "RelativeHumidity_AGL-2m", tuple(GDPS_FILE_HOURS)
    ),
    "WIND_10maboveground": VariableSpec(
        "WIND_10maboveground", "WindSpeed_AGL-10m", tuple(GDPS_FILE_HOURS)
    ),
    "WDIR_10maboveground": VariableSpec(
        "WDIR_10maboveground", "WindDir_AGL-10m", tuple(GDPS_FILE_HOURS)
    ),
    "GUST_10maboveground": VariableSpec(
        "GUST_10maboveground", "WindGust_AGL-10m", tuple(GDPS_FILE_HOURS)
    ),
    "PRATE_surface": VariableSpec(
        "PRATE_surface", "PrecipRate_Sfc", tuple(GDPS_FILE_HOURS)
    ),
    "APCP_surface": VariableSpec(
        "APCP_surface", "Precip-Accum_Sfc", tuple(GDPS_FILE_HOURS)
    ),
    "UVI_surface": VariableSpec("UVI_surface", "UVIndex_Sfc", tuple(GDPS_FILE_HOURS)),
    "DSWRF_surface": VariableSpec(
        "DSWRF_surface",
        "DownwardShortwaveRadiationFlux-Accum_Sfc",
        tuple(GDPS_FILE_HOURS),
    ),
    "PRES_surface": VariableSpec(
        "PRES_surface", "Pressure_Sfc", tuple(GDPS_FILE_HOURS)
    ),
    "TCDC_surface": VariableSpec(
        "TCDC_surface", "TotalCloudCover_Sfc", tuple(GDPS_FILE_HOURS)
    ),
    "PRMSL_meansealevel": VariableSpec(
        "PRMSL_meansealevel", "Pressure_MSL", tuple(GDPS_FILE_HOURS)
    ),
    "TCIOZ_surface": VariableSpec("TCIOZ_surface", "O3_EAtm", tuple(GDPS_FILE_HOURS)),
    "PTYPE_surface": VariableSpec(
        "PTYPE_surface",
        "PrecipType-Instant_Sfc",
        tuple(GDPS_REDUCED_FILE_HOURS),
        nearest=True,
    ),
    "CAPE_surface": VariableSpec(
        "CAPE_surface", "CAPE_Sfc", tuple(GDPS_REDUCED_FILE_HOURS)
    ),
    "CIN_surface": VariableSpec(
        "CIN_surface", "CIN_Sfc", tuple(GDPS_REDUCED_FILE_HOURS)
    ),
    "VVEL_500mb": VariableSpec(
        "VVEL_500mb", "VerticalVelocity_IsbL-0500", tuple(GDPS_3_HOUR_FILE_HOURS)
    ),
    "KX_surface": VariableSpec(
        "KX_surface", "KIndex_Sfc", tuple(GDPS_REDUCED_FILE_HOURS)
    ),
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


@dataclass(frozen=True)
class SelectedPoint:
    """A point chosen for validation output."""

    y: int
    x: int
    note: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spot-check GDPS final Zarr values against raw GRIB2 files."
    )
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument("--grib-root", type=Path, default=DEFAULT_GRIB_ROOT)
    parser.add_argument(
        "--base-time-pickle", type=Path, default=DEFAULT_BASE_TIME_PICKLE
    )
    parser.add_argument(
        "--coord-zarr-path",
        type=Path,
        default=DEFAULT_COORD_ZARR_PATH,
        help="Optional GDPS process Zarr group containing latitude/longitude arrays.",
    )
    parser.add_argument(
        "--variables",
        default="PTYPE_surface,APCP_surface,UVI_surface",
        help="Comma-separated final GDPS variables to check.",
    )
    parser.add_argument(
        "--source-hours",
        default="1,3,84,168,174,240",
        help="Comma-separated forecast hours expected to align exactly with raw GRIBs.",
    )
    parser.add_argument(
        "--interp-hours",
        default="2,85,86,169,170,171,172,173",
        help="Comma-separated forecast hours expected to be produced by interpolation.",
    )
    parser.add_argument(
        "--ptype-hours",
        default="1,2,3,4,5,84,85,86,87,168,169,170,171,172,173,174,228,229,230,231,232,233,234,235,236,237,238,239,240",
        help="Comma-separated forecast hours to print in the detailed PTYPE table.",
    )
    parser.add_argument(
        "--detail-vars",
        default="PTYPE_surface,APCP_surface,UVI_surface",
        help="Comma-separated variables to print as point detail tables.",
    )
    parser.add_argument(
        "--find-ptype-points",
        type=int,
        default=1,
        help="Automatically select this many grid points where raw PTYPE changes over time.",
    )
    parser.add_argument(
        "--ptype-search-hours",
        default="3,6,9,12,24,36,48,60,72,84,87,96,120,144,168,174,180,204,228,240",
        help="Comma-separated raw PTYPE source forecast hours used to find mixed-ptype locations.",
    )
    parser.add_argument(
        "--ptype-search-candidates",
        type=int,
        default=20000,
        help="Number of random grid cells to sample when finding mixed-ptype locations.",
    )
    parser.add_argument(
        "--y",
        type=int,
        help="Latitude/y grid index for point-specific output.",
    )
    parser.add_argument(
        "--x",
        type=int,
        help="Longitude/x grid index for point-specific output.",
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
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=180,
        help="Allowed final time-coordinate drift. Float32 final storage can quantize epoch seconds.",
    )
    parser.add_argument("--show-passes", action="store_true")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_variables(raw: str) -> list[str]:
    """Parse and validate comma-separated variable names."""
    variables = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(variables) - set(VARIABLE_SPECS))
    if unknown:
        raise ValueError(f"Unknown GDPS variables: {unknown}")
    return variables


def normalize_longitude(lon: float, longitudes: np.ndarray) -> float:
    """Normalize longitude to the coordinate convention used by the source grid."""
    lon_min = float(np.nanmin(longitudes))
    lon_max = float(np.nanmax(longitudes))
    if lon_min >= 0 and lon < 0:
        return lon % 360
    if lon_max <= 180 and lon > 180:
        return ((lon + 180) % 360) - 180
    return lon


def nearest_point_from_lat_lon(
    coord_zarr_path: Path, lat: float, lon: float
) -> tuple[int, int, float, float]:
    """Map latitude/longitude to nearest GDPS grid indices."""
    if not coord_zarr_path.exists():
        raise FileNotFoundError(
            f"Coordinate Zarr path does not exist: {coord_zarr_path}"
        )

    coord_root = zarr.open(str(coord_zarr_path), mode="r")
    if "latitude" not in coord_root or "longitude" not in coord_root:
        raise ValueError(
            f"Coordinate Zarr missing latitude/longitude arrays: {coord_zarr_path}"
        )

    latitudes = np.asarray(coord_root["latitude"][:], dtype=np.float64)
    longitudes = np.asarray(coord_root["longitude"][:], dtype=np.float64)
    target_lon = normalize_longitude(lon, longitudes)
    y = int(np.nanargmin(np.abs(latitudes - lat)))
    x = int(np.nanargmin(np.abs(longitudes - target_lon)))
    return y, x, float(latitudes[y]), float(longitudes[x])


def load_base_time(path: Path) -> pd.Timestamp:
    """Load the GDPS base run time from the ingest pickle."""
    with path.open("rb") as file:
        return pd.Timestamp(pickle.load(file))


def resolve_grib_run_dir(grib_root: Path, base_time: pd.Timestamp) -> Path:
    """Resolve either a parent GRIB cache path or a single run directory."""
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
    """Validate the final Zarr layout expected by GDPS ingest."""
    if not isinstance(root, zarr.Array):
        raise TypeError("Expected final GDPS Zarr to be a root zarr.Array.")
    if root.ndim != 4:
        raise ValueError(
            "Expected final GDPS Zarr shape (variable, time, latitude, longitude)."
        )
    if root.shape[0] != len(ZARR_VARS):
        raise ValueError(f"Expected {len(ZARR_VARS)} variables, found {root.shape[0]}.")


def infer_forecast_offset(final_time_len: int, source_hours: list[int]) -> int:
    """Infer the final-array index offset where forecast hour zero would sit."""
    if not source_hours:
        raise ValueError("Cannot infer offset without source hours.")
    max_hour = max(source_hours)
    offset = final_time_len - max_hour - 1
    if offset < 0:
        raise ValueError(
            f"Final time length {final_time_len} is too short for max forecast hour {max_hour}."
        )
    return offset


def expected_unix_seconds(
    base_time: pd.Timestamp, forecast_offset: int, time_len: int
) -> np.ndarray:
    """Build the expected hourly final time axis from the run time and offset."""
    hours = np.arange(-forecast_offset, time_len - forecast_offset, dtype=np.int64)
    base_seconds = int(base_time.timestamp())
    return base_seconds + hours * 3600


def grib_file_path(
    run_dir: Path, base_time: pd.Timestamp, spec: VariableSpec, hour: int
) -> Path:
    """Find a raw GRIB2 file for a variable and forecast hour."""
    run_token = base_time.strftime("%Y%m%dT%HZ")
    filename = f"{run_token}_MSC_GDPS_{spec.grib_token}_LatLon0.15_PT{hour:03d}H.grib2"
    direct = run_dir / filename
    if direct.exists():
        return direct

    matches = sorted(
        run_dir.glob(f"*MSC_GDPS_{spec.grib_token}_LatLon0.15_PT{hour:03d}H.grib2")
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing GRIB for {spec.zarr_name} hour {hour}: {direct}"
        )
    return matches[0]


@lru_cache(maxsize=64)
def load_grib_values(path: str) -> np.ndarray:
    """Load a single-field GRIB file as a numpy array."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        data_vars = list(ds.data_vars)
        if len(data_vars) != 1:
            raise ValueError(f"Expected one data variable in {path}, found {data_vars}")
        return np.asarray(ds[data_vars[0]].values)
    finally:
        ds.close()


def raw_grib_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Read the raw GRIB value transformed into the final Zarr source units."""
    path = grib_file_path(run_dir, base_time, spec, hour)
    current = float(load_grib_values(str(path))[y, x])
    if spec.zarr_name != "APCP_surface":
        return current

    previous_hours = [
        source_hour for source_hour in spec.source_hours if source_hour < hour
    ]
    previous = 0.0
    previous_hour = 0
    if previous_hours:
        previous_hour = previous_hours[-1]
        previous_path = grib_file_path(run_dir, base_time, spec, previous_hour)
        previous = float(load_grib_values(str(previous_path))[y, x])

    time_step = hour - previous_hour
    return max((current - previous) / time_step, 0.0)


def stored_expected(value: float) -> float:
    """Match final ingest storage rounding."""
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


def bracketing_source_hours(
    source_hours: tuple[int, ...], hour: int
) -> tuple[int, int] | None:
    """Return source hours surrounding an interpolated target hour."""
    if hour in source_hours:
        return None
    left = [source_hour for source_hour in source_hours if source_hour < hour]
    right = [source_hour for source_hour in source_hours if source_hour > hour]
    if not left or not right:
        return None
    return left[-1], right[0]


def nearest_source_hour_for_ptype(hour: int) -> int | None:
    """Return the precipitation-type source hour used by nearest-neighbor interpolation."""
    source_hours = VARIABLE_SPECS["PTYPE_surface"].source_hours
    if hour < source_hours[0] or hour > source_hours[-1]:
        return None
    return min(
        source_hours,
        key=lambda source_hour: (abs(source_hour - hour), source_hour),
    )


def target_source_hour_for_ptype(hour: int) -> int | None:
    """Return hour only when a raw PTYPE GRIB exists at the exact target hour."""
    source_hours = VARIABLE_SPECS["PTYPE_surface"].source_hours
    return hour if hour in source_hours else None


def target_source_hour(spec: VariableSpec, hour: int) -> int | None:
    """Return hour only when a raw GRIB exists at the exact target hour."""
    return hour if hour in spec.source_hours else None


def expected_nan_aware_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Mirror interp_time_map_blocks_nan for one variable and grid point."""
    if spec.nearest:
        source_by_distance = sorted(
            spec.source_hours,
            key=lambda source_hour: (abs(source_hour - hour), source_hour),
        )
        for source_hour in source_by_distance:
            value = raw_grib_value(run_dir, base_time, spec, source_hour, y, x)
            if is_finite(value):
                return value
        return math.nan

    left = nearest_finite_source_value(
        run_dir,
        base_time,
        spec,
        [
            source_hour
            for source_hour in reversed(spec.source_hours)
            if source_hour <= hour
        ],
        y,
        x,
    )
    right = nearest_finite_source_value(
        run_dir,
        base_time,
        spec,
        [source_hour for source_hour in spec.source_hours if source_hour >= hour],
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


def nearest_finite_source_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    candidate_hours: list[int],
    y: int,
    x: int,
) -> tuple[int, float] | None:
    """Return the first finite source value from ordered candidate hours."""
    for source_hour in candidate_hours:
        value = raw_grib_value(run_dir, base_time, spec, source_hour, y, x)
        if is_finite(value):
            return source_hour, value
    return None


def expected_interpolated_value(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
    y: int,
    x: int,
) -> float:
    """Compute the expected final value for a non-source forecast hour."""
    bracket = bracketing_source_hours(spec.source_hours, hour)
    if bracket is None:
        raise ValueError(
            f"Hour {hour} is not an interpolated hour for {spec.zarr_name}."
        )

    left_hour, right_hour = bracket
    left_value = raw_grib_value(run_dir, base_time, spec, left_hour, y, x)
    right_value = raw_grib_value(run_dir, base_time, spec, right_hour, y, x)

    if not is_finite(left_value) or not is_finite(right_value):
        return expected_nan_aware_value(run_dir, base_time, spec, hour, y, x)

    if spec.nearest:
        left_distance = abs(hour - left_hour)
        right_distance = abs(right_hour - hour)
        return left_value if left_distance <= right_distance else right_value

    weight = (hour - left_hour) / (right_hour - left_hour)
    return (1 - weight) * left_value + weight * right_value


def sample_points(
    y_size: int, x_size: int, count: int, seed: int
) -> list[tuple[int, int]]:
    """Pick deterministic random grid points."""
    rng = random.Random(seed)
    return [(rng.randrange(y_size), rng.randrange(x_size)) for _ in range(count)]


def mixed_ptype_summary(values: np.ndarray) -> tuple[int, str]:
    """Summarize finite ptype values for one candidate point."""
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
    """Find grid points where raw PTYPE has multiple finite values over time."""
    if count <= 0:
        return []
    if candidates <= 0:
        raise ValueError("--ptype-search-candidates must be positive.")

    spec = VARIABLE_SPECS["PTYPE_surface"]
    source_hours = sorted(set(search_hours) & set(spec.source_hours))
    if len(source_hours) < 2:
        raise ValueError(
            "At least two valid PTYPE source hours are needed for automatic point search."
        )

    rng = random.Random(seed)
    candidate_count = min(candidates, y_size * x_size)
    candidate_points_seen = set()
    point_list = []
    while len(point_list) < candidate_count:
        point = (rng.randrange(y_size), rng.randrange(x_size))
        if point in candidate_points_seen:
            continue
        candidate_points_seen.add(point)
        point_list.append(point)

    ys = np.asarray([point[0] for point in point_list], dtype=np.intp)
    xs = np.asarray([point[1] for point in point_list], dtype=np.intp)
    values_by_hour = []
    for hour in source_hours:
        values_by_hour.append(raw_grib_array(run_dir, base_time, spec, hour)[ys, xs])

    point_values = np.stack(values_by_hour, axis=0)
    mixed_points = []
    for point_idx, (y, x) in enumerate(point_list):
        unique_count, unique_text = mixed_ptype_summary(point_values[:, point_idx])
        if unique_count > 1:
            mixed_points.append(
                SelectedPoint(
                    y=y,
                    x=x,
                    note=f"auto mixed raw PTYPE over hours={source_hours} values={unique_text}",
                )
            )

    return mixed_points[:count]


def raw_grib_array(
    run_dir: Path,
    base_time: pd.Timestamp,
    spec: VariableSpec,
    hour: int,
) -> np.ndarray:
    """Load the raw GRIB array for a variable and forecast hour."""
    path = grib_file_path(run_dir, base_time, spec, hour)
    return load_grib_values(str(path))


def points_from_args(
    args: argparse.Namespace,
    run_dir: Path,
    base_time: pd.Timestamp,
    y_size: int,
    x_size: int,
    ptype_search_hours: list[int],
) -> list[SelectedPoint]:
    """Resolve requested points from explicit input or automatic ptype search."""
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("--lat and --lon must be provided together.")
        y, x, grid_lat, grid_lon = nearest_point_from_lat_lon(
            args.coord_zarr_path, args.lat, args.lon
        )
        return [
            SelectedPoint(
                y=y,
                x=x,
                note=f"requested lat/lon=({args.lat}, {args.lon}) nearest_grid=({grid_lat}, {grid_lon})",
            )
        ]

    if args.y is not None or args.x is not None:
        if args.y is None or args.x is None:
            raise ValueError("--y and --x must be provided together.")
        y = int(args.y)
        x = int(args.x)
        if y < 0 or y >= y_size or x < 0 or x >= x_size:
            raise ValueError(
                f"Point y={y} x={x} is outside raw grid shape {(y_size, x_size)}."
            )
        return [SelectedPoint(y=y, x=x, note="requested grid indices")]

    selected = find_mixed_ptype_points(
        run_dir=run_dir,
        base_time=base_time,
        y_size=y_size,
        x_size=x_size,
        search_hours=ptype_search_hours,
        count=args.find_ptype_points,
        candidates=args.ptype_search_candidates,
        seed=args.seed,
    )
    if selected:
        return selected

    return [
        SelectedPoint(
            y=228, x=102, note="fallback grid indices; no mixed PTYPE point found"
        )
    ]


def print_ptype_point_table(
    root: zarr.Array,
    run_dir: Path,
    base_time: pd.Timestamp,
    forecast_offset: int,
    y: int,
    x: int,
    hours: list[int],
    tolerance: float,
) -> tuple[int, int]:
    """Print precipitation type GRIB and final Zarr values at one grid point."""
    spec = VARIABLE_SPECS["PTYPE_surface"]
    var_index = ZARR_VARS.index("PTYPE_surface")
    failures = 0
    rows = 0

    print()
    print(f"PTYPE point table y={y} x={x}")
    print(
        "hour valid_time           target_grib_ptype interp_grib_hour "
        "interp_grib_ptype zarr_ptype expected_ptype abs_diff status"
    )
    for hour in hours:
        time_index = forecast_offset + hour
        if time_index < 0 or time_index >= root.shape[1]:
            continue

        target_raw_hour = target_source_hour_for_ptype(hour)
        interp_raw_hour = nearest_source_hour_for_ptype(hour)
        zarr_value = float(root[var_index, time_index, y, x])
        target_grib_value = (
            raw_grib_value(run_dir, base_time, spec, target_raw_hour, y, x)
            if target_raw_hour is not None
            else math.nan
        )
        if interp_raw_hour is None:
            interp_grib_value = math.nan
            expected_value = math.nan
            interp_raw_hour_text = "NA"
            diff = math.nan
            status = "NO_GRIB"
        else:
            interp_grib_value = raw_grib_value(
                run_dir, base_time, spec, interp_raw_hour, y, x
            )
            expected_value = stored_expected(
                expected_nan_aware_value(run_dir, base_time, spec, hour, y, x)
            )
            interp_raw_hour_text = str(interp_raw_hour)
            passed, diff = compare_values(zarr_value, expected_value, tolerance)
            failures += 0 if passed else 1
            status = "PASS" if passed else "FAIL"

        rows += 1
        valid_time = base_time + pd.Timedelta(hours=hour)

        print(
            f"{hour:>4d} {valid_time:%Y-%m-%d %H:%M} "
            f"{target_grib_value:>17.8g} {interp_raw_hour_text:>16s} "
            f"{interp_grib_value:>17.8g} {zarr_value:>10.8g} "
            f"{expected_value:>14.8g} {diff:>8.3g} {status}"
        )

    return failures, rows


def print_continuous_point_table(
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
    """Print raw/interpolated GRIB and final Zarr values at one grid point."""
    spec = VARIABLE_SPECS[variable]
    var_index = ZARR_VARS.index(variable)
    failures = 0
    rows = 0

    print()
    print(f"{variable} point table y={y} x={x}")
    print(
        "hour valid_time           target_grib_value left_hour left_grib_value "
        "right_hour right_grib_value zarr_value expected_value abs_diff status"
    )
    for hour in hours:
        time_index = forecast_offset + hour
        if time_index < 0 or time_index >= root.shape[1]:
            continue

        exact_hour = target_source_hour(spec, hour)
        bracket = bracketing_source_hours(spec.source_hours, hour)
        zarr_value = float(root[var_index, time_index, y, x])
        target_grib_value = (
            raw_grib_value(run_dir, base_time, spec, exact_hour, y, x)
            if exact_hour is not None
            else math.nan
        )

        if exact_hour is not None:
            left_hour = exact_hour
            right_hour = exact_hour
            left_grib_value = target_grib_value
            right_grib_value = target_grib_value
            expected_value = stored_expected(target_grib_value)
        elif bracket is None:
            left_hour = None
            right_hour = None
            left_grib_value = math.nan
            right_grib_value = math.nan
            expected_value = math.nan
        else:
            left_hour, right_hour = bracket
            left_grib_value = raw_grib_value(run_dir, base_time, spec, left_hour, y, x)
            right_grib_value = raw_grib_value(
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
        left_hour_text = str(left_hour) if left_hour is not None else "NA"
        right_hour_text = str(right_hour) if right_hour is not None else "NA"

        print(
            f"{hour:>4d} {valid_time:%Y-%m-%d %H:%M} "
            f"{target_grib_value:>17.8g} {left_hour_text:>9s} "
            f"{left_grib_value:>15.8g} {right_hour_text:>10s} "
            f"{right_grib_value:>16.8g} {zarr_value:>10.8g} "
            f"{expected_value:>14.8g} {diff:>8.3g} {status}"
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
    diffs = np.abs(actual - expected)
    max_diff = float(np.nanmax(diffs))
    passed = bool(max_diff <= tolerance_seconds)
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} time-axis max_abs_diff_seconds={max_diff:.1f} tolerance={tolerance_seconds}"
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
    """Compare final Zarr values at raw source forecast hours."""
    results = []
    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        var_index = ZARR_VARS.index(variable)
        for hour in source_hours:
            if hour not in spec.source_hours:
                continue
            time_index = forecast_offset + hour
            if time_index < 0 or time_index >= root.shape[1]:
                continue
            for y, x in points:
                zarr_value = float(root[var_index, time_index, y, x])
                raw_value = raw_grib_value(run_dir, base_time, spec, hour, y, x)
                if is_finite(raw_value):
                    expected_value = stored_expected(raw_value)
                    note = ""
                else:
                    expected_value = stored_expected(
                        expected_nan_aware_value(run_dir, base_time, spec, hour, y, x)
                    )
                    note = "raw_source_nan_filled_by_interp"
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
                        note,
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
    """Compare final Zarr values at hours filled by temporal interpolation."""
    results = []
    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        var_index = ZARR_VARS.index(variable)
        for hour in interp_hours:
            bracket = bracketing_source_hours(spec.source_hours, hour)
            if bracket is None:
                continue
            time_index = forecast_offset + hour
            if time_index < 0 or time_index >= root.shape[1]:
                continue
            note = f"bracket={bracket[0]}-{bracket[1]} method={'nearest' if spec.nearest else 'linear'}"
            for y, x in points:
                zarr_value = float(root[var_index, time_index, y, x])
                expected_value = stored_expected(
                    expected_interpolated_value(run_dir, base_time, spec, hour, y, x)
                )
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
                        note,
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

        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.kind:6s} {result.variable:22s} h={result.hour:03d} "
            f"y={result.y:04d} x={result.x:04d} zarr={result.zarr_value:.8g} "
            f"expected={result.expected_value:.8g} abs_diff={result.abs_diff:.8g} {result.note}"
        )
    return failures


def main() -> None:
    """Run GDPS spot checks."""
    args = parse_args()
    variables = parse_csv_variables(args.variables)
    detail_vars = parse_csv_variables(args.detail_vars)
    source_hours = parse_csv_ints(args.source_hours)
    interp_hours = parse_csv_ints(args.interp_hours)
    ptype_search_hours = parse_csv_ints(args.ptype_search_hours)
    ptype_hours = sorted(
        set(parse_csv_ints(args.ptype_hours)) | set(ptype_search_hours)
    )

    base_time = load_base_time(args.base_time_pickle)
    run_dir = resolve_grib_run_dir(args.grib_root, base_time)
    root = zarr.open(str(args.zarr_path), mode="r")
    validate_root_array(root)

    forecast_offset = infer_forecast_offset(root.shape[1], GDPS_FILE_HOURS)
    first_spec = VARIABLE_SPECS[variables[0]]
    first_path = grib_file_path(
        run_dir, base_time, first_spec, first_spec.source_hours[0]
    )
    raw_shape = load_grib_values(str(first_path)).shape
    y_size = min(root.shape[2], raw_shape[0])
    x_size = min(root.shape[3], raw_shape[1])
    selected_points = points_from_args(
        args, run_dir, base_time, y_size, x_size, ptype_search_hours
    )
    random_points = sample_points(y_size, x_size, args.points, args.seed)
    selected_point_tuples = [(point.y, point.x) for point in selected_points]
    points = [
        *selected_point_tuples,
        *[point for point in random_points if point not in selected_point_tuples],
    ]

    print(f"Final Zarr: {args.zarr_path} shape={root.shape} chunks={root.chunks}")
    print(f"GRIB run dir: {run_dir}")
    print(f"Base time: {base_time}")
    print(f"Forecast hour offset: {forecast_offset}")
    for point in selected_points:
        print(f"Selected point: y={point.y} x={point.x} ({point.note})")
    print(f"Check points: {points}")
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
    for point in selected_points:
        for detail_var in detail_vars:
            if detail_var == "PTYPE_surface":
                table_failures, table_rows = print_ptype_point_table(
                    root,
                    run_dir,
                    base_time,
                    forecast_offset,
                    point.y,
                    point.x,
                    ptype_hours,
                    args.tolerance,
                )
            else:
                table_failures, table_rows = print_continuous_point_table(
                    root,
                    run_dir,
                    base_time,
                    forecast_offset,
                    detail_var,
                    point.y,
                    point.x,
                    ptype_hours,
                    args.tolerance,
                )
            failures += table_failures
            detail_rows += table_rows
    total = len(source_results) + len(interp_results) + detail_rows + 1
    passed = total - failures
    print()
    print(
        f"Summary: passed={passed} failed={failures} total={total} tolerance={args.tolerance}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
