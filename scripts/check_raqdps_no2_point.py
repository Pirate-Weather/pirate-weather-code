#!/usr/bin/env python3
"""Check RAQDPS NO2 values at one lat/lon and two target times."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ECCODES_DIR = REPO_ROOT / ".build" / "ingest-test" / "toolchain"
os.environ.setdefault("ECCODES_DIR", str(DEFAULT_ECCODES_DIR))
os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    f"{DEFAULT_ECCODES_DIR / 'lib'}:{DEFAULT_ECCODES_DIR / 'lib64'}",
)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import zarr  # noqa: E402

from API.raqdps_utils import RAQDPS_OUTPUT_VARS, build_raqdps_url  # noqa: E402

DEFAULT_ZARR = "/home/reya/Weather/Prod/v30/RAQDPS.zarr"
DEFAULT_LAT = 45.4766681
DEFAULT_LON = -73.5550823
GRID_RUN = datetime(2026, 7, 6, 12, tzinfo=timezone.utc)
TARGETS = (
    ("2026-07-06T12:00Z", datetime(2026, 7, 6, 12, tzinfo=timezone.utc)),
    (
        "2026-07-09 08:00 EDT",
        datetime(2026, 7, 9, 8, tzinfo=timezone(timedelta(hours=-4))),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", default=DEFAULT_ZARR, help="RAQDPS final Zarr path")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument(
        "--grid-grib",
        default="/tmp/raqdps_no2_grid_20260706T12Z_f000.grib2",
        help="Cached/downloaded RAQDPS GRIB used only for grid coordinates",
    )
    parser.add_argument(
        "--time-tolerance-seconds",
        type=int,
        default=300,
        help="Nearest-time tolerance for float32 epoch storage",
    )
    return parser.parse_args()


def ensure_grid_grib(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = build_raqdps_url(GRID_RUN, "NO2", 0)
    print(f"Downloading RAQDPS grid GRIB: {url}")
    urlretrieve(url, path)
    return path


def nearest_grid_point(
    grid_grib: Path, lat: float, lon: float
) -> tuple[int, int, float, float, float]:
    ds = xr.open_dataset(grid_grib, engine="cfgrib", decode_times=False)
    try:
        grid_lat = ds["latitude"].values
        grid_lon = ds["longitude"].values
    finally:
        ds.close()

    lat_rad = np.deg2rad(grid_lat)
    lon_rad = np.deg2rad(grid_lon)
    target_lat_rad = np.deg2rad(lat)
    target_lon_rad = np.deg2rad(lon)
    haversine = (
        np.sin((lat_rad - target_lat_rad) / 2) ** 2
        + np.cos(target_lat_rad)
        * np.cos(lat_rad)
        * np.sin((lon_rad - target_lon_rad) / 2) ** 2
    )
    y_idx, x_idx = np.unravel_index(int(np.nanargmin(haversine)), haversine.shape)
    distance_km = float(6371.0 * 2 * np.arcsin(np.sqrt(haversine[y_idx, x_idx])))
    return (
        y_idx,
        x_idx,
        float(grid_lat[y_idx, x_idx]),
        float(grid_lon[y_idx, x_idx]),
        distance_km,
    )


def nearest_time_index(
    times: np.ndarray, target: datetime, tolerance_seconds: int
) -> tuple[int, float]:
    target_utc = target.astimezone(timezone.utc)
    deltas = times.astype(np.float64) - target_utc.timestamp()
    time_idx = int(np.nanargmin(np.abs(deltas)))
    delta_seconds = float(deltas[time_idx])
    if abs(delta_seconds) > tolerance_seconds:
        raise ValueError(
            f"No stored time within {tolerance_seconds}s of {target_utc.isoformat()}; "
            f"nearest delta is {delta_seconds:.0f}s"
        )
    return time_idx, delta_seconds


def main() -> None:
    args = parse_args()
    root = zarr.open(args.zarr, mode="r")
    no2_index = RAQDPS_OUTPUT_VARS.index("NO2")
    times = root[0, :, 0, 0]
    y_idx, x_idx, grid_lat, grid_lon, distance_km = nearest_grid_point(
        ensure_grid_grib(Path(args.grid_grib)), args.lat, args.lon
    )

    print(f"Zarr: {args.zarr}")
    print(
        f"Requested point: {args.lat:.7f}, {args.lon:.7f}; "
        f"nearest grid y={y_idx}, x={x_idx} at "
        f"{grid_lat:.6f}, {grid_lon:.6f} ({distance_km:.2f} km)"
    )

    for label, target in TARGETS:
        time_idx, delta_seconds = nearest_time_index(
            times, target, args.time_tolerance_seconds
        )
        value = float(root[no2_index, time_idx, y_idx, x_idx])
        target_utc = target.astimezone(timezone.utc)
        print(
            f"NO2 {label} ({target_utc:%Y-%m-%dT%H:%MZ}): "
            f"time_idx={time_idx}, stored_delta={delta_seconds:.0f}s, "
            f"value={value:g} µg/m³"
        )


if __name__ == "__main__":
    main()
