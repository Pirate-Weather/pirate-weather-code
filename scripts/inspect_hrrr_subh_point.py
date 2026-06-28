#!/usr/bin/env python3
"""Print all HRRR SubH times for one variable at an x/y grid point.

When REFD_1000maboveground is selected, also print its converted
precipitation rate from API.utils.precip.dbz_to_rate.

Example:
    python scripts/inspect_hrrr_subh_point.py \
        --variable REFD_1000maboveground --x 907 --y 545
"""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.constants.api_const import (  # noqa: E402
    PRECIP_TYPES,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
)
from API.constants.shared_const import KELVIN_TO_CELSIUS  # noqa: E402
from API.utils.precip import dbz_to_rate  # noqa: E402

DEFAULT_DOWNLOAD_DIR = "/mnt/nvme/data/Prod/SubH/v30"
DEFAULT_X_INDEX = 907
DEFAULT_Y_INDEX = 545
SUBH_INTERVAL_SECONDS = 15 * 60
REFLECTIVITY_VARIABLE = "REFD_1000maboveground"

VARIABLES = (
    "time",
    "GUST_surface",
    "PRES_surface",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    REFLECTIVITY_VARIABLE,
    "APCP_surface",
    "VIS_surface",
    "SPFH_2maboveground",
    "DSWRF_surface",
)


def default_download_dir() -> Path:
    return Path(os.getenv("forecast_path", DEFAULT_DOWNLOAD_DIR)).expanduser()


def find_zarr_store(download_dir: Path) -> Path:
    if download_dir.name.endswith(".zarr"):
        return download_dir
    return download_dir / "SubH.zarr"


def format_value(value: np.float32 | np.float64) -> str:
    if np.isnan(value):
        return "nan"
    return f"{float(value):.8g}"


def format_timestamp(value: np.float32) -> str:
    if np.isnan(value):
        return "nan"
    rounded_value = round(float(value) / SUBH_INTERVAL_SECONDS) * SUBH_INTERVAL_SECONDS
    return datetime.fromtimestamp(rounded_value, tz=UTC).isoformat()


def read_point_variable(array, variable: str, x_index: int, y_index: int):
    variable_index = VARIABLES.index(variable)
    return np.asarray(array[variable_index, :, y_index, x_index])


def derive_precip_types(array, x_index: int, y_index: int, dbz_values: np.ndarray):
    flag_variables = (
        "CSNOW_surface",
        "CICEP_surface",
        "CFRZR_surface",
        "CRAIN_surface",
    )
    type_labels = np.array(
        [
            PRECIP_TYPES["none"],
            PRECIP_TYPES["snow"],
            PRECIP_TYPES["ice"],
            PRECIP_TYPES["sleet"],
            PRECIP_TYPES["rain"],
        ]
    )
    flags = np.column_stack(
        [
            np.zeros_like(dbz_values),
            *(
                read_point_variable(array, variable, x_index, y_index)
                for variable in flag_variables
            ),
        ]
    )
    precip_types = type_labels[np.argmax(flags, axis=1)]

    missing_type = (precip_types == PRECIP_TYPES["none"]) & (dbz_values > 0)
    temperatures_c = (
        read_point_variable(array, "TMP_2maboveground", x_index, y_index)
        - KELVIN_TO_CELSIUS
    )
    precip_types[missing_type] = np.where(
        temperatures_c[missing_type] >= TEMP_THRESHOLD_RAIN_C,
        PRECIP_TYPES["rain"],
        np.where(
            temperatures_c[missing_type] <= TEMP_THRESHOLD_SNOW_C,
            PRECIP_TYPES["snow"],
            PRECIP_TYPES["sleet"],
        ),
    )
    return precip_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print values from a downloaded HRRR SubH Zarr store at x/y."
    )
    parser.add_argument(
        "download_dir",
        nargs="?",
        type=Path,
        default=default_download_dir(),
        help="directory containing SubH.zarr, or the SubH.zarr path itself",
    )
    parser.add_argument(
        "--variable",
        required=True,
        choices=VARIABLES[1:],
        help="weather variable to print",
    )
    parser.add_argument("--x", type=int, default=DEFAULT_X_INDEX, help="x grid index")
    parser.add_argument("--y", type=int, default=DEFAULT_Y_INDEX, help="y grid index")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store_path = find_zarr_store(args.download_dir.expanduser())
    if not store_path.is_dir():
        raise SystemExit(f"SubH Zarr store does not exist: {store_path}")

    array = zarr.open(zarr.storage.LocalStore(store_path), mode="r")
    if array.ndim != 4:
        raise SystemExit(f"Expected a 4D SubH array, found shape {array.shape}")
    if array.shape[0] != len(VARIABLES):
        raise SystemExit(f"Expected {len(VARIABLES)} variables, found {array.shape[0]}")

    _, _, y_size, x_size = array.shape
    if not 0 <= args.x < x_size or not 0 <= args.y < y_size:
        raise SystemExit(
            f"Point x={args.x}, y={args.y} is outside "
            f"x=0..{x_size - 1}, y=0..{y_size - 1}"
        )

    times = read_point_variable(array, "time", args.x, args.y)
    values = read_point_variable(array, args.variable, args.x, args.y)

    rates = None
    precip_types = None
    if args.variable == REFLECTIVITY_VARIABLE:
        precip_types = derive_precip_types(array, args.x, args.y, values)
        rates = dbz_to_rate(values.astype(float), precip_types)

    print(f"Store: {store_path}")
    print(f"Shape: {array.shape}")
    print(f"Point: x={args.x}, y={args.y}")
    print(f"Variable: {args.variable}")

    for time_index, (timestamp, value) in enumerate(zip(times, values, strict=True)):
        prefix = f"time[{time_index}] {format_timestamp(timestamp)}: "
        if rates is None or precip_types is None:
            print(f"{prefix}{format_value(value)}")
        else:
            print(
                f"{prefix}{format_value(value)} dBZ -> "
                f"{format_value(rates[time_index])} mm/h "
                f"({precip_types[time_index]})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
