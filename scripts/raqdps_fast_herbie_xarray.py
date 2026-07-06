#!/usr/bin/env python3
"""Open RAQDPS GRIB2 files with Herbie FastHerbie as xarray datasets.

This module is intended for interactive workbook/notebook testing. Example:

    from scripts.raqdps_fast_herbie_xarray import open_raqdps_dataset

    ds = open_raqdps_dataset(variables=["O3", "NO2"], forecast_hours=range(6))

Import the helpers directly, or run the file as a small smoke test.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ECCODES_DIR = REPO_ROOT / ".build" / "ingest-test" / "toolchain"
DEFAULT_SAVE_DIR = Path(os.getenv("RAQDPS_HERBIE_SAVE_DIR", "/tmp/raqdps-herbie"))
RAQDPS_LEVEL = "Sfc"
RAQDPS_VARIABLES = ("PM2.5", "PM10", "NO2", "O3", "SO2")
RAQDPS_FORECAST_HOURS = tuple(range(73))


def configure_eccodes(eccodes_dir: Path | str = DEFAULT_ECCODES_DIR) -> None:
    """Set ecCodes environment defaults before Herbie/cfgrib imports."""
    eccodes_path = Path(eccodes_dir)
    os.environ.setdefault("ECCODES_DIR", str(eccodes_path))
    os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")
    os.environ.setdefault(
        "LD_LIBRARY_PATH", f"{eccodes_path / 'lib'}:{eccodes_path / 'lib64'}"
    )


def register_raqdps_template() -> None:
    """Register the repo-local RAQDPS template with the installed Herbie package."""
    configure_eccodes()
    import herbie.models as herbie_model_templates

    from API.raqdps_herbie_template import raqdps

    herbie_model_templates.raqdps = raqdps


def latest_raqdps_run(
    count: int = 10, save_dir: Path | str = DEFAULT_SAVE_DIR
) -> datetime:
    """Return the newest recent 00/12 UTC RAQDPS run with f072 O3 available."""
    configure_eccodes()
    register_raqdps_template()

    from herbie import Herbie

    from API.raqdps_utils import candidate_raqdps_runs, herbie_naive_utc, normalize_utc

    for run_time in candidate_raqdps_runs(count=count):
        h = Herbie(
            herbie_naive_utc(run_time),
            model="raqdps",
            product="10km/grib2",
            fxx=72,
            variable="O3",
            level=RAQDPS_LEVEL,
            priority=["msc"],
            save_dir=save_dir,
            verbose=False,
        )
        if h.grib is not None:
            return normalize_utc(run_time)

    raise RuntimeError("No recent RAQDPS run with f072 O3 data was found")


def open_raqdps_variable(
    variable: str,
    *,
    run_time: datetime | None = None,
    forecast_hours: Iterable[int] = RAQDPS_FORECAST_HOURS,
    save_dir: Path | str = DEFAULT_SAVE_DIR,
    max_threads: int = 6,
    xarray_threads: int | None = 2,
    remove_grib: bool = False,
    convert_units: bool = True,
):
    """Open one RAQDPS variable as an xarray Dataset using FastHerbie."""
    configure_eccodes()
    register_raqdps_template()

    from herbie import FastHerbie

    from API.raqdps_utils import convert_to_ug_m3, herbie_naive_utc

    if run_time is None:
        run_time = latest_raqdps_run(save_dir=save_dir)

    fh = FastHerbie(
        [herbie_naive_utc(run_time)],
        fxx=list(forecast_hours),
        model="raqdps",
        product="10km/grib2",
        variable=variable,
        level=RAQDPS_LEVEL,
        priority=["msc"],
        save_dir=save_dir,
        max_threads=max_threads,
        verbose=False,
    )
    ds = fh.xarray(
        None,
        max_threads=xarray_threads,
        backend_kwargs={"indexpath": ""},
        remove_grib=remove_grib,
    )

    if isinstance(ds, list):
        raise ValueError(f"Expected one RAQDPS hypercube for {variable}, got {len(ds)}")

    grib_var_name = next(iter(ds.data_vars))
    data_array = ds[grib_var_name].astype("float32")
    converted = convert_to_ug_m3(data_array, variable) if convert_units else data_array
    converted.name = variable
    converted.attrs.update(ds[grib_var_name].attrs)
    if convert_units:
        converted.attrs["units"] = "µg m-3"
    converted.attrs["raqdps_native_variable"] = grib_var_name

    return converted.to_dataset()


def open_raqdps_dataset(
    *,
    variables: Iterable[str] = RAQDPS_VARIABLES,
    run_time: datetime | None = None,
    forecast_hours: Iterable[int] = RAQDPS_FORECAST_HOURS,
    save_dir: Path | str = DEFAULT_SAVE_DIR,
    max_threads: int = 6,
    xarray_threads: int | None = 2,
    remove_grib: bool = False,
    convert_units: bool = True,
):
    """Open and merge multiple converted RAQDPS pollutant variables."""
    import xarray as xr

    from API.raqdps_utils import normalize_utc

    if run_time is None:
        run_time = latest_raqdps_run(save_dir=save_dir)

    forecast_hours = tuple(forecast_hours)
    datasets = [
        open_raqdps_variable(
            variable,
            run_time=run_time,
            forecast_hours=forecast_hours,
            save_dir=save_dir,
            max_threads=max_threads,
            xarray_threads=xarray_threads,
            remove_grib=remove_grib,
            convert_units=convert_units,
        )
        for variable in variables
    ]
    ds = xr.merge(datasets, compat="override")
    ds.attrs["raqdps_run_time_utc"] = normalize_utc(run_time).isoformat()
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-time", help="RAQDPS run time, e.g. 2026-07-06T12:00Z")
    parser.add_argument("--variables", nargs="+", default=["O3"])
    parser.add_argument("--forecast-hours", nargs="+", type=int, default=list(range(3)))
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--max-threads", type=int, default=6)
    parser.add_argument("--xarray-threads", type=int, default=2)
    parser.add_argument("--remove-grib", action="store_true")
    parser.add_argument("--native-units", action="store_true")
    return parser.parse_args()


def parse_run_time(value: str | None) -> datetime | None:
    """Parse an optional ISO-like UTC run time string."""
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def main() -> None:
    args = parse_args()
    ds = open_raqdps_dataset(
        variables=args.variables,
        run_time=parse_run_time(args.run_time),
        forecast_hours=args.forecast_hours,
        save_dir=args.save_dir,
        max_threads=args.max_threads,
        xarray_threads=args.xarray_threads,
        remove_grib=args.remove_grib,
        convert_units=not args.native_units,
    )
    print(ds)


if __name__ == "__main__":
    main()
