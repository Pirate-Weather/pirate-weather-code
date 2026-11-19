#!/usr/bin/env python3
"""Compare hourly RDAQA and RAQDPS PM2.5 at one lat/lon.

Downloads the GRIB2 files straight from the MSC Datamart (it keeps about 30
days) and prints one row per hour, for checking both against a station by hand.
RDAQA and RAQDPS share the same 10 km rotated grid, so one nearest-cell lookup
serves both.

``--analysis`` picks the RDAQA files: ``prelim`` (default, posted ~40 min after
the hour), ``final`` (~2 h after), or ``both``, which adds a Final-Prelim column.

For each hour, RAQDPS comes from the newest 00/12 UTC run at least
``--min-lead`` hours before it. The default 0 takes the freshest run (leads
0-11 h); about 5 matches what the API had, since a run is published roughly
4.5 h after its cycle.

Needs cfgrib and eccodes (the ingest environment), not just the API's packages.

Example:
    scripts/compare_rdaqa_raqdps_point.py --lat 45.4343 --lon -75.6764 \\
        --hours 48 --tz America/Toronto --analysis both --csv /tmp/ottawa.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
from zoneinfo import ZoneInfo

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

from API.raqdps_utils import build_raqdps_url, convert_to_output_units  # noqa: E402

DATAMART = "https://dd.weather.gc.ca"
VARIABLE = "PM2.5"
RAQDPS_MAX_LEAD = 72
# How far back from now to look for the newest RDAQA hour (final lags ~2 h)
LATEST_PROBE_HOURS = 6
# RDAQA analysis -> filename separator and prefix
ANALYSIS_NAMES = {"prelim": "-Prelim_", "final": "_"}
ANALYSIS_CHOICES = {
    "prelim": ("prelim",),
    "final": ("final",),
    "both": ("prelim", "final"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument(
        "--analysis",
        choices=sorted(ANALYSIS_CHOICES),
        default="prelim",
        help="RDAQA analysis to read (default prelim; both adds Final-Prelim)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours to compare, ending at --end (default 24)",
    )
    parser.add_argument(
        "--end",
        help="Last hour to compare, UTC (e.g. 2026-11-18T12). Default: the "
        "newest hour with every requested RDAQA analysis on the Datamart",
    )
    parser.add_argument(
        "--min-lead",
        type=int,
        default=0,
        help="Minimum RAQDPS lead in hours (default 0; ~5 matches the API)",
    )
    parser.add_argument(
        "--tz",
        default="UTC",
        help="Time zone for the printed times, e.g. America/Toronto",
    )
    parser.add_argument("--csv", type=Path, help="Also write the rows to this CSV")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rdaqa_raqdps_compare",
        help="Where downloaded GRIB2 files are kept between runs",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def rdaqa_url(valid_time: datetime, analysis: str) -> str:
    """Datamart URL of one RDAQA PM2.5 analysis hour."""
    name = f"RDAQA{ANALYSIS_NAMES[analysis]}{VARIABLE}"
    return (
        f"{DATAMART}/{valid_time:%Y%m%d}/WXO-DD/model_rdaqa/10km/{valid_time:%H}/"
        f"{valid_time:%Y%m%dT%HZ}_MSC_{name}_Sfc_RLatLon0.09_PT0H.grib2"
    )


def raqdps_run_for(valid_time: datetime, min_lead: int) -> tuple[datetime, int]:
    """Newest 00/12 UTC RAQDPS run at least ``min_lead`` hours before ``valid_time``."""
    latest_run = valid_time - timedelta(hours=min_lead)
    run = latest_run.replace(hour=12 if latest_run.hour >= 12 else 0)
    return run, int((valid_time - run).total_seconds() // 3600)


def url_exists(url: str) -> bool:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=30) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False


def latest_rdaqa_hour(analyses: tuple[str, ...]) -> datetime:
    """Newest hour for which every requested analysis is on the Datamart."""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    for back in range(LATEST_PROBE_HOURS + 1):
        candidate = now - timedelta(hours=back)
        if all(url_exists(rdaqa_url(candidate, a)) for a in analyses):
            return candidate
    raise RuntimeError(
        f"No RDAQA {'/'.join(analyses)} {VARIABLE} file in the last "
        f"{LATEST_PROBE_HOURS} hours"
    )


def fetch(url: str, cache_dir: Path) -> Path | None:
    """Download ``url`` into the cache, or return None if it is not on the Datamart."""
    path = cache_dir / url.rsplit("/", 1)[1]
    if path.exists():
        return path
    tmp = path.with_suffix(".part")
    try:
        urlretrieve(url, tmp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    tmp.rename(path)
    return path


def fetch_raqdps(
    valid_time: datetime, min_lead: int, cache_dir: Path
) -> tuple[Path | None, datetime, int]:
    """Download RAQDPS for ``valid_time`` from the newest run that has it.

    Starts at the newest run at least ``min_lead`` hours before ``valid_time``
    and steps back 12 h at a time while that run's file is not on the Datamart
    (a run is posted ~4.5 h after its cycle), up to the last 72 h lead.
    Returns the path (None if no run has it) and the run and lead used.
    """
    first_run, first_lead = raqdps_run_for(valid_time, min_lead)
    run, lead = first_run, first_lead
    while lead <= RAQDPS_MAX_LEAD:
        path = fetch(build_raqdps_url(run, VARIABLE, lead), cache_dir)
        if path is not None:
            return path, run, lead
        run -= timedelta(hours=12)
        lead += 12
    return None, first_run, first_lead


def open_field(path: Path) -> xr.Dataset:
    return xr.open_dataset(
        path,
        engine="cfgrib",
        decode_times=False,
        backend_kwargs={"indexpath": ""},
    )


def nearest_cell(
    path: Path, lat: float, lon: float
) -> tuple[int, int, float, float, float]:
    """Nearest grid cell (y, x, lat, lon, distance km) by great-circle distance."""
    ds = open_field(path)
    try:
        grid_lat = ds["latitude"].values
        grid_lon = ds["longitude"].values
    finally:
        ds.close()
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    lat_rad, lon_rad = np.deg2rad(grid_lat), np.deg2rad(grid_lon)
    t_lat, t_lon = np.deg2rad(lat), np.deg2rad(lon)
    hav = (
        np.sin((lat_rad - t_lat) / 2) ** 2
        + np.cos(t_lat) * np.cos(lat_rad) * np.sin((lon_rad - t_lon) / 2) ** 2
    )
    y, x = np.unravel_index(int(np.nanargmin(hav)), hav.shape)
    distance_km = float(6371.0 * 2 * np.arcsin(np.sqrt(hav[y, x])))
    return int(y), int(x), float(grid_lat[y, x]), float(grid_lon[y, x]), distance_km


def point_value(path: Path | None, y: int, x: int) -> float:
    """PM2.5 in µg/m³ at one cell, or NaN when the file is missing."""
    if path is None:
        return float("nan")
    ds = open_field(path)
    try:
        field = ds[next(iter(ds.data_vars))]
        return float(convert_to_output_units(field.values[y, x], VARIABLE))
    finally:
        ds.close()


def fmt(value: float) -> str:
    return "—" if np.isnan(value) else f"{value:.1f}"


def mean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def main() -> None:
    args = parse_args()
    if args.hours < 1:
        raise SystemExit("--hours must be at least 1")
    if not 0 <= args.min_lead <= RAQDPS_MAX_LEAD - 12:
        raise SystemExit(f"--min-lead must be between 0 and {RAQDPS_MAX_LEAD - 12}")
    tz = ZoneInfo(args.tz)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    analyses = ANALYSIS_CHOICES[args.analysis]
    # RAQDPS is compared against final when it is read, else prelim
    reference = "final" if "final" in analyses else "prelim"

    if args.end:
        end = datetime.fromisoformat(args.end).replace(
            minute=0, second=0, microsecond=0
        )
        end = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
    else:
        end = latest_rdaqa_hour(analyses)
    hours = [end - timedelta(hours=h) for h in range(args.hours - 1, -1, -1)]

    urls = {a: [rdaqa_url(t, a) for t in hours] for a in analyses}

    print(
        f"Downloading {len(hours) * (len(analyses) + 1)} files to {args.cache_dir} ..."
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = {
            source: list(pool.map(lambda u: fetch(u, args.cache_dir), source_urls))
            for source, source_urls in urls.items()
        }
        # RAQDPS falls back to an older run when the newest is not posted yet
        raqdps = list(
            pool.map(lambda t: fetch_raqdps(t, args.min_lead, args.cache_dir), hours)
        )
    paths["raqdps"] = [path for path, _, _ in raqdps]
    runs = [(run, lead) for _, run, lead in raqdps]

    grid_file = next(
        (p for source_paths in paths.values() for p in source_paths if p), None
    )
    if grid_file is None:
        raise SystemExit("None of the requested files are on the Datamart")
    y, x, cell_lat, cell_lon, dist = nearest_cell(grid_file, args.lat, args.lon)
    print(
        f"Nearest cell y={y} x={x} at {cell_lat:.4f}, {cell_lon:.4f} "
        f"({dist:.1f} km from {args.lat}, {args.lon})\n"
    )

    values = {
        source: np.array([point_value(p, y, x) for p in source_paths])
        for source, source_paths in paths.items()
    }

    # Output columns: (label, CSV name, values)
    columns = [(f"RDAQA {a.title()}", f"rdaqa_{a}_pm25", values[a]) for a in analyses]
    if args.analysis == "both":
        columns.append(
            ("Final-Prelim", "final_minus_prelim", values["final"] - values["prelim"])
        )
    columns += [
        ("RAQDPS", "raqdps_pm25", values["raqdps"]),
        (
            f"RAQDPS-{reference.title()}",
            f"raqdps_minus_{reference}",
            values["raqdps"] - values[reference],
        ),
    ]
    widths = [max(len(label) + 2, 9) for label, _, _ in columns]

    header = (
        f"{'Time (' + args.tz + ')':<26}"
        + "".join(f"{label:>{w}}" for (label, _, _), w in zip(columns, widths))
        + "   RAQDPS run (lead)"
    )
    print(f"PM2.5 µg/m³\n{header}\n{'-' * len(header)}")
    for i, (t, (run, lead)) in enumerate(zip(hours, runs, strict=True)):
        print(
            f"{t.astimezone(tz):%Y-%m-%d %H:%M %Z}".ljust(26)
            + "".join(f"{fmt(col[i]):>{w}}" for (_, _, col), w in zip(columns, widths))
            + f"   {run:%m-%d %HZ} (f{lead:03d})"
        )
    # Means over the hours every column has, so a gap in one source does not
    # shift its mean against the others
    common = np.all([np.isfinite(col) for _, _, col in columns], axis=0)
    print("-" * len(header))
    print(
        f"{'Mean':<26}"
        + "".join(
            f"{fmt(mean_or_nan(col[common])):>{w}}"
            for (_, _, col), w in zip(columns, widths)
        )
    )
    if not common.all():
        print(f"Means use the {int(common.sum())} hours with every column present")

    missing = {
        source: int(np.isnan(vals).sum())
        for source, vals in values.items()
        if np.isnan(vals).any()
    }
    if missing:
        print(
            "Missing on the Datamart: "
            + ", ".join(f"{n} {source} hours" for source, n in missing.items())
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["time"] + [name for _, name, _ in columns] + ["raqdps_run", "lead_h"]
            )
            for i, (t, (run, lead)) in enumerate(zip(hours, runs, strict=True)):
                writer.writerow(
                    [t.astimezone(tz).isoformat()]
                    + [
                        "" if np.isnan(col[i]) else round(col[i], 2)
                        for _, _, col in columns
                    ]
                    + [f"{run:%m-%d %HZ}", lead]
                )
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
