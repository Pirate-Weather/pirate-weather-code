#!/usr/bin/env python3
"""Fetch Pirate Weather's required ERA5 variables for CSV points into Zarr."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import sys
from collections.abc import Iterator
from dask.diagnostics import ProgressBar
from dataclasses import dataclass
from pathlib import Path

import dask
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.constants.model_const import ERA5  # noqa: E402

DEFAULT_SOURCE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
TIME_CHUNK_HOURS = 24


@dataclass(frozen=True)
class Point:
    row_number: int
    label: str
    latitude: float
    longitude: float


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the ERA5 variables required by Pirate Weather at points from "
            "a CSV and write a local API-compatible Zarr store."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV with lat/lon columns.")
    parser.add_argument("output_path", type=Path, help="Local output Zarr directory.")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        required=True,
        help="First UTC date to fetch, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        required=True,
        help="Last UTC date to fetch (inclusive), in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--spatial-chunk",
        "--point-chunk",
        dest="spatial_chunk",
        type=int,
        default=1,
        help=(
            "Number of latitude and longitude cells per output chunk. "
            "Time chunks are always 24 hours."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of Dask worker threads used while downloading.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Source ERA5 Zarr URL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_path if it already exists.",
    )
    return parser.parse_args()


def detect_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        f"Could not find any of the required CSV columns: {', '.join(candidates)}"
    )


def slugify(value: str) -> str:
    cleaned = []
    for character in value.strip():
        if character.isalnum() or character in {"-", "_"}:
            cleaned.append(character)
        elif character in {" ", "."}:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "point"


def load_points(csv_path: Path) -> list[Point]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file is missing a header row.")

        latitude_column = detect_column(
            reader.fieldnames, ("lat", "latitude", "slatitude")
        )
        longitude_column = detect_column(
            reader.fieldnames,
            ("lon", "lng", "longitude", "long", "slongitude", "slongitud"),
        )

        label_column = None
        for candidate in (
            "id",
            "locationid",
            "name",
            "label",
            "site",
            "station",
        ):
            try:
                label_column = detect_column(reader.fieldnames, (candidate,))
                break
            except ValueError:
                continue

        points = []
        labels = set()
        for row_number, row in enumerate(reader, start=2):
            latitude_value = row.get(latitude_column, "").strip()
            longitude_value = row.get(longitude_column, "").strip()
            if not latitude_value or not longitude_value:
                raise ValueError(f"Row {row_number} is missing latitude or longitude.")

            latitude = float(latitude_value)
            longitude = float(longitude_value)
            if not -90 <= latitude <= 90:
                raise ValueError(
                    f"Row {row_number} latitude must be between -90 and 90."
                )
            if not -180 <= longitude <= 360:
                raise ValueError(
                    f"Row {row_number} longitude must be between -180 and 360."
                )

            raw_label = row.get(label_column, "") if label_column else ""
            label = slugify(raw_label) if raw_label else f"row_{row_number}"
            if label in labels:
                label = f"{label}_row_{row_number}"
            labels.add(label)
            points.append(Point(row_number, label, latitude, longitude))

    if not points:
        raise ValueError("CSV file does not contain any data rows.")
    return points


def date_range(start_date: dt.date, end_date: dt.date) -> Iterator[dt.date]:
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def nearest_grid_indices(
    source_latitudes: np.ndarray,
    source_longitudes: np.ndarray,
    points: list[Point],
) -> tuple[np.ndarray, np.ndarray]:
    requested_latitudes = np.asarray([point.latitude for point in points])
    requested_longitudes = np.mod(
        np.asarray([point.longitude for point in points]), 360
    )

    latitude_indices = np.abs(
        source_latitudes[:, np.newaxis] - requested_latitudes
    ).argmin(axis=0)
    longitude_distance = np.abs(source_longitudes[:, np.newaxis] - requested_longitudes)
    longitude_distance = np.minimum(longitude_distance, 360 - longitude_distance)
    longitude_indices = longitude_distance.argmin(axis=0)
    return latitude_indices, longitude_indices


def build_daily_dataset(
    source: xr.Dataset,
    points: list[Point],
    latitude_indices: np.ndarray,
    longitude_indices: np.ndarray,
    target_date: dt.date,
) -> xr.Dataset:
    start = np.datetime64(target_date.isoformat(), "ns")
    stop = start + np.timedelta64(TIME_CHUNK_HOURS, "h")
    expected_times = np.arange(start, stop, np.timedelta64(1, "h"))
    day = source[list(ERA5)].sel(time=slice(start, stop - np.timedelta64(1, "h")))

    actual_times = day["time"].values.astype("datetime64[ns]")
    if not np.array_equal(actual_times, expected_times):
        raise ValueError(
            f"ERA5 does not contain all 24 hourly values for {target_date.isoformat()}."
        )

    unique_latitude_indices = np.unique(latitude_indices)
    unique_longitude_indices = np.unique(longitude_indices)
    selected = day.isel(
        latitude=unique_latitude_indices,
        longitude=unique_longitude_indices,
    )

    data_vars = {
        name: selected[name].transpose("time", "latitude", "longitude") for name in ERA5
    }
    result = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": actual_times,
            "latitude": selected["latitude"].values,
            "longitude": selected["longitude"].values,
            "requested_point": np.arange(len(points), dtype=np.int64),
            "point_label": (
                "requested_point",
                [point.label for point in points],
            ),
            "requested_latitude": (
                "requested_point",
                [point.latitude for point in points],
            ),
            "requested_longitude": (
                "requested_point",
                [point.longitude for point in points],
            ),
            "selected_latitude": (
                "requested_point",
                source["latitude"].values[latitude_indices],
            ),
            "selected_longitude": (
                "requested_point",
                source["longitude"].values[longitude_indices],
            ),
        },
    )
    result.attrs.update(
        {
            "description": "ERA5 point data required by Pirate Weather",
            "layout": "Pirate Weather API-compatible latitude/longitude grid",
            "time_zone": "UTC",
        }
    )
    return result


def write_points_zarr(
    source: xr.Dataset,
    points: list[Point],
    output_path: Path,
    start_date: dt.date,
    end_date: dt.date,
    spatial_chunk: int,
    workers: int,
    source_url: str = DEFAULT_SOURCE,
) -> None:
    if start_date > end_date:
        raise ValueError("--start-date must be on or before --end-date.")
    if spatial_chunk < 1:
        raise ValueError("--spatial-chunk must be at least 1.")
    if workers < 1:
        raise ValueError("--workers must be at least 1.")

    missing_variables = set(ERA5) - set(source.data_vars)
    if missing_variables:
        missing = ", ".join(sorted(missing_variables))
        raise ValueError(f"ERA5 source is missing required variables: {missing}")

    latitude_indices, longitude_indices = nearest_grid_indices(
        source["latitude"].values,
        source["longitude"].values,
        points,
    )
    latitude_count = len(np.unique(latitude_indices))
    longitude_count = len(np.unique(longitude_indices))
    latitude_chunk = min(spatial_chunk, latitude_count)
    longitude_chunk = min(spatial_chunk, longitude_count)
    encoding = {
        name: {"chunks": (TIME_CHUNK_HOURS, latitude_chunk, longitude_chunk)}
        for name in ERA5
    }
    created_utc = dt.datetime.now(dt.UTC).isoformat()
    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=workers):
            for day_index, target_date in enumerate(date_range(start_date, end_date)):
                daily = build_daily_dataset(
                    source,
                    points,
                    latitude_indices,
                    longitude_indices,
                    target_date,
                ).load()
                daily.attrs["source"] = source_url
                daily.attrs["created_utc"] = created_utc

                if day_index == 0:
                    daily.to_zarr(
                        output_path,
                        mode="w",
                        encoding=encoding,
                        consolidated=True,
                        zarr_format=3,
                    )
                else:
                    daily.to_zarr(
                        output_path,
                        mode="a",
                        append_dim="time",
                        consolidated=True,
                        zarr_format=3,
                    )
                # print(
                #     f"Wrote {target_date.isoformat()} "
                #     f"({TIME_CHUNK_HOURS} hours, {latitude_count} latitudes, "
                #     f"{longitude_count} longitudes)",
                #     flush=True,
                # )


def main() -> int:
    args = parse_args()
    if args.output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output path already exists: {args.output_path}. "
                "Use --overwrite to replace it."
            )
        if args.output_path.is_dir():
            shutil.rmtree(args.output_path)
        else:
            args.output_path.unlink()

    points = load_points(args.csv_path)
    print(f"Opening ERA5 source for {len(points)} points", flush=True)
    source = xr.open_zarr(
        args.source,
        chunks={},
        storage_options={
            "token": "anon",
            "skip_instance_cache": True,
        },
    )
    try:
        write_points_zarr(
            source=source,
            points=points,
            output_path=args.output_path,
            start_date=args.start_date,
            end_date=args.end_date,
            spatial_chunk=args.spatial_chunk,
            workers=args.workers,
            source_url=args.source,
        )
    except Exception:
        if args.output_path.exists():
            shutil.rmtree(args.output_path)
        raise
    finally:
        source.close()

    print(f"Finished writing {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
