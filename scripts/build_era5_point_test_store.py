"""Build one week of the ERA5 point-read archive from Google ARCO ERA5.

Google's ARCO store (``full_37-1h-0p25deg-chunk-1``) has one chunk per hour
covering the whole globe, so a 25-hour time machine point read fetches ~450
global fields (7.6-8.5 s cold, 1.6 s warm cache). This script rewrites one week
into a point-read layout: one sharded int16 ``(variable, time, latitude,
longitude)`` array (``ERA5_ARCHIVE`` in ``API/constants/model_const.py``) whose
inner chunk holds every variable and the whole week for a 10x10 cell square,
grouped into 100x100 cell shards (~64 MB uncompressed). A point read is one
inner chunk (two when a request crosses a week boundary in a full archive).

Stored variables are the 18 the API reads from Google, with the four rain and
snow rates merged into rain and snow (the API only uses the sums), the API's 3x3
precipitation probability precomputed (so a read never needs the neighbouring
chunk), and direct solar radiation and cloud base height added. Values are
quantized to int16 in API units (``ERA5_ARCHIVE_ENCODING``) and the smooth fields
stored as hour-to-hour differences (``ERA5_ARCHIVE_DELTA_VARS``); readers decode
with ``API.io.ZarrHelpers.decode_era5_archive``. Measured on a June and a
January week: ~950 MiB per week against 1885 MiB for bitrounded float32, about
4.3 TB for 1940-2025, with ~1.6x the point-read throughput on OVH RBX High
Performance storage.

Weeks are aligned to 1940-01-01 (ERA5's first day, a Monday), so a full
archive built week by week keeps the same chunk boundaries.

The grid is Google's: latitude 90 -> -90 and longitude 0 -> 359.75 at 0.25 deg,
with latitude padded from 721 to 730 rows of fill to complete the last chunk.

Memory: each variable is converted to int16 as soon as it is downloaded, so the
week is held as ~6.7 GB of int16 rather than ~14 GB of float32, and the store
is written one band of 100x100 shards at a time.

Usage:
    python scripts/build_era5_point_test_store.py --out <dir> [--week-start 2015-06-29]
        [--upload bucket/prefix]   # S3 settings from s3_endpoint/s3_region, AWS_KEY/AWS_SECRET
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.constants.model_const import (  # noqa: E402
    ERA5_ARCHIVE,
    ERA5_ARCHIVE_DELTA_VARS,
    ERA5_ARCHIVE_ENCODING,
    ERA5_ARCHIVE_FILL,
    ERA5_ARCHIVE_NAN_VARS,
)
from API.ingest_utils import (  # noqa: E402
    encode_era5_archive_field,
    open_remote,
    publish_store,
    write_era5_archive_zarr,
)
from API.io.ZarrHelpers import decode_era5_archive  # noqa: E402

logger = logging.getLogger("era5_point_test_store")

GOOGLE_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_WEEK_EPOCH = dt.date(1940, 1, 1)
WEEK_HOURS = 7 * 24
ERA5_N_LAT = 721
ERA5_N_LON = 1440
FINAL_CHUNK = 10  # 10x10 cells (2.5 x 2.5 deg) per inner chunk
SHARD = 100  # 100x100 cells per shard: 19 x 168 x 100 x 100 x 2 B = 64 MB
# Google answers each hourly global chunk in ~0.1-0.3 s; many in flight keeps
# the 168-object-per-variable load bandwidth bound rather than latency bound.
SOURCE_READ_THREADS = 64
STORE_NAME = "ERA5_archive_test.zarr"
# Matches ERA5_PRECIP_PROB_THRESHOLD_M in API/request/grid_indexing.py, which the
# API used to apply to Google's total_precipitation (m) at read time.
PRECIP_PROB_THRESHOLD_M = 0.0001

# Google ARCO source variables for each stored variable, summed, and the factor
# from Google's units to the stored API units.
ARCHIVE_SOURCES = {
    "instantaneous_10m_wind_gust": (("instantaneous_10m_wind_gust",), 1.0),
    "mean_sea_level_pressure": (("mean_sea_level_pressure",), 0.01),  # Pa -> hPa
    "2m_temperature": (("2m_temperature",), 1.0),
    "2m_dewpoint_temperature": (("2m_dewpoint_temperature",), 1.0),
    "10m_u_component_of_wind": (("10m_u_component_of_wind",), 1.0),
    "10m_v_component_of_wind": (("10m_v_component_of_wind",), 1.0),
    "precipitation_type": (("precipitation_type",), 1.0),
    "total_precipitation": (("total_precipitation",), 1000.0),  # m -> mm
    # kg m-2 s-1 -> mm/h
    "rain_rate": (("large_scale_rain_rate", "convective_rain_rate"), 3600.0),
    "snowfall_rate_water_equivalent": (
        (
            "large_scale_snowfall_rate_water_equivalent",
            "convective_snowfall_rate_water_equivalent",
        ),
        3600.0,
    ),
    "total_column_ozone": (("total_column_ozone",), 46696.0),  # kg m-2 -> DU
    "total_cloud_cover": (("total_cloud_cover",), 100.0),  # 0-1 -> %
    # Hourly accumulations in J m-2 -> mean W m-2 over the hour.
    "downward_uv_radiation_at_the_surface": (
        ("downward_uv_radiation_at_the_surface",),
        1 / 3600,
    ),
    "surface_solar_radiation_downwards": (
        ("surface_solar_radiation_downwards",),
        1 / 3600,
    ),
    "convective_available_potential_energy": (
        ("convective_available_potential_energy",),
        1.0,
    ),
    "surface_pressure": (("surface_pressure",), 0.01),  # Pa -> hPa
    "total_sky_direct_solar_radiation_at_surface": (
        ("total_sky_direct_solar_radiation_at_surface",),
        1 / 3600,
    ),
    "cloud_base_height": (("cloud_base_height",), 1.0),  # m, NaN where there is none
}
DERIVED_VARS = ("precipitation_probability",)
ARCHIVE_UNITS = {
    "instantaneous_10m_wind_gust": "m/s",
    "mean_sea_level_pressure": "hPa",
    "2m_temperature": "K",
    "2m_dewpoint_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "precipitation_type": "ERA5 ptype code",
    "total_precipitation": "mm",
    "rain_rate": "mm/h",
    "snowfall_rate_water_equivalent": "mm/h",
    "total_column_ozone": "DU",
    "total_cloud_cover": "%",
    "downward_uv_radiation_at_the_surface": "W/m2",
    "surface_solar_radiation_downwards": "W/m2",
    "convective_available_potential_energy": "J/kg",
    "surface_pressure": "hPa",
    "precipitation_probability": "%",
    "total_sky_direct_solar_radiation_at_surface": "W/m2",
    "cloud_base_height": "m",
}


def week_start_for(day: dt.date) -> dt.date:
    """Monday starting the 1940-aligned week that contains ``day``."""
    weeks = (day - ERA5_WEEK_EPOCH).days // 7
    return ERA5_WEEK_EPOCH + dt.timedelta(days=7 * weeks)


def precipitation_probability(total_precipitation_m: np.ndarray) -> np.ndarray:
    """The API's ERA5 precipitation probability (%): the share of the 3x3
    neighbourhood above ``PRECIP_PROB_THRESHOLD_M``, ``(time, lat, lon)``.

    As in ``_load_era5_slice``, longitude wraps and the polar rows use the six
    cells that exist.
    """
    hits = (total_precipitation_m > PRECIP_PROB_THRESHOLD_M).astype(np.int8)
    across = hits + np.roll(hits, 1, axis=2) + np.roll(hits, -1, axis=2)
    del hits
    count = across.copy()
    count[:, 1:] += across[:, :-1]
    count[:, :-1] += across[:, 1:]
    del across
    cells = np.full(total_precipitation_m.shape[1], 9.0, np.float32)
    cells[[0, -1]] = 6.0
    return np.float32(100.0) * count / cells[None, :, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", required=True, help="parent directory for the store")
    parser.add_argument("--week-start", default="2015-06-29")
    parser.add_argument("--upload", help="bucket/prefix to publish to (S3)")
    parser.add_argument("--write-workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    variables = sorted(ERA5_ARCHIVE, key=ERA5_ARCHIVE.get)
    if set(variables) != set(ARCHIVE_SOURCES) | set(DERIVED_VARS):
        raise KeyError("ARCHIVE_SOURCES does not cover ERA5_ARCHIVE")

    requested = dt.date.fromisoformat(args.week_start)
    week_start = week_start_for(requested)
    if week_start != requested:
        logger.info("Using aligned week start %s (requested %s)", week_start, requested)
    start = np.datetime64(week_start.isoformat(), "h")
    stop = start + np.timedelta64(WEEK_HOURS - 1, "h")

    # Open Google ARCO and select the week.
    t0 = time.perf_counter()
    # Opened without dask: Google's store has 273 variables x 1.3 M hourly
    # chunks, and building dask chunks for all of them took 16.5 GB before any
    # data was read. Only the selected week is turned into dask chunks.
    ds = xr.open_zarr(GOOGLE_ERA5_URL, storage_options={"token": "anon"}, chunks=None)
    source_vars = sorted({v for names, _ in ARCHIVE_SOURCES.values() for v in names})
    missing = [var for var in source_vars if var not in ds]
    if missing:
        raise KeyError(f"Google ERA5 is missing {missing}")
    week = ds[source_vars].sel(time=slice(start, stop)).chunk({"time": 1})
    times = week["time"].values
    if len(times) != WEEK_HOURS or np.any(np.diff(times) != np.timedelta64(1, "h")):
        raise ValueError(f"Expected {WEEK_HOURS} hourly steps, got {len(times)}")
    if week.sizes["latitude"] != ERA5_N_LAT or week.sizes["longitude"] != ERA5_N_LON:
        raise ValueError(f"Unexpected ERA5 grid {dict(week.sizes)}")
    lats = week["latitude"].values
    lons = week["longitude"].values
    logger.info("Opened %s week %s .. %s", GOOGLE_ERA5_URL, times[0], times[-1])

    # Spot-check cells: both poles, the dateline and random cells. Their source
    # values are kept to compare with the decoded store.
    rng = np.random.default_rng(0)
    spot_cells = [
        (0, 0),
        (ERA5_N_LAT - 1, ERA5_N_LON - 1),
        (360, 0),
        (360, ERA5_N_LON - 1),
    ] + [
        (int(rng.integers(ERA5_N_LAT)), int(rng.integers(ERA5_N_LON)))
        for _ in range(12)
    ]
    spot_y = np.array([y for y, _ in spot_cells])
    spot_x = np.array([x for _, x in spot_cells])
    spot_source = {}

    # Download, convert and encode each variable into one padded int16 array.
    padded_lat = -(-ERA5_N_LAT // FINAL_CHUNK) * FINAL_CHUNK
    stacked = np.empty((len(variables), WEEK_HOURS, padded_lat, ERA5_N_LON), np.int16)
    stacked[:, :, ERA5_N_LAT:, :] = ERA5_ARCHIVE_FILL
    buffers = [
        np.empty((WEEK_HOURS, ERA5_N_LAT, ERA5_N_LON), np.float32) for _ in range(2)
    ]

    def store_encoded(name: str, values: np.ndarray) -> None:
        spot_source[name] = values[:, spot_y, spot_x].astype(np.float64)
        stacked[variables.index(name), :, :ERA5_N_LAT, :] = encode_era5_archive_field(
            name, values
        )

    # da.store writes each hourly source chunk straight into its slice: loading
    # with .values first held every chunk plus a concatenated copy per variable.
    with dask.config.set(scheduler="threads", num_workers=SOURCE_READ_THREADS):
        for name, (sources, factor) in ARCHIVE_SOURCES.items():
            t_var = time.perf_counter()
            for buffer, var in zip(buffers, sources):
                da.store(
                    week[var].transpose("time", "latitude", "longitude").data,
                    buffer,
                    lock=False,
                )
                # An hour of a field that is NaN everywhere means the download or
                # the source is incomplete (cloud base is NaN only where there is
                # no cloud base; encode_era5_archive_field rejects other NaNs).
                if not all(np.isfinite(hour).any() for hour in buffer):
                    raise ValueError(
                        f"ERA5 {var} has an empty hour; not writing the store"
                    )
            values = buffers[0] if len(sources) == 1 else buffers[0] + buffers[1]
            if name == "total_precipitation":
                store_encoded(
                    "precipitation_probability", precipitation_probability(values)
                )
            store_encoded(name, values * np.float32(factor))
            del values
            logger.info(
                "Read and encoded %s in %.1f s", name, time.perf_counter() - t_var
            )
    del buffers
    logger.info("Source read %.0f s", time.perf_counter() - t0)

    # Write the store.
    store_path = os.path.join(args.out, STORE_NAME)
    unix_times = (times - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    t_write = time.perf_counter()
    write_era5_archive_zarr(
        stacked,
        store_path,
        FINAL_CHUNK,
        SHARD,
        args.write_workers,
        {
            "time": [round(t) for t in unix_times],
            "variables": variables,
            "units": ARCHIVE_UNITS,
            "encoding": {name: list(ERA5_ARCHIVE_ENCODING[name]) for name in variables},
            "delta_variables": list(ERA5_ARCHIVE_DELTA_VARS),
            "nan_variables": list(ERA5_ARCHIVE_NAN_VARS),
            "source": GOOGLE_ERA5_URL,
            "week_start": week_start.isoformat(),
            "precipitation_probability_threshold_m": PRECIP_PROB_THRESHOLD_M,
            # The API locates ERA5 cells from the grid.
            "latitude_first": float(lats[0]),
            "longitude_first": float(lons[0]),
            "grid_step": float(lons[1] - lons[0]),
            "n_latitude": ERA5_N_LAT,
            "n_longitude": ERA5_N_LON,
        },
    )
    logger.info("Wrote %s in %.0f s", store_path, time.perf_counter() - t_write)

    # Reopen, decode the spot cells as a reader would and compare with the
    # source. Allowed error: half a step for linear fields; for sqrt fields the
    # rounding error of sqrt(value) * scale, sqrt(v) / scale + 1 / (4 scale^2).
    import zarr

    array = zarr.open_array(store_path, mode="r")
    for k, (y, x) in enumerate(spot_cells):
        decoded = decode_era5_archive(array[:, :, y, x], dict(array.attrs))
        for i, name in enumerate(variables):
            source = spot_source[name][:, k]
            kind, step_or_scale, _ = ERA5_ARCHIVE_ENCODING[name]
            finite = np.isfinite(source)
            if not np.array_equal(np.isfinite(decoded[i]), finite):
                raise ValueError(
                    f"Stored {name} at ({y}, {x}) has NaN where the source does not"
                )
            if kind == "lin":
                expected = source
                allowed = 0.5 * step_or_scale * 1.001 + 1e-6 * np.abs(source)
            else:
                expected = np.clip(source, 0, None)  # tiny negative accumulations are 0
                allowed = (
                    np.sqrt(expected) / step_or_scale + 0.25 / step_or_scale**2 + 1e-6
                )
            error = np.abs(decoded[i] - expected)
            if np.any(error[finite] > allowed[finite]):
                raise ValueError(
                    f"Stored {name} at ({y}, {x}) does not match the source"
                )
    logger.info(
        "Spot checks passed (%d cells x %d variables)", len(spot_cells), len(variables)
    )

    files = [p for p in Path(store_path).rglob("*") if p.is_file()]
    logger.info(
        "Store: %d objects, %.0f MiB",
        len(files),
        sum(p.stat().st_size for p in files) / 2**20,
    )

    # Publish
    if args.upload:
        remote = open_remote(
            args.upload, "S3", os.getenv("AWS_KEY", ""), os.getenv("AWS_SECRET", "")
        )
        t_up = time.perf_counter()
        version = publish_store(store_path, remote, STORE_NAME, week_start.isoformat())
        logger.info(
            "Published %s/%s in %.0f s",
            args.upload,
            version,
            time.perf_counter() - t_up,
        )


if __name__ == "__main__":
    main()
