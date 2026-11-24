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

The API reads these weeks first when ``ERA5_ARCHIVE`` points at the published
location (``API.io.ZarrHelpers.read_era5_archive_point``) and falls back to Google
for any week that is missing.

Two ways to publish a week (``--publish``):

* ``immutable``: ``ERA5_archive/<week>.zarr``, written once. Only a complete week
  of final ERA5 may be published this way; readers cache it for good.
* ``versioned``: ``ERA5_archive/<week>/<stamp>.zarr`` plus the pointer
  ``ERA5_archive/<week>.current.json`` (``publish_store``), for weeks that still
  change: preliminary ERA5T weeks, and the current week, which is published with
  the hours Google has so far (the rest are fill, and ``complete_through`` gives
  the last real hour). Readers re-resolve the pointer every few minutes and never
  see a store change under them. ``scripts/era5_archive_daily.py`` decides which.

Usage:
    python scripts/build_era5_point_test_store.py --out <dir> [--week-start 2015-06-29]
        [--upload bucket/prefix | /local/dir] [--publish immutable|versioned]
        # S3 settings from s3_endpoint/s3_region/s3_storage_class, AWS_KEY/AWS_SECRET
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
    ERA5_ARCHIVE_SOURCES,
    ERA5_ARCHIVE_STORE_DIR,
    ERA5_ARCHIVE_WEEK_EPOCH,
    ERA5_ARCHIVE_WEEK_HOURS,
)
from API.ingest_utils import (  # noqa: E402
    encode_era5_archive_field,
    open_remote,
    publish_store,
    upload_store,
    write_era5_archive_zarr,
)
from API.io.ZarrHelpers import decode_era5_archive  # noqa: E402

logger = logging.getLogger("era5_point_test_store")

GOOGLE_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_WEEK_EPOCH = dt.date.fromisoformat(ERA5_ARCHIVE_WEEK_EPOCH)
WEEK_HOURS = ERA5_ARCHIVE_WEEK_HOURS
ERA5_N_LAT = 721
ERA5_N_LON = 1440
FINAL_CHUNK = 10  # 10x10 cells (2.5 x 2.5 deg) per inner chunk
SHARD = 100  # 100x100 cells per shard: 19 x 168 x 100 x 100 x 2 B = 64 MB
# Google answers each hourly global chunk in ~0.1-0.3 s; many in flight keeps
# the 168-object-per-variable load bandwidth bound rather than latency bound.
SOURCE_READ_THREADS = 64
# Matches ERA5_PRECIP_PROB_THRESHOLD_M in API/request/grid_indexing.py, which the
# API used to apply to Google's total_precipitation (m) at read time.
PRECIP_PROB_THRESHOLD_M = 0.0001
# ERA5 has isolated spurious CAPE spikes over steep terrain: 2011-04-30 09 UTC
# has 24 cells over the Peruvian Andes at up to 466,656 J/kg, where observed CAPE
# tops out near 8,000 J/kg and ERA5's ordinary maximum in the tested weeks was
# ~9,700. Such values are capped (and counted in the log) rather than stored; a
# cap well above any real value leaves genuine extremes untouched. Other fields
# keep encode_era5_archive_field's strict range check, so new oddities still fail.
CAPE_MAX_J_PER_KG = 20000.0

# Google's ARCO store is a copy of the Copernicus Climate Data Store's ERA5 single
# levels, and its ERA5T now and then lacks a whole day that its metadata claims
# (google-research/arco-era5 issues #142 to #164: 2026-06-18, 07-06, 07-25, 08-31,
# 10-10, 10-14, 10-20..23, 11-04, 11-14, each backfilled only after a report). Those
# hours read as NaN. Once Google has claimed such a day for COPERNICUS_AFTER_DAYS,
# the builder fetches it from Copernicus instead; before that it fails the week as
# before, so the next daily run picks up Google's own backfill. On 2026-11-10 the
# two sources were bit-for-bit identical for every variable below, with the same
# grid, NaN cells and accumulation hours, so a filled hour needs no conversion.
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_DATASET = "reanalysis-era5-single-levels"
COPERNICUS_AFTER_DAYS = 3
# Google's variable names are the Copernicus request names; the NetCDF Copernicus
# returns uses the GRIB short names.
CDS_NETCDF_NAMES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_dewpoint_temperature": "d2m",
    "2m_temperature": "t2m",
    "cloud_base_height": "cbh",
    "convective_available_potential_energy": "cape",
    "convective_rain_rate": "crr",
    "convective_snowfall_rate_water_equivalent": "csfr",
    "downward_uv_radiation_at_the_surface": "uvb",
    "instantaneous_10m_wind_gust": "i10fg",
    "large_scale_rain_rate": "lsrr",
    "large_scale_snowfall_rate_water_equivalent": "lssfr",
    "mean_sea_level_pressure": "msl",
    "precipitation_type": "ptype",
    "surface_pressure": "sp",
    "surface_solar_radiation_downwards": "ssrd",
    "total_cloud_cover": "tcc",
    "total_column_ozone": "tco3",
    "total_precipitation": "tp",
    "total_sky_direct_solar_radiation_at_surface": "fdir",
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


def days_for_copernicus(
    empty_hours: np.ndarray, era5t_stop: np.datetime64, after_days: int
) -> list[dt.date]:
    """Days of ``empty_hours`` (Google hours that are NaN everywhere) to fetch from
    Copernicus. Raises while any of them is younger than ``after_days`` past
    Google's ERA5T end, so Google gets that long to backfill its own gap."""
    days = sorted({hour.astype("datetime64[D]").item() for hour in empty_hours})
    stop_day = era5t_stop.astype("datetime64[D]").item()
    recent = [day for day in days if (stop_day - day).days < after_days]
    if recent:
        raise ValueError(
            f"Google ERA5 has empty hours on {[d.isoformat() for d in recent]}; "
            f"waiting until {after_days} days past ERA5T before using Copernicus"
        )
    return days


def download_copernicus_day(
    day: dt.date, variables: list[str], work: Path
) -> xr.Dataset:
    """One day of ERA5 hours from the Copernicus CDS, with Google's names and a
    ``time`` axis. Needs CDSAPI_KEY, on an account that has accepted the ERA5
    licence. The request waits in the CDS queue: ~2 min on 2026-11-24 for a day
    of every variable (a 550 MB zip, ~2.4 GB unpacked in ``work``)."""
    import zipfile

    import cdsapi

    key = os.getenv("CDSAPI_KEY")
    if not key:
        raise RuntimeError(f"Google ERA5 is missing {day}; set CDSAPI_KEY to fetch it")
    # The store is written to --out only later, so it may not exist yet.
    work.mkdir(parents=True, exist_ok=True)
    target = work / f"copernicus-{day.isoformat()}.zip"
    cdsapi.Client(url=os.getenv("CDSAPI_URL", CDS_URL), key=key).retrieve(
        CDS_DATASET,
        {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [f"{day.year}"],
            "month": [f"{day.month:02d}"],
            "day": [f"{day.day:02d}"],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "data_format": "netcdf",
            "download_format": "zip",
        },
        str(target),
    )
    # Copernicus splits the NetCDF by GRIB step type: one file of instantaneous
    # fields, one of hourly accumulations.
    unpacked = work / f"copernicus-{day.isoformat()}"
    with zipfile.ZipFile(target) as archive:
        archive.extractall(unpacked)
    target.unlink()
    parts = [xr.open_dataset(path) for path in sorted(unpacked.glob("*.nc"))]
    return copernicus_to_google(xr.merge(parts, compat="override"), variables)


def copernicus_to_google(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Rename a Copernicus ERA5 NetCDF to Google's variable names and ``time``."""
    missing = [var for var in variables if CDS_NETCDF_NAMES[var] not in ds]
    if missing:
        raise KeyError(f"Copernicus ERA5 is missing {missing}")
    renamed = ds[[CDS_NETCDF_NAMES[var] for var in variables]].rename(
        {CDS_NETCDF_NAMES[var]: var for var in variables}
    )
    return renamed.rename(valid_time="time")


def open_archive_remote(upload: str):
    """The archive's bucket/prefix on S3, or a local directory (an absolute path)."""
    if os.path.isabs(upload):
        return open_remote(upload, "Download")
    return open_remote(
        upload, "S3", os.getenv("AWS_KEY", ""), os.getenv("AWS_SECRET", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", required=True, help="parent directory for the store")
    parser.add_argument("--week-start", default="2015-06-29")
    parser.add_argument("--upload", help="bucket/prefix (S3) or local directory")
    parser.add_argument(
        "--publish", choices=("immutable", "versioned"), default="immutable"
    )
    parser.add_argument("--write-workers", type=int, default=8)
    parser.add_argument(
        "--copernicus-after-days",
        type=int,
        default=COPERNICUS_AFTER_DAYS,
        help="days past Google's ERA5T end before a Google gap is fetched from Copernicus",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    variables = sorted(ERA5_ARCHIVE, key=ERA5_ARCHIVE.get)
    if set(variables) != set(ERA5_ARCHIVE_SOURCES) | set(DERIVED_VARS):
        raise KeyError("ERA5_ARCHIVE_SOURCES does not cover ERA5_ARCHIVE")

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
    source_vars = sorted(
        {v for names, _ in ERA5_ARCHIVE_SOURCES.values() for v in names}
    )
    missing = [var for var in source_vars if var not in ds]
    if missing:
        raise KeyError(f"Google ERA5 is missing {missing}")
    # Google extends final ERA5 (valid_time_stop, ~3 months behind) with the
    # preliminary ERA5T (valid_time_stop_era5t, ~5 days behind), which ECMWF later
    # replaces. Its time axis runs to 2050 with NaN past the data, so the week is
    # cut at the last ERA5T day; a week past the final date is tagged ERA5T so it
    # can be rebuilt once final ERA5 covers it.
    final_stop = np.datetime64(ds.attrs["valid_time_stop"]) + np.timedelta64(23, "h")
    era5t_stop = np.datetime64(ds.attrs["valid_time_stop_era5t"]) + np.timedelta64(
        23, "h"
    )
    available_stop = min(stop, era5t_stop)
    if available_stop < start:
        raise ValueError(f"Google has no ERA5 for the week of {week_start} yet")
    week = ds[source_vars].sel(time=slice(start, available_stop)).chunk({"time": 1})
    times = week["time"].values
    n_hours = int((available_stop - start) / np.timedelta64(1, "h")) + 1
    if len(times) != n_hours or np.any(np.diff(times) != np.timedelta64(1, "h")):
        raise ValueError(f"Expected {n_hours} hourly steps, got {len(times)}")
    if week.sizes["latitude"] != ERA5_N_LAT or week.sizes["longitude"] != ERA5_N_LON:
        raise ValueError(f"Unexpected ERA5 grid {dict(week.sizes)}")
    lats = week["latitude"].values
    lons = week["longitude"].values
    partial = n_hours < WEEK_HOURS
    era5t = bool(times[-1] > final_stop)
    if args.publish == "immutable" and (partial or era5t):
        # Readers cache an immutable week for good, so it must never change.
        raise ValueError(
            f"Week {week_start} is {'partial' if partial else 'ERA5T'}; "
            "only complete final-ERA5 weeks are published immutable (use --publish versioned)"
        )
    logger.info(
        "Opened %s week %s .. %s (%d h, %s)",
        GOOGLE_ERA5_URL,
        times[0],
        times[-1],
        n_hours,
        "ERA5T" if era5t else "final ERA5",
    )

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

    # Download, convert and encode each variable into one padded int16 array. A
    # partial week keeps the whole week's shape (so its chunks line up with the
    # complete week that replaces it) with fill after the last hour.
    padded_lat = -(-ERA5_N_LAT // FINAL_CHUNK) * FINAL_CHUNK
    stacked = np.full(
        (len(variables), WEEK_HOURS, padded_lat, ERA5_N_LON),
        ERA5_ARCHIVE_FILL,
        np.int16,
    )
    buffers = [
        np.empty((n_hours, ERA5_N_LAT, ERA5_N_LON), np.float32) for _ in range(2)
    ]

    def store_encoded(name: str, values: np.ndarray) -> None:
        spot_source[name] = values[:, spot_y, spot_x].astype(np.float64)
        stacked[variables.index(name), :n_hours, :ERA5_N_LAT, :] = (
            encode_era5_archive_field(name, values)
        )

    # Days fetched from Copernicus, downloaded once for every variable, and the
    # hours they filled (recorded in the store's attributes).
    copernicus_days: dict[dt.date, xr.Dataset] = {}
    copernicus_hours: set[str] = set()

    def fill_from_copernicus(var: str, buffer: np.ndarray, empty: np.ndarray) -> None:
        for day in days_for_copernicus(
            times[empty], era5t_stop, args.copernicus_after_days
        ):
            if day not in copernicus_days:
                t_day = time.perf_counter()
                day_ds = download_copernicus_day(day, source_vars, Path(args.out))
                if not (
                    np.array_equal(day_ds["latitude"].values, lats)
                    and np.array_equal(day_ds["longitude"].values, lons)
                ):
                    raise ValueError(
                        f"Copernicus ERA5 for {day} is not on Google's grid"
                    )
                copernicus_days[day] = day_ds
                logger.info(
                    "Fetched %s from Copernicus in %.0f s",
                    day,
                    time.perf_counter() - t_day,
                )
        for i in empty:
            hour = times[i]
            day_ds = copernicus_days[hour.astype("datetime64[D]").item()]
            buffer[i] = (
                day_ds[var].sel(time=hour).transpose("latitude", "longitude").values
            )
            copernicus_hours.add(str(hour.astype("datetime64[h]")))
        logger.warning(
            "Filled %d empty Google hours of %s from Copernicus", len(empty), var
        )

    # da.store writes each hourly source chunk straight into its slice: loading
    # with .values first held every chunk plus a concatenated copy per variable.
    with dask.config.set(scheduler="threads", num_workers=SOURCE_READ_THREADS):
        for name, (sources, factor) in ERA5_ARCHIVE_SOURCES.items():
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
                # Google's gaps are filled from Copernicus once old enough.
                empty = np.flatnonzero([not np.isfinite(hour).any() for hour in buffer])
                if empty.size:
                    fill_from_copernicus(var, buffer, empty)
                if not all(np.isfinite(hour).any() for hour in buffer):
                    raise ValueError(
                        f"ERA5 {var} has an empty hour; not writing the store"
                    )
            values = buffers[0] if len(sources) == 1 else buffers[0] + buffers[1]
            if name == "total_precipitation":
                store_encoded(
                    "precipitation_probability", precipitation_probability(values)
                )
            values = values * np.float32(factor)
            if name == "convective_available_potential_energy":
                spurious = int((values > CAPE_MAX_J_PER_KG).sum())
                if spurious:
                    logger.warning(
                        "Capping %d spurious ERA5 CAPE values (max %.0f J/kg) at %.0f",
                        spurious,
                        float(np.nanmax(values)),
                        CAPE_MAX_J_PER_KG,
                    )
                    np.minimum(values, CAPE_MAX_J_PER_KG, out=values)
            store_encoded(name, values)
            del values
            logger.info(
                "Read and encoded %s in %.1f s", name, time.perf_counter() - t_var
            )
    del buffers
    logger.info("Source read %.0f s", time.perf_counter() - t0)

    # Write the store.
    store_path = os.path.join(args.out, f"{week_start.isoformat()}.zarr")
    week_times = start + np.arange(WEEK_HOURS).astype("timedelta64[h]")
    unix_times = (week_times - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(
        1, "s"
    )
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
            # Hours Google lacked and Copernicus supplied (identical data, see
            # COPERNICUS_AFTER_DAYS); empty when the week is all Google's.
            "copernicus_filled_hours": sorted(copernicus_hours),
            "copernicus_dataset": CDS_DATASET if copernicus_hours else None,
            "era5t": era5t,
            "google_valid_time_stop": ds.attrs["valid_time_stop"],
            "google_valid_time_stop_era5t": ds.attrs["valid_time_stop_era5t"],
            # The last hour with data (the week's last hour unless it is partial).
            "complete_through": round(unix_times[n_hours - 1]),
            "hours_available": n_hours,
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
        decoded = decode_era5_archive(array[:, :n_hours, y, x], dict(array.attrs))
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

    # Publish. upload_store writes the chunks, then the array metadata, then the
    # root zarr.json, and readers treat a store without its root metadata as
    # missing, so a half-uploaded week is never read. A versioned week goes to a
    # new path and only then does its pointer move (publish_store), so a reader
    # never sees a store change under it; the pointer's run_time is the last hour
    # with data, which the daily job compares with Google's to decide on a rebuild.
    if args.upload:
        remote = open_archive_remote(args.upload)
        t_up = time.perf_counter()
        name = f"{ERA5_ARCHIVE_STORE_DIR}/{week_start.isoformat()}.zarr"
        if args.publish == "immutable":
            upload_store(store_path, remote, name)
            published = name
        else:
            complete_through = str(times[-1].astype("datetime64[h]"))
            published = publish_store(store_path, remote, name, complete_through)
        logger.info(
            "Published %s/%s (%s) in %.0f s",
            args.upload,
            published,
            args.publish,
            time.perf_counter() - t_up,
        )


if __name__ == "__main__":
    main()
