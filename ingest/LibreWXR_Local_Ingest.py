"""LibreWXR radar nowcast export for the minutely forecast.

A sidecar to the LibreWXR data pipeline (https://github.com/JoshuaKimsey/LibreWXR).
It polls the pipeline's cache directory (``LIBREWXR_CACHE_DIR``, where the pipeline
rewrites ``state.json`` after every fetch cycle) and, whenever a new nowcast has been
generated, composites the latest observed radar frame (T+0) and every extrapolated
nowcast step (T+10 ... T+90 min at the default 9 frames) onto one global 0.04 degree
lat/lon grid as composite reflectivity in dBZ. The result is published as
``LibreWXR.zarr`` and ``LibreWXR.time.pickle`` in the same layout as the other final
stores, so the API reads it like any other model.

Values follow LibreWXR's own point sampling (``librewxr.mcp.sampling``): the finest
radar region covering a cell wins, the stored uint8 pixel ``(dBZ + 32) * 2`` is decoded,
and the RRQPE display calibration shift is undone. Only the radar extrapolation is
exported, never the NWP blend LibreWXR uses for its tiles, so cells outside radar
coverage are NaN and the API falls back to its models there. Inside coverage, no echo
and reflectivity below ``REFC_THRESHOLD`` are 0 dBZ, as for HRRR.

Runs for ever by default; ``--once`` does a single pass (for a scheduler or a test).
Needs the ``librewxr`` package importable, with the same ``LIBREWXR_*`` settings as
the pipeline (region list, RRQPE offset), e.g. by running it from the pipeline's
environment file.

Alexander Rey, November 2026
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd

from API.constants.shared_const import (
    INGEST_VERSION_STR,
    LIBREWXR_GRID_HEIGHT,
    LIBREWXR_GRID_NORTH,
    LIBREWXR_GRID_RESOLUTION_DEG,
    LIBREWXR_GRID_WEST,
    LIBREWXR_GRID_WIDTH,
    LIBREWXR_MAX_NOWCAST_AGE_SECONDS,
    REFC_THRESHOLD,
)
from API.ingest_utils import (
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    configure_zarr_writes,
    open_remote,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

logger = logging.getLogger(__name__)

# %% Store constants
STORE_NAME = "LibreWXR.zarr"
TIME_PICKLE_NAME = "LibreWXR.time.pickle"
# The published store's only variable; column 1 of the API's LIBREWXR index map
zarr_vars = ("REFC_composite",)

# LibreWXR stores reflectivity as uint8 pixel = (dBZ + 32) * 2 (0 = no echo)
LIBREWXR_PIXEL_DBZ_OFFSET = 32.0
LIBREWXR_PIXEL_PER_DBZ = 2.0
# Reflectivity above this is not weather (large hail tops out near 75 dBZ); a
# saturated pixel (255 = 95.5 dBZ) is a decode sentinel and becomes NaN
LIBREWXR_MAX_VALID_DBZ = 80.0

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

librewxr_cache_dir = os.getenv("LIBREWXR_CACHE_DIR", default="")
poll_seconds = float(os.getenv("LIBREWXR_POLL_SECONDS", default="15"))

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/LibreWXR"
)
forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/LibreWXR")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


# %% LibreWXR cache access
def load_librewxr():
    """Import the LibreWXR pieces the export needs.

    Deferred so the module (its constants in particular) imports without
    LibreWXR installed, and so the LIBREWXR_* environment is read once at
    run time by its settings object.
    """
    from librewxr.config import settings
    from librewxr.data import coverage
    from librewxr.data.master_state import load_state, state_mtime
    from librewxr.data.regions import REGIONS
    from librewxr.sources import (
        collect_radar_coverage_metadata,
        enabled_regions_with_always_on,
    )
    from librewxr.tiles import coordinates

    return {
        "settings": settings,
        "coverage": coverage,
        "coordinates": coordinates,
        "REGIONS": REGIONS,
        "load_state": load_state,
        "state_mtime": state_mtime,
        "collect_radar_coverage_metadata": collect_radar_coverage_metadata,
        "enabled_regions_with_always_on": enabled_regions_with_always_on,
    }


def prepare_coverage_masks(lib: dict, cache_dir: Path) -> None:
    """Install the radar coverage masks (station ranges, product polygons).

    The pipeline persists them under ``<cache_dir>/coverage_masks``; they are
    memory-mapped from there when their signature matches this process's
    settings and rebuilt in memory otherwise (a few seconds).
    """
    started = time.time()
    enabled = lib["enabled_regions_with_always_on"](lib["settings"])
    station_map, range_overrides, polygons = lib["collect_radar_coverage_metadata"](
        lib["settings"]
    )
    if not lib["coverage"].load_masks(
        cache_dir, enabled, station_map, range_overrides, polygons
    ):
        lib["coverage"].build_coverage_masks(
            station_map, range_overrides=range_overrides, coverage_polygons=polygons
        )
        logger.info("Built coverage masks in %.1f s", time.time() - started)


def read_nowcast_snapshot(
    lib: dict, cache_dir: Path
) -> tuple[int, list[tuple[int, dict]]]:
    """The latest radar timestamp and the frames to export from ``state.json``.

    Returns ``(latest_ts, steps)`` where ``steps`` is ``[(valid_ts, regions)]``
    for T+0 (the latest observed radar frame) followed by every nowcast step,
    and ``regions`` maps a region name to ``(memmap path, dtype, shape)``.
    Raises ``LookupError`` while the pipeline has not published a nowcast yet.
    """
    payload = lib["load_state"](cache_dir)
    if payload is None:
        raise LookupError("no state.json yet")
    stores = payload["stores"]
    frame_store = stores.get("frame_store")
    nowcast = stores.get("nowcast_store")
    if not frame_store or not frame_store.get("frames"):
        raise LookupError("no radar frames in state.json")
    if not nowcast or not nowcast.get("frames"):
        raise LookupError("no nowcast frames in state.json")

    def region_files(memmap_dir: str, regions: dict) -> dict:
        return {
            name: (os.path.join(memmap_dir, basename), dtype, tuple(shape))
            for name, (basename, dtype, shape) in regions.items()
        }

    latest = max(frame_store["frames"], key=lambda f: f["timestamp"])
    latest_ts = int(latest["timestamp"])
    steps = [(latest_ts, region_files(frame_store["memmap_dir"], latest["regions"]))]
    for frame in sorted(nowcast["frames"], key=lambda f: f["timestamp"]):
        ts = int(frame["timestamp"])
        if ts <= latest_ts:
            # A nowcast left over from an older radar frame; the next cycle replaces it
            continue
        steps.append((ts, region_files(nowcast["memmap_dir"], frame["regions"])))
    if len(steps) < 2:
        raise LookupError(f"nowcast in state.json is not newer than radar {latest_ts}")
    return latest_ts, steps


# %% Global grid
def grid_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes (north to south) and longitudes (west to east)."""
    res = LIBREWXR_GRID_RESOLUTION_DEG
    lats = LIBREWXR_GRID_NORTH - (np.arange(LIBREWXR_GRID_HEIGHT) + 0.5) * res
    lons = LIBREWXR_GRID_WEST + (np.arange(LIBREWXR_GRID_WIDTH) + 0.5) * res
    return lats.astype(np.float32), lons.astype(np.float32)


def region_row_col_span(region) -> tuple[slice, slice]:
    """The global grid rows and columns a region's geographic bounds can touch."""
    res = LIBREWXR_GRID_RESOLUTION_DEG
    row_start = int(np.floor((LIBREWXR_GRID_NORTH - region.north) / res))
    row_stop = int(np.ceil((LIBREWXR_GRID_NORTH - region.south) / res))
    col_start = int(np.floor((region.west - LIBREWXR_GRID_WEST) / res))
    col_stop = int(np.ceil((region.east - LIBREWXR_GRID_WEST) / res))
    return (
        slice(max(0, row_start), min(LIBREWXR_GRID_HEIGHT, row_stop)),
        slice(max(0, col_start), min(LIBREWXR_GRID_WIDTH, col_stop)),
    )


def decode_dbz(pixels: np.ndarray, region_name: str, settings) -> np.ndarray:
    """uint8 radar pixels to composite reflectivity in dBZ (float32).

    Pixel 0 is "no echo". RRQPE frames carry LibreWXR's display calibration
    shift (``rrqpe_dbz_offset``), which is removed here so the store holds
    true values, as LibreWXR's own point tool reports them. Anything below
    ``REFC_THRESHOLD`` is zeroed, the same convention as the HRRR stores, and
    an impossible value (a saturated pixel) is NaN so the API falls back there.
    """
    dbz = pixels.astype(np.float32) / LIBREWXR_PIXEL_PER_DBZ - LIBREWXR_PIXEL_DBZ_OFFSET
    if region_name == "RRQPE":
        dbz -= float(settings.rrqpe_dbz_offset)
    dbz[(pixels == 0) | (dbz < REFC_THRESHOLD)] = 0.0
    dbz[dbz > LIBREWXR_MAX_VALID_DBZ] = np.nan
    return dbz


def composite_step(
    lib: dict,
    regions: dict,
    lats: np.ndarray,
    lons: np.ndarray,
    out: np.ndarray,
) -> int:
    """Fill ``out`` (NaN, shape grid) with one step's radar composite.

    Regions are laid down finest resolution first, as LibreWXR's renderer and
    point sampler do, and a cell takes the nearest pixel of the first region
    whose coverage mask (station range or product polygon) contains it. Cells
    no region covers stay NaN. Returns the number of regions used.
    """
    REGIONS = lib["REGIONS"]
    coordinates = lib["coordinates"]
    sample_coverage = lib["coverage"].sample_coverage
    settings = lib["settings"]

    claimed = np.zeros(out.shape, dtype=bool)
    used = 0
    for name in sorted(regions, key=lambda n: REGIONS[n].pixel_size):
        region = REGIONS.get(name)
        if region is None:
            logger.warning("Unknown LibreWXR region %s, skipping", name)
            continue
        path, dtype, shape = regions[name]
        pixels = np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=shape)

        rows, cols = region_row_col_span(region)
        lat_sub = lats[rows]
        lon_sub = lons[cols]
        if lat_sub.size == 0 or lon_sub.size == 0:
            continue
        if region.proj == "laea":
            col_grid, row_grid = coordinates._laea_pixel_coords(
                lon_sub, lat_sub, region
            )
        elif region.proj == "tmerc":
            col_grid, row_grid = coordinates._tmerc_pixel_coords(
                lon_sub, lat_sub, region
            )
        else:
            lat_grid, lon_grid = np.meshgrid(lat_sub, lon_sub, indexing="ij")
            col_grid = (lon_grid - region.west) / region.pixel_size
            row_grid = (region.north - lat_grid) / region._ps_y
        with np.errstate(invalid="ignore"):
            inside = (
                (row_grid >= 0)
                & (row_grid < shape[0])
                & (col_grid >= 0)
                & (col_grid < shape[1])
            )
        inside &= ~claimed[rows, cols]
        if not inside.any():
            continue
        lat_grid, lon_grid = np.meshgrid(lat_sub, lon_sub, indexing="ij")
        covered = np.zeros(inside.shape, dtype=bool)
        covered[inside] = sample_coverage(name, lat_grid[inside], lon_grid[inside])
        if not covered.any():
            continue
        r = row_grid[covered].astype(np.int32)
        c = col_grid[covered].astype(np.int32)
        values = decode_dbz(np.asarray(pixels[r, c]), name, settings)

        block = out[rows, cols]
        block[covered] = values
        out[rows, cols] = block
        claimed_block = claimed[rows, cols]
        claimed_block |= covered
        claimed[rows, cols] = claimed_block
        used += 1
        del pixels
    return used


# %% Export
def export_nowcast(
    lib: dict, cache_dir: Path, forecast_remote, zarr_store_workers: int
) -> bool:
    """Composite the current nowcast and publish it; False when there is nothing new."""
    T0 = time.time()
    latest_ts, steps = read_nowcast_snapshot(lib, cache_dir)
    run_time = pd.Timestamp(latest_ts, unit="s")

    # %% Skip the run if this nowcast has already been published
    previous_run_time = read_published_run_time(forecast_remote, TIME_PICKLE_NAME)
    if previous_run_time is not None and previous_run_time >= run_time:
        return False
    age = time.time() - latest_ts
    if age > LIBREWXR_MAX_NOWCAST_AGE_SECONDS:
        logger.warning(
            "LibreWXR nowcast base %s is %.0f min old; not publishing",
            run_time,
            age / 60,
        )
        return False
    logger.info("Processing LibreWXR nowcast %s (%d steps)", run_time, len(steps))

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(forecast_process_dir, exist_ok=True)

    # %% Composite every step onto the global grid: (variable, time, y, x)
    lats, lons = grid_coordinates()
    stacked = np.full(
        (len(zarr_vars), len(steps), LIBREWXR_GRID_HEIGHT, LIBREWXR_GRID_WIDTH),
        np.nan,
        dtype=np.float32,
    )
    times = []
    for index, (valid_ts, regions) in enumerate(steps):
        step_start = time.time()
        used = composite_step(lib, regions, lats, lons, stacked[0, index])
        times.append(valid_ts)
        logger.info(
            "Step %d/%d valid %s: %d regions in %.1f s",
            index + 1,
            len(steps),
            pd.Timestamp(valid_ts, unit="s"),
            used,
            time.time() - step_start,
        )

    # %% Validate before writing: coverage must exist and times must be ordered
    covered_fraction = float(np.isfinite(stacked[0, 0]).mean())
    if covered_fraction == 0.0:
        raise ValueError("LibreWXR composite has no radar coverage at T+0")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"LibreWXR step times are not increasing: {times}")
    logger.info(
        "Composite done in %.1f s: %.1f%% of the globe has radar coverage",
        time.time() - T0,
        covered_fraction * 100,
    )

    # %% Write the final store (no time interpolation for 10-minute data)
    final_chunk = FINAL_CHUNK_SIZES["LibreWXR"]
    shard = FINAL_SHARD_SIZES["LibreWXR"]
    stacked_dask = da.from_array(
        stacked, chunks=(len(zarr_vars), len(steps), shard, shard)
    )
    write_final_zarr(
        stacked_dask,
        forecast_process_dir + "/" + STORE_NAME,
        final_chunk,
        shard,
        zarr_store_workers,
        zarr_vars,
        times=times,
    )
    del stacked_dask, stacked

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [STORE_NAME],
        run_time,
        TIME_PICKLE_NAME,
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    logger.info("LibreWXR nowcast %s published in %.1f s", run_time, time.time() - T0)
    return True


def main(argv: list[str] | None = None) -> int:
    """Poll the LibreWXR cache and export each new nowcast."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--once", action="store_true", help="do one pass instead of polling for ever"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    if not librewxr_cache_dir:
        logger.error("LIBREWXR_CACHE_DIR is not set")
        return 2
    cache_dir = Path(librewxr_cache_dir)
    lib = load_librewxr()
    prepare_coverage_masks(lib, cache_dir)
    zarr_store_workers = configure_zarr_writes()
    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )
    logger.info(
        "Watching %s every %.0f s, publishing to %s",
        cache_dir,
        poll_seconds,
        forecast_path,
    )

    last_mtime = None
    while True:
        mtime = lib["state_mtime"](cache_dir)
        if mtime is not None and mtime != last_mtime:
            try:
                if not export_nowcast(
                    lib, cache_dir, forecast_remote, zarr_store_workers
                ):
                    logger.info("No Update to LibreWXR")
                last_mtime = mtime
            except LookupError as exc:
                # The pipeline is still warming up; state.json is rewritten every cycle
                logger.info("LibreWXR not ready: %s", exc)
                last_mtime = mtime
            except (OSError, ValueError):
                # A memmap replaced mid-read or an invalid composite: the next
                # state.json rewrite retries from scratch
                logger.exception("LibreWXR export failed; retrying on the next cycle")
                last_mtime = mtime
        if args.once:
            return 0
        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
