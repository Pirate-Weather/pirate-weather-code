"""LibreWXR radar nowcast export for the minutely forecast.

Run by a LibreWXR data pipeline's nowcast hook (``LIBREWXR_HOOK_NOWCAST_CMD``,
https://github.com/JoshuaKimsey/LibreWXR) once per nowcast, after the pipeline
has written the ``state.json`` snapshot that names the nowcast memmaps. It
composites the latest observed radar frame (T+0) and every extrapolated nowcast
step (T+10 ... T+90 min at the default 9 frames) onto one global 0.04 degree
lat/lon grid as composite reflectivity in dBZ and publishes the result as
``LibreWXR.zarr`` and ``LibreWXR.time.pickle`` in the same layout as the other
final stores, so the API reads it like any other model.

Values follow LibreWXR's own point sampling (``librewxr.mcp.sampling``): the finest
radar region covering a cell wins, the stored uint8 pixel ``(dBZ + 32) * 2`` is decoded,
and the RRQPE display calibration shift is undone. Only the radar extrapolation is
exported, never the NWP blend LibreWXR uses for its tiles, so cells outside radar
coverage are NaN and the API falls back to its models there. Inside coverage, no echo
and reflectivity below ``REFC_THRESHOLD`` are 0 dBZ, as for HRRR.

The same run exports LibreWXR's satellite lightning (GOES-18/19 GLM and, with EUMETSAT
credentials, Meteosat-12 LI flashes, ``<cache>/lightning/current.npz``) as two more
variables at T+0: the distance and bearing from each cell to the nearest recent strike,
which the API serves as ``currently.nearestStormDistance`` / ``nearestStormBearing``.

One pass per run, like the other ingests: it reads the cache the hook names
(``LIBREWXR_CACHE_DIR``), exits early when that nowcast is already published, and
exits non-zero on failure so the pipeline logs it. It runs inside the pipeline's
environment, so it needs the ``librewxr`` package (region definitions, projections,
coverage masks) and the store writer's dependencies; ``Docker/pirate-librewxr-dockerfile``
builds that image. The publishing settings are the usual ingest ones (``save_type``,
``forecast_path``, ``forecast_process_dir``, ``AWS_KEY`` / ``AWS_SECRET``, ``s3_endpoint``).

Alexander Rey, November 2026
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from API.constants.shared_const import (
    INGEST_VERSION_STR,
    LIBREWXR_GRID_HEIGHT,
    LIBREWXR_GRID_NORTH,
    LIBREWXR_GRID_RESOLUTION_DEG,
    LIBREWXR_GRID_WEST,
    LIBREWXR_GRID_WIDTH,
    LIBREWXR_MAX_NOWCAST_AGE_SECONDS,
    LIGHTNING_RECENT_WINDOW_SECONDS,
    LIGHTNING_SEARCH_DISTANCE_M,
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
from API.utils.storm_proc import _bearing_deg, _haversine_distance_m

logger = logging.getLogger(__name__)

# %% Store constants
STORE_NAME = "LibreWXR.zarr"
TIME_PICKLE_NAME = "LibreWXR.time.pickle"
# The published variables; zarr_vars[i] is column i + 1 of the API's LIBREWXR map
zarr_vars = ("REFC_composite", "Lightning_Distance", "Lightning_Direction")
RADAR_VARS = 1

# LibreWXR stores reflectivity as uint8 pixel = (dBZ + 32) * 2 (0 = no echo)
LIBREWXR_PIXEL_DBZ_OFFSET = 32.0
LIBREWXR_PIXEL_PER_DBZ = 2.0
# Reflectivity above this is not weather (large hail tops out near 75 dBZ); a
# saturated pixel (255 = 95.5 dBZ) is a decode sentinel and becomes NaN
LIBREWXR_MAX_VALID_DBZ = 80.0

# Lightning satellites: the satellite id LibreWXR stores with each flash, its name in
# the artifact's metadata, its sub-satellite longitude, and the great-circle angle
# from the sub-satellite point inside which a feed with no strikes is trusted to
# mean no lightning. On 2026-11-24, 99.9 % of 2 h of strikes lay within 55-60
# degrees for GOES GLM and 78 degrees for Meteosat-12 LI (99 % within 58); the
# limits stay inside those, where detection is reliable, and beyond them a strike
# is still used but its absence says nothing.
LIGHTNING_SATELLITES = (
    (18, "goes18", -137.2, 55.0),
    (19, "goes19", -75.2, 55.0),
    (12, "meteosat12", 0.0, 65.0),
)
# A feed whose newest data is older than this is treated as down: its coverage
# is dropped, so the API falls back to GFS there. Meteosat-12 normally runs
# 15-25 minutes behind; three missed 10-minute files is an outage.
LIGHTNING_MAX_FEED_LAG_SECONDS = 45 * 60
LIGHTNING_ARTIFACT = Path("lightning") / "current.npz"
EARTH_RADIUS_M = 6_371_000.0
# The nearest-strike search runs per 1 degree tile (25 x 25 cells): only tiles
# that can hold a cell within LIGHTNING_SEARCH_DISTANCE_M of a strike are queried.
LIGHTNING_TILE_CELLS = 25

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

librewxr_cache_dir = os.getenv("LIBREWXR_CACHE_DIR", default="")

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
    nowcast_frames = sorted(nowcast["frames"], key=lambda f: f["timestamp"])
    # The pipeline also dumps state.json mid-cycle (on radar frame merges), when
    # the new radar frame is in but the nowcast still belongs to the previous
    # one. Exporting that would publish a stale nowcast under the new run time
    # and then skip the real one as "No Update", so wait for the dump that
    # follows the nowcast: every step must be later than the latest frame.
    if int(nowcast_frames[0]["timestamp"]) <= latest_ts:
        raise LookupError(
            f"nowcast starts at {nowcast_frames[0]['timestamp']}, not after the "
            f"latest radar frame {latest_ts}; waiting for this cycle's nowcast"
        )
    steps = [(latest_ts, region_files(frame_store["memmap_dir"], latest["regions"]))]
    for frame in nowcast_frames:
        ts = int(frame["timestamp"])
        steps.append((ts, region_files(nowcast["memmap_dir"], frame["regions"])))
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


# %% Satellite lightning
def read_lightning_artifact(cache_dir: Path) -> tuple[np.ndarray, dict] | None:
    """The flash points and per-satellite metadata LibreWXR holds, or None.

    ``points`` is a structured array (``time_s``, ``lat``, ``lon``, ``energy``,
    ``satellite``) of every flash in LibreWXR's retention window (2 h); the
    metadata's ``satellites`` entry gives each feed's newest data end (``last_e``).
    A missing or unreadable artifact (lightning disabled, first minutes after a
    pipeline start, a torn write) is not a reason to hold back the radar
    nowcast, so it returns None and the lightning fields are published as NaN,
    where the API keeps its GFS storm distance.
    """
    path = cache_dir / LIGHTNING_ARTIFACT
    try:
        with np.load(path, allow_pickle=False) as loaded:
            points = loaded["points"]
            meta = json.loads(str(loaded["meta"]))
    except FileNotFoundError:
        logger.warning("No LibreWXR lightning artifact at %s", path)
        return None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        logger.warning("Unreadable LibreWXR lightning artifact %s: %s", path, exc)
        return None
    return points, meta.get("satellites", {})


def unit_vectors(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Points on the unit sphere, so a k-d tree's chord distance orders by arc length."""
    lat = np.deg2rad(lats.astype(np.float64))
    lon = np.deg2rad(lons.astype(np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def chord_for_distance(distance_m: float) -> float:
    """Unit-sphere chord length of a great-circle distance."""
    return 2.0 * np.sin(min(distance_m / EARTH_RADIUS_M, np.pi) / 2.0)


def lightning_storm_fields(
    points: np.ndarray,
    satellites: dict,
    now: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) and bearing (degrees) from every cell to the nearest recent strike.

    Each satellite feed contributes its strikes from the last
    ``LIGHTNING_RECENT_WINDOW_SECONDS`` of its own data and, while it is not
    stale, its coverage. A covered cell with no strike within
    ``LIGHTNING_SEARCH_DISTANCE_M`` is +inf (lightning rules a nearer storm out);
    an uncovered one is NaN unless a strike is in range. Bearings point from the
    cell towards the strike, as the GFS storm direction does.
    """
    distance = np.full((lats.size, lons.size), np.nan, dtype=np.float32)
    direction = np.full((lats.size, lons.size), np.nan, dtype=np.float32)
    cos_lat = np.cos(np.deg2rad(lats.astype(np.float64)))[:, np.newaxis]

    recent = []
    for satellite_id, name, sub_lon, coverage_deg in LIGHTNING_SATELLITES:
        last_e = satellites.get(name, {}).get("last_e")
        if last_e is None:
            logger.info("Lightning: no %s feed in the artifact", name)
            continue
        lag = now - float(last_e)
        if lag > LIGHTNING_MAX_FEED_LAG_SECONDS:
            logger.warning(
                "Lightning: %s data ends %.0f min ago; dropping its coverage",
                name,
                lag / 60,
            )
            continue
        # Great-circle angle from the sub-satellite point (on the equator):
        # cos(angle) = cos(lat) * cos(lon - sub_lon)
        cos_lon = np.cos(np.deg2rad(lons.astype(np.float64) - sub_lon))[np.newaxis, :]
        covered = cos_lat * cos_lon >= np.cos(np.deg2rad(coverage_deg))
        distance[covered] = np.inf
        in_window = (
            (points["satellite"] == satellite_id)
            & (points["time_s"] > float(last_e) - LIGHTNING_RECENT_WINDOW_SECONDS)
            & (points["time_s"] <= float(last_e))
        )
        recent.append(points[in_window])
        logger.info(
            "Lightning: %s %d recent strikes, data %.0f min old",
            name,
            int(in_window.sum()),
            lag / 60,
        )
        del covered

    strikes = np.concatenate(recent) if recent else points[:0]
    if strikes.size == 0:
        return distance, direction

    tree = cKDTree(unit_vectors(strikes["lat"], strikes["lon"]))
    search_chord = chord_for_distance(LIGHTNING_SEARCH_DISTANCE_M)

    # Pick the 1 degree tiles whose centre is within the search distance plus the
    # tile's half-diagonal of some strike; no cell outside them can be in range
    tile = LIGHTNING_TILE_CELLS
    tile_lats = lats.reshape(-1, tile).mean(axis=1)
    tile_lons = lons.reshape(-1, tile).mean(axis=1)
    tile_lat_grid, tile_lon_grid = np.meshgrid(tile_lats, tile_lons, indexing="ij")
    # Largest at the equator, where a degree of longitude is longest
    tile_size_m = tile * np.deg2rad(LIBREWXR_GRID_RESOLUTION_DEG) * EARTH_RADIUS_M
    half_diagonal_m = np.sqrt(2.0) * tile_size_m / 2.0
    tile_chord, _ = tree.query(
        unit_vectors(tile_lat_grid.ravel(), tile_lon_grid.ravel()),
        distance_upper_bound=chord_for_distance(
            LIGHTNING_SEARCH_DISTANCE_M + half_diagonal_m
        ),
        workers=-1,
    )
    near_tiles = np.isfinite(tile_chord).reshape(tile_lat_grid.shape)
    near_cells = np.repeat(np.repeat(near_tiles, tile, axis=0), tile, axis=1)
    rows, cols = np.nonzero(near_cells)
    del near_cells

    cell_lats = lats[rows]
    cell_lons = lons[cols]
    chord, nearest = tree.query(
        unit_vectors(cell_lats, cell_lons),
        distance_upper_bound=search_chord,
        workers=-1,
    )
    found = np.isfinite(chord)
    nearest = nearest[found]
    rows, cols = rows[found], cols[found]
    cell_lats, cell_lons = cell_lats[found], cell_lons[found]
    distance[rows, cols] = _haversine_distance_m(
        cell_lats, cell_lons, strikes["lat"][nearest], strikes["lon"][nearest]
    )
    direction[rows, cols] = _bearing_deg(
        cell_lats, cell_lons, strikes["lat"][nearest], strikes["lon"][nearest]
    )
    return distance, direction


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
        (RADAR_VARS, len(steps), LIBREWXR_GRID_HEIGHT, LIBREWXR_GRID_WIDTH),
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

    # %% Nearest recent lightning strike, at T+0 only
    lightning_start = time.time()
    lightning = read_lightning_artifact(cache_dir)
    if lightning is None:
        lightning_distance = np.full(stacked.shape[2:], np.nan, dtype=np.float32)
        lightning_direction = np.full(stacked.shape[2:], np.nan, dtype=np.float32)
    else:
        points, satellites = lightning
        lightning_distance, lightning_direction = lightning_storm_fields(
            points, satellites, time.time(), lats, lons
        )
        del points
    logger.info(
        "Lightning fields in %.1f s: %.1f%% of the globe covered, %.2f%% with a "
        "strike within %.0f km",
        time.time() - lightning_start,
        float((~np.isnan(lightning_distance)).mean() * 100),
        float(np.isfinite(lightning_distance).mean() * 100),
        LIGHTNING_SEARCH_DISTANCE_M / 1000,
    )

    # %% Write the final store (no time interpolation for 10-minute data)
    final_chunk = FINAL_CHUNK_SIZES["LibreWXR"]
    shard = FINAL_SHARD_SIZES["LibreWXR"]
    spatial_chunks = (shard, shard)
    radar_dask = da.from_array(
        stacked, chunks=(RADAR_VARS, len(steps)) + spatial_chunks
    )
    # The lightning fields have no nowcast steps; the NaN steps are built lazily
    # per shard so they never take a full-grid array of memory
    lightning_t0 = da.stack(
        [
            da.from_array(field[np.newaxis], chunks=(1,) + spatial_chunks)
            for field in (lightning_distance, lightning_direction)
        ]
    )
    lightning_steps = da.full(
        (2, len(steps) - 1) + stacked.shape[2:],
        np.nan,
        dtype=np.float32,
        chunks=(2, len(steps) - 1) + spatial_chunks,
    )
    stacked_dask = da.concatenate(
        (radar_dask, da.concatenate((lightning_t0, lightning_steps), axis=1)), axis=0
    ).rechunk((len(zarr_vars), len(steps)) + spatial_chunks)
    write_final_zarr(
        stacked_dask,
        forecast_process_dir + "/" + STORE_NAME,
        final_chunk,
        shard,
        zarr_store_workers,
        zarr_vars,
        times=times,
    )
    del stacked_dask, stacked, lightning_distance, lightning_direction

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


def main() -> int:
    """Export the nowcast the LibreWXR cache currently holds; one pass."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

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

    try:
        if not export_nowcast(lib, cache_dir, forecast_remote, zarr_store_workers):
            logger.info("No Update to LibreWXR")
    except LookupError as exc:
        # The pipeline is still warming up, or this snapshot predates its
        # nowcast; the hook runs again for the next one
        logger.info("LibreWXR not ready: %s", exc)
    except (OSError, ValueError):
        # A memmap replaced mid-read or an invalid composite: the next nowcast
        # starts from scratch, and the pipeline logs this exit
        logger.exception("LibreWXR export failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
