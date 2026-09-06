"""Grid indexing and data fetch helpers."""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from API.constants.grid_const import (
    HRRR_X_MAX,
    HRRR_X_MIN,
    HRRR_Y_MAX,
    HRRR_Y_MIN,
    NBM_X_MAX,
    NBM_X_MIN,
    NBM_Y_MAX,
    NBM_Y_MIN,
    RTMA_RU_AXIS,
    RTMA_RU_CENTRAL_LAT,
    RTMA_RU_CENTRAL_LONG,
    RTMA_RU_DELTA,
    RTMA_RU_MIN_X,
    RTMA_RU_MIN_Y,
    RTMA_RU_PARALLEL,
    RTMA_RU_X_MAX,
    RTMA_RU_X_MIN,
    RTMA_RU_Y_MAX,
    RTMA_RU_Y_MIN,
)
from API.constants.model_const import ERA5, ERA5_SOURCE_VARS
from API.constants.shared_const import HISTORY_PERIODS
from API.utils.geo import is_in_north_america, lambertGridMatch
from API.utils.timing import StepTimer

ERA5_PRECIP_PROB_THRESHOLD_M = 0.0001  # m, matching ERA5 total_precipitation units
SILAM_LAT_START = -89.6
SILAM_LON_START = -179.8
SILAM_GRID_DELTA = 0.2
SILAM_LAT_COUNT = 897
SILAM_LON_COUNT = 1800


def _normalize_longitude_180(lon):
    """Normalize longitude values to [-180, 180)."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def _lat_lon_to_unit_xyz(lat, lon) -> np.ndarray:
    """Convert latitude/longitude degrees to unit-sphere XYZ coordinates."""
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))
    lon_rad = np.deg2rad(_normalize_longitude_180(lon))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [
            np.ravel(cos_lat * np.cos(lon_rad)),
            np.ravel(cos_lat * np.sin(lon_rad)),
            np.ravel(np.sin(lat_rad)),
        ]
    )


def _lat_lon_grid_lookup_cache(lat_lon_grid: Any) -> dict[str, Any]:
    """Return a cached spherical KD-tree for a 2-D curved/rotated grid."""
    if isinstance(lat_lon_grid, dict):
        cache = lat_lon_grid.get("_lookup_cache")
        if cache is not None:
            return cache

    latitude = np.asarray(lat_lon_grid["latitude"], dtype=float)
    # Ensure longitudes are normalized to -180..180 matching target coords
    longitude = _normalize_longitude_180(lat_lon_grid["longitude"])

    if latitude.shape != longitude.shape:
        raise ValueError(
            f"Latitude/Longitude shape mismatch: lat={latitude.shape} lon={longitude.shape}"
        )

    cache = {
        "tree": cKDTree(_lat_lon_to_unit_xyz(latitude, longitude)),
        "shape": latitude.shape,
        "latitude": latitude,
        "longitude": longitude,
    }
    if isinstance(lat_lon_grid, dict):
        lat_lon_grid["_lookup_cache"] = cache
    return cache


def _nearest_2d_grid_coords(
    lat: float,
    lon: float,
    lat_lon_grid: Any,
    max_distance: float = 0.020,
    model_name: str = "2D",
) -> tuple[int, int, float, float]:
    """Return x/y and nearest geographic coordinates for a 2D projected or rotated grid."""
    cache = _lat_lon_grid_lookup_cache(lat_lon_grid)
    target_xyz = _lat_lon_to_unit_xyz(np.array([lat]), np.array([lon]))

    dist, flat_idx = cache["tree"].query(target_xyz, k=1)

    if float(dist[0]) > max_distance:
        raise ValueError(
            f"Location ({lat:.3f}, {lon:.3f}) is outside the {model_name} domain "
            f"(nearest grid point chord distance {float(dist[0]):.4f} > {max_distance})"
        )

    y_idx, x_idx = np.unravel_index(int(flat_idx[0]), cache["shape"])
    return (
        int(x_idx),
        int(y_idx),
        float(cache["latitude"][y_idx, x_idx]),
        float(cache["longitude"][y_idx, x_idx]),
    )


def _nearest_regular_grid_index(
    value: float,
    start: float,
    delta: float,
    count: int,
) -> int:
    """Return the nearest bounded index on a regular one-dimensional grid."""
    index = math.floor(((value - start) / delta) + 0.5)
    return max(0, min(count - 1, index))


def _silam_grid_coords(lat: float, az_lon: float) -> tuple[int, int, float, float]:
    """Return x/y and nearest gridpoint coordinates for the SILAM global grid."""
    y_silam = _nearest_regular_grid_index(
        lat,
        SILAM_LAT_START,
        SILAM_GRID_DELTA,
        SILAM_LAT_COUNT,
    )
    x_silam = _nearest_regular_grid_index(
        az_lon,
        SILAM_LON_START,
        SILAM_GRID_DELTA,
        SILAM_LON_COUNT,
    )
    silam_lat = SILAM_LAT_START + y_silam * SILAM_GRID_DELTA
    silam_lon = SILAM_LON_START + x_silam * SILAM_GRID_DELTA
    return x_silam, y_silam, silam_lat, silam_lon


@dataclass
class ZarrSources:
    subh: Any
    hrrr_6h: Any
    hrrr: Any
    nbm: Any
    nbm_fire: Any
    gfs: Any
    ecmwf: Any
    gefs: Any
    hrdps: Any = None
    gdps: Any = None
    geps: Any = None
    reps: Any = None
    rtma_ru: Any = None
    wmo_alerts: Any = None
    era5_data: Any = None
    dwd_mosmix: Any = None
    aigfs: Any = None
    aigefs: Any = None
    ecmwf_aifs: Any = None
    raqdps: Any = None
    silam: Any = None
    raqdps_lat_lon: Any = None


@dataclass
class GridIndexingResult:
    dataOut: np.ndarray | bool
    dataOut_h2: np.ndarray | bool
    dataOut_hrrrh: np.ndarray | bool
    dataOut_nbm: np.ndarray | bool
    dataOut_nbmFire: np.ndarray | bool
    dataOut_gfs: np.ndarray | bool
    dataOut_ecmwf: np.ndarray | bool
    dataOut_gefs: np.ndarray | bool
    dataOut_hrdps: np.ndarray | bool
    dataOut_gdps: np.ndarray | bool
    dataOut_geps: np.ndarray | bool
    dataOut_reps: np.ndarray | bool
    dataOut_rtma_ru: np.ndarray | bool
    dataOut_dwd_mosmix: np.ndarray | bool
    dataOut_aigfs: np.ndarray | bool
    dataOut_aigefs: np.ndarray | bool
    dataOut_aifs: np.ndarray | bool
    era5_merged: np.ndarray | bool
    subhRunTime: float | None
    hrrrhRunTime: float | None
    h2RunTime: float | None
    nbmRunTime: float | None
    nbmFireRunTime: float | None
    gfsRunTime: float | None
    ecmwfRunTime: float | None
    gefsRunTime: float | None
    hrdpsRunTime: float | None
    gdpsRunTime: float | None
    gepsRunTime: float | None
    repsRunTime: float | None
    dwdMosmixRunTime: float | None
    aigfsRunTime: float | None
    aigefsRunTime: float | None
    aifsRunTime: float | None
    x_rtma: float | None
    y_rtma: float | None
    rtma_lat: float | None
    rtma_lon: float | None
    x_nbm: float | None
    y_nbm: float | None
    nbm_lat: float | None
    nbm_lon: float | None
    x_p: float | None
    y_p: float | None
    gfs_lat: float | None
    gfs_lon: float | None
    x_p_eur: float | None
    y_p_eur: float | None
    lats_ecmwf: np.ndarray | None
    lons_ecmwf: np.ndarray | None
    x_dwd: float | None
    y_dwd: float | None
    dwd_lat: float | None
    dwd_lon: float | None
    # Canadian / ensemble grid coordinates
    x_gdps: float | None
    y_gdps: float | None
    gdps_lat: float | None
    gdps_lon: float | None
    x_geps: float | None
    y_geps: float | None
    geps_lat: float | None
    geps_lon: float | None
    sourceIDX: dict
    WMO_alertDat: str | None
    # Air quality model outputs
    dataOut_raqdps: np.ndarray | bool = False
    dataOut_silam: np.ndarray | bool = False
    raqdpsRunTime: float | None = None
    silamRunTime: float | None = None
    x_raqdps: float | None = None
    y_raqdps: float | None = None
    raqdps_lat: float | None = None
    raqdps_lon: float | None = None
    x_silam: float | None = None
    y_silam: float | None = None
    silam_lat: float | None = None
    silam_lon: float | None = None


def _load_era5_slice(era5_data, lat: float, lon: float, base_day_utc, num_hours: int):
    """Load the ERA5 point slice needed for the requested hourly grid."""
    abslat_era5 = np.abs(era5_data["ERA5_lats"] - lat)
    abslon_era5 = np.abs(era5_data["ERA5_lons"] - lon)
    y_p = np.argmin(abslat_era5)
    x_p = np.argmin(abslon_era5)
    t_p = np.argmin(
        np.abs(
            era5_data["ERA5_times"] - np.datetime64(base_day_utc.replace(tzinfo=None))
        )
    )

    precip_amount_var = "total_precipitation"
    if precip_amount_var not in era5_data["dsERA5"]:
        raise KeyError(f"Expected ERA5 precipitation variable '{precip_amount_var}'")

    dataOut_ERA5_xr = era5_data["dsERA5"][list(ERA5_SOURCE_VARS)].isel(
        latitude=y_p, longitude=x_p, time=slice(t_p, t_p + num_hours)
    )
    dataOut_ERA5 = xr.concat(
        [dataOut_ERA5_xr[var] for var in ERA5_SOURCE_VARS], dim="variable"
    )
    unix_times_era5 = (
        era5_data["ERA5_times"][t_p : t_p + num_hours].astype("datetime64[s]")
        - np.datetime64("1970-01-01T00:00:00")
    ).astype(np.int64)  # Use cached time
    era5_merged = np.vstack((unix_times_era5, dataOut_ERA5.values)).T

    n_lat = era5_data["ERA5_lats"].size
    n_lon = era5_data["ERA5_lons"].size
    y_indices = np.arange(max(y_p - 1, 0), min(y_p + 2, n_lat))
    x_indices = np.array([(x_p - 1) % n_lon, x_p, (x_p + 1) % n_lon])

    precip_window = (
        era5_data["dsERA5"][precip_amount_var]
        .isel(
            time=slice(t_p, t_p + num_hours),
            latitude=y_indices,
            longitude=x_indices,
        )
        .transpose("time", "latitude", "longitude")
        .values
    )

    # Estimate precipitation probability as the percentage of valid cells in the
    # 3x3 neighbourhood exceeding the measurable-precipitation threshold.
    # The threshold units must match total_precipitation units.
    valid = np.isfinite(precip_window)
    hits = valid & (precip_window > ERA5_PRECIP_PROB_THRESHOLD_M)
    denom = valid.sum(axis=(1, 2))
    hit_count = hits.sum(axis=(1, 2))
    precip_prob = np.divide(
        100.0 * hit_count,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )

    if precip_prob.shape[0] != era5_merged.shape[0]:
        raise ValueError(
            "ERA5 precipitation probability length does not match point slice length"
        )

    era5_merged = np.column_stack((era5_merged, precip_prob))

    # Round the precipitation_type variable to nearest integer
    # to avoid issues with interpolation producing non-integer values.
    era5_merged[:, ERA5["precipitation_type"]] = np.rint(
        era5_merged[:, ERA5["precipitation_type"]]
    )
    return era5_merged


def _era5_cache_stats(era5_data) -> dict[str, int] | None:
    cache_store = era5_data.get("ERA5_cache_store") if era5_data else None
    if cache_store is None or not hasattr(cache_store, "cache_stats"):
        return None
    return cache_store.cache_stats()


def _cache_stats_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


async def calculate_grid_indexing(
    *,
    lat: float,
    lon: float,
    az_lon: float,
    utc_time: datetime.datetime,
    now_time: datetime.datetime,
    time_machine: bool,
    ex_hrrr: int,
    ex_nbm: int,
    ex_gfs: int,
    ex_ecmwf: int,
    ex_gefs: int,
    ex_rtma_ru: int,
    ex_dwd_mosmix: int,
    ex_aigfs: int,
    ex_aigefs: int,
    ex_aifs: int,
    ex_raqdps: int = 0,
    ex_silam: int = 0,
    inc_aimodels: int = 0,
    read_wmo_alerts: bool = True,
    base_day_utc: datetime.datetime | None = None,
    num_hours: int = 0,
    zarr_sources: ZarrSources = None,
    weather=None,
    timing_start: datetime.datetime | None = None,
    timing_enabled: bool = False,
    logger: logging.Logger | None = None,
) -> GridIndexingResult:
    """Compute grid coordinates and pull the zarr slices for the request."""
    timer = StepTimer(timing_start, timing_enabled)
    sourceIDX = {}
    readRTMA_RU = False
    readNBM = False
    readGFS = False
    readECMWF = False
    readGEFS = False
    readHRDPS = False
    readGDPS = False
    readGEPS = False
    readREPS = False
    readHRRR = False
    readERA5 = False
    readDWD_MOSMIX = False
    readAIGFS = False
    readAIGEFS = False
    readAIFS = False

    def _get_grid_coords(
        lat,
        lon,
        central_lon_deg,
        central_lat_deg,
        std_parallel_deg,
        semimajor_axis,
        min_x_grid,
        min_y_grid,
        delta,
        x_min_bound,
        y_min_bound,
        x_max_bound,
        y_max_bound,
    ):
        grid_lat, grid_lon, x, y = lambertGridMatch(
            math.radians(central_lon_deg),
            math.radians(central_lat_deg),
            math.radians(std_parallel_deg),
            semimajor_axis,
            lat,
            lon,
            min_x_grid,
            min_y_grid,
            delta,
        )

        in_bounds = (
            (x >= x_min_bound)
            and (y >= y_min_bound)
            and (x <= x_max_bound)
            and (y <= y_max_bound)
        )

        return grid_lat, grid_lon, x, y, in_bounds

    if (
        az_lon < -134
        or az_lon > -61
        or lat < 21
        or lat > 53
        or ex_hrrr == 1
        or time_machine
    ):
        dataOut = False
        dataOut_hrrrh = False
        dataOut_h2 = False
    else:
        hrrr_lat, hrrr_lon, x_hrrr, y_hrrr, hrrr_in_bounds = _get_grid_coords(
            lat,
            lon,
            262.5,
            38.5,
            38.5,
            6371229,
            -2697500,
            -1587300,
            3000,
            HRRR_X_MIN,
            HRRR_Y_MIN,
            HRRR_X_MAX,
            HRRR_Y_MAX,
        )

        if not hrrr_in_bounds:
            dataOut = False
            dataOut_h2 = False
            dataOut_hrrrh = False
        else:
            readHRRR = True

        sourceIDX["hrrr"] = {}
        sourceIDX["hrrr"]["x"] = int(x_hrrr)
        sourceIDX["hrrr"]["y"] = int(y_hrrr)
        sourceIDX["hrrr"]["lat"] = round(hrrr_lat, 2)
        sourceIDX["hrrr"]["lon"] = round(((hrrr_lon + 180) % 360) - 180, 2)

    timer.log("### RTMA_RU Start ###")

    if (
        az_lon < -138.3
        or az_lon > -59
        or lat < 19.3
        or lat > 57
        or time_machine
        or ex_rtma_ru == 1
    ):
        dataOut_rtma_ru = False
        x_rtma = None
        y_rtma = None
        rtma_lat = None
        rtma_lon = None
    else:
        rtma_lat, rtma_lon, x_rtma, y_rtma, rtma_in_bounds = _get_grid_coords(
            lat,
            lon,
            RTMA_RU_CENTRAL_LONG,
            RTMA_RU_CENTRAL_LAT,
            RTMA_RU_PARALLEL,
            RTMA_RU_AXIS,
            RTMA_RU_MIN_X,
            RTMA_RU_MIN_Y,
            RTMA_RU_DELTA,
            RTMA_RU_X_MIN,
            RTMA_RU_Y_MIN,
            RTMA_RU_X_MAX,
            RTMA_RU_Y_MAX,
        )

        if not rtma_in_bounds:
            dataOut_rtma_ru = False
        else:
            readRTMA_RU = True
            dataOut_rtma_ru = None

    timer.log("### NBM Start ###")

    if (
        az_lon < -138.3
        or az_lon > -59
        or lat < 19.3
        or lat > 57
        or ex_nbm == 1
        or time_machine
    ):
        dataOut_nbm = False
        dataOut_nbmFire = False
        x_nbm = None
        y_nbm = None
        nbm_lat = None
        nbm_lon = None
    else:
        nbm_lat, nbm_lon, x_nbm, y_nbm, nbm_in_bounds = _get_grid_coords(
            lat,
            lon,
            265,
            25,
            25.0,
            6371200,
            -3271152.8,
            -263793.46,
            2539.703000,
            NBM_X_MIN,
            NBM_Y_MIN,
            NBM_X_MAX,
            NBM_Y_MAX,
        )

        if not nbm_in_bounds:
            dataOut_nbm = False
            dataOut_nbmFire = False
        else:
            timer.log("### NBM Detail Start ###")
            readNBM = True
            dataOut_nbm = None
            dataOut_nbmFire = None

    timer.log("### GFS/GEFS Start ###")

    lats_gfs = np.arange(-90, 90, 0.25)
    lons_gfs = np.arange(0, 360, 0.25)
    abslat = np.abs(lats_gfs - lat)
    abslon = np.abs(lons_gfs - lon)
    y_p = np.argmin(abslat)
    x_p = np.argmin(abslon)
    gfs_lat = lats_gfs[y_p]
    gfs_lon = lons_gfs[x_p]

    x_hrdps = None
    y_hrdps = None
    hrdps_lat = None
    hrdps_lon = None
    x_reps = None
    y_reps = None
    reps_lat = None
    reps_lon = None

    if (now_time - utc_time) > datetime.timedelta(hours=10 * 24):
        dataOut_gfs = False
        readERA5 = True
        readGFS = False
        ex_gfs = 1
    elif ex_gfs:
        dataOut_gfs = False
        readGFS = False
    else:
        readGFS = True
        dataOut_gfs = None

    timer.log("### GFS Detail END ###")

    timer.log("### ECMWF Detail Start ###")

    dataOut_ecmwf = False
    lats_ecmwf = None
    lons_ecmwf = None
    x_p_eur = None
    y_p_eur = None
    if ex_ecmwf == 1 or time_machine or zarr_sources.ecmwf is None:
        dataOut_ecmwf = False
    else:
        readECMWF = True
        lats_ecmwf = np.arange(90, -90, -0.25)
        lons_ecmwf = np.arange(-180, 180, 0.25)
        abslat_ecmwf = np.abs(lats_ecmwf - lat)
        abslon_ecmwf = np.abs(lons_ecmwf - az_lon)
        y_p_eur = np.argmin(abslat_ecmwf)
        x_p_eur = np.argmin(abslon_ecmwf)

    timer.log("### ECMWF Detail END ###")

    timer.log("### GEFS Detail Start ###")

    if ex_gefs == 1 or time_machine:
        dataOut_gefs = False
    else:
        readGEFS = True
        dataOut_gefs = None

    timer.log("### GEFS Detail END ###")

    timer.log("### Canadian Models Detail Start ###")

    dataOut_hrdps = False
    dataOut_gdps = False
    dataOut_geps = False
    dataOut_reps = False
    if not time_machine:
        if zarr_sources.hrdps is not None:
            try:
                (
                    x_hrdps,
                    y_hrdps,
                    hrdps_lat,
                    hrdps_lon,
                ) = _nearest_2d_grid_coords(
                    lat, lon, zarr_sources.hrdps, max_distance=0.005, model_name="HRDPS"
                )
                readHRDPS = True
                dataOut_hrdps = None
            except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
                logger.debug("HRDPS grid lookup failed: %s", exc)
        # GDPS is a regular lat/lon grid: 2400 x 1201 @ 0.15° resolution
        if zarr_sources.gdps is not None:
            try:
                # Build GDPS lat/lon arrays (90 -> -90, -180 -> 179.85)
                lats_gdps = np.linspace(90.0, -90.0, 1201)
                lons_gdps = np.linspace(-180.0, 180.0 - 0.15, 2400)

                # Convert input lon to 0..360 for GDPS indexing
                target_lon_360 = (az_lon + 360.0) % 360.0

                abslat = np.abs(lats_gdps - lat)
                abslon = np.abs(lons_gdps - target_lon_360)
                y_gdps = int(np.argmin(abslat))
                x_gdps = int(np.argmin(abslon))
                gdps_lat = float(lats_gdps[y_gdps])
                gdps_lon = float(lons_gdps[x_gdps])
                readGDPS = True
                dataOut_gdps = None
            except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
                logger.debug("GDPS grid lookup failed: %s", exc)
        if zarr_sources.gdps is not None:
            # If the above block didn't set coords, ensure read flags set
            readGDPS = True
            dataOut_gdps = None
        if zarr_sources.geps is not None:
            try:
                # GEPS: 720 x 361, 0.5° resolution, lon 0..359.5, lat -90..90
                lats_geps = np.linspace(-90.0, 90.0, 361)
                lons_geps = np.linspace(0.0, 360.0 - 0.5, 720)
                # Convert input lon to 0..360 for GEPS indexing
                lon360 = (az_lon + 360.0) % 360.0
                abslat = np.abs(lats_geps - lat)
                abslon = np.abs(lons_geps - lon360)
                y_geps = int(np.argmin(abslat))
                x_geps = int(np.argmin(abslon))
                geps_lat = float(lats_geps[y_geps])
                geps_lon = float(lons_geps[x_geps])
                readGEPS = True
                dataOut_geps = None
            except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
                logger.debug("GEPS grid lookup failed: %s", exc)
        if zarr_sources.reps is not None:
            try:
                (
                    x_reps,
                    y_reps,
                    reps_lat,
                    reps_lon,
                ) = _nearest_2d_grid_coords(
                    lat, lon, zarr_sources.reps, max_distance=0.020, model_name="REPS"
                )
                readREPS = True
                dataOut_reps = None
            except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
                logger.debug("REPS grid lookup failed: %s", exc)

    timer.log("### Canadian Models Detail END ###")

    timer.log("### DWD MOSMIX Detail Start ###")

    # DWD MOSMIX uses the same 0.25° GFS grid (interpolated during ingest)
    # DWD MOSMIX-S stations are located worldwide, with coverage in Europe, USA,
    # Australia, India, Brazil, Africa and other regions. Some variables like
    # solar radiation may only be available for European stations.
    dataOut_dwd_mosmix = False
    x_dwd = None
    y_dwd = None
    dwd_lat = None
    dwd_lon = None
    if ex_dwd_mosmix == 1 or time_machine or zarr_sources.dwd_mosmix is None:
        dataOut_dwd_mosmix = False
    else:
        # DWD MOSMIX is interpolated onto the GFS 0.25° grid
        # Use the same lat/lon coordinates as GFS
        readDWD_MOSMIX = True
        x_dwd = x_p
        y_dwd = y_p
        dwd_lat = gfs_lat
        dwd_lon = gfs_lon

    timer.log("### DWD MOSMIX Detail END ###")

    timer.log("### AI Models Detail Start ###")

    ai_models_requested = bool(inc_aimodels) and not time_machine
    is_na = is_in_north_america(lat, az_lon)

    if ai_models_requested and is_na:
        if ex_aigfs != 1 and zarr_sources.aigfs is not None:
            readAIGFS = True
        if ex_aigefs != 1 and zarr_sources.aigefs is not None:
            readAIGEFS = True
    elif ai_models_requested and not is_na:
        if ex_aifs != 1 and zarr_sources.ecmwf_aifs is not None:
            if x_p_eur is None or y_p_eur is None:
                lats_ecmwf = np.arange(90, -90, -0.25)
                lons_ecmwf = np.arange(-180, 180, 0.25)
                abslat_ecmwf = np.abs(lats_ecmwf - lat)
                abslon_ecmwf = np.abs(lons_ecmwf - az_lon)
                y_p_eur = np.argmin(abslat_ecmwf)
                x_p_eur = np.argmin(abslon_ecmwf)
            readAIFS = True

    timer.log("### AI Models Detail END ###")

    timer.log("### AQ Models Detail Start ###")

    readRAQDPS = False
    readSILAM = False
    x_raqdps = None
    y_raqdps = None
    raqdps_lat_val = None
    raqdps_lon_val = None
    x_silam = None
    y_silam = None
    silam_lat_val = None
    silam_lon_val = None

    # RAQDPS: Canadian regional air quality model; uses a rotated lat-lon 2D grid.
    # Available only when the lat/lon pickle has been loaded.
    if (
        ex_raqdps != 1
        and zarr_sources.raqdps is not None
        and zarr_sources.raqdps_lat_lon is not None
    ):
        try:
            (
                x_raqdps,
                y_raqdps,
                raqdps_lat_val,
                raqdps_lon_val,
            ) = _nearest_2d_grid_coords(
                lat,
                lon,
                zarr_sources.raqdps_lat_lon,
                max_distance=0.020,
                model_name="RAQDPS",
            )
            readRAQDPS = True
        except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("RAQDPS grid lookup failed: %s", exc)

    # SILAM: Global air quality model; uses a regular 0.2° lat/lon grid.
    if ex_silam != 1 and zarr_sources.silam is not None:
        try:
            x_silam, y_silam, silam_lat_val, silam_lon_val = _silam_grid_coords(
                lat,
                az_lon,
            )
            readSILAM = True
        except (IndexError, KeyError, ValueError, TypeError, AttributeError) as exc:
            logger.debug("SILAM grid lookup failed: %s", exc)

    timer.log("### AQ Models Detail END ###")

    if readERA5:
        era5_read_start = time.perf_counter()
        cache_stats_before = _era5_cache_stats(zarr_sources.era5_data)
        try:
            ERA5_MERGED = await asyncio.to_thread(
                _load_era5_slice,
                zarr_sources.era5_data,
                lat=lat,
                lon=lon,
                base_day_utc=base_day_utc,
                num_hours=num_hours,
            )
        finally:
            if timing_enabled:
                elapsed_ms = (time.perf_counter() - era5_read_start) * 1000
                cache_delta = _cache_stats_delta(
                    cache_stats_before,
                    _era5_cache_stats(zarr_sources.era5_data),
                )
                if cache_delta is None:
                    logger.info("ERA5 read: %.1f ms", elapsed_ms)
                else:
                    reads = cache_delta["hits"] + cache_delta["misses"]
                    hit_rate = 100 * cache_delta["hits"] / reads if reads else 0
                    logger.info(
                        "ERA5 read: %.1f ms cache_hits=%d cache_misses=%d "
                        "evictions=%d hit_rate=%.1f%%",
                        elapsed_ms,
                        cache_delta["hits"],
                        cache_delta["misses"],
                        cache_delta["evictions"],
                        hit_rate,
                    )

    else:
        ERA5_MERGED = False

    zarrTasks = {}
    if readHRRR:
        zarrTasks["SubH"] = weather.zarr_read("SubH", zarr_sources.subh, x_hrrr, y_hrrr)
        zarrTasks["HRRR_6H"] = weather.zarr_read(
            "HRRR_6H", zarr_sources.hrrr_6h, x_hrrr, y_hrrr
        )
        zarrTasks["HRRR"] = weather.zarr_read("HRRR", zarr_sources.hrrr, x_hrrr, y_hrrr)
    if readNBM:
        zarrTasks["NBM"] = weather.zarr_read("NBM", zarr_sources.nbm, x_nbm, y_nbm)
    if readGFS:
        zarrTasks["GFS"] = weather.zarr_read("GFS", zarr_sources.gfs, x_p, y_p)
    if readECMWF:
        zarrTasks["ECMWF"] = weather.zarr_read(
            "ECMWF", zarr_sources.ecmwf, x_p_eur, y_p_eur
        )
    if readGEFS:
        zarrTasks["GEFS"] = weather.zarr_read("GEFS", zarr_sources.gefs, x_p, y_p)
    if readHRDPS:
        zarrTasks["HRDPS"] = weather.zarr_read(
            "HRDPS", zarr_sources.hrdps, x_hrdps, y_hrdps
        )
    if readGDPS:
        zarrTasks["GDPS"] = weather.zarr_read("GDPS", zarr_sources.gdps, x_p, y_p)
    if readGEPS:
        zarrTasks["GEPS"] = weather.zarr_read("GEPS", zarr_sources.geps, x_p, y_p)
    if readREPS:
        zarrTasks["REPS"] = weather.zarr_read("REPS", zarr_sources.reps, x_reps, y_reps)
    if readRTMA_RU:
        zarrTasks["RTMA_RU"] = weather.zarr_read(
            "RTMA_RU", zarr_sources.rtma_ru, x_rtma, y_rtma
        )
    if readDWD_MOSMIX:
        zarrTasks["DWD_MOSMIX"] = weather.zarr_read(
            "DWD_MOSMIX", zarr_sources.dwd_mosmix, x_dwd, y_dwd
        )
    if readAIGFS:
        zarrTasks["AIGFS"] = weather.zarr_read("AIGFS", zarr_sources.aigfs, x_p, y_p)
    if readAIGEFS:
        zarrTasks["AIGEFS"] = weather.zarr_read("AIGEFS", zarr_sources.aigefs, x_p, y_p)
    if readAIFS:
        zarrTasks["ECMWF_AIFS"] = weather.zarr_read(
            "ECMWF_AIFS", zarr_sources.ecmwf_aifs, x_p_eur, y_p_eur
        )
    if readRAQDPS:
        zarrTasks["RAQDPS"] = weather.zarr_read_max_square(
            "RAQDPS", zarr_sources.raqdps, x_raqdps, y_raqdps
        )
    if readSILAM:
        zarrTasks["SILAM"] = weather.zarr_read_max_square(
            "SILAM", zarr_sources.silam, x_silam, y_silam
        )

    WMO_alertDat = None
    if read_wmo_alerts:
        wmo_alerts_lats = np.arange(-60, 85, 0.0625)
        wmo_alerts_lons = np.arange(-180, 180, 0.0625)
        wmo_abslat = np.abs(wmo_alerts_lats - lat)
        wmo_abslon = np.abs(wmo_alerts_lons - az_lon)
        wmo_alerts_y_p = np.argmin(wmo_abslat)
        wmo_alerts_x_p = np.argmin(wmo_abslon)
        WMO_alertDat = zarr_sources.wmo_alerts[wmo_alerts_y_p, wmo_alerts_x_p]
        if timing_enabled:
            print(WMO_alertDat)

    results = await asyncio.gather(*zarrTasks.values())
    zarr_results = {key: result for key, result in zip(zarrTasks.keys(), results)}

    subhRunTime = None
    hrrrhRunTime = None
    h2RunTime = None
    nbmRunTime = None
    nbmFireRunTime = None
    gfsRunTime = None
    ecmwfRunTime = None
    gefsRunTime = None
    hrdpsRunTime = None
    gdpsRunTime = None
    gepsRunTime = None
    repsRunTime = None
    dwdMosmixRunTime = None
    aigfsRunTime = None
    aigefsRunTime = None
    aifsRunTime = None

    if readHRRR:
        dataOut = zarr_results["SubH"]
        dataOut_h2 = zarr_results["HRRR_6H"]
        dataOut_hrrrh = zarr_results["HRRR"]
        if (
            (dataOut is not False)
            and (dataOut_h2 is not False)
            and (dataOut_hrrrh is not False)
        ):
            subhRunTime = dataOut[0, 0]
            if (
                utc_time
                - datetime.datetime.fromtimestamp(
                    subhRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=4):
                dataOut = False
            hrrrhRunTime = dataOut_hrrrh[HISTORY_PERIODS["HRRR"], 0]
            if (
                utc_time
                - datetime.datetime.fromtimestamp(
                    hrrrhRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=16):
                dataOut_hrrrh = False
            h2RunTime = dataOut_h2[0, 0]
            if (
                utc_time
                - datetime.datetime.fromtimestamp(
                    h2RunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=46):
                dataOut_h2 = False
        else:
            dataOut = False
            dataOut_h2 = False
            dataOut_hrrrh = False

    if readNBM:
        dataOut_nbm = zarr_results["NBM"]
        dataOut_nbmFire = False
        if dataOut_nbm is not False:
            nbmRunTime = dataOut_nbm[HISTORY_PERIODS["NBM"], 0]
            try:
                timestamp_dt = datetime.datetime.fromtimestamp(
                    nbmRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                # Exclude hourly NBM if older than 2 days
                if (utc_time - timestamp_dt) > datetime.timedelta(days=2):
                    dataOut_nbm = False
                    nbmRunTime = None
                    logger.warning("OLD NBM")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse NBM runtime for freshness check")

        if dataOut_nbm is not False:
            sourceIDX["nbm"] = {}
            sourceIDX["nbm"]["x"] = int(x_nbm)
            sourceIDX["nbm"]["y"] = int(y_nbm)
            sourceIDX["nbm"]["lat"] = round(nbm_lat, 2)
            sourceIDX["nbm"]["lon"] = round(((nbm_lon + 180) % 360) - 180, 2)

    if readGFS:
        dataOut_gfs = zarr_results["GFS"]
        if dataOut_gfs is not False:
            gfsRunTime = dataOut_gfs[HISTORY_PERIODS["GFS"] - 1, 0]
            try:
                timestamp_dt = datetime.datetime.fromtimestamp(
                    gfsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                # Exclude 6-hourly GFS if older than 5 days
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_gfs = False
                    gfsRunTime = None
                    logger.warning("OLD GFS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse GFS runtime for freshness check")

    if readECMWF:
        dataOut_ecmwf = zarr_results["ECMWF"]
        if dataOut_ecmwf is not False:
            ecmwfRunTime = dataOut_ecmwf[HISTORY_PERIODS["ECMWF"] - 3, 0]
            try:
                timestamp_dt = datetime.datetime.fromtimestamp(
                    ecmwfRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                # Exclude 12-hourly ECMWF if older than 5 days
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_ecmwf = False
                    ecmwfRunTime = None
                    logger.warning("OLD ECMWF")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse ECMWF runtime for freshness check")

        if dataOut_ecmwf is not False:
            sourceIDX["ecmwf_ifs"] = {}
            sourceIDX["ecmwf_ifs"]["x"] = int(x_p_eur)
            sourceIDX["ecmwf_ifs"]["y"] = int(y_p_eur)
            sourceIDX["ecmwf_ifs"]["lat"] = round(lats_ecmwf[y_p_eur], 2)
            sourceIDX["ecmwf_ifs"]["lon"] = round(lons_ecmwf[x_p_eur], 2)

    if readGEFS:
        dataOut_gefs = zarr_results["GEFS"]
        if dataOut_gefs is not False:
            try:
                gefsRunTime = dataOut_gefs[HISTORY_PERIODS["GEFS"] - 3, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    gefsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                # Exclude 6-hourly GEFS if older than 5 days
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_gefs = False
                    gefsRunTime = None
                    logger.warning("OLD GEFS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse GEFS runtime for freshness check")
        else:
            gefsRunTime = None

    if readHRDPS:
        dataOut_hrdps = zarr_results["HRDPS"]
        if dataOut_hrdps is not False:
            try:
                hrdpsRunTime = dataOut_hrdps[HISTORY_PERIODS["HRDPS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    hrdpsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_hrdps = False
                    hrdpsRunTime = None
                    logger.warning("OLD HRDPS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse HRDPS runtime for freshness check")
            else:
                sourceIDX["hrdps"] = {}
                sourceIDX["hrdps"]["x"] = int(x_hrdps)
                sourceIDX["hrdps"]["y"] = int(y_hrdps)
                sourceIDX["hrdps"]["lat"] = round(hrdps_lat, 2)
                sourceIDX["hrdps"]["lon"] = round(((hrdps_lon + 180) % 360) - 180, 2)
    else:
        dataOut_hrdps = False

    if readGDPS:
        dataOut_gdps = zarr_results["GDPS"]
        if dataOut_gdps is not False:
            try:
                gdpsRunTime = dataOut_gdps[HISTORY_PERIODS["GDPS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    gdpsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_gdps = False
                    gdpsRunTime = None
                    logger.warning("OLD GDPS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse GDPS runtime for freshness check")
            else:
                sourceIDX["gdps"] = {}
                sourceIDX["gdps"]["x"] = int(x_gdps)
                sourceIDX["gdps"]["y"] = int(y_gdps)
                sourceIDX["gdps"]["lat"] = round(gdps_lat, 2)
                sourceIDX["gdps"]["lon"] = round(gdps_lon, 2)
    else:
        dataOut_gdps = False

    if readGEPS:
        dataOut_geps = zarr_results["GEPS"]
        if dataOut_geps is not False:
            try:
                gepsRunTime = dataOut_geps[HISTORY_PERIODS["GEPS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    gepsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_geps = False
                    gepsRunTime = None
                    logger.warning("OLD GEPS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse GEPS runtime for freshness check")
            else:
                sourceIDX["geps"] = {}
                sourceIDX["geps"]["x"] = int(x_geps)
                sourceIDX["geps"]["y"] = int(y_geps)
                sourceIDX["geps"]["lat"] = round(geps_lat, 2)
                sourceIDX["geps"]["lon"] = round(geps_lon, 2)
    else:
        dataOut_geps = False

    if readREPS:
        dataOut_reps = zarr_results["REPS"]
        if dataOut_reps is not False:
            try:
                repsRunTime = dataOut_reps[HISTORY_PERIODS["REPS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    repsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_reps = False
                    repsRunTime = None
                    logger.warning("OLD REPS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse REPS runtime for freshness check")
            else:
                sourceIDX["reps"] = {}
                sourceIDX["reps"]["x"] = int(x_reps)
                sourceIDX["reps"]["y"] = int(y_reps)
                sourceIDX["reps"]["lat"] = round(reps_lat, 2)
                sourceIDX["reps"]["lon"] = round(((reps_lon + 180) % 360) - 180, 2)
    else:
        dataOut_reps = False

    if readRTMA_RU:
        dataOut_rtma_ru = zarr_results["RTMA_RU"]
        if dataOut_rtma_ru is not False:
            rtma_ru_time = dataOut_rtma_ru[0, 0]
            if (
                utc_time
                - datetime.datetime.fromtimestamp(
                    rtma_ru_time.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=1):
                dataOut_rtma_ru = False
                logger.warning("OLD RTMA_RU")
    else:
        dataOut_rtma_ru = False

    if readDWD_MOSMIX:
        dataOut_dwd_mosmix = zarr_results["DWD_MOSMIX"]
        if dataOut_dwd_mosmix is not False:
            # Check if the data point has any valid (non-NaN) data
            # DWD zarr files are mostly empty, so we need to verify actual data exists
            if np.all(np.isnan(dataOut_dwd_mosmix[:, 1:])):
                # All data is NaN, treat as no data available
                dataOut_dwd_mosmix = False
            elif len(dataOut_dwd_mosmix) > HISTORY_PERIODS["DWD_MOSMIX"]:
                # Bounds check before accessing the specific index
                # Negative 1 is because the 19Z forecast contains data starting at hour 1
                dwdMosmixRunTime = dataOut_dwd_mosmix[
                    HISTORY_PERIODS["DWD_MOSMIX"] - 1, 0
                ]

                # Validate the timestamp is valid (not 0, NaN, or unreasonably old/future)
                # A timestamp of 0 results in "1970-01-01 00Z" which indicates missing data
                # Note: DWD MOSMIX may show timestamps up to 48 hours in the future when
                # historical data is unavailable (uses HISTORY_PERIODS offset on forecast-only data)
                if np.isnan(dwdMosmixRunTime) or dwdMosmixRunTime <= 0:
                    # Invalid timestamp (NaN or zero), treat as no data available
                    logger.debug(
                        f"DWD MOSMIX timestamp invalid (NaN or zero): {dwdMosmixRunTime}"
                    )
                    dataOut_dwd_mosmix = False
                    dwdMosmixRunTime = None
                else:
                    timestamp_dt = datetime.datetime.fromtimestamp(
                        dwdMosmixRunTime.astype(int), datetime.UTC
                    ).replace(tzinfo=None)
                    time_diff = utc_time - timestamp_dt

                    if (
                        time_diff > datetime.timedelta(days=7)  # Too old
                        or time_diff
                        < datetime.timedelta(hours=-72)  # Allow up to 72h future
                    ):
                        # Invalid timestamp, treat as no data available
                        logger.debug(
                            f"DWD MOSMIX timestamp invalid (too old/future): "
                            f"{dwdMosmixRunTime} ({timestamp_dt}), "
                            f"time_diff={time_diff}"
                        )
                        dataOut_dwd_mosmix = False
                        dwdMosmixRunTime = None
                    else:
                        sourceIDX["dwd_mosmix"] = {}
                        sourceIDX["dwd_mosmix"]["x"] = int(x_dwd)
                        sourceIDX["dwd_mosmix"]["y"] = int(y_dwd)
                        sourceIDX["dwd_mosmix"]["lat"] = round(dwd_lat, 2)
                        sourceIDX["dwd_mosmix"]["lon"] = round(
                            ((dwd_lon + 180) % 360) - 180, 2
                        )
            else:
                # Data array too short, treat as no data available
                dataOut_dwd_mosmix = False

    if readAIGFS:
        dataOut_aigfs = zarr_results["AIGFS"]
        if dataOut_aigfs is not False:
            try:
                aigfsRunTime = dataOut_aigfs[HISTORY_PERIODS["AIGFS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    aigfsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_aigfs = False
                    aigfsRunTime = None
                    logger.warning("OLD AIGFS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse AIGFS runtime for freshness check")
    else:
        dataOut_aigfs = False

    if readAIGEFS:
        dataOut_aigefs = zarr_results["AIGEFS"]
        if dataOut_aigefs is not False:
            try:
                aigefsRunTime = dataOut_aigefs[HISTORY_PERIODS["AIGEFS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    aigefsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_aigefs = False
                    aigefsRunTime = None
                    logger.warning("OLD AIGEFS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse AIGEFS runtime for freshness check")
    else:
        dataOut_aigefs = False

    if readAIFS:
        dataOut_aifs = zarr_results["ECMWF_AIFS"]
        if dataOut_aifs is not False:
            try:
                aifsRunTime = dataOut_aifs[HISTORY_PERIODS["ECMWF_AIFS"] - 1, 0]
                timestamp_dt = datetime.datetime.fromtimestamp(
                    aifsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_aifs = False
                    aifsRunTime = None
                    logger.warning("OLD ECMWF_AIFS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse ECMWF_AIFS runtime for freshness check")
    else:
        dataOut_aifs = False

    # --- AQ model results ---
    dataOut_raqdps = False
    dataOut_silam = False
    raqdpsRunTime = None
    silamRunTime = None

    if "RAQDPS" in zarr_results:
        dataOut_raqdps = zarr_results["RAQDPS"]
        if isinstance(dataOut_raqdps, np.ndarray):
            try:
                raqdpsRunTime = float(dataOut_raqdps[HISTORY_PERIODS["RAQDPS"], 0])
                timestamp_dt = datetime.datetime.fromtimestamp(
                    int(raqdpsRunTime), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_raqdps = False
                    raqdpsRunTime = None
                    logger.warning("OLD RAQDPS")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse RAQDPS runtime for freshness check")

    if "SILAM" in zarr_results:
        dataOut_silam = zarr_results["SILAM"]
        if isinstance(dataOut_silam, np.ndarray):
            try:
                silamRunTime = float(dataOut_silam[HISTORY_PERIODS["SILAM"] - 1, 0])
                timestamp_dt = datetime.datetime.fromtimestamp(
                    int(silamRunTime), datetime.UTC
                ).replace(tzinfo=None)
                if (utc_time - timestamp_dt) > datetime.timedelta(days=5):
                    dataOut_silam = False
                    silamRunTime = None
                    logger.warning("OLD SILAM")
            except (ValueError, TypeError, AttributeError):
                logger.debug("Failed to parse SILAM runtime for freshness check")

    return GridIndexingResult(
        dataOut=dataOut,
        dataOut_h2=dataOut_h2,
        dataOut_hrrrh=dataOut_hrrrh,
        dataOut_nbm=dataOut_nbm,
        dataOut_nbmFire=dataOut_nbmFire,
        dataOut_gfs=dataOut_gfs,
        dataOut_ecmwf=dataOut_ecmwf,
        dataOut_gefs=dataOut_gefs,
        dataOut_hrdps=dataOut_hrdps,
        dataOut_gdps=dataOut_gdps,
        dataOut_geps=dataOut_geps,
        dataOut_reps=dataOut_reps,
        dataOut_rtma_ru=dataOut_rtma_ru,
        dataOut_dwd_mosmix=dataOut_dwd_mosmix,
        dataOut_aigfs=dataOut_aigfs,
        dataOut_aigefs=dataOut_aigefs,
        dataOut_aifs=dataOut_aifs,
        era5_merged=ERA5_MERGED,
        subhRunTime=subhRunTime,
        hrrrhRunTime=hrrrhRunTime,
        h2RunTime=h2RunTime,
        nbmRunTime=nbmRunTime,
        nbmFireRunTime=nbmFireRunTime,
        gfsRunTime=gfsRunTime,
        ecmwfRunTime=ecmwfRunTime,
        gefsRunTime=gefsRunTime,
        hrdpsRunTime=hrdpsRunTime,
        gdpsRunTime=gdpsRunTime,
        gepsRunTime=gepsRunTime,
        repsRunTime=repsRunTime,
        dwdMosmixRunTime=dwdMosmixRunTime,
        aigfsRunTime=aigfsRunTime,
        aigefsRunTime=aigefsRunTime,
        aifsRunTime=aifsRunTime,
        x_rtma=x_rtma,
        y_rtma=y_rtma,
        rtma_lat=rtma_lat,
        rtma_lon=rtma_lon,
        x_nbm=x_nbm,
        y_nbm=y_nbm,
        nbm_lat=nbm_lat,
        nbm_lon=nbm_lon,
        x_p=x_p,
        y_p=y_p,
        gfs_lat=gfs_lat,
        gfs_lon=gfs_lon,
        x_p_eur=x_p_eur,
        y_p_eur=y_p_eur,
        lats_ecmwf=lats_ecmwf,
        lons_ecmwf=lons_ecmwf,
        x_dwd=x_dwd,
        y_dwd=y_dwd,
        dwd_lat=dwd_lat,
        dwd_lon=dwd_lon,
        x_gdps=locals().get("x_gdps", None),
        y_gdps=locals().get("y_gdps", None),
        gdps_lat=locals().get("gdps_lat", None),
        gdps_lon=locals().get("gdps_lon", None),
        x_geps=locals().get("x_geps", None),
        y_geps=locals().get("y_geps", None),
        geps_lat=locals().get("geps_lat", None),
        geps_lon=locals().get("geps_lon", None),
        sourceIDX=sourceIDX,
        WMO_alertDat=WMO_alertDat,
        dataOut_raqdps=dataOut_raqdps,
        dataOut_silam=dataOut_silam,
        raqdpsRunTime=raqdpsRunTime,
        silamRunTime=silamRunTime,
        x_raqdps=x_raqdps,
        y_raqdps=y_raqdps,
        raqdps_lat=raqdps_lat_val,
        raqdps_lon=raqdps_lon_val,
        x_silam=x_silam,
        y_silam=y_silam,
        silam_lat=silam_lat_val,
        silam_lon=silam_lon_val,
    )
