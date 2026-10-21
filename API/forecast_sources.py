import datetime
import logging
from dataclasses import dataclass

import numpy as np

from API.constants.model_const import (
    AIGEFS,
    AIGFS,
    DWD_MOSMIX,
    ECMWF,
    ECMWF_AIFS,
    GDPS,
    GEFS,
    GEPS,
    GFS,
    HRDPS,
    REPS,
    URMA,
)
from API.constants.shared_const import MISSING_DATA
from API.request.grid_indexing import GridIndexingResult
from API.utils.geo import rounder

URMA_TIME_TOLERANCE_SECONDS = 300


@dataclass
class SourceMetadata:
    source_list: list[str]
    source_times: dict[str, str]
    source_idx: dict[str, dict]

    def add(
        self,
        source: str,
        *,
        time_value: str | None = None,
        time_key: str | None = None,
    ) -> None:
        if source not in self.source_list:
            self.source_list.append(source)
        if time_value is not None:
            self.source_times[time_key or source] = time_value

    def drop(self, source: str, *, time_key: str | None = None) -> None:
        if source in self.source_list:
            self.source_list.remove(source)
        self.source_times.pop(time_key or source, None)

    def has_sources(self, *sources: str) -> bool:
        return all(src in self.source_list for src in sources)


@dataclass
class MergeResult:
    hrrr: np.ndarray | None
    nbm: np.ndarray | None
    gfs: np.ndarray | None
    ecmwf: np.ndarray | None
    gefs: np.ndarray | None
    hrdps: np.ndarray | None
    gdps: np.ndarray | None
    geps: np.ndarray | None
    reps: np.ndarray | None
    dwd_mosmix: np.ndarray | None
    aigfs: np.ndarray | None
    aigefs: np.ndarray | None
    aifs: np.ndarray | None
    metadata: SourceMetadata
    urma: np.ndarray | None = None


def nearest_index(a, v) -> int:
    """Find the nearest index in array to value using binary search."""
    idx = np.searchsorted(a, v)
    idx = np.clip(idx, 1, len(a) - 1)
    left, right = a[idx - 1], a[idx]
    return idx if abs(right - v) < abs(v - left) else idx - 1


def _aligned_rows(
    times: np.ndarray, base_timestamp: float, num_hours: int
) -> tuple[int, int, int]:
    """Return ``(dest_start, src_start, rows)`` placing source rows at their valid hour.

    Stores that keep a short history (GFS, ECMWF) start before local midnight
    and are sliced from the row nearest midnight. Forecast-only stores start after
    it, so their first row is placed at its own hour and the earlier hours stay
    missing for the next source in the per-variable priority order to fill.
    """
    first_time = times[0]
    if np.isfinite(first_time) and first_time > base_timestamp:
        dest_start = round((first_time - base_timestamp) / 3600)
        src_start = 0
    else:
        dest_start = 0
        src_start = nearest_index(times, base_timestamp)
    rows = max(0, min(len(times) - src_start, num_hours - dest_start))
    return dest_start, src_start, rows


def _format_run_time(
    run_time: float | np.generic | None,
    *,
    offset_hours: int = 0,
    round_to: int | None = None,
    fmt: str = "%Y-%m-%d %HZ",
) -> str | None:
    if run_time is None:
        return None

    runtime = datetime.datetime.fromtimestamp(int(run_time), datetime.UTC).replace(
        tzinfo=None
    )
    if offset_hours:
        runtime -= datetime.timedelta(hours=offset_hours)
    rounded = rounder(runtime, to=round_to) if round_to else rounder(runtime)
    return rounded.strftime(fmt)


def _format_rtma_time(data_out_rtma: np.ndarray) -> str | None:
    if not isinstance(data_out_rtma, np.ndarray):
        return None

    rtma_timestamp = datetime.datetime.fromtimestamp(
        int(data_out_rtma[0, 0]), datetime.UTC
    ).replace(tzinfo=None)
    rounded_rtma_time = rounder(rtma_timestamp, to=15)
    return rounded_rtma_time.strftime("%Y-%m-%d %H:%MZ")


def build_source_metadata(
    *,
    grid_result: GridIndexingResult,
    era5_merged: np.ndarray | bool,
    use_etopo: bool,
    time_machine: bool,
) -> SourceMetadata:
    """Construct the source list, times, and indexes from grid results."""
    metadata = SourceMetadata([], {}, dict(grid_result.sourceIDX))

    if use_etopo:
        metadata.add("ETOPO1")

    if isinstance(era5_merged, np.ndarray):
        metadata.add("era5")

    if isinstance(grid_result.dataOut, np.ndarray):
        metadata.add(
            "hrrrsubh",
            time_value=_format_run_time(grid_result.subhRunTime),
            time_key="hrrr_subh",
        )

    if isinstance(grid_result.dataOut_rtma_ru, np.ndarray) and not time_machine:
        metadata.add(
            "rtma_ru", time_value=_format_rtma_time(grid_result.dataOut_rtma_ru)
        )
        metadata.source_idx["rtma_ru"] = {
            "x": int(grid_result.x_rtma),
            "y": int(grid_result.y_rtma),
            "lat": round(float(grid_result.rtma_lat), 2),
            "lon": round(((float(grid_result.rtma_lon) + 180) % 360) - 180, 2),
        }

    if isinstance(grid_result.dataOut_hrrrh, np.ndarray):
        if not time_machine:
            metadata.add(
                "hrrr_0-18",
                time_value=_format_run_time(grid_result.hrrrhRunTime),
            )
        else:
            metadata.add("hrrr")

    if isinstance(grid_result.dataOut_nbm, np.ndarray):
        if not time_machine:
            metadata.add("nbm", time_value=_format_run_time(grid_result.nbmRunTime))
        else:
            metadata.add("nbm")

    if isinstance(grid_result.dataOut_dwd_mosmix, np.ndarray) and not time_machine:
        metadata.add(
            "dwd_mosmix", time_value=_format_run_time(grid_result.dwdMosmixRunTime)
        )

    if isinstance(grid_result.dataOut_ecmwf, np.ndarray) and not time_machine:
        metadata.add("ecmwf_ifs", time_value=_format_run_time(grid_result.ecmwfRunTime))

    if isinstance(grid_result.dataOut_h2, np.ndarray):
        metadata.add(
            "hrrr_18-48",
            time_value=_format_run_time(
                grid_result.h2RunTime, offset_hours=18, round_to=None
            ),
        )

    if isinstance(grid_result.dataOut_gfs, np.ndarray):
        metadata.add("gfs", time_value=_format_run_time(grid_result.gfsRunTime))
        metadata.source_idx["gfs"] = {
            "x": int(grid_result.x_p),
            "y": int(grid_result.y_p),
            "lat": round(float(grid_result.gfs_lat), 2),
            "lon": round(((float(grid_result.gfs_lon) + 180) % 360) - 180, 2),
        }

    if isinstance(grid_result.dataOut_gefs, np.ndarray):
        metadata.add("gefs", time_value=_format_run_time(grid_result.gefsRunTime))

    if isinstance(grid_result.dataOut_hrdps, np.ndarray):
        metadata.add("hrdps", time_value=_format_run_time(grid_result.hrdpsRunTime))

    if isinstance(grid_result.dataOut_gdps, np.ndarray):
        metadata.add("gdps", time_value=_format_run_time(grid_result.gdpsRunTime))

    if isinstance(grid_result.dataOut_geps, np.ndarray):
        metadata.add("geps", time_value=_format_run_time(grid_result.gepsRunTime))

    if isinstance(grid_result.dataOut_reps, np.ndarray):
        metadata.add("reps", time_value=_format_run_time(grid_result.repsRunTime))

    if isinstance(grid_result.dataOut_aigfs, np.ndarray):
        metadata.add("aigfs", time_value=_format_run_time(grid_result.aigfsRunTime))

    if isinstance(grid_result.dataOut_aigefs, np.ndarray):
        metadata.add("aigefs", time_value=_format_run_time(grid_result.aigefsRunTime))

    if isinstance(grid_result.dataOut_aifs, np.ndarray):
        metadata.add("ecmwf_aifs", time_value=_format_run_time(grid_result.aifsRunTime))

    if isinstance(grid_result.dataOut_raqdps, np.ndarray):
        metadata.add("raqdps", time_value=_format_run_time(grid_result.raqdpsRunTime))
        if grid_result.x_raqdps is not None and grid_result.y_raqdps is not None:
            metadata.source_idx["raqdps"] = {
                "x": int(grid_result.x_raqdps),
                "y": int(grid_result.y_raqdps),
                "lat": round(float(grid_result.raqdps_lat), 2)
                if grid_result.raqdps_lat is not None
                else None,
                "lon": round(float(grid_result.raqdps_lon), 2)
                if grid_result.raqdps_lon is not None
                else None,
            }

    if isinstance(grid_result.dataOut_silam, np.ndarray):
        metadata.add("silam", time_value=_format_run_time(grid_result.silamRunTime))
        if grid_result.x_silam is not None and grid_result.y_silam is not None:
            metadata.source_idx["silam"] = {
                "x": int(grid_result.x_silam),
                "y": int(grid_result.y_silam),
                "lat": round(float(grid_result.silam_lat), 2)
                if grid_result.silam_lat is not None
                else None,
                "lon": round(float(grid_result.silam_lon), 2)
                if grid_result.silam_lon is not None
                else None,
            }

    return metadata


def _merge_aigfs_as_gfs(
    data_aigfs: np.ndarray, base_timestamp: float, num_hours: int
) -> np.ndarray:
    merged = np.full((num_hours, max(GFS.values()) + 1), MISSING_DATA)
    dest_start, src_start, rows = _aligned_rows(
        data_aigfs[:, 0], base_timestamp, num_hours
    )
    src = data_aigfs[src_start : src_start + rows, :]
    out = merged[dest_start : dest_start + rows]
    out[:, 0] = src[:, 0]
    out[:, GFS["pressure"]] = src[:, AIGFS["pressure"]]
    out[:, GFS["temp"]] = src[:, AIGFS["temp"]]
    out[:, GFS["wind_u"]] = src[:, AIGFS["wind_u"]]
    out[:, GFS["wind_v"]] = src[:, AIGFS["wind_v"]]
    out[:, GFS["accum"]] = src[:, AIGFS["accum"]]
    out[:, GFS["intensity"]] = src[:, AIGFS["accum"]]
    return merged


def _merge_aigefs_as_gefs(
    data_aigefs: np.ndarray, base_timestamp: float, num_hours: int
) -> np.ndarray:
    merged = np.full((num_hours, max(GEFS.values()) + 1), MISSING_DATA)
    dest_start, src_start, rows = _aligned_rows(
        data_aigefs[:, 0], base_timestamp, num_hours
    )
    src = data_aigefs[src_start : src_start + rows, :]
    out = merged[dest_start : dest_start + rows]
    out[:, 0] = src[:, 0]
    out[:, GEFS["prob"]] = src[:, AIGEFS["prob"]]
    out[:, GEFS["accum"]] = src[:, AIGEFS["accum"]]
    out[:, GEFS["error"]] = src[:, AIGEFS["error"]]
    return merged


def _merge_aifs_as_ecmwf(
    data_aifs: np.ndarray, base_timestamp: float, num_hours: int
) -> np.ndarray:
    merged = np.full((num_hours, max(ECMWF.values()) + 1), MISSING_DATA)
    dest_start, src_start, rows = _aligned_rows(
        data_aifs[:, 0], base_timestamp, num_hours
    )
    src = data_aifs[src_start : src_start + rows, :]
    out = merged[dest_start : dest_start + rows]
    out[:, 0] = src[:, 0]
    out[:, ECMWF["pressure"]] = src[:, ECMWF_AIFS["pressure"]]
    out[:, ECMWF["temp"]] = src[:, ECMWF_AIFS["temp"]]
    out[:, ECMWF["dew"]] = src[:, ECMWF_AIFS["dew"]]
    out[:, ECMWF["wind_u"]] = src[:, ECMWF_AIFS["wind_u"]]
    out[:, ECMWF["wind_v"]] = src[:, ECMWF_AIFS["wind_v"]]
    out[:, ECMWF["ptype"]] = src[:, ECMWF_AIFS["ptype"]]
    out[:, ECMWF["cloud"]] = src[:, ECMWF_AIFS["cloud"]]
    out[:, ECMWF["prob"]] = src[:, ECMWF_AIFS["prob"]]
    out[:, ECMWF["accum_mean"]] = src[:, ECMWF_AIFS["accum_mean"]]
    out[:, ECMWF["accum_stddev"]] = src[:, ECMWF_AIFS["accum_stddev"]]
    out[:, ECMWF["accum"]] = src[:, ECMWF_AIFS["accum_mean"]]
    out[:, ECMWF["intensity"]] = src[:, ECMWF_AIFS["accum_mean"]]
    return merged


def add_etopo_source(
    metadata: SourceMetadata,
    *,
    x_idx: int,
    y_idx: int,
    lat_val: float,
    lon_val: float,
) -> None:
    metadata.source_idx["etopo"] = {
        "x": int(x_idx),
        "y": int(y_idx),
        "lat": round(float(lat_val), 4),
        "lon": round(float(lon_val), 4),
    }


def _merge_simple_source(
    data: np.ndarray,
    base_timestamp: float,
    num_hours: int,
    target_columns: int,
    *,
    source_columns: int | None = None,
) -> np.ndarray:
    merged = np.full((num_hours, target_columns), MISSING_DATA)
    dest_start, src_start, rows = _aligned_rows(data[:, 0], base_timestamp, num_hours)
    copy_columns = source_columns or target_columns
    merged[dest_start : dest_start + rows, 0:copy_columns] = data[
        src_start : src_start + rows, 0:copy_columns
    ]
    return merged


def _merge_hrrr_blocks(
    data_hrrrh: np.ndarray,
    data_h2: np.ndarray,
    base_timestamp: float,
    num_hours: int,
) -> np.ndarray:
    # The HRRR 18-48 h store continues where the hourly 0-18 h store ends.
    h2_start_idx = nearest_index(data_h2[:, 0], data_hrrrh[-1, 0]) + 1
    common_cols = min(data_hrrrh.shape[1], data_h2.shape[1])
    hrrr_combined = np.concatenate(
        (data_hrrrh[:, 0:common_cols], data_h2[h2_start_idx:, 0:common_cols]),
        axis=0,
    )
    return _merge_simple_source(
        hrrr_combined,
        base_timestamp,
        num_hours,
        data_h2.shape[1],
        source_columns=common_cols,
    )


def _merge_urma_source(
    data: np.ndarray, base_timestamp: float, num_hours: int, min_timestamp: float
) -> tuple[np.ndarray | None, float | None]:
    """Align hourly analyses by their valid time without filling missing observations."""
    merged = np.full((num_hours, max(URMA.values()) + 1), MISSING_DATA)
    latest_time = None
    for row in data:
        timestamp = float(row[0])
        if not np.isfinite(timestamp):
            continue
        hour = round((timestamp - base_timestamp) / 3600)
        aligned_time = base_timestamp + hour * 3600
        if (
            not 0 <= hour < num_hours
            or abs(timestamp - aligned_time) > URMA_TIME_TOLERANCE_SECONDS
            or aligned_time < min_timestamp
        ):
            continue
        if not np.isfinite(row[1:]).any():
            continue
        merged[hour] = row[: merged.shape[1]]
        merged[hour, 0] = aligned_time
        latest_time = max(latest_time or aligned_time, aligned_time)
    return (merged, latest_time) if latest_time is not None else (None, None)


def merge_hourly_models(
    *,
    metadata: SourceMetadata,
    num_hours: int,
    base_day_utc_grib,
    data_hrrrh: np.ndarray | None,
    data_h2: np.ndarray | None,
    data_nbm: np.ndarray | None,
    data_gfs: np.ndarray | None,
    data_ecmwf: np.ndarray | None,
    data_gefs: np.ndarray | None,
    data_hrdps: np.ndarray | None,
    data_gdps: np.ndarray | None,
    data_geps: np.ndarray | None,
    data_reps: np.ndarray | None,
    data_dwd_mosmix: np.ndarray | None,
    data_aigfs: np.ndarray | None,
    data_aigefs: np.ndarray | None,
    data_aifs: np.ndarray | None,
    logger: logging.Logger,
    loc_tag: str,
    data_urma: np.ndarray | None = None,
    urma_min_timestamp: float | None = None,
) -> MergeResult:
    hrrr_merged = None
    nbm_merged = None
    gfs_merged = None
    ecmwf_merged = None
    gefs_merged = None
    hrdps_merged = None
    gdps_merged = None
    geps_merged = None
    reps_merged = None
    dwd_mosmix_merged = None
    aigfs_merged = None
    aigefs_merged = None
    aifs_merged = None
    urma_merged = None

    if isinstance(data_urma, np.ndarray) and urma_min_timestamp is not None:
        urma_merged, latest_urma_time = _merge_urma_source(
            data_urma, base_day_utc_grib, num_hours, urma_min_timestamp
        )
        if urma_merged is not None:
            metadata.add("urma", time_value=_format_run_time(latest_urma_time))

    try:
        if (
            metadata.has_sources("hrrr_0-18", "hrrr_18-48")
            and isinstance(data_hrrrh, np.ndarray)
            and isinstance(data_h2, np.ndarray)
        ):
            hrrr_merged = _merge_hrrr_blocks(
                data_hrrrh, data_h2, base_day_utc_grib, num_hours
            )
            if not np.isfinite(hrrr_merged[:, 0]).any():
                hrrr_merged = None
                metadata.drop("hrrr_18-48")
                metadata.drop("hrrr_0-18")
                logger.error("HRRR data not available for the requested time range.")

        if "nbm" in metadata.source_list and isinstance(data_nbm, np.ndarray):
            nbm_merged = _merge_simple_source(
                data_nbm, base_day_utc_grib, num_hours, data_nbm.shape[1]
            )
            if not np.isfinite(nbm_merged[:, 0]).any():
                nbm_merged = None
                metadata.drop("nbm")
                logger.error("NBM data not available for the requested time range.")

    except Exception:
        logger.exception(
            "HRRR or NBM data not available, falling back to GFS %s", loc_tag
        )
        metadata.drop("hrrr_18-48")
        metadata.drop("nbm")
        metadata.drop("hrrr_0-18")
        metadata.drop("hrrrsubh", time_key="hrrr_subh")

    if "dwd_mosmix" in metadata.source_list and isinstance(data_dwd_mosmix, np.ndarray):
        dwd_mosmix_merged = _merge_simple_source(
            data_dwd_mosmix,
            base_day_utc_grib,
            num_hours,
            max(DWD_MOSMIX.values()) + 1,
        )

    if "gfs" in metadata.source_list and isinstance(data_gfs, np.ndarray):
        gfs_merged = _merge_simple_source(
            data_gfs,
            base_day_utc_grib,
            num_hours,
            max(GFS.values()) + 1,
            source_columns=data_gfs.shape[1],
        )

    if "ecmwf_ifs" in metadata.source_list and isinstance(data_ecmwf, np.ndarray):
        ecmwf_merged = _merge_simple_source(
            data_ecmwf,
            base_day_utc_grib,
            num_hours,
            max(ECMWF.values()) + 1,
            source_columns=data_ecmwf.shape[1],
        )

    if "gefs" in metadata.source_list and isinstance(data_gefs, np.ndarray):
        gefs_merged = _merge_simple_source(
            data_gefs, base_day_utc_grib, num_hours, data_gefs.shape[1]
        )

    if "hrdps" in metadata.source_list and isinstance(data_hrdps, np.ndarray):
        hrdps_merged = _merge_simple_source(
            data_hrdps,
            base_day_utc_grib,
            num_hours,
            max(HRDPS.values()) + 1,
            source_columns=data_hrdps.shape[1],
        )

    if "gdps" in metadata.source_list and isinstance(data_gdps, np.ndarray):
        gdps_merged = _merge_simple_source(
            data_gdps,
            base_day_utc_grib,
            num_hours,
            max(GDPS.values()) + 1,
            source_columns=data_gdps.shape[1],
        )

    if "geps" in metadata.source_list and isinstance(data_geps, np.ndarray):
        geps_merged = _merge_simple_source(
            data_geps,
            base_day_utc_grib,
            num_hours,
            max(GEPS.values()) + 1,
            source_columns=data_geps.shape[1],
        )

    if "reps" in metadata.source_list and isinstance(data_reps, np.ndarray):
        reps_merged = _merge_simple_source(
            data_reps,
            base_day_utc_grib,
            num_hours,
            max(REPS.values()) + 1,
            source_columns=data_reps.shape[1],
        )

    if "aigfs" in metadata.source_list and isinstance(data_aigfs, np.ndarray):
        aigfs_merged = _merge_aigfs_as_gfs(data_aigfs, base_day_utc_grib, num_hours)

    if "aigefs" in metadata.source_list and isinstance(data_aigefs, np.ndarray):
        aigefs_merged = _merge_aigefs_as_gefs(data_aigefs, base_day_utc_grib, num_hours)

    if "ecmwf_aifs" in metadata.source_list and isinstance(data_aifs, np.ndarray):
        aifs_merged = _merge_aifs_as_ecmwf(data_aifs, base_day_utc_grib, num_hours)

    return MergeResult(
        hrrr=hrrr_merged,
        nbm=nbm_merged,
        gfs=gfs_merged,
        ecmwf=ecmwf_merged,
        gefs=gefs_merged,
        hrdps=hrdps_merged,
        gdps=gdps_merged,
        geps=geps_merged,
        reps=reps_merged,
        dwd_mosmix=dwd_mosmix_merged,
        aigfs=aigfs_merged,
        aigefs=aigefs_merged,
        aifs=aifs_merged,
        metadata=metadata,
        urma=urma_merged,
    )
