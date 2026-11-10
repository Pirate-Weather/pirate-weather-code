import datetime
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from API.api_utils import relative_humidity_from_dewpoint_c
from API.constants.model_const import (
    AIGEFS,
    AIGFS,
    CFS,
    DWD_MOSMIX,
    ECMWF,
    ECMWF_AIFS,
    GDPS,
    GEFS,
    GEPS,
    GFS,
    HRDPS,
    REPS,
    RRFS,
    URMA,
)
from API.constants.shared_const import MISSING_DATA, PUBLISH_SCHEDULE_HOURS
from API.request.grid_indexing import GridIndexingResult
from API.utils.geo import rounder

SECONDS_PER_HOUR = 3600

URMA_TIME_TOLERANCE_SECONDS = 300

# The sources the per-element selections report as having supplied a value, as
# the columns of the per-hour usage mask. The keys are the selection code's own
# names: "hrrr" and "rrfs" stand for both of their hourly stores, "hrrrsubh" and
# "rrfssubh" for the sub-hourly ones (select_used_sources maps them to flag names).
USED_SOURCE_KEYS = (
    "rrfs",
    "rrfssubh",
    "urma",
    "rtma_ru",
    "hrrrsubh",
    "librewxr",
    "nbm",
    "hrrr",
    "hrdps",
    "reps",
    "gdps",
    "geps",
    "ecmwf_ifs",
    "era5",
    "gfs",
    "gefs",
    "dwd_mosmix",
    "raqdps",
    "silam",
    "cfs",
)
USED_SOURCE_COLUMN = {key: column for column, key in enumerate(USED_SOURCE_KEYS)}
# The hourly stacks and the daily member masks call ECMWF IFS "ecmwf"
USED_SOURCE_COLUMN["ecmwf"] = USED_SOURCE_COLUMN["ecmwf_ifs"]

# sourceTimes keys that differ from the flags.sources name they belong to
_SOURCE_TIME_KEYS = {"hrrr_subh": "hrrrsubh", "rrfs_subh": "rrfssubh"}


@dataclass
class SourceMetadata:
    source_list: list[str]
    source_times: dict[str, str]
    source_idx: dict[str, dict]
    # The sources that supplied at least one returned value and their run
    # times (version 2 flags.sources / sourceTimes), in source_list order; set by
    # select_used_sources. source_list stays the sources that were available.
    used_sources: list[str] | None = None
    used_source_times: dict[str, str] | None = None

    # Run time of each source as unix seconds (the time sourceTimes shows), keyed
    # by flags.sources name; next_update_time projects the next refresh from it.
    source_run_times: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        source: str,
        *,
        run_time: datetime.datetime | None = None,
        time_fmt: str = "%Y-%m-%d %HZ",
        time_key: str | None = None,
    ) -> None:
        if source not in self.source_list:
            self.source_list.append(source)
        if run_time is not None:
            self.source_times[time_key or source] = run_time.strftime(time_fmt)
            self.source_run_times[source] = int(run_time.timestamp())

    def drop(self, source: str, *, time_key: str | None = None) -> None:
        if source in self.source_list:
            self.source_list.remove(source)
        self.source_times.pop(time_key or source, None)
        self.source_run_times.pop(source, None)

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
    rrfs: np.ndarray | None = None
    cfs: np.ndarray | None = None


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
        # A day that starts after the store ends (a future request, #689) has
        # no rows here: the nearest row would be the store's last hour, copied
        # onto local midnight. Only the end matters; an all-NaN row inside the
        # store (NaN time) must not drop the whole store.
        if not np.any(times >= base_timestamp - SECONDS_PER_HOUR + 1):
            return 0, 0, 0
        src_start = nearest_index(times, base_timestamp)
    rows = max(0, min(len(times) - src_start, num_hours - dest_start))
    return dest_start, src_start, rows


def _rounded_run_time(
    run_time: float | np.generic | None,
    *,
    offset_hours: int = 0,
    round_to: int | None = None,
) -> datetime.datetime | None:
    """A store's run time (unix seconds) as a UTC datetime, rounded as sourceTimes shows it."""
    if run_time is None:
        return None

    runtime = datetime.datetime.fromtimestamp(int(run_time), datetime.UTC)
    if offset_hours:
        runtime -= datetime.timedelta(hours=offset_hours)
    return rounder(runtime, to=round_to) if round_to else rounder(runtime)


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

    # Listed only if it has data for the request: a recent time-machine request
    # can read ERA5 rows that are all NaN (hours ERA5 hasn't reached yet, which
    # GFS history fills), and code that branches on "era5" assumes values.
    if isinstance(era5_merged, np.ndarray) and np.isfinite(era5_merged[:, 1:]).any():
        metadata.add("era5")

    # RRFS (include=rrfs). Each block is listed under the grid it came from
    # ("rrfs" for 3 km CONUS, "rrfs_na" for 13 km North America), like HRRR's
    # 0-18 and 18-48 h stores.
    rrfs_grid = grid_result.rrfs_grid
    if isinstance(grid_result.dataOut_rrfs_subh, np.ndarray):
        metadata.add(
            "rrfssubh",
            run_time=_rounded_run_time(grid_result.rrfsSubhRunTime),
            time_key="rrfs_subh",
        )
    if isinstance(grid_result.dataOut_rrfs, np.ndarray):
        metadata.add(
            f"{rrfs_grid}_0-18", run_time=_rounded_run_time(grid_result.rrfsRunTime)
        )
    if isinstance(grid_result.dataOut_rrfs_6h, np.ndarray):
        metadata.add(
            f"{rrfs_grid}_18-83",
            run_time=_rounded_run_time(grid_result.rrfs6hRunTime),
        )

    if isinstance(grid_result.dataOut, np.ndarray):
        metadata.add(
            "hrrrsubh",
            run_time=_rounded_run_time(grid_result.subhRunTime),
            time_key="hrrr_subh",
        )

    if isinstance(grid_result.dataOut_librewxr, np.ndarray):
        metadata.add(
            "librewxr",
            run_time=_rounded_run_time(grid_result.librewxrRunTime, round_to=10),
            time_fmt="%Y-%m-%d %H:%MZ",
        )

    if isinstance(grid_result.dataOut_rtma_ru, np.ndarray) and not time_machine:
        # RTMA-RU is an analysis: its run time is the valid time (15-minute cycles)
        metadata.add(
            "rtma_ru",
            run_time=_rounded_run_time(grid_result.dataOut_rtma_ru[0, 0], round_to=15),
            time_fmt="%Y-%m-%d %H:%MZ",
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
                run_time=_rounded_run_time(grid_result.hrrrhRunTime),
            )
        else:
            metadata.add("hrrr")

    if isinstance(grid_result.dataOut_nbm, np.ndarray):
        if not time_machine:
            metadata.add("nbm", run_time=_rounded_run_time(grid_result.nbmRunTime))
        else:
            metadata.add("nbm")

    if isinstance(grid_result.dataOut_dwd_mosmix, np.ndarray) and not time_machine:
        metadata.add(
            "dwd_mosmix", run_time=_rounded_run_time(grid_result.dwdMosmixRunTime)
        )

    if isinstance(grid_result.dataOut_ecmwf, np.ndarray) and not time_machine:
        metadata.add("ecmwf_ifs", run_time=_rounded_run_time(grid_result.ecmwfRunTime))

    if isinstance(grid_result.dataOut_h2, np.ndarray):
        metadata.add(
            "hrrr_18-48",
            # The 18-48 h store starts at lead 18
            run_time=_rounded_run_time(grid_result.h2RunTime, offset_hours=18),
        )

    if isinstance(grid_result.dataOut_gfs, np.ndarray):
        metadata.add("gfs", run_time=_rounded_run_time(grid_result.gfsRunTime))
        metadata.source_idx["gfs"] = {
            "x": int(grid_result.x_p),
            "y": int(grid_result.y_p),
            "lat": round(float(grid_result.gfs_lat), 2),
            "lon": round(((float(grid_result.gfs_lon) + 180) % 360) - 180, 2),
        }

    if isinstance(grid_result.dataOut_gefs, np.ndarray):
        metadata.add("gefs", run_time=_rounded_run_time(grid_result.gefsRunTime))

    # CFS is only read for a future day past the GFS store (grid_indexing)
    if isinstance(grid_result.dataOut_cfs, np.ndarray):
        metadata.add("cfs", run_time=_rounded_run_time(grid_result.cfsRunTime))

    if isinstance(grid_result.dataOut_hrdps, np.ndarray):
        metadata.add("hrdps", run_time=_rounded_run_time(grid_result.hrdpsRunTime))

    if isinstance(grid_result.dataOut_gdps, np.ndarray):
        metadata.add("gdps", run_time=_rounded_run_time(grid_result.gdpsRunTime))

    if isinstance(grid_result.dataOut_geps, np.ndarray):
        metadata.add("geps", run_time=_rounded_run_time(grid_result.gepsRunTime))

    if isinstance(grid_result.dataOut_reps, np.ndarray):
        metadata.add("reps", run_time=_rounded_run_time(grid_result.repsRunTime))

    if isinstance(grid_result.dataOut_aigfs, np.ndarray):
        metadata.add("aigfs", run_time=_rounded_run_time(grid_result.aigfsRunTime))

    if isinstance(grid_result.dataOut_aigefs, np.ndarray):
        metadata.add("aigefs", run_time=_rounded_run_time(grid_result.aigefsRunTime))

    if isinstance(grid_result.dataOut_aifs, np.ndarray):
        metadata.add("ecmwf_aifs", run_time=_rounded_run_time(grid_result.aifsRunTime))

    if isinstance(grid_result.dataOut_raqdps, np.ndarray):
        metadata.add("raqdps", run_time=_rounded_run_time(grid_result.raqdpsRunTime))
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
        metadata.add("silam", run_time=_rounded_run_time(grid_result.silamRunTime))
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


def member_mask_columns(index_map: dict) -> list[int]:
    """The member bitmask word columns of a model's column map, in word order."""
    keys = sorted(key for key in index_map if key.startswith("member_mask_"))
    return [index_map[key] for key in keys]


def _copy_member_mask_columns(out, src, target_map: dict, source_map: dict) -> None:
    """Copy the AI ensemble's member bitmask words into the columns of the model
    it stands in for (equal member counts, see ENSEMBLE_MEMBER_COUNTS). A store
    from before the masks is narrower and leaves the columns missing, so the daily
    probability falls back to the hourly maximum."""
    source_columns = member_mask_columns(source_map)
    if src.shape[1] > max(source_columns):
        out[:, member_mask_columns(target_map)] = src[:, source_columns]


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
    _copy_member_mask_columns(out, src, GEFS, AIGEFS)
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
    _copy_member_mask_columns(out, src, ECMWF, ECMWF_AIFS)
    # The AIFS store has no humidity column, unlike the IFS store it stands in for.
    # Temperatures are already in Celsius here (convert_data_to_celsius runs first).
    out[:, ECMWF["humidity"]] = (
        relative_humidity_from_dewpoint_c(
            src[:, ECMWF_AIFS["temp"]], src[:, ECMWF_AIFS["dew"]]
        )
        * 100.0
    )
    return merged


def _used_flag_names(key, source_list, *, time_machine, rrfs_grid):
    """The flags.sources names a selection key stands for."""
    if key == "hrrr":
        # Both HRRR stores are merged into one hourly array (forecast only)
        return ("hrrr",) if time_machine else ("hrrr_0-18", "hrrr_18-48")
    if key == "rrfssubh" and "rrfssubh" not in source_list:
        # The minutely RRFS fields were interpolated from the hourly store
        key = "rrfs"
    if key == "rrfs":
        return (f"{rrfs_grid}_0-18", f"{rrfs_grid}_18-83")
    return (key,)


def select_used_sources(
    metadata: SourceMetadata,
    used_hours: np.ndarray,
    *,
    minutely_used: set[str],
    current_used: set[str],
    daily_used: set[str],
    ai_blends: dict | None,
    ex_currently: bool,
    ex_minutely: bool,
    ex_hourly: bool,
    ex_daily: bool,
    day_night: bool,
    time_machine: bool,
    base_time_offset: int,
    output_hours: int,
    rrfs_grid: str | None,
) -> list[str]:
    """Pick the sources that supplied at least one returned value (#554).

    ``metadata.source_list`` is every source read for the request; version 2
    flags show only those whose values reached a returned block:
    - hourly: the source of each value in the returned hours (``used_hours``);
    - daily / day_night: every hour, as each feeds a day's aggregate, plus the
      ensembles whose member share became a day's probability (``daily_used``);
    - minutely, and currently (whose precipitation is the first minute's):
      the minutely selections (``minutely_used``);
    - currently: its own per-field selections (``current_used``);
    - ETOPO1, for the elevation every response carries.

    Args:
        metadata: The request's source metadata. Its ``used_sources`` and
            ``used_source_times`` are set from its ``source_list`` (keeping
            that order) and ``source_times``.
        used_hours: (hours, USED_SOURCE_KEYS) bool mask of the source each
            hourly value came from (the hourly block, plus air quality).
        minutely_used, current_used, daily_used: The selection keys the
            minutely, currently and daily member-share selections took.
        ai_blends: ``{key: (ai_name, ai_hours, base_hours)}`` under
            include=aimodels, where an AI model was merged into the array of
            the conventional model ``key``: the hours where each of the two
            supplied values, so each is listed only if it supplied a counted
            hour. Minutely and currently selections count as the current hour
            and the next.
        ex_currently, ex_minutely, ex_hourly, ex_daily: The blocks excluded.
        day_night: Whether the day_night block is returned.
        time_machine: Whether this is a time machine request.
        base_time_offset: The current hour's row (forecast requests).
        output_hours: The hourly rows returned (48, or 168 with extend=hourly).
        rrfs_grid: The RRFS grid read ("rrfs" or "rrfs_na"), if any.

    Returns:
        The used sources (also set on ``metadata``).
    """
    hourly_start = 0 if time_machine else base_time_offset
    current_rows = slice(hourly_start, hourly_start + 2)
    counted = np.zeros_like(used_hours)
    if not ex_daily or day_night:
        counted[:] = used_hours
        for key in daily_used:
            counted[:, USED_SOURCE_COLUMN[key]] = True
    elif not ex_hourly:
        hourly_rows = slice(hourly_start, hourly_start + output_hours)
        counted[hourly_rows] = used_hours[hourly_rows]
    selections = []
    if not ex_currently or not ex_minutely:
        selections.append(minutely_used)
    if not ex_currently:
        selections.append(current_used)
    for keys in selections:
        for key in keys:
            counted[current_rows, USED_SOURCE_COLUMN[key]] = True

    ai_blends = ai_blends or {}
    used_names = set()
    for column in np.flatnonzero(counted.any(axis=0)):
        key = USED_SOURCE_KEYS[column]
        blend = ai_blends.get(key)
        if blend is not None:
            ai_name, ai_hours, base_hours = blend
            hours = counted[:, column]
            if (hours & ai_hours).any():
                used_names.add(ai_name)
            if not (hours & base_hours).any():
                continue
        used_names.update(
            _used_flag_names(
                key,
                metadata.source_list,
                time_machine=time_machine,
                rrfs_grid=rrfs_grid,
            )
        )

    # Every response carries the elevation
    used_names.add("ETOPO1")

    metadata.used_sources = [
        source for source in metadata.source_list if source in used_names
    ]
    metadata.used_source_times = {
        time_key: run_time
        for time_key, run_time in metadata.source_times.items()
        if _SOURCE_TIME_KEYS.get(time_key, time_key) in used_names
    }
    return metadata.used_sources


def next_update_time(metadata: SourceMetadata, now_timestamp: float) -> int | None:
    """When the forecast should next refresh: version 2 ``flags.nextUpdate`` (#668).

    The earliest expected publish of a new run among the sources used in the
    response (``metadata.used_sources``), from each one's run time and its
    ``PUBLISH_SCHEDULE_HOURS`` entry. A run that is overdue (a late ingest, or a
    cycle that never came) rolls forward by whole cycles to the next scheduled
    publish, so the value is never in the past. Sources without a schedule
    (ETOPO1, ERA5) or without a run time are skipped.

    Args:
        metadata: The request's source metadata, after ``select_used_sources``.
        now_timestamp: The request's current time, unix seconds (UTC).

    Returns:
        The expected publish time in unix seconds, or None if no used source
        has a schedule.
    """
    next_update = None
    for source in metadata.used_sources or ():
        schedule = PUBLISH_SCHEDULE_HOURS.get(source)
        run_time = metadata.source_run_times.get(source)
        if schedule is None or run_time is None:
            continue
        interval_hours, delay_hours = schedule
        interval_seconds = interval_hours * 3600
        expected = run_time + (interval_hours + delay_hours) * 3600
        if expected <= now_timestamp:
            missed_cycles = (
                math.floor((now_timestamp - expected) / interval_seconds) + 1
            )
            expected += missed_cycles * interval_seconds
        if next_update is None or expected < next_update:
            next_update = expected
    return None if next_update is None else int(next_update)


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
    # A store can be wider than the API's column map while an older API serves a
    # newer store (columns are only ever appended); the extra columns are ignored
    copy_columns = min(source_columns or target_columns, target_columns)
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


def _merge_rrfs_blocks(
    data_rrfs: np.ndarray | None,
    data_rrfs_6h: np.ndarray | None,
    base_timestamp: float,
    num_hours: int,
) -> np.ndarray:
    """Join the RRFS hourly (1-18 h) and 6-hourly (18-83 h) stores.

    The two come from different runs (the 6-hourly store only updates at 00,
    06, 12 and 18z), so the hourly rows are kept and the 6-hourly store
    continues from the first time after them. Either store may be missing on
    its own (the AWS bucket drops cycles); the other is then used alone.
    """
    if data_rrfs is None:
        rrfs_combined = data_rrfs_6h
    elif data_rrfs_6h is None:
        rrfs_combined = data_rrfs
    else:
        last_hourly_time = np.nanmax(data_rrfs[:, 0])
        later_rows = data_rrfs_6h[:, 0] > last_hourly_time
        rrfs_combined = np.concatenate(
            (data_rrfs[np.isfinite(data_rrfs[:, 0])], data_rrfs_6h[later_rows]),
            axis=0,
        )
    return _merge_simple_source(
        rrfs_combined, base_timestamp, num_hours, max(RRFS.values()) + 1
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
    data_rrfs: np.ndarray | None = None,
    data_rrfs_6h: np.ndarray | None = None,
    rrfs_grid: str | None = None,
    data_cfs: np.ndarray | None = None,
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
    rrfs_merged = None
    cfs_merged = None

    if rrfs_grid is not None and (data_rrfs is not None or data_rrfs_6h is not None):
        rrfs_merged = _merge_rrfs_blocks(
            data_rrfs, data_rrfs_6h, base_day_utc_grib, num_hours
        )
        if not np.isfinite(rrfs_merged[:, 0]).any():
            rrfs_merged = None
            metadata.drop(f"{rrfs_grid}_0-18")
            metadata.drop(f"{rrfs_grid}_18-83")
            metadata.drop("rrfssubh", time_key="rrfs_subh")
            logger.warning("RRFS data not available for the requested time range.")

    if isinstance(data_urma, np.ndarray) and urma_min_timestamp is not None:
        urma_merged, latest_urma_time = _merge_urma_source(
            data_urma, base_day_utc_grib, num_hours, urma_min_timestamp
        )
        if urma_merged is not None:
            metadata.add("urma", run_time=_rounded_run_time(latest_urma_time))

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
            data_gefs,
            base_day_utc_grib,
            num_hours,
            max(GEFS.values()) + 1,
            source_columns=data_gefs.shape[1],
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

    # CFS holds only the requested day's window (zarr_read_window), so its rows
    # align like any forecast-only store's
    if "cfs" in metadata.source_list and isinstance(data_cfs, np.ndarray):
        cfs_merged = _merge_simple_source(
            data_cfs,
            base_day_utc_grib,
            num_hours,
            max(CFS.values()) + 1,
            source_columns=data_cfs.shape[1],
        )

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
        rrfs=rrfs_merged,
        cfs=cfs_merged,
    )
