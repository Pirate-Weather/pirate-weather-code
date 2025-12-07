import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from API.constants.model_const import DWD_MOSMIX, ECMWF, GFS
from API.constants.shared_const import MISSING_DATA
from API.request.grid_indexing import GridIndexingResult
from API.utils.geo import rounder


@dataclass
class SourceMetadata:
    source_list: List[str]
    source_times: Dict[str, str]
    source_idx: Dict[str, dict]

    def add(
        self,
        source: str,
        *,
        time_value: Optional[str] = None,
        time_key: Optional[str] = None,
    ) -> None:
        if source not in self.source_list:
            self.source_list.append(source)
        if time_value is not None:
            self.source_times[time_key or source] = time_value

    def drop(self, source: str, *, time_key: Optional[str] = None) -> None:
        if source in self.source_list:
            self.source_list.remove(source)
        self.source_times.pop(time_key or source, None)

    def has_sources(self, *sources: str) -> bool:
        return all(src in self.source_list for src in sources)


@dataclass
class MergeResult:
    hrrr: Optional[np.ndarray]
    nbm: Optional[np.ndarray]
    nbm_fire: Optional[np.ndarray]
    gfs: Optional[np.ndarray]
    ecmwf: Optional[np.ndarray]
    gefs: Optional[np.ndarray]
    dwd_mosmix: Optional[np.ndarray]
    metadata: SourceMetadata


def nearest_index(a, v) -> int:
    """Find the nearest index in array to value using binary search."""
    idx = np.searchsorted(a, v)
    idx = np.clip(idx, 1, len(a) - 1)
    left, right = a[idx - 1], a[idx]
    return idx if abs(right - v) < abs(v - left) else idx - 1


def _format_run_time(
    run_time: Optional[Union[float, np.generic]],
    *,
    offset_hours: int = 0,
    round_to: Optional[int] = None,
    fmt: str = "%Y-%m-%d %HZ",
) -> Optional[str]:
    if run_time is None:
        return None

    runtime = datetime.datetime.fromtimestamp(int(run_time), datetime.UTC).replace(
        tzinfo=None
    )
    if offset_hours:
        runtime -= datetime.timedelta(hours=offset_hours)
    rounded = rounder(runtime, to=round_to) if round_to else rounder(runtime)
    return rounded.strftime(fmt)


def _format_rtma_time(data_out_rtma: np.ndarray) -> Optional[str]:
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
    era5_merged: Union[np.ndarray, bool],
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

    if isinstance(grid_result.dataOut_nbmFire, np.ndarray) and not time_machine:
        metadata.add(
            "nbm_fire", time_value=_format_run_time(grid_result.nbmFireRunTime)
        )

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

    return metadata


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


def _merge_hrrr_blocks(
    data_hrrrh: np.ndarray,
    data_h2: np.ndarray,
    hrrr_start_idx: int,
    h2_start_idx: int,
    num_hours: int,
) -> np.ndarray:
    merged = np.full((num_hours, data_h2.shape[1]), MISSING_DATA)
    common_cols = min(data_hrrrh.shape[1], data_h2.shape[1])
    hrrr_rows = len(data_hrrrh) - hrrr_start_idx
    h2_rows = len(data_h2) - h2_start_idx
    total_rows = min(hrrr_rows + h2_rows, num_hours)
    merged[0:total_rows, 0:common_cols] = np.concatenate(
        (
            data_hrrrh[hrrr_start_idx:, 0:common_cols],
            data_h2[h2_start_idx:, 0:common_cols],
        ),
        axis=0,
    )[0:total_rows, :]
    return merged


def _merge_simple_source(
    data: np.ndarray,
    start_idx: int,
    num_hours: int,
    target_columns: int,
    *,
    source_columns: Optional[int] = None,
) -> np.ndarray:
    merged = np.full((num_hours, target_columns), MISSING_DATA)
    end_idx = min(len(data), num_hours + start_idx)
    copy_columns = source_columns or target_columns
    merged[0 : (end_idx - start_idx), 0:copy_columns] = data[
        start_idx:end_idx, 0:copy_columns
    ]
    return merged


def merge_hourly_models(
    *,
    metadata: SourceMetadata,
    num_hours: int,
    base_day_utc_grib,
    data_hrrrh: Optional[np.ndarray],
    data_h2: Optional[np.ndarray],
    data_nbm: Optional[np.ndarray],
    data_nbm_fire: Optional[np.ndarray],
    data_gfs: Optional[np.ndarray],
    data_ecmwf: Optional[np.ndarray],
    data_gefs: Optional[np.ndarray],
    data_dwd_mosmix: Optional[np.ndarray],
    logger: logging.Logger,
    loc_tag: str,
) -> MergeResult:
    hrrr_merged = None
    nbm_merged = None
    nbm_fire_merged = None
    gfs_merged = None
    ecmwf_merged = None
    gefs_merged = None
    dwd_mosmix_merged = None

    try:
        if (
            metadata.has_sources("hrrr_0-18", "hrrr_18-48")
            and isinstance(data_hrrrh, np.ndarray)
            and isinstance(data_h2, np.ndarray)
        ):
            hrrr_start_idx = nearest_index(data_hrrrh[:, 0], base_day_utc_grib)
            h2_start_idx = nearest_index(data_h2[:, 0], data_hrrrh[-1, 0]) + 1

            if (h2_start_idx < 1) or (hrrr_start_idx < 2):
                metadata.drop("hrrr_18-48")
                metadata.drop("hrrr_0-18")
                logger.error("HRRR data not available for the requested time range.")
            else:
                hrrr_merged = _merge_hrrr_blocks(
                    data_hrrrh, data_h2, hrrr_start_idx, h2_start_idx, num_hours
                )

        if "nbm" in metadata.source_list and isinstance(data_nbm, np.ndarray):
            nbm_start_idx = nearest_index(data_nbm[:, 0], base_day_utc_grib)
            if nbm_start_idx < 1:
                metadata.drop("nbm")
                logger.error("NBM data not available for the requested time range.")
            else:
                nbm_merged = _merge_simple_source(
                    data_nbm, nbm_start_idx, num_hours, data_nbm.shape[1]
                )

        if "nbm_fire" in metadata.source_list and isinstance(data_nbm_fire, np.ndarray):
            nbm_fire_start_idx = nearest_index(data_nbm_fire[:, 0], base_day_utc_grib)
            if nbm_fire_start_idx < 1:
                metadata.drop("nbm_fire")
                logger.error(
                    "NBM Fire data not available for the requested time range."
                )
            else:
                nbm_fire_merged = _merge_simple_source(
                    data_nbm_fire, nbm_fire_start_idx, num_hours, data_nbm_fire.shape[1]
                )

    except Exception:
        logger.exception(
            "HRRR or NBM data not available, falling back to GFS %s", loc_tag
        )
        metadata.drop("hrrr_18-48")
        metadata.drop("nbm_fire")
        metadata.drop("nbm")
        metadata.drop("hrrr_0-18")
        metadata.drop("hrrrsubh", time_key="hrrr_subh")

    if "dwd_mosmix" in metadata.source_list and isinstance(data_dwd_mosmix, np.ndarray):
        dwd_start_idx = nearest_index(data_dwd_mosmix[:, 0], base_day_utc_grib)

        # Check if we have valid data at the start index
        if dwd_start_idx >= len(data_dwd_mosmix):
            logger.warning(
                f"DWD MOSMIX start index ({dwd_start_idx}) exceeds data length ({len(data_dwd_mosmix)}) {loc_tag}"
            )
            metadata.drop("dwd_mosmix")
        elif dwd_start_idx < 0:
            logger.warning(
                f"DWD MOSMIX start index is negative ({dwd_start_idx}) {loc_tag}"
            )
            metadata.drop("dwd_mosmix")
        else:
            # Check if the data at start_idx is actually at or after the base_day_utc_grib
            data_time = data_dwd_mosmix[dwd_start_idx, 0]
            time_diff_hours = (data_time - base_day_utc_grib) / 3600

            logger.debug(
                f"DWD MOSMIX merge: start_idx={dwd_start_idx}, "
                f"base_day_utc={base_day_utc_grib}, data_time={data_time}, "
                f"diff={time_diff_hours:.1f}h {loc_tag}"
            )

            # If the data starts significantly after the base day, we need to account for the offset
            # Use timestamp-based alignment instead of index-based copying
            dwd_mosmix_merged = np.full(
                (num_hours, max(DWD_MOSMIX.values()) + 1), MISSING_DATA
            )

            # Copy data row by row, aligning by timestamp
            for i in range(len(data_dwd_mosmix)):
                row_time = data_dwd_mosmix[i, 0]
                # Calculate which output hour this row corresponds to
                hour_offset = int(round((row_time - base_day_utc_grib) / 3600))

                # Only copy if within valid output range
                if 0 <= hour_offset < num_hours:
                    copy_columns = data_dwd_mosmix.shape[1]
                    dwd_mosmix_merged[hour_offset, 0:copy_columns] = data_dwd_mosmix[
                        i, 0:copy_columns
                    ]

            # Log if merged data is mostly NaN (could indicate data quality issues)
            if dwd_mosmix_merged is not None:
                non_nan_count = np.count_nonzero(~np.isnan(dwd_mosmix_merged[:, 1:]))
                total_count = dwd_mosmix_merged[:, 1:].size
                if total_count > 0:
                    nan_pct = 100 * (1 - non_nan_count / total_count)
                    if nan_pct > 90:
                        logger.warning(
                            f"DWD MOSMIX merged data is {nan_pct:.1f}% NaN "
                            f"({non_nan_count}/{total_count} valid) {loc_tag}"
                        )
                    # Check if temperature data exists and looks valid (should be in Celsius, typically -50 to 50)
                    temp_col = DWD_MOSMIX["temp"]
                    if temp_col < dwd_mosmix_merged.shape[1]:
                        temps = dwd_mosmix_merged[:, temp_col]
                        valid_temps = temps[~np.isnan(temps)]
                        if len(valid_temps) > 0:
                            temp_mean = np.mean(valid_temps)
                            # If mean temp is > 100, it's likely still in Kelvin (conversion didn't happen)
                            if temp_mean > 100:
                                logger.error(
                                    f"DWD MOSMIX temperature appears to be in Kelvin "
                                    f"(mean={temp_mean:.1f}K). Conversion may have failed. {loc_tag}"
                                )
                            elif temp_mean < -100:
                                logger.warning(
                                    f"DWD MOSMIX temperature is unusually low "
                                    f"(mean={temp_mean:.1f}C). Possible data issue. {loc_tag}"
                                )

    if "gfs" in metadata.source_list and isinstance(data_gfs, np.ndarray):
        gfs_start_idx = nearest_index(data_gfs[:, 0], base_day_utc_grib)
        gfs_merged = _merge_simple_source(
            data_gfs,
            gfs_start_idx,
            num_hours,
            max(GFS.values()) + 1,
            source_columns=data_gfs.shape[1],
        )

    if "ecmwf_ifs" in metadata.source_list and isinstance(data_ecmwf, np.ndarray):
        ecmwf_start_idx = nearest_index(data_ecmwf[:, 0], base_day_utc_grib)
        ecmwf_merged = _merge_simple_source(
            data_ecmwf,
            ecmwf_start_idx,
            num_hours,
            max(ECMWF.values()) + 1,
            source_columns=data_ecmwf.shape[1],
        )

    if "gefs" in metadata.source_list and isinstance(data_gefs, np.ndarray):
        gefs_start_idx = nearest_index(data_gefs[:, 0], base_day_utc_grib)
        gefs_merged = _merge_simple_source(
            data_gefs, gefs_start_idx, num_hours, data_gefs.shape[1]
        )

    return MergeResult(
        hrrr=hrrr_merged,
        nbm=nbm_merged,
        nbm_fire=nbm_fire_merged,
        gfs=gfs_merged,
        ecmwf=ecmwf_merged,
        gefs=gefs_merged,
        dwd_mosmix=dwd_mosmix_merged,
        metadata=metadata,
    )
