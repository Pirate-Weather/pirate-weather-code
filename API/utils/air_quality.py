"""Air-quality helpers for API response shaping."""

from __future__ import annotations

from typing import Any

import numpy as np

from API.ingest_utils import calculate_aqi

RAQDPS_AQ_INDEX = {
    "pm2_5": 1,
    "pm10": 2,
    "no2": 3,
    "o3": 4,
    "so2": 5,
}

SILAM_AQ_INDEX = {
    "pm2_5": 1,
    "pm10": 2,
    "o3": 3,
    "no2": 4,
    "so2": 5,
    "co": 6,
}

AQ_BASE_FIELDS = ("usEpaAqi", "pm2_5")
AQ_DETAIL_FIELDS = ("pm10", "o3", "no2", "so2", "co")


def _aligned_series(
    hour_array_grib: np.ndarray,
    model_data: Any,
    variable_index: int,
) -> np.ndarray:
    out = np.full(len(hour_array_grib), np.nan, dtype=np.float32)
    if not isinstance(model_data, np.ndarray) or model_data.ndim < 2:
        return out

    time_to_value = {}
    for row in model_data:
        if len(row) <= variable_index:
            continue
        ts = int(row[0])
        value = row[variable_index]
        if np.isnan(value):
            continue
        time_to_value[ts] = float(value)

    for i, ts in enumerate(hour_array_grib):
        out[i] = time_to_value.get(int(ts), np.nan)

    return out


def build_air_quality_series(
    *,
    hour_array_grib: np.ndarray,
    data_raqdps: Any,
    data_silam: Any,
) -> dict[str, np.ndarray]:
    """Build hourly AQ series with RAQDPS prioritized over SILAM."""
    pm25_raqdps = _aligned_series(
        hour_array_grib, data_raqdps, RAQDPS_AQ_INDEX["pm2_5"]
    )
    pm25_silam = _aligned_series(hour_array_grib, data_silam, SILAM_AQ_INDEX["pm2_5"])
    pm25 = np.where(~np.isnan(pm25_raqdps), pm25_raqdps, pm25_silam)

    pm10_raqdps = _aligned_series(hour_array_grib, data_raqdps, RAQDPS_AQ_INDEX["pm10"])
    pm10_silam = _aligned_series(hour_array_grib, data_silam, SILAM_AQ_INDEX["pm10"])
    pm10 = np.where(~np.isnan(pm10_raqdps), pm10_raqdps, pm10_silam)

    o3_raqdps = _aligned_series(hour_array_grib, data_raqdps, RAQDPS_AQ_INDEX["o3"])
    o3_silam = _aligned_series(hour_array_grib, data_silam, SILAM_AQ_INDEX["o3"])
    o3 = np.where(~np.isnan(o3_raqdps), o3_raqdps, o3_silam)

    no2_raqdps = _aligned_series(hour_array_grib, data_raqdps, RAQDPS_AQ_INDEX["no2"])
    no2_silam = _aligned_series(hour_array_grib, data_silam, SILAM_AQ_INDEX["no2"])
    no2 = np.where(~np.isnan(no2_raqdps), no2_raqdps, no2_silam)

    so2_raqdps = _aligned_series(hour_array_grib, data_raqdps, RAQDPS_AQ_INDEX["so2"])
    so2_silam = _aligned_series(hour_array_grib, data_silam, SILAM_AQ_INDEX["so2"])
    so2 = np.where(~np.isnan(so2_raqdps), so2_raqdps, so2_silam)

    co = _aligned_series(hour_array_grib, data_silam, SILAM_AQ_INDEX["co"])

    pm25_3d = pm25[:, np.newaxis, np.newaxis]
    pm10_3d = pm10[:, np.newaxis, np.newaxis]
    o3_3d = o3[:, np.newaxis, np.newaxis]
    no2_3d = no2[:, np.newaxis, np.newaxis]
    so2_3d = so2[:, np.newaxis, np.newaxis]
    co_3d = co[:, np.newaxis, np.newaxis]
    aqi = calculate_aqi(
        pm25_3d, pm10_3d, o3_3d, no2_3d, so2_3d, co_3d, use_nowcast=True
    )[:, 0, 0]

    return {
        "usEpaAqi": aqi.astype(np.float32),
        "pm2_5": pm25.astype(np.float32),
        "pm10": pm10.astype(np.float32),
        "o3": o3.astype(np.float32),
        "no2": no2.astype(np.float32),
        "so2": so2.astype(np.float32),
        "co": co.astype(np.float32),
    }


def enrich_hourly_with_air_quality(
    *,
    hour_list: list[dict[str, Any]],
    aq_series: dict[str, np.ndarray],
    include_details: bool,
) -> None:
    fields = AQ_BASE_FIELDS + (AQ_DETAIL_FIELDS if include_details else ())
    for idx, item in enumerate(hour_list):
        for field in fields:
            item[field] = float(np.round(aq_series[field][idx], 2))


def enrich_current_with_air_quality(
    *,
    currently: dict[str, Any],
    hour_array_grib: np.ndarray,
    aq_series: dict[str, np.ndarray],
    include_details: bool,
) -> None:
    if len(hour_array_grib) == 0:
        return

    fields = AQ_BASE_FIELDS + (AQ_DETAIL_FIELDS if include_details else ())
    for field in fields:
        currently[field] = float(np.round(aq_series[field][0], 2))


def enrich_daily_with_air_quality(
    *,
    day_list: list[dict[str, Any]],
    hour_list: list[dict[str, Any]],
    hourly_day_index: np.ndarray,
    aq_series: dict[str, np.ndarray],
    include_details: bool,
) -> None:
    fields = AQ_BASE_FIELDS + (AQ_DETAIL_FIELDS if include_details else ())
    if not day_list or not hour_list:
        return

    hour_times = np.array(
        [int(item.get("time", 0)) for item in hour_list], dtype=np.int64
    )

    for day_idx, day_item in enumerate(day_list):
        mask = hourly_day_index[: len(hour_times)] == day_idx
        if not np.any(mask):
            continue
        for field in fields:
            values = aq_series[field][: len(hour_times)][mask]
            times = hour_times[mask]
            valid = ~np.isnan(values)
            if not np.any(valid):
                continue
            valid_values = values[valid]
            valid_times = times[valid]
            max_idx = int(np.argmax(valid_values))
            min_idx = int(np.argmin(valid_values))
            day_item[f"{field}Avg"] = float(np.round(np.nanmean(valid_values), 2))
            day_item[f"{field}Min"] = float(np.round(valid_values[min_idx], 2))
            day_item[f"{field}MinTime"] = int(valid_times[min_idx])
            day_item[f"{field}Max"] = float(np.round(valid_values[max_idx], 2))
            day_item[f"{field}MaxTime"] = int(valid_times[max_idx])
