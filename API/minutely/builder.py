"""Minutely interpolation and object construction helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from API.api_utils import fast_nearest_interp
from API.constants.api_const import (
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
)
from API.constants.forecast_const import DATA_MINUTELY
from API.constants.model_const import ECMWF, ERA5, GEFS, GFS, HRRR, HRRR_SUBH, NBM
from API.constants.shared_const import MISSING_DATA
from API.utils.precip import dbz_to_rate


def build_minutely_block(
    *,
    minute_array_grib: np.ndarray,
    source_list: List[str],
    hrrr_subh_data: Optional[np.ndarray],
    hrrr_merged: Optional[np.ndarray],
    nbm_data: Optional[np.ndarray],
    gefs_data: Optional[np.ndarray],
    gfs_data: Optional[np.ndarray],
    ecmwf_data: Optional[np.ndarray],
    era5_data: Optional[np.ndarray],
    prep_intensity_unit: float,
    version: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    list,
    list,
    np.ndarray,
    list,
    list,
    Optional[np.ndarray],
]:
    """Compute minutely interpolations and output objects."""
    InterTminute = np.zeros((61, 5))
    InterPminute = np.full((61, max(DATA_MINUTELY.values()) + 1), MISSING_DATA)

    # Build interpolation scaffolding
    gefsMinuteInterpolation = (
        np.zeros((len(minute_array_grib), len(gefs_data[0, :])))
        if "gefs" in source_list and gefs_data is not None and len(gefs_data) > 0
        else None
    )
    gfsMinuteInterpolation = (
        np.zeros((len(minute_array_grib), len(gfs_data[0, :])))
        if "gfs" in source_list and gfs_data is not None and len(gfs_data) > 0
        else None
    )
    ecmwfMinuteInterpolation = (
        np.zeros((len(minute_array_grib), len(ecmwf_data[0, :])))
        if "ecmwf_ifs" in source_list and ecmwf_data is not None and len(ecmwf_data) > 0
        else None
    )
    nbmMinuteInterpolation = (
        np.zeros((len(minute_array_grib), 18)) if nbm_data is not None else None
    )

    hrrrSubHInterpolation: Optional[np.ndarray] = None

    if "hrrrsubh" in source_list and hrrr_subh_data is not None:
        hrrrSubHInterpolation = np.zeros(
            (len(minute_array_grib), len(hrrr_subh_data[0, :]))
        )
        for i in range(len(hrrr_subh_data[0, :]) - 1):
            hrrrSubHInterpolation[:, i + 1] = np.interp(
                minute_array_grib,
                hrrr_subh_data[:, 0].squeeze(),
                hrrr_subh_data[:, i + 1],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        # Fallback to hourly HRRR if SubH is out of range
        if np.isnan(hrrrSubHInterpolation[1, 1]) and hrrr_merged is not None:
            for key in [
                "gust",
                "pressure",
                "temp",
                "dew",
                "wind_u",
                "wind_v",
                "intensity",
                "snow",
                "ice",
                "freezing_rain",
                "rain",
                "refc",
                "solar",
                "vis",
            ]:
                hrrrSubHInterpolation[:, HRRR_SUBH[key]] = np.interp(
                    minute_array_grib,
                    hrrr_merged[:, 0].squeeze(),
                    hrrr_merged[:, HRRR[key]],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

        if "gefs" in source_list and gefsMinuteInterpolation is not None:
            gefsMinuteInterpolation[:, GEFS["error"]] = np.interp(
                minute_array_grib,
                gefs_data[:, 0].squeeze(),
                gefs_data[:, GEFS["error"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
    else:
        if "gefs" in source_list and gefsMinuteInterpolation is not None:
            for i in range(len(gefs_data[0, :]) - 1):
                gefsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    gefs_data[:, 0].squeeze(),
                    gefs_data[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

        if "gfs" in source_list and gfsMinuteInterpolation is not None:
            for i in range(len(gfs_data[0, :]) - 1):
                gfsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    gfs_data[:, 0].squeeze(),
                    gfs_data[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

    if "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        for i in range(len(ecmwf_data[0, :]) - 1):
            if i + 1 == ECMWF["ptype"]:
                ecmwfMinuteInterpolation[:, i + 1] = fast_nearest_interp(
                    minute_array_grib,
                    ecmwf_data[:, 0].squeeze(),
                    ecmwf_data[:, i + 1],
                )
            else:
                ecmwfMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    ecmwf_data[:, 0].squeeze(),
                    ecmwf_data[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

    if (
        "nbm" in source_list
        and nbmMinuteInterpolation is not None
        and nbm_data is not None
    ):
        for i in [
            NBM["accum"],
            NBM["prob"],
            NBM["rain"],
            NBM["freezing_rain"],
            NBM["snow"],
            NBM["ice"],
        ]:
            nbmMinuteInterpolation[:, i] = np.interp(
                minute_array_grib,
                nbm_data[:, 0].squeeze(),
                nbm_data[:, i],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

    era5_MinuteInterpolation = (
        np.zeros((len(minute_array_grib), max(ERA5.values())))
        if "era5" in source_list and era5_data is not None and len(era5_data) > 0
        else None
    )

    if "era5" in source_list and era5_MinuteInterpolation is not None:
        for i in [
            ERA5["large_scale_rain_rate"],
            ERA5["convective_rain_rate"],
            ERA5["large_scale_snowfall_rate_water_equivalent"],
            ERA5["convective_snowfall_rate_water_equivalent"],
        ]:
            era5_MinuteInterpolation[:, i] = np.interp(
                minute_array_grib,
                era5_data[:, 0].squeeze(),
                era5_data[:, i],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        era5_MinuteInterpolation[:, ERA5["precipitation_type"]] = fast_nearest_interp(
            minute_array_grib,
            era5_data[:, 0].squeeze(),
            era5_data[:, ERA5["precipitation_type"]],
        )

    InterPminute[:, DATA_MINUTELY["time"]] = minute_array_grib

    if "nbm" in source_list:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            nbmMinuteInterpolation[:, NBM["prob"]] * 0.01
        )
    elif "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["prob"]] = ecmwfMinuteInterpolation[
            :, ECMWF["prob"]
        ]
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["prob"]] = gefsMinuteInterpolation[
            :, GEFS["prob"]
        ]
    else:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    InterPminute[
        InterPminute[:, DATA_MINUTELY["prob"]] < 0.05, DATA_MINUTELY["prob"]
    ] = 0

    if "hrrrsubh" in source_list:
        for i in [
            HRRR_SUBH["snow"],
            HRRR_SUBH["ice"],
            HRRR_SUBH["freezing_rain"],
            HRRR_SUBH["rain"],
        ]:
            InterTminute[:, i - 7] = hrrrSubHInterpolation[:, i]
    elif "nbm" in source_list:
        InterTminute[:, 1] = nbmMinuteInterpolation[:, NBM["snow"]]
        InterTminute[:, 2] = nbmMinuteInterpolation[:, NBM["ice"]]
        InterTminute[:, 3] = nbmMinuteInterpolation[:, NBM["freezing_rain"]]
        InterTminute[:, 4] = nbmMinuteInterpolation[:, NBM["rain"]]
    elif "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        ptype_ecmwf = ecmwfMinuteInterpolation[:, ECMWF["ptype"]]
        InterTminute[:, 1] = np.where(np.isin(ptype_ecmwf, [5, 6, 9]), 1, 0)
        InterTminute[:, 2] = np.where(np.isin(ptype_ecmwf, [4, 8, 10]), 1, 0)
        InterTminute[:, 3] = np.where(np.isin(ptype_ecmwf, [3, 12]), 1, 0)
        InterTminute[:, 4] = np.where(np.isin(ptype_ecmwf, [1, 2, 7, 11]), 1, 0)
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
            InterTminute[:, i - 3] = gefsMinuteInterpolation[:, i]
    elif "gfs" in source_list and gfsMinuteInterpolation is not None:
        for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
            InterTminute[:, i - 11] = gfsMinuteInterpolation[:, i]
    elif "era5" in source_list and era5_MinuteInterpolation is not None:
        ptype_era5 = era5_MinuteInterpolation[:, ERA5["precipitation_type"]]
        InterTminute[:, 1] = np.where(np.isin(ptype_era5, [5, 6, 9]), 1, 0)
        InterTminute[:, 2] = np.where(np.isin(ptype_era5, [4, 8, 10]), 1, 0)
        InterTminute[:, 3] = np.where(np.isin(ptype_era5, [3, 12]), 1, 0)
        InterTminute[:, 4] = np.where(np.isin(ptype_era5, [1, 2, 7, 11]), 1, 0)

    maxPchance = (
        np.argmax(InterTminute, axis=1)
        if not np.any(np.isnan(InterTminute))
        else np.full(len(minute_array_grib), 5)
    )
    pTypes = ["none", "snow", "sleet", "sleet", "rain", MISSING_DATA]
    pTypesText = ["Clear", "Snow", "Sleet", "Sleet", "Rain", MISSING_DATA]
    pTypesIcon = ["clear", "snow", "sleet", "sleet", "rain", MISSING_DATA]

    minuteType = [pTypes[maxPchance[idx]] for idx in range(61)]
    precipTypes = np.array(minuteType)

    if "hrrrsubh" in source_list:
        temp_arr = hrrrSubHInterpolation[:, HRRR_SUBH["temp"]]
        refc_arr = hrrrSubHInterpolation[:, HRRR_SUBH["refc"]]

        mask = (precipTypes == "none") & (refc_arr > 0)
        precipTypes[mask] = np.where(
            temp_arr[mask] >= TEMP_THRESHOLD_RAIN_C,
            "rain",
            np.where(temp_arr[mask] <= TEMP_THRESHOLD_SNOW_C, "snow", "sleet"),
        )
        # Keep precipTypes as a numpy array; no need to convert to list and back
        # precipTypes = np.array(minuteType)

        InterPminute[:, DATA_MINUTELY["intensity"]] = dbz_to_rate(refc_arr, precipTypes)
    elif "nbm" in source_list:
        InterPminute[:, DATA_MINUTELY["intensity"]] = nbmMinuteInterpolation[
            :, NBM["accum"]
        ]
    elif "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            ecmwfMinuteInterpolation[:, ECMWF["intensity"]] * 3600
        )
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = gefsMinuteInterpolation[
            :, GEFS["accum"]
        ]
    elif "gfs" in source_list and gfsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = dbz_to_rate(
            gfsMinuteInterpolation[:, GFS["refc"]], precipTypes
        )
    elif "era5" in source_list and era5_MinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            era5_MinuteInterpolation[
                :, ERA5["large_scale_snowfall_rate_water_equivalent"]
            ]
            + era5_MinuteInterpolation[
                :, ERA5["convective_snowfall_rate_water_equivalent"]
            ]
            + era5_MinuteInterpolation[:, ERA5["large_scale_rain_rate"]]
            + era5_MinuteInterpolation[:, ERA5["convective_rain_rate"]]
        ) * 3600

    if "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["error"]] = (
            ecmwfMinuteInterpolation[:, ECMWF["accum_stddev"]] * 1000
        )
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["error"]] = gefsMinuteInterpolation[
            :, GEFS["error"]
        ]
    else:
        InterPminute[:, DATA_MINUTELY["error"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    minuteKeys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
    ]
    if version >= 2:
        minuteKeys += ["rainIntensity", "snowIntensity", "sleetIntensity"]

    InterPminute[:, DATA_MINUTELY["rain_intensity"]] = 0
    InterPminute[:, DATA_MINUTELY["snow_intensity"]] = 0
    InterPminute[:, DATA_MINUTELY["ice_intensity"]] = 0

    rain_mask_min = maxPchance == PRECIP_IDX["rain"]
    InterPminute[rain_mask_min, DATA_MINUTELY["rain_intensity"]] = InterPminute[
        rain_mask_min, DATA_MINUTELY["intensity"]
    ]

    snow_mask_min = maxPchance == PRECIP_IDX["snow"]
    InterPminute[snow_mask_min, DATA_MINUTELY["snow_intensity"]] = (
        InterPminute[snow_mask_min, DATA_MINUTELY["intensity"]] * 10
    )

    sleet_mask_min = (maxPchance == PRECIP_IDX["ice"]) | (
        maxPchance == PRECIP_IDX["sleet"]
    )
    InterPminute[sleet_mask_min, DATA_MINUTELY["ice_intensity"]] = InterPminute[
        sleet_mask_min, DATA_MINUTELY["intensity"]
    ]

    minuteTimes = InterPminute[:, DATA_MINUTELY["time"]]
    minuteIntensity = np.maximum(InterPminute[:, DATA_MINUTELY["intensity"]], 0)
    minuteProbability = np.minimum(
        np.maximum(InterPminute[:, DATA_MINUTELY["prob"]], 0), 1
    )
    minuteIntensityError = np.maximum(InterPminute[:, DATA_MINUTELY["error"]], 0)

    minuteRainIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["rain_intensity"]], 0
    )
    minuteSnowIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["snow_intensity"]], 0
    )
    minuteSleetIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["ice_intensity"]], 0
    )

    minuteRainIntensity[np.abs(minuteRainIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0
    minuteSnowIntensity[np.abs(minuteSnowIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0
    minuteSleetIntensity[np.abs(minuteSleetIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = (
        0.0
    )
    minuteProbability[np.abs(minuteProbability) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0
    minuteIntensityError[np.abs(minuteIntensityError) < PRECIP_NOISE_THRESHOLD_MMH] = (
        0.0
    )
    minuteIntensity[np.abs(minuteIntensity) < PRECIP_NOISE_THRESHOLD_MMH] = 0.0

    zero_type_mask = maxPchance == 0
    minuteRainIntensity[zero_type_mask] = 0.0
    minuteSnowIntensity[zero_type_mask] = 0.0
    minuteSleetIntensity[zero_type_mask] = 0.0
    minuteProbability[zero_type_mask] = 0.0
    minuteIntensityError[zero_type_mask] = 0.0
    minuteIntensity[zero_type_mask] = 0.0

    minuteIntensity_display = np.round(minuteIntensity * prep_intensity_unit, 4)
    minuteIntensityError_display = np.round(
        minuteIntensityError * prep_intensity_unit, 4
    )
    minuteRainIntensity_display = np.round(minuteRainIntensity * prep_intensity_unit, 4)
    minuteSnowIntensity_display = np.round(minuteSnowIntensity * prep_intensity_unit, 4)
    minuteSleetIntensity_display = np.round(
        minuteSleetIntensity * prep_intensity_unit, 4
    )
    minuteProbability_display = np.round(minuteProbability, 2)

    minuteItems = []
    minuteItems_si = []
    all_minute_keys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
        "rainIntensity",
        "snowIntensity",
        "sleetIntensity",
    ]
    for idx in range(61):
        values = [
            int(minuteTimes[idx]),
            float(minuteIntensity_display[idx]),
            float(minuteProbability_display[idx]),
            float(minuteIntensityError_display[idx]),
            minuteType[idx],
        ]
        if version >= 2:
            values += [
                float(minuteRainIntensity_display[idx]),
                float(minuteSnowIntensity_display[idx]),
                float(minuteSleetIntensity_display[idx]),
            ]
        minuteItems.append(dict(zip(minuteKeys, values)))

        values_si = [
            int(minuteTimes[idx]),
            float(minuteIntensity[idx]),
            float(minuteProbability[idx]),
            float(minuteIntensityError[idx]),
            minuteType[idx],
            float(minuteRainIntensity[idx]),
            float(minuteSnowIntensity[idx]),
            float(minuteSleetIntensity[idx]),
        ]
        minuteItems_si.append(dict(zip(all_minute_keys, values_si)))

    return (
        InterPminute,
        InterTminute,
        minuteItems,
        minuteItems_si,
        maxPchance,
        pTypesText,
        pTypesIcon,
        hrrrSubHInterpolation,
    )
