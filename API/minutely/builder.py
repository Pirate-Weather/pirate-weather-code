"""Minutely interpolation and object construction helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from API.api_utils import (
    fast_nearest_interp,
    map_wmo4677_to_ptype,
    zero_small_values,
)
from API.constants.api_const import (
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
)
from API.constants.forecast_const import DATA_MINUTELY
from API.constants.model_const import (
    DWD_MOSMIX,
    ECMWF,
    ERA5,
    GEFS,
    GFS,
    HRRR,
    HRRR_SUBH,
    NBM,
)
from API.constants.shared_const import MISSING_DATA
from API.utils.precip import dbz_to_rate
from API.utils.source_priority import should_gfs_precede_dwd


def _interp_gefs(minute_array_grib, gefs_data):
    """
    Interpolate GEFS data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        gefs_data: GEFS data array.

    Returns:
        Interpolated GEFS data.
    """
    if gefs_data is None or len(gefs_data) == 0:
        return None

    gefsMinuteInterpolation = np.zeros((len(minute_array_grib), len(gefs_data[0, :])))
    for i in range(len(gefs_data[0, :]) - 1):
        gefsMinuteInterpolation[:, i + 1] = np.interp(
            minute_array_grib,
            gefs_data[:, 0].squeeze(),
            gefs_data[:, i + 1],
            left=MISSING_DATA,
            right=MISSING_DATA,
        )
    return gefsMinuteInterpolation


def _interp_gfs(minute_array_grib, gfs_data):
    """
    Interpolate GFS data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        gfs_data: GFS data array.

    Returns:
        Interpolated GFS data.
    """
    if gfs_data is None or len(gfs_data) == 0:
        return None

    gfsMinuteInterpolation = np.zeros((len(minute_array_grib), len(gfs_data[0, :])))
    for i in range(len(gfs_data[0, :]) - 1):
        gfsMinuteInterpolation[:, i + 1] = np.interp(
            minute_array_grib,
            gfs_data[:, 0].squeeze(),
            gfs_data[:, i + 1],
            left=MISSING_DATA,
            right=MISSING_DATA,
        )
    return gfsMinuteInterpolation


def _interp_ecmwf(minute_array_grib, ecmwf_data):
    """
    Interpolate ECMWF data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        ecmwf_data: ECMWF data array.

    Returns:
        Interpolated ECMWF data.
    """
    if ecmwf_data is None or len(ecmwf_data) == 0:
        return None

    ecmwfMinuteInterpolation = np.zeros((len(minute_array_grib), len(ecmwf_data[0, :])))
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
    return ecmwfMinuteInterpolation


def _interp_nbm(minute_array_grib, nbm_data):
    """
    Interpolate NBM data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        nbm_data: NBM data array.

    Returns:
        Interpolated NBM data.
    """
    if nbm_data is None:
        return None

    nbmMinuteInterpolation = np.zeros((len(minute_array_grib), 18))
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
    return nbmMinuteInterpolation


def _interp_hrrr(minute_array_grib, hrrr_subh_data, hrrr_merged):
    """
    Interpolate HRRR data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        hrrr_subh_data: HRRR sub-hourly data array.
        hrrr_merged: HRRR merged data array.

    Returns:
        Interpolated HRRR data.
    """
    if hrrr_subh_data is None:
        return None

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
    return hrrrSubHInterpolation


def _interp_era5(minute_array_grib, era5_data):
    """
    Interpolate ERA5 data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        era5_data: ERA5 data array.

    Returns:
        Interpolated ERA5 data.
    """
    if era5_data is None or len(era5_data) == 0:
        return None

    era5_MinuteInterpolation = np.zeros((len(minute_array_grib), max(ERA5.values())))
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
    return era5_MinuteInterpolation


def _interp_dwd_mosmix(minute_array_grib, dwd_mosmix_data):
    """
    Interpolate DWD MOSMIX data to minutely intervals.

    Args:
        minute_array_grib: Minutely time array.
        dwd_mosmix_data: DWD MOSMIX data array.

    Returns:
        Interpolated DWD MOSMIX data.
    """
    if dwd_mosmix_data is None or len(dwd_mosmix_data) == 0:
        return None

    dwd_mosmix_MinuteInterpolation = np.zeros(
        (len(minute_array_grib), max(DWD_MOSMIX.values()) + 1)
    )

    # Interpolate precipitation accumulation
    dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["accum"]] = np.interp(
        minute_array_grib,
        dwd_mosmix_data[:, 0].squeeze(),
        dwd_mosmix_data[:, DWD_MOSMIX["accum"]],
        left=MISSING_DATA,
        right=MISSING_DATA,
    )

    # Interpolate precipitation accumulation
    dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["temp"]] = np.interp(
        minute_array_grib,
        dwd_mosmix_data[:, 0].squeeze(),
        dwd_mosmix_data[:, DWD_MOSMIX["temp"]],
        left=MISSING_DATA,
        right=MISSING_DATA,
    )

    # Use nearest neighbor for precipitation type (categorical data)
    dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["ptype"]] = fast_nearest_interp(
        minute_array_grib,
        dwd_mosmix_data[:, 0].squeeze(),
        dwd_mosmix_data[:, DWD_MOSMIX["ptype"]],
    )

    return dwd_mosmix_MinuteInterpolation


def _calculate_prob(
    minute_array_grib,
    source_list,
    nbmMinuteInterpolation,
    ecmwfMinuteInterpolation,
    gefsMinuteInterpolation,
):
    """
    Calculate precipitation probability.

    Args:
        minute_array_grib: Minutely time array.
        source_list: List of data sources.
        nbmMinuteInterpolation: NBM interpolated data.
        ecmwfMinuteInterpolation: ECMWF interpolated data.
        gefsMinuteInterpolation: GEFS interpolated data.

    Returns:
        Array of precipitation probabilities.
    """
    InterPminute_prob = np.full(len(minute_array_grib), MISSING_DATA)

    if "nbm" in source_list and nbmMinuteInterpolation is not None:
        InterPminute_prob = nbmMinuteInterpolation[:, NBM["prob"]] * 0.01
    elif "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        InterPminute_prob = ecmwfMinuteInterpolation[:, ECMWF["prob"]]
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        InterPminute_prob = gefsMinuteInterpolation[:, GEFS["prob"]]

    InterPminute_prob[InterPminute_prob < 0.05] = 0
    return InterPminute_prob


def _calculate_precip_type_probs(
    source_list,
    hrrrSubHInterpolation,
    nbmMinuteInterpolation,
    dwd_mosmix_MinuteInterpolation,
    ecmwfMinuteInterpolation,
    gefsMinuteInterpolation,
    gfsMinuteInterpolation,
    era5_MinuteInterpolation,
    lat,
    lon,
):
    """
    Calculate precipitation type probabilities.

    Args:
        source_list: List of data sources.
        hrrrSubHInterpolation: HRRR sub-hourly interpolated data.
        nbmMinuteInterpolation: NBM interpolated data.
        dwd_mosmix_MinuteInterpolation: DWD MOSMIX interpolated data.
        ecmwfMinuteInterpolation: ECMWF interpolated data.
        gefsMinuteInterpolation: GEFS interpolated data.
        gfsMinuteInterpolation: GFS interpolated data.
        era5_MinuteInterpolation: ERA5 interpolated data.
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        Array of precipitation type probabilities.
    """
    InterTminute = np.zeros((61, 5))

    # Helper functions to process each data source
    def _process_ecmwf():
        if "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
            ptype_ecmwf = ecmwfMinuteInterpolation[:, ECMWF["ptype"]]
            InterTminute[:, 1] = np.where(np.isin(ptype_ecmwf, [5, 6, 9]), 1, 0)
            InterTminute[:, 2] = np.where(np.isin(ptype_ecmwf, [4, 8, 10]), 1, 0)
            InterTminute[:, 3] = np.where(np.isin(ptype_ecmwf, [3, 12]), 1, 0)
            InterTminute[:, 4] = np.where(np.isin(ptype_ecmwf, [1, 2, 7, 11]), 1, 0)
            return True
        return False

    def _process_dwd_mosmix():
        if "dwd_mosmix" in source_list and dwd_mosmix_MinuteInterpolation is not None:
            # Map WMO 4677 codes to precip-type categories via centralized helper
            # Pass temperature for validation to prevent unrealistic frozen precip at warm temps
            ptype_dwd = dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["ptype"]]
            temp_dwd = dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["temp"]]
            mapped = map_wmo4677_to_ptype(np.round(ptype_dwd), temperature_c=temp_dwd)
            InterTminute[:, 1] = (mapped == 1).astype(int)
            InterTminute[:, 2] = (mapped == 2).astype(int)
            InterTminute[:, 3] = (mapped == 3).astype(int)
            InterTminute[:, 4] = (mapped == 4).astype(int)
            return True
        return False

    def _process_gfs():
        if "gfs" in source_list and gfsMinuteInterpolation is not None:
            for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
                InterTminute[:, i - 11] = gfsMinuteInterpolation[:, i]
            return True
        return False

    def _process_gefs():
        if "gefs" in source_list and gefsMinuteInterpolation is not None:
            for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
                InterTminute[:, i - 3] = gefsMinuteInterpolation[:, i]
            return True
        return False

    def _process_era5():
        if "era5" in source_list and era5_MinuteInterpolation is not None:
            ptype_era5 = era5_MinuteInterpolation[:, ERA5["precipitation_type"]]
            InterTminute[:, 1] = np.where(np.isin(ptype_era5, [5, 6, 9]), 1, 0)
            InterTminute[:, 2] = np.where(np.isin(ptype_era5, [4, 8, 10]), 1, 0)
            InterTminute[:, 3] = np.where(np.isin(ptype_era5, [3, 12]), 1, 0)
            InterTminute[:, 4] = np.where(np.isin(ptype_era5, [1, 2, 7, 11]), 1, 0)
            return True
        return False

    # Process high-priority sources first (same for all regions)
    if "hrrrsubh" in source_list and hrrrSubHInterpolation is not None:
        for i in [
            HRRR_SUBH["snow"],
            HRRR_SUBH["ice"],
            HRRR_SUBH["freezing_rain"],
            HRRR_SUBH["rain"],
        ]:
            InterTminute[:, i - 7] = hrrrSubHInterpolation[:, i]
        return InterTminute

    if "nbm" in source_list and nbmMinuteInterpolation is not None:
        InterTminute[:, 1] = nbmMinuteInterpolation[:, NBM["snow"]]
        InterTminute[:, 2] = nbmMinuteInterpolation[:, NBM["ice"]]
        InterTminute[:, 3] = nbmMinuteInterpolation[:, NBM["freezing_rain"]]
        InterTminute[:, 4] = nbmMinuteInterpolation[:, NBM["rain"]]
        return InterTminute

    # Determine priority order based on location
    # In North America: ECMWF > GFS > DWD MOSMIX > GEFS > ERA5
    # Rest of world: DWD MOSMIX > ECMWF > GFS > GEFS > ERA5
    gfs_before_dwd = should_gfs_precede_dwd(lat, lon)

    if gfs_before_dwd:
        # North America priority
        processors = [_process_ecmwf, _process_gfs, _process_dwd_mosmix, _process_gefs, _process_era5]
    else:
        # Rest of world priority
        processors = [_process_dwd_mosmix, _process_ecmwf, _process_gfs, _process_gefs, _process_era5]

    # Try each processor in priority order
    for processor in processors:
        if processor():
            break

    return InterTminute


def _calculate_intensity(
    source_list,
    precipTypes,
    hrrrSubHInterpolation,
    nbmMinuteInterpolation,
    dwd_mosmix_MinuteInterpolation,
    ecmwfMinuteInterpolation,
    gefsMinuteInterpolation,
    gfsMinuteInterpolation,
    era5_MinuteInterpolation,
):
    """
    Calculate precipitation intensity.

    Args:
        source_list: List of data sources.
        precipTypes: Array of precipitation types.
        hrrrSubHInterpolation: HRRR sub-hourly interpolated data.
        nbmMinuteInterpolation: NBM interpolated data.
        dwd_mosmix_MinuteInterpolation: DWD MOSMIX interpolated data.
        ecmwfMinuteInterpolation: ECMWF interpolated data.
        gefsMinuteInterpolation: GEFS interpolated data.
        gfsMinuteInterpolation: GFS interpolated data.
        era5_MinuteInterpolation: ERA5 interpolated data.

    Returns:
        Tuple containing intensity array and updated precipitation types.
    """
    intensity = np.full(len(precipTypes), MISSING_DATA)

    if "hrrrsubh" in source_list and hrrrSubHInterpolation is not None:
        temp_arr = hrrrSubHInterpolation[:, HRRR_SUBH["temp"]]
        refc_arr = hrrrSubHInterpolation[:, HRRR_SUBH["refc"]]
        mask = (precipTypes == "none") & (refc_arr > 0)
        precipTypes[mask] = np.where(
            temp_arr[mask] >= TEMP_THRESHOLD_RAIN_C,
            "rain",
            np.where(temp_arr[mask] <= TEMP_THRESHOLD_SNOW_C, "snow", "sleet"),
        )
        intensity = dbz_to_rate(refc_arr, precipTypes)
    elif "nbm" in source_list and nbmMinuteInterpolation is not None:
        intensity = nbmMinuteInterpolation[:, NBM["accum"]]
    elif "dwd_mosmix" in source_list and dwd_mosmix_MinuteInterpolation is not None:
        # DWD MOSMIX RR1c is in kg/m^2 = mm (hourly accumulation)
        intensity = np.nan_to_num(
            dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["accum"]]
        )
        temp_arr = dwd_mosmix_MinuteInterpolation[:, DWD_MOSMIX["temp"]]
        mask = (precipTypes == "none") & (intensity > 0)
        precipTypes[mask] = np.where(
            temp_arr[mask] >= TEMP_THRESHOLD_RAIN_C,
            "rain",
            np.where(temp_arr[mask] <= TEMP_THRESHOLD_SNOW_C, "snow", "sleet"),
        )
    elif "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        intensity = ecmwfMinuteInterpolation[:, ECMWF["intensity"]] * 3600
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        intensity = gefsMinuteInterpolation[:, GEFS["accum"]]
    elif "gfs" in source_list and gfsMinuteInterpolation is not None:
        intensity = dbz_to_rate(gfsMinuteInterpolation[:, GFS["refc"]], precipTypes)
    elif "era5" in source_list and era5_MinuteInterpolation is not None:
        intensity = (
            era5_MinuteInterpolation[
                :, ERA5["large_scale_snowfall_rate_water_equivalent"]
            ]
            + era5_MinuteInterpolation[
                :, ERA5["convective_snowfall_rate_water_equivalent"]
            ]
            + era5_MinuteInterpolation[:, ERA5["large_scale_rain_rate"]]
            + era5_MinuteInterpolation[:, ERA5["convective_rain_rate"]]
        ) * 3600

    return intensity, precipTypes


def _calculate_error(
    minute_array_grib,
    source_list,
    ecmwfMinuteInterpolation,
    gefsMinuteInterpolation,
):
    """
    Calculate precipitation intensity error.

    Args:
        minute_array_grib: Minutely time array.
        source_list: List of data sources.
        ecmwfMinuteInterpolation: ECMWF interpolated data.
        gefsMinuteInterpolation: GEFS interpolated data.

    Returns:
        Array of precipitation intensity errors.
    """
    error = np.ones(len(minute_array_grib)) * MISSING_DATA

    if "ecmwf_ifs" in source_list and ecmwfMinuteInterpolation is not None:
        error = ecmwfMinuteInterpolation[:, ECMWF["accum_stddev"]] * 1000
    elif "gefs" in source_list and gefsMinuteInterpolation is not None:
        error = gefsMinuteInterpolation[:, GEFS["error"]]

    return error


def _process_minute_items(
    InterPminute,
    minuteType,
    prep_intensity_unit,
    version,
):
    """
    Process minutely items for output.

    Args:
        InterPminute: Minutely interpolated data.
        minuteType: List of precipitation types.
        prep_intensity_unit: Precipitation intensity unit.
        version: API version.

    Returns:
        Tuple containing minute items and SI minute items.
    """
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

    minuteSleetIntensity = np.maximum(
        InterPminute[:, DATA_MINUTELY["ice_intensity"]], 0
    )

    minuteRainIntensity = zero_small_values(
        minuteRainIntensity, threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    minuteSnowIntensity = zero_small_values(
        minuteSnowIntensity, threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    minuteSleetIntensity = zero_small_values(
        minuteSleetIntensity, threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    minuteProbability = zero_small_values(
        minuteProbability, threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    minuteIntensityError = zero_small_values(
        minuteIntensityError, threshold=PRECIP_NOISE_THRESHOLD_MMH
    )
    minuteIntensity = zero_small_values(
        minuteIntensity, threshold=PRECIP_NOISE_THRESHOLD_MMH
    )

    # If type is none, zero out everything
    # We need to reconstruct maxPchance or pass it in?
    # Or just rely on minuteType being 'none'
    zero_type_mask = np.array(minuteType) == "none"

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

    minuteKeys = [
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
        minuteItems_si.append(dict(zip(minuteKeys, values_si)))

    return minuteItems, minuteItems_si


def build_minutely_block(
    *,
    minute_array_grib: np.ndarray,
    source_list: List[str],
    hrrr_subh_data: Optional[np.ndarray],
    hrrr_merged: Optional[np.ndarray],
    nbm_data: Optional[np.ndarray],
    dwd_mosmix_data: Optional[np.ndarray],
    gefs_data: Optional[np.ndarray],
    gfs_data: Optional[np.ndarray],
    ecmwf_data: Optional[np.ndarray],
    era5_data: Optional[np.ndarray],
    prep_intensity_unit: float,
    version: float,
    lat: float,
    lon: float,
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
    """
    Compute minutely interpolations and output objects.

    This function coordinates the interpolation of minutely weather data
    from various sources, calculates probabilities, intensities, and errors,
    and constructs the final minutely data arrays and lists.

    Args:
        minute_array_grib: Minutely time array.
        source_list: List of data sources.
        hrrr_subh_data: HRRR sub-hourly data.
        hrrr_merged: HRRR merged data.
        nbm_data: NBM data.
        dwd_mosmix_data: DWD MOSMIX data.
        gefs_data: GEFS data.
        gfs_data: GFS data.
        ecmwf_data: ECMWF data.
        era5_data: ERA5 data.
        prep_intensity_unit: Precipitation intensity unit.
        version: API version.
        lat: Latitude of the location.
        lon: Longitude of the location.

    Returns:
        Tuple containing minutely data arrays and lists.
    """
    InterPminute = np.full((61, max(DATA_MINUTELY.values()) + 1), MISSING_DATA)
    InterPminute[:, DATA_MINUTELY["time"]] = minute_array_grib

    # Interpolate data sources
    gefsMinuteInterpolation = (
        _interp_gefs(minute_array_grib, gefs_data) if "gefs" in source_list else None
    )
    gfsMinuteInterpolation = (
        _interp_gfs(minute_array_grib, gfs_data) if "gfs" in source_list else None
    )
    ecmwfMinuteInterpolation = (
        _interp_ecmwf(minute_array_grib, ecmwf_data)
        if "ecmwf_ifs" in source_list
        else None
    )
    nbmMinuteInterpolation = (
        _interp_nbm(minute_array_grib, nbm_data) if "nbm" in source_list else None
    )
    hrrrSubHInterpolation = (
        _interp_hrrr(minute_array_grib, hrrr_subh_data, hrrr_merged)
        if "hrrrsubh" in source_list
        else None
    )
    era5_MinuteInterpolation = (
        _interp_era5(minute_array_grib, era5_data) if "era5" in source_list else None
    )
    dwd_mosmix_MinuteInterpolation = (
        _interp_dwd_mosmix(minute_array_grib, dwd_mosmix_data)
        if "dwd_mosmix" in source_list
        else None
    )

    # Handle GEFS error interpolation inside HRRR block logic from original code
    # The original code updated gefsMinuteInterpolation inside the HRRR block.
    # We need to replicate that if it's critical.
    if "hrrrsubh" in source_list and hrrrSubHInterpolation is not None:
        if (
            "gefs" in source_list
            and gefsMinuteInterpolation is not None
            and gefs_data is not None
        ):
            gefsMinuteInterpolation[:, GEFS["error"]] = np.interp(
                minute_array_grib,
                gefs_data[:, 0].squeeze(),
                gefs_data[:, GEFS["error"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

    # Calculate Probability
    InterPminute[:, DATA_MINUTELY["prob"]] = _calculate_prob(
        minute_array_grib,
        source_list,
        nbmMinuteInterpolation,
        ecmwfMinuteInterpolation,
        gefsMinuteInterpolation,
    )

    # Calculate Precip Type Probs
    InterTminute = _calculate_precip_type_probs(
        source_list,
        hrrrSubHInterpolation,
        nbmMinuteInterpolation,
        dwd_mosmix_MinuteInterpolation,
        ecmwfMinuteInterpolation,
        gefsMinuteInterpolation,
        gfsMinuteInterpolation,
        era5_MinuteInterpolation,
        lat,
        lon,
    )

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

    # Calculate Intensity (and update precipTypes for HRRR/DWD MOSMIX temperature-based fallback)
    intensity, precipTypes = _calculate_intensity(
        source_list,
        precipTypes,
        hrrrSubHInterpolation,
        nbmMinuteInterpolation,
        dwd_mosmix_MinuteInterpolation,
        ecmwfMinuteInterpolation,
        gefsMinuteInterpolation,
        gfsMinuteInterpolation,
        era5_MinuteInterpolation,
    )
    InterPminute[:, DATA_MINUTELY["intensity"]] = intensity

    # Update minuteType list from updated precipTypes array (for HRRR/DWD MOSMIX logic)
    minuteType = precipTypes.tolist()

    # Recalculate maxPchance from updated precipTypes to ensure type-specific intensities
    # are distributed correctly when intensity calculation updates the precipitation type
    # (e.g., for DWD MOSMIX temperature-based fallback or HRRR radar-based typing)
    #
    # This recalculation is necessary because _calculate_intensity() may update precipTypes
    # based on temperature when WMO codes indicate "none" but accumulation > 0. Without this
    # recalculation, the type-specific intensities (rain/snow/ice) would all be zero even
    # though the total precipIntensity is non-zero.
    #
    # Note: If precipTypes contains MISSING_DATA (np.nan), it becomes string "nan" in the array
    # and will default to 0 (none) via dict.get(), which is the correct behavior
    ptype_to_idx = PRECIP_IDX.copy()  # Copy the dictionary
    maxPchance = np.array([ptype_to_idx.get(ptype, 0) for ptype in precipTypes])

    # Calculate Error
    InterPminute[:, DATA_MINUTELY["error"]] = _calculate_error(
        minute_array_grib,
        source_list,
        ecmwfMinuteInterpolation,
        gefsMinuteInterpolation,
    )

    # Distribute intensity to specific types
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

    # Process into output lists
    minuteItems, minuteItems_si = _process_minute_items(
        InterPminute,
        minuteType,
        prep_intensity_unit,
        version,
    )

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
