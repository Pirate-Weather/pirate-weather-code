import logging
from typing import Any, Dict, List, Optional

import numpy as np

from API.api_utils import fast_nearest_interp  # Using api_utils for now as it was there
from API.constants.forecast_const import DATA_MINUTELY
from API.constants.model_const import (
    ECMWF,
    ERA5,
    GEFS,
    GFS,
    HRRR,
    HRRR_SUBH,
    NBM,
)
from API.constants.shared_const import (
    MISSING_DATA,
    PRECIP_IDX,
    PRECIP_NOISE_THRESHOLD_MMH,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
)
from API.processing.utils import dbz_to_rate

logger = logging.getLogger("pirate-weather-api")


def process_minutely(
    sourceList: List[str],
    minute_array_grib: np.ndarray,
    version: float,
    prepIntensityUnit: float,
    dataOut_gefs: Optional[np.ndarray] = None,
    dataOut_gfs: Optional[np.ndarray] = None,
    dataOut_ecmwf: Optional[np.ndarray] = None,
    dataOut_nbm: Optional[np.ndarray] = None,
    dataOut_hrrr_subh: Optional[np.ndarray] = None,  # dataOut in original
    HRRR_Merged: Optional[np.ndarray] = None,
    ERA5_MERGED: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    # Initialize interpolation arrays
    gefsMinuteInterpolation = None
    gfsMinuteInterpolation = None
    ecmwfMinuteInterpolation = None
    nbmMinuteInterpolation = None
    hrrrSubHInterpolation = None
    era5_MinuteInterpolation = None

    if "gefs" in sourceList and dataOut_gefs is not None:
        gefsMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_gefs[0, :]))
        )

    if "gfs" in sourceList and dataOut_gfs is not None:
        gfsMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_gfs[0, :]))
        )

    if "ecmwf_ifs" in sourceList and dataOut_ecmwf is not None:
        ecmwfMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_ecmwf[0, :]))
        )

    if (
        "nbm" in sourceList
    ):  # NBM is always initialized in original code if sourceList check passes, but dataOut_nbm might be needed
        nbmMinuteInterpolation = np.zeros((len(minute_array_grib), 18))

    # Interpolate for minutely
    if "hrrrsubh" in sourceList and dataOut_hrrr_subh is not None:
        hrrrSubHInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_hrrr_subh[0, :]))
        )
        for i in range(len(dataOut_hrrr_subh[0, :]) - 1):
            hrrrSubHInterpolation[:, i + 1] = np.interp(
                minute_array_grib,
                dataOut_hrrr_subh[:, 0].squeeze(),
                dataOut_hrrr_subh[:, i + 1],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        # Check for nan, which means SubH is out of range, and fall back to regular HRRR
        if np.isnan(hrrrSubHInterpolation[1, 1]) and HRRR_Merged is not None:
            # Mapping HRRR columns to HRRR_SUBH columns for fallback
            # Note: This assumes HRRR_Merged has the correct columns.
            # The original code manually maps specific columns.

            hrrrSubHInterpolation[:, HRRR_SUBH["gust"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["gust"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["pressure"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["pressure"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["temp"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["temp"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["dew"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["dew"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["wind_u"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["wind_u"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["wind_v"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["wind_v"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["intensity"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["intensity"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["snow"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["snow"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["ice"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["ice"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["freezing_rain"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["freezing_rain"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["rain"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["rain"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["refc"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["refc"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["solar"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["solar"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["vis"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["vis"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        if (
            "gefs" in sourceList
            and gefsMinuteInterpolation is not None
            and dataOut_gefs is not None
        ):
            gefsMinuteInterpolation[:, GEFS["error"]] = np.interp(
                minute_array_grib,
                dataOut_gefs[:, 0].squeeze(),
                dataOut_gefs[:, GEFS["error"]],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

    else:  # Use GFS/GEFS
        if (
            "gefs" in sourceList
            and gefsMinuteInterpolation is not None
            and dataOut_gefs is not None
        ):
            for i in range(len(dataOut_gefs[0, :]) - 1):
                gefsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_gefs[:, 0].squeeze(),
                    dataOut_gefs[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

        if (
            "gfs" in sourceList
            and gfsMinuteInterpolation is not None
            and dataOut_gfs is not None
        ):
            for i in range(len(dataOut_gfs[0, :]) - 1):
                gfsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_gfs[:, 0].squeeze(),
                    dataOut_gfs[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

    if (
        "ecmwf_ifs" in sourceList
        and ecmwfMinuteInterpolation is not None
        and dataOut_ecmwf is not None
    ):
        for i in range(len(dataOut_ecmwf[0, :]) - 1):
            if i + 1 == ECMWF["ptype"]:
                ecmwfMinuteInterpolation[:, i + 1] = fast_nearest_interp(
                    minute_array_grib,
                    dataOut_ecmwf[:, 0].squeeze(),
                    dataOut_ecmwf[:, i + 1],
                )
            else:
                ecmwfMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_ecmwf[:, 0].squeeze(),
                    dataOut_ecmwf[:, i + 1],
                    left=MISSING_DATA,
                    right=MISSING_DATA,
                )

    if (
        "nbm" in sourceList
        and nbmMinuteInterpolation is not None
        and dataOut_nbm is not None
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
                dataOut_nbm[:, 0].squeeze(),
                dataOut_nbm[:, i],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

    if "era5" in sourceList and ERA5_MERGED is not None:
        era5_MinuteInterpolation = np.zeros(
            (len(minute_array_grib), max(ERA5.values()) + 1)
        )  # +1 to be safe with max index
        for i in [
            ERA5["large_scale_rain_rate"],
            ERA5["convective_rain_rate"],
            ERA5["large_scale_snowfall_rate_water_equivalent"],
            ERA5["convective_snowfall_rate_water_equivalent"],
        ]:
            era5_MinuteInterpolation[:, i] = np.interp(
                minute_array_grib,
                ERA5_MERGED[:, 0].squeeze(),
                ERA5_MERGED[:, i],
                left=MISSING_DATA,
                right=MISSING_DATA,
            )

        era5_MinuteInterpolation[:, ERA5["precipitation_type"]] = fast_nearest_interp(
            minute_array_grib,
            ERA5_MERGED[:, 0].squeeze(),
            ERA5_MERGED[:, ERA5["precipitation_type"]],
        )
        era5_MinuteInterpolation[:, ERA5["prob"]] = np.interp(
            minute_array_grib,
            ERA5_MERGED[:, 0].squeeze(),
            ERA5_MERGED[:, ERA5["prob"]],
            left=MISSING_DATA,
            right=MISSING_DATA,
        )

    # InterPminute calculation
    InterPminute = np.full(
        (len(minute_array_grib), max(DATA_MINUTELY.values()) + 1), MISSING_DATA
    )
    InterPminute[:, DATA_MINUTELY["time"]] = minute_array_grib

    InterTminute = np.zeros((len(minute_array_grib), 5))  # Type

    # precipProbability
    if "nbm" in sourceList and nbmMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            nbmMinuteInterpolation[:, NBM["prob"]] * 0.01
        )
    elif "ecmwf_ifs" in sourceList and ecmwfMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["prob"]] = ecmwfMinuteInterpolation[
            :, ECMWF["prob"]
        ]
    elif "gefs" in sourceList and gefsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["prob"]] = gefsMinuteInterpolation[
            :, GEFS["prob"]
        ]
    elif "era5" in sourceList and era5_MinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            era5_MinuteInterpolation[:, ERA5["prob"]] * 0.01
        )
    else:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    # Less than 5% set to 0
    InterPminute[
        InterPminute[:, DATA_MINUTELY["prob"]] < 0.05, DATA_MINUTELY["prob"]
    ] = 0

    # Precipitation Type
    if "hrrrsubh" in sourceList and hrrrSubHInterpolation is not None:
        for i in [
            HRRR_SUBH["snow"],
            HRRR_SUBH["ice"],
            HRRR_SUBH["freezing_rain"],
            HRRR_SUBH["rain"],
        ]:
            InterTminute[:, i - 7] = hrrrSubHInterpolation[:, i]
    elif "nbm" in sourceList and nbmMinuteInterpolation is not None:
        InterTminute[:, 1] = nbmMinuteInterpolation[:, NBM["snow"]]
        InterTminute[:, 2] = nbmMinuteInterpolation[:, NBM["ice"]]
        InterTminute[:, 3] = nbmMinuteInterpolation[:, NBM["freezing_rain"]]
        InterTminute[:, 4] = nbmMinuteInterpolation[:, NBM["rain"]]
    elif "ecmwf_ifs" in sourceList and ecmwfMinuteInterpolation is not None:
        ptype_ecmwf = ecmwfMinuteInterpolation[:, ECMWF["ptype"]]
        InterTminute[:, 1] = np.where(np.isin(ptype_ecmwf, [5, 6, 9]), 1, 0)
        InterTminute[:, 2] = np.where(np.isin(ptype_ecmwf, [4, 8, 10]), 1, 0)
        InterTminute[:, 3] = np.where(np.isin(ptype_ecmwf, [3, 12]), 1, 0)
        InterTminute[:, 4] = np.where(np.isin(ptype_ecmwf, [1, 2, 7, 11]), 1, 0)
    elif "gefs" in sourceList and gefsMinuteInterpolation is not None:
        for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
            InterTminute[:, i - 3] = gefsMinuteInterpolation[:, i]
    elif "gfs" in sourceList and gfsMinuteInterpolation is not None:
        for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
            InterTminute[:, i - 11] = gfsMinuteInterpolation[:, i]
    elif "era5" in sourceList and era5_MinuteInterpolation is not None:
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

    minuteType = [pTypes[maxPchance[idx]] for idx in range(len(minute_array_grib))]
    precipTypes = np.array(minuteType)

    # Intensity
    if "hrrrsubh" in sourceList and hrrrSubHInterpolation is not None:
        temp_arr = hrrrSubHInterpolation[:, HRRR_SUBH["temp"]]
        refc_arr = hrrrSubHInterpolation[:, HRRR_SUBH["refc"]]
        mask = (precipTypes == "none") & (refc_arr > 0)
        precipTypes[mask] = np.where(
            temp_arr[mask] >= TEMP_THRESHOLD_RAIN_C,
            "rain",
            np.where(temp_arr[mask] <= TEMP_THRESHOLD_SNOW_C, "snow", "sleet"),
        )
        minuteType = precipTypes.tolist()
        precipTypes = np.array(minuteType)
        InterPminute[:, DATA_MINUTELY["intensity"]] = dbz_to_rate(refc_arr, precipTypes)
    elif "nbm" in sourceList and nbmMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = nbmMinuteInterpolation[
            :, NBM["accum"]
        ]
    elif "ecmwf_ifs" in sourceList and ecmwfMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            ecmwfMinuteInterpolation[:, ECMWF["intensity"]] * 3600
        )
    elif "gefs" in sourceList and gefsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = gefsMinuteInterpolation[
            :, GEFS["accum"]
        ]
    elif "gfs" in sourceList and gfsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["intensity"]] = dbz_to_rate(
            gfsMinuteInterpolation[:, GFS["refc"]], precipTypes
        )
    elif "era5" in sourceList and era5_MinuteInterpolation is not None:
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

    # Error
    if "ecmwf_ifs" in sourceList and ecmwfMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["error"]] = (
            ecmwfMinuteInterpolation[:, ECMWF["accum_stddev"]] * 1000
        )
    elif "gefs" in sourceList and gefsMinuteInterpolation is not None:
        InterPminute[:, DATA_MINUTELY["error"]] = gefsMinuteInterpolation[
            :, GEFS["error"]
        ]
    else:
        InterPminute[:, DATA_MINUTELY["error"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    # Calculate type-specific intensities
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

    # Prepare output arrays
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

    # Noise thresholding
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

    # Display units conversion
    minuteIntensity_display = np.round(minuteIntensity * prepIntensityUnit, 4)
    minuteIntensityError_display = np.round(minuteIntensityError * prepIntensityUnit, 4)
    minuteRainIntensity_display = np.round(minuteRainIntensity * prepIntensityUnit, 4)
    minuteSnowIntensity_display = np.round(minuteSnowIntensity * prepIntensityUnit, 4)
    minuteSleetIntensity_display = np.round(minuteSleetIntensity * prepIntensityUnit, 4)
    minuteProbability_display = np.round(minuteProbability, 2)

    minuteItems = []
    minuteItems_si = []
    minuteKeys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
    ]
    if version >= 2:
        minuteKeys += ["rainIntensity", "snowIntensity", "sleetIntensity"]

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

    for idx in range(len(minute_array_grib)):
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

    return {
        "minuteItems": minuteItems,
        "minuteItems_si": minuteItems_si,
        "InterPminute": InterPminute,
        "InterTminute": InterTminute,
        "hrrrSubHInterpolation": hrrrSubHInterpolation,
        "nbmMinuteInterpolation": nbmMinuteInterpolation,
        "ecmwfMinuteInterpolation": ecmwfMinuteInterpolation,
        "gefsMinuteInterpolation": gefsMinuteInterpolation,
        "gfsMinuteInterpolation": gfsMinuteInterpolation,
        "era5_MinuteInterpolation": era5_MinuteInterpolation,
        "minuteRainIntensity": minuteRainIntensity,
        "minuteSnowIntensity": minuteSnowIntensity,
        "minuteSleetIntensity": minuteSleetIntensity,
        "maxPchance": maxPchance,
    }
