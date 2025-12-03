"""Currently block helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import metpy as mp
import numpy as np
from metpy.calc import relative_humidity_from_dewpoint

from API.api_utils import (
    calculate_apparent_temperature,
    clipLog,
    estimate_visibility_gultepe_rh_pr_numpy,
)
from API.constants.clip_const import (
    CLIP_CAPE,
    CLIP_CLOUD,
    CLIP_FIRE,
    CLIP_HUMIDITY,
    CLIP_OZONE,
    CLIP_PRESSURE,
    CLIP_SMOKE,
    CLIP_SOLAR,
    CLIP_TEMP,
    CLIP_UV,
    CLIP_VIS,
    CLIP_WIND,
)
from API.constants.forecast_const import (
    DATA_CURRENT,
    DATA_DAY,
    DATA_MINUTELY,
)
from API.constants.model_const import (
    ECMWF,
    ERA5,
    GFS,
    HRRR,
    HRRR_SUBH,
    NBM,
    NBM_FIRE_INDEX,
    RTMA_RU,
)
from API.constants.shared_const import MISSING_DATA
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    WIND_THRESHOLDS,
)
from API.PirateText import calculate_text
from API.PirateTextHelper import estimate_snow_height


@dataclass
class CurrentSection:
    """Container for currently results."""

    currently: dict
    interp_current: np.ndarray
    summary_key: Optional[str] = None


def build_current_section(
    *,
    sourceList,
    hour_array_grib: np.ndarray,
    minute_array_grib: np.ndarray,
    InterPminute: np.ndarray,
    minuteItems: list,
    minuteRainIntensity: np.ndarray,
    minuteSnowIntensity: np.ndarray,
    minuteSleetIntensity: np.ndarray,
    InterSday: np.ndarray,
    dayZeroRain: float,
    dayZeroSnow: float,
    dayZeroIce: float,
    prepAccumUnit: float,
    prepIntensityUnit: float,
    windUnit: float,
    visUnits: float,
    tempUnits: float,
    humidUnit: float,
    extraVars,
    summaryText: bool,
    translation,
    icon: str,
    unitSystem: str,
    version: int,
    timeMachine: bool,
    tmExtra: bool,
    lat: float,
    lon_IN: float,
    tz_name,
    tz_offset: float,
    ETOPO: float,
    elevUnit: float,
    dataOut_rtma_ru,
    hrrrSubHInterpolation,
    HRRR_Merged,
    NBM_Merged,
    ECMWF_Merged,
    GFS_Merged,
    ERA5_MERGED,
    NBM_Fire_Merged,
    logger,
    loc_tag: str,
    log_timing: Optional[Callable[[str], None]] = None,
    include_currently: bool = True,
) -> CurrentSection:
    """Calculate the currently block and return it alongside the raw array."""
    if log_timing:
        log_timing("Current Start")

    current_summary_key: Optional[str] = None

    if np.min(np.abs(hour_array_grib - minute_array_grib[0])) < 120:
        currentIDX_hrrrh = np.argmin(np.abs(hour_array_grib - minute_array_grib[0]))
        interpFac1 = 0
        interpFac2 = 1
    else:
        currentIDX_hrrrh = np.searchsorted(
            hour_array_grib, minute_array_grib[0], side="left"
        )

        interpFac1 = 1 - (
            abs(minute_array_grib[0] - hour_array_grib[currentIDX_hrrrh - 1])
            / (
                hour_array_grib[currentIDX_hrrrh]
                - hour_array_grib[currentIDX_hrrrh - 1]
            )
        )

        interpFac2 = 1 - (
            abs(minute_array_grib[0] - hour_array_grib[currentIDX_hrrrh])
            / (
                hour_array_grib[currentIDX_hrrrh]
                - hour_array_grib[currentIDX_hrrrh - 1]
            )
        )

    currentIDX_hrrrh_A = np.max((currentIDX_hrrrh - 1, 0))

    InterPcurrent = np.zeros(shape=max(DATA_CURRENT.values()) + 1)
    InterPcurrent[DATA_CURRENT["time"]] = int(minute_array_grib[0])

    has_hrrr_merged = (
        HRRR_Merged is not None
        and ("hrrr_0-18" in sourceList)
        and ("hrrr_18-48" in sourceList)
    )

    def select_value(strategies, default=MISSING_DATA):
        for predicate, getter in strategies:
            if predicate():
                return getter()
        return default

    def interp_scalar(merged, key):
        return (
            merged[currentIDX_hrrrh_A, key] * interpFac1
            + merged[currentIDX_hrrrh, key] * interpFac2
        )

    def interp_uv_magnitude(merged, u_key, v_key):
        return math.sqrt(
            (
                merged[currentIDX_hrrrh_A, u_key] * interpFac1
                + merged[currentIDX_hrrrh, u_key] * interpFac2
            )
            ** 2
            + (
                merged[currentIDX_hrrrh_A, v_key] * interpFac1
                + merged[currentIDX_hrrrh, v_key] * interpFac2
            )
            ** 2
        )

    def bearing_from_components(u_val, v_val):
        return np.rad2deg(np.mod(np.arctan2(u_val, v_val) + np.pi, 2 * np.pi))

    def calculate_ecmwf_relative_humidity():
        humid_fac1 = relative_humidity_from_dewpoint(
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["temp"]] * mp.units.units.degC,
            ECMWF_Merged[currentIDX_hrrrh_A, ECMWF["dew"]] * mp.units.units.degC,
            phase="auto",
        ).magnitude
        humid_fac2 = relative_humidity_from_dewpoint(
            ECMWF_Merged[currentIDX_hrrrh, ECMWF["temp"]] * mp.units.units.degC,
            ECMWF_Merged[currentIDX_hrrrh, ECMWF["dew"]] * mp.units.units.degC,
            phase="auto",
        ).magnitude

        return (humid_fac1 * interpFac1 + humid_fac2 * interpFac2) * 100 * humidUnit

    def calculate_era5_relative_humidity():
        humid_fac1 = relative_humidity_from_dewpoint(
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["2m_temperature"]]
            * mp.units.units.degC,
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["2m_dewpoint_temperature"]]
            * mp.units.units.degC,
            phase="auto",
        ).magnitude
        humid_fac2 = relative_humidity_from_dewpoint(
            ERA5_MERGED[currentIDX_hrrrh, ERA5["2m_temperature"]] * mp.units.units.degC,
            ERA5_MERGED[currentIDX_hrrrh, ERA5["2m_dewpoint_temperature"]]
            * mp.units.units.degC,
            phase="auto",
        ).magnitude

        return (humid_fac1 * interpFac1 + humid_fac2 * interpFac2) * 100 * humidUnit

    InterPcurrent[DATA_CURRENT["temp"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: dataOut_rtma_ru[0, RTMA_RU["temp"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: hrrrSubHInterpolation[0, HRRR_SUBH["temp"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["temp"]),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: interp_scalar(ECMWF_Merged, ECMWF["temp"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["temp"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(ERA5_MERGED, ERA5["2m_temperature"]),
            ),
        ]
    )

    InterPcurrent[DATA_CURRENT["temp"]] = clipLog(
        InterPcurrent[DATA_CURRENT["temp"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Temperature Current",
    )

    InterPcurrent[DATA_CURRENT["dew"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: dataOut_rtma_ru[0, RTMA_RU["dew"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: hrrrSubHInterpolation[0, HRRR_SUBH["dew"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["dew"]),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: interp_scalar(ECMWF_Merged, ECMWF["dew"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["dew"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(ERA5_MERGED, ERA5["2m_dewpoint_temperature"]),
            ),
        ]
    )

    InterPcurrent[DATA_CURRENT["dew"]] = clipLog(
        InterPcurrent[DATA_CURRENT["dew"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Dewpoint Current",
    )

    InterPcurrent[DATA_CURRENT["humidity"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: dataOut_rtma_ru[0, RTMA_RU["humidity"]],
            ),
            (
                lambda: has_hrrr_merged,
                lambda: interp_scalar(HRRR_Merged, HRRR["humidity"]) * humidUnit,
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["humidity"]) * humidUnit,
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["humidity"]) * humidUnit,
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                calculate_ecmwf_relative_humidity,
            ),
            (lambda: "era5" in sourceList, calculate_era5_relative_humidity),
        ]
    )

    InterPcurrent[DATA_CURRENT["humidity"]] = clipLog(
        InterPcurrent[DATA_CURRENT["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
        "Humidity Current",
    )

    InterPcurrent[DATA_CURRENT["pressure"]] = select_value(
        [
            (
                lambda: has_hrrr_merged,
                lambda: interp_scalar(HRRR_Merged, HRRR["pressure"]),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: interp_scalar(ECMWF_Merged, ECMWF["pressure"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["pressure"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(
                    ERA5_MERGED,
                    ERA5["mean_sea_level_pressure"],
                ),
            ),
        ]
    )

    InterPcurrent[DATA_CURRENT["pressure"]] = clipLog(
        InterPcurrent[DATA_CURRENT["pressure"]],
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Pressure Current",
    )

    InterPcurrent[DATA_CURRENT["wind"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: math.sqrt(
                    dataOut_rtma_ru[0, RTMA_RU["wind_u"]] ** 2
                    + dataOut_rtma_ru[0, RTMA_RU["wind_v"]] ** 2
                ),
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: math.sqrt(
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_u"]] ** 2
                    + hrrrSubHInterpolation[0, HRRR_SUBH["wind_v"]] ** 2
                ),
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["wind"]),
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: interp_uv_magnitude(
                    ECMWF_Merged,
                    ECMWF["wind_u"],
                    ECMWF["wind_v"],
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_uv_magnitude(GFS_Merged, GFS["wind_u"], GFS["wind_v"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_uv_magnitude(
                    ERA5_MERGED,
                    ERA5["10m_u_component_of_wind"],
                    ERA5["10m_v_component_of_wind"],
                ),
            ),
        ]
    )

    InterPcurrent[DATA_CURRENT["wind"]] = clipLog(
        InterPcurrent[DATA_CURRENT["wind"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "WindSpeed Current",
    )

    InterPcurrent[DATA_CURRENT["gust"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: dataOut_rtma_ru[0, RTMA_RU["gust"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: hrrrSubHInterpolation[0, HRRR_SUBH["gust"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["gust"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["gust"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(
                    ERA5_MERGED,
                    ERA5["instantaneous_10m_wind_gust"],
                ),
            ),
        ],
        default=MISSING_DATA,
    )

    InterPcurrent[DATA_CURRENT["gust"]] = clipLog(
        InterPcurrent[DATA_CURRENT["gust"]],
        CLIP_WIND["min"],
        CLIP_WIND["max"],
        "Gust Current",
    )

    if "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["intensity"]] = (
            (
                ERA5_MERGED[currentIDX_hrrrh_A, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh_A, ERA5["convective_rain_rate"]]
                + ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["large_scale_snowfall_rate_water_equivalent"],
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["convective_snowfall_rate_water_equivalent"],
                ]
            )
            * interpFac1
            + (
                ERA5_MERGED[currentIDX_hrrrh, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh, ERA5["convective_rain_rate"]]
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["large_scale_snowfall_rate_water_equivalent"]
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["convective_snowfall_rate_water_equivalent"]
                ]
            )
            * interpFac2
        ) * 3600

        InterPcurrent[DATA_CURRENT["rain_intensity"]] = (
            (
                ERA5_MERGED[currentIDX_hrrrh_A, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh_A, ERA5["convective_rain_rate"]]
            )
            * interpFac1
            + (
                ERA5_MERGED[currentIDX_hrrrh, ERA5["large_scale_rain_rate"]]
                + ERA5_MERGED[currentIDX_hrrrh, ERA5["convective_rain_rate"]]
            )
            * interpFac2
        ) * 3600

        era5_current_snow_we = (
            (
                ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["large_scale_snowfall_rate_water_equivalent"],
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh_A,
                    ERA5["convective_snowfall_rate_water_equivalent"],
                ]
            )
            * interpFac1
            + (
                ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["large_scale_snowfall_rate_water_equivalent"]
                ]
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["convective_snowfall_rate_water_equivalent"]
                ]
            )
            * interpFac2
        ) * 3600

        InterPcurrent[DATA_CURRENT["snow_intensity"]] = estimate_snow_height(
            np.array([era5_current_snow_we]),
            np.array([InterPcurrent[DATA_CURRENT["temp"]]]),
            np.array([InterPcurrent[DATA_CURRENT["wind"]]]),
        )[0]

        InterPcurrent[DATA_CURRENT["ice_intensity"]] = 0
    else:
        InterPcurrent[DATA_CURRENT["intensity"]] = InterPminute[
            0, DATA_MINUTELY["intensity"]
        ]
        InterPcurrent[DATA_CURRENT["prob"]] = InterPminute[0, DATA_MINUTELY["prob"]]
        InterPcurrent[DATA_CURRENT["error"]] = InterPminute[0, DATA_MINUTELY["error"]]

    InterPcurrent[DATA_CURRENT["bearing"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: bearing_from_components(
                    dataOut_rtma_ru[0, RTMA_RU["wind_u"]],
                    dataOut_rtma_ru[0, RTMA_RU["wind_v"]],
                ),
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: bearing_from_components(
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_u"]],
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_v"]],
                ),
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: NBM_Merged[currentIDX_hrrrh, NBM["bearing"]],
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: bearing_from_components(
                    ECMWF_Merged[currentIDX_hrrrh, ECMWF["wind_u"]],
                    ECMWF_Merged[currentIDX_hrrrh, ECMWF["wind_v"]],
                ),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: bearing_from_components(
                    GFS_Merged[currentIDX_hrrrh, GFS["wind_u"]],
                    GFS_Merged[currentIDX_hrrrh, GFS["wind_v"]],
                ),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: bearing_from_components(
                    ERA5_MERGED[currentIDX_hrrrh, ERA5["10m_u_component_of_wind"]],
                    ERA5_MERGED[currentIDX_hrrrh, ERA5["10m_v_component_of_wind"]],
                ),
            ),
        ],
        default=MISSING_DATA,
    )

    InterPcurrent[DATA_CURRENT["cloud"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: dataOut_rtma_ru[0, RTMA_RU["cloud"]] * 0.01,
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["cloud"]) * 0.01,
            ),
            (
                lambda: "ecmwf_ifs" in sourceList,
                lambda: interp_scalar(ECMWF_Merged, ECMWF["cloud"]) * 0.01,
            ),
            (
                lambda: has_hrrr_merged,
                lambda: interp_scalar(HRRR_Merged, HRRR["cloud"]) * 0.01,
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["cloud"]) * 0.01,
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(
                    ERA5_MERGED,
                    ERA5["total_cloud_cover"],
                ),
            ),
        ],
        default=MISSING_DATA,
    )

    InterPcurrent[DATA_CURRENT["cloud"]] = clipLog(
        InterPcurrent[DATA_CURRENT["cloud"]],
        CLIP_CLOUD["min"],
        CLIP_CLOUD["max"],
        "Cloud Current",
    )

    if "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["uv"]] = clipLog(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["uv"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["uv"]] * interpFac2
            )
            * 18.9
            * 0.025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )
    elif "era5" in sourceList:
        InterPcurrent[DATA_CURRENT["uv"]] = clipLog(
            (
                ERA5_MERGED[
                    currentIDX_hrrrh_A, ERA5["downward_uv_radiation_at_the_surface"]
                ]
                * interpFac1
                + ERA5_MERGED[
                    currentIDX_hrrrh, ERA5["downward_uv_radiation_at_the_surface"]
                ]
                * interpFac2
            )
            / 3600
            * 40
            * 0.0025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Current",
        )
    else:
        InterPcurrent[DATA_CURRENT["uv"]] = MISSING_DATA

    station_pressure_value = MISSING_DATA
    if "rtma_ru" in sourceList:
        station_pressure_value = dataOut_rtma_ru[0, RTMA_RU["pressure"]]
    elif "gfs" in sourceList:
        station_pressure_value = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["station_pressure"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["station_pressure"]] * interpFac2
        )
    elif "era5" in sourceList:
        station_pressure_value = (
            ERA5_MERGED[currentIDX_hrrrh_A, ERA5["surface_pressure"]] * interpFac1
            + ERA5_MERGED[currentIDX_hrrrh, ERA5["surface_pressure"]] * interpFac2
        )

    InterPcurrent[DATA_CURRENT["station_pressure"]] = clipLog(
        station_pressure_value,
        CLIP_PRESSURE["min"],
        CLIP_PRESSURE["max"],
        "Station Pressure Current",
    )

    InterPcurrent[DATA_CURRENT["vis"]] = select_value(
        [
            (
                lambda: "rtma_ru" in sourceList,
                lambda: 16090
                if dataOut_rtma_ru[0, RTMA_RU["vis"]] >= 15999
                else dataOut_rtma_ru[0, RTMA_RU["vis"]],
            ),
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: hrrrSubHInterpolation[0, HRRR_SUBH["vis"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["vis"]),
            ),
            (
                lambda: has_hrrr_merged,
                lambda: interp_scalar(HRRR_Merged, HRRR["vis"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["vis"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: estimate_visibility_gultepe_rh_pr_numpy(
                    ERA5_MERGED[currentIDX_hrrrh_A, :] * interpFac1
                    + ERA5_MERGED[currentIDX_hrrrh, :] * interpFac2,
                    var_index=ERA5,
                    var_axis=1,
                ),
            ),
        ],
        default=MISSING_DATA,
    )

    InterPcurrent[DATA_CURRENT["vis"]] = np.clip(
        InterPcurrent[DATA_CURRENT["vis"]], CLIP_VIS["min"], CLIP_VIS["max"]
    )

    InterPcurrent[DATA_CURRENT["ozone"]] = clipLog(
        select_value(
            [
                (
                    lambda: "gfs" in sourceList,
                    lambda: interp_scalar(GFS_Merged, GFS["ozone"]),
                ),
                (
                    lambda: "era5" in sourceList,
                    lambda: interp_scalar(
                        ERA5_MERGED,
                        ERA5["total_column_ozone"],
                    )
                    * 46696,
                ),
            ],
            default=MISSING_DATA,
        ),
        CLIP_OZONE["min"],
        CLIP_OZONE["max"],
        "Ozone Current",
    )

    if "gfs" in sourceList:
        InterPcurrent[DATA_CURRENT["storm_dist"]] = np.maximum(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["storm_dist"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["storm_dist"]] * interpFac2
            ),
            0,
        )
        InterPcurrent[DATA_CURRENT["storm_dir"]] = GFS_Merged[
            currentIDX_hrrrh, GFS["storm_dir"]
        ]
    else:
        InterPcurrent[DATA_CURRENT["storm_dist"]] = MISSING_DATA
        InterPcurrent[DATA_CURRENT["storm_dir"]] = MISSING_DATA

    if has_hrrr_merged:
        InterPcurrent[DATA_CURRENT["smoke"]] = clipLog(
            (
                HRRR_Merged[currentIDX_hrrrh_A, HRRR["smoke"]] * interpFac1
                + HRRR_Merged[currentIDX_hrrrh, HRRR["smoke"]] * interpFac2
            ),
            CLIP_SMOKE["min"],
            CLIP_SMOKE["max"],
            "Smoke Current",
        )
    else:
        InterPcurrent[DATA_CURRENT["smoke"]] = MISSING_DATA

    InterPcurrent[DATA_CURRENT["solar"]] = select_value(
        [
            (
                lambda: "hrrrsubh" in sourceList,
                lambda: hrrrSubHInterpolation[0, HRRR_SUBH["solar"]],
            ),
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["solar"]),
            ),
            (
                lambda: has_hrrr_merged,
                lambda: interp_scalar(HRRR_Merged, HRRR["solar"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["solar"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(
                    ERA5_MERGED,
                    ERA5["surface_solar_radiation_downwards"],
                )
                / 3600,
            ),
        ],
        default=MISSING_DATA,
    )

    InterPcurrent[DATA_CURRENT["solar"]] = clipLog(
        InterPcurrent[DATA_CURRENT["solar"]],
        CLIP_SOLAR["min"],
        CLIP_SOLAR["max"],
        "Solar Current",
    )

    InterPcurrent[DATA_CURRENT["cape"]] = select_value(
        [
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["cape"]),
            ),
            (
                lambda: has_hrrr_merged,
                lambda: interp_scalar(HRRR_Merged, HRRR["cape"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["cape"]),
            ),
            (
                lambda: "era5" in sourceList,
                lambda: interp_scalar(
                    ERA5_MERGED,
                    ERA5["convective_available_potential_energy"],
                ),
            ),
        ]
    )

    InterPcurrent[DATA_CURRENT["cape"]] = clipLog(
        InterPcurrent[DATA_CURRENT["cape"]],
        CLIP_CAPE["min"],
        CLIP_CAPE["max"],
        "CAPE Current",
    )

    InterPcurrent[DATA_CURRENT["apparent"]] = calculate_apparent_temperature(
        InterPcurrent[DATA_CURRENT["temp"]],
        InterPcurrent[DATA_CURRENT["humidity"]],
        InterPcurrent[DATA_CURRENT["wind"]],
        InterPcurrent[DATA_CURRENT["solar"]],
    )

    InterPcurrent[DATA_CURRENT["feels_like"]] = select_value(
        [
            (
                lambda: "nbm" in sourceList,
                lambda: interp_scalar(NBM_Merged, NBM["apparent"]),
            ),
            (
                lambda: "gfs" in sourceList,
                lambda: interp_scalar(GFS_Merged, GFS["apparent"]),
            ),
            (lambda: timeMachine, lambda: InterPcurrent[DATA_CURRENT["apparent"]]),
        ],
        default=MISSING_DATA,
    )

    InterPcurrent[DATA_CURRENT["feels_like"]] = clipLog(
        InterPcurrent[DATA_CURRENT["feels_like"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )

    if "nbm_fire" in sourceList:
        InterPcurrent[DATA_CURRENT["fire"]] = clipLog(
            (
                NBM_Fire_Merged[currentIDX_hrrrh_A, NBM_FIRE_INDEX] * interpFac1
                + NBM_Fire_Merged[currentIDX_hrrrh, NBM_FIRE_INDEX] * interpFac2
            ),
            CLIP_FIRE["min"],
            CLIP_FIRE["max"],
            "Fire index Current",
        )
    else:
        InterPcurrent[DATA_CURRENT["fire"]] = MISSING_DATA

    curr_temp_si = InterPcurrent[DATA_CURRENT["temp"]] 
    curr_dew_si = InterPcurrent[DATA_CURRENT["dew"]]
    curr_wind_si = InterPcurrent[DATA_CURRENT["wind"]]
    curr_vis_si = InterPcurrent[DATA_CURRENT["vis"]]

    if tempUnits == 0:
        curr_temp_display = np.round(
            (InterPcurrent[DATA_CURRENT["temp"]]) * 9 / 5 + 32, 2
        )
        curr_apparent_display = np.round(
            (InterPcurrent[DATA_CURRENT["apparent"]]) * 9 / 5 + 32,
            2,
        )
        curr_dew_display = np.round(
            (InterPcurrent[DATA_CURRENT["dew"]]) * 9 / 5 + 32, 2
        )
        curr_feels_like_display = np.round(
            (InterPcurrent[DATA_CURRENT["feels_like"]]) * 9 / 5
            + 32,
            2,
        )
    else:
        curr_temp_display = np.round(InterPcurrent[DATA_CURRENT["temp"]], 2)
        curr_apparent_display = np.round(
            InterPcurrent[DATA_CURRENT["apparent"]], 2
        )
        curr_dew_display = np.round(InterPcurrent[DATA_CURRENT["dew"]], 2)
        curr_feels_like_display = np.round(
            InterPcurrent[DATA_CURRENT["feels_like"]], 2
        )

    curr_storm_dist_display = np.round(
        InterPcurrent[DATA_CURRENT["storm_dist"]] * visUnits, 2
    )
    curr_rain_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["rain_intensity"]] * prepIntensityUnit, 4
    )
    curr_snow_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["snow_intensity"]] * prepIntensityUnit, 4
    )
    curr_ice_intensity_display = np.round(
        InterPcurrent[DATA_CURRENT["ice_intensity"]] * prepIntensityUnit, 4
    )
    curr_pressure_display = np.round(InterPcurrent[DATA_CURRENT["pressure"]] / 100, 2)
    curr_wind_display = np.round(InterPcurrent[DATA_CURRENT["wind"]] * windUnit, 2)
    curr_gust_display = np.round(InterPcurrent[DATA_CURRENT["gust"]] * windUnit, 2)
    curr_vis_display = np.round(InterPcurrent[DATA_CURRENT["vis"]] * visUnits, 2)
    curr_station_pressure_display = np.round(
        InterPcurrent[DATA_CURRENT["station_pressure"]] / 100, 2
    )

    curr_humidity_display = np.round(InterPcurrent[DATA_CURRENT["humidity"]], 2)
    curr_cloud_display = np.round(InterPcurrent[DATA_CURRENT["cloud"]], 2)
    curr_uv_display = np.round(InterPcurrent[DATA_CURRENT["uv"]], 2)
    curr_ozone_display = np.round(InterPcurrent[DATA_CURRENT["ozone"]], 2)
    curr_smoke_display = np.round(InterPcurrent[DATA_CURRENT["smoke"]], 2)
    curr_fire_display = np.round(InterPcurrent[DATA_CURRENT["fire"]], 2)
    curr_solar_display = np.round(InterPcurrent[DATA_CURRENT["solar"]], 2)
    bearing_val = np.mod(InterPcurrent[DATA_CURRENT["bearing"]], 360)
    curr_bearing_display = (
        int(np.round(bearing_val, 0)) if not np.isnan(bearing_val) else np.nan
    )
    curr_cape_display = (
        int(np.round(InterPcurrent[DATA_CURRENT["cape"]], 0))
        if not np.isnan(InterPcurrent[DATA_CURRENT["cape"]])
        else 0
    )

    dayZeroIce = float(np.round(dayZeroIce * prepAccumUnit, 4))
    dayZeroRain = float(np.round(dayZeroRain * prepAccumUnit, 4))
    dayZeroSnow = float(np.round(dayZeroSnow * prepAccumUnit, 4))

    if (
        (minuteItems[0]["precipIntensity"])
        > (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepIntensityUnit)
    ) & (minuteItems[0]["precipType"] is not None):
        cIcon = minuteItems[0]["precipType"]
        cText = (
            minuteItems[0]["precipType"][0].upper() + minuteItems[0]["precipType"][1:]
        )

    elif InterPcurrent[DATA_CURRENT["vis"]] < FOG_THRESHOLD_METERS:
        cIcon = "fog"
        cText = "Fog"
    elif InterPcurrent[DATA_CURRENT["wind"]] > WIND_THRESHOLDS["light"]:
        cIcon = "wind"
        cText = "Windy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
        cIcon = "cloudy"
        cText = "Cloudy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        cText = "Partly Cloudy"

        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            cIcon = "partly-cloudy-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            cIcon = "partly-cloudy-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            cIcon = "partly-cloudy-night"
    else:
        cText = "Clear"
        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            cIcon = "clear-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            cIcon = "clear-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            cIcon = "clear-night"

    if log_timing:
        log_timing("Object Start")

    InterPcurrent[DATA_CURRENT["rain_intensity"]] = 0
    InterPcurrent[DATA_CURRENT["snow_intensity"]] = 0
    InterPcurrent[DATA_CURRENT["ice_intensity"]] = 0

    InterPcurrent[DATA_CURRENT["rain_intensity"]] = minuteRainIntensity[0]
    InterPcurrent[DATA_CURRENT["snow_intensity"]] = minuteSnowIntensity[0]
    InterPcurrent[DATA_CURRENT["ice_intensity"]] = minuteSleetIntensity[0]

    InterPcurrent[((InterPcurrent > -0.01) & (InterPcurrent < 0.01))] = 0

    currently = dict()
    currently["time"] = int(minute_array_grib[0])
    currently["summary"] = cText
    currently["icon"] = cIcon
    currently["nearestStormDistance"] = curr_storm_dist_display
    currently["nearestStormBearing"] = (
        int(InterPcurrent[DATA_CURRENT["storm_dir"]])
        if not np.isnan(InterPcurrent[DATA_CURRENT["storm_dir"]])
        else np.nan
    )
    currently["precipIntensity"] = minuteItems[0]["precipIntensity"]
    currently["precipProbability"] = minuteItems[0]["precipProbability"]
    currently["precipIntensityError"] = minuteItems[0]["precipIntensityError"]
    currently["precipType"] = minuteItems[0]["precipType"]
    currently["rainIntensity"] = curr_rain_intensity_display
    currently["snowIntensity"] = curr_snow_intensity_display
    currently["iceIntensity"] = curr_ice_intensity_display
    currently["temperature"] = curr_temp_display
    currently["apparentTemperature"] = curr_apparent_display
    currently["dewPoint"] = curr_dew_display
    currently["humidity"] = curr_humidity_display
    currently["pressure"] = curr_pressure_display
    currently["windSpeed"] = curr_wind_display
    currently["windGust"] = curr_gust_display
    currently["windBearing"] = curr_bearing_display
    currently["cloudCover"] = curr_cloud_display
    currently["uvIndex"] = curr_uv_display
    currently["visibility"] = curr_vis_display
    currently["ozone"] = curr_ozone_display
    currently["smoke"] = curr_smoke_display
    currently["fireIndex"] = curr_fire_display
    currently["feelsLike"] = curr_feels_like_display
    currently["currentDayIce"] = dayZeroIce
    currently["currentDayLiquid"] = dayZeroRain
    currently["currentDaySnow"] = dayZeroSnow
    currently["solar"] = curr_solar_display
    currently["cape"] = curr_cape_display

    if "stationPressure" in extraVars:
        currently["stationPressure"] = curr_station_pressure_display

    if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
        currentDay = False
    elif (
        InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
        and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
    ):
        currentDay = True
    elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
        currentDay = False
    else:
        currentDay = True

    currently_si = dict(currently)
    currently_si["icon"] = currently["icon"]
    currently_si["precipType"] = currently["precipType"]
    currently_si["windSpeed"] = curr_wind_si
    currently_si["visibility"] = curr_vis_si
    currently_si["temperature"] = curr_temp_si
    currently_si["dewPoint"] = curr_dew_si
    currently_si["cloudCover"] = InterPcurrent[DATA_CURRENT["cloud"]]
    currently_si["humidity"] = InterPcurrent[DATA_CURRENT["humidity"]]
    currently_si["smoke"] = InterPcurrent[DATA_CURRENT["smoke"]]
    currently_si["cape"] = InterPcurrent[DATA_CURRENT["cape"]]
    currently_si["rainIntensity"] = InterPcurrent[DATA_CURRENT["rain_intensity"]]
    currently_si["snowIntensity"] = InterPcurrent[DATA_CURRENT["snow_intensity"]]
    currently_si["iceIntensity"] = InterPcurrent[DATA_CURRENT["ice_intensity"]]
    currently_si["liquidAccumulation"] = 0
    currently_si["snowAccumulation"] = 0
    currently_si["iceAccumulation"] = 0

    if include_currently:
        try:
            if summaryText:
                currentText, currentIcon = calculate_text(
                    currently_si,
                    currentDay,
                    "current",
                    icon,
                )
                current_summary_key = currentText
                currently["summary"] = translation.translate(["title", currentText])
                currently["icon"] = currentIcon
        except Exception:
            logger.exception("CURRENTLY TEXT GEN ERROR %s", loc_tag)

        if version < 2:
            currently.pop("smoke", None)
            currently.pop("currentDayIce", None)
            currently.pop("currentDayLiquid", None)
            currently.pop("currentDaySnow", None)
            currently.pop("fireIndex", None)
            currently.pop("feelsLike", None)
            currently.pop("solar", None)
            currently.pop("cape", None)
            currently.pop("rainIntensity", None)
            currently.pop("snowIntensity", None)
            currently.pop("iceIntensity", None)

        if timeMachine and not tmExtra:
            currently.pop("nearestStormDistance", None)
            currently.pop("nearestStormBearing", None)
            currently.pop("precipProbability", None)
            currently.pop("precipIntensityError", None)
            currently.pop("humidity", None)
            currently.pop("uvIndex", None)
            currently.pop("visibility", None)
            currently.pop("ozone", None)

    return CurrentSection(
        currently=currently,
        interp_current=InterPcurrent,
        summary_key=current_summary_key,
    )
