"""Grid indexing and data fetch helpers."""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import xarray as xr

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
from API.constants.model_const import ERA5
from API.constants.shared_const import HISTORY_PERIODS
from API.utils.geo import lambertGridMatch
from API.utils.timing import StepTimer


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
    rtma_ru: Any
    wmo_alerts: Any
    era5_data: Any
    dwd_mosmix: Any = None


@dataclass
class GridIndexingResult:
    dataOut: Union[np.ndarray, bool]
    dataOut_h2: Union[np.ndarray, bool]
    dataOut_hrrrh: Union[np.ndarray, bool]
    dataOut_nbm: Union[np.ndarray, bool]
    dataOut_nbmFire: Union[np.ndarray, bool]
    dataOut_gfs: Union[np.ndarray, bool]
    dataOut_ecmwf: Union[np.ndarray, bool]
    dataOut_gefs: Union[np.ndarray, bool]
    dataOut_rtma_ru: Union[np.ndarray, bool]
    dataOut_dwd_mosmix: Union[np.ndarray, bool]
    era5_merged: Union[np.ndarray, bool]
    subhRunTime: Union[float, None]
    hrrrhRunTime: Union[float, None]
    h2RunTime: Union[float, None]
    nbmRunTime: Union[float, None]
    nbmFireRunTime: Union[float, None]
    gfsRunTime: Union[float, None]
    ecmwfRunTime: Union[float, None]
    gefsRunTime: Union[float, None]
    dwdMosmixRunTime: Union[float, None]
    x_rtma: Union[float, None]
    y_rtma: Union[float, None]
    rtma_lat: Union[float, None]
    rtma_lon: Union[float, None]
    x_nbm: Union[float, None]
    y_nbm: Union[float, None]
    nbm_lat: Union[float, None]
    nbm_lon: Union[float, None]
    x_p: Union[float, None]
    y_p: Union[float, None]
    gfs_lat: Union[float, None]
    gfs_lon: Union[float, None]
    x_p_eur: Union[float, None]
    y_p_eur: Union[float, None]
    lats_ecmwf: Union[np.ndarray, None]
    lons_ecmwf: Union[np.ndarray, None]
    x_dwd: Union[float, None]
    y_dwd: Union[float, None]
    dwd_lat: Union[float, None]
    dwd_lon: Union[float, None]
    sourceIDX: dict
    WMO_alertDat: Union[str, None]


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
    read_wmo_alerts: bool,
    base_day_utc: datetime.datetime,
    zarr_sources: ZarrSources,
    weather,
    timing_start: datetime.datetime,
    timing_enabled: bool,
    logger: logging.Logger,
) -> GridIndexingResult:
    """Compute grid coordinates and pull the zarr slices for the request."""
    timer = StepTimer(timing_start, timing_enabled)
    sourceIDX = dict()
    readRTMA_RU = False
    readNBM = False
    readGFS = False
    readECMWF = False
    readGEFS = False
    readHRRR = False
    readERA5 = False
    readDWD_MOSMIX = False

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

        sourceIDX["hrrr"] = dict()
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
    if ex_ecmwf == 1:
        dataOut_ecmwf = False
    elif time_machine:
        dataOut_ecmwf = False
    elif zarr_sources.ecmwf is None:
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

    if ex_gefs == 1:
        dataOut_gefs = False
    elif time_machine:
        dataOut_gefs = False
    else:
        readGEFS = True
        dataOut_gefs = None

    timer.log("### GEFS Detail END ###")

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
    if ex_dwd_mosmix == 1:
        dataOut_dwd_mosmix = False
    elif time_machine:
        dataOut_dwd_mosmix = False
    elif zarr_sources.dwd_mosmix is None:
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

    if readERA5:
        abslat_era5 = np.abs(zarr_sources.era5_data["ERA5_lats"] - lat)
        abslon_era5 = np.abs(zarr_sources.era5_data["ERA5_lons"] - lon)
        y_p = np.argmin(abslat_era5)
        x_p = np.argmin(abslon_era5)
        t_p = np.argmin(
            np.abs(
                zarr_sources.era5_data["ERA5_times"]
                - np.datetime64(base_day_utc.replace(tzinfo=None))
            )
        )
        dataOut_ERA5_xr = zarr_sources.era5_data["dsERA5"][ERA5.keys()].isel(
            latitude=y_p, longitude=x_p, time=slice(t_p, t_p + 25)
        )
        dataOut_ERA5 = xr.concat(
            [dataOut_ERA5_xr[var] for var in ERA5.keys()], dim="variable"
        )
        unix_times_era5 = (
            dataOut_ERA5_xr["time"].astype("datetime64[s]")
            - np.datetime64("1970-01-01T00:00:00")
        ).astype(np.int64)
        ERA5_MERGED = np.vstack((unix_times_era5, dataOut_ERA5.values)).T
    else:
        ERA5_MERGED = False

    zarrTasks = dict()
    if readHRRR:
        zarrTasks["SubH"] = weather.zarr_read("SubH", zarr_sources.subh, x_hrrr, y_hrrr)
        zarrTasks["HRRR_6H"] = weather.zarr_read(
            "HRRR_6H", zarr_sources.hrrr_6h, x_hrrr, y_hrrr
        )
        zarrTasks["HRRR"] = weather.zarr_read("HRRR", zarr_sources.hrrr, x_hrrr, y_hrrr)
    if readNBM:
        zarrTasks["NBM"] = weather.zarr_read("NBM", zarr_sources.nbm, x_nbm, y_nbm)
        zarrTasks["NBM_Fire"] = weather.zarr_read(
            "NBM_Fire", zarr_sources.nbm_fire, x_nbm, y_nbm
        )
    if readGFS:
        zarrTasks["GFS"] = weather.zarr_read("GFS", zarr_sources.gfs, x_p, y_p)
    if readECMWF:
        zarrTasks["ECMWF"] = weather.zarr_read(
            "ECMWF", zarr_sources.ecmwf, x_p_eur, y_p_eur
        )
    if readGEFS:
        zarrTasks["GEFS"] = weather.zarr_read("GEFS", zarr_sources.gefs, x_p, y_p)
    if readRTMA_RU:
        zarrTasks["RTMA_RU"] = weather.zarr_read(
            "RTMA_RU", zarr_sources.rtma_ru, x_rtma, y_rtma
        )
    if readDWD_MOSMIX:
        zarrTasks["DWD_MOSMIX"] = weather.zarr_read(
            "DWD_MOSMIX", zarr_sources.dwd_mosmix, x_dwd, y_dwd
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
    dwdMosmixRunTime = None

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
        dataOut_nbmFire = zarr_results["NBM_Fire"]
        if dataOut_nbm is not False:
            nbmRunTime = dataOut_nbm[HISTORY_PERIODS["NBM"], 0]
        sourceIDX["nbm"] = dict()
        sourceIDX["nbm"]["x"] = int(x_nbm)
        sourceIDX["nbm"]["y"] = int(y_nbm)
        sourceIDX["nbm"]["lat"] = round(nbm_lat, 2)
        sourceIDX["nbm"]["lon"] = round(((nbm_lon + 180) % 360) - 180, 2)
        if dataOut_nbmFire is not False:
            nbmFireRunTime = dataOut_nbmFire[HISTORY_PERIODS["NBM"] - 6, 0]

    if readGFS:
        dataOut_gfs = zarr_results["GFS"]
        if dataOut_gfs is not False:
            gfsRunTime = dataOut_gfs[HISTORY_PERIODS["GFS"] - 1, 0]

    if readECMWF:
        dataOut_ecmwf = zarr_results["ECMWF"]
        if dataOut_ecmwf is not False:
            ecmwfRunTime = dataOut_ecmwf[HISTORY_PERIODS["ECMWF"] - 3, 0]
            sourceIDX["ecmwf_ifs"] = dict()
            sourceIDX["ecmwf_ifs"]["x"] = int(x_p_eur)
            sourceIDX["ecmwf_ifs"]["y"] = int(y_p_eur)
            sourceIDX["ecmwf_ifs"]["lat"] = round(lats_ecmwf[y_p_eur], 2)
            sourceIDX["ecmwf_ifs"]["lon"] = round(lons_ecmwf[x_p_eur], 2)

    if readGEFS:
        dataOut_gefs = zarr_results["GEFS"]
        gefsRunTime = dataOut_gefs[HISTORY_PERIODS["GEFS"] - 3, 0]

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
                dwdMosmixRunTime = dataOut_dwd_mosmix[HISTORY_PERIODS["DWD_MOSMIX"], 0]

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
                        or time_diff < datetime.timedelta(hours=-72)  # Allow up to 72h future
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
                        sourceIDX["dwd_mosmix"] = dict()
                        sourceIDX["dwd_mosmix"]["x"] = int(x_dwd)
                        sourceIDX["dwd_mosmix"]["y"] = int(y_dwd)
                        sourceIDX["dwd_mosmix"]["lat"] = round(dwd_lat, 2)
                        sourceIDX["dwd_mosmix"]["lon"] = round(
                            ((dwd_lon + 180) % 360) - 180, 2
                        )
            else:
                # Data array too short, treat as no data available
                dataOut_dwd_mosmix = False

    return GridIndexingResult(
        dataOut=dataOut,
        dataOut_h2=dataOut_h2,
        dataOut_hrrrh=dataOut_hrrrh,
        dataOut_nbm=dataOut_nbm,
        dataOut_nbmFire=dataOut_nbmFire,
        dataOut_gfs=dataOut_gfs,
        dataOut_ecmwf=dataOut_ecmwf,
        dataOut_gefs=dataOut_gefs,
        dataOut_rtma_ru=dataOut_rtma_ru,
        dataOut_dwd_mosmix=dataOut_dwd_mosmix,
        era5_merged=ERA5_MERGED,
        subhRunTime=subhRunTime,
        hrrrhRunTime=hrrrhRunTime,
        h2RunTime=h2RunTime,
        nbmRunTime=nbmRunTime,
        nbmFireRunTime=nbmFireRunTime,
        gfsRunTime=gfsRunTime,
        ecmwfRunTime=ecmwfRunTime,
        gefsRunTime=gefsRunTime,
        dwdMosmixRunTime=dwdMosmixRunTime,
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
        sourceIDX=sourceIDX,
        WMO_alertDat=WMO_alertDat,
    )
