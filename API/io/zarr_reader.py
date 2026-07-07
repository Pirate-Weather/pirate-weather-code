"""Helpers for loading and reading Zarr datasets."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import aiobotocore.session as _aio_session
import numpy as np
import s3fs
import zarr

from API.constants.api_const import MAX_ZARR_READ_RETRIES
from API.constants.shared_const import INGEST_VERSION_STR, MISSING_DATA
from API.io.ZarrHelpers import _add_custom_header, init_ERA5, setup_testing_zipstore


def _default_logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class ZarrStores:
    ETOPO_f: Optional[Any] = None
    SubH_Zarr: Optional[Any] = None
    HRRR_6H_Zarr: Optional[Any] = None
    GFS_Zarr: Optional[Any] = None
    ECMWF_Zarr: Optional[Any] = None
    NBM_Zarr: Optional[Any] = None
    NBM_Fire_Zarr: Optional[Any] = None
    GEFS_Zarr: Optional[Any] = None
    HRRR_Zarr: Optional[Any] = None
    NWS_Alerts_Zarr: Optional[Any] = None
    WMO_Alerts_Zarr: Optional[Any] = None
    RTMA_RU_Zarr: Optional[Any] = None
    ERA5_Data: Optional[Any] = None
    DWD_MOSMIX_Zarr: Optional[Any] = None
    AIGFS_Zarr: Optional[Any] = None
    AIGEFS_Zarr: Optional[Any] = None
    ECMWF_AIFS_Zarr: Optional[Any] = None
    RAQDPS_Zarr: Optional[Any] = None
    SILAM_Zarr: Optional[Any] = None
    RAQDPS_LatLon: Optional[Any] = None


async def get_zarr(store, X, Y):
    """Asynchronously retrieve zarr data at given coordinates."""
    return store[:, :, X, Y]


def has_interior_nan_holes(arr: np.ndarray) -> tuple[bool, Optional[int]]:
    """Detect an interior block of NaNs in a 2D array."""
    mask = np.isnan(arr)
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    diff = padded[:, 1:].astype(int) - padded[:, :-1].astype(int)
    starts = diff == 1
    ends = diff == -1
    interior_starts = starts[:, 1:-1]
    interior_ends = ends[:, 1:-1]
    row_has_start = interior_starts.any(axis=1)
    row_has_end = interior_ends.any(axis=1)
    matching_rows = np.flatnonzero(row_has_start & row_has_end)
    if matching_rows.size:
        return True, int(matching_rows[0])
    return False, None


def _interp_row(row: np.ndarray) -> np.ndarray:
    """Fill only strictly interior NaN-runs in a 1D array."""
    n = row.size
    x = np.arange(n)
    mask = np.isnan(row)
    if mask.any() and not mask.all():
        good = ~mask
        row[mask] = np.interp(
            x[mask], x[good], row[good], left=MISSING_DATA, right=MISSING_DATA
        )
    return row


class WeatherParallel(object):
    """Helper class for parallel zarr reading operations."""

    def __init__(
        self,
        loc_tag: str = "",
        *,
        logger: Optional[logging.Logger] = None,
        timing_enabled: bool = False,
    ):
        self.loc_tag = loc_tag
        self.logger = logger or _default_logger()
        self.timing_enabled = timing_enabled

    async def zarr_read(self, model, opened_zarr, x, y):
        if self.timing_enabled:
            self.logger.debug("### %s Reading! %s", model, self.loc_tag)
        err_count = 0
        data_out = False
        while err_count < MAX_ZARR_READ_RETRIES:
            try:
                if model == "DWD_MOSMIX":
                    # DWD MOSMIX has data saved in a group
                    data_out = await asyncio.to_thread(
                        lambda: (
                            opened_zarr["__xarray_dataarray_variable__"][
                                :,
                                :,
                                y,
                                x,
                            ].T
                        )
                    )
                else:
                    data_out = await asyncio.to_thread(
                        lambda: opened_zarr[:, :, y, x].T
                    )

                has_missing_data, missing_row = has_interior_nan_holes(data_out.T)
                if has_missing_data:
                    if model == "DWD_MOSMIX":
                        # DWD MOSMIX station data may have gaps (e.g. missing station
                        # reports or the 6-hourly period beyond 240 h). Linearly
                        # interpolating those gaps flattens the diurnal cycle (e.g.
                        # temperatures in tropical locations like Rio de Janeiro appear
                        # averaged/flat). Leave the NaN values intact so the merge
                        # logic falls back to GFS/ECMWF for those hours instead.
                        self.logger.debug(
                            "### %s Has missing data (row %s), leaving NaN for merge fallback",
                            model,
                            missing_row,
                        )
                    else:
                        self.logger.warning(
                            "### %s Interpolating missing data (row %s)!",
                            model,
                            missing_row,
                        )
                        data_out = np.apply_along_axis(_interp_row, 0, data_out)

                if self.timing_enabled:
                    self.logger.debug("### %s Done! %s", model, self.loc_tag)
                return data_out

            except Exception:
                self.logger.exception("### %s Failure! %s", model, self.loc_tag)
                err_count += 1

        self.logger.error("### %s Failure! %s", model, self.loc_tag)
        data_out = False
        return data_out


def update_zarr_store(
    initial_run: bool,
    *,
    stage: str,
    save_dir: str,
    use_etopo: bool,
    save_type: str,
    s3_bucket: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    logger: Optional[logging.Logger] = None,
) -> ZarrStores:
    """Load zarr data stores from static file paths."""
    logger = logger or _default_logger()
    stores = ZarrStores()
    ingest_version = INGEST_VERSION_STR

    # Load ETOPO on initial run if enabled
    if initial_run and use_etopo:
        etopo_path = os.path.join(save_dir, "ETOPO_DA_C.zarr")
        if os.path.exists(etopo_path):
            stores.ETOPO_f = zarr.open(zarr.storage.LocalStore(etopo_path), mode="r")
            logger.info("Loaded ETOPO from: %s", etopo_path)

    # Open the Google ERA5 dataset for Dev and TimeMachine
    if stage in ("DEV", "TIMEMACHINE"):
        era5_cache_dir = os.environ.get(
            "ERA5_CACHE_DIR", os.path.join(save_dir, "ERA5_cache")
        )
        stores.ERA5_Data = init_ERA5(era5_cache_dir)
        logger.info("ERA5 disk cache: %s", era5_cache_dir)

    # If TimeMachine, load GFS
    if stage == "TIMEMACHINE":
        gfs_path = os.path.join(save_dir, "GFS.zarr")
        if os.path.exists(gfs_path):
            stores.GFS_Zarr = zarr.open(zarr.storage.LocalStore(gfs_path), mode="r")
            logger.info("Loaded GFS from: %s", gfs_path)

    # Use local stores for Dev and Prod
    if stage in ("DEV", "PROD"):
        local_stores = [
            ("GFS_Zarr", "GFS.zarr"),
            ("NWS_Alerts_Zarr", "NWS_Alerts.zarr"),
            ("SubH_Zarr", "SubH.zarr"),
            ("HRRR_6H_Zarr", "HRRR_6H.zarr"),
            ("ECMWF_Zarr", "ECMWF.zarr"),
            ("NBM_Zarr", "NBM.zarr"),
            ("NBM_Fire_Zarr", "NBM_Fire.zarr"),
            ("GEFS_Zarr", "GEFS.zarr"),
            ("HRRR_Zarr", "HRRR.zarr"),
            ("WMO_Alerts_Zarr", "WMO_Alerts.zarr"),
            ("RTMA_RU_Zarr", "RTMA_RU.zarr"),
            ("DWD_MOSMIX_Zarr", "DWD_MOSMIX.zarr"),
            ("AIGFS_Zarr", "AIGFS.zarr"),
            ("AIGEFS_Zarr", "AIGEFS.zarr"),
            ("ECMWF_AIFS_Zarr", "ECMWF_AIFS.zarr"),
            ("RAQDPS_Zarr", "RAQDPS.zarr"),
            ("SILAM_Zarr", "SILAM.zarr"),
        ]
        for attr, fname in local_stores:
            _load_local_store(stores, attr, save_dir, fname, logger=logger)

        # Load AQ model lat/lon pickles
        _load_aq_lat_lon_pickles(stores, save_dir, logger=logger)

    # Use S3 stores for Testing and TM Testing
    if stage in ("TESTING", "TM_TESTING"):
        logger.info("Setting up S3 zarrs")
        if save_type == "S3":
            # Create an aiobotocore session and register the apikey hook on
            # it *before* any client is created.  Session-level registration
            # propagates to every client the session creates, which avoids a
            # race condition where the first HeadObject fires before a
            # client-level hook is active.
            aio_sess = _aio_session.AioSession()
            aio_sess.register("before-send.s3", _add_custom_header)
            s3 = s3fs.S3FileSystem(
                anon=True,
                asynchronous=False,
                endpoint_url="https://api.pirateweather.net/files/",
                skip_instance_cache=True,
                session=aio_sess,
            )
        elif save_type == "S3Zarr":
            s3 = s3fs.S3FileSystem(
                key=aws_access_key_id, secret=aws_secret_access_key, version_aware=True
            )
        else:
            s3 = None

        gfs_store = setup_testing_zipstore(
            s3, s3_bucket, ingest_version, save_type, "GFS"
        )
        stores.GFS_Zarr = zarr.open(gfs_store, mode="r")
        era5_cache_dir = os.environ.get(
            "ERA5_CACHE_DIR", os.path.join(save_dir, "ERA5_cache")
        )
        stores.ERA5_Data = init_ERA5(era5_cache_dir)
        logger.info("GFS Read")
        logger.info("ERA5 Read; disk cache: %s", era5_cache_dir)

        if stage == "TESTING":
            testing_stores = [
                ("NWS_Alerts_Zarr", "NWS_Alerts"),
                ("SubH_Zarr", "SubH"),
                ("HRRR_6H_Zarr", "HRRR_6H"),
                ("GEFS_Zarr", "GEFS"),
                ("NBM_Zarr", "NBM"),
                ("NBM_Fire_Zarr", "NBM_Fire"),
                ("HRRR_Zarr", "HRRR"),
                ("WMO_Alerts_Zarr", "WMO_Alerts"),
                ("RTMA_RU_Zarr", "RTMA_RU"),
                ("ECMWF_Zarr", "ECMWF"),
                ("DWD_MOSMIX_Zarr", "DWD_MOSMIX"),
                ("AIGFS_Zarr", "AIGFS"),
                ("AIGEFS_Zarr", "AIGEFS"),
                ("ECMWF_AIFS_Zarr", "ECMWF_AIFS"),
            ]
            for attr, name in testing_stores:
                setattr(
                    stores,
                    attr,
                    _testing_store(
                        s3, s3_bucket, ingest_version, save_type, name, logger=logger
                    ),
                )
            if use_etopo:
                stores.ETOPO_f = _testing_store(
                    s3,
                    s3_bucket,
                    ingest_version,
                    save_type,
                    "ETOPO_DA_C",
                    logger=logger,
                )

    logger.info("Zarr stores loaded")
    return stores


def _load_aq_lat_lon_pickles(
    stores: ZarrStores,
    save_dir: str,
    *,
    logger: logging.Logger,
) -> None:
    """Load AQ model lat/lon pickle files when available."""
    for attr, fname in (("RAQDPS_LatLon", "RAQDPS.lat_lon.pickle"),):
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    setattr(stores, attr, pickle.load(fh))
                logger.info("Loaded %s from: %s", attr, path)
            except Exception as exc:
                logger.warning("Could not load %s: %s", attr, exc)


def _load_local_store(
    stores: ZarrStores,
    attr_name: str,
    save_dir: str,
    filename: str,
    *,
    logger: logging.Logger,
) -> None:
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        try:
            setattr(
                stores, attr_name, zarr.open(zarr.storage.LocalStore(path), mode="r")
            )
            logger.info("Loaded %s from: %s", attr_name, path)
        except Exception as exc:  # keep compatibility with ECMWF failure handling
            logger.info("%s not available: %s", attr_name, exc)
            setattr(stores, attr_name, None)
    else:
        logger.info("%s not found: %s", attr_name, path)


def _testing_store(
    s3: Optional[s3fs.S3FileSystem],
    s3_bucket: str,
    ingest_version: str,
    save_type: str,
    name: str,
    *,
    logger: logging.Logger,
):
    store = setup_testing_zipstore(s3, s3_bucket, ingest_version, save_type, name)
    logger.info("%s Read", name)
    return zarr.open(store, mode="r")
