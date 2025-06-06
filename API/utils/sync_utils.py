"""Utility helpers for syncing Zarr data from S3 or local storage."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import subprocess
from typing import Iterable, Tuple, Any

import boto3
from functools import lru_cache
import s3fs
import zarr
from boto3.s3.transfer import TransferConfig


class S3ZipStore(zarr.storage.ZipStore):
    """Simple wrapper to allow FastAPI to read zipped Zarr files from S3."""

    def __init__(self, path: s3fs.S3File) -> None:
        super().__init__(path="", mode="r")
        self.path = path


@lru_cache(maxsize=1)
def get_s3_client() -> boto3.client:
    """Return a cached boto3 S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_KEY", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET", ""),
    )


def add_custom_header(request: Any, **_: Any) -> None:
    """Attach an API key header to signed S3 requests."""

    request.headers["apikey"] = os.environ.get("PW_API", "")


def download_if_newer(
    s3_bucket: str,
    s3_object_key: str,
    local_file_path: str,
    local_lmdb_path: str,
    initial_download: bool,
) -> None:
    """Download an object from S3 if it has been updated."""

    config = (
        TransferConfig(use_threads=True, max_bandwidth=None)
        if initial_download
        else TransferConfig(use_threads=False, max_bandwidth=100000000)
    )

    if os.environ.get("save_type", "S3") == "S3":
        s3_client = get_s3_client()
        s3_response = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
        s3_last_modified = s3_response["LastModified"].timestamp()
    else:
        s3_last_modified = os.path.getmtime(os.path.join(s3_bucket, s3_object_key))

    new_file = False
    pickle_path = f"{local_file_path}.modtime.pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            local_last_modified = pickle.load(file)
        if s3_last_modified > local_last_modified:
            if os.environ.get("save_type", "S3") == "S3":
                s3_client.download_file(s3_bucket, s3_object_key, local_file_path, Config=config)
            else:
                shutil.copy(os.path.join(s3_bucket, s3_object_key), local_file_path)
            new_file = True
            with open(pickle_path, "wb") as file:
                pickle.dump(s3_last_modified, file)
    else:
        if os.environ.get("save_type", "S3") == "S3":
            s3_client.download_file(s3_bucket, s3_object_key, local_file_path, Config=config)
        else:
            shutil.copy(os.path.join(s3_bucket, s3_object_key), local_file_path)
        with open(pickle_path, "wb") as file:
            pickle.dump(s3_last_modified, file)
        new_file = True

    if new_file:
        with open(f"{local_lmdb_path}.lock", "w"):
            pass
        shutil.move(local_file_path, f"{local_lmdb_path}_{s3_last_modified}")
        os.remove(f"{local_lmdb_path}.lock")


def find_largest_integer_directory(
    parent_dir: str,
    key_string: str,
    initial_run: bool,
) -> Tuple[str | None, Iterable[str]]:
    """Return the most recent matching directory and directories to remove."""

    largest_value = -1.0
    largest_dir: str | None = None
    old_dirs: list[str] = []
    stage = os.environ.get("STAGE", "PROD")

    for entry in os.listdir(parent_dir):
        if key_string in entry and "TMP" not in entry:
            old_dirs.append(entry)
            try:
                value = float(entry[-12:])
                if value > largest_value:
                    largest_value = value
                    largest_dir = entry
            except ValueError:
                continue

    if stage == "PROD" and largest_dir in old_dirs:
        old_dirs.remove(largest_dir)
    if (not initial_run) and not old_dirs:
        largest_dir = None

    return largest_dir, old_dirs


logger = logging.getLogger("dataSync")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def update_zarr_store(initial_run: bool) -> None:
    """Load the latest Zarr archives into memory."""

    import numpy as np  # Local import to avoid heavy dependency at module load

    global ETOPO_f, SubH_Zarr, HRRR_6H_Zarr, GFS_Zarr, NBM_Zarr, NBM_Fire_Zarr
    global GEFS_Zarr, HRRR_Zarr, NWS_Alerts_Zarr

    stage = os.environ.get("STAGE", "PROD")
    os.makedirs("/tmp/empty", exist_ok=True)

    def _open_latest(key: str) -> Tuple[str | None, Iterable[str]]:
        latest, old = find_largest_integer_directory("/tmp", key, initial_run)
        if latest is not None:
            z = zarr.open(zarr.storage.ZipStore(f"/tmp/{latest}", mode="r"), mode="r")
            logger.info("Loading new: %s", latest)
            return z, old
        return None, old

    NWS_Alerts_Zarr, old_alert = _open_latest("NWS_Alerts.zarr")
    SubH_Zarr, old_subh = _open_latest("SubH.zarr")
    HRRR_6H_Zarr, old_hrrr6h = _open_latest("HRRR_6H.zarr")
    GFS_Zarr, old_gfs = _open_latest("GFS.zarr")
    NBM_Zarr, old_nbm = _open_latest("NBM.zarr")
    NBM_Fire_Zarr, old_nbm_fire = _open_latest("NBM_Fire.zarr")
    GEFS_Zarr, old_gefs = _open_latest("GEFS.zarr")
    HRRR_Zarr, old_hrrr = _open_latest("HRRR.zarr")

    if initial_run and os.environ.get("useETOPO", True):
        ETOPO_f, old_etopo = _open_latest("ETOPO_DA_C.zarr")
        logger.info("ETOPO Setup")
    else:
        old_etopo = []

    for old in [*old_alert, *old_subh, *old_hrrr6h, *old_gfs, *old_nbm,
                *old_nbm_fire, *old_gefs, *old_hrrr, *old_etopo]:
        if stage == "PROD":
            logger.info("Removing old: %s", old)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old}", shell=True)

    logger.info("Refreshed Zarrs")
