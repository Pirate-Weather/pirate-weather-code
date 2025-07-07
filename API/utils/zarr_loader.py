import logging
import os
import subprocess
import zarr

from .s3_utils import S3ZipStore, DownloadSpec, download_if_newer, S3_BUCKET
from .constants import ZARR_DOWNLOADS

logger = logging.getLogger(__name__)


def find_largest_integer_directory(parent_dir: str, key_string: str, initial_run: bool):
    """Return the latest matching directory and a list of older directories."""
    largest_value = -1
    largest_dir = None
    old_dirs = []

    stage = os.environ.get("STAGE", "PROD")

    for entry in os.listdir(parent_dir):
        if (key_string in entry) and ("TMP" not in entry):
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


def update_zarr_store(initial_run: bool):
    """Refresh all zarr datasets stored under ``/tmp``."""
    global ETOPO_f, SubH_Zarr, HRRR_6H_Zarr, GFS_Zarr, NBM_Zarr, NBM_Fire_Zarr, GEFS_Zarr, HRRR_Zarr, NWS_Alerts_Zarr

    stage = os.environ.get("STAGE", "PROD")
    os.makedirs("/tmp/empty", exist_ok=True)

    latest_Alert, old_Alert = find_largest_integer_directory("/tmp", "NWS_Alerts.zarr", initial_run)
    if latest_Alert is not None:
        NWS_Alerts_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_Alert), mode="r")
        logger.info("Loading new: %s", latest_Alert)
    for old_dir in old_Alert:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_SubH, old_SubH = find_largest_integer_directory("/tmp", "SubH.zarr", initial_run)
    if latest_SubH is not None:
        SubH_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_SubH), mode="r")
        logger.info("Loading new: %s", latest_SubH)
    for old_dir in old_SubH:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_HRRR_6H, old_HRRR_6H = find_largest_integer_directory("/tmp", "HRRR_6H.zarr", initial_run)
    if latest_HRRR_6H is not None:
        HRRR_6H_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_HRRR_6H), mode="r")
        logger.info("Loading new: %s", latest_HRRR_6H)
    for old_dir in old_HRRR_6H:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_GFS, old_GFS = find_largest_integer_directory("/tmp", "GFS.zarr", initial_run)
    if latest_GFS is not None:
        GFS_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_GFS), mode="r")
        logger.info("Loading new: %s", latest_GFS)
    for old_dir in old_GFS:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_NBM, old_NBM = find_largest_integer_directory("/tmp", "NBM.zarr", initial_run)
    if latest_NBM is not None:
        NBM_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_NBM), mode="r")
        logger.info("Loading new: %s", latest_NBM)
    for old_dir in old_NBM:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_NBM_Fire, old_NBM_Fire = find_largest_integer_directory("/tmp", "NBM_Fire.zarr", initial_run)
    if latest_NBM_Fire is not None:
        NBM_Fire_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_NBM_Fire), mode="r")
        logger.info("Loading new: %s", latest_NBM_Fire)
    for old_dir in old_NBM_Fire:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_GEFS, old_GEFS = find_largest_integer_directory("/tmp", "GEFS.zarr", initial_run)
    if latest_GEFS is not None:
        GEFS_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_GEFS), mode="r")
        logger.info("Loading new: %s", latest_GEFS)
    for old_dir in old_GEFS:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    latest_HRRR, old_HRRR = find_largest_integer_directory("/tmp", "HRRR.zarr", initial_run)
    if latest_HRRR is not None:
        HRRR_Zarr = zarr.open(S3ZipStore("/tmp/" + latest_HRRR), mode="r")
        logger.info("Loading new: %s", latest_HRRR)
    for old_dir in old_HRRR:
        if stage == "PROD":
            logger.info("Removing old: %s", old_dir)
            subprocess.run(f"nice -n 20 rm -rf /tmp/{old_dir}", shell=True)

    if initial_run and os.getenv("useETOPO", "True"):
        latest_ETOPO, _ = find_largest_integer_directory("/tmp", "ETOPO_DA_C.zarr", initial_run)
        if latest_ETOPO is not None:
            ETOPO_f = zarr.open(S3ZipStore("/tmp/" + latest_ETOPO), mode="r")
            logger.info("ETOPO Setup")

    logger.info("Refreshed Zarrs")


def sync_zarr_datasets(initial_run: bool, use_etopo: bool = True) -> None:
    """Download new zarr archives and refresh stores."""

    stage = os.environ.get("STAGE", "PROD")
    if stage == "PROD":
        names = [
            "SubH",
            "HRRR_6H",
            "GFS",
            "NBM",
            "NBM_Fire",
            "GEFS",
            "HRRR",
            "NWS_Alerts",
        ]
        if use_etopo:
            names.append("ETOPO")

        for name in names:
            object_key, tmp_file, local_store = ZARR_DOWNLOADS[name]
            download_if_newer(
                DownloadSpec(
                    bucket=S3_BUCKET,
                    object_key=object_key,
                    local_file=tmp_file,
                    local_store=local_store,
                    initial_download=initial_run,
                )
            )
            logger.info("%s Download!", name)

    if stage in {"PROD", "DEV"}:
        update_zarr_store(initial_run)
