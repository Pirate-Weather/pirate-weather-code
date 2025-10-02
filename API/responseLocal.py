import asyncio
import datetime
import logging

# Standard library imports
import math
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from typing import Union

# Third-party imports
import boto3
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from astral import LocationInfo, moon
from astral.sun import sun
from boto3.s3.transfer import TransferConfig
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from fastapi_utils.tasks import repeat_every
from API.PirateDailyText import calculate_day_text
from API.PirateMinutelyText import calculate_minutely_text
from API.PirateText import calculate_text
from API.PirateTextHelper import estimate_snow_height
from pirateweather_translations.dynamic_loader import load_all_translations
from API.PirateWeeklyText import calculate_weekly_text
from pytz import timezone, utc
from API.timemachine import TimeMachine
from timezonefinder import TimezoneFinder

from API.constants.api_const import (
    API_VERSION,
    APPARENT_TEMP_CONSTS,
    DBZ_CONST,
    GLOBE_TEMP_CONST,
    LARGEST_DIR_INIT,
    MAX_S3_RETRIES,
    NICE_PRIORITY,
    PRECIP_IDX,
    S3_BASE_DELAY,
    S3_MAX_BANDWIDTH,
    SOLAR_IRRADIANCE_CONST,
    SOLAR_RAD_CONST,
    TEMP_THRESHOLD_RAIN_C,
    TEMP_THRESHOLD_SNOW_C,
    TEMPERATURE_UNITS_THRESH,
    WBGT_CONST,
)
from API.constants.clip_const import (
    CLIP_CLOUD,
    CLIP_FEELS_LIKE,
    CLIP_FIRE,
    CLIP_GLOBAL,
    CLIP_HUMIDITY,
    CLIP_OZONE,
    CLIP_PRESSURE,
    CLIP_PROB,
    CLIP_SMOKE,
    CLIP_TEMP,
    CLIP_UV,
    CLIP_VIS,
    CLIP_WIND,
)
from API.constants.forecast_const import (
    DATA_CURRENT,
    DATA_DAY,
    DATA_HOURLY,
    DATA_MINUTELY,
)
from API.constants.grid_const import (
    HRRR_X_MAX,
    HRRR_X_MIN,
    HRRR_Y_MAX,
    HRRR_Y_MIN,
    NBM_X_MAX,
    NBM_X_MIN,
    NBM_Y_MAX,
    NBM_Y_MIN,
    US_BOUNDING_BOX,
)

# Project imports
from API.constants.model_const import GEFS, GFS, HRRR, HRRR_SUBH, NBM, NBM_FIRE_INDEX
from API.constants.shared_const import (
    INGEST_VERSION_STR,
    KELVIN_TO_CELSIUS,
    MISSING_DATA,
    REFC_THRESHOLD,
)
from API.constants.text_const import (
    CLOUD_COVER_THRESHOLDS,
    DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM,
    FOG_THRESHOLD_METERS,
    HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM,
    PRECIP_PROB_THRESHOLD,
    WIND_THRESHOLDS,
)

Translations = load_all_translations()

lock = threading.Lock()

aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
pw_api_key = os.environ.get("PW_API", "")
save_type = os.getenv("save_type", default="S3")
s3_bucket = os.getenv("s3_bucket", default="piratezarr2")
useETOPO = os.getenv("useETOPO", default=True)
TIMING = os.environ.get("TIMING", False)

force_now = os.getenv("force_now", default=False)

# Version code for ingest files
ingestVersion = INGEST_VERSION_STR


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    # include timestamp, level, logger name, module, line number, message
    fmt = "%(asctime)s %(levelname)s [%(name)s:%(module)s:%(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S%z"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


class S3ZipStore(zarr.storage.ZipStore):
    def __init__(self, path: s3fs.S3File) -> None:
        super().__init__(path="", mode="r")
        self.path = path


def _add_custom_header(request, **kwargs):
    request.headers["apikey"] = pw_api_key


def _retry_s3_operation(
    operation, max_retries=MAX_S3_RETRIES, base_delay=S3_BASE_DELAY
):
    """Retry S3 operations with exponential backoff for rate limiting."""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            # Check if it's a rate limiting error
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"S3 rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
            # Re-raise the exception if it's not rate limiting or max retries reached
            raise e
    raise Exception(f"Failed after {max_retries} attempts")


def download_if_newer(
    s3_bucket, s3_object_key, local_file_path, local_lmdb_path, initialDownload
):
    if initialDownload:
        config = TransferConfig(use_threads=True, max_bandwidth=None)
    else:
        config = TransferConfig(use_threads=False, max_bandwidth=S3_MAX_BANDWIDTH)

    # Initialize the S3 client
    if save_type == "S3":
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # Get the last modified timestamp of the S3 object
        # Use retry logic to handle rate limiting
        s3_response = _retry_s3_operation(
            lambda: s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
        )
        s3_last_modified = s3_response["LastModified"].timestamp()
    else:
        # If saved locally, get the last modified timestamp of the local file
        s3_last_modified = os.path.getmtime(s3_bucket + "/" + s3_object_key)

    newFile = False

    # Check if the local file exists
    # Read pickle with last modified time
    if os.path.exists(local_file_path + ".modtime.pickle"):
        # Open the file in binary mode
        with open(local_file_path + ".modtime.pickle", "rb") as file:
            # Deserialize and retrieve the variable from the file
            local_last_modified = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if s3_last_modified > local_last_modified:
            # Download the file
            if save_type == "S3":
                _retry_s3_operation(
                    lambda: s3_client.download_file(
                        s3_bucket, s3_object_key, local_file_path, Config=config
                    )
                )
            else:
                # Copy the local file over
                shutil.copy(s3_bucket + "/" + s3_object_key, local_file_path)

            newFile = True
            with open(local_file_path + ".modtime.pickle", "wb") as file:
                # Serialize and write the variable to the file
                pickle.dump(s3_last_modified, file)

        else:
            (f"{s3_object_key} is already up to date.")

    else:
        # Download the file
        if save_type == "S3":
            _retry_s3_operation(
                lambda: s3_client.download_file(
                    s3_bucket, s3_object_key, local_file_path, Config=config
                )
            )
        else:
            # Otherwise copy local file
            shutil.copy(s3_bucket + "/" + s3_object_key, local_file_path)

        with open(local_file_path + ".modtime.pickle", "wb") as file:
            # Serialize and write the variable to the file
            pickle.dump(s3_last_modified, file)

        newFile = True
        # Untar the file
        # shutil.unpack_archive(local_file_path, extract_path, 'tar')

    if newFile:
        # Write a file to show an update is in progress, do not reload
        with open(local_lmdb_path + ".lock", "w"):
            pass

        # Rename
        shutil.move(local_file_path, local_lmdb_path + "_" + str(s3_last_modified))

        # ZipZarr.close()
        # os.remove(local_file_path)
        os.remove(local_lmdb_path + ".lock")


logger = logging.getLogger("dataSync")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def find_largest_integer_directory(parent_dir, key_string, initialRun):
    largest_value = LARGEST_DIR_INIT
    largest_dir = None
    old_dirs = []

    STAGE = os.environ.get("STAGE", "PROD")

    for entry in os.listdir(parent_dir):
        # entry_path = os.path.join(parent_dir, entry)
        if (key_string in entry) & ("TMP" not in entry):
            old_dirs.append(entry)
            try:
                # Extract the integer value from the directory name
                value = float(
                    entry[-12:]
                )  # No constant needed, this is a filename slice

                if value > largest_value:
                    largest_value = value
                    largest_dir = entry
            except ValueError:
                # If the directory name is not an integer, skip it
                continue

    # Remove the latest dir from old_dirs
    if STAGE == "PROD":
        old_dirs.remove(largest_dir)

    if (not initialRun) & (len(old_dirs) == 0):
        largest_dir = None

    return largest_dir, old_dirs


def update_zarr_store(initialRun):
    global ETOPO_f
    global SubH_Zarr
    global HRRR_6H_Zarr
    global GFS_Zarr
    global NBM_Zarr
    global NBM_Fire_Zarr
    global GEFS_Zarr
    global HRRR_Zarr
    global NWS_Alerts_Zarr

    STAGE = os.environ.get("STAGE", "PROD")
    # Create empty dir
    os.makedirs("/tmp/empty", exist_ok=True)

    # Find the latest file that's ready
    latest_Alert, old_Alert = find_largest_integer_directory(
        "/tmp", "NWS_Alerts.zarr", initialRun
    )
    if latest_Alert is not None:
        NWS_Alerts_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_Alert, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_Alert)
    for old_dir in old_Alert:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n {NICE_PRIORITY} rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n {NICE_PRIORITY} rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_SubH, old_SubH = find_largest_integer_directory(
        "/tmp", "SubH.zarr", initialRun
    )
    if latest_SubH is not None:
        SubH_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_SubH, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_SubH)
    for old_dir in old_SubH:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_HRRR_6H, old_HRRR_6H = find_largest_integer_directory(
        "/tmp", "HRRR_6H.zarr", initialRun
    )
    if latest_HRRR_6H is not None:
        HRRR_6H_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_HRRR_6H, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_HRRR_6H)
    for old_dir in old_HRRR_6H:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_GFS, old_GFS = find_largest_integer_directory("/tmp", "GFS.zarr", initialRun)
    if latest_GFS is not None:
        GFS_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_GFS, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_GFS)
    for old_dir in old_GFS:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_NBM, old_NBM = find_largest_integer_directory("/tmp", "NBM.zarr", initialRun)
    if latest_NBM is not None:
        NBM_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_NBM, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_NBM)
    for old_dir in old_NBM:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_NBM_Fire, old_NBM_Fire = find_largest_integer_directory(
        "/tmp", "NBM_Fire.zarr", initialRun
    )
    if latest_NBM_Fire is not None:
        NBM_Fire_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_NBM_Fire, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_NBM_Fire)
    for old_dir in old_NBM_Fire:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_GEFS, old_GEFS = find_largest_integer_directory(
        "/tmp", "GEFS.zarr", initialRun
    )
    if latest_GEFS is not None:
        GEFS_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_GEFS, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_GEFS)
    for old_dir in old_GEFS:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    latest_HRRR, old_HRRR = find_largest_integer_directory(
        "/tmp", "HRRR.zarr", initialRun
    )
    if latest_HRRR is not None:
        HRRR_Zarr = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_HRRR, mode="r"), mode="r"
        )
        logger.info("Loading new: " + latest_HRRR)
    for old_dir in old_HRRR:
        if STAGE == "PROD":
            logger.info("Removing old: " + old_dir)
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
            subprocess.run(command, shell=True)

    if (initialRun) and (useETOPO):
        latest_ETOPO, old_ETOPO = find_largest_integer_directory(
            "/tmp", "ETOPO_DA_C.zarr", initialRun
        )
        ETOPO_f = zarr.open(
            zarr.storage.ZipStore("/tmp/" + latest_ETOPO, mode="r"), mode="r"
        )

    print("Refreshed Zarrs")


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()


def solar_rad(D_t, lat, t_t):
    """
    returns The theortical clear sky short wave radiation
    https://www.mdpi.com/2072-4292/5/10/4735/htm
    """

    d = 1 + SOLAR_RAD_CONST["eccentricity"] * math.sin(
        (2 * math.pi * (D_t - SOLAR_RAD_CONST["offset"])) / 365
    )
    r = SOLAR_RAD_CONST["r"]
    S_0 = SOLAR_RAD_CONST["S0"]
    delta = SOLAR_RAD_CONST["delta_factor"] * math.sin(
        (2 * math.pi * (D_t + SOLAR_RAD_CONST["delta_offset"])) / 365
    )
    radLat = np.deg2rad(lat)
    solarHour = math.pi * ((t_t - SOLAR_RAD_CONST["hour_offset"]) / 12)
    cosTheta = math.sin(delta) * math.sin(radLat) + math.cos(delta) * math.cos(
        radLat
    ) * math.cos(solarHour)
    R_s = r * (S_0 / d**2) * cosTheta

    if R_s < 0:
        R_s = 0

    return R_s


def toTimestamp(d):
    return d.timestamp()


# If testing, read zarrs directly from S3
# This should be implemented as a fallback at some point
STAGE = os.environ.get("STAGE", "PROD")
if STAGE == "TESTING":
    print("Setting up S3 zarrs")
    # If S3, use that, otherwise use local
    if save_type == "S3":
        # s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key, asynchronous=False)
        s3 = s3fs.S3FileSystem(
            anon=True,
            asynchronous=False,
            endpoint_url="https://api.pirateweather.net/files/",
        )
        s3.s3.meta.events.register("before-sign.s3.*", _add_custom_header)

        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/" + ingestVersion + "/NWS_Alerts.zarr.zip"
                )
            )
            store = S3ZipStore(f)
        # Try an old ingest version for testing
        except FileNotFoundError:
            ingestVersion = "v27"
            print("Using old ingest version: " + ingestVersion)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/" + ingestVersion + "/NWS_Alerts.zarr.zip"
                )
            )
            store = S3ZipStore(f)

    elif save_type == "S3Zarr":
        s3 = s3fs.S3FileSystem(
            key=aws_access_key_id, secret=aws_secret_access_key, version_aware=True
        )

        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingestVersion
                    + "/NWS_Alerts.zarr.zip"
                )
            )
            store = S3ZipStore(f)
        except FileNotFoundError:
            ingestVersion = "v27"
            print("Using old ingest version: " + ingestVersion)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingestVersion
                    + "/NWS_Alerts.zarr.zip"
                )
            )
            store = S3ZipStore(f)

    else:
        f = s3_bucket + "NWS_Alerts.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    NWS_Alerts_Zarr = zarr.open(store, mode="r")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open("s3://ForecastTar_v2/" + ingestVersion + "/SubH.zarr.zip")
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/SubH.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    else:
        f = s3_bucket + "SubH_v2.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    SubH_Zarr = zarr.open(store, mode="r")
    print("SubH Read")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://ForecastTar_v2/" + ingestVersion + "/HRRR_6H.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/HRRR_6H.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    else:
        f = s3_bucket + "HRRR_6H.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    HRRR_6H_Zarr = zarr.open(store, mode="r")
    print("HRRR_6H Read")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open("s3://ForecastTar_v2/" + ingestVersion + "/GFS.zarr.zip")
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/GFS.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    else:
        f = s3_bucket + "GFS.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    GFS_Zarr = zarr.open(store, mode="r")
    print("GFS Read")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open("s3://ForecastTar_v2/" + ingestVersion + "/GEFS.zarr.zip")
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/GEFS.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    else:
        f = s3_bucket + "GEFS.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    GEFS_Zarr = zarr.open(store, mode="r")
    print("GEFS Read")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open("s3://ForecastTar_v2/" + ingestVersion + "/NBM.zarr.zip")
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        # print('USE VERSION NBM')
        # f = s3.open("s3://" + s3_bucket + "/NBM.zarr.zip",
        #             version_id="sfWxulLYHDWCQTiM2u0v.x_Sg4pTwpG7")
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/NBM.zarr.zip"
            )
        )

        store = S3ZipStore(f)
    else:
        f = s3_bucket + "NBM.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    NBM_Zarr = zarr.open(store, mode="r")
    print("NBM Read")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://ForecastTar_v2/" + ingestVersion + "/NBM_Fire.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/NBM_Fire.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    else:
        f = s3_bucket + "NBM_Fire.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    NBM_Fire_Zarr = zarr.open(store, mode="r")
    print("NBM Fire Read")

    if save_type == "S3":
        f = _retry_s3_operation(
            lambda: s3.open("s3://ForecastTar_v2/" + ingestVersion + "/HRRR.zarr.zip")
        )
        store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        f = _retry_s3_operation(
            lambda: s3.open(
                "s3://"
                + s3_bucket
                + "/ForecastTar_v2/"
                + ingestVersion
                + "/HRRR.zarr.zip"
            )
        )
        store = S3ZipStore(f)
    else:
        f = s3_bucket + "HRRR.zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    HRRR_Zarr = zarr.open(store, mode="r")
    print("HRRR Read")

    if useETOPO:
        if save_type == "S3":
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/" + ingestVersion + "/ETOPO_DA_C.zarr.zip"
                )
            )
            store = S3ZipStore(f)
        elif save_type == "S3Zarr":
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingestVersion
                    + "/ETOPO_DA_C.zarr.zip"
                )
            )
            store = S3ZipStore(f)
        else:
            f = s3_bucket + "ETOPO_DA_C.zarr.zip"
            store = zarr.storage.ZipStore(f, mode="r")

        ETOPO_f = zarr.open(store, mode="r")
    print("ETOPO Read")


async def get_zarr(store, X, Y):
    return store[:, :, X, Y]


lats_etopo = np.arange(-90, 90, 0.01666667)
lons_etopo = np.arange(-180, 180, 0.01666667)

tf = TimezoneFinder(in_memory=True)


def get_offset(*, lat, lng, utcTime, tf):
    # tf = TimezoneFinder()
    """
    returns a location's time zone offset from UTC in minutes.
    """

    today = utcTime
    tz_target = timezone(tf.timezone_at(lng=lng, lat=lat))
    # ATTENTION: tz_target could be None! handle error case
    today_target = tz_target.localize(today)
    today_utc = utc.localize(today)
    return (today_utc - today_target).total_seconds() / 60, tz_target


def has_interior_nan_holes(arr: np.ndarray) -> bool:
    """
    Return True if `arr` (2D: rows × cols) contains at least one
    contiguous block of NaNs that:
      - does *not* touch the first or last column
      - has at least one NaN
    """
    # 1) make a mask of NaNs
    mask = np.isnan(arr)

    # 2) pad left/right with False so that edges never count as run boundaries
    #    padded.shape == (rows, cols+2)
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)

    # 3) compute a 1D diff along each row:
    #    diff == +1  → run *start* (False→True)
    #    diff == -1  → run *end*   (True→False)
    #    diff.shape == (rows, cols+1)
    diff = padded[:, 1:].astype(int) - padded[:, :-1].astype(int)
    starts = diff == 1  # potential run‐starts
    ends = diff == -1  # potential run‐ends

    # 4) ignore any that occur at the very first or last original column:
    #    we only want starts/ends in columns 1…(cols-2)
    interiorStarts = starts[:, 1:-1]
    interiorEnds = ends[:, 1:-1]

    # 5) a row has an interior hole iff it has at least one interior start
    #    *and* at least one interior end.  If any row meets that, we’re done.
    rowHasStart = interiorStarts.any(axis=1)
    rowHasEnd = interiorEnds.any(axis=1)

    return bool(np.any(rowHasStart & rowHasEnd))


# Interpolation function to interpolate nans in a row, keeping nan's at the start and end
def _interp_row(row: np.ndarray) -> np.ndarray:
    """
    Fill only strictly interior NaN‐runs in a 1D array
    (i.e. ignore any NaNs at index 0 or -1) by linear interpolation.
    """
    n = row.size
    x = np.arange(n)

    # mask of all NaNs
    mask = np.isnan(row)

    if mask.any() and not mask.all():
        good = ~mask

        # interp only at mask positions, using the remaining points
        row[mask] = np.interp(x[mask], x[good], row[good], left=np.nan, right=np.nan)

    return row


class WeatherParallel(object):
    async def zarr_read(self, model, opened_zarr, x, y):
        if TIMING:
            print("### " + model + " Reading!")
            print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None))

        errCount = 0
        dataOut = False
        # Try to read Zarr file
        while errCount < 4:
            try:
                dataOut = await asyncio.to_thread(lambda: opened_zarr[:, :, y, x].T)

                # Fake some bad data for testing
                # if model == "GFS":
                # dataOut[10:100, 4] = np.nan

                # Check for missing/ bad data and interpolate
                # This should not occur, but good to have a fallback
                if has_interior_nan_holes(dataOut.T):
                    print("### " + model + " Interpolating missing data!")

                    # Print the location of the missing data
                    if TIMING:
                        print(
                            "### " + model + " Missing data at: ",
                            np.argwhere(np.isnan(dataOut)),
                        )

                    dataOut = np.apply_along_axis(_interp_row, 0, dataOut)

                if TIMING:
                    print("### " + model + " Done!")
                    print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None))
                return dataOut

            except Exception:
                print("### " + model + " Failure!")
                errCount = errCount + 1
                print(traceback.print_exc())

        print("### " + model + " Failure!")
        dataOut = False
        return dataOut


def cull(lng, lat):
    """Accepts a list of lat/lng tuples.
    returns the list of tuples that are within the bounding box for the US.
    NB. THESE ARE NOT NECESSARILY WITHIN THE US BORDERS!
    https://gist.github.com/jsundram/1251783
    """

    ### TODO: Add Alaska somehow

    top = US_BOUNDING_BOX["top"]
    left = US_BOUNDING_BOX["left"]
    right = US_BOUNDING_BOX["right"]
    bottom = US_BOUNDING_BOX["bottom"]

    inside_box = 0
    if (bottom <= lat <= top) and (left <= lng <= right):
        inside_box = 1

    return inside_box


def lambertGridMatch(
    central_longitude,
    central_latitude,
    standard_parallel,
    semimajor_axis,
    lat,
    lon,
    hrrr_minX,
    hrrr_minY,
    hrrr_delta,
):
    # From https://en.wikipedia.org/wiki/Lambert_conformal_conic_projection

    hrr_n = math.sin(standard_parallel)
    hrrr_F = (
        math.cos(standard_parallel)
        * (math.tan(0.25 * math.pi + 0.5 * standard_parallel)) ** hrr_n
    ) / hrr_n
    hrrr_p = (
        semimajor_axis
        * hrrr_F
        * 1
        / (math.tan(0.25 * math.pi + 0.5 * math.radians(lat)) ** hrr_n)
    )
    hrrr_p0 = (
        semimajor_axis
        * hrrr_F
        * 1
        / (math.tan(0.25 * math.pi + 0.5 * central_latitude) ** hrr_n)
    )

    x_hrrrLoc = hrrr_p * math.sin(hrr_n * (math.radians(lon) - central_longitude))
    y_hrrrLoc = hrrr_p0 - hrrr_p * math.cos(
        hrr_n * (math.radians(lon) - central_longitude)
    )

    x_hrrr = round((x_hrrrLoc - hrrr_minX) / hrrr_delta)
    y_hrrr = round((y_hrrrLoc - hrrr_minY) / hrrr_delta)

    x_grid = x_hrrr * hrrr_delta + hrrr_minX
    y_grid = y_hrrr * hrrr_delta + hrrr_minY

    hrrr_p2 = math.copysign(math.sqrt(x_grid**2 + (hrrr_p0 - y_grid) ** 2), hrr_n)

    lat_grid = math.degrees(
        2 * math.atan((semimajor_axis * hrrr_F / hrrr_p2) ** (1 / hrr_n)) - math.pi / 2
    )

    hrrr_theta = math.atan((x_grid) / (hrrr_p0 - y_grid))

    lon_grid = math.degrees(central_longitude + hrrr_theta / hrr_n)

    return lat_grid, lon_grid, x_hrrr, y_hrrr


def rounder(t):
    if t.minute >= 30:
        # Round up to the next hour
        rounded_dt = t.replace(second=0, microsecond=0, minute=0) + datetime.timedelta(
            hours=1
        )
    else:
        # Round down to the current hour
        rounded_dt = t.replace(second=0, microsecond=0, minute=0)
    return rounded_dt


def unix_to_day_of_year_and_lst(dt, longitude):
    # Calculate the day of the year
    day_of_year = dt.timetuple().tm_yday

    # Calculate UTC time in hours
    utc_time = dt.hour + dt.minute / 60 + dt.second / 3600
    print(utc_time)

    # Calculate Local Solar Time (LST) considering the longitude
    lst = utc_time + (longitude / 15)
    print(lst)

    return day_of_year, lst


def solar_irradiance(latitude, longitude, unix_time):
    G_sc = SOLAR_IRRADIANCE_CONST["GSC"]

    # Get the day of the year and Local Solar Time (LST)
    day_of_year, local_solar_time = unix_to_day_of_year_and_lst(unix_time, longitude)

    # Calculate solar declination (delta) in radians
    delta = math.radians(SOLAR_IRRADIANCE_CONST["declination"]) * math.sin(
        math.radians(360 / 365 * (284 + day_of_year))
    )

    # Calculate hour angle (H) in degrees, then convert to radians
    H = math.radians(15 * (local_solar_time - 12))

    # Convert latitude to radians
    phi = math.radians(latitude)

    # Calculate solar elevation angle (alpha)
    sin_alpha = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(
        delta
    ) * math.cos(H)

    # Calculate air mass (AM)
    AM = 1 / sin_alpha if sin_alpha > 0 else float("inf")
    G_0 = G_sc * (
        1
        + SOLAR_IRRADIANCE_CONST["g0_coeff"]
        * math.cos(math.radians(360 * day_of_year / 365))
    )
    G = (
        G_0 * sin_alpha * math.exp(-SOLAR_IRRADIANCE_CONST["am_coeff"] * AM)
        if sin_alpha > 0
        else 0
    )

    return G


def calculate_globe_temperature(
    air_temperature, solar_radiation, wind_speed, globe_diameter=0.15, emissivity=0.95
):
    """
    Estimate the globe temperature based on ambient temperature, solar radiation, and wind speed.

    Parameters:
    air_temperature (float): Ambient air temperature in degrees Celsius.
    solar_radiation (float): Solar radiation in watts per square meter (W/m²).
    wind_speed (float): Wind speed in meters per second (m/s).
    globe_diameter (float, optional): Diameter of the globe thermometer in meters (default is 0.15m).
    emissivity (float, optional): Emissivity of the globe (default is 0.95 for a black globe).

    Returns:
    float: Estimated globe temperature in degrees Celsius.
    """
    globe_temperature = air_temperature + (
        GLOBE_TEMP_CONST["factor"] * (solar_radiation ** GLOBE_TEMP_CONST["temp_exp"])
    ) / (
        emissivity
        * (globe_diameter ** GLOBE_TEMP_CONST["diam_exp"])
        * (wind_speed ** GLOBE_TEMP_CONST["wind_exp"])
    )
    return globe_temperature


def calculate_wbgt(
    temperature,
    humidity,
    wind_speed=None,
    solar_radiation=None,
    globe_temperature=None,
    in_sun=False,
):
    """
    Calculate the Wet-Bulb Globe Temperature (WBGT).

    Parameters:
    temperature (float): The ambient air temperature in degrees Celsius.
    humidity (float): The relative humidity as a percentage (0-100).
    wind_speed (float, optional): The wind speed in meters per second. Required if `in_sun` is True.
    solar_radiation (float, optional): Solar radiation in watts per square meter (W/m²). Used to calculate globe temperature if `globe_temperature` is not provided.
    globe_temperature (float, optional): The globe temperature in degrees Celsius. Required if `in_sun` is True and `solar_radiation` is not provided.
    in_sun (bool, optional): If True, calculates WBGT for sunny conditions using wind_speed and globe_temperature.

    Returns:
    float: The Wet-Bulb Globe Temperature in degrees Celsius.
    """
    if in_sun:
        if globe_temperature is None:
            if wind_speed is None or solar_radiation is None:
                raise ValueError(
                    "Wind speed and solar radiation must be provided if globe temperature is not provided for outdoor WBGT calculation."
                )
            globe_temperature = calculate_globe_temperature(
                temperature, solar_radiation, wind_speed
            )
        wbgt = (
            WBGT_CONST["temp_weight"] * temperature
            + WBGT_CONST["globe_weight"] * globe_temperature
            + WBGT_CONST["wind_weight"] * wind_speed
        )
    else:
        wbgt = WBGT_CONST["temp_weight"] * temperature + WBGT_CONST[
            "humidity_weight"
        ] * (humidity / 100.0 * temperature)

    return wbgt


def dbz_to_rate(dbz_array, precip_type_array, min_dbz=REFC_THRESHOLD):
    """
    Convert dBZ to precipitation rate (mm/h) using a Z-R relationship with soft threshold.

    Args:
        dbz_array (np.ndarray): Radar reflectivity in dBZ.
        precip_type_array (np.ndarray): Array of precipitation types ('rain' or 'snow').
        min_dbz (float): Minimum dBZ for soft thresholding. Values below this are scaled linearly.

    Returns:
        np.ndarray: Precipitation rate in mm/h.
    """
    # Ensure no negative dBZ values
    dbz_array = np.maximum(dbz_array, 0.0)

    # Convert dBZ to Z
    z_array = 10 ** (dbz_array / 10.0)

    # Initialize rate coefficients for rain
    a_array = np.full_like(dbz_array, DBZ_CONST["rain_a"], dtype=float)
    b_array = np.full_like(dbz_array, DBZ_CONST["rain_b"], dtype=float)
    snow_mask = precip_type_array == "snow"
    a_array[snow_mask] = DBZ_CONST["snow_a"]
    b_array[snow_mask] = DBZ_CONST["snow_b"]

    # Compute precipitation rate
    rate_array = (z_array / a_array) ** (1.0 / b_array)

    # Apply soft threshold for sub-threshold dBZ values
    below_threshold = dbz_array < min_dbz
    rate_array[below_threshold] *= dbz_array[below_threshold] / min_dbz

    # Final check: ensure no negative rates
    rate_array = np.maximum(rate_array, 0.0)
    return rate_array


@app.get("/timemachine/{apikey}/{location}", response_class=ORJSONResponse)
@app.get("/forecast/{apikey}/{location}", response_class=ORJSONResponse)
async def PW_Forecast(
    request: Request,
    location: str,
    units: Union[str, None] = None,
    extend: Union[str, None] = None,
    exclude: Union[str, None] = None,
    lang: Union[str, None] = None,
    version: Union[str, None] = None,
    tmextra: Union[str, None] = None,
    apikey: Union[str, None] = None,
    icon: Union[str, None] = None,
    extraVars: Union[str, None] = None,
) -> dict:
    global ETOPO_f
    global SubH_Zarr
    global HRRR_6H_Zarr
    global GFS_Zarr
    global NBM_Zarr
    global NBM_Fire_Zarr
    global GEFS_Zarr
    global HRRR_Zarr
    global NWS_Alerts_Zarr

    readHRRR = False
    readGFS = False
    readNBM = False
    readGEFS = False

    STAGE = os.environ.get("STAGE", "PROD")

    # Timing Check
    T_Start = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)

    # Current time
    if force_now is False:
        nowTime = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    else:
        # Force now for testing with static inputs
        nowTime = datetime.datetime.fromtimestamp(int(force_now), datetime.UTC).replace(
            tzinfo=None
        )

        print("Forced Current Time to:")
        print(nowTime)

    ### If developing in REPL, uncomment to provide static variables
    # location = "47.1756,27.594,1741126460"
    # units = "ca"
    # extend = None
    # exclude = None
    # lang = "en"
    # version = "2"
    # tmextra: None
    # apikey: None

    locationReq = location.split(",")

    # Get the location
    try:
        lat = float(locationReq[0])
        lon_IN = float(locationReq[1])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Location Specification")
        # return {
        #     'statusCode': 400,
        #     'body': json.dumps('Invalid Location Specification')
        # }
    lon = lon_IN % 360  # 0-360
    az_Lon = ((lon + 180) % 360) - 180  # -180-180

    if (lon_IN < -180) or (lon > 360):
        # print('ERROR')
        raise HTTPException(status_code=400, detail="Invalid Longitude")
    if (lat < -90) or (lat > 90):
        # print('ERROR')
        raise HTTPException(status_code=400, detail="Invalid Latitude")

    if len(locationReq) == 2:
        if STAGE == "TIMEMACHINE":
            raise HTTPException(status_code=400, detail="Missing Time Specification")

        else:
            utcTime = nowTime

    elif len(locationReq) == 3:
        # If time is specified as a unix time
        if locationReq[2].lstrip("-+").isnumeric():
            if float(locationReq[2]) > 0:
                utcTime = datetime.datetime.fromtimestamp(
                    float(locationReq[2]), datetime.UTC
                ).replace(tzinfo=None)
            elif float(locationReq[2]) < -100000:  # Very negatime time
                utcTime = datetime.datetime.fromtimestamp(
                    float(locationReq[2]), datetime.UTC
                ).replace(tzinfo=None)
            elif float(locationReq[2]) < 0:  # Negatime time
                utcTime = nowTime + datetime.timedelta(seconds=float(locationReq[2]))

        else:
            try:
                utcTime = datetime.datetime.strptime(
                    locationReq[2], "%Y-%m-%dT%H:%M:%S%z"
                )
                # Since it is in UTC time already
                utcTime = utcTime.replace(tzinfo=None)
            except Exception:
                try:
                    utcTime = datetime.datetime.strptime(
                        locationReq[2], "%Y-%m-%dT%H:%M:%S%Z"
                    )
                    # Since it is in UTC time already
                    utcTime = utcTime.replace(tzinfo=None)
                except Exception:
                    try:
                        localTime = datetime.datetime.strptime(
                            locationReq[2], "%Y-%m-%dT%H:%M:%S"
                        )

                        # If no time zome specified, assume local time, and convert
                        tz_offsetLocIN = {
                            "lat": lat,
                            "lng": az_Lon,
                            "utcTime": localTime,
                            "tf": tf,
                        }

                        tz_offsetIN, tz_name = get_offset(**tz_offsetLocIN)
                        utcTime = localTime - datetime.timedelta(minutes=tz_offsetIN)

                    except Exception:
                        # print('ERROR')
                        raise HTTPException(
                            status_code=400, detail="Invalid Time Specification"
                        )

    else:
        raise HTTPException(
            status_code=400, detail="Invalid Time or Location Specification"
        )

    timeMachine = False
    timeMachineNear = False
    # Set up translations
    if not lang:
        lang = "en"

    if icon != "pirate":
        icon = "darksky"

    # Check if langugage is supported
    if lang not in Translations:
        # Throw an error
        raise HTTPException(status_code=400, detail="Language Not Supported")

    translation = Translations[lang]

    if utcTime < datetime.datetime(2024, 5, 1):
        timeMachine = True

        if (
            ("localhost" in str(request.url))
            or ("timemachine" in str(request.url))
            or ("127.0.0.1" in str(request.url))
        ):
            TM_Response = await TimeMachine(
                lat, lon, az_Lon, utcTime, tf, units, exclude, lang, API_VERSION
            )

            return TM_Response
        else:
            raise HTTPException(
                status_code=400,
                detail="Requested Time is in the Past. Please Use Timemachine.",
            )
    elif (nowTime - utcTime) > datetime.timedelta(hours=24):
        # More than 47 hours ago must be time machine request
        if (
            ("localhost" in str(request.url))
            or ("timemachine" in str(request.url))
            or ("127.0.0.1" in str(request.url))
        ):
            timeMachine = True
        else:
            raise HTTPException(
                status_code=400,
                detail="Requested Time is in the Past. Please Use Timemachine.",
            )
            # lock.acquire(blocking=True, timeout=60)
    elif nowTime < utcTime:
        if (utcTime - nowTime) < datetime.timedelta(hours=1):
            utcTime = nowTime
        else:
            raise HTTPException(
                status_code=400, detail="Requested Time is in the Future"
            )
    elif (nowTime - utcTime) < datetime.timedelta(hours=24):
        # If within the last 24 hours, it may or may not be a timemachine request
        if "timemachine" in str(request.url):
            timeMachineNear = True
            # This results in the API using the live zip file, but only doing a 24 hour forecast from midnight of the requested day
            if TIMING:
                print("Near term timemachine request")
                # Print how far in the past it is
                print((nowTime - utcTime))
        # Otherwise, just a normal request

    # Timing Check
    if TIMING:
        print("Request process time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Calculate the timezone offset
    tz_offsetLoc = {"lat": lat, "lng": az_Lon, "utcTime": utcTime, "tf": tf}
    tz_offset, tz_name = get_offset(**tz_offsetLoc)

    tzReq = tf.timezone_at(lat=lat, lng=az_Lon)

    # Timing Check
    if TIMING:
        print("Timezone offset time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Set defaults
    if not extend:
        extendFlag = 0
    else:
        if extend == "hourly":
            extendFlag = 1
        else:
            extendFlag = 0

    if not version:
        version = 1

    version = float(version)

    # Check if extra information should be included with time machine
    if not tmextra:
        tmExtra = False
    else:
        tmExtra = True

    if not exclude:
        excludeParams = ""
    else:
        excludeParams = exclude

    if not extraVars:
        extraVars = []
    else:
        extraVars = extraVars.split(",")

    exCurrently = 0
    exMinutely = 0
    exHourly = 0
    exDaily = 0
    exFlags = 0
    exAlerts = 0
    exNBM = 0
    exHRRR = 0
    exGEFS = 0
    summaryText = True

    if "currently" in excludeParams:
        exCurrently = 1
    if "minutely" in excludeParams:
        exMinutely = 1
    if "hourly" in excludeParams:
        exHourly = 1
    if "daily" in excludeParams:
        exDaily = 1
    if "flags" in excludeParams:
        exFlags = 1
    if "alerts" in excludeParams:
        exAlerts = 1
    if "nbm" in excludeParams:
        exNBM = 1
    if "hrrr" in excludeParams:
        exHRRR = 1
    if "gefs" in excludeParams:
        exGEFS = 1
    if "summary" in excludeParams:
        summaryText = False

    # Set up timemache params
    if timeMachine and not tmExtra:
        exMinutely = 1

    if timeMachine:
        exAlerts = 1

    # Exclude Alerts outside US
    if exAlerts == 0:
        if cull(az_Lon, lat) == 0:
            exAlerts = 1

    # Default to US
    unitSystem = "us"
    windUnit = 2.234  # mph
    prepIntensityUnit = 0.0394  # inches/hour
    prepAccumUnit = 0.0394  # inches
    tempUnits = 0  # F. This is harder
    pressUnits = 0.01  # Hectopascals
    visUnits = 0.00062137  # miles
    humidUnit = 0.01  # %
    elevUnit = 3.28084  # ft

    if units:
        unitSystem = units[0:2]

        if unitSystem == "ca":
            windUnit = 3.600  # kph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        elif unitSystem == "uk":
            windUnit = 2.234  # mph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.00062137  # miles
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        elif unitSystem == "si":
            windUnit = 1  # m/s
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = KELVIN_TO_CELSIUS  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km
            humidUnit = 0.01  # %
            elevUnit = 1  # m

    weather = WeatherParallel()

    zarrTasks = dict()

    # Base times
    pytzTZ = timezone(tzReq)

    # utcTime  = datetime.datetime(year=2024, month=3, day=8, hour=6, minute=15)
    baseTime = utc.localize(
        datetime.datetime(
            year=utcTime.year,
            month=utcTime.month,
            day=utcTime.day,
            hour=utcTime.hour,
            minute=utcTime.minute,
        )
    ).astimezone(pytzTZ)
    baseHour = pytzTZ.localize(
        datetime.datetime(
            year=baseTime.year,
            month=baseTime.month,
            day=baseTime.day,
            hour=baseTime.hour,
        )
    )

    baseDay = baseTime.replace(hour=0, minute=0, second=0, microsecond=0)

    baseDayUTC = baseDay.astimezone(utc)

    # Find UTC time for the base day
    baseDayUTC_Grib = (
        (
            np.datetime64(baseDay.astimezone(utc).replace(tzinfo=None))
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
        )
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Timing Check
    if TIMING:
        print("### HRRR Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    sourceIDX = dict()

    # Ignore areas outside of HRRR coverage
    if az_Lon < -134 or az_Lon > -61 or lat < 21 or lat > 53 or exHRRR == 1:
        dataOut = False
        dataOut_hrrrh = False
        dataOut_h2 = False

    else:
        # HRRR
        central_longitude_hrrr = math.radians(262.5)
        central_latitude_hrrr = math.radians(38.5)
        standard_parallel_hrrr = math.radians(38.5)
        semimajor_axis_hrrr = 6371229
        hrrr_minX = -2697500
        hrrr_minY = -1587300
        hrrr_delta = 3000

        hrrr_lat, hrrr_lon, x_hrrr, y_hrrr = lambertGridMatch(
            central_longitude_hrrr,
            central_latitude_hrrr,
            standard_parallel_hrrr,
            semimajor_axis_hrrr,
            lat,
            lon,
            hrrr_minX,
            hrrr_minY,
            hrrr_delta,
        )

        if (
            (x_hrrr < HRRR_X_MIN)
            or (y_hrrr < HRRR_Y_MIN)
            or (x_hrrr > HRRR_X_MAX)
            or (y_hrrr > HRRR_Y_MAX)
        ):
            dataOut = False
            dataOut_h2 = False
            dataOut_hrrrh = False
        else:
            # Subh
            # Check if timemachine request, use different sources
            if timeMachine:
                date_range = pd.date_range(
                    start=baseDayUTC - datetime.timedelta(hours=1),
                    end=baseDayUTC + datetime.timedelta(days=1, hours=1),
                    freq="1h",
                ).to_list()
                if utcTime < datetime.datetime(2025, 6, 10):
                    zarrList = [
                        "s3://"
                        + s3_bucket
                        + "/HRRRH/HRRRH_Hist"
                        + t.strftime("%Y%m%dT%H0000Z")
                        + ".zarr/"
                        for t in date_range
                    ]
                    consolidateZarr = True
                else:
                    zarrList = [
                        "s3://"
                        + s3_bucket
                        + "/Hist_v2/HRRR/HRRR_Hist_v2"
                        + t.strftime("%Y%m%dT%H0000Z")
                        + ".zarr/"
                        for t in date_range
                    ]
                    consolidateZarr = False

                now = time.time()
                HRRRdropvars = []
                if utcTime < datetime.datetime(2025, 7, 7):
                    HRRRdropvars.append("DSWRF_surface")
                    HRRRdropvars.append("CAPE_surface")
                with xr.open_mfdataset(
                    zarrList,
                    engine="zarr",
                    consolidated=consolidateZarr,
                    decode_cf=False,
                    parallel=True,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                    cache=False,
                    drop_variables=HRRRdropvars,
                ) as xr_mf:
                    # Correct for Pressure Switch
                    if "PRES_surface" in xr_mf.data_vars:
                        HRRRHzarrVars = (
                            "time",
                            "VIS_surface",
                            "GUST_surface",
                            "PRES_surface",
                            "TMP_2maboveground",
                            "DPT_2maboveground",
                            "RH_2maboveground",
                            "UGRD_10maboveground",
                            "VGRD_10maboveground",
                            "PRATE_surface",
                            "APCP_surface",
                            "CSNOW_surface",
                            "CICEP_surface",
                            "CFRZR_surface",
                            "CRAIN_surface",
                            "TCDC_entireatmosphere",
                            "MASSDEN_8maboveground",
                            "REFC_entireatmosphere",
                        )
                    else:
                        HRRRHzarrVars = (
                            "time",
                            "VIS_surface",
                            "GUST_surface",
                            "MSLMA_meansealevel",
                            "TMP_2maboveground",
                            "DPT_2maboveground",
                            "RH_2maboveground",
                            "UGRD_10maboveground",
                            "VGRD_10maboveground",
                            "PRATE_surface",
                            "APCP_surface",
                            "CSNOW_surface",
                            "CICEP_surface",
                            "CFRZR_surface",
                            "CRAIN_surface",
                            "TCDC_entireatmosphere",
                            "MASSDEN_8maboveground",
                            "REFC_entireatmosphere",
                        )

                    dataOut_hrrrh = np.zeros((len(xr_mf.time), len(HRRRHzarrVars)))

                    # Add time
                    dataOut_hrrrh[:, 0] = xr_mf.time.compute().data

                    for vIDX, v in enumerate(HRRRHzarrVars[1:]):
                        if v in xr_mf.data_vars:
                            dataOut_hrrrh[:, vIDX + 1] = (
                                xr_mf[v][:, y_hrrr, x_hrrr].compute().data
                            )
                        else:
                            print("Variable not in HRRR Zarr:")
                            print(v)
                    now2 = time.time()

                # Timing Check
                if TIMING:
                    print("HRRRH Hist Time")
                    print(now2 - now)

                dataOut = False
                dataOut_h2 = False

                subhRunTime = 0
                hrrrhRunTime = 0
                h2RunTime = 0

                readHRRR = False
            else:
                readHRRR = True

        sourceIDX["hrrr"] = dict()
        sourceIDX["hrrr"]["x"] = int(x_hrrr)
        sourceIDX["hrrr"]["y"] = int(y_hrrr)
        sourceIDX["hrrr"]["lat"] = round(hrrr_lat, 2)
        sourceIDX["hrrr"]["lon"] = round(((hrrr_lon + 180) % 360) - 180, 2)

    # Timing Check
    if TIMING:
        print("### NBM Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)
    # Ignore areas outside of NBM coverage
    if az_Lon < -138.3 or az_Lon > -59 or lat < 19.3 or lat > 57 or exNBM == 1:
        dataOut_nbm = False
        dataOut_nbmFire = False
    else:
        # NBM
        central_longitude_nbm = math.radians(265)
        central_latitude_nbm = math.radians(25)
        standard_parallel_nbm = math.radians(25.0)
        semimajor_axis_nbm = 6371200
        nbm_minX = -3271152.8
        nbm_minY = -263793.46
        nbm_delta = 2539.703000

        nbm_lat, nbm_lon, x_nbm, y_nbm = lambertGridMatch(
            central_longitude_nbm,
            central_latitude_nbm,
            standard_parallel_nbm,
            semimajor_axis_nbm,
            lat,
            lon,
            nbm_minX,
            nbm_minY,
            nbm_delta,
        )

        if (
            (x_nbm < NBM_X_MIN)
            or (y_nbm < NBM_Y_MIN)
            or (x_nbm > NBM_X_MAX)
            or (y_nbm > NBM_Y_MAX)
        ):
            dataOut_nbm = False
            dataOut_nbmFire = False
        else:
            # Timing Check
            if TIMING:
                print("### NBM Detail Start ###")
                print(
                    datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
                )

            if timeMachine:
                date_range = pd.date_range(
                    start=baseDayUTC - datetime.timedelta(hours=1),
                    end=baseDayUTC + datetime.timedelta(days=1, hours=1),
                    freq="1h",
                ).to_list()

                if utcTime < datetime.datetime(2025, 6, 10):
                    zarrList = [
                        "s3://"
                        + s3_bucket
                        + "/NBM/NBM_Hist"
                        + t.strftime("%Y%m%dT%H0000Z")
                        + ".zarr/"
                        for t in date_range
                    ]
                    consolidateZarr = True

                else:
                    zarrList = [
                        "s3://"
                        + s3_bucket
                        + "/Hist_v2/NBM/NBM_Hist"
                        + t.strftime("%Y%m%dT%H0000Z")
                        + ".zarr/"
                        for t in date_range
                    ]
                    consolidateZarr = False

                now = time.time()

                NBMdropvars = []
                if utcTime < datetime.datetime(2025, 10, 5):
                    NBMdropvars.append("DSWRF_surface")
                    NBMdropvars.append("CAPE_surface")

                with xr.open_mfdataset(
                    zarrList,
                    engine="zarr",
                    consolidated=consolidateZarr,
                    decode_cf=False,
                    parallel=True,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                    cache=False,
                    drop_variables=NBMdropvars,
                ) as xr_mf:
                    now2 = time.time()
                    if TIMING:
                        print("NBM Open Time")
                        print(now2 - now)

                    # Correct for Pressure Switch
                    NBMzarrVars = (
                        "time",
                        "GUST_10maboveground",
                        "TMP_2maboveground",
                        "APTMP_2maboveground",
                        "DPT_2maboveground",
                        "RH_2maboveground",
                        "WIND_10maboveground",
                        "WDIR_10maboveground",
                        "APCP_surface",
                        "TCDC_surface",
                        "VIS_surface",
                        "PWTHER_surfaceMreserved",
                        "PPROB",
                        "PACCUM",
                        "PTYPE_prob_GE_1_LT_2_prob_fcst_1_1_surface",
                        "PTYPE_prob_GE_3_LT_4_prob_fcst_1_1_surface",
                        "PTYPE_prob_GE_5_LT_7_prob_fcst_1_1_surface",
                        "PTYPE_prob_GE_8_LT_9_prob_fcst_1_1_surface",
                    )

                    dataOut_nbm = np.zeros((len(xr_mf.time), len(NBMzarrVars)))
                    # Add time
                    dataOut_nbm[:, 0] = xr_mf.time.compute().data

                    for vIDX, v in enumerate(NBMzarrVars[1:]):
                        dataOut_nbm[:, vIDX + 1] = (
                            xr_mf[v][:, y_nbm, x_nbm].compute().data
                        )
                    now3 = time.time()

                if TIMING:
                    print("NBM Hist Time")
                    print(now3 - now)

                dataOut_nbmFire = False

                nbmRunTime = 0
                nbmFireRunTime = 0

                readNBM = False
            else:
                readNBM = True

    # Timing Check
    if TIMING:
        print("### GFS/GEFS Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # GFS
    lats_gfs = np.arange(-90, 90, 0.25)
    lons_gfs = np.arange(0, 360, 0.25)

    abslat = np.abs(lats_gfs - lat)
    abslon = np.abs(lons_gfs - lon)
    y_p = np.argmin(abslat)
    x_p = np.argmin(abslon)

    gfs_lat = lats_gfs[y_p]
    gfs_lon = lons_gfs[x_p]

    # Timing Check
    if TIMING:
        print("### GFS Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    if timeMachine:
        now = time.time()
        # Create list of zarrs
        # Negative 1 since the first timestep of a model run is used
        hours_to_subtract = (baseDayUTC.hour - 1) % 6
        rounded_time = baseDayUTC - datetime.timedelta(
            hours=hours_to_subtract + 1,
            minutes=baseDayUTC.minute,
            seconds=baseDayUTC.second,
            microseconds=baseDayUTC.microsecond,
        )

        date_range = pd.date_range(
            start=rounded_time,
            end=rounded_time + datetime.timedelta(days=1, hours=6),
            freq="6h",
        ).to_list()

        # Select either <v2.7 or >=v2.7 bucket
        if utcTime < datetime.datetime(2025, 6, 10):
            zarrList = [
                "s3://"
                + s3_bucket
                + "/GFS/GFS_Hist"
                + t.strftime("%Y%m%dT%H0000Z")
                + ".zarr/"
                for t in date_range
            ]
            consolidateZarr = True
        else:
            zarrList = [
                "s3://"
                + s3_bucket
                + "/Hist_v2/GFS/GFS_Hist_v2"
                + t.strftime("%Y%m%dT%H0000Z")
                + ".zarr/"
                for t in date_range
            ]
            consolidateZarr = False

        GFSdropvars = []

        # Check if before October 8, 2025, and drop "DSWRF_surface", "CAPE_surface" if so
        # This avoids issues with missing variable in earlier files
        if utcTime < datetime.datetime(2025, 10, 5):
            GFSdropvars.append("DSWRF_surface")
            GFSdropvars.append("CAPE_surface")
            GFSdropvars.append("PRES_station")
            GFSdropvars.append("DUVB_surface")

        # Fix an issue with the chunking of "PRES_surface" during september 2025
        if (utcTime >= datetime.datetime(2025, 9, 1)) and (
            utcTime < datetime.datetime(2025, 10, 1)
        ):
            GFSdropvars.append("PRES_surface")

        with xr.open_mfdataset(
            zarrList,
            engine="zarr",
            consolidated=consolidateZarr,
            decode_cf=False,
            parallel=True,
            storage_options={"key": aws_access_key_id, "secret": aws_secret_access_key},
            cache=False,
            drop_variables=GFSdropvars,
        ) as xr_mf:
            now2 = time.time()
            if TIMING:
                print("GFS Open Time")
                print(now2 - now)

            # Correct for Pressure Switch
            if "PRES_surface" in xr_mf.data_vars:
                GFSzarrVars = (
                    "time",
                    "VIS_surface",
                    "GUST_surface",
                    "PRES_surface",
                    "TMP_2maboveground",
                    "DPT_2maboveground",
                    "RH_2maboveground",
                    "APTMP_2maboveground",
                    "UGRD_10maboveground",
                    "VGRD_10maboveground",
                    "PRATE_surface",
                    "APCP_surface",
                    "CSNOW_surface",
                    "CICEP_surface",
                    "CFRZR_surface",
                    "CRAIN_surface",
                    "TOZNE_entireatmosphere_consideredasasinglelayer_",
                    "TCDC_entireatmosphere",
                    "DUVB_surface",
                    "Storm_Distance",
                    "Storm_Direction",
                    "REFC_entireatmosphere",
                )
            else:
                GFSzarrVars = (
                    "time",
                    "VIS_surface",
                    "GUST_surface",
                    "PRMSL_meansealevel",
                    "TMP_2maboveground",
                    "DPT_2maboveground",
                    "RH_2maboveground",
                    "APTMP_2maboveground",
                    "UGRD_10maboveground",
                    "VGRD_10maboveground",
                    "PRATE_surface",
                    "APCP_surface",
                    "CSNOW_surface",
                    "CICEP_surface",
                    "CFRZR_surface",
                    "CRAIN_surface",
                    "TOZNE_entireatmosphere_consideredasasinglelayer_",
                    "TCDC_entireatmosphere",
                    "DUVB_surface",
                    "Storm_Distance",
                    "Storm_Direction",
                    "REFC_entireatmosphere",
                    "PRES_station",
                )

            dataOut_gfs = np.full((len(xr_mf.time), len(GFSzarrVars)), np.nan)
            # Add time
            dataOut_gfs[:, 0] = xr_mf.time.compute().data

            for vIDX, v in enumerate(GFSzarrVars[1:]):
                if v in xr_mf.data_vars:
                    dataOut_gfs[:, vIDX + 1] = xr_mf[v][:, y_p, x_p].compute().data
                else:
                    print("Variable not in GFS Zarr:")
                    print(v)
            now3 = time.time()

        if TIMING:
            print("GFS Hist Time")
            print(now3 - now)

        gfsRunTime = 0

        readGFS = False
    else:
        readGFS = True

    # Timing Check
    if TIMING:
        print("### GFS Detail END ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # GEFS
    # Timing Check
    if TIMING:
        print("### GEFS Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)
    if exGEFS == 1:
        dataOut_gefs = False
    else:
        if timeMachine:
            now = time.time()
            # Create list of zarrs
            # Negative 3 since the first timestep (hour 3) of a model run is used
            hours_to_subtract = (baseDayUTC.hour - 3) % 6
            rounded_time = baseDayUTC - datetime.timedelta(
                hours=hours_to_subtract + 3,
                minutes=baseDayUTC.minute,
                seconds=baseDayUTC.second,
                microseconds=baseDayUTC.microsecond,
            )

            date_range = pd.date_range(
                start=rounded_time,
                end=rounded_time + datetime.timedelta(days=1, hours=6),
                freq="6h",
            ).to_list()

            if utcTime < datetime.datetime(2025, 6, 10):
                zarrList = [
                    "s3://"
                    + s3_bucket
                    + "/GEFS/GEFS_HistProb_"
                    + t.strftime("%Y%m%dT%H0000Z")
                    + ".zarr/"
                    for t in date_range
                ]
                consolidateZarr = True
            else:
                zarrList = [
                    "s3://"
                    + s3_bucket
                    + "/Hist_v2/GEFS/GEFS_HistProb_"
                    + t.strftime("%Y%m%dT%H0000Z")
                    + ".zarr/"
                    for t in date_range
                ]
                consolidateZarr = False
            with xr.open_mfdataset(
                zarrList,
                engine="zarr",
                consolidated=consolidateZarr,
                decode_cf=False,
                parallel=True,
                storage_options={
                    "key": aws_access_key_id,
                    "secret": aws_secret_access_key,
                },
                cache=False,
            ) as xr_mf:
                GEFSzarrVars = (
                    "time",
                    "Precipitation_Prob",
                    "APCP_Mean",
                    "APCP_StdDev",
                    "CSNOW_Prob",
                    "CICEP_Prob",
                    "CFRZR_Prob",
                    "CRAIN_Prob",
                )

                dataOut_gefs = np.zeros((len(xr_mf.time), len(GEFSzarrVars)))
                # Add time
                dataOut_gefs[:, 0] = xr_mf.time.compute().data
                for vIDX, v in enumerate(GEFSzarrVars[1:]):
                    dataOut_gefs[:, vIDX + 1] = xr_mf[v][:, y_p, x_p].compute().data
                now2 = time.time()

            if TIMING:
                print("GEFS Hist Time")
                print(now2 - now)

            gefsRunTime = 0

            readGEFS = False
        else:
            readGEFS = True

    # Timing Check
    if TIMING:
        print("### GEFS Detail Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    sourceIDX["gfs"] = dict()
    sourceIDX["gfs"]["x"] = int(x_p)
    sourceIDX["gfs"]["y"] = int(y_p)
    sourceIDX["gfs"]["lat"] = round(gfs_lat, 2)
    sourceIDX["gfs"]["lon"] = round(((gfs_lon + 180) % 360) - 180, 2)

    if readHRRR:
        zarrTasks["SubH"] = weather.zarr_read("SubH", SubH_Zarr, x_hrrr, y_hrrr)

        # HRRR_6H
        zarrTasks["HRRR_6H"] = weather.zarr_read(
            "HRRR_6H", HRRR_6H_Zarr, x_hrrr, y_hrrr
        )

        # HRRR
        zarrTasks["HRRR"] = weather.zarr_read("HRRR", HRRR_Zarr, x_hrrr, y_hrrr)

    if readNBM:
        zarrTasks["NBM"] = weather.zarr_read("NBM", NBM_Zarr, x_nbm, y_nbm)
        zarrTasks["NBM_Fire"] = weather.zarr_read(
            "NBM_Fire", NBM_Fire_Zarr, x_nbm, y_nbm
        )

    if readGFS:
        zarrTasks["GFS"] = weather.zarr_read("GFS", GFS_Zarr, x_p, y_p)

    if readGEFS:
        zarrTasks["GEFS"] = weather.zarr_read("GEFS", GEFS_Zarr, x_p, y_p)

    results = await asyncio.gather(*zarrTasks.values())
    zarr_results = {key: result for key, result in zip(zarrTasks.keys(), results)}

    if readHRRR:
        dataOut = zarr_results["SubH"]
        dataOut_h2 = zarr_results["HRRR_6H"]
        dataOut_hrrrh = zarr_results["HRRR"]

        if (
            (dataOut is not False)
            and (dataOut_h2 is not False)
            and (dataOut_hrrrh is not False)
        ):
            # Calculate run times from specific time step for each model
            subhRunTime = dataOut[0, 0]

            # Check if the model times are valid for the request time
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    subhRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=4):
                dataOut = False
                print("OLD SubH")

            hrrrhRunTime = dataOut_hrrrh[48, 0]
            # print( datetime.datetime.fromtimestamp(dataOut_hrrrh[35, 0].astype(int)))
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    hrrrhRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=16):
                dataOut_hrrrh = False
                print("OLD HRRRH")

            h2RunTime = dataOut_h2[0, 0]
            if (
                utcTime
                - datetime.datetime.fromtimestamp(
                    h2RunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ) > datetime.timedelta(hours=46):
                dataOut_h2 = False
                print("OLD HRRR_6H")

    if readNBM:
        dataOut_nbm = zarr_results["NBM"]
        dataOut_nbmFire = zarr_results["NBM_Fire"]

        if dataOut_nbm is not False:
            nbmRunTime = dataOut_nbm[48, 0]

        sourceIDX["nbm"] = dict()
        sourceIDX["nbm"]["x"] = int(x_nbm)
        sourceIDX["nbm"]["y"] = int(y_nbm)
        sourceIDX["nbm"]["lat"] = round(nbm_lat, 2)
        sourceIDX["nbm"]["lon"] = round(((nbm_lon + 180) % 360) - 180, 2)

        # Timing Check
        if TIMING:
            print("### NMB Detail End ###")
            print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

        if dataOut_nbmFire is not False:
            nbmFireRunTime = dataOut_nbmFire[42, 0]  # 48-6

    if readGFS:
        dataOut_gfs = zarr_results["GFS"]
        if dataOut_gfs is not False:
            gfsRunTime = dataOut_gfs[47, 0]  # 48-1

    if readGEFS:
        dataOut_gefs = zarr_results["GEFS"]
        gefsRunTime = dataOut_gefs[45, 0]  # 48-3

    sourceTimes = dict()
    if timeMachine is False:
        if useETOPO:
            sourceList = ["ETOPO1", "gfs"]
        else:
            sourceList = ["gfs"]
    else:
        sourceList = ["gfs"]

    # Timing Check
    if TIMING:
        print("### Sources Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # If point is not in HRRR coverage or HRRR-subh is more than 4 hours old, the fallback to GFS
    if isinstance(dataOut, np.ndarray):
        sourceList.append("hrrrsubh")
        sourceTimes["hrrr_subh"] = rounder(
            datetime.datetime.fromtimestamp(
                subhRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

    if (isinstance(dataOut_hrrrh, np.ndarray)) & (not timeMachine):
        sourceList.append("hrrr_0-18")
        sourceTimes["hrrr_0-18"] = rounder(
            datetime.datetime.fromtimestamp(
                hrrrhRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_hrrrh, np.ndarray)) & (timeMachine):
        sourceList.append("hrrr")

    if (isinstance(dataOut_nbm, np.ndarray)) & (not timeMachine):
        sourceList.append("nbm")
        sourceTimes["nbm"] = rounder(
            datetime.datetime.fromtimestamp(
                nbmRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_nbm, np.ndarray)) & (timeMachine):
        sourceList.append("nbm")

    if (isinstance(dataOut_nbmFire, np.ndarray)) & (not timeMachine):
        sourceList.append("nbm_fire")
        sourceTimes["nbm_fire"] = rounder(
            datetime.datetime.fromtimestamp(
                nbmFireRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

    # If point is not in HRRR coverage or HRRR-hrrrh is more than 16 hours old, the fallback to GFS
    if isinstance(dataOut_h2, np.ndarray):
        sourceList.append("hrrr_18-48")
        # Subtract 18 hours since we're using the 18h time steo
        sourceTimes["hrrr_18-48"] = rounder(
            datetime.datetime.fromtimestamp(
                h2RunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
            - datetime.timedelta(hours=18)
        ).strftime("%Y-%m-%d %HZ")

    # Always include GFS
    if timeMachine is False:
        sourceTimes["gfs"] = rounder(
            datetime.datetime.fromtimestamp(
                gfsRunTime.astype(int), datetime.UTC
            ).replace(tzinfo=None)
        ).strftime("%Y-%m-%d %HZ")

        if isinstance(dataOut_gefs, np.ndarray):
            sourceList.append("gefs")
            sourceTimes["gefs"] = rounder(
                datetime.datetime.fromtimestamp(
                    gefsRunTime.astype(int), datetime.UTC
                ).replace(tzinfo=None)
            ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_gefs, np.ndarray)) & (timeMachine):
        sourceList.append("gefs")

    # Timing Check
    if TIMING:
        print("### ETOPO Start ###")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    ## ELEVATION
    abslat = np.abs(lats_etopo - lat)
    abslon = np.abs(lons_etopo - az_Lon)
    y_p = np.argmin(abslat)
    x_p = np.argmin(abslon)

    if (useETOPO) and ((STAGE == "PROD") or (STAGE == "DEV")):
        ETOPO = int(ETOPO_f[y_p, x_p])
    else:
        ETOPO = 0

    if ETOPO < 0:
        ETOPO = 0

    if useETOPO:
        sourceIDX["etopo"] = dict()
        sourceIDX["etopo"]["x"] = int(x_p)
        sourceIDX["etopo"]["y"] = int(y_p)
        sourceIDX["etopo"]["lat"] = round(lats_etopo[y_p], 4)
        sourceIDX["etopo"]["lon"] = round(lons_etopo[x_p], 4)

    # Timing Check
    if TIMING:
        print("Base Times")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Number of hours to start at
    if timeMachine:
        baseTimeOffset = 0
    else:
        baseTimeOffset = (baseHour - baseDay).seconds / 3600

    # Merge hourly models onto a consistent time grid, starting from midnight on the requested day
    # Note that baseTime is the requested time, in TZ aware datetime format
    ### Minutely
    minute_array = np.arange(
        baseTime.astimezone(utc).replace(tzinfo=None),
        baseTime.astimezone(utc).replace(tzinfo=None) + datetime.timedelta(minutes=61),
        datetime.timedelta(minutes=1),
    )

    minute_array_grib = (
        (minute_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    InterTminute = np.zeros((61, 5))  # Type
    InterPminute = np.full((61, 4), np.nan)  # Time, Intensity,Probability

    # Setup the time parameters for output and processing
    if timeMachine:
        daily_days = 1  # Number of days to output
        daily_day_hours = 1  # Additional hours to use in the processing
        ouputHours = 24
        ouputDays = 1

    elif timeMachineNear:
        daily_days = 8
        daily_day_hours = 5
        ouputHours = 24
        ouputDays = 1

    else:
        daily_days = 8
        daily_day_hours = 5

        if extendFlag:
            ouputHours = 168
        else:
            ouputHours = 48
        ouputDays = 8

    hour_array = np.arange(
        baseDay.astimezone(utc).replace(tzinfo=None),
        baseDay.astimezone(utc).replace(tzinfo=None)
        + datetime.timedelta(days=daily_days)
        + datetime.timedelta(hours=daily_day_hours),
        datetime.timedelta(hours=1),
    )

    numHours = len(hour_array)

    InterPhour = np.full((numHours, 28), np.nan)  # Time, Intensity,Probability

    hour_array_grib = (
        (hour_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Timing Check
    if TIMING:
        print("Nearest IDX Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # HRRR
    if timeMachine is False:
        # Since the forecast files are pre-processed, they'll always be hourly and the same length. This avoids interpolation
        try:  # Add a fallback to GFS if these don't work
            # HRRR
            if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
                HRRR_StartIDX = nearest_index(dataOut_hrrrh[:, 0], baseDayUTC_Grib)
                H2_StartIDX = nearest_index(dataOut_h2[:, 0], dataOut_hrrrh[-1, 0]) + 1

                if (H2_StartIDX < 1) or (HRRR_StartIDX < 2):
                    if "hrrr_18-48" in sourceTimes:
                        sourceTimes.pop("hrrr_18-48", None)
                    if "hrrr_18-48" in sourceTimes:
                        sourceTimes.pop("hrrr_18-48", None)
                    if "hrrr_0-18" in sourceTimes:
                        sourceTimes.pop("hrrr_0-18", None)
                    if "hrrr_0-18" in sourceTimes:
                        sourceTimes.pop("hrrr_0-18", None)

                    # Log the error
                    logger.error(
                        "HRRR data not available for the requested time range."
                    )

                else:
                    HRRR_Merged = np.full((numHours, dataOut_h2.shape[1]), np.nan)
                    # The 0-18 hour HRRR data (dataOut_hrrrh) has fewer columns than the 18-48 hour data (dataOut_h2)
                    # when in timeMachine mode. Only concatenate the common columns (0-17).
                    common_cols = min(dataOut_hrrrh.shape[1], dataOut_h2.shape[1])
                    HRRR_Merged[
                        0 : (67 - HRRR_StartIDX) + (31 - H2_StartIDX), 0:common_cols
                    ] = np.concatenate(
                        (
                            dataOut_hrrrh[HRRR_StartIDX:, 0:common_cols],
                            dataOut_h2[H2_StartIDX:, 0:common_cols],
                        ),
                        axis=0,
                    )

            # NBM
            if "nbm" in sourceList:
                NBM_StartIDX = nearest_index(dataOut_nbm[:, 0], baseDayUTC_Grib)

                if NBM_StartIDX < 1:
                    if "nbm" in sourceList:
                        sourceList.remove("nbm")
                    if "nbm" in sourceTimes:
                        sourceTimes.pop("nbm", None)
                    logger.error("NBM data not available for the requested time range.")
                else:
                    NBM_Merged = np.full((numHours, dataOut_nbm.shape[1]), np.nan)
                    NBM_Merged[0 : (242 - NBM_StartIDX), :] = dataOut_nbm[
                        NBM_StartIDX : (numHours + NBM_StartIDX), :
                    ]

            # NBM FIre
            if "nbm_fire" in sourceList:
                NBM_Fire_StartIDX = nearest_index(
                    dataOut_nbmFire[:, 0], baseDayUTC_Grib
                )

                if NBM_Fire_StartIDX < 1:
                    if "nbm_fire" in sourceList:
                        sourceList.remove("nbm_fire")
                    if "nbm_fire" in sourceTimes:
                        sourceTimes.pop("nbm_fire", None)

                    logger.error(
                        "NBM Fire data not available for the requested time range."
                    )
                else:
                    NBM_Fire_Merged = np.full(
                        (numHours, dataOut_nbmFire.shape[1]), np.nan
                    )

                    NBM_Fire_Merged[0 : (229 - NBM_Fire_StartIDX), :] = dataOut_nbmFire[
                        NBM_Fire_StartIDX : (numHours + NBM_Fire_StartIDX), :
                    ]

        except Exception:
            print("HRRR or NBM data not available, falling back to GFS")
            print(traceback.print_exc())
            if "hrrr_18-48" in sourceTimes:
                sourceTimes.pop("hrrr_18-48", None)
            if "nbm_fire" in sourceTimes:
                sourceTimes.pop("nbm_fire", None)
            if "nbm" in sourceTimes:
                sourceTimes.pop("nbm", None)
            if "hrrr_0-18" in sourceTimes:
                sourceTimes.pop("hrrr_0-18", None)
            if "hrrr_subh" in sourceTimes:
                sourceTimes.pop("hrrr_subh", None)

            if "hrrrsubh" in sourceList:
                sourceList.remove("hrrrsubh")
            if "hrrr_0-18" in sourceList:
                sourceList.remove("hrrr_0-18")
            if "nbm" in sourceList:
                sourceList.remove("nbm")
            if "nbm_fire" in sourceList:
                sourceList.remove("nbm_fire")
            if "hrrr_18-48" in sourceList:
                sourceList.remove("hrrr_18-48")

        # GFS
        GFS_StartIDX = nearest_index(dataOut_gfs[:, 0], baseDayUTC_Grib)
        GFS_EndIDX = min((len(dataOut_gfs), (numHours + GFS_StartIDX)))
        GFS_Merged = np.full((numHours, max(GFS.values()) + 1), np.nan)
        GFS_Merged[0 : (GFS_EndIDX - GFS_StartIDX), 0 : dataOut_gfs.shape[1]] = (
            dataOut_gfs[GFS_StartIDX:GFS_EndIDX, 0 : dataOut_gfs.shape[1]]
        )

        # GEFS
        if "gefs" in sourceList:
            GEFS_StartIDX = nearest_index(dataOut_gefs[:, 0], baseDayUTC_Grib)
            GEFS_Merged = dataOut_gefs[GEFS_StartIDX : (numHours + GEFS_StartIDX), :]

    # Interpolate if Time Machine
    else:
        GFS_Merged = np.full((len(hour_array_grib), max(GFS.values()) + 1), np.nan)
        for i in range(0, len(dataOut_gfs[0, :])):
            GFS_Merged[:, i] = np.interp(
                hour_array_grib,
                dataOut_gfs[:, 0].squeeze(),
                dataOut_gfs[:, i],
                left=np.nan,
                right=np.nan,
            )

        if "gefs" in sourceList:
            GEFS_Merged = np.zeros((len(hour_array_grib), dataOut_gefs.shape[1]))
            for i in range(0, len(dataOut_gefs[0, :])):
                GEFS_Merged[:, i] = np.interp(
                    hour_array_grib,
                    dataOut_gefs[:, 0].squeeze(),
                    dataOut_gefs[:, i],
                    left=np.nan,
                    right=np.nan,
                )
        if "nbm" in sourceList:
            NBM_Merged = np.zeros((len(hour_array_grib), dataOut_nbm.shape[1]))
            for i in range(0, len(dataOut_nbm[0, :])):
                NBM_Merged[:, i] = np.interp(
                    hour_array_grib,
                    dataOut_nbm[:, 0].squeeze(),
                    dataOut_nbm[:, i],
                    left=np.nan,
                    right=np.nan,
                )
        if "hrrr" in sourceList:
            HRRR_Merged = np.zeros((len(hour_array_grib), dataOut_hrrrh.shape[1]))
            for i in range(0, len(dataOut_hrrrh[0, :])):
                HRRR_Merged[:, i] = np.interp(
                    hour_array_grib,
                    dataOut_hrrrh[:, 0].squeeze(),
                    dataOut_hrrrh[:, i],
                    left=np.nan,
                    right=np.nan,
                )

    # Timing Check
    if TIMING:
        print("Array start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    InterPhour[:, DATA_HOURLY["time"]] = hour_array_grib

    # Daily array, 12 to 12
    # Have to redo the localize because of dayligt saving time
    day_array_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_4am_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=4
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_6am_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=6
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    day_array_6pm_grib = np.array(
        [
            pytzTZ.localize(
                datetime.datetime(
                    year=baseTime.year, month=baseTime.month, day=baseTime.day, hour=18
                )
                + datetime.timedelta(days=i)
            )
            .astimezone(utc)
            .timestamp()
            for i in range(10)
        ]
    ).astype(np.int32)

    # day_array_grib = (np.datetime64(day_array) - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))).astype(
    #    'timedelta64[s]').astype(np.int32)

    #    baseDay_6am_Local = datetime.datetime(year=baseTimeLocal.year, month=baseTimeLocal.month, day=baseTimeLocal.day,
    #                                          hour=6, minute=0, second=0)
    #    baseDayUTC_6am = baseDay_6am_Local - datetime.timedelta(minutes=tz_offset)
    #
    #    day_array_6am = np.arange(baseDayUTC_6am, baseDayUTC_6am + datetime.timedelta(days=9), datetime.timedelta(days=1))
    #    day_array_6am_grib = (day_array_6am - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))).astype(
    #        'timedelta64[s]').astype(np.int32)
    #
    #    baseDay_6pm_Local = datetime.datetime(year=baseTimeLocal.year, month=baseTimeLocal.month, day=baseTimeLocal.day,
    #                                          hour=18, minute=0, second=0)
    #    baseDayUTC_6pm = baseDay_6pm_Local - datetime.timedelta(minutes=tz_offset)
    #    day_array_6pm = np.arange(baseDayUTC_6pm, baseDayUTC_6pm + datetime.timedelta(days=9), datetime.timedelta(days=1))
    #    day_array_6pm_grib = (day_array_6pm - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))).astype(
    #        'timedelta64[s]').astype(np.int32)

    # Which hours map to which days
    hourlyDayIndex = np.full(len(hour_array_grib), int(MISSING_DATA))
    hourlyDay4amIndex = np.full(len(hour_array_grib), int(MISSING_DATA))
    hourlyHighIndex = np.full(len(hour_array_grib), int(MISSING_DATA))
    hourlyLowIndex = np.full(len(hour_array_grib), int(MISSING_DATA))

    # Zero to 9 to account for the four horus in day 8
    for d in range(0, 9):
        hourlyDayIndex[
            np.where(
                (hour_array_grib >= day_array_grib[d])
                & (hour_array_grib < day_array_grib[d + 1])
            )
        ] = d
        hourlyDay4amIndex[
            np.where(
                (hour_array_grib >= day_array_4am_grib[d])
                & (hour_array_grib < day_array_4am_grib[d + 1])
            )
        ] = d
        hourlyHighIndex[
            np.where(
                (hour_array_grib > day_array_6am_grib[d])
                & (hour_array_grib <= day_array_6pm_grib[d])
            )
        ] = d
        hourlyLowIndex[
            np.where(
                (hour_array_grib > day_array_6pm_grib[d])
                & (hour_array_grib <= day_array_6am_grib[d + 1])
            )
        ] = d

    if not timeMachine:
        hourlyDayIndex = hourlyDayIndex.astype(int)
        hourlyDay4amIndex = hourlyDay4amIndex.astype(int)
        hourlyHighIndex = hourlyHighIndex.astype(int)
        hourlyLowIndex = hourlyLowIndex.astype(int)
    else:
        # When running in timemachine mode, don't try to parse through different times, use the current 24h day for everything
        hourlyDayIndex = np.full(len(hour_array_grib), int(0))
        hourlyDay4amIndex = np.full(len(hour_array_grib), int(0))
        hourlyHighIndex = np.full(len(hour_array_grib), int(0))
        hourlyLowIndex = np.full(len(hour_array_grib), int(0))

    # +1 to account for the extra 4 hours of summary
    InterSday = np.zeros(shape=(daily_days + 1, 21))

    # Timing Check
    if TIMING:
        print("Sunrise start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    loc = LocationInfo("name", "region", tz_name, lat, az_Lon)

    # Calculate Sunrise, Sunset, Moon Phase
    for i in range(0, daily_days + 1):
        try:
            s = sun(
                loc.observer, date=baseDay + datetime.timedelta(days=i)
            )  # Use local to get the correct date

            InterSday[i, DATA_DAY["sunrise"]] = (
                (
                    np.datetime64(s["sunrise"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
            InterSday[i, DATA_DAY["sunset"]] = (
                (
                    np.datetime64(s["sunset"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )

            InterSday[i, DATA_DAY["dawn"]] = (
                (
                    np.datetime64(s["dawn"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
            InterSday[i, DATA_DAY["dusk"]] = (
                (
                    np.datetime64(s["dusk"].astimezone(utc).replace(tzinfo=None))
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )

        except ValueError:
            # If always sunny, (northern hemisphere during the summer) OR southern hemi during the winter
            if ((lat > 0) & (baseDay.month >= 4) & (baseDay.month <= 9)) or (
                (lat < 0) & (baseDay.month <= 3) | (baseDay.month >= 10)
            ):
                # Set sunrise to one second after midnight
                InterSday[i, DATA_DAY["sunrise"]] = day_array_grib[i] + np.timedelta64(
                    1, "s"
                ).astype("timedelta64[s]").astype(np.int32)
                # Set sunset to one second before midnight the following day
                InterSday[i, DATA_DAY["sunset"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                # Set sunrise to one second after midnight
                InterSday[i, DATA_DAY["dawn"]] = day_array_grib[i] + np.timedelta64(
                    1, "s"
                ).astype("timedelta64[s]").astype(np.int32)
                # Set sunset to one second before midnight the following day
                InterSday[i, DATA_DAY["dusk"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

            # Else
            else:
                # Set sunrise to two seconds before midnight
                InterSday[i, DATA_DAY["sunrise"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(2, "s").astype("timedelta64[s]").astype(np.int32)
                )
                # Set sunset to one seconds before midnight
                InterSday[i, DATA_DAY["sunset"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                InterSday[i, DATA_DAY["dawn"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(2, "s").astype("timedelta64[s]").astype(np.int32)
                )
                # Set sunset to one seconds before midnight
                InterSday[i, DATA_DAY["dusk"]] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

        m = moon.phase(baseDay + datetime.timedelta(days=i))
        InterSday[i, DATA_DAY["moon_phase"]] = m / 27.99

    # Timing Check
    if TIMING:
        print("Interpolation Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Interpolate for minutely
    # Concatenate HRRR and HRRR2
    if "gefs" in sourceList:
        gefsMinuteInterpolation = np.zeros(
            (len(minute_array_grib), len(dataOut_gefs[0, :]))
        )

    gfsMinuteInterpolation = np.zeros((len(minute_array_grib), len(dataOut_gfs[0, :])))

    nbmMinuteInterpolation = np.zeros((len(minute_array_grib), 18))

    if "hrrrsubh" in sourceList:
        hrrrSubHInterpolation = np.zeros((len(minute_array_grib), len(dataOut[0, :])))
        for i in range(len(dataOut[0, :]) - 1):
            hrrrSubHInterpolation[:, i + 1] = np.interp(
                minute_array_grib,
                dataOut[:, 0].squeeze(),
                dataOut[:, i + 1],
                left=np.nan,
                right=np.nan,
            )

        # Check for nan, which means SubH is out of range, and fall back to regular HRRR
        if np.isnan(hrrrSubHInterpolation[1, 1]):
            hrrrSubHInterpolation[:, HRRR_SUBH["gust"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["gust"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["pressure"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["pressure"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["temp"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["temp"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["dew"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["dew"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["wind_u"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["wind_u"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["wind_v"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["wind_v"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["intensity"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["intensity"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["snow"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["snow"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["ice"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["ice"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["freezing_rain"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["freezing_rain"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["rain"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["rain"]],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, HRRR_SUBH["refc"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["refc"]],
                left=np.nan,
                right=np.nan,
            )

            # Visibility is at a weird index
            hrrrSubHInterpolation[:, HRRR_SUBH["vis"]] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, HRRR["vis"]],
                left=np.nan,
                right=np.nan,
            )
        if "gefs" in sourceList:
            gefsMinuteInterpolation[:, GEFS["error"]] = np.interp(
                minute_array_grib,
                dataOut_gefs[:, 0].squeeze(),
                dataOut_gefs[:, GEFS["error"]],
                left=np.nan,
                right=np.nan,
            )

    else:  # Use GEFS
        if "gefs" in sourceList:
            for i in range(len(dataOut_gefs[0, :]) - 1):
                gefsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_gefs[:, 0].squeeze(),
                    dataOut_gefs[:, i + 1],
                    left=np.nan,
                    right=np.nan,
                )

        else:  # GFS Fallback
            # This could be optimized by only interpolating the necessary columns
            for i in range(len(dataOut_gfs[0, :]) - 1):
                gfsMinuteInterpolation[:, i + 1] = np.interp(
                    minute_array_grib,
                    dataOut_gfs[:, 0].squeeze(),
                    dataOut_gfs[:, i + 1],
                    left=np.nan,
                    right=np.nan,
                )

    if "nbm" in sourceList:
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
                left=np.nan,
                right=np.nan,
            )

    # Timing Check
    if TIMING:
        print("Minutely Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    InterPminute[:, DATA_MINUTELY["time"]] = minute_array_grib

    # "precipProbability"
    # Use NBM where available
    if "nbm" in sourceList:
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            nbmMinuteInterpolation[:, NBM["prob"]] * 0.01
        )
    elif "gefs" in sourceList:
        InterPminute[:, DATA_MINUTELY["prob"]] = gefsMinuteInterpolation[
            :, GEFS["prob"]
        ]
    else:  # Missing (-999) fallback
        InterPminute[:, DATA_MINUTELY["prob"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    # Less than 5% set to 0
    InterPminute[
        InterPminute[:, DATA_MINUTELY["prob"]] < 0.05, DATA_MINUTELY["prob"]
    ] = 0

    # Precipitation Type
    # IF HRRR, use that, otherwise GEFS
    if "hrrrsubh" in sourceList:
        for i in [
            HRRR_SUBH["snow"],
            HRRR_SUBH["ice"],
            HRRR_SUBH["freezing_rain"],
            HRRR_SUBH["rain"],
        ]:
            InterTminute[:, i - 7] = hrrrSubHInterpolation[:, i]
    elif "nbm" in sourceList:
        # 14 = Rain (1,2), 15 = Freezing Rain/ Ice (3,4), 16 = Snow (5,6,7), 17 = Ice (8,9)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-201.shtml

        # Snow
        InterTminute[:, 1] = nbmMinuteInterpolation[:, NBM["snow"]]
        # Ice
        InterTminute[:, 2] = nbmMinuteInterpolation[:, NBM["ice"]]
        # Freezing Rain
        InterTminute[:, 3] = nbmMinuteInterpolation[:, NBM["freezing_rain"]]
        # Rain
        InterTminute[:, 4] = nbmMinuteInterpolation[:, NBM["rain"]]
    elif "gefs" in sourceList:
        for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
            InterTminute[:, i - 3] = gefsMinuteInterpolation[:, i]
    else:  # GFS Fallback
        for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
            InterTminute[:, i - 11] = gfsMinuteInterpolation[:, i]

    # If all nan, set pchance to -999, otherwise determine the predominant type
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

    if "hrrrsubh" in sourceList:
        # Get temperature and reflectivity arrays first.
        temp_arr = hrrrSubHInterpolation[:, HRRR_SUBH["temp"]]
        refc_arr = hrrrSubHInterpolation[:, HRRR_SUBH["refc"]]

        # Mask: only assign type if current type is "none" AND reflectivity shows precip
        mask = (precipTypes == "none") & (refc_arr > 0)

        # Assign rain, snow, sleet based on temperature thresholds
        precipTypes[mask] = np.where(
            temp_arr[mask] >= TEMP_THRESHOLD_RAIN_C,
            "rain",
            np.where(temp_arr[mask] <= TEMP_THRESHOLD_SNOW_C, "snow", "sleet"),
        )

        # Update lists and arrays
        minuteType = precipTypes.tolist()
        precipTypes = np.array(minuteType)

        # Now convert reflectivity to precipitation intensity using estimated types
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            dbz_to_rate(refc_arr, precipTypes) * prepIntensityUnit
        )
    elif "nbm" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            nbmMinuteInterpolation[:, NBM["accum"]] * prepIntensityUnit
        )
    elif "gefs" in sourceList:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            gefsMinuteInterpolation[:, GEFS["accum"]] * prepIntensityUnit
        )
    else:
        InterPminute[:, DATA_MINUTELY["intensity"]] = (
            dbz_to_rate(gfsMinuteInterpolation[:, GFS["refc"]], precipTypes)
            * prepIntensityUnit
        )

    if "hrrrsubh" not in sourceList:
        # Set intensity to zero if POP == 0
        InterPminute[
            InterPminute[:, DATA_MINUTELY["prob"]] == 0, DATA_MINUTELY["intensity"]
        ] = 0

    # "precipIntensityError"
    if "gefs" in sourceList:
        InterPminute[:, DATA_MINUTELY["error"]] = (
            gefsMinuteInterpolation[:, GEFS["error"]] * prepIntensityUnit
        )
    else:  # Missing
        InterPminute[:, DATA_MINUTELY["error"]] = (
            np.ones(len(minute_array_grib)) * MISSING_DATA
        )

    # Create list of icons based off of maxPchance
    minuteKeys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
    ]

    # Assign pfactors for rain and snow for intensity
    pFacMinute = np.zeros((len(minute_array_grib)))
    pFacMinute[
        (
            (maxPchance == PRECIP_IDX["rain"])
            | (maxPchance == PRECIP_IDX["ice"])
            | (maxPchance == PRECIP_IDX["sleet"])
        )
    ] = 1  # Rain, Ice
    # Note, this means that intensity is always in liquid water equivalent
    pFacMinute[(maxPchance == PRECIP_IDX["snow"])] = 1  # Snow

    if "hrrrsubh" in sourceList:
        # Sometimes reflectivity shows precipitation when the type is none which causes the intensity to suddenly drop to 0
        # Setting the pFacMinute for None type to 1 will prevent this issue
        # Is worth testing to see if this causes unintended side effects
        pFacMinute[(maxPchance == PRECIP_IDX["none"])] = 1  # None

    minuteTimes = InterPminute[:, DATA_MINUTELY["time"]]
    minuteIntensity = np.maximum(
        np.round(InterPminute[:, DATA_MINUTELY["intensity"]] * pFacMinute, 4), 0
    )
    minuteProbability = np.minimum(
        np.maximum(np.round(InterPminute[:, DATA_MINUTELY["prob"]], 2), 0), 1
    )
    minuteIntensityError = np.maximum(
        np.round(InterPminute[:, DATA_MINUTELY["error"]], 2), 0
    )

    # Convert nan to -999 for json
    minuteIntensity[np.isnan(minuteIntensity)] = MISSING_DATA
    minuteProbability[np.isnan(minuteProbability)] = MISSING_DATA
    minuteIntensityError[np.isnan(minuteIntensityError)] = MISSING_DATA

    minuteDict = [
        dict(
            zip(
                minuteKeys,
                [
                    int(minuteTimes[idx]),
                    float(minuteIntensity[idx]),
                    float(minuteProbability[idx]),
                    float(minuteIntensityError[idx]),
                    minuteType[idx],
                ],
            )
        )
        for idx in range(61)
    ]

    # Timing Check
    if TIMING:
        print("Hourly start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    ## Approach
    # Use NBM where available
    # Use GFS past the end of NBM
    # Use HRRRH/ HRRRH2 if requested (?)
    # Use HRRR for some other variables

    # Precipitation Type
    # NBM
    maxPchanceHour = np.full((len(hour_array_grib), 3), MISSING_DATA)

    if "nbm" in sourceList:
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        InterThour[:, 1] = NBM_Merged[:, NBM["snow"]]
        InterThour[:, 2] = NBM_Merged[:, NBM["ice"]]
        InterThour[:, 3] = NBM_Merged[:, NBM["freezing_rain"]]
        InterThour[:, 4] = NBM_Merged[:, NBM["rain"]]

        # 14 = Rain (1,2), 15 = Freezing Rain/ Ice (3,4), 16 = Snow (5,6,7), 17 = Ice (8,9)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-201.shtml

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 0] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 0] = MISSING_DATA

    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterThour = np.zeros(shape=(len(hour_array), 5))
        InterThour[:, 1] = HRRR_Merged[:, HRRR["snow"]]
        InterThour[:, 2] = HRRR_Merged[:, HRRR["ice"]]
        InterThour[:, 3] = HRRR_Merged[:, HRRR["freezing_rain"]]
        InterThour[:, 4] = HRRR_Merged[:, HRRR["rain"]]

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0
        maxPchanceHour[:, 1] = np.argmax(InterThour, axis=1)
        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 1] = MISSING_DATA

    # GEFS
    if "gefs" in sourceList:
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        for i in [GEFS["snow"], GEFS["ice"], GEFS["freezing_rain"], GEFS["rain"]]:
            InterThour[:, i - 3] = GEFS_Merged[:, i]

        # 4 = Snow, 5 = Sleet, 6 = Freezing Rain, 7 = Rain

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 2] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 2] = MISSING_DATA
    else:  # GFS Fallback
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        for i in [GFS["snow"], GFS["ice"], GFS["freezing_rain"], GFS["rain"]]:
            InterThour[:, i - 11] = GFS_Merged[:, i]

        # 12 = Snow, 13 = Sleet, 14 = Freezing Rain, 15 = Rain

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 2] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 2] = MISSING_DATA

    # Intensity
    # NBM
    prcipIntensityHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        prcipIntensityHour[:, 0] = NBM_Merged[:, NBM["intensity"]]
    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        prcipIntensityHour[:, 1] = HRRR_Merged[:, HRRR["intensity"]] * 3600
    # GEFS
    if "gefs" in sourceList:
        prcipIntensityHour[:, 2] = GEFS_Merged[:, GEFS["accum"]]
    else:  # GFS Fallback
        prcipIntensityHour[:, 2] = GFS_Merged[:, GFS["intensity"]] * 3600

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["intensity"]] = (
        np.choose(np.argmin(np.isnan(prcipIntensityHour), axis=1), prcipIntensityHour.T)
        * prepIntensityUnit
    )

    # Set zero as the floor
    InterPhour[:, DATA_HOURLY["intensity"]] = np.maximum(
        InterPhour[:, DATA_HOURLY["intensity"]], 0
    )

    # Use the same type value as the intensity
    InterPhour[:, DATA_HOURLY["type"]] = np.choose(
        np.argmin(np.isnan(prcipIntensityHour), axis=1), maxPchanceHour.T
    )
    # Probability
    # NBM
    prcipProbabilityHour = np.full((len(hour_array_grib), 2), np.nan)
    if "nbm" in sourceList:
        prcipProbabilityHour[:, 0] = NBM_Merged[:, NBM["prob"]] * 0.01
    # GEFS
    if "gefs" in sourceList:
        prcipProbabilityHour[:, 1] = GEFS_Merged[:, GEFS["prob"]]

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["prob"]] = np.choose(
        np.argmin(np.isnan(prcipProbabilityHour), axis=1), prcipProbabilityHour.T
    )
    # Cap at 1
    InterPhour[:, DATA_HOURLY["prob"]] = clipLog(
        InterPhour[:, DATA_HOURLY["prob"]],
        CLIP_PROB["min"],
        CLIP_PROB["max"],
        "Probability Hour",
    )

    # Less than 5% set to 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] < 0.05, DATA_HOURLY["prob"]] = 0

    # Set intensity to zero if POP == 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] == 0, 2] = 0

    # Intensity Error
    # GEFS
    if "gefs" in sourceList:
        InterPhour[:, DATA_HOURLY["error"]] = np.maximum(
            GEFS_Merged[:, GEFS["error"]] * prepIntensityUnit, 0
        )

    ### Temperature
    TemperatureHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        TemperatureHour[:, 0] = NBM_Merged[:, NBM["temp"]]

    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        TemperatureHour[:, 1] = HRRR_Merged[:, HRRR["temp"]]

    if "gfs" in sourceList:
        TemperatureHour[:, 2] = GFS_Merged[:, GFS["temp"]]

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["temp"]] = np.choose(
        np.argmin(np.isnan(TemperatureHour), axis=1), TemperatureHour.T
    )

    # Clip between -90 and 60
    InterPhour[:, DATA_HOURLY["temp"]] = clipLog(
        InterPhour[:, DATA_HOURLY["temp"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Temperature Hour",
    )

    ### Dew Point
    DewPointHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        DewPointHour[:, 0] = NBM_Merged[:, NBM["dew"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        DewPointHour[:, 1] = HRRR_Merged[:, HRRR["dew"]]
    if "gfs" in sourceList:
        DewPointHour[:, 2] = GFS_Merged[:, GFS["dew"]]
    InterPhour[:, DATA_HOURLY["dew"]] = np.choose(
        np.argmin(np.isnan(DewPointHour), axis=1), DewPointHour.T
    )

    # Clip between -90 and 60 C
    InterPhour[:, DATA_HOURLY["dew"]] = clipLog(
        InterPhour[:, DATA_HOURLY["dew"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Dew Point Hour",
    )

    ### Humidity
    HumidityHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        HumidityHour[:, 0] = NBM_Merged[:, NBM["humidity"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        HumidityHour[:, 1] = HRRR_Merged[:, HRRR["humidity"]]
    if "gfs" in sourceList:
        HumidityHour[:, 2] = GFS_Merged[:, GFS["humidity"]]
    InterPhour[:, DATA_HOURLY["humidity"]] = (
        np.choose(np.argmin(np.isnan(HumidityHour), axis=1), HumidityHour.T) * humidUnit
    )

    # Clip between 0 and 1
    InterPhour[:, DATA_HOURLY["humidity"]] = clipLog(
        InterPhour[:, DATA_HOURLY["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
        "Humidity Hour",
    )

    ### Pressure
    PressureHour = np.full((len(hour_array_grib), 2), np.nan)
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        PressureHour[:, 0] = HRRR_Merged[:, HRRR["pressure"]]
    if "gfs" in sourceList:
        PressureHour[:, 1] = GFS_Merged[:, GFS["pressure"]]
    InterPhour[:, DATA_HOURLY["pressure"]] = np.choose(
        np.argmin(np.isnan(PressureHour), axis=1), PressureHour.T
    )

    # Clip between 800 and 1100
    InterPhour[:, DATA_HOURLY["pressure"]] = (
        clipLog(
            InterPhour[:, DATA_HOURLY["pressure"]],
            CLIP_PRESSURE["min"],
            CLIP_PRESSURE["max"],
            "Pressure Hour",
        )
        * pressUnits
    )

    ### Wind Speed
    WindSpeedHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        WindSpeedHour[:, 0] = NBM_Merged[:, NBM["wind"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindSpeedHour[:, 1] = np.sqrt(
            HRRR_Merged[:, HRRR["wind_u"]] ** 2 + HRRR_Merged[:, HRRR["wind_v"]] ** 2
        )
    if "gfs" in sourceList:
        WindSpeedHour[:, 2] = np.sqrt(
            GFS_Merged[:, GFS["wind_u"]] ** 2 + GFS_Merged[:, GFS["wind_v"]] ** 2
        )

    InterPhour[:, DATA_HOURLY["wind"]] = np.choose(
        np.argmin(np.isnan(WindSpeedHour), axis=1), WindSpeedHour.T
    )

    # Clip between 0 and 400
    InterPhour[:, DATA_HOURLY["wind"]] = (
        clipLog(
            InterPhour[:, DATA_HOURLY["wind"]],
            CLIP_WIND["min"],
            CLIP_WIND["max"],
            "Wind Speed",
        )
        * windUnit
    )

    ### Wind Gust
    WindGustHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        WindGustHour[:, 0] = NBM_Merged[:, NBM["gust"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindGustHour[:, 1] = HRRR_Merged[:, HRRR["gust"]]
    if "gfs" in sourceList:
        WindGustHour[:, 2] = GFS_Merged[:, GFS["gust"]]
    InterPhour[:, DATA_HOURLY["gust"]] = np.choose(
        np.argmin(np.isnan(WindGustHour), axis=1), WindGustHour.T
    )
    # Clip between 0 and 400
    InterPhour[:, DATA_HOURLY["gust"]] = (
        clipLog(
            InterPhour[:, DATA_HOURLY["gust"]],
            CLIP_WIND["min"],
            CLIP_WIND["max"],
            "Wind Gust Hour",
        )
        * windUnit
    )

    ### Wind Bearing
    WindBearingHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        WindBearingHour[:, 0] = NBM_Merged[:, NBM["bearing"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindBearingHour[:, 1] = np.rad2deg(
            np.mod(
                np.arctan2(
                    HRRR_Merged[:, HRRR["wind_u"]], HRRR_Merged[:, HRRR["wind_v"]]
                )
                + np.pi,
                2 * np.pi,
            )
        )
    if "gfs" in sourceList:
        WindBearingHour[:, 2] = np.rad2deg(
            np.mod(
                np.arctan2(GFS_Merged[:, GFS["wind_u"]], GFS_Merged[:, GFS["wind_v"]])
                + np.pi,
                2 * np.pi,
            )
        )
    InterPhour[:, DATA_HOURLY["bearing"]] = np.mod(
        np.choose(np.argmin(np.isnan(WindBearingHour), axis=1), WindBearingHour.T), 360
    )

    ### Cloud Cover
    CloudCoverHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        CloudCoverHour[:, 0] = NBM_Merged[:, NBM["cloud"]]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        CloudCoverHour[:, 1] = HRRR_Merged[:, HRRR["cloud"]]
    if "gfs" in sourceList:
        CloudCoverHour[:, 2] = GFS_Merged[:, GFS["cloud"]]
    InterPhour[:, DATA_HOURLY["cloud"]] = np.maximum(
        np.choose(np.argmin(np.isnan(CloudCoverHour), axis=1), CloudCoverHour.T) * 0.01,
        0,
    )
    # Clip between 0 and 1
    InterPhour[:, DATA_HOURLY["cloud"]] = clipLog(
        InterPhour[:, DATA_HOURLY["cloud"]],
        CLIP_CLOUD["min"],
        CLIP_CLOUD["max"],
        "Cloud Cover Hour",
    )

    ### UV Index
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["uv"]] = clipLog(
            GFS_Merged[:, GFS["uv"]] * 18.9 * 0.025,
            CLIP_UV["min"],
            CLIP_UV["max"],
            "UV Hour",
        )

        # Fix small negative zero
        # InterPhour[InterPhour[:, 14]<0, 14] = 0

    ### Visibility
    VisibilityHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        VisibilityHour[:, 0] = NBM_Merged[:, NBM["vis"]]

        # Filter out missing visibility values
        VisibilityHour[VisibilityHour[:, 0] < -1, 0] = np.nan
        VisibilityHour[VisibilityHour[:, 0] > 1e6, 0] = np.nan
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        VisibilityHour[:, 1] = HRRR_Merged[:, HRRR["vis"]]
    if "gfs" in sourceList:
        VisibilityHour[:, 2] = GFS_Merged[:, GFS["vis"]]

    InterPhour[:, DATA_HOURLY["vis"]] = (
        np.clip(
            np.choose(np.argmin(np.isnan(VisibilityHour), axis=1), VisibilityHour.T),
            CLIP_VIS["min"],
            CLIP_VIS["max"],
        )
        * visUnits
    )

    ### Ozone Index
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["ozone"]] = clipLog(
            GFS_Merged[:, GFS["ozone"]],
            CLIP_OZONE["min"],
            CLIP_OZONE["max"],
            "Ozone Hour",
        )

    ### Precipitation Accumulation
    PrecpAccumHour = np.full((len(hour_array_grib), 4), np.nan)
    # NBM
    if "nbm" in sourceList:
        PrecpAccumHour[:, 0] = NBM_Merged[:, NBM["intensity"]]
    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        PrecpAccumHour[:, 1] = HRRR_Merged[:, HRRR["accum"]]
    # GEFS
    if "gefs" in sourceList:
        PrecpAccumHour[:, 2] = GEFS_Merged[:, GEFS["accum"]]
    # GFS
    if "gfs" in sourceList:
        PrecpAccumHour[:, 3] = GFS_Merged[:, GFS["accum"]]

    InterPhour[:, DATA_HOURLY["accum"]] = np.maximum(
        np.choose(np.argmin(np.isnan(PrecpAccumHour), axis=1), PrecpAccumHour.T)
        * prepAccumUnit,
        0,
    )

    # Set accumulation to zero if POP == 0
    InterPhour[InterPhour[:, DATA_HOURLY["prob"]] == 0, DATA_HOURLY["accum"]] = 0

    ### Near Storm Distance
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["storm_dist"]] = np.maximum(
            GFS_Merged[:, GFS["storm_dist"]] * visUnits, 0
        )

    ### Near Storm Direction
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["storm_dir"]] = GFS_Merged[:, GFS["storm_dir"]]

    # Air quality/ smoke
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPhour[:, DATA_HOURLY["smoke"]] = clipLog(
            HRRR_Merged[:, HRRR["smoke"]],
            CLIP_SMOKE["min"],
            CLIP_SMOKE["max"],
            "Air quality Hour",
        )  # Maximum US AQI value for PM2.5 (smoke) is 500 which corresponds to 500 PM2.5
    else:
        InterPhour[:, DATA_HOURLY["smoke"]] = MISSING_DATA

    # Fire Index
    if "nbm_fire" in sourceList:
        InterPhour[:, DATA_HOURLY["fire"]] = clipLog(
            NBM_Fire_Merged[:, NBM_FIRE_INDEX],
            CLIP_FIRE["min"],
            CLIP_FIRE["max"],
            "Fire Hour",
        )

    # Convert wind speed from its display unit to m/s for the apparent temperature
    windSpeedMps = InterPhour[:, DATA_HOURLY["wind"]] / windUnit

    # Calculate the apparent temperature
    InterPhour[:, DATA_HOURLY["apparent"]] = calculate_apparent_temperature(
        InterPhour[:, DATA_HOURLY["temp"]],  # Air temperature in Kelvin
        InterPhour[:, DATA_HOURLY["humidity"]],  # Relative humidity (0.0 to 1.0)
        windSpeedMps,  # Wind speed in meters per second
    )

    ### Feels Like Temperature
    AppTemperatureHour = np.full((len(hour_array_grib), 2), np.nan)
    if "nbm" in sourceList:
        AppTemperatureHour[:, 0] = NBM_Merged[:, NBM["apparent"]]

    if "gfs" in sourceList:
        AppTemperatureHour[:, 1] = GFS_Merged[:, GFS["apparent"]]

    # Take first non-NaN value
    InterPhour[:, DATA_HOURLY["feels_like"]] = np.choose(
        np.argmin(np.isnan(AppTemperatureHour), axis=1), AppTemperatureHour.T
    )

    # Clip between -90 and 60
    InterPhour[:, DATA_HOURLY["feels_like"]] = clipLog(
        InterPhour[:, DATA_HOURLY["feels_like"]],
        CLIP_FEELS_LIKE["min"],
        CLIP_FEELS_LIKE["max"],
        "Feels Like Hour",
    )

    # Station Pressure
    if "gfs" in sourceList:
        InterPhour[:, DATA_HOURLY["station_pressure"]] = (
            clipLog(
                GFS_Merged[:, GFS["station_pressure"]],
                CLIP_PRESSURE["min"],
                CLIP_PRESSURE["max"],
                "Station Pressure Hour",
            )
            * pressUnits
        )

    # Set temperature units
    if tempUnits == 0:
        InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["humidity"]] = (
            InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["humidity"]]
            - KELVIN_TO_CELSIUS
        ) * 9 / 5 + 32
        InterPhour[:, DATA_HOURLY["feels_like"]] = (
            InterPhour[:, DATA_HOURLY["feels_like"]] - KELVIN_TO_CELSIUS
        ) * 9 / 5 + 32
    else:
        InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["humidity"]] = (
            InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["humidity"]] - tempUnits
        )
        InterPhour[:, DATA_HOURLY["feels_like"]] = (
            InterPhour[:, DATA_HOURLY["feels_like"]] - tempUnits
        )

    # Add a global check for weird values, since nothing should ever be greater than 10000
    # Keep time col
    InterPhourData = InterPhour[:, DATA_HOURLY["type"] :]
    InterPhourData[InterPhourData > CLIP_GLOBAL["max"]] = np.nan
    InterPhourData[InterPhourData < CLIP_GLOBAL["min"]] = np.nan
    InterPhour[:, 1:] = InterPhourData

    hourList = []
    hourIconList = []
    hourTextList = []

    # Find snow and liqiud precip
    # Set to zero as baseline
    InterPhour[:, DATA_HOURLY["rain"]] = 0
    InterPhour[:, DATA_HOURLY["snow"]] = 0
    InterPhour[:, DATA_HOURLY["ice"]] = 0

    # Accumulations in liquid equivalent
    InterPhour[InterPhour[:, DATA_HOURLY["type"]] == 4, DATA_HOURLY["rain"]] = (
        InterPhour[InterPhour[:, DATA_HOURLY["type"]] == 4, DATA_HOURLY["accum"]]
    )  # rain

    # Use the new snow height estimation for snow accumulation.
    snow_indices = np.where(InterPhour[:, DATA_HOURLY["type"]] == 1)[0]
    if snow_indices.size > 0:
        # Extract and convert data for all snow events in a vectorized way
        liquid_mm = InterPhour[snow_indices, DATA_HOURLY["accum"]] / prepAccumUnit
        if tempUnits == 0:  # Fahrenheit
            temp_c = (InterPhour[snow_indices, DATA_HOURLY["temp"]] - 32) * 5 / 9
        else:
            temp_c = InterPhour[snow_indices, DATA_HOURLY["temp"]]
        wind_mps = InterPhour[snow_indices, DATA_HOURLY["wind"]] / windUnit
        # Calculate snow height for all snow indices in a vectorized operation.
        snow_mm_values = estimate_snow_height(liquid_mm, temp_c, wind_mps)
        # Convert output to requested units and assign back to the main array
        InterPhour[snow_indices, DATA_HOURLY["snow"]] = snow_mm_values * prepAccumUnit

    InterPhour[
        (
            (InterPhour[:, DATA_HOURLY["type"]] == 2)
            | (InterPhour[:, DATA_HOURLY["type"]] == 3)
        ),
        DATA_HOURLY["ice"],
    ] = (
        InterPhour[
            (
                (InterPhour[:, DATA_HOURLY["type"]] == 2)
                | (InterPhour[:, DATA_HOURLY["type"]] == 3)
            ),
            DATA_HOURLY["accum"],
        ]
        * 1
    )  # Ice

    # Rain
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepRain = InterPhour[:, DATA_HOURLY["rain"]].copy()
    # Everything that isn't the current day
    dayZeroPrepRain[hourlyDayIndex != 0] = 0
    # Everything after the request time
    if not (timeMachine or timeMachineNear):
        dayZeroPrepRain[int(baseTimeOffset) :] = 0

    # Snow
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepSnow = InterPhour[:, DATA_HOURLY["snow"]].copy()
    # Everything that isn't the current day
    dayZeroPrepSnow[hourlyDayIndex != 0] = 0
    # Everything after the request time
    if not (timeMachine or timeMachineNear):
        dayZeroPrepSnow[int(baseTimeOffset) :] = 0

    # Sleet
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepSleet = InterPhour[:, DATA_HOURLY["ice"]].copy()
    # Everything that isn't the current day
    dayZeroPrepSleet[hourlyDayIndex != 0] = 0
    # Everything after the request time
    if not (timeMachine or timeMachineNear):
        dayZeroPrepSleet[int(baseTimeOffset) :] = 0

    # Accumulations in liquid equivalent
    dayZeroRain = dayZeroPrepRain.sum().round(4)  # rain
    dayZeroSnow = dayZeroPrepSnow.sum().round(4)  # Snow
    dayZeroIce = dayZeroPrepSleet.sum().round(4)  # Ice

    # Zero prep intensity and accum before forecast time
    InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["intensity"]] = 0
    InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["accum"]] = 0
    InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["rain"]] = 0
    InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["snow"]] = 0
    InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["ice"]] = 0

    # Zero prep prob before forecast time
    InterPhour[0 : int(baseTimeOffset), DATA_HOURLY["prob"]] = 0

    # Assign pfactors for rain and snow for intensity
    pFacHour = np.zeros((len(hour_array)))
    pFacHour[
        (
            (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["rain"])
            | (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["ice"])
            | (InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["sleet"])
        )
    ] = 1  # Rain, Ice
    # NOTE, this means that intensity is always liquid water equivalent.
    pFacHour[(InterPhour[:, DATA_HOURLY["type"]] == PRECIP_IDX["snow"])] = 1  # Snow

    InterPhour[:, DATA_HOURLY["intensity"]] = (
        InterPhour[:, DATA_HOURLY["intensity"]] * pFacHour
    )

    # pTypeMap = {0: 'none', 1: 'snow', 2: 'sleet', 3: 'sleet', 4: 'rain'}
    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])
    PTypeHour = pTypeMap[InterPhour[:, 1].astype(int)]
    PTextHour = pTextMap[InterPhour[:, 1].astype(int)]

    # Round all to 2 except precipitations
    InterPhour[:, DATA_HOURLY["prob"]] = InterPhour[:, DATA_HOURLY["prob"]].round(2)
    InterPhour[:, DATA_HOURLY["temp"] : DATA_HOURLY["accum"]] = InterPhour[
        :, DATA_HOURLY["temp"] : DATA_HOURLY["accum"]
    ].round(2)
    InterPhour[:, DATA_HOURLY["storm_dist"] : DATA_HOURLY["rain"]] = InterPhour[
        :, DATA_HOURLY["storm_dist"] : DATA_HOURLY["rain"]
    ].round(2)
    InterPhour[:, DATA_HOURLY["fire"]] = InterPhour[:, DATA_HOURLY["fire"]].round(2)
    InterPhour[:, DATA_HOURLY["station_pressure"]] = InterPhour[
        :, DATA_HOURLY["station_pressure"]
    ].round(2)

    # Round to 4
    InterPhour[:, DATA_HOURLY["type"] : DATA_HOURLY["prob"]] = InterPhour[
        :, DATA_HOURLY["type"] : DATA_HOURLY["prob"]
    ].round(4)
    InterPhour[:, DATA_HOURLY["error"] : DATA_HOURLY["temp"]] = InterPhour[
        :, DATA_HOURLY["error"] : DATA_HOURLY["temp"]
    ].round(4)
    InterPhour[:, DATA_HOURLY["accum"]] = InterPhour[:, DATA_HOURLY["accum"]].round(4)
    InterPhour[:, DATA_HOURLY["rain"] : DATA_HOURLY["fire"]] = InterPhour[
        :, DATA_HOURLY["rain"] : DATA_HOURLY["fire"]
    ].round(4)

    # Fix very small neg from interp to solve -0
    InterPhour[((InterPhour > -0.001) & (InterPhour < 0.001))] = 0

    # Replace NaN with -999 for json
    InterPhour[np.isnan(InterPhour)] = MISSING_DATA

    # Timing Check
    if TIMING:
        print("Hourly Loop start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # for idx in range(int(baseTimeOffset), hourly_hours + int(baseTimeOffset)):
    # For day 0 summary, need to calculate hourly data from midnight local
    for idx in range(0, numHours):
        # Check if day or night
        if hour_array_grib[idx] < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]:
            isDay = False
        elif (
            hour_array_grib[idx] >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            and hour_array_grib[idx]
            <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
        ):
            isDay = True
        elif hour_array_grib[idx] > InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]:
            isDay = False

        # Set text
        if InterPhour[idx, DATA_HOURLY["prob"]] >= PRECIP_PROB_THRESHOLD and (
            (
                (
                    InterPhour[idx, DATA_HOURLY["rain"]]
                    + InterPhour[idx, DATA_HOURLY["ice"]]
                )
                > (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit)
            )
            or (
                InterPhour[idx, DATA_HOURLY["snow"]]
                > (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit)
            )
        ):
            # If more than 30% chance of precip at any point throughout the day, then the icon for whatever is happening
            # Thresholds set in mm
            hourIcon = PTypeHour[idx]
            hourText = PTextHour[idx]
        # If visibility <1000 and during the day
        # elif InterPhour[idx,14]<1000 and (hour_array_grib[idx]>InterPday[dCount,16] and hour_array_grib[idx]<InterPday[dCount,17]):
        elif InterPhour[idx, DATA_HOURLY["vis"]] < (FOG_THRESHOLD_METERS * visUnits):
            hourIcon = "fog"
            hourText = "Fog"
        # If wind is greater than 10 m/s
        elif InterPhour[idx, DATA_HOURLY["wind"]] > (
            WIND_THRESHOLDS["light"] * windUnit
        ):
            hourIcon = "wind"
            hourText = "Windy"
        elif InterPhour[idx, DATA_HOURLY["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
            hourIcon = "cloudy"
            hourText = "Cloudy"
        elif (
            InterPhour[idx, DATA_HOURLY["cloud"]]
            > CLOUD_COVER_THRESHOLDS["partly_cloudy"]
        ):
            hourText = "Partly Cloudy"

            if (
                hour_array_grib[idx]
                < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            ):
                # Before sunrise
                hourIcon = "partly-cloudy-night"
            elif (
                hour_array_grib[idx]
                >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
                and hour_array_grib[idx]
                <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunrise before sunset
                hourIcon = "partly-cloudy-day"
            elif (
                hour_array_grib[idx]
                > InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunset
                hourIcon = "partly-cloudy-night"
        else:
            hourText = "Clear"

            if (
                hour_array_grib[idx]
                < InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
            ):
                # Before sunrise
                hourIcon = "clear-night"
            elif (
                hour_array_grib[idx]
                >= InterSday[hourlyDayIndex[idx], DATA_DAY["sunrise"]]
                and hour_array_grib[idx]
                <= InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunrise before sunset
                hourIcon = "clear-day"
            elif (
                hour_array_grib[idx]
                > InterSday[hourlyDayIndex[idx], DATA_DAY["sunset"]]
            ):
                # After sunset
                hourIcon = "clear-night"

        hourItem = {
            "time": int(hour_array_grib[idx]),
            "summary": hourText,
            "icon": hourIcon,
            "precipIntensity": InterPhour[idx, DATA_HOURLY["intensity"]],
            "precipProbability": InterPhour[idx, DATA_HOURLY["prob"]],
            "precipIntensityError": InterPhour[idx, DATA_HOURLY["error"]],
            "precipAccumulation": InterPhour[idx, DATA_HOURLY["rain"]]
            + InterPhour[idx, DATA_HOURLY["snow"]]
            + InterPhour[idx, DATA_HOURLY["ice"]],
            "precipType": PTypeHour[idx],
            "temperature": InterPhour[idx, DATA_HOURLY["temp"]],
            "apparentTemperature": InterPhour[idx, DATA_HOURLY["apparent"]],
            "dewPoint": InterPhour[idx, DATA_HOURLY["dew"]],
            "humidity": InterPhour[idx, DATA_HOURLY["humidity"]],
            "pressure": InterPhour[idx, DATA_HOURLY["pressure"]],
            "windSpeed": InterPhour[idx, DATA_HOURLY["wind"]],
            "windGust": InterPhour[idx, DATA_HOURLY["gust"]],
            "windBearing": int(InterPhour[idx, DATA_HOURLY["bearing"]]),
            "cloudCover": InterPhour[idx, DATA_HOURLY["cloud"]],
            "uvIndex": InterPhour[idx, DATA_HOURLY["uv"]],
            "visibility": InterPhour[idx, DATA_HOURLY["vis"]],
            "ozone": InterPhour[idx, DATA_HOURLY["ozone"]],
            "smoke": InterPhour[idx, DATA_HOURLY["smoke"]],
            "liquidAccumulation": InterPhour[idx, DATA_HOURLY["rain"]],
            "snowAccumulation": InterPhour[idx, DATA_HOURLY["snow"]],
            "iceAccumulation": InterPhour[idx, DATA_HOURLY["ice"]],
            "nearestStormDistance": InterPhour[idx, DATA_HOURLY["storm_dist"]],
            "nearestStormBearing": int(InterPhour[idx, DATA_HOURLY["storm_dir"]]),
            "fireIndex": InterPhour[idx, DATA_HOURLY["fire"]],
            "feelsLike": InterPhour[idx, DATA_HOURLY["feels_like"]],
        }

        # Add station pressure if requested
        if "stationPressure" in extraVars:
            hourItem["stationPressure"] = InterPhour[
                idx, DATA_HOURLY["station_pressure"]
            ]

        try:
            hourText, hourIcon = calculate_text(
                hourItem,
                prepAccumUnit,
                visUnits,
                windUnit,
                tempUnits,
                isDay,
                InterPhour[idx, DATA_HOURLY["rain"]],
                InterPhour[idx, DATA_HOURLY["snow"]],
                InterPhour[idx, DATA_HOURLY["ice"]],
                "hour",
                InterPhour[idx, DATA_HOURLY["intensity"]],
                icon,
            )

            if summaryText:
                hourItem["summary"] = translation.translate(["title", hourText])
                hourItem["icon"] = hourIcon

        except Exception:
            print("HOURLY TEXT GEN ERROR:")
            print(traceback.print_exc())

        if version < 2:
            hourItem.pop("liquidAccumulation", None)
            hourItem.pop("snowAccumulation", None)
            hourItem.pop("iceAccumulation", None)
            hourItem.pop("nearestStormDistance", None)
            hourItem.pop("nearestStormBearing", None)
            hourItem.pop("fireIndex", None)
            hourItem.pop("feelsLike", None)

        if timeMachine and not tmExtra:
            hourItem.pop("uvIndex", None)
            hourItem.pop("ozone", None)

        hourList.append(hourItem)

        hourIconList.append(hourIcon)
        hourTextList.append(hourItem["summary"])

    # Daily calculations #################################################
    # Timing Check
    if TIMING:
        print("Daily start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    mean_results = []
    sum_results = []
    max_results = []
    min_results = []
    argmax_results = []
    argmin_results = []
    high_results = []
    low_results = []
    arghigh_results = []
    arglow_results = []
    mean_4am_results = []
    sum_4am_results = []
    max_4am_results = []
    maxPchanceDay = np.zeros((daily_days))

    # Pre-calculate masks for each group to avoid redundant computation
    masks = [hourlyDayIndex == day_index for day_index in range(daily_days)]
    for mask in masks:
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        mean_results.append(np.mean(filtered_data, axis=0))
        sum_results.append(np.sum(filtered_data, axis=0))
        max_results.append(np.max(filtered_data, axis=0))
        min_results.append(np.min(filtered_data, axis=0))
        maxTime = np.argmax(filtered_data, axis=0)
        minTime = np.argmin(filtered_data, axis=0)
        argmax_results.append(filtered_data[maxTime, 0])
        argmin_results.append(filtered_data[minTime, 0])

    # Icon/ summary parameters go from 4 am to 4 am
    masks = [hourlyDay4amIndex == day_index for day_index in range(daily_days)]
    for mIDX, mask in enumerate(masks):
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        mean_4am_results.append(np.mean(filtered_data, axis=0))
        sum_4am_results.append(np.sum(filtered_data, axis=0))
        max_4am_results.append(np.max(filtered_data, axis=0))

        dailyTypeCount = Counter(filtered_data[:, 1]).most_common(2)

        # Check if the most common type is zero, in that case return the second most common
        if dailyTypeCount[0][0] == 0:
            if len(dailyTypeCount) == 2:
                maxPchanceDay[mIDX] = dailyTypeCount[1][0]
            else:
                maxPchanceDay[mIDX] = dailyTypeCount[0][
                    0
                ]  # If all ptypes are none, then really shouldn't be any precipitation

        else:
            maxPchanceDay[mIDX] = dailyTypeCount[0][0]

    # Daily High
    masks = [hourlyHighIndex == day_index for day_index in range(daily_days)]

    for mask in masks:
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        high_results.append(np.max(filtered_data, axis=0))
        maxTime = np.argmax(filtered_data, axis=0)
        arghigh_results.append(filtered_data[maxTime, 0])

    # Daily Low
    masks = [hourlyLowIndex == day_index for day_index in range(daily_days)]

    for mask in masks:
        filtered_data = InterPhour[mask]

        # Calculate and store each statistic for the current group
        low_results.append(np.min(filtered_data, axis=0))
        minTime = np.argmin(filtered_data, axis=0)
        arglow_results.append(filtered_data[minTime, 0])

    # Convert lists to numpy arrays if necessary
    InterPday = np.array(mean_results)
    InterPdaySum = np.array(sum_results)
    InterPdayMax = np.array(max_results)
    InterPdayMin = np.array(min_results)
    InterPdayMaxTime = np.array(argmax_results)
    InterPdayMinTime = np.array(argmin_results)
    InterPdayHigh = np.array(high_results)
    InterPdayLow = np.array(low_results)
    InterPdayHighTime = np.array(arghigh_results)
    InterPdayLowTime = np.array(arglow_results)
    InterPday4am = np.array(mean_4am_results)
    InterPdaySum4am = np.array(sum_4am_results)
    InterPdayMax4am = np.array(max_4am_results)

    # Select the daily accum type:
    # Start with the most common type for the day as a baseline

    # The logic here is trying to guess what the most "useful" type of precipitation would be, while avoiding strange results
    # First, if there is a ton of rain, that should show up even if there's a lot of snow "hours"
    # Then, since snow is 10x rain, the rain icon shouldn't appear is there is much snow,
    # otherwise it looks like an unreasonable amount of rain. So snow greater than 1 cm takes priority over rain.
    # Finally, if there is much ice at all, that takes priority over rain or snow.

    # Improved logic: if all types are present, use sleet (3).
    all_types = (
        (InterPdaySum[:, DATA_DAY["rain"]] > 0)
        & (InterPdaySum[:, DATA_DAY["snow"]] > 0)
        & (InterPdaySum[:, DATA_DAY["ice"]] > 0)
    )
    maxPchanceDay[all_types] = 3

    # Otherwise, use the type with the most accumulation.
    # 21: rain, 22: snow, 23: ice
    precip_accum = np.stack(
        [
            InterPdaySum[:, DATA_DAY["rain"]],  # rain
            InterPdaySum[:, DATA_DAY["snow"]],  # snow
            InterPdaySum[:, DATA_DAY["ice"]],  # ice
        ],
        axis=1,
    )
    # 4: rain, 1: snow, 2: ice (map index to type)

    type_map = np.array([PRECIP_IDX["rain"], PRECIP_IDX["snow"], PRECIP_IDX["ice"]])
    dominant_type = type_map[np.argmax(precip_accum, axis=1)]

    # Only update where not all types are present.
    not_all_types = ~all_types
    has_precip = np.max(precip_accum, axis=1) > 0
    update_mask = not_all_types & has_precip
    maxPchanceDay[update_mask] = dominant_type[update_mask]

    # The following thresholds are applied after the dominant type (by volume) is determined.
    # They serve to highlight significant precipitation events, overriding the volume-based
    # determination if a certain threshold is met. The priority for these overrides is:
    # Ice > Snow > Rain.
    # If more than 10 mm of rain is forecast, then rain.
    maxPchanceDay[InterPdaySum[:, DATA_DAY["rain"]] > (10 * prepAccumUnit)] = (
        PRECIP_IDX["rain"]
    )

    # If more than 5 mm of snow is forecast, then snow.
    maxPchanceDay[InterPdaySum[:, DATA_DAY["snow"]] > (5 * prepAccumUnit)] = PRECIP_IDX[
        "snow"
    ]

    # Else, if more than 1 mm of ice is forecast, then ice.
    maxPchanceDay[InterPdaySum[:, DATA_DAY["ice"]] > (1 * prepAccumUnit)] = PRECIP_IDX[
        "ice"
    ]

    # Process Daily Data for ouput
    dayList = []
    dayIconList = []
    dayTextList = []

    maxPchanceDay = np.array(maxPchanceDay).astype(int)
    PTypeDay = pTypeMap[maxPchanceDay]
    PTextDay = pTextMap[maxPchanceDay]

    # Round
    # Round all to 2 except precipitations
    InterPday[:, 5:18] = InterPday[:, 5:18].round(2)
    InterPday[:, DATA_DAY["station_pressure"]] = InterPday[
        :, DATA_DAY["station_pressure"]
    ].round(2)

    InterPdayMax[:, DATA_DAY["prob"]] = InterPdayMax[:, DATA_DAY["prob"]].round(2)
    InterPdayMax[:, 5:18] = InterPdayMax[:, 5:18].round(2)
    InterPdayMax[:, DATA_DAY["fire"]] = InterPdayMax[:, DATA_DAY["fire"]].round(2)

    InterPdayMin[:, 5:18] = InterPdayMin[:, 5:18].round(2)
    InterPdaySum[:, 5:18] = InterPdaySum[:, 5:18].round(2)
    InterPdayHigh[:, 5:18] = InterPdayHigh[:, 5:18].round(2)
    InterPdayLow[:, 5:18] = InterPdayLow[:, 5:18].round(2)

    InterPday[:, 1:5] = InterPday[:, 1:5].round(4)
    InterPdaySum[:, 1:5] = InterPdaySum[:, 1:5].round(4)
    InterPdayMax[:, 1:3] = InterPdayMax[:, 1:3].round(4)
    InterPdayMax[:, 4:5] = InterPdayMax[:, 4:5].round(4)
    InterPdaySum[:, 21:24] = InterPdaySum[:, 21:24].round(4)
    InterPdayMax[:, 21:24] = InterPdayMax[:, 21:24].round(4)

    if TIMING:
        print("Daily Loop start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    for idx in range(0, daily_days):
        if InterPdayMax4am[idx, DATA_DAY["prob"]] > PRECIP_PROB_THRESHOLD and (
            (
                (
                    InterPdaySum4am[idx, DATA_DAY["rain"]]
                    + InterPdaySum4am[idx, DATA_DAY["ice"]]
                )
                > (DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit)
            )
            or (
                InterPdaySum4am[idx, DATA_DAY["snow"]]
                > (DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM * prepAccumUnit)
            )
        ):
            # If more than 30% chance of precip at any point throughout the day, and either more than 1 mm of rain or 5 mm of snow
            # Thresholds set in mm
            dayIcon = PTypeDay[idx]
            dayText = PTextDay[idx]

            # Fallback if no ptype for some reason. This should never occur though
            if dayIcon == "none":
                if tempUnits == 0:
                    tempThresh = TEMPERATURE_UNITS_THRESH["f"]
                else:
                    tempThresh = TEMPERATURE_UNITS_THRESH["c"]

                if InterPday[idx, DATA_DAY["temp"]] > tempThresh:
                    dayIcon = "rain"
                    dayText = "Rain"
                else:
                    dayIcon = "snow"
                    dayText = "Snow"

        elif InterPday4am[idx, DATA_DAY["vis"]] < (FOG_THRESHOLD_METERS * visUnits):
            dayIcon = "fog"
            dayText = "Fog"
        elif InterPday4am[idx, DATA_DAY["wind"]] > (
            WIND_THRESHOLDS["light"] * windUnit
        ):
            dayIcon = "wind"
            dayText = "Windy"
        elif InterPday4am[idx, DATA_DAY["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
            dayIcon = "cloudy"
            dayText = "Cloudy"
        elif (
            InterPday4am[idx, DATA_DAY["cloud"]]
            > CLOUD_COVER_THRESHOLDS["partly_cloudy"]
        ):
            dayIcon = "partly-cloudy-day"
            dayText = "Partly Cloudy"
        else:
            dayIcon = "clear-day"
            dayText = "Clear"

        # Temperature High is daytime high, so 6 am to 6 pm
        # First index is 6 am, then index 2
        # Nightime is index 1, 3, etc.
        dayObject = {
            "time": int(day_array_grib[idx]),
            "summary": dayText,
            "icon": dayIcon,
            "dawnTime": int(InterSday[idx, DATA_DAY["dawn"]]),
            "sunriseTime": int(InterSday[idx, DATA_DAY["sunrise"]]),
            "sunsetTime": int(InterSday[idx, DATA_DAY["sunset"]]),
            "duskTime": int(InterSday[idx, DATA_DAY["dusk"]]),
            "moonPhase": InterSday[idx, DATA_DAY["moon_phase"]].round(2),
            "precipIntensity": InterPday[idx, DATA_DAY["intensity"]],
            "precipIntensityMax": InterPdayMax[idx, DATA_DAY["intensity"]],
            "precipIntensityMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["intensity"]]),
            "precipProbability": InterPdayMax[idx, DATA_DAY["prob"]],
            "precipAccumulation": round(
                InterPdaySum[idx, DATA_DAY["rain"]]
                + InterPdaySum[idx, DATA_DAY["snow"]]
                + InterPdaySum[idx, DATA_DAY["ice"]],
                4,
            ),
            "precipType": PTypeDay[idx],
            "temperatureHigh": InterPdayHigh[idx, DATA_DAY["temp"]],
            "temperatureHighTime": int(InterPdayHighTime[idx, DATA_DAY["temp"]]),
            "temperatureLow": InterPdayLow[idx, DATA_DAY["temp"]],
            "temperatureLowTime": int(InterPdayLowTime[idx, DATA_DAY["temp"]]),
            "apparentTemperatureHigh": InterPdayHigh[idx, DATA_DAY["apparent"]],
            "apparentTemperatureHighTime": int(
                InterPdayHighTime[idx, DATA_DAY["apparent"]]
            ),
            "apparentTemperatureLow": InterPdayLow[idx, DATA_DAY["apparent"]],
            "apparentTemperatureLowTime": int(
                InterPdayLowTime[idx, DATA_DAY["apparent"]]
            ),
            "dewPoint": InterPday[idx, DATA_DAY["dew"]],
            "humidity": InterPday[idx, DATA_DAY["humidity"]],
            "pressure": InterPday[idx, DATA_DAY["pressure"]],
            "windSpeed": InterPday[idx, DATA_DAY["wind"]],
            "windGust": InterPday[idx, DATA_DAY["gust"]],
            "windGustTime": int(InterPdayMaxTime[idx, DATA_DAY["gust"]]),
            "windBearing": int(InterPday[idx, DATA_DAY["bearing"]]),
            "cloudCover": InterPday[idx, DATA_DAY["cloud"]],
            "uvIndex": InterPdayMax[idx, DATA_DAY["uv"]],
            "uvIndexTime": int(InterPdayMaxTime[idx, DATA_DAY["uv"]]),
            "visibility": InterPday[idx, DATA_DAY["vis"]],
            "temperatureMin": InterPdayMin[idx, DATA_DAY["temp"]],
            "temperatureMinTime": int(InterPdayMinTime[idx, DATA_DAY["temp"]]),
            "temperatureMax": InterPdayMax[idx, DATA_DAY["temp"]],
            "temperatureMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["temp"]]),
            "apparentTemperatureMin": InterPdayMin[idx, DATA_DAY["apparent"]],
            "apparentTemperatureMinTime": int(
                InterPdayMinTime[idx, DATA_DAY["apparent"]]
            ),
            "apparentTemperatureMax": InterPdayMax[idx, DATA_DAY["apparent"]],
            "apparentTemperatureMaxTime": int(
                InterPdayMaxTime[idx, DATA_DAY["apparent"]]
            ),
            "smokeMax": InterPdayMax[idx, DATA_DAY["smoke"]],
            "smokeMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["smoke"]]),
            "liquidAccumulation": InterPdaySum[idx, DATA_DAY["rain"]],
            "snowAccumulation": InterPdaySum[idx, DATA_DAY["snow"]],
            "iceAccumulation": InterPdaySum[idx, DATA_DAY["ice"]],
            "fireIndexMax": InterPdayMax[idx, DATA_DAY["fire"]],
            "fireIndexMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["fire"]]),
        }

        # Add station pressure if requested
        if "stationPressure" in extraVars:
            dayObject["stationPressure"] = InterPday[idx, DATA_DAY["station_pressure"]]

        try:
            if idx < 8:
                # Calculate the day summary from 4 to 4
                dayIcon, dayText = calculate_day_text(
                    hourList[((idx) * 24) + 4 : ((idx + 1) * 24) + 4],
                    prepAccumUnit,
                    visUnits,
                    windUnit,
                    tempUnits,
                    True,
                    str(tz_name),
                    int(time.time()),
                    "day",
                    icon,
                )

                # Translate the text
                if summaryText:
                    dayObject["summary"] = translation.translate(["sentence", dayText])
                    dayObject["icon"] = dayIcon
        except Exception:
            print("DAILY TEXT GEN ERROR:")
            print(traceback.print_exc())

        if version < 2:
            dayObject.pop("dawnTime", None)
            dayObject.pop("duskTime", None)
            dayObject.pop("smokeMax", None)
            dayObject.pop("smokeMaxTime", None)
            dayObject.pop("liquidAccumulation", None)
            dayObject.pop("snowAccumulation", None)
            dayObject.pop("iceAccumulation", None)
            dayObject.pop("fireIndexMax", None)
            dayObject.pop("fireIndexMaxTime", None)

        if timeMachine and not tmExtra:
            dayObject.pop("precipProbability", None)
            dayObject.pop("humidity", None)
            dayObject.pop("uvIndex", None)
            dayObject.pop("uvIndexTime", None)
            dayObject.pop("visibility", None)

        dayList.append(dayObject)

        dayTextList.append(dayObject["summary"])
        dayIconList.append(dayIcon)

    # Timing Check
    if TIMING:
        print("Alert Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    alertDict = []
    # If alerts are requested and in the US
    try:
        if (
            (not timeMachine)
            and (exAlerts == 0)
            and (az_Lon > -127)
            and (az_Lon < -65)
            and (lat > 24)
            and (lat < 50)
        ):
            # Read in NetCDF
            # Find NetCDF Point based on alerts grid
            alerts_lons = np.arange(-127, -65, 0.025)
            alerts_lats = np.arange(24, 50, 0.025)

            abslat = np.abs(alerts_lats - lat)
            abslon = np.abs(alerts_lons - az_Lon)
            alerts_y_p = np.argmin(abslat)
            alerts_x_p = np.argmin(abslon)

            alertList = []

            alertDat = NWS_Alerts_Zarr[alerts_y_p, alerts_x_p]

            if alertDat == "":
                alertList = []
            else:
                # Match if any alerts
                alerts = str(alertDat).split("|")
                # Loop through each alert
                for alert in alerts:
                    # Extract alert details
                    alertDetails = alert.split("}{")

                    alertOnset = datetime.datetime.strptime(
                        alertDetails[3], "%Y-%m-%dT%H:%M:%S%z"
                    ).astimezone(utc)
                    alertEnd = datetime.datetime.strptime(
                        alertDetails[4], "%Y-%m-%dT%H:%M:%S%z"
                    ).astimezone(utc)

                    # Format description newlines
                    alertDescript = alertDetails[1]
                    # Step 1: Replace double newlines with a single newline
                    formatted_text = re.sub(r"(?<!\n)\n(?!\n)", " ", alertDescript)

                    # Step 2: Replace remaining single newlines with a space
                    formatted_text = re.sub(r"\n\n", "\n", formatted_text)

                    alertDict = {
                        "title": alertDetails[0],
                        "regions": [s.lstrip() for s in alertDetails[2].split(";")],
                        "severity": alertDetails[5],
                        "time": int(
                            (
                                alertOnset
                                - datetime.datetime(1970, 1, 1, 0, 0, 0).astimezone(utc)
                            ).total_seconds()
                        ),
                        "expires": int(
                            (
                                alertEnd
                                - datetime.datetime(1970, 1, 1, 0, 0, 0).astimezone(utc)
                            ).total_seconds()
                        ),
                        "description": formatted_text,
                        "uri": alertDetails[6],
                    }

                    alertList.append(dict(alertDict))
        else:
            alertList = []

    except Exception:
        print("An Alert error occurred:")
        print(traceback.print_exc())

    # Timing Check
    if TIMING:
        print("Current Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    # Currently data, find points for linear averaging
    # Use GFS, since should also be there and the should cover all times... this could be an issue at some point

    # If within 2 minutes of a hour, do not using rounding
    if np.min(np.abs(GFS_Merged[:, 0] - minute_array_grib[0])) < 120:
        currentIDX_hrrrh = np.argmin(np.abs(GFS_Merged[:, 0] - minute_array_grib[0]))
        interpFac1 = 0
        interpFac2 = 1
    else:
        currentIDX_hrrrh = np.searchsorted(
            GFS_Merged[:, 0], minute_array_grib[0], side="left"
        )

        # Find weighting factors for hourly data
        # Weighting factors for linear interpolation
        interpFac1 = 1 - (
            abs(minute_array_grib[0] - GFS_Merged[currentIDX_hrrrh - 1, 0])
            / (GFS_Merged[currentIDX_hrrrh, 0] - GFS_Merged[currentIDX_hrrrh - 1, 0])
        )

        interpFac2 = 1 - (
            abs(minute_array_grib[0] - GFS_Merged[currentIDX_hrrrh, 0])
            / (GFS_Merged[currentIDX_hrrrh, 0] - GFS_Merged[currentIDX_hrrrh - 1, 0])
        )

    currentIDX_hrrrh_A = np.max((currentIDX_hrrrh - 1, 0))

    InterPcurrent = np.zeros(shape=22)  # Time, Intensity,Probability
    InterPcurrent[DATA_CURRENT["time"]] = int(minute_array_grib[0])

    # Get prep probability, intensity and error from minutely
    InterPcurrent[DATA_CURRENT["intensity"]] = InterPminute[
        0, DATA_MINUTELY["intensity"]
    ]
    InterPcurrent[DATA_CURRENT["prob"]] = InterPminute[
        0, DATA_MINUTELY["prob"]
    ]  # "precipProbability"
    InterPcurrent[DATA_CURRENT["error"]] = InterPminute[
        0, DATA_MINUTELY["error"]
    ]  # "precipIntensityError"

    # Temperature from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = hrrrSubHInterpolation[
            0, HRRR_SUBH["temp"]
        ]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["temp"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["temp"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["temp"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["temp"]] * interpFac2
        )

    # Clip between -90 and 60
    InterPcurrent[DATA_CURRENT["temp"]] = clipLog(
        InterPcurrent[DATA_CURRENT["temp"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Temperature Current",
    )

    # Dewpoint from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = hrrrSubHInterpolation[0, HRRR_SUBH["dew"]]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["dew"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["dew"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["dew"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["dew"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["dew"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["dew"]] * interpFac2
        )

        # Clip between -90 and 60
        InterPcurrent[DATA_CURRENT["dew"]] = clipLog(
            InterPcurrent[DATA_CURRENT["dew"]],
            CLIP_TEMP["min"],
            CLIP_TEMP["max"],
            "Dewpoint Current",
        )

    # humidity, NBM then HRRR, then GFS
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["humidity"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["humidity"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["humidity"]] * interpFac2
        ) * humidUnit
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["humidity"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["humidity"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["humidity"]] * interpFac2
        ) * humidUnit
    else:
        InterPcurrent[DATA_CURRENT["humidity"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["humidity"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["humidity"]] * interpFac2
        ) * humidUnit

    # Clip between 0 and 1
    InterPcurrent[DATA_CURRENT["humidity"]] = clipLog(
        InterPcurrent[DATA_CURRENT["humidity"]],
        CLIP_HUMIDITY["min"],
        CLIP_HUMIDITY["max"],
        "Humidity Current",
    )

    # Pressure from HRRR, then GFS
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["pressure"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["pressure"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["pressure"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["pressure"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["pressure"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["pressure"]] * interpFac2
        )

    # Clip between 800 and 1100
    InterPcurrent[DATA_CURRENT["pressure"]] = (
        clipLog(
            InterPcurrent[DATA_CURRENT["pressure"]],
            CLIP_PRESSURE["min"],
            CLIP_PRESSURE["max"],
            "Pressure Current",
        )
        * pressUnits
    )

    # WindSpeed from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            hrrrSubHInterpolation[0, HRRR_SUBH["wind_u"]] ** 2
            + hrrrSubHInterpolation[0, HRRR_SUBH["wind_v"]] ** 2
        )
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["wind"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["wind"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["wind"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["wind"]] = math.sqrt(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["wind_u"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["wind_u"]] * interpFac2
            )
            ** 2
            + (
                GFS_Merged[currentIDX_hrrrh_A, GFS["wind_v"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["wind_v"]] * interpFac2
            )
            ** 2
        )
    InterPcurrent[DATA_CURRENT["wind"]] = (
        clipLog(
            InterPcurrent[DATA_CURRENT["wind"]],
            CLIP_WIND["min"],
            CLIP_WIND["max"],
            "WindSpeed Current",
        )
        * windUnit
    )

    # Gust from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = hrrrSubHInterpolation[
            0, HRRR_SUBH["gust"]
        ]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["gust"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["gust"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["gust"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["gust"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["gust"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["gust"]] * interpFac2
        )

    # Clip between 0 and 400
    InterPcurrent[DATA_CURRENT["gust"]] = (
        clipLog(
            InterPcurrent[DATA_CURRENT["gust"]],
            CLIP_WIND["min"],
            CLIP_WIND["max"],
            "Gust Current",
        )
        * windUnit
    )

    # WindDir from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_u"]],
                    hrrrSubHInterpolation[0, HRRR_SUBH["wind_v"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["bearing"]] = NBM_Merged[
            currentIDX_hrrrh, NBM["bearing"]
        ]
    else:
        InterPcurrent[DATA_CURRENT["bearing"]] = np.rad2deg(
            np.mod(
                np.arctan2(
                    GFS_Merged[currentIDX_hrrrh, GFS["wind_u"]],
                    GFS_Merged[currentIDX_hrrrh, GFS["wind_v"]],
                )
                + np.pi,
                2 * np.pi,
            )
        )

    # Cloud, NBM then HRRR, then GFS
    if "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["cloud"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["cloud"]] * interpFac2
        ) * 0.01
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["cloud"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["cloud"]] * interpFac2
        ) * 0.01
    else:
        InterPcurrent[DATA_CURRENT["cloud"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["cloud"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["cloud"]] * interpFac2
        ) * 0.01

    # Clip
    InterPcurrent[DATA_CURRENT["cloud"]] = clipLog(
        InterPcurrent[DATA_CURRENT["cloud"]],
        CLIP_CLOUD["min"],
        CLIP_CLOUD["max"],
        "Cloud Current",
    )

    # UV Index from GFS
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

    # Station Pressure from GFS
    InterPcurrent[DATA_CURRENT["station_pressure"]] = (
        clipLog(
            (
                GFS_Merged[currentIDX_hrrrh_A, GFS["station_pressure"]] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, GFS["station_pressure"]] * interpFac2
            ),
            CLIP_PRESSURE["min"],
            CLIP_PRESSURE["max"],
            "Station Pressure Current",
        )
        * pressUnits
    )

    # VIS, SubH, NBM then HRRR, then GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = hrrrSubHInterpolation[0, HRRR_SUBH["vis"]]
    elif "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["vis"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["vis"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["vis"]] * interpFac2
        )
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[DATA_CURRENT["vis"]] = (
            HRRR_Merged[currentIDX_hrrrh_A, HRRR["vis"]] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, HRRR["vis"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["vis"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["vis"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["vis"]] * interpFac2
        )

    InterPcurrent[DATA_CURRENT["vis"]] = np.clip(InterPcurrent[14], 0, 16090) * visUnits

    # Ozone from GFS
    InterPcurrent[DATA_CURRENT["ozone"]] = clipLog(
        GFS_Merged[currentIDX_hrrrh_A, GFS["ozone"]] * interpFac1
        + GFS_Merged[currentIDX_hrrrh, GFS["ozone"]] * interpFac2,
        CLIP_OZONE["min"],
        CLIP_OZONE["max"],
        "Ozone Current",
    )

    # Storm Distance from GFS
    InterPcurrent[DATA_CURRENT["storm_dist"]] = np.maximum(
        (
            GFS_Merged[currentIDX_hrrrh_A, GFS["storm_dist"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["storm_dist"]] * interpFac2
        )
        * visUnits,
        0,
    )

    # Storm Bearing from GFS
    InterPcurrent[DATA_CURRENT["storm_dir"]] = GFS_Merged[
        currentIDX_hrrrh, GFS["storm_dir"]
    ]

    # Smoke from HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
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

    # Convert wind speed from its display unit to m/s for the apparent temperature function
    currentWindSpeedMps = InterPcurrent[9] / windUnit

    # Calculate the apparent temperature
    InterPcurrent[DATA_CURRENT["apparent"]] = calculate_apparent_temperature(
        InterPcurrent[DATA_CURRENT["temp"]],  # Air temperature in Kelvin
        InterPcurrent[DATA_CURRENT["humidity"]],  # Relative humidity (0.0 to 1.0)
        currentWindSpeedMps,  # Wind speed in meters per second
    )

    if "nbm" in sourceList:
        InterPcurrent[DATA_CURRENT["feels_like"]] = (
            NBM_Merged[currentIDX_hrrrh_A, NBM["apparent"]] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, NBM["apparent"]] * interpFac2
        )
    else:
        InterPcurrent[DATA_CURRENT["feels_like"]] = (
            GFS_Merged[currentIDX_hrrrh_A, GFS["apparent"]] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, GFS["apparent"]] * interpFac2
        )

    # Clip
    InterPcurrent[DATA_CURRENT["feels_like"]] = clipLog(
        InterPcurrent[DATA_CURRENT["feels_like"]],
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )

    # Fire index from NBM Fire
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

    # Current temperature in Celsius
    curr_temp = (
        InterPcurrent[DATA_CURRENT["temp"]] - KELVIN_TO_CELSIUS
    )  # temperature in Celsius

    # Put temperature into units
    if tempUnits == 0:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            InterPcurrent[DATA_CURRENT["temp"]] - KELVIN_TO_CELSIUS
        ) * 9 / 5 + 32  # "temperature"
        InterPcurrent[DATA_CURRENT["apparent"]] = (
            InterPcurrent[DATA_CURRENT["apparent"]] - KELVIN_TO_CELSIUS
        ) * 9 / 5 + 32  # "apparentTemperature"
        InterPcurrent[DATA_CURRENT["dew"]] = (
            InterPcurrent[DATA_CURRENT["dew"]] - KELVIN_TO_CELSIUS
        ) * 9 / 5 + 32  # "dewPoint"
        InterPcurrent[DATA_CURRENT["feels_like"]] = (
            InterPcurrent[DATA_CURRENT["feels_like"]] - KELVIN_TO_CELSIUS
        ) * 9 / 5 + 32  # "FeelsLike"

    else:
        InterPcurrent[DATA_CURRENT["temp"]] = (
            InterPcurrent[DATA_CURRENT["temp"]] - tempUnits
        )  # "temperature"
        InterPcurrent[DATA_CURRENT["apparent"]] = (
            InterPcurrent[DATA_CURRENT["apparent"]] - tempUnits
        )  # "apparentTemperature"
        InterPcurrent[DATA_CURRENT["dew"]] = (
            InterPcurrent[DATA_CURRENT["dew"]] - tempUnits
        )  # "dewPoint"
        InterPcurrent[DATA_CURRENT["feels_like"]] = (
            InterPcurrent[DATA_CURRENT["feels_like"]] - tempUnits
        )  # "FeelsLike"

    if (
        (minuteDict[0]["precipIntensity"])
        > (HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM * prepIntensityUnit)
    ) & (minuteDict[0]["precipType"] is not None):
        # If more than 25% chance of precip, then the icon for whatever is happening, so long as the icon exists
        cIcon = minuteDict[0]["precipType"]
        cText = minuteDict[0]["precipType"][0].upper() + minuteDict[0]["precipType"][1:]

        # Because soemtimes there's precipitation not no type, don't use an icon in those cases

    # If visibility <1km and during the day
    elif InterPcurrent[DATA_CURRENT["vis"]] < (FOG_THRESHOLD_METERS * visUnits):
        cIcon = "fog"
        cText = "Fog"
    elif InterPcurrent[DATA_CURRENT["wind"]] > (WIND_THRESHOLDS["light"] * windUnit):
        cIcon = "wind"
        cText = "Windy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["cloudy"]:
        cIcon = "cloudy"
        cText = "Cloudy"
    elif InterPcurrent[DATA_CURRENT["cloud"]] > CLOUD_COVER_THRESHOLDS["partly_cloudy"]:
        cText = "Partly Cloudy"

        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            # Before sunrise
            cIcon = "partly-cloudy-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            # After sunrise before sunset
            cIcon = "partly-cloudy-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            # After sunset
            cIcon = "partly-cloudy-night"
    else:
        cText = "Clear"
        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            # Before sunrise
            cIcon = "clear-night"
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            # After sunrise before sunset
            cIcon = "clear-day"
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            # After sunset
            cIcon = "clear-night"

    # Timing Check
    if TIMING:
        print("Object Start")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    InterPcurrent = InterPcurrent.round(2)
    InterPcurrent[np.isnan(InterPcurrent)] = MISSING_DATA

    # Fix small neg zero
    InterPcurrent[((InterPcurrent > -0.01) & (InterPcurrent < 0.01))] = 0

    # Convert intensity to accumulation based on type
    currnetRainAccum = 0
    currnetSnowAccum = 0
    currnetIceAccum = 0

    if minuteDict[0]["precipType"] in ("rain", "none"):
        currnetRainAccum = (
            minuteDict[0]["precipIntensity"] / prepIntensityUnit * prepAccumUnit
        )
    elif minuteDict[0]["precipType"] == "snow":
        # Use the new snow height estimation (in mm), then convert to requested units
        curr_liquid = minuteDict[0]["precipIntensity"] / prepIntensityUnit
        currnetSnowAccum = (
            estimate_snow_height(curr_liquid, curr_temp, currentWindSpeedMps)
            * prepAccumUnit
        )
    elif minuteDict[0]["precipType"] == "sleet":
        currnetIceAccum = (
            minuteDict[0]["precipIntensity"] / prepIntensityUnit * prepAccumUnit
        )

    ### RETURN ###
    returnOBJ = dict()

    returnOBJ["latitude"] = round(float(lat), 4)
    returnOBJ["longitude"] = round(float(lon_IN), 4)
    returnOBJ["timezone"] = str(tz_name)
    returnOBJ["offset"] = float(tz_offset / 60)
    returnOBJ["elevation"] = round(float(ETOPO * elevUnit))

    if exCurrently != 1:
        returnOBJ["currently"] = dict()
        returnOBJ["currently"]["time"] = int(minute_array_grib[0])
        returnOBJ["currently"]["summary"] = cText
        returnOBJ["currently"]["icon"] = cIcon
        returnOBJ["currently"]["nearestStormDistance"] = InterPcurrent[
            DATA_CURRENT["storm_dist"]
        ]
        returnOBJ["currently"]["nearestStormBearing"] = int(
            InterPcurrent[DATA_CURRENT["storm_dir"]].round()
        )
        returnOBJ["currently"]["precipIntensity"] = minuteDict[0]["precipIntensity"]
        returnOBJ["currently"]["precipProbability"] = minuteDict[0]["precipProbability"]
        returnOBJ["currently"]["precipIntensityError"] = minuteDict[0][
            "precipIntensityError"
        ]
        returnOBJ["currently"]["precipType"] = minuteDict[0]["precipType"]
        returnOBJ["currently"]["temperature"] = InterPcurrent[DATA_CURRENT["temp"]]
        returnOBJ["currently"]["apparentTemperature"] = InterPcurrent[
            DATA_CURRENT["apparent"]
        ]
        returnOBJ["currently"]["dewPoint"] = InterPcurrent[DATA_CURRENT["dew"]]
        returnOBJ["currently"]["humidity"] = InterPcurrent[DATA_CURRENT["humidity"]]
        returnOBJ["currently"]["pressure"] = InterPcurrent[DATA_CURRENT["pressure"]]
        returnOBJ["currently"]["windSpeed"] = InterPcurrent[DATA_CURRENT["wind"]]
        returnOBJ["currently"]["windGust"] = InterPcurrent[DATA_CURRENT["gust"]]
        returnOBJ["currently"]["windBearing"] = int(
            np.mod(InterPcurrent[DATA_CURRENT["bearing"]], 360).round()
        )
        returnOBJ["currently"]["cloudCover"] = InterPcurrent[DATA_CURRENT["cloud"]]
        returnOBJ["currently"]["uvIndex"] = InterPcurrent[DATA_CURRENT["uv"]]
        returnOBJ["currently"]["visibility"] = InterPcurrent[DATA_CURRENT["vis"]]
        returnOBJ["currently"]["ozone"] = InterPcurrent[DATA_CURRENT["ozone"]]
        returnOBJ["currently"]["smoke"] = InterPcurrent[
            DATA_CURRENT["smoke"]
        ]  # kg/m3 to ug/m3
        returnOBJ["currently"]["fireIndex"] = InterPcurrent[DATA_CURRENT["fire"]]
        returnOBJ["currently"]["feelsLike"] = InterPcurrent[DATA_CURRENT["feels_like"]]
        returnOBJ["currently"]["currentDayIce"] = dayZeroIce
        returnOBJ["currently"]["currentDayLiquid"] = dayZeroRain
        returnOBJ["currently"]["currentDaySnow"] = dayZeroSnow

        if "stationPressure" in extraVars:
            returnOBJ["currently"]["stationPressure"] = InterPcurrent[
                DATA_CURRENT["station_pressure"]
            ]

        # Update the text
        if InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunrise"]]:
            # Before sunrise
            currentDay = False
        elif (
            InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunrise"]]
            and InterPcurrent[DATA_CURRENT["time"]] < InterSday[0, DATA_DAY["sunset"]]
        ):
            # After sunrise before sunset
            currentDay = True
        elif InterPcurrent[DATA_CURRENT["time"]] > InterSday[0, DATA_DAY["sunset"]]:
            # After sunset
            currentDay = False

        try:
            currentText, currentIcon = calculate_text(
                returnOBJ["currently"],
                prepAccumUnit,
                visUnits,
                windUnit,
                tempUnits,
                currentDay,
                currnetRainAccum,
                currnetSnowAccum,
                currnetIceAccum,
                "current",
                minuteDict[0]["precipIntensity"],
                icon,
            )
            if summaryText:
                returnOBJ["currently"]["summary"] = translation.translate(
                    ["title", currentText]
                )
                returnOBJ["currently"]["icon"] = currentIcon
        except Exception:
            print("CURRENTLY TEXT GEN ERROR:")
            print(traceback.print_exc())

        if version < 2:
            returnOBJ["currently"].pop("smoke", None)
            returnOBJ["currently"].pop("currentDayIce", None)
            returnOBJ["currently"].pop("currentDayLiquid", None)
            returnOBJ["currently"].pop("currentDaySnow", None)
            returnOBJ["currently"].pop("fireIndex", None)
            returnOBJ["currently"].pop("feelsLike", None)

        if timeMachine and not tmExtra:
            returnOBJ["currently"].pop("nearestStormDistance", None)
            returnOBJ["currently"].pop("nearestStormBearing", None)
            returnOBJ["currently"].pop("precipProbability", None)
            returnOBJ["currently"].pop("precipIntensityError", None)
            returnOBJ["currently"].pop("humidity", None)
            returnOBJ["currently"].pop("uvIndex", None)
            returnOBJ["currently"].pop("visibility", None)
            returnOBJ["currently"].pop("ozone", None)

    if exMinutely != 1:
        returnOBJ["minutely"] = dict()
        try:
            if summaryText:
                minuteText, minuteIcon = calculate_minutely_text(
                    minuteDict, currentText, currentIcon, icon, prepIntensityUnit
                )
                returnOBJ["minutely"]["summary"] = translation.translate(
                    ["sentence", minuteText]
                )
                returnOBJ["minutely"]["icon"] = minuteIcon
            else:
                returnOBJ["minutely"]["summary"] = pTypesText[
                    int(Counter(maxPchance).most_common(1)[0][0])
                ]
                returnOBJ["minutely"]["icon"] = pTypesIcon[
                    int(Counter(maxPchance).most_common(1)[0][0])
                ]

        except Exception:
            print("MINUTELY TEXT GEN ERROR:")
            print(traceback.print_exc())
            returnOBJ["minutely"]["summary"] = pTypesText[
                int(Counter(maxPchance).most_common(1)[0][0])
            ]
            returnOBJ["minutely"]["icon"] = pTypesIcon[
                int(Counter(maxPchance).most_common(1)[0][0])
            ]

        returnOBJ["minutely"]["data"] = minuteDict

    if exHourly != 1:
        returnOBJ["hourly"] = dict()
        if (not timeMachine) or (tmExtra):
            try:
                hourIcon, hourText = calculate_day_text(
                    hourList[int(baseTimeOffset) : int(baseTimeOffset) + 24],
                    prepAccumUnit,
                    visUnits,
                    windUnit,
                    tempUnits,
                    True,
                    str(tz_name),
                    int(time.time()),
                    "hour",
                    icon,
                )
                if summaryText:
                    returnOBJ["hourly"]["summary"] = translation.translate(
                        ["sentence", hourText]
                    )
                    returnOBJ["hourly"]["icon"] = hourIcon
                else:
                    returnOBJ["hourly"]["summary"] = max(
                        set(hourTextList), key=hourTextList.count
                    )
                    returnOBJ["hourly"]["icon"] = max(
                        set(hourIconList), key=hourIconList.count
                    )

            except Exception:
                print("TEXT GEN ERROR:")
                print(traceback.print_exc())
                returnOBJ["hourly"]["summary"] = max(
                    set(hourTextList), key=hourTextList.count
                )
                returnOBJ["hourly"]["icon"] = max(
                    set(hourIconList), key=hourIconList.count
                )

        # Final hourly cleanup.
        fieldsToRemove = []

        # Remove 'smoke' if the version is less than 2.
        if version < 2:
            fieldsToRemove.append("smoke")

        # Remove extra fields for basic Time Machine requests.
        if timeMachine and not tmExtra:
            fieldsToRemove.extend(
                [
                    "precipProbability",
                    "precipIntensityError",
                    "humidity",
                    "visibility",
                ]
            )

        # Apply all identified removals to the final hourList.
        if fieldsToRemove:
            for hourItem in hourList:
                for field in fieldsToRemove:
                    hourItem.pop(field, None)

        # If a timemachine request, do not offset to now
        if timeMachine or timeMachineNear:
            returnOBJ["hourly"]["data"] = hourList[0:ouputHours]
        else:
            returnOBJ["hourly"]["data"] = hourList[
                int(baseTimeOffset) : int(baseTimeOffset) + ouputHours
            ]

    if exDaily != 1:
        returnOBJ["daily"] = dict()
        if (not timeMachine) or (tmExtra):
            try:
                if summaryText:
                    weekText, weekIcon = calculate_weekly_text(
                        dayList, prepAccumUnit, tempUnits, str(tz_name), icon
                    )
                    returnOBJ["daily"]["summary"] = translation.translate(
                        ["sentence", weekText]
                    )
                    returnOBJ["daily"]["icon"] = weekIcon
                else:
                    returnOBJ["daily"]["summary"] = max(
                        set(dayTextList), key=dayTextList.count
                    )
                    returnOBJ["daily"]["icon"] = max(
                        set(dayIconList), key=dayIconList.count
                    )

            except Exception:
                print("DAILY SUMMARY TEXT GEN ERROR:")
                print(traceback.print_exc())
                returnOBJ["daily"]["summary"] = max(
                    set(dayTextList), key=dayTextList.count
                )
                returnOBJ["daily"]["icon"] = max(
                    set(dayIconList), key=dayIconList.count
                )
        returnOBJ["daily"]["data"] = dayList[0:ouputDays]

    if exAlerts != 1:
        returnOBJ["alerts"] = alertList

    # Timing Check
    if TIMING:
        print("Final Time")
        print(datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start)

    if exFlags != 1:
        returnOBJ["flags"] = dict()
        returnOBJ["flags"]["sources"] = sourceList
        returnOBJ["flags"]["sourceTimes"] = sourceTimes
        returnOBJ["flags"]["nearest-station"] = int(0)
        returnOBJ["flags"]["units"] = unitSystem
        returnOBJ["flags"]["version"] = API_VERSION
        if version >= 2:
            returnOBJ["flags"]["sourceIDX"] = sourceIDX
            returnOBJ["flags"]["processTime"] = (
                datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
            ).microseconds

        # if timeMachine:
        # lock.release()

    return ORJSONResponse(
        content=returnOBJ,
        headers={
            "X-Node-ID": platform.node(),
            "X-Response-Time": str(
                (
                    datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - T_Start
                ).microseconds
            ),
            "Cache-Control": "max-age=900, must-revalidate",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


@app.on_event("startup")
def initialDataSync() -> None:
    global zarrReady

    zarrReady = False
    print("Initial Download")

    STAGE = os.environ.get("STAGE", "PROD")
    if STAGE == "PROD":
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/SubH.zarr.zip",
            "/tmp/SubH_TMP.zarr.zip",
            "/tmp/SubH.zarr.prod.zip",
            True,
        )
        print("SubH Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/HRRR_6H.zarr.zip",
            "/tmp/HRRR_6H_TMP.zarr.zip",
            "/tmp/HRRR_6H.zarr.prod.zip",
            True,
        )
        print("HRRR_6H Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/GFS.zarr.zip",
            "/tmp/GFS.zarr_TMP.zip",
            "/tmp/GFS.zarr.prod.zip",
            True,
        )
        print("GFS Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/NBM.zarr.zip",
            "/tmp/NBM.zarr_TMP.zip",
            "/tmp/NBM.zarr.prod.zip",
            True,
        )
        print("NBM Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/NBM_Fire.zarr.zip",
            "/tmp/NBM_Fire_TMP.zarr.zip",
            "/tmp/NBM_Fire.zarr.prod.zip",
            True,
        )
        print("NBM_Fire Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/GEFS.zarr.zip",
            "/tmp/GEFS_TMP.zarr.zip",
            "/tmp/GEFS.zarr.prod.zip",
            True,
        )
        print("GEFS  Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/HRRR.zarr.zip",
            "/tmp/HRRR_TMP.zarr.zip",
            "/tmp/HRRR.zarr.prod.zip",
            True,
        )
        print("HRRR  Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar_v2/" + ingestVersion + "/NWS_Alerts.zarr.zip",
            "/tmp/NWS_Alerts_TMP.zarr.zip",
            "/tmp/NWS_Alerts.zarr.prod.zip",
            True,
        )
        print("Alerts Download!")

        if useETOPO:
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/ETOPO_DA_C.zarr.zip",
                "/tmp/ETOPO_DA_C_TMP.zarr.zip",
                "/tmp/ETOPO_DA_C.zarr.prod.zip",
                True,
            )
            print("ETOPO Download!")
    else:
        print(STAGE)
    if (STAGE == "PROD") or (STAGE == "DEV"):
        update_zarr_store(True)

    zarrReady = True

    print("Initial Download End!")


@app.on_event("startup")
@repeat_every(seconds=60 * 5, logger=logger)  # 5 Minute
def dataSync() -> None:
    global zarrReady

    logger.info(zarrReady)

    STAGE = os.environ.get("STAGE", "PROD")

    if zarrReady:
        if STAGE == "PROD":
            time.sleep(20)
            logger.info("Starting Update")

            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/SubH.zarr.zip",
                "/tmp/SubH_TMP.zarr.zip",
                "/tmp/SubH.zarr.prod.zip",
                False,
            )
            logger.info("SubH Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/HRRR_6H.zarr.zip",
                "/tmp/HRRR_6H_TMP.zarr.zip",
                "/tmp/HRRR_6H.zarr.prod.zip",
                False,
            )
            logger.info("HRRR_6H Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/GFS.zarr.zip",
                "/tmp/GFS.zarr_TMP.zip",
                "/tmp/GFS.zarr.prod.zip",
                False,
            )
            logger.info("GFS Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/NBM.zarr.zip",
                "/tmp/NBM.zarr_TMP.zip",
                "/tmp/NBM.zarr.prod.zip",
                False,
            )
            logger.info("NBM Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/NBM_Fire.zarr.zip",
                "/tmp/NBM_Fire_TMP.zarr.zip",
                "/tmp/NBM_Fire.zarr.prod.zip",
                False,
            )
            logger.info("NBM_Fire Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/GEFS.zarr.zip",
                "/tmp/GEFS_TMP.zarr.zip",
                "/tmp/GEFS.zarr.prod.zip",
                False,
            )
            logger.info("GEFS  Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/HRRR.zarr.zip",
                "/tmp/HRRR_TMP.zarr.zip",
                "/tmp/HRRR.zarr.prod.zip",
                False,
            )
            logger.info("HRRR  Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar_v2/" + ingestVersion + "/NWS_Alerts.zarr.zip",
                "/tmp/NWS_Alerts_TMP.zarr.zip",
                "/tmp/NWS_Alerts.zarr.prod.zip",
                False,
            )
            logger.info("Alerts Download!")

            if useETOPO:
                download_if_newer(
                    s3_bucket,
                    "ForecastTar_v2/" + ingestVersion + "/ETOPO_DA_C.zarr.zip",
                    "/tmp/ETOPO_DA_C_TMP.zarr.zip",
                    "/tmp/ETOPO_DA_C.zarr.prod.zip",
                    False,
                )
                logger.info("ETOPO Download!")
        else:
            print(STAGE)

        if (STAGE == "PROD") or (STAGE == "DEV"):
            update_zarr_store(False)

    logger.info("Sync End!")


def calculate_apparent_temperature(airTemp, humidity, wind):
    """
    Calculates the apparent temperature temperature based on air temperature, wind speed and humidity
    Formula from: https://github.com/breezy-weather/breezy-weather/discussions/1085
    AT = Ta + 0.33 * rh / 100 * 6.105 * exp(17.27 * Ta / (237.7 + Ta)) - 0.70 * ws - 4.00

    Parameters:
    - airTemperature (float): Air temperature
    - humidity (float): Relative humidity
    - windSpeed (float): Wind speed in meters per second

    Returns:
    - float: Apparent temperature
    """

    # Convert air_temp from Kelvin to Celsius for the formula parts that use Celsius
    airTempC = airTemp - KELVIN_TO_CELSIUS

    # Calculate water vapor pressure 'e'
    # Ensure humidity is not 0 for calculation, replace with a small non-zero value if needed
    # The original equation does not guard for zero humidity. If relative_humidity_0_1 is 0, e will be 0.
    e = (
        humidity
        * APPARENT_TEMP_CONSTS["e_const"]
        * np.exp(
            APPARENT_TEMP_CONSTS["exp_a"]
            * airTempC
            / (APPARENT_TEMP_CONSTS["exp_b"] + airTempC)
        )
    )

    # Calculate apparent temperature in Celsius
    apparentTempC = (
        airTempC
        + APPARENT_TEMP_CONSTS["humidity_factor"] * e
        - APPARENT_TEMP_CONSTS["wind_factor"] * wind
        + APPARENT_TEMP_CONSTS["const"]
    )

    # Convert back to Kelvin
    apparentTempK = apparentTempC + KELVIN_TO_CELSIUS

    # Clip between -90 and 60
    return clipLog(
        apparentTempK,
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )


def clipLog(data, min, max, name):
    """
    Clip the data between min and max. Log if there is an error
    """

    # Print if the clipping is larger than 25 of the min
    if data.min() < (min - 0.25):
        # Print the data and the index it occurs
        logger.error("Min clipping required for " + name)
        logger.error("Min Value: " + str(data.min()))
        if isinstance(data, np.ndarray):
            logger.error("Min Index: " + str(np.where(data == data.min())))

        # Replace values below the threshold with np.nan
        if np.isscalar(data):
            if data < min:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data < min] = np.nan

    else:
        data = np.clip(data, a_min=min, a_max=None)

    # Same for max
    if data.max() > (max + 0.25):
        logger.error("Max clipping required for " + name)
        logger.error("Max Value: " + str(data.max()))

        # Print the data and the index it occurs
        if isinstance(data, np.ndarray):
            logger.error("Max Index: " + str(np.where(data == data.max())))

        # Replace values above the threshold with np.nan
        if np.isscalar(data):
            if data > max:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data > max] = np.nan

    else:
        data = np.clip(data, a_min=None, a_max=max)

    return data


def nearest_index(a, v):
    # Slightly faster than a simple linear search for large arrays
    # Find insertion point
    idx = np.searchsorted(a, v)
    # Clip so we don’t run off the ends
    idx = np.clip(idx, 1, len(a) - 1)
    # Look at neighbors, pick the closer one
    left, right = a[idx - 1], a[idx]
    return idx if abs(right - v) < abs(v - left) else idx - 1
