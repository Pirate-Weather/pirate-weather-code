import asyncio
import datetime
import logging
import math
import os
import pickle
import platform
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from typing import Union

import boto3
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from astral import LocationInfo, moon
from astral.sun import sun
from boto3.s3.transfer import TransferConfig
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from fastapi_utils.tasks import repeat_every
from PirateText import calculate_text
from PirateSimpleDaily import calculate_simple_day_text
from pytz import timezone, utc
from timemachine import TimeMachine
from timezonefinder import TimezoneFinder


from pirateweather_translations.dynamic_loader import load_all_translations

Translations = load_all_translations()

lock = threading.Lock()

aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
save_type = os.getenv("save_type", default="S3")
s3_bucket = os.getenv("s3_bucket", default="piratezarr2")
useETOPO = os.getenv("useETOPO", default=True)
print(os.environ.get("TIMING", False))
TIMING = os.environ.get("TIMING", False)

force_now = os.getenv("force_now", default=False)


def download_if_newer(
    s3_bucket, s3_object_key, local_file_path, local_lmdb_path, initialDownload
):
    if initialDownload:
        config = TransferConfig(use_threads=True, max_bandwidth=None)
    else:
        config = TransferConfig(use_threads=False, max_bandwidth=100000000)

    # Initialize the S3 client
    if save_type == "S3":
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # Get the last modified timestamp of the S3 object
        s3_response = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
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
                s3_client.download_file(
                    s3_bucket, s3_object_key, local_file_path, Config=config
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
            s3_client.download_file(
                s3_bucket, s3_object_key, local_file_path, Config=config
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

        # local_lmdb_path_tmp = local_lmdb_path + "_TMP"

        # if initialDownload:
        #     command = f"unzip -q -o {local_file_path} -d {local_lmdb_path_tmp}"
        # else:
        #     command = f"nice -n 20 ionice -c 3 unzip -q -o {local_file_path} -d {local_lmdb_path_tmp}"

        # process = subprocess.Popen(command, shell=True)
        # subprocess.run(["ionice", "-c", "3", "unzip", "-q", "-o", local_file_path, "-d", local_lmdb_path_tmp], shell=True, check=True)
        # subprocess.run(command, shell=True)

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
    largest_value = -1
    largest_dir = None
    old_dirs = []

    STAGE = os.environ.get("STAGE", "PROD")

    for entry in os.listdir(parent_dir):
        # entry_path = os.path.join(parent_dir, entry)
        if (key_string in entry) & ("TMP" not in entry):
            old_dirs.append(entry)
            try:
                # Extract the integer value from the directory name
                value = float(entry[-12:])

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
            # command = f"nice -n 20 rsync -a --bwlimit=200 --delete /tmp/empty/ /tmp/{old_dir}/"
            # subprocess.run(command, shell=True)
            command = f"nice -n 20 rm -rf /tmp/{old_dir}"
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
        logger.info("ETOPO Setup")

    print("Refreshed Zarrs")


app = FastAPI()


def solar_rad(D_t, lat, t_t):
    """
    returns The theortical clear sky short wave radiation
    https://www.mdpi.com/2072-4292/5/10/4735/htm
    """

    d = 1 + 0.0167 * math.sin((2 * math.pi * (D_t - 93.5365)) / 365)
    r = 0.75
    S_0 = 1367
    delta = 0.4096 * math.sin((2 * math.pi * (D_t + 284)) / 365)
    radLat = np.deg2rad(lat)
    solarHour = math.pi * ((t_t - 12) / 12)
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
# STAGE = os.environ.get("STAGE", "PROD")
# if STAGE == "TESTING":
#     print("Setting up S3 zarrs")
#
#     # If S3, use that, otherwise use local
#     if save_type == "S3":
#         s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/NWS_Alerts.zarr.zip")
#     else:
#         f = "/tmp/NWS_Alerts.zarr.zip"
#
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     NWS_Alerts_Zarr = zarr.open(store, mode="r")
#     print("Alerts Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/SubH.zarr.zip")
#     else:
#         f = "/tmp/SubH.zarr.zip"
#
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     SubH_Zarr = zarr.open(store, mode="r")
#     print("SubH Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/HRRR_6H.zarr.zip")
#     else:
#         f = "/tmp/HRRR_6H.zarr.zip"
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     HRRR_6H_Zarr = zarr.open(store, mode="r")
#     print("HRRR_6H Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/GFS.zarr.zip")
#     else:
#         f = "/tmp/GFS.zarr.zip"
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     GFS_Zarr = zarr.open(store, mode="r")
#     print("GFS Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/GEFS.zarr.zip")
#     else:
#         f = "/tmp/GEFS.zarr.zip"
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     GEFS_Zarr = zarr.open(store, mode="r")
#     print("GEFS Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/NBM.zarr.zip")
#     else:
#         f = "/tmp/NBM.zarr.zip"
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     NBM_Zarr = zarr.open(store, mode="r")
#     print("NBM Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/NBM_Fire.zarr.zip")
#     else:
#         f = "/tmp/NBM_Fire.zarr.zip"
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     NBM_Fire_Zarr = zarr.open(store, mode="r")
#     print("NBM Fire Read")
#
#     if save_type == "S3":
#         f = s3.open("s3://" + s3_bucket + "/ForecastTar/HRRR.zarr.zip")
#     else:
#         f = "/tmp/HRRR.zarr.zip"
#     fs = ZipFileSystem(f, mode="r")
#     store = FSMap("", fs, check=False)
#     HRRR_Zarr = zarr.open(store, mode="r")
#     print("HRRR Read")
#
#     if useETOPO:
#         if save_type == "S3":
#             f = s3.open("s3://" + s3_bucket + "/ForecastTar/ETOPO_DA_C.zarr.zip")
#         else:
#             f = "/tmp/ETOPO_DA_C.zarr.zip"
#         fs = ZipFileSystem(f, mode="r")
#         store = FSMap("", fs, check=False)
#         ETOPO_f = zarr.open(store, mode="r")
#     print("ETOPO Read")


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


def arrayInterp(hour_array_grib, modelData, modelIndex):
    modelInterp = np.interp(
        hour_array_grib,
        modelData[:, 0],
        modelData[:, modelIndex],
        left=np.nan,
        right=np.nan,
    )

    return modelInterp


class WeatherParallel(object):
    async def zarr_read(self, model, opened_zarr, x, y):
        if TIMING:
            print("### " + model + " Reading!")
            print(datetime.datetime.utcnow())

        errCount = 0
        dataOut = False
        # Try to read HRRR Zarr
        while errCount < 4:
            try:
                dataOut = await asyncio.to_thread(lambda: opened_zarr[:, :, y, x].T)
                if TIMING:
                    print("### " + model + " Done!")
                    print(datetime.datetime.utcnow())
                return dataOut

            except Exception:
                errCount = errCount + 1

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

    top = 49.3457868  # north lat
    left = -124.7844079  # west long
    right = -66.9513812  # east long
    bottom = 24.7433195  # south lat

    inside_box = 0
    if (bottom <= lat <= top) and (left <= lng <= right):
        inside_box = 1

    return inside_box


def find_nearest(array, value):
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array) - 1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx - 1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx - 1]
        else:
            idx_nearest = idx_sorted[idx]
    return idx_nearest


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
    # Constants
    G_sc = 1367  # Solar constant in W/m^2

    # Get the day of the year and Local Solar Time (LST)
    day_of_year, local_solar_time = unix_to_day_of_year_and_lst(unix_time, longitude)

    # Calculate solar declination (delta) in radians
    delta = math.radians(23.45) * math.sin(
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
    AM = 1 / sin_alpha if sin_alpha > 0 else float("inf")  # Avoid division by zero

    # Calculate extraterrestrial solar irradiance (G_0)
    G_0 = G_sc * (1 + 0.033 * math.cos(math.radians(360 * day_of_year / 365)))

    # Calculate clear-sky solar irradiance (G)
    G = (
        G_0 * sin_alpha * math.exp(-0.14 * AM) if sin_alpha > 0 else 0
    )  # Ensure no negative irradiance

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
    globe_temperature = air_temperature + (1.5 * 10**8 * (solar_radiation**0.6)) / (
        emissivity * (globe_diameter**0.4) * (wind_speed**0.6)
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

        wbgt = 0.7 * temperature + 0.2 * globe_temperature + 0.1 * wind_speed
    else:
        wbgt = 0.7 * temperature + 0.3 * (humidity / 100.0 * temperature)

    return wbgt


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

    print(os.environ.get("STAGE", "PROD"))
    STAGE = os.environ.get("STAGE", "PROD")

    # Timing Check
    T_Start = datetime.datetime.utcnow()

    # Current time
    if force_now is False:
        nowTime = datetime.datetime.utcnow()
    else:
        # Force now for testing with static inputs
        nowTime = datetime.datetime.utcfromtimestamp(int(force_now))
        print("Forced Current Time to:")
        print(nowTime)

    ### If developing in REPL, uncomment to provide static variables
    # location = "47.1756,27.594,1730869200"
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
                utcTime = datetime.datetime.utcfromtimestamp(float(locationReq[2]))
            elif float(locationReq[2]) < -100000:  # Very negatime time
                utcTime = datetime.datetime.utcfromtimestamp(float(locationReq[2]))
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

    if utcTime < datetime.datetime(2024, 5, 1):
        timeMachine = True
        # print(request.url)
        if (
            ("localhost" in str(request.url))
            or ("timemachine" in str(request.url))
            or ("127.0.0.1" in str(request.url))
        ):
            TM_Response = await TimeMachine(
                lat,
                lon,
                az_Lon,
                utcTime,
                tf,
                units,
                exclude,
            )

            return TM_Response
        else:
            raise HTTPException(
                status_code=400,
                detail="Requested Time is in the Past. Please Use Timemachine.",
            )

    elif (nowTime - utcTime) > datetime.timedelta(hours=25):
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

    # Timing Check
    if TIMING:
        print("Request process time")
        print(datetime.datetime.utcnow() - T_Start)

    # Calculate the timezone offset
    tz_offsetLoc = {"lat": lat, "lng": az_Lon, "utcTime": utcTime, "tf": tf}
    tz_offset, tz_name = get_offset(**tz_offsetLoc)

    tzReq = tf.timezone_at(lat=lat, lng=az_Lon)

    # Timing Check
    if TIMING:
        print("Timezone offset time")
        print(datetime.datetime.utcnow() - T_Start)

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

    # Set up translations
    if not lang:
        lang = "en"

    # Check if langugage is supported
    if lang not in Translations:
        # Throw an error
        raise HTTPException(status_code=400, detail="Language Not Supported")

    translation = Translations[lang]
    if not translation:
        translation = Translations["en"]

    # Check if extra information should be included with time machine
    if not tmextra:
        tmExtra = False
    else:
        tmExtra = True

    if not exclude:
        excludeParams = ""
    else:
        excludeParams = exclude

    exCurrently = 0
    exMinutely = 0
    exHourly = 0
    exDaily = 0
    exFlags = 0
    exAlerts = 0
    exNBM = 0
    exHRRR = 0
    exGEFS = 0

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

    # Set up timemache params
    if timeMachine and not tmExtra:
        exMinutely = 1

    if timeMachine:
        exAlerts = 1

    # Exclude Alerts outside US
    if exAlerts == 0:
        if cull(az_Lon, lat) == 0:
            exAlerts = 1

    # Default to US :(
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
            tempUnits = 273.15  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        elif unitSystem == "uk":
            windUnit = 2.234  # mph
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = 273.15  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.00062137  # miles
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        elif unitSystem == "us":
            windUnit = 2.234  # mph
            prepIntensityUnit = 0.0394  # inches/hour
            prepAccumUnit = 0.0394  # inches
            tempUnits = 0  # F. This is harder
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.00062137  # miles
            humidUnit = 0.01  # %
            elevUnit = 3.28084  # ft
        elif unitSystem == "si":
            windUnit = 1  # m/s
            prepIntensityUnit = 1  # mm/h
            prepAccumUnit = 0.1  # cm
            tempUnits = 273.15  # Celsius
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.001  # km
            humidUnit = 0.01  # %
            elevUnit = 1  # m
        else:
            unitSystem = "us"
            windUnit = 2.234  # mph
            prepIntensityUnit = 0.0394  # inches/hour
            prepAccumUnit = 0.0394  # inches
            tempUnits = 0  # F. This is harder
            pressUnits = 0.01  # Hectopascals
            visUnits = 0.00062137  # miles
            humidUnit = 0.01  # %
            elevUnit = 3.28084  # ft

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
            np.datetime64(baseDay.astimezone(utc))
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
        )
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Timing Check
    if TIMING:
        print("### HRRR Start ###")
        print(datetime.datetime.utcnow() - T_Start)

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

        if (x_hrrr < 1) or (y_hrrr < 1) or (x_hrrr > 1799) or (y_hrrr > 1059):
            dataOut = False
            dataOut_h2 = False
            dataOut_hrrrh = False
        else:
            # Subh
            # Check if timemachine request, use different sources
            if timeMachine:
                date_range = pd.date_range(
                    start=baseDayUTC,
                    end=baseDayUTC + datetime.timedelta(days=1),
                    freq="1h",
                ).to_list()
                zarrList = [
                    "s3://"
                    + s3_bucket
                    + "/HRRRH/HRRRH_Hist"
                    + t.strftime("%Y%m%dT%H0000Z")
                    + ".zarr/"
                    for t in date_range
                ]

                now = time.time()
                with xr.open_mfdataset(
                    zarrList,
                    engine="zarr",
                    consolidated=True,
                    decode_cf=False,
                    parallel=True,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                    cache=False,
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
                        )

                    dataOut_hrrrh = np.zeros((len(xr_mf.time), len(HRRRHzarrVars)))

                    # Add time
                    dataOut_hrrrh[:, 0] = xr_mf.time.compute().data

                    for vIDX, v in enumerate(HRRRHzarrVars[1:]):
                        dataOut_hrrrh[:, vIDX + 1] = (
                            xr_mf[v][:, y_hrrr, x_hrrr].compute().data
                        )
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
        print(datetime.datetime.utcnow() - T_Start)
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

        if (x_nbm < 1) or (y_nbm < 1) or (x_nbm > 2344) or (y_nbm > 1596):
            dataOut_nbm = False
            dataOut_nbmFire = False
        else:
            # Timing Check
            if TIMING:
                print("### NMB Detail Start ###")
                print(datetime.datetime.utcnow() - T_Start)

            if timeMachine:
                print("NBM")
                date_range = pd.date_range(
                    start=baseDayUTC,
                    end=baseDayUTC + datetime.timedelta(days=1),
                    freq="1h",
                ).to_list()
                zarrList = [
                    "s3://"
                    + s3_bucket
                    + "/NBM/NBM_Hist"
                    + t.strftime("%Y%m%dT%H0000Z")
                    + ".zarr/"
                    for t in date_range
                ]

                now = time.time()
                with xr.open_mfdataset(
                    zarrList,
                    engine="zarr",
                    consolidated=True,
                    decode_cf=False,
                    parallel=True,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                    cache=False,
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
        print(datetime.datetime.utcnow() - T_Start)

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
        print(datetime.datetime.utcnow() - T_Start)

    if timeMachine:
        print("GFS")
        now = time.time()
        # Create list of zarrs
        hours_to_subtract = baseDayUTC.hour % 6
        rounded_time = baseDayUTC - datetime.timedelta(
            hours=hours_to_subtract,
            minutes=baseDayUTC.minute,
            seconds=baseDayUTC.second,
            microseconds=baseDayUTC.microsecond,
        )

        date_range = pd.date_range(
            start=rounded_time, end=rounded_time + datetime.timedelta(days=1), freq="6h"
        ).to_list()

        zarrList = [
            "s3://"
            + s3_bucket
            + "/GFS/GFS_Hist"
            + t.strftime("%Y%m%dT%H0000Z")
            + ".zarr/"
            for t in date_range
        ]
        with xr.open_mfdataset(
            zarrList,
            engine="zarr",
            consolidated=True,
            decode_cf=False,
            parallel=True,
            storage_options={"key": aws_access_key_id, "secret": aws_secret_access_key},
            cache=False,
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
                )
            else:
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
                )

            dataOut_gfs = np.zeros((len(xr_mf.time), len(GFSzarrVars)))
            # Add time
            dataOut_gfs[:, 0] = xr_mf.time.compute().data
            for vIDX, v in enumerate(GFSzarrVars[1:]):
                dataOut_gfs[:, vIDX + 1] = xr_mf[v][:, y_p, x_p].compute().data
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
        print(datetime.datetime.utcnow() - T_Start)

    # GEFS
    # Timing Check
    if TIMING:
        print("### GEFS Detail Start ###")
        print(datetime.datetime.utcnow() - T_Start)
    if exGEFS == 1:
        dataOut_gefs = False
    else:
        if timeMachine:
            now = time.time()
            # Create list of zarrs
            hours_to_subtract = baseDayUTC.hour % 6
            rounded_time = baseDayUTC - datetime.timedelta(
                hours=hours_to_subtract,
                minutes=baseDayUTC.minute,
                seconds=baseDayUTC.second,
                microseconds=baseDayUTC.microsecond,
            )

            date_range = pd.date_range(
                start=rounded_time,
                end=rounded_time + datetime.timedelta(days=1),
                freq="6h",
            ).to_list()
            zarrList = [
                "s3://"
                + s3_bucket
                + "/GEFS/GEFS_HistProb_"
                + t.strftime("%Y%m%dT%H0000Z")
                + ".zarr/"
                for t in date_range
            ]

            with xr.open_mfdataset(
                zarrList,
                engine="zarr",
                consolidated=True,
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
        print(datetime.datetime.utcnow() - T_Start)

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
                utcTime - datetime.datetime.utcfromtimestamp(subhRunTime.astype(int))
            ) > datetime.timedelta(hours=4):
                dataOut = False
                print("OLD SubH")

            hrrrhRunTime = dataOut_hrrrh[36, 0]
            # print( datetime.datetime.utcfromtimestamp(dataOut_hrrrh[35, 0].astype(int)))
            if (
                utcTime - datetime.datetime.utcfromtimestamp(hrrrhRunTime.astype(int))
            ) > datetime.timedelta(hours=16):
                dataOut_hrrrh = False
                print("OLD HRRRH")

            h2RunTime = dataOut_h2[0, 0]
            if (
                utcTime - datetime.datetime.utcfromtimestamp(h2RunTime.astype(int))
            ) > datetime.timedelta(hours=46):
                dataOut_h2 = False
                print("OLD HRRR_6H")

    if readNBM:
        dataOut_nbm = zarr_results["NBM"]
        dataOut_nbmFire = zarr_results["NBM_Fire"]

        if dataOut_nbm is not False:
            nbmRunTime = dataOut_nbm[36, 0]

        sourceIDX["nbm"] = dict()
        sourceIDX["nbm"]["x"] = int(x_nbm)
        sourceIDX["nbm"]["y"] = int(y_nbm)
        sourceIDX["nbm"]["lat"] = round(nbm_lat, 2)
        sourceIDX["nbm"]["lon"] = round(((nbm_lon + 180) % 360) - 180, 2)

        # Timing Check
        if TIMING:
            print("### NMB Detail End ###")
            print(datetime.datetime.utcnow() - T_Start)

        if dataOut_nbmFire is not False:
            # for i in range(0,50):
            # print( datetime.datetime.utcfromtimestamp(dataOut_nbmFire[i, 0].astype(int)))
            nbmFireRunTime = dataOut_nbmFire[30, 0]

    if readGFS:
        dataOut_gfs = zarr_results["GFS"]
        gfsRunTime = dataOut_gfs[35, 0]

    if readGEFS:
        dataOut_gefs = zarr_results["GEFS"]
        gefsRunTime = dataOut_gefs[33, 0]

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
        print(datetime.datetime.utcnow() - T_Start)

    # If point is not in HRRR coverage or HRRR-subh is more than 4 hours old, the fallback to GFS
    if isinstance(dataOut, np.ndarray):
        sourceList.append("hrrrsubh")
        sourceTimes["hrrr_subh"] = rounder(
            datetime.datetime.utcfromtimestamp(subhRunTime.astype(int))
        ).strftime("%Y-%m-%d %HZ")

    if (isinstance(dataOut_hrrrh, np.ndarray)) & (not timeMachine):
        sourceList.append("hrrr_0-18")
        sourceTimes["hrrr_0-18"] = rounder(
            datetime.datetime.utcfromtimestamp(hrrrhRunTime.astype(int))
        ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_hrrrh, np.ndarray)) & (timeMachine):
        sourceList.append("hrrr")

    if (isinstance(dataOut_nbm, np.ndarray)) & (not timeMachine):
        sourceList.append("nbm")
        sourceTimes["nbm"] = rounder(
            datetime.datetime.utcfromtimestamp(nbmRunTime.astype(int))
        ).strftime("%Y-%m-%d %HZ")
    elif (isinstance(dataOut_nbm, np.ndarray)) & (timeMachine):
        sourceList.append("nbm")

    if (isinstance(dataOut_nbmFire, np.ndarray)) & (not timeMachine):
        sourceList.append("nbm_fire")
        sourceTimes["nbm_fire"] = rounder(
            datetime.datetime.utcfromtimestamp(nbmFireRunTime.astype(int))
        ).strftime("%Y-%m-%d %HZ")

    # If point is not in HRRR coverage or HRRR-hrrrh is more than 16 hours old, the fallback to GFS
    if isinstance(dataOut_h2, np.ndarray):
        sourceList.append("hrrr_18-48")
        # Stbtract 18 hours since we're using the 18h time steo
        sourceTimes["hrrr_18-48"] = rounder(
            datetime.datetime.utcfromtimestamp(h2RunTime.astype(int))
            - datetime.timedelta(hours=18)
        ).strftime("%Y-%m-%d %HZ")

    # Always include GFS
    if timeMachine is False:
        sourceTimes["gfs"] = rounder(
            datetime.datetime.utcfromtimestamp(gfsRunTime.astype(int))
        ).strftime("%Y-%m-%d %HZ")

        if isinstance(dataOut_gefs, np.ndarray):
            sourceList.append("gefs")
            sourceTimes["gefs"] = rounder(
                datetime.datetime.utcfromtimestamp(gefsRunTime.astype(int))
            ).strftime("%Y-%m-%d %HZ")

    # Timing Check
    if TIMING:
        print("### ETOPO Start ###")
        print(datetime.datetime.utcnow() - T_Start)

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
        print(datetime.datetime.utcnow() - T_Start)

    # Number of hours to start at
    if timeMachine:
        baseTimeOffset = 0
    else:
        baseTimeOffset = (baseHour - baseDay).seconds / 3600

    # Merge hourly models onto a consistent time grid, starting from midnight on the requested day
    numHours = 193  # Number of hours to merge

    ### Minutely
    minute_array = np.arange(
        baseTime.astimezone(utc),
        baseTime + datetime.timedelta(minutes=61),
        datetime.timedelta(minutes=1),
    )
    minute_array_grib = (
        (minute_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    InterTminute = np.zeros((61, 5))  # Type
    InterPminute = np.full((61, 4), np.nan)  # Time, Intensity,Probability

    if timeMachine:
        hourly_hours = 24
        daily_days = 1
        daily_day_hours = 1
    elif extendFlag == 1:
        hourly_hours = 169
        daily_days = 8
        daily_day_hours = 1
    else:
        hourly_hours = 48
        daily_days = 8
        daily_day_hours = 1

    hour_array = np.arange(
        baseDay.astimezone(utc),
        baseDay.astimezone(utc)
        + datetime.timedelta(days=daily_days)
        + datetime.timedelta(hours=daily_day_hours),
        datetime.timedelta(hours=1),
    )

    InterPhour = np.full((len(hour_array), 27), np.nan)  # Time, Intensity,Probability

    hour_array_grib = (
        (hour_array - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0)))
        .astype("timedelta64[s]")
        .astype(np.int32)
    )

    # Timing Check
    if TIMING:
        print("Nearest IDX Start")
        print(datetime.datetime.utcnow() - T_Start)

    # HRRR
    if timeMachine is False:
        # Since the forecast files are pre-processed, they'll always be hourly and the same lenght. This avoids interpolation
        try:  # Add a fallback to GFS if these don't work
            # HRRR
            if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
                HRRR_StartIDX = find_nearest(dataOut_hrrrh[:, 0], baseDayUTC_Grib)
                H2_StartIDX = find_nearest(dataOut_h2[:, 0], dataOut_hrrrh[-1, 0])

                HRRR_Merged = np.full((numHours, dataOut_h2.shape[1]), np.nan)
                HRRR_Merged[0 : (55 - HRRR_StartIDX) + (31 - H2_StartIDX), :] = (
                    np.concatenate(
                        (dataOut_hrrrh[HRRR_StartIDX:, :], dataOut_h2[H2_StartIDX:, :]),
                        axis=0,
                    )
                )

            # NBM
            if "nbm" in sourceList:
                NBM_StartIDX = find_nearest(dataOut_nbm[:, 0], baseDayUTC_Grib)
                NBM_Merged = np.full((numHours, dataOut_nbm.shape[1]), np.nan)
                NBM_Merged[0 : (230 - NBM_StartIDX), :] = dataOut_nbm[
                    NBM_StartIDX : (numHours + NBM_StartIDX), :
                ]

            # NBM FIre
            if "nbm_fire" in sourceList:
                NBM_Fire_StartIDX = find_nearest(dataOut_nbmFire[:, 0], baseDayUTC_Grib)
                NBM_Fire_Merged = np.full((numHours, dataOut_nbmFire.shape[1]), np.nan)
                NBM_Fire_Merged[0 : (217 - NBM_Fire_StartIDX), :] = dataOut_nbmFire[
                    NBM_Fire_StartIDX : (numHours + NBM_Fire_StartIDX), :
                ]
        except Exception:
            sourceTimes.pop("hrrr_18-48")
            sourceTimes.pop("nbm_fire")
            sourceTimes.pop("nbm")
            sourceTimes.pop("hrrr_0-18")
            sourceTimes.pop("hrrr_subh")
            sourceList.remove("hrrrsubh")
            sourceList.remove("hrrr_0-18")
            sourceList.remove("nbm")
            sourceList.remove("nbm_fire")
            sourceList.remove("hrrr_18-48")

        # GFS
        GFS_StartIDX = find_nearest(dataOut_gfs[:, 0], baseDayUTC_Grib)
        GFS_EndIDX = min((len(dataOut_gfs), (numHours + GFS_StartIDX)))
        GFS_Merged = np.zeros((numHours, dataOut_gfs.shape[1]))
        GFS_Merged[0 : (GFS_EndIDX - GFS_StartIDX), :] = dataOut_gfs[
            GFS_StartIDX:GFS_EndIDX, :
        ]

        # GEFS
        if "gefs" in sourceList:
            GEFS_StartIDX = find_nearest(dataOut_gefs[:, 0], baseDayUTC_Grib)
            GEFS_Merged = dataOut_gefs[GEFS_StartIDX : (numHours + GEFS_StartIDX), :]

    # Interpolate if Time Machine
    else:
        GFS_Merged = np.zeros((len(hour_array_grib), dataOut_gfs.shape[1]))
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
        print(datetime.datetime.utcnow() - T_Start)

    InterPhour[:, 0] = hour_array_grib

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
            for i in range(9)
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
            for i in range(9)
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
            for i in range(9)
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
            for i in range(9)
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
    hourlyDayIndex = np.full(len(hour_array_grib), int(-999))
    hourlyDay4amIndex = np.full(len(hour_array_grib), int(-999))
    hourlyHighIndex = np.full(len(hour_array_grib), int(-999))
    hourlyLowIndex = np.full(len(hour_array_grib), int(-999))

    for d in range(0, 8):
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

    InterSday = np.zeros(shape=(daily_days, 21))

    # Timing Check
    if TIMING:
        print("Sunrise start")
        print(datetime.datetime.utcnow() - T_Start)

    loc = LocationInfo("name", "region", tz_name, lat, az_Lon)

    # Calculate Sunrise, Sunset, Moon Phase
    for i in range(0, daily_days):
        try:
            s = sun(
                loc.observer, date=baseDay + datetime.timedelta(days=i)
            )  # Use local to get the correct date

            InterSday[i, 17] = (
                (
                    np.datetime64(s["sunrise"])
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
            InterSday[i, 18] = (
                (
                    np.datetime64(s["sunset"])
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )

            InterSday[i, 15] = (
                (
                    np.datetime64(s["dawn"])
                    - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
                )
                .astype("timedelta64[s]")
                .astype(np.int32)
            )
            InterSday[i, 16] = (
                (
                    np.datetime64(s["dusk"])
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
                InterSday[i, 17] = day_array_grib[i] + np.timedelta64(1, "s").astype(
                    "timedelta64[s]"
                ).astype(np.int32)
                # Set sunset to one second before midnight the following day
                InterSday[i, 18] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                # Set sunrise to one second after midnight
                InterSday[i, 15] = day_array_grib[i] + np.timedelta64(1, "s").astype(
                    "timedelta64[s]"
                ).astype(np.int32)
                # Set sunset to one second before midnight the following day
                InterSday[i, 16] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

            # Else
            else:
                # Set sunrise to two seconds before midnight
                InterSday[i, 17] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(2, "s").astype("timedelta64[s]").astype(np.int32)
                )
                # Set sunset to one seconds before midnight
                InterSday[i, 18] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

                InterSday[i, 15] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(2, "s").astype("timedelta64[s]").astype(np.int32)
                )
                # Set sunset to one seconds before midnight
                InterSday[i, 16] = (
                    day_array_grib[i]
                    + np.timedelta64(1, "D").astype("timedelta64[s]").astype(np.int32)
                    - np.timedelta64(1, "s").astype("timedelta64[s]").astype(np.int32)
                )

        m = moon.phase(baseDay + datetime.timedelta(days=i))
        InterSday[i, 19] = m / 27.99

    # Timing Check
    if TIMING:
        print("Interpolation Start")
        print(datetime.datetime.utcnow() - T_Start)

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
            hrrrSubHInterpolation[:, 1] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 2],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 2] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 3],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 3] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 4],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 4] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 5],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 5] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 7],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 6] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 8],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 7] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 9],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 8] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 11],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 9] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 12],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 10] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 13],
                left=np.nan,
                right=np.nan,
            )
            hrrrSubHInterpolation[:, 11] = np.interp(
                minute_array_grib,
                HRRR_Merged[:, 0].squeeze(),
                HRRR_Merged[:, 14],
                left=np.nan,
                right=np.nan,
            )

        if "gefs" in sourceList:
            gefsMinuteInterpolation[:, 3] = np.interp(
                minute_array_grib,
                dataOut_gefs[:, 0].squeeze(),
                dataOut_gefs[:, 3],
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
        for i in [8, 12, 14, 15, 16, 17]:
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
        print(datetime.datetime.utcnow() - T_Start)

    InterPminute[:, 0] = minute_array_grib

    # "precipProbability"
    # Use NBM where available
    if "nbm" in sourceList:
        InterPminute[:, 2] = nbmMinuteInterpolation[:, 12] * 0.01
    elif "gefs" in sourceList:
        InterPminute[:, 2] = gefsMinuteInterpolation[:, 1]
    else:  # Missing (-999) fallback
        InterPminute[:, 2] = np.ones(len(minute_array_grib)) * -999

    # Less than 5% set to 0
    InterPminute[InterPminute[:, 2] < 0.05, 2] = 0

    # Prep Intensity
    # Kind of complex, process:
    # 1. If probability >0:
    # 2. If HRRR intensity >0, use that, else use NBM, unless one isn't available, then use the other one or GEFS

    # probMask = np.where(InterPminute[:, 2] > 0)
    #
    # if ('hrrrsubh' in sourceList) or ('nbm' in sourceList):
    #     subHMask = np.full(len(InterPminute), False)
    #
    #     if ('hrrrsubh' in sourceList):
    #         subHMask = np.where(hrrrSubHInterpolation[:, 7] > 0)
    #         InterPminute[subHMask, 1] = hrrrSubHInterpolation[subHMask, 7] * 3600 * prepIntensityUnit
    #
    #     if ('nbm' in sourceList):
    #         InterPminute[probMask & ~subHMask, 1] = nbmMinuteInterpolation[probMask & ~subHMask,8] * prepIntensityUnit
    # elif  ('hrrrsubh' in sourceList):
    #     InterPminute[:, 1] = hrrrSubHInterpolation[:, 7] * 3600 * prepIntensityUnit
    # elif ('nbm' in sourceList):
    #     InterPminute[:, 1] = nbmMinuteInterpolation[:,8] * prepIntensityUnit
    # else:
    #     InterPminute[:, 1] = gefsMinuteInterpolation[:, 2] * 1 * prepIntensityUnit

    # Keep it simple for now
    if "hrrrsubh" in sourceList:
        InterPminute[:, 1] = hrrrSubHInterpolation[:, 7] * 3600 * prepIntensityUnit
    elif "nbm" in sourceList:
        InterPminute[:, 1] = nbmMinuteInterpolation[:, 8] * prepIntensityUnit
    elif "gefs" in sourceList:
        InterPminute[:, 1] = gefsMinuteInterpolation[:, 2] * 1 * prepIntensityUnit
    else:  # GFS fallback
        InterPminute[:, 1] = gfsMinuteInterpolation[:, 10] * 3600 * prepIntensityUnit

    if "hrrrsubh" not in sourceList:
        # Set intensity to zero if POP == 0
        InterPminute[InterPminute[:, 2] == 0, 1] = 0

    # "precipIntensityError"
    if "gefs" in sourceList:
        InterPminute[:, 3] = gefsMinuteInterpolation[:, 3] * prepIntensityUnit
    else:  # Missing
        InterPminute[:, 3] = np.ones(len(minute_array_grib)) * -999

    # Precipitation Type
    # IF HRRR, use that, otherwise GEFS
    if "hrrrsubh" in sourceList:
        for i in [8, 9, 10, 11]:
            InterTminute[:, i - 7] = hrrrSubHInterpolation[:, i]
    elif "nbm" in sourceList:
        InterTminute[:, 1] = nbmMinuteInterpolation[:, 16]
        InterTminute[:, 2] = nbmMinuteInterpolation[:, 17]
        InterTminute[:, 3] = nbmMinuteInterpolation[:, 15]
        InterTminute[:, 4] = nbmMinuteInterpolation[:, 14]
    elif "gefs" in sourceList:
        for i in [4, 5, 6, 7]:
            InterTminute[:, i - 3] = gefsMinuteInterpolation[:, i]
    else:  # GFS Fallback
        for i in [12, 13, 14, 15]:
            InterTminute[:, i - 11] = gfsMinuteInterpolation[:, i]

    # If all nan, set pchance to -999
    if np.any(np.isnan(InterTminute)):
        maxPchance = np.full(len(minute_array_grib), 5)
    else:
        maxPchance = np.argmax(InterTminute, axis=1)

    # Create list of icons based off of maxPchance
    minuteKeys = [
        "time",
        "precipIntensity",
        "precipProbability",
        "precipIntensityError",
        "precipType",
    ]
    pTypes = ["none", "snow", "sleet", "sleet", "rain", -999]
    pTypesText = ["Clear", "Snow", "Sleet", "Sleet", "Rain", -999]
    pTypesIcon = ["clear", "snow", "sleet", "sleet", "rain", -999]

    minuteType = [pTypes[maxPchance[idx]] for idx in range(61)]

    # Assign pfactors for rain and snow for intensity
    pFacMinute = np.zeros((len(minute_array_grib)))
    pFacMinute[((maxPchance == 4) | (maxPchance == 2) | (maxPchance == 3))] = (
        1  # Rain, Ice
    )
    # Note, this means that intensity is always in liquid water equivalent
    pFacMinute[(maxPchance == 1)] = 1  # Snow

    minuteTimes = InterPminute[:, 0]
    minuteIntensity = np.maximum(np.round(InterPminute[:, 1] * pFacMinute, 4), 0)
    minuteProbability = np.minimum(np.maximum(np.round(InterPminute[:, 2], 2), 0), 1)
    minuteIntensityError = np.maximum(np.round(InterPminute[:, 3], 2), 0)

    # Convert nan to -999 for json
    minuteIntensity[np.isnan(minuteIntensity)] = -999
    minuteProbability[np.isnan(minuteProbability)] = -999
    minuteIntensityError[np.isnan(minuteIntensityError)] = -999

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
        print(datetime.datetime.utcnow() - T_Start)

    ## Approach
    # Use NBM where available
    # Use GFS past the end of NBM
    # Use HRRRH/ HRRRH2 if requested (?)
    # Use HRRR for some other variables

    ###  probVars
    ### ('time', 'Precipitation_Prob', 'APCP_Mean', 'APCP_StdDev', 'CSNOW_Prob', 'CICEP_Prob', 'CFRZR_Prob', 'CRAIN_Prob')

    # Precipitation Type
    # NBM
    maxPchanceHour = np.full((len(hour_array_grib), 3), -999)

    if "nbm" in sourceList:
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        InterThour[:, 1] = NBM_Merged[:, 16]
        InterThour[:, 2] = NBM_Merged[:, 17]
        InterThour[:, 3] = NBM_Merged[:, 15]
        InterThour[:, 4] = NBM_Merged[:, 14]

        # 14 = Rain (1,2), 15 = Freezing Rain/ Ice (3,4), 16 = Snow (5,6,7), 17 = Ice (8,9)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-201.shtml

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 0] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 0] = -999

    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterThour = np.zeros(shape=(len(hour_array), 5))
        InterThour[:, 1] = HRRR_Merged[:, 11]
        InterThour[:, 2] = HRRR_Merged[:, 12]
        InterThour[:, 3] = HRRR_Merged[:, 13]
        InterThour[:, 4] = HRRR_Merged[:, 14]

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0
        maxPchanceHour[:, 1] = np.argmax(InterThour, axis=1)
        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 1] = -999

    # GEFS
    if "gefs" in sourceList:
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        for i in [4, 5, 6, 7]:
            InterThour[:, i - 3] = GEFS_Merged[:, i]

        # 4 = Snow, 5 = Sleet, 6 = Freezing Rain, 7 = Rain

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 2] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 2] = -999
    else:  # GFS Fallback
        InterThour = np.zeros(shape=(len(hour_array), 5))  # Type
        for i in [12, 13, 14, 15]:
            InterThour[:, i - 11] = GFS_Merged[:, i]

        # 12 = Snow, 13 = Sleet, 14 = Freezing Rain, 15 = Rain

        # Fix rounding issues
        InterThour[InterThour < 0.01] = 0

        maxPchanceHour[:, 2] = np.argmax(InterThour, axis=1)

        # Put Nan's where they exist in the original data
        maxPchanceHour[np.isnan(InterThour[:, 1]), 2] = -999

    # Intensity
    # NBM
    prcipIntensityHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        prcipIntensityHour[:, 0] = NBM_Merged[:, 13]
    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        prcipIntensityHour[:, 1] = HRRR_Merged[:, 9] * 3600
    # GEFS
    if "gefs" in sourceList:
        prcipIntensityHour[:, 2] = GEFS_Merged[:, 2]
    else:  # GFS Fallback
        prcipIntensityHour[:, 2] = GFS_Merged[:, 10] * 3600
    # Take first non-NaN value
    InterPhour[:, 2] = (
        np.choose(np.argmin(np.isnan(prcipIntensityHour), axis=1), prcipIntensityHour.T)
        * prepIntensityUnit
    )

    # Set zero as the floor
    InterPhour[:, 2] = np.maximum(InterPhour[:, 2], 0)

    # Use the same type value as the intensity
    InterPhour[:, 1] = np.choose(
        np.argmin(np.isnan(prcipIntensityHour), axis=1), maxPchanceHour.T
    )

    # Probability
    # NBM
    prcipProbabilityHour = np.full((len(hour_array_grib), 2), np.nan)
    if "nbm" in sourceList:
        prcipProbabilityHour[:, 0] = NBM_Merged[:, 12] * 0.01
    # GEFS
    if "gefs" in sourceList:
        prcipProbabilityHour[:, 1] = GEFS_Merged[:, 1]

    # Take first non-NaN value
    InterPhour[:, 3] = np.choose(
        np.argmin(np.isnan(prcipProbabilityHour), axis=1), prcipProbabilityHour.T
    )
    # Cap at 1
    InterPhour[:, 3] = np.clip(InterPhour[:, 3], 0, 1)

    # Less than 5% set to 0
    InterPhour[InterPhour[:, 3] < 0.05, 3] = 0

    # Set intensity to zero if POP == 0
    InterPhour[InterPhour[:, 3] == 0, 2] = 0

    # Intensity Error
    # GEFS
    if "gefs" in sourceList:
        InterPhour[:, 4] = np.maximum(GEFS_Merged[:, 2] * prepIntensityUnit, 0)

    ### Temperature
    TemperatureHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        TemperatureHour[:, 0] = NBM_Merged[:, 2]

    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        TemperatureHour[:, 1] = HRRR_Merged[:, 4]

    if "gfs" in sourceList:
        TemperatureHour[:, 2] = GFS_Merged[:, 4]

    # Take first non-NaN value
    InterPhour[:, 5] = np.choose(
        np.argmin(np.isnan(TemperatureHour), axis=1), TemperatureHour.T
    )

    # Clip between -90 and 60
    InterPhour[:, 5] = np.clip(InterPhour[:, 5], -183, 333)

    ### Dew Point
    DewPointHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        DewPointHour[:, 0] = NBM_Merged[:, 4]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        DewPointHour[:, 1] = HRRR_Merged[:, 5]
    if "gfs" in sourceList:
        DewPointHour[:, 2] = GFS_Merged[:, 5]
    InterPhour[:, 7] = np.choose(
        np.argmin(np.isnan(DewPointHour), axis=1), DewPointHour.T
    )

    # Clip between -90 and 60 C
    InterPhour[:, 7] = np.clip(InterPhour[:, 7], -183, 333)

    ### Humidity
    HumidityHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        HumidityHour[:, 0] = NBM_Merged[:, 5]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        HumidityHour[:, 1] = HRRR_Merged[:, 6]
    if "gfs" in sourceList:
        HumidityHour[:, 2] = GFS_Merged[:, 6]
    InterPhour[:, 8] = (
        np.choose(np.argmin(np.isnan(HumidityHour), axis=1), HumidityHour.T) * humidUnit
    )

    # Clip between 0 and 1
    InterPhour[:, 8] = np.clip(InterPhour[:, 8], 0, 1)

    ### Pressure
    PressureHour = np.full((len(hour_array_grib), 2), np.nan)
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        PressureHour[:, 0] = HRRR_Merged[:, 3]
    if "gfs" in sourceList:
        PressureHour[:, 1] = GFS_Merged[:, 3]
    InterPhour[:, 9] = (
        np.choose(np.argmin(np.isnan(PressureHour), axis=1), PressureHour.T)
        * pressUnits
    )

    # Clip between 800 and 1100
    InterPhour[:, 9] = np.clip(InterPhour[:, 9], 800, 1100)

    ### Wind Speed
    WindSpeedHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        WindSpeedHour[:, 0] = NBM_Merged[:, 6]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindSpeedHour[:, 1] = np.sqrt(HRRR_Merged[:, 7] ** 2 + HRRR_Merged[:, 8] ** 2)
    if "gfs" in sourceList:
        WindSpeedHour[:, 2] = np.sqrt(GFS_Merged[:, 8] ** 2 + GFS_Merged[:, 9] ** 2)

    InterPhour[:, 10] = np.choose(
        np.argmin(np.isnan(WindSpeedHour), axis=1), WindSpeedHour.T
    )

    # Clip between 0 and 400
    InterPhour[:, 10] = np.clip(InterPhour[:, 10], 0, 120) * windUnit

    ### Wind Gust
    WindGustHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        WindGustHour[:, 0] = NBM_Merged[:, 1]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindGustHour[:, 1] = HRRR_Merged[:, 2]
    if "gfs" in sourceList:
        WindGustHour[:, 2] = GFS_Merged[:, 2]
    InterPhour[:, 11] = np.choose(
        np.argmin(np.isnan(WindGustHour), axis=1), WindGustHour.T
    )
    # Clip between 0 and 400
    InterPhour[:, 11] = np.clip(InterPhour[:, 11], 0, 120) * windUnit

    ### Wind Bearing
    WindBearingHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        WindBearingHour[:, 0] = NBM_Merged[:, 7]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        WindBearingHour[:, 1] = np.rad2deg(
            np.mod(np.arctan2(HRRR_Merged[:, 7], HRRR_Merged[:, 8]) + np.pi, 2 * np.pi)
        )
    if "gfs" in sourceList:
        WindBearingHour[:, 2] = np.rad2deg(
            np.mod(np.arctan2(GFS_Merged[:, 8], GFS_Merged[:, 9]) + np.pi, 2 * np.pi)
        )
    InterPhour[:, 12] = np.mod(
        np.choose(np.argmin(np.isnan(WindBearingHour), axis=1), WindBearingHour.T), 360
    )

    ### Cloud Cover
    CloudCoverHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        CloudCoverHour[:, 0] = NBM_Merged[:, 9]
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        CloudCoverHour[:, 1] = HRRR_Merged[:, 15]
    if "gfs" in sourceList:
        CloudCoverHour[:, 2] = GFS_Merged[:, 17]
    InterPhour[:, 13] = np.maximum(
        np.choose(np.argmin(np.isnan(CloudCoverHour), axis=1), CloudCoverHour.T) * 0.01,
        0,
    )
    # Clip between 0 and 1
    InterPhour[:, 13] = np.clip(InterPhour[:, 13], 0, 1)

    ### UV Index
    if "gfs" in sourceList:
        InterPhour[:, 14] = np.clip(GFS_Merged[:, 18] * 18.9 * 0.025, 0, 15)

        # Fix small negative zero
        # InterPhour[InterPhour[:, 14]<0, 14] = 0

    ### Visibility
    VisibilityHour = np.full((len(hour_array_grib), 3), np.nan)
    if "nbm" in sourceList:
        VisibilityHour[:, 0] = NBM_Merged[:, 10]

        # Filter out missing visibility values
        VisibilityHour[VisibilityHour[:, 0] < -1, 0] = np.nan
        VisibilityHour[VisibilityHour[:, 0] > 1e6, 0] = np.nan
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        VisibilityHour[:, 1] = HRRR_Merged[:, 1]
    if "gfs" in sourceList:
        VisibilityHour[:, 2] = GFS_Merged[:, 1]

    InterPhour[:, 15] = (
        np.clip(
            np.choose(np.argmin(np.isnan(VisibilityHour), axis=1), VisibilityHour.T),
            0,
            16090,
        )
        * visUnits
    )

    ### Ozone Index
    if "gfs" in sourceList:
        InterPhour[:, 16] = np.clip(GFS_Merged[:, 16], 0, 500)

    ### Precipitation Accumulation
    PrecpAccumHour = np.full((len(hour_array_grib), 4), np.nan)
    # NBM
    if "nbm" in sourceList:
        PrecpAccumHour[:, 0] = NBM_Merged[:, 13]
    # HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        PrecpAccumHour[:, 1] = HRRR_Merged[:, 10]
    # GEFS
    if "gefs" in sourceList:
        PrecpAccumHour[:, 2] = GEFS_Merged[:, 2]
    # GFS
    if "gfs" in sourceList:
        PrecpAccumHour[:, 3] = GFS_Merged[:, 11]

    InterPhour[:, 17] = np.maximum(
        np.choose(np.argmin(np.isnan(PrecpAccumHour), axis=1), PrecpAccumHour.T)
        * prepAccumUnit,
        0,
    )

    # Set accumulation to zero if POP == 0
    InterPhour[InterPhour[:, 3] == 0, 17] = 0

    ### Near Storm Distance
    if "gfs" in sourceList:
        InterPhour[:, 18] = np.maximum(GFS_Merged[:, 19] * visUnits, 0)

    ### Near Storm Direction
    if "gfs" in sourceList:
        InterPhour[:, 19] = GFS_Merged[:, 20]

    # Air quality
    if version >= 2:
        if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
            InterPhour[:, 20] = np.clip(
                HRRR_Merged[:, 16] * 1e9, 0, 200
            )  # Change from kg/m3 to ug/m3
        else:
            InterPhour[:, 20] = -999

    # Fire Index
    if "nbm_fire" in sourceList:
        InterPhour[:, 24] = np.clip(NBM_Fire_Merged[:, 1], 0, 100)

    # Apparent Temperature, Radiative temperature formula
    # https: // github.com / breezy - weather / breezy - weather / discussions / 1085
    # AT = Ta + 0.33 × rh / 100 × 6.105 × exp(17.27 × Ta / (237.7 + Ta)) − 0.70 × ws − 4.00
    e = (
        InterPhour[:, 8]
        * 6.105
        * np.exp(
            17.27 * (InterPhour[:, 5] - 273.15) / (237.7 + (InterPhour[:, 5] - 273.15))
        )
    )
    InterPhour[:, 6] = (
        (InterPhour[:, 5] - 273.15)
        + 0.33 * e
        - 0.70 * (InterPhour[:, 10] / windUnit)
        - 4.00
    ) + 273.15

    ### Feels Like Temperature
    AppTemperatureHour = np.full((len(hour_array_grib), 2), np.nan)
    if "nbm" in sourceList:
        AppTemperatureHour[:, 0] = NBM_Merged[:, 3]

    if "gfs" in sourceList:
        AppTemperatureHour[:, 1] = GFS_Merged[:, 7]

    # Take first non-NaN value
    InterPhour[:, 25] = np.choose(
        np.argmin(np.isnan(AppTemperatureHour), axis=1), AppTemperatureHour.T
    )

    # Clip between -90 and 60
    InterPhour[:, 25] = np.clip(InterPhour[:, 25], -183, 333)

    # Set temperature units
    if tempUnits == 0:
        InterPhour[:, 5:8] = (InterPhour[:, 5:8] - 273.15) * 9 / 5 + 32
        InterPhour[:, 25] = (InterPhour[:, 25] - 273.15) * 9 / 5 + 32
    else:
        InterPhour[:, 5:8] = InterPhour[:, 5:8] - tempUnits
        InterPhour[:, 25] = InterPhour[:, 25] - tempUnits

    # Add a global check for weird values, since nothing should ever be greater than 10000
    # Keep time col
    InterPhourData = InterPhour[:, 1:]
    InterPhourData[InterPhourData > 10000] = np.nan
    InterPhourData[InterPhourData < -1000] = np.nan
    InterPhour[:, 1:] = InterPhourData

    hourList = []
    hourIconList = []
    hourTextList = []

    # Find snow and liqiud precip
    # Set to zero as baseline
    InterPhour[:, 21] = 0
    InterPhour[:, 22] = 0
    InterPhour[:, 23] = 0

    # Accumulations in liquid equivalent
    InterPhour[InterPhour[:, 1] == 4, 21] = InterPhour[
        InterPhour[:, 1] == 4, 17
    ]  # rain

    # 10:1 Snow factor applied here!
    InterPhour[InterPhour[:, 1] == 1, 22] = (
        InterPhour[InterPhour[:, 1] == 1, 17] * 10
    )  # Snow

    InterPhour[((InterPhour[:, 1] == 2) | (InterPhour[:, 1] == 3)), 23] = (
        InterPhour[((InterPhour[:, 1] == 2) | (InterPhour[:, 1] == 3)), 17] * 1
    )  # Ice

    # Rain
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepRain = InterPhour[:, 21].copy()
    # Everything that isn't the current day
    dayZeroPrepRain[hourlyDayIndex != 0] = 0
    # Everything after the request time
    dayZeroPrepRain[int(baseTimeOffset) :] = 0

    # Snow
    # Calculate prep accumulation for current day before zeroing
    dayZeroPrepSnow = InterPhour[:, 22].copy()
    # Everything that isn't the current day
    dayZeroPrepSnow[hourlyDayIndex != 0] = 0
    # Everything after the request time
    dayZeroPrepSnow[int(baseTimeOffset) :] = 0

    # Sleet
    # Calculate prep accumilation for current day before zeroing
    dayZeroPrepSleet = InterPhour[:, 23].copy()
    # Everything that isn't the current day
    dayZeroPrepSleet[hourlyDayIndex != 0] = 0
    # Everything after the request time
    dayZeroPrepSleet[int(baseTimeOffset) :] = 0

    # Accumulations in liquid equivalent
    dayZeroRain = dayZeroPrepRain.sum().round(4)  # rain
    dayZeroSnow = dayZeroPrepSnow.sum().round(4)  # Snow
    dayZeroIce = dayZeroPrepSleet.sum().round(4)  # Ice

    # Zero prep intensity and accum before forecast time
    InterPhour[0 : int(baseTimeOffset), 2] = 0
    InterPhour[0 : int(baseTimeOffset), 17] = 0
    InterPhour[0 : int(baseTimeOffset), 21] = 0
    InterPhour[0 : int(baseTimeOffset), 22] = 0
    InterPhour[0 : int(baseTimeOffset), 23] = 0

    # Zero prep prob before forecast time
    InterPhour[0 : int(baseTimeOffset), 3] = 0

    # Assign pfactors for rain and snow for intensity
    pFacHour = np.zeros((len(hour_array)))
    pFacHour[
        ((InterPhour[:, 1] == 4) | (InterPhour[:, 1] == 2) | (InterPhour[:, 1] == 3))
    ] = 1  # Rain, Ice
    # NOTE, this means that intensity is always liquid water equivalent.
    pFacHour[(InterPhour[:, 1] == 1)] = 1  # Snow

    InterPhour[:, 2] = InterPhour[:, 2] * pFacHour

    # pTypeMap = {0: 'none', 1: 'snow', 2: 'sleet', 3: 'sleet', 4: 'rain'}
    pTypeMap = np.array(["none", "snow", "sleet", "sleet", "rain"])
    pTextMap = np.array(["None", "Snow", "Sleet", "Sleet", "Rain"])
    PTypeHour = pTypeMap[InterPhour[:, 1].astype(int)]
    PTextHour = pTextMap[InterPhour[:, 1].astype(int)]

    # Round all to 2 except precipitations
    InterPhour[:, 3] = InterPhour[:, 3].round(2)
    InterPhour[:, 5:17] = InterPhour[:, 5:17].round(2)
    InterPhour[:, 18:21] = InterPhour[:, 18:21].round(2)
    InterPhour[:, 24:26] = InterPhour[:, 24:26].round(2)

    # Round to 4
    InterPhour[:, 1:3] = InterPhour[:, 1:3].round(4)
    InterPhour[:, 4:5] = InterPhour[:, 4:5].round(4)
    InterPhour[:, 17] = InterPhour[:, 17].round(4)
    InterPhour[:, 21:24] = InterPhour[:, 21:24].round(4)

    # Fix very small neg from interp to solve -0
    InterPhour[((InterPhour > -0.001) & (InterPhour < 0.001))] = 0

    # Replace NaN with -999 for json
    InterPhour[np.isnan(InterPhour)] = -999

    # Timing Check
    if TIMING:
        print("Hourly Loop start")
        print(datetime.datetime.utcnow() - T_Start)

    for idx in range(int(baseTimeOffset), hourly_hours + int(baseTimeOffset)):
        # Check if day or night
        if hour_array_grib[idx] < InterSday[hourlyDayIndex[idx], 17]:
            isDay = False
        elif (
            hour_array_grib[idx] >= InterSday[hourlyDayIndex[idx], 17]
            and hour_array_grib[idx] <= InterSday[hourlyDayIndex[idx], 18]
        ):
            isDay = True
        elif hour_array_grib[idx] > InterSday[hourlyDayIndex[idx], 18]:
            isDay = False

        # Set text
        if InterPhour[idx, 3] >= 0.3 and (
            ((InterPhour[idx, 21] + InterPhour[idx, 23]) > (0.02 * prepAccumUnit))
            or (InterPhour[idx, 22] > (0.02 * prepAccumUnit))
        ):
            # If more than 30% chance of precip at any point throughout the day, then the icon for whatever is happening
            # Thresholds set in mm
            hourIcon = PTypeHour[idx]
            hourText = PTextHour[idx]
        # If visibility <1000 and during the day
        # elif InterPhour[idx,14]<1000 and (hour_array_grib[idx]>InterPday[dCount,16] and hour_array_grib[idx]<InterPday[dCount,17]):
        elif InterPhour[idx, 15] < (1000 * visUnits):
            hourIcon = "fog"
            hourText = "Fog"
        # If wind is greater than 10 m/s
        elif InterPhour[idx, 10] > (10 * windUnit):
            hourIcon = "wind"
            hourText = "Windy"
        elif InterPhour[idx, 13] > 0.75:
            hourIcon = "cloudy"
            hourText = "Cloudy"
        elif InterPhour[idx, 13] > 0.375:
            hourText = "Partly Cloudy"

            if hour_array_grib[idx] < InterSday[hourlyDayIndex[idx], 17]:
                # Before sunrise
                hourIcon = "partly-cloudy-night"
            elif (
                hour_array_grib[idx] >= InterSday[hourlyDayIndex[idx], 17]
                and hour_array_grib[idx] <= InterSday[hourlyDayIndex[idx], 18]
            ):
                # After sunrise before sunset
                hourIcon = "partly-cloudy-day"
            elif hour_array_grib[idx] > InterSday[hourlyDayIndex[idx], 18]:
                # After sunset
                hourIcon = "partly-cloudy-night"
        else:
            hourText = "Clear"

            if hour_array_grib[idx] < InterSday[hourlyDayIndex[idx], 17]:
                # Before sunrise
                hourIcon = "clear-night"
            elif (
                hour_array_grib[idx] >= InterSday[hourlyDayIndex[idx], 17]
                and hour_array_grib[idx] <= InterSday[hourlyDayIndex[idx], 18]
            ):
                # After sunrise before sunset
                hourIcon = "clear-day"
            elif hour_array_grib[idx] > InterSday[hourlyDayIndex[idx], 18]:
                # After sunset
                hourIcon = "clear-night"

        if timeMachine and not tmExtra:
            hourItem = {
                "time": int(hour_array_grib[idx]),
                "summary": hourText,
                "icon": hourIcon,
                "precipIntensity": InterPhour[idx, 2],
                "precipAccumulation": InterPhour[idx, 21]
                + InterPhour[idx, 22]
                + InterPhour[idx, 23],
                "precipType": PTypeHour[idx],
                "temperature": InterPhour[idx, 5],
                "apparentTemperature": InterPhour[idx, 6],
                "dewPoint": InterPhour[idx, 7],
                "pressure": InterPhour[idx, 9],
                "windSpeed": InterPhour[idx, 10],
                "windGust": InterPhour[idx, 11],
                "windBearing": InterPhour[idx, 12],
                "cloudCover": InterPhour[idx, 13],
                "snowAccumulation": InterPhour[idx, 22],
            }

        elif version >= 2:
            hourItem = {
                "time": int(hour_array_grib[idx]),
                "summary": hourText,
                "icon": hourIcon,
                "precipIntensity": InterPhour[idx, 2],
                "precipProbability": InterPhour[idx, 3],
                "precipIntensityError": InterPhour[idx, 4],
                "precipAccumulation": InterPhour[idx, 21]
                + InterPhour[idx, 22]
                + InterPhour[idx, 23],
                "precipType": PTypeHour[idx],
                "temperature": InterPhour[idx, 5],
                "apparentTemperature": InterPhour[idx, 6],
                "dewPoint": InterPhour[idx, 7],
                "humidity": InterPhour[idx, 8],
                "pressure": InterPhour[idx, 9],
                "windSpeed": InterPhour[idx, 10],
                "windGust": InterPhour[idx, 11],
                "windBearing": InterPhour[idx, 12],
                "cloudCover": InterPhour[idx, 13],
                "uvIndex": InterPhour[idx, 14],
                "visibility": InterPhour[idx, 15],
                "ozone": InterPhour[idx, 16],
                "smoke": InterPhour[idx, 20],
                "liquidAccumulation": InterPhour[idx, 21],
                "snowAccumulation": InterPhour[idx, 22],
                "iceAccumulation": InterPhour[idx, 23],
                "nearestStormDistance": InterPhour[idx, 18],
                "nearestStormBearing": InterPhour[idx, 19],
                "fireIndex": InterPhour[idx, 24],
                "feelsLike": InterPhour[idx, 25],
            }

        else:
            hourItem = {
                "time": int(hour_array_grib[idx]),
                "summary": hourText,
                "icon": hourIcon,
                "precipIntensity": InterPhour[idx, 2],
                "precipProbability": InterPhour[idx, 3],
                "precipIntensityError": InterPhour[idx, 4],
                "precipAccumulation": InterPhour[idx, 21]
                + InterPhour[idx, 22]
                + InterPhour[idx, 23],
                "precipType": PTypeHour[idx],
                "temperature": InterPhour[idx, 5],
                "apparentTemperature": InterPhour[idx, 6],
                "dewPoint": InterPhour[idx, 7],
                "humidity": InterPhour[idx, 8],
                "pressure": InterPhour[idx, 9],
                "windSpeed": InterPhour[idx, 10],
                "windGust": InterPhour[idx, 11],
                "windBearing": InterPhour[idx, 12],
                "cloudCover": InterPhour[idx, 13],
                "uvIndex": InterPhour[idx, 14],
                "visibility": InterPhour[idx, 15],
                "ozone": InterPhour[idx, 16],
            }

        try:
            hourText, hourIcon = calculate_text(
                hourItem,
                prepAccumUnit,
                visUnits,
                windUnit,
                tempUnits,
                isDay,
                InterPhour[idx, 21],
                InterPhour[idx, 22],
                InterPhour[idx, 23],
                "hour",
                mode="title",
            )
            hourItem["summary"] = translation.translate(hourText)
            hourItem["icon"] = hourIcon
        except Exception as e:
            print("TEXT GEN ERROR:")
            print(e)

        hourList.append(hourItem)

        hourIconList.append(hourIcon)
        hourTextList.append(hourItem["summary"])

    # Daily calculations #################################################
    # Timing Check
    if TIMING:
        print("Daily start")
        print(datetime.datetime.utcnow() - T_Start)

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
    maxPchanceDay = []
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
        # maxPchanceDay.append(stats.mode(filtered_data[:,1], axis=0)[0])

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

    # First, add a fallback if any precipitation is expected
    maxPchanceDay[((maxPchanceDay == 0) & (InterPdaySum[:, 21] > 0))] = 4
    maxPchanceDay[((maxPchanceDay == 0) & (InterPdaySum[:, 22] > 0))] = 1
    maxPchanceDay[((maxPchanceDay == 0) & (InterPdaySum[:, 23] > 0))] = 2

    # Then, if more than 10 mm of rain is forecast, then rain
    maxPchanceDay[InterPdaySum[:, 21] > (10 * prepAccumUnit)] = 4

    # If more than 5 mm of snow is forecast, then snow
    maxPchanceDay[InterPdaySum[:, 22] > (5 * prepAccumUnit)] = 1

    # Else, if more than 1 mm of ice is forecast, then ice
    maxPchanceDay[InterPdaySum[:, 23] > (1 * prepAccumUnit)] = 2

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
    InterPdayMax[:, 3] = InterPdayMax[:, 3].round(2)
    InterPdayMax[:, 5:18] = InterPdayMax[:, 5:18].round(2)
    InterPdayMax[:, 24] = InterPdayMax[:, 24].round(2)

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
        print(datetime.datetime.utcnow() - T_Start)

    for idx in range(0, daily_days):
        if InterPdayMax4am[idx, 3] > 0.3 and (
            (
                (InterPdaySum4am[idx, 21] + InterPdaySum4am[idx, 23])
                > (1 * prepAccumUnit)
            )
            or (InterPdaySum4am[idx, 22] > (10 * prepAccumUnit))
        ):
            # If more than 30% chance of precip at any point throughout the day, and either more than 1 mm of rain or 5 mm of snow
            # Thresholds set in mm
            dayIcon = PTypeDay[idx]
            dayText = PTextDay[idx]

            # Fallback if no ptype for some reason. This should never occur though
            if dayIcon == "none":
                if tempUnits == 0:
                    tempThresh = 32
                else:
                    tempThresh = 0

                if InterPday[idx, 5] > tempThresh:
                    dayIcon = "rain"
                    dayText = "Rain"
                else:
                    dayIcon = "snow"
                    dayText = "Snow"

        elif InterPday4am[idx, 15] < (1000 * visUnits):
            dayIcon = "fog"
            dayText = "Fog"
        elif InterPday4am[idx, 10] > (10 * windUnit):
            dayIcon = "wind"
            dayText = "Windy"
        elif InterPday4am[idx, 13] > 0.75:
            dayIcon = "cloudy"
            dayText = "Cloudy"
        elif InterPday4am[idx, 13] > 0.375:
            dayIcon = "partly-cloudy-day"
            dayText = "Partly Cloudy"
        else:
            dayIcon = "clear-day"
            dayText = "Clear"

        # Temperature High is daytime high, so 6 am to 6 pm
        # First index is 6 am, then index 2
        # Nightime is index 1, 3, etc.
        if timeMachine and not tmExtra:
            dayObject = {
                "time": int(day_array_grib[idx]),
                "summary": dayText,
                "icon": dayIcon,
                "sunriseTime": int(InterSday[idx, 17]),
                "sunsetTime": int(InterSday[idx, 18]),
                "moonPhase": InterSday[idx, 19].round(2),
                "precipIntensity": InterPday[idx, 2],
                "precipIntensityMax": InterPdayMax[idx, 2],
                "precipIntensityMaxTime": int(InterPdayMaxTime[idx, 2]),
                "precipAccumulation": round(
                    InterPdaySum[idx, 21]
                    + InterPdaySum[idx, 22]
                    + InterPdaySum[idx, 23],
                    4,
                ),
                "precipType": PTypeDay[idx],
                "temperatureHigh": InterPdayHigh[idx, 5],
                "temperatureHighTime": int(InterPdayHighTime[idx, 5]),
                "temperatureLow": InterPdayLow[idx, 5],
                "temperatureLowTime": int(InterPdayLowTime[idx, 5]),
                "apparentTemperatureHigh": InterPdayHigh[idx, 6],
                "apparentTemperatureHighTime": int(InterPdayHighTime[idx, 6]),
                "apparentTemperatureLow": InterPdayLow[idx, 6],
                "apparentTemperatureLowTime": int(InterPdayLowTime[idx, 6]),
                "dewPoint": InterPday[idx, 7],
                "pressure": InterPday[idx, 9],
                "windSpeed": InterPday[idx, 10],
                "windGust": InterPday[idx, 11],
                "windGustTime": int(InterPdayMaxTime[idx, 11]),
                "windBearing": InterPday[idx, 12],
                "cloudCover": InterPday[idx, 13],
                "temperatureMin": InterPdayMin[idx, 5],
                "temperatureMinTime": int(InterPdayMinTime[idx, 5]),
                "temperatureMax": InterPdayMax[idx, 5],
                "temperatureMaxTime": int(InterPdayMaxTime[idx, 5]),
                "apparentTemperatureMin": InterPdayMin[idx, 6],
                "apparentTemperatureMinTime": int(InterPdayMinTime[idx, 6]),
                "apparentTemperatureMax": InterPdayMax[idx, 6],
                "apparentTemperatureMaxTime": int(InterPdayMaxTime[idx, 6]),
                "snowAccumulation": InterPdaySum[idx, 22],
            }
        else:
            if version >= 2:
                dayObject = {
                    "time": int(day_array_grib[idx]),
                    "summary": dayText,
                    "icon": dayIcon,
                    "dawnTime": int(InterSday[idx, 15]),
                    "sunriseTime": int(InterSday[idx, 17]),
                    "sunsetTime": int(InterSday[idx, 18]),
                    "duskTime": int(InterSday[idx, 16]),
                    "moonPhase": InterSday[idx, 19].round(2),
                    "precipIntensity": InterPday[idx, 2],
                    "precipIntensityMax": InterPdayMax[idx, 2],
                    "precipIntensityMaxTime": int(InterPdayMaxTime[idx, 2]),
                    "precipProbability": InterPdayMax[idx, 3],
                    "precipAccumulation": round(
                        InterPdaySum[idx, 21]
                        + InterPdaySum[idx, 22]
                        + InterPdaySum[idx, 23],
                        4,
                    ),
                    "precipType": PTypeDay[idx],
                    "temperatureHigh": InterPdayHigh[idx, 5],
                    "temperatureHighTime": int(InterPdayHighTime[idx, 5]),
                    "temperatureLow": InterPdayLow[idx, 5],
                    "temperatureLowTime": int(InterPdayLowTime[idx, 5]),
                    "apparentTemperatureHigh": InterPdayHigh[idx, 6],
                    "apparentTemperatureHighTime": int(InterPdayHighTime[idx, 6]),
                    "apparentTemperatureLow": InterPdayLow[idx, 6],
                    "apparentTemperatureLowTime": int(InterPdayLowTime[idx, 6]),
                    "dewPoint": InterPday[idx, 7],
                    "humidity": InterPday[idx, 8],
                    "pressure": InterPday[idx, 9],
                    "windSpeed": InterPday[idx, 10],
                    "windGust": InterPday[idx, 11],
                    "windGustTime": int(InterPdayMaxTime[idx, 11]),
                    "windBearing": InterPday[idx, 12],
                    "cloudCover": InterPday[idx, 13],
                    "uvIndex": InterPdayMax[idx, 14],
                    "uvIndexTime": int(InterPdayMaxTime[idx, 14]),
                    "visibility": InterPday[idx, 15],
                    "temperatureMin": InterPdayMin[idx, 5],
                    "temperatureMinTime": int(InterPdayMinTime[idx, 5]),
                    "temperatureMax": InterPdayMax[idx, 5],
                    "temperatureMaxTime": int(InterPdayMaxTime[idx, 5]),
                    "apparentTemperatureMin": InterPdayMin[idx, 6],
                    "apparentTemperatureMinTime": int(InterPdayMinTime[idx, 6]),
                    "apparentTemperatureMax": InterPdayMax[idx, 6],
                    "apparentTemperatureMaxTime": int(InterPdayMaxTime[idx, 6]),
                    "smokeMax": InterPdayMax[idx, 20],
                    "smokeMaxTime": int(InterPdayMaxTime[idx, 20]),
                    "liquidAccumulation": InterPdaySum[idx, 21],
                    "snowAccumulation": InterPdaySum[idx, 22],
                    "iceAccumulation": InterPdaySum[idx, 23],
                    "fireIndexMax": InterPdayMax[idx, 24],
                    "fireIndexMaxTime": InterPdayMaxTime[idx, 24],
                }
            else:
                dayObject = {
                    "time": int(day_array_grib[idx]),
                    "summary": dayText,
                    "icon": dayIcon,
                    "sunriseTime": int(InterSday[idx, 17]),
                    "sunsetTime": int(InterSday[idx, 18]),
                    "moonPhase": InterSday[idx, 19].round(2),
                    "precipIntensity": InterPday[idx, 2],
                    "precipIntensityMax": InterPdayMax[idx, 2],
                    "precipIntensityMaxTime": int(InterPdayMaxTime[idx, 2]),
                    "precipProbability": InterPdayMax[idx, 3],
                    "precipAccumulation": round(
                        InterPdaySum[idx, 21]
                        + InterPdaySum[idx, 22]
                        + InterPdaySum[idx, 23],
                        4,
                    ),
                    "precipType": PTypeDay[idx],
                    "temperatureHigh": InterPdayHigh[idx, 5],
                    "temperatureHighTime": int(InterPdayHighTime[idx, 5]),
                    "temperatureLow": InterPdayLow[idx, 5],
                    "temperatureLowTime": int(InterPdayLowTime[idx, 5]),
                    "apparentTemperatureHigh": InterPdayHigh[idx, 6],
                    "apparentTemperatureHighTime": int(InterPdayHighTime[idx, 6]),
                    "apparentTemperatureLow": InterPdayLow[idx, 6],
                    "apparentTemperatureLowTime": int(InterPdayLowTime[idx, 6]),
                    "dewPoint": InterPday[idx, 7],
                    "humidity": InterPday[idx, 8],
                    "pressure": InterPday[idx, 9],
                    "windSpeed": InterPday[idx, 10],
                    "windGust": InterPday[idx, 11],
                    "windGustTime": int(InterPdayMaxTime[idx, 11]),
                    "windBearing": InterPday[idx, 12],
                    "cloudCover": InterPday[idx, 13],
                    "uvIndex": InterPdayMax[idx, 14],
                    "uvIndexTime": int(InterPdayMaxTime[idx, 14]),
                    "visibility": InterPday[idx, 15],
                    "temperatureMin": InterPdayMin[idx, 5],
                    "temperatureMinTime": int(InterPdayMinTime[idx, 5]),
                    "temperatureMax": InterPdayMax[idx, 5],
                    "temperatureMaxTime": int(InterPdayMaxTime[idx, 5]),
                    "apparentTemperatureMin": InterPdayMin[idx, 6],
                    "apparentTemperatureMinTime": int(InterPdayMinTime[idx, 6]),
                    "apparentTemperatureMax": InterPdayMax[idx, 6],
                    "apparentTemperatureMaxTime": int(InterPdayMaxTime[idx, 6]),
                }

        try:
            # Update the text
            dayText, dayIcon = calculate_simple_day_text(
                dayObject,
                prepAccumUnit,
                visUnits,
                windUnit,
                tempUnits,
                True,
                InterPdaySum[idx, 21],
                InterPdaySum[idx, 22],
                InterPdaySum[idx, 23],
                "day",
            )
            dayObject["summary"] = translation.translate(dayText)
            dayObject["icon"] = dayIcon
        except Exception as e:
            print("TEXT GEN ERROR:")
            print(e)

        dayList.append(dayObject)

        dayTextList.append(dayObject["summary"])
        dayIconList.append(dayIcon)

    # Timing Check
    if TIMING:
        print("Alert Start")
        print(datetime.datetime.utcnow() - T_Start)

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

    except Exception as error:
        print("An Alert error occurred:", error)

    # Timing Check
    if TIMING:
        print("Current Start")
        print(datetime.datetime.utcnow() - T_Start)

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

    InterPcurrent = np.zeros(shape=21)  # Time, Intensity,Probability
    InterPcurrent[0] = int(minute_array_grib[0])

    # Get prep probability, type, and intensity from minutely
    InterPcurrent[1] = InterPminute[0, 1]
    InterPcurrent[2] = InterPminute[0, 2]  # "precipProbability"
    InterPcurrent[3] = InterPminute[0, 3]  # "precipIntensityError"

    # Temperature from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[4] = hrrrSubHInterpolation[0, 3]
    elif "nbm" in sourceList:
        InterPcurrent[4] = (
            NBM_Merged[currentIDX_hrrrh - 1, 2] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 2] * interpFac2
        )
    else:
        InterPcurrent[4] = (
            GFS_Merged[currentIDX_hrrrh - 1, 4] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 4] * interpFac2
        )

    # Clip between -90 and 60
    InterPcurrent[4] = np.clip(InterPcurrent[4], -183, 333)

    # Dewpoint from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[6] = hrrrSubHInterpolation[0, 4]
    elif "nbm" in sourceList:
        InterPcurrent[6] = (
            NBM_Merged[currentIDX_hrrrh - 1, 4] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 4] * interpFac2
        )
    else:
        InterPcurrent[6] = (
            GFS_Merged[currentIDX_hrrrh - 1, 5] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 5] * interpFac2
        )

    # Clip between -90 and 60
    InterPcurrent[6] = np.clip(InterPcurrent[6], -183, 333)

    # humidity, NBM then HRRR, then GFS
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[7] = (
            HRRR_Merged[currentIDX_hrrrh - 1, 6] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, 6] * interpFac2
        ) * humidUnit
    elif "nbm" in sourceList:
        InterPcurrent[7] = (
            NBM_Merged[currentIDX_hrrrh - 1, 5] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 5] * interpFac2
        ) * humidUnit
    else:
        InterPcurrent[7] = (
            GFS_Merged[currentIDX_hrrrh - 1, 6] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 6] * interpFac2
        ) * humidUnit

    # Clip between 0 and 1
    InterPcurrent[7] = np.clip(InterPcurrent[7], 0, 1)

    # Pressure from HRRR, then GFS
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[8] = (
            HRRR_Merged[currentIDX_hrrrh - 1, 3] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, 3] * interpFac2
        )
    else:
        InterPcurrent[8] = (
            GFS_Merged[currentIDX_hrrrh - 1, 3] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 3] * interpFac2
        )

    # Clip between 800 and 1100
    InterPcurrent[8] = np.clip(InterPcurrent[8], 80000, 110000) * pressUnits

    # WindSpeed from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[9] = math.sqrt(
            hrrrSubHInterpolation[0, 5] ** 2 + hrrrSubHInterpolation[0, 6] ** 2
        )
    elif "nbm" in sourceList:
        InterPcurrent[9] = (
            NBM_Merged[currentIDX_hrrrh - 1, 6] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 6] * interpFac2
        )
    else:
        InterPcurrent[9] = math.sqrt(
            (
                GFS_Merged[currentIDX_hrrrh - 1, 8] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, 8] * interpFac2
            )
            ** 2
            + (
                GFS_Merged[currentIDX_hrrrh - 1, 9] * interpFac1
                + GFS_Merged[currentIDX_hrrrh, 9] * interpFac2
            )
            ** 2
        )
    InterPcurrent[9] = np.clip(InterPcurrent[9], 0, 120) * windUnit

    # Guest from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[10] = hrrrSubHInterpolation[0, 1]
    elif "nbm" in sourceList:
        InterPcurrent[10] = (
            NBM_Merged[currentIDX_hrrrh - 1, 1] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 1] * interpFac2
        )
    else:
        InterPcurrent[10] = (
            GFS_Merged[currentIDX_hrrrh - 1, 2] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 2] * interpFac2
        )

    # Clip between 0 and 400
    InterPcurrent[10] = np.clip(InterPcurrent[10], 0, 120) * windUnit

    # WindDir from subH, then NBM, the GFS
    if "hrrrsubh" in sourceList:
        InterPcurrent[11] = np.rad2deg(
            np.mod(
                np.arctan2(hrrrSubHInterpolation[0, 5], hrrrSubHInterpolation[0, 6])
                + np.pi,
                2 * np.pi,
            )
        )
    elif "nbm" in sourceList:
        InterPcurrent[11] = NBM_Merged[currentIDX_hrrrh - 1, 7]
    else:
        InterPcurrent[11] = np.rad2deg(
            np.mod(
                np.arctan2(
                    GFS_Merged[currentIDX_hrrrh, 8], GFS_Merged[currentIDX_hrrrh, 9]
                )
                + np.pi,
                2 * np.pi,
            )
        )

    # Cloud, NBM then HRRR, then GFS
    if "nbm" in sourceList:
        InterPcurrent[12] = (
            NBM_Merged[currentIDX_hrrrh - 1, 9] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 9] * interpFac2
        ) * 0.01
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[12] = (
            HRRR_Merged[currentIDX_hrrrh - 1, 15] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, 15] * interpFac2
        ) * 0.01
    else:
        InterPcurrent[12] = (
            GFS_Merged[currentIDX_hrrrh - 1, 17] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 17] * interpFac2
        ) * 0.01

    # Clip
    InterPcurrent[12] = np.clip(InterPcurrent[12], 0, 15)

    # UV Index from subH, then NBM, the GFS
    InterPcurrent[13] = np.clip(
        (
            GFS_Merged[currentIDX_hrrrh - 1, 18] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 18] * interpFac2
        )
        * 18.9
        * 0.025
    )

    # VIS, NBM then HRRR, then GFS
    if "nbm" in sourceList:
        InterPcurrent[14] = (
            NBM_Merged[currentIDX_hrrrh - 1, 10] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 10] * interpFac2
        )
    elif ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[14] = (
            HRRR_Merged[currentIDX_hrrrh - 1, 1] * interpFac1
            + HRRR_Merged[currentIDX_hrrrh, 1] * interpFac2
        )
    else:
        InterPcurrent[14] = (
            GFS_Merged[currentIDX_hrrrh - 1, 1] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 1] * interpFac2
        )

    InterPcurrent[14] = np.clip(InterPcurrent[14], 0, 16090) * visUnits

    # Ozone from GFS
    InterPcurrent[15] = np.clip(
        GFS_Merged[currentIDX_hrrrh - 1, 16] * interpFac1
        + GFS_Merged[currentIDX_hrrrh, 16] * interpFac2,
        0,
        500,
    )  # "   "ozone"

    # Storm Distance from GFS
    InterPcurrent[16] = np.maximum(
        (
            GFS_Merged[currentIDX_hrrrh - 1, 19] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 19] * interpFac2
        )
        * visUnits,
        0,
    )

    # Storm Bearing from GFS
    InterPcurrent[17] = GFS_Merged[currentIDX_hrrrh, 20]

    # Smoke from HRRR
    if ("hrrr_0-18" in sourceList) and ("hrrr_18-48" in sourceList):
        InterPcurrent[18] = np.clip(
            (
                (
                    HRRR_Merged[currentIDX_hrrrh - 1, 16] * interpFac1
                    + HRRR_Merged[currentIDX_hrrrh, 16] * interpFac2
                )
                * 1e9
            ),
            0,
            200,
        )
    else:
        InterPcurrent[18] = -999

    # Apparent Temperature, Radiative temperature formula
    # https: // github.com / breezy - weather / breezy - weather / discussions / 1085
    # AT = Ta + 0.33 × (rh / 100 × 6.105 × exp(17.27 × Ta / (237.7 + Ta))) − 0.70 × ws − 4.00

    eCurrent = (
        InterPcurrent[7]
        * 6.105
        * np.exp(
            17.27 * (InterPcurrent[4] - 273.15) / (237.7 + (InterPcurrent[4] - 273.15))
        )
    )

    InterPcurrent[5] = np.clip(
        (
            (
                (InterPcurrent[4] - 273.15)
                + 0.33 * eCurrent
                - 0.70 * (InterPcurrent[9] / windUnit)
                - 4.00
            )
            + 273.15
        ),
        -183,
        333,
    )

    # Where Ta is the ambient temperature in °C
    # e is the water vapor pressure in hPa
    # ws is the wind speed in m/s
    # Q is the solar radiation per unit area of body surface in w/m²
    if "nbm" in sourceList:
        InterPcurrent[20] = (
            NBM_Merged[currentIDX_hrrrh - 1, 3] * interpFac1
            + NBM_Merged[currentIDX_hrrrh, 3] * interpFac2
        )
    else:
        InterPcurrent[20] = (
            GFS_Merged[currentIDX_hrrrh - 1, 7] * interpFac1
            + GFS_Merged[currentIDX_hrrrh, 7] * interpFac2
        )

    # Clip
    InterPcurrent[20] = np.clip(InterPcurrent[20], -183, 333)

    # Fire index from NBM Fire
    if "nbm_fire" in sourceList:
        InterPcurrent[19] = np.clip(
            (
                NBM_Fire_Merged[currentIDX_hrrrh - 1, 1] * interpFac1
                + NBM_Fire_Merged[currentIDX_hrrrh, 1] * interpFac2
            ),
            0,
            100,
        )
    else:
        InterPcurrent[19] = -999

    # Put temperature into units
    if tempUnits == 0:
        InterPcurrent[4] = (InterPcurrent[4] - 273.15) * 9 / 5 + 32  # "temperature"
        InterPcurrent[5] = (
            InterPcurrent[5] - 273.15
        ) * 9 / 5 + 32  # "apparentTemperature"
        InterPcurrent[6] = (InterPcurrent[6] - 273.15) * 9 / 5 + 32  # "dewPoint"
        InterPcurrent[20] = (InterPcurrent[20] - 273.15) * 9 / 5 + 32  # "FeelsLike"

    else:
        InterPcurrent[4] = InterPcurrent[4] - tempUnits  # "temperature"
        InterPcurrent[5] = InterPcurrent[5] - tempUnits  # "apparentTemperature"
        InterPcurrent[6] = InterPcurrent[6] - tempUnits  # "dewPoint"
        InterPcurrent[20] = InterPcurrent[20] - tempUnits  # "FeelsLike"

    if ((minuteDict[0]["precipIntensity"]) > (0.02 * prepIntensityUnit)) & (
        minuteDict[0]["precipType"] is not None
    ):
        # If more than 25% chance of precip, then the icon for whatever is happening, so long as the icon exists
        cIcon = minuteDict[0]["precipType"]
        cText = minuteDict[0]["precipType"][0].upper() + minuteDict[0]["precipType"][1:]

        # Because soemtimes there's precipitation not no type, don't use an icon in those cases

    # If visibility <1km and during the day
    # elif InterPcurrent[14]<1000 and (InterPcurrent[0]>InterPday[0,16] and InterPcurrent[0]<InterPday[0,17]):
    elif InterPcurrent[14] < (1000 * visUnits):
        cIcon = "fog"
        cText = "Fog"
    elif InterPcurrent[9] > (10 * windUnit):
        cIcon = "wind"
        cText = "Windy"
    elif InterPcurrent[12] > 0.75:
        cIcon = "cloudy"
        cText = "Cloudy"
    elif InterPcurrent[12] > 0.375:
        cText = "Partly Cloudy"

        if InterPcurrent[0] < InterSday[0, 17]:
            # Before sunrise
            cIcon = "partly-cloudy-night"
        elif (
            InterPcurrent[0] > InterSday[0, 17] and InterPcurrent[0] < InterSday[0, 18]
        ):
            # After sunrise before sunset
            cIcon = "partly-cloudy-day"
        elif InterPcurrent[0] > InterSday[0, 18]:
            # After sunset
            cIcon = "partly-cloudy-night"
    else:
        cText = "Clear"
        if InterPcurrent[0] < InterSday[0, 17]:
            # Before sunrise
            cIcon = "clear-night"
        elif (
            InterPcurrent[0] > InterSday[0, 17] and InterPcurrent[0] < InterSday[0, 18]
        ):
            # After sunrise before sunset
            cIcon = "clear-day"
        elif InterPcurrent[0] > InterSday[0, 18]:
            # After sunset
            cIcon = "clear-night"

    # Timing Check
    if TIMING:
        print("Object Start")
        print(datetime.datetime.utcnow() - T_Start)

    InterPcurrent = InterPcurrent.round(2)
    InterPcurrent[np.isnan(InterPcurrent)] = -999

    # Fix small neg zero
    InterPcurrent[((InterPcurrent > -0.01) & (InterPcurrent < 0.01))] = 0

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

        if (not timeMachine) or (tmExtra):
            returnOBJ["currently"]["nearestStormDistance"] = InterPcurrent[16]
            returnOBJ["currently"]["nearestStormBearing"] = InterPcurrent[17].round()
        returnOBJ["currently"]["precipIntensity"] = minuteDict[0]["precipIntensity"]

        if (not timeMachine) or (tmExtra):
            returnOBJ["currently"]["precipProbability"] = minuteDict[0][
                "precipProbability"
            ]
            returnOBJ["currently"]["precipIntensityError"] = minuteDict[0][
                "precipIntensityError"
            ]
        returnOBJ["currently"]["precipType"] = minuteDict[0]["precipType"]
        returnOBJ["currently"]["temperature"] = InterPcurrent[4]
        returnOBJ["currently"]["apparentTemperature"] = InterPcurrent[5]
        returnOBJ["currently"]["dewPoint"] = InterPcurrent[6]

        if (not timeMachine) or (tmExtra):
            returnOBJ["currently"]["humidity"] = InterPcurrent[7]
        returnOBJ["currently"]["pressure"] = InterPcurrent[8]
        returnOBJ["currently"]["windSpeed"] = InterPcurrent[9]
        returnOBJ["currently"]["windGust"] = InterPcurrent[10]
        returnOBJ["currently"]["windBearing"] = np.mod(InterPcurrent[11], 360).round()
        returnOBJ["currently"]["cloudCover"] = InterPcurrent[12]

        if (not timeMachine) or (tmExtra):
            returnOBJ["currently"]["uvIndex"] = InterPcurrent[13]
            returnOBJ["currently"]["visibility"] = InterPcurrent[14]
            returnOBJ["currently"]["ozone"] = InterPcurrent[15]

        if version >= 2:
            returnOBJ["currently"]["smoke"] = InterPcurrent[18]  # kg/m3 to ug/m3
            returnOBJ["currently"]["fireIndex"] = InterPcurrent[19]
            returnOBJ["currently"]["feelsLike"] = InterPcurrent[20]
            returnOBJ["currently"]["currentDayIce"] = dayZeroIce
            returnOBJ["currently"]["currentDayLiquid"] = dayZeroRain
            returnOBJ["currently"]["currentDaySnow"] = dayZeroSnow

        # Update the text
        if InterPcurrent[0] < InterSday[0, 17]:
            # Before sunrise
            currentDay = False
        elif (
            InterPcurrent[0] > InterSday[0, 17] and InterPcurrent[0] < InterSday[0, 18]
        ):
            # After sunrise before sunset
            currentDay = True
        elif InterPcurrent[0] > InterSday[0, 18]:
            # After sunset
            currentDay = False

        # Convert intensity to accumulation based on type
        currnetRainAccum = 0
        currnetSnowAccum = 0
        currnetIceAccum = 0

        if minuteDict[0]["precipType"] == "rain":
            currnetRainAccum = (
                minuteDict[0]["precipIntensity"] / prepIntensityUnit * prepAccumUnit
            )
        elif minuteDict[0]["precipType"] == "snow":
            currnetSnowAccum = (
                minuteDict[0]["precipIntensity"] / prepIntensityUnit * prepAccumUnit
            ) * 10  # 1:10 since intensity is in liquid water equivalent
        elif minuteDict[0]["precipType"] == "sleet":
            currnetIceAccum = (
                minuteDict[0]["precipIntensity"] / prepIntensityUnit * prepAccumUnit
            )

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
                mode="title",
            )
            returnOBJ["currently"]["summary"] = translation.translate(currentText)
            returnOBJ["currently"]["icon"] = currentIcon
        except Exception as e:
            print("TEXT GEN ERROR:")
            print(e)

    if exMinutely != 1:
        returnOBJ["minutely"] = dict()
        try:
            returnOBJ["minutely"]["summary"] = translation.translate(currentText)
            returnOBJ["minutely"]["icon"] = currentIcon
        except Exception as e:
            print("TEXT GEN ERROR:")
            print(e)
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
            returnOBJ["hourly"]["summary"] = max(
                set(hourTextList), key=hourTextList.count
            )
            returnOBJ["hourly"]["icon"] = max(set(hourIconList), key=hourIconList.count)
        returnOBJ["hourly"]["data"] = hourList

    if exDaily != 1:
        returnOBJ["daily"] = dict()
        if (not timeMachine) or (tmExtra):
            returnOBJ["daily"]["summary"] = max(set(dayTextList), key=dayTextList.count)
            returnOBJ["daily"]["icon"] = max(set(dayIconList), key=dayIconList.count)
        returnOBJ["daily"]["data"] = dayList

    if exAlerts != 1:
        returnOBJ["alerts"] = alertList

    # Timing Check
    if TIMING:
        print("Final Time")
        print(datetime.datetime.utcnow() - T_Start)

    if exFlags != 1:
        returnOBJ["flags"] = dict()
        returnOBJ["flags"]["sources"] = sourceList
        returnOBJ["flags"]["sourceTimes"] = sourceTimes
        returnOBJ["flags"]["nearest-station"] = int(0)
        returnOBJ["flags"]["units"] = unitSystem
        returnOBJ["flags"]["version"] = "V2.5.5i"

        if version >= 2:
            returnOBJ["flags"]["sourceIDX"] = sourceIDX
            returnOBJ["flags"]["processTime"] = (
                datetime.datetime.utcnow() - T_Start
            ).microseconds

        # if timeMachine:
        # lock.release()

    return ORJSONResponse(
        content=returnOBJ,
        headers={
            "X-Node-ID": platform.node(),
            "X-Response-Time": str((datetime.datetime.utcnow() - T_Start).microseconds),
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
    print(STAGE)
    if STAGE == "PROD":
        download_if_newer(
            s3_bucket,
            "ForecastTar/SubH.zarr.zip",
            "/tmp/SubH_TMP.zarr.zip",
            "/tmp/SubH.zarr.prod.zip",
            True,
        )
        print("SubH Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/HRRR_6H.zarr.zip",
            "/tmp/HRRR_6H_TMP.zarr.zip",
            "/tmp/HRRR_6H.zarr.prod.zip",
            True,
        )
        print("HRRR_6H Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/GFS.zarr.zip",
            "/tmp/GFS.zarr_TMP.zip",
            "/tmp/GFS.zarr.prod.zip",
            True,
        )
        print("GFS Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/NBM.zarr.zip",
            "/tmp/NBM.zarr_TMP.zip",
            "/tmp/NBM.zarr.prod.zip",
            True,
        )
        print("NBM Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/NBM_Fire.zarr.zip",
            "/tmp/NBM_Fire_TMP.zarr.zip",
            "/tmp/NBM_Fire.zarr.prod.zip",
            True,
        )
        print("NBM_Fire Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/GEFS.zarr.zip",
            "/tmp/GEFS_TMP.zarr.zip",
            "/tmp/GEFS.zarr.prod.zip",
            True,
        )
        print("GEFS  Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/HRRR.zarr.zip",
            "/tmp/HRRR_TMP.zarr.zip",
            "/tmp/HRRR.zarr.prod.zip",
            True,
        )
        print("HRRR  Download!")
        download_if_newer(
            s3_bucket,
            "ForecastTar/NWS_Alerts.zarr.zip",
            "/tmp/NWS_Alerts_TMP.zarr.zip",
            "/tmp/NWS_Alerts.zarr.prod.zip",
            True,
        )
        print("Alerts Download!")

        if useETOPO:
            download_if_newer(
                s3_bucket,
                "ForecastTar/ETOPO_DA_C.zarr.zip",
                "/tmp/ETOPO_DA_C_TMP.zarr.zip",
                "/tmp/ETOPO_DA_C.zarr.prod.zip",
                True,
            )
            print("ETOPO Download!")

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

    print(STAGE)

    if zarrReady:
        if STAGE == "PROD":
            time.sleep(20)
            logger.info("Starting Update")

            download_if_newer(
                s3_bucket,
                "ForecastTar/SubH.zarr.zip",
                "/tmp/SubH_TMP.zarr.zip",
                "/tmp/SubH.zarr.prod.zip",
                False,
            )
            logger.info("SubH Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/HRRR_6H.zarr.zip",
                "/tmp/HRRR_6H_TMP.zarr.zip",
                "/tmp/HRRR_6H.zarr.prod.zip",
                False,
            )
            logger.info("HRRR_6H Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/GFS.zarr.zip",
                "/tmp/GFS.zarr_TMP.zip",
                "/tmp/GFS.zarr.prod.zip",
                False,
            )
            logger.info("GFS Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/NBM.zarr.zip",
                "/tmp/NBM.zarr_TMP.zip",
                "/tmp/NBM.zarr.prod.zip",
                False,
            )
            logger.info("NBM Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/NBM_Fire.zarr.zip",
                "/tmp/NBM_Fire_TMP.zarr.zip",
                "/tmp/NBM_Fire.zarr.prod.zip",
                False,
            )
            logger.info("NBM_Fire Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/GEFS.zarr.zip",
                "/tmp/GEFS_TMP.zarr.zip",
                "/tmp/GEFS.zarr.prod.zip",
                False,
            )
            logger.info("GEFS  Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/HRRR.zarr.zip",
                "/tmp/HRRR_TMP.zarr.zip",
                "/tmp/HRRR.zarr.prod.zip",
                False,
            )
            logger.info("HRRR  Download!")
            download_if_newer(
                s3_bucket,
                "ForecastTar/NWS_Alerts.zarr.zip",
                "/tmp/NWS_Alerts_TMP.zarr.zip",
                "/tmp/NWS_Alerts.zarr.prod.zip",
                False,
            )
            logger.info("Alerts Download!")

            if useETOPO:
                download_if_newer(
                    s3_bucket,
                    "ForecastTar/ETOPO_DA_C.zarr.zip",
                    "/tmp/ETOPO_DA_C_TMP.zarr.zip",
                    "/tmp/ETOPO_DA_C.zarr.prod.zip",
                    False,
                )
                logger.info("ETOPO Download!")

        if (STAGE == "PROD") or (STAGE == "DEV"):
            update_zarr_store(False)

    logger.info("Sync End!")
