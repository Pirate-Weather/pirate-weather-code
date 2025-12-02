# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import requests
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingestVersion = "v27"

# FMI GRIBs are opened with the `cfgrib` engine (used via xarray) in this script.
# In most workflows `wgrib2` is not required. If you need low-level inspection or
# format conversions, install `wgrib2` or use `pygrib`. The `wgrib2_path` default
# is provided only as a hint and is not used by the current code path.
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2"
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/FMI"
)
forecast_process_path = forecast_process_dir + "/FMI_Process"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/FMI")

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and final chunk size
processChunk = 100
finalChunk = 3

# Define the variables to be saved in the final Zarr store
# These names are based on your provided GFS naming convention.
zarrVars = (
    "time",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "GUST_surface",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "APCP_surface",
    "CAPE_surface",
    "TCDC_entireatmosphere",
    "VIS_surface",
    "DSWRF_surface",
)

hisPeriod = 48

# Map of GRIB parameter short names (as exposed by cfgrib/xarray) to our target
# Zarr variable names. Confirm these keys by inspecting the GRIB (for example:
# `xr.open_dataset(path, engine='cfgrib')` and check `dataset.variables`), or
# with tools like `pygrib`/`wgrib2 -v`. Keep this mapping aligned with the
# source GRIB short names to ensure variables are renamed correctly.
grib_to_zarr_map = {
    "pressure": "PRMSL_meansealevel",
    "temperature": "TMP_2maboveground",
    "dewpoint": "DPT_2maboveground",
    "humidity": "RH_2maboveground",
    "windgust": "GUST_surface",
    "windums": "UGRD_10maboveground",
    "windvms": "VGRD_10maboveground",
    "precipitationamount": "APCP_surface",
    "cape": "CAPE_surface",
    "totalcloudcover": "TCDC_entireatmosphere",
    "visibility": "VIS_surface",
    "radiationnetsurfaceswaccumulation": "DSWRF_surface",
}


# Create directories for processing, but avoid deleting higher-level folders.
os.makedirs(forecast_process_dir, exist_ok=True)
os.makedirs(forecast_process_path, exist_ok=True)
os.makedirs(tmpDIR, exist_ok=True)

if saveType == "Download":
    os.makedirs(os.path.join(forecast_path, ingestVersion), exist_ok=True)

T0 = time.time()


# Function to get the latest model run time (origintime)
def get_latest_origintime():
    """
    Find the latest model run time, rounded to the most recent 3-hour interval.
    FMI Harmonie runs on 3-hour cycles (00, 03, 06, ... 21 UTC). We use a small
    buffer (45 minutes) to avoid picking a run that may still be uploading.
    """
    now_utc = datetime.now(timezone.utc)

    # Calculate the latest 3-hour block
    latest_run_hour = (now_utc.hour // 3) * 3

    # The run time is the beginning of the 3-hour block
    latest_origintime = now_utc.replace(
        hour=latest_run_hour, minute=0, second=0, microsecond=0
    )

    # If the current time is too close to the latest run hour, the data might not be ready yet.
    # We'll subtract 3 hours to get the previous run, which should be available.
    # The 'timedelta(minutes=45)' is a reasonable buffer to give the model time to process and upload the data.
    if (now_utc - latest_origintime) < timedelta(minutes=45):
        latest_origintime -= timedelta(hours=3)

    return latest_origintime


# %% Construct the FMI download URL
base_url = "https://opendata.fmi.fi/download"
producer = "harmonie_scandinavia_surface"
parameters = "Pressure,Temperature,DewPoint,Humidity,WindUMS,WindVMS,PrecipitationAmount,CAPE,TotalCloudCover,Visibility,WindGust,RadiationNetSurfaceSWAccumulation"

# Get the latest model run time
origintime = get_latest_origintime()
starttime = origintime
endtime = origintime + timedelta(hours=72)  # FMI Harmonie has a 72-hour forecast

# Bounding box for the FMI request. Format is `lon_min,lat_min,lon_max,lat_max`
# in WGS84 (EPSG:4326). Consider moving this to configuration or environment
# variables for different deployments/regions.
bbox = "-18.118179851154,49.765786639371,54.237,75.227023343491"
projection = "EPSG:4326"
file_format = "grib2"
timestep = 60  # 60 minutes

download_params = {
    "producer": producer,
    "param": parameters,
    "bbox": bbox,
    "origintime": origintime.isoformat(),
    "starttime": starttime.isoformat(),
    "endtime": endtime.isoformat(),
    "format": file_format,
    "projection": projection,
    "levels": 0,
    "timestep": timestep,
}

# Construct the final URL
grib_url = f"{base_url}?{'&'.join([f'{key}={value}' for key, value in download_params.items()])}"

# Check if this is newer than the current file
if saveType == "S3":
    if s3.exists(forecast_path + "/" + ingestVersion + "/FMI.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingestVersion + "/FMI.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= origintime:
            logger.info("No Update to FMI, ending")
            sys.exit()
else:
    if os.path.exists(forecast_path + "/" + ingestVersion + "/FMI.time.pickle"):
        with open(
            forecast_path + "/" + ingestVersion + "/FMI.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= origintime:
            logger.info("No Update to FMI, ending")
            sys.exit()

logger.info(f"Found latest GRIB file: {grib_url}")
logger.info(f"Base time: {origintime}")

# %% Download the GRIB file
local_grib_file = os.path.join(
    tmpDIR, f"fmi_harmonie_surface_{origintime.strftime('%Y%m%d_%H%M')}.grib2"
)
logger.info(f"Downloading to {local_grib_file}")
try:
    with requests.get(grib_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(local_grib_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logger.info("Download complete.")
except requests.exceptions.RequestException as e:
    logger.error(f"Error downloading the GRIB file: {e}")
    sys.exit(1)

# %% Process the GRIB file
try:
    # Use cfgrib to open the GRIB file. If cfgrib returns multiple messages or
    # variables, `filter_by_keys` helps limit what is loaded. If a parameter is
    # missing here, inspect the GRIB's available short names and update
    # `grib_to_zarr_map` accordingly.
    xarray_fmi_data = xr.open_dataset(
        local_grib_file,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": list(grib_to_zarr_map.keys())}},
    )

    # Rename variables according to our map
    xarray_fmi_data = xarray_fmi_data.rename(
        {
            grib_short_name: zarr_name
            for grib_short_name, zarr_name in grib_to_zarr_map.items()
            if grib_short_name in xarray_fmi_data
        }
    )

    # Ensure a consistent time coordinate, as in the GFS script
    xarray_fmi_data = xarray_fmi_data.assign_coords({"time": xarray_fmi_data.time.data})

    logger.info("Successfully opened and processed GRIB file with xarray.")

except Exception as e:
    logger.error(f"Error processing GRIB file: {e}")
    sys.exit(1)

# %% Save the dataset with compression and filters
# Rechunk before writing to make the Zarr store more efficient for reads and
# downstream processing. Chunk sizes are tuned heuristically for our workflow.
xarray_fmi_data = xarray_fmi_data.chunk(
    chunks={"time": xarray_fmi_data.time.size, "x": processChunk, "y": processChunk}
)

# Save to a clear Zarr path inside the process folder
process_zarr = os.path.join(forecast_process_path, "FMI.zarr")

with ProgressBar():
    xarray_fmi_data.to_zarr(process_zarr, mode="w", consolidated=False, compute=True)
logger.info("Saved Zarr data to disk at %s", process_zarr)


# %% Final output handling and cleanup
# Final output handling. We support either uploading to S3 or copying to a local
# deployment directory. Note: writing many small files to S3 with `s3.open` can
# be slower than using zarr's native S3 stores (`S3Map`/`s3fs.S3Map`) or writing
# the zarr store directly to S3 via zarr's store implementations. The current
# approach is portable and straightforward; consider optimizing for large
# deployments.
if saveType == "S3":
    # Upload the Zarr directory to S3 by walking its files. This is more portable
    # than attempting to call a recursive put on the filesystem object.
    dest_base = f"{forecast_path}/{ingestVersion}/FMI.zarr"
    for root, dirs, files in os.walk(process_zarr):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, process_zarr)
            s3_path = os.path.join(dest_base, rel_path)
            with open(local_path, "rb") as lf, s3.open(s3_path, "wb") as sf:
                sf.write(lf.read())

    # write the time pickle to S3
    with s3.open(f"{forecast_path}/{ingestVersion}/FMI.time.pickle", "wb") as f:
        pickle.dump(origintime, f)
    logger.info("Uploaded Zarr and time pickle to S3 at %s", dest_base)
else:
    # Local filesystem: save the time pickle and copy the zarr directory
    local_time_pickle = os.path.join(forecast_process_dir, "FMI.time.pickle")
    with open(local_time_pickle, "wb") as file:
        pickle.dump(origintime, file)

    dest_dir = os.path.join(forecast_path, ingestVersion)
    os.makedirs(dest_dir, exist_ok=True)

    shutil.copy(local_time_pickle, os.path.join(dest_dir, "FMI.time.pickle"))
    shutil.copytree(process_zarr, os.path.join(dest_dir, "FMI.zarr"), dirs_exist_ok=True)
    logger.info("Copied Zarr and time pickle to %s", dest_dir)

# Clean up process folder (only the process zarr and temp GRIB)
# We intentionally do NOT remove the parent `forecast_process_dir` so
# configuration and other process-level files are preserved across runs.
try:
    if os.path.exists(process_zarr):
        shutil.rmtree(process_zarr)
    if os.path.exists(local_grib_file):
        os.remove(local_grib_file)
except Exception:
    logger.warning("Failed to fully clean up temporary files.")

T1 = time.time()
logger.info(f"Total processing time: {T1 - T0} seconds")

logger.info("Script finished successfully.")
