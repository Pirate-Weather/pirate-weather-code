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

# NOTE: You may need a similar tool to wgrib2 for FMI data if it needs to be processed.
# Assuming standard GRIB format readable by cfgrib, wgrib2 might not be necessary.
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
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

# A dictionary to map FMI GRIB short names to our desired Zarr names.
# This mapping is crucial and needs to be verified by inspecting the actual GRIB file.
# The keys are placeholders and must be replaced with the actual short names from the FMI data.
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


# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)

if saveType == "Download":
    if not os.path.exists(forecast_path + "/" + ingestVersion):
        os.makedirs(forecast_path + "/" + ingestVersion)

T0 = time.time()


# Function to get the latest model run time (origintime)
def get_latest_origintime():
    """
    Finds the latest available model run time, rounded to the nearest 3-hour interval.
    The FMI Harmonie model runs are typically at 00, 03, 06, 09, 12, 15, 18, and 21 UTC.
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

# A more robust solution for the bounding box might be needed, but this is a good start
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
    # Use cfgrib to open the GRIB file.
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
# We'll rechunk to make sure it's optimized for saving
xarray_fmi_data = xarray_fmi_data.chunk(
    chunks={"time": xarray_fmi_data.time.size, "x": processChunk, "y": processChunk}
)

with ProgressBar():
    xarray_fmi_data.to_zarr(
        forecast_process_path + "_.zarr", mode="w", consolidated=False, compute=True
    )
logger.info("Saved Zarr data to disk.")


# %% Final output handling and cleanup
# This section is directly modeled after the GFS script's final steps
if saveType == "S3":
    # Upload the Zarr directory to S3
    s3.put(
        forecast_process_path + "_.zarr",
        forecast_path + "/" + ingestVersion + "/FMI.zarr",
        recursive=True,
    )

    with open(forecast_process_dir + "/FMI.time.pickle", "wb") as file:
        pickle.dump(origintime, file)

    s3.put_file(
        forecast_process_dir + "/FMI.time.pickle",
        forecast_path + "/" + ingestVersion + "/FMI.time.pickle",
    )
else:
    with open(forecast_process_dir + "/FMI.time.pickle", "wb") as file:
        pickle.dump(origintime, file)

    shutil.move(
        forecast_process_dir + "/FMI.time.pickle",
        forecast_path + "/" + ingestVersion + "/FMI.time.pickle",
    )

    shutil.copytree(
        forecast_process_path + "_.zarr",
        forecast_path + "/" + ingestVersion + "/FMI.zarr",
        dirs_exist_ok=True,
    )

# Clean up temporary files and directories
shutil.rmtree(forecast_process_dir)

T1 = time.time()
logger.info(f"Total processing time: {T1 - T0} seconds")

logger.info("Script finished successfully.")
