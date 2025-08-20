# %% RTMA Rapid Update Processing script
# Based on RTMA_Local_Ingest.py
# Alexander Rey, August 2025

# %% Import modules
import os
import pickle
import sys
import traceback
import warnings
import requests
import logging

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from metpy.calc import relative_humidity_from_specific_humidity

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# %% Setup paths and parameters
ingestVersion = "v27"

analysis_process_dir = os.getenv(
    "analysis_process_dir", default="/home/ubuntu/Weather/RTMA-RU"
)
analysis_path = os.getenv("analysis_path", default="/home/ubuntu/Weather/Prod/RTMA-RU")
tmpDIR = analysis_process_dir + "/Downloads"

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


def get_latest_rtma_ru_data():
    """
    Find the latest available RTMA-RU file by looking back to the previous
    15-minute interval to account for data delay.
    """
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod/"

    # RTMA-RU runs every 15 minutes. We need to find the latest
    # available file, which is typically from the previous 15-minute interval.
    now = pd.Timestamp.now(tz="utc")

    # Adjust to the latest *previous* 15-minute interval.
    # This ensures we are always looking for a file that should have been published.
    file_time = now.floor("15T") - pd.Timedelta(minutes=15)

    yyyymmdd = file_time.strftime("%Y%m%d")
    hh = file_time.strftime("%H")

    file_name = f"rtma.t{hh}z.ru.2dvaranl_ndfd_conus.grib2"
    full_url = f"{base_url}rtma.{yyyymmdd}/{file_name}"

    os.makedirs(tmpDIR, exist_ok=True)
    local_path = os.path.join(tmpDIR, file_name)

    logging.info(f"Attempting to download from: {full_url}")

    try:
        response = requests.get(full_url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info("Download successful.")

        # Convert GRIB2 to xarray dataset
        ds = xr.open_dataset(local_path, engine="cfgrib")

        return ds

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {e}")
        traceback.print_exc()
        raise
    except Exception as e:
        logging.error(f"Failed to download or process RTMA-RU data: {e}")
        traceback.print_exc()
        raise


def convert_specific_to_relative_humidity(ds):
    """
    Convert specific humidity (shum) to relative humidity.

    This calculation requires temperature and pressure.
    """
    try:
        T = ds["tmp"].metpy.sel(vertical="2 m").metpy.quantify()  # Temperature
        q = ds["shum"].metpy.sel(vertical="2 m").metpy.quantify()  # Specific humidity
        p = ds["pressfc"].metpy.quantify()  # Surface pressure

        rh = relative_humidity_from_specific_humidity(p, T, q)
        ds["rh"] = (("latitude", "longitude"), rh.magnitude)
        ds["rh"].attrs["units"] = "%"
        ds["rh"].attrs["long_name"] = "Relative Humidity"

    except Exception as e:
        logging.error(f"Error converting specific humidity: {e}")
        traceback.print_exc()
        ds["rh"] = xr.full_like(ds["tmp"], np.nan)
        ds["rh"].attrs["units"] = "%"
        ds["rh"].attrs["long_name"] = "Relative Humidity (Calculation Failed)"
    return ds


def main():
    """
    Main function to orchestrate the RTMA-RU data ingestion.
    """
    ds = None
    try:
        ds = get_latest_rtma_ru_data()
    except Exception as e:
        logging.error(f"Failed to download RTMA-RU data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Add a global time coordinate for consistency
    ds = ds.assign_coords(time=ds.valid_time)

    # Convert specific humidity to relative humidity
    ds = convert_specific_to_relative_humidity(ds)

    # Rename variables for consistency
    ds = ds.rename(
        {
            "TMP": "TMP_2maboveground",
            "DPT": "DPT_2maboveground",
            "GUST": "GUST_surface",
            "UGRD": "UGRD_10maboveground",
            "VGRD": "VGRD_10maboveground",
            "VIS": "VIS_surface",
            "TCDC": "TCDC_entireatmosphere",
            "RH": "RH_2maboveground",
            "PRES": "PRES_surface",
        }
    )

    # Select and rename variables for consistency
    variables_to_keep = [
        "time",
        "latitude",
        "longitude",
        "pressfc",
        "TMP_2maboveground",
        "DPT_2maboveground",
        "GUST_surface",
        "UGRD_10maboveground",
        "VGRD_10maboveground",
        "RH_2maboveground",
        "VIS_surface",
        "TCDC_entireatmosphere",
    ]

    # Filter for only the variables we need
    ds = ds[variables_to_keep]

    # Process and save the data
    base_time = pd.Timestamp.now(tz="utc")
    try:
        if saveType == "S3":
            # Save to S3
            ds.to_zarr(
                store=s3fs.S3Map(
                    root=f"{analysis_path}/{ingestVersion}/RTMA-RU.zarr", s3=s3
                ),
                mode="w",
            )
        else:
            # Save locally
            ds.to_zarr(f"{analysis_path}/{ingestVersion}/RTMA-RU.zarr", mode="w")

        # Write most recent time to a file
        with open(f"{analysis_path}/{ingestVersion}/RTMA-RU.time.pickle", "wb") as file:
            pickle.dump(base_time, file)

        logging.info(f"RTMA-RU data successfully processed for {base_time}")

    except Exception as e:
        logging.error(f"Failed to save RTMA-RU data: {e}")
        traceback.print_exc()
        sys.exit(1)
