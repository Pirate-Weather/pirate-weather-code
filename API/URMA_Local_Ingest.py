# %% URMA Historical Ingest script
# Alexander Rey, August 2025
#
# This script is designed to download URMA (Unrestricted Mesoscale Analysis)
# data from a specified historical period and append it to a Zarr archive.
# It accounts for the 6-hour delay in URMA data availability.

# %% Import modules
import os
import pickle
import traceback
import warnings
import numpy as np
import logging

import pandas as pd
import s3fs
import xarray as xr
from herbie import Herbie
from metpy.calc import relative_humidity_from_specific_humidity

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# %% Setup paths and parameters
ingestVersion = "v27"

analysis_process_dir = os.getenv(
    "analysis_process_dir", default="/home/ubuntu/Weather/URMA-Historic"
)
historic_path = os.getenv("historic_path", default="/home/ubuntu/Weather/History/URMA")
tmpDIR = analysis_process_dir + "/Downloads"

saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


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
    Main function to orchestrate the URMA data ingestion for a historical period.
    """
    # Create the date range to download. This example downloads the past 7 days.
    # The start time is set to 6 hours ago to ensure the most recent data is available.
    start_time = pd.Timestamp.utcnow().floor("1H") - pd.Timedelta(days=7)
    end_time = pd.Timestamp.utcnow().floor("1H") - pd.Timedelta(hours=6)

    # Generate a list of dates/times for each hour
    dates = pd.date_range(start=start_time, end=end_time, freq="1H")

    # Define variable mapping from GRIB short names to desired names
    variable_mapping = {
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

    # Define the final variable list for selection
    variables_to_keep = [
        "time",
        "latitude",
        "longitude",
        "PRES_surface",
        "TMP_2maboveground",
        "DPT_2maboveground",
        "GUST_surface",
        "UGRD_10maboveground",
        "VGRD_10maboveground",
        "RH_2maboveground",
        "VIS_surface",
        "TCDC_entireatmosphere",
    ]

    for date in dates:
        logging.info(f"Processing data for: {date}")
        try:
            H = Herbie(date, model="urma")

            # Check if the file exists
            if H.grib_path is None:
                logging.warning(f"File not found for {date}, skipping...")
                continue

            # Read the GRIB2 file using xarray and cfgrib
            ds = H.xarray(variable_mapping.keys(), verbose=False)

            # Convert specific humidity to relative humidity
            ds = convert_specific_to_relative_humidity(ds)

            # Rename variables using the defined mapping
            ds = ds.rename(variable_mapping)

            # Filter for only the variables we need
            ds = ds[variables_to_keep]

            # Set up the Zarr store path
            store_path = f"{historic_path}/{ingestVersion}/URMA.zarr"
            if saveType == "S3":
                store = s3fs.S3Map(root=store_path, s3=s3)
            else:
                store = store_path
                os.makedirs(os.path.dirname(store_path), exist_ok=True)

            # Check if Zarr store exists to decide between 'w' and 'a' mode
            if saveType == "S3" and s3.exists(store_path):
                mode = "a"
            elif saveType != "S3" and os.path.exists(store_path):
                mode = "a"
            else:
                mode = "w"

            # Save or append the data to the Zarr store
            ds.to_zarr(store=store, mode=mode, append_dim="time", consolidated=True)

            logging.info(f"Successfully processed and saved data for {date}")

        except Exception as e:
            logging.error(f"Failed to process URMA data for {date}: {e}")
            traceback.print_exc()
            continue

    # After the loop, save the latest processed time
    base_time = pd.Timestamp.utcnow().floor("1H") - pd.Timedelta(hours=6)
    time_file_path = f"{historic_path}/{ingestVersion}/URMA.time.pickle"
    if saveType == "S3":
        with s3.open(time_file_path, "wb") as f:
            pickle.dump(base_time, f)
    else:
        os.makedirs(os.path.dirname(time_file_path), exist_ok=True)
        with open(time_file_path, "wb") as f:
            pickle.dump(base_time, f)