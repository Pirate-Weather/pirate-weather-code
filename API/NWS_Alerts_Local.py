# %% Read and Process NWS Alerts
# Created on: 2023-01-17

# Import Modules
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET


import geopandas as gp
import numpy as np
import nwswx
import pandas as pd
import requests
import s3fs
import zarr
from numpy.dtypes import StringDType

# %% Setup paths and parameters
wgrib2_path = os.getenv("wgrib2_path", default="/home/ubuntu/wgrib2_build/bin/wgrib2 ")

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/NWS_Alerts"
)
forecast_process_path = forecast_process_dir + "/NWS_Alerts_Process"
hist_process_path = forecast_process_dir + "/NWS_Alerts_Historic"
tmpDIR = forecast_process_dir + "/Downloads"

forecast_path = os.getenv(
    "forecast_path", default="/home/ubuntu/Weather/Prod/NWS_Alerts"
)
historic_path = os.getenv(
    "historic_path", default="/home/ubuntu/Weather/History/NWS_Alerts"
)


saveType = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Define the processing and history chunk size
processChunk = 100

# Define the final x/y chunksize
finalChunk = 3

hisPeriod = 36

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)

if saveType == "Download":
    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

# %% Download watch warning kml
warningURL = (
    "https://tgftp.nws.noaa.gov/SL.us008001/DF.sha/DC.cap/DS.WWA/current_all.tar.gz"
)

r = requests.get(warningURL, allow_redirects=True)

# Save to file in tmpDIR
savePath = os.path.join(forecast_process_dir, "current_all.tar.gz")
open(savePath, "wb").write(r.content)

tar = tarfile.open(savePath, "r:gz")
tar.extractall(path=forecast_process_dir)
tar.close()

# %% Read in KMZ using geopandas
nws_alert_gdf = gp.read_file(os.path.join(forecast_process_dir, "current_all.shp"))

# %% Setup NWS ID
nws = nwswx.WxAPI("api@alexanderrey.ca")

# Active alerts
alertsIN = nws.active_alerts(return_format=nwswx.formats.JSONLD)

nws_alerts = alertsIN["@graph"]

# If more than one page of alerts, loop through an append
while "pagination" in alertsIN:
    # Get the next page of alerts
    result = requests.get(alertsIN["pagination"]["next"])

    # Convert to JSON
    alertsIN = result.json()
    nws_alerts.extend(alertsIN["features"])

    print("AWS Alerts: ", len(nws_alerts))
nws_alert_df = pd.DataFrame.from_records(nws_alerts)

nws_alert_df["CAP_ID"] = nws_alert_df["id"]

# Merge where available
nws_alert_merged = nws_alert_gdf.merge(
    nws_alert_df.drop_duplicates(subset="CAP_ID"), how="left", on="CAP_ID"
)

ns = {"ns": "urn:oasis:names:tc:emergency:cap:1.2"}

# print(list(nws_alert_merged.columns.values))

# Call the NWS API for any missing alerts
for row in range(0, len(nws_alert_merged)):
    if pd.isna(nws_alert_merged.loc[row, "headline"]):
        try:
            response = requests.get(
                "https://api.weather.gov/alerts/"
                + nws_alert_merged.loc[row, "CAP_ID"]
                + ".cap"
            )
            root = ET.fromstring(response.content)

            nws_alert_merged.loc[row, "headline"] = root.find(
                ".//ns:info/ns:headline", namespaces=ns
            ).text
            nws_alert_merged.loc[row, "event"] = root.find(
                ".//ns:info/ns:event", namespaces=ns
            ).text
            nws_alert_merged.loc[row, "description"] = root.find(
                ".//ns:info/ns:description", namespaces=ns
            ).text
            nws_alert_merged.loc[row, "areaDesc"] = root.find(
                ".//ns:area/ns:areaDesc", namespaces=ns
            ).text
            nws_alert_merged.loc[row, "severity"] = root.find(
                ".//ns:info/ns:severity", namespaces=ns
            ).text

        except Exception:
            nws_alert_merged.loc[row, "headline"] = nws_alert_merged.loc[
                row, "PROD_TYPE"
            ]
            nws_alert_merged.loc[row, "event"] = nws_alert_merged.loc[row, "PROD_TYPE"]
            nws_alert_merged.loc[row, "description"] = nws_alert_merged.loc[
                row, "PROD_TYPE"
            ]
            nws_alert_merged.loc[row, "areaDesc"] = nws_alert_merged.loc[row, "WFO"]
            nws_alert_merged.loc[row, "severity"] = nws_alert_merged.loc[row, "SIG"]


# Convert to geodataframe
nws_alert_merged_gdf = gp.GeoDataFrame(nws_alert_merged, geometry="geometry_x")

# %% Create grid of points
xs = np.arange(-127, -65, 0.025)
ys = np.arange(24, 50, 0.025)

lons, lats = np.meshgrid(xs, ys)

# Create GeoSeries of Points
gridPointsSeries = gp.GeoDataFrame(
    geometry=gp.points_from_xy(lons.flatten(), lats.flatten()), crs="EPSG:4326"
)
gridPointsSeries["INDEX"] = gridPointsSeries.index
points_in_polygons = gp.sjoin(
    gridPointsSeries, nws_alert_merged_gdf, predicate="within", how="inner"
)

# Create a formatted string ton save all the relevant in the zarr array
points_in_polygons["string"] = (
    points_in_polygons["event"].astype(str) + "}"
    "{"
    + points_in_polygons["description"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["areaDesc"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["effective"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["EXPIRATION"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["severity"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["URL"].astype(str)
)


float_rows = points_in_polygons[
    points_in_polygons["string"].apply(lambda x: isinstance(x, float))
]

# Print the filtered rows
print(float_rows)

# Combine the formatted strings using "|" as a spacer
df = points_in_polygons.groupby("INDEX").agg({"string": "|".join}).reset_index()


# Merge back into primary geodataframe
gridPointsSeries = gridPointsSeries.merge(df, on="INDEX", how="left")

# Set empty data as blank
gridPointsSeries.loc[gridPointsSeries["string"].isna(), ["string"]] = ""

# Concert to string
gridPointsSeries["string"] = gridPointsSeries["string"].astype(str)

# %% XR approach
gridPoints_XR = gridPointsSeries["string"].to_xarray()

# Reshape to 2D
gridPoints_XR2 = gridPoints_XR.values.astype(StringDType()).reshape(lons.shape)

# Write to zarr
# Save as zarr
if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/NWS_Alerts.zarr.zip", mode="a"
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/NWS_Alerts.zarr")


# Create a Zarr array in the store with zstd compression
# with ProgressBar():
zarr_array = zarr.create_array(
    data=gridPoints_XR2,
    store=zarr_store,
    chunks=(4, 4),
)

if saveType == "S3":
    zarr_store.close()

# Test Read
# zip_store_read = zarr.storage.ZipStore(
#     merge_process_dir + "/NWS_Alerts.zarr.zip", compression=0, mode="r"
# )
# alertsReadTest = zarr.open_array(zip_store_read)
# print(alertsReadTest[600, 1500:])


# Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/NWS_Alerts.zarr.zip",
        forecast_path + "/ForecastTar/NWS_Alerts.zarr.zip",
    )
else:
    # Copy the zarr file to the final location
    shutil.copytree(forecast_process_dir + "/NWS_Alerts.zarr",
    forecast_path + "/NWS_Alerts.zarr",
                    dirs_exist_ok=True)

# Clean up
shutil.rmtree(forecast_process_dir)
