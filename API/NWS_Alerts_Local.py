# %% Read and Process NWS Alerts
# Created on: 2023-01-17

# Import Modules
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET

import dask
import dask.array as da
import geopandas as gp
import numcodecs
import numpy as np
import nwswx
import pandas as pd
import requests
import s3fs
import zarr
from numcodecs import Blosc

s3_bucket = "s3://piratezarr2"
s3_save_path = "/ForecastProd/Alerts/NWS_"
merge_process_dir = os.getenv("merge_process_dir", default="/home/ubuntu/data/")
saveType = os.getenv("save_type", default="S3")
s3_bucket = os.getenv("save_path", default="s3://piratezarr2")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


# Create new directory for processing if it does not exist
if not os.path.exists(merge_process_dir):
    os.makedirs(merge_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(merge_process_dir)
    os.makedirs(merge_process_dir)


if saveType == "Download":
    if not os.path.exists(s3_bucket):
        os.makedirs(s3_bucket)
    if not os.path.exists(s3_bucket + "/ForecastTar"):
        os.makedirs(s3_bucket + "/ForecastTar")

# %% Download watch warning kml
warningURL = (
    "https://tgftp.nws.noaa.gov/SL.us008001/DF.sha/DC.cap/DS.WWA/current_all.tar.gz"
)

r = requests.get(warningURL, allow_redirects=True)

# Save to file in tmpDIR
savePath = os.path.join(merge_process_dir, "current_all.tar.gz")
open(savePath, "wb").write(r.content)

tar = tarfile.open(savePath, "r:gz")
tar.extractall(path=merge_process_dir)
tar.close()

# %% Read in KMZ using geopandas
nws_alert_gdf = gp.read_file(os.path.join(merge_process_dir, "current_all.shp"))

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
points_in_polygons["string"] = points_in_polygons["event"].astype(
    str
) + "}" "{" + points_in_polygons["description"].astype(
    str
) + "}" + "{" + points_in_polygons["areaDesc"].astype(
    str
) + "}" + "{" + points_in_polygons["effective"].astype(
    str
) + "}" + "{" + points_in_polygons["EXPIRATION"].astype(
    str
) + "}" + "{" + points_in_polygons["severity"].astype(
    str
) + "}" + "{" + points_in_polygons["URL"].astype(str)


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

# %% Create Dask Array of Strings
dask.config.set({"dataframe.convert-string": True})
grid_alerts_dask = da.from_array(gridPointsSeries["string"], chunks=4)
grid_alerts_square = da.reshape(grid_alerts_dask, lons.shape)
grid_alerts_chunk = grid_alerts_square.rechunk(4)

compressor = Blosc(cname="zstd", clevel=1)  # Use zstd compression

# Save as zarr
zip_store = zarr.ZipStore(
    merge_process_dir + "/NWS_Alerts.zarr.zip", compression=0, mode="w"
)
grid_alerts_chunk.to_zarr(
    zip_store,
    overwrite=True,
    compressor=compressor,
    object_codec=numcodecs.vlen.VLenUTF8(),
)
zip_store.close()


# Upload to S3
if saveType == "S3":
    # Upload to S3
    s3.put_file(
        merge_process_dir + "/NWS_Alerts.zarr.zip",
        s3_bucket + "/ForecastTar/NWS_Alerts.zarr.zip",
    )
else:
    # Move to local
    shutil.move(
        merge_process_dir + "/NWS_Alerts.zarr.zip",
        s3_bucket + "/ForecastTar/NWS_Alerts.zarr.zip",
    )

    # Clean up
    shutil.rmtree(merge_process_dir)
