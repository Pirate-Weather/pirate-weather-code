# %% Read and Process NWS Alerts
# Created on: 2023-01-17

# Import Modules
import os
import re
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
from zarr.core.dtype import VariableLengthUTF8

from API.constants.shared_const import INGEST_VERSION_STR

# %% Setup paths and parameters
ingestVersion = INGEST_VERSION_STR

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
    if not os.path.exists(forecast_path + "/" + ingestVersion):
        os.makedirs(forecast_path + "/" + ingestVersion)
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

# Download zone to county mapping file (optional, used as fallback)
# Note: This mapping may not always be needed as affectedZones often contains both
zone_to_county = {}
zone_county_url = "https://www.weather.gov/source/gis/Shapefiles/County/bp18mr25.dbf"
zone_county_path = os.path.join(tmpDIR, "bp18mr25.dbf")

try:
    r_zone = requests.get(zone_county_url, allow_redirects=True)
    with open(zone_county_path, "wb") as f:
        f.write(r_zone.content)

    # Read the zone to county mapping
    zone_county_gdf = gp.read_file(zone_county_path)

    # Try to build mapping from available columns
    # Look for zone codes and county codes in the shapefile
    for col_name in zone_county_gdf.columns:
        print(f"Zone-County mapping column: {col_name}")

    # Note: The exact column names may vary. This is a best-effort mapping.
    # Most alerts will already have both zone and county in affectedZones.
    if "ZONE" in zone_county_gdf.columns and "FIPS" in zone_county_gdf.columns:
        # Use FIPS code if available
        for idx, row in zone_county_gdf.head(100).iterrows():
            zone = row.get("ZONE", "")
            fips = row.get("FIPS", "")
            if zone and fips:
                zone_to_county[zone] = fips
                if idx < 5:  # Debug: print first few mappings
                    print(f"Zone mapping: {zone} -> {fips}")
except Exception as e:
    print(f"Warning: Could not load zone-to-county mapping: {e}")
    zone_to_county = {}


# Function to build user-friendly alert URL
def build_alert_url(affected_zones, cap_id):
    """
    Build a user-friendly alert URL from affectedZones.

    Args:
        affected_zones: List of zone URLs or string
        cap_id: Alert CAP ID

    Returns:
        User-friendly URL string
    """
    try:
        # Handle different formats of affectedZones
        if pd.isna(affected_zones):
            return f"https://api.weather.gov/alerts/{cap_id}"

        # Convert to list if it's a string
        if isinstance(affected_zones, str):
            zones_list = [affected_zones]
        else:
            zones_list = (
                list(affected_zones)
                if hasattr(affected_zones, "__iter__")
                else [str(affected_zones)]
            )

        # Extract zone codes from URLs (e.g., https://api.weather.gov/zones/forecast/NCZ039 -> NCZ039)
        zone_codes = []
        county_codes = []

        for zone_url in zones_list:
            if isinstance(zone_url, str):
                # Extract the zone code from the URL
                parts = zone_url.split("/")
                if len(parts) > 0:
                    code = parts[-1]
                    # Check if it's a zone (ends with Z followed by digits) or county (ends with C followed by digits)
                    if "Z" in code and code[-3:].isdigit():
                        zone_codes.append(code)
                    elif "C" in code and code[-3:].isdigit():
                        county_codes.append(code)

        # If we don't have a zone, try to extract from first element
        if not zone_codes and zones_list:
            first_zone = str(zones_list[0])
            # Try to extract pattern like NCZ039
            match = re.search(r"([A-Z]{2}[ZC]\d{3})", first_zone)
            if match:
                code = match.group(1)
                if "Z" in code:
                    zone_codes.append(code)
                elif "C" in code:
                    county_codes.append(code)

        # If we have a zone, try to find or map to a county
        if zone_codes:
            warnzone = zone_codes[0]

            # First, check if we already have a county code
            if county_codes:
                warncounty = county_codes[0]
                return f"https://forecast.weather.gov/showsigwx.php?warnzone={warnzone}&warncounty={warncounty}"

            # Try to map zone to county using the mapping file
            if warnzone in zone_to_county:
                warncounty = zone_to_county[warnzone]
                return f"https://forecast.weather.gov/showsigwx.php?warnzone={warnzone}&warncounty={warncounty}"

    except Exception:
        pass

    # Fallback to API URL
    return f"https://api.weather.gov/alerts/{cap_id}"


# Add user-friendly URL column
nws_alert_df["ALERT_URL"] = nws_alert_df.apply(
    lambda row: build_alert_url(row.get("affectedZones"), row.get("id")), axis=1
)

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

# Create a formatted string to save all the relevant in the zarr array
# Use ALERT_URL if available, otherwise fall back to URL from shapefile
points_in_polygons["FINAL_URL"] = points_in_polygons.apply(
    lambda row: (
        row["ALERT_URL"] if pd.notna(row.get("ALERT_URL")) else row.get("URL", "")
    ),
    axis=1,
)

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
    + points_in_polygons["FINAL_URL"].astype(str)
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
gridPoints_XR2 = gridPoints_XR.values.astype(VariableLengthUTF8).reshape(lons.shape)

# Write to zarr
# Save as zarr
if saveType == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/NWS_Alerts.zarr.zip", mode="a", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/NWS_Alerts.zarr")


# Create a Zarr array in the store with zstd compression
zarr_array = zarr.create_array(
    store=zarr_store,
    shape=gridPoints_XR2.shape,
    dtype=zarr.dtype.VariableLengthUTF8(),
    chunks=(10, 10),
    overwrite=True,
)

# Save the data
zarr_array[:] = gridPoints_XR2

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
        forecast_path + "/" + ingestVersion + "/NWS_Alerts.zarr.zip",
    )
else:
    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/NWS_Alerts.zarr",
        forecast_path + "/" + ingestVersion + "/NWS_Alerts.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)
