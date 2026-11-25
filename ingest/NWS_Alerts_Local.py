# %% Read and Process NWS Alerts
# Created on: 2023-01-17

# Import Modules
import logging
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET

import geopandas as gp
import numpy as np
import nwswx
import pandas as pd
import requests
import zarr
from zarr.core.dtype import VariableLengthUTF8

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    close_store,
    configure_zarr_limits,
    open_remote,
    positive_int_env,
    publish_store,
    tune_nofile_limit,
)

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

# Logging
logger = logging.getLogger(__name__)

# Contact the NWS API asks automated clients to send in the User-Agent.
NWS_USERAGENT_ID = "api@alexanderrey.ca"
# /alerts/active normally returns every active alert in one response; this only
# bounds the pagination loop if NWS starts paging it (or keeps returning "next").
NWS_ALERTS_MAX_PAGES = 20
NWS_ALERTS_PAGE_TIMEOUT_S = 60
# What the API shows for an NWS alert, apart from its onset time. Messages that
# agree on all of these look identical to a user (see the collapse step below).
NWS_ALERT_DISPLAY_KEY = ["event", "description", "areaDesc", "EXPIRATION", "severity"]


def main() -> None:
    """Download NWS alerts and write the alerts zarr grid."""
    zarr_store_workers = positive_int_env("zarr_store_workers", 2)
    zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

    tune_nofile_limit()
    zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
        zarr_store_workers, zarr_async_concurrency
    )
    logging.basicConfig(level=logging.INFO)

    # Create new directory for processing if it does not exist
    if not os.path.exists(forecast_process_dir):
        os.makedirs(forecast_process_dir)
    else:
        # If it does exist, remove it
        shutil.rmtree(forecast_process_dir)
        os.makedirs(forecast_process_dir)

    if not os.path.exists(tmpDIR):
        os.makedirs(tmpDIR)

    if saveType == "Download" and not os.path.exists(historic_path):
        os.makedirs(historic_path)

    # %% Download watch warning kml
    warningURL = (
        "https://tgftp.nws.noaa.gov/SL.us008001/DF.sha/DC.cap/DS.WWA/current_all.tar.gz"
    )

    r = requests.get(warningURL, allow_redirects=True)

    # Save to file in tmpDIR
    savePath = os.path.join(forecast_process_dir, "current_all.tar.gz")
    with open(savePath, "wb") as f:
        f.write(r.content)

    with tarfile.open(savePath, "r:gz") as tar:
        # Python 3.14 changes default extract behavior; be explicit and keep compatibility.
        try:
            tar.extractall(path=forecast_process_dir, filter="data")
        except TypeError:
            tar.extractall(path=forecast_process_dir)

    # %% Read in KMZ using geopandas
    nws_alert_gdf = gp.read_file(os.path.join(forecast_process_dir, "current_all.shp"))

    # %% Setup NWS ID
    nws = nwswx.WxAPI(NWS_USERAGENT_ID)

    # Active alerts
    alertsIN = nws.active_alerts(return_format=nwswx.formats.JSONLD)

    nws_alerts = alertsIN["@graph"]

    # Follow any "pagination.next" links in the same JSON-LD format, whose
    # records are under "@graph". (A request without the Accept header returns
    # GeoJSON, with the fields nested under "features[].properties".) Cursor
    # pages can keep linking to an empty page, so stop on one.
    n_pages = 1
    next_url = alertsIN.get("pagination", {}).get("next")
    while next_url:
        if n_pages >= NWS_ALERTS_MAX_PAGES:
            # Alerts on unread pages still reach the grid via the shapefile
            # and the per-alert .cap fallback below.
            logger.warning(
                "NWS active alerts: stopped after %d pages (%d records); not read: %s",
                n_pages,
                len(nws_alerts),
                next_url,
            )
            break
        result = requests.get(
            next_url,
            headers={
                "Accept": nwswx.formats.JSONLD,
                "User-Agent": f"nwswx [{NWS_USERAGENT_ID}]",
            },
            timeout=NWS_ALERTS_PAGE_TIMEOUT_S,
        )
        result.raise_for_status()
        alertsIN = result.json()
        n_pages += 1
        if not alertsIN["@graph"]:
            break
        nws_alerts.extend(alertsIN["@graph"])
        next_url = alertsIN.get("pagination", {}).get("next")
    logger.info(
        "NWS active alerts: %d records from %d page(s)", len(nws_alerts), n_pages
    )
    nws_alert_df = pd.DataFrame.from_records(nws_alerts)

    nws_alert_df["CAP_ID"] = nws_alert_df["id"]
    nws_alert_df = nws_alert_df.drop_duplicates(subset="CAP_ID")

    # Drop superseded messages. An update or correction (e.g. a CCA) lists the
    # messages it replaces in "references", but NWS can keep both active, and
    # each would otherwise be shown as its own alert (issue 698). Shapefile rows
    # for a superseded CAP_ID must go too, or the .cap fallback below would
    # fetch the old message and add it back.
    superseded_cap_ids = {
        reference["identifier"]
        for references in nws_alert_df["references"]
        if isinstance(references, list)
        for reference in references
    }
    n_superseded = nws_alert_df["CAP_ID"].isin(superseded_cap_ids).sum()
    nws_alert_df = nws_alert_df[~nws_alert_df["CAP_ID"].isin(superseded_cap_ids)]
    nws_alert_gdf = nws_alert_gdf[~nws_alert_gdf["CAP_ID"].isin(superseded_cap_ids)]
    logger.info("NWS alerts: dropped %d superseded messages", n_superseded)

    # Merge where available
    nws_alert_merged = nws_alert_gdf.merge(nws_alert_df, how="left", on="CAP_ID")

    ns = {"ns": "urn:oasis:names:tc:emergency:cap:1.2"}

    # Call the NWS API for any missing alerts
    for row in range(len(nws_alert_merged)):
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

            except (requests.RequestException, ET.ParseError, AttributeError):
                nws_alert_merged.loc[row, "headline"] = nws_alert_merged.loc[
                    row, "PROD_TYPE"
                ]
                nws_alert_merged.loc[row, "event"] = nws_alert_merged.loc[
                    row, "PROD_TYPE"
                ]
                nws_alert_merged.loc[row, "description"] = nws_alert_merged.loc[
                    row, "PROD_TYPE"
                ]
                nws_alert_merged.loc[row, "areaDesc"] = nws_alert_merged.loc[row, "WFO"]
                nws_alert_merged.loc[row, "severity"] = nws_alert_merged.loc[row, "SIG"]

    # Collapse messages that look the same to a user. Offices resend products
    # ("RESENT", a new CAP_ID with the same text), and separate VTEC events can
    # carry identical text; either way the API would show the same alert twice.
    # Keep the most recently sent message (then the latest effective time), so a
    # correction wins over the message it corrects. This works per CAP_ID, since
    # one alert has a shapefile row per zone polygon and all of them are kept.
    alerts_by_cap = nws_alert_merged.drop_duplicates(subset="CAP_ID").copy()
    for time_column in ("sent", "effective"):
        alerts_by_cap[time_column + "_utc"] = pd.to_datetime(
            alerts_by_cap[time_column], utc=True, errors="coerce", format="ISO8601"
        )
    newest_first = alerts_by_cap.sort_values(
        ["sent_utc", "effective_utc"], ascending=False, na_position="last"
    )
    kept_cap_ids = newest_first.drop_duplicates(subset=NWS_ALERT_DISPLAY_KEY)["CAP_ID"]
    logger.info(
        "NWS alerts: collapsed %d messages that repeat another alert's text; "
        "%d alerts kept",
        len(alerts_by_cap) - len(kept_cap_ids),
        len(kept_cap_ids),
    )
    nws_alert_merged = nws_alert_merged[nws_alert_merged["CAP_ID"].isin(kept_cap_ids)]

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

    # Create a formatted string to save all relevant fields in the zarr array.
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

    # Normalize possible null/non-string values before joining grouped strings.
    points_in_polygons["string"] = points_in_polygons["string"].fillna("").astype(str)
    points_in_polygons = points_in_polygons[points_in_polygons["string"] != ""]

    # Combine formatted strings using "|" as a separator. Drop exact repeats
    # (dict.fromkeys keeps the first, in order): the API rebuilds the alert URI
    # from the request location, so repeated entries would be identical alerts.
    df = points_in_polygons.groupby("INDEX", as_index=False)["string"].agg(
        lambda values: "|".join(dict.fromkeys(v for v in values if v))
    )

    # Merge back into primary geodataframe
    gridPointsSeries = gridPointsSeries.merge(df, on="INDEX", how="left")

    # Set empty data as blank
    gridPointsSeries["string"] = gridPointsSeries["string"].fillna("")

    # Concert to string
    gridPointsSeries["string"] = gridPointsSeries["string"].astype(str)

    # %% XR approach
    gridPoints_XR = gridPointsSeries["string"].to_xarray()

    # Reshape to 2D
    gridPoints_XR2 = gridPoints_XR.values.astype(VariableLengthUTF8).reshape(lons.shape)

    # Write to zarr
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/NWS_Alerts.zarr")
    zarr_array = zarr.create_array(
        store=zarr_store,
        shape=gridPoints_XR2.shape,
        dtype=zarr.dtype.VariableLengthUTF8(),
        # The API reads this with zarr-python: tensorstore cannot read
        # variable-length strings. The whole grid is written in one call below.
        chunks=(FINAL_CHUNK_SIZES["ALERTS"],) * 2,
        shards=(FINAL_SHARD_SIZES["ALERTS"],) * 2,
        config={"write_empty_chunks": True},
        overwrite=True,
    )
    zarr_array[:] = gridPoints_XR2
    close_store(zarr_store)

    # Publish to S3 or the local publish directory
    forecast_remote = open_remote(
        forecast_path + "/" + ingestVersion,
        saveType,
        aws_access_key_id,
        aws_secret_access_key,
    )
    publish_store(
        forecast_process_dir + "/NWS_Alerts.zarr", forecast_remote, "NWS_Alerts.zarr"
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)


if __name__ == "__main__":
    main()
