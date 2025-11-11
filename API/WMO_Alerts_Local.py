"""
Alexander Rey, October 2025

NOTE: This script only processes alerts where the CAP contains polygon data. This does not include every alert.
NOTE: US Alerts are handled separately in NWS_Alerts_Local.py, since polygons are not always included in the CAP messages.

Retrieve CAP alert polygons from all RSS feeds and return them as a GeoDataFrame.

This convenience function automates the entire workflow:

1. Download the WMO ``sources.json`` file to determine which feed
   identifiers are currently operational.
2. For each ``sourceId``, fetch the corresponding RSS feed located
   at ``https://severeweather.wmo.int/v2/cap-alerts/{sourceId}/rss.xml``.
3. Parse each feed for item links and download every CAP XML
   document referenced.
4. Extract all polygons defined in the CAP documents and assemble
   them into a ``geopandas.GeoDataFrame`` with columns for
   ``source_id``, ``event``, ``area_desc`` and a geometry column.

Parameters
----------
timeout : float, optional
    Socket timeout (in seconds) applied to HTTP requests.  A value
    of 30 seconds is used by default.

Returns
-------
geopandas.GeoDataFrame
    A GeoDataFrame where each row corresponds to a single polygon
    extracted from a CAP message.  The geometry column contains
    ``shapely.geometry.Polygon`` objects in EPSG:4326.

Notes
-----
* If a feed or CAP message fails to download or parse, the error
  is logged and processing continues with subsequent feeds.
* The function may take several minutes to run depending on the
  number of feeds and the volume of CAP alerts published.
* Polygons are not simplified or validated beyond ensuring that
  they contain at least three vertices and are closed.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import aiohttp
import geopandas as gpd
import numpy as np
import s3fs
import zarr
from shapely.geometry import Polygon
from zarr.core.dtype import VariableLengthUTF8

from API.constants.shared_const import INGEST_VERSION_STR

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/WMO_Alerts"
)
forecast_process_path = forecast_process_dir + "/WMO_Alerts_Process"
hist_process_path = forecast_process_dir + "/WMO_Alerts_Historic"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/WMO_Alerts")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/WMO_Alerts")


save_type = os.getenv("save_type", default="Download")
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

if save_type == "Download":
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)
    if not os.path.exists(historic_path):
        os.makedirs(historic_path)

# Configure a basic logger for debug output.  Consumers of this module
# can override or extend the logging configuration as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _cap_text(elem, tag: str, ns: dict) -> str:
    """Get text for a CAP tag under elem, handling namespaces gracefully."""
    if ns:  # e.g., {'cap': 'urn:oasis:names:tc:emergency:cap:1.2'}
        # Use the prefix and pass the mapping
        return (elem.findtext(f"cap:{tag}", default="", namespaces=ns) or "").strip()
    # No namespace: plain tag
    return (elem.findtext(tag, default="") or "").strip()


def _extract_polygons_from_cap(cap_xml: str, source_id: str, cap_link: str):
    results = []

    try:
        root = ET.fromstring(cap_xml)
    except ET.ParseError as exc:
        print(f"Failed to parse {source_id}: {exc}")
        return results

    # Detect namespace (CAP 1.1 or 1.2)
    ns = {"cap": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}

    # --- Skip duplicate languages ---
    seen_languages = set()

    for info in root.findall(".//cap:info" if ns else ".//info", ns):
        lang_elem = info.find("cap:language" if ns else "language", ns)
        lang = (
            (lang_elem.text or "").strip().lower()
            if lang_elem is not None
            else "unknown"
        )

        # Use only whatever language is first seen
        if not seen_languages:
            seen_languages.add(lang)
        elif lang not in seen_languages:
            seen_languages.add(lang)
            continue  # Skip additional languages

        urgency = _cap_text(info, "urgency", ns)
        if urgency.lower() == "past":  # handle case-insensitive variants
            continue

        # All CAP feeds seem to have the event field, some have headline and description, some just one or the other
        # If there is a headline and description, use headline for event and description for description
        # If there is a headline but no description, use headline for description and event for event
        # If there is a description but no headline, use description for description and event for event
        # Treat blank strings as missing
        event = _cap_text(info, "event", ns) or None
        headline = _cap_text(info, "headline", ns) or None
        description = _cap_text(info, "description", ns) or None

        description_text = description or headline

        if headline and description:
            event_text = headline
        else:
            event_text = event

        severity = _cap_text(info, "severity", ns)

        # If "effective" is in the CAP, use it; otherwise fall back to "onset"
        if _cap_text(info, "effective", ns):
            effective = _cap_text(info, "effective", ns)
        else:
            effective = _cap_text(info, "onset", ns)
        expires = _cap_text(info, "expires", ns)

        for area in info.findall("cap:area" if ns else "area", ns):
            area_desc = area.findtext(
                "cap:areaDesc" if ns else "areaDesc", "", ns
            ).strip()
            for poly_elem in area.findall("cap:polygon" if ns else "polygon", ns):
                polygon_text = (poly_elem.text or "").strip()
                if not polygon_text:
                    continue
                coords = []
                for part in polygon_text.replace(";", " ").split():
                    if "," in part:
                        lat_str, lon_str = part.split(",", 1)
                    else:
                        continue
                    try:
                        lat, lon = float(lat_str), float(lon_str)
                    except ValueError:
                        continue
                    coords.append((lon, lat))
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    try:
                        poly = Polygon(coords)
                        results.append(
                            (
                                source_id,
                                event_text,
                                description_text,
                                severity,
                                effective,
                                expires,
                                area_desc,
                                poly,
                                cap_link,
                            )
                        )
                    except Exception as e:
                        print(f"Polygon construction failed: {e}")
                        continue

    return results


def find_cap_expires(item, ns):
    # Try with a declared CAP namespace first (most correct)
    cap_uri = ns.get("cap")  # e.g., 'urn:oasis:names:tc:emergency:cap:1.2'
    if cap_uri:
        e = item.find(f"{{{cap_uri}}}expires")
        if e is not None and e.text:
            return e.text.strip()

    # Try using a prefix (works only if feed didn't declare ns and used literal tags)
    e = item.find("cap:expires")
    if e is not None and e.text:
        return e.text.strip()

    # Last resort: scan children and match by localname suffix
    for e in item.iter():
        if e.tag.endswith("expires") and e.text:
            return e.text.strip()

    return None


# %% WMO Alert Processing

# Async HTTP helpers
# -------------------------------
DEFAULT_TIMEOUT = 30
MAX_CONCURRENCY = 20  # tune between 8–32
MAX_RETRIES = 3
BACKOFF_BASE = 0.6


class HttpError(Exception):
    pass


async def _fetch_json(
    session: aiohttp.ClientSession, url: str, timeout: float = DEFAULT_TIMEOUT
) -> Dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with asyncio.timeout(timeout):
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise HttpError(f"{url} -> {resp.status}")
                    return await resp.json()
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            await asyncio.sleep(BACKOFF_BASE * attempt)


async def _fetch_text(
    session: aiohttp.ClientSession, url: str, timeout: float = DEFAULT_TIMEOUT
) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with asyncio.timeout(timeout):
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise HttpError(f"{url} -> {resp.status}")
                    return await resp.text()
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            await asyncio.sleep(BACKOFF_BASE * attempt)


# -------------------------------
# Parsing helpers
# -------------------------------
def _find_feed_namespaces(feed_bytes: bytes) -> Dict[str, str]:
    ns = {}
    for event, elem in ET.iterparse(io.BytesIO(feed_bytes), events=("start-ns",)):
        prefix, uri = elem
        ns[prefix or "default"] = uri
    return ns


def _rss_item_links_and_guids(feed_content: bytes) -> List[Tuple[str, Optional[str]]]:
    try:
        root = ET.fromstring(feed_content)
    except ET.ParseError:
        return []
    out = []
    for item in root.findall(".//item"):
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip() or None
        if link:
            out.append((link, guid))
    return out


# -------------------------------
# Main async pipeline
# -------------------------------
async def gather_cap_polygons_async(timeout: float = 30.0) -> gpd.GeoDataFrame:
    base_url = "https://severeweather.wmo.int"
    sources_url = f"{base_url}/json/sources.json"

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY, ttl_dns_cache=300)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(
        connector=connector, raise_for_status=False
    ) as session:
        sources_data = await _fetch_json(session, sources_url, timeout)
        sources: Iterable[Dict] = sources_data.get("sources", [])

        # Pull the “current alerts” list once
        wmo_all = (
            await _fetch_json(session, f"{base_url}/v2/json/wmo_all.json", timeout)
        ).get("items", [])
        # Build quick lookups
        current_ids = {item.get("id") for item in wmo_all if item.get("id")}
        current_agencies = set()
        for record in wmo_all:
            capURL = record.get("capURL", "") or ""
            url = record.get("url", "") or ""
            if capURL:
                current_agencies.add(capURL.split("/")[0])
            elif url:
                current_agencies.add(url.split("/")[0])

        excluded_ids = {
            "co-ungrd-es",
            "mv-ndmc-en",
            "us-noaa-nws-en-marine",
            "us-noaa-nws-en",
            "cn-cma-xx",
            "mo-smg-xx",
        }

        source_ids: List[str] = []
        for entry in sources:
            src = entry.get("source", {})
            sid = src.get("sourceId")
            status = src.get("capAlertFeedStatus")
            if not sid or status != "operating":
                continue
            if sid in excluded_ids:
                continue
            if sid in current_agencies:
                source_ids.append(sid)

        rows: List[Dict[str, Optional[str]]] = []
        geometries: List[Polygon] = []

        async def process_feed(sid: str):
            feed_url = f"{base_url}/v2/cap-alerts/{sid}/rss.xml"
            async with sem:
                feed_text = await _fetch_text(session, feed_url, timeout)
            if not feed_text:
                return

            feed_bytes = feed_text.encode("utf-8", "ignore")
            items = _rss_item_links_and_guids(feed_bytes)

            # Filter to items that correspond to "current" WMO ids (guid exact match is fastest; fallback loose contains)
            filtered = []
            for link, guid in items:
                if not link:
                    continue
                if guid and guid in current_ids:
                    filtered.append(link)
                elif guid:
                    # guard against feeds where guid differs slightly
                    if any(guid in cid for cid in current_ids):
                        filtered.append(link)

            # Fetch all CAP XMLs for this feed concurrently
            async def fetch_and_extract(cap_link: str):
                async with sem:
                    cap_xml = await _fetch_text(session, cap_link, timeout)
                if not cap_xml:
                    return []
                return _extract_polygons_from_cap(cap_xml, sid, cap_link)

            tasks = [asyncio.create_task(fetch_and_extract(link)) for link in filtered]
            for coro in asyncio.as_completed(tasks):
                try:
                    poly_entries = await coro
                except Exception:
                    continue
                for (
                    src_id,
                    event,
                    description,
                    severity,
                    effective,
                    expires,
                    area_desc,
                    poly,
                    cap_link,
                ) in poly_entries:
                    rows.append(
                        {
                            "source_id": src_id,
                            "event": event,
                            "description": description,
                            "severity": severity,
                            "effective": effective,
                            "expires": expires,
                            "area_desc": area_desc,
                            "URL": cap_link,  # optionally store cap_link; add it to fetch_and_extract signature if you want it
                        }
                    )
                    geometries.append(poly)

        # Kick off all feed tasks
        await asyncio.gather(*(process_feed(sid) for sid in source_ids))

        return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


# %% Main script starts here

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s"
)
wmo_gdf = asyncio.run(gather_cap_polygons_async(timeout=30.0))
print(
    f"Built GeoDataFrame with {len(wmo_gdf)} polygons from {wmo_gdf['source_id'].nunique()} sources."
)

# Then save a zarr zip the same way NWS alerts does

# Create a grid of points, using a 0.125 degree spacing
ys = np.arange(-60, 85, 0.0625)
xs = np.arange(-180, 180, 0.0625)

lons, lats = np.meshgrid(xs, ys)

# Create GeoSeries of Points
gridPointsSeries = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(lons.flatten(), lats.flatten()), crs="EPSG:4326"
)
gridPointsSeries["INDEX"] = gridPointsSeries.index
points_in_polygons = gpd.sjoin(
    gridPointsSeries, wmo_gdf, predicate="within", how="inner"
)

# Create a formatted string ton save all the relevant in the zarr array
points_in_polygons["string"] = (
    points_in_polygons["event"].astype(str) + "}"
    "{"
    + points_in_polygons["description"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["area_desc"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["effective"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["expires"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["severity"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["URL"].astype(str)
)

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
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/WMO_Alerts.zarr.zip", mode="w", compression=0
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/WMO_Alerts.zarr")


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


if save_type == "S3":
    zarr_store.close()

## TEST READ
# Find the index for 45.6060335,-73.7919091
# lat = 16.687474
# az_Lon =82.144343
#
# alerts_lats = np.arange(-60, 85, 0.0625)
# alerts_lons = np.arange(-180, 180, 0.0625)
# abslat = np.abs(alerts_lats - lat)
# abslon = np.abs(alerts_lons - az_Lon)
# alerts_y_p = np.argmin(abslat)
# alerts_x_p = np.argmin(abslon)
#
# # gridPoints_XR2[alerts_y_p, alerts_x_p]
# zip_store_read = zarr.storage.ZipStore(
#     forecast_process_dir + "/WMO_Alerts.zarr.zip", compression=0, mode="r"
# )
# alertsReadTest = zarr.open_array(zip_store_read)
# print(alertsReadTest[alerts_y_p, alerts_x_p])

# Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/WMO_Alerts.zarr.zip",
        forecast_path + "/" + ingest_version + "/WMO_Alerts.zarr.zip",
    )
else:
    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/WMO_Alerts.zarr",
        forecast_path + "/" + ingest_version + "/WMO_Alerts.zarr",
        dirs_exist_ok=True,
    )

# Clean up
shutil.rmtree(forecast_process_dir)
