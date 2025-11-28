"""
Alexander Rey, October 2025

NOTE: This script processes alerts where the CAP contains polygon data or geocode information.
NOTE: Geocodes (EMMA_ID) are automatically converted to polygons using MeteoAlarm geocodes.
NOTE: US Alerts are handled separately in NWS_Alerts_Local.py, since polygons are not always included in the CAP messages.

Retrieve CAP alert polygons and geocodes from all RSS feeds and return them as a GeoDataFrame.

This convenience function automates the entire workflow:

1. Download the WMO ``sources.json`` file to determine which feed
   identifiers are currently operational.
2. Load MeteoAlarm geocodes from the official MeteoAlarm repository for
   improved EMMA_ID to polygon mapping.
3. For each ``sourceId``, fetch the corresponding RSS feed located
   at ``https://severeweather.wmo.int/v2/cap-alerts/{sourceId}/rss.xml``.
4. Parse each feed for item links and download every CAP XML
   document referenced.
5. Extract all polygons and geocodes defined in the CAP documents.
6. Convert EMMA_ID geocodes to polygon geometries when MeteoAlarm data provides a match.
7. Assemble them into a ``geopandas.GeoDataFrame`` with columns for
   ``source_id``, ``event``, ``area_desc``, ``geocode_name``, ``geocode_value`` and a geometry column.

Parameters
----------
timeout : float, optional
    Socket timeout (in seconds) applied to HTTP requests.  A value
    of 30 seconds is used by default.

Returns
-------
geopandas.GeoDataFrame
    A GeoDataFrame where each row corresponds to a single polygon
    extracted from a CAP message or converted from a geocode.  The geometry column contains
    ``shapely.geometry.Polygon`` objects in EPSG:4326. Geocode information
    is included in ``geocode_name`` and ``geocode_value`` columns when available.

Notes
-----
* If a feed or CAP message fails to download or parse, the error
  is logged and processing continues with subsequent feeds.
* The function may take several minutes to run depending on the
  number of feeds and the volume of CAP alerts published.
* Polygons are not simplified or validated beyond ensuring that
  they contain at least three vertices and are closed.
* Geocode information (EMMA_ID) is extracted and converted to polygon geometries when
    MeteoAlarm geocodes provide a match.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import aiohttp
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import zarr
from shapely.geometry import Polygon
from zarr.core.dtype import VariableLengthUTF8

from API.constants.shared_const import INGEST_VERSION_STR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
METEOALARM_ALIASES_PATH = os.path.join(DATA_DIR, "meteoalarm_aliases.csv")

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
        logger.warning("Failed to parse %s: %s", source_id, exc)
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

            # Extract all geocode entries for this area
            geocode_entries: List[Tuple[Optional[str], Optional[str]]] = []
            seen_geocodes: set[Tuple[str, str]] = set()
            for geocode_elem in area.findall("cap:geocode" if ns else "geocode", ns):
                value_name = geocode_elem.findtext(
                    "cap:valueName" if ns else "valueName", "", ns
                ).strip()
                value = geocode_elem.findtext(
                    "cap:value" if ns else "value", "", ns
                ).strip()
                if not value:
                    continue

                normalized = (value_name.upper(), value.upper())
                if normalized in seen_geocodes:
                    continue
                seen_geocodes.add(normalized)
                geocode_entries.append((value_name or None, value))

            if not geocode_entries:
                geocode_entries.append((None, None))

            # Process polygons if available
            has_polygon = False
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
                        has_polygon = True
                        for geocode_name, geocode_value in geocode_entries:
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
                                    geocode_name or "",
                                    geocode_value or "",
                                )
                            )
                    except Exception as e:
                        logger.warning("Polygon construction failed: %s", e)
                        continue

            # If no polygon was found but geocode exists, still create an entry
            # Use a None/empty polygon placeholder - will be filtered later
            if not has_polygon:
                for geocode_name, geocode_value in geocode_entries:
                    if not geocode_name or not geocode_value:
                        continue
                    # Create a minimal point geometry as placeholder (0,0) - will be filtered
                    # This allows geocode-only alerts to be stored for future processing
                    results.append(
                        (
                            source_id,
                            event_text,
                            description_text,
                            severity,
                            effective,
                            expires,
                            area_desc,
                            None,  # No polygon geometry
                            cap_link,
                            geocode_name,
                            geocode_value,
                        )
                    )

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


def _apply_meteoalarm_aliases(
    gdf: gpd.GeoDataFrame, alias_csv_path: str
) -> gpd.GeoDataFrame:
    """Duplicate MeteoAlarm polygons for alias codes defined in CSV."""
    if gdf is None or gdf.empty:
        return gdf

    if not alias_csv_path or not os.path.exists(alias_csv_path):
        logger.info(
            "MeteoAlarm alias file not found at %s; skipping alias expansion",
            alias_csv_path,
        )
        return gdf

    try:
        alias_df = pd.read_csv(alias_csv_path, dtype=str)
    except Exception as exc:
        logger.warning(
            "Could not read MeteoAlarm alias CSV %s: %s", alias_csv_path, exc
        )
        return gdf

    if alias_df.empty:
        return gdf

    required_columns = {"CODE", "ALIAS_CODE"}
    missing_columns = required_columns.difference(alias_df.columns)
    if missing_columns:
        logger.warning(
            "MeteoAlarm alias CSV missing required columns: %s", missing_columns
        )
        return gdf

    alias_df = alias_df.dropna(subset=["CODE", "ALIAS_CODE"])
    if alias_df.empty:
        return gdf

    alias_df["CODE"] = alias_df["CODE"].str.strip().str.upper()
    alias_df["ALIAS_CODE"] = alias_df["ALIAS_CODE"].str.strip().str.upper()
    alias_df = alias_df[
        (alias_df["CODE"] != "")
        & (alias_df["CODE"] != "NAN")
        & (alias_df["ALIAS_CODE"] != "")
        & (alias_df["ALIAS_CODE"] != "NAN")
    ]
    if alias_df.empty:
        return gdf

    id_column = 'code'
    if id_column is None:
        logger.warning(
            "Could not identify MeteoAlarm geocode column; skipping alias expansion"
        )
        return gdf

    normalized_codes = gdf[id_column].fillna("").astype(str).str.strip()
    normalized_upper = normalized_codes.str.upper()
    existing_codes = {code for code in normalized_upper if code}

    new_rows: List[pd.DataFrame] = []
    added_features = 0

    # Optional alias metadata columns
    alias_type_column = "ALIAS_TYPE" if "ALIAS_TYPE" in alias_df.columns else None

    for code_value, group in alias_df.groupby("CODE"):
        base_mask = normalized_upper == code_value
        if not base_mask.any():
            logger.debug(
                "Alias source code %s not present in MeteoAlarm dataset", code_value
            )
            continue

        base_rows = gdf.loc[base_mask]

        for _, alias_row in group.iterrows():
            alias_code = alias_row["ALIAS_CODE"]
            if alias_code == "NAN" or alias_code in existing_codes:
                continue

            alias_copy = base_rows.copy(deep=True)
            alias_copy[id_column] = alias_code

            if alias_type_column:
                alias_value = alias_row.get(alias_type_column, "")
                alias_value = (
                    alias_value.strip() if isinstance(alias_value, str) else alias_value
                )
            else:
                alias_value = None

            alias_copy["alias_type"] = alias_value
            alias_copy["alias_source_code"] = code_value

            new_rows.append(alias_copy)
            existing_codes.add(alias_code)
            added_features += len(alias_copy)

    if not new_rows:
        return gdf

    combined = pd.concat([gdf, *new_rows], ignore_index=True)
    logger.info(
        "Added %d MeteoAlarm alias polygons for %d alias definitions",
        added_features,
        len(new_rows),
    )
    return gpd.GeoDataFrame(combined, geometry=gdf.geometry.name, crs=gdf.crs)


def load_meteoalarm_geocodes(cache_dir: str = None) -> Optional[gpd.GeoDataFrame]:
    """
    Load MeteoAlarm geocodes from the official MeteoAlarm repository.

    These geocodes provide EMMA_ID to polygon mappings for European weather alerts.
    Uses caching to avoid re-downloading on every run.

    Parameters
    ----------
    cache_dir : str, optional
        Directory to cache the MeteoAlarm geocodes file. If None, uses forecast_process_dir.

    Returns
    -------
    GeoDataFrame or None
        A GeoDataFrame with MeteoAlarm regions and EMMA_ID codes, or None if loading fails.
    """
    if cache_dir is None:
        cache_dir = forecast_process_dir

    cache_file = os.path.join(cache_dir, "meteoalarm_geocodes_cache.geojson")
    cache_max_age_days = 30

    # Check if cached file exists and is recent enough
    use_cache = False
    if os.path.exists(cache_file):
        file_age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if file_age_days < cache_max_age_days:
            use_cache = True
            logger.info(
                "Using cached MeteoAlarm geocodes (age: %.1f days)", file_age_days
            )

    try:
        if use_cache:
            # Load from cache
            gdf = gpd.read_file(cache_file)
            logger.info("Loaded %d MeteoAlarm regions from cache", len(gdf))
        else:
            # Download fresh data
            meteoalarm_url = "https://gitlab.com/meteoalarm-pm-group/documents/-/raw/master/MeteoAlarm_Geocodes_2025_10_06.json?inline=false"
            logger.info("Downloading MeteoAlarm geocodes from GitLab...")
            gdf = gpd.read_file(meteoalarm_url)
            logger.info("Downloaded %d MeteoAlarm regions", len(gdf))

            # Save to cache for future use
            try:
                gdf.to_file(cache_file, driver="GeoJSON")
                logger.info("Cached MeteoAlarm geocodes to %s", cache_file)
            except Exception as cache_error:
                logger.warning("Could not cache MeteoAlarm geocodes: %s", cache_error)

        # Ensure CRS is EPSG:4326
        if gdf.crs is None or gdf.crs.to_string().upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf = _apply_meteoalarm_aliases(gdf, METEOALARM_ALIASES_PATH)
        logger.info("MeteoAlarm geocodes ready (CRS: %s)", gdf.crs)
        return gdf
    except Exception as e:
        logger.warning("Could not load MeteoAlarm geocodes: %s", e)
        return None


def geocode_to_polygon(
    geocode_value: str,
    geocode_name: str,
    meteoalarm_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Optional[Polygon]:
    """
    Convert a geocode to a polygon geometry.

    Currently supports EMMA_ID geocodes using MeteoAlarm polygons. Other geocode types are logged
    but not converted to polygons.

    Known unsupported types (will be logged for future analysis):
    - AMOC-AreaCode: Australian weather district codes
    - UGC: Universal Geographic Code (US/Canada zones)
    - SAME: Specific Area Message Encoding (US FIPS codes)
    - Country-specific codes (Japan JMA, China, New Zealand, etc.)

    Uses optimized lookups with early returns to minimize processing time.

    Parameters
    ----------
    geocode_value : str
        The geocode value (e.g., "FR433", "IT003")
    geocode_name : str
        The geocode type (e.g., "EMMA_ID", "AMOC-AreaCode")
    meteoalarm_gdf : GeoDataFrame or None, optional
        GeoDataFrame containing MeteoAlarm geocode boundaries

    Returns
    -------
    Polygon or None
        The polygon geometry for the geocode, or None if not found/supported
    """
    if not geocode_value:
        return None

    # For EMMA_ID, try MeteoAlarm geocodes first (most accurate)
    if (geocode_name == "EMMA_ID" or  geocode_name == "NUTS3") and meteoalarm_gdf is not None:
        try:
            # MeteoAlarm geocodes may have different field names, try common ones
            # Typically they use "emma_id", "EMMA_ID", "geocode", or similar
            match = meteoalarm_gdf[meteoalarm_gdf['code'] == geocode_value]
            if not match.empty:
                logger.info(
                    "Matched %s to %s in MeteoAlarm geocodes",
                    geocode_name,
                    geocode_value,
                )
                return match.geometry.iloc[0]
        except Exception as e:
            logger.warning("Error matching EMMA_ID in MeteoAlarm geocodes: %s", e)

    # Log unsupported geocode types for future analysis
    else:
        logger.info(
            "Geocode type %s with value %s not currently supported for polygon conversion",
            geocode_name,
            geocode_value,
        )

    return None


# -------------------------------
# Main async pipeline
# -------------------------------
async def gather_cap_polygons_async(timeout: float = 30.0) -> gpd.GeoDataFrame:
    base_url = "https://severeweather.wmo.int"
    sources_url = f"{base_url}/json/sources.json"

    # Load MeteoAlarm geocodes for EMMA_ID lookups
    meteoalarm_gdf = load_meteoalarm_geocodes()

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
                    geocode_name,
                    geocode_value,
                ) in poly_entries:
                    # If no polygon but we have a geocode, try to convert it
                    if poly is None and geocode_name and geocode_value:
                        poly = geocode_to_polygon(
                            geocode_value,
                            geocode_name,
                            meteoalarm_gdf,
                        )

                    # Include all entries that have a polygon geometry
                    # (either from CAP or converted from geocode)
                    if poly is not None:
                        rows.append(
                            {
                                "source_id": src_id,
                                "event": event,
                                "description": description,
                                "severity": severity,
                                "effective": effective,
                                "expires": expires,
                                "area_desc": area_desc,
                                "URL": cap_link,
                                "geocode_name": geocode_name,
                                "geocode_value": geocode_value,
                            }
                        )
                        geometries.append(poly)

        # Kick off all feed tasks

        # Use only "cz-chmi-cs" for testing
        # source_ids = ["cz-chmi-cs"]

        await asyncio.gather(*(process_feed(sid) for sid in source_ids))

        return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


# %% Main script starts here

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s"
)
wmo_gdf = asyncio.run(gather_cap_polygons_async(timeout=30.0))
logger.info(
    "Built GeoDataFrame with %d polygons from %d sources.",
    len(wmo_gdf),
    wmo_gdf["source_id"].nunique(),
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
    + "}"
    + "{"
    + points_in_polygons["geocode_name"].astype(str)
    + "}"
    + "{"
    + points_in_polygons["geocode_value"].astype(str)
)

# Combine the formatted strings using "~" as a spacer, since it doesn't seem to be used in CAP messages
df = points_in_polygons.groupby("INDEX").agg({"string": "~".join}).reset_index()


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
# lat = 42.4974694
# az_Lon = -94.1680158

# alerts_lats = np.arange(-60, 85, 0.0625)
# alerts_lons = np.arange(-180, 180, 0.0625)
# abslat = np.abs(alerts_lats - lat)
# abslon = np.abs(alerts_lons - az_Lon)
# alerts_y_p = np.argmin(abslat)
# alerts_x_p = np.argmin(abslon)

# gridPoints_XR2[alerts_y_p, alerts_x_p]

# # Find locations in wmo_gdf where source_id'=='fr-meteofrance-xx'
# fr_meteofrance_alerts = wmo_gdf.loc[wmo_gdf["source_id"] == "fr-meteofrance-xx"]

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
