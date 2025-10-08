"""
Utility for building a GeoDataFrame of alert coverage polygons from the
World Meteorological Organization (WMO) Severe Weather Information
Centre (SWIC) RSS feeds.

The SWIC portal publishes a page of RSS feeds where each feed
corresponds to a particular alerting authority (e.g. a national
meteorological service).  Each feed item links to an individual CAP
(Common Alerting Protocol) message.  CAP messages can contain one or
more `<area>` elements with embedded `<polygon>` tags describing the
geographic coverage of the alert.  This module provides a function
that walks through all of the feeds advertised on the SWIC “Alert
Hub CAP Feeds” page, fetches the CAP messages and extracts the
coverage polygons into a single GeoDataFrame.

Notes
-----
* The function relies on the ``requests``, ``shapely`` and
  ``geopandas`` packages.  These packages must be installed in the
  Python environment where this module is executed.
* Polygons are returned in geographic (longitude/latitude) order and
  use EPSG:4326 as the default CRS.
* Sources that are marked as non‑operational (``capAlertFeedStatus``
  not equal to ``"operating"``) in the WMO ``sources.json`` are
  skipped.  Two additional source identifiers (``co-ungrd-es`` and
  ``mv-ndmc-en``) are explicitly excluded because the SWIC site
  omits them from the feed table.
* The CAP protocol allows multiple polygons and multiple areas per
  alert.  Each polygon becomes a separate row in the resulting
  GeoDataFrame with a corresponding source identifier, event name and
  area description.

Example
-------
#>>> from API.alert_polygons import build_alert_polygon_geodataframe
#>>> gdf = build_alert_polygon_geodataframe()
#>>> print(gdf.head())
"""

from __future__ import annotations

import io
import json
import logging
from typing import Iterable, List, Dict, Optional

import requests
import xml.etree.ElementTree as ET

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except ImportError as exc:
    raise ImportError(
        "The 'geopandas' and 'shapely' libraries are required to use "
        "this module. Please install them via pip or conda."
    ) from exc


# Configure a basic logger for debug output.  Consumers of this module
# can override or extend the logging configuration as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _fetch_json(url: str, timeout: float = 30.0) -> Dict:
    """Helper to fetch JSON data from a URL.

    Parameters
    ----------
    url : str
        The URL to request.
    timeout : float, optional
        Request timeout in seconds, by default 30 seconds.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    requests.HTTPError
        If the request results in a non‑200 response.
    ValueError
        If the response body cannot be decoded as JSON.
    """
    logger.debug("Fetching JSON from %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON from %s", url)
        raise ValueError(f"Invalid JSON received from {url}") from exc


def _extract_polygons_from_cap(cap_xml: str, source_id: str):
    from xml.etree import ElementTree as ET
    from shapely.geometry import Polygon

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
        if lang in seen_languages:
            continue  # Skip duplicate language section
        seen_languages.add(lang)

        urgency = info.findtext("cap:urgency" if ns else "urgency", "").strip()
        if urgency == "Past":
            continue

        event = info.findtext("cap:event" if ns else "event", "").strip()
        headline = info.findtext("cap:headline" if ns else "headline", "").strip()
        description = info.findtext(
            "cap:description" if ns else "description", ""
        ).strip()
        severity = info.findtext("cap:severity" if ns else "severity", "").strip()
        effective = info.findtext("cap:effective" if ns else "effective", "").strip()
        expires = info.findtext("cap:expires" if ns else "expires", "").strip()

        for area in info.findall("cap:area" if ns else "area", ns):
            area_desc = area.findtext("cap:areaDesc" if ns else "areaDesc", "").strip()
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
                                event,
                                headline,
                                description,
                                severity,
                                effective,
                                expires,
                                area_desc,
                                poly,
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


def build_alert_polygon_geodataframe(timeout: float = 30.0) -> gpd.GeoDataFrame:
    """Retrieve CAP alert polygons from all RSS feeds and return them as a GeoDataFrame.

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
    base_url = "https://severeweather.wmo.int"
    sources_url = f"{base_url}/json/sources.json"

    # Download the list of sources
    try:
        sources_data = _fetch_json(sources_url, timeout=timeout)
    except Exception as exc:
        logger.error("Failed to fetch sources JSON from %s: %s", sources_url, exc)
        raise
    sources: Iterable[Dict] = sources_data.get("sources", [])

    # Download list of current alerts:
    wmo_all_data = _fetch_json(
        "https://severeweather.wmo.int" + "/v2/json/wmo_all.json", timeout=timeout
    )
    wmo_all: Iterable[Dict] = wmo_all_data.get("items", [])

    # Loop through each wmo record and build a list of agencies and ids
    wmo_id = []
    agency = []
    for record in wmo_all:
        wmo_id.append(record.get("id"))
        # Agency is everything before the first / in the capURL or url, whichever is present
        capURL = record.get("capURL", "")
        url = record.get("url", "")
        if capURL:
            agency.append(capURL.split("/")[0])
        elif url:
            agency.append(url.split("/")[0])

    # Find unique agencies
    unique_agency = list(set(agency))
    unique_ids = list(set(wmo_id))

    # Build a list of source IDs to query.  Only include feeds marked
    # as operating.  Skip certain identifiers that are excluded from
    # the SWIC table. Also exclude NWS since it's handled separately.
    # Also exclude China CMA since their CAPs do not have locations.
    # Also exclude Hong Kong and Macao since their CAPs do not have expiry times.
    excluded_ids = {
        "co-ungrd-es",
        "mv-ndmc-en",
        "us-noaa-nws-en-marine",
        "us-noaa-nws-en",
        "cn-cma-xx",
        "mo-smg-xx",
    }
    # Count the number of unique ids, excluding ones from an excluded agency

    source_ids: List[str] = []
    for entry in sources:
        src = entry.get("source", {})
        sid = src.get("sourceId")
        status = src.get("capAlertFeedStatus")
        if not sid or status != "operating":
            continue
        if sid in excluded_ids:
            continue
        if (
            sid in unique_agency
        ):  # Only include if there is a current alert from that agency
            source_ids.append(sid)

    # Container for results
    rows: List[Dict[str, Optional[str]]] = []
    geometries: List[Polygon] = []

    for sid in source_ids:
        print(sid)
        feed_url = f"{base_url}/v2/cap-alerts/{sid}/rss.xml"
        logger.info("Processing feed %s", feed_url)
        try:
            feed_resp = requests.get(feed_url, timeout=timeout)
            feed_resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to fetch feed for %s: %s", sid, exc)
            continue

        # Parse the RSS feed.  We'll use ElementTree since feedparser
        # will ignore the CAP-specific elements.
        try:
            feed_root = ET.fromstring(feed_resp.content)
        except ET.ParseError as exc:
            logger.warning("Failed to parse RSS feed for %s: %s", sid, exc)
            continue

        # Get the namespace for the overall feed
        ns = {}
        for event, elem in ET.iterparse(
            io.BytesIO(feed_resp.content), events=("start-ns",)
        ):
            prefix, uri = elem
            # Empty prefix means default ns; store as 'default' so we can inspect
            ns[prefix or "default"] = uri

        # Iterate over all items
        itemCount = 0
        geomCount = 0
        for item in feed_root.findall(".//item"):
            link_elem = item.find("link")
            cap_link = (
                link_elem.text.strip()
                if link_elem is not None and link_elem.text
                else None
            )
            if not cap_link:
                continue

            # Check if the alert guid is in the current wmo_all list
            guid_elem = item.find("guid")
            guid = (
                guid_elem.text.strip()
                if guid_elem is not None and guid_elem.text
                else None
            )

            if not any(guid in id for id in unique_ids):
                continue

            try:
                cap_resp = requests.get(cap_link, timeout=timeout)
                cap_resp.raise_for_status()
            except Exception as exc:
                logger.debug("Failed to fetch CAP message %s: %s", cap_link, exc)
                continue
            # Extract polygons from this CAP
            itemCount = itemCount + 1
            poly_entries = _extract_polygons_from_cap(cap_resp.text, sid)

            for (
                src_id,
                event,
                headline,
                description,
                severity,
                effective,
                expires,
                area_desc,
                poly,
            ) in poly_entries:
                rows.append(
                    {
                        "source_id": src_id,
                        "event": event,
                        "headline": headline,
                        "description": description,
                        "severity": severity,
                        "effective": effective,
                        "expires": expires,
                        "area_desc": area_desc,
                        "URL": cap_link,
                    }
                )
                geometries.append(poly)

                geomCount = geomCount + 1
        print(
            " Processed "
            + str(itemCount)
            + " items, found "
            + str(geomCount)
            + " polygons"
        )
    # Create the GeoDataFrame
    if not geometries:
        # Return an empty GeoDataFrame with the expected columns if no polygons were found
        return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
    return gdf


# %% TODO: Take the polyongs, generate a grid, and determine which points are inside which polygons.
# Then save a zarr zip the same way NWS alerts does
