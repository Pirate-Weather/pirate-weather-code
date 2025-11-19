"""Helper functions for WMO alert ingest testing.

This module extracts specific functions from WMO_Alerts_Local.py that are needed
for testing without executing the full ingest pipeline.
"""

from xml.etree import ElementTree as ET

from shapely.geometry import Polygon


def _cap_text(elem, tag: str, ns: dict) -> str:
    """Get text for a CAP tag under elem, handling namespaces gracefully."""
    if ns:  # e.g., {'cap': 'urn:oasis:names:tc:emergency:cap:1.2'}
        # Use the prefix and pass the mapping
        return (elem.findtext(f"cap:{tag}", default="", namespaces=ns) or "").strip()
    # No namespace: plain tag
    return (elem.findtext(tag, default="") or "").strip()


def _extract_polygons_from_cap(cap_xml: str, source_id: str, cap_link: str):
    """Extract polygons from a CAP XML document.

    This is a copy of the function from WMO_Alerts_Local.py for testing purposes.
    """
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
