import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapely.geometry import Polygon


def _cap_text(elem, tag: str, ns: dict) -> str:
    """Get text for a CAP tag under elem, handling namespaces gracefully."""
    if ns:  # e.g., {'cap': 'urn:oasis:names:tc:emergency:cap:1.2'}
        # Use the prefix and pass the mapping
        return (elem.findtext(f"cap:{tag}", default="", namespaces=ns) or "").strip()
    # No namespace: plain tag
    return (elem.findtext(tag, default="") or "").strip()


def _extract_polygons_from_cap_test(cap_xml: str, source_id: str, cap_link: str):
    """Test version of _extract_polygons_from_cap - copied from WMO_Alerts_Local.py"""
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

            # Extract geocode information if present
            geocode_name = ""
            geocode_value = ""
            for geocode_elem in area.findall("cap:geocode" if ns else "geocode", ns):
                value_name = geocode_elem.findtext(
                    "cap:valueName" if ns else "valueName", "", ns
                ).strip()
                value = geocode_elem.findtext(
                    "cap:value" if ns else "value", "", ns
                ).strip()
                if value_name and value:
                    geocode_name = value_name
                    geocode_value = value
                    break  # Use the first geocode found

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
                                geocode_name,
                                geocode_value,
                            )
                        )
                    except Exception as e:
                        print(f"Polygon construction failed: {e}")
                        continue

            # If no polygon was found but geocode exists, still create an entry
            if not has_polygon and geocode_name and geocode_value:
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


def test_extract_geocode_with_polygon():
    """Test extraction of geocode from CAP XML with polygon."""
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <info>
    <language>en-US</language>
    <event>Severe Weather Warning</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Observed</certainty>
    <effective>2025-01-01T00:00:00Z</effective>
    <expires>2025-01-01T12:00:00Z</expires>
    <headline>Test Alert</headline>
    <description>Test description</description>
    <area>
      <areaDesc>Test Area</areaDesc>
      <polygon>45.0,9.0 45.0,10.0 46.0,10.0 46.0,9.0 45.0,9.0</polygon>
      <geocode>
        <valueName>EMMA_ID</valueName>
        <value>IT003</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(cap_xml, "test-source", "http://test.com/alert")

    assert len(results) == 1
    result = results[0]
    assert result[0] == "test-source"  # source_id
    assert result[1] == "Test Alert"  # event_text
    assert result[2] == "Test description"  # description_text
    assert result[3] == "Severe"  # severity
    assert result[4] == "2025-01-01T00:00:00Z"  # effective
    assert result[5] == "2025-01-01T12:00:00Z"  # expires
    assert result[6] == "Test Area"  # area_desc
    assert result[7] is not None  # polygon (should be a Polygon object)
    assert result[8] == "http://test.com/alert"  # cap_link
    assert result[9] == "EMMA_ID"  # geocode_name
    assert result[10] == "IT003"  # geocode_value


def test_extract_geocode_without_polygon():
    """Test extraction of geocode from CAP XML without polygon."""
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <info>
    <language>fr-FR</language>
    <event>Alerte Météo</event>
    <urgency>Expected</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2025-01-02T00:00:00Z</effective>
    <expires>2025-01-02T12:00:00Z</expires>
    <headline>Test French Alert</headline>
    <description>Test French description</description>
    <area>
      <areaDesc>Haute-Saône</areaDesc>
      <geocode>
        <valueName>NUTS3</valueName>
        <value>FR433</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(cap_xml, "fr-source", "http://test.fr/alert")

    assert len(results) == 1
    result = results[0]
    assert result[0] == "fr-source"  # source_id
    assert result[1] == "Test French Alert"  # event_text
    assert result[2] == "Test French description"  # description_text
    assert result[3] == "Moderate"  # severity
    assert result[4] == "2025-01-02T00:00:00Z"  # effective
    assert result[5] == "2025-01-02T12:00:00Z"  # expires
    assert result[6] == "Haute-Saône"  # area_desc
    assert result[7] is None  # polygon (should be None for geocode-only)
    assert result[8] == "http://test.fr/alert"  # cap_link
    assert result[9] == "NUTS3"  # geocode_name
    assert result[10] == "FR433"  # geocode_value


def test_extract_no_geocode_with_polygon():
    """Test extraction without geocode but with polygon."""
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <info>
    <language>en-US</language>
    <event>Test Event</event>
    <urgency>Immediate</urgency>
    <severity>Minor</severity>
    <certainty>Observed</certainty>
    <effective>2025-01-03T00:00:00Z</effective>
    <expires>2025-01-03T12:00:00Z</expires>
    <headline>Test Headline</headline>
    <description>Test description</description>
    <area>
      <areaDesc>Region without geocode</areaDesc>
      <polygon>40.0,8.0 40.0,9.0 41.0,9.0 41.0,8.0 40.0,8.0</polygon>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(cap_xml, "test-no-geo", "http://test.com/nogeo")

    assert len(results) == 1
    result = results[0]
    assert result[0] == "test-no-geo"  # source_id
    assert result[7] is not None  # polygon (should be present)
    assert result[9] == ""  # geocode_name (should be empty)
    assert result[10] == ""  # geocode_value (should be empty)


def test_extract_multiple_geocodes_uses_first():
    """Test that when multiple geocodes exist, only the first is used."""
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <info>
    <language>it-IT</language>
    <event>Multiple Geocode Test</event>
    <urgency>Future</urgency>
    <severity>Minor</severity>
    <certainty>Possible</certainty>
    <effective>2025-01-04T00:00:00Z</effective>
    <expires>2025-01-04T12:00:00Z</expires>
    <headline>Test Multiple</headline>
    <description>Test description</description>
    <area>
      <areaDesc>Lombardia</areaDesc>
      <polygon>45.0,9.0 45.0,10.0 46.0,10.0 46.0,9.0 45.0,9.0</polygon>
      <geocode>
        <valueName>EMMA_ID</valueName>
        <value>IT003</value>
      </geocode>
      <geocode>
        <valueName>OTHER_CODE</valueName>
        <value>OTHER_VALUE</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(cap_xml, "test-multi", "http://test.com/multi")

    assert len(results) == 1
    result = results[0]
    assert result[9] == "EMMA_ID"  # geocode_name (first one)
    assert result[10] == "IT003"  # geocode_value (first one)


def test_extract_past_urgency_skipped():
    """Test that alerts with urgency 'past' are skipped."""
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <info>
    <language>en-US</language>
    <event>Past Event</event>
    <urgency>Past</urgency>
    <severity>Severe</severity>
    <certainty>Observed</certainty>
    <effective>2025-01-01T00:00:00Z</effective>
    <expires>2025-01-01T12:00:00Z</expires>
    <headline>Past Alert</headline>
    <description>Past description</description>
    <area>
      <areaDesc>Past Area</areaDesc>
      <geocode>
        <valueName>TEST_CODE</valueName>
        <value>TEST123</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(cap_xml, "test-past", "http://test.com/past")

    assert len(results) == 0  # Should be skipped

