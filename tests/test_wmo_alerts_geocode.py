import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Conditional import for shapely - only needed for test functions that use it
try:
    from shapely.geometry import Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

    # Mock Polygon class for when shapely is not available
    class Polygon:
        def __init__(self, *args, **kwargs):
            pass


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

    root = ET.fromstring(cap_xml)

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

    results = _extract_polygons_from_cap_test(
        cap_xml, "test-source", "http://test.com/alert"
    )

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

    results = _extract_polygons_from_cap_test(
        cap_xml, "fr-source", "http://test.fr/alert"
    )

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

    results = _extract_polygons_from_cap_test(
        cap_xml, "test-no-geo", "http://test.com/nogeo"
    )

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

    results = _extract_polygons_from_cap_test(
        cap_xml, "test-multi", "http://test.com/multi"
    )

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

    results = _extract_polygons_from_cap_test(
        cap_xml, "test-past", "http://test.com/past"
    )

    assert len(results) == 0  # Should be skipped


def _geocode_to_polygon_test(geocode_value, geocode_name, nuts_gdf):
    """Test version of geocode_to_polygon function.

    This is a copy of the function from WMO_Alerts_Local.py to avoid
    importing the module which has side effects (creates directories).
    """
    if nuts_gdf is None or not geocode_value:
        return None

    if geocode_name == "NUTS3":
        # Direct NUTS3 code lookup - optimized with boolean indexing
        match = nuts_gdf[nuts_gdf["NUTS_ID"] == geocode_value]
        if not match.empty:
            return match.geometry.iloc[0]

    elif geocode_name == "EMMA_ID":
        # EMMA_ID format: [Country][Number] (e.g., IT003, FR433, DE001)
        # Try multiple strategies with early returns for efficiency:

        # Strategy 1: Direct match (some EMMA IDs align with NUTS codes)
        match = nuts_gdf[nuts_gdf["NUTS_ID"] == geocode_value]
        if not match.empty:
            return match.geometry.iloc[0]

        # Strategy 2: Country-based prefix matching
        # Extract country code (first 2 chars)
        if len(geocode_value) >= 2:
            country = geocode_value[:2]

            # Filter by country first to reduce search space
            country_regions = nuts_gdf[nuts_gdf["CNTR_CODE"] == country]

            if not country_regions.empty:
                # Strategy 3: Prefix matching for NUTS2 alignment
                # EMMA regions often align with NUTS2, so try prefix matching
                nuts2_prefix = (
                    geocode_value[:4] if len(geocode_value) >= 4 else geocode_value[:3]
                )
                prefix_match = country_regions[
                    country_regions["NUTS_ID"].str.startswith(nuts2_prefix)
                ]

                if not prefix_match.empty:
                    # Use union of matching regions for better coverage
                    return prefix_match.geometry.union_all()

                # Last resort: return first matching country region
                # This is very approximate but better than excluding the alert
                return country_regions.geometry.iloc[0]

    # Fallback: try direct lookup regardless of geocode_name
    match = nuts_gdf[nuts_gdf["NUTS_ID"] == geocode_value]
    if not match.empty:
        return match.geometry.iloc[0]

    return None


def test_geocode_to_polygon_nuts3():
    """Test NUTS3 geocode to polygon conversion with mocked NUTS GeoDataFrame.

    This test uses a mock GeoDataFrame to test the geocode_to_polygon function
    without requiring external HTTP requests to Eurostat.
    """
    # Skip if shapely/geopandas not available (these are ingest dependencies)
    if not SHAPELY_AVAILABLE:
        return

    try:
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return  # Skip test if geopandas not available

    # Create a small mock NUTS GeoDataFrame with test polygons
    mock_nuts_data = {
        "NUTS_ID": ["FR433", "FR101", "IT003", "DE001", "IT001", "IT002"],
        "CNTR_CODE": ["FR", "FR", "IT", "DE", "IT", "IT"],
        "geometry": [
            ShapelyPolygon(
                [(6.0, 47.0), (7.0, 47.0), (7.0, 48.0), (6.0, 48.0)]
            ),  # FR433 Haute-Saône
            ShapelyPolygon(
                [(2.0, 48.5), (3.0, 48.5), (3.0, 49.5), (2.0, 49.5)]
            ),  # FR101 Paris
            ShapelyPolygon(
                [(9.0, 45.0), (10.0, 45.0), (10.0, 46.0), (9.0, 46.0)]
            ),  # IT003 Lombardia approx
            ShapelyPolygon(
                [(6.0, 50.0), (7.0, 50.0), (7.0, 51.0), (6.0, 51.0)]
            ),  # DE001
            ShapelyPolygon(
                [(7.0, 45.0), (8.0, 45.0), (8.0, 46.0), (7.0, 46.0)]
            ),  # IT001
            ShapelyPolygon(
                [(8.0, 45.0), (9.0, 45.0), (9.0, 46.0), (8.0, 46.0)]
            ),  # IT002
        ],
    }
    mock_nuts_gdf = gpd.GeoDataFrame(mock_nuts_data, crs="EPSG:4326")

    # Test NUTS3 direct lookup - should find FR433
    result = _geocode_to_polygon_test("FR433", "NUTS3", mock_nuts_gdf)
    assert result is not None
    assert result.is_valid

    # Test NUTS3 lookup for FR101 (Paris)
    result = _geocode_to_polygon_test("FR101", "NUTS3", mock_nuts_gdf)
    assert result is not None
    assert result.is_valid

    # Test NUTS3 lookup for non-existent code
    result = _geocode_to_polygon_test("XX999", "NUTS3", mock_nuts_gdf)
    assert result is None

    # Test with None GeoDataFrame
    result = _geocode_to_polygon_test("FR433", "NUTS3", None)
    assert result is None

    # Test with empty geocode value
    result = _geocode_to_polygon_test("", "NUTS3", mock_nuts_gdf)
    assert result is None


def test_geocode_to_polygon_emma_id():
    """Test EMMA_ID geocode to polygon conversion with mocked NUTS GeoDataFrame.

    This test uses a mock GeoDataFrame to test the EMMA_ID conversion logic
    without requiring external HTTP requests to Eurostat.
    """
    # Skip if shapely/geopandas not available (these are ingest dependencies)
    if not SHAPELY_AVAILABLE:
        return

    try:
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return  # Skip test if geopandas not available

    # Create a mock NUTS GeoDataFrame with Italian regions for EMMA_ID testing
    mock_nuts_data = {
        "NUTS_ID": ["IT003", "ITC41", "ITC42", "ITC43", "FR433", "DE001"],
        "CNTR_CODE": ["IT", "IT", "IT", "IT", "FR", "DE"],
        "geometry": [
            ShapelyPolygon(
                [(9.0, 45.0), (10.0, 45.0), (10.0, 46.0), (9.0, 46.0)]
            ),  # IT003 direct match
            ShapelyPolygon(
                [(9.5, 45.5), (10.5, 45.5), (10.5, 46.5), (9.5, 46.5)]
            ),  # ITC41
            ShapelyPolygon(
                [(8.5, 45.0), (9.5, 45.0), (9.5, 46.0), (8.5, 46.0)]
            ),  # ITC42
            ShapelyPolygon(
                [(8.0, 44.5), (9.0, 44.5), (9.0, 45.5), (8.0, 45.5)]
            ),  # ITC43
            ShapelyPolygon(
                [(6.0, 47.0), (7.0, 47.0), (7.0, 48.0), (6.0, 48.0)]
            ),  # FR433
            ShapelyPolygon(
                [(6.0, 50.0), (7.0, 50.0), (7.0, 51.0), (6.0, 51.0)]
            ),  # DE001
        ],
    }
    mock_nuts_gdf = gpd.GeoDataFrame(mock_nuts_data, crs="EPSG:4326")

    # Test EMMA_ID with direct match (IT003 exists in NUTS)
    result = _geocode_to_polygon_test("IT003", "EMMA_ID", mock_nuts_gdf)
    assert result is not None
    assert result.is_valid

    # Test EMMA_ID with country-based fallback (IT999 doesn't exist but IT country does)
    result = _geocode_to_polygon_test("IT999", "EMMA_ID", mock_nuts_gdf)
    assert result is not None  # Should fall back to first Italian region
    assert result.is_valid

    # Test EMMA_ID for non-existent country
    result = _geocode_to_polygon_test("XX999", "EMMA_ID", mock_nuts_gdf)
    assert result is None

    # Test with None GeoDataFrame
    result = _geocode_to_polygon_test("IT003", "EMMA_ID", None)
    assert result is None


def test_geocode_to_polygon_unsupported_type():
    """Test that unsupported geocode types return None but are logged.

    This verifies that unsupported types like AMOC-AreaCode don't cause errors
    and return None gracefully.
    """
    # Skip if shapely/geopandas not available
    if not SHAPELY_AVAILABLE:
        return

    try:
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return

    # Create a minimal mock GeoDataFrame
    mock_nuts_data = {
        "NUTS_ID": ["FR433"],
        "CNTR_CODE": ["FR"],
        "geometry": [
            ShapelyPolygon([(6.0, 47.0), (7.0, 47.0), (7.0, 48.0), (6.0, 48.0)]),
        ],
    }
    mock_nuts_gdf = gpd.GeoDataFrame(mock_nuts_data, crs="EPSG:4326")

    # Test AMOC-AreaCode (Australian) - should return None
    result = _geocode_to_polygon_test("NSW_FW001", "AMOC-AreaCode", mock_nuts_gdf)
    assert result is None

    # Test UGC (US/Canada) - should return None
    result = _geocode_to_polygon_test("CAZ006", "UGC", mock_nuts_gdf)
    assert result is None

    # Test SAME (US FIPS) - should return None
    result = _geocode_to_polygon_test("006001", "SAME", mock_nuts_gdf)
    assert result is None


def test_extract_french_nuts3_multi_area_alert():
    """Test extraction of a French alert with multiple NUTS3 geocodes.

    This test uses a real-world French Meteo-France alert that has multiple
    areas with NUTS3 geocodes but no polygon data. Based on the alert:
    https://github.com/user-attachments/files/23634546/07-39995720e024962746976cbdafc71a4f.xml
    """
    # Simplified version of the French alert with key areas
    cap_xml = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>2.49.0.0.250.0.FR.20251119160107.266025</identifier>
  <sender>vigilance@meteo.fr</sender>
  <sent>2025-11-19T16:01:07+01:00</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-GB</language>
    <category>Met</category>
    <event>Moderate snow-ice warning</event>
    <responseType>Monitor</responseType>
    <urgency>Future</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2025-11-20T00:00:00+01:00</effective>
    <onset>2025-11-20T00:00:00+01:00</onset>
    <expires>2025-11-21T00:00:00+01:00</expires>
    <senderName>METEO-FRANCE</senderName>
    <headline>Moderate snow-ice warning</headline>
    <description>Moderate damages may occur.</description>
    <instruction>Be careful, keep informed.</instruction>
    <web>https://vigilance.meteofrance.fr/</web>
    <contact>METEO-FRANCE</contact>
    <area>
      <areaDesc>Aisne</areaDesc>
      <geocode>
        <valueName>NUTS3</valueName>
        <value>FR221</value>
      </geocode>
    </area>
    <area>
      <areaDesc>Paris</areaDesc>
      <geocode>
        <valueName>NUTS3</valueName>
        <value>FR101</value>
      </geocode>
    </area>
    <area>
      <areaDesc>Haute-Saône</areaDesc>
      <geocode>
        <valueName>NUTS3</valueName>
        <value>FR433</value>
      </geocode>
    </area>
    <area>
      <areaDesc>Savoie</areaDesc>
      <geocode>
        <valueName>NUTS3</valueName>
        <value>FR717</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(
        cap_xml, "fr-meteofrance-en", "https://vigilance.meteofrance.fr/alert.xml"
    )

    # Should have 4 areas, all with NUTS3 geocodes
    assert len(results) == 4

    # Check that all results have the expected structure
    for result in results:
        assert result[0] == "fr-meteofrance-en"  # source_id
        assert result[1] == "Moderate snow-ice warning"  # event
        assert result[3] == "Moderate"  # severity
        assert "2025-11-20" in result[4]  # effective date
        assert "2025-11-21" in result[5]  # expires date
        assert result[7] is None  # No polygon (only geocode)
        assert result[9] == "NUTS3"  # geocode_name
        assert result[10].startswith("FR")  # geocode_value is a French NUTS3 code

    # Check specific area descriptions and geocodes
    area_descs = [r[6] for r in results]
    geocodes = [r[10] for r in results]

    assert "Paris" in area_descs
    assert "Haute-Saône" in area_descs
    assert "FR101" in geocodes  # Paris
    assert "FR433" in geocodes  # Haute-Saône
    assert "FR717" in geocodes  # Savoie
    assert "FR221" in geocodes  # Aisne


def test_extract_italian_emma_id_alert():
    """Test extraction of an Italian alert with EMMA_ID geocode.

    This test uses a real-world Italian MeteoAlarm alert that has
    EMMA_ID geocode but no polygon data. Based on the alert:
    https://github.com/user-attachments/files/23634547/49-d0899b149ee9273d569ed762d735ce6e.xml
    """
    cap_xml = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>2.49.0.0.380.3.IT.251119104948.038</identifier>
  <sender>aerocnmca.1sv.prv1@aeronautica.difesa.it</sender>
  <sent>2025-11-19T10:49:49+01:00</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-GB</language>
    <category>Met</category>
    <event>Yellow Snow-ice Warning</event>
    <responseType>Monitor</responseType>
    <urgency>Future</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <audience>Private</audience>
    <effective>2025-11-20T01:00:00+01:00</effective>
    <onset>2025-11-20T01:00:00+01:00</onset>
    <expires>2025-11-21T00:59:00+01:00</expires>
    <senderName>Italian Air Force National Meteorological Service</senderName>
    <headline>Yellow Snow-ice Warning for Italy - Lombardia</headline>
    <description>Moderate intensity weather phenomena expected</description>
    <instruction>BE AWARE, keep up to date with the latest weather forecast.</instruction>
    <web>https://meteoalarm.org/en/live/region/IT?s=lombardia</web>
    <area>
      <areaDesc>Lombardia</areaDesc>
      <geocode>
        <valueName>EMMA_ID</valueName>
        <value>IT003</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(
        cap_xml, "it-meteoam-en", "https://meteoalarm.org/alert.xml"
    )

    # Should have 1 area with EMMA_ID geocode
    assert len(results) == 1

    result = results[0]
    assert result[0] == "it-meteoam-en"  # source_id
    # When headline and description both exist, headline becomes the event
    assert (
        result[1] == "Yellow Snow-ice Warning for Italy - Lombardia"
    )  # event (headline)
    assert result[3] == "Moderate"  # severity
    assert "2025-11-20" in result[4]  # effective date
    assert "2025-11-21" in result[5]  # expires date
    assert result[6] == "Lombardia"  # area description
    assert result[7] is None  # No polygon (only geocode)
    assert result[9] == "EMMA_ID"  # geocode_name
    assert result[10] == "IT003"  # geocode_value (Lombardia)
