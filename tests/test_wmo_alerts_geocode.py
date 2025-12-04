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


# Path to local data files
DATA_DIR = Path(__file__).resolve().parent.parent / "API" / "data"
METEOALARM_ALIASES_PATH = DATA_DIR / "meteoalarm_aliases.csv"
METEOALARM_GEOJSON_PATH = DATA_DIR / "meteoalarm_geocodes.json"


def _cap_text(elem, tag: str, ns: dict) -> str:
    """Get text for a CAP tag under elem, handling namespaces gracefully."""
    if ns:  # e.g., {'cap': 'urn:oasis:names:tc:emergency:cap:1.2'}
        # Use the prefix and pass the mapping
        return (elem.findtext(f"cap:{tag}", default="", namespaces=ns) or "").strip()
    # No namespace: plain tag
    return (elem.findtext(tag, default="") or "").strip()


def _extract_polygons_from_cap_test(cap_xml: str, source_id: str, cap_link: str):
    """Test version of _extract_polygons_from_cap - synced with WMO_Alerts_Local.py

    NOTE: This function should be kept in sync with the production code in
    API/WMO_Alerts_Local.py. If you modify this function, update the production
    code as well, and vice versa.
    """
    results = []

    root = ET.fromstring(cap_xml)

    # Detect namespace (CAP 1.1 or 1.2)
    ns = {"cap": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}

    # --- Skip duplicate languages ---
    # We only want to process ONE info block per language to avoid duplicates.
    # Canadian CAP files often have multiple info blocks for the same language.
    seen_languages = set()

    for info in root.findall(".//cap:info" if ns else ".//info", ns):
        lang_elem = info.find("cap:language" if ns else "language", ns)
        lang = (
            (lang_elem.text or "").strip().lower()
            if lang_elem is not None
            else "unknown"
        )

        # Skip if we've already processed an info block for this language
        if lang in seen_languages:
            continue
        seen_languages.add(lang)

        # Only process the first language seen, skip subsequent languages
        if len(seen_languages) > 1:
            continue

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

            # Extract all geocode entries for this area with deduplication
            geocode_entries = []
            seen_geocodes = set()
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
                        # When polygon exists, create only ONE entry per polygon
                        # Geocodes are not needed since the polygon defines the area
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
                                "",
                                "",
                            )
                        )
                    except Exception as e:
                        print(f"Polygon construction failed: {e}")
                        continue

            # If no polygon was found but geocode exists, still create an entry
            if not has_polygon:
                for geocode_name, geocode_value in geocode_entries:
                    if not geocode_name or not geocode_value:
                        continue
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
    assert result[9] == ""  # geocode_name (empty when polygon exists)
    assert result[10] == ""  # geocode_value (empty when polygon exists)


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


def test_extract_multiple_geocodes_no_duplicates():
    """Test that multiple geocodes with polygon produce only one result with no geocode info."""
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
    # When polygon exists, geocodes are not included
    assert result[9] == ""  # geocode_name (empty when polygon exists)
    assert result[10] == ""  # geocode_value (empty when polygon exists)


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


def _load_local_meteoalarm_geocodes():
    """Load meteoalarm geocodes from local bundled files.

    Returns a GeoDataFrame with meteoalarm regions including alias codes.
    This mirrors the load_meteoalarm_geocodes function from WMO_Alerts_Local.py.
    """
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        return None

    if not METEOALARM_GEOJSON_PATH.exists():
        return None

    gdf = gpd.read_file(METEOALARM_GEOJSON_PATH)

    # Ensure CRS is EPSG:4326
    if gdf.crs is None or gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Apply aliases if available
    if METEOALARM_ALIASES_PATH.exists():
        try:
            alias_df = pd.read_csv(METEOALARM_ALIASES_PATH, dtype=str)
            if not alias_df.empty:
                required_columns = {"CODE", "ALIAS_CODE"}
                if required_columns.issubset(alias_df.columns):
                    alias_df = alias_df.dropna(subset=["CODE", "ALIAS_CODE"])
                    alias_df["CODE"] = alias_df["CODE"].str.strip().str.upper()
                    alias_df["ALIAS_CODE"] = (
                        alias_df["ALIAS_CODE"].str.strip().str.upper()
                    )
                    alias_df = alias_df[
                        (alias_df["CODE"] != "")
                        & (alias_df["CODE"] != "NAN")
                        & (alias_df["ALIAS_CODE"] != "")
                        & (alias_df["ALIAS_CODE"] != "NAN")
                    ]

                    # Build alias entries
                    id_column = "code"
                    normalized_codes = gdf[id_column].fillna("").astype(str).str.strip()
                    normalized_upper = normalized_codes.str.upper()
                    existing_codes = {code for code in normalized_upper if code}

                    new_rows = []
                    for code_value, group in alias_df.groupby("CODE"):
                        base_mask = normalized_upper == code_value
                        if not base_mask.any():
                            continue
                        base_rows = gdf.loc[base_mask]
                        for _, alias_row in group.iterrows():
                            alias_code = alias_row["ALIAS_CODE"]
                            if alias_code == "NAN" or alias_code in existing_codes:
                                continue
                            alias_copy = base_rows.copy(deep=True)
                            alias_copy[id_column] = alias_code
                            new_rows.append(alias_copy)
                            existing_codes.add(alias_code)

                    if new_rows:
                        combined = pd.concat([gdf, *new_rows], ignore_index=True)
                        gdf = gpd.GeoDataFrame(
                            combined, geometry=gdf.geometry.name, crs=gdf.crs
                        )
        except Exception:
            pass  # Ignore alias errors

    return gdf


def _geocode_to_polygon_test(geocode_value, geocode_name, meteoalarm_gdf):
    """Test version of geocode_to_polygon function.

    This mirrors the geocode_to_polygon function from WMO_Alerts_Local.py.
    Uses the 'code' column to match geocodes in the MeteoAlarm GeoDataFrame.
    """
    if meteoalarm_gdf is None or not geocode_value:
        return None

    # For EMMA_ID or NUTS3, try MeteoAlarm geocodes
    if geocode_name == "EMMA_ID" or geocode_name == "NUTS3":
        try:
            match = meteoalarm_gdf[meteoalarm_gdf["code"] == geocode_value]
            if not match.empty:
                return match.geometry.iloc[0]
        except Exception:
            pass

    return None


def test_geocode_to_polygon_nuts3():
    """Test NUTS3 geocode to polygon conversion using local MeteoAlarm data.

    This test uses the actual bundled meteoalarm_geocodes.json file with alias
    expansion to test NUTS3 geocode lookups. NUTS3 codes are available through
    the alias mapping (e.g., FR082 -> FR433).
    """
    # Skip if shapely/geopandas not available (these are ingest dependencies)
    if not SHAPELY_AVAILABLE:
        return

    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        return  # Skip test if geopandas not available

    # Load actual local MeteoAlarm geocodes with aliases
    meteoalarm_gdf = _load_local_meteoalarm_geocodes()
    if meteoalarm_gdf is None:
        return  # Skip if data files not available

    # Test NUTS3 lookup for FR433 (Haute-Saône) - available via alias from FR082
    result = _geocode_to_polygon_test("FR433", "NUTS3", meteoalarm_gdf)
    assert result is not None, "FR433 should be found via alias from FR082"
    assert result.is_valid

    # Test NUTS3 lookup for FR101 (Paris) - direct EMMA_ID code
    result = _geocode_to_polygon_test("FR101", "NUTS3", meteoalarm_gdf)
    assert result is not None, "FR101 should be found as direct EMMA_ID"
    assert result.is_valid

    # Test NUTS3 lookup for non-existent code
    result = _geocode_to_polygon_test("XX999", "NUTS3", meteoalarm_gdf)
    assert result is None

    # Test with None GeoDataFrame
    result = _geocode_to_polygon_test("FR433", "NUTS3", None)
    assert result is None

    # Test with empty geocode value
    result = _geocode_to_polygon_test("", "NUTS3", meteoalarm_gdf)
    assert result is None


def test_geocode_to_polygon_emma_id():
    """Test EMMA_ID geocode to polygon conversion using local MeteoAlarm data.

    This test uses the actual bundled meteoalarm_geocodes.json file to test
    EMMA_ID geocode lookups for real regions like IT003 (Lombardia).
    """
    # Skip if shapely/geopandas not available (these are ingest dependencies)
    if not SHAPELY_AVAILABLE:
        return

    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        return  # Skip test if geopandas not available

    # Load actual local MeteoAlarm geocodes
    meteoalarm_gdf = _load_local_meteoalarm_geocodes()
    if meteoalarm_gdf is None:
        return  # Skip if data files not available

    # Test EMMA_ID with direct match (IT003 - Lombardia)
    result = _geocode_to_polygon_test("IT003", "EMMA_ID", meteoalarm_gdf)
    assert result is not None, "IT003 (Lombardia) should be found in MeteoAlarm data"
    assert result.is_valid

    # Test EMMA_ID for FR082 (Haute-Saône) - direct EMMA_ID
    result = _geocode_to_polygon_test("FR082", "EMMA_ID", meteoalarm_gdf)
    assert result is not None, "FR082 (Haute-Saône) should be found in MeteoAlarm data"
    assert result.is_valid

    # Test EMMA_ID for non-existent code
    result = _geocode_to_polygon_test("XX999", "EMMA_ID", meteoalarm_gdf)
    assert result is None

    # Test with None GeoDataFrame
    result = _geocode_to_polygon_test("IT003", "EMMA_ID", None)
    assert result is None


def test_geocode_to_polygon_unsupported_type():
    """Test that unsupported geocode types return None gracefully.

    This verifies that unsupported types like AMOC-AreaCode don't cause errors
    and return None gracefully.
    """
    # Skip if shapely/geopandas not available
    if not SHAPELY_AVAILABLE:
        return

    try:
        import geopandas as gpd  # noqa: F401
    except ImportError:
        return

    # Load actual local MeteoAlarm geocodes
    meteoalarm_gdf = _load_local_meteoalarm_geocodes()
    if meteoalarm_gdf is None:
        return  # Skip if data files not available

    # Test AMOC-AreaCode (Australian) - should return None
    result = _geocode_to_polygon_test("NSW_FW001", "AMOC-AreaCode", meteoalarm_gdf)
    assert result is None

    # Test UGC (US/Canada) - should return None
    result = _geocode_to_polygon_test("CAZ006", "UGC", meteoalarm_gdf)
    assert result is None

    # Test SAME (US FIPS) - should return None
    result = _geocode_to_polygon_test("006001", "SAME", meteoalarm_gdf)
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


def test_extract_canadian_alert_no_duplicates():
    """Test that Canadian alerts with many geocodes per area don't create duplicates.

    Canadian CAP alerts often have multiple geocodes per area (profile:CAP-CP:Location
    codes for individual municipalities). Each area should produce only ONE result
    when a polygon is present, not one result per geocode.

    This test is based on a real Environment Canada fog advisory. The original
    bug would create 21 duplicate entries (13 from the first area + 8 from the
    second area) instead of the expected 2 entries (one per area).
    """
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>test-canadian-alert</identifier>
  <sender>cap-pac@canada.ca</sender>
  <info>
    <language>en-CA</language>
    <event>fog</event>
    <urgency>Future</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2025-12-03T12:10:08-00:00</effective>
    <expires>2025-12-03T20:31:08-00:00</expires>
    <headline>yellow advisory - fog - in effect</headline>
    <description>Visibility will be near zero in fog.</description>
    <area>
      <areaDesc>North Coast - inland including Terrace</areaDesc>
      <polygon>55.28,-128.02 54.54,-127.69 54.23,-128.61 54.07,-129.09 55.28,-128.02</polygon>
      <geocode>
        <valueName>layer:EC-MSC-SMC:1.0:CLC</valueName>
        <value>089210</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949011</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949013</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949018</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949028</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949035</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949039</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949804</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949805</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949807</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949815</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949816</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949844</value>
      </geocode>
    </area>
    <area>
      <areaDesc>North Coast - inland including Kitimat</areaDesc>
      <polygon>53.79,-128.89 53.85,-128.93 54.07,-129.09 54.23,-128.61 53.79,-128.89</polygon>
      <geocode>
        <valueName>layer:EC-MSC-SMC:1.0:CLC</valueName>
        <value>089220</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5945006</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949005</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949013</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949020</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5949803</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5951031</value>
      </geocode>
      <geocode>
        <valueName>profile:CAP-CP:Location:0.3</valueName>
        <value>5951053</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(
        cap_xml, "ca-msc-en", "https://test.gc.ca/alert.xml"
    )

    # Should have exactly 2 results (one per area), NOT 21 (one per geocode)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # First area
    assert results[0][0] == "ca-msc-en"  # source_id
    assert results[0][1] == "yellow advisory - fog - in effect"  # event
    assert results[0][6] == "North Coast - inland including Terrace"  # area_desc
    assert results[0][7] is not None  # polygon should exist
    assert results[0][9] == ""  # geocode_name (empty when polygon exists)
    assert results[0][10] == ""  # geocode_value (empty when polygon exists)

    # Second area
    assert results[1][6] == "North Coast - inland including Kitimat"  # area_desc
    assert results[1][7] is not None  # polygon should exist
    assert results[1][9] == ""  # geocode_name (empty when polygon exists)
    assert results[1][10] == ""  # geocode_value (empty when polygon exists)


def test_extract_duplicate_info_blocks_same_language():
    """Test that duplicate info blocks with the same language don't create duplicates.

    This test is based on a real Canadian Environment Canada alert pattern where
    the same info block appears multiple times with the same language (e.g., en-CA).
    This was causing duplicate alerts to appear in the API output.

    The fix ensures only ONE info block per language is processed, and only the
    first language encountered is used (to avoid processing both en-CA and fr-CA).
    """
    cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>test-barrie-alert</identifier>
  <sender>cap-pac@canada.ca</sender>
  <info>
    <language>en-CA</language>
    <event>Winter Storm Warning</event>
    <urgency>Future</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <effective>2025-12-03T12:00:00-00:00</effective>
    <expires>2025-12-03T20:00:00-00:00</expires>
    <headline>Winter Storm Warning in effect</headline>
    <description>Heavy snow expected.</description>
    <area>
      <areaDesc>Barrie - Orillia</areaDesc>
      <polygon>44.5,-79.5 44.5,-79.0 44.0,-79.0 44.0,-79.5 44.5,-79.5</polygon>
      <geocode>
        <valueName>layer:EC-MSC-SMC:1.0:CLC</valueName>
        <value>061310</value>
      </geocode>
    </area>
  </info>
  <info>
    <language>en-CA</language>
    <event>Winter Storm Warning</event>
    <urgency>Future</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <effective>2025-12-03T12:00:00-00:00</effective>
    <expires>2025-12-03T20:00:00-00:00</expires>
    <headline>Winter Storm Warning in effect</headline>
    <description>Heavy snow expected.</description>
    <area>
      <areaDesc>Barrie - Orillia</areaDesc>
      <polygon>44.5,-79.5 44.5,-79.0 44.0,-79.0 44.0,-79.5 44.5,-79.5</polygon>
      <geocode>
        <valueName>layer:EC-MSC-SMC:1.0:CLC</valueName>
        <value>061310</value>
      </geocode>
    </area>
  </info>
  <info>
    <language>fr-CA</language>
    <event>Avertissement de tempête hivernale</event>
    <urgency>Future</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <effective>2025-12-03T12:00:00-00:00</effective>
    <expires>2025-12-03T20:00:00-00:00</expires>
    <headline>Avertissement de tempête hivernale en vigueur</headline>
    <description>Fortes chutes de neige prévues.</description>
    <area>
      <areaDesc>Barrie - Orillia</areaDesc>
      <polygon>44.5,-79.5 44.5,-79.0 44.0,-79.0 44.0,-79.5 44.5,-79.5</polygon>
      <geocode>
        <valueName>layer:EC-MSC-SMC:1.0:CLC</valueName>
        <value>061310</value>
      </geocode>
    </area>
  </info>
</alert>"""

    results = _extract_polygons_from_cap_test(
        cap_xml, "ca-msc-en", "https://test.gc.ca/barrie-alert.xml"
    )

    # Should have exactly 1 result - only the first en-CA info block should be processed
    # The second en-CA info block and the fr-CA block should be skipped
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    result = results[0]
    assert result[0] == "ca-msc-en"  # source_id
    assert result[1] == "Winter Storm Warning in effect"  # event (headline)
    assert result[3] == "Severe"  # severity
    assert result[6] == "Barrie - Orillia"  # area_desc
    assert result[7] is not None  # polygon should exist
    assert result[9] == ""  # geocode_name (empty when polygon exists)
    assert result[10] == ""  # geocode_value (empty when polygon exists)
