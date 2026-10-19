"""WMO CAP area parsing and MeteoAlarm geocode-to-polygon resolution."""

import pytest

from API import WMO_Alerts_Local as wmo

# Row layout returned by _extract_polygons_from_cap.
AREA_DESC, POLYGON, GEOCODE_NAME, GEOCODE_VALUE = 6, 7, 9, 10


def _cap(areas_xml: str, language: str = "en-CA") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>test-alert</identifier>
  <info>
    <language>{language}</language>
    <event>fog</event>
    <urgency>Future</urgency>
    <severity>Moderate</severity>
    <effective>2025-12-03T12:10:08-00:00</effective>
    <expires>2025-12-03T20:31:08-00:00</expires>
    <headline>yellow advisory - fog - in effect</headline>
    <description>Visibility will be near zero in fog.</description>
    {areas_xml}
  </info>
</alert>"""


def _geocodes(name: str, values: list[str]) -> str:
    return "".join(
        f"<geocode><valueName>{name}</valueName><value>{v}</value></geocode>"
        for v in values
    )


def test_extract_canadian_alert_no_duplicates():
    """Test that Canadian alerts with many geocodes per area don't create duplicates.

    Canadian CAP alerts often have multiple geocodes per area (profile:CAP-CP:Location
    codes for individual municipalities). Each area should produce only ONE result
    when a polygon is present, not one result per geocode (#439).

    This test is based on a real Environment Canada fog advisory. The original
    bug would create 21 duplicate entries (13 from the first area + 8 from the
    second area) instead of the expected 2 entries (one per area).
    """
    location = "profile:CAP-CP:Location:0.3"
    cap_xml = _cap(
        f"""
    <area>
      <areaDesc>North Coast - inland including Terrace</areaDesc>
      <polygon>55.28,-128.02 54.54,-127.69 54.23,-128.61 54.07,-129.09 55.28,-128.02</polygon>
      {_geocodes("layer:EC-MSC-SMC:1.0:CLC", ["089210"])}
      {_geocodes(location, [str(5949011 + i) for i in range(12)])}
    </area>
    <area>
      <areaDesc>North Coast - inland including Kitimat</areaDesc>
      <polygon>53.79,-128.89 53.85,-128.93 54.07,-129.09 54.23,-128.61 53.79,-128.89</polygon>
      {_geocodes("layer:EC-MSC-SMC:1.0:CLC", ["089220"])}
      {_geocodes(location, [str(5945006 + i) for i in range(7)])}
    </area>"""
    )

    rows = wmo._extract_polygons_from_cap(cap_xml, "ca-msc-en", "https://x/a.xml")

    assert [(r[AREA_DESC], r[GEOCODE_NAME], r[GEOCODE_VALUE]) for r in rows] == [
        ("North Coast - inland including Terrace", "", ""),
        ("North Coast - inland including Kitimat", "", ""),
    ]
    assert all(r[POLYGON] is not None and r[POLYGON].is_valid for r in rows)
    assert rows[0][1] == "yellow advisory - fog - in effect"


def test_geocode_only_cap_yields_one_row_per_area_code():
    """Météo-France CAPs carry no polygons, only one NUTS3 geocode per area.
    Each area must become its own geocode row (later resolved to a polygon);
    dropping or merging them would leave départements without their warning.
    Based on a real Météo-France snow-ice warning.
    """
    areas = [
        ("Aisne", "FR221"),
        ("Paris", "FR101"),
        ("Haute-Saône", "FR433"),
        ("Savoie", "FR717"),
    ]
    cap_xml = _cap(
        "".join(
            f"<area><areaDesc>{desc}</areaDesc>{_geocodes('NUTS3', [code])}</area>"
            for desc, code in areas
        ),
        language="en-GB",
    )

    rows = wmo._extract_polygons_from_cap(cap_xml, "fr-meteofrance-en", "u")

    assert [
        (r[AREA_DESC], r[POLYGON], r[GEOCODE_NAME], r[GEOCODE_VALUE]) for r in rows
    ] == [(desc, None, "NUTS3", code) for desc, code in areas]


@pytest.fixture(scope="module")
def meteoalarm_gdf():
    gdf = wmo.load_meteoalarm_geocodes()
    assert gdf is not None, "bundled API/data MeteoAlarm geocodes failed to load"
    return gdf


@pytest.mark.parametrize(
    ("code", "name", "expected_centroid"),
    [
        # FR433 is only an alias of EMMA_ID FR082 in meteoalarm_aliases.csv.
        pytest.param("FR433", "NUTS3", (6.094, 47.642), id="nuts3-alias"),
        pytest.param("IT003", "EMMA_ID", (9.769, 45.62), id="emma-id-lombardia"),
        pytest.param("CAZ006", "UGC", None, id="unsupported-type"),
        pytest.param("XX999", "NUTS3", None, id="unknown-code"),
    ],
)
def test_geocode_resolves_to_bundled_meteoalarm_polygon(
    meteoalarm_gdf, code, name, expected_centroid
):
    """Geocode-only European alerts are only gridded if their code resolves
    to a polygon from the bundled MeteoAlarm data, including NUTS3 codes that
    exist only via the alias table; unsupported or unknown codes must yield
    None rather than a wrong region.
    """
    polygon = wmo.geocode_to_polygon(code, name, meteoalarm_gdf)

    if expected_centroid is None:
        assert polygon is None
    else:
        assert polygon.is_valid
        centroid = polygon.centroid
        assert (round(centroid.x, 3), round(centroid.y, 3)) == expected_centroid
