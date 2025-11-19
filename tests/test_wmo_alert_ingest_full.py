"""Test for WMO alert ingest functionality with full dependencies.

This test module requires the ingest dependencies (geopandas, zarr, etc.)
and is marked with @pytest.mark.ingest to allow it to be skipped when
running the main test suite.

To run only ingest tests:
    pytest -v -m ingest

To run all tests except ingest tests:
    pytest -v -m "not ingest"
"""

import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import zarr
from shapely.geometry import Point

# Mark all tests in this module as requiring ingest dependencies
pytestmark = pytest.mark.ingest


def test_wmo_alert_dwd_xml_exists():
    """Test that the German DWD test WMO alert XML file exists."""
    test_xml_path = (
        Path(__file__).resolve().parent / "data" / "test_wmo_alert_dwd_germany.xml"
    )
    assert test_xml_path.exists(), f"Test XML file not found at {test_xml_path}"
    assert test_xml_path.is_file(), "Test XML path is not a file"


def test_extract_polygons_from_dwd_cap():
    """Test extracting polygons from a German DWD CAP XML message.

    This test uses a real-world alert from Deutscher Wetterdienst (DWD)
    for icy surfaces that was reported to have issues appearing in the API.
    """
    from tests.wmo_test_helpers import _extract_polygons_from_cap

    # Read the test XML file
    test_xml_path = (
        Path(__file__).resolve().parent / "data" / "test_wmo_alert_dwd_germany.xml"
    )
    with open(test_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Extract polygons
    source_id = "de-dwd-en"
    cap_link = "https://test.example.com/dwd_alert.xml"
    results = _extract_polygons_from_cap(xml_content, source_id, cap_link)

    # Verify we got results - DWD alert has multiple polygonal areas
    assert len(results) > 0, f"Expected at least 1 polygon, got {len(results)}"

    # Check the structure of the first result
    first_result = results[0]
    assert len(first_result) == 9, "Each result should have 9 elements"

    # Unpack the result
    (
        result_source_id,
        event,
        description,
        severity,
        effective,
        expires,
        area_desc,
        polygon,
        result_cap_link,
    ) = first_result

    # Verify the extracted data from DWD alert
    assert result_source_id == source_id
    assert event == "Official WARNING of ICY SURFACES" or event == "icy surfaces"
    assert description is not None and len(description) > 0
    assert severity == "Minor"
    assert effective == "2025-11-19T20:06:00+01:00"
    assert expires == "2025-11-20T09:00:00+01:00"
    assert area_desc == "polygonal event area"
    assert polygon is not None
    assert result_cap_link == cap_link

    # Verify the polygon is valid
    assert polygon.is_valid, "Polygon should be valid"
    assert polygon.geom_type == "Polygon"

    # Check coordinates are in the expected range (Germany)
    bounds = polygon.bounds
    assert 5.0 <= bounds[0] <= 15.0, f"min longitude {bounds[0]} not in Germany range"
    assert 5.0 <= bounds[2] <= 15.0, f"max longitude {bounds[2]} not in Germany range"
    assert 47.0 <= bounds[1] <= 55.0, f"min latitude {bounds[1]} not in Germany range"
    assert 47.0 <= bounds[3] <= 55.0, f"max latitude {bounds[3]} not in Germany range"


def test_dwd_alert_geodataframe_creation():
    """Test creating a GeoDataFrame from DWD alert data."""
    from tests.wmo_test_helpers import _extract_polygons_from_cap

    # Read the test XML file
    test_xml_path = (
        Path(__file__).resolve().parent / "data" / "test_wmo_alert_dwd_germany.xml"
    )
    with open(test_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Extract polygons
    source_id = "de-dwd-en"
    cap_link = "https://test.example.com/dwd_alert.xml"
    results = _extract_polygons_from_cap(xml_content, source_id, cap_link)

    # Create a GeoDataFrame similar to what the full ingest does
    rows = []
    geometries = []

    for (
        src_id,
        event,
        description,
        severity,
        effective,
        expires,
        area_desc,
        poly,
        url,
    ) in results:
        rows.append(
            {
                "source_id": src_id,
                "event": event,
                "description": description,
                "severity": severity,
                "effective": effective,
                "expires": expires,
                "area_desc": area_desc,
                "URL": url,
            }
        )
        geometries.append(poly)

    gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")

    # Verify GeoDataFrame structure
    assert len(gdf) > 0, "GeoDataFrame should have at least one row"
    assert "source_id" in gdf.columns
    assert "event" in gdf.columns
    assert "description" in gdf.columns
    assert "severity" in gdf.columns
    assert "effective" in gdf.columns
    assert "expires" in gdf.columns
    assert "area_desc" in gdf.columns
    assert "URL" in gdf.columns
    assert "geometry" in gdf.columns

    # Verify CRS
    assert gdf.crs.to_string() == "EPSG:4326"

    # All severities should be "Minor" for this DWD alert
    assert all(gdf["severity"] == "Minor")


def test_dwd_alert_zarr_creation_and_retrieval():
    """Test creating a zarr store from DWD alert data and retrieving it.

    This test validates that the DWD icy surfaces alert can be:
    1. Parsed from CAP XML
    2. Converted to a spatial grid
    3. Stored in zarr format
    4. Retrieved from zarr for locations within the alert area
    """
    from tests.wmo_test_helpers import _extract_polygons_from_cap

    # Read the test XML file
    test_xml_path = (
        Path(__file__).resolve().parent / "data" / "test_wmo_alert_dwd_germany.xml"
    )
    with open(test_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Extract polygons
    source_id = "de-dwd-en"
    cap_link = "https://test.example.com/dwd_alert.xml"
    results = _extract_polygons_from_cap(xml_content, source_id, cap_link)

    # Create a GeoDataFrame
    rows = []
    geometries = []

    for (
        src_id,
        event,
        description,
        severity,
        effective,
        expires,
        area_desc,
        poly,
        url,
    ) in results:
        rows.append(
            {
                "source_id": src_id,
                "event": event,
                "description": description,
                "severity": severity,
                "effective": effective,
                "expires": expires,
                "area_desc": area_desc,
                "URL": url,
            }
        )
        geometries.append(poly)

    wmo_gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")

    # Get bounding box for test grid (cover Germany region)
    all_bounds = wmo_gdf.total_bounds  # (minx, miny, maxx, maxy)

    # Expand bounds slightly and align to grid
    min_lon = np.floor(all_bounds[0] / 0.0625) * 0.0625 - 0.125
    max_lon = np.ceil(all_bounds[2] / 0.0625) * 0.0625 + 0.125
    min_lat = np.floor(all_bounds[1] / 0.0625) * 0.0625 - 0.125
    max_lat = np.ceil(all_bounds[3] / 0.0625) * 0.0625 + 0.125

    # Create a grid of points (using 0.0625° spacing like production)
    ys = np.arange(min_lat, max_lat, 0.0625)
    xs = np.arange(min_lon, max_lon, 0.0625)

    lons, lats = np.meshgrid(xs, ys)

    # Create GeoSeries of Points
    gridPointsSeries = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons.flatten(), lats.flatten()), crs="EPSG:4326"
    )
    gridPointsSeries["INDEX"] = gridPointsSeries.index

    # Spatial join to find points within polygons
    points_in_polygons = gpd.sjoin(
        gridPointsSeries, wmo_gdf, predicate="within", how="inner"
    )

    assert len(points_in_polygons) > 0, "No points found within alert polygons"

    # Create a formatted string to save all relevant data in the zarr array
    points_in_polygons["string"] = (
        points_in_polygons["event"].astype(str)
        + "}{"
        + points_in_polygons["description"].astype(str)
        + "}{"
        + points_in_polygons["area_desc"].astype(str)
        + "}{"
        + points_in_polygons["effective"].astype(str)
        + "}{"
        + points_in_polygons["expires"].astype(str)
        + "}{"
        + points_in_polygons["severity"].astype(str)
        + "}{"
        + points_in_polygons["URL"].astype(str)
    )

    # Combine the formatted strings using "~" as a spacer
    df = points_in_polygons.groupby("INDEX").agg({"string": "~".join}).reset_index()

    # Merge back into primary geodataframe
    gridPointsSeries = gridPointsSeries.merge(df, on="INDEX", how="left")

    # Set empty data as blank
    gridPointsSeries.loc[gridPointsSeries["string"].isna(), ["string"]] = ""

    # Convert to string
    gridPointsSeries["string"] = gridPointsSeries["string"].astype(str)

    # Reshape to 2D
    gridPoints_XR = gridPointsSeries["string"].to_xarray()

    # Test in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "WMO_Alerts.zarr")
        zarr_store = zarr.storage.LocalStore(zarr_path)

        # Create a Zarr array in the store
        from zarr.core.dtype import VariableLengthUTF8

        gridPoints_XR2 = gridPoints_XR.values.astype(VariableLengthUTF8).reshape(
            lons.shape
        )

        zarr_array = zarr.create_array(
            store=zarr_store,
            shape=gridPoints_XR2.shape,
            dtype=zarr.dtype.VariableLengthUTF8(),
            chunks=(10, 10),
            overwrite=True,
        )

        # Save the data
        zarr_array[:] = gridPoints_XR2

        # Verify we can read it back
        read_store = zarr.storage.LocalStore(zarr_path)
        read_array = zarr.open_array(read_store, mode="r")

        # Find a point within one of the alert areas
        # Use the centroid of the first polygon
        test_point = wmo_gdf.geometry.iloc[0].centroid
        test_lat = test_point.y
        test_lon = test_point.x

        alerts_lats = ys
        alerts_lons = xs
        abslat = np.abs(alerts_lats - test_lat)
        abslon = np.abs(alerts_lons - test_lon)
        alerts_y_p = np.argmin(abslat)
        alerts_x_p = np.argmin(abslon)

        # Read the alert data for this point
        alert_data = read_array[alerts_y_p, alerts_x_p]

        # The point should be within one of the alert polygons
        assert alert_data != "", f"Expected alert data at ({test_lat}, {test_lon})"
        assert "icy surfaces" in alert_data.lower() or "icy" in alert_data.lower()
        assert "Minor" in alert_data


def test_dwd_alert_response_integration():
    """Test that DWD alert data can be integrated into a response.

    This test validates that a point query within the DWD alert area
    would correctly return alert information in the API response format.
    """
    from tests.wmo_test_helpers import _extract_polygons_from_cap

    # Read the test XML file
    test_xml_path = (
        Path(__file__).resolve().parent / "data" / "test_wmo_alert_dwd_germany.xml"
    )
    with open(test_xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Extract polygons
    source_id = "de-dwd-en"
    cap_link = "https://test.example.com/dwd_alert.xml"
    results = _extract_polygons_from_cap(xml_content, source_id, cap_link)

    # Create a GeoDataFrame
    rows = []
    geometries = []

    for (
        src_id,
        event,
        description,
        severity,
        effective,
        expires,
        area_desc,
        poly,
        url,
    ) in results:
        rows.append(
            {
                "source_id": src_id,
                "event": event,
                "description": description,
                "severity": severity,
                "effective": effective,
                "expires": expires,
                "area_desc": area_desc,
                "URL": url,
            }
        )
        geometries.append(poly)

    wmo_gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")

    # Test a point that should be within the alert area (centroid of first polygon)
    test_point = wmo_gdf.geometry.iloc[0].centroid
    test_lat = test_point.y
    test_lon = test_point.x

    # Simulate the response integration code from responseLocal.py
    wmo_alerts_lats = np.arange(-60, 85, 0.0625)
    wmo_alerts_lons = np.arange(-180, 180, 0.0625)
    wmo_abslat = np.abs(wmo_alerts_lats - test_lat)
    wmo_abslon = np.abs(wmo_alerts_lons - test_lon)
    wmo_alerts_y_p = np.argmin(wmo_abslat)
    wmo_alerts_x_p = np.argmin(wmo_abslon)

    # Verify the indices are within reasonable bounds
    assert 0 <= wmo_alerts_y_p < len(wmo_alerts_lats)
    assert 0 <= wmo_alerts_x_p < len(wmo_alerts_lons)

    # Test a point that should NOT be in the alert area (far from Germany)
    test_lat_outside = 40.0
    test_lon_outside = -100.0

    point_outside = Point(test_lon_outside, test_lat_outside)
    point_gdf = gpd.GeoDataFrame(
        [{"name": "test"}], geometry=[point_outside], crs="EPSG:4326"
    )

    # Check if point is within any alert polygon
    result = gpd.sjoin(point_gdf, wmo_gdf, predicate="within", how="inner")
    assert len(result) == 0, "Point outside Germany should not be in alert area"

    # Test a point that SHOULD be in the alert area
    point_inside = test_point
    point_gdf_inside = gpd.GeoDataFrame(
        [{"name": "test"}], geometry=[point_inside], crs="EPSG:4326"
    )

    result_inside = gpd.sjoin(
        point_gdf_inside, wmo_gdf, predicate="within", how="inner"
    )
    assert len(result_inside) > 0, "Point in alert area should be detected"
    assert result_inside.iloc[0]["event"] in [
        "icy surfaces",
        "Official WARNING of ICY SURFACES",
    ]
