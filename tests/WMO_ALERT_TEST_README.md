# WMO Alert Ingest Test

This directory contains tests for the WMO (World Meteorological Organization) alert ingest functionality.

## Overview

The WMO alert ingest system processes weather alerts from various meteorological agencies around the world using the Common Alerting Protocol (CAP) format. This test suite verifies that the system correctly:

1. Parses CAP XML files with polygon data
2. Extracts alert information including event type, severity, effective/expiry times
3. Creates GeoDataFrames for spatial processing
4. Generates zarr data stores for efficient storage and retrieval
5. Integrates alert data into API responses

## Test Files

### `test_wmo_alert_ingest.py`
Main test module containing 5 test functions:

- **test_wmo_alert_xml_exists**: Verifies the test XML file is present
- **test_extract_polygons_from_cap**: Tests polygon extraction from CAP XML
- **test_wmo_alert_geodataframe_creation**: Tests GeoDataFrame creation from alert data
- **test_wmo_alert_zarr_creation**: Tests zarr storage creation and reading
- **test_wmo_alert_response_integration**: Tests integration with the API response system

### `wmo_test_helpers.py`
Helper module containing extracted functions from `WMO_Alerts_Local.py` for testing purposes. This allows testing the polygon extraction logic without triggering the full ingest pipeline that requires specific directory permissions.

### `data/test_wmo_alert_with_polygon.xml`
Sample CAP XML file based on a real French Météo-France alert (Vigilance jaune neige-verglas - Yellow alert for snow and ice). The file has been modified to include polygon data for two test areas:

1. **Paris Region Test Area**: 48.8°N to 48.9°N, 2.2°E to 2.4°E
2. **Northern France Test Area**: 49.0°N to 49.2°N, 2.0°E to 2.3°E

## CAP XML Structure

The test XML follows the CAP 1.2 standard and includes:

- **Alert metadata**: identifier, sender, sent time, status, message type
- **Info block**: language, category, event type, urgency, severity, certainty
- **Temporal data**: effective time, onset time, expiry time
- **Area information**: area description, polygon coordinates, geocodes
- **Alert content**: headline, description, instructions, web links

## Running the Tests

```bash
# Run all WMO alert tests
python -m pytest tests/test_wmo_alert_ingest.py -v

# Run a specific test
python -m pytest tests/test_wmo_alert_ingest.py::test_extract_polygons_from_cap -v

# Run with verbose output
python -m pytest tests/test_wmo_alert_ingest.py -vv
```

## Test Data

The test uses a moderate severity snow/ice alert for France with:
- **Event**: Vigilance jaune neige-verglas (Yellow snow-ice alert)
- **Severity**: Moderate
- **Effective time**: 2025-11-20T00:00:00+01:00
- **Expiry time**: 2025-11-21T00:00:00+01:00
- **Coordinate system**: EPSG:4326 (WGS84)

## Integration with WMO_Alerts_Local.py

The full ingest system (`API/WMO_Alerts_Local.py`) performs these steps:

1. Downloads the WMO sources.json file to identify operational feeds
2. Fetches RSS feeds for each source
3. Downloads CAP XML documents from feed links
4. Extracts polygons and creates a GeoDataFrame
5. Creates a grid of points (0.0625° spacing)
6. Performs spatial join to find points within alert polygons
7. Saves data to zarr format for efficient retrieval

These tests verify steps 3-7 work correctly using a static test file instead of downloading from live sources.

## Dependencies

- geopandas: Spatial data operations
- shapely: Polygon geometry
- numpy: Numerical operations
- zarr: Data storage
- xarray: Multi-dimensional arrays
- pytest: Test framework

## Notes

- The test avoids importing `WMO_Alerts_Local.py` directly because it runs initialization code that requires specific directory permissions
- The helper module (`wmo_test_helpers.py`) contains copies of essential functions for isolated testing
- Test data uses realistic coordinates for locations in France
- Grid spacing in tests (0.0625°) matches the production system
