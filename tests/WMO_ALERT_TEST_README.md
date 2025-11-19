# WMO Alert Ingest Test

This directory contains tests for the WMO (World Meteorological Organization) alert ingest functionality.

## Test Files

### `test_wmo_alert_ingest.py` (marked with @pytest.mark.ingest)
Basic test module with 5 test functions using a French Météo-France alert:
- Tests CAP XML parsing and polygon extraction
- Tests GeoDataFrame creation
- Tests zarr storage and retrieval
- Tests API response integration

Uses `test_wmo_alert_with_polygon.xml` - a modified French alert with polygon data for Paris region.

### `test_wmo_alert_ingest_full.py` (marked with @pytest.mark.ingest)
Full ingest test module with 5 test functions using a German DWD alert:
- Tests parsing of real-world DWD (Deutscher Wetterdienst) alert
- Tests spatial operations on actual alert polygons
- Tests zarr storage with realistic data
- Validates alert detection for points within affected areas

Uses `test_wmo_alert_dwd_germany.xml` - a real DWD icy surfaces warning that was reported to have issues appearing in the API.

## Running Tests

Both test modules are marked with `@pytest.mark.ingest` because they require ingest dependencies (geopandas, zarr, s3fs, aiohttp, etc.).

### Run only ingest tests:
```bash
pytest -v -m ingest
```

### Run all tests except ingest tests:
```bash
pytest -v -m "not ingest"
```

### Run specific test file:
```bash
pytest tests/test_wmo_alert_ingest_full.py -v
```

### Run both WMO test files:
```bash
pytest tests/test_wmo_alert_ingest.py tests/test_wmo_alert_ingest_full.py -v
```

## Configuration

The `pytest.ini` file in the repository root defines the `ingest` marker:

```ini
[pytest]
markers =
    ingest: marks tests as requiring ingest dependencies (deselect with '-m "not ingest"')
```

## Test Data Files

### `data/test_wmo_alert_with_polygon.xml`
- **Source**: Modified French Météo-France alert
- **Event**: Vigilance jaune neige-verglas (Yellow snow-ice alert)
- **Severity**: Moderate
- **Coverage**: Paris region (48.8°N-48.9°N, 2.2°E-2.4°E) and Northern France
- **Use**: Basic functionality testing

### `data/test_wmo_alert_dwd_germany.xml`
- **Source**: Real German DWD (Deutscher Wetterdienst) alert
- **Event**: Official WARNING of ICY SURFACES
- **Severity**: Minor
- **Coverage**: Multiple areas across Germany
- **Effective**: 2025-11-19T20:06:00+01:00 to 2025-11-20T09:00:00+01:00
- **Use**: Real-world alert testing for reported API issues

## Dependencies

These tests require the following ingest dependencies:
- geopandas
- shapely
- zarr
- numpy
- xarray
- s3fs
- aiohttp
- pytest

## Helper Module

### `wmo_test_helpers.py`
Contains extracted functions from `WMO_Alerts_Local.py`:
- `_cap_text()` - Extract text from CAP XML tags
- `_extract_polygons_from_cap()` - Parse CAP XML and extract polygon geometries

This allows testing without importing the full module which has side effects (directory creation, environment setup).

## Integration with Full Ingest

The test suite validates the WMO alert ingest pipeline without requiring:
- Network access to live WMO feeds
- S3 or cloud storage
- Specific directory permissions
- Production environment setup

Instead, tests use static XML files to verify:
- CAP XML parsing correctness
- Spatial operations (point-in-polygon)
- Data transformation (GeoDataFrame → zarr)
- Alert data formatting for API responses

## Notes

- Grid spacing (0.0625°) matches the production system
- Coordinate system: EPSG:4326 (WGS84)
- Alert data format: `event}{description}{area_desc}{effective}{expires}{severity}{URL`
- Multiple alerts at same location are joined with `~` separator
