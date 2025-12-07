# DWD MOSMIX Stations Feature

## Overview

This feature adds information about DWD MOSMIX weather stations to the API response's `flags` section when using V2 query parameters. This allows users to see which DWD MOSMIX stations are contributing data to their specific location's forecast.

## How It Works

### Ingest Process

1. **Station Metadata Collection**: During the DWD MOSMIX data ingest process, the script parses station information from the MOSMIX-S KML file, including:
   - Station ID
   - Station Name
   - Latitude
   - Longitude

2. **Grid Mapping**: The `build_grid_to_stations_map()` function creates a mapping between GFS 0.25° grid cells and nearby stations within a 50km radius.

3. **Storage**: The station mapping is saved as a pickle file (`DWD_MOSMIX_stations.pickle`) alongside the Zarr data files.

### API Response

When a user makes a request to the API with version >= 2 and DWD MOSMIX data is available for their location, the response will include a `stations` field in the `flags` section:

```json
{
  "flags": {
    "sources": ["dwd_mosmix", "gfs"],
    "version": "V2.9.0c",
    "stations": [
      {
        "id": "10382",
        "name": "Berlin-Tempelhof",
        "lat": 52.4675,
        "lon": 13.4021
      },
      {
        "id": "10384",
        "name": "Berlin-Schönefeld",
        "lat": 52.3807,
        "lon": 13.5226
      }
    ],
    ...
  }
}
```

## Implementation Details

### Key Components

1. **DWD_Mosmix_Local_Ingest.py**
   - `build_grid_to_stations_map()`: Creates the grid-to-stations mapping using BallTree for efficient spatial queries
   - Saves the mapping to a pickle file during ingest

2. **responseLocal.py**
   - Loads the station mapping on startup
   - Looks up stations for the requested grid cell
   - Adds stations to the flags section when version >= 2

### Requirements

- Station map must exist at: `{save_dir}/{ingest_version}/DWD_MOSMIX_stations.pickle`
- Request must use version >= 2 (add `?version=2` to the URL)
- DWD MOSMIX data must be available for the requested location

## Example Usage

### API Request
```bash
curl "https://api.pirateweather.net/forecast/YOUR_API_KEY/52.52,13.405?version=2"
```

### Response (relevant section)
```json
{
  "flags": {
    "sources": ["dwd_mosmix", "gfs"],
    "version": "V2.9.0c",
    "stations": [
      {
        "id": "10382",
        "name": "Berlin-Tempelhof",
        "lat": 52.4675,
        "lon": 13.4021
      }
    ],
    "sourceIDX": {
      "dwd_mosmix": {
        "x": 53,
        "y": 570,
        "lat": 52.5,
        "lon": 13.4
      }
    }
  }
}
```

## Technical Notes

- The 50km radius is defined by `DWD_RADIUS` in `API/ingest_utils.py`
- Longitude values are converted from [0, 360] to [-180, 180] format for API output
- Grid cells with no nearby stations will not have the `stations` field
- The feature only appears when DWD MOSMIX is actually used as a data source for the forecast

## Testing

Run the station mapping tests with:
```bash
pytest tests/test_dwd_station_mapping.py -v
```

Tests cover:
- Station data structure validation
- Coordinate validation and conversion
- Grid cell mapping format
- Longitude conversion edge cases
- Grid resolution verification
