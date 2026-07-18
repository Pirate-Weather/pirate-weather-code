# Adding a New Variable to an Existing Model

This document outlines the steps to add a new meteorological variable to an existing weather model within the Pirate Weather codebase and integrate it into the API response. This process involves updating the data ingestion logic and then modifying the API's forecasting logic.

**Note:** The Pirate Weather codebase has been refactored into a modular structure. The main API response logic is now split across multiple modules in the `API/` directory, including:
- `API/hourly/` - Hourly forecast generation
- `API/daily/` - Daily forecast aggregation
- `API/current/` - Current conditions
- `API/minutely/` - Minutely precipitation forecasts
- `API/constants/` - Shared constants and configuration
- `API/utils/` - Utility functions
- `API/legacy/` - Backward compatibility helpers

**Important:** The codebase makes extensive use of [NumPy](https://numpy.org/) for efficient array operations. If you're unfamiliar with NumPy, see the [NumPy Guide](NUMPY_GUIDE.md) for common patterns used in Pirate Weather.

## Phase 1: Modifying the Data Ingestion Script

This phase focuses on ensuring the new variable is retrieved and saved alongside the existing data for a specific model.

1.  **Locate the Relevant Ingestion Script**:
    * Navigate to the `API/` directory and identify the Python script responsible for ingesting data from the specific model you want to modify (e.g., `GFS_Local_Ingest.py`, `HRRR_Local_Ingest.py`).

2.  **Add the New Variable to Data Retrieval**:
    * Within the ingestion script, locate the section where the meteorological variables are defined for retrieval. This often involves a list or dictionary used by `Herbie` (e.g., `grib_filters` or a similar configuration).
    * Add the GRIB2 name (or the corresponding key) for your new variable to this list. Ensure you understand its unit and expected data type.
    * **Example (Conceptual)**:
        ```
        # In API/GFS_Local_Ingest.py, find where variables are defined
        variables_to_fetch = [
            'TMP_2maboveground',
            'PRATE_surface',
            # Add your new variable here
            'HPBL_surface' # Example: Planetary Boundary Layer Height
        ]
        ```

3.  **Process and Save the New Variable**:
    * Ensure that after fetching, the new variable is correctly processed and included when the data is converted to Zarr format. The existing scripts typically handle this automatically if the variable is included in the initial `Herbie` request and the `xarray` dataset construction.
    * Verify that the new variable's data is consistent with the existing Zarr structure (e.g., same dimensions, appropriate chunking if explicitly controlled).
    * Make sure it's not an accumulated or averaged variable, and if so, process accordingly.

4.  **Update Ingestion Dependencies (If Necessary)**:
    * If adding this new variable requires any new Python libraries for its processing or handling that are not already in your ingestion Docker container, add them to `Docker/requirements-ingest.txt`.

5.  **Reprocess Existing Data (Important)**:
    * To ensure the API can access the new variable for past model runs, you will likely need to re-run the ingestion script for historical data or for recent model runs that are missing this variable. This will overwrite existing Zarr files with updated versions that include your new data.

## Phase 2: Integrating into the API

This phase involves making the new variable available in the API's forecasting logic and ensuring it appears in the JSON response, with version control.

1.  **Update Data Constants**:
    * In `API/constants/forecast_const.py`, locate the appropriate data dictionary (`DATA_HOURLY`, `DATA_CURRENT`, `DATA_MINUTELY`, or `DATA_DAY`). These dictionaries define the column indices for variables in the interpolation arrays.
    * Add your new variable to the appropriate dictionary with the next available index.
    * **Example**:
        ```python
        # In API/constants/forecast_const.py
        DATA_HOURLY = {
            "time": 0,
            "type": 1,
            # ... other variables ...
            "ice_intensity": 31,
            "boundary_layer_height": 32,  # New variable added here
        }
        ```

2.  **Extend Data Arrays and Add to Data Inputs**:
    * The main interpolation arrays (`InterPhour`, `InterPminute`, `InterPcurrent`) are now automatically sized based on the constants in `forecast_const.py`.
    * Add your new variable to the data preparation logic in `API/data_inputs.py` or `API/forecast_sources.py`:
        * If it's a simple model variable, add it to the appropriate inputs dictionary in `prepare_data_inputs()` in `API/data_inputs.py`
        * If it requires special merging logic from multiple models, update `merge_hourly_models()` in `API/forecast_sources.py`
    * **Example (adding to data inputs)**:
        ```python
        # In API/data_inputs.py, within prepare_data_inputs()
        # Model constants are defined in API/constants/model_const.py
        # For example, GFS["temp"] or HRRR["gust"]
        
        boundary_layer_inputs = {
            "gfs": gfs_merged[:, GFS["boundary_layer"]] if "gfs" in source_list else None,
            "hrrr": hrrr_merged[:, HRRR["boundary_layer"]] if "hrrr" in source_list else None,
            # Add other models as needed
        }
        inputs["boundary_layer_inputs"] = boundary_layer_inputs
        ```
    * **Note**: You'll also need to add the new variable index to the model constant dictionaries in `API/constants/model_const.py` (e.g., `GFS["boundary_layer"] = 25`)
    * Apply clipping and validation as needed using functions from `API/api_utils.py` (e.g., `clipLog()`)
    * Ensure units are consistent with other API response variables. Conversion factors are defined in `API/constants/api_const.py`

3.  **Add to API Response Object**:
    * Based on which forecast section your variable belongs to, update the appropriate builder module:
        * **Hourly**: `API/hourly/block.py` - function `build_hourly_block()` and `build_hourly_objects()`
        * **Daily**: `API/daily/builder.py` - function `build_daily_section()`
        * **Currently**: `API/current/metrics.py` - function `build_current_section()`
        * **Minutely**: `API/minutely/builder.py` - function `build_minutely_block()`
    
    * **Example (adding to hourly response in `API/hourly/block.py`)**:
        ```python
        # In build_hourly_objects() function
        # Add the new variable to the hourly item dictionary
        hourItem = {
            "time": int(hour_array_grib[idx]),
            # ... other items ...
            "planetaryBoundaryLayerHeight": hourly_display[idx, DATA_HOURLY["boundary_layer_height"]],
        }
        ```
    
    * For daily forecasts, you may want to compute aggregates (min, max, mean) using the helper functions in `API/daily/builder.py`
    * **Example (adding to daily response)**:
        ```python
        # In build_daily_section() function
        # Use _aggregate_stats() to compute daily statistics
        boundary_layer_stats = _aggregate_stats(
            InterPhour[:, DATA_HOURLY["boundary_layer_height"]],
            hourlyDayIndex,
            daily_days,
            calc_max=True,
            calc_mean=True
        )
        
        # Add to daily object
        dayObject = {
            "time": int(day_array_grib[d]),
            # ... other items ...
            "boundaryLayerHeightMax": boundary_layer_stats[2][d],  # max
            "boundaryLayerHeightMean": boundary_layer_stats[0][d],  # mean
        }
        ```

4.  **Implement Version-Based Exclusion (CRUCIAL)**:
    * The new data point should *only* be included when the `version` query string is `2` (or a higher specified version). Add a conditional check in the appropriate builder to remove it otherwise. This prevents breaking older clients that might not expect the new field.
    * **Example (within hourly builder)**:
        ```python
        # In build_hourly_objects() or similar function
        if version < 2:
            hourItem.pop("planetaryBoundaryLayerHeight", None)
        ```
    * Apply this `pop` logic to all relevant response sections (hourly, daily, currently)

5.  **Add Rounding Rules (Optional)**:
    * If your new variable requires specific rounding precision, add it to `ROUNDING_RULES` in `API/constants/api_const.py`
    * **Example**:
        ```python
        ROUNDING_RULES = {
            # ... existing rules ...
            "planetaryBoundaryLayerHeight": 0,  # Round to integer meters
        }
        ```

6.  **Update Summary/Icon Generation (Optional)**:
    * If your new variable significantly impacts weather conditions (e.g., a severe weather index), consider integrating it into the text generation functions:
        * `API/PirateText.py` - `calculate_text()` for hourly/current summaries
        * `API/PirateMinutelyText.py` - `calculate_minutely_text()` for minutely summaries
        * `API/PirateDailyText.py` - `calculate_day_text()` for daily summaries
        * `API/PirateWeeklyText.py` - `calculate_weekly_text()` for weekly summaries
    * Helper functions are available in `API/PirateTextHelper.py`
    * This is usually more complex and should only be done if the variable directly influences the 'overall' weather description.

## Testing

Thorough testing is paramount when adding new variables to ensure data integrity and API stability.

* **Unit Tests**: If applicable, add unit tests for any new helper functions or complex logic introduced for the variable. Tests are located in the `tests/` directory.
* **Integration Tests**: Modify existing integration tests (e.g., in `tests/test_s3_live.py`) to assert that the new variable appears in the API response and has expected values.
* **Comparison Tests**: If you have `test_compare_production.py` set up, ensure your changes don't cause unexpected differences for existing fields. You might need to temporarily disable this for the new field until production is updated.
* **Manual Testing**: Issue cURL requests to your local API with and without the `version=2` (or relevant version) query string to confirm the variable's presence/absence as expected. Check different locations and forecast times.
* **Linting and Formatting**: Run the linting and formatting tools before submitting:
    ```bash
    scripts/lint
    scripts/format
    ```
* **Run Tests**: Execute the test suite to verify no regressions:
    ```bash
    pytest
    ```
---

## Air Quality Fields

Air quality data requires `version=2` (or higher) in the request. The top-level AQI field is present by default when AQ model data is available. Per-pollutant concentration fields are gated behind `include=airqualitydetails`.

### Summary field (version ≥ 2)

| Field | Sections | Description |
|-------|----------|-------------|
| `airQualityIndex` | currently, hourly, daily | Numeric AQI value. Scale depends on unit system (see below). Integer. |
| `airQualityIndexMax` | daily | Daily maximum AQI. |
| `airQualityIndexMaxTime` | daily | Unix timestamp of daily AQI peak. |

### Detail fields (`include=airqualitydetails`, version ≥ 2)

| Field | Sections | Unit | Description |
|-------|----------|------|-------------|
| `pm25` | currently, hourly | µg/m³ | PM2.5 concentration. |
| `pm25Max` | daily | µg/m³ | Daily max PM2.5. |
| `pm25MaxTime` | daily | Unix s | Time of daily max PM2.5. |
| `pm10` | currently, hourly | µg/m³ | PM10 concentration. |
| `pm10Max` | daily | µg/m³ | Daily max PM10. |
| `pm10MaxTime` | daily | Unix s | Time of daily max PM10. |
| `ozoneConcentration` | currently, hourly | ppb | O3 concentration. |
| `ozoneConcentrationMax` | daily | ppb | Daily max O3. |
| `ozoneConcentrationMaxTime` | daily | Unix s | Time of daily max O3. |
| `no2Concentration` | currently, hourly | ppb | NO2 concentration. |
| `no2ConcentrationMax` | daily | ppb | Daily max NO2. |
| `no2ConcentrationMaxTime` | daily | Unix s | Time of daily max NO2. |
| `so2Concentration` | currently, hourly | ppb | SO2 concentration. |
| `so2ConcentrationMax` | daily | ppb | Daily max SO2. |
| `so2ConcentrationMaxTime` | daily | Unix s | Time of daily max SO2. |
| `coConcentration` | currently, hourly | ppb | CO concentration (SILAM only). |
| `coConcentrationMax` | daily | ppb | Daily max CO. |
| `coConcentrationMaxTime` | daily | Unix s | Time of daily max CO. |

### AQI system selection

| `units` query parameter | AQI system |
|------------------------|-----------|
| `us` (default) | US EPA AQI, 0–500+ scale |
| `ca` | Canadian AQHI, 1–10+ scale |
| `uk` or `si` | EU CAQI, 0–100+ scale |

### Model priority

1. **RAQDPS** (regional, Canada + CONUS) is preferred when available.
2. **SILAM** (global) is used as a fallback.
3. If both models are excluded (`exclude=raqdps,silam`) all AQ fields are absent.

CO (`coConcentration`) is sourced exclusively from SILAM. RAQDPS does not publish CO.

### Flags

When AQ data is loaded, `flags.sources` and `flags.sourceTimes` will include `raqdps` and/or `silam` entries with their respective model run timestamps.
