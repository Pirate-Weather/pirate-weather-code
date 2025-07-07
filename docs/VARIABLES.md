# Adding a New Variable to an Existing Model

This document outlines the steps to add a new meteorological variable to an existing weather model within the Pirate Weather codebase and integrate it into the API response. This process involves updating the data ingestion logic and then modifying the API's forecasting logic.

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

## Phase 2: Integrating into the API (`responseLocal.py`)

This phase involves making the new variable available in the API's forecasting logic and ensuring it appears in the JSON response, with version control.

1.  **Update Zarr Variable Tuples (if applicable)**:
    * In `API/responseLocal.py`, locate the `zarrVars` tuples (e.g., `GFSzarrVars`, `HRRRHzarrVars`, `NBMzarrVars`). These tuples define the order and names of variables read from the Zarr files.
    * Add your new variable's internal name to the appropriate tuple, maintaining the correct index.
    * **Example**:
        ```
        GFSzarrVars = (
            "time",
            "VIS_surface",
            # ... other variables ...
            "HPBL_surface", # New variable added here
        )
        ```

2.  **Extend Data Arrays (`InterPhour`, `InterPminute`, `InterPcurrent`)**:
    * Identify which time-series array (hourly, minutely, or currently) your new variable will populate. Most new variables will be hourly.
    * Increase the size of the relevant `np.zeros` array to accommodate the new column.
    * Populate the new column with data from your model. This will often involve similar interpolation/selection logic as existing variables (e.g., `np.choose` to pick the best source based on model hierarchy).
    * **Example (adding to `InterPhour`)**:
        ```
        # Increase the size of InterPhour array by 1 for the new variable
        InterPhour = np.full((len(hour_array_grib), 27), np.nan) # Change 27 to 28 for new variable

        # ... existing variable assignments ...

        # Assign the new variable's data
        # Assuming HPBL is index 26 (if 27 columns initially)
        if "gfs" in sourceList: # Or relevant source
            InterPhour[:, 26] = GFS_Merged[:, <HPBL_INDEX_IN_GFS_ZARR_VARS_TUPLE>]
            # Apply unit conversions or clipping if necessary
            InterPhour[:, 26] = clipLog(InterPhour[:, 26], min_val, max_val, "HPBL Hour")
        ```
    * Ensure units are consistent with other API response variables. Apply conversion factors (`elevUnit`, `tempUnits`, etc.) if needed.

3.  **Add to API Response Object**:
    * Locate the dictionary creation for the `hourly` data (`hourItem`), `daily` data (`dayObject`), and `currently` data (`returnOBJ["currently"]`).
    * Add a new key-value pair for your new variable. Use a descriptive name that will appear in the API JSON response.
    * **Example (adding to `hourItem`)**:
        ```
        hourItem = {
            "time": int(hour_array_grib[idx]),
            # ... other items ...
            "planetaryBoundaryLayerHeight": InterPhour[idx, 26], # New item
        }
        ```
    * Repeat this for `dayObject` and `returnOBJ["currently"]` if the variable is relevant for those sections. For daily, you might want `min`, `max`, or `mean` values.

4.  **Implement Version-Based Exclusion (CRUCIAL)**:
    * The new data point should *only* be included when the `version` query string is `2` (or a higher specified version), add a conditional check to remove it otherwise. This prevents breaking older clients that might not expect the new field.
    * **Example (within `hourList` loop, or after `dayList` is populated)**:
        ```
        if version < 2:
            hourItem.pop("planetaryBoundaryLayerHeight", None)
        ```
    * Apply this `pop` logic to all relevant dictionaries (`hourItem`, `dayObject`, `returnOBJ["currently"]`).

5.  **Update Summary/Icon Generation (Optional)**:
    * If your new variable significantly impacts weather conditions (e.g., a severe weather index), consider integrating it into the `calculate_text`, `calculate_minutely_text`, `calculate_day_text`, or `calculate_weekly_text` functions to affect the `summary` and `icon` fields. This is usually more complex and should only be done if the variable directly influences the 'overall' weather description.

## Testing

Thorough testing is paramount when adding new variables to ensure data integrity and API stability.

* **Unit Tests**: If applicable, add unit tests for any new helper functions or complex logic introduced for the variable.
* **Integration Tests**: Modify existing integration tests (e.g., in `tests/test_s3_live.py`) to assert that the new variable appears in the API response and has expected values.
* **Comparison Tests**: If you have `test_compare_production.py` setup, ensure your changes don't cause unexpected differences for existing fields. You might need to temporarily disable this for the new field until production is updated.
* **Manual Testing**: Issue cURL requests to your local API with and without the `version=2` (or relevant version) query string to confirm the variable's presence/absence as expected. Check different locations and forecast times.