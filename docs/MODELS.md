# Integrating a New Weather Model

To integrate a new weather model into the Pirate Weather code, you'll primarily work with the data ingestion scripts and the main API response logic. This process involves setting up data retrieval, processing the data into a usable format, and then integrating it into the API's forecasting logic.

An important high level note: One of the key goals of this API is to be efficient and provide fast retrievals, which might explain some of the design choices here (ex. combining all the variables within one chunk, using Dask for many things). The goal is for processing to fit within 16 GB of RAM, and to keep response times under 50 ms.  

**Note:** The Pirate Weather codebase has been refactored into a modular structure. The API response generation is now split across multiple modules:
- `API/responseLocal.py` - Main orchestration and FastAPI endpoints
- `API/hourly/` - Hourly forecast generation
- `API/daily/` - Daily forecast aggregation
- `API/current/` - Current conditions
- `API/minutely/` - Minutely precipitation forecasts
- `API/constants/` - Shared constants (forecast indices, model constants, clipping rules)
- `API/io/` - Zarr reading and data I/O helpers
- `API/forecast_sources.py` - Model merging and source priority logic
- `API/data_inputs.py` - Data preparation and input organization
- `API/request/` - Request preprocessing and grid indexing
- `API/utils/` - Utility functions (time indexing, solar calculations, geography)

Here's a general outline for integrating a new model:


### Phase 1: Data Ingestion

This phase focuses on retrieving the raw model data, processing it, and saving it in a format optimized for the API ([Zarr](https://zarr.readthedocs.io/en/stable/)).

* **Identify Data Source and Retrieval Method**:

  * Determine where the new model's data is hosted (e.g., NOAA NCEP, ECMWF).

  * Choose an appropriate data retrieval library. Existing ingest scripts heavily use [Herbie](https://github.com/blaylockbk/Herbie). If Herbie supports your model, it's the preferred choice, since it makes future maintenance easier.

  * If Herbie isn't suitable, the ingest containers do have internet access, so any download client should work.

* **Create a New Ingestion Script**:

  * Working off existing scripts like `API/HRRR_Local_Ingest.py`, create a new Python script (e.g., `API/YOUR_MODEL_Local_Ingest.py`). Every model has its own quirks, but HRRR provides a good starting point with a minimum of weirdness.

  * This script should, while staying within 16 GB of RAM:
 
    * Define the model name and relevant parameters (e.g., `model_name`, `product`, `model_type`).
  
    * Use `Herbie` (or your chosen method) to download GRIB2 (or other format) files for specific forecast hours and variables.
  
    * Select the necessary meteorological variables (e.g., temperature, precipitation, wind components) and map them to consistent internal names. Refer to `API/constants/forecast_const.py` to see which variables are used in the API response. It isn't necessary for a model to cover everything, but ideally it should have enough to be consistent for a forecast.
 
    * Process the GRIB2 data using tools like `cfgrib` (often used implicitly by `xarray` with GRIB engines) or `wgrib2` for specific operations if needed. CFGRIB is preferred, but most of the models use `wgrib2` at the moment since it is so efficient. The key processing steps are:
    
		1.  Merge all the time step files into a single NetCDF or Zarr file ([Example](https://github.com/Pirate-Weather/pirate-weather-code/blob/aeef857c7c6133ccef2f439efe5c718c75813caf/API/HRRR_Local_Ingest.py#L191)) 
		2.  Ensure that winds are earth centred ([Example](https://github.com/Pirate-Weather/pirate-weather-code/blob/aeef857c7c6133ccef2f439efe5c718c75813caf/API/HRRR_Local_Ingest.py#L215)) 
		3.  De-accumulate any variables that need it, notably precipitation ([Example](https://github.com/Pirate-Weather/pirate-weather-code/blob/aeef857c7c6133ccef2f439efe5c718c75813caf/API/HRRR_Local_Ingest.py#L262)). Notably, the API uses trailing precipitation accumulations, so precipitation between 0H and 1H is saved in the 1H slot. This might change someday, but will be on the response side, not the ingest.
		4.  De-average any variables that need it (such as UV) ([Example](https://github.com/Pirate-Weather/pirate-weather-code/blob/aeef857c7c6133ccef2f439efe5c718c75813caf/API/GFS_Local_Ingest.py#L417)) 
		5.  Save the Zarr archives locally in the `/mnt/nvme/` directory, which is typically mounted from a Docker volume. The `save_path` and `save_type` environment variables control this, compressed and chunked along the time dimension ([Example](https://github.com/Pirate-Weather/pirate-weather-code/blob/aeef857c7c6133ccef2f439efe5c718c75813caf/API/HRRR_Local_Ingest.py#L282)) 

    * Download 1-hourly forecasts to use as an approximation of observed data. Some day a better approach could be more accurate here, but it's a reasonably good proxy.

    * Merge 48 hours worth of "observed" data with the forecast data to create a cohesive time series.  

    * Merge variables together along an axis to create a 4D array (parameter, time, X, Y), chunked so that all variables and all times are stored in the same chunk (-1, -1, 3, 3). This step might have to be done in Dask to reduce memory requirements.
   
    * Save the final Zarr array to disk and upload to S3! 

* **Update Docker Configuration**:

  * If your ingestion script requires new Python libraries, add them to `Docker/requirements-ingest.txt`
 
  * Ensure your Docker Compose setup (`docker-compose_oph`) includes the necessary volume mounts for the new model's data, similar to how existing models are handled.

### Phase 2: API Integration

This phase involves modifying the API's core logic to load the new model's data and incorporate its forecasts into the API responses.

* **Add Model Constant**:

  * In `API/constants/model_const.py`, add a constant identifier for your new model following the existing pattern.
  
  * **Example**:
    ```python
    # In API/constants/model_const.py
    YOUR_MODEL = "your_model_name"
    ```

* **Declare New Global Zarr Variable**:

  * In `API/responseLocal.py` (or more appropriately in `API/io/zarr_reader.py` if following the modular pattern), declare a new global variable for your model's Zarr store, for example, `YOUR_MODEL_Zarr`.

* **Update `update_zarr_store` Function**:

  * Modify the `update_zarr_store(initialRun)` function in `API/io/zarr_reader.py` (or `API/responseLocal.py` depending on code version) to find and open your new model's Zarr file.

  * Add logic to locate the latest version of your model's Zarr store (similar to how existing models are handled).

  * Include `zarr.open` calls for your model's Zarr store, similar to existing models.

  * Add download/sync logic if using S3 or remote storage.

* **Integrate into `PW_Forecast` Endpoint**:

  * **Request Preprocessing**: In `API/request/preprocess.py`, the `prepare_initial_request()` function handles initial request validation and parameter parsing. No changes are typically needed here unless your model requires special request handling.

  * **Grid Indexing**: In `API/request/grid_indexing.py`, the `calculate_grid_indexing()` function determines which grid points to read from each model.
  
    * If your model uses a different grid than existing ones (e.g., a new projection or resolution), you may need to:
      1. Add grid constants to `API/constants/grid_const.py`
      2. Add a new grid matching function in `API/request/grid_indexing.py` (similar to `lambertGridMatch` for Lambert Conformal grids)
      3. Update the `ZarrSources` dataclass to include your model
    
    * The grid matching should be fast and not rely on reading the entire grid file. Pre-compute coordinate lookups if needed.

  * **Asynchronous Zarr Read**: In `API/responseLocal.py`, within the `PW_Forecast()` function:
    
    * Add your model to the `ZarrSources` object instantiation
    * The zarr reading is handled asynchronously using the framework established in `calculate_grid_indexing()`
    
    * **Example**:
      ```python
      # In PW_Forecast function
      zarr_sources = ZarrSources(
          gfs=GFS_Zarr if not readGFS else None,
          hrrr=HRRR_Zarr if not readHRRR else None,
          # ... other models ...
          your_model=YOUR_MODEL_Zarr if not readYOUR_MODEL else None,
      )
      ```

  * **Source List and Metadata**: Update the source list logic in `API/forecast_sources.py`:
    
    * Add your model to `build_source_metadata()` to track which models are available and their forecast times
    * This function returns a `SourceMetadata` object with source lists, times, and indices

  * **Data Merging and Interpolation**: This is a critical step.

    * **Determine Model Hierarchy**: Decide where your model fits in the priority hierarchy:
      * Short-term high-resolution (like HRRR): 0-18 hours
      * Medium-term regional (like NBM): 18-240 hours  
      * Global long-term (like GFS/GEFS): beyond 240 hours
      * Specialty models (like DWD MOSMIX for Europe, ECMWF for global high-quality)

    * **Update Model Merging**: In `API/forecast_sources.py`, the `merge_hourly_models()` function handles combining multiple model sources with priority logic:
      
      ```python
      # In merge_hourly_models() function
      # Add your model to the appropriate priority tier
      # Example for a high-resolution short-term model:
      if "your_model" in source_list:
          # Merge your model data with appropriate priority
          # Use np.where or np.choose to select between models
          merged_temp = np.where(
              np.isnan(hrrr_temp),
              your_model_temp,
              hrrr_temp
          )
      ```

    * **Update Data Inputs**: In `API/data_inputs.py`, the `prepare_data_inputs()` function organizes raw model data into structured inputs for the forecast builders:
      
      ```python
      # In prepare_data_inputs() function
      temperature_inputs = {
          "gfs": GFS_Merged[:, GFS_TEMP_IDX] if "gfs" in source_list else None,
          "hrrr": HRRR_Merged[:, HRRR_TEMP_IDX] if "hrrr" in source_list else None,
          "your_model": YOUR_MODEL_Merged[:, YOUR_MODEL_TEMP_IDX] if "your_model" in source_list else None,
          # ... other models
      }
      ```
    
    * **Variable Mapping**: Map your model's raw variable names to the standardized names used in `DATA_HOURLY`, `DATA_CURRENT`, etc. (defined in `API/constants/forecast_const.py`)

    * **Unit Conversions**: Apply necessary unit conversions. Conversion factors are defined in `API/constants/api_const.py` (e.g., `CONVERSION_FACTORS`)

    * **Clipping and Validation**: Use utility functions from `API/api_utils.py` for data validation:
      * `clipLog()` - Log and clip values to valid ranges
      * Clipping constants are in `API/constants/clip_const.py`

  * **Summary and Icon Generation**: If your model provides unique insights relevant to the `summary` and `icon` fields, integrate its data into the text generation functions:
    
    * `API/PirateText.py` - `calculate_text()` for hourly/current summaries
    * `API/PirateMinutelyText.py` - `calculate_minutely_text()` for minutely summaries
    * `API/PirateDailyText.py` - `calculate_day_text()` for daily summaries
    * `API/PirateWeeklyText.py` - `calculate_weekly_text()` for weekly summaries
    * `API/PirateTextHelper.py` - Helper functions for text generation

* **Testing**:

  * Add new tests in the `tests/` directory to verify that your model's data is correctly ingested, loaded, and contributes accurately to the API responses.
  
  * Existing test files that can serve as templates:
    * `tests/test_s3_live.py` - Integration tests with live data
    * `tests/test_compare_production.py` - Production comparison tests
  
  * Run linting and formatting:
    ```bash
    scripts/lint
    scripts/format
    ```
  
  * Run the test suite:
    ```bash
    pytest
    ```

This overview provides a roadmap for integrating a new model. Be prepared for iterative testing and debugging, as weather data processing can be complex due to varying grids, units, and data formats.