# Integrating a New Weather Model

To integrate a new weather model into the Pirate Weather code, you'll primarily work with the data ingestion scripts and the main API response logic. This process involves setting up data retrieval, processing the data into a usable format, and then integrating it into the API's forecasting logic.

An important high level note: One of the key goals of this API is to be efficient and provide fast retrievals, which might explain some of the design choices here (ex. combining all the variables within one chunk, using Dask for many things). The goal is for processing to fit within 16 GB of RAM, and to keep response times under 50 ms.  

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
  
    * Select the necessary meteorological variables (e.g., temperature, precipitation, wind components) and map them to consistent internal names. Refer to `responseLocal.py` to see which variables are used in the API response. It isn't necessary for a model to cover everything, but ideally it should have enough to be consistent for a forecast.
 
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

* **Declare New Global Zarr Variable**:

  * In `API/responseLocal.py`, declare a new global variable for your model's Zarr store, for example, `YOUR_MODEL_Zarr`.

* **Update `update_zarr_store` Function**:

  * Modify the `update_zarr_store(initialRun)` function to find and open your new model's Zarr file.

  * Add logic to `find_largest_integer_directory` to locate the latest version of your model's Zarr store.

  * Include `zarr.open` calls for your model's Zarr store, similar to `GFS_Zarr = zarr.open(...)`.

  * Add `download_if_newer` calls in `initialDataSync` and `dataSync` to ensure your model's data is regularly downloaded and updated.

* **Integrate into `PW_Forecast` Endpoint**:

  * **Conditional Data Reading**: In the `PW_Forecast` function, add a flag (e.g., `readYOUR_MODEL = False`) and corresponding `exclude` parameter logic if you want to allow users to exclude your model's data.

  * **Grid Matching**: If your model uses a different grid than existing ones (e.g., HRRR's Lambert Conformal vs. GFS's Lat/Lon), you'll need to define a new grid matching function (similar to `lambertGridMatch`) or adapt `y_p`, `x_p` logic for latitude/longitude lookup. This should be fast and not rely on reading the entire grid file to use a KNN match.

  * **Asynchronous Zarr Read**: Add a new task to the `zarrTasks` dictionary within the `WeatherParallel` class instance to asynchronously read data from your new model's Zarr store.

  * **Source List and Times**: Update `sourceList` and `sourceTimes` to include your new model as a data source.

  * **Data Merging and Interpolation**: This is a critical step.

    * Determine where your model fits into the hierarchy (e.g., short-term like HRRR, mid-term like NBM, or global like GFS/GEFS).

    * Modify the hourly and minutely interpolation/merging sections (`InterPhour`, `InterPminute`) to incorporate your model's data. This often involves `np.choose` with `np.argmin(np.isnan(...))` to select the first non-NaN value from a prioritized list of models.

    * Ensure all necessary variables (temperature, humidity, wind, precipitation, etc.) are pulled from your model where it is the primary source or fallback.

    * Pay close attention to units and conversions (`windUnit`, `prepIntensityUnit`, `tempUnits`, etc.).

  * **Variable Mapping**: Map your model's raw variable names to the standardized names used in the API response (e.g., `TMP_2maboveground` becomes `temperature`).

  * **Summary and Icon Generation**: If your model provides unique insights relevant to the `summary` and `icon` fields, integrate its data into the `calculate_text`, `calculate_minutely_text`, `calculate_day_text`, and `calculate_weekly_text` functions.

* **Testing**:

  * Add new tests in the `tests` directory to verify that your model's data is correctly ingested, loaded, and contributes accurately to the API responses. `test_compare_production.py` and `test_s3_live.py` can serve as inspiration.

This overview provides a roadmap for integrating a new model. Be prepared for iterative testing and debugging, as weather data processing can be complex due to varying grids, units, and data formats.