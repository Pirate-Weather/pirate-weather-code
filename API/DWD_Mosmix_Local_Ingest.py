# %% DWD MOSMIX-S Hourly Processing script - Gridded Interpolation
# Alexander Rey, June 2025

# %% Import modules
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import zarr.storage
import requests
import io
import zipfile
from pykml import parser
from lxml import etree

import dask.array as da # For Dask operations
from scipy.interpolate import griddata # For numerical point-to-grid interpolation
from scipy.spatial import cKDTree # For categorical (string) nearest-neighbor interpolation

import json # For saving station metadata as JSON
import s3fs # For S3 operations

warnings.filterwarnings("ignore", "This pattern is interpreted")


# Define KML and DWD namespaces for easier parsing
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
DWD_NAMESPACE = (
    "{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}"
)

# Define the comprehensive list of variables to be saved in the final Zarr dataset.
# This list matches the schema expected by responseLocal.py, adding NaNs for unavailable data.
# Note: PTYPE_surface will be string, numerical variables will be float.
zarrVars = (
    "time",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "PRES_meansealevel",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "GUST_surface",
    "PRATE_surface",
    "APCP_surface",
    "TCDC_entireatmosphere",
    "VIS_surface",
    "PPROB",
    "PTYPE_surface", # This will be interpolated as string type
    "APTMP_2maboveground", # Placeholder
    "CSNOW_surface", # Placeholder
    "CICEP_surface", # Placeholder
    "CFRZR_surface", # Placeholder
    "CRAIN_surface", # Placeholder
    "REFC_entireatmosphere", # Placeholder
    "DSWRF_surface", # Placeholder
    "CAPE_surface", # Placeholder
    "TOZNE_entireatmosphere_consideredasasinglelayer_", # Placeholder
    "DUVB_surface", # Placeholder
    "MASSDEN_8maboveground", # Placeholder
    "Storm_Distance", # Placeholder
    "Storm_Direction", # Placeholder
    "FOSINDX_surface", # Placeholder
    "PWTHER_surfaceMreserved", # Placeholder
    "WIND_10maboveground", # Placeholder
    "WDIR_10maboveground", # Placeholder
)

# Define the subset of variables from MOSMIX_S that the API will actively use and interpolate.
# These should be the *standardized* names after conversion from DWD's original names.
# PTYPE_surface (string type) will be handled separately.
api_target_numerical_variables = [
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "PRES_meansealevel",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "GUST_surface",
    "PRATE_surface",
    "TCDC_entireatmosphere",
    "VIS_surface",
    "PPROB",
]


# --- Helper Functions (adapted from previous turns) ---

def _get_precip_type_from_ww(ww_code):
    """
    Determines precipitation type (rain, snow, sleet, or none) from DWD WW code.
    Args: ww_code (int or float): The WW (present weather) code.
    Returns: str: "rain", "snow", "sleet", or "none".
    """
    if np.isnan(ww_code): return "none"
    ww_code = int(ww_code)
    if 0 <= ww_code <= 49: return "none"
    elif 50 <= ww_code <= 59: return "rain"
    elif 60 <= ww_code <= 69: return "sleet" if ww_code in [66, 67] else "rain"
    elif 70 <= ww_code <= 79: return "sleet" if ww_code in [79] else "snow"
    elif 80 <= ww_code <= 89: return "sleet" if ww_code in [87, 88] else ("snow" if ww_code in [83, 84, 85, 86] else "rain")
    elif 90 <= ww_code <= 99: return "sleet" if ww_code in [95, 96, 97, 99] else "rain"
    return "none"


def parse_mosmix_kml(kml_filepath):
    """Parses a DWD MOSMIX KML/KMZ file and extracts forecast data into a Pandas DataFrame."""
    if kml_filepath.lower().endswith(".kmz"):
        with zipfile.ZipFile(kml_filepath, "r") as kmz_file:
            kml_content = next((kmz_file.read(name) for name in kmz_file.namelist() if name.lower().endswith(".kml")), None)
            if kml_content is None: raise ValueError(f"No KML file found inside {kml_filepath}.")
            root = parser.parse(io.BytesIO(kml_content)).getroot()
    else:
        with open(kml_filepath, "rb") as f: root = parser.parse(f).getroot()

    data_records = []
    global_metadata = {}
    product_definition_elem = root.find(f"./{KML_NAMESPACE}Document/{KML_NAMESPACE}ExtendedData/{DWD_NAMESPACE}ProductDefinition")
    if product_definition_elem is not None:
        global_metadata["IssueTime"] = pd.to_datetime(product_definition_elem.findtext(f"{DWD_NAMESPACE}IssueTime")) if product_definition_elem.findtext(f"{DWD_NAMESPACE}IssueTime") else pd.NaT

    global_forecast_times = []
    time_step_elements = root.findall(f"./{KML_NAMESPACE}Document/{KML_NAMESPACE}ExtendedData/{DWD_NAMESPACE}ProductDefinition/{DWD_NAMESPACE}ForecastTimeSteps/{DWD_NAMESPACE}TimeStep")
    if not time_step_elements: raise ValueError("No global forecast time steps found in KML.")
    for ts_elem in time_step_elements: global_forecast_times.append(pd.to_datetime(ts_elem.text))
    global_forecast_times = pd.Series(global_forecast_times).dropna().drop_duplicates().sort_values().tolist()

    for placemark in root.findall(f".//{KML_NAMESPACE}Placemark"):
        station_id = (placemark.find(f"{KML_NAMESPACE}name").text.strip() if placemark.find(f"{KML_NAMESPACE}name") is not None else "Unknown ID")
        station_name = (placemark.find(f"{KML_NAMESPACE}description").text.strip() if placemark.find(f"{KML_NAMESPACE}description") is not None else "Unknown Name")
        coordinates_elem = placemark.find(f"{KML_NAMESPACE}Point/{KML_NAMESPACE}coordinates")
        lon, lat, alt = None, None, None
        if coordinates_elem is not None and coordinates_elem.text:
            coords = coordinates_elem.text.strip().split(",")
            lon, lat, alt = float(coords[0]), float(coords[1]), (float(coords[2]) if len(coords) > 2 else 0.0)

        extended_data_pm = placemark.find(f"{KML_NAMESPACE}ExtendedData")
        if extended_data_pm is None: continue

        station_forecast_data = {}
        for forecast_elem in extended_data_pm.findall(f"{DWD_NAMESPACE}Forecast"):
            element_name = forecast_elem.get(f"{DWD_NAMESPACE}elementName")
            value_elem = forecast_elem.find(f"{DWD_NAMESPACE}value")
            if element_name and value_elem is not None and value_elem.text:
                param_values = [float(val) if val != "-" else np.nan for val in value_elem.text.strip().split()]
                if len(param_values) == len(global_forecast_times): station_forecast_data[element_name] = param_values
        
        for i, ftime in enumerate(global_forecast_times):
            record = {"station_id": station_id, "station_name": station_name, "longitude": lon, "latitude": lat, "altitude": alt, "time": ftime}
            for param, values in station_forecast_data.items(): record[param] = values[i]
            data_records.append(record)

    df = pd.DataFrame(data_records)
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["time"]).sort_values(by=["station_id", "time"]).reset_index(drop=True)
    return df, global_metadata


# New helper function to calculate saturation vapor pressure
def _calculate_saturation_vapor_pressure(T_kelvin):
    """
    Calculates saturation vapor pressure (in hPa) from temperature in Kelvin.
    Uses Magnus formula over water.
    """
    T_celsius = T_kelvin - 273.15
    # Clamp T_celsius to a reasonable range for numerical stability if needed, though typically not for standard atmospheric temps
    T_celsius = np.clip(T_celsius, -100, 100) # Example clipping
    return 6.1094 * np.exp((17.625 * T_celsius) / (243.04 + T_celsius))

# New function to calculate relative humidity
def calculate_relative_humidity(T_kelvin, Td_kelvin):
    """
    Calculates relative humidity (%) from temperature (T) and dew point temperature (Td), both in Kelvin.
    Returns NaN if T or Td are NaN.
    """
    # Handle NaN values explicitly
    if np.isnan(T_kelvin) or np.isnan(Td_kelvin):
        return np.nan

    # Calculate saturation vapor pressure at actual temperature
    es_T = _calculate_saturation_vapor_pressure(T_kelvin)
    # Calculate actual vapor pressure at dew point temperature
    e_Td = _calculate_saturation_vapor_pressure(Td_kelvin)

    # Relative Humidity in percentage
    # Handle potential division by zero or very small values for es_T
    # If es_T is zero or very close to zero, it means temperature is extremely low,
    # and division might lead to Inf.
    if np.isclose(es_T, 0.0):
        return 0.0 if np.isclose(e_Td, 0.0) else np.nan # Or 0 if both are essentially zero, else NaN

    rh = (e_Td / es_T) * 100

    # Clip values to be between 0 and 100%
    rh = np.clip(rh, 0, 100)
    return rh


def convert_df_to_xarray(df, global_metadata=None):
    """Converts a Pandas DataFrame to an xarray Dataset, preparing for gridding."""
    if df.empty: return xr.Dataset()

    ds = df.set_index(["station_id", "time"])[[col for col in df.columns if col not in ["station_id", "time", "station_name", "longitude", "latitude", "altitude"]]].to_xarray()
    unique_stations_df = df[["station_id", "station_name", "longitude", "latitude", "altitude"]].drop_duplicates(subset=["station_id"]).set_index("station_id")

    ds = ds.assign_coords(
        station_name=("station_id", unique_stations_df["station_name"].reindex(ds.station_id).values),
        longitude=("station_id", unique_stations_df["longitude"].reindex(ds.station_id).values),
        latitude=("station_id", unique_stations_df["latitude"].reindex(ds.station_id).values),
        altitude=("station_id", unique_stations_df["altitude"].reindex(ds.station_id).values),
    )

    # Standardize variable names and units (all values remain raw as per original model units for interpolation)
    if "TTT" in ds.data_vars: ds["TMP_2maboveground"] = ds["TTT"]
    if "Td" in ds.data_vars: ds["DPT_2maboveground"] = ds["Td"]
    # Removed direct check for "Rh" as it will now be calculated

    # --- Calculation of Relative Humidity (RH_2maboveground) ---
    # User stated MOSMIX data does not have Rh directly, so calculate from TTT and Td
    if "TMP_2maboveground" in ds.data_vars and "DPT_2maboveground" in ds.data_vars:
        # Apply the calculation function across the xarray DataArrays
        ds["RH_2maboveground"] = xr.apply_ufunc(
            calculate_relative_humidity,
            ds["TMP_2maboveground"],
            ds["DPT_2maboveground"],
            vectorize=True,  # Apply function element-wise over array dimensions
            dask='parallelized', # Enable Dask for parallel computation
            output_dtypes=[np.float32]
        )
    else:
        # If TTT or Td are missing, RH cannot be calculated, so fill with NaN
        print("Warning: Cannot calculate Relative Humidity (RH_2maboveground) due to missing Temperature (TTT) or Dew Point (Td) data. Filling with NaN.")
        ds["RH_2maboveground"] = (("station_id", "time"), np.full((ds.sizes["station_id"], ds.sizes["time"]), np.nan, dtype=np.float32))

    if "PPPP" in ds.data_vars: ds["PRES_meansealevel"] = ds["PPPP"]
    if "FF" in ds.data_vars and "DD" in ds.data_vars:
        wind_speed_ms = ds["FF"]; wind_direction_rad = np.deg2rad(270 - ds["DD"])
        ds["UGRD_10maboveground"] = wind_speed_ms * np.cos(wind_direction_rad)
        ds["VGRD_10maboveground"] = wind_speed_ms * np.sin(wind_direction_rad)
    if "FX1" in ds.data_vars: ds["GUST_surface"] = ds["FX1"]
    if "RR1c" in ds.data_vars: ds["PRATE_surface"] = ds["RR1c"]; ds["APCP_surface"] = ds["RR1c"]
    if "N" in ds.data_vars: ds["TCDC_entireatmosphere"] = ds["N"]
    if "VV" in ds.data_vars: ds["VIS_surface"] = ds["VV"]
    if "R101" in ds.data_vars: ds["PPROB"] = ds["R101"]
    if "WW" in ds.data_vars: ds["PTYPE_surface"] = ds["WW"].map(_get_precip_type_from_ww)

    # Remove original DWD variables and ensure 'Rh' is explicitly handled if it were ever parsed.
    dwd_original_vars_to_drop = ["PPPP", "TX", "TTT", "Td", "Rh", "FF", "DD", "FX1", "RR1c", "VV", "N", "R101", "WW"]
    for var in dwd_original_vars_to_drop:
        if var in ds.data_vars: ds = ds.drop_vars(var)
    
    if global_metadata:
        attrs_to_add = {}
        for k, v in global_metadata.items():
            if isinstance(v, pd.Timestamp): attrs_to_add[k] = v.isoformat()
            elif isinstance(v, list) and all(isinstance(item, dict) and "referenceTime" in item and isinstance(item["referenceTime"], pd.Timestamp) for item in v):
                attrs_to_add[k] = [{m_k: (m_v.isoformat() if isinstance(m_v, pd.Timestamp) else m_v) for m_k, m_v in model.items()} for model in v]
            else: attrs_to_add[k] = v
        ds.attrs.update(attrs_to_add)
    return ds


def save_to_zarr(xarray_dataset, zarr_path):
    """Saves an xarray Dataset to a Zarr store."""
    if xarray_dataset.dims:
        xarray_dataset.to_zarr(zarr_path, mode="w", compute=True)
        print(f"Data successfully saved to Zarr at: {zarr_path}")
    else:
        print("Xarray Dataset is empty (no dimensions found), nothing to save to Zarr.")


# --- NEW FUNCTIONS FOR GRIDDING ---

def load_gfs_grid_coordinates(gfs_zarr_path):
    """
    Loads latitude and longitude coordinates from a GFS Zarr dataset.
    This defines the target grid for interpolation.
    """
    try:
        # Assuming GFS Zarr has 'latitude' and 'longitude' as dimensions/coords.
        # Use chunks={} to load entire coordinates into memory, as they are relatively small.
        with xr.open_zarr(gfs_zarr_path, consolidated=False, chunks={}) as gfs_ds:
            gfs_lats = gfs_ds.latitude.values
            gfs_lons = gfs_ds.longitude.values
            print(f"Loaded GFS grid: {len(gfs_lats)} latitudes, {len(gfs_lons)} longitudes.")
            return gfs_lats, gfs_lons
    except Exception as e:
        print(f"Error loading GFS grid from {gfs_zarr_path}: {e}")
        sys.exit(1)


def interpolate_dwd_to_grid(dwd_ds_point, gfs_lats, gfs_lons, numerical_variables_to_interpolate, all_zarr_schema_vars):
    """
    Interpolates DWD point data (from stations) onto a GFS-like rectilinear grid.
    Leverages Dask for parallel processing over time steps using scipy.interpolate.griddata for numerical
    and cKDTree for categorical (string) data.
    """
    # Create a meshgrid for the target GFS grid points (used by scipy.interpolate.griddata)
    gfs_lon_grid, gfs_lat_grid = np.meshgrid(gfs_lons, gfs_lats)
    gfs_grid_points_flat = np.c_[gfs_lat_grid.ravel(), gfs_lon_grid.ravel()] # Shape (num_grid_points, 2)

    # Extract spatial coordinates of DWD stations (constant for all time steps)
    dwd_station_lats = dwd_ds_point.latitude.values
    dwd_station_lons = dwd_ds_point.longitude.values
    dwd_spatial_points = np.c_[dwd_station_lats, dwd_station_lons] # Shape (num_stations, 2)

    # Initialize a new xarray Dataset to store the gridded data
    gridded_dwd_ds = xr.Dataset(
        coords={
            "time": dwd_ds_point.time,
            "latitude": gfs_lats,
            "longitude": gfs_lons
        }
    )
    
    # Define output chunking for the gridded data.
    # Chunk over time, and a portion of lat/lon for parallel processing of grid cells.
    output_grid_chunk_shape = (1, len(gfs_lats) // 10, len(gfs_lons) // 10) 


    # --- Interpolate Numerical Variables ---
    def _interpolate_single_time_slice_numerical(dwd_data_slice_numerical, dwd_spatial_points, gfs_grid_points_flat, gfs_lats_size, gfs_lons_size):
        """Applies griddata for one numerical variable at one time step."""
        # Filter out NaN values from station data for this time slice
        valid_indices = ~np.isnan(dwd_data_slice_numerical)
        
        # If no valid data points for this time slice, return NaNs for the entire grid slice
        if not np.any(valid_indices):
            return np.full((gfs_lats_size, gfs_lons_size), np.nan, dtype=np.float32)

        # Get valid station points and their corresponding values
        station_points_for_interp = dwd_spatial_points[valid_indices]
        values_at_stations = dwd_data_slice_numerical[valid_indices]

        # Perform the interpolation using 'nearest' neighbor method
        # 'nearest' is robust and performs well for irregularly spaced points like stations.
        interpolated_values_flat = griddata(
            station_points_for_interp, # Source points (lat/lon of valid stations)
            values_at_stations,        # Values at those stations
            gfs_grid_points_flat,      # Target grid points (flat list of lat/lon for GFS grid)
            method='nearest',          # Or 'linear' if desired and data density allows.
            fill_value=np.nan          # Fill outside convex hull with NaN
        )
        
        # Reshape the flat interpolated array back to the 2D grid shape (latitude, longitude)
        return interpolated_values_flat.reshape(gfs_lats_size, gfs_lons_size)

    for var_name in numerical_variables_to_interpolate:
        if var_name in dwd_ds_point.data_vars:
            print(f"Scheduling interpolation for numerical variable: {var_name}")
            
            # Use `xr.apply_ufunc` to apply the interpolation function across time slices in parallel.
            # `input_core_dims=[['station_id']]` means the `_interpolate_single_time_slice_numerical`
            # function operates on 1D arrays corresponding to each time slice (across stations).
            # `output_core_dims=[['latitude', 'longitude']]` means the output is a 2D grid.
            gridded_data_var_da = xr.apply_ufunc(
                _interpolate_single_time_slice_numerical,
                dwd_ds_point[var_name],                 # Input DataArray (station_id, time)
                input_core_dims=[['station_id']],       # Apply function along 'station_id' dim for each time slice
                output_core_dims=[['latitude', 'longitude']], # Output has new 'latitude', 'longitude' dims
                exclude_dims={'station_id'},            # 'station_id' is processed internally, excluded from output dims
                dask='parallelized',                    # Enable Dask parallelization
                output_dtypes=[np.float32],             # Specify output dtype
                kwargs={
                    'dwd_spatial_points': dwd_spatial_points,
                    'gfs_grid_points_flat': gfs_grid_points_flat,
                    'gfs_lats_size': len(gfs_lats),
                    'gfs_lons_size': len(gfs_lons),
                }
            ).chunk({'time': output_grid_chunk_shape[0], 
                     'latitude': output_grid_chunk_shape[1], 
                     'longitude': output_grid_chunk_shape[2]})
            
            gridded_dwd_ds[var_name] = gridded_data_var_da
        else:
            print(f"Variable {var_name} not found in DWD data, adding as NaN placeholder to gridded dataset.")
            gridded_dwd_ds[var_name] = (('time', 'latitude', 'longitude'), 
                                        da.full((len(dwd_ds_point.time), len(gfs_lats), len(gfs_lons)), np.nan, dtype=np.float32, chunks=output_grid_chunk_shape))

    # --- Interpolate Categorical (String) Variables like PTYPE_surface ---
    # `griddata` does not work with strings. Nearest neighbor using cKDTree is suitable.
    def _interpolate_single_time_slice_categorical(dwd_data_slice_ptype, dwd_spatial_points, gfs_grid_points_flat, gfs_lats_size, gfs_lons_size):
        """Applies nearest neighbor for one categorical (string) variable at one time step."""
        # Filter out "none" or empty strings from station data for this time slice
        # Assuming "none" is the default or invalid state for precipType.
        valid_indices = (dwd_data_slice_ptype != "none") & (dwd_data_slice_ptype != "") & (~pd.isna(dwd_data_slice_ptype))
        
        # If no valid categorical data points for this time slice, return "none" for the entire grid slice
        if not np.any(valid_indices):
            return np.full((gfs_lats_size, gfs_lons_size), "none", dtype=object)

        # Build KDTree with only valid station points for this time slice
        station_points_for_kdtree = dwd_spatial_points[valid_indices]
        kdtree = cKDTree(station_points_for_kdtree)
        
        # Query KDTree for nearest neighbor for each GFS grid point
        # `distances` are not needed, just `indices` to the nearest valid station.
        _, nearest_indices = kdtree.query(gfs_grid_points_flat, k=1)
        
        # Get the corresponding precipitation types from the original valid DWD data slice
        # The result might contain values outside the valid_indices if nearest_indices point to filtered out stations.
        # So, we map directly from the valid subset of types using the derived indices.
        original_valid_types = dwd_data_slice_ptype[valid_indices]
        nearest_types = original_valid_types[nearest_indices]
        
        # Reshape to 2D grid and return
        return nearest_types.reshape(gfs_lats_size, gfs_lons_size)

    # Schedule interpolation for PTYPE_surface
    if "PTYPE_surface" in dwd_ds_point.data_vars:
        print("Scheduling interpolation for PTYPE_surface (string type)...")
        gridded_ptype_var_da = xr.apply_ufunc(
            _interpolate_single_time_slice_categorical,
            dwd_ds_point["PTYPE_surface"],
            input_core_dims=[['station_id']],
            output_core_dims=[['latitude', 'longitude']],
            exclude_dims={'station_id'},
            dask='parallelized',
            output_dtypes=[object], # Output dtype must be object for strings
            kwargs={
                'dwd_spatial_points': dwd_spatial_points,
                'gfs_grid_points_flat': gfs_grid_points_flat,
                'gfs_lats_size': len(gfs_lats),
                'gfs_lons_size': len(gfs_lons),
            }
        ).chunk({'time': output_grid_chunk_shape[0], 
                 'latitude': output_grid_chunk_shape[1], 
                 'longitude': output_grid_chunk_shape[2]})
        
        gridded_dwd_ds["PTYPE_surface"] = gridded_ptype_var_da
    else:
        print("PTYPE_surface not found in DWD data, adding as 'none' placeholder to gridded dataset.")
        gridded_dwd_ds["PTYPE_surface"] = (('time', 'latitude', 'longitude'),
                                           da.full((len(dwd_ds_point.time), len(gfs_lats), len(gfs_lons)), "none", dtype=object, chunks=output_grid_chunk_shape))


    # --- Add placeholder variables for all other zarrVars not directly interpolated ---
    # This ensures the final Zarr schema matches the comprehensive zarrVars tuple.
    for var_name in all_zarr_schema_vars:
        if var_name not in gridded_dwd_ds.data_vars and var_name != "time":
            print(f"Adding placeholder NaN variable: {var_name}")
            # Determine dtype: object for strings, float for numerical
            dtype_for_placeholder = object if var_name == "PTYPE_surface" else np.float32
            
            gridded_dwd_ds[var_name] = (('time', 'latitude', 'longitude'),
                                        da.full((len(dwd_ds_point.time), len(gfs_lats), len(gfs_lons)), np.nan, dtype=dtype_for_placeholder, chunks=output_grid_chunk_shape))

    return gridded_dwd_ds


# --- Main Ingest Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    dwd_mosmix_url = "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz"
    downloaded_kmz_file = "MOSMIX_S_LATEST_240.kmz"
    
    # Paths for processed data
    forecast_process_dir = os.getenv("forecast_process_dir", "/home/ubuntu/Weather/DWD")
    tmpDIR = os.getenv("tmp_dir", os.path.join(forecast_process_dir, "Downloads")) # Use os.getenv for tmp_dir as well
    forecast_path = os.getenv("forecast_path", "/home/ubuntu/Weather/Prod/DWD")
    
    # Path to an existing GFS Zarr file to get the reference grid
    # This environment variable should point to a GFS Zarr dataset that contains 'latitude' and 'longitude' coordinates.
    gfs_zarr_reference_path = os.getenv("GFS_ZARR_PATH", "/path/to/your/GFS.zarr") # **CRITICAL: SET THIS ENV VAR**
    
    saveType = os.getenv("save_type", "Download")
    aws_access_key_id = os.environ.get("AWS_KEY", "")
    aws_secret_access_key = os.environ.get("AWS_SECRET", "")

    s3 = None
    if saveType == "S3":
        s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


    # Create necessary directories
    os.makedirs(forecast_process_dir, exist_ok=True)
    if os.path.exists(forecast_process_dir):
        shutil.rmtree(forecast_process_dir) # Clean previous run
    os.makedirs(forecast_process_dir)
    os.makedirs(tmpDIR, exist_ok=True)
    if saveType == "Download":
        os.makedirs(forecast_path, exist_ok=True)

    T0 = time.time()

    # --- Step 1: Download and Parse DWD MOSMIX-S KML/KMZ ---
    print(f"\n--- Attempting to download DWD MOSMIX data from: {dwd_mosmix_url} ---")
    try:
        response = requests.get(dwd_mosmix_url, stream=True)
        response.raise_for_status()
        with open(os.path.join(tmpDIR, downloaded_kmz_file), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully to: {os.path.join(tmpDIR, downloaded_kmz_file)}")
    except Exception as e:
        print(f"Error downloading DWD data: {e}. Exiting.")
        sys.exit(1)

    print(f"\n--- Reading and Parsing KML from {os.path.join(tmpDIR, downloaded_kmz_file)} ---")
    df_data, global_metadata = parse_mosmix_kml(os.path.join(tmpDIR, downloaded_kmz_file))
    if df_data.empty:
        print("No data extracted from KML/KMZ. Exiting.")
        sys.exit(1)
        
    # NEW CODE: Ensure unique stations based on latitude and longitude
    initial_station_count = df_data['station_id'].nunique()
    # Keep the first occurrence if lat/lon duplicates exist
    df_data_unique_stations = df_data.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    unique_station_count = df_data_unique_stations['station_id'].nunique()

    if initial_station_count > unique_station_count:
        print(f"Removed {initial_station_count - unique_station_count} duplicate stations based on latitude/longitude.")
        print(f"Proceeding with {unique_station_count} unique stations for interpolation.")
    else:
        print("No duplicate stations found based on latitude/longitude in the raw data.")

    df_data = df_data_unique_stations # Use the filtered DataFrame for further processing
    
    
    base_time = global_metadata.get("IssueTime", datetime.utcnow())
    print(f"Base time for this ingest run (from KML metadata): {base_time}")

    # Check if this is newer than the current file (prevents re-processing old data)
    final_time_pickle_path = os.path.join(forecast_path, "DWD.time.pickle")
    if saveType == "S3":
        s3_time_pickle_key = os.path.join("ForecastTar_v2", "DWD.time.pickle")
        try:
            with s3.open(f"{os.getenv('s3_bucket', 'your-s3-bucket')}/{s3_time_pickle_key}", "rb") as f:
                previous_base_time = pickle.load(f)
            if previous_base_time >= base_time:
                print("No new update to DWD found, ending script.")
                sys.exit()
        except FileNotFoundError:
            print("Previous DWD time pickle not found in S3, proceeding with ingest.")
        except Exception as e:
            print(f"Error checking previous DWD time in S3: {e}. Proceeding.")
    else:
        if os.path.exists(final_time_pickle_path):
            with open(final_time_pickle_path, "rb") as file:
                previous_base_time = pickle.load(file)
            if previous_base_time >= base_time:
                print("No new update to DWD found locally, ending script.")
                sys.exit()

    # --- Step 2: Convert Pandas DataFrame to xarray Dataset (point-based) ---
    print("\n--- Converting DataFrame to xarray Dataset (point-based) ---")
    dwd_ds_point = convert_df_to_xarray(df_data, global_metadata=global_metadata)
    if not dwd_ds_point.dims:
        print("Point-based xarray Dataset is empty. Exiting.")
        sys.exit(1)
    print("Point-based DWD Dataset preview:\n", dwd_ds_point)

    # --- Step 3: Load GFS Grid Coordinates ---
    print(f"\n--- Loading GFS grid coordinates from: {gfs_zarr_reference_path} ---")
    gfs_lats, gfs_lons = load_gfs_grid_coordinates(gfs_zarr_reference_path)
    if gfs_lats is None or gfs_lons is None:
        print("Failed to load GFS grid coordinates. Exiting.")
        sys.exit(1)

    # --- Step 4: Interpolate DWD Point Data to GFS Grid ---
    print("\n--- Interpolating DWD point data to GFS grid ---")
    # This is the core interpolation step, parallelized by Dask.
    # The output `gridded_dwd_ds` will be the 4D (variable, time, latitude, longitude) dataset.
    gridded_dwd_ds = interpolate_dwd_to_grid(dwd_ds_point, gfs_lats, gfs_lons, api_target_numerical_variables, zarrVars)

    print("\nGridded DWD Dataset preview (after interpolation):\n", gridded_dwd_ds)
    print(f"\nGridded DWD Dataset dimensions: {gridded_dwd_ds.dims}")
    print(f"Gridded DWD Dataset data variables: {list(gridded_dwd_ds.data_vars.keys())}")


    # --- Step 5: Save Gridded DWD Dataset to Zarr ---
    # The final gridded Zarr will be saved here.
    gridded_zarr_output_full_path = os.path.join(forecast_process_dir, "DWD_Gridded.zarr")
    print(f"\n--- Saving Gridded xarray Dataset to Zarr: {gridded_zarr_output_full_path} ---")
    save_to_zarr(gridded_dwd_ds, gridded_zarr_output_full_path)

    # --- Step 6: Save Original Station Metadata (for nearest station flag) ---
    # Extract only the essential station metadata for lookup
    station_metadata_df = df_data[['station_id', 'station_name', 'latitude', 'longitude', 'altitude']].drop_duplicates(subset='station_id').set_index('station_id')
    station_metadata_path = os.path.join(forecast_process_dir, "DWD_Station_Metadata.json")
    print(f"\n--- Saving original station metadata to: {station_metadata_path} ---")
    station_metadata_df.to_json(station_metadata_path, orient="index") # Save as JSON
    print("Station metadata saved.")


    # --- Step 7: Upload to S3 or Copy Locally (Final Transfer) ---
    # The main gridded Zarr will be transferred.
    final_gridded_zarr_target_path = os.path.join(forecast_path, "DWD.zarr") # API expects DWD.zarr
    final_time_pickle_path_dest = os.path.join(forecast_path, "DWD.time.pickle")

    if saveType == "S3":
        s3_bucket_name = os.getenv("s3_bucket", "your-s3-bucket") # Ensure S3 bucket is configured
        s3_gridded_zarr_key = os.path.join("ForecastTar_v2", "DWD.zarr") # Example S3 key path
        s3_time_pickle_key_upload = os.path.join("ForecastTar_v2", "DWD.time.pickle")
        s3_station_metadata_key = os.path.join("ForecastTar_v2", "DWD_Station_Metadata.json")

        # Upload Gridded Zarr directory
        print(f"Uploading gridded Zarr from {gridded_zarr_output_full_path} to S3://{s3_bucket_name}/{s3_gridded_zarr_key}")
        s3.put(gridded_zarr_output_full_path, f"{s3_bucket_name}/{s3_gridded_zarr_key}", recursive=True)
        
        # Upload time pickle
        temp_time_pickle_source_path = os.path.join(forecast_process_dir, "DWD.time.pickle")
        with open(temp_time_pickle_source_path, "wb") as file: pickle.dump(base_time, file)
        print(f"Uploading time pickle from {temp_time_pickle_source_path} to S3://{s3_bucket_name}/{s3_time_pickle_key_upload}")
        s3.put_file(temp_time_pickle_source_path, f"{s3_bucket_name}/{s3_time_pickle_key_upload}")

        # Upload station metadata
        print(f"Uploading station metadata from {station_metadata_path} to S3://{s3_bucket_name}/{s3_station_metadata_key}")
        s3.put_file(station_metadata_path, f"{s3_bucket_name}/{s3_station_metadata_key}")

        print("All files uploaded to S3.")

    else: # saveType == "Download" (local copy)
        # Move time pickle
        temp_time_pickle_source_path = os.path.join(forecast_process_dir, "DWD.time.pickle")
        with open(temp_time_pickle_source_path, "wb") as file: pickle.dump(base_time, file)
        shutil.move(temp_time_pickle_source_path, final_time_pickle_path_dest)
        
        # Copy gridded Zarr directory
        shutil.copytree(gridded_zarr_output_full_path, final_gridded_zarr_target_path, dirs_exist_ok=True)
        
        # Copy station metadata
        shutil.copy(station_metadata_path, os.path.join(forecast_path, "DWD_Station_Metadata.json"))
        print(f"All files copied locally to {forecast_path}.")


    # --- Final Cleanup ---
    if os.path.exists(forecast_process_dir):
        shutil.rmtree(forecast_process_dir)
        print(f"\nCleaned up processing directory: {forecast_process_dir}")
    if os.path.exists(tmpDIR):
        shutil.rmtree(tmpDIR)
        print(f"Cleaned up downloads directory: {tmpDIR}")

    T_end = time.time()
    print(f"\nTotal script execution time: {T_end - T0:.2f} seconds")
