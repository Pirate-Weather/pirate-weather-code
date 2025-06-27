# %% DWD MOSMIX-S Hourly Processing script - Gridded Interpolation
# This script downloads the latest DWD MOSMIX-S KML/KMZ data,
# parses the point-based station forecasts, cleans and standardizes the data,
# interpolates it onto a specified GFS reference grid, and then saves the
# resulting gridded (4D) data to a Zarr store for API consumption.
# It also stores original station metadata for lookup purposes.
#
# Author: Alexander Rey
# Date: June 2025

# %% Import modules
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import requests
import io
import zipfile
from pykml import parser  # For parsing KML/KMZ files

import dask.array as da  # For Dask array operations and parallelization
from scipy.interpolate import griddata  # For numerical point-to-grid interpolation
from scipy.spatial import (
    cKDTree,
)  # For categorical (string) nearest-neighbor interpolation

import s3fs  # For S3 operations (uploading processed data)

# Suppress specific warnings that might arise from pandas/xarray operations with NaNs
warnings.filterwarnings("ignore", "This pattern is interpreted")


# Define KML and DWD namespaces for easier parsing of the XML structure.
# These namespaces are crucial for correctly locating elements within the KML file.
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
DWD_NAMESPACE = (
    "{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}"
)

# Define the comprehensive list of variables to be saved in the final Zarr dataset.
# This list matches the schema expected by the downstream API (responseLocal.py).
# Variables not directly available from MOSMIX-S will be filled with NaNs or default values
# during the interpolation process to maintain a consistent schema.
zarrVars = (
    "time",
    "TMP_2maboveground",  # Temperature at 2m (Kelvin)
    "DPT_2maboveground",  # Dew Point Temperature at 2m (Kelvin)
    "RH_2maboveground",  # Relative Humidity at 2m (calculated, %)
    "PRES_meansealevel",  # Pressure at Mean Sea Level (Pa)
    "UGRD_10maboveground",  # U-component of Wind at 10m (m/s)
    "VGRD_10maboveground",  # V-component of Wind at 10m (m/s)
    "GUST_surface",  # Wind Gust at surface (m/s)
    "PRATE_surface",  # Precipitation Rate at surface (mm/h, 1-hour accumulation)
    "APCP_surface",  # Accumulated Precipitation at surface (mm, 1-hour accumulation)
    "TCDC_entireatmosphere",  # Total Cloud Cover (0-100%)
    "VIS_surface",  # Visibility at surface (meters)
    "PPROB",  # Probability of Precipitation (NaN placeholder for MOSMIX-S)
    "PTYPE_surface",  # Precipitation Type (string: "rain", "snow", "sleet", "none")
    # Placeholders for variables not directly available from DWD MOSMIX-S.
    # These will be initialized with NaNs to ensure schema compatibility.
    "APTMP_2maboveground",  # Apparent Temperature
    "CSNOW_surface",  # Categorical Snow
    "CICEP_surface",  # Categorical Ice Pellets
    "CFRZR_surface",  # Categorical Freezing Rain
    "CRAIN_surface",  # Categorical Rain
    "REFC_entireatmosphere",  # Radar Reflectivity
    "DSWRF_surface",  # Downward Shortwave Radiation Flux
    "CAPE_surface",  # Convective Available Potential Energy
    "TOZNE_entireatmosphere_consideredasasinglelayer_",  # Total Ozone
    "DUVB_surface",  # UV-B Radiation
    "MASSDEN_8maboveground",  # Mass Density (e.g., for smoke)
    "Storm_Distance",  # Distance to nearest storm
    "Storm_Direction",  # Direction to nearest storm
    "FOSINDX_surface",  # Fire Outbreak Spreading Index
    "PWTHER_surfaceMreserved",  # NBM specific Present Weather
    "WIND_10maboveground",  # Scalar Wind Speed at 10m
    "WDIR_10maboveground",  # Scalar Wind Direction at 10m
)

# Define the subset of standardized variables that are directly sourced from
# MOSMIX-S (or calculated from it) and will be actively interpolated onto the grid.
# PTYPE_surface is handled separately due to its string data type.
api_target_numerical_variables = [
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",  # Relative Humidity is calculated from TTT and Td
    "PRES_meansealevel",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "GUST_surface",
    "PRATE_surface",
    "TCDC_entireatmosphere",
    "VIS_surface",
    # PPROB (from R101) is excluded as it's not in MOSMIX-S data.
]


# --- Core Helper Functions ---


def _get_precip_type_from_ww(ww_code):
    """
    Determines precipitation type (rain, snow, sleet, or none) from DWD WW code.
    This is a simplified mapping based on common WMO present weather codes.

    Args:
        ww_code (int or float): The WW (present weather) code. Can be NaN if missing.

    Returns:
        str: "rain", "snow", "sleet", or "none".
    """
    if np.isnan(ww_code):
        return "none"

    ww_code = int(ww_code)

    # WMO Present Weather Codes (simplified categories)
    if 0 <= ww_code <= 49:  # No precipitation, fog, haze, etc.
        return "none"
    elif 50 <= ww_code <= 59:  # Drizzle
        return "rain"
    elif 60 <= ww_code <= 69:  # Rain
        if ww_code in [66, 67]:  # Freezing Rain
            return "sleet"
        return "rain"
    elif 70 <= ww_code <= 79:  # Snow, ice pellets
        if ww_code in [79]:  # Ice Pellets
            return "sleet"
        return "snow"
    elif 80 <= ww_code <= 89:  # Showers
        if ww_code in [83, 84, 85, 86]:  # Snow/sleet showers
            return "snow"
        if ww_code in [87, 88]:  # Hail showers
            return "sleet"
        return "rain"
    elif 90 <= ww_code <= 99:  # Thunderstorms
        if ww_code in [95, 96, 97, 99]:  # Thunderstorm with hail, ice pellets
            return "sleet"
        return "rain"

    return "none"  # Fallback for any unmapped codes


def _calculate_saturation_vapor_pressure(T_kelvin):
    """
    Calculates saturation vapor pressure (in hPa) from temperature in Kelvin.
    Uses the Magnus formula (based on Goff-Gratch equation approximation) over water.

    Args:
        T_kelvin (np.ndarray or float): Temperature in Kelvin.

    Returns:
        np.ndarray or float: Saturation vapor pressure in hPa.
    """
    T_celsius = T_kelvin - 273.15
    # Clip T_celsius to a reasonable range to prevent numerical issues with exp() for extreme values.
    T_celsius = np.clip(T_celsius, -100, 100)
    return 6.1094 * np.exp((17.625 * T_celsius) / (243.04 + T_celsius))


def calculate_relative_humidity(T_kelvin, Td_kelvin):
    """
    Calculates relative humidity (%) from temperature (T) and dew point temperature (Td),
    both in Kelvin.

    Args:
        T_kelvin (np.ndarray or float): Temperature in Kelvin.
        Td_kelvin (np.ndarray or float): Dew point temperature in Kelvin.

    Returns:
        np.ndarray or float: Relative humidity in percentage (0-100%). Returns NaN if T or Td are NaN.
    """
    # Use np.where to handle NaN values gracefully in a vectorized manner.
    rh = np.where(
        np.isnan(T_kelvin) | np.isnan(Td_kelvin),  # Condition for NaN output
        np.nan,  # Value if condition is True (NaN input)
        (
            _calculate_saturation_vapor_pressure(Td_kelvin)  # Actual vapor pressure
            / _calculate_saturation_vapor_pressure(
                T_kelvin
            )  # Saturation vapor pressure
        )
        * 100,  # Convert to percentage
    )
    # Clip values to be strictly between 0 and 100%
    rh = np.clip(rh, 0, 100)
    return rh


def parse_mosmix_kml(kml_filepath):
    """
    Parses a DWD MOSMIX KML/KMZ file and extracts forecast data into a Pandas DataFrame.
    It reads global forecast time steps and station-specific parameters.

    Args:
        kml_filepath (str): Path to the KML or KMZ file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The extracted forecast data.
            - dict: A dictionary of global metadata (e.g., IssueTime).
    """
    # Handle KMZ (zipped KML) files or direct KML files
    if kml_filepath.lower().endswith(".kmz"):
        with zipfile.ZipFile(kml_filepath, "r") as kmz_file:
            # Find the first KML file within the KMZ archive
            kml_content = next(
                (
                    kmz_file.read(name)
                    for name in kmz_file.namelist()
                    if name.lower().endswith(".kml")
                ),
                None,
            )
            if kml_content is None:
                raise ValueError(f"Error: No KML file found inside {kml_filepath}.")
            root = parser.parse(io.BytesIO(kml_content)).getroot()
    else:
        # Parse directly from KML file
        with open(kml_filepath, "rb") as f:
            root = parser.parse(f).getroot()

    data_records = []
    global_metadata = {}

    # Extract global metadata (e.g., IssueTime) from Document level
    product_definition_elem = root.find(
        f"./{KML_NAMESPACE}Document/{KML_NAMESPACE}ExtendedData/{DWD_NAMESPACE}ProductDefinition"
    )
    if product_definition_elem is not None:
        issue_time_text = product_definition_elem.findtext(f"{DWD_NAMESPACE}IssueTime")
        global_metadata["IssueTime"] = (
            pd.to_datetime(issue_time_text) if issue_time_text else pd.NaT
        )

    # Extract global forecast time steps from Document level
    global_forecast_times = []
    time_step_elements = root.findall(
        f"./{KML_NAMESPACE}Document/{KML_NAMESPACE}ExtendedData/{DWD_NAMESPACE}ProductDefinition/{DWD_NAMESPACE}ForecastTimeSteps/{DWD_NAMESPACE}TimeStep"
    )
    if not time_step_elements:
        raise ValueError("Error: No global forecast time steps found in KML Document.")

    for ts_elem in time_step_elements:
        global_forecast_times.append(pd.to_datetime(ts_elem.text))
    # Filter out invalid times and ensure unique, sorted list
    global_forecast_times = (
        pd.Series(global_forecast_times)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    print(f"Found {len(global_forecast_times)} global forecast time steps.")

    # Iterate through Placemarks to extract station-specific data
    for placemark in root.findall(f".//{KML_NAMESPACE}Placemark"):
        # Extract station ID and name
        station_id = (
            placemark.find(f"{KML_NAMESPACE}name").text.strip()
            if placemark.find(f"{KML_NAMESPACE}name") is not None
            else "Unknown ID"
        )
        station_name = (
            placemark.find(f"{KML_NAMESPACE}description").text.strip()
            if placemark.find(f"{KML_NAMESPACE}description") is not None
            else "Unknown Name"
        )

        # Extract coordinates (longitude, latitude, altitude)
        coordinates_elem = placemark.find(
            f"{KML_NAMESPACE}Point/{KML_NAMESPACE}coordinates"
        )
        lon, lat, alt = None, None, None
        if coordinates_elem is not None and coordinates_elem.text:
            coords = coordinates_elem.text.strip().split(",")
            try:
                lon = float(coords[0])
                lat = float(coords[1])
                alt = float(coords[2]) if len(coords) > 2 else 0.0
            except ValueError:
                print(
                    f"Warning: Could not parse coordinates for station ID {station_id}: {coordinates_elem.text}. Setting to NaN."
                )
                lon, lat, alt = np.nan, np.nan, np.nan

        extended_data_pm = placemark.find(f"{KML_NAMESPACE}ExtendedData")
        if extended_data_pm is None:
            print(
                f"Warning: No ExtendedData found for station ID {station_id}. Skipping."
            )
            continue

        station_forecast_data = {}  # Stores {parameter_name: [value1, value2, ...]}
        # Parse dwd:Forecast elements within this Placemark's ExtendedData
        for forecast_elem in extended_data_pm.findall(f"{DWD_NAMESPACE}Forecast"):
            element_name = forecast_elem.get(f"{DWD_NAMESPACE}elementName")
            value_elem = forecast_elem.find(f"{DWD_NAMESPACE}value")

            if element_name and value_elem is not None and value_elem.text:
                # Convert space-separated string values to a list of floats, handling '-' as NaN
                param_values = [
                    float(val) if val != "-" else np.nan
                    for val in value_elem.text.strip().split()
                ]

                # Crucial check: Ensure the number of values matches global time steps
                if len(param_values) == len(global_forecast_times):
                    station_forecast_data[element_name] = param_values
                else:
                    print(
                        f"Warning: Mismatch in number of values for '{element_name}' for station {station_id}. Skipping parameter."
                    )

        # Create a record for each (time, station) pair
        for i, ftime in enumerate(global_forecast_times):
            record = {
                "station_id": station_id,
                "station_name": station_name,
                "longitude": lon,
                "latitude": lat,
                "altitude": alt,
                "time": ftime,
            }
            for param, values in station_forecast_data.items():
                record[param] = values[i]
            data_records.append(record)

    df = pd.DataFrame(data_records)
    # Ensure 'time' column is datetime and timezone-naive UTC, then sort
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(None)
    df = (
        df.dropna(subset=["time"])
        .sort_values(by=["station_id", "time"])
        .reset_index(drop=True)
    )
    return df, global_metadata


def convert_df_to_xarray(df, global_metadata=None):
    """
    Converts a Pandas DataFrame (point-based) to an xarray Dataset.
    Standardizes variable names and units, and calculates derived variables.

    Args:
        df (pd.DataFrame): Input DataFrame from KML parsing.
        global_metadata (dict, optional): Global metadata to add as Dataset attributes.

    Returns:
        xr.Dataset: Point-based xarray Dataset with 'station_id' and 'time' dimensions.
    """
    if df.empty:
        print("DataFrame is empty, cannot convert to xarray Dataset.")
        return xr.Dataset()

    # Create a base xarray Dataset with station_id and time as dimensions
    # Identify unique station details to become coordinates associated with 'station_id'
    unique_stations_df = (
        df[["station_id", "station_name", "longitude", "latitude", "altitude"]]
        .drop_duplicates(subset=["station_id"])
        .set_index("station_id")
    )

    # Identify columns that represent forecast data variables
    data_vars_cols = [
        col
        for col in df.columns
        if col
        not in [
            "station_id",
            "time",
            "station_name",
            "longitude",
            "latitude",
            "altitude",
        ]
    ]

    # Convert DataFrame to xarray Dataset, setting 'station_id' and 'time' as multi-index
    ds = df.set_index(["station_id", "time"])[data_vars_cols].to_xarray()

    # Assign static station properties as coordinates to the 'station_id' dimension
    ds = ds.assign_coords(
        station_name=(
            "station_id",
            unique_stations_df["station_name"].reindex(ds.station_id).values,
        ),
        longitude=(
            "station_id",
            unique_stations_df["longitude"].reindex(ds.station_id).values,
        ),
        latitude=(
            "station_id",
            unique_stations_df["latitude"].reindex(ds.station_id).values,
        ),
        altitude=(
            "station_id",
            unique_stations_df["altitude"].reindex(ds.station_id).values,
        ),
    )

    # --- Standardize variable names and units from DWD KML to API schema ---
    # Values remain in their original DWD units (Kelvin, Pa, m/s, mm/h) before interpolation.
    if "TTT" in ds.data_vars:
        ds["TMP_2maboveground"] = ds["TTT"]
    if "Td" in ds.data_vars:
        ds["DPT_2maboveground"] = ds["Td"]

    # Calculate Relative Humidity (RH_2maboveground) from Temperature (TTT) and Dew Point (Td)
    if "TMP_2maboveground" in ds.data_vars and "DPT_2maboveground" in ds.data_vars:
        ds["RH_2maboveground"] = xr.apply_ufunc(
            calculate_relative_humidity,
            ds["TMP_2maboveground"],
            ds["DPT_2maboveground"],
            vectorize=True,  # Allows element-wise application over arrays
            dask="parallelized",  # Enables Dask for parallel computation
            output_dtypes=[np.float32],  # Specify output data type
        )
    else:
        print(
            "Warning: Cannot calculate Relative Humidity (RH_2maboveground) due to missing Temperature (TTT) or Dew Point (Td) data. Filling with NaN."
        )
        # Initialize RH with NaNs if calculation is not possible
        ds["RH_2maboveground"] = (
            ("station_id", "time"),
            np.full(
                (ds.sizes["station_id"], ds.sizes["time"]), np.nan, dtype=np.float32
            ),
        )

    if "PPPP" in ds.data_vars:
        ds["PRES_meansealevel"] = ds["PPPP"]

    # Calculate U- and V-components of wind from speed (FF) and direction (DD)
    if "FF" in ds.data_vars and "DD" in ds.data_vars:
        wind_speed_ms = ds["FF"]
        # Convert meteorological direction (where wind comes from, 0=N, 90=E)
        # to mathematical/cartesian direction (where wind goes, 0=E, 90=N) and then to radians.
        wind_direction_rad = np.deg2rad(270 - ds["DD"])
        ds["UGRD_10maboveground"] = wind_speed_ms * np.cos(wind_direction_rad)
        ds["VGRD_10maboveground"] = wind_speed_ms * np.sin(wind_direction_rad)

    if "FX1" in ds.data_vars:
        ds["GUST_surface"] = ds["FX1"]

    # Precipitation Rate (RR1c) and Accumulated Precipitation (APCP_surface)
    # MOSMIX-S RR1c is typically 1-hour accumulation, so PRATE_surface and APCP_surface can be the same.
    if "RR1c" in ds.data_vars:
        ds["PRATE_surface"] = ds["RR1c"]
        ds["APCP_surface"] = ds["RR1c"]

    if "N" in ds.data_vars:
        ds["TCDC_entireatmosphere"] = ds["N"]
    if "VV" in ds.data_vars:
        ds["VIS_surface"] = ds["VV"]

    # PPROB (R101) is not in MOSMIX-S, so no processing from source. Will be NaN placeholder later.

    # Precipitation Type (WW)
    if "WW" in ds.data_vars:
        ds["PTYPE_surface"] = ds["WW"].map(_get_precip_type_from_ww)

    # Remove original DWD variables from the dataset after conversion/derivation
    dwd_original_vars_to_drop = [
        "PPPP",
        "TX",
        "TTT",
        "Td",
        "FF",
        "DD",
        "FX1",
        "RR1c",
        "VV",
        "N",
        "WW",
    ]
    for var in dwd_original_vars_to_drop:
        if var in ds.data_vars:
            ds = ds.drop_vars(var)
    # Explicitly drop 'Rh' and 'R101' if they somehow were parsed or created,
    # as we rely on calculation for RH and R101 is not from MOSMIX-S.
    if "Rh" in ds.data_vars:
        ds = ds.drop_vars("Rh")
    if "R101" in ds.data_vars:
        ds = ds.drop_vars("R101")

    # Add global metadata as attributes to the Dataset for traceability
    if global_metadata:
        attrs_to_add = {}
        for k, v in global_metadata.items():
            if isinstance(v, pd.Timestamp):
                attrs_to_add[k] = v.isoformat()
            elif isinstance(v, list) and all(
                isinstance(item, dict)
                and "referenceTime" in item
                and isinstance(item["referenceTime"], pd.Timestamp)
                for item in v
            ):
                # Convert list of model dicts (containing timestamps) to Zarr-friendly string representation
                attrs_to_add[k] = [
                    {
                        m_k: (m_v.isoformat() if isinstance(m_v, pd.Timestamp) else m_v)
                        for m_k, m_v in model.items()
                    }
                    for model in v
                ]
            else:
                attrs_to_add[k] = v
        ds.attrs.update(attrs_to_add)
    return ds


def save_to_zarr(xarray_dataset, zarr_path):
    """
    Saves an xarray Dataset to a Zarr store.

    Args:
        xarray_dataset (xr.Dataset): The dataset to save.
        zarr_path (str): The file path for the Zarr store.
    """
    if xarray_dataset.dims:  # Check if dataset is not empty
        xarray_dataset.to_zarr(zarr_path, mode="w", compute=True)
        print(f"Data successfully saved to Zarr at: {zarr_path}")
    else:
        print("Xarray Dataset is empty (no dimensions found), nothing to save to Zarr.")


# --- Functions for Gridding and Interpolation ---


def load_gfs_grid_coordinates(gfs_zarr_path):
    """
    Loads latitude and longitude coordinates from a GFS Zarr dataset.
    These coordinates define the target grid for interpolation.

    Args:
        gfs_zarr_path (str): Path to an existing GFS Zarr dataset.

    Returns:
        tuple: A tuple containing (gfs_lats, gfs_lons) as NumPy arrays.
               Exits if the GFS Zarr cannot be loaded.
    """
    try:
        # Open the GFS Zarr dataset, ensuring coordinates are loaded into memory.
        with xr.open_zarr(gfs_zarr_path, consolidated=False, chunks={}) as gfs_ds:
            gfs_lats = gfs_ds.latitude.values
            gfs_lons = gfs_ds.longitude.values
            print(
                f"Loaded GFS grid: {len(gfs_lats)} latitudes, {len(gfs_lons)} longitudes."
            )
            return gfs_lats, gfs_lons
    except Exception as e:
        print(f"Error loading GFS grid from {gfs_zarr_path}: {e}")
        sys.exit(1)


# Helper for numerical interpolation (to be used with xr.apply_ufunc)
def _interpolate_single_time_slice_numerical(
    dwd_data_slice_numerical,
    dwd_spatial_points,
    gfs_grid_points_flat,
    gfs_lats_size,
    gfs_lons_size,
):
    """
    Applies `scipy.interpolate.griddata` for one numerical variable at one time step.
    This function is designed to be called by `xr.apply_ufunc` for parallel processing.

    Args:
        dwd_data_slice_numerical (np.ndarray): 1D array of numerical values for a single time step across DWD stations.
        dwd_spatial_points (np.ndarray): 2D array of (latitude, longitude) of DWD stations.
        gfs_grid_points_flat (np.ndarray): Flattened 2D array of (latitude, longitude) for the GFS grid.
        gfs_lats_size (int): Number of latitudes in the GFS grid.
        gfs_lons_size (int): Number of longitudes in the GFS grid.

    Returns:
        np.ndarray: 2D array (latitude, longitude) of interpolated values for the time slice.
    """
    # Filter out NaN values from station data for this specific time slice
    valid_indices = ~np.isnan(dwd_data_slice_numerical)

    # If no valid data points exist for this time slice, return an array of NaNs
    if not np.any(valid_indices):
        return np.full((gfs_lats_size, gfs_lons_size), np.nan, dtype=np.float32)

    # Get valid station points and their corresponding values for interpolation
    station_points_for_interp = dwd_spatial_points[valid_indices]
    values_at_stations = dwd_data_slice_numerical[valid_indices]

    # Perform the interpolation using 'nearest' neighbor method.
    # 'nearest' is chosen for robustness with irregularly spaced points and computational efficiency.
    interpolated_values_flat = griddata(
        station_points_for_interp,  # Source points (lat/lon of valid DWD stations)
        values_at_stations,  # Values at those valid stations
        gfs_grid_points_flat,  # Target grid points (flat list of lat/lon for GFS grid)
        method="nearest",  # Interpolation method
        fill_value=np.nan,  # Fill grid points outside the convex hull of source points with NaN
    )

    # Reshape the flat interpolated array back to the 2D grid shape (latitude, longitude)
    return interpolated_values_flat.reshape(gfs_lats_size, gfs_lons_size)


# Helper for categorical (string) interpolation (to be used with xr.apply_ufunc)
def _interpolate_single_time_slice_categorical(
    dwd_data_slice_ptype,
    dwd_spatial_points,
    gfs_grid_points_flat,
    gfs_lats_size,
    gfs_lons_size,
):
    """
    Applies nearest neighbor interpolation for one categorical (string) variable at one time step.
    Uses `scipy.spatial.cKDTree` for efficient nearest neighbor lookup.

    Args:
        dwd_data_slice_ptype (np.ndarray): 1D array of string values for a single time step across DWD stations.
        dwd_spatial_points (np.ndarray): 2D array of (latitude, longitude) of DWD stations.
        gfs_grid_points_flat (np.ndarray): Flattened 2D array of (latitude, longitude) for the GFS grid.
        gfs_lats_size (int): Number of latitudes in the GFS grid.
        gfs_lons_size (int): Number of longitudes in the GFS grid.

    Returns:
        np.ndarray: 2D array (latitude, longitude) of interpolated string values for the time slice.
    """
    # Filter out "none" or empty strings from station data for this time slice to build KDTree only on valid points.
    valid_indices = (
        (dwd_data_slice_ptype != "none")
        & (dwd_data_slice_ptype != "")
        & (~pd.isna(dwd_data_slice_ptype))
    )

    # If no valid categorical data points exist, return an array of "none" for the entire grid slice
    if not np.any(valid_indices):
        return np.full((gfs_lats_size, gfs_lons_size), "none", dtype=object)

    # Build KDTree using only the valid station points for this time slice
    station_points_for_kdtree = dwd_spatial_points[valid_indices]
    kdtree = cKDTree(station_points_for_kdtree)

    # Query the KDTree to find the nearest valid DWD station for each GFS grid point
    # Returns distances and indices of the nearest neighbors. We only need the indices.
    _, nearest_indices = kdtree.query(gfs_grid_points_flat, k=1)

    # Get the corresponding precipitation types from the original valid DWD data slice
    original_valid_types = dwd_data_slice_ptype[valid_indices]
    nearest_types = original_valid_types[nearest_indices]

    # Reshape the flat array of nearest types back to the 2D grid shape
    return nearest_types.reshape(gfs_lats_size, gfs_lons_size)


def interpolate_dwd_to_grid(
    dwd_ds_point,
    gfs_lats,
    gfs_lons,
    numerical_variables_to_interpolate,
    all_zarr_schema_vars,
):
    """
    Orchestrates the interpolation of DWD point data (from stations) onto a GFS-like rectilinear grid.
    Handles both numerical and categorical (string) variables.

    Args:
        dwd_ds_point (xr.Dataset): DWD data with 'station_id' and 'time' dimensions.
        gfs_lats (np.ndarray): Latitude coordinates of the target GFS grid.
        gfs_lons (np.ndarray): Longitude coordinates of the target GFS grid.
        numerical_variables_to_interpolate (list): List of numerical variable names to interpolate.
        all_zarr_schema_vars (tuple): Comprehensive list of all variables expected in the final Zarr schema.

    Returns:
        xr.Dataset: Gridded DWD data with 'time', 'latitude', 'longitude' dimensions, and interpolated variables.
    """
    # Create a meshgrid for the target GFS grid points (for scipy.interpolate.griddata and cKDTree)
    gfs_lon_grid, gfs_lat_grid = np.meshgrid(gfs_lons, gfs_lats)
    gfs_grid_points_flat = np.c_[gfs_lat_grid.ravel(), gfs_lon_grid.ravel()]

    # Extract DWD station coordinates (constant for all time steps)
    dwd_station_lats = dwd_ds_point.latitude.values
    dwd_station_lons = dwd_ds_point.longitude.values
    dwd_spatial_points = np.c_[dwd_station_lats, dwd_station_lons]

    # Initialize a new xarray Dataset to store the gridded data.
    # Its coordinates are the new grid (time, latitude, longitude).
    gridded_dwd_ds = xr.Dataset(
        coords={"time": dwd_ds_point.time, "latitude": gfs_lats, "longitude": gfs_lons}
    )

    # Define optimal chunking for the output gridded data.
    # Chunk over time dimension (e.g., 1 time step per chunk) and potentially subdivide lat/lon.
    # This enables Dask to parallelize computations efficiently.
    output_grid_chunk_shape = (1, len(gfs_lats) // 10, len(gfs_lons) // 10)

    # --- Interpolate Numerical Variables ---
    for var_name in numerical_variables_to_interpolate:
        if var_name in dwd_ds_point.data_vars:
            print(f"Scheduling interpolation for numerical variable: {var_name}")
            # Use `xr.apply_ufunc` to apply the `_interpolate_single_time_slice_numerical` function
            # across each time slice of the input DWD point data in parallel using Dask.
            gridded_data_var_da = xr.apply_ufunc(
                _interpolate_single_time_slice_numerical,
                dwd_ds_point[var_name],  # Input DataArray (station_id, time)
                input_core_dims=[
                    ["station_id"]
                ],  # Function operates on the 'station_id' dimension per time slice
                output_core_dims=[
                    ["latitude", "longitude"]
                ],  # Output has new 'latitude', 'longitude' dimensions
                exclude_dims={
                    "station_id"
                },  # 'station_id' is an internal detail, excluded from output dims
                dask="parallelized",  # Enable Dask for parallel execution
                output_dtypes=[
                    np.float32
                ],  # Specify the output data type (float for numerical data)
                kwargs={  # Pass fixed arguments to the helper function
                    "dwd_spatial_points": dwd_spatial_points,
                    "gfs_grid_points_flat": gfs_grid_points_flat,
                    "gfs_lats_size": len(gfs_lats),
                    "gfs_lons_size": len(gfs_lons),
                },
            ).chunk(
                {
                    "time": output_grid_chunk_shape[
                        0
                    ],  # Rechunk the Dask array after creation
                    "latitude": output_grid_chunk_shape[1],
                    "longitude": output_grid_chunk_shape[2],
                }
            )

            gridded_dwd_ds[var_name] = gridded_data_var_da
        else:
            print(
                f"Variable {var_name} not found in DWD data, adding as NaN placeholder to gridded dataset."
            )
            # If a numerical variable is missing, add it as a Dask array filled with NaNs
            gridded_dwd_ds[var_name] = (
                ("time", "latitude", "longitude"),
                da.full(
                    (len(dwd_ds_point.time), len(gfs_lats), len(gfs_lons)),
                    np.nan,
                    dtype=np.float32,
                    chunks=output_grid_chunk_shape,
                ),
            )

    # --- Interpolate Categorical (String) Variables like PTYPE_surface ---
    if "PTYPE_surface" in dwd_ds_point.data_vars:
        print("Scheduling interpolation for PTYPE_surface (string type)...")
        # Use `xr.apply_ufunc` for categorical interpolation. Output dtype must be `object` for strings.
        gridded_ptype_var_da = xr.apply_ufunc(
            _interpolate_single_time_slice_categorical,
            dwd_ds_point["PTYPE_surface"],
            input_core_dims=[["station_id"]],
            output_core_dims=[["latitude", "longitude"]],
            exclude_dims={"station_id"},
            dask="parallelized",
            output_dtypes=[object],  # Output dtype is `object` for string arrays
            kwargs={
                "dwd_spatial_points": dwd_spatial_points,
                "gfs_grid_points_flat": gfs_grid_points_flat,
                "gfs_lats_size": len(gfs_lats),
                "gfs_lons_size": len(gfs_lons),
            },
        ).chunk(
            {
                "time": output_grid_chunk_shape[0],
                "latitude": output_grid_chunk_shape[1],
                "longitude": output_grid_chunk_shape[2],
            }
        )

        gridded_dwd_ds["PTYPE_surface"] = gridded_ptype_var_da
    else:
        print(
            "PTYPE_surface not found in DWD data, adding as 'none' placeholder to gridded dataset."
        )
        # If PTYPE_surface is missing, add it as a Dask array filled with "none" strings.
        gridded_dwd_ds["PTYPE_surface"] = (
            ("time", "latitude", "longitude"),
            da.full(
                (len(dwd_ds_point.time), len(gfs_lats), len(gfs_lons)),
                "none",
                dtype=object,
                chunks=output_grid_chunk_shape,
            ),
        )

    # --- Add Placeholder Variables for All Other Zarr Schema Variables ---
    # This loop ensures that `gridded_dwd_ds` contains all variables defined in `zarrVars`,
    # filling with NaNs if they were not directly interpolated from MOSMIX-S data.
    for var_name in all_zarr_schema_vars:
        if var_name not in gridded_dwd_ds.data_vars and var_name != "time":
            print(f"Adding placeholder NaN variable: {var_name}")
            # Determine appropriate dtype for the placeholder (object for strings, float for numerical)
            dtype_for_placeholder = (
                object if var_name == "PTYPE_surface" else np.float32
            )  # PTYPE is the only string variable in schema

            gridded_dwd_ds[var_name] = (
                ("time", "latitude", "longitude"),
                da.full(
                    (len(dwd_ds_point.time), len(gfs_lats), len(gfs_lons)),
                    np.nan,
                    dtype=dtype_for_placeholder,
                    chunks=output_grid_chunk_shape,
                ),
            )

    return gridded_dwd_ds


# --- Main Ingest Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # URL for the DWD MOSMIX-S latest 240-hour forecast KMZ file
    dwd_mosmix_url = "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz"
    downloaded_kmz_file = "MOSMIX_S_LATEST_240.kmz"

    # Base directory for all processing and output files.
    forecast_process_dir = os.getenv("forecast_process_dir", "/home/ubuntu/Weather/DWD")
    # Temporary directory for downloaded files.
    tmpDIR = os.getenv("tmp_dir", os.path.join(forecast_process_dir, "Downloads"))
    # Final destination directory for processed Zarr files.
    forecast_path = os.getenv("forecast_path", "/home/ubuntu/Weather/Prod/DWD")

    # Path to an existing GFS Zarr file that will be used as the reference grid.
    # This environment variable MUST be set for the script to run correctly.
    gfs_zarr_reference_path = os.getenv(
        "GFS_ZARR_PATH", "/path/to/your/GFS.zarr"
    )  # TODO: Set this environment variable!

    # Defines where the final Zarr file should be saved: "Download" (local) or "S3" (AWS S3).
    saveType = os.getenv("save_type", "Download")
    # AWS Credentials for S3 operations (should be set as environment variables).
    aws_access_key_id = os.environ.get("AWS_KEY", "")
    aws_secret_access_key = os.environ.get("AWS_SECRET", "")

    # Initialize S3 filesystem object if saving to S3.
    s3 = None
    if saveType == "S3":
        s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    # --- Directory Setup and Cleanup ---
    # Create the main processing directory; remove and recreate if it already exists for a clean slate.
    os.makedirs(forecast_process_dir, exist_ok=True)
    if os.path.exists(forecast_process_dir):
        shutil.rmtree(forecast_process_dir)  # Cleans up any previous partial runs
    os.makedirs(forecast_process_dir)

    # Create the temporary download directory.
    os.makedirs(tmpDIR, exist_ok=True)

    # Create the final forecast output directory if saving locally.
    if saveType == "Download":
        os.makedirs(forecast_path, exist_ok=True)

    T0 = time.time()  # Start timer for script execution

    # --- Step 1: Download and Parse DWD MOSMIX-S KML/KMZ Data ---
    print(f"\n--- Attempting to download DWD MOSMIX data from: {dwd_mosmix_url} ---")
    try:
        response = requests.get(dwd_mosmix_url, stream=True)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        with open(os.path.join(tmpDIR, downloaded_kmz_file), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(
            f"File downloaded successfully to: {os.path.join(tmpDIR, downloaded_kmz_file)}"
        )
    except Exception as e:
        print(f"Error downloading DWD data: {e}. Exiting.")
        sys.exit(1)

    print(
        f"\n--- Reading and Parsing KML from {os.path.join(tmpDIR, downloaded_kmz_file)} ---"
    )
    df_data, global_metadata = parse_mosmix_kml(
        os.path.join(tmpDIR, downloaded_kmz_file)
    )
    if df_data.empty:
        print("No data extracted from KML/KMZ. Exiting.")
        sys.exit(1)

    # --- Ensure unique stations based on latitude and longitude ---
    # This step filters out stations that might have different IDs but identical lat/lon coordinates.
    initial_station_count = df_data["station_id"].nunique()
    # Use drop_duplicates on the lat/lon subset, keeping the first occurrence.
    df_data_unique_stations = df_data.drop_duplicates(
        subset=["latitude", "longitude"], keep="first"
    )
    unique_station_count = df_data_unique_stations["station_id"].nunique()

    if initial_station_count > unique_station_count:
        print(
            f"Removed {initial_station_count - unique_station_count} duplicate stations based on latitude/longitude."
        )
        print(
            f"Proceeding with {unique_station_count} unique stations for interpolation."
        )
    else:
        print(
            "No duplicate stations found based on latitude/longitude in the raw data."
        )

    df_data = df_data_unique_stations  # Use the filtered DataFrame for all subsequent processing

    # Extract the base time from the KML metadata (IssueTime) for update checks.
    base_time = global_metadata.get("IssueTime", datetime.utcnow())
    print(f"Base time for this ingest run (from KML metadata): {base_time}")

    # --- Check for Updates (Skip if no new data) ---
    # This prevents redundant processing if the latest downloaded data is not newer than the last ingested.
    final_time_pickle_path = os.path.join(forecast_path, "DWD.time.pickle")
    if saveType == "S3":
        s3_bucket_name = os.getenv(
            "s3_bucket", "your-s3-bucket"
        )  # Ensure S3 bucket is configured
        s3_time_pickle_key = os.path.join("ForecastTar_v2", "DWD.time.pickle")
        try:
            with s3.open(f"{s3_bucket_name}/{s3_time_pickle_key}", "rb") as f:
                previous_base_time = pickle.load(f)
            if previous_base_time >= base_time:
                print("No new update to DWD found in S3, ending script.")
                sys.exit()
        except FileNotFoundError:
            print("Previous DWD time pickle not found in S3, proceeding with ingest.")
        except Exception as e:
            print(f"Error checking previous DWD time in S3: {e}. Proceeding.")
    else:  # saveType == "Download" (local check)
        if os.path.exists(final_time_pickle_path):
            with open(final_time_pickle_path, "rb") as file:
                previous_base_time = pickle.load(file)
            if previous_base_time >= base_time:
                print("No new update to DWD found locally, ending script.")
                sys.exit()

    # --- Step 2: Convert Pandas DataFrame to xarray Dataset (Point-Based Format) ---
    # This step transforms the flat DataFrame into an xarray Dataset with station_id and time as dimensions.
    print("\n--- Converting DataFrame to xarray Dataset (point-based) ---")
    dwd_ds_point = convert_df_to_xarray(df_data, global_metadata=global_metadata)
    if (
        not dwd_ds_point.dims
    ):  # Check if the resulting dataset has dimensions (i.e., is not empty)
        print("Point-based xarray Dataset is empty after conversion. Exiting.")
        sys.exit(1)
    print("Point-based DWD Dataset preview:\n", dwd_ds_point)

    # --- Step 3: Load GFS Grid Coordinates for Target Interpolation Grid ---
    # This provides the target latitude and longitude arrays for the interpolation.
    print(f"\n--- Loading GFS grid coordinates from: {gfs_zarr_reference_path} ---")
    gfs_lats, gfs_lons = load_gfs_grid_coordinates(gfs_zarr_reference_path)
    if gfs_lats is None or gfs_lons is None:
        print("Failed to load GFS grid coordinates. Exiting.")
        sys.exit(1)

    # --- Step 4: Interpolate DWD Point Data to the GFS Grid ---
    # This is the core transformation where point data is converted to gridded data (4D array).
    # Dask is leveraged internally by `xr.apply_ufunc` for parallel processing.
    print("\n--- Interpolating DWD point data to GFS grid ---")
    gridded_dwd_ds = interpolate_dwd_to_grid(
        dwd_ds_point, gfs_lats, gfs_lons, api_target_numerical_variables, zarrVars
    )

    print("\nGridded DWD Dataset preview (after interpolation):\n", gridded_dwd_ds)
    print(f"\nGridded DWD Dataset dimensions: {gridded_dwd_ds.dims}")
    print(
        f"Gridded DWD Dataset data variables: {list(gridded_dwd_ds.data_vars.keys())}"
    )

    # --- Step 5: Save Gridded DWD Dataset to Zarr ---
    # The final gridded and interpolated data is saved as a Zarr store.
    gridded_zarr_output_full_path = os.path.join(
        forecast_process_dir, "DWD_Gridded.zarr"
    )
    print(
        f"\n--- Saving Gridded xarray Dataset to Zarr: {gridded_zarr_output_full_path} ---"
    )
    save_to_zarr(gridded_dwd_ds, gridded_zarr_output_full_path)

    # --- Step 6: Save Original Station Metadata (for API's 'nearest station' logic) ---
    # This saves a separate JSON file containing the original unique station details,
    # which can be used by the API to find the nearest station to a query point.
    station_metadata_df = (
        df_data[["station_id", "station_name", "latitude", "longitude", "altitude"]]
        .drop_duplicates(subset="station_id")
        .set_index("station_id")
    )
    station_metadata_path = os.path.join(
        forecast_process_dir, "DWD_Station_Metadata.json"
    )
    print(f"\n--- Saving original station metadata to: {station_metadata_path} ---")
    station_metadata_df.to_json(
        station_metadata_path, orient="index"
    )  # Saves as a dictionary-like JSON (station_id as key)
    print("Station metadata saved.")

    # --- Step 7: Final Data Transfer (Upload to S3 or Copy Locally) ---
    # This step moves the processed Zarr file and associated metadata to their final destination.
    final_gridded_zarr_target_path = os.path.join(
        forecast_path, "DWD.zarr"
    )  # Standardized name for API consumption
    final_time_pickle_path_dest = os.path.join(forecast_path, "DWD.time.pickle")

    if saveType == "S3":
        s3_bucket_name = os.getenv(
            "s3_bucket", "your-s3-bucket"
        )  # Ensure this env var is set
        s3_gridded_zarr_key = os.path.join(
            "ForecastTar_v2", "DWD.zarr"
        )  # Example S3 key path for gridded Zarr
        s3_time_pickle_key_upload = os.path.join(
            "ForecastTar_v2", "DWD.time.pickle"
        )  # S3 key for time pickle
        s3_station_metadata_key = os.path.join(
            "ForecastTar_v2", "DWD_Station_Metadata.json"
        )  # S3 key for station metadata

        print(
            f"Uploading gridded Zarr from {gridded_zarr_output_full_path} to S3://{s3_bucket_name}/{s3_gridded_zarr_key}"
        )
        s3.put(
            gridded_zarr_output_full_path,
            f"{s3_bucket_name}/{s3_gridded_zarr_key}",
            recursive=True,
        )

        # Save and upload time pickle
        temp_time_pickle_source_path = os.path.join(
            forecast_process_dir, "DWD.time.pickle"
        )
        with open(temp_time_pickle_source_path, "wb") as file:
            pickle.dump(base_time, file)
        print(
            f"Uploading time pickle from {temp_time_pickle_source_path} to S3://{s3_bucket_name}/{s3_time_pickle_key_upload}"
        )
        s3.put_file(
            temp_time_pickle_source_path,
            f"{s3_bucket_name}/{s3_time_pickle_key_upload}",
        )

        # Upload station metadata JSON
        print(
            f"Uploading station metadata from {station_metadata_path} to S3://{s3_bucket_name}/{s3_station_metadata_key}"
        )
        s3.put_file(
            station_metadata_path, f"{s3_bucket_name}/{s3_station_metadata_key}"
        )

        print("All files uploaded to S3.")

    else:  # saveType == "Download" (local copy operation)
        # Save and move time pickle to final local path
        temp_time_pickle_source_path = os.path.join(
            forecast_process_dir, "DWD.time.pickle"
        )
        with open(temp_time_pickle_source_path, "wb") as file:
            pickle.dump(base_time, file)
        shutil.move(temp_time_pickle_source_path, final_time_pickle_path_dest)

        # Copy gridded Zarr directory to final local path
        shutil.copytree(
            gridded_zarr_output_full_path,
            final_gridded_zarr_target_path,
            dirs_exist_ok=True,
        )

        # Copy station metadata JSON to final local path
        shutil.copy(
            station_metadata_path,
            os.path.join(forecast_path, "DWD_Station_Metadata.json"),
        )
        print(f"All files copied locally to {forecast_path}.")

    # --- Final Cleanup ---
    # Remove temporary processing directories.
    if os.path.exists(forecast_process_dir):
        shutil.rmtree(forecast_process_dir)
        print(f"\nCleaned up processing directory: {forecast_process_dir}")
    if os.path.exists(tmpDIR):
        shutil.rmtree(tmpDIR)
        print(f"Cleaned up downloads directory: {tmpDIR}")

    T_end = time.time()  # End timer
    print(f"\nTotal script execution time: {T_end - T0:.2f} seconds")
