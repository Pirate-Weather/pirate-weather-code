# %% DWD MOSMIX-S Hourly Processing script - Gridded Interpolation
# This script downloads the latest DWD MOSMIX-S KML/KMZ data,
# parses the point-based station forecasts, cleans and standardizes the data,
# interpolates it onto a specified GFS reference grid, and then saves the
# resulting gridded (4D) data to a Zarr store for API consumption.
# It also stores original station metadata for lookup purposes.
#
# Author: Alexander Rey
# Date: June 2025


# NOTE: You may ask yourself, why does it require 20 GB of RAM to process a 40 MB KMZ?
# The issue is that Zarr doesn't nicely support sparse arrays, and the DWD MOSMIX-S is mostly empty
# As soon as this issue is resolved in Zarr, this script can be optimized further.
# https://github.com/zarr-developers/zarr-specs/issues/245

# NOTE 2: Multiple stations can map to the same grid cell (e.g., OSAKA station ID 47772 and
# OSAKA AIRPORT station ID 47771 are ~10km apart). This function uses the closest station for interpolation.

# %% Import modules
import io
import logging
import os
import pickle
import shutil
import sys
import time
import zipfile
from datetime import UTC, datetime

import dask.array as da
import numpy as np
import pandas as pd
import requests
import s3fs
import xarray as xr
import zarr
import zarr.storage
from dask.diagnostics import ProgressBar
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from pykml import parser
from sklearn.neighbors import BallTree
from tqdm import tqdm  # safe to ignore if not using log="tqdm"

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    DWD_RADIUS,
    FINAL_CHUNK_SIZES,
    interpolate_temporal_gaps_efficiently,
)

# Distance to interpolate stations to grid (km)
radius_km = DWD_RADIUS

# Set up basic logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define KML and DWD namespaces for easier parsing of the XML structure.
# These namespaces are crucial for correctly locating elements within the KML file.
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
DWD_NAMESPACE = (
    "{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}"
)

# Define the comprehensive list of variables to be saved in the final Zarr dataset.
# during the interpolation process to maintain a consistent schema.
zarr_vars = (
    "time",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "PRES_meansealevel",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "GUST_surface",
    "APCP_surface",
    "TCDC_entireatmosphere",
    "VIS_surface",
    "PTYPE_surface",
    "DSWRF_surface",
)

# Define the subset of standardized variables that are directly sourced from
# MOSMIX-S (or calculated from it) and will be actively interpolated onto the grid.
api_target_numerical_variables = [
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",  # Relative Humidity is calculated
    "PRES_meansealevel",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "GUST_surface",
    "APCP_surface",  # Now in cm
    "PTYPE_surface",
    "TCDC_entireatmosphere",
    "VIS_surface",
    "DSWRF_surface",  # New: calculated from Rad1h
]


# --- Core Helper Functions ---
# Relative humidity is calculated using MetPy's `relative_humidity_from_dewpoint`.
# This removes the need for custom vapor-pressure approximations and leverages
# a tested implementation that supports pint units.


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
                logging.error(f"Error: No KML file found inside {kml_filepath}.")
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
        logging.error("Error: No global forecast time steps found in KML Document.")
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
    logging.info(f"Found {len(global_forecast_times)} global forecast time steps.")

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
                logging.warning(
                    f"Warning: Could not parse coordinates for station ID {station_id}: {coordinates_elem.text}. Setting to NaN."
                )
                lon, lat, alt = np.nan, np.nan, np.nan

        extended_data_pm = placemark.find(f"{KML_NAMESPACE}ExtendedData")
        if extended_data_pm is None:
            logging.warning(
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
                    logging.warning(
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


def process_dwd_df(df, global_metadata=None):
    """
    Processes the DataFrame extracted from DWD MOSMIX KML parsing.

    Args:
        df (pd.DataFrame): Input DataFrame from KML parsing.
        global_metadata (dict, optional): Global metadata to add as Dataset attributes.

    Returns:
        xr.Dataset: Point-based xarray Dataset with 'station_id' and 'time' dimensions.
    """

    # --- Standardize variable names and units from DWD KML to API schema ---
    # Values remain in their original DWD units (Kelvin, Pa, m/s, mm/h) before interpolation.
    if "TTT" in df.columns:
        df["TMP_2maboveground"] = df["TTT"]
    if "Td" in df.columns:
        df["DPT_2maboveground"] = df["Td"]

    # Calculate Relative Humidity (RH_2maboveground) from Temperature (TTT) and Dew Point (Td)
    if "TMP_2maboveground" in df.columns and "DPT_2maboveground" in df.columns:
        # Use MetPy which expects pint quantities. We multiply by units.kelvin
        # and return the dimensionless magnitude (0-1) scaled to percent (0-100).
        def _rh_metpy(t, td):
            t_q = t * units.kelvin
            td_q = td * units.kelvin
            rh_q = relative_humidity_from_dewpoint(t_q, td_q)
            # Convert to percent
            return (rh_q.magnitude * 100).astype(np.float32)

        df["RH_2maboveground"] = _rh_metpy(
            df["TMP_2maboveground"].to_numpy(),
            df["DPT_2maboveground"].to_numpy(),
        ).astype(np.float32)

    df["PRES_meansealevel"] = df["PPPP"]

    # Calculate U- and V-components of wind from speed (FF) and direction (DD)
    wind_speed_ms = df["FF"]
    # Convert meteorological direction (where wind comes from, 0=N, 90=E)
    # to mathematical/cartesian direction (where wind goes, 0=E, 90=N) and then to radians.
    wind_direction_rad = np.deg2rad(270 - df["DD"])
    df["UGRD_10maboveground"] = wind_speed_ms * np.cos(wind_direction_rad)
    df["VGRD_10maboveground"] = wind_speed_ms * np.sin(wind_direction_rad)

    df["GUST_surface"] = df["FX1"]

    # Accumulated Precipitation (RR1c in Kg/m2)
    # MOSMIX-S RR1c is 1-hour accumulation.
    df["APCP_surface"] = df["RR1c"]

    df["TCDC_entireatmosphere"] = df["N"]
    df["VIS_surface"] = df["VV"]

    # Precipitation Type (WW)
    # From: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
    df["PTYPE_surface"] = df["ww"]

    # Downward Short-Wave Radiation Flux (DSWRF_surface) from Rad1h
    # Rad1h unit is kJ/m^2 (hourly sum). Convert to W/m^2 (Joules per second per square meter).
    # Conversion: (kJ/m^2 * 1000 J/kJ) / 3600 s/hour = J/m^2/s = W/m^2. So, divide by 3.6.
    df["DSWRF_surface"] = df["Rad1h"] / 3.6

    return df


def interpolate_dwd_to_grid_knearest_dask(
    df,
    var_cols,
    radius_km,
    k_max=4,
    time_col="time",
    lat_col="latitude",
    lon_col="longitude",
    station_col="station_id",
    dtype=np.float32,
    log="print",  # "print" | "tqdm" | "none"
    print_every=10,  # used if log="print"
    time_chunk=24,  # Dask chunk size in time
):
    """
    Vectorized nearest-neighbour grid interpolation with dense, Dask-backed output.

    Each dataframe row is assigned to up to k_max nearest grid cells
    within radius_km. For each (time, lat, lon) grid cell, ONLY THE NEAREST
    station is used, provided it is within radius_km. Otherwise, the cell is NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: station_col, time_col, lat_col, lon_col, var_cols.
        Lat/lon in degrees. Lon assumed in [0, 360) for the GFS grid.
    var_cols : str or list of str
        Variables to interpolate.
    radius_km : float
        Max radius in km for a station to influence a grid cell.
    k_max : int
        Maximum number of grid cells to which each station can contribute.
    time_col, lat_col, lon_col, station_col : str
        Column names.
    dtype : numpy dtype
        Output dtype (np.float32 recommended).
    log : {"print","tqdm","none"}
        Logging mode.
    print_every : int
        Print frequency if log="print".
    time_chunk : int
        Dask chunk size for the time dimension.

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, lat, lon); each data_var is a dense,
        Dask-backed 3D array. Each grid cell/time uses the closest station
        within radius_km, or NaN if none.
    """
    if isinstance(var_cols, str):
        var_cols = [var_cols]

    def _log(msg):
        if log == "print":
            print(msg)

    # -------------------------------------------------
    # 0) Define global 0.25° GFS grid
    # -------------------------------------------------
    _log("Building 0.25° GFS grid…")

    gfs_lats = np.arange(-90, 90, 0.25)
    gfs_lons = np.arange(0, 360, 0.25)
    ny = gfs_lats.size
    nx = gfs_lons.size

    lon_grid, lat_grid = np.meshgrid(gfs_lons, gfs_lats)
    grid_coords_deg = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    grid_coords_rad = np.radians(grid_coords_deg)

    # -------------------------------------------------
    # 1) Station → k-nearest neighbours on grid
    # -------------------------------------------------
    _log("Querying k-nearest neighbours for stations…")

    tree = BallTree(grid_coords_rad, metric="haversine")

    station_meta = df.drop_duplicates(subset=station_col)[
        [station_col, lat_col, lon_col]
    ]
    station_ids = station_meta[station_col].to_numpy()
    pts_rad = np.radians(station_meta[[lat_col, lon_col]].to_numpy())

    # k_max nearest neighbours per station (in radians)
    dist_rad, inds_flat = tree.query(pts_rad, k=k_max)  # (n_stations, k_max)

    radius_rad = radius_km / 6371.0
    mask_nn = dist_rad <= radius_rad
    inds_flat = np.where(
        mask_nn, inds_flat, -1
    )  # -1 marks "no neighbour within radius"

    station_indexer = {sid: i for i, sid in enumerate(station_ids)}

    # -------------------------------------------------
    # 2) Time indexing & sorting
    # -------------------------------------------------
    _log("Indexing time dimension…")

    df_sorted = df.sort_values(time_col)
    unique_times = df_sorted[time_col].drop_duplicates().to_numpy()
    nt = len(unique_times)

    if nt == 0:
        return xr.Dataset(
            coords={"time": unique_times, "lat": gfs_lats, "lon": gfs_lons}
        )

    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    t_idx = df_sorted[time_col].map(time_to_idx).to_numpy()  # shape (n_rows,)

    stn_full = df_sorted[station_col].to_numpy()
    stn_idx_full = np.array([station_indexer[s] for s in stn_full], dtype=np.int64)

    # -------------------------------------------------
    # 3) Expand row → neighbour grid indices (vectorized)
    # -------------------------------------------------
    _log("Linking rows to neighbour grid cells…")

    # Each row maps to k_max neighbours for its station
    neigh_flat = inds_flat[stn_idx_full]  # (n_rows, k_max)
    neigh_time = np.repeat(t_idx[:, None], k_max, axis=1)  # (n_rows, k_max)
    neigh_dist = dist_rad[stn_idx_full]  # (n_rows, k_max)

    # Flatten and filter invalid neighbours
    flat_flat = neigh_flat.ravel()
    flat_time = neigh_time.ravel()
    flat_dist = neigh_dist.ravel()

    valid = flat_flat >= 0
    if not np.any(valid):
        # No neighbours within radius_km at all
        ds = xr.Dataset()
        for col in var_cols:
            arr = np.full((nt, ny, nx), np.nan, dtype=dtype)
            darr = da.from_array(arr, chunks=(min(time_chunk, nt), ny, nx))
            ds[col] = xr.DataArray(
                darr,
                dims=("time", "lat", "lon"),
                coords={"time": unique_times, "lat": gfs_lats, "lon": gfs_lons},
                name=col,
            )
        _log("Done (no valid neighbours within radius; all NaNs).")
        return ds

    flat_flat = flat_flat[valid].astype(np.int64)
    flat_time = flat_time[valid].astype(np.int64)
    flat_dist = flat_dist[valid].astype(np.float32)

    # Grid indices (y, x)
    iy, ix = np.unravel_index(flat_flat, (ny, nx))
    iy = iy.astype(np.int64)
    ix = ix.astype(np.int64)

    # Linear index for (time, y, x)
    n_cells = nt * ny * nx
    linear_index = flat_time * (ny * nx) + iy * nx + ix  # shape (n_valid,)

    # Row indices expanded to neighbours
    row_idx = np.arange(df_sorted.shape[0], dtype=np.int64)
    row_idx_rep = np.repeat(row_idx[:, None], k_max, axis=1).ravel()[valid]

    # -------------------------------------------------
    # 4) For each grid cell/time, find the nearest station (single pass)
    # -------------------------------------------------
    _log("Computing nearest-station per grid cell/time (argmin)…")

    # Sort candidates by cell index then distance
    # lexsort uses last key as primary sort key -> primary: linear_index, secondary: flat_dist
    order = np.lexsort((flat_dist, linear_index))
    lin_sorted = linear_index[order]
    dist_sorted = flat_dist[order]
    row_sorted = row_idx_rep[order]

    # Take the first candidate per unique cell index → nearest station
    uniq_lin, idx_first = np.unique(lin_sorted, return_index=True)
    best_lin = uniq_lin
    best_dist = dist_sorted[idx_first]
    best_row = row_sorted[idx_first]

    # Enforce radius cutoff in radians
    mask_radius = best_dist <= radius_rad
    best_lin = best_lin[mask_radius]
    best_row = best_row[mask_radius]

    # If still nothing, everything is NaN
    if best_lin.size == 0:
        ds = xr.Dataset()
        for col in var_cols:
            arr = np.full((nt, ny, nx), np.nan, dtype=dtype)
            darr = da.from_array(arr, chunks=(min(time_chunk, nt), ny, nx))
            ds[col] = xr.DataArray(
                darr,
                dims=("time", "lat", "lon"),
                coords={"time": unique_times, "lat": gfs_lats, "lon": gfs_lons},
                name=col,
            )
        _log("Done (no winners within radius; all NaNs).")
        return ds

    # -------------------------------------------------
    # 5) Build dense NumPy arrays for each variable and wrap with Dask
    # -------------------------------------------------
    _log(f"Building dense Dask arrays for {len(var_cols)} variable(s)…")

    data_vars = {}
    var_iter = tqdm(var_cols, desc="Variables") if log == "tqdm" else var_cols

    for col in var_iter:
        vals = df_sorted[col].to_numpy().astype(dtype)

        # Allocate flattened grid
        arr_flat = np.full(n_cells, np.nan, dtype=dtype)

        # Assign values of chosen rows to their grid cells
        # Note: if the chosen row's value is NaN, the cell is NaN.
        arr_flat[best_lin] = vals[best_row]

        # Reshape back to (time, lat, lon)
        arr = arr_flat.reshape(nt, ny, nx)

        # Wrap in Dask
        chunk_t = min(time_chunk, nt) if nt > 0 else 1
        darr = da.from_array(arr, chunks=(chunk_t, ny, nx))

        data_vars[col] = xr.DataArray(
            darr,
            dims=("time", "lat", "lon"),
            coords={"time": unique_times, "lat": gfs_lats, "lon": gfs_lons},
            name=col,
        )

    _log("Building xarray.Dataset (dense, dask-backed)…")

    ds = xr.Dataset(data_vars=data_vars)

    _log("Done. Dataset is lazy; ready for .to_zarr() or .compute().")

    return ds


def build_grid_to_stations_map(
    df,
    radius_km=DWD_RADIUS,
    lat_col="latitude",
    lon_col="longitude",
    station_col="station_id",
    station_name_col="station_name",
    log="print",
):
    """
    Build a mapping from grid cells to nearby stations within radius.

    For each grid cell in the 0.25° GFS grid, stores a list of stations
    that contributed data to that cell (i.e., stations within radius_km).

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: station_col, station_name_col, lat_col, lon_col
    radius_km : float
        Max radius in km for stations to be associated with a grid cell
    lat_col, lon_col, station_col, station_name_col : str
        Column names
    log : {"print","tqdm","none"}
        Logging mode

    Returns
    -------
    dict
        A dictionary where keys are (y_idx, x_idx) tuples and values are
        lists of dicts with station info: [{"id": "...", "name": "...", "lat": ..., "lon": ...}, ...]
    """

    def _log(msg):
        if log == "print":
            print(msg)

    _log("Building grid-to-stations mapping...")

    # Define global 0.25° GFS grid
    gfs_lats = np.arange(-90, 90, 0.25)
    gfs_lons = np.arange(0, 360, 0.25)
    ny = gfs_lats.size
    nx = gfs_lons.size

    lon_grid, lat_grid = np.meshgrid(gfs_lons, gfs_lats)
    grid_coords_deg = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    grid_coords_rad = np.radians(grid_coords_deg)

    # Build BallTree for grid cells
    tree = BallTree(grid_coords_rad, metric="haversine")

    # Get unique stations with metadata
    station_meta = df.drop_duplicates(subset=station_col)[
        [station_col, station_name_col, lat_col, lon_col]
    ]

    # For each station, find all grid cells within radius
    radius_rad = radius_km / 6371.0

    # Query all grid cells within radius for each station
    # Store closest station: key=(y, x), value=(distance, station_dict)
    grid_best_match = {}

    station_iter = (
        tqdm(
            station_meta.itertuples(index=False),
            total=len(station_meta),
            desc="Mapping stations to grid",
        )
        if log == "tqdm"
        else station_meta.itertuples(index=False)
    )

    for station in station_iter:
        station_id = getattr(station, station_col)
        station_name = getattr(station, station_name_col)
        station_lat = getattr(station, lat_col)
        station_lon = getattr(station, lon_col)

        # Convert station coords to radians for query
        stn_rad = np.radians([[station_lat, station_lon]])

        # Find all grid cells within radius
        indices, distances = tree.query_radius(
            stn_rad, r=radius_rad, return_distance=True
        )

        # Unwrap arrays (query_radius returns arrays of arrays)
        indices = indices[0]
        distances = distances[0]

        # Convert longitude back to [-180, 180] for API output
        output_lon = ((station_lon + 180) % 360) - 180

        station_info = {
            "id": station_id,
            "name": station_name,
            "lat": round(station_lat, 4),
            "lon": round(output_lon, 4),
        }

        # Convert flat indices to (y, x) grid indices
        for flat_idx, dist in zip(indices, distances):
            y_idx, x_idx = np.unravel_index(flat_idx, (ny, nx))
            grid_key = (y_idx, x_idx)

            # Check if this station is closer than the existing one for this grid cell
            if grid_key not in grid_best_match or dist < grid_best_match[grid_key][0]:
                grid_best_match[grid_key] = (dist, station_info)

    # Convert best matches to the expected format: keys are (y_idx, x_idx), values are lists of dicts
    grid_to_stations = {k: [v[1]] for k, v in grid_best_match.items()}

    _log(f"Mapped {len(station_meta)} stations to {len(grid_to_stations)} grid cells.")

    return grid_to_stations


# --- Main Ingest Execution Block ---
# This script follows the same top-level script style as other ingest scripts in
# the repository (no `main()` wrapper). It expects environment variables to be
# set for paths and S3 options when run as a script.

# --- Configuration ---
# URL for the DWD MOSMIX-S latest 240-hour forecast KMZ file
dwd_mosmix_url = "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz"
downloaded_kmz_file = "MOSMIX_S_LATEST_240.kmz"

# Script ingest version string
ingest_version = INGEST_VERSION_STR

# Base directory for all processing and output files.
forecast_process_dir = os.getenv("forecast_process_dir", "/mnt/nvme/data/DWD_MOSMIX")
forecast_process_path = forecast_process_dir + "/DWD_Process"
# Temporary directory for downloaded files.
tmp_dir = os.getenv("tmp_dir", os.path.join(forecast_process_dir, "Downloads"))
# Final destination directory for processed Zarr files.
forecast_path = os.getenv("forecast_path", "/mnt/nvme/data/Prod/DWD")
historic_path = os.getenv("historic_path", "/mnt/nvme/data/DWD/Historic")

# Defines where the final Zarr file should be saved: "Download" (local) or "S3" (AWS S3).
save_type = os.getenv("save_type", "Download")
# AWS Credentials for S3 operations (should be set as environment variables).
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

# Initialize S3 filesystem object if saving to S3.
s3 = None
if save_type == "S3":
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)


# --- Directory Setup and Cleanup ---
# Create the main processing directory; clear its contents if it already exists to avoid
# deleting the directory itself (safer for processes that depend on the parent dir).
def _clear_directory(path):
    """Remove all contents of a directory but keep the directory itself."""
    if not os.path.exists(path):
        return
    for entry in os.scandir(path):
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)
        except Exception:
            logging.warning(f"Could not remove {entry.path} during clear_directory.")


if os.path.exists(forecast_process_dir):
    _clear_directory(forecast_process_dir)
else:
    os.makedirs(forecast_process_dir, exist_ok=True)

# Create the temporary download directory.
os.makedirs(tmp_dir, exist_ok=True)

# Create the final forecast output directory if saving locally (versioned by ingest).
if save_type == "Download":
    os.makedirs(os.path.join(forecast_path, ingest_version), exist_ok=True)
    os.makedirs(historic_path, exist_ok=True)

start_time = time.time()  # Start timer for script execution


# --- Step 1: Download and Parse DWD MOSMIX-S KML/KMZ Data ---
def download_mosmix_file(url, local_path):
    """
    Downloads a MOSMIX file from a URL to a local path.
    """
    try:
        response = requests.get(url, stream=True)
        # Check if the file exists (404 handling)
        if response.status_code == 404:
            logging.warning(f"File not found: {url}")
            return False

        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"File downloaded successfully to: {local_path}")
        return True
    except Exception as e:
        logging.warning(f"Error downloading {url}: {e}")
        return False


# --- Step 1: Download and Parse DWD MOSMIX-S KML/KMZ Data ---
logging.info(f"\n--- Attempting to download DWD MOSMIX data from: {dwd_mosmix_url} ---")
if not download_mosmix_file(dwd_mosmix_url, os.path.join(tmp_dir, downloaded_kmz_file)):
    logging.critical("Error downloading latest DWD data. Exiting.")
    sys.exit(1)

logging.info(
    f"\n--- Reading and Parsing KML from {os.path.join(tmp_dir, downloaded_kmz_file)} ---"
)
df_data, global_metadata = parse_mosmix_kml(os.path.join(tmp_dir, downloaded_kmz_file))
if df_data.empty:
    logging.critical("No data extracted from KML/KMZ. Exiting.")
    sys.exit(1)

# Convert longitudes from [-180, 180] to [0, 360] for consistency with GFS grid
df_data["longitude"] = df_data["longitude"] % 360

# Extract the base time from the KML metadata (IssueTime) for update checks.
base_time = global_metadata.get("IssueTime", datetime.now(UTC))
logging.info(f"Base time for this ingest run (from KML metadata): {base_time}")

# --- Check for Updates (Skip if no new data) ---
if save_type == "S3":
    # Check if the file exists and load it
    if s3.exists(forecast_path + "/" + ingest_version + "/DWD_MOSMIX.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/DWD_MOSMIX.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to DWD_MOSMIX, ending")
            sys.exit()

else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/DWD_MOSMIX.time.pickle"):
        # Open the file in binary mode
        with open(
            forecast_path + "/" + ingest_version + "/DWD_MOSMIX.time.pickle", "rb"
        ) as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print("No Update to DWD_MOSMIX, ending")
            sys.exit()


# --- Step 2: Process Pandas DataFrame ---
logging.info("\n--- Converting DataFrame to xarray Dataset (point-based) ---")


def process_and_interpolate_df(df_input):
    """
    Wrapper to process DWD DataFrame, optionally temporally interpolate,
    and then spatially interpolate to grid.
    """
    # 1. Process units and variables
    df_processed = process_dwd_df(df_input)

    # 2. Interpolate to Grid
    gridded_ds = interpolate_dwd_to_grid_knearest_dask(
        df_processed,
        var_cols=zarr_vars,
        radius_km=radius_km,
        time_col="time",
        lat_col="latitude",
        lon_col="longitude",
        station_col="station_id",
        log="none",  # Reduce noise
    )

    return gridded_ds


# Process the latest forecast
gridded_dwd_ds_forecast = process_and_interpolate_df(df_data)

# Chunk
gridded_dwd_ds_forecast = gridded_dwd_ds_forecast.chunk(
    chunks={"time": 240, "lat": CHUNK_SIZES["DWD"], "lon": CHUNK_SIZES["DWD"]}
)

# --- Step 3: Historic Data ---
# Logic to download and process previous runs
history_period = HISTORY_PERIODS["DWD_MOSMIX"]
historic_datasets = []

# Loop back through history
for i in range(history_period, 0, -1):
    # Calculate target time
    target_time = base_time - pd.Timedelta(hours=i)

    time_str = target_time.strftime("%Y%m%d%H")

    # Path for cached Zarr file
    hist_zarr_filename = f"DWD_Hist_{time_str}.zarr"

    # Check if cached file exists
    if save_type == "S3":
        hist_zarr_path = f"{historic_path}/{hist_zarr_filename}"
        cached_exists = s3.exists(hist_zarr_path.replace(".zarr", ".done"))
    else:
        hist_zarr_path = os.path.join(historic_path, hist_zarr_filename)
        cached_exists = os.path.exists(hist_zarr_path.replace(".zarr", ".done"))

    if cached_exists:
        logging.info(f"Loading cached historic run: {time_str}")
        try:
            if save_type == "S3":
                store = zarr.storage.FsspecStore.from_url(
                    hist_zarr_path,
                    storage_options={
                        "key": aws_access_key_id,
                        "secret": aws_secret_access_key,
                    },
                )
            else:
                store = zarr.storage.LocalStore(hist_zarr_path)

            ds_hist = xr.open_dataset(store, engine="zarr", chunks="auto")
            historic_datasets.append(ds_hist)
            continue
        except Exception as e:
            logging.warning(f"Error loading cached {hist_zarr_path}: {e}")
            # Fallback to re-download if load fails? Or just skip?
            # Let's try to re-process.
            pass

    # If not cached, download and process
    logging.info(f"Processing historic run: {time_str}")

    # Construct URL for historical file
    # Pattern: MOSMIX_S_2025121018_240.kmz
    hist_filename = f"MOSMIX_S_{time_str}_240.kmz"
    hist_url = f"https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/{hist_filename}"
    local_hist_path = os.path.join(tmp_dir, hist_filename)

    # Check if download is needed
    if not os.path.exists(local_hist_path):
        if not download_mosmix_file(hist_url, local_hist_path):
            continue

    # Parse the file
    try:
        df_hist, _ = parse_mosmix_kml(local_hist_path)
        if df_hist.empty:
            continue

        # Standardize longitudes
        df_hist["longitude"] = df_hist["longitude"] % 360

        # Files are offset by 1 hour (so T21 forecast is in the T20 file)
        target_time = target_time.tz_localize(None)
        target_time = target_time + pd.Timedelta(hours=1)

        # Filter dataframe to this specific time
        df_hist_filtered = df_hist[df_hist["time"] == target_time].copy()

        if df_hist_filtered.empty:
            logging.warning(
                f"No data found for target time {target_time} in {hist_filename}"
            )
            continue

        # Process this single timestep (no temporal interp needed for single step)
        ds_hist = process_and_interpolate_df(df_hist_filtered)

        # Cache the processed dataset
        logging.info(f"Caching processed run to {hist_zarr_path}")
        if save_type == "S3":
            store = zarr.storage.FsspecStore.from_url(
                hist_zarr_path,
                storage_options={
                    "key": aws_access_key_id,
                    "secret": aws_secret_access_key,
                },
            )
        else:
            store = zarr.storage.LocalStore(hist_zarr_path)

        # Rechunk to process chunks
        ds_hist_chunk = ds_hist.chunk(
            chunks={"time": 1, "lat": CHUNK_SIZES["DWD"], "lon": CHUNK_SIZES["DWD"]}
        )

        ds_hist_chunk.to_zarr(store, mode="w", consolidated=False)

        # Create .done file
        if save_type == "S3":
            s3.touch(hist_zarr_path.replace(".zarr", ".done"))
        else:
            with open(hist_zarr_path.replace(".zarr", ".done"), "w") as f:
                f.write("Done")

        # Re-open lazily to use in current run (avoids memory hogging)
        ds_hist_lazy = xr.open_dataset(store, engine="zarr", chunks="auto")
        historic_datasets.append(ds_hist_lazy)

        # Cleanup
        os.remove(local_hist_path)

    except Exception as e:
        logging.warning(f"Error processing {hist_filename}: {e}")

# Combine datasets
if historic_datasets:
    logging.info(f"Combining {len(historic_datasets)} historic datasets with forecast")
    # Concatenate historic datasets along time dimension
    ds_history = xr.concat(historic_datasets, dim="time")

    # Concatenate history and forecast
    gridded_dwd_ds = xr.concat([ds_history, gridded_dwd_ds_forecast], dim="time")

    # Sort by time just in case
    gridded_dwd_ds = gridded_dwd_ds.sortby("time")
else:
    gridded_dwd_ds = gridded_dwd_ds_forecast

# --- Step 4.5: Build Grid-to-Stations Mapping ---
logging.info("\n--- Building grid-to-stations mapping ---")
grid_to_stations_map = build_grid_to_stations_map(
    df_data,  # Use original df_data which has station_name
    radius_km=radius_km,
    lat_col="latitude",
    lon_col="longitude",
    station_col="station_id",
    station_name_col="station_name",
    log="tqdm",
)

# Save the station map to a pickle file
station_map_file = os.path.join(forecast_process_dir, "DWD_MOSMIX_stations.pickle")
with open(station_map_file, "wb") as f:
    pickle.dump(grid_to_stations_map, f)
logging.info(f"Station map saved to {station_map_file}")

# --- Step 5: Temporal Interpolation and Saving ---
# Rechunk the data to be more manageable for processing
gridded_dwd_ds_chunk = gridded_dwd_ds.chunk(
    {
        "time": len(gridded_dwd_ds.time),
        "lat": CHUNK_SIZES["DWD"],
        "lon": CHUNK_SIZES["DWD"],
    }
)

# Strip existing encoding so Zarr writes the new chunks we specified
for var in gridded_dwd_ds_chunk.variables:
    if "chunks" in gridded_dwd_ds_chunk[var].encoding:
        del gridded_dwd_ds_chunk[var].encoding["chunks"]
    if "preferred_chunks" in gridded_dwd_ds_chunk[var].encoding:
        del gridded_dwd_ds_chunk[var].encoding["preferred_chunks"]

# Write chunked to disk as interim step
with ProgressBar():
    gridded_dwd_ds_chunk.to_zarr(forecast_process_path + "_chunk.zarr", mode="w")

# Get the range from combined dataset
min_time = gridded_dwd_ds.time.min().values
max_time = gridded_dwd_ds.time.max().values

# Delete from memory
del gridded_dwd_ds_chunk
del gridded_dwd_ds
del historic_datasets
try:
    del ds_history
except NameError:
    pass

# Read back in as xarray
ds_chunk_disk = xr.open_zarr(forecast_process_path + "_chunk.zarr", chunks="auto")

# Interpolate max 3 hour gaps
# Define variables that shouldn't be smoothed linearly (e.g. codes, types)
nearest_neighbor_vars = ["PTYPE_surface"]

logging.info("Interpolating gaps and extrapolating edges...")

# Apply efficient interpolation
ds_chunk_disk_interp = interpolate_temporal_gaps_efficiently(
    ds_chunk_disk, nearest_vars=nearest_neighbor_vars, max_gap_hours=3, time_dim="time"
)

# 1. Prepare Target Time Grid (Hourly)

# Create continuous hourly range
hourly_times = pd.date_range(start=min_time, end=max_time, freq="1h")

# Convert timestamps to Unix seconds for interpolation function
unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")

stacked_times = ds_chunk_disk_interp.time.values
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (hourly_times.values - unix_epoch) / one_second

ds_rename = ds_chunk_disk_interp.rename({"time": "stacked_time"})

# Add a 3D time array
time3d = (
    ((ds_rename["stacked_time"] - unix_epoch) / np.timedelta64(1, "s"))
    .astype("float32")  # 1D ('time',)
    .expand_dims(
        lat=ds_rename.lat,  # add ('latitude',)
        lon=ds_rename.lon,
    )  # add ('longitude',)
    .transpose("stacked_time", "lat", "lon")  # order dims
)


# Add the time array to the dataset
ds_rename["time"] = time3d

# Set time to nan where TMP_2maboveground is nan
ds_rename["time"] = ds_rename["time"].where(
    ~np.isnan(ds_rename["TMP_2maboveground"]), other=np.nan
)

# Set the order correctly
vars_in = [
    v for v in zarr_vars if v in ds_rename.data_vars
]  # keep only those that exist
ds_stack = ds_rename[vars_in].to_array(dim="var", name="var")

# Rechunk the data to be more manageable for processing
ds_chunk = ds_stack.chunk(
    {
        "var": -1,
        "stacked_time": len(stacked_timesUnix),
        "lat": FINAL_CHUNK_SIZES["DWD"],
        "lon": FINAL_CHUNK_SIZES["DWD"],
    }
)


# Create a zarr backed dask array
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/DWD_MOSMIX.zarr.zip", mode="a"
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/DWD_MOSMIX.zarr")


# 1. Interpolate the stacked array to be hourly along the time axis
# 2. Pad to chunk size
# 3. Create the zarr array
# 4. Rechunk it to match the final array
# 5. Write it out to the zarr array

with ProgressBar():
    # 4. Rechunk it to match the final array
    # 5. Write it out to the zarr array
    ds_chunk.to_zarr(zarr_store, mode="w")


if save_type == "S3":
    zarr_store.close()


### Test read back
# localTest = zarr.open(
#     forecast_process_dir + "/DWD_MOSMIX.zarr",
#     mode="r",
# )
# --- Step 6: Upload to S3 or Move to Final Location ---

# %% Upload to S3
if save_type == "S3":
    # Upload to S3
    s3.put_file(
        forecast_process_dir + "/DWD_MOSMIX.zarr.zip",
        forecast_path + "/" + ingest_version + "/DWD_MOSMIX.zarr.zip",
    )

    # Upload station map
    s3.put_file(
        forecast_process_dir + "/DWD_MOSMIX_stations.pickle",
        forecast_path + "/" + ingest_version + "/DWD_MOSMIX_stations.pickle",
    )

    # Write most recent forecast time
    with open(forecast_process_dir + "/DWD_MOSMIX.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    s3.put_file(
        forecast_process_dir + "/DWD_MOSMIX.time.pickle",
        forecast_path + "/" + ingest_version + "/DWD_MOSMIX.time.pickle",
    )


else:
    # Write most recent forecast time
    with open(forecast_process_dir + "/DWD_MOSMIX.time.pickle", "wb") as file:
        # Serialize and write the variable to the file
        pickle.dump(base_time, file)

    shutil.move(
        forecast_process_dir + "/DWD_MOSMIX.time.pickle",
        forecast_path + "/" + ingest_version + "/DWD_MOSMIX.time.pickle",
    )

    # Copy station map to final location
    shutil.copy(
        forecast_process_dir + "/DWD_MOSMIX_stations.pickle",
        forecast_path + "/" + ingest_version + "/DWD_MOSMIX_stations.pickle",
    )

    # Copy the zarr file to the final location
    shutil.copytree(
        forecast_process_dir + "/DWD_MOSMIX.zarr",
        forecast_path + "/" + ingest_version + "/DWD_MOSMIX.zarr",
        dirs_exist_ok=True,
    )


# --- Final Cleanup ---
_clear_directory(forecast_process_dir)
_clear_directory(tmp_dir)

end_time = time.time()  # End timer
logging.info(f"\nTotal script execution time: {end_time - start_time:.2f} seconds")
