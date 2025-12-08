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


# NOTE 2: At some point it would be better to include short term (48 hour) historic data here as well
# This would keep the sources more consistent for the API users, instead of mixing MOSMIX-S forecast and
# something else for recent past. Since it's an hourly model, it would be merging the first step from the last 48 runs


# NOTE 3: Multiple stations can map to the same grid cell (e.g., OSAKA station ID 47772 and
# OSAKA AIRPORT station ID 47771 are ~10km apart). The interpolation uses inverse distance
# weighting (IDW) to properly average values from all contributing stations rather than
# taking just the last value. This prevents temperature jumps and other data artifacts
# in the API output.

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

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import CHUNK_SIZES, DWD_RADIUS, FINAL_CHUNK_SIZES

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


def fill_station_time_series(
    df: pd.DataFrame,
    var_cols,
    time_col: str = "time",
    station_col: str = "station_id",
    max_gap: int | None = None,
    fill_ends: bool = False,
    log="print",  # "print" | "tqdm" | None
    print_every=50,  # applies only if log="print"
):
    """
    Interpolates missing values separately for each station over time.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing station data.
    - var_cols (list[str]): A list of column names to interpolate.
    - time_col (str): The name of the time column.
    - station_col (str): The name of the station identifier column.
    - max_gap (int | None): The maximum number of consecutive NaNs to fill.
    - fill_ends (bool): If True, fill NaNs at the beginning and end of series.
    - log (str): The logging mode ('print', 'tqdm', or None).
    - print_every (int): Frequency of progress printing if log is 'print'.

    Progress Monitoring:
      log="tqdm"  → progress bar over stations
      log="print" → prints every N stations
      log=None    → silent
    """
    if isinstance(var_cols, str):
        var_cols = [var_cols]

    # Sort input for deterministic behavior
    df_sorted = df.sort_values([station_col, time_col]).copy()

    # Unique station groups
    stations = df_sorted[station_col].unique()
    n_stations = len(stations)

    if log == "tqdm":
        station_iter = tqdm(stations, desc="Interpolating stations", unit="stations")
    else:
        station_iter = stations

    def _interp_one_station(station_id, idx):
        df_s = df_sorted.loc[df_sorted[station_col] == station_id]

        for col in var_cols:
            s = df_s[col]

            # skip if nothing to fill
            if s.notna().sum() < 2:
                continue

            if np.issubdtype(df_s[time_col].dtype, np.datetime64):
                s2 = s.copy()
                s2.index = df_s[time_col]
                s2 = s2.interpolate(
                    method="time", limit=max_gap, limit_direction="both"
                )
            else:
                s2 = s.interpolate(
                    method="linear", limit=max_gap, limit_direction="both"
                )

            if fill_ends:
                s2 = s2.ffill().bfill()

            df_sorted.loc[df_s.index, col] = s2.to_numpy()

        if log == "print" and idx % print_every == 0:
            print(f"[{idx}/{n_stations}] Interpolated station {station_id}")

    # Loop stations with progress feedback
    for i, st in enumerate(station_iter):
        _interp_one_station(st, i)

    if log == "print":
        print("Interpolation complete.")

    return df_sorted


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
    Vectorized k-nearest grid interpolation with dense, Dask-backed output.

    Each dataframe row is assigned to up to k_max nearest grid cells
    within radius_km. Output variables are (time, lat, lon) Dask arrays.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: station_col, time_col, lat_col, lon_col, var_cols.
        Lat/lon in degrees. Lon assumed in [0, 360) for the GFS grid.
    var_cols : str or list of str
        Variables to interpolate.
    radius_km : float
        Max radius in km for grid cell to be influenced.
    k_max : int
        Maximum number of nearest grid cells per station.
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
        Dask-backed 3D array.
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

    # k_max nearest neighbours per station
    dist_rad, inds_flat = tree.query(pts_rad, k=k_max)  # (n_stations, k_max)

    radius_rad = radius_km / 6371.0
    mask = dist_rad <= radius_rad
    inds_flat = np.where(mask, inds_flat, -1)  # -1 marks "no neighbour within radius"

    station_indexer = {sid: i for i, sid in enumerate(station_ids)}

    # -------------------------------------------------
    # 2) Time indexing & sorting
    # -------------------------------------------------
    _log("Indexing time dimension…")

    df_sorted = df.sort_values(time_col)
    unique_times = df_sorted[time_col].drop_duplicates().to_numpy()
    nt = len(unique_times)

    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    t_idx = df_sorted[time_col].map(time_to_idx).to_numpy()  # shape (n_rows,)

    stn_full = df_sorted[station_col].to_numpy()
    stn_idx_full = np.array([station_indexer[s] for s in stn_full], dtype=int)

    # -------------------------------------------------
    # 3) Expand row → neighbour grid indices (vectorized)
    # -------------------------------------------------
    _log("Linking rows to neighbour grid cells…")

    # Each row maps to k_max neighbours for its station
    neigh_flat = inds_flat[stn_idx_full]  # (n_rows, k_max)
    neigh_time = np.repeat(t_idx[:, None], k_max, 1)  # (n_rows, k_max)

    # Get distances for each row's neighbours (for inverse distance weighting)
    neigh_dist = dist_rad[stn_idx_full]  # (n_rows, k_max)

    # Flatten and filter invalid neighbours
    flat_flat = neigh_flat.ravel()
    flat_time = neigh_time.ravel()
    flat_dist = neigh_dist.ravel()
    valid = flat_flat >= 0
    flat_flat = flat_flat[valid]
    flat_time = flat_time[valid]
    flat_dist = flat_dist[valid]

    # Grid indices (y, x)
    iy, ix = np.unravel_index(flat_flat, (ny, nx))

    # -------------------------------------------------
    # 4) Build dense NumPy arrays and wrap them in Dask
    # -------------------------------------------------
    _log(f"Building dense Dask arrays for {len(var_cols)} variable(s)…")

    data_vars = {}
    var_iter = tqdm(var_cols, desc="Variables") if log == "tqdm" else var_cols

    for col in var_iter:
        vals = df_sorted[col].to_numpy().astype(dtype)

        # Replicate each row's value across its k_max neighbours
        vals_rep = np.repeat(vals[:, None], k_max, axis=1).ravel()
        vals_rep = vals_rep[valid]

        # Drop NaNs
        val_mask = ~np.isnan(vals_rep)
        t_final = flat_time[val_mask]
        y_final = iy[val_mask]
        x_final = ix[val_mask]
        v_final = vals_rep[val_mask]
        d_final = flat_dist[val_mask]

        # Allocate dense arrays for values, weights, and weight sums
        arr = np.full((nt, ny, nx), 0.0, dtype=dtype)
        weight_sum = np.full((nt, ny, nx), 0.0, dtype=dtype)

        # Use inverse distance weighting to handle multiple stations per grid cell
        # Add small epsilon to avoid division by zero for exact matches
        epsilon = 1e-10
        weights = 1.0 / (d_final + epsilon)

        # Accumulate weighted values and weights using np.add.at for proper handling of duplicates
        np.add.at(arr, (t_final, y_final, x_final), v_final * weights)
        np.add.at(weight_sum, (t_final, y_final, x_final), weights)

        # Compute weighted average where we have data
        valid_cells = weight_sum > 0
        arr[valid_cells] = arr[valid_cells] / weight_sum[valid_cells]
        arr[~valid_cells] = np.nan

        # Wrap in Dask
        chunk_t = min(time_chunk, nt) if nt > 0 else 1
        darr = da.from_array(arr, chunks=(chunk_t, ny, nx))

        data_vars[col] = xr.DataArray(
            darr,
            dims=("time", "lat", "lon"),
            coords={
                "time": unique_times,
                "lat": gfs_lats,
                "lon": gfs_lons,
            },
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
    grid_to_stations = {}

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
        indices = tree.query_radius(stn_rad, r=radius_rad)[0]

        # Convert flat indices to (y, x) grid indices
        for flat_idx in indices:
            y_idx, x_idx = np.unravel_index(flat_idx, (ny, nx))

            if (y_idx, x_idx) not in grid_to_stations:
                grid_to_stations[(y_idx, x_idx)] = []

            # Add station info to this grid cell
            # Convert longitude back to [-180, 180] for API output
            output_lon = ((station_lon + 180) % 360) - 180

            grid_to_stations[(y_idx, x_idx)].append(
                {
                    "id": station_id,
                    "name": station_name,
                    "lat": round(station_lat, 4),
                    "lon": round(output_lon, 4),
                }
            )

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
forecast_process_dir = os.getenv("forecast_process_dir", "/mnt/nvme/data/DWD")
# Temporary directory for downloaded files.
tmp_dir = os.getenv("tmp_dir", os.path.join(forecast_process_dir, "Downloads"))
# Final destination directory for processed Zarr files.
forecast_path = os.getenv("forecast_path", "/mnt/nvme/data/Prod/DWD")

# Defines where the final Zarr file should be saved: "Download" (local) or "S3" (AWS S3).
save_type = os.getenv("save_type", "Download")
# AWS Credentials for S3 operations (should be set as environment variables).
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

# Initialize S3 filesystem object if saving to S3.
s3 = None
if save_type == "S3":
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# zarr import for S3 stores (imports are at top)


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

start_time = time.time()  # Start timer for script execution

# --- Step 1: Download and Parse DWD MOSMIX-S KML/KMZ Data ---
logging.info(f"\n--- Attempting to download DWD MOSMIX data from: {dwd_mosmix_url} ---")
try:
    response = requests.get(dwd_mosmix_url, stream=True)
    response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    with open(os.path.join(tmp_dir, downloaded_kmz_file), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logging.info(
        f"File downloaded successfully to: {os.path.join(tmp_dir, downloaded_kmz_file)}"
    )
except Exception as e:
    logging.critical(f"Error downloading DWD data: {e}. Exiting.")
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

df_process = process_dwd_df(df_data)

# Fill missing data per station using time interpolation
df_filled = fill_station_time_series(
    df_process,
    var_cols=api_target_numerical_variables,  # whatever variables you will grid
    time_col="time",
    station_col="station_id",
    max_gap=None,  # or an integer if you want to limit gap length
    fill_ends=False,  # or True if you want to fill edges too
    log="tqdm",
)


# --- Step 4: Interpolate DWD Point Data to the GFS Grid ---
logging.info("\n--- Interpolating DWD point data to GFS grid ---")
gridded_dwd_ds = interpolate_dwd_to_grid_knearest_dask(
    df_filled,
    api_target_numerical_variables,
    50,
    time_col="time",
    lat_col="latitude",
    lon_col="longitude",
    station_col="station_id",
    log="tqdm",
)

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

# Reformat the dataset to have a stacked time dimension and add a 3D time variable
# Get the actual stacked times from the concatenated dataset
unix_epoch = np.datetime64(0, "s")
one_second = np.timedelta64(1, "s")

stacked_times = gridded_dwd_ds.time.values
stacked_timesUnix = (stacked_times - unix_epoch) / one_second

# Rename time dimension to match later processing
ds_rename = gridded_dwd_ds.rename({"time": "stacked_time"})

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

# Set the order correctly
vars_in = [
    v for v in zarr_vars if v in ds_rename.data_vars
]  # keep only those that exist
ds_stack = ds_rename[vars_in].to_array(dim="var", name="var")

# Rechunk the data to be more manageable for processing
ds_chunk = ds_stack.chunk(
    {
        "var": 1,
        "stacked_time": len(stacked_timesUnix),
        "lat": CHUNK_SIZES["DWD"],
        "lon": CHUNK_SIZES["DWD"],
    }
)

# Interim zarr save of the stacked array. Not necessary for local, but speeds things up on S3
with ProgressBar():
    ds_chunk.to_zarr(forecast_process_dir + "/DWD_MOSMIX_stack.zarr", mode="w")

# Read in stacked 4D array back in
daskVarArrayStackDisk = da.from_zarr(
    forecast_process_dir + "/DWD_MOSMIX_stack.zarr",
    component="__xarray_dataarray_variable__",
)


# --- Step 5: Save Gridded DWD Dataset to Zarr ---

if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        forecast_process_dir + "/DWD_MOSMIX.zarr.zip", mode="a"
    )
else:
    zarr_store = zarr.storage.LocalStore(forecast_process_dir + "/DWD_MOSMIX.zarr")


# Create the zarr array
zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        len(zarr_vars),
        len(stacked_timesUnix),
        daskVarArrayStackDisk.shape[2],
        daskVarArrayStackDisk.shape[3],
    ),
    chunks=(
        len(zarr_vars),
        len(stacked_timesUnix),
        FINAL_CHUNK_SIZES["DWD"],
        FINAL_CHUNK_SIZES["DWD"],
    ),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
)

with ProgressBar():
    daskVarArrayStackDisk.round(5).rechunk(
        (
            len(zarr_vars),
            len(stacked_timesUnix),
            FINAL_CHUNK_SIZES["DWD"],
            FINAL_CHUNK_SIZES["DWD"],
        )
    ).to_zarr(zarr_array, overwrite=True, compute=True)


if save_type == "S3":
    zarr_store.close()


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
