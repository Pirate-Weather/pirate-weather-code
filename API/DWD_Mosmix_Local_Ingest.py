"""DWD MOSMIX-S station forecast ingest.

Downloads the latest DWD MOSMIX-S KMZ (point forecasts for ~5400 stations
worldwide, issued hourly, hourly steps to 240 h), parses the station forecasts,
standardizes them to the API's GRIB-style variable names, fills short per-station
gaps and grids each station onto the nearest cells of the global 0.25 degree GFS
grid. Publishes ``DWD_MOSMIX.zarr``, the grid-cell-to-station map
``DWD_MOSMIX_stations.pickle`` and ``DWD_MOSMIX.time.pickle`` (the KML IssueTime).
The store starts at the first MOSMIX-S time step (IssueTime + 1 h).

Why does it need ~20 GB of RAM to process a 40 MB KMZ? Zarr does not support
sparse arrays and the gridded MOSMIX-S data is mostly empty; see
https://github.com/zarr-developers/zarr-specs/issues/245.

Multiple stations can map to the same grid cell (e.g. OSAKA 47772 and OSAKA
AIRPORT 47771 are ~10 km apart); the closest station wins.

Alexander Rey, June 2025
"""

import io
import logging
import os
import pickle
import time
import zipfile

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
from API.ingest_utils import (
    CHUNK_SIZES,
    DWD_RADIUS,
    FINAL_CHUNK_SIZES,
    clear_directory,
    interpolate_temporal_gaps_efficiently,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
)

logger = logging.getLogger(__name__)

# %% Model constants
# DWD publishes the newest MOSMIX-S run under a fixed "LATEST" name
DWD_MOSMIX_URL = "https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz"
DOWNLOADED_KMZ_FILE = "MOSMIX_S_LATEST_240.kmz"

# KML and DWD namespaces used to locate elements within the KML file
KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"
DWD_NAMESPACE = (
    "{https://opendata.dwd.de/weather/lib/pointforecast_dwd_extension_V1_0.xsd}"
)

# Stations are gridded onto the global 0.25 degree GFS grid so the API can look
# them up with the same indices as GFS. Longitudes are in [0, 360).
GFS_GRID_LATS = np.arange(-90, 90, 0.25)
GFS_GRID_LONS = np.arange(0, 360, 0.25)
EARTH_RADIUS_KM = 6371.0
# Distance to interpolate stations to grid (km)
radius_km = DWD_RADIUS

# Rad1h is an hourly sum in kJ/m^2; dividing by 3.6 gives the mean flux in W/m^2
# ((kJ/m^2 * 1000 J/kJ) / 3600 s).
KJ_PER_M2_PER_HOUR_TO_W_PER_M2 = 3.6
# Longest run of missing hourly rows filled within one station's series before
# gridding; longer gaps null the series so the API falls back to GFS/ECMWF.
STATION_MAX_GAP_HOURS = 6
# Longest run of missing time steps filled per grid cell after gridding
GRID_MAX_GAP_HOURS = 3
# Precipitation type (WW codes) is categorical, so its gaps take the nearest value
nearest_neighbor_vars = ["PTYPE_surface"]

# Variables in the published store; the order is the API's DWD_MOSMIX index map
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

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# Base directory for all processing and output files
forecast_process_dir = os.getenv("forecast_process_dir", "/mnt/nvme/data/DWD_MOSMIX")
forecast_process_path = forecast_process_dir + "/DWD_Process"
# Temporary directory for downloaded files
tmp_dir = os.getenv("tmp_dir", os.path.join(forecast_process_dir, "Downloads"))
# Final destination directory for processed Zarr files
forecast_path = os.getenv("forecast_path", "/mnt/nvme/data/Prod/DWD")

# "Download" (local) or "S3"
save_type = os.getenv("save_type", "Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

DWD_DOWNLOAD_RETRIES = positive_int_env("dwd_download_retries", 3)
DWD_RETRY_SLEEP_SECONDS = positive_int_env("dwd_retry_sleep_seconds", 30)
# (connect, read) timeouts for the KMZ request
DWD_REQUEST_TIMEOUT_SECONDS = (10, 300)


def build_gfs_grid_tree() -> BallTree:
    """Build a haversine BallTree over the cell centres of the 0.25 degree GFS grid."""
    lon_grid, lat_grid = np.meshgrid(GFS_GRID_LONS, GFS_GRID_LATS)
    grid_coords_deg = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return BallTree(np.radians(grid_coords_deg), metric="haversine")


def download_mosmix_file(url: str, local_path: str) -> None:
    """Download the MOSMIX KMZ with request timeouts and bounded retries.

    Raises the last error after ``DWD_DOWNLOAD_RETRIES`` attempts; a missing
    KMZ means there is nothing valid to publish.
    """
    for attempt in range(1, DWD_DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(
                url, stream=True, timeout=DWD_REQUEST_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.writelines(response.iter_content(chunk_size=8192))
            logger.info("DWD MOSMIX KMZ downloaded to %s", local_path)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == DWD_DOWNLOAD_RETRIES:
                raise RuntimeError(
                    f"DWD MOSMIX download from {url} failed after "
                    f"{DWD_DOWNLOAD_RETRIES} attempts"
                ) from exc
            sleep_s = DWD_RETRY_SLEEP_SECONDS * attempt
            logger.warning(
                "DWD MOSMIX download attempt %d/%d failed (%s). Retrying in %ss",
                attempt,
                DWD_DOWNLOAD_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)


def parse_mosmix_kml(kml_filepath: str):
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
                raise ValueError(f"No KML file found inside {kml_filepath}.")
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
        logger.error("Error: No global forecast time steps found in KML Document.")
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
    logger.info("Found %d global forecast time steps.", len(global_forecast_times))

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
                logger.warning(
                    "Could not parse coordinates for station ID %s: %s. Setting to NaN.",
                    station_id,
                    coordinates_elem.text,
                )
                lon, lat, alt = np.nan, np.nan, np.nan

        extended_data_pm = placemark.find(f"{KML_NAMESPACE}ExtendedData")
        if extended_data_pm is None:
            logger.warning(
                "No ExtendedData found for station ID %s. Skipping.", station_id
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
                    logger.warning(
                        "Mismatch in number of values for '%s' for station %s. "
                        "Skipping parameter.",
                        element_name,
                        station_id,
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

    # Downward Short-Wave Radiation Flux (DSWRF_surface) from the hourly Rad1h sum
    df["DSWRF_surface"] = df["Rad1h"] / KJ_PER_M2_PER_HOUR_TO_W_PER_M2

    return df


def fill_station_gaps(
    df,
    station_col="station_id",
    time_col="time",
    max_gap_hours=STATION_MAX_GAP_HOURS,
):
    """
    Fill NaN gaps in each station's time series before spatial gridding.

    DWD MOSMIX station data has two common sources of NaN gaps:

    * ``APCP_surface`` (RR1c): DWD MOSMIX omits this parameter for some hours,
      leaving gaps of up to 5–6 consecutive NaNs.  These are filled by **linear
      interpolation** — between two dry periods (0 → 0) this naturally gives 0;
      between non-zero values (e.g. 0.5 → 0.1) it provides a physically reasonable
      estimate rather than incorrectly reporting no precipitation.  After
      interpolation any remaining NaNs (gaps longer than ``max_gap_hours``) are
      filled with **0**, since an isolated missing report surrounded by no-rain
      hours almost certainly means no precipitation was forecast.
    * ``PTYPE_surface`` (ww): categorical precipitation-type code filled by
      forward- then backward-propagation so that the nearest valid code is used.
    * All other continuous variables (temperature, humidity, wind U/V, etc.):
            gaps of up to ``max_gap_hours`` consecutive NaN *timesteps* are filled by
            **linear interpolation** within each station's own time series. If any
            longer gap is present for a station/variable pair, that entire variable
            series for that station is replaced with NaN so downstream merge logic can
            fall back to GFS/ECMWF instead of mixing observed values with an unreliable
            partially filled segment.

    .. note::
        ``max_gap_hours`` counts consecutive NaN *rows*, not wall-clock hours.
        DWD MOSMIX-S produces one row per hour for the first 240 forecast hours,
        so for that dataset the two are equivalent.

    Not all stations report the full suite of variables — solar radiation is
    only available for European stations, and wind gusts (FX1) are absent for
    some networks.  Those absent columns are entirely NaN for the affected
    stations and represent unavailable data, not gaps to fill.  The function
    skips any column that is entirely NaN for a given station, and skips the
    station entirely if all NaN columns fall into that category.

    This function is called on the processed DataFrame (after ``process_dwd_df``)
    but *before* spatial gridding, so that ``interpolate_dwd_to_grid_knearest_dask``
    always receives hole-free station series and the nearest-station-wins strategy
    never leaves a grid cell NaN due to a single missing report.

    Parameters
    ----------
    df : pd.DataFrame
        Processed station DataFrame with one row per (station, time) pair.
    station_col : str
        Column identifying each station.
    time_col : str
        Column containing the forecast timestamp (used for ordering only).
    max_gap_hours : int
        Maximum number of consecutive NaN timesteps to fill by interpolation
        for continuous variables. If a longer gap exists, the entire station /
        variable series is replaced with NaN (except APCP_surface and
        PTYPE_surface, which retain their special-case handling).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN gaps filled.
    """
    # Precipitation column: interpolate first, then zero-fill any remaining NaNs.
    # Linear interpolation handles both dry-period gaps (0→0 stays 0) and
    # active-precipitation gaps (e.g. 0.5→0.1 gives a smooth estimate).
    PRECIP_VARS = {"APCP_surface"}

    # Categorical columns: propagate the nearest valid code rather than
    # interpolating, because averaging integer codes is not physically meaningful.
    CATEGORICAL_VARS = {"PTYPE_surface"}

    # Non-data columns: skip entirely.
    SKIP_COLS = {station_col, time_col, "latitude", "longitude", "altitude"}

    def _has_oversized_nan_gap(series):
        nan_mask = series.isna().to_numpy()
        if not nan_mask.any():
            return False

        padded = np.pad(nan_mask.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        gap_starts = np.flatnonzero(transitions == 1)
        gap_ends = np.flatnonzero(transitions == -1)
        return np.any((gap_ends - gap_starts) > max_gap_hours)

    df = df.copy()

    # Pre-compute the subset of columns that need gap-filling so that the
    # per-station NaN check below is as cheap as possible.
    fill_cols = [
        col
        for col in df.columns
        if col not in SKIP_COLS
        and (col in CATEGORICAL_VARS or pd.api.types.is_float_dtype(df[col]))
    ]

    # Pre-compute a boolean mask: which rows have at least one NaN in a fill column.
    # Used to quickly skip stations that are already complete.
    has_any_nan = df[fill_cols].isna().any(axis=1)

    for grp_idx in df.groupby(station_col, sort=False).groups.values():
        # Fast path: skip the station entirely if it has no NaN in any fill column.
        if not has_any_nan.loc[grp_idx].any():
            continue

        grp = df.loc[grp_idx]
        sorted_idx = grp.index

        # Second fast path: only process columns that have a genuine mix of NaN
        # and valid values for this station.  Columns where every value is NaN
        # (e.g. FX1 / solar radiation for non-European stations) represent
        # unavailable data — they should be left as NaN, not filled.
        # Compute NaN counts for all fill columns in one vectorised pass instead
        # of calling .isna() column-by-column (≈2.5× faster per station).
        nan_counts = grp[fill_cols].isna().sum()
        cols_to_fill = nan_counts[
            (nan_counts > 0) & (nan_counts < len(grp))
        ].index.tolist()
        if not cols_to_fill:
            continue  # all NaN columns are entirely absent — nothing to fill

        for col in cols_to_fill:
            s = grp[col]

            if col in PRECIP_VARS:
                # Interpolate short gaps; zero-fill any remaining NaNs (isolated
                # missing reports in an otherwise dry period).
                filled = s.interpolate(
                    method="linear",
                    limit=max_gap_hours,
                    limit_direction="both",
                ).fillna(0.0)
                df.loc[sorted_idx, col] = filled.values
            elif col in CATEGORICAL_VARS:
                # Forward-fill then backward-fill within the station series.
                # limit= counts consecutive NaN rows; assumes uniform hourly spacing.
                df.loc[sorted_idx, col] = (
                    s.ffill(limit=max_gap_hours).bfill(limit=max_gap_hours).values
                )
            else:
                if _has_oversized_nan_gap(s):
                    df.loc[sorted_idx, col] = np.nan
                else:
                    # Linear interpolation capped at max_gap_hours consecutive NaN rows.
                    # limit_direction="both" also fills NaNs at the start/end of the series.
                    df.loc[sorted_idx, col] = s.interpolate(
                        method="linear",
                        limit=max_gap_hours,
                        limit_direction="both",
                    ).values

    return df


def interpolate_dwd_to_grid_knearest_dask(
    df,
    var_cols,
    radius_km,
    grid_tree,
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

    NaN gaps within each station's time series should be filled before calling
    this function (see ``fill_station_gaps``), so that the nearest station always
    has a valid value and no abrupt source switches occur.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: station_col, time_col, lat_col, lon_col, var_cols.
        Lat/lon in degrees. Lon assumed in [0, 360) for the GFS grid.
    var_cols : str or list of str
        Variables to interpolate.
    radius_km : float
        Max radius in km for a station to influence a grid cell.
    grid_tree : sklearn.neighbors.BallTree
        Haversine tree over the GFS grid cells, from ``build_gfs_grid_tree``.
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
            logger.info(msg)

    gfs_lats = GFS_GRID_LATS
    gfs_lons = GFS_GRID_LONS
    ny = gfs_lats.size
    nx = gfs_lons.size

    # -------------------------------------------------
    # 1) Station → k-nearest neighbours on grid
    # -------------------------------------------------
    _log("Querying k-nearest neighbours for stations…")

    tree = grid_tree

    station_meta = df.drop_duplicates(subset=station_col)[
        [station_col, lat_col, lon_col]
    ]
    station_ids = station_meta[station_col].to_numpy()
    pts_rad = np.radians(station_meta[[lat_col, lon_col]].to_numpy())

    # k_max nearest neighbours per station (in radians)
    dist_rad, inds_flat = tree.query(pts_rad, k=k_max)  # (n_stations, k_max)

    radius_rad = radius_km / EARTH_RADIUS_KM
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

    # Use station_id as a deterministic final tie-breaker when two stations are
    # exactly the same distance from the same grid cell at the same time.
    station_ids_full = df_sorted[station_col].astype(str).to_numpy()
    station_sort_key = pd.to_numeric(station_ids_full, errors="coerce")
    if pd.isna(station_sort_key).any():
        station_sort_key = station_ids_full
    station_sort_rep = np.repeat(station_sort_key[:, None], k_max, axis=1).ravel()[
        valid
    ]

    # -------------------------------------------------
    # 4) For each grid cell/time, find the nearest station (single pass)
    # -------------------------------------------------
    _log("Computing nearest-station per grid cell/time (argmin)…")

    # Sort candidates by cell index, then distance, then station_id.
    # lexsort uses the last key as the primary sort key.
    order = np.lexsort((station_sort_rep, flat_dist, linear_index))
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
    grid_tree,
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
    grid_tree : sklearn.neighbors.BallTree
        Haversine tree over the GFS grid cells, from ``build_gfs_grid_tree``.
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
            logger.info(msg)

    _log("Building grid-to-stations mapping...")

    ny = GFS_GRID_LATS.size
    nx = GFS_GRID_LONS.size
    tree = grid_tree

    # Get unique stations with metadata
    station_meta = df.drop_duplicates(subset=station_col)[
        [station_col, station_name_col, lat_col, lon_col]
    ]

    # For each station, find all grid cells within radius
    radius_rad = radius_km / EARTH_RADIUS_KM

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


def process_and_interpolate_df(df_input, grid_tree):
    """Standardize the station DataFrame, fill per-station gaps and grid it."""
    df_processed = process_dwd_df(df_input)

    # Fill per-station NaN gaps before gridding so that nearest-station-wins
    # always finds a valid value and never leaves a grid cell NaN due to a
    # single missing report (e.g. RR1c 5–6 h gaps, DD 3–4 h gaps).
    df_processed = fill_station_gaps(df_processed)

    gridded_ds = interpolate_dwd_to_grid_knearest_dask(
        df_processed,
        var_cols=zarr_vars,
        radius_km=radius_km,
        grid_tree=grid_tree,
        time_col="time",
        lat_col="latitude",
        lon_col="longitude",
        station_col="station_id",
        log="none",
    )

    return gridded_ds


def main() -> None:
    """Run the DWD MOSMIX ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    s3 = None
    if save_type == "S3":
        s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    # Empty the processing directory but keep the directory itself, which is
    # safer for processes that depend on the parent dir.
    clear_directory(forecast_process_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Download and parse the latest MOSMIX-S KMZ
    kmz_path = os.path.join(tmp_dir, DOWNLOADED_KMZ_FILE)
    logger.info("Downloading DWD MOSMIX data from %s", DWD_MOSMIX_URL)
    download_mosmix_file(DWD_MOSMIX_URL, kmz_path)

    logger.info("Reading and parsing KML from %s", kmz_path)
    df_data, global_metadata = parse_mosmix_kml(kmz_path)
    if df_data.empty:
        raise RuntimeError(f"No station data extracted from DWD MOSMIX {kmz_path}")

    # Convert longitudes from [-180, 180] to [0, 360] for consistency with GFS grid
    df_data["longitude"] = df_data["longitude"] % 360

    # %% Determine the model run from the KML IssueTime
    # The run time is what the API and the update check compare, so a KML
    # without one cannot be published.
    base_time = global_metadata.get("IssueTime", pd.NaT)
    if pd.isna(base_time):
        raise RuntimeError(f"DWD MOSMIX {kmz_path} has no IssueTime")

    # %% Skip the run if this issue has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/DWD_MOSMIX.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to DWD_MOSMIX, ending")
        return
    logger.info("Processing DWD_MOSMIX run %s", base_time)

    # %% Standardize, fill station gaps and grid onto the GFS grid
    # One tree serves both the gridding and the station map
    grid_tree = build_gfs_grid_tree()

    # The store holds the latest forecast only; its first row is the first
    # MOSMIX-S forecast time step (one hour after IssueTime).
    gridded_dwd_ds = process_and_interpolate_df(df_data, grid_tree)

    # %% Build the grid-cell-to-station map the API uses for station metadata
    logger.info("Building grid-to-stations mapping")
    grid_to_stations_map = build_grid_to_stations_map(
        df_data,  # Use original df_data which has station_name
        grid_tree,
        radius_km=radius_km,
        lat_col="latitude",
        lon_col="longitude",
        station_col="station_id",
        station_name_col="station_name",
        log="tqdm",
    )

    station_map_file = os.path.join(forecast_process_dir, "DWD_MOSMIX_stations.pickle")
    with open(station_map_file, "wb") as f:
        pickle.dump(grid_to_stations_map, f)
    logger.info("Station map saved to %s", station_map_file)

    # %% Write the gridded data to disk, then fill short time gaps per cell
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

    # Writing the dense gridded arrays to disk first keeps the gap filling below
    # from holding them all in memory
    with ProgressBar():
        gridded_dwd_ds_chunk.to_zarr(forecast_process_path + "_chunk.zarr", mode="w")
    del gridded_dwd_ds_chunk
    del gridded_dwd_ds

    ds_chunk_disk = xr.open_zarr(forecast_process_path + "_chunk.zarr", chunks="auto")

    logger.info("Interpolating gaps and extrapolating edges")
    ds_chunk_disk_interp = interpolate_temporal_gaps_efficiently(
        ds_chunk_disk,
        nearest_vars=nearest_neighbor_vars,
        max_gap_hours=GRID_MAX_GAP_HOURS,
        time_dim="time",
    )

    # %% Stack into the (variable, time, y, x) layout the API reads
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_times = ds_chunk_disk_interp.time.values
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second

    ds_rename = ds_chunk_disk_interp.rename({"time": "stacked_time"})

    # The API reads time from the store like any other variable
    time3d = (
        ((ds_rename["stacked_time"] - unix_epoch) / one_second)
        .astype("float32")
        .expand_dims(lat=ds_rename.lat, lon=ds_rename.lon)
        .transpose("stacked_time", "lat", "lon")
    )
    ds_rename["time"] = time3d

    # Cells with no station have no temperature; mark their time missing too
    ds_rename["time"] = ds_rename["time"].where(
        ~np.isnan(ds_rename["TMP_2maboveground"]), other=np.nan
    )

    vars_in = [v for v in zarr_vars if v in ds_rename.data_vars]
    ds_stack = ds_rename[vars_in].to_array(dim="var", name="var")

    ds_chunk = ds_stack.chunk(
        {
            "var": -1,
            "stacked_time": len(stacked_timesUnix),
            "lat": FINAL_CHUNK_SIZES["DWD"],
            "lon": FINAL_CHUNK_SIZES["DWD"],
        }
    )

    # %% Write the final store
    dwd_store_name = "DWD_MOSMIX.zarr.zip" if save_type == "S3" else "DWD_MOSMIX.zarr"
    if save_type == "S3":
        zarr_store = zarr.storage.ZipStore(
            forecast_process_dir + "/" + dwd_store_name, mode="a"
        )
    else:
        zarr_store = zarr.storage.LocalStore(
            forecast_process_dir + "/" + dwd_store_name
        )

    # Unlike the GRIB ingests this writes with xarray rather than dask, which
    # also stores the coordinate metadata.
    with ProgressBar():
        ds_chunk.to_zarr(zarr_store, mode="w")

    if save_type == "S3":
        zarr_store.close()

    # %% Publish the store and station map, then the run time the API uses to detect them
    publish_ingest_run(
        [dwd_store_name, "DWD_MOSMIX_stations.pickle"],
        base_time,
        "DWD_MOSMIX.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    clear_directory(forecast_process_dir)
    clear_directory(tmp_dir)
    logger.info("DWD_MOSMIX run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
