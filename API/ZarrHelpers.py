# Helpers for working with Zarr data in the Pirate Weather API
# Alexander Rey, October 2025

import os
import random
import time

import numpy as np
import s3fs
import xarray as xr
import zarr

from API.constants.api_const import (
    MAX_S3_RETRIES,
    S3_BASE_DELAY,
)

pw_api_key = os.environ.get("PW_API", "")


def _add_custom_header(request, **kwargs):
    request.headers["apikey"] = pw_api_key


class S3ZipStore(zarr.storage.ZipStore):
    def __init__(self, path: s3fs.S3File) -> None:
        super().__init__(path="", mode="r")
        self.path = path


def _retry_s3_operation(
    operation, max_retries=MAX_S3_RETRIES, base_delay=S3_BASE_DELAY
):
    """Retry S3 operations with exponential backoff for rate limiting."""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            # Check if it's a rate limiting error
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"S3 rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
            # Re-raise the exception if it's not rate limiting or max retries reached
            raise e
    raise Exception(f"Failed after {max_retries} attempts")


def setup_testing_zipstore(s3, s3_bucket, ingest_version, save_type, model_name):
    """Sets up a zarr store from a zipped zarr file in S3 or locally.

    Parameters:
        - s3 (s3fs.S3FileSystem): An s3fs filesystem object.
        - s3_bucket (str): The S3 bucket name or local path.
        - ingest_version (str): The version string for the data.
        - save_type (str): The type of storage ("S3", "S3Zarr", or local path).
        - model_name (str): The name of the model.

    Returns:
        - store: A zarr store object.
    """

    if save_type == "S3":
        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
        # Try an old ingest version for testing
        except FileNotFoundError:
            ingest_version = "v28"
            print("Using old ingest version: " + ingest_version)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
    elif save_type == "S3Zarr":
        try:
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)
        except FileNotFoundError:
            ingest_version = "v28"
            print("Using old ingest version: " + ingest_version)
            f = _retry_s3_operation(
                lambda: s3.open(
                    "s3://"
                    + s3_bucket
                    + "/ForecastTar_v2/"
                    + ingest_version
                    + "/"
                    + model_name
                    + ".zarr.zip"
                )
            )
            store = S3ZipStore(f)

    else:
        f = s3_bucket + model_name + ".zarr.zip"
        store = zarr.storage.ZipStore(f, mode="r")

    return store


## Estimate Visibility for ERA5
# https://chatgpt.com/share/68fa2b7f-0840-800d-b90e-81f5edb4bc90
def estimate_visibility_from_numpy(
    arr: np.ndarray,
    var_index: dict,
    var_axis: int = 0,   # 0 => shape (n_vars, n_time); set to 1 if (n_time, n_vars)
    params: dict | None = None,
) -> np.ndarray:
    """
    Heuristic visibility proxy (km) from a NumPy array built from ERA5 variables.

    Parameters
    ----------
    arr : np.ndarray
        Array with variables stacked along `var_axis` and time along the other axis.
        Typically shape = (n_vars, n_time).
    var_index : dict[str,int]
        Mapping from variable name -> 1-based index in `arr` (as in your ERA5 dict).
    var_axis : int
        Axis along which variables are stacked (0 or 1).
    params : dict
        Optional thresholds/parameters override.

    Returns
    -------
    np.ndarray
        Visibility in km, shape = (n_time,)
    """
    # Thresholds/parameters (same logic as xarray version)
    p = {
        # Fog/haze
        "dpd_fog_K": 1.0,
        "dpd_haze_K": 2.0,
        "lcc_fog_min": 0.6,
        "no_precip_mm_h": 0.1,
        # Precip→vis bins (mm/h)
        "rain_bins_mm_h":  np.array([0.5, 5.0, 20.0]),
        "rain_vis_km":     np.array([10.0, 8.0, 3.0, 1.0]),
        "snow_bins_mm_h":  np.array([0.1, 1.0, 5.0]),
        "snow_vis_km":     np.array([10.0, 5.0, 2.0, 0.8]),
        # Clear sky vis (km)
        "vis_fog_km": 0.5,
        "vis_haze_km": 5.0,
        "vis_clear_km": 10.0,
        # Ceiling caps
        "cap1_cbase_m": 50,   "cap1_vis_km": 1.0,
        "cap2_cbase_m": 100,  "cap2_vis_km": 3.0,
        # Wind relaxation
        "wind_relax_ms": 8.0,
        "wind_relax_min_vis_km": 2.0,
        # Output clamp
        "min_vis_km": 0.2,
        "max_vis_km": 16.0,
    }
    if params:
        p.update(params)

    # Helper to get a variable vector (time,) from 1-based index mapping
    def get(name):
        idx1 = var_index.get(name)
        if idx1 is None:
            return None
        i = idx1 - 1  # convert 1-based -> 0-based
        if var_axis == 0:
            out = arr[i, ...]
        else:
            out = arr[..., i]
        # Ensure shape (time,)
        return np.asarray(out)

    # Pull what we need (some optional)
    t2m  = get("2m_temperature")
    td2m = get("2m_dewpoint_temperature")
    if t2m is None or td2m is None:
        raise ValueError("Need both '2m_temperature' and '2m_dewpoint_temperature' in var_index.")

    lcc   = get("low_cloud_cover")
    cbase = get("cloud_base_height")
    u10   = get("10m_u_component_of_wind")
    v10   = get("10m_v_component_of_wind")

    # Rates in m s-1 → mm h-1
    def mmh(x):
        return x * 3600.0 * 1000.0

    ls_rain = get("large_scale_rain_rate")
    cv_rain = get("convective_rain_rate")
    ls_snow = get("large_scale_snowfall_rate_water_equivalent")
    cv_snow = get("convective_snowfall_rate_water_equivalent")

    # Determine time length and make safe defaults
    n_time = t2m.shape[0]
    def zeros(): return np.zeros(n_time, dtype=float)
    def full(v): return np.full(n_time, v, dtype=float)

    if lcc   is None: lcc   = zeros()
    if cbase is None: cbase = full(1e9)
    if u10   is None: u10   = zeros()
    if v10   is None: v10   = zeros()
    if ls_rain is None: ls_rain = zeros()
    if cv_rain is None: cv_rain = zeros()
    if ls_snow is None: ls_snow = zeros()
    if cv_snow is None: cv_snow = zeros()

    rain_rate_mm_h = mmh(ls_rain + cv_rain)
    snow_rate_mm_h = mmh(ls_snow + cv_snow)

    # --- Precip → visibility via piecewise-constant bins ---
    def binned_vis(rate, edges, values):
        # values length = len(edges)+1
        vis = np.full_like(rate, values[-1], dtype=float)
        # fill from highest edge downward so lower bins overwrite
        for i in range(len(edges)-1, -1, -1):
            vis = np.where(rate <= edges[i], values[i], vis)
        return vis

    vis_rain = binned_vis(rain_rate_mm_h, p["rain_bins_mm_h"], p["rain_vis_km"])
    vis_snow = binned_vis(snow_rate_mm_h, p["snow_bins_mm_h"], p["snow_vis_km"])
    vis_precip = np.minimum(vis_rain, vis_snow)

    # --- Clear-sky component from dewpoint depression + low cloud ---
    dpd = t2m - td2m
    no_precip = (rain_rate_mm_h <= p["no_precip_mm_h"]) & (snow_rate_mm_h <= p["no_precip_mm_h"])
    fog_flag  = (dpd <= p["dpd_fog_K"])  & (lcc >= p["lcc_fog_min"]) & no_precip
    haze_flag = (dpd <= p["dpd_haze_K"]) & no_precip

    vis_clear = np.where(
        fog_flag, p["vis_fog_km"],
        np.where(haze_flag, p["vis_haze_km"], p["vis_clear_km"])
    )

    # --- Combine (more limiting wins) ---
    vis = np.minimum(vis_clear, vis_precip)

    # --- Ceiling penalties ---
    vis = np.where(cbase < p["cap1_cbase_m"], np.minimum(vis, p["cap1_vis_km"]), vis)
    mid_cap = (cbase >= p["cap1_cbase_m"]) & (cbase < p["cap2_cbase_m"])
    vis = np.where(mid_cap, np.minimum(vis, p["cap2_vis_km"]), vis)

    # --- Wind mixing relaxation for dense fog ---
    wind10 = np.hypot(u10, v10)
    relax = (wind10 >= p["wind_relax_ms"]) & fog_flag
    vis = np.where(relax, np.maximum(vis, p["wind_relax_min_vis_km"]), vis)

    # --- Clamp & return ---
    np.clip(vis, p["min_vis_km"], p["max_vis_km"], out=vis) * 1000 # Return in m
    return vis

# Function to initialize in ERA5 xarray dataset
def init_ERA5():
    # Open the ERA5 dataset from Google Cloud
    dsERA5 = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks={"time": 25},
        storage_options=dict(token="anon"),
    )

    ERA5_lats = dsERA5["latitude"][:]
    ERA5_lons = dsERA5["longitude"][:]
    ERA5_times = dsERA5["time"][:]

    ERA5_Data = {
        "dsERA5": dsERA5,
        "ERA5_lats": ERA5_lats,
        "ERA5_lons": ERA5_lons,
        "ERA5_times": ERA5_times,
    }

    return ERA5_Data
