# %% Script to contain the helper functions as part of the data ingest for Pirate Weather
# Alexander Rey. July 2025

import logging
import re
import sys
import time
from typing import Iterable, Optional, Union

import cartopy.crs as ccrs
import dask.array as da
import numpy as np
import xarray as xr
from herbie import Path

# Import atmospheric calculation constants
from API.constants.shared_const import (
    BOLTON_CONST,
    CLOUD_RH_CRITICAL,
    CLOUD_RH_EXPONENT,
    FREEZING_LEVEL_HIGH,
    FREEZING_LEVEL_SURFACE,
    FREEZING_LEVEL_TEMP_TOLERANCE,
    GRAVITY,
    KELVIN_TO_CELSIUS,
    MISSING_DATA,
    REFC_THRESHOLD,
    WATER_VAPOR_GAS_CONSTANT_RATIO,
)

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Shared ingest constants
CHUNK_SIZES = {
    "NBM": 100,
    "HRRR": 100,
    "HRRR_6H": 100,
    "GFS": 50,
    "GEFS": 100,
    "HGEFS": 100,
    "ECMWF": 100,
    "NBM_Fire": 100,
    "RTMA": 100,
    "DWD": 100,
}

FINAL_CHUNK_SIZES = {
    "NBM": 3,
    "HRRR": 5,
    "HRRR_6H": 5,
    "GFS": 3,
    "GEFS": 3,
    "HGEFS": 3,
    "ECMWF": 3,
    "NBM_Fire": 5,
    "RTMA": 25,
    "DWD": 5,
}

FORECAST_LEAD_RANGES = {
    "GFS_1": list(range(1, 121)),
    "GFS_2": list(range(123, 241, 3)),
    "GEFS": list(range(3, 241, 3)),
    "NBM_FIRE": list(range(6, 192, 6)),
    "HRRR_1H": list(range(1, 19)),
    "HRRR_6H": list(range(18, 49)),
    "ECMWF_AIFS": list(range(0, 241, 6)),
    "ECMWF_IFS_1": list(range(3, 144, 3)),
    "ECMWF_IFS_2": list(range(144, 241, 6)),
    "AIGFS": list(range(0, 241, 6)),
    "AIGEFS": list(range(0, 241, 6)),
    "HGEFS": list(range(0, 241, 6)),
}

# Radius, in km, used for DWD model nearest-neighbor selection
DWD_RADIUS = 50

VALID_DATA_MIN = -100
VALID_DATA_MAX = 120000


def mask_invalid_data(daskArray, ignoreAxis=None):
    """Masks invalid data in a dask array, ignoring the time dimension."""
    # TODO: Update to mask for each variable according to reasonable values, as opposed to this global mask
    valid_mask = (daskArray >= VALID_DATA_MIN) & (daskArray <= VALID_DATA_MAX)
    # Ignore times by setting first dimension to True
    valid_mask[0, :, :, :] = True

    # Also ignore the specified axis if provided
    if ignoreAxis is not None:
        for i in ignoreAxis:
            valid_mask[i, :, :, :] = True
    return da.where(valid_mask, daskArray, MISSING_DATA)


def mask_invalid_refc(xrArr: "xr.DataArray") -> "xr.DataArray":
    """Masks REFC values less than 5, setting them to 0.

    Args:
        xrArr: The input xarray DataArray with REFC values.

    Returns:
        The masked xarray DataArray.
    """
    return xrArr.where(xrArr >= REFC_THRESHOLD, 0)


# Function to get the list of GRIB files from the forecast subscription, used by NBM
def getGribList(FH_forecastsub, matchStrings):
    try:
        gribList = [
            str(Path(x.get_localFilePath(matchStrings)).expand())
            for x in FH_forecastsub.file_exists
        ]
    except Exception:
        logger.warning("Download Failure 1, wait 20 seconds and retry")
        time.sleep(20)
        FH_forecastsub.download(matchStrings, verbose=False)
        try:
            gribList = [
                str(Path(x.get_localFilePath(matchStrings)).expand())
                for x in FH_forecastsub.file_exists
            ]
        except Exception:
            logger.warning("Download Failure 2, wait 20 seconds and retry")
            time.sleep(20)
            FH_forecastsub.download(matchStrings, verbose=False)
            try:
                gribList = [
                    str(Path(x.get_localFilePath(matchStrings)).expand())
                    for x in FH_forecastsub.file_exists
                ]
            except Exception:
                logger.warning("Download Failure 3, wait 20 seconds and retry")
                time.sleep(20)
                FH_forecastsub.download(matchStrings, verbose=False)
                try:
                    gribList = [
                        str(Path(x.get_localFilePath(matchStrings)).expand())
                        for x in FH_forecastsub.file_exists
                    ]
                except Exception:
                    logger.warning("Download Failure 4, wait 20 seconds and retry")
                    time.sleep(20)
                    FH_forecastsub.download(matchStrings, verbose=False)
                    try:
                        gribList = [
                            str(Path(x.get_localFilePath(matchStrings)).expand())
                            for x in FH_forecastsub.file_exists
                        ]
                    except Exception:
                        logger.warning("Download Failure 5, wait 20 seconds and retry")
                        time.sleep(20)
                        FH_forecastsub.download(matchStrings, verbose=False)
                        try:
                            gribList = [
                                str(Path(x.get_localFilePath(matchStrings)).expand())
                                for x in FH_forecastsub.file_exists
                            ]
                        except Exception:
                            logger.critical("Download Failure 6, Fail")
                            exit(1)
    return gribList


def validate_grib_stats(gribCheck):
    """
    Inspect gribCheck.stdout (from `wgrib2 … -stats`) for min/max values,
    print any out-of-range records, and exit(10) if invalid data is found.

    Expects:
      - gribCheck.stdout: the full stdout string
      - globals: VALID_DATA_MIN, VALID_DATA_MAX
    """
    # extract all mins and maxs
    minValues = [float(m) for m in re.findall(r"min=([-\d\.eE]+)", gribCheck.stdout)]
    maxValues = [float(M) for M in re.findall(r"max=([-\d\.eE]+)", gribCheck.stdout)]

    # extract variable names (4th field)
    varNames = re.findall(r"(?m)^(?:[^:]+:){3}([^:]+):", gribCheck.stdout)
    # ensure we found at least one variable
    if not varNames:
        logger.error("Error: no variables found in GRIB stats output.")
        sys.exit(10)

    # extract forecast lead times (6th field)
    varTimes = re.findall(r"(?m)^(?:[^:]+:){5}([^:]+):", gribCheck.stdout)

    # find any indices where data is out of range
    # TODO: This would be better if we checked against a dictionary of valid ranges defined per variable
    invalidIdxs = [
        i
        for i, (mn, mx) in enumerate(zip(minValues, maxValues))
        if mn < VALID_DATA_MIN or mx > VALID_DATA_MAX
    ]

    if invalidIdxs:
        logger.error("Invalid data found in grib files:")
        for i in invalidIdxs:
            logger.error("  Variable : %s", varNames[i])
            logger.error("  Time     : %s", varTimes[i])
            logger.error("  Min/Max  : %s / %s", minValues[i], maxValues[i])
            logger.error("---")
        logger.error("Exiting due to invalid data in grib files.")
        sys.exit(10)

    else:
        logger.info("All grib files passed validation checks.")
        # compute overall min/max for each variable across all times
        varExtremes = {}
        for var, mn, mx in zip(varNames, minValues, maxValues):
            lo, hi = varExtremes.setdefault(var, [mn, mx])
            varExtremes[var][0] = min(lo, mn)
            varExtremes[var][1] = max(hi, mx)

        # print overall extremes
        logger.info("Overall min/max for each variable across all times:")
        for var, (mn, mx) in varExtremes.items():
            logger.info("  %s: min=%s, max=%s", var, mn, mx)

    # all good
    return True


def pad_to_chunk_size(dask_array: da.Array, final_chunk: int) -> da.Array:
    """Pad a 4D dask array so its Y and X dimensions are multiples of final_chunk.

    This ensures efficient zarr storage by aligning spatial dimensions to chunk boundaries.
    Padding is done with NaN values on the right and bottom edges.

    Args:
        dask_array: 4D dask array with shape (var, time, y, x).
        final_chunk: The target chunk size for spatial dimensions.

    Returns:
        Padded dask array with y and x dimensions that are multiples of final_chunk.
        If no padding is needed, returns the original array.
    """
    y, x = dask_array.shape[2], dask_array.shape[3]
    pad_y = (-y) % final_chunk  # 0..(final_chunk - 1)
    pad_x = (-x) % final_chunk  # 0..(final_chunk - 1)

    # Only pad if necessary
    if pad_y or pad_x:
        return da.pad(
            dask_array,
            ((0, 0), (0, 0), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=np.nan,
        )
    return dask_array


def earth_relative_wind_components(
    ugrd: xr.DataArray, vgrd: xr.DataArray
) -> tuple[
    np.ndarray, np.ndarray
]:  # Based off: https://unidata.github.io/python-gallery/examples/500hPa_Absolute_Vorticity_winds.html#function-to-compute-earth-relative-winds
    """Calculate north-relative wind components from grid-relative components.

    Uses Cartopy to transform vectors from the model's grid-relative projection
    to a standard Plate Carree projection (earth-relative).

    Args:
        ugrd: Xarray DataArray of the grid-relative u-component of the wind.
        vgrd: Xarray DataArray of the grid-relative v-component of the wind.

    Returns:
        A tuple containing two numpy arrays: the earth-relative u-component (ut)
        and v-component (vt) of the wind.
    """
    data_crs = ugrd.metpy_crs.metpy.cartopy_crs

    x = ugrd.x.values
    y = ugrd.y.values

    xx, yy = np.meshgrid(x, y)

    ut, vt = ccrs.PlateCarree().transform_vectors(
        data_crs, xx, yy, ugrd.values, vgrd.values
    )

    return ut, vt


def interp_time_take_blend(
    arr: da.Array,
    stacked_timesUnix: np.ndarray,
    hourly_timesUnix: np.ndarray,
    nearest_vars: Optional[Union[int, Iterable[int]]] = None,  # var indices using NN
    dtype: str = "float32",
    fill_value: float = np.nan,
    time_axis: int = 1,
) -> da.Array:
    r"""Interpolate model data along the time dimension via gather-and-blend.

    The helper assumes the input has shape \(V, T, Y, X\) and that time is
    the second axis. It gathers the values for the two surrounding stored
    times using ``da.take``, then linearly blends them using the fractional
    offset between ``stacked_timesUnix`` and ``hourly_timesUnix``. Points that
    fall outside the time range defined by ``stacked_timesUnix`` are filled with
    ``fill_value``. Optionally, a subset of variable indices may be overridden
    with nearest-neighbor interpolation instead of the blended values.

    Args:
        arr: Chunked Dask array containing the forecast in \(V, T, Y, X\)
            order. The time axis must already be a single chunk so gather
            operations stay within chunk boundaries.
        stacked_timesUnix: Known source timestamps to interpolate from (monotonic
            increasing unix seconds).
        hourly_timesUnix: Desired target timestamps; must lie within the range
            spanned by ``stacked_timesUnix`` when possible.
        nearest_vars: Optional variable indices that should use the closer
            neighbor directly instead of linear blending (e.g., categorical
            flags). Can be a single index or an iterable.
        dtype: Output dtype for the interpolated array.
        fill_value: Value used for times outside the available range.
        time_axis: Axis index for time (must be 1 in this helper).

    Returns:
        A dask array shaped \(V, T_new, Y, X\) with interpolated (or overridden)
        values and ``dtype``.
    """
    if arr.ndim != 4:
        raise ValueError("Expected arr with dims (V, T, Y, X).")

    VAX, TAX = 0, time_axis
    if TAX != 1:
        raise NotImplementedError("This helper assumes time_axis == 1 for (V,T,Y,X).")

    # Precompute the two neighbor‐indices and the weights
    x_a = np.array(stacked_timesUnix)
    x_b = np.array(hourly_timesUnix)

    idx = np.searchsorted(x_a, x_b) - 1
    idx0 = np.clip(idx, 0, len(x_a) - 2)
    idx1 = idx0 + 1
    w = (x_b - x_a[idx0]) / (x_a[idx1] - x_a[idx0])  # float array, shape (T_new,)

    # boolean mask of “in‐range” points
    valid = (x_b >= x_a[0]) & (x_b <= x_a[-1])  # shape (T_new,)

    T_new = int(len(idx0))
    if not (len(idx1) == T_new and len(w) == T_new and len(valid) == T_new):
        raise ValueError("idx0, idx1, w, and valid must all have length T_new.")

    # Ensure time axis already fits in one chunk so gather (`da.take`) stays within chunk boundaries
    time_chunks = arr.chunks[TAX]
    if len(time_chunks) != 1:
        raise ValueError(
            "time axis must be a single chunk; please rechunk with ``arr.rechunk({time_axis: -1})`` "
            f"before calling (got {len(time_chunks)} chunks)."
        )
    arr_t = arr

    # Gather neighbors along time
    y0 = da.take(arr_t, idx0, axis=TAX)  # (V, T_new, Y, X)
    y1 = da.take(arr_t, idx1, axis=TAX)

    # Weighted blend
    w_r = da.asarray(w, chunks=(T_new,))[None, :, None, None]
    out = (1 - w_r) * y0 + w_r * y1
    out = out.astype(dtype, copy=False)

    # Mask invalid new times
    if (~valid).any():
        valid_r = da.asarray(valid, chunks=(T_new,))[None, :, None, None]
        out = da.where(valid_r, out, fill_value)

    # Optional nearest-neighbor override for specified variable indices
    if nearest_vars is not None:
        # Precompute nearest indices once: closer of idx0 / idx1
        nearest_idx = np.where(w < 0.5, idx0, idx1).astype(idx0.dtype)

        # Normalize indices as a sorted, unique list
        if isinstance(nearest_vars, int):
            nearest_vars = [nearest_vars]
        nv = sorted(set(int(i) for i in nearest_vars))

        # Compute nearest only for needed variables (cheap if few vars)
        # Shape of each nearest slice: (1, T_new, Y, X)
        take_nn = da.take(arr_t, nearest_idx, axis=TAX)
        # Replace in 'out' per variable index
        pieces = []
        prev = 0
        for i in nv:
            if i < 0 or i >= arr.shape[VAX]:
                raise IndexError(
                    f"nearest_vars index {i} out of range for V={arr.shape[VAX]}"
                )
            if i > prev:
                pieces.append(out[prev:i])  # unchanged segment
            pieces.append(
                take_nn[i : i + 1].astype(dtype, copy=False)
            )  # nearest segment
            prev = i + 1
        if prev < arr.shape[VAX]:
            pieces.append(out[prev:])
        out = da.concatenate(pieces, axis=VAX)

    return out


def interpolate_temporal_gaps_efficiently(
    ds_chunked, nearest_vars=None, max_gap_hours=3, time_dim="time"
):
    """
    Interpolates temporal gaps and extrapolates edges efficiently in a sparse Dask/Xarray dataset.

    Logic:
    1. Re-chunks data to be contiguous in time ('pencils').
    2. Short-circuits empty (all-NaN) spatial chunks.
    3. Interpolates internal gaps (Linear by default, Nearest for specific vars).
    4. Extrapolates edges (Nearest neighbor / Forward & Back fill) for ALL vars.

    Args:
        ds_chunked (xr.Dataset): Input dataset. Must be chunked with time as a single chunk.
        nearest_vars (list): List of variable names to use 'nearest' interpolation for
                             (e.g., flags, codes). Default is 'linear' for others.
        max_gap_hours (int): Max gap size to interpolate internally.
                             Extrapolation is applied to the ends regardless of gap size.
        time_dim (str): Name of the time dimension.

    Returns:
        xr.Dataset: Processed dataset.
    """

    if nearest_vars is None:
        nearest_vars = []

    # Helper function applied to each Dask block
    def _interpolate_block(block, time_coords, method):
        # OPTIMIZATION: Short-circuit empty blocks
        # If the entire spatial chunk is NaN, return immediately.
        if np.all(np.isnan(block)):
            return block

        # Wrap numpy block in DataArray for convenient Xarray methods
        da_temp = xr.DataArray(
            block, dims=(time_dim, "y", "x"), coords={time_dim: time_coords}
        )

        # 1. Interpolate Internal Gaps
        # use_coordinate=True ensures we respect actual time steps, not just index count
        filled = da_temp.interpolate_na(
            dim=time_dim, method=method, limit=max_gap_hours, use_coordinate=True
        )

        # 2. Extrapolate Edges (Nearest Neighbour)
        # We use ffill (forward) and bfill (backward) to extend the last known
        # valid value to the start/end of the series.
        filled = filled.ffill(time_dim).bfill(time_dim)

        return filled.values

    # Function to map over every variable
    def _process_variable(da_var):
        # Skip coordinate variables or vars without time
        if time_dim not in da_var.dims:
            return da_var

        # Determine interpolation method for this specific variable
        # Default is 'linear', unless specified in nearest_vars
        interp_method = "nearest" if da_var.name in nearest_vars else "linear"

        time_coords = da_var[time_dim].values

        # dask.map_blocks lets us run the logic on every chunk in parallel
        processed_data = da_var.data.map_blocks(
            _interpolate_block,
            time_coords=time_coords,
            method=interp_method,
            dtype=da_var.dtype,
            chunks=da_var.chunks,
        )

        return da_var.copy(data=processed_data)

    # Execute
    return ds_chunked.map(_process_variable)


def calculate_freezing_level(
    temperature_levels: xr.DataArray,
    geopotential_levels: xr.DataArray,
    pressure_levels: list,
) -> xr.DataArray:
    """Calculate freezing level height from temperature and geopotential at pressure levels.

    Finds the altitude where temperature = 273.15 K (0°C) using linear interpolation
    between pressure levels.

    Args:
        temperature_levels: Temperature at pressure levels (K), shape (step/time, level, lat, lon)
        geopotential_levels: Geopotential at pressure levels (m²/s²), shape (step/time, level, lat, lon)
        pressure_levels: List of pressure levels in hPa (e.g., [1000, 925, 850, ...])

    Returns:
        Freezing level height in meters, shape (step/time, lat, lon)
    """
    # Convert geopotential to height (divide by gravity)
    height_levels = geopotential_levels / GRAVITY  # meters

    # Initialize output with NaN
    freezing_level = xr.full_like(temperature_levels.isel(level=0, drop=True), np.nan)

    # Loop through adjacent pressure levels to find freezing level
    # Process from surface to top of atmosphere
    for i in range(len(pressure_levels) - 1):
        # Get temperatures and heights at two adjacent levels
        t_lower = temperature_levels.isel(level=i)
        t_upper = temperature_levels.isel(level=i + 1)
        h_lower = height_levels.isel(level=i)
        h_upper = height_levels.isel(level=i + 1)

        # Check if freezing level is between these two levels
        # (temperature crosses KELVIN_TO_CELSIUS)
        crosses = ((t_lower >= KELVIN_TO_CELSIUS) & (t_upper < KELVIN_TO_CELSIUS)) | (
            (t_lower < KELVIN_TO_CELSIUS) & (t_upper >= KELVIN_TO_CELSIUS)
        )

        # Linear interpolation where temperature crosses freezing
        # Only process if there are any crossings to improve performance
        if crosses.any():
            # Avoid division by zero
            temp_diff = t_upper - t_lower
            # Only interpolate where temp changes significantly
            valid = np.abs(temp_diff) > FREEZING_LEVEL_TEMP_TOLERANCE

            fraction = xr.where(valid, (KELVIN_TO_CELSIUS - t_lower) / temp_diff, 0)
            interp_height = h_lower + fraction * (h_upper - h_lower)

            # Update freezing level where it crosses and hasn't been set yet
            # This ensures we get the first (lowest) crossing
            freezing_level = xr.where(
                crosses & np.isnan(freezing_level), interp_height, freezing_level
            )

    # Handle edge cases where no crossing was found
    # If all temps are below freezing, set to surface
    # If all temps are above freezing, set to high altitude
    all_cold = temperature_levels.min(dim="level") < KELVIN_TO_CELSIUS
    all_warm = temperature_levels.max(dim="level") >= KELVIN_TO_CELSIUS

    freezing_level = xr.where(
        np.isnan(freezing_level) & all_cold, FREEZING_LEVEL_SURFACE, freezing_level
    )
    freezing_level = xr.where(
        np.isnan(freezing_level) & all_warm, FREEZING_LEVEL_HIGH, freezing_level
    )

    return freezing_level


def calculate_cloud_cover_from_rh(
    temperature_levels: xr.DataArray,
    specific_humidity_levels: xr.DataArray,
    pressure_levels: list,
) -> xr.DataArray:
    """Calculate total cloud cover from relative humidity profiles using simplified Slingo method.

    Computes relative humidity at each pressure level, applies an empirical cloud fraction
    formula, and integrates vertically to estimate total cloud cover.

    Args:
        temperature_levels: Temperature at pressure levels (K), shape (step/time, level, lat, lon)
        specific_humidity_levels: Specific humidity at pressure levels (kg/kg), shape (step/time, level, lat, lon)
        pressure_levels: List of pressure levels in hPa (e.g., [1000, 925, 850, ...])

    Returns:
        Total cloud cover (fraction 0-1), shape (step/time, lat, lon)
    """
    # Calculate saturation vapor pressure using Bolton's formula
    # e_s = base_pressure * exp(temp_coeff * (T - KELVIN_TO_CELSIUS) / (T - KELVIN_TO_CELSIUS + temp_offset))
    T_celsius = temperature_levels - KELVIN_TO_CELSIUS
    e_sat = BOLTON_CONST["base_pressure"] * np.exp(
        BOLTON_CONST["temp_coeff"]
        * T_celsius
        / (T_celsius + BOLTON_CONST["temp_offset"])
    )  # hPa

    # Convert specific humidity to mixing ratio
    # q = w / (1 + w) => w = q / (1 - q)
    mixing_ratio = specific_humidity_levels / (1 - specific_humidity_levels)

    # Calculate actual vapor pressure
    # e = (w * P) / (WATER_VAPOR_GAS_CONSTANT_RATIO + w) where P is pressure in hPa
    pressure_array = xr.DataArray(
        pressure_levels, dims="level", coords={"level": temperature_levels.level}
    )

    e_actual = (mixing_ratio * pressure_array) / (
        WATER_VAPOR_GAS_CONSTANT_RATIO + mixing_ratio
    )

    # Calculate relative humidity (0-1)
    rh = e_actual / e_sat
    rh = xr.where(rh > 1, 1, rh)  # Cap at 100%
    rh = xr.where(rh < 0, 0, rh)  # Floor at 0%

    # Apply simplified cloud fraction formula based on RH
    # Cloud fraction increases sigmoidally as RH approaches 100%
    # Using a simplified version of Xu-Randall (1996)
    # C = max(0, (RH - RH_crit) / (1 - RH_crit))^CLOUD_RH_EXPONENT
    cloud_fraction = xr.where(
        rh > CLOUD_RH_CRITICAL,
        ((rh - CLOUD_RH_CRITICAL) / (1 - CLOUD_RH_CRITICAL)) ** CLOUD_RH_EXPONENT,
        0.0,
    )

    # Vertical integration using random overlap assumption
    # Total cloud cover = 1 - product(1 - cloud_fraction_i)
    # This prevents unrealistic 100% cloud cover from multiple layers
    cloud_free = 1 - cloud_fraction
    total_cloud_free = cloud_free.prod(dim="level")
    total_cloud_cover = 1 - total_cloud_free

    # Ensure result is between 0 and 1
    total_cloud_cover = xr.where(total_cloud_cover < 0, 0, total_cloud_cover)
    total_cloud_cover = xr.where(total_cloud_cover > 1, 1, total_cloud_cover)

    return total_cloud_cover


def derive_precip_type(
    apcp: xr.DataArray,
    temp_surface: Optional[xr.DataArray] = None,
    temp_levels: Optional[xr.DataArray] = None,
    geopotential_levels: Optional[xr.DataArray] = None,
    pressure_levels: Optional[list] = None,
    apcp_threshold: float = 0.0001,
    rain_thresh_c: float = 5.0,
    snow_thresh_c: float = -10.0,
    warm_layer_min_m: float = 200.0,
    near_surface_freeze_m: float = 100.0,
) -> xr.DataArray:
    """Derive categorical precip type from precipitation and temperature.

    Returns integer codes: 1=snow, 2=freezing rain, 3=sleet, 4=rain, 0=no/insignificant precip.

    The rules are conservative and tunable via the threshold parameters. Inputs
    expect temperatures in Kelvin (consistent with other helpers).
    """

    # Default output: 0 (no precip)
    out = xr.full_like(apcp, 0).astype("int8")

    # Mask where precipitation is meaningful
    has_precip = apcp > apcp_threshold

    # Determine surface temperature (use lowest pressure level as proxy if needed)
    if temp_surface is None and temp_levels is not None:
        temp_surface = temp_levels.isel(level=0)

    if temp_surface is None:
        # No temperature info: mark precip as rain conservatively
        return xr.where(has_precip, 4, out)

    temp_surf_c = temp_surface - KELVIN_TO_CELSIUS

    # Strong-warm/strong-cold shortcuts
    out = xr.where((temp_surf_c >= rain_thresh_c) & has_precip, 4, out)
    out = xr.where((temp_surf_c <= snow_thresh_c) & has_precip, 1, out)

    # Remaining points to classify
    mid_mask = has_precip & (out == 0)

    if temp_levels is None:
        # Without vertical profile: decide by sign of surface temp
        out = xr.where(mid_mask & (temp_surf_c > 0), 4, out)
        out = xr.where(mid_mask & (temp_surf_c <= 0), 1, out)
        return out

    # Convert levels to Celsius
    temp_levels_c = temp_levels - KELVIN_TO_CELSIUS

    # Compute heights if geopotential available. If `pressure_levels` is
    # supplied, attach as a coordinate for clarity (non-fatal on mismatch).
    if geopotential_levels is not None:
        height_levels = geopotential_levels / GRAVITY
        if pressure_levels is not None:
            try:
                height_levels = height_levels.assign_coords(
                    level=("level", pressure_levels)
                )
            except Exception:
                # ignore if coords already set or lengths mismatch
                pass
    else:
        height_levels = None

    # Warm layer detection: any level above 0C
    warm_present = (temp_levels_c > 0).any(dim="level")

    # Approximate warm-layer thickness and base/top heights when heights are available
    warm_thickness = None
    warm_base = None
    if height_levels is not None:
        warm_heights = height_levels.where(temp_levels_c > 0)
        # max - min across level, result will be NaN where no warm levels
        warm_top = warm_heights.max(dim="level")
        warm_base = warm_heights.min(dim="level")
        warm_thickness = warm_top - warm_base

    # Freezing rain: warm layer aloft (sufficient thickness + base above near-surface) + surface <= 0
    fr_mask = mid_mask & warm_present & (temp_surf_c <= 0)
    if warm_thickness is not None:
        fr_mask = fr_mask & (warm_thickness >= warm_layer_min_m)
    if warm_base is not None:
        fr_mask = fr_mask & (warm_base >= near_surface_freeze_m)
    out = xr.where(fr_mask, 2, out)

    # Sleet: warm layer aloft with surface > 0 and sufficient warm-layer thickness
    sleet_mask = mid_mask & warm_present & (temp_surf_c > 0)
    if warm_thickness is not None:
        sleet_mask = sleet_mask & (warm_thickness >= warm_layer_min_m)
    out = xr.where(sleet_mask, 3, out)

    # Any remaining mid_mask: snow if surface <=0 else rain
    remaining = mid_mask & (out == 0)
    out = xr.where(remaining & (temp_surf_c <= 0), 1, out)
    out = xr.where(remaining & (temp_surf_c > 0), 4, out)

    return out
