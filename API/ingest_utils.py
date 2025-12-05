# %% Script to contain the helper functions as part of the data ingest for Pirate Weather
# Alexander Rey. July 17 2025

import re
import sys
import time
from typing import Iterable, Optional, Union

import cartopy.crs as ccrs
import dask.array as da
import numpy as np
import xarray as xr
from herbie import Path

from API.constants.shared_const import MISSING_DATA, REFC_THRESHOLD

# Shared ingest constants
CHUNK_SIZES = {
    "NBM": 100,
    "HRRR": 100,
    "HRRR_6H": 100,
    "GFS": 50,
    "GEFS": 100,
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
        print("Download Failure 1, wait 20 seconds and retry")
        time.sleep(20)
        FH_forecastsub.download(matchStrings, verbose=False)
        try:
            gribList = [
                str(Path(x.get_localFilePath(matchStrings)).expand())
                for x in FH_forecastsub.file_exists
            ]
        except Exception:
            print("Download Failure 2, wait 20 seconds and retry")
            time.sleep(20)
            FH_forecastsub.download(matchStrings, verbose=False)
            try:
                gribList = [
                    str(Path(x.get_localFilePath(matchStrings)).expand())
                    for x in FH_forecastsub.file_exists
                ]
            except Exception:
                print("Download Failure 3, wait 20 seconds and retry")
                time.sleep(20)
                FH_forecastsub.download(matchStrings, verbose=False)
                try:
                    gribList = [
                        str(Path(x.get_localFilePath(matchStrings)).expand())
                        for x in FH_forecastsub.file_exists
                    ]
                except Exception:
                    print("Download Failure 4, wait 20 seconds and retry")
                    time.sleep(20)
                    FH_forecastsub.download(matchStrings, verbose=False)
                    try:
                        gribList = [
                            str(Path(x.get_localFilePath(matchStrings)).expand())
                            for x in FH_forecastsub.file_exists
                        ]
                    except Exception:
                        print("Download Failure 5, wait 20 seconds and retry")
                        time.sleep(20)
                        FH_forecastsub.download(matchStrings, verbose=False)
                        try:
                            gribList = [
                                str(Path(x.get_localFilePath(matchStrings)).expand())
                                for x in FH_forecastsub.file_exists
                            ]
                        except Exception:
                            print("Download Failure 6, Fail")
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
        print("Error: no variables found in GRIB stats output.")
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
        print("Invalid data found in grib files:")
        for i in invalidIdxs:
            print(f"  Variable : {varNames[i]}")
            print(f"  Time     : {varTimes[i]}")
            print(f"  Min/Max  : {minValues[i]} / {maxValues[i]}")
            print("---")
        print("Exiting due to invalid data in grib files.")
        sys.exit(10)

    else:
        print("All grib files passed validation checks.")
        # compute overall min/max for each variable across all times
        varExtremes = {}
        for var, mn, mx in zip(varNames, minValues, maxValues):
            lo, hi = varExtremes.setdefault(var, [mn, mx])
            varExtremes[var][0] = min(lo, mn)
            varExtremes[var][1] = max(hi, mx)

        # print overall extremes
        print("Overall min/max for each variable across all times:")
        for var, (mn, mx) in varExtremes.items():
            print(f"  {var}: min={mn}, max={mx}")

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
    """
    Interpolate along `time_axis` of a 4D array (V, T, Y, X) using gather+blend,
    with optional nearest-neighbor override for selected variable indices.

    Returns a Dask array with shape (V, T_new, Y, X) and dtype `dtype`.
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

    # Ensure time is one chunk so gather (`da.take`) stays within chunk boundaries
    arr_t = arr.rechunk({TAX: -1})

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
