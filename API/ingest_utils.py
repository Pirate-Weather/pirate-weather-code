# %% Script to contain the helper functions as part of the data ingest for Pirate Weather
# Alexander Rey. July 17 2025

import dask.array as da
import numpy as np
import time
from herbie import Path

VALID_DATA_MIN = -100
VALID_DATA_MAX = 120000


def mask_invalid_data(dask_array):
    """Masks invalid data in a dask array, ignoring the time dimension."""
    # TODO: Update to mask for each variable according to reasonable values, as opposed to this global mask
    valid_mask = (dask_array >= VALID_DATA_MIN) & (dask_array <= VALID_DATA_MAX)
    # Ignore times by setting first dimension to True
    valid_mask[0, :, :, :] = True
    return da.where(valid_mask, dask_array, np.nan)


# Linear interpolation of time blocks in a dask array
def interp_time_block(y_block, idx0, idx1, w, valid):
    """
    y_block: np.ndarray of shape (Vb, T_old, Yb, Xb)
    idx0, idx1, w, valid: 1D NumPy arrays of length T_new
    """
    # 1) pull out the two knot‐time slices
    y0 = y_block[:, idx0, ...]  # → (Vb, T_new, Yb, Xb)
    y1 = y_block[:, idx1, ...]

    # 2) build the broadcastable weights
    w_r = w[None, :, None, None]
    omw_r = (1 - w)[None, :, None, None]

    # 3) linear blend
    y_interp = omw_r * y0 + w_r * y1

    # 4) zero‐out (or NaN‐out) anything outside the original time range
    #    here we choose NaN so it’s clear these were out-of-range
    if not np.all(valid):
        # valid==False where x_b is outside [x_a[0], x_a[-1]]
        inv = ~valid
        y_interp[:, inv, :, :] = np.nan

    return y_interp


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
