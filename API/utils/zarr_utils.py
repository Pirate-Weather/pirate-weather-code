"""Helper utilities for working with Zarr data."""

import asyncio
from typing import Any

import numpy as np
import zarr


async def get_zarr(store: zarr.hierarchy.Group, x: int, y: int) -> np.ndarray:
    """Return a slice from an opened Zarr store."""

    return store[:, :, x, y]


def arrayInterp(hour_array_grib: np.ndarray, modelData: np.ndarray, modelIndex: int) -> np.ndarray:
    """Interpolate a model variable onto a common time grid."""

    return np.interp(
        hour_array_grib,
        modelData[:, 0],
        modelData[:, modelIndex],
        left=np.nan,
        right=np.nan,
    )


class WeatherParallel:
    """Async helper for reading data from Zarr."""

    async def zarr_read(
        self, model: str, opened_zarr: zarr.hierarchy.Group, x: int, y: int
    ) -> Any:
        """Read a single point from a Zarr store asynchronously."""

        errCount = 0
        while errCount < 4:
            try:
                dataOut = await asyncio.to_thread(lambda: opened_zarr[:, :, y, x].T)
                return dataOut
            except Exception:
                errCount += 1
        return False
