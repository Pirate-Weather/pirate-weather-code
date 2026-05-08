import asyncio
import datetime
import logging
import numpy as np

from API.constants.api_const import MAX_ZARR_READ_RETRIES
from API.processing.utils import TIMING, has_interior_nan_holes, _interp_row

logger = logging.getLogger("pirate-weather-api")

class WeatherParallel(object):
    """Helper class for parallel zarr reading operations."""

    def __init__(self, loc_tag: str = "") -> None:
        self.loc_tag = loc_tag

    async def zarr_read(self, model, opened_zarr, x, y):
        if TIMING:
            logger.debug(f"### {model} Reading!")
            logger.debug(datetime.datetime.now(datetime.UTC).replace(tzinfo=None))

        err_count = 0
        data_out = False
        # Try to read Zarr file
        while err_count < MAX_ZARR_READ_RETRIES:
            try:
                data_out = await asyncio.to_thread(lambda: opened_zarr[:, :, y, x].T)

                # Check for missing/ bad data and interpolate
                # This should not occur, but good to have a fallback
                has_missing_data, missing_row = has_interior_nan_holes(data_out.T)
                if has_missing_data:
                    logger.warning(
                        f"### {model} Interpolating missing data (row {missing_row})!"
                    )

                    # Print the location of the missing data
                    if TIMING:
                        logger.debug(
                            f"### {model} Missing data at: {np.argwhere(np.isnan(data_out))}"
                        )

                    data_out = np.apply_along_axis(_interp_row, 0, data_out)

                if TIMING:
                    logger.debug(f"### {model} Done!")
                    logger.debug(
                        datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                    )
                return data_out

            except Exception:
                logger.exception("### %s Failure! %s", model, self.loc_tag)
                err_count += 1

        logger.error("### %s Failure! %s", model, self.loc_tag)
        data_out = False
        return data_out


async def get_zarr(store, X, Y):
    """Asynchronously retrieve zarr data at given coordinates.

    Args:
        store: Zarr store to read from
        X: X coordinate
        Y: Y coordinate

    Returns:
        Zarr data at the specified coordinates
    """
    return store[:, :, X, Y]
