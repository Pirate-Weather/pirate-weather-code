"""Utility subpackage for Pirate Weather."""

from .indices import HourlyIndex
from .time_utils import (
    calculate_globe_temperature,
    calculate_wbgt,
    get_offset,
    rounder,
    solar_irradiance,
    solar_rad,
    unix_to_day_of_year_and_lst,
    tf,
)
from .zarr_utils import WeatherParallel, arrayInterp, get_zarr
from .sync_utils import (
    S3ZipStore,
    add_custom_header,
    download_if_newer,
    find_largest_integer_directory,
    update_zarr_store,
    logger,
)

__all__ = [
    "HourlyIndex",
    "calculate_globe_temperature",
    "calculate_wbgt",
    "get_offset",
    "rounder",
    "solar_irradiance",
    "solar_rad",
    "unix_to_day_of_year_and_lst",
    "tf",
    "WeatherParallel",
    "arrayInterp",
    "get_zarr",
    "S3ZipStore",
    "add_custom_header",
    "download_if_newer",
    "find_largest_integer_directory",
    "update_zarr_store",
    "logger",
]
