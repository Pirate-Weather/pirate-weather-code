"""Helper functions for determining data source priority based on location."""

from API.utils.geo import is_in_north_america


def should_gfs_precede_dwd(lat, lon):
    """
    Determine if GFS should be prioritized over DWD MOSMIX.

    In North America, GFS should always precede DWD MOSMIX.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        bool: True if GFS should precede DWD MOSMIX, False otherwise.
    """
    return is_in_north_america(lat, lon)
