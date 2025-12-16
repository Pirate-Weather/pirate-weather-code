"""Helper functions for determining data source priority based on location."""

from API.utils.geo import is_in_north_america


def get_source_order_for_variable(lat, lon, has_ecmwf=True):
    """
    Get the priority order for data sources based on location.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        has_ecmwf (bool): Whether ECMWF has data for this variable.

    Returns:
        tuple: Ordered tuple of source names for priority selection.

    Priority rules:
        - North America (all variables):
          RTMA_RU > HRRR_SUBH > NBM > HRRR > ECMWF > GFS > DWD_MOSMIX > ERA5
        - Rest of world (all variables):
          RTMA_RU > HRRR_SUBH > NBM > HRRR > DWD_MOSMIX > ECMWF > GFS > ERA5
    """
    in_north_america = is_in_north_america(lat, lon)

    if in_north_america:
        # North America: DWD MOSMIX below GFS
        if has_ecmwf:
            return (
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "ecmwf_ifs",
                "gfs",
                "dwd_mosmix",
                "era5",
            )
        else:
            # For variables where ECMWF doesn't have data
            return (
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "gfs",
                "dwd_mosmix",
                "era5",
            )
    else:
        # Rest of world: DWD MOSMIX above ECMWF and GFS
        if has_ecmwf:
            return (
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "dwd_mosmix",
                "ecmwf_ifs",
                "gfs",
                "era5",
            )
        else:
            # For variables where ECMWF doesn't have data
            return (
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "dwd_mosmix",
                "gfs",
                "era5",
            )


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
