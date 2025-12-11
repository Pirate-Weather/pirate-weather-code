"""Helper functions for determining data source priority based on location."""

from API.utils.geo import is_in_north_america


def get_source_order_for_variable(lat, lon, variable_type="standard"):
    """
    Get the priority order for data sources based on location and variable type.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        variable_type (str): Type of variable. Options:
            - "standard": Variables where both ECMWF and DWD MOSMIX have data
              (temp, dew, pressure, wind, bearing, cloud, accum)
            - "dwd_only": Variables where DWD MOSMIX has data but ECMWF doesn't
              (humidity, gust, vis, solar)

    Returns:
        tuple: Ordered tuple of source names for priority selection.

    Priority rules:
        - Standard variables in North America:
          RTMA_RU > HRRR_SUBH > NBM > HRRR > ECMWF > DWD_MOSMIX > GFS > ERA5
        - Standard variables elsewhere:
          RTMA_RU > HRRR_SUBH > NBM > HRRR > DWD_MOSMIX > ECMWF > GFS > ERA5
        - DWD-only variables in North America:
          RTMA_RU > HRRR_SUBH > NBM > HRRR > GFS > DWD_MOSMIX > ERA5
        - DWD-only variables elsewhere:
          RTMA_RU > HRRR_SUBH > NBM > HRRR > DWD_MOSMIX > GFS > ERA5
    """
    in_north_america = is_in_north_america(lat, lon)

    if variable_type == "dwd_only":
        # For variables where DWD MOSMIX has data but ECMWF doesn't
        # In North America, DWD MOSMIX should be below GFS
        if in_north_america:
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
            return (
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "dwd_mosmix",
                "gfs",
                "era5",
            )
    else:
        # Standard variables where both ECMWF and DWD MOSMIX have data
        if in_north_america:
            return (
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "ecmwf_ifs",
                "dwd_mosmix",
                "gfs",
                "era5",
            )
        else:
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


def should_ecmwf_precede_dwd(lat, lon):
    """
    Determine if ECMWF should be prioritized over DWD MOSMIX.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        bool: True if ECMWF should precede DWD MOSMIX, False otherwise.
    """
    return is_in_north_america(lat, lon)


def should_gfs_precede_dwd_for_var(lat, lon, variable_type):
    """
    Determine if GFS should be prioritized over DWD MOSMIX for a specific variable.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        variable_type (str): Type of variable ("standard" or "dwd_only").

    Returns:
        bool: True if GFS should precede DWD MOSMIX for this variable, False otherwise.
    """
    # For DWD-only variables in North America, GFS should precede DWD MOSMIX
    return is_in_north_america(lat, lon) and variable_type == "dwd_only"
