# %% Script to contain the helper functions as part of the API for Pirate Weather
# Alexander Rey. October 2025
import logging

import numpy as np

from API.constants.api_const import APPARENT_TEMP_CONSTS, APPARENT_TEMP_SOLAR_CONSTS
from API.constants.clip_const import CLIP_TEMP
from API.constants.shared_const import KELVIN_TO_CELSIUS

logger = logging.getLogger(__name__)


def calculate_apparent_temperature(air_temp, humidity, wind, solar=None):
    """
    Calculates the apparent temperature based on air temperature, wind speed, humidity and solar radiation if provided.

    Parameters:
    - air_temp (float): Air temperature in Celsuis
    - humidity (float): Relative humidity in %
    - wind (float): Wind speed in meters per second

    Returns:
    - float: Apparent temperature in Kelvin
    """

    # Convert air_temp from Kelvin to Celsius for the formula parts that use Celsius
    air_temp_c = air_temp - KELVIN_TO_CELSIUS

    # Calculate water vapor pressure 'e'
    # Ensure humidity is not 0 for calculation, replace with a small non-zero value if needed
    # The original equation does not guard for zero humidity. If relative_humidity_0_1 is 0, e will be 0.
    e = (
        humidity
        * APPARENT_TEMP_CONSTS["e_const"]
        * np.exp(
            APPARENT_TEMP_CONSTS["exp_a"]
            * air_temp_c
            / (APPARENT_TEMP_CONSTS["exp_b"] + air_temp_c)
        )
    )

    if solar is None:
        # Calculate apparent temperature in Celsius
        apparent_temp_c = (
            air_temp_c
            + APPARENT_TEMP_CONSTS["humidity_factor"] * e
            - APPARENT_TEMP_CONSTS["wind_factor"] * wind
            + APPARENT_TEMP_CONSTS["const"]
        )
    else:
        # Calculate the effective solar term 'q' used in the apparent temperature formula.
        # The model's `solar` value is Downward Short-Wave Radiation Flux in W/m^2.
        # `q_factor` scales that irradiance to the empirical Q used in the formula
        # (for example q_factor=0.1 reduces the raw W/m^2 to a smaller effective value).
        # Tuning `q_factor` controls how strongly solar irradiance influences apparent temp.
        q = solar * APPARENT_TEMP_SOLAR_CONSTS["q_factor"]

        # Calculate apparent temperature in Celsius using solar radiation
        apparent_temp_c = (
            air_temp_c
            + APPARENT_TEMP_SOLAR_CONSTS["humidity_factor"] * e
            - APPARENT_TEMP_SOLAR_CONSTS["wind_factor"] * wind
            + (APPARENT_TEMP_SOLAR_CONSTS["solar_factor"] * q) / (wind + 10)
            + APPARENT_TEMP_SOLAR_CONSTS["const"]
        )

    # Convert back to Kelvin
    apparent_temp_k = apparent_temp_c + KELVIN_TO_CELSIUS

    # Clip between -90 and 60
    return clipLog(
        apparent_temp_k,
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )


def clipLog(data, min_val, max_val, name):
    """
    Clip the data between min and max. Log if there is an error
    """

    # Print if the clipping is larger than 25 of the min
    if data.min() < (min_val - 0.25):
        # Print the data and the index it occurs
        logger.error("Min clipping required for " + name)
        logger.error("Min Value: " + str(data.min()))
        if isinstance(data, np.ndarray):
            logger.error("Min Index: " + str(np.where(data == data.min())))

        # Replace values below the threshold with np.nan
        if np.isscalar(data):
            if data < min_val:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data < min_val] = np.nan

    else:
        data = np.clip(data, a_min=min_val, a_max=None)

    # Same for max
    if data.max() > (max_val + 0.25):
        logger.error("Max clipping required for " + name)
        logger.error("Max Value: " + str(data.max()))

        # Print the data and the index it occurs
        if isinstance(data, np.ndarray):
            logger.error("Max Index: " + str(np.where(data == data.max())))

        # Replace values above the threshold with np.nan
        if np.isscalar(data):
            if data > max_val:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data > max_val] = np.nan

    else:
        data = np.clip(data, a_min=None, a_max=max_val)

    return data
