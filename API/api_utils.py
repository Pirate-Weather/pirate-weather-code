# %% Script to contain the helper functions as part of the API for Pirate Weather
# Alexander Rey. October 2025
import logging

import numpy as np

from API.constants.api_const import APPARENT_TEMP_CONSTS, APPARENT_TEMP_SOLAR_CONSTS
from API.constants.clip_const import CLIP_TEMP
from API.constants.shared_const import KELVIN_TO_CELSIUS

logger = logging.getLogger(__name__)


def calculate_apparent_temperature(airTemp, humidity, wind):
    """
    Calculates the apparent temperature temperature based on air temperature, wind speed and humidity
    Formula from: https://github.com/breezy-weather/breezy-weather/discussions/1085
    AT = Ta + 0.33 * rh / 100 * 6.105 * exp(17.27 * Ta / (237.7 + Ta)) - 0.70 * ws - 4.00

    Parameters:
    - airTemperature (float): Air temperature
    - humidity (float): Relative humidity
    - windSpeed (float): Wind speed in meters per second

    Returns:
    - float: Apparent temperature
    """

    # Convert air_temp from Kelvin to Celsius for the formula parts that use Celsius
    airTempC = airTemp - KELVIN_TO_CELSIUS

    # Calculate water vapor pressure 'e'
    # Ensure humidity is not 0 for calculation, replace with a small non-zero value if needed
    # The original equation does not guard for zero humidity. If relative_humidity_0_1 is 0, e will be 0.
    e = (
        humidity
        * APPARENT_TEMP_CONSTS["e_const"]
        * np.exp(
            APPARENT_TEMP_CONSTS["exp_a"]
            * airTempC
            / (APPARENT_TEMP_CONSTS["exp_b"] + airTempC)
        )
    )

    # Calculate apparent temperature in Celsius
    apparentTempC = (
        airTempC
        + APPARENT_TEMP_CONSTS["humidity_factor"] * e
        - APPARENT_TEMP_CONSTS["wind_factor"] * wind
        + APPARENT_TEMP_CONSTS["const"]
    )

    # Convert back to Kelvin
    apparentTempK = apparentTempC + KELVIN_TO_CELSIUS

    # Clip between -90 and 60
    return clipLog(
        apparentTempK,
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )


def calculate_apparent_temperature_solar(airTemp, humidity, wind, solar):
    """
    Calculates the apparent temperature temperature based on air temperature, wind speed and humidity and solar radiation
    Formula from: https://github.com/breezy-weather/breezy-weather/discussions/1085
    AT = Ta + 0.348 * rh / 100 * 6.105 * exp(17.27 * Ta / (237.7 + Ta)) - 0.70 * ws + 0.70 * Q / (ws + 10) - 4.25

    Parameters:
    - airTemperature (float): Air temperature
    - humidity (float): Relative humidity
    - windSpeed (float): Wind speed in meters per second
    - solar (float): Solar radiation in W/m^2

    Returns:
    - float: Apparent temperature
    """

    # Convert air_temp from Kelvin to Celsius for the formula parts that use Celsius
    airTempC = airTemp - KELVIN_TO_CELSIUS

    # Calculate water vapor pressure 'e'
    # Ensure humidity is not 0 for calculation, replace with a small non-zero value if needed
    # The original equation does not guard for zero humidity. If relative_humidity_0_1 is 0, e will be 0.
    e = (
        humidity
        * APPARENT_TEMP_SOLAR_CONSTS["e_const"]
        * np.exp(
            APPARENT_TEMP_SOLAR_CONSTS["exp_a"]
            * airTempC
            / (APPARENT_TEMP_SOLAR_CONSTS["exp_b"] + airTempC)
        )
    )

    # Calculate apparent temperature in Celsius
    apparentTempC = (
        airTempC
        + APPARENT_TEMP_SOLAR_CONSTS["humidity_factor"] * e
        - APPARENT_TEMP_SOLAR_CONSTS["wind_factor"] * wind
        + (APPARENT_TEMP_SOLAR_CONSTS["wind_factor"] * solar) / (wind + 10)
        + APPARENT_TEMP_SOLAR_CONSTS["const"]
    )

    # Convert back to Kelvin
    apparentTempK = apparentTempC + KELVIN_TO_CELSIUS

    # Clip between -90 and 60
    return clipLog(
        apparentTempK,
        CLIP_TEMP["min"],
        CLIP_TEMP["max"],
        "Apparent Temperature Current",
    )


def clipLog(data, min, max, name):
    """
    Clip the data between min and max. Log if there is an error
    """

    # Print if the clipping is larger than 25 of the min
    if data.min() < (min - 0.25):
        # Print the data and the index it occurs
        logger.error("Min clipping required for " + name)
        logger.error("Min Value: " + str(data.min()))
        if isinstance(data, np.ndarray):
            logger.error("Min Index: " + str(np.where(data == data.min())))

        # Replace values below the threshold with np.nan
        if np.isscalar(data):
            if data < min:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data < min] = np.nan

    else:
        data = np.clip(data, a_min=min, a_max=None)

    # Same for max
    if data.max() > (max + 0.25):
        logger.error("Max clipping required for " + name)
        logger.error("Max Value: " + str(data.max()))

        # Print the data and the index it occurs
        if isinstance(data, np.ndarray):
            logger.error("Max Index: " + str(np.where(data == data.max())))

        # Replace values above the threshold with np.nan
        if np.isscalar(data):
            if data > max:
                data = np.nan
        else:
            data = np.array(data, dtype=float)
            data[data > max] = np.nan

    else:
        data = np.clip(data, a_min=None, a_max=max)

    return data
