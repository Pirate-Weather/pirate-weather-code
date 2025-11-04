"""
Constants for the API file
"""

# Grouped solar radiation constants
SOLAR_RAD_CONST = {
    "eccentricity": 0.0167,
    "offset": 93.5365,
    "r": 0.75,
    "S0": 1367,
    "delta_factor": 0.4096,
    "delta_offset": 284,
    "hour_offset": 12,
}

# Grouped solar irradiance constants
SOLAR_IRRADIANCE_CONST = {
    "GSC": 1367,
    "declination": 23.45,
    "am_coeff": 0.14,
    "g0_coeff": 0.033,
}

# Grouped globe temperature constants
GLOBE_TEMP_CONST = {
    "diameter_default": 0.15,
    "emissivity_default": 0.95,
    "factor": 1.5e8,
    "temp_exp": 0.6,
    "diam_exp": 0.4,
    "wind_exp": 0.6,
}

# Grouped WBGT constants
WBGT_CONST = {
    "temp_weight": 0.7,
    "globe_weight": 0.2,
    "wind_weight": 0.1,
    "humidity_weight": 0.3,
}

# Grouped DBZ constants
DBZ_CONST = {
    "rain_a": 200.0,
    "rain_b": 1.6,
    "snow_a": 600.0,
    "snow_b": 2.0,
}

# Grouped apparent temperature constants
APPARENT_TEMP_CONSTS = {
    "humidity_factor": 0.33,
    "wind_factor": 0.70,
    "const": -4.00,
    "exp_a": 17.27,
    "exp_b": 237.7,
    "e_const": 6.105,
}

# Grouped apparent temperature solar constants
APPARENT_TEMP_SOLAR_CONSTS = {
    "humidity_factor": 0.348,
    "wind_factor": 0.70,
    "solar_factor": 0.70,
    "q_factor": 0.10,
    "const": -4.25,
}

MAGNUS_FORMULA_CONSTS = {
    "dew_factor": 17.625,
    "temp_factor": 243.04,
}

PRECIP_IDX = {
    "none": 0,
    "snow": 1,
    "ice": 2,
    "sleet": 3,
    "rain": 4,
}

# API versioning and ingest version constants
# Version scheme is: Major.Minor.Patch
API_VERSION = "V2.8g"

# Command priorities
NICE_PRIORITY = 20

# Generic API constants
MAX_S3_RETRIES = 5
S3_BASE_DELAY = 1
S3_MAX_BANDWIDTH = 100000000
LARGEST_DIR_INIT = -1

# Temperature thresholds
TEMPERATURE_UNITS_THRESH = {"c": 0, "f": 32}
TEMP_THRESHOLD_RAIN_C = 274.15
TEMP_THRESHOLD_SNOW_C = 272.15

# Define rounding rules for all numeric fields
ROUNDING_RULES = {
    # Coordinates and general
    "latitude": 4,
    "longitude": 4,
    "offset": 2,
    "elevation": 0,
    # Precipitation
    "precipIntensity": 4,
    "precipIntensityError": 4,
    "precipIntensityMax": 4,
    "precipProbability": 2,
    "precipAccumulation": 4,
    "rainIntensity": 4,
    "snowIntensity": 4,
    "iceIntensity": 4,
    "rainAccumulation": 4,
    "snowAccumulation": 4,
    "iceAccumulation": 4,
    "liquidAccumulation": 4,
    "sleetAccumulation": 4,
    "currentDayIce": 4,
    "currentDayLiquid": 4,
    "currentDaySnow": 4,
    # Temperature
    "temperature": 2,
    "temperatureHigh": 2,
    "temperatureLow": 2,
    "temperatureMin": 2,
    "temperatureMax": 2,
    "apparentTemperature": 2,
    "apparentTemperatureHigh": 2,
    "apparentTemperatureLow": 2,
    "apparentTemperatureMin": 2,
    "apparentTemperatureMax": 2,
    "dewPoint": 2,
    "feelsLike": 2,
    # Atmospheric
    "humidity": 2,
    "pressure": 2,
    "stationPressure": 2,
    "cloudCover": 2,
    "visibility": 2,
    "ozone": 2,
    "uvIndex": 2,
    # Wind
    "windSpeed": 2,
    "windGust": 2,
    "windBearing": 0,
    # Storm
    "nearestStormDistance": 2,
    "nearestStormBearing": 0,
    # Other
    "moonPhase": 2,
    "smoke": 2,
    "fireIndex": 2,
    "solar": 2,
    "cape": 0,
    "sunriseTime": 0,
    "sunsetTime": 0,
    "moonriseTime": 0,
    "moonsetTime": 0,
    "precipIntensityMaxTime": 0,
    "temperatureHighTime": 0,
    "temperatureLowTime": 0,
    "temperatureMinTime": 0,
    "temperatureMaxTime": 0,
    "apparentTemperatureHighTime": 0,
    "apparentTemperatureLowTime": 0,
    "apparentTemperatureMinTime": 0,
    "apparentTemperatureMaxTime": 0,
    "windGustTime": 0,
    "uvIndexTime": 0,
    "time": 0,
}