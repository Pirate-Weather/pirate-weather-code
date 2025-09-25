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

# Grouped DB constants
DB_CONST = {
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

PRECIP_IDX = {
    "none": 0,
    "snow": 1,
    "ice": 2,
    "sleet": 3,
    "rain": 4,
}

# API versioning and ingest version constants
API_VERSION = "V2.7.7"

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
