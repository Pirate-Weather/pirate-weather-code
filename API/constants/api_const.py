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

PRECIP_NOISE_THRESHOLD_MMH = (
    0.01  # Threshold in mm/h to filter out noise in precipitation intensity
)

# API versioning and ingest version constants
# Version scheme is: Major.Minor.Patch
API_VERSION = "V2.8.4a"

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

# Zarr read retry constants
MAX_ZARR_READ_RETRIES = 4

# Grouped coordinate constants
COORDINATE_CONST = {
    "longitude_min": -180,
    "longitude_max": 360,
    "latitude_min": -90,
    "latitude_max": 90,
    "longitude_offset": 180,
}

# Grouped time machine constants
TIME_MACHINE_CONST = {
    "threshold_hours": 25,
    "very_negative_threshold": -100000,
}

# Filename constants
FILENAME_TIMESTAMP_SLICE_LENGTH = 12

# Grouped unit conversion constants
UNIT_CONVERSION_CONST = {
    "seconds_to_minutes": 60,
    "hours_to_minutes": 60,
    "seconds_to_hours": 3600,
    "longitude_to_hours": 15,
}

# Grouped Etopo resolution constants
ETOPO_CONST = {
    "lat_resolution": 0.01666667,
    "lon_resolution": 0.01666667,
}

# Grouped Lambert projection constants
LAMBERT_CONST = {
    "pi_factor": 0.25,
    "half_pi_factor": 0.5,
}

# Grouped DBZ conversion constants
DBZ_CONVERSION_CONST = {
    "divisor": 10.0,
    "min_value": 0.0,
    "exponent": 1.0,
}

# Grouped conversion factor constants
CONVERSION_FACTORS = {
    "humidity_percentage": 100.0,
    "pressure_to_hpa": 100,
    "cloud_cover_percentage": 0.01,
    "joules_to_watts": 3600,
    "ozone_to_dobson": 46696,
}

# Grouped RTMA_RU visibility constants
RTMA_RU_VIS_CONST = {
    "max_threshold": 15999,
    "converted_value": 16090,
}

# Grouped UV index constants
UV_INDEX_CONST = {
    "gfs_factor": 18.9,
    "gfs_multiplier": 0.025,
    "era5_factor": 40,
    "era5_multiplier": 0.0025,
}

# Default rounding interval (minutes)
DEFAULT_ROUNDING_INTERVAL = 60

# Grouped solar calculation constants
SOLAR_CALC_CONST = {
    "day_of_year_base": 284,
    "degrees_per_year": 360,
    "days_per_year": 365,
    "hour_factor": 15,
    "hour_offset": 12,
}

# WBGT temperature units
WBGT_PERCENTAGE_DIVISOR = 100.0

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
    "rainIntensityMax": 4,
    "snowIntensity": 4,
    "snowIntensityMax": 4,
    "iceIntensity": 4,
    "iceIntensityMax": 4,
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
    "smokeMax": 2,
    "fireIndex": 2,
    "fireIndexMax": 2,
    "solar": 2,
    "solarMax": 2,
    "cape": 0,
    "capeMax": 0,
    "sunriseTime": 0,
    "sunsetTime": 0,
    "moonriseTime": 0,
    "moonsetTime": 0,
    "dawnTime": 0,
    "duskTime": 0,
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
    "fireIndexMaxTime": 0,
    "solarMaxTime": 0,
    "capeMaxTime": 0,
    "smokeMaxTime": 0,
}
