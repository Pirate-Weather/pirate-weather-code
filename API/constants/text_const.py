# Constants for text generation in Pirate Weather

# Cloud cover thresholds (% as fraction)
CLOUD_COVER_THRESHOLDS = {
    "cloudy": 0.875,
    "mostly_cloudy": 0.625,
    "partly_cloudy": 0.375,
    "mostly_clear": 0.125,
    "clear": 0.0,
}

# Precipitation intensity thresholds (mm/h liquid equivalent)
PRECIP_INTENSITY_THRESHOLDS = {
    "light": 0.4,
    "mid": 2.5,
    "heavy": 10.0,
}

# Snow intensity thresholds (mm/h liquid equivalent)
SNOW_INTENSITY_THRESHOLDS = {
    "light": 0.13,
    "mid": 0.83,
    "heavy": 3.33,
}

# Icon thresholds for precipitation accumulation (mm liquid equivalent)
HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM = 0.2
HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM = 0.02
DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM = 10.0
DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM = 1.0

# Visibility thresholds (meters)
FOG_THRESHOLD_METERS = 1000
MIST_THRESHOLD_METERS = 10000
SMOKE_CONCENTRATION_THRESHOLD_UGM3 = 25
TEMP_DEWPOINT_SPREAD_FOR_FOG = 2
TEMP_DEWPOINT_SPREAD_FOR_MIST = 3

# Wind thresholds (m/s)
WIND_THRESHOLDS = {
    "light": 6.7056,
    "mid": 10.0,
    "heavy": 17.8816,
}

# Other constants
DEFAULT_VISIBILITY = 10000
DEFAULT_POP = 1
DEFAULT_HUMIDITY = 0.5
PRECIP_PROB_THRESHOLD = 0.25

# CAPE thresholds
CAPE_THRESHOLDS = {
    "low": 1000,
    "high": 2500,
}

# Lifted Index threshold
LIFTED_INDEX_THRESHOLD = -4
