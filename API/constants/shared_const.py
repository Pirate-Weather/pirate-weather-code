"""
Shared constants
"""

import numpy as np

# Invalid data
MISSING_DATA = np.nan

# Minimum reflectivity threshold (dBZ)
REFC_THRESHOLD = 5.0

# Ingest version
INGEST_VERSION_STR = "v30"

# Convert Kelvin to Celsius
KELVIN_TO_CELSIUS = 273.15

HISTORY_PERIODS = {
    "NBM": 48,
    "HRRR": 48,
    "HRRR_6H": 48,
    "GFS": 288,  # GFS has a 12-day history, allowing 10 days of local retrievals. Beyond that is Google ERA5
    "AIGFS": 48,
    "AIGEFS": 48,
    "GEFS": 48,
    "ECMWF": 48,
    "NBM_Fire": 48,
    "DWD_MOSMIX": 48,  # History period offset (like other models)
    "ECMWF_AIFS": 48,
    "AQM": 24,  # Store last 24 hours of AQM data to calculate US EPA AQI values
    "RDAQA": 24,  # Store last 24 hours of RDAQA data to calculate US EPA AQI values
}  # Note: SILAM and IS4FIRES are not included since they are behind real-time and include observations in their forecasts
