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

# Physical constants for atmospheric calculations
GRAVITY = 9.80665  # Standard gravity (m/s²)
WATER_VAPOR_GAS_CONSTANT_RATIO = 0.622  # Ratio of gas constants (R_d/R_v)

# Bolton's formula constants for saturation vapor pressure
BOLTON_CONST = {
    "base_pressure": 6.112,  # Base saturation vapor pressure (hPa)
    "temp_coeff": 17.67,  # Temperature coefficient
    "temp_offset": 243.5,  # Temperature offset (°C)
}

# Cloud formation constants
CLOUD_RH_CRITICAL = 0.75  # Critical relative humidity threshold for cloud formation
CLOUD_RH_EXPONENT = 2  # Exponent for cloud fraction formula

# Freezing level default bounds
FREEZING_LEVEL_SURFACE = 0.0  # Surface level (m) when all temps below freezing
FREEZING_LEVEL_HIGH = 15000.0  # High altitude (m) when all temps above freezing
FREEZING_LEVEL_TEMP_TOLERANCE = (
    0.01  # Temperature difference tolerance (K) for interpolation
)

HISTORY_PERIODS = {
    "NBM": 48,
    "HRRR": 48,
    "HRRR_6H": 48,
    "GFS": 288,  # GFS has a 12-day history, allowing 10 days of local retrievals. Beyond that is Google ERA5
    "AIGFS": 48,
    "AIGEFS": 24,  # Only 24 hours is stored on NOMADS
    "GEFS": 48,
    "ECMWF": 48,
    "NBM_Fire": 48,
    "DWD_MOSMIX": 48,  # History period offset (like other models)
    "ECMWF_AIFS": 48,
    "HGEFS": 12,  # HGEFS has a shorter history period on NOMADS
}
