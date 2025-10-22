"""
Shared constants
"""

# Invalid data
MISSING_DATA = -999

# Minimum reflectivity threshold (dBZ)
REFC_THRESHOLD = 5.0

# Ingest version
INGEST_VERSION_STR = "v29"

# Convert Kelvin to Celsius
KELVIN_TO_CELSIUS = 273.15

HISTORY_PERIODS = {
    "NBM": 48,
    "HRRR": 48,
    "HRRR_6H": 48,
    "GFS": 288,  # GFS has a 12-day history, allowing 10 days of local retrievals. Beyond that is Google ERA5
    "GEFS": 48,
    "ECMWF": 48,
    "NBM_Fire": 48,
}
