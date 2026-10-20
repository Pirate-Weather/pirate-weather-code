"""
Shared constants
"""

import numpy as np

# Invalid data
MISSING_DATA = np.nan

# Minimum reflectivity threshold (dBZ)
REFC_THRESHOLD = 5.0

# An ensemble member counts towards precipitation probability (GEFS, AIGEFS,
# GEPS, REPS) when its hourly precipitation rate exceeds this (mm/h).
ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H = 0.1

# Ingest version
INGEST_VERSION_STR = "v31"

# Convert Kelvin to Celsius
KELVIN_TO_CELSIUS = 273.15

# Hours of short-term history stored ahead of the forecast. Only these sources
# keep it; every other model store starts at its first forecast lead.
HISTORY_PERIODS = {
    # GFS has a 12-day history, allowing 10 days of local retrievals.
    # Beyond that is Google ERA5.
    "GFS": 288,
    "ECMWF": 48,
    "REPS": 48,
    "SILAM": 48,  # Store last 48 hours of SILAM data for historic AQI lookbacks
    # URMA has roughly 12 days of history, allowing 10 days of local retrievals.
    # URMA data has a 6 hour delay, so account for that in the history period to keep consistency with GFS.
    "URMA": 282,
}

# First forecast lead (hours) of the forecast-only model stores. The API recovers
# the model run time as the first stored valid time minus this lead.
FIRST_FORECAST_LEAD_HOURS = {
    "HRRR": 1,
    "NBM": 1,
    "GEFS": 3,
    "HRDPS": 1,
    "GDPS": 1,
    "GEPS": 3,
    "DWD_MOSMIX": 1,
    "AIGFS": 6,
    "AIGEFS": 6,
    "ECMWF_AIFS": 6,
    "RAQDPS": 0,
}
