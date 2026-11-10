"""
Shared constants
"""

import numpy as np

# Invalid data
MISSING_DATA = np.nan

# Minimum reflectivity threshold (dBZ)
REFC_THRESHOLD = 5.0

# LibreWXR radar composite grid (ingest/LibreWXR_Local_Ingest.py): a global
# regular lat/lon grid at the RRQPE native-downsampled resolution, rows from
# north to south and columns from west to east, cell centres half a cell in.
LIBREWXR_GRID_RESOLUTION_DEG = 0.04
LIBREWXR_GRID_NORTH = 90.0
LIBREWXR_GRID_WEST = -180.0
LIBREWXR_GRID_HEIGHT = 4500
LIBREWXR_GRID_WIDTH = 9000
# A nowcast whose radar frame is older than this is not published or served:
# the pipeline has stalled, so the minutely block falls back to the models.
LIBREWXR_MAX_NOWCAST_AGE_SECONDS = 3600

# An ensemble member counts towards precipitation probability (GEFS, AIGEFS,
# GEPS, REPS) when its hourly precipitation rate exceeds this (mm/h).
ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H = 0.1

# Per-hour ensemble member bitmasks (Precipitation_Member_Mask_<word>): bit m of
# word w is set when member 24 w + m exceeds ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H.
# The words are float32 integers, exact below 2^24, so 24 members per word. The
# API ORs the words over a day's hours and popcounts them to get the share of
# members with precipitation at any hour of the day, which the max of the hourly
# probabilities understates when members disagree on timing.
MEMBER_MASK_BITS_PER_WORD = 24
MEMBER_MASK_VAR_PREFIX = "Precipitation_Member_Mask_"
# Members per ensemble. The API divides a popcount by these, and the ingests fail
# before publishing when a run has a different count. The AI ensembles share the
# GEFS / ECMWF columns under include=aimodels, so their counts must match.
ENSEMBLE_MEMBER_COUNTS = {
    "GEFS": 30,
    "AIGEFS": 30,
    "GEPS": 21,
    "REPS": 21,
    # The open-data enfo files hold the 50 perturbed members; the control
    # member is not among them (runs of 2026-11-09/10 had 50)
    "ECMWF_IFS": 50,
    "ECMWF_AIFS": 50,
}


def member_mask_var_names(member_count: int) -> tuple[str, ...]:
    """Store variable names holding the member bitmask words for ``member_count``
    members: ceil(member_count / 24) words, numbered from 0."""
    word_count = -(-member_count // MEMBER_MASK_BITS_PER_WORD)
    return tuple(f"{MEMBER_MASK_VAR_PREFIX}{word}" for word in range(word_count))


# Ingest version
INGEST_VERSION_STR = "v32"

# Convert Kelvin to Celsius
KELVIN_TO_CELSIUS = 273.15

# Hours of short-term history stored ahead of the forecast. Only these sources
# keep it; every other model store starts at its first forecast lead.
HISTORY_PERIODS = {
    # GFS has a 12-day history, allowing 10 days of local retrievals.
    # Beyond that is Google ERA5.
    "GFS": 288,
    "ECMWF": 48,
    "SILAM": 48,  # Store last 48 hours of SILAM data for historic AQI lookbacks
    # URMA has roughly 12 days of history, allowing 10 days of local retrievals.
    # URMA data has a 6 hour delay, so account for that in the history period to keep consistency with GFS.
    "URMA": 282,
}

# First forecast lead (hours) of the forecast-only model stores. The API recovers
# the model run time as the first stored valid time minus this lead.
FIRST_FORECAST_LEAD_HOURS = {
    "HRRR": 1,
    "RRFS": 1,
    "RRFS_6H": 18,
    "RRFS_NA": 1,
    "RRFS_NA_6H": 18,
    # RRFS sub-hourly starts at the first 15-minute step
    "RRFS_SubH": 0.25,
    "NBM": 1,
    "GEFS": 3,
    "HRDPS": 1,
    "GDPS": 1,
    "GEPS": 3,
    "REPS": 3,
    "DWD_MOSMIX": 1,
    "AIGFS": 6,
    "AIGEFS": 6,
    "ECMWF_AIFS": 6,
    "RAQDPS": 0,
}
