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
    # CFSv2: the four members of one 00Z cycle (ingest/CFS_Local_Ingest.py)
    "CFS": 4,
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
    "CFS": 6,
}

# When each forecast source normally publishes its next run, for version 2
# flags.nextUpdate (#668): {flags.sources name: (cycle interval, hours from a
# run time to its publish)}, so a run's successor is expected at
# run time + interval + delay. The run time is what flags.sourceTimes shows: the
# model cycle, except for the analyses and nowcasts (rtma_ru: the analysis valid
# time; librewxr: the radar frame the nowcast starts from).
# Delay = the ingest's start after its cycle (the Ofelia schedule in
# pirate-compose_oph.local.yml, mirrored by the RBX CronJobs) + its run time,
# from the publish stamps of the 2026-10-31 to 11-04 local stack runs (store
# version stamp minus run time) and the durations in TODO.md's Done log. Every
# ingest takes the newest complete cycle, so a late one is caught at its next
# scheduled start. This table must move with the schedules.
PUBLISH_SCHEDULE_HOURS = {
    # Hourly HRRR (and its 15-minute store) start at :45 and take the cycle
    # from 1 h 45 min before: published 1.78-1.79 h after the cycle.
    "hrrr_0-18": (1, 1.8),
    "hrrrsubh": (1, 1.8),
    # HRRR 18-48 h starts at 02:30/08:30/14:30/20:30 for the 00/06/12/18Z
    # cycles: published 2.54 h after.
    "hrrr_18-48": (6, 2.6),
    # NBM starts at :45 with the same hour's cycle: published 1.2 h after.
    "nbm": (1, 1.25),
    # RTMA-RU starts at :10/:25/:40/:55 with the analysis 25 min before:
    # published 25 min after the valid time.
    "rtma_ru": (0.25, 0.42),
    # LibreWXR publishes from its nowcast hook every 10 minutes: one sample,
    # 3.7 min after the radar frame time.
    "librewxr": (10 / 60, 5 / 60),
    # DWD MOSMIX starts on the hour with the last hour's issue: 1.1-1.17 h.
    "dwd_mosmix": (1, 1.25),
    # NCEP global: GFS and AIGFS start 5 h after the cycle (GFS published at
    # 5.44 h, AIGFS 5.09 h), GEFS 7 h (7.24-7.29 h), AIGEFS at +8 h 20 min
    # (8.72-8.73 h).
    "gfs": (6, 5.5),
    "aigfs": (6, 5.25),
    "gefs": (6, 7.25),
    "aigefs": (6, 8.75),
    # ECMWF IFS starts at 10:30/22:30 for 00/12Z (the 0.1 degree 240 h file is
    # on AWS near +10 h) and takes 2,100-3,500 s. AIFS starts at +8 h: 8.17 h.
    "ecmwf_ifs": (12, 11.5),
    "ecmwf_aifs": (6, 8.25),
    # MSC: HRDPS and REPS start at +4 h (published 4.26 h and 4.18 h), GDPS
    # +5 h (5.59 h; 1,450 s since the parallel downloads), GEPS +7 h (7.16 h),
    # RAQDPS at +4 h 15 min.
    "hrdps": (6, 4.25),
    "reps": (6, 4.25),
    "gdps": (12, 5.5),
    "geps": (12, 7.25),
    "raqdps": (12, 4.5),
    # FMI posts the 00Z SILAM run near 12:55 the next day; the ingest starts at
    # 14:00 (+38 h), and at 19:00 for the days it is posted near 17:50.
    "silam": (24, 38.25),
    # CFS (future days past the GFS store only) takes the 00Z cycle once a day
    # at 12:30 UTC and publishes about 3 min later. The 15:30 retry for a late
    # cycle is not modelled: an overdue run rolls forward to the next 12:30.
    "cfs": (24, 12.55),
    # RRFS (include=rrfs) posting on AWS is irregular, and each ingest takes
    # the newest cycle whose last lead exists (hourly stores checked at :50,
    # the 6-hourly 18-83 h stores at :20). No measured runs; estimates.
    "rrfs_0-18": (1, 2.0),
    "rrfs_na_0-18": (1, 2.0),
    "rrfssubh": (1, 2.0),
    "rrfs_18-83": (6, 3.5),
    "rrfs_na_18-83": (6, 3.5),
}
