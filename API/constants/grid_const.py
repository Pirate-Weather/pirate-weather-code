# US bounding box for cull
US_BOUNDING_BOX = {
    "top": 49.3457868,  # north lat
    "left": -124.7844079,  # west long
    "right": -66.9513812,  # east long
    "bottom": 24.7433195,  # south lat
}
# Grid boundary constants for model index checks
HRRR_X_MIN = 1
HRRR_Y_MIN = 1
HRRR_X_MAX = 1799
HRRR_Y_MAX = 1059

NBM_X_MIN = 1
NBM_Y_MIN = 1
NBM_X_MAX = 2344
NBM_Y_MAX = 1596

# RTMA Rapid Update grid constants (approximate 2.5 km resolution)
# These are nominal bounds used for nearest-index lookup when exact coordinates
# are not present in the zarr store. If exact coordinates are present in the
# zarr, prefer reading them from the file instead of these constants.
RTMA_LAT_MAX = 55.0
RTMA_LAT_MIN = 20.0
RTMA_LON_MIN = -130.0
RTMA_LON_MAX = -60.0
RTMA_DEG_STEP = 0.0225  # ~2.5 km spacing in degrees
