"""
Constants for variable clipping
"""

# Grouped clip constants
CLIP_PROB = {"min": 0, "max": 1}
CLIP_TEMP = {"min": 183, "max": 333}
CLIP_HUMIDITY = {"min": 0, "max": 1}
CLIP_PRESSURE = {"min": 80000, "max": 110000}  # Pascals (800-1100 hPa)
CLIP_WIND = {"min": 0, "max": 120}
CLIP_CLOUD = {"min": 0, "max": 1}
CLIP_UV = {"min": 0, "max": 15}
CLIP_VIS = {"min": 0, "max": 16090}
CLIP_OZONE = {"min": 0, "max": 500}
CLIP_FIRE = {"min": 0, "max": 100}
CLIP_SMOKE = {"min": 0, "max": 500}
CLIP_FEELS_LIKE = {"min": 183, "max": 333}
CLIP_GLOBAL = {"min": -1000, "max": 10000}
CLIP_SOLAR = {"min": 0, "max": 10000}
CLIP_CAPE = {"min": 0, "max": 10000}
