"""
Constants for variable clipping
"""

# Grouped clip constants
CLIP_PROB = {"min": 0, "max": 1}
CLIP_TEMP = {"min": 183, "max": 333}
CLIP_HUMIDITY = {"min": 0, "max": 1}
CLIP_PRESSURE = {"min": 800, "max": 1100}
CLIP_WIND = {"min": 0, "max": 120}
CLIP_CLOUD = {"min": 0, "max": 1}
CLIP_UV = {"min": 0, "max": 15}
CLIP_VIS = {"min": 0, "max": 16090}
CLIP_OZONE = {"min": 0}  # max is DATA_POINT_CLIPS["ozone"]
CLIP_FIRE = {"min": 0}  # max is DATA_POINT_CLIPS["fire"]
CLIP_SMOKE = {"min": 0}  # max is DATA_POINT_CLIPS["smoke"]
CLIP_FEELS_LIKE = {"min": 183, "max": 333}
CLIP_GLOBAL = {"min": -1000, "max": 10000}

DATA_POINT_CLIPS = {
    "smoke": 500,
    "ozone": 500,
    "fire": 100,
}
