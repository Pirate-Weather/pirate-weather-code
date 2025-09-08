"""
Constants for timemachine.py
"""

# Apparent temperature and wind chill constants
APPARENT_TEMP_WINDCHILL_CONST = {
    "threshold_k": 283.15,  # 10C in K
    "windchill_1": 13.12,
    "windchill_2": 0.6215,
    "windchill_3": 11.37,
    "windchill_4": 0.3965,
    "windchill_exp": 0.16,
    "windchill_kph_conv": 3.6,
    "apparent_temp_const": 5 / 9,
    "apparent_temp_2": 6.11,
    "apparent_temp_3": 5417.7530,
    "apparent_temp_4": 273.16,
    "apparent_temp_5": 10,
}

ICE_ACCUMULATION = 0
DAILY_PRECIP_THRESHOLD = 0.5
