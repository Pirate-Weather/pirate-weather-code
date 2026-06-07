# Constants for text generation in Pirate Weather

# Cloud cover thresholds (% as fraction)
CLOUD_COVER_THRESHOLDS = {
    "cloudy": 0.875,
    "mostly_cloudy": 0.625,
    "partly_cloudy": 0.375,
    "mostly_clear": 0.125,
    "clear": 0.0,
}
# Cloud cover thresholds for daily text generation
CLOUD_COVER_DAILY_THRESHOLDS = {
    "cloudy": 1.0,
    "mostly_cloudy": 0.75,
    "partly_cloudy": 0.50,
    "mostly_clear": 0.25,
    "clear": 0.0,
}

# Precipitation intensity thresholds (mm/h liquid equivalent)
PRECIP_INTENSITY_THRESHOLDS = {
    "light": 0.4,
    "mid": 2.5,
    "heavy": 10.0,
}

# Snow intensity thresholds (mm/h of snow)
SNOW_INTENSITY_THRESHOLDS = {
    "light": 1.30,
    "mid": 8.30,
    "heavy": 33.30,
}

# Icon thresholds for precipitation accumulation
HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM = 0.2  # In snow units (mm of snow)
HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM = 0.02  # (mm liquid)
DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM = 5.0  # In snow units (mm of snow)
DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM = 1.0  # (mm liquid)

# Text thresholds for precipitation accumulation (only rain/sleet)
DAILY_PRECIP_ACCUM_TEXT_THRESHOLD_MM = 0.2  # (mm rain/sleet)

# Visibility thresholds (metres)
FOG_THRESHOLD_METERS = 1000
MIST_THRESHOLD_METERS = 5000
SMOKE_CONCENTRATION_THRESHOLD_UGM3 = 25
TEMP_DEWPOINT_SPREAD_FOR_FOG = 2
TEMP_DEWPOINT_SPREAD_FOR_MIST = 3

# Wind thresholds (m/s)
WIND_THRESHOLDS = {
    "light": 6.7056,
    "mid": 10.0,
    "heavy": 17.8816,
}

# Other constants
DEFAULT_VISIBILITY = 10000
DEFAULT_POP = 1
DEFAULT_HUMIDITY = 0.5
PRECIP_PROB_THRESHOLD = 0.25

# CAPE thresholds (J/kg)
CAPE_THRESHOLDS = {
    "low": 1250,
    "high": 2500,
}

# Lifted Index thresholds (K) — more negative = more unstable
LI_THRESHOLDS = {
    "possible": -3,    # Marginally unstable; isolated thunderstorms possible
    "thunderstorm": -6,  # Very unstable; thunderstorms likely
}

# Convective Inhibition thresholds (J/kg, negative by convention)
CIN_THRESHOLDS = {
    "moderate": -50,    # Moderate cap; reduces but does not prevent convection
    "suppressed": -200,  # Strong cap; suppresses convection
}

# Vertical velocity (omega) thresholds (Pa/s) — negative = upward motion
VV_THRESHOLDS = {
    "strong_upward": -0.5,  # Forced ascent that can confirm a marginal storm signal
}

# K Index thresholds (°C / K equivalent)
# K = (T850 − T500) + Td850 − (T700 − Td700)
KI_THRESHOLDS = {
    "possible": 20,    # Isolated thunderstorms possible
    "thunderstorm": 35,  # Numerous thunderstorms likely
}

# Maximum temperature–dewpoint spread (°C) above which the atmosphere is
# considered too dry for thunderstorms. Only applied when temperature > 0 °C
# so that thundersnow can still be detected at sub-zero temperatures.
MAX_DEWPOINT_DEPRESSION_FOR_STORM = 20

# Temperature thresholds
WARM_TEMPERATURE_THRESHOLD = {"c": 20, "f": 68}
LIQUID_DENSITY_CONVERSION = 1000

# Snow density and snow height calculation constants
SNOW_DENSITY_CONST = {
    "max_kelvin": 275.65,
    "low_temp_threshold": 260.15,
    "density_base": 500,
    "low_temp_exp_coeff": 0.904,
    "low_temp_exp_factor": 0.008,
    "wind_exp": 1.7,
    "high_temp_exp_coeff": 0.951,
    "high_temp_exp_factor": 0.008,
    "high_temp_power_base": 278.15,
    "high_temp_power_exp": -1.15,
    "high_temp_exp_factor2": 1.4,
    "min_density": 50,
}

# Tolerance to show less-than for snow accumulations in mm
LESS_THAN_TOLERANCE = 50
