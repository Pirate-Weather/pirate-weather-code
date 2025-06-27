from enum import IntEnum


class GFSFields(IntEnum):
    TIME = 0
    VIS_SURFACE = 1
    GUST_SURFACE = 2
    PRES_SURFACE = 3
    TMP_2M = 4
    DPT_2M = 5
    RH_2M = 6
    APTMP_2M = 7
    UGRD_10M = 8
    VGRD_10M = 9
    PRATE_SURFACE = 10
    APCP_SURFACE = 11
    CSNOW_SURFACE = 12
    CICEP_SURFACE = 13
    CFRZR_SURFACE = 14
    CRAIN_SURFACE = 15
    TOZNE = 16
    TCDC = 17
    DUVB = 18
    STORM_DISTANCE = 19
    STORM_DIRECTION = 20


class NBMFields(IntEnum):
    TIME = 0
    GUST_10M = 1
    TMP_2M = 2
    APTMP_2M = 3
    DPT_2M = 4
    RH_2M = 5
    WIND_10M = 6
    WDIR_10M = 7
    APCP_SURFACE = 8
    TCDC_SURFACE = 9
    VIS_SURFACE = 10
    PWTHER_RESERVED = 11
    PPROB = 12
    PACCUM = 13
    PTYPE_1 = 14
    PTYPE_3 = 15
    PTYPE_5 = 16
    PTYPE_8 = 17


class HRRRFields(IntEnum):
    TIME = 0
    VIS_SURFACE = 1
    GUST_SURFACE = 2
    PRESSURE = 3
    TMP_2M = 4
    DPT_2M = 5
    RH_2M = 6
    UGRD_10M = 7
    VGRD_10M = 8
    PRATE_SURFACE = 9
    APCP_SURFACE = 10
    CSNOW_SURFACE = 11
    CICEP_SURFACE = 12
    CFRZR_SURFACE = 13
    CRAIN_SURFACE = 14
    TCDC = 15
    MASSDEN_8M = 16


class GEFSFields(IntEnum):
    TIME = 0
    PRECIP_PROB = 1
    APCP_MEAN = 2
    APCP_STDDEV = 3
    CSNOW_PROB = 4
    CICEP_PROB = 5
    CFRZR_PROB = 6
    CRAIN_PROB = 7


class NBMFireFields(IntEnum):
    """Indexes for NBM fire-weather data."""

    TIME = 0
    FIRE_INDEX = 1


class MinuteFields(IntEnum):
    """Indexes for minute-level precipitation data arrays."""

    TIME = 0
    INTENSITY = 1
    PROBABILITY = 2
    INTENSITY_ERROR = 3


class HourFields(IntEnum):
    """Indexes for hourly interpolation arrays."""

    TIME = 0
    PTYPE = 1
    PRECIP_INTENSITY = 2
    PRECIP_PROB = 3
    INTENSITY_ERROR = 4
    TEMPERATURE = 5
    APPARENT_TEMP = 6
    DEW_POINT = 7
    HUMIDITY = 8
    PRESSURE = 9
    WIND_SPEED = 10
    WIND_GUST = 11
    WIND_BEARING = 12
    CLOUD_COVER = 13
    UV_INDEX = 14
    VISIBILITY = 15
    OZONE = 16
    PRECIP_ACCUM = 17
    STORM_DISTANCE = 18
    STORM_BEARING = 19
    SMOKE = 20
    RAIN_ACCUM = 21
    SNOW_ACCUM = 22
    ICE_ACCUM = 23
    FIRE_INDEX = 24
    FEELS_LIKE = 25


class CurrentFields(IntEnum):
    """Indexes for the current conditions array."""

    TIME = 0
    PRECIP_INTENSITY = 1
    PRECIP_PROBABILITY = 2
    INTENSITY_ERROR = 3
    TEMPERATURE = 4
    APPARENT_TEMP = 5
    DEW_POINT = 6
    HUMIDITY = 7
    PRESSURE = 8
    WIND_SPEED = 9
    WIND_GUST = 10
    WIND_BEARING = 11
    CLOUD_COVER = 12
    UV_INDEX = 13
    VISIBILITY = 14
    OZONE = 15
    STORM_DISTANCE = 16
    STORM_BEARING = 17
    SMOKE = 18
    FIRE_INDEX = 19
    FEELS_LIKE = 20


class SunFields(IntEnum):
    """Indexes for sunrise/sunset related arrays."""

    DAWN = 15
    DUSK = 16
    SUNRISE = 17
    SUNSET = 18
    MOON_PHASE = 19
