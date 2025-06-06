"""Indices used for hourly and daily forecast arrays."""
from enum import IntEnum


class HourlyIndex(IntEnum):
    TIME = 0
    PRECIP_TYPE = 1
    PRECIP_INTENSITY = 2
    PRECIP_PROBABILITY = 3
    PRECIP_INTENSITY_ERROR = 4
    TEMPERATURE = 5
    APPARENT_TEMP_RADIATIVE = 6
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
    PRECIP_ACCUMULATION = 17
    NEAR_STORM_DISTANCE = 18
    NEAR_STORM_BEARING = 19
    SMOKE = 20
    LIQUID_ACCUM = 21
    SNOW_ACCUM = 22
    ICE_ACCUM = 23
    FIRE_INDEX = 24
    FEELS_LIKE = 25


class DailyIndex(IntEnum):
    """Indices used for daily forecast arrays."""

    TIME = HourlyIndex.TIME
    PRECIP_TYPE = HourlyIndex.PRECIP_TYPE
    PRECIP_INTENSITY = HourlyIndex.PRECIP_INTENSITY
    PRECIP_PROBABILITY = HourlyIndex.PRECIP_PROBABILITY
    PRECIP_INTENSITY_ERROR = HourlyIndex.PRECIP_INTENSITY_ERROR
    TEMPERATURE = HourlyIndex.TEMPERATURE
    APPARENT_TEMP_RADIATIVE = HourlyIndex.APPARENT_TEMP_RADIATIVE
    DEW_POINT = HourlyIndex.DEW_POINT
    HUMIDITY = HourlyIndex.HUMIDITY
    PRESSURE = HourlyIndex.PRESSURE
    WIND_SPEED = HourlyIndex.WIND_SPEED
    WIND_GUST = HourlyIndex.WIND_GUST
    WIND_BEARING = HourlyIndex.WIND_BEARING
    CLOUD_COVER = HourlyIndex.CLOUD_COVER
    UV_INDEX = HourlyIndex.UV_INDEX
    VISIBILITY = HourlyIndex.VISIBILITY
    OZONE = HourlyIndex.OZONE
    PRECIP_ACCUMULATION = HourlyIndex.PRECIP_ACCUMULATION
    NEAR_STORM_DISTANCE = HourlyIndex.NEAR_STORM_DISTANCE
    NEAR_STORM_BEARING = HourlyIndex.NEAR_STORM_BEARING
    SMOKE = HourlyIndex.SMOKE
    LIQUID_ACCUM = HourlyIndex.LIQUID_ACCUM
    SNOW_ACCUM = HourlyIndex.SNOW_ACCUM
    ICE_ACCUM = HourlyIndex.ICE_ACCUM
    FIRE_INDEX = HourlyIndex.FIRE_INDEX
    FEELS_LIKE = HourlyIndex.FEELS_LIKE


class SunIndex(IntEnum):
    """Indices used for sunrise and sunset arrays."""

    DAWN = 15
    DUSK = 16
    SUNRISE = 17
    SUNSET = 18
    MOON_PHASE = 19
