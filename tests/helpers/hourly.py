"""Keyword-argument builder for ``API.hourly.block.build_hourly_block``."""

from unittest.mock import MagicMock

import numpy as np

from API.constants.forecast_const import DATA_DAY, DATA_HOURLY

# Per-source input stacks (num_hours, n_sources) that prepare_data_inputs returns.
_STACKED_INPUTS = (
    "prcipIntensity_inputs",
    "prcipProbability_inputs",
    "prcipType_inputs",
    "temperature_inputs",
    "dew_inputs",
    "humidity_inputs",
    "pressure_inputs",
    "wind_inputs",
    "gust_inputs",
    "bearing_inputs",
    "cloud_inputs",
    "uv_inputs",
    "vis_inputs",
    "ozone_inputs",
    "smoke_inputs",
    "accum_inputs",
    "fire_inputs",
    "feels_like_inputs",
    "solar_inputs",
    "cape_inputs",
    "error_inputs",
    "lifted_index_inputs",
    "vertical_velocity_inputs",
    "convective_inhibition_inputs",
    "k_index_inputs",
)


def hourly_block_kwargs(num_hours, **overrides):
    """Return SI-unit ``build_hourly_block`` kwargs for one day of hourly data
    with every model input missing (NaN); pass overrides for the fields a test
    cares about, e.g. ``**prepare_data_inputs(...)``."""
    kwargs = {
        "source_list": [],
        "InterPhour": np.zeros((num_hours, max(DATA_HOURLY.values()) + 1)),
        "hour_array_grib": np.arange(num_hours) * 3600,
        "hour_array": np.arange(num_hours),
        "InterSday": np.zeros((1, max(DATA_DAY.values()) + 1)),
        "hourlyDayIndex": np.zeros(num_hours, dtype=int),
        "baseTimeOffset": 0,
        "timeMachine": False,
        "tmExtra": False,
        "prepIntensityUnit": 1.0,
        "prepAccumUnit": 1.0,
        "windUnit": 1.0,
        "visUnits": 1.0,
        "tempUnits": 1,
        "humidUnit": 1.0,
        "extraVars": [],
        "summaryText": False,
        "icon": "default",
        "translation": MagicMock(),
        "unitSystem": "si",
        "is_all_night": False,
        "tz_name": "UTC",
        "InterThour_inputs": {},
        "nearstorm_inputs": {
            "dist": np.full((num_hours, 1), np.nan),
            "dir": np.full((num_hours, 1), np.nan),
        },
        "station_pressure_inputs": None,
        "era5_rain_intensity": None,
        "era5_snow_water_equivalent": None,
        "version": 2,
    }
    kwargs.update({name: np.full((num_hours, 1), np.nan) for name in _STACKED_INPUTS})
    kwargs.update(overrides)
    return kwargs
