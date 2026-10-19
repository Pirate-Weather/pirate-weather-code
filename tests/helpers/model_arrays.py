"""Builders for merged model arrays and hourly-input calls used by API tests."""

import numpy as np

from API.data_inputs import prepare_data_inputs


def model_array(model: dict, num_hours: int, **columns: float) -> np.ndarray:
    """Return an all-NaN merged array for ``model`` with named columns filled.

    ``model`` is a column map from ``API.constants.model_const`` (e.g. ``GFS``);
    each keyword sets that column to a constant for every hour.
    """
    array = np.full((num_hours, max(model.values()) + 1), np.nan)
    for name, value in columns.items():
        array[:, model[name]] = value
    return array


def hourly_inputs(
    *,
    lat: float,
    lon: float,
    source_list: list[str],
    prioritize_ai_models: bool = False,
    extra_vars: list[str] | None = None,
    **merged: np.ndarray,
) -> dict:
    """Call ``prepare_data_inputs`` with every source not given defaulting to None.

    Merged arrays are passed by source name without the ``_merged`` suffix
    (``gfs=...``, ``dwd_mosmix=...``); ``num_hours`` is taken from them.
    """
    kwargs = {
        f"{name}_merged": None
        for name in ("nbm", "hrrr", "dwd_mosmix", "ecmwf", "gefs", "gfs", "era5")
    }
    kwargs.update({f"{name}_merged": array for name, array in merged.items()})
    num_hours = next(iter(merged.values())).shape[0]
    return prepare_data_inputs(
        source_list=source_list,
        extra_vars=extra_vars or [],
        num_hours=num_hours,
        lat=lat,
        lon=lon,
        prioritize_ai_models=prioritize_ai_models,
        **kwargs,
    )
