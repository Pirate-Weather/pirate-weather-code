import datetime

import numpy as np
import pytest
import xarray as xr

from API.constants.model_const import ERA5
from API.request.grid_indexing import _load_era5_slice


def test_load_era5_slice_uses_requested_num_hours():
    hours = 60
    requested_hours = 49
    times = np.array(
        [
            np.datetime64("2024-01-01T00:00:00") + np.timedelta64(hour, "h")
            for hour in range(hours)
        ]
    )
    latitudes = np.array([40.0])
    longitudes = np.array([286.0])

    data_vars = {}
    for var_name, var_index in ERA5.items():
        values = np.full((hours, 1, 1), float(var_index))
        if var_name == "precipitation_type":
            values[:, 0, 0] = 4.6
        data_vars[var_name] = (("time", "latitude", "longitude"), values)

    ds_era5 = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    era5_data = {
        "ERA5_lats": latitudes,
        "ERA5_lons": longitudes,
        "ERA5_times": times,
        "dsERA5": ds_era5,
    }

    result = _load_era5_slice(
        era5_data,
        lat=40.0,
        lon=286.0,
        base_day_utc=datetime.datetime(2024, 1, 1, 0, 0, 0),
        num_hours=requested_hours,
    )

    assert result.shape[0] == requested_hours
    np.testing.assert_array_equal(
        result[:, 0],
        np.array(
            [
                int(
                    (time - np.datetime64("1970-01-01T00:00:00"))
                    / np.timedelta64(1, "s")
                )
                for time in times[:requested_hours]
            ],
            dtype=np.int64,
        ),
    )
    assert np.all(result[:, ERA5["precipitation_type"]] == 5)
    assert np.all(result[:, ERA5["prob"]] == 100.0)


def test_load_era5_slice_estimates_precip_probability_from_neighbourhood():
    hours = 3
    times = np.array(
        [
            np.datetime64("2024-01-01T00:00:00") + np.timedelta64(hour, "h")
            for hour in range(hours)
        ]
    )
    latitudes = np.array([40.0, 39.75])
    longitudes = np.array([0.0, 90.0, 180.0, 270.0])

    data_vars = {}
    for var_name, var_index in ERA5.items():
        if var_name == "prob":
            continue
        values = np.full((hours, latitudes.size, longitudes.size), float(var_index))
        if var_name == "precipitation_type":
            values[:] = 4.6
        if var_name == "total_precipitation":
            values[:] = 0.0
            values[0, :, [3, 0, 1]] = 0.0002
            values[1, 0, [3, 0, 1]] = 0.0002
            values[1, 1, 3] = np.nan
            values[2, :, [3, 0, 1]] = np.nan
        data_vars[var_name] = (("time", "latitude", "longitude"), values)

    ds_era5 = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    era5_data = {
        "ERA5_lats": latitudes,
        "ERA5_lons": longitudes,
        "ERA5_times": times,
        "dsERA5": ds_era5,
    }

    result = _load_era5_slice(
        era5_data,
        lat=40.0,
        lon=0.0,
        base_day_utc=datetime.datetime(2024, 1, 1, 0, 0, 0),
        num_hours=hours,
    )

    np.testing.assert_allclose(
        result[:, ERA5["prob"]],
        np.array([100.0, 60.0, 0.0]),
    )


def test_load_era5_slice_raises_clear_error_without_precip_amount_variable():
    times = np.array([np.datetime64("2024-01-01T00:00:00")])
    latitudes = np.array([40.0])
    longitudes = np.array([286.0])

    data_vars = {}
    for var_name, var_index in ERA5.items():
        if var_name in {"prob", "total_precipitation"}:
            continue
        data_vars[var_name] = (
            ("time", "latitude", "longitude"),
            np.full((1, 1, 1), float(var_index)),
        )

    era5_data = {
        "ERA5_lats": latitudes,
        "ERA5_lons": longitudes,
        "ERA5_times": times,
        "dsERA5": xr.Dataset(
            data_vars=data_vars,
            coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        ),
    }

    with pytest.raises(KeyError, match="total_precipitation"):
        _load_era5_slice(
            era5_data,
            lat=40.0,
            lon=286.0,
            base_day_utc=datetime.datetime(2024, 1, 1, 0, 0, 0),
            num_hours=1,
        )
