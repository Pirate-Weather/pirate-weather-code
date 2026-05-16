import datetime

import numpy as np
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
