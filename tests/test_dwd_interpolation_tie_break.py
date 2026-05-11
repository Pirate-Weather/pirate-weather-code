import ast
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree


def load_interpolate_dwd_to_grid_knearest_dask():
    source_path = (
        Path(__file__).resolve().parents[1] / "API" / "DWD_Mosmix_Local_Ingest.py"
    )
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))

    for node in module.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "interpolate_dwd_to_grid_knearest_dask"
        ):
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {
                "da": da,
                "np": np,
                "pd": pd,
                "xr": xr,
                "BallTree": BallTree,
                "tqdm": lambda iterable, **kwargs: iterable,
            }
            exec(compile(function_module, str(source_path), "exec"), namespace)
            return namespace["interpolate_dwd_to_grid_knearest_dask"]

    raise AssertionError("interpolate_dwd_to_grid_knearest_dask not found")


interpolate_dwd_to_grid_knearest_dask = load_interpolate_dwd_to_grid_knearest_dask()


def test_equal_distance_tie_prefers_lowest_station_id():
    df = pd.DataFrame(
        {
            "station_id": ["100", "99"],
            "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
            "latitude": [0.0, 0.0],
            "longitude": [10.125, 9.875],
            "TMP_2maboveground": [2.0, 1.0],
        }
    )

    ds = interpolate_dwd_to_grid_knearest_dask(
        df,
        var_cols=["TMP_2maboveground"],
        radius_km=20.0,
        k_max=4,
        time_col="time",
        lat_col="latitude",
        lon_col="longitude",
        station_col="station_id",
        log="none",
    )

    result = (
        ds["TMP_2maboveground"].isel(time=0).sel(lat=0.0, lon=10.0).compute().item()
    )

    assert result == 1.0