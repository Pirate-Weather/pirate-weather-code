import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_fill_station_gaps():
    source_path = Path(__file__).resolve().parents[1] / "API" / "DWD_Mosmix_Local_Ingest.py"
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "fill_station_gaps":
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace = {"np": np, "pd": pd}
            exec(compile(function_module, str(source_path), "exec"), namespace)
            return namespace["fill_station_gaps"]

    raise AssertionError("fill_station_gaps not found")


fill_station_gaps = load_fill_station_gaps()


def test_large_continuous_gap_nulls_entire_station_variable():
    times = pd.date_range("2026-01-01", periods=8, freq="1h")
    df = pd.DataFrame(
        {
            "station_id": ["A"] * 8 + ["B"] * 8,
            "time": list(times) * 2,
            "latitude": [50.0] * 16,
            "longitude": [10.0] * 16,
            "altitude": [100.0] * 16,
            "TMP_2maboveground": [
                1.0,
                2.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                7.0,
                8.0,
                1.0,
                2.0,
                np.nan,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            "VIS_surface": [
                10.0,
                11.0,
                np.nan,
                np.nan,
                14.0,
                15.0,
                16.0,
                17.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
            ],
        }
    )

    result = fill_station_gaps(df, max_gap_hours=3)

    station_a = result[result["station_id"] == "A"].reset_index(drop=True)
    station_b = result[result["station_id"] == "B"].reset_index(drop=True)

    assert station_a["TMP_2maboveground"].isna().all()
    np.testing.assert_allclose(
        station_a["VIS_surface"].to_numpy(),
        np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
    )
    np.testing.assert_allclose(
        station_b["TMP_2maboveground"].to_numpy(),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )


def test_precip_and_ptype_keep_special_case_behavior_for_long_gaps():
    times = pd.date_range("2026-01-01", periods=6, freq="1h")
    df = pd.DataFrame(
        {
            "station_id": ["A"] * 6,
            "time": times,
            "latitude": [50.0] * 6,
            "longitude": [10.0] * 6,
            "altitude": [100.0] * 6,
            "APCP_surface": [0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
            "PTYPE_surface": [1.0, np.nan, np.nan, np.nan, np.nan, 2.0],
        }
    )

    result = fill_station_gaps(df, max_gap_hours=3)

    np.testing.assert_allclose(
        result["APCP_surface"].to_numpy(),
        np.zeros(6, dtype=np.float64),
    )
    np.testing.assert_allclose(
        result["PTYPE_surface"].to_numpy(),
        np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]),
    )