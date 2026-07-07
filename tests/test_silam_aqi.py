"""Tests for SILAM air-quality ingest output variables.

SILAM ingest stores pollutant concentrations directly. AQI is calculated outside
of ingest, so these tests verify output variable/unit expectations without
importing or exercising any ingest-time AQI helper.
"""

import ast
from pathlib import Path

import numpy as np

SILAM_SCRIPT = Path(__file__).resolve().parents[1] / "API" / "FMI_Silam_Local_Ingest.py"


def _script_source() -> str:
    return SILAM_SCRIPT.read_text(encoding="utf-8")


def _constant_value(name: str):
    tree = ast.parse(_script_source())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Constant not found: {name}")


def test_silam_outputs_concentration_variables_not_aqi():
    zarr_vars = _constant_value("zarr_vars")

    assert "aqi" not in {var.lower() for var in zarr_vars}
    assert zarr_vars == (
        "time",
        "cnc_PM2_5",
        "cnc_PM10",
        "PM_FRP_column",
        "BLH",
        "cnc_O3",
        "cnc_NO2",
        "cnc_SO2",
        "cnc_CO",
    )


def test_silam_ingest_does_not_import_shared_aqi_calculator():
    source = _script_source()

    assert "calculate_aqi" not in source
    assert "calculate_nowcast_concentration" not in source


def test_silam_particulate_conversion_factors_are_mass_concentrations():
    assert _constant_value("KG_M3_TO_UG_M3") == 1e9
    assert _constant_value("KG_M2_TO_UG_M2") == 1e9

    pm25_kg_m3 = np.array([1e-9, 2.5e-8], dtype=np.float32)
    converted_pm25 = pm25_kg_m3 * _constant_value("KG_M3_TO_UG_M3")

    np.testing.assert_allclose(converted_pm25, np.array([1.0, 25.0], dtype=np.float32))


def test_silam_gas_conversion_factor_outputs_ppb():
    assert _constant_value("MOL_MOL_TO_PPB") == 1e9

    vmr = np.array([40e-9, 120e-9], dtype=np.float32)
    converted_ppb = vmr * _constant_value("MOL_MOL_TO_PPB")

    np.testing.assert_allclose(converted_ppb, np.array([40.0, 120.0], dtype=np.float32))
