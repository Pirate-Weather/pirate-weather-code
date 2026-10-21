"""Ingest output layout vs. API variable-index contract."""

import importlib

import pytest

from API.constants import model_const
from API.raqdps_utils import RAQDPS_OUTPUT_VARS

_HRRR_LAYOUT = {
    "vis": "VIS_surface",
    "gust": "GUST_surface",
    "pressure": "MSLMA_meansealevel",
    "temp": "TMP_2maboveground",
    "dew": "DPT_2maboveground",
    "humidity": "RH_2maboveground",
    "wind_u": "UGRD_10maboveground",
    "wind_v": "VGRD_10maboveground",
    "intensity": "PRATE_surface",
    "accum": "APCP_surface",
    "snow": "CSNOW_surface",
    "ice": "CICEP_surface",
    "freezing_rain": "CFRZR_surface",
    "rain": "CRAIN_surface",
    "cloud": "TCDC_entireatmosphere",
    "smoke": "MASSDEN_8maboveground",
    "refc": "REFD_1000maboveground",
    "solar": "DSWRF_surface",
    "cape": "CAPE_surface",
}

_ANALYSIS_LAYOUT = {
    "vis": "vis",
    "gust": "i10fg",
    "pressure": "sp",
    "temp": "t2m",
    "dew": "d2m",
    "humidity": "rh",
    "cloud": "tcc",
    "wind_u": "u10",
    "wind_v": "v10",
}

# (ingest module, API index map in model_const, API key -> source variable the
# key is meant to read).
LAYOUTS = [
    pytest.param("HRRR_Local_Ingest", "HRRR", _HRRR_LAYOUT, id="hrrr"),
    pytest.param("HRRR_6H_Local_Ingest", "HRRR", _HRRR_LAYOUT, id="hrrr_6h"),
    pytest.param(
        "SubH_Local_Ingest",
        "HRRR_SUBH",
        {
            "gust": "GUST_surface",
            "pressure": "PRES_surface",
            "temp": "TMP_2maboveground",
            "dew": "DPT_2maboveground",
            "wind_u": "UGRD_10maboveground",
            "wind_v": "VGRD_10maboveground",
            "intensity": "PRATE_surface",
            "snow": "CSNOW_surface",
            "ice": "CICEP_surface",
            "freezing_rain": "CFRZR_surface",
            "rain": "CRAIN_surface",
            "refc": "REFD_1000maboveground",
            "accum": "APCP_surface",
            "vis": "VIS_surface",
            "specific_humidity": "SPFH_2maboveground",
            "solar": "DSWRF_surface",
        },
        id="hrrr_subh",
    ),
    pytest.param("RTMA-RU_Local_Ingest", "RTMA_RU", _ANALYSIS_LAYOUT, id="rtma_ru"),
    pytest.param("URMA_Local_Ingest", "URMA", _ANALYSIS_LAYOUT, id="urma"),
    pytest.param(
        "DWD_Mosmix_Local_Ingest",
        "DWD_MOSMIX",
        {
            "temp": "TMP_2maboveground",
            "dew": "DPT_2maboveground",
            "humidity": "RH_2maboveground",
            "pressure": "PRES_meansealevel",
            "wind_u": "UGRD_10maboveground",
            "wind_v": "VGRD_10maboveground",
            "gust": "GUST_surface",
            "accum": "APCP_surface",
            "cloud": "TCDC_entireatmosphere",
            "vis": "VIS_surface",
            "ptype": "PTYPE_surface",
            "solar": "DSWRF_surface",
        },
        id="dwd_mosmix",
    ),
    pytest.param(
        "FMI_Silam_Local_Ingest",
        "SILAM",
        {
            "time": "time",
            "pm25": "cnc_PM2_5",
            "pm10": "cnc_PM10",
            "pm_frp_column": "PM_FRP_column",
            "blh": "BLH",
            "o3": "cnc_O3",
            "no2": "cnc_NO2",
            "so2": "cnc_SO2",
            "co": "cnc_CO",
        },
        id="silam",
    ),
    pytest.param(
        None,
        "RAQDPS",
        {
            "time": "time",
            "pm25": "PM2.5",
            "pm10": "PM10",
            "no2": "NO2",
            "o3": "O3",
            "so2": "SO2",
        },
        id="raqdps",
    ),
]


@pytest.mark.parametrize(("ingest_module", "const_name", "expected"), LAYOUTS)
def test_api_index_map_matches_ingest_variable_order(
    ingest_module, const_name, expected
):
    """The API reads ingest zarrs by integer position via model_const maps.
    If an ingest adds, removes or reorders a variable without updating the
    map, the API silently serves the wrong field (e.g. dew point as
    temperature) with no error. Every key must index the variable it names.
    """
    if ingest_module is None:
        # RAQDPS stacks variables in raqdps_utils.RAQDPS_OUTPUT_VARS order.
        zarr_vars = RAQDPS_OUTPUT_VARS
    else:
        zarr_vars = importlib.import_module(f"ingest.{ingest_module}").zarr_vars
    index_map = getattr(model_const, const_name)

    assert zarr_vars[0] == "time"
    actual = {
        key: zarr_vars[idx] if idx < len(zarr_vars) else f"<out of range: {idx}>"
        for key, idx in index_map.items()
    }
    assert actual == expected
