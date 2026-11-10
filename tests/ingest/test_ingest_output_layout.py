"""Ingest output layout vs. API variable-index contract."""

import importlib

import pytest

from API.constants import model_const
from API.constants.shared_const import (
    ENSEMBLE_MEMBER_COUNTS,
    MEMBER_MASK_BITS_PER_WORD,
    member_mask_var_names,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    EXACT_BITS,
    FINAL_SHARD_SIZES,
    KEEPBITS,
)
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

# RRFS keeps HRRR's order with its own MSLP and cloud-level names, then adds
# CIN and the lifted index; the CONUS and North American stores share it.
_RRFS_LAYOUT = {
    **_HRRR_LAYOUT,
    "pressure": "MSLET_meansealevel",
    "cloud": "TCDC_entireatmosphere_consideredasasinglelayer_",
    "cin": "CIN_surface",
    "lifted_index": "LFTX_500M1000mb",
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

_CANADIAN_ENSEMBLE_LAYOUT = {
    "prob": "Precipitation_Prob",
    "accum": "APCP_Mean",
    "error": "APCP_StdDev",
    "freezing_rain": "AFRAIN_Mean",
    "ice": "AICEP_Mean",
    "rain": "ARAIN_Mean",
    "snow": "ASNOW_Mean",
    "member_mask_0": "Precipitation_Member_Mask_0",
}

# (ingest module, or (module, name of its store variable tuple) when it is not
# zarr_vars; API index map in model_const; API key -> source variable the key
# is meant to read).
LAYOUTS = [
    pytest.param("HRRR_Local_Ingest", "HRRR", _HRRR_LAYOUT, id="hrrr"),
    pytest.param("HRRR_6H_Local_Ingest", "HRRR", _HRRR_LAYOUT, id="hrrr_6h"),
    pytest.param("RRFS_Local_Ingest", "RRFS", _RRFS_LAYOUT, id="rrfs"),
    pytest.param("RRFS_6H_Local_Ingest", "RRFS", _RRFS_LAYOUT, id="rrfs_6h"),
    pytest.param("RRFS_NA_Local_Ingest", "RRFS", _RRFS_LAYOUT, id="rrfs_na"),
    pytest.param("RRFS_NA_6H_Local_Ingest", "RRFS", _RRFS_LAYOUT, id="rrfs_na_6h"),
    pytest.param(
        "LibreWXR_Local_Ingest", "LIBREWXR", {"refc": "REFC_composite"}, id="librewxr"
    ),
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
    pytest.param(
        "RRFS_SubH_Local_Ingest",
        "RRFS_SUBH",
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
        id="rrfs_subh",
    ),
    pytest.param(
        "ECMWF_Local_Ingest",
        "ECMWF",
        {
            "pressure": "msl",
            "temp": "t2m",
            "dew": "d2m",
            "wind_u": "u10",
            "wind_v": "v10",
            "intensity": "tprate",
            "accum": "tp",
            "ptype": "ptype",
            "cloud": "tcc",
            "prob": "Precipitation_Prob",
            "accum_mean": "APCP_Mean",
            "accum_stddev": "APCP_StdDev",
            "humidity": "rh",
            "member_mask_0": "Precipitation_Member_Mask_0",
            "member_mask_1": "Precipitation_Member_Mask_1",
            "member_mask_2": "Precipitation_Member_Mask_2",
        },
        id="ecmwf",
    ),
    pytest.param(
        "ECMWF_AIFS_Local_Ingest",
        "ECMWF_AIFS",
        {
            "pressure": "msl",
            "temp": "t2m",
            "dew": "d2m",
            "wind_u": "u10",
            "wind_v": "v10",
            "ptype": "ptype",
            "cloud": "tcc",
            "prob": "Precipitation_Prob",
            "accum_mean": "APCP_Mean",
            "accum_stddev": "APCP_StdDev",
            "member_mask_0": "Precipitation_Member_Mask_0",
            "member_mask_1": "Precipitation_Member_Mask_1",
            "member_mask_2": "Precipitation_Member_Mask_2",
        },
        id="ecmwf_aifs",
    ),
    # The ensemble ingests publish statistics (probVars), not the per-member
    # zarr_vars they read
    pytest.param(
        ("GEFS_Local_Ingest", "probVars"),
        "GEFS",
        {
            "prob": "Precipitation_Prob",
            "accum": "APCP_Mean",
            "error": "APCP_StdDev",
            "snow": "CSNOW_Prob",
            "ice": "CICEP_Prob",
            "freezing_rain": "CFRZR_Prob",
            "rain": "CRAIN_Prob",
            "member_mask_0": "Precipitation_Member_Mask_0",
            "member_mask_1": "Precipitation_Member_Mask_1",
        },
        id="gefs",
    ),
    pytest.param(
        ("AIGEFS_Local_Ingest", "probVars"),
        "AIGEFS",
        {
            "prob": "Precipitation_Prob",
            "accum": "APCP_Mean",
            "error": "APCP_StdDev",
            "member_mask_0": "Precipitation_Member_Mask_0",
            "member_mask_1": "Precipitation_Member_Mask_1",
        },
        id="aigefs",
    ),
    pytest.param(
        ("GEPS_Local_Ingest", "probVars"), "GEPS", _CANADIAN_ENSEMBLE_LAYOUT, id="geps"
    ),
    pytest.param(
        ("REPS_Local_Ingest", "probVars"), "REPS", _CANADIAN_ENSEMBLE_LAYOUT, id="reps"
    ),
    pytest.param(
        "CFS_Local_Ingest",
        "CFS",
        {
            "temp": "TMP_2maboveground",
            "dew": "DPT_2maboveground",
            "humidity": "RH_2maboveground",
            "pressure": "PRMSL_meansealevel",
            "station_pressure": "PRES_surface",
            "wind_u": "UGRD_10maboveground",
            "wind_v": "VGRD_10maboveground",
            "intensity": "PRATE_surface",
            "cloud": "TCDC_entireatmosphere",
            "solar": "DSWRF_surface",
            "prob": "Precipitation_Prob",
            "error": "APCP_StdDev",
            "member_mask_0": "Precipitation_Member_Mask_0",
        },
        id="cfs",
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
    Stores no longer hold time (it is the array's time attribute, which the
    reader puts in column 0), so the store's variable i is column i + 1.
    """
    if ingest_module is None:
        # RAQDPS's column order, time first, is raqdps_utils.RAQDPS_OUTPUT_VARS.
        columns = RAQDPS_OUTPUT_VARS
    else:
        if isinstance(ingest_module, tuple):
            ingest_module, store_vars_name = ingest_module
        else:
            store_vars_name = "zarr_vars"
        zarr_vars = getattr(
            importlib.import_module(f"ingest.{ingest_module}"), store_vars_name
        )
        assert "time" not in zarr_vars
        # write_final_zarr bitrounds by name and raises on one KEEPBITS lacks
        assert set(zarr_vars) <= set(KEEPBITS), set(zarr_vars) - set(KEEPBITS)
        columns = ("time", *zarr_vars)
    index_map = getattr(model_const, const_name)

    assert columns[0] == "time"
    actual = {
        key: columns[idx] if idx < len(columns) else f"<out of range: {idx}>"
        for key, idx in index_map.items()
    }
    assert actual == expected


def test_member_mask_words_cover_every_member_and_stay_exact():
    """The API popcounts the mask words and divides by ENSEMBLE_MEMBER_COUNTS, so a
    model with more members than its words hold, a word bitrounded to fewer than
    its 24 bits, or an AI ensemble whose count differs from the model whose
    columns it borrows (forecast_sources copies them across) would each give a
    wrong daily probability with no error.
    """
    for model, count in ENSEMBLE_MEMBER_COUNTS.items():
        words = member_mask_var_names(count)
        assert len(words) * MEMBER_MASK_BITS_PER_WORD >= count, model
        assert all(KEEPBITS[word] == EXACT_BITS for word in words), model
    # A float32 holds one word's integers exactly
    assert MEMBER_MASK_BITS_PER_WORD <= EXACT_BITS + 1
    assert ENSEMBLE_MEMBER_COUNTS["AIGEFS"] == ENSEMBLE_MEMBER_COUNTS["GEFS"]
    assert ENSEMBLE_MEMBER_COUNTS["ECMWF_AIFS"] == ENSEMBLE_MEMBER_COUNTS["ECMWF_IFS"]
    for model in ("GEFS", "AIGEFS", "GEPS", "REPS", "ECMWF", "ECMWF_AIFS", "CFS"):
        index_map = getattr(model_const, model)
        mask_columns = [v for k, v in index_map.items() if k.startswith("member_mask_")]
        count_key = {"ECMWF": "ECMWF_IFS"}.get(model, model)
        assert len(mask_columns) == len(
            member_mask_var_names(ENSEMBLE_MEMBER_COUNTS[count_key])
        ), model
        # Appended after every earlier column, so older stores still read
        assert min(mask_columns) > max(
            v for k, v in index_map.items() if not k.startswith("member_mask_")
        ), model


def test_processing_tiles_split_into_whole_final_shards():
    """Protect the ingest memory limit: when a shard straddles processing tiles, every
    tile stays in memory until its neighbours are written (NBM 8.9 GiB against 3.7 GiB,
    GFS 10.4 GiB against 3.3 GiB), with no error and an identical store, so a chunk or
    shard change would otherwise only show up as an out-of-memory job in production.
    """
    # RAQDPS shards are only ~18 MB, so its 250-cell tiles have never mattered
    misaligned_on_purpose = {"RAQDPS"}
    for model, shard in FINAL_SHARD_SIZES.items():
        if model in CHUNK_SIZES and model not in misaligned_on_purpose:
            assert CHUNK_SIZES[model] % shard == 0, model
