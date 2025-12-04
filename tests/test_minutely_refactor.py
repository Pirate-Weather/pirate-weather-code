import numpy as np

from API.constants.forecast_const import DATA_MINUTELY
from API.constants.model_const import HRRR, HRRR_SUBH
from API.minutely.builder import build_minutely_block


def test_build_minutely_block_structure():
    # Mock inputs
    minute_array_grib = np.arange(0, 61 * 60, 60)
    source_list = ["hrrrsubh", "hrrr_0-18"]

    # Mock HRRR SubH data (time, vars...)
    hrrr_subh_data = np.zeros((61, max(HRRR_SUBH.values()) + 1))
    hrrr_subh_data[:, 0] = minute_array_grib

    # Mock HRRR Merged data
    hrrr_merged = np.zeros((61, max(HRRR.values()) + 1))
    hrrr_merged[:, 0] = minute_array_grib

    kwargs = {
        "minute_array_grib": minute_array_grib,
        "source_list": source_list,
        "hrrr_subh_data": hrrr_subh_data,
        "hrrr_merged": hrrr_merged,
        "nbm_data": None,
        "gefs_data": None,
        "gfs_data": None,
        "ecmwf_data": None,
        "era5_data": None,
        "prep_intensity_unit": 1.0,
        "version": 2,
    }

    result = build_minutely_block(**kwargs)

    (
        InterPminute,
        InterTminute,
        minuteItems,
        minuteItems_si,
        maxPchance,
        pTypesText,
        pTypesIcon,
        hrrrSubHInterpolation,
    ) = result

    assert isinstance(InterPminute, np.ndarray)
    assert InterPminute.shape == (61, max(DATA_MINUTELY.values()) + 1)
    assert isinstance(minuteItems, list)
    assert len(minuteItems) == 61
    assert isinstance(minuteItems[0], dict)
    assert "precipIntensity" in minuteItems[0]


def test_build_minutely_block_empty():
    # Test with minimal inputs
    minute_array_grib = np.arange(0, 61 * 60, 60)
    source_list = []

    kwargs = {
        "minute_array_grib": minute_array_grib,
        "source_list": source_list,
        "hrrr_subh_data": None,
        "hrrr_merged": None,
        "nbm_data": None,
        "gefs_data": None,
        "gfs_data": None,
        "ecmwf_data": None,
        "era5_data": None,
        "prep_intensity_unit": 1.0,
        "version": 2,
    }

    result = build_minutely_block(**kwargs)

    InterPminute = result[0]
    assert isinstance(InterPminute, np.ndarray)
