import numpy as np

from API.constants.forecast_const import DATA_MINUTELY
from API.constants.model_const import DWD_MOSMIX, ECMWF, GEFS, GFS, HRRR, HRRR_SUBH
from API.minutely.builder import _interp_dwd_mosmix, build_minutely_block


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
        "dwd_mosmix_data": None,
        "gefs_data": None,
        "gfs_data": None,
        "ecmwf_data": None,
        "era5_data": None,
        "prep_intensity_unit": 1.0,
        "version": 2,
        "lat": 40.7128,  # New York City
        "lon": -74.0060,
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
        "dwd_mosmix_data": None,
        "gefs_data": None,
        "gfs_data": None,
        "ecmwf_data": None,
        "era5_data": None,
        "prep_intensity_unit": 1.0,
        "version": 2,
        "lat": 40.7128,  # New York City
        "lon": -74.0060,
    }

    result = build_minutely_block(**kwargs)

    InterPminute = result[0]
    assert isinstance(InterPminute, np.ndarray)


def test_interp_dwd_mosmix_no_rr1c_returns_none():
    """_interp_dwd_mosmix should return None when the accum column is all NaN,
    so the minutely elif chain falls through to ECMWF/GFS (matching hourly behaviour)."""
    minute_array_grib = np.arange(0, 61 * 60, 60, dtype=float)

    n_rows = 61
    n_cols = max(DWD_MOSMIX.values()) + 2  # +1 for time col (col 0)
    dwd_data = np.ones((n_rows, n_cols))
    dwd_data[:, 0] = minute_array_grib  # time
    dwd_data[:, DWD_MOSMIX["accum"]] = np.nan  # no RR1c at all

    result = _interp_dwd_mosmix(minute_array_grib, dwd_data)
    assert result is None, "Expected None when RR1c is entirely absent"


def test_interp_dwd_mosmix_with_rr1c_returns_array():
    """_interp_dwd_mosmix should return an array when at least some accum values exist."""
    minute_array_grib = np.arange(0, 61 * 60, 60, dtype=float)

    n_rows = 61
    n_cols = max(DWD_MOSMIX.values()) + 2
    dwd_data = np.ones((n_rows, n_cols))
    dwd_data[:, 0] = minute_array_grib
    dwd_data[:, DWD_MOSMIX["accum"]] = 0.5  # valid non-NaN accum

    result = _interp_dwd_mosmix(minute_array_grib, dwd_data)
    assert result is not None, "Expected array when RR1c has valid values"
    assert result.shape[0] == len(minute_array_grib)


def test_build_minutely_dwd_no_rr1c_falls_back_to_ecmwf():
    """When DWD MOSMIX has no RR1c, the minutely block should fall back to ECMWF."""
    minute_array_grib = np.arange(0, 61 * 60, 60, dtype=float)
    source_list = ["dwd_mosmix", "ecmwf_ifs"]

    # DWD MOSMIX data — all NaN for the accum column
    n_cols_dwd = max(DWD_MOSMIX.values()) + 2
    dwd_data = np.ones((61, n_cols_dwd))
    dwd_data[:, 0] = minute_array_grib
    dwd_data[:, DWD_MOSMIX["accum"]] = np.nan
    dwd_data[:, DWD_MOSMIX["temp"]] = 300.0
    dwd_data[:, DWD_MOSMIX["ptype"]] = 0.0

    # ECMWF data with a known non-zero precipitation intensity
    n_cols_ecmwf = max(ECMWF.values()) + 2
    ecmwf_data = np.zeros((61, n_cols_ecmwf))
    ecmwf_data[:, 0] = minute_array_grib
    ecmwf_data[:, ECMWF["intensity"]] = 0.001  # non-zero precipitation

    result = build_minutely_block(
        minute_array_grib=minute_array_grib,
        source_list=source_list,
        hrrr_subh_data=None,
        hrrr_merged=None,
        nbm_data=None,
        dwd_mosmix_data=dwd_data,
        gefs_data=None,
        gfs_data=None,
        ecmwf_data=ecmwf_data,
        era5_data=None,
        prep_intensity_unit=1.0,
        version=2,
        lat=-22.9,  # Rio de Janeiro
        lon=-43.2,
    )
    InterPminute = result[0]
    # With ECMWF as fallback, precipIntensity should be non-zero
    precip_intensity = InterPminute[:, DATA_MINUTELY["intensity"]]
    assert np.any(precip_intensity > 0), (
        "Expected non-zero precipIntensity from ECMWF fallback when DWD has no RR1c"
    )


def test_build_minutely_aigefs_ptype_falls_back_to_temperature():
    minute_array_grib = np.arange(0, 61 * 60, 60, dtype=float)
    source_list = ["gefs", "gfs"]

    gefs_data = np.full((61, max(GEFS.values()) + 1), np.nan)
    gefs_data[:, 0] = minute_array_grib
    gefs_data[:, GEFS["accum"]] = 0.4
    gefs_data[:, GEFS["prob"]] = 0.8

    gfs_data = np.full((61, max(GFS.values()) + 1), np.nan)
    gfs_data[:, 0] = minute_array_grib
    gfs_data[:, GFS["temp"]] = -5.0

    result = build_minutely_block(
        minute_array_grib=minute_array_grib,
        source_list=source_list,
        hrrr_subh_data=None,
        hrrr_merged=None,
        nbm_data=None,
        dwd_mosmix_data=None,
        gefs_data=gefs_data,
        gfs_data=gfs_data,
        ecmwf_data=None,
        era5_data=None,
        prep_intensity_unit=1.0,
        version=2,
        lat=45.0,
        lon=-75.0,
        prioritize_ai_models=True,
    )

    InterPminute = result[0]
    assert np.any(InterPminute[:, DATA_MINUTELY["snow_intensity"]] > 0)
    assert np.all(InterPminute[:, DATA_MINUTELY["rain_intensity"]] == 0)
