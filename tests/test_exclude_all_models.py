"""Tests for the validation that prevents excluding all forecast model sources."""

from API.constants.model_const import FORECAST_SOURCES


def _has_forecast_source(source_list):
    """Return True if at least one forecast source is in source_list."""
    return any(src in source_list for src in FORECAST_SOURCES)


def test_no_forecast_source_when_only_rtma_ru():
    """RTMA-RU alone is not a forecast model; validation should fail."""
    source_list = ["rtma_ru"]
    assert not _has_forecast_source(source_list)


def test_no_forecast_source_when_only_hrrrsubh():
    """hrrrsubh alone is not a forecast model; validation should fail."""
    source_list = ["hrrrsubh"]
    assert not _has_forecast_source(source_list)


def test_no_forecast_source_when_only_etopo():
    """etopo alone is not a forecast model; validation should fail."""
    source_list = ["etopo"]
    assert not _has_forecast_source(source_list)


def test_no_forecast_source_empty_list():
    """Empty source list should fail validation."""
    source_list = []
    assert not _has_forecast_source(source_list)


def test_no_forecast_source_when_rtma_and_hrrrsubh():
    """RTMA-RU + hrrrsubh together still have no forecast source."""
    source_list = ["rtma_ru", "hrrrsubh", "etopo"]
    assert not _has_forecast_source(source_list)


def test_gfs_is_forecast_source():
    """GFS alone satisfies the forecast source requirement."""
    source_list = ["gfs"]
    assert _has_forecast_source(source_list)


def test_ecmwf_is_forecast_source():
    """ECMWF alone satisfies the forecast source requirement."""
    source_list = ["ecmwf_ifs"]
    assert _has_forecast_source(source_list)


def test_gefs_is_forecast_source():
    """GEFS alone satisfies the forecast source requirement."""
    source_list = ["gefs"]
    assert _has_forecast_source(source_list)


def test_dwd_mosmix_is_forecast_source():
    """DWD MOSMIX alone satisfies the forecast source requirement."""
    source_list = ["dwd_mosmix"]
    assert _has_forecast_source(source_list)


def test_nbm_is_forecast_source():
    """NBM alone satisfies the forecast source requirement."""
    source_list = ["nbm"]
    assert _has_forecast_source(source_list)


def test_hrrr_blocks_are_forecast_sources():
    """HRRR (0-18 and 18-48) together satisfy the forecast source requirement."""
    source_list = ["hrrr_0-18", "hrrr_18-48"]
    assert _has_forecast_source(source_list)


def test_hrrr_timemachine_is_forecast_source():
    """HRRR in time machine mode satisfies the forecast source requirement."""
    source_list = ["hrrr"]
    assert _has_forecast_source(source_list)


def test_era5_is_forecast_source():
    """ERA5 (used in time machine mode) satisfies the forecast source requirement."""
    source_list = ["era5"]
    assert _has_forecast_source(source_list)


def test_gfs_with_rtma_is_forecast_source():
    """GFS + RTMA-RU: has a forecast source."""
    source_list = ["rtma_ru", "gfs", "hrrrsubh"]
    assert _has_forecast_source(source_list)


def test_full_non_us_source_list():
    """A realistic non-US source list (no HRRR/NBM) satisfies the requirement."""
    source_list = ["etopo", "gfs", "ecmwf_ifs", "gefs", "dwd_mosmix"]
    assert _has_forecast_source(source_list)


def test_all_models_excluded_non_us_location():
    """If GFS, ECMWF, GEFS, DWD are all excluded on non-US location, no forecast source remains."""
    # Non-US location: HRRR and NBM are not available, and all others excluded
    source_list = ["etopo"]  # only elevation data
    assert not _has_forecast_source(source_list)
