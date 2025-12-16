"""Tests for North America source priority adjustment."""

import numpy as np

from API.constants.model_const import DWD_MOSMIX, ECMWF
from API.data_inputs import prepare_data_inputs
from API.utils.geo import is_in_north_america


def test_is_in_north_america_usa():
    """Test that US locations are detected as North America."""
    # New York City
    assert is_in_north_america(40.7128, -74.0060) is True

    # Los Angeles
    assert is_in_north_america(34.0522, -118.2437) is True

    # Anchorage, Alaska
    assert is_in_north_america(61.2181, -149.9003) is True


def test_is_in_north_america_canada():
    """Test that Canadian locations are detected as North America."""
    # Toronto
    assert is_in_north_america(43.6532, -79.3832) is True

    # Vancouver
    assert is_in_north_america(49.2827, -123.1207) is True

    # Ottawa
    assert is_in_north_america(45.4215, -75.6972) is True


def test_is_in_north_america_mexico():
    """Test that Mexican locations are detected as North America."""
    # Mexico City
    assert is_in_north_america(19.4326, -99.1332) is True

    # Tijuana
    assert is_in_north_america(32.5149, -117.0382) is True


def test_is_not_in_north_america_europe():
    """Test that European locations are not detected as North America."""
    # London
    assert is_in_north_america(51.5074, -0.1278) is False

    # Berlin
    assert is_in_north_america(52.5200, 13.4050) is False

    # Paris
    assert is_in_north_america(48.8566, 2.3522) is False


def test_is_not_in_north_america_asia():
    """Test that Asian locations are not detected as North America."""
    # Tokyo
    assert is_in_north_america(35.6762, 139.6503) is False

    # Beijing
    assert is_in_north_america(39.9042, 116.4074) is False


def test_is_not_in_north_america_oceania():
    """Test that Oceania locations are not detected as North America."""
    # Sydney
    assert is_in_north_america(-33.8688, 151.2093) is False

    # Auckland
    assert is_in_north_america(-36.8485, 174.7633) is False


def test_is_not_in_north_america_south_america():
    """Test that South American locations are not detected as North America."""
    # Buenos Aires
    assert is_in_north_america(-34.6037, -58.3816) is False

    # Sao Paulo
    assert is_in_north_america(-23.5505, -46.6333) is False


def test_is_not_in_north_america_us_minor_outlying():
    """Test that US Minor Outlying Islands are not detected as North America."""
    # Wake Island (in Pacific, very far from mainland)
    assert is_in_north_america(19.2806, 166.6500) is False

    # Midway Atoll (in Pacific)
    assert is_in_north_america(28.2072, -177.3735) is False


def test_ecmwf_priority_in_north_america():
    """Test that ECMWF has higher priority than DWD MOSMIX in North America."""
    num_hours = 10

    # Create both DWD MOSMIX and ECMWF data with different temperatures
    dwd_merged = np.full((num_hours, max(DWD_MOSMIX.values()) + 1), np.nan)
    dwd_merged[:, 0] = np.arange(num_hours) * 3600
    dwd_merged[:, DWD_MOSMIX["temp"]] = 10.0  # DWD says 10°C

    ecmwf_merged = np.full((num_hours, max(ECMWF.values()) + 1), np.nan)
    ecmwf_merged[:, 0] = np.arange(num_hours) * 3600
    ecmwf_merged[:, ECMWF["temp"]] = 15.0  # ECMWF says 15°C

    # Test with New York location (North America)
    inputs = prepare_data_inputs(
        source_list=["dwd_mosmix", "ecmwf_ifs"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=dwd_merged,
        ecmwf_merged=ecmwf_merged,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=40.7128,  # New York
        lon=-74.0060,
    )

    # In North America, ECMWF should be prioritized (first non-NaN column)
    temp_inputs = inputs["temperature_inputs"]
    first_temp = temp_inputs[0, 0]

    # Should use ECMWF (15°C) not DWD MOSMIX (10°C)
    assert not np.isnan(first_temp), "Temperature should not be NaN"
    assert np.isclose(first_temp, 15.0, atol=0.1), (
        f"Should use ECMWF temperature (15°C) in North America, got {first_temp}°C"
    )


def test_dwd_priority_outside_north_america():
    """Test that DWD MOSMIX has higher priority than ECMWF outside North America."""
    num_hours = 10

    # Create both DWD MOSMIX and ECMWF data with different temperatures
    dwd_merged = np.full((num_hours, max(DWD_MOSMIX.values()) + 1), np.nan)
    dwd_merged[:, 0] = np.arange(num_hours) * 3600
    dwd_merged[:, DWD_MOSMIX["temp"]] = 10.0  # DWD says 10°C

    ecmwf_merged = np.full((num_hours, max(ECMWF.values()) + 1), np.nan)
    ecmwf_merged[:, 0] = np.arange(num_hours) * 3600
    ecmwf_merged[:, ECMWF["temp"]] = 15.0  # ECMWF says 15°C

    # Test with Berlin location (Europe, not North America)
    inputs = prepare_data_inputs(
        source_list=["dwd_mosmix", "ecmwf_ifs"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=dwd_merged,
        ecmwf_merged=ecmwf_merged,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=52.5200,  # Berlin
        lon=13.4050,
    )

    # Outside North America, DWD MOSMIX should be prioritized (first non-NaN column)
    temp_inputs = inputs["temperature_inputs"]
    first_temp = temp_inputs[0, 0]

    # Should use DWD MOSMIX (10°C) not ECMWF (15°C)
    assert not np.isnan(first_temp), "Temperature should not be NaN"
    assert np.isclose(first_temp, 10.0, atol=0.1), (
        f"Should use DWD MOSMIX temperature (10°C) outside North America, got {first_temp}°C"
    )


def test_priority_with_all_models_north_america():
    """Test priority with NBM, HRRR, DWD MOSMIX, ECMWF for North America."""
    num_hours = 5

    # Import NBM, GFS, and HRRR constants
    from API.constants.model_const import GFS, HRRR, NBM

    # Create data for all models with distinct temperatures
    nbm_merged = np.full((num_hours, max(NBM.values()) + 1), np.nan)
    nbm_merged[:, 0] = np.arange(num_hours) * 3600
    nbm_merged[:, NBM["temp"]] = 20.0  # NBM temp

    hrrr_merged = np.full((num_hours, max(HRRR.values()) + 1), np.nan)
    hrrr_merged[:, 0] = np.arange(num_hours) * 3600
    hrrr_merged[:, HRRR["temp"]] = 18.0  # HRRR temp

    dwd_merged = np.full((num_hours, max(DWD_MOSMIX.values()) + 1), np.nan)
    dwd_merged[:, 0] = np.arange(num_hours) * 3600
    dwd_merged[:, DWD_MOSMIX["temp"]] = 12.0

    ecmwf_merged = np.full((num_hours, max(ECMWF.values()) + 1), np.nan)
    ecmwf_merged[:, 0] = np.arange(num_hours) * 3600
    ecmwf_merged[:, ECMWF["temp"]] = 14.0

    gfs_merged = np.full((num_hours, max(GFS.values()) + 1), np.nan)
    gfs_merged[:, 0] = np.arange(num_hours) * 3600
    gfs_merged[:, GFS["temp"]] = 10.0  # GFS temp

    # Test with Toronto location
    inputs = prepare_data_inputs(
        source_list=[
            "nbm",
            "hrrr_0-18",
            "hrrr_18-48",
            "dwd_mosmix",
            "ecmwf_ifs",
            "gfs",
        ],
        nbm_merged=nbm_merged,
        nbm_fire_merged=None,
        hrrr_merged=hrrr_merged,
        dwd_mosmix_merged=dwd_merged,
        ecmwf_merged=ecmwf_merged,
        gefs_merged=None,
        gfs_merged=gfs_merged,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=43.6532,  # Toronto
        lon=-79.3832,
    )

    temp_inputs = inputs["temperature_inputs"]

    # Priority in North America: NBM > HRRR > ECMWF > GFS > DWD MOSMIX
    # First column should be NBM (20.0)
    assert np.isclose(temp_inputs[0, 0], 20.0, atol=0.1), "First priority should be NBM"
    # Second column should be HRRR (18.0)
    assert np.isclose(temp_inputs[0, 1], 18.0, atol=0.1), (
        "Second priority should be HRRR"
    )
    # Third column should be ECMWF (14.0) in North America
    assert np.isclose(temp_inputs[0, 2], 14.0, atol=0.1), (
        "Third priority should be ECMWF in North America"
    )
    # Fourth column should be GFS (10.0) in North America
    assert np.isclose(temp_inputs[0, 3], 10.0, atol=0.1), (
        "Fourth priority should be GFS in North America"
    )
    # Fifth column should be DWD MOSMIX (12.0)
    assert np.isclose(temp_inputs[0, 4], 12.0, atol=0.1), (
        "Fifth priority should be DWD MOSMIX"
    )


def test_priority_with_all_models_europe():
    """Test priority with HRRR, DWD MOSMIX, ECMWF for Europe (no NBM/HRRR)."""
    num_hours = 5

    # Import GFS constants
    from API.constants.model_const import GFS

    # Create data for models available in Europe
    dwd_merged = np.full((num_hours, max(DWD_MOSMIX.values()) + 1), np.nan)
    dwd_merged[:, 0] = np.arange(num_hours) * 3600
    dwd_merged[:, DWD_MOSMIX["temp"]] = 12.0

    ecmwf_merged = np.full((num_hours, max(ECMWF.values()) + 1), np.nan)
    ecmwf_merged[:, 0] = np.arange(num_hours) * 3600
    ecmwf_merged[:, ECMWF["temp"]] = 14.0

    gfs_merged = np.full((num_hours, max(GFS.values()) + 1), np.nan)
    gfs_merged[:, 0] = np.arange(num_hours) * 3600
    gfs_merged[:, GFS["temp"]] = 10.0  # GFS temp

    # Test with Berlin location
    inputs = prepare_data_inputs(
        source_list=["dwd_mosmix", "ecmwf_ifs", "gfs"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=dwd_merged,
        ecmwf_merged=ecmwf_merged,
        gefs_merged=None,
        gfs_merged=gfs_merged,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=52.5200,  # Berlin
        lon=13.4050,
    )

    temp_inputs = inputs["temperature_inputs"]

    # Priority outside North America: DWD MOSMIX > ECMWF > GFS
    # First column should be DWD MOSMIX (12.0)
    assert np.isclose(temp_inputs[0, 0], 12.0, atol=0.1), (
        "First priority should be DWD MOSMIX in Europe"
    )
    # Second column should be ECMWF (14.0)
    assert np.isclose(temp_inputs[0, 1], 14.0, atol=0.1), (
        "Second priority should be ECMWF"
    )
    # Third column should be GFS (10.0)
    assert np.isclose(temp_inputs[0, 2], 10.0, atol=0.1), "Third priority should be GFS"
