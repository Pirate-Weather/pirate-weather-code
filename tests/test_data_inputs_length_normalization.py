import numpy as np
import pytest

from API.constants.model_const import ERA5
from API.data_inputs import _normalize_length, prepare_data_inputs


@pytest.mark.parametrize(
    ("values", "num_hours"),
    [
        (np.arange(3), 5),
        (np.arange(5), 3),
    ],
)
def test_normalize_length_logs_warning_for_mismatched_input(caplog, values, num_hours):
    with caplog.at_level("WARNING"):
        _normalize_length(num_hours, values, label="test_input")

    assert "Normalizing data input length for test_input" in caplog.text


def test_normalize_length_does_not_log_when_length_matches(caplog):
    with caplog.at_level("WARNING"):
        result = _normalize_length(4, np.arange(4), label="matched_input")

    assert caplog.text == ""
    np.testing.assert_array_equal(result, np.arange(4))


def test_prepare_data_inputs_normalizes_short_era5_series_to_num_hours():
    num_hours = 49
    source_hours = 25
    era5_merged = np.full((source_hours, max(ERA5.values()) + 1), np.nan)

    era5_merged[:, ERA5["precipitation_type"]] = np.arange(source_hours)
    era5_merged[:, ERA5["large_scale_rain_rate"]] = 0.001
    era5_merged[:, ERA5["convective_rain_rate"]] = 0.002
    era5_merged[:, ERA5["large_scale_snowfall_rate_water_equivalent"]] = 0.0
    era5_merged[:, ERA5["convective_snowfall_rate_water_equivalent"]] = 0.0
    era5_merged[:, ERA5["2m_temperature"]] = 1.0
    era5_merged[:, ERA5["2m_dewpoint_temperature"]] = 0.0
    era5_merged[:, ERA5["mean_sea_level_pressure"]] = 101325.0
    era5_merged[:, ERA5["10m_u_component_of_wind"]] = 1.0
    era5_merged[:, ERA5["10m_v_component_of_wind"]] = 1.0
    era5_merged[:, ERA5["instantaneous_10m_wind_gust"]] = 2.0
    era5_merged[:, ERA5["total_cloud_cover"]] = 0.5
    era5_merged[:, ERA5["downward_uv_radiation_at_the_surface"]] = 100.0
    era5_merged[:, ERA5["total_column_ozone"]] = 0.25
    era5_merged[:, ERA5["total_precipitation"]] = 0.001
    era5_merged[:, ERA5["prob"]] = 75.0
    era5_merged[:, ERA5["surface_solar_radiation_downwards"]] = 120.0
    era5_merged[:, ERA5["convective_available_potential_energy"]] = 50.0
    era5_merged[:, ERA5["surface_pressure"]] = 100000.0

    inputs = prepare_data_inputs(
        source_list=["era5"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=None,
        ecmwf_merged=None,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=era5_merged,
        extra_vars=[],
        num_hours=num_hours,
        lat=40.0,
        lon=-75.0,
    )

    assert inputs["InterThour_inputs"]["era5_ptype"].shape == (num_hours,)
    assert inputs["prcipIntensity_inputs"].shape == (num_hours, 1)
    assert inputs["prcipProbability_inputs"].shape == (num_hours, 1)
    assert inputs["prcipType_inputs"].shape == (num_hours, 1)
    assert inputs["temperature_inputs"].shape[0] == num_hours
    assert inputs["era5_rain_intensity"].shape == (num_hours,)

    assert np.allclose(
        inputs["InterThour_inputs"]["era5_ptype"][:source_hours],
        np.arange(source_hours),
    )
    assert np.isnan(inputs["InterThour_inputs"]["era5_ptype"][source_hours:]).all()
    assert np.allclose(inputs["prcipProbability_inputs"][:source_hours, 0], 0.75)
    assert np.isnan(inputs["prcipProbability_inputs"][source_hours:, 0]).all()
    assert np.isnan(inputs["temperature_inputs"][source_hours:, 0]).all()
    assert np.isnan(inputs["era5_rain_intensity"][source_hours:]).all()
