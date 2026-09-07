import numpy as np
import pytest

from API.api_utils import map_canadian_precip_type_to_ptype
from API.constants.model_const import ERA5, GDPS, GEPS, HRDPS, REPS
from API.data_inputs import _normalize_length, prepare_data_inputs
from API.responseLocal import convert_data_to_celsius


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


def test_prepare_data_inputs_includes_canadian_model_fields():
    num_hours = 3

    hrdps_merged = np.full((num_hours, max(HRDPS.values()) + 1), np.nan)
    hrdps_merged[:, HRDPS["temp"]] = 12.0
    hrdps_merged[:, HRDPS["dew"]] = 8.0
    hrdps_merged[:, HRDPS["rh"]] = 70.0
    hrdps_merged[:, HRDPS["wind"]] = 4.0
    hrdps_merged[:, HRDPS["wind_dir"]] = 90.0
    hrdps_merged[:, HRDPS["gust"]] = 5.0
    hrdps_merged[:, HRDPS["pressure"]] = 100000.0
    hrdps_merged[:, HRDPS["cloud"]] = 40.0
    hrdps_merged[:, HRDPS["uv"]] = 2.5
    hrdps_merged[:, HRDPS["solar"]] = 300.0
    hrdps_merged[:, HRDPS["cape"]] = 420.0
    hrdps_merged[:, HRDPS["intensity"]] = 1.0
    hrdps_merged[:, HRDPS["accum"]] = 3.0
    hrdps_merged[:, HRDPS["ptype"]] = 1.0

    gdps_merged = np.full((num_hours, max(GDPS.values()) + 1), np.nan)
    gdps_merged[:, GDPS["temp"]] = 11.0
    gdps_merged[:, GDPS["dew"]] = 7.0
    gdps_merged[:, GDPS["rh"]] = 68.0
    gdps_merged[:, GDPS["wind"]] = 3.5
    gdps_merged[:, GDPS["wind_dir"]] = 180.0
    gdps_merged[:, GDPS["gust"]] = 4.5
    gdps_merged[:, GDPS["station_pressure"]] = 101000.0
    gdps_merged[:, GDPS["cloud"]] = 50.0
    gdps_merged[:, GDPS["uv"]] = 1.5
    gdps_merged[:, GDPS["solar"]] = 280.0
    gdps_merged[:, GDPS["ozone"]] = 45.0
    gdps_merged[:, GDPS["cape"]] = 380.0
    gdps_merged[:, GDPS["intensity"]] = 0.8
    gdps_merged[:, GDPS["accum"]] = 2.5
    gdps_merged[:, GDPS["type"]] = 2.0

    geps_merged = np.full((num_hours, max(GEPS.values()) + 1), np.nan)
    geps_merged[:, GEPS["prob"]] = 65.0
    geps_merged[:, GEPS["accum"]] = 1.5
    geps_merged[:, GEPS["error"]] = 0.6
    geps_merged[:, GEPS["freezing_rain"]] = 0.2
    geps_merged[:, GEPS["ice"]] = 0.1
    geps_merged[:, GEPS["rain"]] = 0.6
    geps_merged[:, GEPS["snow"]] = 0.4

    reps_merged = np.full((num_hours, max(REPS.values()) + 1), np.nan)
    reps_merged[:, REPS["prob"]] = 70.0
    reps_merged[:, REPS["accum"]] = 1.8
    reps_merged[:, REPS["error"]] = 0.5
    reps_merged[:, REPS["freezing_rain"]] = 0.3
    reps_merged[:, REPS["ice"]] = 0.2
    reps_merged[:, REPS["rain"]] = 0.7
    reps_merged[:, REPS["snow"]] = 0.5

    inputs = prepare_data_inputs(
        source_list=["hrdps", "gdps", "geps", "reps"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrdps_merged=hrdps_merged,
        reps_merged=reps_merged,
        hrrr_merged=None,
        dwd_mosmix_merged=None,
        ecmwf_merged=None,
        gdps_merged=gdps_merged,
        geps_merged=geps_merged,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=45.0,
        lon=-75.0,
    )

    assert np.allclose(
        inputs["temperature_inputs"][:, 0], hrdps_merged[:, HRDPS["temp"]]
    )
    assert np.allclose(
        inputs["pressure_inputs"][:, 0], hrdps_merged[:, HRDPS["pressure"]]
    )
    assert np.allclose(inputs["wind_inputs"][:, 0], hrdps_merged[:, HRDPS["wind"]])
    assert np.allclose(inputs["gust_inputs"][:, 0], hrdps_merged[:, HRDPS["gust"]])
    assert np.allclose(
        inputs["cloud_inputs"][:, 0], hrdps_merged[:, HRDPS["cloud"]] * 0.01
    )
    assert np.allclose(inputs["solar_inputs"][:, 0], hrdps_merged[:, HRDPS["solar"]])
    assert np.allclose(inputs["cape_inputs"][:, 0], hrdps_merged[:, HRDPS["cape"]])
    assert np.allclose(
        inputs["prcipProbability_inputs"][:, 0], geps_merged[:, GEPS["prob"]] * 0.01
    )
    assert np.allclose(inputs["accum_inputs"][:, 0], hrdps_merged[:, HRDPS["accum"]])
    expected_hrdps_ptype = map_canadian_precip_type_to_ptype(
        np.round(hrdps_merged[:, HRDPS["ptype"]])
    )
    assert np.allclose(inputs["prcipType_inputs"][:, 0], expected_hrdps_ptype)


def test_convert_data_to_celsius_handles_canadian_models():
    hrdps = np.full((3, max(HRDPS.values()) + 1), np.nan)
    hrdps[:, HRDPS["temp"]] = 295.15
    hrdps[:, HRDPS["dew"]] = 288.15

    gdps = np.full((3, max(GDPS.values()) + 1), np.nan)
    gdps[:, GDPS["temp"]] = 300.15
    gdps[:, GDPS["dew"]] = 293.15

    convert_data_to_celsius(
        dataOut=None,
        dataOut_h2=None,
        dataOut_hrrrh=None,
        dataOut_nbm=None,
        dataOut_gfs=None,
        dataOut_ecmwf=None,
        dataOut_rtma_ru=None,
        era5_merged=None,
        dataOut_dwd_mosmix=None,
        dataOut_aigfs=None,
        dataOut_aifs=None,
        dataOut_hrdps=hrdps,
        dataOut_gdps=gdps,
    )

    hrdps[:, HRDPS["temp"]] = 295.15
    hrdps[:, HRDPS["dew"]] = 288.15
    gdps[:, GDPS["temp"]] = 300.15
    gdps[:, GDPS["dew"]] = 293.15

    convert_data_to_celsius(
        dataOut=None,
        dataOut_h2=None,
        dataOut_hrrrh=None,
        dataOut_nbm=None,
        dataOut_gfs=None,
        dataOut_ecmwf=None,
        dataOut_rtma_ru=None,
        era5_merged=None,
        dataOut_dwd_mosmix=None,
        dataOut_aigfs=None,
        dataOut_aifs=None,
        dataOut_hrdps=hrdps,
        dataOut_gdps=gdps,
    )

    assert np.allclose(hrdps[:, HRDPS["temp"]], 22.0)
    assert np.allclose(hrdps[:, HRDPS["dew"]], 15.0)
    assert np.allclose(gdps[:, GDPS["temp"]], 27.0)
    assert np.allclose(gdps[:, GDPS["dew"]], 20.0)


def test_prepare_data_inputs_keeps_canadian_uv_index_direct():
    num_hours = 2
    hrdps_merged = np.full((num_hours, max(HRDPS.values()) + 1), np.nan)
    hrdps_merged[:, HRDPS["uv"]] = 4.5

    gdps_merged = np.full((num_hours, max(GDPS.values()) + 1), np.nan)
    gdps_merged[:, GDPS["uv"]] = 5.5

    inputs = prepare_data_inputs(
        source_list=["hrdps", "gdps"],
        nbm_merged=None,
        nbm_fire_merged=None,
        hrdps_merged=hrdps_merged,
        reps_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=None,
        ecmwf_merged=None,
        gdps_merged=gdps_merged,
        geps_merged=None,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=45.0,
        lon=-75.0,
    )

    assert np.allclose(inputs["uv_inputs"][:, 0], np.array([4.5, 4.5]))
    assert np.allclose(inputs["uv_inputs"][:, 1], np.array([5.5, 5.5]))
