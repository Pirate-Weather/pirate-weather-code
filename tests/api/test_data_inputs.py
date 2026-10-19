"""Model data preparation for the hourly block: input stacking, length
normalization, and Kelvin to Celsius conversion."""

import numpy as np
import pytest

from API.constants.api_const import PRECIP_IDX
from API.constants.model_const import ERA5, GDPS, GEPS, HRDPS, REPS, URMA
from API.data_inputs import prepare_data_inputs

# Importing responseLocal only touches the network when STAGE is TESTING/TM_TESTING.
from API.responseLocal import convert_data_to_celsius


def test_prepare_data_inputs_normalizes_short_era5_series_to_num_hours():
    """Regression for #624: extended time machine requests can get an ERA5 slice
    shorter than num_hours; every input must still be num_hours long (padded
    with NaN) so the hourly block does not fail on mismatched shapes."""
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
    """Canadian models must feed the hourly inputs with HRDPS ahead of GDPS and
    REPS ahead of GEPS, with their native units handled: cloud % -> fraction and
    the UV index used directly (no GFS-style flux scaling)."""
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
    geps_merged[:, GEPS["prob"]] = 0.65
    geps_merged[:, GEPS["accum"]] = 1.5
    geps_merged[:, GEPS["error"]] = 0.6
    geps_merged[:, GEPS["freezing_rain"]] = 0.2
    geps_merged[:, GEPS["ice"]] = 0.1
    geps_merged[:, GEPS["rain"]] = 0.6
    geps_merged[:, GEPS["snow"]] = 0.4

    reps_merged = np.full((num_hours, max(REPS.values()) + 1), np.nan)
    reps_merged[:, REPS["prob"]] = 0.70
    reps_merged[:, REPS["accum"]] = 1.8
    reps_merged[:, REPS["error"]] = 0.5
    reps_merged[:, REPS["freezing_rain"]] = 0.3
    reps_merged[:, REPS["ice"]] = 0.2
    reps_merged[:, REPS["rain"]] = 0.7
    reps_merged[:, REPS["snow"]] = 0.5

    inputs = prepare_data_inputs(
        source_list=["hrdps", "gdps", "geps", "reps"],
        nbm_merged=None,
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

    def first_choice(name):
        return inputs[name][:, 0]

    np.testing.assert_array_equal(first_choice("temperature_inputs"), 12.0)
    np.testing.assert_array_equal(first_choice("pressure_inputs"), 100000.0)
    np.testing.assert_array_equal(first_choice("wind_inputs"), 4.0)
    np.testing.assert_array_equal(first_choice("gust_inputs"), 5.0)
    np.testing.assert_allclose(first_choice("cloud_inputs"), 0.4)
    np.testing.assert_array_equal(first_choice("solar_inputs"), 300.0)
    np.testing.assert_array_equal(first_choice("cape_inputs"), 420.0)
    np.testing.assert_array_equal(first_choice("prcipProbability_inputs"), 0.70)
    np.testing.assert_array_equal(first_choice("accum_inputs"), 1.8)
    # REPS rain is the only component rate above the 0.5 threshold.
    np.testing.assert_array_equal(first_choice("prcipType_inputs"), PRECIP_IDX["rain"])
    np.testing.assert_array_equal(inputs["uv_inputs"], [[2.5, 1.5]] * num_hours)


@pytest.mark.parametrize(
    ("argument", "model_idx"),
    [
        pytest.param("dataOut_urma", URMA, id="urma"),
        pytest.param("dataOut_hrdps", HRDPS, id="hrdps"),
        pytest.param("dataOut_gdps", GDPS, id="gdps"),
    ],
)
def test_convert_data_to_celsius_converts_model(argument, model_idx):
    """Models ingested in Kelvin must be converted before use; a model missing
    from the conversion list would report temperatures near 290 °C."""
    data = np.full((2, max(model_idx.values()) + 1), np.nan)
    data[:, model_idx["temp"]] = 288.15
    data[:, model_idx["dew"]] = 283.15
    model_arrays = dict.fromkeys(
        (
            "dataOut",
            "dataOut_h2",
            "dataOut_hrrrh",
            "dataOut_nbm",
            "dataOut_gfs",
            "dataOut_ecmwf",
            "dataOut_rtma_ru",
            "era5_merged",
            "dataOut_dwd_mosmix",
            "dataOut_aigfs",
            "dataOut_aifs",
            "dataOut_hrdps",
            "dataOut_gdps",
        )
    )
    model_arrays[argument] = data

    convert_data_to_celsius(**model_arrays)

    np.testing.assert_allclose(data[:, model_idx["temp"]], 15.0)
    np.testing.assert_allclose(data[:, model_idx["dew"]], 10.0)
