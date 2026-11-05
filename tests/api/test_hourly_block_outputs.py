"""Hourly block output values: ERA5 precipitation probability and temperature
clipping."""

import numpy as np
import pytest

from API.constants.clip_const import CLIP_TEMP
from API.constants.model_const import ERA5
from API.data_inputs import prepare_data_inputs
from API.hourly.block import build_hourly_block
from tests.helpers.hourly import hourly_block_kwargs


def test_build_hourly_block_outputs_era5_precip_probability():
    """Regression for #664: ERA5 stores precipitation probability as 0-100, and
    time machine responses must report it as the API's 0-1 fraction."""
    num_hours = 3
    era5_merged = np.zeros((num_hours, max(ERA5.values()) + 1))
    era5_merged[:, ERA5["prob"]] = [0.0, 56.0, 100.0]
    era5_merged[:, ERA5["mean_sea_level_pressure"]] = 101325.0
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

    hour_list, *_ = build_hourly_block(
        **hourly_block_kwargs(
            num_hours, source_list=["era5"], timeMachine=True, **inputs
        )
    )

    assert [hour["precipProbability"] for hour in hour_list] == [0.0, 0.56, 1.0]


@pytest.mark.parametrize(
    ("model_temp", "expected_temp"),
    [
        pytest.param(
            CLIP_TEMP["max"] + 1.0, CLIP_TEMP["max"], id="slightly_high_clamped"
        ),
        # More than 25 % above the limit is treated as corrupt data, not clamped.
        pytest.param(1000.0, np.nan, id="implausible_becomes_missing"),
    ],
)
def test_hourly_temperature_clipping(model_temp, expected_temp):
    """A model temperature slightly past the plausible range is clamped to the
    limit, but a wildly wrong one must be reported as missing instead of as a
    believable 60 °C."""
    num_hours = 3

    hour_list, *_ = build_hourly_block(
        **hourly_block_kwargs(
            num_hours, temperature_inputs=np.full((num_hours, 1), model_temp)
        )
    )

    np.testing.assert_array_equal(
        [hour["temperature"] for hour in hour_list], np.full(num_hours, expected_temp)
    )
