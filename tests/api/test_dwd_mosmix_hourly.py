"""DWD MOSMIX unit handling on the way into the hourly inputs."""

import numpy as np

from API.constants.api_const import PRECIP_IDX
from API.constants.model_const import DWD_MOSMIX
from tests.helpers.model_arrays import hourly_inputs, model_array


def test_dwd_mosmix_units_and_present_weather_reach_hourly_inputs():
    """DWD is the only source that reports cloud cover in percent and precip type
    as WMO 4677 present-weather codes; without conversion it would publish
    cloudCover 95 instead of 0.95 or a snow code above freezing.
    """
    dwd = model_array(DWD_MOSMIX, 2, cloud=95.0, vis=4500.0, ptype=71.0)  # 71: snow
    dwd[:, DWD_MOSMIX["temp"]] = [-5.0, 8.0]

    inputs = hourly_inputs(
        lat=52.52, lon=13.405, source_list=["dwd_mosmix"], dwd_mosmix=dwd
    )

    np.testing.assert_allclose(inputs["cloud_inputs"][:, 0], [0.95, 0.95])
    np.testing.assert_array_equal(inputs["vis_inputs"][:, 0], [4500.0, 4500.0])
    np.testing.assert_array_equal(inputs["temperature_inputs"][:, 0], [-5.0, 8.0])
    # A snow code at 8 C is implausible and is re-labelled as rain.
    np.testing.assert_array_equal(
        inputs["prcipType_inputs"][:, 0], [PRECIP_IDX["snow"], PRECIP_IDX["rain"]]
    )
