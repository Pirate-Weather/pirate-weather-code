import numpy as np

from API.PirateTextHelper import estimate_snow_density, estimate_snow_height


def test_snow_density_and_height_golden_values():
    """Pin the new-snow density formula, which turns liquid precipitation into
    the snow depth behind every snow accumulation forecast and its error range.

    Covers the warm branch (with and without wind compaction) and the cold
    branch below -13 C, as arrays because the API calls it vectorised. A silent
    change here shifts every snow total.
    """
    temperatures_c = np.array([0.0, -5.0, 2.0, -20.0])
    wind_speeds_mps = np.array([0.0, 5.0, 10.0, 2.0])
    liquid_mm = np.array([10.0, 5.0, 15.0, 4.0])

    np.testing.assert_allclose(
        estimate_snow_density(temperatures_c, wind_speeds_mps),
        [118.3810, 119.3624, 285.6412, 59.5971],
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        estimate_snow_height(liquid_mm, temperatures_c, wind_speeds_mps),
        [84.47300, 41.88923, 52.51342, 67.11742],
        rtol=1e-6,
    )
