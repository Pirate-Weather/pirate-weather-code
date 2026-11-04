"""clipLog keeps every forecast field in range, so its mistakes reach the response."""

import numpy as np
import pytest

from API.api_utils import clipLog


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param([40.0, 81.0, 1000.0], [40.0, 80.0, np.nan], id="above-max"),
        pytest.param([40.0, 9.0, -100.0], [40.0, 10.0, np.nan], id="below-min"),
        pytest.param([np.nan, 40.0, 1000.0], [np.nan, 40.0, np.nan], id="beside-a-nan"),
        pytest.param([81.0, 9.0], [80.0, 10.0], id="only-clamping"),
    ],
)
def test_only_implausible_values_become_missing(data, expected):
    """One bad value used to blank every value past the limit, so a 1000 C hour also
    lost its 61 C neighbours, which should clamp. A NaN in the array did the reverse
    by turning the check off, and 1000 C was published as a plausible 60 C. Range 10
    to 80: the plausible limits are 5.5 and 100.
    """
    result = clipLog(np.array(data), 10.0, 80.0, "test")

    np.testing.assert_allclose(result, expected)


def test_negative_minimum_has_no_clamping_margin():
    """0.75 * min is inside the range when the minimum is negative, so it can't be
    the plausible limit: valid values must pass untouched and anything lower is
    missing, as before.
    """
    result = clipLog(np.array([-85.0, -91.0, 30.0]), -90.0, 60.0, "test")

    np.testing.assert_allclose(result, [-85.0, np.nan, 30.0])


def test_scalar_values():
    """currently fields pass single values rather than arrays."""
    assert clipLog(61.0, -90.0, 60.0, "test") == 60.0
    assert np.isnan(clipLog(1000.0, -90.0, 60.0, "test"))
