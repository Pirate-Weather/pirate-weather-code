"""Test that grid constants are correctly set for RTMA-RU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from API.constants.grid_const import (
    NBM_X_MAX,
    NBM_Y_MAX,
    RTMA_RU_X_MAX,
    RTMA_RU_X_MIN,
    RTMA_RU_Y_MAX,
    RTMA_RU_Y_MIN,
)


def test_rtma_ru_grid_dimensions():
    """Test that RTMA-RU grid dimensions are correct.

    According to the netCDF metadata from rtma2p5_ru.t1730z.2dvaranl_ndfd.grb2:
    - dimensions: x = 2345; y = 1597;

    With 0-based indexing in numpy/zarr arrays:
    - Valid x indices: 0 to 2344
    - Valid y indices: 0 to 1596

    With the MIN threshold of 1 (matching NBM pattern):
    - RTMA_RU_X_MIN = 1, RTMA_RU_X_MAX = 2344
    - RTMA_RU_Y_MIN = 1, RTMA_RU_Y_MAX = 1596
    """
    # RTMA-RU should have same max dimensions as NBM since they share
    # the same Lambert Conformal Conic projection and grid
    assert RTMA_RU_X_MAX == NBM_X_MAX == 2344
    assert RTMA_RU_Y_MAX == NBM_Y_MAX == 1596

    # MIN values should be 1 to match the pattern of other grids
    assert RTMA_RU_X_MIN == 1
    assert RTMA_RU_Y_MIN == 1


def test_rtma_ru_grid_valid_range():
    """Test that the valid index range is reasonable.

    The actual grid has 2345 x 1597 points, so with indices starting
    at 1 and max at 2344/1596, we have 2344 valid x indices and
    1596 valid y indices, which leaves a small margin but is acceptable
    given the MIN threshold of 1.
    """
    x_range = RTMA_RU_X_MAX - RTMA_RU_X_MIN + 1
    y_range = RTMA_RU_Y_MAX - RTMA_RU_Y_MIN + 1

    # Should be close to the actual dimensions (2345 x 1597)
    # With MIN=1, we lose the 0 index, so range should be 2344 x 1596
    assert x_range == 2344
    assert y_range == 1596
