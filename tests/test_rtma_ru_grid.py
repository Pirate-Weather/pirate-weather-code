"""Tests for RTMA-RU grid coordinate calculations."""

import math

from API.constants.grid_const import (
    RTMA_RU_AXIS,
    RTMA_RU_CENTRAL_LAT,
    RTMA_RU_CENTRAL_LONG,
    RTMA_RU_DELTA,
    RTMA_RU_MIN_X,
    RTMA_RU_MIN_Y,
    RTMA_RU_PARALLEL,
)


def lambert_grid_match(
    central_longitude,
    central_latitude,
    standard_parallel,
    semimajor_axis,
    lat,
    lon,
    min_x,
    min_y,
    delta,
):
    """
    Calculate grid position using Lambert Conformal Conic projection.

    This is a simplified version of the lambertGridMatch function from responseLocal.py
    for testing purposes.
    """
    hrr_n = math.sin(standard_parallel)
    hrrr_F = (
        math.cos(standard_parallel)
        * (math.tan(0.25 * math.pi + 0.5 * standard_parallel)) ** hrr_n
    ) / hrr_n
    hrrr_p = (
        semimajor_axis
        * hrrr_F
        * 1
        / (math.tan(0.25 * math.pi + 0.5 * math.radians(lat)) ** hrr_n)
    )
    hrrr_p0 = (
        semimajor_axis
        * hrrr_F
        * 1
        / (math.tan(0.25 * math.pi + 0.5 * central_latitude) ** hrr_n)
    )

    x_loc = hrrr_p * math.sin(hrr_n * (math.radians(lon) - central_longitude))
    y_loc = hrrr_p0 - hrrr_p * math.cos(hrr_n * (math.radians(lon) - central_longitude))

    x_grid = round((x_loc - min_x) / delta)
    y_grid = round((y_loc - min_y) / delta)

    return x_grid, y_grid


def test_rtma_ru_grid_matches_nbm():
    """
    Test that RTMA-RU uses the same grid parameters as NBM.

    RTMA-RU and NBM share the same NDFD grid (Lambert Conformal Conic
    with 2539.703 meter resolution). This test ensures the RTMA_RU_DELTA
    constant matches NBM's grid resolution.
    """
    # NBM grid delta (the reference standard)
    nbm_delta = 2539.703000

    # RTMA-RU should use the same delta
    assert RTMA_RU_DELTA == nbm_delta, (
        f"RTMA_RU_DELTA ({RTMA_RU_DELTA}) should match NBM grid delta ({nbm_delta})"
    )


def test_vancouver_coordinates():
    """
    Test grid coordinate calculation for Vancouver, BC.

    This test case is based on the issue report where Vancouver coordinates
    (49.245, -123.115) were returning incorrect temperature data due to
    wrong grid cell selection.
    """
    # Vancouver coordinates from issue report
    lat = 49.245
    lon = -123.115

    # Calculate RTMA-RU grid position
    x_rtma, y_rtma = lambert_grid_match(
        central_longitude=math.radians(RTMA_RU_CENTRAL_LONG),
        central_latitude=math.radians(RTMA_RU_CENTRAL_LAT),
        standard_parallel=math.radians(RTMA_RU_PARALLEL),
        semimajor_axis=RTMA_RU_AXIS,
        lat=lat,
        lon=lon,
        min_x=RTMA_RU_MIN_X,
        min_y=RTMA_RU_MIN_Y,
        delta=RTMA_RU_DELTA,
    )

    # Calculate NBM grid position (should match)
    nbm_delta = 2539.703000
    x_nbm, y_nbm = lambert_grid_match(
        central_longitude=math.radians(265.0),
        central_latitude=math.radians(25.0),
        standard_parallel=math.radians(25.0),
        semimajor_axis=6371200,
        lat=lat,
        lon=lon,
        min_x=-3271152.8,
        min_y=-263793.46,
        delta=nbm_delta,
    )

    # RTMA-RU and NBM should produce the same grid coordinates
    assert x_rtma == x_nbm, (
        f"Vancouver RTMA-RU x={x_rtma} should match NBM x={x_nbm}"
    )
    assert y_rtma == y_nbm, (
        f"Vancouver RTMA-RU y={y_rtma} should match NBM y={y_nbm}"
    )

    # Expected grid position for Vancouver with correct delta
    assert x_rtma == 109, f"Expected x=109, got x={x_rtma}"
    assert y_rtma == 9601, f"Expected y=9601, got y={y_rtma}"


def test_rtma_ru_grid_consistency():
    """Test that RTMA-RU grid parameters are consistent with NBM."""
    # All projection parameters should match NBM
    assert RTMA_RU_CENTRAL_LONG == 265.0
    assert RTMA_RU_CENTRAL_LAT == 25.0
    assert RTMA_RU_PARALLEL == 25.0
    assert RTMA_RU_AXIS == 6371200
    assert RTMA_RU_MIN_X == -3271152.8
    assert RTMA_RU_MIN_Y == -263793.46
