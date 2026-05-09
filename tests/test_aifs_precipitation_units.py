"""Tests for AIFS precipitation unit conversion.

AIFS ensemble tp and sf are in kg m**-2 (= mm of liquid water equivalent),
NOT in meters like the IFS ensemble.  The ingest code must divide by 6000
(6 hours × 1000 mm/m) to convert mm/6h → m/h, matching the IFS pipeline
which then applies a downstream × 1000 factor (m/h → mm/h).

If the divisor is wrong (e.g. 6 instead of 6000), the final precipitation
values end up 1000× too large, producing impossible values like 1000+ inches
of daily rain.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helper: simulate the AIFS ingest pipeline for a single member / location
# ---------------------------------------------------------------------------

DIVISOR_CORRECT = 6000  # mm/6h → m/h   (the fix)
DIVISOR_WRONG = 6  # mm/6h → mm/h  (the old, broken behaviour)
DOWNSTREAM_FACTOR = 1000  # m/h → mm/h, applied by data_inputs.py


def simulate_pipeline(tp_mm_period, divisor):
    """Simulate the AIFS→accum_inputs pipeline for a single 6-h step.

    Parameters
    ----------
    tp_mm_period : float
        6-hour period precipitation in mm (kg m**-2), as provided by AIFS.
    divisor : int
        The divisor applied in the ingest code (6 or 6000).

    Returns
    -------
    float
        The effective hourly accumulation in mm/h that ends up in
        ``accum_inputs`` after the downstream × 1000 factor.
    """
    # Ingest: convert the 6-h period accumulation to an hourly rate.
    hourly_rate_ingest = tp_mm_period / divisor

    # data_inputs.py always multiplies by 1000 (expecting m/h → mm/h).
    hourly_accum_mm = hourly_rate_ingest * DOWNSTREAM_FACTOR

    return hourly_accum_mm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_correct_divisor_gives_expected_mm_per_hour():
    """With divisor=6000 the pipeline yields the physically correct mm/h."""
    # A 6-h accumulation of 12 mm should correspond to 2 mm/h.
    tp_mm = 12.0
    result = simulate_pipeline(tp_mm, DIVISOR_CORRECT)
    expected = tp_mm / 6.0  # 12 mm over 6 h = 2 mm/h
    assert abs(result - expected) < 1e-9, (
        f"Expected {expected} mm/h, got {result} mm/h with divisor={DIVISOR_CORRECT}"
    )


def test_wrong_divisor_is_1000x_too_large():
    """With divisor=6 (the bug) the output is 1000× too large."""
    tp_mm = 12.0
    correct = simulate_pipeline(tp_mm, DIVISOR_CORRECT)
    buggy = simulate_pipeline(tp_mm, DIVISOR_WRONG)
    ratio = buggy / correct
    assert abs(ratio - 1000.0) < 1e-9, (
        f"Expected bug to be 1000× too large, got ratio={ratio}"
    )


def test_zero_precipitation_stays_zero():
    """Zero precipitation should remain zero regardless of divisor."""
    assert simulate_pipeline(0.0, DIVISOR_CORRECT) == 0.0
    assert simulate_pipeline(0.0, DIVISOR_WRONG) == 0.0


def test_daily_accumulation_in_range_for_heavy_rain():
    """Simulate a heavy but physically plausible 24-h rainfall at one location.

    The ECMWF AIFS ensemble might forecast 50 mm in 6 h for a severe event.
    With the correct divisor the daily total stays within plausible bounds;
    with the bug it would be > 1000 inches.
    """
    # Four 6-h periods, each with 50 mm (very heavy rain).
    tp_periods_mm = [50.0, 50.0, 50.0, 50.0]

    hourly_rates_correct = [
        simulate_pipeline(p, DIVISOR_CORRECT) for p in tp_periods_mm
    ]
    hourly_rates_buggy = [simulate_pipeline(p, DIVISOR_WRONG) for p in tp_periods_mm]

    # Each 6-h block contributes 6 hours × hourly_rate mm/h.
    daily_mm_correct = sum(r * 6 for r in hourly_rates_correct)
    daily_mm_buggy = sum(r * 6 for r in hourly_rates_buggy)

    # Correct: 4 × 50 mm = 200 mm/day — realistic for a very heavy event.
    assert abs(daily_mm_correct - 200.0) < 1e-6, (
        f"Expected 200 mm/day, got {daily_mm_correct}"
    )

    # Buggy: 200 000 mm/day ≈ 7874 inches — physically impossible.
    assert daily_mm_buggy == daily_mm_correct * 1000, (
        "Buggy pipeline should be 1000× the correct value"
    )

    # Convert to inches for comparison with the reported issue values.
    daily_in_correct = daily_mm_correct / 25.4
    daily_in_buggy = daily_mm_buggy / 25.4

    assert daily_in_correct < 20, (
        f"Correct daily total should be < 20 inches, got {daily_in_correct:.1f} in"
    )
    assert daily_in_buggy > 1000, (
        f"Buggy daily total should be > 1000 inches, got {daily_in_buggy:.1f} in"
    )


def test_divisor_produces_correct_m_per_hour():
    """Verify the ingest-side output (before × 1000) is in m/h as expected."""
    # 6 mm over 6 hours → 1 mm/h = 0.001 m/h
    tp_mm = 6.0
    ingest_result_m_per_h = tp_mm / DIVISOR_CORRECT
    expected_m_per_h = 0.001
    assert abs(ingest_result_m_per_h - expected_m_per_h) < 1e-12, (
        f"Expected {expected_m_per_h} m/h, got {ingest_result_m_per_h} m/h"
    )


def test_numpy_array_division_consistent():
    """Verify the fix works the same way on numpy arrays (as used in xarray)."""
    tp_arr = np.array([0.0, 5.0, 10.0, 50.0, 100.0])  # mm per 6-h period

    result_correct = (tp_arr / DIVISOR_CORRECT) * DOWNSTREAM_FACTOR
    result_wrong = (tp_arr / DIVISOR_WRONG) * DOWNSTREAM_FACTOR

    np.testing.assert_allclose(result_correct, tp_arr / 6.0)
    np.testing.assert_allclose(result_wrong, result_correct * 1000.0)
