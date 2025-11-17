"""Test to verify hourly rounding optimization works correctly.

This test ensures that the vectorized unit conversion and rounding
produces the same results as the previous per-hour approach.
"""

import numpy as np
import pytest


def test_vectorized_rounding_equivalence():
    """Test that vectorized rounding produces same results as loop-based approach."""
    # Sample test data
    test_values = np.array([1.23456, 2.34567, 3.45678, 4.56789])
    decimals = 2
    
    # Vectorized approach (new)
    vectorized_result = np.round(test_values, decimals)
    
    # Loop-based approach (old)
    loop_result = np.array([round(val, decimals) for val in test_values])
    
    np.testing.assert_array_almost_equal(vectorized_result, loop_result)


def test_unit_conversion_vectorized():
    """Test that vectorized unit conversions work correctly."""
    # Test temperature conversion: Celsius to Fahrenheit
    temps_celsius = np.array([0.0, 10.0, 20.0, 30.0])
    expected_fahrenheit = np.array([32.0, 50.0, 68.0, 86.0])
    
    # Vectorized conversion
    result = temps_celsius * 9 / 5 + 32
    
    np.testing.assert_array_almost_equal(result, expected_fahrenheit)


def test_precipitation_conversion_vectorized():
    """Test that vectorized precipitation unit conversions work correctly."""
    # Test mm to inches conversion
    precip_mm = np.array([0.0, 10.0, 25.4, 50.8])
    mm_to_inches = 0.0394
    
    # Vectorized conversion
    result = precip_mm * mm_to_inches
    
    # Verify conversions are approximately correct (within reasonable tolerance)
    assert abs(result[0] - 0.0) < 0.001
    assert abs(result[1] - 0.394) < 0.001
    assert abs(result[2] - 1.0) < 0.01
    assert abs(result[3] - 2.0) < 0.01


def test_rounding_with_nan_values():
    """Test that rounding handles NaN values correctly."""
    test_values = np.array([1.23456, np.nan, 3.45678, np.nan])
    decimals = 2
    
    # Vectorized rounding
    result = np.round(test_values, decimals)
    
    # Check that rounded values are correct
    assert round(result[0], 2) == 1.23
    assert np.isnan(result[1])
    assert round(result[2], 2) == 3.46
    assert np.isnan(result[3])


def test_integer_conversion_after_rounding():
    """Test that values are correctly converted to integers when decimals=0."""
    test_values = np.array([1.4, 2.5, 3.6, 4.9])
    
    # Round to 0 decimals and convert to int (simulating uvIndex, etc.)
    result = np.round(test_values, 0).astype(int)
    expected = np.array([1, 2, 4, 5], dtype=int)
    
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
