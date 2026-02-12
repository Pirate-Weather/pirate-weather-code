# NumPy Usage Guide for Pirate Weather

## Overview

The Pirate Weather codebase makes extensive use of [NumPy](https://numpy.org/) for efficient numerical computations and array operations. NumPy is a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

This guide provides a quick reference to the NumPy patterns and operations commonly used throughout the Pirate Weather codebase. For comprehensive documentation, refer to the [official NumPy documentation](https://numpy.org/doc/stable/).

## Why NumPy?

NumPy is essential for Pirate Weather because:
- **Performance**: Operations are vectorized and implemented in C, making them much faster than native Python loops
- **Memory Efficiency**: NumPy arrays use contiguous memory blocks, reducing overhead
- **Array Operations**: Weather data is naturally multi-dimensional (time, latitude, longitude, variables)
- **Broadcasting**: Enables operations on arrays of different shapes without explicit loops

## Common NumPy Operations in Pirate Weather

### 1. Array Creation and Initialization

```python
import numpy as np

# Create an empty array filled with a specific value (common for missing data)
InterPhour = np.full((168, 32), MISSING_DATA)  # 168 hours, 32 variables

# Create an array of zeros (used for temporary calculations)
InterSday = np.zeros(shape=(8, 21))  # 8 days, 21 variables

# Create a time array using arange
hour_array = np.arange(start_time, end_time, datetime.timedelta(hours=1))
```

### 2. Array Indexing and Slicing

Weather data in Pirate Weather is typically stored as 2D arrays where:
- **First dimension**: Time (hours, minutes, or days)
- **Second dimension**: Variables (temperature, humidity, etc.)

```python
# Access a specific variable for all time steps
all_temperatures = InterPhour[:, DATA_HOURLY["temp"]]

# Access a specific time step for all variables
hour_zero_data = InterPhour[0, :]

# Access a specific value
current_temp = InterPhour[0, DATA_HOURLY["temp"]]

# Slice a range of time steps
first_24_hours = InterPhour[0:24, :]
```

### 3. Vectorized Operations

Instead of looping through arrays, NumPy allows operations on entire arrays at once:

```python
# Convert all temperatures from Kelvin to Celsius (vectorized)
temps_celsius = InterPhour[:, DATA_HOURLY["temp"]] - KELVIN_TO_CELSIUS

# Convert to Fahrenheit (all values at once)
temps_fahrenheit = temps_celsius * 9 / 5 + 32

# Apply rounding to all values
temps_rounded = np.round(temps_fahrenheit, 2)
```

### 4. Boolean Indexing and Masking

Filter or modify data based on conditions:

```python
# Find where values are NaN (missing)
nan_mask = np.isnan(temperature_array)

# Replace NaN values with a default
temperature_array[nan_mask] = 0.0

# Set values below threshold to zero
precipitation[precipitation < 0.01] = 0.0

# Conditional operations
# Set negative values to NaN
wind_speed[wind_speed < 0] = np.nan
```

### 5. Priority-Based Model Selection

A common pattern in Pirate Weather is selecting the best available data source:

```python
# Select first non-NaN value from prioritized models
# Priority: HRRR -> NBM -> GFS
merged_temp = np.choose(
    np.argmin([
        np.isnan(hrrr_temp),
        np.isnan(nbm_temp),
        np.isnan(gfs_temp)
    ], axis=0),
    [hrrr_temp, nbm_temp, gfs_temp]
)

# Using np.where for two sources
final_temp = np.where(
    np.isnan(hrrr_temp),  # condition
    gfs_temp,              # value if True
    hrrr_temp              # value if False
)
```

### 6. Statistical Operations

Calculate statistics across time or space:

```python
# Daily statistics from hourly data
daily_max_temp = np.max(hourly_temps[day_mask])
daily_min_temp = np.min(hourly_temps[day_mask])
daily_mean_temp = np.mean(hourly_temps[day_mask])

# Find the index of maximum/minimum
max_temp_hour = np.argmax(hourly_temps[day_mask])
min_temp_hour = np.argmin(hourly_temps[day_mask])

# Sum for accumulations (e.g., precipitation)
daily_precip_total = np.sum(hourly_precip[day_mask])
```

### 7. Array Stacking and Concatenation

Combine data from multiple sources:

```python
# Stack arrays horizontally (side by side)
combined_data = np.column_stack([array1, array2, array3])

# Stack arrays vertically (top to bottom)
extended_forecast = np.vstack([short_term, long_term])

# Create a priority stack for model selection
model_stack = np.column_stack([hrrr_data, nbm_data, gfs_data])
```

### 8. Handling NaN (Missing Data)

```python
# Check if any values are NaN
has_missing = np.any(np.isnan(data_array))

# Count NaN values
num_missing = np.sum(np.isnan(data_array))

# Replace NaN with a specific value
clean_data = np.nan_to_num(data_array, nan=0.0)

# Get non-NaN values
valid_data = data_array[~np.isnan(data_array)]
```

### 9. Mathematical Functions

```python
# Trigonometric functions (for wind direction, solar angles)
wind_u = wind_speed * np.sin(np.radians(wind_direction))
wind_v = wind_speed * np.cos(np.radians(wind_direction))

# Calculate wind speed from components
wind_speed = np.sqrt(u_component**2 + v_component**2)

# Exponential and logarithmic
dew_point = np.log(humidity / 100.0)  # Simplified example

# Clipping values to a range
clipped_values = np.clip(values, min_value, max_value)
```

### 10. Array Reshaping

```python
# Flatten a 2D array to 1D
flat_array = multi_dim_array.flatten()

# Reshape to different dimensions
# Convert hourly data (168,) to weeks × hours (7, 24)
weekly_grid = hourly_data.reshape(7, 24)

# Get array dimensions
num_hours, num_variables = InterPhour.shape
```

## Performance Tips

1. **Avoid Python loops**: Use vectorized NumPy operations instead
   ```python
   # Slow - Python loop
   for i in range(len(temps)):
       temps[i] = temps[i] * 9/5 + 32
   
   # Fast - Vectorized
   temps = temps * 9/5 + 32
   ```

2. **Pre-allocate arrays**: Create arrays with `np.zeros()` or `np.full()` instead of growing them
   ```python
   # Slow - Growing array
   result = []
   for val in data:
       result.append(process(val))
   
   # Fast - Pre-allocated
   result = np.zeros(len(data))
   for i, val in enumerate(data):
       result[i] = process(val)
   ```

3. **Use in-place operations** when possible to save memory:
   ```python
   # Creates new array
   temps = temps + 273.15
   
   # Modifies existing array (saves memory)
   temps += 273.15
   ```

## Common Pitfalls

1. **Integer vs Float Division**: NumPy respects Python's division rules
   ```python
   # Returns integer (wrong for temperatures!)
   result = np.array([5, 9]) / 2  # array([2, 4])
   
   # Returns float (correct)
   result = np.array([5, 9]) / 2.0  # array([2.5, 4.5])
   ```

2. **NaN Comparisons**: NaN values don't equal themselves
   ```python
   # Wrong - always False
   if value == np.nan:
   
   # Correct
   if np.isnan(value):
   ```

3. **Boolean Indexing Returns Copy**: Modifying won't affect original
   ```python
   # This doesn't modify the original array
   temps[temps < 0] = temps[temps < 0] + 273.15
   
   # This does
   mask = temps < 0
   temps[mask] += 273.15
   ```

## Key Modules Used in Pirate Weather

- **Data Processing**: `API/data_inputs.py`, `API/forecast_sources.py`
- **Hourly Forecasts**: `API/hourly/block.py`, `API/hourly/builder.py`
- **Daily Aggregations**: `API/daily/builder.py`
- **Utilities**: `API/api_utils.py`, `API/utils/`

## Additional Resources

- [NumPy Official Documentation](https://numpy.org/doc/stable/)
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [Array Programming with NumPy](https://www.nature.com/articles/s41586-020-2649-2) - The Nature paper describing NumPy
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) - Free online book

## Finding NumPy Usage in the Codebase

To see real examples of NumPy usage in Pirate Weather:

```bash
# Find files using NumPy
grep -r "import numpy" API/

# Find specific NumPy operations
grep -r "np.where\|np.choose\|np.isnan" API/

# See vectorized conversions
grep -r "vectorized" API/
```

Look at these files for extensive NumPy examples:
- `API/data_inputs.py` - Priority stacking and model merging
- `API/hourly/block.py` - Vectorized unit conversions
- `API/daily/builder.py` - Statistical aggregations
- `API/api_utils.py` - General array operations
