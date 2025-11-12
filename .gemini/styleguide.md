# Pirate Weather Python Style Guide

# Introduction
This style guide outlines the coding conventions for Python code developed at Pirate Weather.
It's based on PEP 8, but with some modifications to address specific needs and
preferences within our organization.

# Key Principles
* **Readability:** Code should be easy to understand for all team members.
* **Maintainability:** Code should be easy to modify and extend.
* **Consistency:** Adhering to a consistent style across all projects improves
  collaboration and reduces errors.
* **Performance:** While readability is paramount, code should be efficient.

# Deviations from PEP 8

## Line Length
* **Maximum line length:** 120 characters (instead of PEP 8's 79).
    * Modern screens allow for wider lines, improving code readability in many cases.
    * Many common patterns in our codebase, like long strings or URLs, often exceed 79 characters.

## Indentation
* **Use 4 spaces per indentation level.** (PEP 8 recommendation)

## Imports
* **Group imports:**
    * Standard library imports
    * Related third party imports
    * Local application/library specific imports
* **Absolute imports:** Always use absolute imports for clarity.
* **Import order within groups:**  Sort alphabetically.

## Naming Conventions

* **Variables:** Use lowercase with underscores (snake_case): `user_name`, `total_count`
* **Constants:**  Use uppercase with underscores: `MAX_VALUE`, `DATABASE_NAME`
* **Functions:** Use lowercase with underscores (snake_case): `calculate_total()`, `process_data()`
* **Classes:** Use CapWords (CamelCase): `UserManager`, `PaymentProcessor`
* **Modules:** Use lowercase with underscores (snake_case): `user_utils`, `payment_gateway`

## Docstrings
* **Use triple double quotes (`"""Docstring goes here."""`) for all docstrings.**
* **First line:** Concise summary of the object's purpose.
* **For complex functions/classes:** Include detailed descriptions of parameters, return values,
  attributes, and exceptions.
* **Use Google style docstrings:** This helps with automated documentation generation.
    ```python
    def my_function(param1, param2):
        """Single-line summary.

        More detailed description, if necessary.

        Parameters:
        - param1 (int): The first parameter.
        - param2 (str): The second parameter.

        Returns:
        - bool: The return value. True for success, False otherwise.

        Raises:
        - ValueError: If `param2` is invalid.
        """
        # function body here
    ```

## Type Hints
* **Use type hints:**  Type hints improve code readability and help catch errors early.
* **Follow PEP 484:**  Use the standard type hinting syntax.

## Comments
* **Write clear and concise comments:** Explain the "why" behind the code, not just the "what".
* **Comment sparingly:** Well-written code should be self-documenting where possible.
* **Use complete sentences:** Start comments with a capital letter and use proper punctuation.

## Logging
* **Use a standard logging framework:**  Pirate Weather uses the built-in `logging` module.
* **Log at appropriate levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL
* **Provide context:** Include relevant information in log messages to aid debugging.

## Error Handling
* **Use specific exceptions:** Avoid using broad exceptions like `Exception`.
* **Handle exceptions gracefully:** Provide informative error messages and avoid crashing the program.
* **Use `try...except` blocks:**  Isolate code that might raise exceptions.

# Tooling
* **Code formatter:**  Ruff - Enforces consistent formatting automatically.
* **Linter:**  Ruff - Identifies potential issues and style violations.

# Example
```python

""" Script to contain the helper functions that can be used to generate the text summary of the forecast data for Pirate Weather """
from collections import Counter
import math

# Cloud Cover Thresholds 
cloudyThreshold = 0.875
mostlyCloudyThreshold = 0.625
partlyCloudyThreshold = 0.375
mostlyClearThreshold = 0.125

# Precipitation Intensity Thresholds (mm/h liquid equivalent)
LIGHT_PRECIP_MM_PER_HOUR = 0.4
MID_PRECIP_MM_PER_HOUR = 2.5
HEAVY_PRECIP_MM_PER_HOUR = 10.0

# Snow Intensity Thresholds (mm/h liquid equivalent)
LIGHT_SNOW_MM_PER_HOUR = 0.13
MID_SNOW_MM_PER_HOUR = 0.83
HEAVY_SNOW_MM_PER_HOUR = 3.33

# Icon Thresholds for Precipitation Accumulation (mm liquid equivalent)
HOURLY_SNOW_ACCUM_ICON_THRESHOLD_MM = 0.2
HOURLY_PRECIP_ACCUM_ICON_THRESHOLD_MM = 0.02

DAILY_SNOW_ACCUM_ICON_THRESHOLD_MM = 10.0
DAILY_PRECIP_ACCUM_ICON_THRESHOLD_MM = 1.0

# Icon Thresholds for Visbility (meters)
DEFAULT_VISIBILITY = 1000


def calculate_wind_text(wind, wind_unit, icon="darksky", mode="both"):
    """
    Calculates the wind text

    Parameters:
    - wind (float) -  The wind speed
    - wind_unit (float) -  The unit of the wind speed
    - mode (str): Determines what gets returned by the function. If set to both the summary and icon for the wind will be returned, if just icon then only the icon is returned and if summary then only the summary is returned.

    Returns:
    - wind_text (str) - The textual representation of the wind
    - wind_icon (str) - The icon representation of the wind
    - icon (str): Which icon set to use - Dark Sky or Pirate Weather
    """
    wind_text = None
    wind_icon = None

    light_wind_thresh = 6.7056 * wind_unit
    mid_wind_thresh = 10 * wind_unit
    heavy_wind_thresh = 17.8816 * wind_unit

    if wind >= light_wind_thresh and wind < mid_wind_thresh:
        wind_text = "light-wind"
        if icon == "pirate":
            wind_icon = "breezy"
        else:
            wind_icon = "wind"
    elif wind >= mid_wind_thresh and wind < heavy_wind_thresh:
        wind_text = "medium-wind"
        wind_icon = "wind"
    elif wind >= heavy_wind_thresh:
        wind_text = "heavy-wind"
        if icon == "pirate":
            wind_icon = "dangerous-wind"
        else:
            wind_icon = "wind"

    if mode == "summary":
        return wind_text
    elif mode == "icon":
        return wind_icon
    else:
        return wind_text, wind_icon
