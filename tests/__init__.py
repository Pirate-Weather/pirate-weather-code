"""Test package initialization.

Suppress noisy warnings during test runs.
"""

import warnings as _warnings

class DiffWarning(UserWarning):
    """Warning emitted when forecast differences are detected."""


_warnings.filterwarnings("ignore")
_warnings.filterwarnings("default", category=DiffWarning)
