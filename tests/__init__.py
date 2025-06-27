"""Test package initialization.

Suppress noisy warnings during test runs."""



class DiffWarning(UserWarning):
    """Warning emitted when forecast differences are detected."""


warnings.filterwarnings("default", category=DiffWarning)

import warnings

warnings.filterwarnings("ignore")
