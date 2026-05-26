"""Test package initialization.

Suppress noisy warnings during test runs.
"""

import os as _os
import warnings as _warnings
from pathlib import Path as _Path

_ENV_DEFAULT_KEYS = frozenset({"PW_API", "AWS_KEY", "AWS_SECRET", "s3_bucket"})


class DiffWarning(UserWarning):
    """Warning emitted when forecast differences are detected."""


def _load_env_defaults() -> None:
    """Load local .env values without overriding existing environment variables."""

    env_path = _Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key not in _ENV_DEFAULT_KEYS or key in _os.environ:
            continue

        _os.environ[key] = value.strip().strip('"').strip("'")


_load_env_defaults()
_warnings.filterwarnings("ignore")
_warnings.filterwarnings("default", category=DiffWarning)
