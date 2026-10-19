"""Opt-in tests that depend on external data.

Everything in this directory talks to something outside the repository: the live
S3 Zarr stores, the production API, NOAA/Herbie downloads, GribStream, or local
production archives. These tests are slow (minutes), memory hungry (~10 GB), and
their results depend on the current state of upstream data, so they are excluded
from the default ``pytest`` run and from CI.

Run them deliberately with::

    PW_RUN_LIVE_TESTS=1 python -m pytest tests/live

Credentials (``PW_API``, ``GRIBSTREAM_API_KEY``, ...) are read from the
environment, falling back to a repository-root ``.env`` file.
"""

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_LIVE_ENV_VAR = "PW_RUN_LIVE_TESTS"
_ENV_FILE_KEYS = frozenset(
    {"PW_API", "AWS_KEY", "AWS_SECRET", "GRIBSTREAM_API_KEY", "s3_bucket"}
)


def _load_env_defaults() -> None:
    """Load credentials from ``.env`` without overriding the real environment."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key in _ENV_FILE_KEYS and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def pytest_collection_modifyitems(config, items):
    """Mark every test here as ``live`` and skip them unless explicitly enabled."""
    live_dir = Path(__file__).resolve().parent
    enabled = os.environ.get(RUN_LIVE_ENV_VAR) == "1"
    if enabled:
        _load_env_defaults()

    skip_live = pytest.mark.skip(
        reason=f"live test: set {RUN_LIVE_ENV_VAR}=1 to run tests that use external data"
    )
    for item in items:
        if live_dir not in Path(item.fspath).resolve().parents:
            continue
        item.add_marker(pytest.mark.live)
        if not enabled:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def pw_api_key() -> str:
    """Return the Pirate Weather API key used for the S3 proxy and production API."""
    key = os.environ.get("PW_API")
    if not key:
        pytest.skip("PW_API is not set")
    return key


@pytest.fixture(scope="session")
def live_client(pw_api_key):
    """A TestClient for ``responseLocal`` reading the live S3 Zarr stores.

    Loading every store takes most of a minute and several GB, so a single app
    is shared across the session.
    """
    from fastapi.testclient import TestClient

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STAGE", os.environ.get("STAGE", "TESTING"))
        mp.setenv("save_type", "S3")
        # responseLocal imports its siblings as top-level modules.
        mp.syspath_prepend(str(REPO_ROOT / "API"))

        response_path = REPO_ROOT / "API" / "responseLocal.py"
        spec = importlib.util.spec_from_file_location("responseLocal", response_path)
        response_local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(response_local)

        with TestClient(response_local.app) as client:
            yield client
