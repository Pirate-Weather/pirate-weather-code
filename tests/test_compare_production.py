import datetime
import json
import os
import warnings
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from tests import DiffWarning
from tests.test_s3_live import _get_client

PW_API = os.environ.get("PW_API")
PROD_BASE = "https://api.pirateweather.net/forecast"
PROD_TIMEMACHINE_BASE = "https://api.pirateweather.net/timemachine"

TIMEMACHINE_TEST_LOCATION = (45.4215, -75.6972)  # Ottawa, Canada
TIMEMACHINE_TEST_DATE = datetime.datetime(2020, 6, 15, tzinfo=datetime.UTC)
TIMEMACHINE_TIMESTAMP = int(TIMEMACHINE_TEST_DATE.timestamp())


class ProductionRequestError(Exception):
    """Raised when a production API request cannot be fulfilled."""


def _fetch_production_json(url: str) -> dict:
    """Fetch JSON from the production API with a 10 second timeout.

    Args:
        url: The URL to fetch JSON data from.

    Returns:
        A dictionary parsed from the JSON response.

    Raises:
        ProductionRequestError: If the request fails due to a network issue,
            an unexpected status code, or invalid JSON in the response.
    """

    try:
        with urlopen(url, timeout=10) as response:
            status_code = response.getcode()
            payload = response.read()
    except URLError as exc:  # pragma: no cover - network failure
        raise ProductionRequestError(exc) from exc

    if status_code != 200:
        raise ProductionRequestError(f"Unexpected status {status_code} from {url}")

    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProductionRequestError(f"Invalid JSON from {url}: {exc}") from exc


def _diff_nested(a: object, b: object, path: str = "") -> dict:
    """Return a mapping of all differences between ``a`` and ``b``.

    The keys of the returned dict are ``/`` separated paths describing the
    location of the difference within the nested structure.
    """

    diffs: dict[str, dict[str, object]] = {}

    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            sub_path = f"{path}/{key}" if path else str(key)
            if key not in a:
                diffs[sub_path] = {"local": None, "prod": b[key]}
            elif key not in b:
                diffs[sub_path] = {"local": a[key], "prod": None}
            else:
                diffs.update(_diff_nested(a[key], b[key], sub_path))
    elif isinstance(a, list) and isinstance(b, list):
        for idx in range(max(len(a), len(b))):
            sub_path = f"{path}[{idx}]"
            try:
                val_a = a[idx]
            except IndexError:
                val_a = None
            try:
                val_b = b[idx]
            except IndexError:
                val_b = None
            if val_a is None or val_b is None:
                diffs[sub_path] = {"local": val_a, "prod": val_b}
            else:
                diffs.update(_diff_nested(val_a, val_b, sub_path))
    else:
        if a != b:
            diffs[path] = {"local": a, "prod": b}

    return diffs


@pytest.mark.skipif(
    not PW_API,
    reason="PW_API environment variable not set",
)
def test_local_vs_production():
    client = _get_client()

    # Houston, TX (29.7604, -95.3698) and London, UK (51.50853, -0.12574)
    for lat, lon in [(29.7604, -95.3698), (51.50853, -0.12574)]:
        local_resp = client.get(f"/forecast/{PW_API}/{lat},{lon}?version=2")
        assert local_resp.status_code == 200
        local_data = local_resp.json()

        prod_url = f"{PROD_BASE}/{PW_API}/{lat},{lon}?version=2"
        try:
            prod_data = _fetch_production_json(prod_url)
        except ProductionRequestError as exc:
            pytest.skip(f"Could not fetch production API: {exc}")

        diffs = _diff_nested(local_data, prod_data)
        if diffs:
            diff_text = json.dumps(diffs, indent=2, sort_keys=True)
            warnings.warn(f"Differences for {lat},{lon}:\n{diff_text}", DiffWarning)


@pytest.mark.skipif(
    not PW_API,
    reason="PW_API environment variable not set",
)
def test_timemachine_vs_production():
    client = _get_client()

    lat, lon = TIMEMACHINE_TEST_LOCATION
    timestamp = TIMEMACHINE_TIMESTAMP

    local_resp = client.get(
        f"/timemachine/{PW_API}/{lat},{lon},{timestamp}?version=2"
    )
    assert local_resp.status_code == 200
    local_data = local_resp.json()

    prod_url = (
        f"{PROD_TIMEMACHINE_BASE}/{PW_API}/{lat},{lon},{timestamp}?version=2"
    )
    try:
        prod_data = _fetch_production_json(prod_url)
    except ProductionRequestError as exc:  # pragma: no cover - network failure
        pytest.skip(f"Could not fetch production API: {exc}")

    diffs = _diff_nested(local_data, prod_data)
    if diffs:
        diff_text = json.dumps(diffs, indent=2, sort_keys=True)
        warnings.warn(
            f"Timemachine differences for {lat},{lon} at {timestamp}:\n{diff_text}",
            DiffWarning,
        )
