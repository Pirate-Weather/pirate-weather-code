import os

import httpx
import pytest

from tests.test_s3_live import _get_client

PW_API = os.environ.get("PW_API")
PROD_BASE = "https://api.pirateweather.net/forecast"


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
    session = httpx.Client()

    for lat, lon in [(45.0, -75.0), (10.0, 10.0)]:
        local_resp = client.get(f"/forecast/{PW_API}/{lat},{lon}")
        assert local_resp.status_code == 200
        local_data = local_resp.json()

        prod_url = f"{PROD_BASE}/{PW_API}/{lat},{lon}"
        try:
            prod_resp = session.get(prod_url, timeout=10)
        except Exception as exc:  # pragma: no cover - network failure
            pytest.skip(f"Could not fetch production API: {exc}")
        assert prod_resp.status_code == 200
        prod_data = prod_resp.json()

        diffs = _diff_nested(local_data, prod_data)
        if diffs:
            print(f"Differences for {lat},{lon}: {diffs}")
