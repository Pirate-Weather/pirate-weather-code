import os
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PW_API = os.environ.get("PW_API")

@pytest.mark.skipif(not PW_API, reason="PW_API environment variable not set")
def test_live_s3_forecast():
    os.environ["STAGE"] = "TESTING"
    os.environ["save_type"] = "S3"
    os.environ.setdefault("useETOPO", "FALSE")

    # Load the responseLocal module directly from the API directory
    api_path = Path(__file__).resolve().parents[1] / "API"
    sys.path.insert(0, str(api_path))
    response_path = api_path / "responseLocal.py"
    spec = importlib.util.spec_from_file_location("responseLocal", response_path)
    responseLocal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(responseLocal)

    client = TestClient(responseLocal.app)
    response = client.get(f"/forecast/{PW_API}/45.0,-75.0")
    assert response.status_code == 200
    data = response.json()
    assert data["latitude"] == pytest.approx(45.0, abs=0.5)
    assert data["longitude"] == pytest.approx(-75.0, abs=0.5)

