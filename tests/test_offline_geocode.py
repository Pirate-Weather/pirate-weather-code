import math

import pytest
from fastapi import HTTPException

from API.utils.offline_geocode import _normalize_country, geocode_city_country


def test_normalize_country_accepts_alpha2_name_and_common_alias():
    assert _normalize_country("US") == "us"
    assert _normalize_country("United States") == "us"
    assert _normalize_country("U.S.A.") == "us"
    assert _normalize_country("United Kingdom") == "gb"


def test_normalize_country_rejects_unknown_country():
    with pytest.raises(HTTPException) as exc:
        _normalize_country("Atlantis")

    assert exc.value.status_code == 400


def test_geocode_city_country_rejects_nan_coordinates(monkeypatch):
    class FakeRow(dict):
        pass

    class FakeResult:
        empty = False
        iloc = [FakeRow(latitude=math.nan, longitude=math.nan)]

    class FakeNominatim:
        def query_location(self, city, top_k=1):
            return FakeResult()

    monkeypatch.setattr(
        "API.utils.offline_geocode._nominatim",
        lambda country_code: FakeNominatim(),
    )

    with pytest.raises(HTTPException) as exc:
        geocode_city_country("Nowhere", "US")

    assert exc.value.status_code == 404
