"""City,Country location parsing via the offline pgeocode lookup."""

import math

import pytest
from fastapi import HTTPException

from API.utils.offline_geocode import _normalize_country, geocode_city_country


@pytest.mark.parametrize(
    ("country", "expected"),
    [
        pytest.param("US", "us", id="alpha2"),
        pytest.param("United States", "us", id="name"),
        pytest.param("U.S.A.", "us", id="punctuated-alpha3"),
        pytest.param("United Kingdom", "gb", id="name-differs-from-code"),
        pytest.param("Atlantis", 400, id="unknown-is-400"),
    ],
)
def test_normalize_country(country, expected):
    """Users write countries many ways; each must resolve to the pgeocode alpha-2
    code, and an unknown country must be a client error rather than a 500.
    """
    if expected == 400:
        with pytest.raises(HTTPException) as exc:
            _normalize_country(country)
        assert exc.value.status_code == 400
    else:
        assert _normalize_country(country) == expected


def test_geocode_city_country_rejects_nan_coordinates(monkeypatch):
    """pgeocode returns a row of NaNs for unknown places instead of an empty
    result; that must be a 404, not a forecast request for lat/lon NaN.
    """

    class FakeResult:
        empty = False
        iloc = ({"latitude": math.nan, "longitude": math.nan},)

    class FakeNominatim:
        def query_location(self, city, top_k=1):
            return FakeResult()

    monkeypatch.setattr(
        "API.utils.offline_geocode._nominatim", lambda country_code: FakeNominatim()
    )

    with pytest.raises(HTTPException) as exc:
        geocode_city_country("Nowhere", "US")

    assert exc.value.status_code == 404
