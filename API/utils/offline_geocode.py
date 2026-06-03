"""Offline city/country geocoding helpers backed by pgeocode."""

from __future__ import annotations

import math
import string
from functools import lru_cache

from fastapi import HTTPException

_COUNTRY_ALIASES = {
    "andorra": "ad",
    "argentina": "ar",
    "american samoa": "as",
    "austria": "at",
    "australia": "au",
    "aland islands": "ax",
    "azerbaijan": "az",
    "bangladesh": "bd",
    "belgium": "be",
    "bulgaria": "bg",
    "bermuda": "bm",
    "brazil": "br",
    "belarus": "by",
    "canada": "ca",
    "switzerland": "ch",
    "chile": "cl",
    "colombia": "co",
    "costa rica": "cr",
    "cyprus": "cy",
    "czechia": "cz",
    "czech republic": "cz",
    "germany": "de",
    "denmark": "dk",
    "dominican republic": "do",
    "algeria": "dz",
    "estonia": "ee",
    "spain": "es",
    "finland": "fi",
    "micronesia": "fm",
    "federated states of micronesia": "fm",
    "faroe islands": "fo",
    "france": "fr",
    "united kingdom": "gb",
    "great britain": "gb",
    "uk": "gb",
    "french guiana": "gf",
    "guernsey": "gg",
    "greenland": "gl",
    "guadeloupe": "gp",
    "guatemala": "gt",
    "guam": "gu",
    "croatia": "hr",
    "haiti": "ht",
    "hungary": "hu",
    "ireland": "ie",
    "isle of man": "im",
    "india": "in",
    "iceland": "is",
    "italy": "it",
    "jersey": "je",
    "japan": "jp",
    "republic of korea": "kr",
    "south korea": "kr",
    "korea": "kr",
    "liechtenstein": "li",
    "sri lanka": "lk",
    "lithuania": "lt",
    "luxembourg": "lu",
    "latvia": "lv",
    "monaco": "mc",
    "republic of moldova": "md",
    "moldova": "md",
    "marshall islands": "mh",
    "north macedonia": "mk",
    "macedonia": "mk",
    "northern mariana islands": "mp",
    "martinique": "mq",
    "malta": "mt",
    "malawi": "mw",
    "mexico": "mx",
    "malaysia": "my",
    "new caledonia": "nc",
    "netherlands": "nl",
    "norway": "no",
    "new zealand": "nz",
    "peru": "pe",
    "philippines": "ph",
    "pakistan": "pk",
    "poland": "pl",
    "saint pierre and miquelon": "pm",
    "puerto rico": "pr",
    "portugal": "pt",
    "palau": "pw",
    "reunion": "re",
    "romania": "ro",
    "serbia": "rs",
    "russian federation": "ru",
    "russia": "ru",
    "sweden": "se",
    "singapore": "sg",
    "slovenia": "si",
    "svalbard and jan mayen islands": "sj",
    "slovakia": "sk",
    "san marino": "sm",
    "thailand": "th",
    "turkey": "tr",
    "ukraine": "ua",
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "us": "us",
    "uruguay": "uy",
    "holy see": "va",
    "vatican": "va",
    "united states virgin islands": "vi",
    "virgin islands": "vi",
    "wallis and futuna islands": "wf",
    "mayotte": "yt",
    "south africa": "za",
}

_ALPHA3_ALIASES = {
    "and": "ad",
    "arg": "ar",
    "aus": "au",
    "aut": "at",
    "bel": "be",
    "bra": "br",
    "can": "ca",
    "che": "ch",
    "chl": "cl",
    "col": "co",
    "cze": "cz",
    "deu": "de",
    "dnk": "dk",
    "esp": "es",
    "fin": "fi",
    "fra": "fr",
    "gbr": "gb",
    "ind": "in",
    "irl": "ie",
    "ita": "it",
    "jpn": "jp",
    "kor": "kr",
    "mex": "mx",
    "mys": "my",
    "nld": "nl",
    "nor": "no",
    "nzl": "nz",
    "per": "pe",
    "phl": "ph",
    "pol": "pl",
    "prt": "pt",
    "rou": "ro",
    "rus": "ru",
    "sgp": "sg",
    "swe": "se",
    "tur": "tr",
    "ukr": "ua",
    "ury": "uy",
    "usa": "us",
    "zaf": "za",
}


def _normalize_country(country: str) -> str:
    normalized = country.strip().lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = " ".join(normalized.split())

    if len(normalized) == 2 and normalized.isalpha():
        return normalized
    if len(normalized) == 3 and normalized.isalpha():
        alpha2 = _ALPHA3_ALIASES.get(normalized)
        if alpha2:
            return alpha2

    alpha2 = _COUNTRY_ALIASES.get(normalized)
    if alpha2:
        return alpha2

    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported Country Specification. Use an ISO 3166-1 alpha-2 code "
            "supported by pgeocode, such as US, CA, GB, FR, or AU."
        ),
    )


@lru_cache(maxsize=128)
def _nominatim(country_code: str):
    try:
        import pgeocode
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pgeocode is required for City,Country location geocoding.",
        )

    try:
        return pgeocode.Nominatim(country_code)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Country Specification: {country_code.upper()}",
        ) from exc


def geocode_city_country(city: str, country: str) -> tuple[float, float]:
    """Resolve a city/country pair to latitude and longitude using pgeocode."""
    city = city.strip()
    if not city:
        raise HTTPException(status_code=400, detail="Invalid City Specification")

    country_code = _normalize_country(country)
    result = _nominatim(country_code).query_location(city, top_k=1)

    if getattr(result, "empty", False):
        raise HTTPException(
            status_code=404,
            detail=f"Location not found for {city}, {country_code.upper()}",
        )

    row = result.iloc[0] if hasattr(result, "iloc") else result
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Location not found for {city}, {country_code.upper()}",
        ) from exc

    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise HTTPException(
            status_code=404,
            detail=f"Location not found for {city}, {country_code.upper()}",
        )

    return lat, lon
