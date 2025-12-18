"""Tests to ensure nearest-station distance unit conversion works for US/UK units."""

from API.constants.api_const import CONVERSION_FACTORS
from API.utils.geo import haversine_distance

# Reuse the sample station data helper from the DWD station mapping tests.
from tests.test_dwd_station_mapping import sample_station_data


def _compute_nearest_distance_km(user, stations):
    distances = [
        haversine_distance(user[0], user[1], s["lat"], s["lon"]) for s in stations
    ]
    return min(distances)


def test_nearest_station_conversion_us_with_sample_data():
    """Use `sample_station_data` to check km -> miles conversion for 'us'."""
    # `sample_station_data` is a pytest fixture; call the wrapped function to get the raw DataFrame
    df = sample_station_data.__wrapped__()
    # Build stations list from DataFrame
    stations = [
        {"id": str(r.station_id), "lat": float(r.latitude), "lon": float(r.longitude)}
        for _, r in df.iterrows()
    ]

    # A user location that is not at a station to test conversion
    user = (50.0, 12.0)
    km = _compute_nearest_distance_km(user, stations)
    km_to_miles = CONVERSION_FACTORS.get("km_to_miles")
    miles_out = round(km * km_to_miles, 2)

    # Expected: min distance is to Munich (48.1, 11.6), ~213.44 km -> 132.52 miles
    assert miles_out == 132.52


def test_nearest_station_conversion_uk2_with_sample_data():
    """Use `sample_station_data` to check km -> miles conversion for 'uk2'."""
    # `sample_station_data` is a pytest fixture; call the wrapped function to get the raw DataFrame
    df = sample_station_data.__wrapped__()
    stations = [
        {"id": str(r.station_id), "lat": float(r.latitude), "lon": float(r.longitude)}
        for _, r in df.iterrows()
    ]

    # A user location that is not at a station to test conversion
    user = (53.0, 11.0)
    km = _compute_nearest_distance_km(user, stations)
    km_to_miles = CONVERSION_FACTORS.get("km_to_miles")
    miles_out = round(km * km_to_miles, 2)

    # Expected: min distance is to Hamburg (53.6, 10.0), ~86.3 km -> 58.51 miles
    assert miles_out == 58.51
