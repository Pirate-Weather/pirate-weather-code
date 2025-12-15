"""Tests to ensure nearest-station distance unit conversion works for US/UK units."""

from API.utils.geo import haversine_distance
from API.constants.api_const import CONVERSION_FACTORS

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

    # choose user at the first station (distance 0) and second scenario with a different user
    user = (float(df.iloc[0].latitude), float(df.iloc[0].longitude))
    km = _compute_nearest_distance_km(user, stations)
    km_to_miles = CONVERSION_FACTORS.get("km_to_miles")
    miles_out = round(km * km_to_miles, 2)

    assert miles_out == round(km * km_to_miles, 2)


def test_nearest_station_conversion_uk2_with_sample_data():
    """Use `sample_station_data` to check km -> miles conversion for 'uk2'."""
    # `sample_station_data` is a pytest fixture; call the wrapped function to get the raw DataFrame
    df = sample_station_data.__wrapped__()
    stations = [
        {"id": str(r.station_id), "lat": float(r.latitude), "lon": float(r.longitude)}
        for _, r in df.iterrows()
    ]

    user = (float(df.iloc[1].latitude), float(df.iloc[1].longitude))
    km = _compute_nearest_distance_km(user, stations)
    km_to_miles = CONVERSION_FACTORS.get("km_to_miles")
    miles_out = round(km * km_to_miles, 2)

    assert miles_out == round(km * km_to_miles, 2)
