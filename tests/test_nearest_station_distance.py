"""Tests for nearest station distance calculation."""

from API.utils.geo import haversine_distance


def test_haversine_distance_berlin_munich():
    """Test haversine distance between Berlin and Munich."""
    # Berlin coordinates
    lat1, lon1 = 52.5, 13.4
    # Munich coordinates
    lat2, lon2 = 48.1, 11.6

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # The actual distance is approximately 505 km
    assert 500 < distance < 510, f"Expected distance ~505 km, got {distance:.2f} km"


def test_haversine_distance_same_point():
    """Test that distance from a point to itself is zero."""
    lat, lon = 40.7128, -74.0060  # New York

    distance = haversine_distance(lat, lon, lat, lon)

    assert distance == 0.0, "Distance from a point to itself should be zero"


def test_haversine_distance_nearby_points():
    """Test distance for nearby points (should be small)."""
    # Two points very close together (~1 km apart)
    lat1, lon1 = 52.5200, 13.4050
    lat2, lon2 = 52.5300, 13.4050

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # Distance should be roughly 1 km (actually about 1.11 km)
    assert 1.0 < distance < 1.5, f"Expected distance ~1 km, got {distance:.2f} km"


def test_haversine_distance_antipodal():
    """Test distance for points on opposite sides of Earth."""
    # North pole
    lat1, lon1 = 90.0, 0.0
    # South pole
    lat2, lon2 = -90.0, 0.0

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # Distance should be approximately half Earth's circumference
    # Earth's circumference is about 40,075 km, so half is about 20,038 km
    assert 20000 < distance < 20100, (
        f"Expected distance ~20,038 km, got {distance:.2f} km"
    )


def test_haversine_distance_across_prime_meridian():
    """Test distance calculation across the prime meridian."""
    # London (just east of prime meridian)
    lat1, lon1 = 51.5074, 0.1278
    # Paris (east of prime meridian)
    lat2, lon2 = 48.8566, 2.3522

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # Actual distance is approximately 340 km
    assert 330 < distance < 350, f"Expected distance ~340 km, got {distance:.2f} km"


def test_haversine_distance_across_dateline():
    """Test distance calculation across the international date line."""
    # Point just west of dateline
    lat1, lon1 = 0.0, 179.5
    # Point just east of dateline
    lat2, lon2 = 0.0, -179.5

    distance = haversine_distance(lat1, lon1, lat2, lon2)

    # These points are actually very close (about 111 km apart at equator)
    assert 100 < distance < 120, f"Expected distance ~111 km, got {distance:.2f} km"


def test_nearest_station_calculation():
    """Test finding the nearest station from a list."""
    # User location (Berlin)
    user_lat, user_lon = 52.5, 13.4

    # Sample stations
    stations = [
        {"id": "10001", "name": "Berlin Station", "lat": 52.5, "lon": 13.4},  # 0 km
        {"id": "10002", "name": "Munich Station", "lat": 48.1, "lon": 11.6},  # ~505 km
        {"id": "10003", "name": "Hamburg Station", "lat": 53.6, "lon": 10.0},  # ~255 km
    ]

    # Calculate distances to each station
    distances = []
    for station in stations:
        distance = haversine_distance(
            user_lat, user_lon, station["lat"], station["lon"]
        )
        distances.append(distance)

    min_distance = min(distances)
    nearest_station_idx = distances.index(min_distance)

    # Berlin station should be the nearest (distance = 0)
    assert nearest_station_idx == 0, "Berlin station should be nearest"
    assert min_distance == 0.0, "Distance to Berlin station should be 0"


def test_nearest_station_multiple_nearby():
    """Test finding nearest station when multiple are nearby."""
    # User location (near Hamburg)
    user_lat, user_lon = 53.5, 10.0

    # Stations at various distances
    stations = [
        {"id": "10001", "name": "Far Station", "lat": 48.1, "lon": 11.6},  # ~570 km
        {"id": "10002", "name": "Near Station", "lat": 53.6, "lon": 10.0},  # ~11 km
        {"id": "10003", "name": "Medium Station", "lat": 52.5, "lon": 13.4},  # ~260 km
    ]

    # Find nearest station
    min_distance = float("inf")
    nearest_station_name = None

    for station in stations:
        distance = haversine_distance(
            user_lat, user_lon, station["lat"], station["lon"]
        )
        if distance < min_distance:
            min_distance = distance
            nearest_station_name = station["name"]

    assert nearest_station_name == "Near Station", "Near Station should be closest"
    assert 10 < min_distance < 15, (
        f"Expected distance ~11 km, got {min_distance:.2f} km"
    )


def test_nearest_station_with_invalid_coordinates():
    """Test handling of stations with missing coordinates."""
    user_lat, user_lon = 52.5, 13.4

    # Mix of valid and invalid stations
    stations = [
        {"id": "10001", "name": "Invalid Station", "lat": None, "lon": 13.4},
        {"id": "10002", "name": "Valid Station", "lat": 52.5, "lon": 13.4},
        {"id": "10003", "name": "Another Invalid", "lat": 52.5, "lon": None},
    ]

    # Calculate distance only for valid stations
    min_distance = float("inf")
    for station in stations:
        station_lat = station.get("lat")
        station_lon = station.get("lon")
        if station_lat is not None and station_lon is not None:
            distance = haversine_distance(user_lat, user_lon, station_lat, station_lon)
            min_distance = min(min_distance, distance)

    # Should find the valid station at distance 0
    assert min_distance == 0.0, "Should find the valid station"


def test_nearest_station_empty_list():
    """Test behavior when no stations are available."""
    user_lat, user_lon = 52.5, 13.4
    stations = []

    # When no stations, distance should remain at default
    nearest_station_distance = -999  # Default value

    if stations:
        min_distance = float("inf")
        for station in stations:
            distance = haversine_distance(
                user_lat, user_lon, station["lat"], station["lon"]
            )
            min_distance = min(min_distance, distance)

        if min_distance != float("inf"):
            nearest_station_distance = int(round(min_distance))

    assert nearest_station_distance == -999, (
        "Should return -999 when no stations available"
    )


def test_distance_rounding():
    """Test that distances are properly rounded to integers."""
    # User location
    user_lat, user_lon = 52.5, 13.4
    # Station about 10.4 km away
    station_lat, station_lon = 52.5, 13.55

    distance = haversine_distance(user_lat, user_lon, station_lat, station_lon)
    rounded_distance = int(round(distance))

    # Should round to nearest integer
    assert isinstance(rounded_distance, int), "Distance should be an integer"
    assert 9 <= rounded_distance <= 11, f"Expected ~10 km, got {rounded_distance} km"
