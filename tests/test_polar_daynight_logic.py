from API.utils.geo import _polar_is_all_day


def test_polar_day_night_logic_northern():
    # High northern latitude in November -> polar night (not polar day)
    lat = 76.417
    month = 11
    assert _polar_is_all_day(lat, month) is False

    # High northern latitude in June -> polar day
    month = 6
    assert _polar_is_all_day(lat, month) is True


def test_polar_day_night_logic_southern():
    # High southern latitude in December -> polar day (southern summer)
    lat = -76.0
    month = 12
    assert _polar_is_all_day(lat, month) is True

    # High southern latitude in June -> polar night
    month = 6
    assert _polar_is_all_day(lat, month) is False


def test_polar_day_night_edge_months():
    # Test boundary months
    lat = 80.0
    assert _polar_is_all_day(lat, 4) is True
    assert _polar_is_all_day(lat, 9) is True
    assert _polar_is_all_day(lat, 3) is False

    lat = -80.0
    assert _polar_is_all_day(lat, 10) is True
    assert _polar_is_all_day(lat, 3) is True
    assert _polar_is_all_day(lat, 4) is False
