import datetime

from API.PirateDailyText import calculate_day_text


def _find_range_in_structure(struct):
    if isinstance(struct, list):
        if len(struct) >= 2 and struct[0] == "centimeters":
            inner = struct[1]
            if isinstance(inner, list) and inner[0] == "range":
                return inner[1], inner[2]
        for item in struct:
            res = _find_range_in_structure(item)
            if res is not None:
                return res
    return None


def test_snow_tolerance_above_5cm_sets_lower_bound_to_one():
    """
    If the displayed snow accumulation (cm) has a very large error such that the
    lower bound would be 0, but the maximum is >= 5 cm (LESS_THAN_TOLERANCE),
    the code should bump the lower bound to 1 (centimeter) rather than leave it
    as 0 which would force a 'less-than' phrasing.
    """

    # Use a fixed UTC start time
    start = int(datetime.datetime(2025, 1, 1, 0, tzinfo=datetime.timezone.utc).timestamp())

    hours = []
    # First hour: large snow accumulation with very large error
    hours.append(
        {
            "time": start,
            "precipType": "snow",
            # 60 mm total snow in this one hour -> 6.0 cm display
            "snowAccumulation": 60.0,
            # Very large intensity error (mm/h) to produce a large snow depth error
            "precipIntensityError": 120.0,
            "liquidAccumulation": 0.0,
            "iceAccumulation": 0.0,
            "precipProbability": 1.0,
            "temperature": 0.0,
            "dewPoint": -1.0,
            "cloudCover": 0.5,
            "windSpeed": 1.0,
            "rainIntensity": 0.0,
            "snowIntensity": 0.0,
            "iceIntensity": 0.0,
            "visibility": 10000,
            "smoke": 0.0,
        }
    )

    # Remaining hours: no precipitation
    for i in range(1, 24):
        hours.append(
            {
                "time": start + i * 3600,
                "precipType": "none",
                "snowAccumulation": 0.0,
                "precipIntensityError": 0.0,
                "liquidAccumulation": 0.0,
                "iceAccumulation": 0.0,
                "precipProbability": 0.0,
                "temperature": 0.0,
                "dewPoint": -1.0,
                "cloudCover": 0.0,
                "windSpeed": 1.0,
                "rainIntensity": 0.0,
                "snowIntensity": 0.0,
                "iceIntensity": 0.0,
                "liquidAccumulation": 0.0,
                "iceAccumulation": 0.0,
                "visibility": 10000,
                "smoke": 0.0,
            }
        )

    icon, summary = calculate_day_text(hours, is_day_time=True, time_zone="UTC", mode="daily", unit_system="si")

    # Find the centimeters range in the nested summary structure
    rng = _find_range_in_structure(summary)
    assert rng is not None, "Expected a centimeters range to be present in the summary"
    low, high = rng

    # With the inputs above: display accum = 6.0 cm, display error = 12.0 cm
    # raw low = floor(6 - 6) = 0, raw high = ceil(6 + 6) = 12
    # Because high >= 5 cm threshold, low should be bumped to 1
    assert low == 1, f"Lower bound should be bumped to 1 but was {low}"
    assert high >= 5, f"Upper bound should be at least 5 cm but was {high}"
