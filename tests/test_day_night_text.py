import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from API.PirateDayNightText import calculate_half_day_text


def test_calculate_half_day_text_clear_day():
    """Test clear day conditions"""
    # Create 13 hours of clear daytime data (4am-4pm)
    hours = []
    for i in range(13):
        hours.append(
            {
                "time": 1730869200 + (i * 3600),  # Starting at 4am
                "cloudCover": 0.1,
                "windSpeed": 5.0,
                "precipIntensity": 0.0,
                "precipAccumulation": 0.0,
                "precipProbability": 0.0,
                "precipType": "none",
                "temperature": 20.0,
                "humidity": 0.5,
                "visibility": 10000.0,
                "dewPoint": 10.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=True,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "clear-day"
    assert text == ["for-day", "clear"]


def test_calculate_half_day_text_rainy_night():
    """Test rainy night conditions"""
    # Create 11 hours of rainy nighttime data (5pm-4am)
    hours = []
    for i in range(11):
        hours.append(
            {
                "time": 1730869200 + (i * 3600),
                "cloudCover": 0.9,
                "windSpeed": 8.0,
                "precipIntensity": 2.0,
                "precipAccumulation": 2.0,
                "precipProbability": 0.8,
                "precipType": "rain",
                "temperature": 15.0,
                "humidity": 0.8,
                "visibility": 5000.0,
                "dewPoint": 13.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=False,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "rain"
    assert text[0] == "for-day"


def test_calculate_half_day_text_partial_rain():
    """Test rain in only part of the period"""
    # Create 13 hours where only the second half has rain
    hours = []
    for i in range(13):
        is_rainy = i >= 7  # Rain in second half
        hours.append(
            {
                "time": 1730869200 + (i * 3600),
                "cloudCover": 0.7 if is_rainy else 0.3,
                "windSpeed": 6.0,
                "precipIntensity": 1.5 if is_rainy else 0.0,
                "precipAccumulation": 1.5 if is_rainy else 0.0,
                "precipProbability": 0.7 if is_rainy else 0.0,
                "precipType": "rain" if is_rainy else "none",
                "temperature": 18.0,
                "humidity": 0.6,
                "visibility": 8000.0,
                "dewPoint": 12.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=True,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "rain"
    # Should indicate rain "during" a specific period (early or late)
    assert text[0] == "during"
    assert text[2] in ["early", "late"]


def test_calculate_half_day_text_cloudy():
    """Test cloudy conditions"""
    hours = []
    for i in range(13):
        hours.append(
            {
                "time": 1730869200 + (i * 3600),
                "cloudCover": 0.9,  # Above 0.875 threshold for "cloudy"
                "windSpeed": 5.0,
                "precipIntensity": 0.0,
                "precipAccumulation": 0.0,
                "precipProbability": 0.0,
                "precipType": "none",
                "temperature": 20.0,
                "humidity": 0.5,
                "visibility": 10000.0,
                "dewPoint": 10.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=True,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "cloudy"
    assert text == ["for-day", "heavy-clouds"]


def test_calculate_half_day_text_windy():
    """Test windy conditions"""
    hours = []
    for i in range(13):
        hours.append(
            {
                "time": 1730869200 + (i * 3600),
                "cloudCover": 0.3,
                "windSpeed": 25.0,  # High wind speed
                "precipIntensity": 0.0,
                "precipAccumulation": 0.0,
                "precipProbability": 0.0,
                "precipType": "none",
                "temperature": 20.0,
                "humidity": 0.5,
                "visibility": 10000.0,
                "dewPoint": 10.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=True,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "wind"
    assert text == ["for-day", "Windy"]


def test_calculate_half_day_text_snow():
    """Test snowy conditions"""
    hours = []
    for i in range(11):
        hours.append(
            {
                "time": 1730869200 + (i * 3600),
                "cloudCover": 0.9,
                "windSpeed": 10.0,
                "precipIntensity": 1.0,
                "precipAccumulation": 1.0,
                "precipProbability": 0.9,
                "precipType": "snow",
                "temperature": -2.0,
                "humidity": 0.8,
                "visibility": 3000.0,
                "dewPoint": -3.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=False,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "snow"
    assert text[0] == "for-day"


def test_calculate_half_day_text_insufficient_data():
    """Test with insufficient data"""
    # Only 5 hours (less than minimum of 8)
    hours = []
    for i in range(5):
        hours.append(
            {
                "time": 1730869200 + (i * 3600),
                "cloudCover": 0.5,
                "windSpeed": 5.0,
                "precipIntensity": 0.0,
                "precipAccumulation": 0.0,
                "precipProbability": 0.0,
                "precipType": "none",
                "temperature": 20.0,
                "humidity": 0.5,
                "visibility": 10000.0,
                "dewPoint": 10.0,
                "smoke": 0.0,
            }
        )

    icon, text = calculate_half_day_text(
        hours=hours,
        precip_accum_unit=1.0,
        vis_units=1.0,
        wind_unit=1.0,
        temp_units=1.0,
        is_day_time=True,
        time_zone="America/New_York",
        curr_time=1730869200,
        icon_set="darksky",
    )

    assert icon == "none"
    assert text == ["unavailable"]
