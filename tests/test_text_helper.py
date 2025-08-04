import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "API"))

import math

from API.PirateTextHelper import (
    calculate_sky_icon,
    calculate_precipitation,
    estimate_snow_height,
    estimate_snow_density,
    humidity_sky_text,
)
from API.PirateMinutelyText import calculate_minutely_text
from API.PirateWeeklyText import calculate_precip_summary, calculate_temp_summary
from API.PirateDailyText import calculate_day_text


def test_calculate_sky_icon_day_night():
    assert calculate_sky_icon(0.9, True) == "cloudy"
    assert calculate_sky_icon(0.9, False) == "cloudy"
    assert calculate_sky_icon(0.5, True) == "partly-cloudy-day"
    assert calculate_sky_icon(0.5, False) == "partly-cloudy-night"
    assert calculate_sky_icon(0.1, True, iconSet="pirate") == "clear-day"


def test_calculate_precip_text_light_rain():
    text, icon = calculate_precipitation(
        precipitationIntensity=0.1,
        precipitationAccumUnit=1.0,
        precipitationType="rain",
        summaryType="hour",
        rainAccumulation=0.1,
        snowAccumulation=0,
        iceAccumulation=0,
        pop=0.5,
        iconSet="pirate",
        isDayTime=True,
    )
    assert text == "possible-very-light-rain"
    assert icon == "possible-rain-day"


def test_estimate_snow_density_and_height():
    density = estimate_snow_density(0, 0)
    assert math.isclose(density, 118.381, rel_tol=1e-3)
    height = estimate_snow_height(10, 0, 0)
    expected_height = 10 * 10 / density
    assert math.isclose(height, expected_height, rel_tol=1e-6)


def test_humidity_sky_text():
    assert humidity_sky_text(25, 1, 0.96) == "high-humidity"
    assert humidity_sky_text(10, 1, 0.1) == "low-humidity"
    assert humidity_sky_text(15, 1, 0.5) is None


def create_minute_data(precip_minutes, precip_type="rain", intensity=0.5):
    """Helper function to create a 61-minute array with specified precipitation."""
    minute_arr = []
    for i in range(61):
        if i in precip_minutes:
            minute_arr.append({"precipIntensity": intensity, "precipType": precip_type})
        else:
            minute_arr.append({"precipIntensity": 0, "precipType": "none"})
    return minute_arr


def test_rain_starting_mid_hour():
    """Test for rain starting in the middle of the hour."""
    # Rain from minute 30 to 60
    precip_minutes = range(30, 61)
    minute_arr = create_minute_data(precip_minutes)

    c_text, c_icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="Clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
    )

    # Expected text: "Light rain starting in 30 min."
    assert c_text[0] == "starting-in"
    assert c_text[1] == "light-rain"
    assert c_text[2] == ["minutes", 30]
    assert c_icon == "rain"


def test_snow_for_full_hour():
    """Test for snow lasting the entire hour."""
    # Snow for all 61 minutes
    precip_minutes = range(61)
    minute_arr = create_minute_data(precip_minutes, precip_type="snow", intensity=0.5)

    c_text, c_icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="Clear",
        currentIcon="clear-day",
        icon="pirate",
        precipIntensityUnit=1.0,
    )

    # Expected text: "Light snow for the hour."
    assert c_text == ["for-hour", "light-snow"]
    assert c_icon == "light-snow"


def test_rain_stopping_mid_hour():
    """Test for rain stopping in the middle of the hour."""
    # Rain for the first 15 minutes (0-14)
    precip_minutes = range(15)
    minute_arr = create_minute_data(precip_minutes)

    c_text, c_icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="Clear",
        currentIcon="clear-day",
        icon="pirate",
        precipIntensityUnit=1.0,
    )

    # Expected text: "Light rain stopping in 15 min."
    assert c_text[0] == "stopping-in"
    assert c_text[1] == "light-rain"
    assert c_text[2] == ["minutes", 15]
    assert c_icon == "light-rain"


def test_no_precipitation():
    """Test for a minutely summary with no precipitation."""
    # No rain at all
    precip_minutes = []
    minute_arr = create_minute_data(precip_minutes)

    c_text, c_icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="Clear",
        currentIcon="clear-day",
        icon="pirate",
        precipIntensityUnit=1.0,
    )

    # Expected text: "Clear for the hour."
    assert c_text == ["for-hour", "Clear"]
    assert c_icon == "clear-day"


def test_rain_stopping_and_starting():
    """Test for rain that stops and then starts again within the hour."""
    # Rain from minute 0-10, then 41-60
    precip_minutes = list(range(11)) + list(range(41, 61))
    minute_arr = create_minute_data(precip_minutes)

    c_text, c_icon = calculate_minutely_text(
        minuteArr=minute_arr,
        currentText="Clear",
        currentIcon="clear-day",
        icon="darksky",
        precipIntensityUnit=1.0,
    )

    # Expected: "Light rain stopping in 11 min., starting again 30 min. later."
    assert c_text[0] == "stopping-then-starting-later"
    assert c_text[1] == "light-rain"
    assert c_text[2] == ["minutes", 11]  # Stops in 11 minutes (after minute 10)
    assert c_text[3] == ["minutes", 30]  # Starts again 30 minutes later (41 - 11)
    assert c_icon == "rain"

def test_no_precipitation_for_week():
    """Test for a weekly summary with no precipitation."""
    icons = ["clear-day"] * 8
    precipitationDays = []

    precipSummary, currentIcon = calculate_precip_summary(
        precipitation=False,
        precipitationDays=precipitationDays,
        icons=icons,
        averageIntensity=0,
        intensityUnit=1.0,
        averagePop=0,
        maxIntensity=0,
        icon="darksky",
    )

    assert precipSummary == ["for-week", "no-precipitation"]
    assert currentIcon == "clear-day"


def test_precipitation_one_day_during_week():
    """Test for a weekly summary with precipitation on one day."""
    precipitationDays = [[0, "monday", {"precipType": "rain", "precipAccumulation": 5, "precipProbability": 0.8}]]
    icons = ["clear-day"] * 8

    precipSummary, currentIcon = calculate_precip_summary(
        precipitation=True,
        precipitationDays=precipitationDays,
        icons=icons,
        averageIntensity=2.0,
        intensityUnit=1.0,
        averagePop=0.8,
        maxIntensity=5.0,
        icon="darksky",
    )

    assert precipSummary == ["during", "medium-rain", "monday"]
    assert currentIcon == "rain"


def test_precipitation_over_weekend():
    """Test for a weekly summary with precipitation over the weekend."""
    precipitationDays = [
        [5, "saturday", {"precipType": "snow", "precipAccumulation": 7, "cape": 100, "liftedIndex": -2, "icon": "snow", "precipProbability": 0.9}],
        [6, "sunday", {"precipType": "snow", "precipAccumulation": 7, "cape": 100, "liftedIndex": -2, "icon": "snow", "precipProbability": 0.9}],
    ]
    icons = ["clear-day"] * 8

    precipSummary, currentIcon = calculate_precip_summary(
        precipitation=True,
        precipitationDays=precipitationDays,
        icons=icons,
        averageIntensity=2.0,
        intensityUnit=1.0,
        averagePop=0.8,
        maxIntensity=5.0,
        icon="darksky",
    )

    assert precipSummary == ['over-weekend', ['and', 'medium-snow', 'possible-heavy-snow']]
    assert currentIcon == "snow"


def test_precipitation_all_week_same_type():
    """Test for a weekly summary with precipitation every day, same type."""
    precipitationDays = [
        [0, "monday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [1, "tuesday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [2, "wednesday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [3, "thursday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [4, "friday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [5, "saturday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [6, "sunday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [7, "next-monday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
    ]
    icons = ["clear-day"] * 8

    precipSummary, currentIcon = calculate_precip_summary(
        precipitation=True,
        precipitationDays=precipitationDays,
        icons=icons,
        averageIntensity=2.0,
        intensityUnit=1.0,
        averagePop=0.8,
        maxIntensity=5.0,
        icon="darksky",
    )

    assert precipSummary == ["for-week", "light-rain"]
    assert currentIcon == "rain"


def test_precipitation_all_week_mixed_type():
    """Test for a weekly summary with precipitation every day, mixed type."""
    precipitationDays = [
        [0, "monday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [1, "tuesday", {"precipType": "snow", "precipAccumulation": 3, "precipProbability": 0.7}],
        [2, "wednesday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [3, "thursday", {"precipType": "snow", "precipAccumulation": 3, "precipProbability": 0.7}],
        [4, "friday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [5, "saturday", {"precipType": "snow", "precipAccumulation": 3, "precipProbability": 0.7}],
        [6, "sunday", {"precipType": "rain", "precipAccumulation": 3, "precipProbability": 0.7}],
        [7, "next-monday", {"precipType": "snow", "precipAccumulation": 3, "precipProbability": 0.7}],
    ]
    icons = ["clear-day"] * 8

    precipSummary, currentIcon = calculate_precip_summary(
        precipitation=True,
        precipitationDays=precipitationDays,
        icons=icons,
        averageIntensity=2.0,
        intensityUnit=1.0,
        averagePop=0.8,
        maxIntensity=5.0,
        icon="darksky",
    )

    assert precipSummary == ["for-week", "mixed-precipitation"]
    assert currentIcon == "sleet"


def test_calculate_temp_summary_rising():
    """Test for a temperature summary with rising temperatures."""
    weekArray = [
        {"temperatureHigh": 10},
        {"temperatureHigh": 12},
        {"temperatureHigh": 14},
        {"temperatureHigh": 16},
        {"temperatureHigh": 18},
        {"temperatureHigh": 20},
        {"temperatureHigh": 22},
        {"temperatureHigh": 24},
    ]
    highTemp = [7, "next-monday", 24, 0]
    lowTemp = [0, "monday", 10, 0]

    tempSummary = calculate_temp_summary(highTemp=highTemp, lowTemp=lowTemp, weekArray=weekArray)

    assert tempSummary == [
        "temperatures-rising",
        ["fahrenheit", 24],
        "next-monday",
    ]

def create_hourly_data(hours_config, start_time_epoch, timezone_str="UTC"):
    """
    Helper to create a list of hourly data dictionaries for testing.
    `hours_config` is a dictionary where keys are hour indices (0-24)
    and values are dictionaries of weather properties to set for that hour.
    """
    start_dt = datetime.datetime.fromtimestamp(
        start_time_epoch, tz=datetime.timezone.utc
    )

    hours_data = []
    for i in range(25):  # Generate 25 hours to be safe for period calculations
        hour_dt = start_dt + datetime.timedelta(hours=i)
        hour_data = {
            "time": int(hour_dt.timestamp()),
            "cloudCover": 0.1,
            "windSpeed": 2,
            "precipIntensity": 0,
            "precipAccumulation": 0,
            "precipType": "none",
            "temperature": 15,  # in C
            "humidity": 0.5,
            "visibility": 16.09,
            "precipProbability": 0,
            "dewPoint": 5,
            "smoke": 0.0,
            "precipIntensityError": 0.0,
        }
        if i in hours_config:
            hour_data.update(hours_config[i])
        hours_data.append(hour_data)
    return hours_data


def test_clear_day():
    """Test for a simple clear day summary."""
    start_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    hours = create_hourly_data({}, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,  # Celsius
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="daily",
        icon_set="pirate",
    )

    assert c_icon == "clear-day"
    assert summary_text == ["sentence", ["for-day", "clear"]]


def test_rain_starting_afternoon():
    """Test for rain starting in the afternoon."""
    start_time = datetime.datetime(2024, 1, 1, 8, 0, 0, tzinfo=datetime.timezone.utc)
    # Rain from 2 PM (hour 14) to 5 PM (hour 17)
    hours_config = {
        i: {
            "precipIntensity": 1.5,
            "precipAccumulation": 1.5,
            "precipType": "rain",
            "precipProbability": 0.9,
        }
        for i in range(6, 10)  # 14:00 to 17:00 local time (UTC in this case)
    }

    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=0,
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="daily",
        icon_set="pirate",
    )

    assert c_icon == "light-rain"
    # Expected: "Light rain starting in the afternoon, continuing until night."
    assert summary_text == [
        "sentence",
        [
            "starting-continuing-until",
            "light-rain",
            "afternoon",
            "night",
        ],
    ]


def test_morning_fog_clearing_later():
    """Test for morning fog that clears up later in the day."""
    start_time = datetime.datetime(2024, 1, 1, 6, 0, 0, tzinfo=datetime.timezone.utc)
    # Fog from 6 AM to 10 AM
    hours_config = {
        i: {"visibility": 0.5, "temperature": 10, "humidity": 0.9, "dewPoint": 9.5}
        for i in range(0, 5)  # 6 AM to 10 AM
    }
    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="daily",
        icon_set="pirate",
    )

    assert c_icon == "fog"
    # Expected: "Foggy in the morning."
    assert summary_text == ["sentence", ["during", "fog", "morning"]]


def test_windy_and_cloudy_all_day():
    """Test for a day that is windy and cloudy throughout."""
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    hours_config = {i: {"windSpeed": 12, "cloudCover": 0.9} for i in range(25)}
    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="daily",
        icon_set="pirate",
    )

    assert c_icon == "wind"
    # Expected: "Overcast and windy throughout the day."
    assert summary_text == ["sentence", ["for-day", ["and", "heavy-clouds", "medium-wind"]]]


def test_intermittent_rain():
    """Test for rain in the morning and again in the evening."""
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    hours_config = {}
    # Rain from 8 AM to 10 AM
    for i in range(8, 11):
        hours_config[i] = {
            "precipIntensity": 1.0,
            "precipAccumulation": 1.0,
            "precipType": "rain",
            "precipProbability": 0.8,
        }
    # Rain from 6 PM to 8 PM
    for i in range(18, 21):
        hours_config[i] = {
            "precipIntensity": 1.0,
            "precipAccumulation": 1.0,
            "precipType": "rain",
            "precipProbability": 0.8,
        }

    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="daily",
        icon_set="pirate",
    )

    assert c_icon == "light-rain"
    # Expected: "Light rain during the morning and evening."
    assert summary_text == [
        "sentence",
        ["during", "light-rain", ["and", "morning", "evening"]],
    ]


def test_intermittent_rain_hourly_mode():
    """Test for intermittent rain using hourly mode."""
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    hours_config = {}
    # Rain from 8 AM to 10 AM
    for i in range(8, 11):
        hours_config[i] = {
            "precipIntensity": 1.0,
            "precipAccumulation": 1.0,
            "precipType": "rain",
            "precipProbability": 0.8,
        }
    # Rain from 6 PM to 8 PM
    for i in range(18, 21):
        hours_config[i] = {
            "precipIntensity": 1.0,
            "precipAccumulation": 1.0,
            "precipType": "rain",
            "precipProbability": 0.8,
        }

    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="hour",
        icon_set="pirate",
    )

    assert c_icon == "light-rain"
    # In hourly mode, periods should be prefixed with "today-"
    assert summary_text == [
        "sentence",
        ["during", "light-rain", ["and", "today-morning", "today-evening"]],
    ]


def test_rain_starts_later_in_period_hourly_mode():
    """Test for rain starting later in the first period in hourly mode."""
    # Forecast starts at 8 AM, rain starts at 9 AM.
    start_time = datetime.datetime(2024, 1, 1, 8, 0, 0, tzinfo=datetime.timezone.utc)
    hours_config = {
        i: {
            "precipIntensity": 1.0,
            "precipAccumulation": 1.0,
            "precipType": "rain",
            "precipProbability": 0.8,
        }
        for i in range(1, 4)  # Rain from 9 AM to 11 AM
    }

    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=True,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="hour",
        icon_set="pirate",
    )

    assert c_icon == "light-rain"
    # Expected: "Light rain during later this morning."
    assert summary_text == [
        "sentence",
        ["during", "light-rain", "later-today-morning"],
    ]


def test_rain_crosses_midnight_hourly_mode():
    """Test for rain that continues past midnight in hourly mode."""
    # Forecast starts at 8 PM, rain from 8 PM to 2 AM.
    start_time = datetime.datetime(2024, 1, 1, 20, 0, 0, tzinfo=datetime.timezone.utc)
    hours_config = {
        i: {
            "precipIntensity": 1.0,
            "precipAccumulation": 1.0,
            "precipType": "rain",
            "precipProbability": 0.8,
        }
        for i in range(0, 7)  # Rain from 8 PM to 2 AM
    }

    hours = create_hourly_data(hours_config, start_time.timestamp())

    c_icon, summary_text = calculate_day_text(
        hours=hours,
        precip_accum_unit=0.1,
        vis_units=0.001,
        wind_unit=1.0,
        temp_units=1,
        is_day_time=False,
        time_zone="UTC",
        curr_time=start_time.timestamp(),
        mode="hour",
        icon_set="pirate",
    )

    assert c_icon == "light-rain"
    # Expected: "Light rain until tomorrow morning."
    assert summary_text == [
        "sentence",
        ["until", "light-rain", "tomorrow-morning"],
    ]