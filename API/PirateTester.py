from PirateText import calculate_text
from pirateweather_translations.dynamic_loader import load_all_translations

Translations = load_all_translations()
translation = Translations["en"]

currently = {
    "time": 1756583640,
    "summary": "Light Precipitation",
    "icon": "partly-cloudy-day",
    "nearestStormDistance": 2.3,
    "nearestStormBearing": 0,
    "precipIntensity": 0.0943,
    "precipProbability": 0.24,
    "precipIntensityError": 0.18,
    "precipType": "none",
    "temperature": 22.37,
    "apparentTemperature": 18.39,
    "dewPoint": 6.46,
    "humidity": 0.37,
    "pressure": 1012.57,
    "windSpeed": 16.71,
    "windGust": 24.06,
    "windBearing": 214,
    "cloudCover": 0.81,
    "uvIndex": 1.18,
    "visibility": 16.09,
    "ozone": 316.71,
}

cText, cIcon = calculate_text(
    currently,
    1,
    1,
    1,
    1,
    True,
    0.0943,
    0,
    0,
    "current",
    0.0943,
    icon="darksky",
)

print(cIcon)
