# %% Script to contain the helper functions that are used in the Pirate Weather API
import numpy as np
import math


def kelvinFromCelsius(celsius):
    return celsius + 273.15


def estimateSnowHeight(precipitationMm, temperatureC, windSpeedMps):
    snowDensityKgM3 = estimateSnowDensity(temperatureC, windSpeedMps)
    return precipitationMm * 10 / snowDensityKgM3

    # This one is too much, with its 10-100x range
    #
    # formula from https://www.omnicalculator.com/other/rain-to-snow#how-many-inches-of-snow-is-equal-to-one-inch-of-rain
    # ratioBase = 10.3 + (-1.21 * temperatureC) + (0.0389 * temperatureC * temperatureC)
    # print(ratioBase)
    # ratio = min(max(ratioBase, 1), 100)
    # snowMm = precipitationMm / ratio
    # return snowMm


# - Returns: kg/m3
def estimateSnowDensity(temperatureC, windSpeedMps):
    # interpolation at  https://docs.google.com/spreadsheets/d/1nrCN37VpoeDgAQHr70HcLDyyt-_dQdsRJMerpKMW0ho/edit?usp=sharing
    # Ratio ranges:
    # 3-30x: https://www.eoas.ubc.ca/courses/atsc113/snow/met_concepts/07-met_concepts/07b-newly-fallen-snow-density/
    # 3-20x: https://www.researchgate.net/figure/Common-densities-of-snow_tbl1_258653078
    # 4-20x: https://www.researchgate.net/figure/Fresh-snow-density-as-a-function-of-air-temperature-and-wind-for-the-3-options-included_fig2_316868161

    # Equations: from ESOLIP, https://www.tandfonline.com/eprint/Qf3k4JEPg3xXRmzp7gQQ/full (https://www.tandfonline.com/doi/pdf/10.1080/02626667.2015.1081203?needAccess=true)
    # Originally from https://sci-hub.hkvisa.net/10.1029/1999jc900011 (Jordan, R.E., Andreas, E.L., and Makshtas, A.P., 1999. Heat budget of snow-covered sea ice at North Pole 4. Journal of Geophysical Research)
    # Problem: These seem to be considering wind speed and it's factor on compacting the snow? Is that okay to use? According to ESOLIP paper probably yes.
    kelvins = kelvinFromCelsius(temperatureC)

    # above 2.5? bring it down, it shouldn't happen, but if it does, let's just assume it's 2.5 deg
    kelvins = min(kelvins, 275.65)

    windSpeedExp17 = pow(windSpeedMps, 1.7)

    snowDensityKgM3 = 1000
    if kelvins <= 260.15:
        snowDensityKgM3 = 500 * (1 - 0.904 * math.exp(-0.008 * windSpeedExp17))
    elif kelvins <= 275.65:
        snowDensityKgM3 = 500 * (
            1
            - 0.951
            * math.exp(-1.4 * pow(278.15 - kelvins, -1.15) - 0.008 * windSpeedExp17)
        )
    else:
        # above 2.5 degrees -> fallback, return precip mm (-> ratio = 1)
        # should not happen - see above
        snowDensityKgM3 = 1000

    # ensure we don't divide by zero - ensure minimum
    snowDensityKgM3 = max(snowDensityKgM3, 50)

    return snowDensityKgM3


def apparent_temp_calculator(temp, humidity, windSpeed, windUnit):
    apparentTemp = temp

    # AT = Ta + 0.33 × rh / 100 × 6.105 × exp(17.27 × Ta / (237.7 + Ta)) − 0.70 × ws − 4.00
    e = humidity * 6.105 * np.exp(17.27 * (temp - 273.15) / (237.7 + (temp - 273.15)))
    apparentTemp = (
        (temp - 273.15) + 0.33 * e - 0.70 * (windSpeed / windUnit) - 4.00
    ) + 273.15

    return apparentTemp


def temp_converter(temp, tempUnits):
    if tempUnits == 0:
        temp = (temp - 273.15) * 9 / 5 + 32
    else:
        temp = temp - 273.15

    return temp
