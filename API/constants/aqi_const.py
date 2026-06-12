"""Air Quality Index (AQI) constants based on EPA standards.

This module contains the concentration breakpoints and corresponding AQI values
for various air pollutants according to the US EPA Air Quality Index.

References:
    - EPA AQI Technical Assistance Document: https://www.airnow.gov/aqi/aqi-basics/
    - EPA AQI Breakpoints: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
"""

# PM2.5 (Fine Particulate Matter, µg/m³)
# Breakpoints for 24-hour average PM2.5 concentrations
PM25_BP = [0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]
PM25_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# PM10 (Coarse Particulate Matter, µg/m³)
# Breakpoints for 24-hour average PM10 concentrations
PM10_BP = [0, 54, 154, 254, 354, 424, 504, 604]
PM10_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# O3 (Ozone, µg/m³)
# Breakpoints for 8-hour average ozone concentrations
O3_BP = [0, 108, 140, 170, 210, 400, 504, 604]
O3_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# NO2 (Nitrogen Dioxide, µg/m³)
# Breakpoints for 1-hour average NO2 concentrations
NO2_BP = [0, 100, 188, 677, 1221, 1880, 2350, 2820]
NO2_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# SO2 (Sulfur Dioxide, µg/m³)
# Breakpoints for 1-hour average SO2 concentrations
SO2_BP = [0, 92, 197, 485, 800, 1574, 2101, 2620]
SO2_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# CO (Carbon Monoxide, µg/m³)
# Breakpoints for 8-hour average CO concentrations
CO_BP = [0, 4400, 9400, 12400, 15400, 30400, 40400, 50400]
CO_AQI = [0, 50, 100, 150, 200, 300, 400, 500]
