# Model variable indices for human-readable access
# Add more mappings as needed for each model

HRRR_SUBH = {
    "gust": 1,
    "pressure": 2,
    "temp": 3,
    "dew": 4,
    "wind_u": 5,
    "wind_v": 6,
    "intensity": 7,
    "snow": 8,
    "ice": 9,
    "freezing_rain": 10,
    "rain": 11,
    "refc": 12,
    "accum": 13,
    "vis": 14,
    "specific_humidity": 15,
    "solar": 16,
}
HRRR = {
    "vis": 1,
    "gust": 2,
    "pressure": 3,
    "temp": 4,
    "dew": 5,
    "humidity": 6,
    "wind_u": 7,
    "wind_v": 8,
    "intensity": 9,
    "accum": 10,
    "snow": 11,
    "ice": 12,
    "freezing_rain": 13,
    "rain": 14,
    "cloud": 15,
    "smoke": 16,
    "refc": 17,
    "solar": 18,
    "cape": 19,
}
# RRFS (ingest/RRFS_*_Local_Ingest.py) is only read with include=rrfs. The CONUS
# 3 km and North American 13 km stores share one layout: HRRR's first 19
# columns (MSLET in place of MSLMA for "pressure"), then surface CIN and the
# 500-1000 mb lifted index. The sub-hourly store matches HRRR_SUBH column for
# column, so the same index map reads it.
RRFS = {
    "vis": 1,
    "gust": 2,
    "pressure": 3,
    "temp": 4,
    "dew": 5,
    "humidity": 6,
    "wind_u": 7,
    "wind_v": 8,
    "intensity": 9,
    "accum": 10,
    "snow": 11,
    "ice": 12,
    "freezing_rain": 13,
    "rain": 14,
    "cloud": 15,
    "smoke": 16,
    "refc": 17,
    "solar": 18,
    "cape": 19,
    "cin": 20,
    "lifted_index": 21,
}
RRFS_SUBH = HRRR_SUBH

GFS = {
    "vis": 1,
    "gust": 2,
    "pressure": 3,
    "temp": 4,
    "dew": 5,
    "humidity": 6,
    "apparent": 7,
    "wind_u": 8,
    "wind_v": 9,
    "intensity": 10,
    "accum": 11,
    "snow": 12,
    "ice": 13,
    "freezing_rain": 14,
    "rain": 15,
    "ozone": 16,
    "cloud": 17,
    "uv": 18,
    "storm_dist": 19,
    "storm_dir": 20,
    "refc": 21,
    "solar": 22,
    "cape": 23,
    "station_pressure": 24,
    "lifted_index": 25,
    "vertical_velocity": 26,
    "cin": 27,
}

GEFS = {
    "prob": 1,
    "accum": 2,
    "error": 3,
    "snow": 4,
    "ice": 5,
    "freezing_rain": 6,
    "rain": 7,
}

AIGEFS = {
    "prob": 1,
    "accum": 2,
    "error": 3,
}

ECMWF = {
    "pressure": 1,
    "temp": 2,
    "dew": 3,
    "wind_u": 4,
    "wind_v": 5,
    "intensity": 6,
    "accum": 7,
    "ptype": 8,
    "cloud": 9,
    "prob": 10,
    "accum_mean": 11,
    "accum_stddev": 12,
}

NBM = {
    "gust": 1,
    "temp": 2,
    "apparent": 3,
    "dew": 4,
    "humidity": 5,
    "wind": 6,
    "bearing": 7,
    "accum": 8,
    "cloud": 9,
    "vis": 10,
    "unused": 11,
    "prob": 12,
    "intensity": 13,
    "rain": 14,
    "freezing_rain": 15,
    "snow": 16,
    "ice": 17,
    "solar": 18,
    "cape": 19,
}

RTMA_RU = {
    "vis": 1,
    "gust": 2,
    "pressure": 3,
    "temp": 4,
    "dew": 5,
    "humidity": 6,
    "cloud": 7,
    "wind_u": 8,
    "wind_v": 9,
}

URMA = {
    "vis": 1,
    "gust": 2,
    "pressure": 3,
    "temp": 4,
    "dew": 5,
    "humidity": 6,
    "cloud": 7,
    "wind_u": 8,
    "wind_v": 9,
}

ERA5 = {
    "instantaneous_10m_wind_gust": 1,
    "mean_sea_level_pressure": 2,
    "2m_temperature": 3,
    "2m_dewpoint_temperature": 4,
    "10m_u_component_of_wind": 5,
    "10m_v_component_of_wind": 6,
    "precipitation_type": 7,
    "total_precipitation": 8,  # Snow + rain in m/liquid water equivalent
    "large_scale_rain_rate": 9,
    "convective_rain_rate": 10,
    "large_scale_snowfall_rate_water_equivalent": 11,
    "convective_snowfall_rate_water_equivalent": 12,
    "total_column_ozone": 13,
    "total_cloud_cover": 14,
    "downward_uv_radiation_at_the_surface": 15,
    "surface_solar_radiation_downwards": 16,
    "convective_available_potential_energy": 17,
    "surface_pressure": 18,
    "prob": 19,
}

ERA5_SOURCE_VARS = tuple(var for var in ERA5 if var != "prob")

# Own ERA5 archive (scripts/build_era5_point_test_store.py), one int16 store per
# 1940-aligned week. Column map as for the forecast stores: time is column 0,
# rebuilt from the store's "time" attribute. Values are in the API units below,
# not Google's (mm rather than m, hPa rather than Pa, W/m2 rather than hourly
# J/m2). rain_rate and snowfall_rate_water_equivalent are the large-scale plus
# convective rates (the API only uses the sums), and precipitation_probability is
# the API's 3x3 neighbourhood probability, precomputed so a point read is one chunk.
ERA5_ARCHIVE = {
    "instantaneous_10m_wind_gust": 1,  # m/s
    "mean_sea_level_pressure": 2,  # hPa
    "2m_temperature": 3,  # K
    "2m_dewpoint_temperature": 4,  # K
    "10m_u_component_of_wind": 5,  # m/s
    "10m_v_component_of_wind": 6,  # m/s
    "precipitation_type": 7,  # ERA5 ptype code
    "total_precipitation": 8,  # mm over the hour
    "rain_rate": 9,  # mm/h
    "snowfall_rate_water_equivalent": 10,  # mm/h
    "total_column_ozone": 11,  # DU
    "total_cloud_cover": 12,  # %
    "downward_uv_radiation_at_the_surface": 13,  # W/m2, mean over the hour
    "surface_solar_radiation_downwards": 14,  # W/m2, mean over the hour
    "convective_available_potential_energy": 15,  # J/kg
    "surface_pressure": 16,  # hPa
    "precipitation_probability": 17,  # %
    "total_sky_direct_solar_radiation_at_surface": 18,  # W/m2, mean over the hour
    "cloud_base_height": 19,  # m above the surface, NaN where there is none
}

# int16 encoding per archive variable: ("lin", step, offset) stores
# round((value - offset) / step); ("sqrt", scale, 0) stores round(sqrt(value) * scale),
# which keeps <= 0.01 mm error below 1 mm and <= 1 % above for precipitation.
# Steps follow the API's precision classes (0.1 K, 0.1 hPa, 0.1 m/s, 1 %), coarser
# where the API reports far less than the source precision (ozone, UV, solar, CAPE).
ERA5_ARCHIVE_ENCODING = {
    "instantaneous_10m_wind_gust": ("lin", 0.1, 0.0),
    "mean_sea_level_pressure": ("lin", 0.1, 1000.0),
    "2m_temperature": ("lin", 0.1, 273.15),
    "2m_dewpoint_temperature": ("lin", 0.1, 273.15),
    "10m_u_component_of_wind": ("lin", 0.1, 0.0),
    "10m_v_component_of_wind": ("lin", 0.1, 0.0),
    "precipitation_type": ("lin", 1.0, 0.0),
    "total_precipitation": ("sqrt", 100.0, 0.0),
    "rain_rate": ("sqrt", 100.0, 0.0),
    "snowfall_rate_water_equivalent": ("sqrt", 100.0, 0.0),
    "total_column_ozone": ("lin", 2.0, 300.0),
    "total_cloud_cover": ("lin", 1.0, 0.0),
    "downward_uv_radiation_at_the_surface": ("lin", 0.5, 0.0),  # 0.05 UV index
    "surface_solar_radiation_downwards": ("lin", 2.0, 0.0),
    "convective_available_potential_energy": ("lin", 10.0, 0.0),
    "surface_pressure": ("lin", 0.1, 750.0),
    "precipitation_probability": ("lin", 0.01, 0.0),
    "total_sky_direct_solar_radiation_at_surface": ("lin", 2.0, 0.0),
    "cloud_base_height": ("lin", 10.0, 0.0),
}

# Smooth in time: these are stored as hour-to-hour differences (first hour
# absolute), which made them 10-25 % smaller; a reader takes the cumulative sum
# from the first hour of the week. Delta makes the intermittent fields larger.
ERA5_ARCHIVE_DELTA_VARS = (
    "instantaneous_10m_wind_gust",
    "mean_sea_level_pressure",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_column_ozone",
    "surface_pressure",
)

# int16 fill: padding rows past the last latitude, and cloud base where there is none.
ERA5_ARCHIVE_FILL = -32768
# The only archive variable where NaN is a value rather than missing data.
ERA5_ARCHIVE_NAN_VARS = ("cloud_base_height",)

# One store per week, ERA5_archive/<week start>.zarr, weeks counted from ERA5's
# first day (1940-01-01, a Monday) so every week keeps the same chunk boundaries.
ERA5_ARCHIVE_STORE_DIR = "ERA5_archive"
ERA5_ARCHIVE_WEEK_EPOCH = "1940-01-01"
ERA5_ARCHIVE_WEEK_HOURS = 7 * 24

# Google ARCO source variables of each archive variable (summed by the builder)
# and the factor from Google's units to the archive's API units. The reader divides
# by the factor to give the API Google's units back, and puts a merged rate in the
# first source column with 0 in the second: the API only uses the large-scale plus
# convective sums, and interpolates linearly, so the result is the same.
ERA5_ARCHIVE_SOURCES = {
    "instantaneous_10m_wind_gust": (("instantaneous_10m_wind_gust",), 1.0),
    "mean_sea_level_pressure": (("mean_sea_level_pressure",), 0.01),  # Pa -> hPa
    "2m_temperature": (("2m_temperature",), 1.0),
    "2m_dewpoint_temperature": (("2m_dewpoint_temperature",), 1.0),
    "10m_u_component_of_wind": (("10m_u_component_of_wind",), 1.0),
    "10m_v_component_of_wind": (("10m_v_component_of_wind",), 1.0),
    "precipitation_type": (("precipitation_type",), 1.0),
    "total_precipitation": (("total_precipitation",), 1000.0),  # m -> mm
    # kg m-2 s-1 -> mm/h
    "rain_rate": (("large_scale_rain_rate", "convective_rain_rate"), 3600.0),
    "snowfall_rate_water_equivalent": (
        (
            "large_scale_snowfall_rate_water_equivalent",
            "convective_snowfall_rate_water_equivalent",
        ),
        3600.0,
    ),
    "total_column_ozone": (("total_column_ozone",), 46696.0),  # kg m-2 -> DU
    "total_cloud_cover": (("total_cloud_cover",), 100.0),  # 0-1 -> %
    # Hourly accumulations in J m-2 -> mean W m-2 over the hour.
    "downward_uv_radiation_at_the_surface": (
        ("downward_uv_radiation_at_the_surface",),
        1 / 3600,
    ),
    "surface_solar_radiation_downwards": (
        ("surface_solar_radiation_downwards",),
        1 / 3600,
    ),
    "convective_available_potential_energy": (
        ("convective_available_potential_energy",),
        1.0,
    ),
    "surface_pressure": (("surface_pressure",), 0.01),  # Pa -> hPa
    "total_sky_direct_solar_radiation_at_surface": (
        ("total_sky_direct_solar_radiation_at_surface",),
        1 / 3600,
    ),
    "cloud_base_height": (("cloud_base_height",), 1.0),  # m, NaN where there is none
}

# DWD MOSMIX variable indices
# These match the zarr_vars order in DWD_Mosmix_Local_Ingest.py:
# time, TMP_2maboveground, DPT_2maboveground, RH_2maboveground,
# PRES_meansealevel, UGRD_10maboveground, VGRD_10maboveground,
# GUST_surface, APCP_surface, TCDC_entireatmosphere, VIS_surface,
# PTYPE_surface, DSWRF_surface
DWD_MOSMIX = {
    "temp": 1,  # TMP_2maboveground (Kelvin)
    "dew": 2,  # DPT_2maboveground (Kelvin)
    "humidity": 3,  # RH_2maboveground (percent 0-100)
    "pressure": 4,  # PRES_meansealevel (Pa)
    "wind_u": 5,  # UGRD_10maboveground (m/s)
    "wind_v": 6,  # VGRD_10maboveground (m/s)
    "gust": 7,  # GUST_surface (m/s)
    "accum": 8,  # APCP_surface (kg/m^2 = mm, RR1c hourly precipitation)
    "cloud": 9,  # TCDC_entireatmosphere (percent 0-100)
    "vis": 10,  # VIS_surface (m)
    "ptype": 11,  # PTYPE_surface (WMO code 4677)
    "solar": 12,  # DSWRF_surface (W/m^2)
}

AIGFS = {
    "pressure": 1,
    "temp": 2,
    "wind_u": 3,
    "wind_v": 4,
    "accum": 5,
}

ECMWF_AIFS = {
    "pressure": 1,
    "temp": 2,
    "dew": 3,
    "wind_u": 4,
    "wind_v": 5,
    "ptype": 6,
    "cloud": 7,
    "prob": 8,
    "accum_mean": 9,
    "accum_stddev": 10,
}

# RAQDPS (Regional Air Quality Deterministic Prediction System) variable indices.
# Matches the stacked-zarr output order from RAQDPS_Local_Ingest.py:
# RAQDPS_OUTPUT_VARS = ("time", "PM2.5", "PM10", "NO2", "O3", "SO2")
RAQDPS = {
    "time": 0,
    "pm25": 1,
    "pm10": 2,
    "no2": 3,
    "o3": 4,
    "so2": 5,
}

# SILAM (System for Integrated modeLling of Atmospheric coMposition) variable indices.
# Matches the stacked-zarr output order from FMI_Silam_Local_Ingest.py:
# zarr_vars = ("time", "cnc_PM2_5", "cnc_PM10", "PM_FRP_column", "BLH",
#              "cnc_O3", "cnc_NO2", "cnc_SO2", "cnc_CO")
SILAM = {
    "time": 0,
    "pm25": 1,
    "pm10": 2,
    "pm_frp_column": 3,
    "blh": 4,
    "o3": 5,
    "no2": 6,
    "so2": 7,
    "co": 8,
}

# Canadian Model variable indices
HRDPS = {
    "gust": 1,
    "pressure": 2,
    "temp": 3,
    "dew": 4,
    "rh": 5,
    "wind": 6,
    "wind_dir": 7,
    "intensity": 8,
    "accum": 9,
    "ptype": 10,
    "cloud": 11,
    "uv": 12,
    "solar": 13,
    "cape": 14,
    "station_pressure": 15,
    "lifted_index": 16,
    "vertical_velocity": 17,
}

GDPS = {
    "temp": 1,
    "dew": 2,
    "rh": 3,
    "wind": 4,
    "wind_dir": 5,
    "gust": 6,
    "intensity": 7,
    "accum": 8,
    "uv": 9,
    "solar": 10,
    "station_pressure": 11,
    "cloud": 12,
    "pressure": 13,
    "ozone": 14,
    "type": 15,
    "cape": 16,
    "cin": 17,
    "vertical_velocity": 18,
    "k_index": 19,
    "lifted_index": 20,
}

# Ensamble variable indices
REPS = {
    "prob": 1,
    "accum": 2,
    "error": 3,
    "freezing_rain": 4,
    "ice": 5,
    "rain": 6,
    "snow": 7,
}

GEPS = {
    "prob": 1,
    "accum": 2,
    "error": 3,
    "freezing_rain": 4,
    "ice": 5,
    "rain": 6,
    "snow": 7,
}

# Sources that can satisfy a forecast or Time Machine request, excluding
# current-conditions-only and elevation data. URMA and ERA5 serve history.
FORECAST_SOURCES = frozenset(
    {
        "hrrr_0-18",
        "hrrr_18-48",
        "hrrr",
        "rrfs_0-18",
        "rrfs_18-83",
        "rrfs_na_0-18",
        "rrfs_na_18-83",
        "nbm",
        "gfs",
        "urma",
        "ecmwf_ifs",
        "gefs",
        "hrdps",
        "gdps",
        "geps",
        "reps",
        "dwd_mosmix",
        "aigfs",
        "aigefs",
        "ecmwf_aifs",
        "era5",
    }
)
# LibreWXR global radar composite (ingest/LibreWXR_Local_Ingest.py): composite
# reflectivity in dBZ at T+0 and each 10-minute nowcast step, NaN outside radar
# coverage.
LIBREWXR = {
    "refc": 1,
}
