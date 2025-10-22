# %% Script to initialize the ERA5 data from the Google Cloud public dataset
# Alexander Rey. October 2025

# Optimize dask for reading large zarr datasets
import dask
import xarray as xr

dask.config.set(
    {
        "array.slicing.split_large_chunks": True,
        "optimization.fuse.active": True,
    }
)


# Function to initialize in ERA5 xarray dataset
def init_ERA5():
    # Open the ERA5 dataset from Google Cloud
    dsERA5 = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks={"time": 48},
        storage_options=dict(token="anon"),
    )

    lats = dsERA5["latitude"][:]
    lons = dsERA5["longitude"][:]
    times = dsERA5["time"][:]

    return dsERA5, lats, lons, times


# Return key ERA5 variables
# zarrVars = [
#     "instantaneous_10m_wind_gust",
#     "mean_sea_level_pressure",
#     "2m_temperature",
#     "2m_dewpoint_temperature",
#     "10m_u_component_of_wind",
#     "10m_v_component_of_wind",
#     "mean_total_precipitation_rate",
#     "total_precipitation",
#     "large_scale_rain_rate",
#     "convective_rain_rate",
#     "large_scale_snowfall_rate_water_equivalent",
#     "convective_snowfall_rate_water_equivalent",
#     "total_column_ozone",
#     "total_cloud_cover",
#     "downward_uv_radiation_at_the_surface",
#     "surface_solar_radiation_downwards",
#     "convective_available_potential_energy",
#     "surface_pressure",
# ]
# # Missing: Vis, humidity, apparent temp, storm distance and direction, smoke, fire, reflectivity,
# import dask
# dask.config.set({
#     "array.slicing.split_large_chunks": True,
#     "optimization.fuse.active": True,
# })
# dsERA5 = xr.open_zarr(
#     'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
#     chunks={'time': 48},
#     storage_options=dict(token='anon'),
# )
#
import numpy as np
startDate = np.datetime64("1990-07-04 00:00:00", 's')
endDate = np.datetime64("1990-07-05 00:00:00", 's')
step = np.timedelta64(1, 'D')
datetimes = np.arange(startDate, endDate + step, step, dtype='datetime64[s]')
#
# # Timing test for debugging
# import time
# start_time = time.time()
#
# print(dsERA5[zarrVars].sel(time=datetimes).isel(latitude=10, longitude=10).compute())
# end_time = time.time()
#
# print(f"Time taken to load and compute: {end_time - start_time} seconds")
