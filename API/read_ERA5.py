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
        chunks={"time": 24},
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
startDate = np.datetime64("2016-02-16 00:00:00", 's') # Snow
endDate = np.datetime64("2016-02-16 00:00:00", 's')
step = np.timedelta64(1, 'h')
datetimes = np.arange(startDate, endDate + 24 * step, step, dtype='datetime64[s]')

dsERA5['snowfall'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum()
dsERA5['total_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum()
dsERA5['mean_snowfall_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum()
dsERA5['convective_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum()
dsERA5['large_scale_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum()
dsERA5['precipitation_type'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()

# Rain
startDate = np.datetime64("2004-09-09 00:00:00", 's') # Snow
endDate = np.datetime64("2004-09-09 00:00:00", 's')
step = np.timedelta64(1, 'h')
datetimes = np.arange(startDate, endDate + 24 * step, step, dtype='datetime64[s]')

dsERA5['total_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum()
A = (dsERA5['convective_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum() +
    dsERA5['large_scale_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum() +
    dsERA5['large_scale_rain_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum() +
    dsERA5['convective_rain_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().sum())*3600
dsERA5['precipitation_type'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()


# Ice?
startDate = np.datetime64("1998-01-05 00:00:00", 's') # Snow
endDate = np.datetime64("1998-01-08 00:00:00", 's')
step = np.timedelta64(1, 'h')
datetimes = np.arange(startDate, endDate + 24 * step, step, dtype='datetime64[s]')

np.round(dsERA5['precipitation_type'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute())
np.round(dsERA5['large_scale_rain_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*10000)
np.round(dsERA5['convective_rain_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*10000)
np.round(dsERA5['large_scale_snowfall_rate_water_equivalent'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*10000)
np.round(dsERA5['convective_snowfall_rate_water_equivalent'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*10000)
np.round(dsERA5['total_precipitation'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*10000)

np.round(dsERA5['large_scale_rain_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*36000)
np.round(dsERA5['large_scale_rain_rate'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()*36000)

# Test vis
dataOut_ERA5_xr =dsERA5[ERA5.keys()].sel(
            latitude=48.64,
            longitude=-77.092 % 360,
            time=datetimes,
            method="nearest")
dataOut_ERA5 = xr.concat([dataOut_ERA5_xr[var] for var in ERA5.keys()], dim='variable')
unix_times_era5 = (
            dataOut_ERA5_xr['time'].values.astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')).astype(
    np.int64)
ERA5_MERGED = np.vstack((unix_times_era5, dataOut_ERA5.values)).T

V = estimate_visibility_from_numpy(ERA5_MERGED, ERA5, var_axis=1)

dsERA5['surface_solar_radiation_downwards'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()


dsERA5['surface_solar_radiation_downwards'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()
dsERA5['convective_available_potential_energy'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute().max()
dsERA5['2m_dewpoint_temperature'].sel(latitude=45.4215, longitude=-75.6972%360, time=datetimes, method="nearest").compute()

startDate = np.datetime64("2022-10-22 20:00:00", 's') # Snow
endDate = np.datetime64("2022-10-23 21:00:00", 's')
step = np.timedelta64(1, 'h')
datetimes = np.arange(startDate, endDate, step, dtype='datetime64[s]')


dsERA5['2m_dewpoint_temperature'].sel(latitude=48.64, longitude=-77.092%360, time=datetimes, method="nearest").compute()

import metpy as mp
from metpy.calc import relative_humidity_from_dewpoint

relative_humidity_from_dewpoint(
            dsERA5['2m_temperature'].sel(latitude=48.64, longitude=-77.092%360, time=datetimes, method="nearest").compute()* mp.units.units.degK,
            dsERA5['2m_dewpoint_temperature'].sel(latitude=48.64, longitude=-77.092%360, time=datetimes, method="nearest").compute() * mp.units.units.degK,
            phase='auto').magnitude


# UV
startDate = np.datetime64("2024-07-07 16:00:00", 's') # Snow
endDate = np.datetime64("2024-07-08 16:00:00", 's')
step = np.timedelta64(1, 'h')
datetimes = np.arange(startDate, endDate, step, dtype='datetime64[s]')


dsERA5['downward_uv_radiation_at_the_surface'].sel(latitude=48.64, longitude=-77.092%360, time=datetimes, method="nearest").compute()



#
# # Timing test for debugging
from dask.distributed import Client
client = Client()                       # start a local cluster
print("Dashboard:", client.dashboard_link)

import dask
import xarray as xr
from dask.distributed import Client
import time
import numpy as np
client = Client()  # optional: launch local Dask cluster

import gcsfs

fs = gcsfs.GCSFileSystem(token="anon", asynchronous=True, block_size=10_485_760)  # 10 MB blocks
mapper = fs.get_mapper("gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
dsERA5 = xr.open_zarr(mapper, consolidated=True, chunks={"time":24})


dsERA5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 48},
    storage_options=dict(token="anon"),
)

import zarr
store = zarr.open_group(
    "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    mode="r",
    storage_options={"token": "anon"},
)
print(list(store.keys()))



from dask.distributed import Client, LocalCluster

cluster = LocalCluster(
    n_workers=1,
    threads_per_worker=18,   # <- 18 threads on a single worker
    processes=False,          # threads, not processes
    memory_limit="32GB",      # adjust to your machine
)
client = Client(cluster)
print(client)



fs = gcsfs.GCSFileSystem(token="anon", asynchronous=True)

# Get a fsspec mapper to the Zarr store
store = fs.get_mapper(
    "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
dsERA5 = xr.open_zarr(store, chunks={"time": 24, "latitude":1, "longitude":1})

dsERA5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 24},
    storage_options=dict(token="anon"), consolidated=True
)



import time

# Read the ERA5 data for the location and time
keys = ERA5.keys()
# Select every 3rd key for testing
selKeys = [key for i, key in enumerate(keys) if i % 1 == 0]

start_time = time.time()

dataOut_ERA5_xr = dsERA5[keys].sel(
    latitude=48.64, longitude=-77.092%360, method="nearest"
).isel(time=slice(500000,500024))
# dataOut_ERA5_xr = dsERA5[keys].isel(time=slice(510000,510024), latitude=100, longitude=100)

end_timeA = time.time()

# dataOut_ERA5_xr.persist(scheduler="threads", num_workers=18)
# arrays = [store[v][500000:500024,100,100] for v in selKeys]

end_timeB = time.time()

dataOut_ERA5 = xr.concat([dataOut_ERA5_xr[var] for var in ERA5.keys()], dim='variable')
end_timeC = time.time()


print(dataOut_ERA5.values)
end_timeD = time.time()
print(f"Time taken to load and compute: {end_timeA - start_time} seconds")
print(f"Time taken to load and compute: {end_timeB - start_time} seconds")
print(f"Time taken to load and compute: {end_timeC - start_time} seconds")

print(f"Time taken to load and compute: {end_timeD - start_time} seconds")


dsERA5[['surface_solar_radiation_downwards', 'latitude','longitude','time']].isel(latitude=178, longitude=1137,time=917692+12).compute()
import datetime as datetime
A =datetime.datetime(
            year=2024,
            month=7,
            day=7,
            hour=7,
            minute=0,
        )
np.argmin(
            np.abs(dsERA5["time"].values - np.datetime64(A))
        )

baseDayUTC_Grib = (
    (
            np.datetime64(A)
            - np.datetime64(datetime.datetime(1970, 1, 1, 0, 0, 0))
    )
    .astype("timedelta64[s]")
    .astype(np.int32)
)
A.timestamp()