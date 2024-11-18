# %% HRRR subhourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023
# Note that because the hourly script saves the 1-h forecast to S3, this script doesn't have to do this


# %% Import modules
from herbie import Herbie, FastHerbie, wgrib2, Path
from herbie.fast import Herbie_latest

import pandas as pd
import s3fs

import zarr
from numcodecs import Blosc, Quantize, BitRound

import dask as dask
import dask.array as da
from dask.diagnostics import ProgressBar

import numpy as np
import xarray as xr
import time

import subprocess
import os
import sys
import shutil
import pickle

import netCDF4 as nc

import dask
import redis

import warnings

warnings.filterwarnings("ignore", 'This pattern is interpreted')
# %% Setup paths and parameters



wgrib2_path = os.getenv('wgrib2_path', default="/home/ubuntu/wgrib2b/grib2/wgrib2/wgrib2 ")
forecast_process_path = os.getenv('forecast_process_path', default='/home/ubuntu/data/SubH_forecast')
merge_process_dir = os.getenv('merge_process_dir', default='/home/ubuntu/data/')
tmpDIR = os.getenv('tmp_dir', default='~/data')
saveType = os.getenv('save_type', default='S3')
s3_bucket = os.getenv('save_path', default='s3://piratezarr2')

s3_save_path = '/ForecastProd/SubH/SubH_'


hisPeriod = 36

s3 = s3fs.S3FileSystem(key="AKIA2HTALZ5LWRCTHC5F",
                       secret="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")

# Create new directory for processing if it does not exist
if not os.path.exists(merge_process_dir):
    os.makedirs(merge_process_dir)
else:
    # If it does exist, remove it
    shutil.rmtree(merge_process_dir)
    os.makedirs(merge_process_dir)
    
    
if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)
if saveType == 'Download':    
  if not os.path.exists(s3_bucket):
    os.makedirs(s3_bucket)
  if not os.path.exists(s3_bucket + '/ForecastTar'):
    os.makedirs(s3_bucket + '/ForecastTar')           
        
# %% Define base time from the most recent run
# base_time = pd.Timestamp("2023-07-01 00:00")
T0 = time.time()

latestRun = Herbie_latest(model="hrrr", n=3, freq="1H", fxx=[6],
                          product="subh", verbose=False, priority='aws', save_dir=tmpDIR)

base_time = latestRun.date

print(base_time)

# Check if this is newer than the current file
if saveType == 'S3':
  # Check if the file exists and load it
  if s3.exists(s3_bucket + '/ForecastTar/SubH.time.pickle'):
      with s3.open(s3_bucket + '/ForecastTar/SubH.time.pickle', 'rb') as f:
          previous_base_time = pickle.load(f)
         

      # Compare timestamps and download if the S3 object is more recent
      if previous_base_time >= base_time:
          print('No Update to SubH, ending')
          sys.exit()

else: 
    if os.path.exists(s3_bucket + '/ForecastTar/SubH.time.pickle'):
        # Open the file in binary mode
        with open(s3_bucket + '/ForecastTar/SubH.time.pickle', 'rb') as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)
    
    

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print('No Update to SubH, ending')
            sys.exit()
            


zarrVars = ('time', 'GUST_surface', 'PRES_surface', 'TMP_2maboveground', 'DPT_2maboveground',
            'UGRD_10maboveground', 'VGRD_10maboveground',
            'PRATE_surface', 'CSNOW_surface', 'CICEP_surface', 'CFRZR_surface',
            'CRAIN_surface')
#

#####################################################################################################
# %% Download forecast data using Herbie Latest
# %% Find the latest run with 240 hours

# Do not include accum since this will only be used for currently = minutely
# Also no humidity, cloud cover, or vis data for some reason

# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP):2 m above ground:)"
matchstring_su = ":((CRAIN|CICEP|CSNOW|CFRZR|PRES|PRATE|GUST):surface:.*min fcst)"
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*min fcst)"

# Merge matchstrings for download
matchStrings = (matchstring_2m + "|" + matchstring_su + "|" + matchstring_10m)

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
hrrr_range1 = range(1, 6)
# Create FastHerbie object
FH_forecastsub = FastHerbie(pd.date_range(start=base_time, periods=1, freq='1H'),
                            model="hrrr", fxx=hrrr_range1,
                            product="subh", verbose=False, priority='aws', save_dir=tmpDIR)

# Download the subsets
FH_forecastsub.download(matchStrings, verbose=False)

# Create list of downloaded grib files
gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = ('cat ' + ' '.join(gribList) + ' | ' + f"{wgrib2_path}" + ' - ' +
       ' -grib ' + forecast_process_path + '_wgrib2_merged.grib2')

# Run wgrib2
spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")

# Check output from wgrib2
# print(spOUT.stdout)

# Use wgrib2 to rotate the wind vectors
# From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
lambertRotation = 'lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000'

cmd2 = (f"{wgrib2_path}" + ' ' + forecast_process_path + '_wgrib2_merged.grib2 ' +
        '-new_grid_winds earth -new_grid ' + lambertRotation +
        ' ' + forecast_process_path + '_wgrib2_merged.regrid')

# Run wgrib2 to rotate winds and save as NetCDF
spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")

# Check output from wgrib2
# print(spOUT2.stdout)

# Convert to NetCDF
cmd3 = (f"{wgrib2_path}" + '  ' + forecast_process_path + '_wgrib2_merged.regrid' +
        ' -netcdf ' + forecast_process_path + '_wgrib2_merged.nc')

# Run wgrib2 to rotate winds and save as NetCDF
spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")

# Check output from wgrib2
# print(spOUT3.stdout)


# %% Create XArray
# Read the netcdf file using xarray
xarray_forecast_merged = xr.open_mfdataset(forecast_process_path + '_wgrib2_merged.nc')

if (len(xarray_forecast_merged.time) != len(hrrr_range1) * 4):
  print(len(xarray_forecast_merged.time))
  print(len(hrrr_range1) * 4)
  
  assert len(xarray_forecast_merged.time) == len(hrrr_range1) * 4, "Incorrect number of timesteps! Exiting"

# Save the dataset with compression and filters for all variables
compressor = Blosc(cname='lz4', clevel=1)
filters = [BitRound(keepbits=12)]

# No chunking since only one time step
encoding = {vname: {'compressor': compressor, 'filters': filters} for vname in zarrVars[1:]}

# with ProgressBar():
# xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
xarray_forecast_merged = xarray_forecast_merged.chunk(chunks={'time': 20, 'x': 90, 'y': 90})
xarray_forecast_merged.to_zarr(forecast_process_path + '_xr_merged.zarr', encoding=encoding, mode='w')

del xarray_forecast_merged

# Remove wgrib2 temp files
os.remove(forecast_process_path + '_wgrib2_merged.grib2')
os.remove(forecast_process_path + '_wgrib2_merged.regrid')
os.remove(forecast_process_path + '_wgrib2_merged.nc')

# %% Format as dask and save as zarr
#####################################################################################################
# Convert from xarray to dask dataframe
daskArrays = []

for dask_var in zarrVars:

    if dask_var == 'time':
        daskArrays.append(
            da.from_array(
                np.tile(np.expand_dims(np.expand_dims(
                    dask.delayed(da.from_zarr)(forecast_process_path + '_xr_merged.zarr', component=dask_var,
                                               inline_array=True).compute().astype(
                        'float32').compute(), axis=1), axis=1),
                        (1, 1059, 1799))))
    else:
        # Concatenate the dask arrays through time
        daskArrays.append(dask.delayed(da.from_zarr)(forecast_process_path + '_xr_merged.zarr', component=dask_var,
                                                     inline_array=True).astype(
            'float32'))

# Stack the DataArrays into a Dask array
stacked_dask_array = dask.delayed(da.stack)(daskArrays, axis=0)

# Chunk the Dask for fast reads at one point
# Variable, time, lat, lon
chunked_dask_array = stacked_dask_array.rechunk((12, 20, 5, 5))

# Setup S3 for dask array save

compressor = Blosc(cname='zstd', clevel=3)  # Use zstd compression

# Save to Zip
zip_store = zarr.ZipStore(merge_process_dir + '/SubH.zarr.zip', compression=0)
chunked_dask_array.to_zarr(zip_store, compressor=compressor, overwrite=True
                           ).compute()
zip_store.close()
# shutil.make_archive('/tmp/SubH.zarr', 'tar', '/tmp/SubH.zarr')

# Upload to S3
if saveType == 'S3':
    # Upload to S3
    s3.put_file(merge_process_dir + '/SubH.zarr.zip', s3_bucket + '/ForecastTar/SubH.zarr.zip')
    
    # Write most recent forecast time
    with open(merge_process_dir + '/SubH.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
   
    s3.put_file(merge_process_dir + '/SubH.time.pickle', s3_bucket + '/ForecastTar/SubH.time.pickle')   
    
else:
    # Move to local
    shutil.move(merge_process_dir + '/SubH.zarr.zip', s3_bucket + '/ForecastTar/SubH.zarr.zip')

    # Write most recent forecast time
    with open(merge_process_dir + '/SubH.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
    shutil.move(merge_process_dir + '/SubH.time.pickle', s3_bucket + '/ForecastTar/SubH.time.pickle')   
      
    # Clean up
    shutil.rmtree(merge_process_dir)    
      

# Test Read
T1 = time.time()
print(T1 - T0)