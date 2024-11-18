# %% HRRRH Hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023

# %% Import modules
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

import pandas as pd
import s3fs

import zarr
from numcodecs import Blosc, Quantize, BitRound

import dask.array as da

import numpy as np
import xarray as xr
import time

import subprocess

import os
import shutil
import sys
import pickle

from dask.diagnostics import ProgressBar
import dask
import redis

import netCDF4 as nc
import warnings

warnings.filterwarnings("ignore", 'This pattern is interpreted')

# %% Setup paths and parameters
wgrib2_path = os.getenv('wgrib2_path', default="/home/ubuntu/wgrib2b/grib2/wgrib2/wgrib2 ")
forecast_process_path = os.getenv('forecast_process_path', default='/home/ubuntu/data/HRRRH_forecast')
hist_process_path = os.getenv('hist_process_path', default='/home/ubuntu/data/HRRRH_hist')
merge_process_dir = os.getenv('merge_process_dir', default='/home/ubuntu/data/')
tmpDIR = os.getenv('tmp_dir', default='~/data')
saveType = os.getenv('save_type', default='S3')
s3_bucket = os.getenv('save_path', default='s3://piratezarr2')

s3_save_path = '/ForecastProd/HRRRH/HRRRH_'


hisPeriod = 36

s3 = s3fs.S3FileSystem(key="AKIA2HTALZ5LWRCTHC5F",
                       secret="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")

# %% Define base time from the most recent run
# base_time = pd.Timestamp("2023-07-01 00:00")
T0 = time.time()

latestRun = Herbie_latest(model="hrrr", n=6, freq="1H", fxx=[18],
                          product="sfc", verbose=False, priority='aws')

base_time = latestRun.date

print(base_time)
# Check if this is newer than the current file
if saveType == 'S3':
  # Check if the file exists and load it
  if s3.exists(s3_bucket + '/ForecastTar/HRRR.time.pickle'):
      with s3.open(s3_bucket + '/ForecastTar/HRRR.time.pickle', 'rb') as f:
          previous_base_time = pickle.load(f)
         

      # Compare timestamps and download if the S3 object is more recent
      if previous_base_time >= base_time:
          print('No Update to HRRR, ending')
          sys.exit()

else: 
    if os.path.exists(s3_bucket + '/ForecastTar/HRRR.time.pickle'):
        # Open the file in binary mode
        with open(s3_bucket + '/ForecastTar/HRRR.time.pickle', 'rb') as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)
    
    

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print('No Update to HRRR, ending')
            sys.exit()
            
            
zarrVars = ('time', 'VIS_surface', 'GUST_surface', 'MSLMA_meansealevel', 'TMP_2maboveground', 'DPT_2maboveground',
            'RH_2maboveground', 'UGRD_10maboveground', 'VGRD_10maboveground',
            'PRATE_surface', 'APCP_surface', 'CSNOW_surface', 'CICEP_surface', 'CFRZR_surface',
            'CRAIN_surface', 'TCDC_entireatmosphere', 'MASSDEN_8maboveground')

# Create new directory for processing if it does not exist
if not os.path.exists(merge_process_dir):
    os.makedirs(merge_process_dir)
if not os.path.exists(tmpDIR):
    os.makedirs(tmpDIR)
if saveType == 'Download':    
  if not os.path.exists(s3_bucket):
    os.makedirs(s3_bucket)
  if not os.path.exists(s3_bucket + '/ForecastTar'):
    os.makedirs(s3_bucket + '/ForecastTar')       
#####################################################################################################
# %% Download forecast data using Herbie Latest
# Find the latest run with 240 hours


# Define the subset of variables to download as a list of strings
matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
matchstring_8m = ":(MASSDEN:8 m above ground:)"
matchstring_su = ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST):surface:.*hour fcst)"
matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
matchstring_cl = "(:TCDC:entire atmosphere:.*hour fcst)"
matchstring_ap = "(:APCP:surface:0-[1-9]*)"
matchstring_sl = "(:MSLMA:)"

# Merge matchstrings for download
matchStrings = (matchstring_2m + "|" + matchstring_su + "|" + matchstring_10m +
                "|" + matchstring_cl + "|" + matchstring_ap + "|" + matchstring_8m + "|" + matchstring_sl)

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
hrrr_range1 = range(1, 19)
# Create FastHerbie object
FH_forecastsub = FastHerbie(pd.date_range(start=base_time, periods=1, freq='1H'),
                            model="hrrr", fxx=hrrr_range1,
                            product="sfc", verbose=False, priority='aws', save_dir=tmpDIR)

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

cmd2 = (f"{wgrib2_path}" + '  ' + forecast_process_path + '_wgrib2_merged.grib2 ' +
        '-new_grid_winds earth -new_grid ' + lambertRotation +
        ' ' + forecast_process_path + '_wgrib2_merged.regrid')

# Run wgrib2 to rotate winds and save as NetCDF
spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")

# Check output from wgrib2
# print(spOUT2.stdout)

# Convert to NetCDF
cmd3 = (f"{wgrib2_path}" + '  ' + forecast_process_path + '_wgrib2_merged.regrid ' +
        ' -netcdf ' + forecast_process_path + '_wgrib2_merged.nc')

# Run wgrib2 to rotate winds and save as NetCDF
spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")

# Check output from wgrib2
# print(spOUT3.stdout)

# %% Create XArray
# Read the netcdf file using xarray
xarray_forecast_merged = xr.open_mfdataset(forecast_process_path + '_wgrib2_merged.nc')

# %% Fix things
# Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
xarray_forecast_merged['APCP_surface'] = xarray_forecast_merged['APCP_surface'].copy(
    data=np.diff(xarray_forecast_merged['APCP_surface'],
                 axis=xarray_forecast_merged['APCP_surface'].get_axis_num('time'), prepend=0))

# %% Save merged and processed xarray dataset to disk using zarr with compression
# Define the path to save the zarr dataset

# Save the dataset with compression and filters for all variables
compressor = Blosc(cname='lz4', clevel=1)
filters = [BitRound(keepbits=9)]

# No chunking since only one time step
encoding = {vname: {'compressor': compressor, 'filters': filters} for vname in zarrVars[1:]}

assert len(xarray_forecast_merged.time) == len(hrrr_range1), "Incorrect number of timesteps! Exiting"

# with ProgressBar():
# xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
xarray_forecast_merged = xarray_forecast_merged.chunk(chunks={'time': 18, 'x': 100, 'y': 100})
xarray_forecast_merged.to_zarr(forecast_process_path + 'merged_zarr.zarr', encoding=encoding, mode='w')

# Clear the xaarray dataset from memory
del xarray_forecast_merged

# Remove wgrib2 temp files
os.remove(forecast_process_path + '_wgrib2_merged.grib2')
os.remove(forecast_process_path + '_wgrib2_merged.regrid')
os.remove(forecast_process_path + '_wgrib2_merged.nc')

print('FORECAST COMPLETE')
################################################################################################
# Historic data
# %% Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
# Create the S3 filesystem

# Saving hourly forecasts means that time machine can grab 24 of them to make a daily forecast
# SubH and 48H forecasts will not be required for time machine then!

# %% Loop through the runs and check if they have already been processed to s3

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
for i in range(hisPeriod, -1, -1):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601
    s3_path = s3_bucket + '/HRRRH/HRRRH_Hist' + (base_time - pd.Timedelta(hours=i)).strftime('%Y%m%dT%H%M%SZ') + '.zarr'

    # Try to open the zarr file to check if it has already been saved
    if saveType == 'S3':
        # Create the S3 filesystem
        s3_path = s3_bucket + '/HRRRH/HRRRH_Hist' + (base_time - pd.Timedelta(hours=i)).strftime(
            '%Y%m%dT%H%M%SZ') + '.zarr'

        if s3.exists(s3_path):
            # Check that all the data is there and that the data is the right shape
            zarrCheck = zarr.open(s3fs.S3Map(root=s3_path, s3=s3, create=True), 'r')

            # # Try to open the zarr file to check if it has already been saved
            if (len(zarrCheck) - 4) == len(zarrVars):  # Subtract 4 for lat, lon, x, and y
                if zarrCheck[zarrVars[-1]].shape[1] == 1059:
                    if zarrCheck[zarrVars[-1]].shape[2] == 1799:
                        # print('Data is there and the right shape')
                        continue

    else:
        # Local Path Setup
        local_path = s3_bucket + '/HRRRH/HRRRH_Hist' + (base_time - pd.Timedelta(hours=i)).strftime(
            '%Y%m%dT%H%M%SZ') + '.zarr'

        # Check if local file exists
        if os.path.exists(local_path):
            # Check that all the data is there and that the data is the right shape
            zarrCheck = zarr.open(local_path, 'r')

            # # Try to open the zarr file to check if it has already been saved
            if (len(zarrCheck) - 4) == len(zarrVars):  # Subtract 4 for lat, lon, x, and y
                if zarrCheck[zarrVars[-1]].shape[1] == 1059:
                    if zarrCheck[zarrVars[-1]].shape[2] == 1799:
                        # print('Data is there and the right shape')
                        continue



    print('Downloading: ' + (base_time - pd.Timedelta(hours=i)).strftime('%Y%m%dT%H%M%SZ'))

    # Create a range of dates for historic data going back 48 hours
    # Since the first hour forecast is used, then the time is an hour behind
    # So data for 18:00 would be the 1st hour of the 17:00 forecast.
    DATES = pd.date_range(
        start=base_time - pd.Timedelta(str(i + 1) + "h"),
        periods=1,
        freq="1H",
    )

    # Create a range of forecast lead times
    # Only want forecast at hour 1- SLightly less accurate than initializing at hour 0 but much avoids precipitation accumulation issues
    fxx = range(1, 2)

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(DATES, model="hrrr", fxx=fxx, product="sfc", verbose=False,
                            priority='aws', save_dir=tmpDIR)

    # Download the subsets
    FH_histsub.download(matchStrings, verbose=False)

    # Use wgrib2 to rotate the wind vectors
    # From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
    lambertRotation = 'lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000'

    cmd2 = (f"{wgrib2_path}" + ' ' + str(FH_histsub.file_exists[0].get_localFilePath(matchStrings)) + ' ' +
            '-new_grid_winds earth -new_grid ' + lambertRotation +
            ' ' + hist_process_path + '_wgrib_merge.regrid')

    # Run wgrib2 to rotate winds and save as NetCDF
    spOUT2 = subprocess.run(cmd2, shell=True, capture_output=True, encoding="utf-8")

    # Convert to NetCDF
    cmd3 = (f"{wgrib2_path}" + ' ' + hist_process_path + '_wgrib_merge.regrid ' +
            ' -netcdf ' + hist_process_path + '_wgrib_merge.nc')

    # Run wgrib2 to rotate winds and save as NetCDF
    spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")

    # %% Merge the  xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib = xr.open_dataset(hist_process_path + '_wgrib_merge.nc')

    # %% Save merged and processed xarray dataset to disk using zarr with compression
    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times
    compressor = Blosc(cname='lz4', clevel=1)
    filters = [BitRound(keepbits=12)]

    # No chunking since only one time step
    encoding = {vname: {'compressor': compressor, 'filters': filters} for vname in zarrVars[1:]}

    # Save as Zarr to s3 for Time Machine
    if saveType == 'S3':
        zarrStore = s3fs.S3Map(root=s3_path, s3=s3, create=True)
    else:
        # Create local Zarr store
        zarrStore = zarr.DirectoryStore(local_path)

    xarray_his_wgrib.to_zarr(
            store=zarrStore, mode='w', consolidated=True, encoding=encoding)

    # Clear the xarray dataset from memory
    del xarray_his_wgrib

    # Remove temp file created by wgrib2
    os.remove(hist_process_path + '_wgrib_merge.regrid')
    os.remove(hist_process_path + '_wgrib_merge.nc')

    print((base_time - pd.Timedelta(hours=i)).strftime('%Y%m%dT%H%M%SZ'))

# %% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################
# %% Merge the historic and forecast datasets and then squash using dask
# Get the s3 paths to the historic data
ncHistWorking_paths = [
    s3_bucket + '/HRRRH/HRRRH_Hist' + (base_time - pd.Timedelta(hours=i)).strftime('%Y%m%dT%H%M%SZ') + '.zarr' for i in
    range(hisPeriod, -1, -1)]
# Dask
daskArrays = []
daskVarArrays = []

for dask_var in zarrVars:
    for local_ncpath in ncHistWorking_paths:
        if saveType == 'S3':
            daskVarArrays.append(da.from_zarr(local_ncpath, component=dask_var, inline_array=True,
                                              storage_options={'key': 'AKIA2HTALZ5LWRCTHC5F',
                                                               'secret': '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'}))
        else:
            daskVarArrays.append(da.from_zarr(local_ncpath, component=dask_var, inline_array=True))
    # Stack historic
    daskVarArraysStack = dask.delayed(da.stack)(daskVarArrays)

    # Add zarr Forecast
    daskForecastArray = dask.delayed(da.from_zarr)(forecast_process_path + 'merged_zarr.zarr', component=dask_var,
                                                   inline_array=True)

    if dask_var == 'time':
        # For some reason this is much faster when using numpy?
        # Also doesn't like being delayed?
        # Convert to float32 to match the other types
        daskArrays.append(
            da.from_array(
                np.tile(np.expand_dims(np.expand_dims(da.concatenate((da.squeeze(daskVarArraysStack.compute()),
                                                                      daskForecastArray.compute()), axis=0).astype(
                    'float32').compute(), axis=1), axis=1),
                        (1, 1059, 1799))))
    else:
        daskArrays.append(
            dask.delayed(da.concatenate)((dask.delayed(da.squeeze)(daskVarArraysStack), daskForecastArray), axis=0))
    daskVarArrays = []

    daskVarArrays = []

# Stack the DataArrays into a Dask array
stacked_dask_array = dask.delayed(da.stack)(daskArrays, axis=0)

# Chunk Dask Array
chunked_dask_array = stacked_dask_array.rechunk((17, 55, 2, 2)).astype('float32')

# Setup S3 for dask array save

# Define the compressor and filter
# filters = [BitRound(keepbits=9)] # Only keep ~ 3 significant digits
compressor = Blosc(cname='zstd', clevel=3)  # Use zstd compression

# Save to Zip
zip_store = zarr.ZipStore(merge_process_dir + '/HRRR.zarr.zip', compression=0)
chunked_dask_array.to_zarr(zip_store, compressor=compressor, overwrite=True
                           ).compute()
zip_store.close()

# Upload to S3
if saveType == 'S3':
    # Upload to S3
    s3.put_file(merge_process_dir + '/HRRR.zarr.zip', s3_bucket + '/ForecastTar/HRRR.zarr.zip')
    
    # Write most recent forecast time
    with open(merge_process_dir + '/HRRR.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
   
    s3.put_file(merge_process_dir + '/HRRR.time.pickle', s3_bucket + '/ForecastTar/HRRR.time.pickle')     
    
else:
    # Move to local
    shutil.move(merge_process_dir + '/HRRR.zarr.zip', s3_bucket + '/ForecastTar/HRRR.zarr.zip')

    # Write most recent forecast time
    with open(merge_process_dir + '/HRRR.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
    shutil.move(merge_process_dir + '/HRRR.time.pickle', s3_bucket + '/ForecastTar/HRRR.time.pickle')

    # Clean up
    shutil.rmtree(merge_process_dir)


# Test Read
T1 = time.time()
print(T1 - T0)