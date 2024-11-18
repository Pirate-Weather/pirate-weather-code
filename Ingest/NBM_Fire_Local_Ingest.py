#%% Script to ingest NBM Fire Index

# Alexander Rey, April 2024

#%% Import modules
from herbie import  Herbie,  Path, FastHerbie
from herbie.fast import Herbie_latest
import pandas as pd
import s3fs

import zarr
import dask
import redis

from numcodecs import Blosc, BitRound

import dask.array as da
from rechunker import rechunk

import numpy as np
import xarray as xr
import time
from datetime import datetime, timedelta
import subprocess

import os
import shutil
import sys
import pickle

from dask.diagnostics import ProgressBar

import netCDF4 as nc

from itertools import chain

from scipy.interpolate import make_interp_spline

import warnings

# Scipy Interp Function
def linInterp1D(block, T_in, T_out):

    interp = make_interp_spline(T_in, block, 3, axis=0)
    interp.extrapolate = False
    interpOut = interp(T_out)
    return interpOut

def linInterp3D(block, T_in, T_out):

    # Filter large values
    bMask = block[:,1,1] < 1e10

    interp = make_interp_spline(T_in[bMask], block[bMask, :, :], 3, axis=0)
    interp.extrapolate = False
    interpOut = interp(T_out)
    return interpOut

def rounder(t):
    if t.minute >= 30:
        if t.hour == 23:
          return t.replace(second=0, microsecond=0, minute=0, hour=0, day=t.day+1)
        else:
          return t.replace(second=0, microsecond=0, minute=0, hour=t.hour+1)
    else:
        return t.replace(second=0, microsecond=0, minute=0)

warnings.filterwarnings("ignore", 'This pattern is interpreted')

#%% Setup paths and parameters
# To be changed in the Docker version
wgrib2_path           = os.getenv('wgrib2_path', default="/home/ubuntu/wgrib2/grib2/wgrib2/wgrib2 ")
forecast_process_path = os.getenv('forecast_process_path', default='/home/ubuntu/data/NBM_Fire_forecast')
hist_process_path     = os.getenv('hist_process_path', default='/home/ubuntu/data/NBM_Fire_hist')
merge_process_dir     = os.getenv('merge_process_dir', default='/home/ubuntu/data/')
tmpDIR = os.getenv('tmp_dir', default='~/data')
saveType = os.getenv('save_type', default='S3')
s3_bucket = os.getenv('save_path', default='s3://piratezarr2')

s3_save_path = '/ForecastProd/NBM_Fire/NBM_Fire_'

hisPeriod = 36

s3 = s3fs.S3FileSystem(key="AKIA2HTALZ5LWRCTHC5F",
                       secret="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")



#%% Define base time from the most recent run
T0 = time.time()

# Find latest 6-hourly run

# Start from now and work backwards in 6 hour increments
current_time = datetime.utcnow()
hour = current_time.hour
# Calculate the most recent hour from 0, 6, 12, or 18 hours ago
if hour < 6:
    recent_hour = 0
elif hour < 12:
    recent_hour = 6
elif hour < 18:
    recent_hour = 12
else:
    recent_hour = 18

# Create a new datetime object with the most recent hour
most_recent_time = datetime(current_time.year, current_time.month, current_time.day, recent_hour, 0, 0)

# Select the most recent 0,6,12,18 run
base_time = False
failCount = 0
while base_time == False:
    latestRuns = Herbie(most_recent_time,
                        model="nbm", fxx=192,
                        product="co", verbose=False, priority=['aws', 'nomdas'], save_dir=tmpDIR)
    if latestRuns.grib:
        base_time = most_recent_time
    else:
        most_recent_time = most_recent_time - timedelta(hours=6)
        failCount = failCount + 1
        print(failCount)

        if failCount == 2:
            print("No recent runs")
            exit(1)


#base_time = pd.Timestamp("2024-03-05 16:00")
# base_time = base_time - pd.Timedelta(1,'h')
print(base_time)

# Check if this is newer than the current file
if saveType == 'S3':
  # Check if the file exists and load it
  if s3.exists(s3_bucket + '/ForecastTar/NBM_Fire.time.pickle'):
      with s3.open(s3_bucket + '/ForecastTar/NBM_Fire.time.pickle', 'rb') as f:
          previous_base_time = pickle.load(f)
         

      # Compare timestamps and download if the S3 object is more recent
      if previous_base_time >= base_time:
          print('No Update to NBM_Fire, ending')
          sys.exit()

else: 
    if os.path.exists(s3_bucket + '/ForecastTar/NBM_Fire.time.pickle'):
        # Open the file in binary mode
        with open(s3_bucket + '/ForecastTar/NBM_Fire.time.pickle', 'rb') as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)
    
        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print('No Update to NBM_Fire, ending')
            sys.exit()

zarrVars = ('time', 'FOSINDX_surface')

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
#%% Download forecast data using Herbie Latest
# Set download rannges
nbm_range = range(6, 192, 6)

# Define the subset of variables to download as a list of strings
matchstring_su = ":FOSINDX:"

# Merge matchstrings for download
matchStrings =  matchstring_su


# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
# Create FastHerbie object
FH_forecastsub = FastHerbie(pd.date_range(start=base_time, periods=1, freq='6h'),
                               model="nbm", fxx=nbm_range,
                               product="co", verbose=False, priority=['aws', 'nomads'], save_dir=tmpDIR)

FH_forecastsub.download(matchStrings, verbose=False)

# Create list of downloaded grib files
try:
    gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]
except:
    print("Download Failure 1, wait 20 seconds and retry")
    time.sleep(20)
    FH_forecastsub.download(matchStrings, verbose=False)
    try:
        gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]
    except:
        print("Download Failure 2, wait 20 seconds and retry")
        time.sleep(20)
        FH_forecastsub.download(matchStrings, verbose=False)
        try:
            gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]
        except:
            print("Download Failure 3, wait 20 seconds and retry")
            time.sleep(20)
            FH_forecastsub.download(matchStrings, verbose=False)
            try:
                gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]
            except:
                print("Download Failure 4, wait 20 seconds and retry")
                time.sleep(20)
                FH_forecastsub.download(matchStrings, verbose=False)
                try:
                    gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]
                except:
                    print("Download Failure 5, wait 20 seconds and retry")
                    time.sleep(20)
                    FH_forecastsub.download(matchStrings, verbose=False)
                    try:
                        gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsub.file_exists]
                    except:
                        print("Download Failure 6, Fail")
                        exit(1)

# Create a string to pass to wgrib2 to merge all gribs into one netcdf
cmd = ('cat '  + ' '.join(gribList) + ' | ' + f"{wgrib2_path}" + ' - ' +
       ' -grib ' + forecast_process_path + '_wgrib2_merged.grib2')
# Run wgrib2
spOUT = subprocess.run(cmd, shell=True,  capture_output=True, encoding="utf-8")

# Check output from wgrib2
#print(spOUT.stdout)

# Use wgrib2 to change the order
cmd2 = (f"{wgrib2_path}" + '  ' + forecast_process_path + '_wgrib2_merged.grib2 ' + ' -ijsmall_grib ' +
          ' 1:2345 1:1597 ' + forecast_process_path + '_wgrib2_merged_order.grib')
spOUT2 = subprocess.run(cmd2, shell=True,  capture_output=True, encoding="utf-8")
os.remove(forecast_process_path + '_wgrib2_merged.grib2')

# Convert to NetCDF
cmd4 = (f"{wgrib2_path}" + '  ' + forecast_process_path + '_wgrib2_merged_order.grib ' +
        ' -set_ext_name 1 -netcdf ' + forecast_process_path + '_wgrib2_merged.nc')

# Run wgrib2 to rotate winds and save as NetCDF
spOUT4 = subprocess.run(cmd4, shell=True,  capture_output=True, encoding="utf-8")
os.remove(forecast_process_path + '_wgrib2_merged_order.grib')

#######
# Use Dask to create a merged array (too large for xarray)
# Dask
chunkx = 100
chunky = 100

# Create base xarray for time interpolation
xarray_forecast_base = xr.open_mfdataset(forecast_process_path + '_wgrib2_merged.nc')

# Check length for errors
assert len(xarray_forecast_base.time) == len(nbm_range), "Incorrect number of timesteps! Exiting"

# Create a new time series
start = xarray_forecast_base.time.min().values
end = xarray_forecast_base.time.max().values
new_hourly_time = pd.date_range(start=start - pd.Timedelta(hisPeriod, 'H'),
                                end = end, freq='H')

stacked_times = np.concatenate((pd.date_range(start=start - pd.Timedelta(hisPeriod, 'H'),
                                              end=start - pd.Timedelta(1, 'H'), freq='6H'),
                                xarray_forecast_base.time.values))

unix_epoch = np.datetime64(0, 's')
one_second = np.timedelta64(1, 's')
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second


# Drop all variables
xarray_forecast_base = xarray_forecast_base.drop_vars([i for i in xarray_forecast_base.data_vars])

# Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
# Note: don't chunk on loading since we don't know how wgrib2 chunked the files. Intead, read the variable into memory and chunk later
with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    for dask_var in zarrVars:

        daskArray = da.from_array(nc.Dataset(forecast_process_path + '_wgrib2_merged.nc')[dask_var],
                                                      lock=True)


        # Rechunk
        daskArray = daskArray.rechunk(chunks=(len(nbm_range), chunkx, chunky))

        #%% Save merged and processed xarray dataset to disk using zarr with compression
        # Define the path to save the zarr dataset
        # Save the dataset with compression and filters for all variables
        if dask_var == 'time':
            # Save the dataset without compression and filters for all variable
            daskArray.to_zarr(forecast_process_path + '_zarrs/' + dask_var + '.zarr',
                              overwrite=True)
        else:
            filters = [BitRound(keepbits=12)]  # Only keep ~ 3 significant digits
            compressor = Blosc(cname='zstd', clevel=1)  # Use zstd compression
            # Save the dataset with compression and filters for all variable
            daskArray.to_zarr(forecast_process_path + '_zarrs/' + dask_var + '.zarr',
                              filters=filters, compression=compressor, overwrite=True)


# Del to free memory
del daskArray, xarray_forecast_base

# Remove wgrib2 temp files
os.remove(forecast_process_path + '_wgrib2_merged.nc')

T1 = time.time()
print(T0 - T1)

################################################################################################
# Historic data
#%% Create a range of dates for historic data going back 48 hours, which should be enough for the daily forecast
#%% Loop through the runs and check if they have already been processed to s3

# Hourly Runs- hisperiod to 1, since the 0th hour run is needed (ends up being basetime -1H since using the 1h forecast)
for i in range(hisPeriod, 1, -6):
    # Define the path to save the zarr dataset with the run time in the filename
    # format the time following iso8601

    if saveType == 'S3':
        s3_path = s3_bucket + '/NBM/NBM_Fire_Hist' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ') + '.zarr'

        # # Try to open the zarr file to check if it has already been saved
        if s3.exists(s3_path):
            continue

    else:
        # Local Path Setup
        local_path = s3_bucket + '/NBM/NBM_Fire_Hist' + (base_time - pd.Timedelta(hours=i)).strftime(
            '%Y%m%dT%H%M%SZ') + '.zarr'

        # Check if local file exists
        if os.path.exists(local_path):
            continue

    print('Downloading: ' + (base_time - pd.Timedelta(hours=i)).strftime('%Y%m%dT%H%M%SZ'))


    # Create a range of dates for historic data going back 48 hours
    # Forward looking, which makes sense since the data at 06Z is the max from 00Z to 06Z
    DATES = pd.date_range(
        start=base_time-pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="1H",
    )

    # Create a range of forecast lead times
    # Only want forecast at hour 1- SLightly less accurate than initializing at hour 0 but much avoids precipitation accumulation issues
    fxx = [6]

    # Create FastHerbie Object.
    FH_histsub = FastHerbie(DATES, model="nbm", fxx=fxx, product="co", verbose=False, priority='aws', save_dir=tmpDIR)

    # Main Vars + Accum
    # Download the subsets
    FH_histsub.download(matchStrings, verbose=False)

    # Use wgrib2 to change the order
    cmd1 = (f"{wgrib2_path}" + '  ' + str(FH_histsub.file_exists[0].get_localFilePath(matchStrings)) + ' -ijsmall_grib ' +
            ' 1:2345 1:1597 ' + hist_process_path + '_wgrib2_merged_order.grib')
    spOUT1 = subprocess.run(cmd1, shell=True, capture_output=True, encoding="utf-8")

    # Convert to NetCDF
    cmd3 = (f"{wgrib2_path}" + ' ' + hist_process_path + '_wgrib2_merged_order.grib ' +
            ' -set_ext_name 1 -netcdf ' + hist_process_path + '_wgrib_merge.nc')
    spOUT3 = subprocess.run(cmd3, shell=True, capture_output=True, encoding="utf-8")

    #%% Merge the  xarrays
    # Read the netcdf file using xarray
    xarray_his_wgrib = xr.open_dataset(hist_process_path + '_wgrib_merge.nc')

    #%% Save merged and processed xarray dataset to disk using zarr with compression
    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times
    compressor = Blosc(cname='lz4', clevel=1)
    filters = [BitRound(keepbits=12)]

    #No chunking since only one time step
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
    os.remove(hist_process_path + '_wgrib2_merged_order.grib')
    os.remove(hist_process_path + '_wgrib_merge.nc')
    # os.remove(hist_process_path + '_ncTemp.nc')

    print((base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ'))




#%% Merge the historic and forecast datasets and then squash using dask
#####################################################################################################
#%% Merge the historic and forecast datasets and then squash using dask

# Create a zarr backed dask array
zarr_store = zarr.DirectoryStore(merge_process_dir + '/NBM_Fire_UnChunk.zarr')

compressor = Blosc(cname='zstd', clevel=3)
filters = [BitRound(keepbits=12)]

# Create a Zarr array in the store with zstd compression. Max length is 195 Forecast Hours  37
zarr_array = zarr.zeros((len(zarrVars), 217, 1597, 2345), chunks=(1, 217, 100, 100),
                        store=zarr_store, compressor=compressor,
                        dtype='float32', overwrite=True)

# Get the s3 paths to the historic data
ncLocalWorking_paths = [s3_bucket + '/NBM/NBM_Fire_Hist' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ')+ '.zarr' for i in range(hisPeriod, 1, -6)]

# Dask
daskArrays = []
daskVarArrays = []

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    for daskVarIDX, dask_var in enumerate(zarrVars):
        for local_ncpath in ncLocalWorking_paths:

            if saveType == 'S3':
                daskVarArrays.append(da.from_zarr(local_ncpath, component=dask_var, inline_array=True,
                                              storage_options={'key': 'AKIA2HTALZ5LWRCTHC5F',
                                                    'secret': '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'}))
            else:
                daskVarArrays.append(da.from_zarr(local_ncpath, component=dask_var, inline_array=True))
        # Add NC Forecast
        daskForecastArray = da.from_zarr(forecast_process_path + '_zarrs/' + dask_var + '.zarr', inline_array=True)

        if dask_var == 'time':
            daskVarArraysStack = da.stack(daskVarArrays)

            # Also doesn't like being delayed?

            daskCatTimes = da.concatenate((da.squeeze(daskVarArraysStack),
                     daskForecastArray), axis=0).astype(
                    'float32')

            #with ProgressBar():
            interpTimes = da.map_blocks(linInterp1D, daskCatTimes.rechunk(len(daskCatTimes)),
                                        stacked_timesUnix, hourly_timesUnix, dtype='float32').compute()

            daskArrayOut = np.tile(np.expand_dims(np.expand_dims(interpTimes, axis=1), axis=1), (1, 1597, 2345))

            da.to_zarr(da.from_array(np.expand_dims(
                daskArrayOut, axis=0)), zarr_array,
                region=(slice(daskVarIDX, daskVarIDX + 1), slice(0, 217), slice(0, 1597), slice(0, 2345)))


        else:
            daskVarArraysAppend = daskVarArrays.append(daskForecastArray)
            varMerged  = da.concatenate(daskVarArrays, axis=0)

           #with ProgressBar():
            da.to_zarr(da.from_array(da.expand_dims(
                da.map_blocks(linInterp3D, varMerged.rechunk((len(stacked_timesUnix), 100, 100)).astype('float32'),
                              stacked_timesUnix, hourly_timesUnix,
                              dtype='float32').compute(), axis=0)), zarr_array,
                       region=(slice(daskVarIDX, daskVarIDX + 1), slice(0, 217), slice(0, 1597), slice(0, 2345)))

        daskVarArrays = []
        varMerged = []

        print(dask_var)



    zarr_store.close()

    shutil.rmtree(forecast_process_path + '_zarrs')

    # Rechunk the zarr array
    encoding = {'compressor': Blosc(cname='zstd', clevel=3)}

    source = zarr.open(merge_process_dir + '/NBM_Fire_UnChunk.zarr')
    intermediate = merge_process_dir + '/NBM_Fire_Mid.zarr'
    target = zarr.ZipStore(merge_process_dir + '/NBM_Fire.zarr.zip', compression=0)
    rechunked = rechunk(source, target_chunks=(len(zarrVars), 217, 2, 2), target_store=target,
                    max_mem='1G',
                    temp_store=intermediate,
                    target_options=encoding)

    #with ProgressBar():
    result = rechunked.execute()

    target.close()


if saveType == 'S3':
    # Upload to S3
    s3.put_file(merge_process_dir + '/NBM_Fire.zarr.zip', s3_bucket + '/ForecastTar/NBM_Fire.zarr.zip')
    
    # Write most recent forecast time
    with open(merge_process_dir + '/NBM_Fire.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
   
    s3.put_file(merge_process_dir + '/NBM_Fire.time.pickle', s3_bucket + '/ForecastTar/NBM_Fire.time.pickle')       
else:
    # Move to local
    shutil.move(merge_process_dir + '/NBM_Fire.zarr.zip', s3_bucket + '/ForecastTar/NBM_Fire.zarr.zip')
    
    # Write most recent forecast time
    with open(merge_process_dir + '/NBM_Fire.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
    shutil.move(merge_process_dir + '/NBM_Fire.time.pickle', s3_bucket + '/ForecastTar/NBM_Fire.time.pickle')        

    # Clean up
    shutil.rmtree(merge_process_dir)
