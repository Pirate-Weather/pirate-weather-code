#%% GEFS Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, September 2023

#%% Import modules
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

import pandas as pd
import s3fs

from numcodecs import Blosc, BitRound

import dask.array as da
from rechunker import rechunk


import numpy as np
import xarray as xr
import time

import subprocess

import os
import shutil
from dask.diagnostics import ProgressBar
import dask
import redis
import sys
import pickle

from scipy.interpolate import make_interp_spline

import zarr
import netCDF4 as nc

import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted')

# Scipy Interp Function
def linInterp(block, T_in, T_out):
    interp = make_interp_spline(T_in, block, 3, axis=0)
    interpOut = interp(T_out)
    return interpOut


#%% Define base time from the most recent run
#
T0 = time.time()

#Setup Paths
# To be changed in the Docker version
wgrib2_path           = os.getenv('wgrib2_path', default="/home/ubuntu/wgrib2/grib2/wgrib2/wgrib2 ")
forecast_process_path = os.getenv('forecast_process_path', default='/home/ubuntu/data/GEFS_forecast')
hist_process_path     = os.getenv('hist_process_path', default='/home/ubuntu/data/GEFS_historic')
merge_process_dir     = os.getenv('merge_process_dir', default='/home/ubuntu/data/')
ncForecastWorking_path = forecast_process_path + '_proc_'
tmpDIR = os.getenv('tmp_dir', default='~/data')
saveType = os.getenv('save_type', default='S3')
s3_bucket = os.getenv('save_path', default='s3://piratezarr2')

latestRun = Herbie_latest(model="gefs", n=3, freq="6H", fxx=[240],
                               product="atmos.25", verbose=False,
                                member="avg", priority='aws')

base_time = latestRun.date
# Check if this is newer than the current file
if saveType == 'S3':
  # Check if the file exists and load it
  if s3.exists(s3_bucket + '/ForecastTar/GEFS.time.pickle'):
      with s3.open(s3_bucket + '/ForecastTar/GEFS.time.pickle', 'rb') as f:
          previous_base_time = pickle.load(f)
         

      # Compare timestamps and download if the S3 object is more recent
      if previous_base_time >= base_time:
          print('No Update to GFS, ending')
          sys.exit()

else: 
    if os.path.exists(s3_bucket + '/ForecastTar/GEFS.time.pickle'):
        # Open the file in binary mode
        with open(s3_bucket + '/ForecastTar/GEFS.time.pickle', 'rb') as file:
            # Deserialize and retrieve the variable from the file
            previous_base_time = pickle.load(file)
    
    

        # Compare timestamps and download if the S3 object is more recent
        if previous_base_time >= base_time:
            print('No Update to GEFS, ending')
            sys.exit()
            
            
print(base_time)
#base_time = pd.Timestamp("2024-02-29 18:00:00")

zarrVars = ('time', 'APCP_surface', 'CSNOW_surface', 'CICEP_surface', 'CFRZR_surface',
            'CRAIN_surface')

probVars = ('time', 'Precipitation_Prob', 'APCP_Mean', 'APCP_StdDev', 'CSNOW_Prob', 'CICEP_Prob', 'CFRZR_Prob',
            'CRAIN_Prob')


hisPeriod = 36
s3_save_path = '/ForecastProd/GEFS/GEFS_Prob_'

s3 = s3fs.S3FileSystem(key="AKIA2HTALZ5LWRCTHC5F",
                       secret="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")

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
#%% Download forecast data for all 30 members to find percentages

# Define the subset of variables to download as a list of strings
matchstring_su = "(:(CRAIN|CICEP|CSNOW|CFRZR):)"
matchstring_ap = "(:APCP:)"

# Merge matchstrings for download
matchStrings = (matchstring_su + "|" + matchstring_ap)

# Create a range of forecast lead times
# Go from 1 to 7 to account for the weird prate approach
gefs_range = range(3,241,3)

# Create FastHerbie object for all 30 members
FH_forecastsubMembers = []
for mem in range (0, 30):
    FH_IN = FastHerbie(pd.date_range(start=base_time, periods=1, freq='6H'),
                               model="gefs", fxx=gefs_range, member=mem+1,
                               product="atmos.25", verbose=False, priority='aws', save_dir=tmpDIR)

    # Check for download length
    assert len(FH_IN.file_exists) == 80, "Incorrect number of timesteps! Exiting"

    FH_forecastsubMembers.append(FH_IN)

    # Download and process the subsets
    FH_forecastsubMembers[mem].download(matchStrings, verbose=False)

    # Create list of downloaded grib files
    gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in FH_forecastsubMembers[mem].file_exists]

    # Create a string to pass to wgrib2 to merge all gribs into one grib
    cmd = 'cat ' + ' '.join(
        gribList) + ' | ' + f"{wgrib2_path}" + ' - ' + '-netcdf ' + forecast_process_path + '_wgrib2_merged_m' + str(
        mem + 1) + '.nc'

    # Run wgrib2 to megre all the grib files
    spOUT = subprocess.run(cmd, shell=True,  capture_output=True, encoding="utf-8")
    #print(spOUT.stdout)

    # Fix precip and chunk each member
    xarray_wgrib = xr.open_dataset(forecast_process_path + '_wgrib2_merged_m' + str(mem + 1) + '.nc')

    # Change from 3 and 6 hour accumulation to 3 hour accumulation
    # Use the difference between 3 and 6 for every other timestep
    apcp_diff_xr = xarray_wgrib['APCP_surface'].diff(dim='time')
    xarray_wgrib['APCP_surface'][slice(1, None, 2), :, :] = apcp_diff_xr[slice(0, None, 2), :, :]

    # Sometimes there will be weird tiny negative values, set them to zero
    xarray_wgrib['APCP_surface'] = np.maximum(xarray_wgrib['APCP_surface'], 0)

    # Divide by 3 to get hourly accum
    xarray_wgrib['APCP_surface'] = xarray_wgrib['APCP_surface'] / 3

    # NOTE: Because the cateogical vars are mixed (0-3 and 0-6) intervals, there can be values even when there's no precip
    for var in ['CRAIN', 'CSNOW', 'CFRZR', 'CICEP']:
        xarray_wgrib[var + '_surface'] = xarray_wgrib[var + '_surface'].where(xarray_wgrib['APCP_surface']!=0, 0)

    # Get a list of all variables in the dataset
    wgribVars = list(xarray_wgrib.data_vars)

    # Define compression and chunking for each variable
    # Compress to save space
    # Save the dataset as a nc file with compression
    encoding = {vname: {'zlib': True, 'complevel': 1, 'chunksizes': (80, 60, 60)} for vname in wgribVars}
    xarray_wgrib.to_netcdf(forecast_process_path + '_xr_m' + str(mem + 1) + '.nc',
                           encoding=encoding)

    # Delete the wgrib netcdf to save space
    subprocess.run('rm ' + forecast_process_path + '_wgrib2_merged_m' + str(mem + 1) + '.nc', shell=True,  capture_output=True, encoding="utf-8")

# Create a new time series
start = xarray_wgrib.time.min().values  # Adjust as necessary
end = xarray_wgrib.time.max().values  # Adjust as necessary
new_hourly_time = pd.date_range(start=start - pd.Timedelta(hisPeriod, 'H'), end=end, freq='H')

# Plus 2 since we start at Hour 3
stacked_times = np.concatenate((pd.date_range(start=start - pd.Timedelta(hisPeriod, 'H'),
                                              end=start - pd.Timedelta(1, 'H'), freq='3H'),
                                xarray_wgrib.time.values))

unix_epoch = np.datetime64(0, 's')
one_second = np.timedelta64(1, 's')
stacked_timesUnix = (stacked_times - unix_epoch) / one_second
hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

ncLocalWorking_paths = [forecast_process_path + '_xr_m' + str(i) + '.nc' for i in range(1, 31, 1)]

# Dask
daskArrays = dict()
daskVarArrays = []

# Combine NetCDF files into a Dask Array, since it works significantly better than the xarray mfdataset appraoach
# Note that the chunks
for dask_var in zarrVars:
    for local_ncpath in ncLocalWorking_paths:
        if dask_var == 'time':
            daskVarArrays.append(da.from_array(nc.Dataset(local_ncpath)[dask_var]))
        else:
            daskVarArrays.append(da.from_array(nc.Dataset(local_ncpath)[dask_var], chunks=(80, 60, 60)))

    # Stack times together, keeping variables separate
    daskArrays[dask_var] = da.stack(daskVarArrays, axis=0)
    daskVarArrays = []

# Dict to hold output dask arrays
daskOutput = dict()

# Find the probably of precipitation greater than 0.1 mm/3h  across all members
daskOutput['Precipitation_Prob'] = ((daskArrays['APCP_surface'])>0.1).sum(axis=0)/30

# Find the average precipitation accumulation and categorical parameters across all members
for var in ['CRAIN', 'CSNOW', 'CFRZR', 'CICEP']:
    daskOutput[var + '_Prob'] =  daskArrays[var + '_surface'].sum(axis=0)/30

# Find the standard deviation of precipitation accumulation across all members
daskOutput['APCP_StdDev'] =  daskArrays['APCP_surface'].std(axis=0)

# Find the average precipitation accumulation across all members
daskOutput['APCP_Mean'] = daskArrays['APCP_surface'].mean(axis=0)

# Copy time over
daskOutput['time'] = daskArrays['time'][1,:]


filters = [BitRound(keepbits=12)] # Only keep ~ 3 significant digits
compressor = Blosc(cname='zstd', clevel=1) # Use zstd compression

for dask_var in probVars:
    #with ProgressBar():
    if dask_var == 'time':
        daskOutput[dask_var].to_zarr(ncForecastWorking_path + dask_var + '.zarr',
                       compression=compressor, overwrite=True)
    else:
        daskOutput[dask_var].to_zarr(ncForecastWorking_path + dask_var + '.zarr',
                       filters=filters, compression=compressor, overwrite=True)


#%% Delete to free memory
del xarray_wgrib, FH_forecastsubMembers, daskOutput, daskArrays, apcp_diff_xr

################################################################################################
# Historic data
#%% Create a range of dates for historic data going back 48 hours
#%% Loop through the runs and check if they have already been processed to s3

# Preprocess function for xarray to add a member dimension
def preprocess(ds):
    return ds.expand_dims('member', axis=0)


# Note: since these files are only 6 hour segments (aka 2 timesteps instead of 80), all the fancy dask stuff isn't necessary
# 6 hour runs
for i in range(hisPeriod, 0, -6):


    # s3_path_NC = s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ') + '.nc'

    # Try to open the zarr file to check if it has already been saved
    if saveType == 'S3':
        # Create the S3 filesystem
        s3_path = s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours=i)).strftime(
            '%Y%m%dT%H%M%SZ') + '.zarr'

        if s3.exists(s3_path):
            continue
    else:
        # Local Path Setup
        local_path = s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours=i)).strftime(
            '%Y%m%dT%H%M%SZ') + '.zarr'

        # Check if local file exists
        if os.path.exists(local_path):
            continue
    print('Downloading: ' + (base_time - pd.Timedelta(hours=i)).strftime('%Y%m%dT%H%M%SZ'))


    # Create a range of dates for historic data
    DATES = pd.date_range(
        start=base_time-pd.Timedelta(str(i) + "h"),
        periods=1,
        freq="6H",
    )

    # Create a range of forecast lead times
    # Forward looking, so 00Z forecast is from 03Z
    # This is what we want for accumilation variables
    FH_forecastsubMembers = []
    for mem in range (0, 30):
        FH_forecastsubMembers.append(FastHerbie(DATES,
                                   model="gefs", fxx=range(3, 7, 3), member=mem+1,
                                   product="atmos.25", verbose=False, priority='aws', save_dir=tmpDIR))
    # Download the subsets
    for mem in range(0, 30):
        # Download the subsets
        FH_forecastsubMembers[mem].download(matchStrings, verbose=False)
        # Create list of downloaded grib files
        gribList = [str(Path(x.get_localFilePath(matchStrings)).expand()) for x in
                    FH_forecastsubMembers[mem].file_exists]

        # Create a string to pass to wgrib2 to merge all gribs into one grib
        cmd = 'cat ' + ' '.join(
            gribList) + ' | ' + f"{wgrib2_path}" + ' - ' + '-netcdf ' + hist_process_path + '_wgrib2_merged_m' + str(
            mem + 1) + '.nc'

        # Run wgrib2 to merge all the grib files
        spOUT = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        #print(spOUT.stdout)

        # Open the NetCDF file with xarray to process and compress
        xarray_hist_wgrib = xr.open_dataset(hist_process_path + '_wgrib2_merged_m' + str(mem + 1) + '.nc')

        # Change from 3 and 6 hour accumulations to 3 hour accumulations
        apcp_hist_diff_xr = xarray_hist_wgrib['APCP_surface'].diff(dim='time')
        xarray_hist_wgrib['APCP_surface'][slice(1, None, 2), :, :] = apcp_hist_diff_xr[slice(0, None, 2), :, :]

        # Sometimes there will be weird negative values, set them to zero
        xarray_hist_wgrib['APCP_surface'] = np.maximum(xarray_hist_wgrib['APCP_surface'], 0)

        # Divide by 3 to get hourly accumilations
        xarray_hist_wgrib['APCP_surface'] = xarray_hist_wgrib['APCP_surface'] /3

        # NOTE: Because the cateogical vars are mixed (0-3 and 0-6) intervals, there can be values even when there's no precip
        for var in ['CRAIN', 'CSNOW', 'CFRZR', 'CICEP']:
            xarray_hist_wgrib[var + '_surface'] = xarray_hist_wgrib[var + '_surface'].where(xarray_hist_wgrib['APCP_surface'] != 0, 0)

        # Get a list of all variables in the dataset
        wgribVars = list(xarray_hist_wgrib.data_vars)


        # Save to NetCDF for prob process
        encoding = {vname: {'zlib': True, 'complevel': 1, 'chunksizes': (2, 90, 90)} for vname in zarrVars[1:]}
        xarray_hist_wgrib.to_netcdf(hist_process_path + '_xr_merged_m' + str(mem + 1) + '.nc',
                                     encoding=encoding)

        # Delete the netcdf to save space
        subprocess.run('rm ' + hist_process_path + '_wgrib2_merged_m' + str(mem + 1) + '.nc', shell=True,
                       capture_output=True, encoding="utf-8")


    #%% Calculate probabilities and standard deviation for historic data
    # Read the merged netcdf files into xarray using the preprocess function, concatenating along the member dimension
    xarray_hist_wgrib_merged = xr.open_mfdataset([hist_process_path + '_xr_merged_m' + str(mem + 1) + '.nc' for mem in range(0, 30)], engine='netcdf4',
                                            preprocess=preprocess, combine='nested', concat_dim='member',
                                            chunks={'member':30, 'time':2, 'latitude':90, 'longitude':90})


    # Create an empty xarray dataset to store the probability of precipitation greater than 1 mm
    xarray_hist_wgrib_prob = xr.Dataset()

    # Find the probably of precipitation greater than 0.1 mm/h across all members
    xarray_hist_wgrib_prob['Precipitation_Prob'] = ((xarray_hist_wgrib_merged['APCP_surface']) > 0.1).sum(dim='member') / 30

    # Find the average precipitation accumulation and categorical parameters across all members
    for var in ['CRAIN', 'CSNOW', 'CFRZR', 'CICEP']:
        xarray_hist_wgrib_prob[var + '_Prob'] = xarray_hist_wgrib_merged[var + '_surface'].sum(dim='member') / 30

    # Find the standard deviation of precipitation accumulation across all members
    xarray_hist_wgrib_prob['APCP_StdDev'] = xarray_hist_wgrib_merged['APCP_surface'].std(dim='member')

    # Find the average precipitation accumulation across all members
    xarray_hist_wgrib_prob['APCP_Mean'] = xarray_hist_wgrib_merged['APCP_surface'].mean(dim='member')

    # Chunk the xarray dataset to speed up processing
    xarray_hist_wgrib_prob = xarray_hist_wgrib_prob.chunk({'time': 2, 'latitude': 90, 'longitude': 90})


    # Save the dataset with compression and filters for all variables
    # Use the same encoding as last time but with larger chuncks to speed up read times
    # Get a list of all variables in the dataset
    compressor = Blosc(cname='lz4', clevel=1)
    filters = [BitRound(keepbits=9)]

    # Don't filter time
    encoding = {vname: {'compressor': compressor, 'filters': filters} for vname in probVars[1:]}

    # Save as zarr for timemachine
    #with ProgressBar():
    # Save as Zarr to s3 for Time Machine
    if saveType == 'S3':
        zarrStore = s3fs.S3Map(root=s3_path, s3=s3, create=True)
    else:
        # Create local Zarr store
        zarrStore = zarr.DirectoryStore(local_path)

    xarray_hist_wgrib_prob.to_zarr(
            store=zarrStore, mode='w', consolidated=True, encoding=encoding)


    # Clear memory
    del xarray_hist_wgrib_prob, xarray_hist_wgrib, xarray_hist_wgrib_merged

#####################################################################################################
#%% Merge the historic and forecast datasets and then squash using dask

# Create a zarr backed dask array
zarr_store = zarr.DirectoryStore(merge_process_dir + '/GEFS_UnChunk.zarr')

compressor = Blosc(cname='lz4', clevel=1)
filters = [BitRound(keepbits=12)]

# Create a Zarr array in the store with zstd compression
zarr_array = zarr.zeros((len(probVars), 274, 721, 1440), chunks=(1, 274, 20, 20), store=zarr_store, compressor=compressor, dtype='float32')


# Get the s3 paths to the historic data
ncLocalWorking_paths = [s3_bucket + '/GEFS/GEFS_HistProb_' + (base_time - pd.Timedelta(hours = i)).strftime('%Y%m%dT%H%M%SZ')+ '.zarr' for i in range(hisPeriod, 0, -6)]

# Dask
daskArrays = []
daskVarArrays = []

for daskVarIDX, dask_var in enumerate(probVars):
    for local_ncpath in ncLocalWorking_paths:

        if saveType == 'S3':
            daskVarArrays.append(da.from_zarr(local_ncpath, component=dask_var, inline_array=True,
                                              storage_options={'key': 'AKIA2HTALZ5LWRCTHC5F',
                                                               'secret': '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'}))
        else:
            daskVarArrays.append(da.from_zarr(local_ncpath, component=dask_var, inline_array=True))

    daskVarArraysStack = da.stack(daskVarArrays, allow_unknown_chunksizes=True)

    # Add forecast as dask array
    daskForecastArray = da.from_zarr(ncForecastWorking_path + dask_var + '.zarr', inline_array=True)

    if dask_var == 'time':
        # For some reason this is much faster when using numpy?
        # Also doesn't like being delayed?
        # Convert to float32 to match the other types
        daskVarArraysShape = da.reshape(daskVarArraysStack, (12, 1), merge_chunks=False)
        daskCatTimes = da.concatenate((da.squeeze(daskVarArraysShape), daskForecastArray), axis=0).astype('float32')


        #with ProgressBar():
        interpTimes = da.map_blocks(linInterp, daskCatTimes.rechunk((len(stacked_timesUnix))),
                                    stacked_timesUnix, hourly_timesUnix, dtype='float32').compute()

        daskArrayOut = np.tile(np.expand_dims(np.expand_dims(interpTimes, axis=1), axis=1), (1, 721, 1440))

        da.to_zarr(da.from_array(np.expand_dims(
            daskArrayOut, axis=0)), zarr_array,
            region=(slice(daskVarIDX, daskVarIDX + 1), slice(0, 274), slice(0, 721), slice(0, 1440)))

    else:
        daskVarArraysShape = da.reshape(daskVarArraysStack, (12, 721, 1440), merge_chunks=False)
        daskArrayOut = da.concatenate((daskVarArraysShape, daskForecastArray), axis=0)

        #with ProgressBar():
        da.to_zarr(da.from_array(da.expand_dims(
            da.map_blocks(linInterp, daskArrayOut[:, :, :].rechunk((len(stacked_timesUnix), 20, 20)).astype('float32'),
                          stacked_timesUnix, hourly_timesUnix,
                          dtype='float32').compute(), axis=0)), zarr_array,
                   region=(slice(daskVarIDX, daskVarIDX + 1), slice(0, 274), slice(0, 721), slice(0, 1440)))

    daskVarArrays = []

    print(dask_var)


zarr_store.close()

# Rechunk the zarr array
# encoding = {'compressor':  Blosc(cname='lz4', clevel=1), 'filters':  [BitRound(keepbits=15)]}
encoding = {'compressor':  Blosc(cname='lz4', clevel=1)}

source = zarr.open(merge_process_dir + '/GEFS_UnChunk.zarr')
intermediate = merge_process_dir + '/GEFS_Mid.zarr'
target = zarr.ZipStore(merge_process_dir + '/GEFS.zarr.zip', compression=0)
rechunked = rechunk(source, target_chunks=(8, 274, 3, 3), target_store=target,
                    max_mem='500M',
                    temp_store=intermediate,
                    target_options= encoding)

#with ProgressBar():
result = rechunked.execute()

# Save to Zip
target.close()

if saveType == 'S3':
    # Upload to S3
    s3.put_file(merge_process_dir + '/GEFS.zarr.zip', s3_bucket + '/ForecastTar/GEFS.zarr.zip')
    
    # Write most recent forecast time
    with open(merge_process_dir + '/GEFS.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
   
    s3.put_file(merge_process_dir + '/GEFS.time.pickle', s3_bucket + '/ForecastTar/GEFS.time.pickle')     
    
else:
    # Move to local
    shutil.move(merge_process_dir + '/GEFS.zarr.zip', s3_bucket + '/ForecastTar/GEFS.zarr.zip')

    # Write most recent forecast time
    with open(merge_process_dir + '/GEFS.time.pickle', 'wb') as file:
      # Serialize and write the variable to the file
      pickle.dump(base_time, file)
      
    shutil.move(merge_process_dir + '/GEFS.time.pickle', s3_bucket + '/ForecastTar/GEFS.time.pickle')   

    # Clean up
    shutil.rmtree(merge_process_dir)

T1 = time.time()

print(T1-T0)