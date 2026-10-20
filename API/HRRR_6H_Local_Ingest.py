# %% HRRR 6-hourly Processing script using Dask, FastHerbie, and wgrib2
# Alexander Rey, November 2023
# Note that because the hourly script saves the 1-h forecast to S3, this script doesn't have to do this


# %% Import modules
import logging
import os
import shutil
import sys
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from herbie import FastHerbie, Path
from herbie.fast import Herbie_latest

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import check_grib_stats, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    mask_invalid_data,
    mask_invalid_refc,
    pad_to_chunk_size,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/ubuntu/Weather/HRRR_6H"
)
forecast_process_path = forecast_process_dir + "/HRRR_6H_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/home/ubuntu/Weather/Prod/HRRR_6H")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


# Define the processing chunk size
process_chunk = CHUNK_SIZES["HRRR_6H"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["HRRR_6H"]


zarr_vars = (
    "time",
    "VIS_surface",
    "GUST_surface",
    "MSLMA_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "TCDC_entireatmosphere",
    "MASSDEN_8maboveground",
    "REFD_1000maboveground",
    "DSWRF_surface",
    "CAPE_surface",
)


def main() -> None:
    """Run the 6-hourly HRRR ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    # %% Define base time from the most recent run
    T0 = time.time()

    latest_run = Herbie_latest(
        model="hrrr",
        n=3,
        freq="6h",
        fxx=[48],
        product="sfc",
        verbose=False,
        priority=["aws", "google", "nomads"],
    )

    base_time = latest_run.date

    logger.info(base_time)
    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/HRRR_6H.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to HRRR_6H, ending")
        return

    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 240 hours

    # Define the subset of variables to download as a list of strings
    matchstring_2m = ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    matchstring_8m = ":(MASSDEN:8 m above ground:)"
    matchstring_su = (
        ":((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST|DSWRF|CAPE):surface:.*hour fcst)"
    )
    matchstring_10m = "(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    matchstring_cl = "(:TCDC:entire atmosphere:.*hour fcst)"
    matchstring_ap = "(:APCP:surface:0-[1-9]*)"
    matchstring_sl = "(:(MSLMA):)"
    matchstring_1000m = "(:REFD:1000 m above ground:)"

    # Merge matchstrings for download
    match_strings = (
        matchstring_2m
        + "|"
        + matchstring_su
        + "|"
        + matchstring_10m
        + "|"
        + matchstring_cl
        + "|"
        + matchstring_ap
        + "|"
        + matchstring_8m
        + "|"
        + matchstring_sl
        + "|"
        + matchstring_1000m
    )

    # Create a range of forecast lead times
    # Go from 1 to 7 to account for the weird prate approach

    hrrr_range1 = FORECAST_LEAD_RANGES["HRRR_6H"]
    # Create FastHerbie object
    FH_forecastsub = FastHerbie(
        pd.date_range(start=base_time, periods=1, freq="1h"),
        model="hrrr",
        fxx=hrrr_range1,
        product="sfc",
        verbose=False,
        priority=["aws", "google", "nomads"],
        save_dir=tmp_dir,
    )

    # Download the subsets
    FH_forecastsub.download(match_strings, verbose=False)

    # Check for download length
    if len(FH_forecastsub.file_exists) != len(hrrr_range1):
        logger.error(
            "Download failed, expected "
            + str(len(hrrr_range1))
            + " files but got "
            + str(len(FH_forecastsub.file_exists))
        )
        sys.exit(1)

    # Create list of downloaded grib files
    grib_list = [
        str(Path(x.get_localFilePath(match_strings)).expand())
        for x in FH_forecastsub.file_exists
    ]

    # Perform a check if any data seems to be invalid
    check_grib_stats(grib_list, wgrib2_path, "HRRR 6-hourly")

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        "cat "
        + " ".join(grib_list)
        + " | "
        + f"{wgrib2_path}"
        + " - "
        + " -grib "
        + forecast_process_path
        + "_wgrib2_merged.grib2"
    )

    # Run wgrib2
    run_checked(cmd, "HRRR 6-hourly wgrib2 GRIB merge")

    # Use wgrib2 to rotate the wind vectors
    # From https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb
    lambertRotation = "lambert:262.500000:38.500000:38.500000:38.500000 237.280472:1799:3000.000000 21.138123:1059:3000.000000"

    cmd2 = (
        f"{wgrib2_path}"
        + " "
        + forecast_process_path
        + "_wgrib2_merged.grib2 "
        + "-new_grid_winds earth -new_grid "
        + lambertRotation
        + " "
        + forecast_process_path
        + "_wgrib2_merged.regrid"
    )

    # Run wgrib2 to rotate winds and save as NetCDF
    run_checked(cmd2, "HRRR 6-hourly wgrib2 wind rotation")

    # Convert to NetCDF
    cmd3 = (
        f"{wgrib2_path}"
        + " "
        + forecast_process_path
        + "_wgrib2_merged.regrid"
        + " -netcdf "
        + forecast_process_path
        + "_wgrib2_merged.nc"
    )

    # Run wgrib2 to rotate winds and save as NetCDF
    run_checked(cmd3, "HRRR 6-hourly wgrib2 NetCDF conversion")

    # %% Create XArray
    # Read the netcdf file using xarray
    xarray_forecast_merged = xr.open_mfdataset(
        forecast_process_path + "_wgrib2_merged.nc"
    )

    # print(xarray_forecast_merged.variables)

    assert len(xarray_forecast_merged.time) == len(hrrr_range1), (
        "Incorrect number of timesteps! Exiting"
    )

    # %% Fix things
    # Fix precipitation accumulation timing to account for everything being a total accumulation from zero to time
    xarray_forecast_merged["APCP_surface"] = xarray_forecast_merged[
        "APCP_surface"
    ].copy(
        data=np.diff(
            xarray_forecast_merged["APCP_surface"],
            axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
            prepend=0,
        )
    )

    # Adjust smoke units to avoid rounding issues
    xarray_forecast_merged["MASSDEN_8maboveground"] = (
        xarray_forecast_merged["MASSDEN_8maboveground"] * 1e9
    )

    # Set REFD values < 5 to 0
    xarray_forecast_merged["REFD_1000maboveground"] = mask_invalid_refc(
        xarray_forecast_merged["REFD_1000maboveground"]
    )

    # No chunking since only one time step

    # with ProgressBar():
    # xarray_forecast_merged.to_netcdf(forecast_process_path + 'merged_netcdf.nc', encoding=encoding)
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": 31, "x": process_chunk, "y": process_chunk}
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_xr_merged.zarr",
            mode="w",
            consolidated=False,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # Remove wgrib2 temp files
    os.remove(forecast_process_path + "_wgrib2_merged.grib2")
    os.remove(forecast_process_path + "_wgrib2_merged.regrid")
    os.remove(forecast_process_path + "_wgrib2_merged.nc")

    # %% Format as dask and save as zarr
    #####################################################################################################\

    # Dask Setup
    daskVarArrayList = []

    # This is quite a bit simpler since there's no historic data being ingested

    for daskVarIDX, dask_var in enumerate(zarr_vars[:]):
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_xr_merged.zarr",
            component=dask_var,
            inline_array=True,
        )

        if dask_var == "time":
            # Create a time array with the same shape
            daskCatTimes = daskForecastArray.astype("float32")

            # Get times as numpy
            npCatTimes = daskCatTimes.compute()

            daskArrayOut = da.from_array(
                np.tile(
                    np.expand_dims(np.expand_dims(npCatTimes, axis=1), axis=1),
                    (1, 1059, 1799),
                )
            ).rechunk((len(npCatTimes), process_chunk, process_chunk))

            daskVarArrayList.append(daskArrayOut)

        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
                    (len(npCatTimes), process_chunk, process_chunk)
                ).astype("float32")
            )

        logger.info(dask_var)

    # Merge the arrays
    # into a single 4D array
    daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

    # Mask out invalid data
    daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

    # Write out to disk
    # This intermediate step is necessary to avoid memory overflow
    # with ProgressBar():
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # Add padding to the zarr store for main forecast chunking
    daskVarArrayStackDisk_main = pad_to_chunk_size(daskVarArrayStackDisk, final_chunk)

    # Write the final store
    hrrr_6h_store_name = "HRRR_6H.zarr.zip" if save_type == "S3" else "HRRR_6H.zarr"
    write_final_zarr(
        daskVarArrayStackDisk_main,
        forecast_process_dir + "/" + hrrr_6h_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [hrrr_6h_store_name],
        base_time,
        "HRRR_6H.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)

    # Test Read
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
