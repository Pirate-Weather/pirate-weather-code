# %% Script to test FastHerbie.py to download GDPS data
# Alexander Rey, April 2026

# %% Import modules
import logging
import os
import re
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
from dask.diagnostics import ProgressBar
from dotenv import find_dotenv, load_dotenv

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import (
    cat_gribs,
    download_and_validate_subset,
    quote_path,
    run_checked,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    clear_directory,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_energy_to_flux,
    interp_time_map_blocks_nan,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

from herbie import HerbieLatest  # noqa: E402

warnings.filterwarnings("ignore", "This pattern is interpreted")

# Logging setup
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

# Note that when running the docker container, this should be: "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GDPS")
forecast_process_path = forecast_process_dir + "/GDPS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GDPS")


save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
herbie_download_retries = positive_int_env("herbie_download_retries", 5)
herbie_retry_sleep_seconds = positive_int_env("herbie_retry_sleep_seconds", 20)
skip_gdps_wgrib2_validation = os.getenv(
    "skip_gdps_wgrib2_validation", "true"
).lower() in {"1", "true", "yes", "on"}


def deaccumulate_accumulated_field(dataset: xr.Dataset, variable_name: str) -> da.Array:
    """Convert a total accumulated field into nonnegative per-hour increments."""
    variable = dataset[variable_name]
    time_axis = variable.get_axis_num("time")
    field_data = da.asarray(variable.data)
    deaccumulated = da.diff(field_data, axis=time_axis, prepend=0)

    time_steps = (dataset.time.diff("time") / np.timedelta64(1, "h")).values.astype(
        "float32"
    )
    time_steps = np.insert(time_steps, 0, time_steps[0])
    broadcast_shape = [1] * variable.ndim
    broadcast_shape[time_axis] = len(time_steps)
    deaccumulated = deaccumulated / time_steps.reshape(broadcast_shape)

    return da.clip(deaccumulated, 0, None)


# Define the processing chunk size
process_chunk = CHUNK_SIZES["GDPS"]

# Define the final x/y chunksize
final_chunk = FINAL_CHUNK_SIZES["GDPS"]


def main() -> None:
    """Run the GDPS forecast ingest."""
    configure_herbie_request_timeouts()
    logging.basicConfig(level=logging.INFO)
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    latest_run = HerbieLatest(
        model="gdps",
        priority=["msc"],
        periods=7,
        fxx=240,
        product="15km/grib2/lat_lon",
        verbose=True,
        variable="AirTemp",
        level="AGL-2m",
        save_dir=herbie_save_dir,
    )

    base_time = latest_run.date

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/GDPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to GDPS, ending")
        return

    zarr_vars = (
        "time",
        "TMP_2maboveground",
        "DPT_2maboveground",
        "RH_2maboveground",
        "WIND_10maboveground",
        "WDIR_10maboveground",
        "GUST_10maboveground",
        "PRATE_surface",
        "APCP_surface",
        "UVI_surface",
        "DSWRF_surface",
        "PRES_surface",
        "TCDC_surface",
        "PRMSL_meansealevel",
        "TCIOZ_surface",
        "PTYPE_surface",
        "CAPE_surface",
        "CIN_surface",
        "VVEL_500mb",
        "KX_surface",
    )

    #####################################################################################################
    # %% Download forecast data using Herbie Latest
    # Find the latest run with 240 hours

    # Define the variables to download as a dictionary of variable and level pairs to match in the grib files
    match_strings = [
        {"variable": "AirTemp", "level": "AGL-2m"},
        {"variable": "DewPoint", "level": "AGL-2m"},
        {"variable": "RelativeHumidity", "level": "AGL-2m"},
        {"variable": "WindSpeed", "level": "AGL-10m"},
        {"variable": "WindDir", "level": "AGL-10m"},
        {"variable": "WindGust", "level": "AGL-10m"},
        {"variable": "PrecipRate", "level": "Sfc"},
        {"variable": "Precip-Accum", "level": "Sfc"},
        {"variable": "PrecipType-Instant", "level": "Sfc"},
        {"variable": "UVIndex", "level": "Sfc"},
        {"variable": "DownwardShortwaveRadiationFlux-Accum", "level": "Sfc"},
        {"variable": "CAPE", "level": "Sfc"},
        {"variable": "Pressure", "level": "Sfc"},
        {"variable": "TotalCloudCover", "level": "Sfc"},
        {"variable": "CIN", "level": "Sfc"},
        {"variable": "Pressure", "level": "MSL"},
        {"variable": "VerticalVelocity", "level": "IsbL-0500"},
        {"variable": "KIndex", "level": "Sfc"},
        {"variable": "O3", "level": "EAtm"},
    ]
    # Range should be 1 hour through hour 83, then 3 hourly through hour 240.
    gdps_range_1 = FORECAST_LEAD_RANGES["GDPS_1"]
    gdps_range_2 = FORECAST_LEAD_RANGES["GDPS_2"]
    gdps_file_range = [*gdps_range_1, *gdps_range_2]
    gdps_reduced_file_range = [
        hour
        for hour in gdps_file_range
        if hour % 3 == 0 and (hour <= 168 or hour % 6 == 0)
    ]
    gdps_3_hour_file_range = [hour for hour in gdps_file_range if hour % 3 == 0]
    gdps_expected_forecast_hours = {
        ("PrecipType-Instant", "Sfc"): gdps_reduced_file_range,
        ("CAPE", "Sfc"): gdps_reduced_file_range,
        ("CIN", "Sfc"): gdps_reduced_file_range,
        ("KIndex", "Sfc"): gdps_reduced_file_range,
        ("VerticalVelocity", "IsbL-0500"): gdps_3_hour_file_range,
    }
    gdps_excluded_stats_variables = {
        ("DownwardShortwaveRadiationFlux-Accum", "Sfc"): ["DSWRF"],
        ("CIN", "Sfc"): ["CIN"],
    }

    # MSC models have each variable in a separate file, so we loop through the variables and levels to download each one and then merge them later
    all_files = []
    expected_total = 0
    for g in match_strings:
        expected_forecast_hours = gdps_expected_forecast_hours.get(
            (g["variable"], g["level"]), gdps_file_range
        )
        excluded_stats_variables = gdps_excluded_stats_variables.get(
            (g["variable"], g["level"])
        )
        grib_files = download_and_validate_subset(
            model="gdps",
            product="15km",
            search=None,
            dataset_name=f"GDPS forecast {g['variable']}:{g['level']}",
            base_time=base_time,
            wgrib2_exe=wgrib2_path.strip(),
            forecast_hours=gdps_file_range,
            expected_forecast_hours=expected_forecast_hours,
            excluded_stats_variables=excluded_stats_variables,
            skip_wgrib2_validation=skip_gdps_wgrib2_validation,
            herbie_save_dir=herbie_save_dir,
            herbie_download_retries=herbie_download_retries,
            herbie_retry_sleep_seconds=herbie_retry_sleep_seconds,
            herbie_kwargs={
                "variable": g["variable"],
                "level": g["level"],
                "verbose": True,
            },
        )

        all_files += grib_files
        expected_total += len(expected_forecast_hours)

        # Log that the download was completed for this variable and level
        logger.info(
            f"Download completed for GDPS forecast {g['variable']}:{g['level']}, "
            f"{len(grib_files)} files downloaded."
        )

    # Deduplicate and sanity-check total files
    all_files = sorted(set(all_files))
    if len(all_files) < expected_total:
        logger.error(
            f"Download incomplete, expected at least {expected_total} files but got {len(all_files)}"
        )
        sys.exit(1)

    # Create ordered list of downloaded grib files from collected paths
    grib_list = all_files

    # Sort the list
    grib_list_sort = sorted(
        grib_list,
        key=lambda f: int(re.search(r"_PT(\d+)H", str(f)).group(1)),
    )

    # Create a string to pass to wgrib2 to merge all gribs into one netcdf
    cmd = (
        f"{cat_gribs(grib_list_sort)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-netcdf {quote_path(forecast_process_path + '_wgrib2_merged.nc')}"
    )

    # Run wgrib2
    run_checked(cmd, "GDPS wgrib2 NetCDF conversion")

    # %% Read the merged netcdf file using xarray (single combined file)
    xarray_forecast_merged = xr.open_dataset(
        forecast_process_path + "_wgrib2_merged.nc"
    )

    if len(xarray_forecast_merged.time) != len(gdps_file_range):
        raise ValueError("Incorrect number of timesteps! Exiting")

    # Determine grid size from merged dataset (supports rotated grids)
    NY = xarray_forecast_merged.dims.get(
        "latitude", xarray_forecast_merged["latitude"].size
    )
    NX = xarray_forecast_merged.dims.get(
        "longitude", xarray_forecast_merged["longitude"].size
    )

    # Create a new hourly time series from the first to the last forecast valid
    # time; the 3-hourly steps after hour 84 are interpolated onto it below.
    start = xarray_forecast_merged.time.min().values
    end = xarray_forecast_merged.time.max().values
    new_hourly_time = pd.date_range(start=start, end=end, freq="h")

    stacked_times = xarray_forecast_merged.time.values
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (stacked_times - unix_epoch) / one_second
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Convert precipitation totals to nonnegative hourly amounts.
    xarray_forecast_merged["APCP_surface"].data = deaccumulate_accumulated_field(
        xarray_forecast_merged, "APCP_surface"
    )

    # DSWRF is accumulated energy in J/m^2. Convert each interval to its average
    # flux in W/m^2 so it matches the solar-radiation contract used by other models.
    xarray_forecast_merged["DSWRF_surface"].data = deaccumulate_energy_to_flux(
        xarray_forecast_merged["DSWRF_surface"].data,
        xarray_forecast_merged.time.values,
        time_axis=xarray_forecast_merged["DSWRF_surface"].get_axis_num("time"),
    )

    # Save the dataset with compression and filters for all variables
    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={"time": 136, "latitude": process_chunk, "longitude": process_chunk}
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_.zarr",
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )

    # %% Delete to free memory
    del (xarray_forecast_merged,)
    T1 = time.time()

    logger.info(T1 - T0)
    os.remove(forecast_process_path + "_wgrib2_merged.nc")

    # %% Stack the forecast variables into a single 4D array using dask
    daskVarArrayList = []

    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            # Broadcast the 1D time axis over the grid so it is stored like a variable
            npForecastTimes = daskForecastArray.astype("float32").compute()

            daskArrayOut = da.from_array(
                np.tile(
                    np.expand_dims(np.expand_dims(npForecastTimes, axis=1), axis=1),
                    (1, NY, NX),
                )
            ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))

            daskVarArrayList.append(daskArrayOut)

        else:
            daskVarArrayList.append(
                daskForecastArray[:, :, :]
                .rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
                .astype("float32")
            )

        logger.info(dask_var)

    # Merge the arrays into a single 4D array
    daskVarArrayListMerge = da.stack(daskVarArrayList, axis=0)

    # Mask out invalid data
    # Ignore storm distance, since it can reach very high values that are still correct
    daskVarArrayListMergeNaN = mask_invalid_data(daskVarArrayListMerge)

    # Write out to disk
    # This intermediate step is necessary to avoid memory overflow
    with (
        ProgressBar(),
        dask.config.set(scheduler="threads", num_workers=zarr_store_workers),
    ):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

    # Read in stacked 4D array back in
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    gdps_store_name = "GDPS.zarr.zip" if save_type == "S3" else "GDPS.zarr"

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            # 1. Interpolate the stacked array to be hourly along the time axis
            daskVarArrayStackDiskInterp = interp_time_map_blocks_nan(
                daskVarArrayStackDisk,
                stacked_timesUnix=stacked_timesUnix,
                hourly_timesUnix=hourly_timesUnix,
                dtype="float32",
                fill_value=np.nan,
                nearest_vars=[zarr_vars.index("PTYPE_surface")],
            )

        # 2. Pad to chunk size
        daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
            daskVarArrayStackDiskInterp, final_chunk
        )

        # 3. Write the final store in source-aligned tiles
        write_final_zarr(
            daskVarArrayStackDiskInterpPad,
            forecast_process_dir + "/" + gdps_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [gdps_store_name],
        base_time,
        "GDPS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)

    # Timing
    T1 = time.time()
    logger.info(T1 - T0)


if __name__ == "__main__":
    main()
