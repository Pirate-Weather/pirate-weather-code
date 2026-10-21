"""HRDPS deterministic forecast ingest.

Downloads the surface fields of the newest MSC High Resolution Deterministic
Prediction System continental run that reaches 48 h (2.5 km rotated
latitude/longitude grid; hourly leads to 48 h) from the MSC Datamart, converts
them to NetCDF with wgrib2, de-accumulates precipitation and solar radiation,
and publishes ``HRDPS.zarr`` plus ``HRDPS.time.pickle``. The store starts at the
first forecast lead (run + 1 h).

Alexander Rey, April 2026
"""

import logging
import os
import re
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
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
    DOWNLOAD_CACHE_MAX_AGE_HOURS,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    broadcast_times_to_grid,
    clear_directory,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_energy_to_flux,
    deaccumulate_to_hourly_rate,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    prune_old_files,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

# Load .env before importing Herbie, which reads its configuration from the
# environment at import time
dotenv_path = find_dotenv(usecwd=True)
loaded = load_dotenv(dotenv_path, override=True)

from herbie import HerbieLatest  # noqa: E402

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# Hourly leads 1-48 h
HRDPS_FORECAST_HOURS = FORECAST_LEAD_RANGES["HRDPS"]
HRDPS_PRODUCT = "continental"
# HerbieLatest steps back 6 hours per period; 7 periods covers the last
# 42 hours.
HRDPS_LATEST_RUN_SEARCH_PERIODS = 7
# The run is complete once the last lead of 2 m temperature is on the Datamart
HRDPS_LATEST_RUN_PROBE = {"variable": "TMP", "level": "AGL-2m"}

# MSC publishes one GRIB file per variable, level and lead. Each entry is the
# (Datamart variable, level) pair of one downloaded field.
HRDPS_MATCH_STRINGS = [
    ("TMP", "AGL-2m"),
    ("DPT", "AGL-2m"),
    ("RH", "AGL-2m"),
    ("WIND", "AGL-10m"),
    ("WDIR", "AGL-10m"),
    ("GUST", "AGL-10m"),
    ("APCP", "Sfc"),
    ("PTYPE", "Sfc"),
    ("PRATE", "Sfc"),
    ("UVIndex", "Sfc"),
    ("DSWRF", "Sfc"),
    ("CAPE", "Sfc"),
    ("PRES", "Sfc"),
    ("TCDC", "Sfc"),
    ("PRMSL", "MSL"),
    ("LFTX", "ISBL_0500"),
    ("VVEL", "ISBL_0500"),
]

# GRIB variables outside the generic wgrib2 stats range for valid reasons:
# DSWRF is accumulated energy (J/m^2, far above the range maximum). CAPE uses
# -1 where no parcel is buoyant (the API clips it to 0) and -999 at the rare
# points where it could not be computed; mask_invalid_data stores those as
# missing so another model fills them.
HRDPS_EXCLUDED_STATS_VARIABLES = {
    ("DSWRF", "Sfc"): ["DSWRF"],
    ("CAPE", "Sfc"): ["CAPE"],
}

# Variables in the published store; the order is the API's HRDPS index map
zarr_vars = (
    "time",
    "GUST_10maboveground",
    "PRMSL_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "WIND_10maboveground",
    "WDIR_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "PTYPE_surface",
    "TCDC_surface",
    "UVI_surface",
    "DSWRF_surface",
    "CAPE_surface",
    "PRES_surface",
    "LFTX_500mb",
    "VVEL_500mb",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# When running the docker container, this should be "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/HRDPS")
forecast_process_path = forecast_process_dir + "/HRDPS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/HRDPS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

HRDPS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
HRDPS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)
# The wgrib2 stats check is on by default; set this only to bypass it
skip_hrdps_wgrib2_validation = os.getenv(
    "skip_hrdps_wgrib2_validation", "false"
).lower() in {"1", "true", "yes", "on"}


def download_hrdps_variable(
    base_time: pd.Timestamp, variable: str, level: str, herbie_save_dir: str
) -> list[str]:
    """Download and validate every lead of one HRDPS field with bounded retries."""
    return download_and_validate_subset(
        model="hrdps",
        product=HRDPS_PRODUCT,
        search=None,
        dataset_name=f"HRDPS forecast {variable}:{level}",
        base_time=base_time,
        forecast_hours=HRDPS_FORECAST_HOURS,
        excluded_stats_variables=HRDPS_EXCLUDED_STATS_VARIABLES.get((variable, level)),
        skip_wgrib2_validation=skip_hrdps_wgrib2_validation,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=HRDPS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=HRDPS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"variable": variable, "level": level},
        wgrib2_exe=wgrib2_path.strip(),
    )


def grib_lead_hour(path: str) -> int:
    """Return the forecast lead of an MSC HRDPS file (``..._PT003H.grib2``)."""
    match = re.search(r"_PT(\d+)H\.grib2$", os.path.basename(path))
    if match is None:
        raise ValueError(f"No forecast lead in HRDPS GRIB file name {path}")
    return int(match.group(1))


def main() -> None:
    """Run the HRDPS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["HRDPS"]
    final_chunk = FINAL_CHUNK_SIZES["HRDPS"]

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    herbie_save_dir = make_herbie_save_dir(tmp_dir)
    prune_old_files(herbie_save_dir, DOWNLOAD_CACHE_MAX_AGE_HOURS)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 48 h lead exists
    latest_run = HerbieLatest(
        model="hrdps",
        priority=["msc"],
        periods=HRDPS_LATEST_RUN_SEARCH_PERIODS,
        fxx=HRDPS_FORECAST_HOURS[-1],
        product=HRDPS_PRODUCT,
        verbose=False,
        save_dir=herbie_save_dir,
        **HRDPS_LATEST_RUN_PROBE,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/HRDPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to HRDPS, ending")
        return
    logger.info("Processing HRDPS run %s", base_time)

    # %% Forecast hours: hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(HRDPS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download each field, then convert all of them to one NetCDF
    grib_list = []
    for variable, level in HRDPS_MATCH_STRINGS:
        grib_files = download_hrdps_variable(
            base_time, variable, level, herbie_save_dir
        )
        grib_list += grib_files
        logger.info(
            "Download completed for HRDPS forecast %s:%s, %s files downloaded.",
            variable,
            level,
            len(grib_files),
        )

    # Feed wgrib2 the files in lead order so every message arrives in valid-time
    # order; the path breaks ties so the merge is deterministic.
    grib_list = sorted(grib_list, key=lambda path: (grib_lead_hour(path), path))

    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        f"{cat_gribs(grib_list)} | "
        f"{quote_path(wgrib2_path.strip())} - "
        f"-netcdf {quote_path(merged_nc_path)}",
        "HRDPS wgrib2 NetCDF conversion",
    )
    xarray_forecast_merged = xr.open_dataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"HRDPS run {base_time} times {xarray_forecast_merged['time'].values} "
            f"do not match the expected forecast leads {HRDPS_FORECAST_HOURS}"
        )

    # HRDPS is on a rotated latitude/longitude grid: latitude and longitude are
    # 2-D coordinates over the y/x dimensions, so the grid size comes from a
    # field's dimensions (either coordinate's size would be NY * NX).
    spatial_dims = xarray_forecast_merged["TMP_2maboveground"].dims[-2:]
    NY, NX = (xarray_forecast_merged.sizes[dim] for dim in spatial_dims)

    # %% Correct the accumulated fields
    # APCP is a total since the run start; difference consecutive leads (1 h
    # apart) to get the hourly rate. GRIB packing can make a difference slightly
    # negative, which the shared helper clamps to zero.
    xarray_forecast_merged["APCP_surface"].data = deaccumulate_to_hourly_rate(
        xarray_forecast_merged["APCP_surface"].data,
        HRDPS_FORECAST_HOURS,
        time_axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
    )

    # DSWRF is accumulated energy in J/m^2. Convert each interval to its average
    # flux in W/m^2 so it matches the solar-radiation contract used by other models.
    xarray_forecast_merged["DSWRF_surface"].data = deaccumulate_energy_to_flux(
        xarray_forecast_merged["DSWRF_surface"].data,
        xarray_forecast_merged.time.values,
        time_axis=xarray_forecast_merged["DSWRF_surface"].get_axis_num("time"),
    )

    xarray_forecast_chunked = xarray_forecast_merged.chunk(
        chunks={
            "time": len(HRDPS_FORECAST_HOURS),
            spatial_dims[0]: process_chunk,
            spatial_dims[1]: process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_chunked.to_zarr(
            forecast_process_path + "_.zarr",
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    # Close the dataset open_dataset returned; closing the chunked copy leaves the
    # NetCDF open, holding its disk space after the remove below until main() returns.
    xarray_forecast_merged.close()

    # The merged NetCDF is large and no longer needed
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second

    daskVarArrayList = []
    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            npForecastTimes = daskForecastArray.astype("float32").compute()
            # Check that the stored times are aligned with the expected forecast times
            validate_stacked_time_alignment(stacked_timesUnix, npForecastTimes)
            # The API reads time from the store like any other variable
            daskVarArrayList.append(
                broadcast_times_to_grid(npForecastTimes, NY, NX, process_chunk)
            )
        else:
            daskVarArrayList.append(
                daskForecastArray.rechunk(
                    (len(stacked_timesUnix), process_chunk, process_chunk)
                ).astype("float32")
            )

    daskVarArrayListMergeNaN = mask_invalid_data(da.stack(daskVarArrayList, axis=0))

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr", overwrite=True, compute=True
        )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Write the final store (HRDPS is already hourly, run + 1 h to + 48 h)
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDisk, final_chunk
    )

    hrdps_store_name = "HRDPS.zarr.zip" if save_type == "S3" else "HRDPS.zarr"
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/" + hrdps_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [hrdps_store_name],
        base_time,
        "HRDPS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up, keeping the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    logger.info("HRDPS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
