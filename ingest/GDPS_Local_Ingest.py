"""GDPS deterministic forecast ingest.

Downloads the surface fields of the newest MSC Global Deterministic Prediction
System run that reaches 240 h (15 km / 0.15 degree global latitude/longitude
grid; hourly leads to 83 h, then 3-hourly to 240 h) from the MSC Datamart,
converts them to NetCDF with wgrib2, de-accumulates precipitation and solar
radiation, interpolates every field to an hourly axis and publishes
``GDPS.zarr`` plus ``GDPS.time.pickle``. The store starts at the first forecast
lead (run + 1 h).

Alexander Rey, April 2026
"""

import logging
import os
import re
import shutil
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
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    broadcast_times_to_grid,
    clear_directory,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    deaccumulate_energy_to_flux,
    deaccumulate_to_hourly_rate,
    interp_time_map_blocks_nan,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
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
# Hourly leads through 83 h, then 3-hourly through 240 h
GDPS_FORECAST_HOURS = [*FORECAST_LEAD_RANGES["GDPS_1"], *FORECAST_LEAD_RANGES["GDPS_2"]]
# Herbie's 15 km global latitude/longitude product ("15km/grib2/lat_lon" is an
# alias for it)
GDPS_PRODUCT = "15km"
# HerbieLatest steps back 6 hours per period; 7 periods covers the last
# 42 hours, enough to always reach a run with a 240 h lead.
GDPS_LATEST_RUN_SEARCH_PERIODS = 7
# The run is complete once the last lead of 2 m temperature is on the Datamart
GDPS_LATEST_RUN_PROBE = {"variable": "AirTemp", "level": "AGL-2m"}

# MSC publishes one GRIB file per variable, level and lead. Each entry is the
# (Datamart variable, level) pair of one downloaded field.
GDPS_MATCH_STRINGS = [
    ("AirTemp", "AGL-2m"),
    ("DewPoint", "AGL-2m"),
    ("RelativeHumidity", "AGL-2m"),
    ("WindSpeed", "AGL-10m"),
    ("WindDir", "AGL-10m"),
    ("WindGust", "AGL-10m"),
    ("PrecipRate", "Sfc"),
    ("Precip-Accum", "Sfc"),
    ("PrecipType-Instant", "Sfc"),
    ("UVIndex", "Sfc"),
    ("DownwardShortwaveRadiationFlux-Accum", "Sfc"),
    ("CAPE", "Sfc"),
    ("Pressure", "Sfc"),
    ("TotalCloudCover", "Sfc"),
    ("CIN", "Sfc"),
    ("Pressure", "MSL"),
    ("VerticalVelocity", "IsbL-0500"),
    ("KIndex", "Sfc"),
    ("O3", "EAtm"),
]

# Some fields are only published on a subset of the leads. They are requested
# for every lead, but only these are expected to exist; wgrib2 leaves the
# missing steps as NaN and the NaN-aware time interpolation fills them.
# Precipitation type and the instability indices are 3-hourly through 168 h,
# then 6-hourly; 500 hPa vertical velocity is 3-hourly throughout.
GDPS_REDUCED_FORECAST_HOURS = [
    hour
    for hour in GDPS_FORECAST_HOURS
    if hour % 3 == 0 and (hour <= 168 or hour % 6 == 0)
]
GDPS_3_HOURLY_FORECAST_HOURS = [hour for hour in GDPS_FORECAST_HOURS if hour % 3 == 0]
GDPS_EXPECTED_FORECAST_HOURS = {
    ("PrecipType-Instant", "Sfc"): GDPS_REDUCED_FORECAST_HOURS,
    ("CAPE", "Sfc"): GDPS_REDUCED_FORECAST_HOURS,
    ("CIN", "Sfc"): GDPS_REDUCED_FORECAST_HOURS,
    ("KIndex", "Sfc"): GDPS_REDUCED_FORECAST_HOURS,
    ("VerticalVelocity", "IsbL-0500"): GDPS_3_HOURLY_FORECAST_HOURS,
}

# GRIB variables outside the generic wgrib2 stats range for valid reasons:
# DSWRF is accumulated energy (J/m^2, far above the range maximum), and CIN and
# the K index (below -170 in cold, dry polar air) are routinely below the range
# minimum.
GDPS_EXCLUDED_STATS_VARIABLES = {
    ("DownwardShortwaveRadiationFlux-Accum", "Sfc"): ["DSWRF"],
    ("CIN", "Sfc"): ["CIN"],
    ("KIndex", "Sfc"): ["KX"],
}

# Variables in the published store; the order is the API's GDPS index map
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

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# When running the docker container, this should be "/build/wgrib2_build/bin/wgrib2 "
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

GDPS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
GDPS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)
# The wgrib2 stats check is on by default; set this only to bypass it
skip_gdps_wgrib2_validation = os.getenv(
    "skip_gdps_wgrib2_validation", "false"
).lower() in {"1", "true", "yes", "on"}


def download_gdps_variable(
    base_time: pd.Timestamp, variable: str, level: str, herbie_save_dir: str
) -> list[str]:
    """Download and validate every lead of one GDPS field with bounded retries."""
    return download_and_validate_subset(
        model="gdps",
        product=GDPS_PRODUCT,
        search=None,
        dataset_name=f"GDPS forecast {variable}:{level}",
        base_time=base_time,
        forecast_hours=GDPS_FORECAST_HOURS,
        expected_forecast_hours=GDPS_EXPECTED_FORECAST_HOURS.get(
            (variable, level), GDPS_FORECAST_HOURS
        ),
        excluded_stats_variables=GDPS_EXCLUDED_STATS_VARIABLES.get((variable, level)),
        skip_wgrib2_validation=skip_gdps_wgrib2_validation,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=GDPS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=GDPS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"variable": variable, "level": level},
        wgrib2_exe=wgrib2_path.strip(),
    )


def grib_lead_hour(path: str) -> int:
    """Return the forecast lead of an MSC GDPS file (``..._PT003H.grib2``)."""
    match = re.search(r"_PT(\d+)H\.grib2$", os.path.basename(path))
    if match is None:
        raise ValueError(f"No forecast lead in GDPS GRIB file name {path}")
    return int(match.group(1))


def main() -> None:
    """Run the GDPS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["GDPS"]
    final_chunk = FINAL_CHUNK_SIZES["GDPS"]

    # Empty the processing directory but keep the Herbie download cache
    clear_directory(forecast_process_dir, keep=tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = HerbieLatest(
        model="gdps",
        priority=["msc"],
        periods=GDPS_LATEST_RUN_SEARCH_PERIODS,
        fxx=GDPS_FORECAST_HOURS[-1],
        product=GDPS_PRODUCT,
        verbose=False,
        save_dir=herbie_save_dir,
        **GDPS_LATEST_RUN_PROBE,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/GDPS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to GDPS, ending")
        return
    logger.info("Processing GDPS run %s", base_time)

    # %% Forecast hours: hourly then 3-hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(GDPS_FORECAST_HOURS, unit="h")
    ).values

    # %% Download each field, then convert all of them to one NetCDF
    grib_list = []
    for variable, level in GDPS_MATCH_STRINGS:
        grib_files = download_gdps_variable(base_time, variable, level, herbie_save_dir)
        grib_list += grib_files
        logger.info(
            "Download completed for GDPS forecast %s:%s, %s files downloaded.",
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
        "GDPS wgrib2 NetCDF conversion",
    )
    xarray_forecast_merged = xr.open_dataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"GDPS run {base_time} times {xarray_forecast_merged['time'].values} "
            f"do not match the expected forecast leads {GDPS_FORECAST_HOURS}"
        )

    # GDPS is on a regular latitude/longitude grid, so the spatial dimensions
    # are (latitude, longitude); read them from a field like HRDPS does.
    spatial_dims = xarray_forecast_merged["TMP_2maboveground"].dims[-2:]
    NY, NX = (xarray_forecast_merged.sizes[dim] for dim in spatial_dims)

    # %% Correct the accumulated fields
    # APCP is a total since the run start. Difference consecutive leads and
    # divide by each lead interval (1 h through 83 h, then 3 h) to get a
    # nonnegative hourly rate.
    xarray_forecast_merged["APCP_surface"].data = deaccumulate_to_hourly_rate(
        xarray_forecast_merged["APCP_surface"].data,
        GDPS_FORECAST_HOURS,
        time_axis=xarray_forecast_merged["APCP_surface"].get_axis_num("time"),
    )

    # DSWRF is accumulated energy in J/m^2. Convert each interval to its average
    # flux in W/m^2 so it matches the solar-radiation contract used by other models.
    xarray_forecast_merged["DSWRF_surface"].data = deaccumulate_energy_to_flux(
        xarray_forecast_merged["DSWRF_surface"].data,
        xarray_forecast_merged.time.values,
        time_axis=xarray_forecast_merged["DSWRF_surface"].get_axis_num("time"),
    )

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(GDPS_FORECAST_HOURS),
            spatial_dims[0]: process_chunk,
            spatial_dims[1]: process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_.zarr",
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    xarray_forecast_merged.close()

    # The merged NetCDF is large and no longer needed
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    # Hourly target axis from the first lead (run + 1 h) to the last; the
    # 3-hourly steps after 83 h are interpolated onto it below.
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0], end=forecast_valid_times[-1], freq="h"
    )
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

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

    # CIN is negative by convention and routinely far below the global
    # VALID_DATA_MIN (-100 J/kg); real runs reach about -1700 J/kg. Masking it
    # would drop exactly the strong caps the thunderstorm suppressor relies on,
    # so it is exempt here and the API bounds it with CLIP_CIN instead.
    daskVarArrayListMergeNaN = mask_invalid_data(
        da.stack(daskVarArrayList, axis=0),
        ignoreAxis=[zarr_vars.index("CIN_surface")],
    )

    # Write the stacked array to disk before interpolating; doing both in one
    # graph overflows memory
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr", overwrite=True, compute=True
        )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Interpolate to hourly and write the final store
    # Precipitation type is categorical, so it takes the nearest step instead of
    # a linear blend.
    daskVarArrayStackDiskInterp = interp_time_map_blocks_nan(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
        nearest_vars=[zarr_vars.index("PTYPE_surface")],
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    gdps_store_name = "GDPS.zarr.zip" if save_type == "S3" else "GDPS.zarr"
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

    # %% Clean up
    # Unlike the other MSC ingests, GDPS also drops its Herbie download cache
    # here; it is only reused when the same run is retried.
    shutil.rmtree(forecast_process_dir)
    logger.info("GDPS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
