"""RTMA Rapid Update analysis ingest.

Downloads the newest RTMA Rapid Update analysis (2.5 km CONUS Lambert conformal
grid, issued every 15 minutes, analysis only) for 2 m temperature, dew point and
specific humidity, 10 m wind and gust, surface pressure, visibility and total
cloud cover. Derives 2 m relative humidity, rotates the winds to earth-relative
and publishes a single-time ``RTMA_RU.zarr`` plus ``RTMA_RU.time.pickle``. The
only time stored is the analysis time; the API uses it for current conditions.

Alexander Rey, September 2025
"""

import logging
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie
from herbie.fast import Herbie_latest
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    earth_relative_wind_components,
    make_herbie_save_dir,
    mask_invalid_data,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# %% Model constants
RTMA_RU_SOURCE_PRIORITY = ["aws", "nomads"]
# RTMA-RU is issued every 15 minutes; search the last five cycles for the newest
# one that is already available.
RTMA_RU_CYCLE_FREQ = "15min"
RTMA_RU_LATEST_SEARCH_CYCLES = 5
# Specific humidity (SPFH) is only downloaded to derive relative humidity.
RTMA_RU_MATCH_STRINGS = (
    ":((DPT|TMP|SPFH):2 m above ground:)"
    "|:(GUST:10 m above ground:)"
    "|:(UGRD:10 m above ground:)"
    "|:(VGRD:10 m above ground:)"
    "|:((VIS|PRES):surface:)"
    "|:TCDC:entire atmosphere"
)

# Variables in the published store (cfgrib short names); the order is the API's
# RTMA_RU index map
zarr_vars = (
    "time",
    "vis",
    "i10fg",
    "sp",
    "t2m",
    "d2m",
    "rh",
    "tcc",
    "u10",
    "v10",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/RTMA_RU"
)
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RTMA_RU")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

# A new cycle arrives every 15 minutes, so keep retries short
RTMA_RU_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RTMA_RU_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)


def download_rtma_ru_analysis(base_time: pd.Timestamp, herbie_save_dir: str) -> None:
    """Download the analysis GRIB subset with bounded retries and a file-count check.

    The wgrib2 stats gate is skipped: RTMA-RU never needed wgrib2, and the
    transform below already masks out-of-range values to NaN, which is the
    behaviour the API relies on for partially missing analysis fields.
    """
    download_and_validate_subset(
        model="rtma_ru",
        product="anl",
        search=RTMA_RU_MATCH_STRINGS,
        dataset_name="RTMA_RU analysis",
        base_time=base_time,
        forecast_hours=[0],
        priority=RTMA_RU_SOURCE_PRIORITY,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=RTMA_RU_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=RTMA_RU_RETRY_SLEEP_SECONDS,
        skip_wgrib2_validation=True,
        wgrib2_exe="",
    )


def main() -> None:
    """Run the RTMA Rapid Update ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["RTMA"]
    final_chunk = FINAL_CHUNK_SIZES["RTMA"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    # %% Determine the model run: the newest available 15-minute analysis
    latest_run = Herbie_latest(
        model="rtma_ru",
        n=RTMA_RU_LATEST_SEARCH_CYCLES,
        freq=RTMA_RU_CYCLE_FREQ,
        product="anl",
        verbose=True,
        priority=RTMA_RU_SOURCE_PRIORITY,
        save_dir=herbie_save_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "RTMA_RU.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to RTMA_RU, ending")
        return
    logger.info("Processing RTMA_RU run %s", base_time)

    # %% Download the analysis and read it with cfgrib
    analysis_time = pd.Timestamp(base_time)
    download_rtma_ru_analysis(analysis_time, herbie_save_dir)

    # Herbie reuses the subset file downloaded above instead of fetching it again
    fh_analysis = Herbie(
        analysis_time,
        model="rtma_ru",
        product="anl",
        verbose=False,
        priority=RTMA_RU_SOURCE_PRIORITY,
        save_dir=herbie_save_dir,
    )
    xarray_herbie_list = fh_analysis.xarray(RTMA_RU_MATCH_STRINGS)

    # cfgrib returns one dataset per level type; merge them into one
    xarray_analysis_merged = xr.merge(xarray_herbie_list, compat="override")
    xarray_analysis_merged = xarray_analysis_merged.assign_coords(
        xarray_herbie_list[0].metpy.parse_cf().coords
    )

    # %% Validate the analysis time against the requested cycle
    grib_analysis_time = pd.Timestamp(xarray_analysis_merged.time.values)
    if grib_analysis_time != analysis_time:
        raise RuntimeError(
            f"RTMA_RU analysis time {grib_analysis_time} does not match the "
            f"requested cycle {analysis_time}"
        )

    # %% Derive relative humidity and earth-relative winds
    # relative_humidity_from_specific_humidity returns a dimensionless fraction (0-1)
    rh_2m = relative_humidity_from_specific_humidity(
        pressure=xarray_analysis_merged["sp"] * units.Pa,
        temperature=xarray_analysis_merged["t2m"] * units.degK,
        specific_humidity=xarray_analysis_merged["sh2"] * units("kg/kg"),
    )
    xarray_analysis_merged["rh"] = rh_2m.metpy.dequantify()

    # The Lambert conformal grid stores grid-relative winds
    u_earth, v_earth = earth_relative_wind_components(
        xarray_analysis_merged["u10"], xarray_analysis_merged["v10"]
    )
    xarray_analysis_merged["u10"].data = u_earth
    xarray_analysis_merged["v10"].data = v_earth

    # %% Replace the time coordinate with a gridded UNIX-time variable
    # The API reads time from the store like any other variable
    model_UNIX_time = xarray_analysis_merged.time.data.astype("datetime64[s]").astype(
        int
    )
    xarray_analysis_merged = xarray_analysis_merged.reset_coords("time", drop=True)
    xarray_analysis_merged["time"] = (
        ("y", "x"),
        np.full(
            (xarray_analysis_merged.sizes["y"], xarray_analysis_merged.sizes["x"]),
            model_UNIX_time,
        ),
    )

    # Specific humidity was only needed for relative humidity
    xarray_analysis_merged = xarray_analysis_merged.drop_vars("sh2")

    # A missing field would shift every later variable in the API's index map
    missing_vars = [v for v in zarr_vars if v not in xarray_analysis_merged.data_vars]
    if missing_vars:
        raise RuntimeError(
            f"RTMA_RU analysis {analysis_time} is missing variables {missing_vars}"
        )
    vars_in = list(zarr_vars)

    # Out-of-range values become NaN
    for var in vars_in:
        if var == "time":
            continue
        ds_clip = xarray_analysis_merged[var]
        if np.issubdtype(ds_clip.dtype, np.number):
            mask = (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
            xarray_analysis_merged[var] = ds_clip.where(mask)

    # %% Stack into the (variable, time, y, x) layout the API reads
    xarray_analysis_stack = (
        xarray_analysis_merged[vars_in]
        .to_stacked_array(new_dim="var", sample_dims=["y", "x"])
        .expand_dims("time", axis=1)
        .chunk(chunks={"var": -1, "time": 1, "x": process_chunk, "y": process_chunk})
        .transpose("var", "time", "y", "x")
    )
    dask_var_array = mask_invalid_data(xarray_analysis_stack.data)
    dask_var_array = pad_to_chunk_size(dask_var_array, final_chunk)

    # %% Write the final store
    rtma_ru_store_name = "RTMA_RU.zarr"
    write_final_zarr(
        dask_var_array,
        forecast_process_dir + "/" + rtma_ru_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [rtma_ru_store_name],
        base_time,
        "RTMA_RU.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("RTMA_RU run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
