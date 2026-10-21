"""GEFS ensemble precipitation ingest.

Downloads the precipitation and precipitation-type fields of all 30 GEFS members
(0.25 degree, 3-hourly to 240 h) for the newest 6-hourly cycle, reduces them to
ensemble statistics (precipitation probability, mean, standard deviation and
precipitation-type fractions), interpolates them to an hourly axis and publishes
``GEFS.zarr`` plus ``GEFS.time.pickle``. The store starts at the first forecast
lead (run + 3 h); the API falls back to other sources for earlier hours.

Alexander Rey, September 2023
"""

import logging
import os
import shutil
import time
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from herbie.fast import Herbie_latest

from API.constants.shared_const import (
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    INGEST_VERSION_STR,
)
from API.ingest_grib_utils import download_and_validate_subset, run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    interp_time_take_blend,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_stacked_time_alignment,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
GEFS_MEMBER_COUNT = 30
# 0.25 degree global latitude/longitude grid
GEFS_GRID_SHAPE = (721, 1440)
GEFS_FORECAST_HOURS = FORECAST_LEAD_RANGES["GEFS"]
GEFS_SOURCE_PRIORITY = ["aws", "nomads"]
# Total precipitation plus the four categorical precipitation types
GEFS_MATCH_STRINGS = "(:(CRAIN|CICEP|CSNOW|CFRZR):)|(:APCP:)"
# GEFS APCP buckets reset every 6 hours: the first lead of a bucket holds a
# 0-3 h total and the second a 0-6 h total.
GEFS_APCP_BUCKET_HOURS = 6
GEFS_LEAD_STEP_HOURS = 3
PRECIP_TYPES = ("CRAIN", "CSNOW", "CFRZR", "CICEP")

# Per-member variables, in the order they are read back after conversion
zarr_vars = (
    "time",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
)

# Variables in the published store; the order is the API's GEFS index map
probVars = (
    "time",
    "Precipitation_Prob",
    "APCP_Mean",
    "APCP_StdDev",
    "CSNOW_Prob",
    "CICEP_Prob",
    "CFRZR_Prob",
    "CRAIN_Prob",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GEFS")
forecast_process_path = forecast_process_dir + "/GEFS_Process"
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GEFS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

GEFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
GEFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_gefs_member(base_time: pd.Timestamp, member: int) -> list[str]:
    """Download and validate one member's forecast GRIB subsets with bounded retries."""
    return download_and_validate_subset(
        model="gefs",
        product="atmos.25",
        search=GEFS_MATCH_STRINGS,
        dataset_name=f"GEFS member {member} forecast",
        base_time=base_time,
        forecast_hours=list(GEFS_FORECAST_HOURS),
        priority=GEFS_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=GEFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=GEFS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"member": member},
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
    )


def main() -> None:
    """Run the GEFS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["GEFS"]
    final_chunk = FINAL_CHUNK_SIZES["GEFS"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run: the newest cycle whose 240 h lead exists
    latest_run = Herbie_latest(
        model="gefs",
        n=3,
        freq="6h",
        fxx=[GEFS_FORECAST_HOURS[-1]],
        product="atmos.25",
        verbose=True,
        member="avg",
        priority=GEFS_SOURCE_PRIORITY,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/GEFS.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to GEFS, ending")
        return
    logger.info("Processing GEFS run %s", base_time)

    # %% Forecast hours: 3-hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(GEFS_FORECAST_HOURS, unit="h")
    ).values
    second_lead_in_bucket = xr.DataArray(
        np.asarray(GEFS_FORECAST_HOURS) % GEFS_APCP_BUCKET_HOURS == 0, dims="time"
    )

    # %% Download, convert and correct each member
    member_zarr_paths = []
    for member in range(1, GEFS_MEMBER_COUNT + 1):
        grib_list = download_gefs_member(base_time, member)

        member_nc_path = forecast_process_path + f"_wgrib2_merged_m{member}.nc"
        run_checked(
            "cat "
            + " ".join(grib_list)
            + f" | {wgrib2_path} - -netcdf {member_nc_path}",
            f"GEFS member {member} wgrib2 NetCDF conversion",
        )
        xarray_wgrib = xr.open_dataset(member_nc_path)

        if not np.array_equal(xarray_wgrib["time"].values, forecast_valid_times):
            raise RuntimeError(
                f"GEFS member {member} times {xarray_wgrib['time'].values} do not "
                f"match the expected forecast leads {GEFS_FORECAST_HOURS}"
            )

        # Turn the 0-3 h / 0-6 h bucket totals into 3-hour totals by differencing
        # the second lead of each bucket, then into an hourly rate.
        apcp = xarray_wgrib["APCP_surface"]
        apcp_3h = xr.where(second_lead_in_bucket, apcp - apcp.shift(time=1), apcp)
        # Packing can leave tiny negative differences
        xarray_wgrib["APCP_surface"] = np.maximum(apcp_3h, 0) / GEFS_LEAD_STEP_HOURS

        # The categorical types are averaged over the same mixed 0-3 h / 0-6 h
        # intervals, so they can be non-zero in a 3-hour step with no precipitation.
        for precip_type in PRECIP_TYPES:
            xarray_wgrib[precip_type + "_surface"] = xarray_wgrib[
                precip_type + "_surface"
            ].where(xarray_wgrib["APCP_surface"] != 0, 0)

        xarray_wgrib_chunked = xarray_wgrib.chunk(
            chunks={
                "time": len(GEFS_FORECAST_HOURS),
                "latitude": process_chunk,
                "longitude": process_chunk,
            }
        )
        member_zarr_path = forecast_process_path + f"_xr_m{member}.zarr"
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            xarray_wgrib_chunked.to_zarr(
                member_zarr_path,
                consolidated=False,
                mode="w",
                chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
            )
        # Close the dataset open_dataset returned; closing the chunked copy
        # leaves the member NetCDF open until the variable is reassigned.
        xarray_wgrib.close()
        member_zarr_paths.append(member_zarr_path)

        # The merged NetCDF is large and no longer needed
        os.remove(member_nc_path)

    # %% Ensemble statistics across members
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    # Hourly target axis from the first lead (run + 3 h) to the last
    new_hourly_time = pd.date_range(
        start=forecast_valid_times[0], end=forecast_valid_times[-1], freq="h"
    )
    hourly_timesUnix = (new_hourly_time - unix_epoch) / one_second

    # Stack members along a new leading axis, one dask array per variable
    # (this works much better than xarray's open_mfdataset here)
    daskArrays = {
        dask_var: da.stack(
            [da.from_zarr(path, dask_var) for path in member_zarr_paths], axis=0
        )
        for dask_var in zarr_vars
    }

    daskOutput = {
        "time": daskArrays["time"][0, :],
        "Precipitation_Prob": (
            daskArrays["APCP_surface"] > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H
        ).sum(axis=0)
        / GEFS_MEMBER_COUNT,
        "APCP_Mean": daskArrays["APCP_surface"].mean(axis=0),
        "APCP_StdDev": daskArrays["APCP_surface"].std(axis=0),
    }
    # Fraction of members forecasting each precipitation type
    for precip_type in PRECIP_TYPES:
        daskOutput[precip_type + "_Prob"] = (
            daskArrays[precip_type + "_surface"].sum(axis=0) / GEFS_MEMBER_COUNT
        )

    # Write each statistic to disk; computing them all at once overflows memory
    for dask_var in probVars:
        stat_array = daskOutput[dask_var]
        to_zarr_kwargs = {}
        if dask_var != "time":
            stat_array = stat_array.rechunk(
                (len(GEFS_FORECAST_HOURS), process_chunk, process_chunk)
            )
            to_zarr_kwargs["dtype"] = "float32"
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            stat_array.to_zarr(
                forecast_process_path + "_" + dask_var + ".zarr",
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                overwrite=True,
                compute=True,
                **to_zarr_kwargs,
            )
    del daskOutput, daskArrays

    # %% Stack the statistics into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in probVars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_" + dask_var + ".zarr", inline_array=True
        )

        if dask_var == "time":
            npCatTimes = daskForecastArray.astype("float32").compute()
            # Check that the stored times are aligned with the expected forecast times
            validate_stacked_time_alignment(stacked_timesUnix, npCatTimes)
            # The API reads time from the store like any other variable
            daskVarArrayList.append(
                da.from_array(
                    np.tile(
                        npCatTimes[:, np.newaxis, np.newaxis], (1, *GEFS_GRID_SHAPE)
                    )
                ).rechunk((len(stacked_timesUnix), process_chunk, process_chunk))
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

    # %% Interpolate to hourly and write the final store
    daskVarArrayStackDiskInterp = interp_time_take_blend(
        daskVarArrayStackDisk,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        dtype="float32",
        fill_value=np.nan,
    )
    daskVarArrayStackDiskInterpPad = pad_to_chunk_size(
        daskVarArrayStackDiskInterp, final_chunk
    )

    gefs_store_name = "GEFS.zarr.zip" if save_type == "S3" else "GEFS.zarr"
    write_final_zarr(
        daskVarArrayStackDiskInterpPad,
        forecast_process_dir + "/" + gefs_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [gefs_store_name],
        base_time,
        "GEFS.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("GEFS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
