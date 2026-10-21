"""URMA analysis ingest with short-term history.

Downloads hourly URMA analyses (2.5 km CONUS Lambert conformal grid, one analysis
per hour, published several hours after its valid time) for 2 m temperature,
dew point and specific humidity, 10 m wind and gust, surface pressure,
visibility and total cloud cover. Each hour is transformed like RTMA-RU
(derived 2 m relative humidity, earth-relative winds) and archived as its own
``URMA_Hist_<time>.zarr``; the last ``HISTORY_PERIODS["URMA"]`` hours are then
merged into ``URMA_Hist.zarr`` and published with ``URMA_Hist.time.pickle``. The
store runs from the newest analysis minus the history period up to the newest
analysis.

Alexander Rey, August 2026
"""

import logging
import os
import shutil
import time
import warnings

import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from herbie import Herbie
from herbie.fast import Herbie_latest
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    archive_tmp_zarr_and_upload,
    broadcast_times_to_grid,
    close_store,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    download_extract_historic_archive,
    earth_relative_wind_components,
    make_herbie_save_dir,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# %% Model constants
URMA_SOURCE_PRIORITY = ["aws", "nomads"]
# URMA is typically published several hours after its valid time. Search far
# enough back (hourly cycles) to cover that delay instead of assuming a recent file.
URMA_LATEST_SEARCH_CYCLES = 12
# Specific humidity (SPFH) is only downloaded to derive relative humidity.
URMA_MATCH_STRINGS = (
    ":((DPT|TMP|SPFH):2 m above ground:)"
    "|:(GUST:10 m above ground:)"
    "|:(UGRD:10 m above ground:)"
    "|:(VGRD:10 m above ground:)"
    "|:((VIS|PRES):surface:)"
    "|:TCDC:entire atmosphere"
)
# Hours of history before the newest analysis kept in the published store
his_period = HISTORY_PERIODS["URMA"]
HOURLY_STORE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# Variables in the published store (cfgrib short names); the order is the API's
# URMA index map
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

historic_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/URMA")
tmp_dir = historic_process_dir + "/Downloads"
hist_process_path = historic_process_dir + "/URMA_Historic"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/URMA")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/Prod/URMA")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

URMA_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
URMA_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)


def download_urma_analysis(valid_time: pd.Timestamp, herbie_save_dir: str) -> None:
    """Download one hourly analysis GRIB subset with bounded retries and a count check.

    The wgrib2 stats gate is skipped: URMA never needed wgrib2, and the
    transform below already masks out-of-range values to NaN, which is the
    behaviour the API relies on for partially missing analysis fields.
    """
    download_and_validate_subset(
        model="urma",
        product="anl",
        search=URMA_MATCH_STRINGS,
        dataset_name=f"URMA analysis {valid_time}",
        base_time=valid_time,
        forecast_hours=[0],
        priority=URMA_SOURCE_PRIORITY,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=URMA_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=URMA_RETRY_SLEEP_SECONDS,
        skip_wgrib2_validation=True,
        wgrib2_exe="",
    )


def main() -> None:
    """Run the URMA ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    # URMA writes its stores with plain zarr/xarray calls, so only the open-file
    # and concurrency limits matter here; the worker count is not used.
    configure_zarr_writes()

    process_chunk = CHUNK_SIZES["URMA"]
    final_chunk = FINAL_CHUNK_SIZES["URMA"]

    # Start each run from an empty processing directory
    shutil.rmtree(historic_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(historic_path, exist_ok=True)
        os.makedirs(publish_dir, exist_ok=True)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    # %% Determine the model run: the newest available hourly analysis
    latest_run = Herbie_latest(
        model="urma",
        n=URMA_LATEST_SEARCH_CYCLES,
        product="anl",
        verbose=True,
        priority=URMA_SOURCE_PRIORITY,
        save_dir=herbie_save_dir,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/URMA_Hist.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to URMA, ending")
        return
    logger.info("Processing URMA run %s", base_time)

    # %% Download, transform and archive every history hour not archived yet
    for i in range(his_period, -1, -1):
        analysis_time = pd.Timestamp(base_time - pd.Timedelta(hours=i))
        timestamp_str = analysis_time.strftime(HOURLY_STORE_TIMESTAMP_FORMAT)

        # A .done marker is written only after an hourly store is complete
        if save_type == "S3":
            s3_path = f"{historic_path}/URMA_Hist_{timestamp_str}.zarr.tar.gz"
            if s3.exists(s3_path.replace(".tar.gz", ".done")):
                logger.info("File already exists in S3, skipping: %s", s3_path)
                continue
        else:
            local_path = f"{historic_path}/URMA_Hist_{timestamp_str}.zarr"
            if os.path.exists(local_path.replace(".zarr", ".done")):
                logger.info("File already exists locally, skipping: %s", local_path)
                continue

        logger.info("Downloading URMA historic timestep: %s", timestamp_str)
        download_urma_analysis(analysis_time, herbie_save_dir)

        # Herbie reuses the subset file downloaded above instead of fetching it again
        fh_analysis = Herbie(
            analysis_time,
            model="urma",
            product="anl",
            verbose=False,
            priority=URMA_SOURCE_PRIORITY,
            save_dir=herbie_save_dir,
        )
        xarray_herbie_list = fh_analysis.xarray(URMA_MATCH_STRINGS)

        # cfgrib returns one dataset per level type; merge them into one
        xarray_analysis_merged = xr.merge(xarray_herbie_list, compat="override")
        xarray_analysis_merged = xarray_analysis_merged.assign_coords(
            xarray_herbie_list[0].metpy.parse_cf().coords
        )

        # Validate the analysis time against the requested hour
        grib_analysis_time = pd.Timestamp(xarray_analysis_merged.time.values)
        if grib_analysis_time != analysis_time:
            raise RuntimeError(
                f"URMA analysis time {grib_analysis_time} does not match the "
                f"requested hour {analysis_time}"
            )

        # Derive relative humidity and earth-relative winds
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

        # Replace the time coordinate with a gridded UNIX-time variable
        # The API reads time from the store like any other variable
        model_UNIX_time = xarray_analysis_merged.time.data.astype(
            "datetime64[s]"
        ).astype(int)
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
        missing_vars = [
            v for v in zarr_vars if v not in xarray_analysis_merged.data_vars
        ]
        if missing_vars:
            raise RuntimeError(
                f"URMA analysis {analysis_time} is missing variables {missing_vars}"
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

        # Stack into the (variable, time, y, x) layout
        xarray_analysis_stack = (
            xarray_analysis_merged[vars_in]
            .to_stacked_array(new_dim="var", sample_dims=["y", "x"])
            .expand_dims("time", axis=1)
            .chunk(
                chunks={"var": -1, "time": 1, "x": process_chunk, "y": process_chunk}
            )
            .transpose("var", "time", "y", "x")
        )
        dask_var_array = mask_invalid_data(xarray_analysis_stack.data)

        # Preserve the final-grid padding while avoiding tens of thousands of tiny
        # files in each hourly store. The merged history is also chunked this way.
        dask_var_array = pad_to_chunk_size(dask_var_array, final_chunk).rechunk(
            (len(vars_in), 1, process_chunk, process_chunk)
        )

        # Write the hour as a directory-backed Zarr group with one named array
        # per variable; the history merge below reads each variable by name.
        hourly_tmp_zarr_path = hist_process_path + "_URMA_Hist_TMP.zarr"
        zarr_store = zarr.storage.LocalStore(hourly_tmp_zarr_path)
        ds_out = xr.Dataset(
            {
                var: (("step", "y", "x"), dask_var_array[var_index])
                for var_index, var in enumerate(vars_in)
            }
        )
        ds_out.to_zarr(
            zarr_store,
            mode="w",
            zarr_format=3,
            encoding={
                var: {
                    "compressors": zarr.codecs.BloscCodec(cname="zstd", clevel=3),
                    "dtype": "float32",
                }
                for var in vars_in
            },
        )
        close_store(zarr_store)

        # Archive the hour, then mark it done so later runs skip it
        if save_type == "S3":
            archive_tmp_zarr_and_upload(
                tmp_zarr_path=hourly_tmp_zarr_path,
                s3_path=s3_path,
                archive_member_name="URMA_Hist.zarr",
                s3=s3,
            )
        else:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            shutil.move(hourly_tmp_zarr_path, local_path)
            done_file = local_path.replace(".zarr", ".done")
            with open(done_file, "w") as f:
                f.write("Done")

    # %% Collect the hourly stores for the history merge
    if save_type == "S3":
        local_temp_dir = hist_process_path + "_s3_temp_downloads"
        os.makedirs(local_temp_dir, exist_ok=True)
        ncHistWorking_paths = []
        for i in range(his_period, -1, -1):
            timestamp = (base_time - pd.Timedelta(hours=i)).strftime(
                HOURLY_STORE_TIMESTAMP_FORMAT
            )
            final_zarr_name = f"URMA_Hist_{timestamp}.zarr"
            extracted_path = download_extract_historic_archive(
                s3=s3,
                historic_path=historic_path,
                final_zarr_name=final_zarr_name,
                extracted_store_name="URMA_Hist.zarr",
                local_temp_dir=local_temp_dir,
                expected_vars=zarr_vars,
            )
            # Every hour was archived above, so a missing one means the archive
            # was lost. The API reads the hourly rows by position, so skipping it
            # would shift every later hour.
            if extracted_path is None:
                raise RuntimeError(
                    f"URMA hourly store {final_zarr_name} is missing from "
                    f"{historic_path}"
                )
            ncHistWorking_paths.append(extracted_path)
    else:
        ncHistWorking_paths = [
            historic_path
            + "/URMA_Hist_"
            + (base_time - pd.Timedelta(hours=i)).strftime(
                HOURLY_STORE_TIMESTAMP_FORMAT
            )
            + ".zarr"
            for i in range(his_period, -1, -1)
        ]

    logger.info(
        "Starting final URMA merge from %d hourly stores.", len(ncHistWorking_paths)
    )

    if not ncHistWorking_paths:
        raise RuntimeError("No hourly URMA stores are available for the final merge")

    # %% Merge the hourly stores into the final store
    first_hour = zarr.open_array(
        ncHistWorking_paths[0],
        path=zarr_vars[0],
        mode="r",
    )
    ny, nx = first_hour.shape[-2:]

    # Create the final array before building the per-variable graphs. Writing one
    # variable at a time bounds scheduler memory for the full history merge.
    final_zarr_path = historic_process_dir + "/URMA_Hist.zarr"
    final_zarr = zarr.open_array(
        final_zarr_path,
        mode="w",
        shape=(len(zarr_vars), len(ncHistWorking_paths), ny, nx),
        chunks=(1, len(ncHistWorking_paths), process_chunk, process_chunk),
        dtype="float32",
    )
    logger.info(
        "Writing final URMA merge with shape %s and chunks %s.",
        final_zarr.shape,
        final_zarr.chunks,
    )

    for var_index, dask_var in enumerate(zarr_vars):
        if dask_var == "time":
            # Read one scalar from each store instead of constructing spatial Dask
            # graphs merely to extract their timestamps.
            np_cat_times = np.asarray(
                [
                    zarr.open_array(local_path, path=dask_var, mode="r")[0, 0, 0]
                    for local_path in ncHistWorking_paths
                ],
                dtype="float32",
            )

            dask_array_out = broadcast_times_to_grid(
                np_cat_times,
                ny,
                nx,
                process_chunk,
            )
        else:
            # Stacking produces shape (n_files, 1, y, x); remove the singleton
            # dimension before writing this variable's final array region.
            dask_var_arrays_stack = da.stack(
                [
                    da.from_zarr(local_path, component=dask_var, inline_array=True)
                    for local_path in ncHistWorking_paths
                ],
                axis=0,
            )
            dask_array_out = (
                dask_var_arrays_stack.squeeze(axis=1)
                .rechunk((len(ncHistWorking_paths), process_chunk, process_chunk))
                .astype("float32")
            )

        da.store(
            dask_array_out[np.newaxis, ...],
            final_zarr,
            regions=(
                slice(var_index, var_index + 1),
                slice(None),
                slice(None),
                slice(None),
            ),
            compute=True,
        )
        logger.info(
            "Merged URMA variable %s (%d/%d).",
            dask_var,
            var_index + 1,
            len(zarr_vars),
        )

    close_store(final_zarr.store)
    logger.info("Final URMA merge completed: %s", final_zarr_path)

    # S3 serves a single zip of the directory store
    if save_type == "S3":
        shutil.make_archive(final_zarr_path, "zip", final_zarr_path)
        urma_store_name = "URMA_Hist.zarr.zip"
    else:
        urma_store_name = "URMA_Hist.zarr"

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [urma_store_name],
        base_time,
        "URMA_Hist.time.pickle",
        process_dir=historic_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(historic_process_dir)
    logger.info("URMA run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
