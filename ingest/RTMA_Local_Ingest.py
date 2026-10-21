"""Hourly RTMA analysis ingest with a fixed 48-hour history.

Downloads hourly RTMA analyses (2.5 km CONUS Lambert conformal ``_wexp`` grid,
one analysis per hour, published about 50 minutes after its valid time) for
2 m temperature, dew point and specific humidity, 10 m wind and gust, surface
pressure, visibility and total cloud cover, plus the hourly RTMA precipitation
analysis. Each hour is transformed like RTMA-RU (derived 2 m relative humidity,
earth-relative winds), given the 1-hour precipitation accumulation ending at
that hour, and archived as its own ``RTMA_Hist_<time>.zarr``.

The published ``RTMA.zarr`` always holds ``RTMA_TIME_STEPS`` hourly rows, oldest
first: row ``i`` is the newest available hourly analysis minus
``RTMA_TIME_STEPS - 1 - i`` hours. It is published with ``RTMA.time.pickle``.
The 15-minute RTMA-RU analysis used for current conditions is a separate ingest.

Alexander Rey, October 2026
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

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_grib_utils import download_and_validate_subset
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    VALID_DATA_MIN,
    archive_tmp_zarr_and_upload,
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
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# %% Model constants
RTMA_SOURCE_PRIORITY = ["aws", "nomads"]
# The hourly analysis appears about 50 minutes after its valid time; search the
# last few hours for the newest one that is already available.
RTMA_CYCLE_FREQ = "1h"
RTMA_LATEST_SEARCH_CYCLES = 4
# Specific humidity (SPFH) is only downloaded to derive relative humidity.
RTMA_ANALYSIS_MATCH_STRINGS = (
    ":((DPT|TMP|SPFH):2 m above ground:)"
    "|:(GUST:10 m above ground:)"
    "|:(UGRD:10 m above ground:)"
    "|:(VGRD:10 m above ground:)"
    "|:((VIS|PRES):surface:)"
    "|:TCDC:entire atmosphere"
)
# The hourly RTMA precipitation product (rtma2p5.<YYYYMMDDHH>.pcp.184.grb2)
# holds one field: the APCP accumulated over the hour ending at <HH>. It is
# downloaded whole (about 0.5 MB): Herbie cannot byte-range subset a file whose
# index has a single message.
# The precipitation product is on NCEP grid 184 (2145 x 1377), which sits inside
# the analysis _wexp grid (2345 x 1597) with the same projection and spacing: its
# first point is exactly 200 cells east of the analysis grid's first point and on
# the same row. It is placed there without resampling; the rest of the _wexp
# grid has no precipitation analysis and stays NaN.
RTMA_PCP_GRID_SHAPE = (1377, 2145)
RTMA_PCP_X_OFFSET = 200
RTMA_PCP_Y_OFFSET = 0

# Hourly rows in the published store: the newest analysis and the 47 h before it
RTMA_TIME_STEPS = 48
HOURLY_STORE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# Variables in the published store (cfgrib short names). tp is the hourly RTMA
# precipitation (kg/m^2 = mm over the hour ending at the row's time).
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
    "tp",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/RTMA")
tmp_dir = forecast_process_dir + "/Downloads"
hist_process_path = forecast_process_dir + "/RTMA_Historic"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RTMA")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/Prod/RTMA")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

RTMA_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RTMA_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)


def download_rtma_product(
    product: str,
    search: str | None,
    valid_time: pd.Timestamp,
    herbie_save_dir: str,
) -> xr.Dataset | list[xr.Dataset]:
    """Download one RTMA GRIB subset (or the whole file) and open it with cfgrib.

    The wgrib2 stats gate is skipped: RTMA never needed wgrib2, and the
    transform below already masks out-of-range values to NaN, which is the
    behaviour the API relies on for partially missing analysis fields.
    """
    download_and_validate_subset(
        model="rtma",
        product=product,
        search=search,
        dataset_name=f"RTMA {product} {valid_time}",
        base_time=valid_time,
        forecast_hours=[0],
        priority=RTMA_SOURCE_PRIORITY,
        herbie_save_dir=herbie_save_dir,
        herbie_download_retries=RTMA_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=RTMA_RETRY_SLEEP_SECONDS,
        skip_wgrib2_validation=True,
        wgrib2_exe="",
    )
    # Herbie reuses the subset file downloaded above instead of fetching it again
    return Herbie(
        valid_time,
        model="rtma",
        product=product,
        verbose=False,
        priority=RTMA_SOURCE_PRIORITY,
        save_dir=herbie_save_dir,
    ).xarray(search)


def read_hourly_precip(
    valid_time: pd.Timestamp, herbie_save_dir: str, analysis_shape: tuple[int, int]
) -> np.ndarray:
    """Return the 1-hour RTMA precipitation ending at ``valid_time`` on the _wexp grid."""
    xarray_pcp = download_rtma_product("pcp", None, valid_time, herbie_save_dir)
    if isinstance(xarray_pcp, list):
        xarray_pcp = xr.merge(xarray_pcp, compat="override")

    pcp_valid_time = pd.Timestamp(xarray_pcp.valid_time.values)
    if pcp_valid_time != valid_time:
        raise RuntimeError(
            f"RTMA precipitation valid time {pcp_valid_time} does not match the "
            f"requested hour {valid_time}"
        )
    tp = xarray_pcp["tp"].values
    if tp.shape != RTMA_PCP_GRID_SHAPE:
        raise RuntimeError(
            f"RTMA precipitation for {valid_time} has grid {tp.shape}, expected "
            f"{RTMA_PCP_GRID_SHAPE}"
        )

    tp_wexp = np.full(analysis_shape, np.nan, dtype="float32")
    tp_wexp[
        RTMA_PCP_Y_OFFSET : RTMA_PCP_Y_OFFSET + RTMA_PCP_GRID_SHAPE[0],
        RTMA_PCP_X_OFFSET : RTMA_PCP_X_OFFSET + RTMA_PCP_GRID_SHAPE[1],
    ] = tp
    return tp_wexp


def write_hourly_store(
    analysis_time: pd.Timestamp,
    herbie_save_dir: str,
    store_path: str,
    process_chunk: int,
    final_chunk: int,
) -> None:
    """Download and transform one hourly analysis and write it as a one-step zarr.

    The group holds one named ``(step, y, x)`` array per variable, padded to the
    final chunk size, which the merge below stacks along time.
    """
    xarray_herbie_list = download_rtma_product(
        "anl", RTMA_ANALYSIS_MATCH_STRINGS, analysis_time, herbie_save_dir
    )

    # cfgrib returns one dataset per level type; merge them into one
    xarray_analysis_merged = xr.merge(xarray_herbie_list, compat="override")
    xarray_analysis_merged = xarray_analysis_merged.assign_coords(
        xarray_herbie_list[0].metpy.parse_cf().coords
    )

    # Validate the analysis time against the requested hour
    grib_analysis_time = pd.Timestamp(xarray_analysis_merged.time.values)
    if grib_analysis_time != analysis_time:
        raise RuntimeError(
            f"RTMA analysis time {grib_analysis_time} does not match the "
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
    model_UNIX_time = xarray_analysis_merged.time.data.astype("datetime64[s]").astype(
        int
    )
    xarray_analysis_merged = xarray_analysis_merged.reset_coords("time", drop=True)
    analysis_shape = (
        xarray_analysis_merged.sizes["y"],
        xarray_analysis_merged.sizes["x"],
    )
    xarray_analysis_merged["time"] = (
        ("y", "x"),
        np.full(analysis_shape, model_UNIX_time),
    )

    # Specific humidity was only needed for relative humidity
    xarray_analysis_merged = xarray_analysis_merged.drop_vars("sh2")

    # Precipitation accumulated over the hour ending at the analysis time
    xarray_analysis_merged["tp"] = (
        ("y", "x"),
        read_hourly_precip(analysis_time, herbie_save_dir, analysis_shape),
    )

    # A missing field would shift every later variable
    missing_vars = [v for v in zarr_vars if v not in xarray_analysis_merged.data_vars]
    if missing_vars:
        raise RuntimeError(
            f"RTMA analysis {analysis_time} is missing variables {missing_vars}"
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
        .chunk(chunks={"var": -1, "time": 1, "x": process_chunk, "y": process_chunk})
        .transpose("var", "time", "y", "x")
    )
    dask_var_array = mask_invalid_data(xarray_analysis_stack.data)
    dask_var_array = pad_to_chunk_size(dask_var_array, final_chunk).rechunk(
        (len(vars_in), 1, process_chunk, process_chunk)
    )

    # One named array per variable; the merge reads each variable by name
    zarr_store = zarr.storage.LocalStore(store_path)
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


def main() -> None:
    """Run the hourly RTMA ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["RTMA"]
    final_chunk = FINAL_CHUNK_SIZES["RTMA_HOURLY"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(historic_path, exist_ok=True)
        os.makedirs(publish_dir, exist_ok=True)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    T0 = time.time()

    # %% Determine the model run: the newest available hourly analysis
    latest_run = Herbie_latest(
        model="rtma",
        n=RTMA_LATEST_SEARCH_CYCLES,
        freq=RTMA_CYCLE_FREQ,
        product="anl",
        verbose=True,
        priority=RTMA_SOURCE_PRIORITY,
        save_dir=herbie_save_dir,
    )
    base_time = pd.Timestamp(latest_run.date)

    # %% Skip the run if this hour has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/RTMA.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to RTMA, ending")
        return
    logger.info("Processing RTMA run %s", base_time)

    # %% Download, transform and archive every hour not archived yet
    # Row i of the store is always base_time - (RTMA_TIME_STEPS - 1 - i) hours
    row_times = [
        base_time - pd.Timedelta(hours=RTMA_TIME_STEPS - 1 - step)
        for step in range(RTMA_TIME_STEPS)
    ]
    for analysis_time in row_times:
        timestamp_str = analysis_time.strftime(HOURLY_STORE_TIMESTAMP_FORMAT)

        # A .done marker is written only after an hourly store is complete
        if save_type == "S3":
            s3_path = f"{historic_path}/RTMA_Hist_{timestamp_str}.zarr.tar.gz"
            if s3.exists(s3_path.replace(".tar.gz", ".done")):
                continue
        else:
            local_path = f"{historic_path}/RTMA_Hist_{timestamp_str}.zarr"
            if os.path.exists(local_path.replace(".zarr", ".done")):
                continue

        logger.info("Processing hourly RTMA analysis %s", timestamp_str)
        hourly_tmp_zarr_path = hist_process_path + "_RTMA_Hist_TMP.zarr"
        write_hourly_store(
            analysis_time,
            herbie_save_dir,
            hourly_tmp_zarr_path,
            process_chunk,
            final_chunk,
        )

        # Archive the hour, then mark it done so later runs skip it
        if save_type == "S3":
            archive_tmp_zarr_and_upload(
                tmp_zarr_path=hourly_tmp_zarr_path,
                s3_path=s3_path,
                archive_member_name="RTMA_Hist.zarr",
                s3=s3,
            )
        else:
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            shutil.move(hourly_tmp_zarr_path, local_path)
            with open(local_path.replace(".zarr", ".done"), "w") as f:
                f.write("Done")

    # %% Collect the hourly stores for the merge
    if save_type == "S3":
        local_temp_dir = hist_process_path + "_s3_temp_downloads"
        os.makedirs(local_temp_dir, exist_ok=True)
        row_store_paths = []
        for analysis_time in row_times:
            final_zarr_name = (
                "RTMA_Hist_"
                + analysis_time.strftime(HOURLY_STORE_TIMESTAMP_FORMAT)
                + ".zarr"
            )
            extracted_path = download_extract_historic_archive(
                s3=s3,
                historic_path=historic_path,
                final_zarr_name=final_zarr_name,
                extracted_store_name="RTMA_Hist.zarr",
                local_temp_dir=local_temp_dir,
                expected_vars=zarr_vars,
            )
            # Every hour was archived above, so a missing one means the archive
            # was lost; fail rather than publish a store with a gap.
            if extracted_path is None:
                raise RuntimeError(
                    f"RTMA hourly store {final_zarr_name} is missing from "
                    f"{historic_path}"
                )
            row_store_paths.append(extracted_path)
    else:
        row_store_paths = [
            historic_path
            + "/RTMA_Hist_"
            + analysis_time.strftime(HOURLY_STORE_TIMESTAMP_FORMAT)
            + ".zarr"
            for analysis_time in row_times
        ]

    # %% Validate each stored analysis time against its row
    expected_times_unix = np.asarray(
        [row_time.timestamp() for row_time in row_times], dtype="float32"
    )
    for store_path, expected_time in zip(
        row_store_paths, expected_times_unix, strict=True
    ):
        stored_time = zarr.open_array(store_path, path="time", mode="r")[0, 0, 0]
        if stored_time != expected_time:
            raise RuntimeError(
                f"{store_path} holds time {stored_time}, expected {expected_time}"
            )

    # %% Stack the hours into the (variable, time, y, x) layout
    # Each store holds (1, y, x) per variable; concatenating gives (time, y, x).
    stacked = da.stack(
        [
            da.concatenate(
                [
                    da.from_zarr(store_path, component=var, inline_array=True)
                    for store_path in row_store_paths
                ],
                axis=0,
            )
            for var in zarr_vars
        ],
        axis=0,
    )
    if stacked.shape[1] != RTMA_TIME_STEPS:
        raise RuntimeError(
            f"RTMA run {base_time} has {stacked.shape[1]} rows, "
            f"expected {RTMA_TIME_STEPS}"
        )
    logger.info(
        "Merging %d hourly RTMA analyses into %s", RTMA_TIME_STEPS, stacked.shape
    )

    # %% Write the final store
    rtma_store_name = "RTMA.zarr.zip" if save_type == "S3" else "RTMA.zarr"
    write_final_zarr(
        stacked,
        forecast_process_dir + "/" + rtma_store_name,
        final_chunk,
        zarr_store_workers,
    )

    # %% Publish the store, then the run time used to detect it
    publish_ingest_run(
        [rtma_store_name],
        base_time,
        "RTMA.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("RTMA run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
