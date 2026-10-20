# ruff: noqa: E402
"""ECCC RAQDPS air quality ingest.

Downloads the newest 00/12 UTC run of Environment and Climate Change Canada's
Regional Air Quality Deterministic Prediction System (10 km rotated lat/lon
grid, hourly leads f000-f072) from the MSC Datamart with Herbie, one GRIB2 file
per variable and lead. Converts PM2.5/PM10 to ug/m3 and NO2/O3/SO2 to ppb and
publishes ``RAQDPS.zarr`` plus ``RAQDPS.lat_lon.pickle`` (the grid the API uses
to locate cells) and ``RAQDPS.time.pickle``. The store starts at f000 (the run
time) and holds the native hourly leads; nothing is interpolated.
"""

import logging
import os
import pickle
import shutil
import time
import urllib.error
import warnings
from datetime import timedelta
from pathlib import Path

# Configure ecCodes before importing Herbie/cfgrib-backed readers.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECCODES_DIR = WORKSPACE_ROOT / ".build" / "ingest-test" / "toolchain"
os.environ.setdefault("ECCODES_DIR", str(DEFAULT_ECCODES_DIR))
os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    f"{DEFAULT_ECCODES_DIR / 'lib'}:{DEFAULT_ECCODES_DIR / 'lib64'}",
)

import dask
import dask.array as da
import herbie.models as herbie_model_templates
import numpy as np
import requests
import s3fs
import xarray as xr
import zarr.storage
from dask.diagnostics import ProgressBar
from herbie import Herbie

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    close_store,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)
from API.raqdps_herbie_template import raqdps
from API.raqdps_utils import (
    RAQDPS_FORECAST_HOURS,
    RAQDPS_LEVEL,
    RAQDPS_OUTPUT_VARS,
    RAQDPS_VARIABLES,
    as_float32_array,
    candidate_raqdps_runs,
    convert_to_output_units,
    herbie_naive_utc,
    normalize_utc,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", "In a future version")

logger = logging.getLogger(__name__)

# Register the repo-local Herbie template without modifying the installed package.
herbie_model_templates.raqdps = raqdps

# %% Model constants
# Variables, output order (time first) and leads f000-f072 live in
# API/raqdps_utils.py, shared with the Herbie template and diagnostics scripts.
# A run is only used once its last lead (f072 O3) is on the Datamart.
RAQDPS_PROBE_VARIABLE = "O3"
RAQDPS_RUN_CANDIDATES = 10
RAQDPS_TIME_PICKLE = "RAQDPS.time.pickle"
RAQDPS_LAT_LON_PICKLE = "RAQDPS.lat_lon.pickle"

# Download failures that are retried: network/HTTP errors, file I/O, and the
# ValueError Herbie raises with errors="raise" when the file is not found
RAQDPS_DOWNLOAD_ERRORS = (OSError, ValueError, requests.RequestException)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/RAQDPS"
)
forecast_process_path = os.path.join(forecast_process_dir, "RAQDPS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RAQDPS")
save_type = os.getenv("save_type", default="Download")

aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
process_chunk = CHUNK_SIZES["RAQDPS"]
final_chunk = FINAL_CHUNK_SIZES["RAQDPS"]
process_chunk_bytes = process_chunk * process_chunk * np.dtype(np.float32).itemsize

RAQDPS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RAQDPS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 10)


def published_path(filename):
    """Return the published output path for one RAQDPS sidecar file."""
    return os.path.join(forecast_path, ingest_version, filename)


def build_herbie(run_time, variable, forecast_hour, *, verbose=False):
    """Create a Herbie object for one RAQDPS field."""
    return Herbie(
        herbie_naive_utc(run_time),
        model="raqdps",
        product="10km/grib2",
        fxx=forecast_hour,
        variable=variable,
        level=RAQDPS_LEVEL,
        priority=["msc"],
        save_dir=tmp_dir,
        verbose=verbose,
    )


def get_latest_raqdps_run():
    """Find the newest RAQDPS 00/12 UTC run whose last lead is available."""
    for candidate in candidate_raqdps_runs(count=RAQDPS_RUN_CANDIDATES):
        try:
            h = build_herbie(
                candidate, RAQDPS_PROBE_VARIABLE, RAQDPS_FORECAST_HOURS[-1]
            )
        except (FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError) as e:
            logger.warning("Unable to probe RAQDPS run %s: %s", candidate, e)
            continue

        if h.grib is not None:
            logger.info("Found latest valid RAQDPS runtime at: %s", candidate)
            return candidate

    raise RuntimeError(
        f"No recent RAQDPS run with f{RAQDPS_FORECAST_HOURS[-1]:03d} "
        f"{RAQDPS_PROBE_VARIABLE} data was found"
    )


def download_raqdps_file(run_time, variable, forecast_hour):
    """Download one RAQDPS GRIB2 file with Herbie, with bounded retries.

    RAQDPS uses a repo-local Herbie template with one full file per variable and
    lead, so the FastHerbie subset helper does not apply. Raises if the file is
    still missing after the last attempt: a gap must not be published.
    """
    for attempt in range(1, RAQDPS_DOWNLOAD_RETRIES + 1):
        try:
            h = build_herbie(run_time, variable, forecast_hour)
            if h.grib is None:
                raise FileNotFoundError(
                    f"RAQDPS file not on the Datamart: run={run_time} "
                    f"variable={variable} f{forecast_hour:03d}"
                )
            # Overwrite on retries so a partial file is not reused
            return h.download(verbose=False, errors="raise", overwrite=attempt > 1)
        except RAQDPS_DOWNLOAD_ERRORS as e:
            if attempt == RAQDPS_DOWNLOAD_RETRIES:
                raise RuntimeError(
                    f"RAQDPS download failed after {attempt} attempts: run={run_time} "
                    f"variable={variable} f{forecast_hour:03d}"
                ) from e
            sleep_s = RAQDPS_RETRY_SLEEP_SECONDS * attempt
            logger.warning(
                "RAQDPS download attempt %d/%d failed for run=%s variable=%s "
                "f%03d (%s). Retrying in %ss",
                attempt,
                RAQDPS_DOWNLOAD_RETRIES,
                run_time,
                variable,
                forecast_hour,
                e,
                sleep_s,
            )
            time.sleep(sleep_s)


def read_raqdps_grib(local_grib_path, variable):
    """Read one RAQDPS GRIB2 field in configured output units as float32."""
    ds = xr.open_dataset(local_grib_path, engine="cfgrib", decode_times=False)
    try:
        grib_var_name = next(iter(ds.data_vars))
        if grib_var_name is None:
            raise ValueError(f"No data variables found in GRIB file: {local_grib_path}")
        data_array = ds[grib_var_name].astype(np.float32)
        return as_float32_array(convert_to_output_units(data_array, variable).values)
    finally:
        ds.close()


def read_raqdps_lat_lon(local_grib_path, expected_shape):
    """Read RAQDPS grid latitude/longitude arrays from one GRIB2 file."""
    ds = xr.open_dataset(local_grib_path, engine="cfgrib", decode_times=False)
    try:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            raise KeyError("RAQDPS GRIB missing latitude/longitude coordinates")

        latitude = as_float32_array(ds[lat_name].values)
        longitude = as_float32_array(ds[lon_name].values)

        if latitude.ndim == 1 and longitude.ndim == 1:
            longitude, latitude = np.meshgrid(longitude, latitude)

        if latitude.shape != expected_shape or longitude.shape != expected_shape:
            raise ValueError(
                "RAQDPS coordinate shape mismatch: "
                f"lat={latitude.shape} lon={longitude.shape} expected={expected_shape}"
            )

        return {
            "latitude": latitude,
            "longitude": longitude,
            "shape": expected_shape,
        }
    finally:
        ds.close()


def load_raqdps_time_slice(run_time, forecast_hour, *, include_lat_lon=False):
    """Download and read every configured variable for one RAQDPS lead.

    Returns the fields by variable and, with ``include_lat_lon``, the grid
    latitude/longitude read from the first file.
    """
    fields = {}
    lat_lon_lookup = None

    for variable in RAQDPS_VARIABLES:
        local_grib_path = download_raqdps_file(run_time, variable, forecast_hour)
        values = read_raqdps_grib(local_grib_path, variable)

        if fields and values.shape != next(iter(fields.values())).shape:
            raise RuntimeError(
                f"RAQDPS run {run_time} f{forecast_hour:03d} {variable} grid "
                f"{values.shape} differs from {next(iter(fields.values())).shape}"
            )
        fields[variable] = values

        if include_lat_lon and lat_lon_lookup is None:
            lat_lon_lookup = read_raqdps_lat_lon(local_grib_path, values.shape)

    return fields, lat_lon_lookup


def variable_store_name(variable):
    """Return the child Zarr store name for an intermediate RAQDPS variable."""
    return f"{variable.replace('.', '_')}.zarr"


def variable_zarr_path(root_path, variable):
    """Return the path for one variable in a variable-separated RAQDPS store."""
    return os.path.join(root_path, variable_store_name(variable))


def build_variable_arrays(time_slices, valid_times, reference_shape):
    """Convert time-slice dictionaries to separate dask arrays per variable."""
    unix_times = np.array([int(valid_time.timestamp()) for valid_time in valid_times])
    variable_arrays = {
        "time": da.from_array(unix_times.astype(np.float32), chunks=(len(unix_times),))
    }

    for variable in RAQDPS_VARIABLES:
        arrays = [
            da.from_array(fields[variable], chunks=(process_chunk, process_chunk))
            for fields in time_slices
        ]

        variable_arrays[variable] = da.stack(arrays, axis=0).astype("float32")

    return variable_arrays


def write_variable_zarrs(variable_arrays, root_path, time_chunks, zarr_store_workers):
    """Write separate intermediate Zarr arrays for each RAQDPS variable."""
    shutil.rmtree(root_path, ignore_errors=True)
    os.makedirs(root_path, exist_ok=True)

    with (
        ProgressBar(),
        dask.config.set(
            scheduler="threads",
            num_workers=zarr_store_workers,
            # Keep Dask's internal Zarr write chunks at least as large as one
            # on-disk Zarr chunk to avoid unsafe auto-rechunk warnings.
            array__chunk_size=time_chunks * process_chunk_bytes,
        ),
    ):
        for variable in RAQDPS_OUTPUT_VARS:
            variable_array = variable_arrays[variable]
            chunks = (
                (time_chunks,)
                if variable == "time"
                else (time_chunks, process_chunk, process_chunk)
            )
            store = zarr.storage.LocalStore(variable_zarr_path(root_path, variable))
            zarr_array = zarr.create_array(
                store=store,
                shape=variable_array.shape,
                chunks=chunks,
                dtype=np.float32,
                overwrite=True,
            )
            variable_array.rechunk(chunks).to_zarr(
                zarr_array, overwrite=True, compute=True
            )
            close_store(store)


def read_intermediate_variable(root_path, variable):
    """Read one variable from a variable-separated RAQDPS intermediate store."""
    return da.from_zarr(variable_zarr_path(root_path, variable), inline_array=True)


def expand_time_array(time_array, reference_shape):
    """Expand a 1D intermediate time array to the final 3D grid shape."""
    if time_array.ndim == 3:
        return time_array

    lat_count, lon_count = reference_shape
    return da.broadcast_to(
        time_array.reshape(time_array.shape[0], 1, 1),
        (time_array.shape[0], lat_count, lon_count),
    )


def prepare_directories():
    """Reset processing directories and ensure output roots exist."""
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    if save_type != "S3":
        os.makedirs(os.path.join(forecast_path, ingest_version), exist_ok=True)


def already_processed(base_time, s3):
    """Return True if the published RAQDPS store is already current."""
    previous_base_time = read_published_run_time(
        published_path(RAQDPS_TIME_PICKLE), save_type, s3
    )
    if previous_base_time is None:
        return False
    if normalize_utc(previous_base_time) < normalize_utc(base_time):
        return False

    # The API needs the lat/lon pickle to locate grid cells, so a current run
    # without it must be reprocessed.
    lat_lon_pickle_path = published_path(RAQDPS_LAT_LON_PICKLE)
    if save_type == "S3":
        lat_lon_exists = s3.exists(lat_lon_pickle_path)
    else:
        lat_lon_exists = os.path.exists(lat_lon_pickle_path)
    if not lat_lon_exists:
        logger.info("RAQDPS time is current, but lat/lon pickle is missing.")
        return False
    return True


def process_forecast(base_time, zarr_store_workers):
    """Download and write the f000-f072 RAQDPS forecast.

    Every lead and variable must be present: the store is not interpolated, so
    a skipped lead would be a silent gap, and the API recovers the run time from
    the first stored valid time (f000, see FIRST_FORECAST_LEAD_HOURS), so a
    missing f000 would make it derive a wrong run time.
    """
    logger.info("Processing RAQDPS forecast from base time: %s", base_time)
    time_slices = []
    valid_times = []
    reference_shape = None
    lat_lon_lookup = None

    for forecast_hour in RAQDPS_FORECAST_HOURS:
        # The grid coordinates are read once, from the first lead
        fields, slice_lat_lon = load_raqdps_time_slice(
            base_time, forecast_hour, include_lat_lon=lat_lon_lookup is None
        )
        slice_shape = next(iter(fields.values())).shape

        if reference_shape is None:
            reference_shape = slice_shape
            lat_lon_lookup = slice_lat_lon
        elif slice_shape != reference_shape:
            raise RuntimeError(
                f"RAQDPS run {base_time} f{forecast_hour:03d} grid {slice_shape} "
                f"differs from {reference_shape}"
            )

        time_slices.append(fields)
        valid_times.append(base_time + timedelta(hours=forecast_hour))

    forecast_arrays = build_variable_arrays(time_slices, valid_times, reference_shape)
    forecast_zarr_path = forecast_process_path + "_forecast.zarr"
    write_variable_zarrs(
        forecast_arrays, forecast_zarr_path, len(valid_times), zarr_store_workers
    )
    return forecast_zarr_path, reference_shape, lat_lon_lookup


def merge_and_publish(
    base_time,
    forecast_zarr_path,
    reference_shape,
    lat_lon_lookup,
    s3,
    zarr_store_workers,
):
    """Stack the forecast variables and publish the final RAQDPS store."""
    logger.info("Stacking RAQDPS forecast variables.")

    dask_variable_list = []
    for variable in RAQDPS_OUTPUT_VARS:
        variable_array = read_intermediate_variable(forecast_zarr_path, variable)
        if variable == "time":
            variable_array = expand_time_array(variable_array, reference_shape)

        merged_variable = variable_array.reshape(
            1,
            variable_array.shape[0],
            reference_shape[0],
            reference_shape[1],
        )
        dask_variable_list.append(merged_variable)

    merged_arrays = da.concatenate(dask_variable_list, axis=0).astype("float32")
    merged_arrays_masked = mask_invalid_data(merged_arrays)

    stack_path = forecast_process_path + "_stack.zarr"
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        merged_arrays_masked.rechunk(
            (
                len(RAQDPS_OUTPUT_VARS),
                merged_arrays_masked.shape[1],
                process_chunk,
                process_chunk,
            )
        ).to_zarr(stack_path, overwrite=True, compute=True)

    stacked_array_disk = da.from_zarr(stack_path)
    stacked_array_padded = pad_to_chunk_size(stacked_array_disk, final_chunk)

    raqdps_store_name = "RAQDPS.zarr.zip" if save_type == "S3" else "RAQDPS.zarr"
    write_final_zarr(
        stacked_array_padded,
        os.path.join(forecast_process_dir, raqdps_store_name),
        final_chunk,
        zarr_store_workers,
    )

    lat_lon_pickle_file_path = os.path.join(forecast_process_dir, RAQDPS_LAT_LON_PICKLE)
    with open(lat_lon_pickle_file_path, "wb") as f:
        pickle.dump(lat_lon_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Publish the store and lat/lon grid, then the run time the API uses to detect them
    publish_ingest_run(
        [raqdps_store_name, RAQDPS_LAT_LON_PICKLE],
        base_time,
        RAQDPS_TIME_PICKLE,
        process_dir=forecast_process_dir,
        publish_dir=os.path.join(forecast_path, ingest_version),
        save_type=save_type,
        s3=s3,
    )


def main():
    """Run the RAQDPS ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Bound Herbie request times
    configure_herbie_request_timeouts()

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    prepare_directories()
    T0 = time.time()

    # %% Determine the model run: the newest run whose last lead is posted
    base_time = get_latest_raqdps_run()

    # %% Skip the run if this cycle (and its lat/lon grid) is already published
    if already_processed(base_time, s3):
        logger.info("RAQDPS store is already up to date. Exiting.")
        return
    logger.info("Processing RAQDPS run %s", base_time)

    try:
        # %% Download, convert and write each lead
        forecast_zarr_path, reference_shape, lat_lon_lookup = process_forecast(
            base_time, zarr_store_workers
        )
        # %% Stack, write the final store and publish
        merge_and_publish(
            base_time,
            forecast_zarr_path,
            reference_shape,
            lat_lon_lookup,
            s3,
            zarr_store_workers,
        )
    finally:
        # %% Clean up
        shutil.rmtree(forecast_process_dir, ignore_errors=True)

    logger.info("RAQDPS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
