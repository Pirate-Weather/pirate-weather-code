# %% Regional Air Quality Deterministic Prediction System (RAQDPS) Ingest Script
# ruff: noqa: E402
# Downloads ECCC RAQDPS forecasts with Herbie, normalizes pollutant units, and
# publishes a 48-hour history plus 72-hour forecast Zarr store.

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
import zipfile
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

import dask  # noqa: E402
import dask.array as da  # noqa: E402
import herbie.models as herbie_model_templates  # noqa: E402
import numpy as np  # noqa: E402
import s3fs  # noqa: E402
import xarray as xr  # noqa: E402
import zarr.storage  # noqa: E402
from dask.diagnostics import ProgressBar  # noqa: E402
from herbie import Herbie  # noqa: E402

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR  # noqa: E402
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    close_store,
    configure_zarr_limits,
    mask_invalid_data,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
)  # noqa: E402
from API.raqdps_herbie_template import raqdps  # noqa: E402
from API.raqdps_utils import (
    RAQDPS_FORECAST_HOURS,
    RAQDPS_LEVEL,
    RAQDPS_OUTPUT_VARS,
    RAQDPS_VARIABLES,
    as_float32_array,
    candidate_raqdps_runs,
    convert_to_ug_m3,
    herbie_naive_utc,
    history_run_for_valid_time,
    history_valid_times,
    normalize_utc,
)  # noqa: E402

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", "In a future version")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Register the repo-local Herbie template without modifying the installed package.
herbie_model_templates.raqdps = raqdps

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/home/reya/Weather/Process/RAQDPS"
)
forecast_process_path = os.path.join(forecast_process_dir, "RAQDPS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")
local_temp_historic_dir = os.path.join(forecast_process_dir, "Historic_Downloads")

forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod")
historic_path = os.getenv("historic_path", default="/home/reya/Weather/History/RAQDPS")
save_type = os.getenv("save_type", default="Download")

aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)

process_chunk = CHUNK_SIZES["RAQDPS"]
final_chunk = FINAL_CHUNK_SIZES["RAQDPS"]
history_period = HISTORY_PERIODS["RAQDPS"]
process_chunk_bytes = process_chunk * process_chunk * np.dtype(np.float32).itemsize


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
    """Find the newest RAQDPS 00/12 UTC run with a f072 ozone file available."""
    for candidate in candidate_raqdps_runs(count=10):
        try:
            h = build_herbie(candidate, "O3", 72)
        except Exception as e:
            logger.warning("Unable to probe RAQDPS run %s: %s", candidate, e)
            continue

        if h.grib is not None:
            logger.info("Found latest valid RAQDPS runtime at: %s", candidate)
            return candidate

    raise RuntimeError("No recent RAQDPS run with f072 O3 data was found")


def download_raqdps_file(run_time, variable, forecast_hour):
    """Download one RAQDPS GRIB2 file with Herbie."""
    try:
        h = build_herbie(run_time, variable, forecast_hour)
        if h.grib is None:
            logger.warning(
                "Missing RAQDPS file for run=%s variable=%s f%03d",
                run_time,
                variable,
                forecast_hour,
            )
            return None
        return h.download(verbose=False, errors="raise")
    except Exception as e:
        logger.warning(
            "Failed RAQDPS download for run=%s variable=%s f%03d: %s",
            run_time,
            variable,
            forecast_hour,
            e,
        )
        return None


def read_raqdps_grib(local_grib_path, variable):
    """Read and normalize one RAQDPS GRIB2 field to a float32 numpy array."""
    ds = xr.open_dataset(local_grib_path, engine="cfgrib", decode_times=False)
    try:
        grib_var_name = list(ds.data_vars)[0]
        data_array = ds[grib_var_name].astype(np.float32)
        return as_float32_array(convert_to_ug_m3(data_array, variable).values)
    finally:
        ds.close()


def load_raqdps_time_slice(run_time, forecast_hour):
    """Download/read all configured variables for one valid RAQDPS hour."""
    fields = {}
    reference_shape = None

    for variable in RAQDPS_VARIABLES:
        local_grib_path = download_raqdps_file(run_time, variable, forecast_hour)
        if local_grib_path is None:
            continue

        try:
            values = read_raqdps_grib(local_grib_path, variable)
        except Exception as e:
            logger.error(
                "Error reading RAQDPS variable=%s run=%s f%03d: %s",
                variable,
                run_time,
                forecast_hour,
                e,
            )
            continue

        fields[variable] = values
        if reference_shape is None:
            reference_shape = values.shape

    if reference_shape is None:
        return None, None

    return fields, reference_shape


def variable_store_name(variable):
    """Return the child Zarr store name for an intermediate RAQDPS variable."""
    return f"{variable.replace('.', '_')}.zarr"


def variable_zarr_path(root_path, variable):
    """Return the path for one variable in a variable-separated RAQDPS store."""
    return os.path.join(root_path, variable_store_name(variable))


def is_variable_zarr_store(root_path):
    """Return True when a path uses the variable-separated intermediate layout."""
    return os.path.isdir(root_path) and all(
        os.path.exists(variable_zarr_path(root_path, variable))
        for variable in RAQDPS_OUTPUT_VARS
    )


def build_variable_arrays(time_slices, valid_times, reference_shape):
    """Convert time-slice dictionaries to separate dask arrays per variable."""
    unix_times = np.array([int(valid_time.timestamp()) for valid_time in valid_times])
    variable_arrays = {
        "time": da.from_array(unix_times.astype(np.float32), chunks=(len(unix_times),))
    }

    for variable in RAQDPS_VARIABLES:
        arrays = []
        for fields in time_slices:
            if variable in fields:
                values = fields[variable]
            else:
                values = np.full(reference_shape, np.nan, dtype=np.float32)
            arrays.append(da.from_array(values, chunks=(process_chunk, process_chunk)))

        variable_arrays[variable] = da.stack(arrays, axis=0).astype("float32")

    return variable_arrays


def write_variable_zarrs(variable_arrays, root_path, time_chunks):
    """Write separate intermediate Zarr arrays for each RAQDPS variable."""
    shutil.rmtree(root_path, ignore_errors=True)
    os.makedirs(root_path, exist_ok=True)

    with ProgressBar():
        with dask.config.set(
            scheduler="threads",
            num_workers=zarr_store_workers,
            # Keep Dask's internal Zarr write chunks aligned with the on-disk
            # CHUNK_SIZES chunks to avoid unsafe auto-rechunk warnings.
            array__chunk_size=process_chunk_bytes,
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
    """Read one variable from a new or legacy RAQDPS intermediate Zarr store."""
    variable_path = variable_zarr_path(root_path, variable)
    if os.path.exists(variable_path):
        return da.from_zarr(variable_path, inline_array=True)

    legacy_stack = da.from_zarr(root_path, inline_array=True)
    variable_index = RAQDPS_OUTPUT_VARS.index(variable)
    return legacy_stack[variable_index]


def expand_time_array(time_array, reference_shape):
    """Expand a 1D intermediate time array to the final 3D grid shape."""
    if time_array.ndim == 3:
        return time_array

    lat_count, lon_count = reference_shape
    return da.broadcast_to(
        time_array.reshape(time_array.shape[0], 1, 1),
        (time_array.shape[0], lat_count, lon_count),
    )


def download_extract_historic_zip(s3_filesystem, s3_zip_path, local_temp_dir):
    """Download and extract an existing RAQDPS historic zip archive."""
    os.makedirs(local_temp_dir, exist_ok=True)
    local_zarr_path = os.path.join(
        local_temp_dir, os.path.basename(s3_zip_path).removesuffix(".zip")
    )

    if os.path.exists(local_zarr_path):
        return local_zarr_path

    if not s3_filesystem.exists(s3_zip_path):
        logger.warning(
            "Historic marker exists, but archive is missing: %s", s3_zip_path
        )
        return None

    local_zip_path = os.path.join(local_temp_dir, os.path.basename(s3_zip_path))
    extract_dir = local_zarr_path + "_extract"

    try:
        s3_filesystem.get_file(s3_zip_path, local_zip_path)
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(local_zip_path) as zip_file:
            zip_file.extractall(extract_dir)

        shutil.move(extract_dir, local_zarr_path)
        return local_zarr_path
    finally:
        if os.path.exists(local_zip_path):
            os.remove(local_zip_path)
        shutil.rmtree(extract_dir, ignore_errors=True)


def prepare_directories():
    """Reset processing directories and ensure output roots exist."""
    if os.path.exists(forecast_process_dir):
        shutil.rmtree(forecast_process_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(local_temp_historic_dir, exist_ok=True)

    if save_type == "S3":
        s3.mkdirs(historic_path, exist_ok=True)
    else:
        os.makedirs(historic_path, exist_ok=True)
        os.makedirs(os.path.join(forecast_path, ingest_version), exist_ok=True)


def already_processed(base_time):
    """Return True if the published RAQDPS store is already current."""
    time_pickle_path = os.path.join(forecast_path, ingest_version, "RAQDPS.time.pickle")
    base_time = normalize_utc(base_time)

    if save_type == "S3":
        if s3.exists(time_pickle_path):
            with s3.open(time_pickle_path, "rb") as f:
                return normalize_utc(pickle.load(f)) >= base_time
        return False

    if os.path.exists(time_pickle_path):
        with open(time_pickle_path, "rb") as f:
            return normalize_utc(pickle.load(f)) >= base_time
    return False


def process_forecast(base_time):
    """Download and write f000-f072 RAQDPS forecast data."""
    logger.info("Processing RAQDPS forecast from base time: %s", base_time)
    time_slices = []
    valid_times = []
    reference_shape = None

    for forecast_hour in RAQDPS_FORECAST_HOURS:
        fields, slice_shape = load_raqdps_time_slice(base_time, forecast_hour)
        valid_time = base_time + timedelta(hours=forecast_hour)

        if fields is None:
            logger.warning(
                "Skipping missing RAQDPS forecast hour: f%03d", forecast_hour
            )
            continue

        if reference_shape is None:
            reference_shape = slice_shape
        elif slice_shape != reference_shape:
            logger.warning(
                "Skipping RAQDPS f%03d due to shape mismatch %s != %s",
                forecast_hour,
                slice_shape,
                reference_shape,
            )
            continue

        time_slices.append(fields)
        valid_times.append(valid_time)

    if not time_slices or reference_shape is None:
        raise RuntimeError("No valid RAQDPS forecast hours were processed")

    forecast_arrays = build_variable_arrays(time_slices, valid_times, reference_shape)
    forecast_zarr_path = forecast_process_path + "_forecast.zarr"
    write_variable_zarrs(forecast_arrays, forecast_zarr_path, len(valid_times))
    return forecast_zarr_path, reference_shape


def process_historic_hour(valid_time, reference_shape):
    """Process one cached historic valid hour and return its zarr path."""
    timestamp_str = valid_time.strftime("%Y%m%dT%H%M%SZ")

    if save_type == "S3":
        s3_path = os.path.join(historic_path, f"RAQDPS_Hist_{timestamp_str}.zarr.zip")
        s3_done_path = s3_path.replace(".zarr.zip", ".done")
        if s3.exists(s3_done_path):
            logger.info("Historic RAQDPS file already exists in S3: %s", timestamp_str)
            historic_zarr_path = download_extract_historic_zip(
                s3, s3_path, local_temp_historic_dir
            )
            if historic_zarr_path is not None and is_variable_zarr_store(
                historic_zarr_path
            ):
                return historic_zarr_path
            logger.info(
                "Regenerating legacy historic RAQDPS store in variable layout: %s",
                timestamp_str,
            )
    else:
        local_path = os.path.join(historic_path, f"RAQDPS_Hist_{timestamp_str}.zarr")
        if os.path.exists(local_path + ".done"):
            logger.info(
                "Historic RAQDPS file already exists locally: %s", timestamp_str
            )
            if is_variable_zarr_store(local_path):
                return local_path
            if os.path.exists(local_path):
                logger.info(
                    "Regenerating legacy historic RAQDPS store in variable layout: %s",
                    timestamp_str,
                )
            else:
                logger.warning(
                    "Historic done marker exists, but zarr is missing: %s", local_path
                )
            shutil.rmtree(local_path, ignore_errors=True)
            if os.path.exists(local_path + ".done"):
                os.remove(local_path + ".done")

    run_time, forecast_hour = history_run_for_valid_time(valid_time)
    logger.info(
        "Processing RAQDPS historic valid=%s from run=%s f%03d",
        valid_time,
        run_time,
        forecast_hour,
    )

    fields, slice_shape = load_raqdps_time_slice(run_time, forecast_hour)
    if fields is None:
        logger.warning("No valid RAQDPS historic data for %s", timestamp_str)
        return None
    if slice_shape != reference_shape:
        logger.warning(
            "Skipping RAQDPS historic %s due to shape mismatch %s != %s",
            timestamp_str,
            slice_shape,
            reference_shape,
        )
        return None

    hist_arrays = build_variable_arrays([fields], [valid_time], reference_shape)
    hist_tmp_path = os.path.join(
        forecast_process_dir, f"RAQDPS_Hist_{timestamp_str}_TMP.zarr"
    )
    write_variable_zarrs(hist_arrays, hist_tmp_path, 1)

    if save_type == "S3":
        zip_export_path = hist_tmp_path.replace("_TMP.zarr", "")
        shutil.make_archive(zip_export_path, "zip", hist_tmp_path)
        s3.put_file(zip_export_path + ".zip", s3_path)

        tmp_done = os.path.join(tmp_dir, f"{timestamp_str}.done")
        with open(tmp_done, "w") as f:
            f.write("Done")
        s3.put_file(tmp_done, s3_done_path)
        os.remove(tmp_done)
        return hist_tmp_path

    final_local_path = os.path.join(historic_path, f"RAQDPS_Hist_{timestamp_str}.zarr")
    shutil.copytree(hist_tmp_path, final_local_path, dirs_exist_ok=True)
    with open(final_local_path + ".done", "w") as f:
        f.write("Done")
    shutil.rmtree(hist_tmp_path)
    return final_local_path


def process_historic(base_time, reference_shape):
    """Process/cache the 48 historical RAQDPS valid hours."""
    historic_zarr_paths = []
    for valid_time in history_valid_times(base_time, history_period):
        historic_path_processed = process_historic_hour(valid_time, reference_shape)
        if historic_path_processed is not None:
            historic_zarr_paths.append(historic_path_processed)
    return historic_zarr_paths


def merge_and_publish(
    base_time, historic_zarr_paths, forecast_zarr_path, reference_shape
):
    """Merge historic and forecast arrays and publish the final RAQDPS store."""
    logger.info("Merging historic and forecast RAQDPS datasets.")

    intermediate_paths = []
    for historic_zarr_path in historic_zarr_paths:
        if os.path.exists(historic_zarr_path):
            intermediate_paths.append(historic_zarr_path)
        else:
            logger.info("Missing historic RAQDPS zarr: %s", historic_zarr_path)

    intermediate_paths.append(forecast_zarr_path)

    dask_variable_list = []
    for variable in RAQDPS_OUTPUT_VARS:
        variable_time_arrays = []
        for intermediate_path in intermediate_paths:
            variable_array = read_intermediate_variable(intermediate_path, variable)
            if variable == "time":
                variable_array = expand_time_array(variable_array, reference_shape)
            variable_time_arrays.append(variable_array)

        merged_variable = da.concatenate(variable_time_arrays, axis=0).reshape(
            1,
            sum(variable_array.shape[0] for variable_array in variable_time_arrays),
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

    if save_type == "S3":
        final_zarr_path = os.path.join(forecast_process_dir, "RAQDPS.zarr.zip")
        zarr_store = zarr.storage.ZipStore(final_zarr_path, mode="a", compression=0)
    else:
        final_zarr_path = os.path.join(forecast_process_dir, "RAQDPS.zarr")
        zarr_store = zarr.storage.LocalStore(final_zarr_path)

    zarr_array = zarr.create_array(
        store=zarr_store,
        shape=stacked_array_padded.shape,
        chunks=(
            len(RAQDPS_OUTPUT_VARS),
            stacked_array_padded.shape[1],
            final_chunk,
            final_chunk,
        ),
        dtype=np.float32,
        overwrite=True,
    )

    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
            stacked_array_padded.round(5).rechunk(
                (
                    len(RAQDPS_OUTPUT_VARS),
                    stacked_array_padded.shape[1],
                    final_chunk,
                    final_chunk,
                )
            ).to_zarr(zarr_array, overwrite=True, compute=True)

    close_store(zarr_store)

    pickle_file_path = os.path.join(tmp_dir, "RAQDPS.time.pickle")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(base_time, f)

    if save_type == "S3":
        s3.put_file(
            final_zarr_path,
            os.path.join(forecast_path, ingest_version, "RAQDPS.zarr.zip"),
        )
        s3.put_file(
            pickle_file_path,
            os.path.join(forecast_path, ingest_version, "RAQDPS.time.pickle"),
        )
        logger.info("RAQDPS S3 target deployment complete.")
    else:
        shutil.move(
            pickle_file_path,
            os.path.join(forecast_path, ingest_version, "RAQDPS.time.pickle"),
        )
        shutil.copytree(
            final_zarr_path,
            os.path.join(forecast_path, ingest_version, "RAQDPS.zarr"),
            dirs_exist_ok=True,
        )
        logger.info("Saved RAQDPS datasets to local environment.")


def main():
    """Run the RAQDPS ingest."""
    prepare_directories()
    start_time = time.time()

    try:
        base_time = get_latest_raqdps_run()
    except RuntimeError as e:
        logger.critical("%s. Exiting.", e)
        sys.exit(1)

    logger.info("RAQDPS base time: %s", base_time)
    if already_processed(base_time):
        logger.info("RAQDPS store is already up to date. Exiting.")
        sys.exit()

    try:
        forecast_zarr_path, reference_shape = process_forecast(base_time)
        historic_zarr_paths = process_historic(base_time, reference_shape)
        merge_and_publish(
            base_time, historic_zarr_paths, forecast_zarr_path, reference_shape
        )
    finally:
        try:
            shutil.rmtree(forecast_process_dir)
        except Exception:
            pass

    end_time = time.time()
    logger.info("Total processing time: %.2f seconds", end_time - start_time)
    logger.info("RAQDPS ingest script finished successfully.")


if __name__ == "__main__":
    main()
