# %% Regional Air Quality Deterministic Prediction System (RAQDPS) Ingest Script
# ruff: noqa: E402
# Downloads ECCC RAQDPS forecasts with Herbie, normalizes pollutant output units, and
# publishes the f000-f072 forecast as a Zarr store.

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
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
    configure_zarr_writes,
    mask_invalid_data,
    pad_to_chunk_size,
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

# %% Setup logging
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

forecast_path = os.getenv("forecast_path", default="/home/reya/Weather/Prod")
save_type = os.getenv("save_type", default="Download")

aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
process_chunk = CHUNK_SIZES["RAQDPS"]
final_chunk = FINAL_CHUNK_SIZES["RAQDPS"]
process_chunk_bytes = process_chunk * process_chunk * np.dtype(np.float32).itemsize
RAQDPS_TIME_PICKLE = "RAQDPS.time.pickle"
RAQDPS_LAT_LON_PICKLE = "RAQDPS.lat_lon.pickle"


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
    """Find the newest RAQDPS 00/12 UTC run with a f072 ozone file available."""
    for candidate in candidate_raqdps_runs(count=10):
        try:
            h = build_herbie(candidate, "O3", 72)
        except (FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError) as e:
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
    except (FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.warning(
            "Failed RAQDPS download for run=%s variable=%s f%03d: %s",
            run_time,
            variable,
            forecast_hour,
            e,
        )
        return None


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
    """Download/read all configured variables for one valid RAQDPS hour."""
    fields = {}
    reference_shape = None
    lat_lon_lookup = None

    for variable in RAQDPS_VARIABLES:
        local_grib_path = download_raqdps_file(run_time, variable, forecast_hour)
        if local_grib_path is None:
            continue

        try:
            values = read_raqdps_grib(local_grib_path, variable)
        except (OSError, ValueError, KeyError) as e:
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

        if include_lat_lon and lat_lon_lookup is None:
            try:
                lat_lon_lookup = read_raqdps_lat_lon(local_grib_path, values.shape)
            except (OSError, ValueError, KeyError) as e:
                logger.warning(
                    "Unable to read RAQDPS lat/lon from %s: %s",
                    local_grib_path,
                    e,
                )

    if reference_shape is None:
        if include_lat_lon:
            return None, None, None
        return None, None

    if include_lat_lon:
        return fields, reference_shape, lat_lon_lookup
    return fields, reference_shape


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
        arrays = []
        for fields in time_slices:
            if variable in fields:
                values = fields[variable]
            else:
                values = np.full(reference_shape, np.nan, dtype=np.float32)
            arrays.append(da.from_array(values, chunks=(process_chunk, process_chunk)))

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
    """Download and write f000-f072 RAQDPS forecast data."""
    logger.info("Processing RAQDPS forecast from base time: %s", base_time)
    time_slices = []
    valid_times = []
    reference_shape = None
    lat_lon_lookup = None

    for forecast_hour in RAQDPS_FORECAST_HOURS:
        slice_lat_lon = None
        if lat_lon_lookup is None:
            fields, slice_shape, slice_lat_lon = load_raqdps_time_slice(
                base_time,
                forecast_hour,
                include_lat_lon=True,
            )
        else:
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

        if lat_lon_lookup is None and slice_lat_lon is not None:
            lat_lon_lookup = slice_lat_lon

        time_slices.append(fields)
        valid_times.append(valid_time)

    if not time_slices or reference_shape is None:
        raise RuntimeError("No valid RAQDPS forecast hours were processed")
    if lat_lon_lookup is None:
        raise RuntimeError("No valid RAQDPS lat/lon grid was read")

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

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    prepare_directories()
    start_time = time.time()

    try:
        base_time = get_latest_raqdps_run()
    except RuntimeError as e:
        logger.critical("%s. Exiting.", e)
        sys.exit(1)

    logger.info("RAQDPS base time: %s", base_time)
    if already_processed(base_time, s3):
        logger.info("RAQDPS store is already up to date. Exiting.")
        return

    try:
        forecast_zarr_path, reference_shape, lat_lon_lookup = process_forecast(
            base_time, zarr_store_workers
        )
        merge_and_publish(
            base_time,
            forecast_zarr_path,
            reference_shape,
            lat_lon_lookup,
            s3,
            zarr_store_workers,
        )
    finally:
        try:
            shutil.rmtree(forecast_process_dir)
        except OSError as e:
            logger.warning(
                "Failed to clean up temporary directory %s: %s",
                forecast_process_dir,
                e,
            )

    end_time = time.time()
    logger.info("Total processing time: %.2f seconds", end_time - start_time)
    logger.info("RAQDPS ingest script finished successfully.")


if __name__ == "__main__":
    main()
