# %% RTMA Rapid Update Processing script using Dask, FastHerbie, and MetPy
# Alexander Rey, September 2025

# %% Import modules
import logging
import os
import shutil
import time
import warnings

# Define ECCODES_DEFINITION_PATH env variable for eccodes
# This is needed in my testing instance- should not be required for the docker image
# os.environ["ECCODES_DEFINITION_PATH"] = (
#    "/home/ubuntu/eccodes-2.40.0-Source/definitions/"
# )
import numpy as np
import s3fs
import xarray as xr
from herbie import Herbie
from herbie.fast import Herbie_latest
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

from API.constants.shared_const import INGEST_VERSION_STR
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
    pad_to_chunk_size,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/RTMA_RU"
)
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RTMA_RU")

# Define the processing and final chunk size
process_chunk = CHUNK_SIZES["RTMA"]
final_chunk = FINAL_CHUNK_SIZES["RTMA"]

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")


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


def main() -> None:
    """Run the RTMA Rapid Update ingest."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    herbie_save_dir = make_herbie_save_dir(tmp_dir)

    # %% Define base time from the most recent run
    t0 = time.time()

    latest_run = Herbie_latest(
        model="rtma_ru",
        n=5,
        freq="15min",
        product="anl",
        verbose=True,
        priority=["aws", "nomads"],
        save_dir=herbie_save_dir,
    )

    base_time = latest_run.date
    logger.info(f"Checking for new RTMA_RU data for base time: {base_time}")

    # Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/RTMA_RU.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to RTMA_RU, ending")
        return

    # %% Download RTMA analysis data using Herbie Latest
    match_strings = (
        ":((DPT|TMP|SPFH):2 m above ground:)"
        "|:(GUST:10 m above ground:)"
        "|:(UGRD:10 m above ground:)"
        "|:(VGRD:10 m above ground:)"
        "|:((VIS|PRES):surface:)"
        "|:TCDC:entire atmosphere"
    )

    fh_analysis = Herbie(
        base_time,
        model="rtma_ru",
        product="anl",
        verbose=False,
        priority=["aws", "nomads"],
        save_dir=herbie_save_dir,
    )

    fh_analysis.download(match_strings, verbose=True)

    logger.info("RTMA_RU GRIB file downloaded successfully.")

    xarray_herbie_list = fh_analysis.xarray(match_strings)

    # Merge the three datasets into one
    xarray_analysis_merged = xr.merge(xarray_herbie_list, compat="override")

    # Assign coordinates from one of the datasets to the merged dataset
    xarray_analysis_merged = xarray_analysis_merged.assign_coords(
        xarray_herbie_list[0].metpy.parse_cf().coords
    )

    # Convert RH from specific humidity and pressure and add it to the dataset
    # relative_humidity_from_specific_humidity returns a dimensionless fraction (0-1)
    rh_2m = relative_humidity_from_specific_humidity(
        pressure=xarray_analysis_merged["sp"] * units.Pa,
        temperature=xarray_analysis_merged["t2m"] * units.degK,
        specific_humidity=xarray_analysis_merged["sh2"] * units("kg/kg"),
    )

    xarray_analysis_merged["rh"] = rh_2m.metpy.dequantify()

    # Convert winds from grid relative to earth relative
    u_earth, v_earth = earth_relative_wind_components(
        xarray_analysis_merged["u10"], xarray_analysis_merged["v10"]
    )

    # Put U and V back into the dataset, replacing the grid relative versions
    xarray_analysis_merged["u10"].data = u_earth
    xarray_analysis_merged["v10"].data = v_earth

    # Drop time as a coordinate
    model_UNIX_time = xarray_analysis_merged.time.data.astype("datetime64[s]").astype(
        int
    )
    xarray_analysis_merged = xarray_analysis_merged.reset_coords("time", drop=True)

    # Add a new data variables for time
    # Same X Y shape as the rest, identical values
    # UNIX time
    xarray_analysis_merged["time"] = (
        ("y", "x"),
        np.full(
            (xarray_analysis_merged.dims["y"], xarray_analysis_merged.dims["x"]),
            model_UNIX_time,
        ),
    )

    # Clip to valid data ranges
    for var in zarr_vars:
        if var == "time":
            continue
        elif var in xarray_analysis_merged.data_vars:
            ds_clip = xarray_analysis_merged[var]
            if np.issubdtype(ds_clip.dtype, np.number):
                mask = (ds_clip >= VALID_DATA_MIN) & (ds_clip <= VALID_DATA_MAX)
                xarray_analysis_merged[var] = ds_clip.where(mask)  # out-of-range → NaN

    # Drop the sh2 variable as we no longer need it
    xarray_analysis_merged = xarray_analysis_merged.drop_vars("sh2")

    # Set the order correctly
    vars_in = [v for v in zarr_vars if v in xarray_analysis_merged.data_vars]

    # Merge the arrays into a single 3D array with the correct order, add a 1 length time dimension, and rechunk
    xarray_analysis_stack = (
        xarray_analysis_merged[vars_in]
        .to_stacked_array(new_dim="var", sample_dims=["y", "x"])
        .expand_dims("time", axis=1)
        .chunk(chunks={"var": -1, "time": 1, "x": final_chunk, "y": final_chunk})
        .transpose("var", "time", "y", "x")
    )

    # Mask out invalid data
    dask_var_array = mask_invalid_data(xarray_analysis_stack)

    # Add padding to the zarr store
    dask_var_array = pad_to_chunk_size(dask_var_array, final_chunk)

    rtma_ru_store_name = "RTMA_RU.zarr.zip" if save_type == "S3" else "RTMA_RU.zarr"
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
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("Cleaning up temporary processing directories.")

    # Test Read
    t1 = time.time()
    logger.info(f"Total script execution time: {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
