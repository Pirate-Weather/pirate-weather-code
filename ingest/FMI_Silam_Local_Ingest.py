"""FMI SILAM global air quality ingest.

Downloads the newest daily FMI SILAM global surface air quality forecast
(SILAM v6.1, 0.2 degree global grid, hourly leads starting at run + 1 h) from
the THREDDS server, converts PM2.5, PM10, fire PM column, boundary layer height
and the O3, NO2, SO2 and CO mixing ratios to the stored units, and merges it
with the first 24 h of each run in the previous 48 h (short-term history,
archived per run for AQI lookbacks). Publishes ``SILAM.zarr`` plus
``SILAM.time.pickle``. The store is not interpolated:
it holds the native hourly steps from run - 47 h onwards.

SILAM (System for Integrated modeLling of Atmospheric coMposition):
https://thredds.silam.fmi.fi/thredds/catalog/catalog.html

Alexander Rey, December 2025
"""

import gc
import logging
import os
import re
import shutil
import time
import warnings
from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree

import dask
import dask.array as da
import numpy as np
import pandas as pd
import requests
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

from API.constants.shared_const import HISTORY_PERIODS, INGEST_VERSION_STR
from API.constants.unit_const import KG_M2_TO_UG_M2, KG_M3_TO_UG_M3, MOL_MOL_TO_PPB
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    archive_tmp_zarr_and_upload,
    configure_zarr_writes,
    download_extract_historic_archive,
    mask_invalid_data,
    pad_to_chunk_size,
    publish_ingest_run,
    read_published_run_time,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
# SILAM steps are hourly means stamped at the end of the hour, so the first
# valid time of a run is run + 1 h.
SILAM_FIRST_LEAD_HOURS = 1
SILAM_STEP_SECONDS = 3600
# SILAM runs once a day; each history run contributes its first 24 h
HISTORIC_STEP_HOURS = 24
HISTORY_PERIOD_HOURS = HISTORY_PERIODS["SILAM"]

SILAM_DOWNLOAD_RETRIES = 3
SILAM_RETRY_SLEEP_SECONDS = 5
SILAM_DOWNLOAD_TIMEOUT_SECONDS = 120
SILAM_CATALOG_TIMEOUT_SECONDS = 30

# Variables in the published store; the order is the API's SILAM index map
zarr_vars = (
    "time",
    "cnc_PM2_5",
    "cnc_PM10",
    "PM_FRP_column",
    "BLH",
    "cnc_O3",
    "cnc_NO2",
    "cnc_SO2",
    "cnc_CO",
)

# SILAM source variables and units:
# - cnc_PM2_5, cnc_PM10: particulate matter in kg/m3 (stored in ug/m3)
# - PM_FRP_column: fire PM column in kg/m2 (stored in ug/m2)
# - BLH: boundary layer height in m
# - vmr_*_gas: gas volume mixing ratios in mol/mol (stored in ppb)
silam_source_vars = (
    "cnc_PM2_5",
    "cnc_PM10",
    "PM_FRP_column",
    "BLH",
    "vmr_O3_gas",
    "vmr_NO2_gas",
    "vmr_SO2_gas",
    "vmr_CO_gas",
)
# SILAM gas mixing ratio -> stored variable name, description
SILAM_GAS_VARIABLES = {
    "vmr_O3_gas": ("cnc_O3", "Ozone"),
    "vmr_NO2_gas": ("cnc_NO2", "Nitrogen dioxide"),
    "vmr_SO2_gas": ("cnc_SO2", "Sulfur dioxide"),
    "vmr_CO_gas": ("cnc_CO", "Carbon monoxide"),
}

base_fileserver_url = "https://thredds.silam.fmi.fi/thredds/fileServer"
base_catalog_url = "https://thredds.silam.fmi.fi/thredds/catalog"
silam_dataset_path = "silam_glob_v6_1_sfc/files"
silam_filename_pattern = re.compile(r"SILAM-AQ-sfc-glob_v6_1_(\d{10})\.nc4$")

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/SILAM")
forecast_process_path = os.path.join(forecast_process_dir, "SILAM_Process")
hist_process_path = os.path.join(forecast_process_dir, "SILAM_Historic")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/SILAM")
historic_path = os.getenv("historic_path", default="/mnt/nvme/data/History/SILAM")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

process_chunk = CHUNK_SIZES["SILAM"]
final_chunk = FINAL_CHUNK_SIZES["SILAM"]

# Open one history run (25 steps) per time chunk
silam_open_chunks = {
    "time": HISTORIC_STEP_HOURS + 1,
    "lat": process_chunk,
    "lon": process_chunk,
}


def get_latest_silam_run() -> datetime:
    """Return the newest SILAM run listed in the THREDDS catalog.

    SILAM availability lags wall-clock time by a variable amount, so the run is
    read from the server catalog. If no catalog can be read the ingest stops:
    a guessed run time could name a file that does not exist or is incomplete.
    """
    catalog_urls = (
        f"{base_catalog_url}/{silam_dataset_path}/catalog.xml",
        f"{base_catalog_url}/{silam_dataset_path}/latest.xml",
    )

    for catalog_url in catalog_urls:
        try:
            response = requests.get(catalog_url, timeout=SILAM_CATALOG_TIMEOUT_SECONDS)
            response.raise_for_status()
            catalog = ElementTree.fromstring(response.content)
        except (ElementTree.ParseError, requests.RequestException) as e:
            logger.warning("Unable to read SILAM catalog %s: %s", catalog_url, e)
            continue

        run_times = []
        for dataset in catalog.iter():
            for dataset_value in (dataset.get("name", ""), dataset.get("urlPath", "")):
                match = silam_filename_pattern.search(dataset_value)
                if match:
                    run_times.append(
                        datetime.strptime(match.group(1), "%Y%m%d%H").replace(
                            tzinfo=UTC
                        )
                    )

        if run_times:
            latest_origintime = max(run_times)
            logger.info(
                "Latest SILAM run from catalog %s: %s", catalog_url, latest_origintime
            )
            return latest_origintime

        logger.warning("No SILAM run files found in catalog %s", catalog_url)

    raise RuntimeError(
        f"Could not determine the latest SILAM run from {list(catalog_urls)}"
    )


def build_silam_download_url(run_time: datetime) -> str:
    """Build the SILAM THREDDS fileServer URL for a model run."""
    run_filename = f"SILAM-AQ-sfc-glob_v6_1_{run_time.strftime('%Y%m%d%H.nc4')}"
    return f"{base_fileserver_url}/{silam_dataset_path}/{run_filename}"


def download_silam_file(url: str, local_path: str) -> bool:
    """Download one SILAM NetCDF file with bounded retries.

    Returns:
        True if the download succeeded, False if the file does not exist (404)
        or every attempt failed.
    """
    for attempt in range(1, SILAM_DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(
                url, stream=True, timeout=SILAM_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                if response.status_code == 404:
                    logger.warning("SILAM file not found: %s", url)
                    return False
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            logger.info("Downloaded SILAM file to: %s", local_path)
            return True
        except (requests.RequestException, OSError) as e:
            logger.warning(
                "Attempt %d/%d failed downloading %s: %s",
                attempt,
                SILAM_DOWNLOAD_RETRIES,
                url,
                e,
            )
            # Do not leave a partial file behind
            if os.path.exists(local_path):
                os.remove(local_path)
            if attempt < SILAM_DOWNLOAD_RETRIES:
                time.sleep(SILAM_RETRY_SLEEP_SECONDS)

    return False


def _make_nan_dataarray(
    time_coord, lat_coord, lon_coord, fill=np.nan, dtype=np.float32
):
    """Return a 3D DataArray filled with `fill` matching provided coords."""
    shape = (len(time_coord), len(lat_coord), len(lon_coord))
    return xr.DataArray(
        np.full(shape, fill, dtype=dtype),
        dims=["time", "latitude", "longitude"],
        coords={"time": time_coord, "latitude": lat_coord, "longitude": lon_coord},
    )


def process_silam_file(
    local_nc_path: str,
    time_slice: slice | None = None,
) -> xr.Dataset:
    """Open a SILAM NetCDF file and convert it to the stored variables and units."""
    # Only read the variables the store needs
    with xr.open_dataset(local_nc_path, engine="netcdf4", decode_cf=False) as meta:
        drop_variables = [
            var_name for var_name in meta.data_vars if var_name not in silam_source_vars
        ]

    xarray_silam_data = xr.open_dataset(
        local_nc_path,
        engine="netcdf4",
        chunks=silam_open_chunks,
        drop_variables=drop_variables,
    )
    logger.info(
        "Opened SILAM file %s: dims %s, variables %s",
        local_nc_path,
        dict(xarray_silam_data.sizes),
        list(xarray_silam_data.data_vars),
    )

    # Select the surface level if a vertical dimension exists
    if "height" in xarray_silam_data.dims:
        xarray_silam_data = xarray_silam_data.isel(height=0)
    elif "level" in xarray_silam_data.dims:
        xarray_silam_data = xarray_silam_data.isel(level=0)

    if "lat" in xarray_silam_data.coords:
        xarray_silam_data = xarray_silam_data.rename({"lat": "latitude"})
    if "lon" in xarray_silam_data.coords:
        xarray_silam_data = xarray_silam_data.rename({"lon": "longitude"})

    if time_slice is not None:
        xarray_silam_data = xarray_silam_data.sel(time=time_slice)

    xarray_processed = xr.Dataset(
        coords={
            "time": xarray_silam_data["time"],
            "latitude": xarray_silam_data["latitude"],
            "longitude": xarray_silam_data["longitude"],
        }
    )

    def _nan_field():
        return _make_nan_dataarray(
            xarray_processed.time,
            xarray_processed.latitude,
            xarray_processed.longitude,
        )

    # Particulate matter: kg/m3 -> ug/m3
    for pm_var, long_name in (
        ("cnc_PM2_5", "PM2.5 concentration"),
        ("cnc_PM10", "PM10 concentration"),
    ):
        if pm_var in xarray_silam_data:
            xarray_processed[pm_var] = (
                xarray_silam_data[pm_var] * KG_M3_TO_UG_M3
            ).astype(np.float32)
            xarray_processed[pm_var].attrs["units"] = "µg/m³"
            xarray_processed[pm_var].attrs["long_name"] = long_name
        else:
            logger.warning("%s not found in SILAM file, storing NaN", pm_var)
            xarray_processed[pm_var] = _nan_field()

    # Fire PM column: kg/m2 -> ug/m2
    if "PM_FRP_column" in xarray_silam_data:
        xarray_processed["PM_FRP_column"] = (
            xarray_silam_data["PM_FRP_column"] * KG_M2_TO_UG_M2
        ).astype(np.float32)
        xarray_processed["PM_FRP_column"].attrs["units"] = "µg/m²"
        xarray_processed["PM_FRP_column"].attrs["long_name"] = "Total PM_FRP column"
    else:
        logger.warning("PM_FRP_column not found in SILAM file, storing NaN")
        xarray_processed["PM_FRP_column"] = _nan_field()

    if "BLH" in xarray_silam_data:
        xarray_processed["BLH"] = xarray_silam_data["BLH"].astype(np.float32)
        xarray_processed["BLH"].attrs["units"] = "m"
        xarray_processed["BLH"].attrs["long_name"] = "Boundary layer height"
    else:
        logger.warning("BLH not found in SILAM file, storing NaN")
        xarray_processed["BLH"] = _nan_field()

    # Gas volume mixing ratios: mol/mol -> ppb
    for silam_var, (output_var, long_name) in SILAM_GAS_VARIABLES.items():
        if silam_var in xarray_silam_data:
            xarray_processed[output_var] = (
                xarray_silam_data[silam_var] * MOL_MOL_TO_PPB
            ).astype(np.float32)
            xarray_processed[output_var].attrs["units"] = "ppb"
            xarray_processed[output_var].attrs["long_name"] = (
                f"{long_name} volume mixing ratio"
            )
        else:
            logger.warning(
                "%s not found in SILAM file, storing NaN for %s", silam_var, output_var
            )
            xarray_processed[output_var] = _nan_field()

    return xarray_processed


def convert_time_coord_to_unix(xarray_processed: xr.Dataset) -> xr.Dataset:
    """Return a dataset with numeric Unix-second time coordinates."""
    time_unix = (
        pd.to_datetime(xarray_processed.time.values).astype("int64") // 1_000_000_000
    )
    return xarray_processed.assign_coords(time=time_unix.astype(np.int64))


def save_processed_zarr(
    xarray_processed: xr.Dataset, zarr_path: str, zarr_store_workers: int
) -> None:
    """Chunk and save a processed SILAM dataset as a zarr group."""
    xarray_processed = xarray_processed.chunk(
        chunks={
            "time": xarray_processed.time.size,
            "latitude": process_chunk,
            "longitude": process_chunk,
        }
    )

    with ProgressBar():
        xarray_processed.to_zarr(
            zarr_path,
            mode="w",
            consolidated=False,
            compute=True,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )


def main() -> None:
    """Run the FMI SILAM air quality ingest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
    zarr_store_workers = configure_zarr_writes()

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(hist_process_path, exist_ok=True)
    if save_type == "S3":
        s3.mkdirs(historic_path, exist_ok=True)
    else:
        os.makedirs(historic_path, exist_ok=True)

    publish_dir = forecast_path + "/" + ingest_version
    if save_type == "Download":
        os.makedirs(publish_dir, exist_ok=True)

    T0 = time.time()

    # %% Determine the model run from the THREDDS catalog
    origintime = get_latest_silam_run()

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        publish_dir + "/SILAM.time.pickle", save_type, s3
    )
    if previous_base_time is not None and previous_base_time >= origintime:
        logger.info("No Update to SILAM, ending")
        return
    logger.info("Processing SILAM run %s", origintime)

    # %% Download and convert the forecast
    download_url = build_silam_download_url(origintime)
    local_nc_path = os.path.join(tmp_dir, "SILAM_latest.nc")
    if not download_silam_file(download_url, local_nc_path):
        raise RuntimeError(
            f"Failed to download SILAM run {origintime} from {download_url}"
        )

    xarray_processed = process_silam_file(local_nc_path)

    # The forecast must be hourly from run + 1 h, or the history/forecast join
    # and the map window would be misaligned
    forecast_times = pd.to_datetime(xarray_processed.time.values)
    expected_first_time = pd.Timestamp(origintime.replace(tzinfo=None)) + pd.Timedelta(
        hours=SILAM_FIRST_LEAD_HOURS
    )
    if forecast_times[0] != expected_first_time or not np.all(
        np.diff(forecast_times.values) == np.timedelta64(SILAM_STEP_SECONDS, "s")
    ):
        raise RuntimeError(
            f"SILAM run {origintime} times {forecast_times.values} are not hourly "
            f"from {expected_first_time}"
        )

    xarray_forecast_processed = convert_time_coord_to_unix(xarray_processed)
    save_processed_zarr(
        xarray_forecast_processed, forecast_process_path + "_.zarr", zarr_store_workers
    )
    lat_count = xarray_forecast_processed.sizes["latitude"]
    lon_count = xarray_forecast_processed.sizes["longitude"]
    xarray_forecast_processed.close()
    xarray_processed.close()
    del xarray_forecast_processed, xarray_processed
    gc.collect()
    logger.info("Saved SILAM forecast zarr to %s_.zarr", forecast_process_path)

    # %% History: archive the first 24 h of each earlier run not yet processed.
    # The API reads the hourly rows by position, so a missing history run would
    # shift every later hour; fail instead of publishing a store with a gap.
    history_offsets = range(HISTORY_PERIOD_HOURS, 0, -HISTORIC_STEP_HOURS)
    for hours_offset in history_offsets:
        hist_start = origintime - timedelta(hours=hours_offset)
        hist_end = hist_start + timedelta(hours=HISTORIC_STEP_HOURS + 1)
        timestamp = hist_start.strftime("%Y%m%dT%H%M%SZ")

        if save_type == "S3":
            s3_path = f"{historic_path}/SILAM_Hist_v3{timestamp}.zarr.tar.gz"
            if s3.exists(s3_path.replace(".tar.gz", ".done")):
                logger.info("Historic SILAM file already exists in S3: %s", s3_path)
                continue
        else:
            local_path = f"{historic_path}/SILAM_Hist_v3{timestamp}.zarr"
            if os.path.exists(local_path.replace(".zarr", ".done")):
                logger.info(
                    "Historic SILAM file already exists locally: %s", local_path
                )
                continue

        hist_url = build_silam_download_url(hist_start)
        hist_nc_path = os.path.join(tmp_dir, f"SILAM_hist_{timestamp}.nc")
        logger.info("Downloading historic SILAM run: %s", timestamp)

        if not download_silam_file(hist_url, hist_nc_path):
            raise RuntimeError(f"Historic SILAM run {timestamp} is missing: {hist_url}")

        # Keep run + 1 h to run + 24 h (the end bound is inclusive)
        slice_end = hist_end - timedelta(seconds=1)
        try:
            xarray_hist_processed = process_silam_file(
                hist_nc_path,
                time_slice=slice(
                    hist_start.replace(tzinfo=None), slice_end.replace(tzinfo=None)
                ),
            )
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Historic SILAM run {timestamp} is unreadable: {hist_nc_path}"
            ) from e

        hist_steps = xarray_hist_processed.sizes.get("time", 0)
        if hist_steps != HISTORIC_STEP_HOURS:
            raise RuntimeError(
                f"Historic SILAM run {timestamp} has {hist_steps} of "
                f"{HISTORIC_STEP_HOURS} hours"
            )

        hist_tmp_zarr_path = hist_process_path + "_SILAM_Hist_TMP.zarr"
        xarray_hist_processed = convert_time_coord_to_unix(xarray_hist_processed)
        save_processed_zarr(
            xarray_hist_processed, hist_tmp_zarr_path, zarr_store_workers
        )

        if save_type == "S3":
            archive_tmp_zarr_and_upload(
                tmp_zarr_path=hist_tmp_zarr_path,
                s3_path=s3_path,
                archive_member_name="SILAM_Hist.zarr",
                s3=s3,
            )
        else:
            os.rename(hist_tmp_zarr_path, local_path)
            with open(local_path.replace(".zarr", ".done"), "w") as f:
                f.write("Done")

        logger.info("Saved historic SILAM slice: %s", timestamp)

    # %% Merge the history and forecast datasets
    if save_type == "S3":
        local_temp_dir = forecast_process_path + "_s3_temp_downloads"
        os.makedirs(local_temp_dir, exist_ok=True)
        historic_zarr_paths = []
        for hours_offset in history_offsets:
            timestamp = (origintime - timedelta(hours=hours_offset)).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            extracted_path = download_extract_historic_archive(
                s3=s3,
                historic_path=historic_path,
                final_zarr_name=f"SILAM_Hist_v3{timestamp}.zarr",
                extracted_store_name="SILAM_Hist.zarr",
                local_temp_dir=local_temp_dir,
                expected_vars=zarr_vars,
            )
            if extracted_path is None:
                raise RuntimeError(
                    f"Historic SILAM archive SILAM_Hist_v3{timestamp} is missing "
                    f"from {historic_path}"
                )
            historic_zarr_paths.append(extracted_path)
    else:
        historic_zarr_paths = []
        for hours_offset in history_offsets:
            timestamp = (origintime - timedelta(hours=hours_offset)).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            local_path = f"{historic_path}/SILAM_Hist_v3{timestamp}.zarr"
            if not os.path.exists(local_path):
                raise RuntimeError(f"Historic SILAM store {local_path} is missing")
            historic_zarr_paths.append(local_path)

    dask_interp_arrays = []
    for dask_var in zarr_vars:
        dask_var_arrays_list = []
        for historic_zarr_path in historic_zarr_paths:
            try:
                dask_var_arrays_list.append(
                    da.from_zarr(
                        historic_zarr_path, component=dask_var, inline_array=True
                    )
                )
            except (FileNotFoundError, KeyError) as e:
                raise RuntimeError(
                    f"Historic SILAM store {historic_zarr_path} is missing {dask_var}"
                ) from e

        dask_forecast_array = da.from_zarr(
            forecast_process_path + "_.zarr", component=dask_var, inline_array=True
        )

        if dask_var == "time":
            dask_time_arrays = [da.squeeze(array) for array in dask_var_arrays_list]
            dask_time_arrays.append(dask_forecast_array)
            # Integer Unix seconds of every stacked step
            stacked_times_unix = da.concatenate(dask_time_arrays, axis=0).compute()
            if np.any(np.diff(stacked_times_unix) != SILAM_STEP_SECONDS):
                raise RuntimeError(
                    f"SILAM run {origintime} stacked times are not hourly: "
                    f"{stacked_times_unix}"
                )
            dask_times_concatenated = da.from_array(
                stacked_times_unix.astype("float32")
            )

            # The API reads time from the store like any other variable
            output_array = da.broadcast_to(
                dask_times_concatenated[:, None, None],
                (dask_times_concatenated.shape[0], lat_count, lon_count),
            ).rechunk((dask_times_concatenated.shape[0], process_chunk, process_chunk))
            dask_interp_arrays.append(output_array)
        else:
            output_array = da.concatenate(
                dask_var_arrays_list + [dask_forecast_array], axis=0
            )
            dask_interp_arrays.append(
                output_array.rechunk(
                    (output_array.shape[0], process_chunk, process_chunk)
                ).astype("float32")
            )

        logger.info("Processed variable: %s", dask_var)

    # %% Stack into the (variable, time, y, x) layout the API reads
    merged_arrays = da.stack(dask_interp_arrays, axis=0)
    merged_arrays_masked = mask_invalid_data(
        merged_arrays, ignoreAxis=[zarr_vars.index("PM_FRP_column")]
    )

    # Write out to disk. This intermediate step avoids memory overflow.
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        merged_arrays_masked.to_zarr(
            forecast_process_path + "_stack.zarr",
            overwrite=True,
            compute=True,
        )

    # %% Write the final store (SILAM is already hourly; no interpolation)
    stacked_array_disk = da.from_zarr(forecast_process_path + "_stack.zarr")
    stacked_array_padded = pad_to_chunk_size(stacked_array_disk, final_chunk)

    silam_store_name = "SILAM.zarr.zip" if save_type == "S3" else "SILAM.zarr"
    with ProgressBar():
        write_final_zarr(
            stacked_array_padded,
            forecast_process_dir + "/" + silam_store_name,
            final_chunk,
            zarr_store_workers,
        )

    # %% Publish the stores, then the run time the API uses to detect them
    publish_ingest_run(
        [silam_store_name],
        origintime,
        "SILAM.time.pickle",
        process_dir=forecast_process_dir,
        publish_dir=publish_dir,
        save_type=save_type,
        s3=s3,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("SILAM run %s published in %.1f s", origintime, time.time() - T0)


if __name__ == "__main__":
    main()
