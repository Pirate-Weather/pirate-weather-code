# %% NOAA Air Quality Model (AQM) Processing script
# This script downloads the latest NOAA Air Quality Model forecast data
# from the NOMADS server (GRIB2 format), processes it, and saves to a Zarr store
# for API consumption.
#
# NOAA AQM (Air Quality Forecast Capability / AQFC) provides CONUS air quality
# forecasts including PM2.5 and Ozone at 12km resolution.
#
# Data source: https://www.nco.ncep.noaa.gov/pmb/products/aqm/
# NOMADS: https://nomads.ncep.noaa.gov/pub/data/nccf/com/aqm/prod/
#
# Model runs: 06Z and 12Z (both 72-hour forecasts)
# Spatial coverage: CONUS only (use SILAM for global coverage)
#
# Author: Alexander Rey
# Date: March 2026

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen

import numpy as np
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import CHUNK_SIZES, calculate_aqi

warnings.filterwarnings("ignore", "This pattern is interpreted")

# %% Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# %% Constants
# NOMADS base URL for AQM GRIB2 files
AQM_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aqm/prod"

# AQM runs at 06Z and 12Z UTC
AQM_RUN_HOURS = [12, 6]

# AQM CONUS grid IDs to try (the exact grid depends on the AQM version)
# Grid 148: CONUS 12km Lambert Conformal (primary), Grid 227: CONUS 1/4 degree
AQM_GRID_IDS = ["148", "227"]

# NOAA AQM variable names in GRIB2 files
# PMTF: Particulate Matter Fine (PM2.5)
#   GRIB2 units: "Particulate matter (fine) [10^-6g/m^3]" (= µg/m³, no conversion needed)
# OZCON: Ozone Concentration
#   GRIB2 units: "Ozone Concentration [ppb]" — must multiply by O3_PPB_TO_UG_M3

# Ozone unit conversion: ppm → µg/m³ at 25°C, 1 atm
# O3 molecular weight = 48 g/mol, molar volume at 25°C = 24.465 L/mol
O3_PPM_TO_UG_M3 = 48.0 / 24.465 * 1000.0  # ≈ 1962 µg/m³ per ppm

# Ozone unit conversion: ppb → µg/m³ at 25°C, 1 atm
O3_PPB_TO_UG_M3 = O3_PPM_TO_UG_M3 / 1000.0  # ≈ 1.962 µg/m³ per ppb


# %% Pure helper functions (importable for testing)


def get_latest_aqm_run():
    """Determine the latest available NOAA AQM model run time.

    AQM runs at 06Z and 12Z UTC. Data is typically available ~5 hours after
    the model run time. Checks in order: today's 12Z, today's 06Z,
    yesterday's 12Z, yesterday's 06Z.

    Returns:
        datetime: The latest available model run time.
    """
    now_utc = datetime.now(timezone.utc)
    availability_delay = timedelta(hours=5)

    # Try today and yesterday, from latest to earliest run
    for days_back in [0, 1]:
        candidate_date = now_utc - timedelta(days=days_back)
        for run_hour in AQM_RUN_HOURS:
            run_time = candidate_date.replace(
                hour=run_hour, minute=0, second=0, microsecond=0
            )
            if now_utc >= run_time + availability_delay:
                return run_time

    # Fallback: yesterday's 12Z
    fallback = (now_utc - timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    return fallback


def build_aqm_url(run_time, variable, grid_id):
    """Build NOMADS download URL for an AQM GRIB2 file.

    Args:
        run_time: Model run datetime (must be 06Z or 12Z).
        variable: Either 'pm25' or 'o3'.
        grid_id: GRIB2 grid identifier string (e.g., '148', '227').

    Returns:
        str: Full HTTPS URL for the GRIB2 file.
    """
    date_str = run_time.strftime("%Y%m%d")
    hour_str = run_time.strftime("%H")
    filename = f"aqm.t{hour_str}z.ave_1hr_{variable}.{grid_id}.grib2"
    return f"{AQM_NOMADS_BASE}/aqm.{date_str}/{filename}"


def convert_o3_to_ug_m3(da, units):
    """Convert ozone DataArray to µg/m³ if necessary.

    NOAA AQM OZCON is always in ppb ("Ozone Concentration [ppb]"), so this
    function normally applies the ppb → µg/m³ conversion. The unit string is
    checked as a safety net in case a future model version changes the units.

    Args:
        da: xarray.DataArray of ozone concentrations.
        units: Unit string from GRIB2 metadata.

    Returns:
        xarray.DataArray: Ozone in µg/m³.
    """
    units_lower = units.lower().strip()

    if "ppb" in units_lower:
        logger.info(f"Converting O3 from ppb to µg/m³ (factor={O3_PPB_TO_UG_M3:.3f})")
        return da * O3_PPB_TO_UG_M3
    elif "ppm" in units_lower:
        logger.info(f"Converting O3 from ppm to µg/m³ (factor={O3_PPM_TO_UG_M3:.1f})")
        return da * O3_PPM_TO_UG_M3
    elif "ug/m" in units_lower or "µg/m" in units_lower or "\u03bcg/m" in units_lower:
        logger.info("O3 already in µg/m³, no conversion needed")
        return da
    else:
        # Unknown units – assume ppb (AQM default) and log a warning
        logger.warning(
            f"Unknown O3 units '{units}'. Assuming ppb and converting to µg/m³."
        )
        return da * O3_PPB_TO_UG_M3


def download_grib2_file(url, dest_path):
    """Download a GRIB2 file from NOMADS.

    Args:
        url: HTTPS URL for the GRIB2 file.
        dest_path: Local file path to save the downloaded file.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    try:
        logger.info(f"Downloading: {url}")
        with urlopen(url, timeout=120) as response:
            data = response.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        logger.info(f"Downloaded {len(data) / 1e6:.1f} MB to {dest_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def read_grib2_variable(grib2_path, short_name_candidates):
    """Read a variable from a GRIB2 file using cfgrib/xarray.

    Tries multiple short name candidates to handle different GRIB2 parameter
    table versions used by different AQM model releases.

    Args:
        grib2_path: Path to the GRIB2 file.
        short_name_candidates: List of cfgrib shortName strings to try.

    Returns:
        tuple: (xarray.DataArray or None, str units or None)
    """
    try:
        ds = xr.open_dataset(
            grib2_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "surface"},
                "indexpath": "",
            },
        )
    except Exception as e:
        logger.warning(f"cfgrib surface-level read failed: {e}. Trying without filter.")
        try:
            ds = xr.open_dataset(
                grib2_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )
        except Exception as e2:
            logger.error(f"Failed to read GRIB2 file {grib2_path}: {e2}")
            return None, None

    # Try known variable short names first, then fall back to any available variable
    for name in short_name_candidates:
        if name in ds:
            da = ds[name]
            units = da.attrs.get("units", "")
            logger.info(f"Found variable '{name}' with units='{units}'")
            return da, units

    # Fallback: use the first non-coordinate data variable
    data_vars = list(ds.data_vars)
    if data_vars:
        name = data_vars[0]
        da = ds[name]
        units = da.attrs.get("units", "")
        logger.info(f"Using fallback variable '{name}' with units='{units}'")
        return da, units

    logger.error(f"No data variables found in {grib2_path}")
    return None, None


def extract_time_series(da, var_name, origintime):
    """Normalize a DataArray to (time, latitude, longitude).

    The cfgrib output may have the time dimension named 'step', 'valid_time',
    or something else depending on how the GRIB2 file was encoded.

    Args:
        da: Input DataArray with shape (..., lat, lon).
        var_name: Human-readable variable name for log messages.
        origintime: Model run time used as a fallback base time.

    Returns:
        xarray.DataArray with dims (time, latitude, longitude), or None on failure.
    """
    if da is None:
        return None

    # Rename spatial dimensions to standard names
    rename_map = {}
    for dim in da.dims:
        if dim.lower() in ("lat", "y"):
            rename_map[dim] = "latitude"
        elif dim.lower() in ("lon", "longitude", "x"):
            rename_map[dim] = "longitude"
    if rename_map:
        da = da.rename(rename_map)

    # Determine time coordinate
    if "valid_time" in da.coords and da["valid_time"].ndim > 0:
        time_coord = da["valid_time"]
        if "step" in da.dims:
            da = da.rename({"step": "time"})
        da = da.assign_coords(time=time_coord)
    elif "step" in da.dims:
        # Convert step offsets to absolute timestamps
        if "time" in da.coords and da["time"].ndim == 0:
            base = da["time"].values
        else:
            base = np.datetime64(origintime.replace(tzinfo=None))
        abs_times = base + da["step"].values
        da = da.rename({"step": "time"}).assign_coords(time=abs_times)

    if "time" not in da.dims:
        logger.warning(f"{var_name}: unexpected dims {da.dims}; wrapping in time dim")
        da = da.expand_dims(dim="time")

    return da.astype(np.float32)


# %% Main execution block
if __name__ == "__main__":
    # %% Setup paths and parameters
    ingestVersion = INGEST_VERSION_STR

    forecast_process_dir = os.getenv(
        "forecast_process_dir", default="/home/ubuntu/Weather/NOAA_AQM"
    )
    forecast_process_path = os.path.join(forecast_process_dir, "NOAA_AQM_Process")
    tmpDIR = os.path.join(forecast_process_dir, "Downloads")

    forecast_path = os.getenv(
        "forecast_path", default="/home/ubuntu/Weather/Prod/NOAA_AQM"
    )

    saveType = os.getenv("save_type", default="Download")
    aws_access_key_id = os.environ.get("AWS_KEY", "")
    aws_secret_access_key = os.environ.get("AWS_SECRET", "")

    s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    # Define the processing chunk size - use HRRR chunk sizes as AQM is CONUS data
    processChunk = CHUNK_SIZES.get("HRRR", 100)

    # Create new directory for processing if it does not exist
    if not os.path.exists(forecast_process_dir):
        os.makedirs(forecast_process_dir)
    else:
        shutil.rmtree(forecast_process_dir)
        os.makedirs(forecast_process_dir)

    if not os.path.exists(tmpDIR):
        os.makedirs(tmpDIR)

    if saveType == "Download":
        if not os.path.exists(os.path.join(forecast_path, ingestVersion)):
            os.makedirs(os.path.join(forecast_path, ingestVersion))

    start_time = time.time()

    # Get the latest model run time
    origintime = get_latest_aqm_run()
    logger.info(f"Latest AQM run time: {origintime}")

    # Check if this is newer than the current file
    if saveType == "S3":
        if s3.exists(forecast_path + "/" + ingestVersion + "/NOAA_AQM.time.pickle"):
            with s3.open(
                forecast_path + "/" + ingestVersion + "/NOAA_AQM.time.pickle", "rb"
            ) as f:
                previous_base_time = pickle.load(f)
            if previous_base_time >= origintime:
                logger.info("No Update to NOAA_AQM, ending")
                sys.exit()
    else:
        if os.path.exists(
            forecast_path + "/" + ingestVersion + "/NOAA_AQM.time.pickle"
        ):
            with open(
                forecast_path + "/" + ingestVersion + "/NOAA_AQM.time.pickle", "rb"
            ) as file:
                previous_base_time = pickle.load(file)
            if previous_base_time >= origintime:
                logger.info("No Update to NOAA_AQM, ending")
                sys.exit()

    # %% Download AQM GRIB2 files from NOMADS
    # Try each possible grid ID until a download succeeds
    pm25_grib_path = None
    o3_grib_path = None

    for grid_id in AQM_GRID_IDS:
        if pm25_grib_path is None:
            pm25_url = build_aqm_url(origintime, "pm25", grid_id)
            pm25_local = os.path.join(tmpDIR, f"aqm_pm25_{grid_id}.grib2")
            if download_grib2_file(pm25_url, pm25_local):
                pm25_grib_path = pm25_local
                logger.info(f"PM2.5 GRIB2 downloaded (grid={grid_id})")

        if o3_grib_path is None:
            o3_url = build_aqm_url(origintime, "o3", grid_id)
            o3_local = os.path.join(tmpDIR, f"aqm_o3_{grid_id}.grib2")
            if download_grib2_file(o3_url, o3_local):
                o3_grib_path = o3_local
                logger.info(f"O3 GRIB2 downloaded (grid={grid_id})")

        if pm25_grib_path and o3_grib_path:
            break

    if pm25_grib_path is None and o3_grib_path is None:
        logger.critical("Failed to download any AQM data from NOMADS. Exiting.")
        sys.exit(1)

    if pm25_grib_path is None:
        logger.warning("PM2.5 GRIB2 download failed; PM2.5 will be NaN.")
    if o3_grib_path is None:
        logger.warning("O3 GRIB2 download failed; O3 will be NaN.")

    # %% Read and process GRIB2 data
    # PM2.5 processing (units: µg/m³)
    pm25_da = None
    if pm25_grib_path:
        pm25_da, pm25_units = read_grib2_variable(
            pm25_grib_path, ["pmtf", "mpm2p5", "PMTF", "pm2p5", "unknownl", "unknown"]
        )
        if pm25_da is not None:
            logger.info(
                f"PM2.5: shape={pm25_da.shape}, units='{pm25_units}', "
                f"range=[{float(pm25_da.min().values):.2f}, "
                f"{float(pm25_da.max().values):.2f}]"
            )

    # O3 processing (convert to µg/m³ if needed)
    o3_da = None
    if o3_grib_path:
        o3_da, o3_units = read_grib2_variable(
            o3_grib_path, ["ozcon", "o3c", "OZCON", "o3", "unknownl", "unknown"]
        )
        if o3_da is not None:
            o3_da = convert_o3_to_ug_m3(o3_da, o3_units)
            logger.info(
                f"O3: shape={o3_da.shape}, "
                f"range=[{float(o3_da.min().values):.2f}, "
                f"{float(o3_da.max().values):.2f}] µg/m³"
            )

    # %% Build a unified time-indexed xarray Dataset
    pm25_ts = extract_time_series(pm25_da, "PM2.5", origintime)
    o3_ts = extract_time_series(o3_da, "O3", origintime)

    # Use whichever successfully-loaded DataArray to define grid coordinates
    reference_da = pm25_ts if pm25_ts is not None else o3_ts
    if reference_da is None:
        logger.critical("No usable AQM data available after processing. Exiting.")
        sys.exit(1)

    time_coord = reference_da["time"]
    lat_coord = reference_da["latitude"]
    lon_coord = reference_da["longitude"]
    fallback_shape = (len(time_coord), len(lat_coord), len(lon_coord))

    def _nan_dataarray(fill=np.nan):
        """Return a NaN-filled DataArray matching the AQM grid."""
        return xr.DataArray(
            np.full(fallback_shape, fill, dtype=np.float32),
            dims=["time", "latitude", "longitude"],
            coords={
                "time": time_coord,
                "latitude": lat_coord,
                "longitude": lon_coord,
            },
        )

    xarray_processed = xr.Dataset(
        coords={"time": time_coord, "latitude": lat_coord, "longitude": lon_coord}
    )

    # PM2.5
    if pm25_ts is not None:
        xarray_processed["cnc_PM2_5"] = pm25_ts
        xarray_processed["cnc_PM2_5"].attrs["units"] = "µg/m³"
        xarray_processed["cnc_PM2_5"].attrs["long_name"] = "PM2.5 concentration"
        xarray_processed["cnc_PM2_5"].attrs["source"] = "NOAA AQM (AQFC)"
    else:
        logger.warning("PM2.5 data unavailable; filling with NaN")
        xarray_processed["cnc_PM2_5"] = _nan_dataarray()

    # O3
    if o3_ts is not None:
        xarray_processed["cnc_O3"] = o3_ts
        xarray_processed["cnc_O3"].attrs["units"] = "µg/m³"
        xarray_processed["cnc_O3"].attrs["long_name"] = "Ozone concentration"
        xarray_processed["cnc_O3"].attrs["source"] = "NOAA AQM (AQFC)"
    else:
        logger.warning("O3 data unavailable; filling with NaN")
        xarray_processed["cnc_O3"] = _nan_dataarray()

    # %% Calculate AQI from PM2.5 and O3
    # NOTE: NOAA AQM provides only PM2.5 and Ozone. The AQI calculated here is
    # limited to these two pollutants; NO2, SO2 and CO are left as NaN. For
    # globally-complete AQI (including all EPA pollutants), combine this data
    # with SILAM concentrations (NO2, SO2, CO) at the API level.
    logger.info(
        "Calculating Air Quality Index (AQI) from PM2.5 and O3 (NO2/SO2/CO unavailable)…"
    )

    pm25_data = xarray_processed["cnc_PM2_5"].values
    o3_data = xarray_processed["cnc_O3"].values
    nan_data = np.full(fallback_shape, np.nan, dtype=np.float32)

    aqi_values = calculate_aqi(
        pm25_data,
        nan_data,  # PM10 not provided by AQM
        o3_data,
        nan_data,  # NO2 not provided by AQM
        nan_data,  # SO2 not provided by AQM
        nan_data,  # CO not provided by AQM
        use_nowcast=True,
    )

    xarray_processed["AQI"] = xr.DataArray(
        aqi_values.astype(np.float32),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": time_coord,
            "latitude": lat_coord,
            "longitude": lon_coord,
        },
        attrs={
            "long_name": "Air Quality Index (PM2.5 and O3 only)",
            "units": "1",
            "method": "EPA NowCast for PM2.5; 8-hour average for O3",
            "note": (
                "AQI is based only on PM2.5 and O3 from NOAA AQM. "
                "For full AQI (including NO2/SO2/CO), combine with SILAM concentrations."
            ),
        },
    )
    logger.info("AQI calculation complete")

    # %% Save the processed data to Zarr
    xarray_processed = xarray_processed.chunk(
        chunks={
            "time": xarray_processed.time.size,
            "latitude": processChunk,
            "longitude": processChunk,
        }
    )

    logger.info(f"Saving processed data to: {forecast_process_path}_.zarr")

    with ProgressBar():
        xarray_processed.to_zarr(
            forecast_process_path + "_.zarr", mode="w", consolidated=False, compute=True
        )
    logger.info("Saved Zarr data to disk.")

    # %% Final output handling and cleanup
    pickle_file_path = os.path.join(forecast_process_dir, "NOAA_AQM.time.pickle")
    with open(pickle_file_path, "wb") as file:
        pickle.dump(origintime, file)

    if saveType == "S3":
        zip_base = os.path.join(forecast_process_dir, "NOAA_AQM.zarr")
        shutil.make_archive(zip_base, "zip", forecast_process_path + "_.zarr")
        zip_path = zip_base + ".zip"

        s3.put_file(
            zip_path,
            os.path.join(forecast_path, ingestVersion, "NOAA_AQM.zarr.zip"),
        )
        s3.put_file(
            pickle_file_path,
            os.path.join(forecast_path, ingestVersion, "NOAA_AQM.time.pickle"),
        )
        logger.info("Uploaded NOAA_AQM zarr zip and time pickle to S3.")
    else:
        shutil.move(
            pickle_file_path,
            os.path.join(forecast_path, ingestVersion, "NOAA_AQM.time.pickle"),
        )
        shutil.copytree(
            forecast_process_path + "_.zarr",
            forecast_path + "/" + ingestVersion + "/NOAA_AQM.zarr",
            dirs_exist_ok=True,
        )
        logger.info(
            f"Saved NOAA_AQM data locally to {forecast_path}/{ingestVersion}/NOAA_AQM.zarr"
        )

    # Cleanup
    try:
        shutil.rmtree(forecast_process_dir)
    except FileNotFoundError:
        logger.debug(
            f"Cleanup directory {forecast_process_dir} not found; nothing to remove."
        )
    except PermissionError as e:
        logger.warning(f"Permission denied removing {forecast_process_dir}: {e}")
    except OSError as e:
        logger.warning(f"OS error while removing {forecast_process_dir}: {e}")

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logger.info("NOAA AQM ingest script finished successfully.")
