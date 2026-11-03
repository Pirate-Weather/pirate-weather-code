"""RRFS North American 6-hourly extended (18-84 h) ingest.

Downloads the 2D fields (2dfld product) of the newest 6-hourly RRFS cycle on the
North American domain (13 km rotated latitude-longitude grid, 1127 x 683; only
the 00/06/12/18 UTC cycles run to 84 h), hourly leads 18-83 h, rotates the winds
to earth-relative and publishes ``RRFS_NA_6H.zarr`` and
``RRFS_NA_6H.time.pickle``. The store starts at lead 18 h (run + 18 h); leads
0-17 come from the hourly RRFS_NA ingest.

Alexander Rey, September 2026
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
import xarray as xr
from herbie.fast import Herbie_latest

from API.constants.shared_const import INGEST_VERSION_STR, KELVIN_TO_CELSIUS
from API.ingest_grib_utils import (
    cat_gribs,
    download_and_validate_subset,
    quote_path,
    run_checked,
)
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    FORECAST_LEAD_RANGES,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    mask_invalid_data,
    mask_invalid_refc,
    open_remote,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    rotated_latlon_grid_geometry,
    write_final_zarr,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")

logger = logging.getLogger(__name__)

# %% Model constants
RRFS_NA_FORECAST_HOURS = FORECAST_LEAD_RANGES["RRFS_NA_6H"]
RRFS_SOURCE_PRIORITY = ["aws", "nomads"]
# Herbie's RRFS template selects the North American domain with domain="na"
RRFS_NA_DOMAIN = "na"
# The NA 2D fields are on a rotated latitude-longitude grid (GRIB grid template
# 1, as wgrib2 -grid reports it): rotated south pole at -35 N, 247 E with no
# extra rotation, 1127 x 683 points 0.1083 deg apart, rotated longitudes from
# 299 (-61) deg and rotated latitudes from -36.9303 deg, rows south to north.
RRFS_NA_NORTH_POLE_LAT = 35.0
RRFS_NA_NORTH_POLE_LON = 247.0 - 180.0
RRFS_NA_GRID_SHAPE = (683, 1127)
RRFS_NA_GRID_STEP_DEG = 0.1083
RRFS_NA_FIRST_RLAT = -36.9303
RRFS_NA_FIRST_RLON = -61.0
# Largest allowed gap (deg) between the grid positions recomputed from these
# constants and the ones wgrib2 writes; they agree to ~1e-11 deg
RRFS_NA_GRID_POSITION_TOLERANCE_DEG = 1e-4

# Herbie match strings for the 2dfld product
RRFS_MATCH_STRINGS = (
    ":((DPT|TMP|APTMP|RH):2 m above ground:)"
    "|:((CRAIN|CICEP|CSNOW|CFRZR|PRATE|VIS|GUST|DSWRF|CAPE|CIN):surface:.*hour fcst)"
    "|(:(UGRD|VGRD):10 m above ground:.*hour fcst)"
    # RRFS labels the column cloud cover "entire atmosphere (considered as a
    # single layer)" (HRRR: "entire atmosphere"); the instantaneous field, not
    # the (N-1)-N h average.
    r"|(:TCDC:entire atmosphere \(considered as a single layer\):\d+ hour fcst:)"
    # Like HRRR, each file holds a 0-N h run total and an (N-1)-N h bucket. This
    # selects the one-hour bucket (17-18 ... 82-83 h), which is already an hourly
    # total; differencing run totals would leave the first lead (18 h) as a
    # 0-18 h total.
    "|(:APCP:surface:[1-9][0-9]*-[0-9]+ hour acc)"
    # RRFS carries several aerosol MASSDEN fields at 8 m; smoke is the
    # particulate organic matter one.
    r"|(:MASSDEN:8 m above ground:\d+ hour fcst:aerosol=Particulate organic matter dry:)"
    # RRFS has no MAPS sea-level pressure (HRRR's MSLMA); MSLET is the
    # Eta-reduced sea-level pressure, the nearest equivalent.
    "|(:MSLET:mean sea level:)"
    "|(:REFD:1000 m above ground:)"
    "|(:LFTX:500-1000 mb:)"
)

# GRIB variables outside the generic wgrib2 stats range for valid reasons: CIN
# is negative by convention and routinely below the range minimum (-100 J/kg).
RRFS_EXCLUDED_STATS_VARIABLES = ["CIN"]

# Smoke (MASSDEN) is in kg/m^3; scale to ug/m^3 so float32 storage keeps precision
MASSDEN_KG_TO_UG = 1e9

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so zarr_vars[i] is column i + 1 of the
# index map, shared with the hourly RRFS_NA store so the API can concatenate
# the two.
zarr_vars = (
    "VIS_surface",
    "GUST_surface",
    "MSLET_meansealevel",
    "TMP_2maboveground",
    "DPT_2maboveground",
    "RH_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
    "PRATE_surface",
    "APCP_surface",
    "CSNOW_surface",
    "CICEP_surface",
    "CFRZR_surface",
    "CRAIN_surface",
    "TCDC_entireatmosphere_consideredasasinglelayer_",
    "MASSDEN_8maboveground",
    "REFD_1000maboveground",
    "DSWRF_surface",
    "CAPE_surface",
    "CIN_surface",
    # wgrib2 -netcdf writes the "500-1000 mb" layer as 500M1000mb
    "LFTX_500M1000mb",
)

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/RRFS_NA_6H"
)
forecast_process_path = forecast_process_dir + "/RRFS_NA_6H_Process"
tmp_dir = forecast_process_dir + "/Downloads"

# Must not sit inside forecast_process_dir, which is deleted at the start of a run
forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/RRFS_NA_6H")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

RRFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 3)
RRFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def download_rrfs_na_6h_subset(base_time: pd.Timestamp) -> list[str]:
    """Download and validate the 18-83 h RRFS NA 2dfld subsets with bounded retries."""
    return download_and_validate_subset(
        model="rrfs",
        product="2dfld",
        search=RRFS_MATCH_STRINGS,
        dataset_name="RRFS_NA 6-hourly",
        base_time=base_time,
        forecast_hours=list(RRFS_NA_FORECAST_HOURS),
        priority=RRFS_SOURCE_PRIORITY,
        herbie_save_dir=tmp_dir,
        herbie_download_retries=RRFS_DOWNLOAD_RETRIES,
        herbie_retry_sleep_seconds=RRFS_RETRY_SLEEP_SECONDS,
        herbie_kwargs={"domain": RRFS_NA_DOMAIN},
        download_verbose=False,
        wgrib2_exe=wgrib2_path.strip(),
        excluded_stats_variables=RRFS_EXCLUDED_STATS_VARIABLES,
    )


def main() -> None:
    """Run the 6-hourly RRFS NA ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()

    process_chunk = CHUNK_SIZES["RRFS_NA_6H"]
    final_chunk = FINAL_CHUNK_SIZES["RRFS_NA_6H"]

    # Start each run from an empty processing directory
    shutil.rmtree(forecast_process_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    forecast_remote = open_remote(
        forecast_path + "/" + ingest_version,
        save_type,
        aws_access_key_id,
        aws_secret_access_key,
    )

    T0 = time.time()

    # %% Determine the model run: the newest 6-hourly cycle whose last lead exists
    latest_run = Herbie_latest(
        model="rrfs",
        n=3,
        freq="6h",
        fxx=[RRFS_NA_FORECAST_HOURS[-1]],
        product="2dfld",
        domain=RRFS_NA_DOMAIN,
        verbose=False,
        priority=RRFS_SOURCE_PRIORITY,
    )
    base_time = latest_run.date

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(
        forecast_remote, "RRFS_NA_6H.time.pickle"
    )
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to RRFS_NA_6H, ending")
        return
    logger.info("Processing RRFS_NA_6H run %s", base_time)

    # %% Forecast hours: hourly leads, stored at their valid times
    forecast_valid_times = (
        pd.Timestamp(base_time) + pd.to_timedelta(RRFS_NA_FORECAST_HOURS, unit="h")
    ).values

    # %% Download, merge and convert to NetCDF
    grib_list = download_rrfs_na_6h_subset(base_time)

    merged_grib_path = forecast_process_path + "_wgrib2_merged.grib2"
    merged_nc_path = forecast_process_path + "_wgrib2_merged.nc"
    run_checked(
        f"{cat_gribs(grib_list)} | {quote_path(wgrib2_path.strip())} - "
        f"-grib {quote_path(merged_grib_path)}",
        "RRFS_NA 6-hourly wgrib2 GRIB merge",
    )

    run_checked(
        f"{wgrib2_path} {merged_grib_path} -netcdf {merged_nc_path}",
        "RRFS_NA 6-hourly wgrib2 NetCDF conversion",
    )

    xarray_forecast_merged = xr.open_mfdataset(merged_nc_path)

    if not np.array_equal(xarray_forecast_merged["time"].values, forecast_valid_times):
        raise RuntimeError(
            f"RRFS_NA 6-hourly run {base_time} times "
            f"{xarray_forecast_merged['time'].values} do not match the expected "
            f"forecast leads {RRFS_NA_FORECAST_HOURS}"
        )

    # %% Rotate the winds to earth-relative
    # The NA U/V are relative to the rotated grid. wgrib2 cannot rotate them in
    # place on this grid (see rotated_latlon_grid_geometry), so the rotation is
    # applied here from the grid geometry. The recomputed grid positions must
    # match the ones wgrib2 wrote, which confirms the grid constants (and so the
    # rotation angles) still describe the published product.
    grid_shape = (
        xarray_forecast_merged.sizes["y"],
        xarray_forecast_merged.sizes["x"],
    )
    if grid_shape != RRFS_NA_GRID_SHAPE:
        raise RuntimeError(
            f"RRFS_NA 6-hourly run {base_time} grid is {grid_shape}, expected "
            f"{RRFS_NA_GRID_SHAPE}; the rotated-grid constants need updating"
        )
    rlat = RRFS_NA_FIRST_RLAT + RRFS_NA_GRID_STEP_DEG * np.arange(grid_shape[0])
    rlon = RRFS_NA_FIRST_RLON + RRFS_NA_GRID_STEP_DEG * np.arange(grid_shape[1])
    grid_lat, grid_lon, wind_angle = rotated_latlon_grid_geometry(
        RRFS_NA_NORTH_POLE_LAT, RRFS_NA_NORTH_POLE_LON, rlat, rlon
    )
    lat_error = np.abs(grid_lat - xarray_forecast_merged["latitude"].values)
    lon_error = np.abs(
        (grid_lon - xarray_forecast_merged["longitude"].values + 180.0) % 360.0 - 180.0
    )
    position_error = max(lat_error.max(), lon_error.max())
    if position_error > RRFS_NA_GRID_POSITION_TOLERANCE_DEG:
        raise RuntimeError(
            f"RRFS_NA 6-hourly run {base_time} grid positions differ from the "
            f"rotated-grid constants by {position_error:.6f} deg; not publishing "
            "winds rotated with the wrong angles"
        )

    cos_angle = xr.DataArray(np.cos(wind_angle), dims=("y", "x"))
    sin_angle = xr.DataArray(np.sin(wind_angle), dims=("y", "x"))
    ugrd_grid_relative = xarray_forecast_merged["UGRD_10maboveground"]
    vgrd_grid_relative = xarray_forecast_merged["VGRD_10maboveground"]
    xarray_forecast_merged["UGRD_10maboveground"] = (
        ugrd_grid_relative * cos_angle - vgrd_grid_relative * sin_angle
    )
    xarray_forecast_merged["VGRD_10maboveground"] = (
        ugrd_grid_relative * sin_angle + vgrd_grid_relative * cos_angle
    )

    # %% Correct RRFS-specific quirks
    # APCP is downloaded as the one-hour bucket, so it needs no differencing.
    xarray_forecast_merged["MASSDEN_8maboveground"] = (
        xarray_forecast_merged["MASSDEN_8maboveground"] * MASSDEN_KG_TO_UG
    )

    # RRFS encodes the 500-1000 mb lifted index offset by 273.15 (as if a
    # Kelvin temperature): its values run ~263-301 and, less 273.15, track the
    # 4-layer lifted index RRFS also publishes (checked 2026-11-03). Remove the
    # offset so the store holds the lifted index itself (K difference, about
    # -20 to 30), the range its KEEPBITS precision assumes.
    xarray_forecast_merged["LFTX_500M1000mb"] = (
        xarray_forecast_merged["LFTX_500M1000mb"] - KELVIN_TO_CELSIUS
    )
    # Every other NA field carries a GRIB bitmap that blanks the grid corners
    # outside the model domain; LFTX has none and holds a constant filler
    # (273.2) there, so blank it wherever the bitmapped 2 m temperature is.
    xarray_forecast_merged["LFTX_500M1000mb"] = xarray_forecast_merged[
        "LFTX_500M1000mb"
    ].where(xarray_forecast_merged["TMP_2maboveground"].notnull())

    # Set REFD values below the reflectivity threshold to 0
    xarray_forecast_merged["REFD_1000maboveground"] = mask_invalid_refc(
        xarray_forecast_merged["REFD_1000maboveground"]
    )

    # wgrib2's NetCDF stores GRIB bitmap points (the NA grid's corners outside
    # the model domain) with _FillValue 9.999e20, and da.from_zarr below reads
    # raw values, so drop it to keep those points NaN. mask_invalid_data would
    # catch 9.999e20, but not in its exempt CIN.
    for var in xarray_forecast_merged.data_vars:
        xarray_forecast_merged[var].encoding.pop("_FillValue", None)

    xarray_forecast_merged = xarray_forecast_merged.chunk(
        chunks={
            "time": len(RRFS_NA_FORECAST_HOURS),
            "x": process_chunk,
            "y": process_chunk,
        }
    )
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        xarray_forecast_merged.to_zarr(
            forecast_process_path + "_xr_merged.zarr",
            mode="w",
            consolidated=False,
            chunkmanager_store_kwargs={"num_workers": zarr_store_workers},
        )
    del xarray_forecast_merged

    # The wgrib2 intermediates are large and no longer needed
    os.remove(merged_grib_path)
    os.remove(merged_nc_path)

    # %% Stack the variables into the (variable, time, y, x) layout the API reads
    daskVarArrayList = []
    for dask_var in zarr_vars:
        daskForecastArray = da.from_zarr(
            forecast_process_path + "_xr_merged.zarr",
            component=dask_var,
            inline_array=True,
        )
        daskVarArrayList.append(
            daskForecastArray.rechunk(
                (len(forecast_valid_times), process_chunk, process_chunk)
            ).astype("float32")
        )

    # CIN is negative by convention and routinely below VALID_DATA_MIN
    # (-100 J/kg); masking it would drop the strong caps, so leave it unmasked.
    daskVarArrayListMergeNaN = mask_invalid_data(
        da.stack(daskVarArrayList, axis=0),
        ignoreAxis=[zarr_vars.index("CIN_surface")],
    )

    # Write the stacked array to disk before the final rechunk; doing both in one
    # graph overflows memory
    with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
        daskVarArrayListMergeNaN.to_zarr(
            forecast_process_path + "_stack.zarr", overwrite=True, compute=True
        )
    daskVarArrayStackDisk = da.from_zarr(forecast_process_path + "_stack.zarr")

    # %% Write the final store (hourly already, so no time interpolation)
    daskVarArrayStackDisk_main = pad_to_chunk_size(daskVarArrayStackDisk, final_chunk)

    rrfs_na_6h_store_name = "RRFS_NA_6H.zarr"
    write_final_zarr(
        daskVarArrayStackDisk_main,
        forecast_process_dir + "/" + rrfs_na_6h_store_name,
        final_chunk,
        FINAL_SHARD_SIZES["RRFS_NA_6H"],
        zarr_store_workers,
        zarr_vars,
        times=forecast_valid_times.astype("datetime64[s]").astype("int64"),
    )

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [rrfs_na_6h_store_name],
        base_time,
        "RRFS_NA_6H.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("RRFS_NA_6H run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
