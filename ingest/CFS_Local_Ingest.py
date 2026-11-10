"""CFS (CFSv2) ingest for months-out future-day requests.

Downloads the time-series files of NOAA's Climate Forecast System (T126 Gaussian
grid, ~0.94 degree, 6-hourly leads) for the four members of the newest 00Z
cycle, capped at six months. State variables are the four-member mean while all
four members run (~111 days) and member 01 alone after that; the precipitation
probability, its spread and the member bitmask exist only in the four-member
range. Everything is interpolated to an hourly axis and published as a
time-chunked ``CFS.zarr`` (``FINAL_TIME_CHUNKS``) plus ``CFS.time.pickle``. The
store starts at the first lead (run + 6 h); the API reads it only for future
days beyond the GFS store.
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import dask.array as da
import numpy as np
import pandas as pd
import requests
import xarray as xr
import zarr
from herbie import Herbie
from metpy.calc import (
    dewpoint_from_specific_humidity,
    relative_humidity_from_specific_humidity,
)
from metpy.units import units

from API.constants.shared_const import (
    ENSEMBLE_MEMBER_COUNTS,
    ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H,
    FIRST_FORECAST_LEAD_HOURS,
    INGEST_VERSION_STR,
    member_mask_var_names,
)
from API.ingest_grib_utils import run_checked
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    FINAL_SHARD_SIZES,
    FINAL_TIME_CHUNKS,
    configure_herbie_request_timeouts,
    configure_zarr_writes,
    interp_time_take_blend,
    mask_invalid_data,
    open_remote,
    pack_member_mask,
    pad_to_chunk_size,
    positive_int_env,
    publish_ingest_run,
    read_published_run_time,
    validate_grib_stats,
    write_final_zarr,
)

logger = logging.getLogger(__name__)

# %% Model constants
CFS_MEMBER_COUNT = ENSEMBLE_MEMBER_COUNTS["CFS"]
CFS_LEAD_STEP_HOURS = 6
CFS_FIRST_LEAD_HOURS = FIRST_FORECAST_LEAD_HOURS["CFS"]
# Six months of forecast (user requirement): nothing past it is downloaded,
# stored or served. Member 01 of a 00Z cycle runs ~295 days.
CFS_FORECAST_DAYS = 184
CFS_MAX_LEAD_HOURS = CFS_FORECAST_DAYS * 24
CFS_FORECAST_HOURS = range(
    CFS_FIRST_LEAD_HOURS, CFS_MAX_LEAD_HOURS + 1, CFS_LEAD_STEP_HOURS
)
# Member 01 must reach this far or the run is not published: a shorter run means
# a truncated download or an upstream change, not a usable months-out forecast.
CFS_MIN_FORECAST_DAYS = 150
# Herbie selects index lines by regex; one alternative per lead keeps every file
# inside the cap (member 01's files hold ~1,180 leads).
CFS_SEARCH = ":(?:" + "|".join(str(hour) for hour in CFS_FORECAST_HOURS) + ") hour fcst"
# Herbie drops the last index message of a file when it is selected (its end
# byte is unknown), so a member 02-04 file can be one lead short, and the u and
# v halves of wnd10m one lead apart. A member keeps the leads all its variables
# share; a larger gap than this means a damaged file.
CFS_MAX_DROPPED_FINAL_LEADS = 1

# Only 00Z cycles are ingested, once a day: their members 02-04 run ~111 days
# (45 days at 06/12/18Z), so the four-member range does not flip between the two,
# and a months-out forecast gains nothing from 6-hourly refreshes.
CFS_CYCLE_HOUR = 0
CFS_LOOKBACK_DAYS = 3
# Member 01's wnd10m is the last file of a 00Z cycle to be posted (~11:25 UTC)
CFS_LAST_POSTED_VARIABLE = "wnd10m"
CFS_SOURCE_PRIORITY = ["aws", "nomads"]

SECONDS_PER_HOUR = 3600
# Floor on the reduced 2 m specific humidity before deriving dew point (kg/kg)
CFS_MIN_SPECIFIC_HUMIDITY_KG_KG = 1e-6
CFS_HUMIDITY_BLOCK_LEADS = 64

# Time-series files (Herbie variable names), each one variable of one member for
# the whole run, and the wgrib2 NetCDF variables each holds
CFS_SOURCE_VARIABLES = {
    "tmp2m": ("TMP_2maboveground",),
    "q2m": ("SPFH_2maboveground",),
    "pressfc": ("PRES_surface",),
    "prmsl": ("PRMSL_meansealevel",),
    "wnd10m": ("UGRD_10maboveground", "VGRD_10maboveground"),
    "prate": ("PRATE_surface",),
    "tcdcclm": ("TCDC_entireatmosphere_consideredasasinglelayer_",),
    "dswsfc": ("DSWRF_surface",),
}

# Per-hour member bitmask word (pack_member_mask) for the daily probability
MEMBER_MASK_VARS = member_mask_var_names(CFS_MEMBER_COUNT)

# Variables in the published store. The API reader puts the valid times (the
# store's time attribute) in column 0, so zarr_vars[i] is column i + 1 of the
# API's CFS index map. Units as in the GFS store, except PRATE_surface, which is
# in mm/h here (GFS stores kg m-2 s-1).
zarr_vars = (
    "TMP_2maboveground",  # K
    "DPT_2maboveground",  # K
    "RH_2maboveground",  # %
    "PRMSL_meansealevel",  # Pa
    "PRES_surface",  # Pa
    "UGRD_10maboveground",  # m/s
    "VGRD_10maboveground",  # m/s
    "PRATE_surface",  # mm/h, ensemble mean
    "TCDC_entireatmosphere",  # %
    "DSWRF_surface",  # W/m2
    "Precipitation_Prob",  # 0-1, NaN after ensemble_end
    "APCP_StdDev",  # mm/h, NaN after ensemble_end
    *MEMBER_MASK_VARS,  # NaN after ensemble_end
)
MEMBER_MASK_INDICES = [zarr_vars.index(v) for v in MEMBER_MASK_VARS]
# Statistics of the 4 members, only defined while all of them run
ENSEMBLE_ONLY_VARS = ("Precipitation_Prob", "APCP_StdDev", *MEMBER_MASK_VARS)
STATE_VARS = tuple(v for v in zarr_vars if v not in ENSEMBLE_ONLY_VARS)
# PRATE, TCDC and DSWRF are labelled "N hour fcst" in the time-series files but
# are means over the 6 hours ending at the lead (as in the flxf files), so each
# hour takes the lead whose interval contains it, which also keeps 6-hour totals.
# The probability, spread and mask describe the same intervals. Cloud and solar
# are blended linearly instead: a smooth curve reads better than 6-hour steps.
INTERVAL_VARS = ("PRATE_surface", "Precipitation_Prob", "APCP_StdDev")
NEXT_LEAD_INDICES = [zarr_vars.index(v) for v in INTERVAL_VARS] + MEMBER_MASK_INDICES

# %% Paths and configuration
ingest_version = INGEST_VERSION_STR

# When running the docker container, this should be "/build/wgrib2_build/bin/wgrib2 "
wgrib2_path = os.getenv(
    "wgrib2_path", default="/home/ubuntu/wgrib2/wgrib2-3.6.0/build/wgrib2/wgrib2 "
)

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/CFS")
forecast_process_path = os.path.join(forecast_process_dir, "CFS_Process")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/CFS")

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")

# Downloads come from AWS (one GET per file) and each file is converted by its
# download thread, so these threads also bound the concurrent wgrib2 decodes.
CFS_MAX_THREADS = positive_int_env("cfs_download_threads", 4)
CFS_DOWNLOAD_RETRIES = positive_int_env("herbie_download_retries", 5)
CFS_RETRY_SLEEP_SECONDS = positive_int_env("herbie_retry_sleep_seconds", 20)


def cfs_herbie(run_time: pd.Timestamp, variable: str, member: int) -> Herbie:
    """Herbie object for one time-series file (one variable, one member)."""
    return Herbie(
        run_time,
        model="cfs",
        product="time_series",
        variable=variable,
        member=member,
        priority=CFS_SOURCE_PRIORITY,
        save_dir=tmp_dir,
        verbose=False,
    )


def download_and_convert_cfs_file(
    run_time: pd.Timestamp, variable: str, member: int
) -> str:
    """Download one time-series file (leads up to the cap) with bounded retries,
    validate its records and convert it to NetCDF; returns the NetCDF path.

    ``-s -stats`` and ``-netcdf`` run in one wgrib2 pass: decoding the JPEG 2000
    records is the cost (~9 s for a member 01 file), so a separate
    ``check_grib_stats`` pass would double it.
    """
    dataset_name = f"CFS {run_time:%Y%m%d%H} member {member:02d} {variable}"
    for attempt in range(1, CFS_DOWNLOAD_RETRIES + 1):
        try:
            herbie_file = cfs_herbie(run_time, variable, member)
            if herbie_file.grib is None or herbie_file.idx is None:
                raise RuntimeError(f"{dataset_name} is not posted with its index")
            grib_path = herbie_file.download(
                CFS_SEARCH, verbose=False, overwrite=attempt > 1
            )
            if not os.path.isfile(grib_path) or os.path.getsize(grib_path) == 0:
                raise RuntimeError(f"{dataset_name} download is missing or empty")
            break
        except (requests.RequestException, RuntimeError, OSError) as exc:
            if attempt == CFS_DOWNLOAD_RETRIES:
                raise
            sleep_s = CFS_RETRY_SLEEP_SECONDS * attempt
            logger.warning(
                "%s download attempt %d/%d failed (%s). Retrying in %ss",
                dataset_name,
                attempt,
                CFS_DOWNLOAD_RETRIES,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    nc_path = f"{forecast_process_path}_{variable}_m{member:02d}.nc"
    grib_check = run_checked(
        f"{wgrib2_path.strip()} {grib_path} -s -stats -netcdf {nc_path}",
        f"{dataset_name} GRIB validation and conversion",
    )
    if not validate_grib_stats(grib_check):
        raise RuntimeError(f"{dataset_name} failed GRIB validation.")
    os.remove(grib_path)
    return nc_path


def valid_lead_count(
    values: np.ndarray, times: np.ndarray, expected_times: np.ndarray, label: str
) -> int:
    """Number of leads with data in one decoded file, checked to be the first of
    ``expected_times`` (6-hourly from run + 6 h) with no data after them (wgrib2
    leaves a missing record all NaN)."""
    has_data = np.isfinite(values).any(axis=(1, 2))
    lead_count = int(has_data.sum())
    if lead_count == 0 or not has_data[:lead_count].all():
        raise RuntimeError(f"{label}: leads with data are not contiguous from 6 h")
    if lead_count > len(expected_times):
        raise RuntimeError(f"{label}: {lead_count} leads, more than the cap")
    if not np.array_equal(times[:lead_count], expected_times[:lead_count]):
        raise RuntimeError(
            f"{label}: leads {times[:3]}... are not 6-hourly from run + 6 h"
        )
    return lead_count


def reduce_cfs_members(
    member_values: list[np.ndarray], precipitation: bool = False
) -> dict[str, np.ndarray]:
    """Reduce the four CFS members of one variable onto member 01's leads.

    ``member_values`` are ``(lead, y, x)`` arrays, member 01 first; each covers
    a prefix of member 01's leads (members 02-04 of a 00Z cycle stop at ~111
    days). The returned ``"mean"`` is the four-member mean at the leads every
    member has and member 01 alone after them. With ``precipitation`` (rates in
    mm/h), the share of members above ``ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H``
    (``Precipitation_Prob``), the members' standard deviation (``APCP_StdDev``)
    and the member bitmask words are returned as well, NaN after the
    four-member leads: a 0/1 probability or a zero spread from one member would
    be misleading.
    """
    if len(member_values) != CFS_MEMBER_COUNT:
        raise RuntimeError(
            f"CFS has {len(member_values)} members, expected {CFS_MEMBER_COUNT}"
        )
    lead_count = member_values[0].shape[0]
    ensemble_count = min(values.shape[0] for values in member_values)
    if max(values.shape[0] for values in member_values) != lead_count:
        raise RuntimeError("CFS member 01 must cover every other member's leads")

    ensemble = np.stack([values[:ensemble_count] for values in member_values])
    reduced = {"mean": member_values[0].astype(np.float32, copy=True)}
    reduced["mean"][:ensemble_count] = ensemble.mean(axis=0)
    if not precipitation:
        return reduced

    after_ensemble = np.full(
        (lead_count - ensemble_count, *ensemble.shape[2:]), np.nan, np.float32
    )
    wet_members = ensemble > ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H
    reduced["Precipitation_Prob"] = np.concatenate(
        [wet_members.sum(axis=0, dtype=np.float32) / CFS_MEMBER_COUNT, after_ensemble]
    )
    reduced["APCP_StdDev"] = np.concatenate(
        [ensemble.std(axis=0).astype(np.float32), after_ensemble]
    )
    words = pack_member_mask(wet_members, CFS_MEMBER_COUNT, label="CFS")
    for name, word in zip(MEMBER_MASK_VARS, words, strict=True):
        reduced[name] = np.concatenate([word, after_ensemble])
    return reduced


def main() -> None:
    """Run the CFS ingest."""
    logging.basicConfig(level=logging.INFO)
    # Bound Herbie request times and pace NOMADS requests (rate-limited per IP)
    configure_herbie_request_timeouts()
    zarr_store_workers = configure_zarr_writes()
    # wgrib2's OpenMP threads do not speed up the JPEG 2000 decode of these
    # files (9.0 s on 1 thread, 10.9 s on 8 for a member 01 file, at 6x the CPU),
    # so each conversion gets one thread and the files run side by side.
    os.environ["OMP_NUM_THREADS"] = "1"

    process_chunk = CHUNK_SIZES["CFS"]
    final_chunk = FINAL_CHUNK_SIZES["CFS"]
    time_chunk, time_shard = FINAL_TIME_CHUNKS["CFS"]

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

    # %% Determine the model run: the newest 00Z cycle whose last-posted file
    # (member 01 wnd10m) exists with its index; one probe per candidate cycle
    today_cycle = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
    today_cycle += pd.Timedelta(hours=CFS_CYCLE_HOUR)
    base_time = None
    for days_back in range(CFS_LOOKBACK_DAYS + 1):
        candidate = today_cycle - pd.Timedelta(days=days_back)
        probe = cfs_herbie(candidate, CFS_LAST_POSTED_VARIABLE, member=1)
        if probe.grib is not None and probe.idx is not None:
            base_time = candidate
            break
        logger.info("CFS %s is not complete yet", candidate)
    if base_time is None:
        raise RuntimeError(
            f"No complete CFS 00Z cycle in the last {CFS_LOOKBACK_DAYS} days"
        )

    # %% Skip the run if this cycle has already been published
    previous_base_time = read_published_run_time(forecast_remote, "CFS.time.pickle")
    if previous_base_time is not None and previous_base_time >= base_time:
        logger.info("No Update to CFS, ending")
        return
    logger.info("Processing CFS run %s", base_time)

    capped_valid_times = (
        base_time + pd.to_timedelta(list(CFS_FORECAST_HOURS), unit="h")
    ).values

    # %% Download, validate and convert every member's files, capped at 6 months
    members = range(1, CFS_MEMBER_COUNT + 1)
    file_keys = [
        (variable, member) for member in members for variable in CFS_SOURCE_VARIABLES
    ]
    with ThreadPoolExecutor(max_workers=CFS_MAX_THREADS) as executor:
        nc_paths = dict(
            zip(
                file_keys,
                executor.map(
                    lambda key: download_and_convert_cfs_file(base_time, *key),
                    file_keys,
                ),
                strict=True,
            )
        )
    logger.info(
        "Downloaded and converted %d CFS files in %.1f s",
        len(nc_paths),
        time.time() - T0,
    )

    # %% Check each member's leads; a member keeps the leads all its variables share
    T1 = time.time()
    member_lead_counts = {}
    gaussian_lat = gaussian_lon = None
    for member in members:
        counts = []
        for variable, nc_variables in CFS_SOURCE_VARIABLES.items():
            with xr.open_dataset(nc_paths[(variable, member)]) as ds:
                if variable != "prmsl":
                    if gaussian_lat is None:
                        gaussian_lat = ds["latitude"].values
                        gaussian_lon = ds["longitude"].values
                    # wgrib2 writes rows south to north (the GFS store's order)
                    if not (
                        np.allclose(ds["latitude"].values, gaussian_lat)
                        and np.allclose(ds["longitude"].values, gaussian_lon)
                    ):
                        raise RuntimeError(
                            f"CFS member {member:02d} {variable} is not on the "
                            "T126 Gaussian grid of the other files"
                        )
                for nc_variable in nc_variables:
                    counts.append(
                        valid_lead_count(
                            ds[nc_variable].values,
                            ds["time"].values,
                            capped_valid_times,
                            f"CFS member {member:02d} {nc_variable}",
                        )
                    )
        if max(counts) - min(counts) > CFS_MAX_DROPPED_FINAL_LEADS:
            raise RuntimeError(
                f"CFS member {member:02d} variables end {max(counts) - min(counts)} "
                "leads apart"
            )
        member_lead_counts[member] = min(counts)
    if gaussian_lat is None or not np.all(np.diff(gaussian_lat) > 0):
        raise RuntimeError("CFS Gaussian latitudes are not south to north")

    lead_count = member_lead_counts[1]
    if lead_count * CFS_LEAD_STEP_HOURS < CFS_MIN_FORECAST_DAYS * 24:
        raise RuntimeError(
            f"CFS member 01 reaches {lead_count * CFS_LEAD_STEP_HOURS} h, short of "
            f"{CFS_MIN_FORECAST_DAYS} days"
        )
    member_lead_counts = {
        member: min(count, lead_count) for member, count in member_lead_counts.items()
    }
    ensemble_lead_count = min(member_lead_counts.values())
    forecast_valid_times = capped_valid_times[:lead_count]
    ensemble_end = forecast_valid_times[ensemble_lead_count - 1]
    logger.info(
        "CFS member leads %s; four-member range to %s, member 01 to %s",
        member_lead_counts,
        ensemble_end,
        forecast_valid_times[-1],
    )

    # %% Reduce the members variable by variable onto one 6-hourly stack
    def read_members(variable: str, nc_variable: str) -> list[np.ndarray]:
        values = []
        for member in members:
            with xr.open_dataset(nc_paths[(variable, member)]) as ds:
                values.append(ds[nc_variable].values[: member_lead_counts[member]])
        return values

    # The 6-hourly stack (13 x 736 leads = 2.8 GB) is a local uncompressed zarr
    # array filled one variable at a time, so it never has to sit in memory
    # next to the members or the hourly write (7.9 GB peak when it was numpy).
    stack_6h = zarr.create_array(
        store=forecast_process_path + "_stack_6h.zarr",
        shape=(len(zarr_vars), lead_count, len(gaussian_lat), len(gaussian_lon)),
        chunks=(1, lead_count, process_chunk, process_chunk),
        dtype="float32",
        fill_value=np.nan,
        compressors=None,
    )
    for variable, nc_variable, store_variable in (
        ("tmp2m", "TMP_2maboveground", "TMP_2maboveground"),
        ("pressfc", "PRES_surface", "PRES_surface"),
        ("wnd10m", "UGRD_10maboveground", "UGRD_10maboveground"),
        ("wnd10m", "VGRD_10maboveground", "VGRD_10maboveground"),
        (
            "tcdcclm",
            "TCDC_entireatmosphere_consideredasasinglelayer_",
            "TCDC_entireatmosphere",
        ),
        ("dswsfc", "DSWRF_surface", "DSWRF_surface"),
    ):
        stack_6h[zarr_vars.index(store_variable)] = reduce_cfs_members(
            read_members(variable, nc_variable)
        )["mean"]

    # PRATE is a mean rate in kg m-2 s-1 over the 6 hours ending at each lead;
    # 1 kg m-2 is 1 mm of water, so seconds per hour gives mm/h. Small negative
    # rates from the spectral model are clipped before the member statistics.
    prate_members = [
        np.maximum(values, 0) * SECONDS_PER_HOUR
        for values in read_members("prate", "PRATE_surface")
    ]
    prate_reduced = reduce_cfs_members(prate_members, precipitation=True)
    del prate_members
    stack_6h[zarr_vars.index("PRATE_surface")] = prate_reduced["mean"]
    for name in ENSEMBLE_ONLY_VARS:
        stack_6h[zarr_vars.index(name)] = prate_reduced[name]
    del prate_reduced

    # PRMSL is on a 1 degree lat/lon grid (360 x 181), unlike every other file.
    # The mean is linear, so the reduced field is regridded once, bilinearly onto
    # the Gaussian grid; a wrap column at 360 = 0 lets 359.06 E interpolate.
    prmsl_mean = reduce_cfs_members(read_members("prmsl", "PRMSL_meansealevel"))["mean"]
    with xr.open_dataset(nc_paths[("prmsl", 1)]) as ds:
        prmsl_lat = ds["latitude"].values
        prmsl_lon = ds["longitude"].values
    prmsl_wrapped = xr.DataArray(
        np.concatenate([prmsl_mean, prmsl_mean[:, :, :1]], axis=2),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": prmsl_lat, "longitude": np.append(prmsl_lon, 360.0)},
    )
    stack_6h[zarr_vars.index("PRMSL_meansealevel")] = prmsl_wrapped.interp(
        latitude=gaussian_lat, longitude=gaussian_lon, method="linear"
    ).values
    del prmsl_mean, prmsl_wrapped

    # Dew point and relative humidity from the reduced 2 m specific humidity,
    # temperature and surface pressure (CFS has no 2 m dew point or humidity).
    # Specific humidity reaches 0 over the Antarctic plateau, where the dew
    # point's log would be -inf; the floor gives a dew point near -80 C there.
    # Computed in blocks of leads: metpy's float64 temporaries on the whole
    # 736-lead field took the run to 6.6 GB.
    specific_humidity = np.maximum(
        reduce_cfs_members(read_members("q2m", "SPFH_2maboveground"))["mean"],
        CFS_MIN_SPECIFIC_HUMIDITY_KG_KG,
    )
    temperature = stack_6h[zarr_vars.index("TMP_2maboveground")]
    surface_pressure = stack_6h[zarr_vars.index("PRES_surface")]
    dewpoint = np.empty_like(temperature)
    relative_humidity = np.empty_like(temperature)
    for first in range(0, lead_count, CFS_HUMIDITY_BLOCK_LEADS):
        block = slice(first, first + CFS_HUMIDITY_BLOCK_LEADS)
        pressure_q = surface_pressure[block] * units.Pa
        specific_humidity_q = specific_humidity[block] * units("kg/kg")
        dewpoint[block] = (
            dewpoint_from_specific_humidity(
                pressure=pressure_q, specific_humidity=specific_humidity_q
            )
            .to("K")
            .magnitude
        )
        # relative_humidity_from_specific_humidity returns a fraction; stored as
        # % like the GFS store, clipped where the member mean supersaturates
        relative_humidity[block] = np.clip(
            relative_humidity_from_specific_humidity(
                pressure=pressure_q,
                temperature=temperature[block] * units.K,
                specific_humidity=specific_humidity_q,
            ).magnitude
            * 100,
            0,
            100,
        )
    stack_6h[zarr_vars.index("DPT_2maboveground")] = dewpoint
    stack_6h[zarr_vars.index("RH_2maboveground")] = relative_humidity
    del specific_humidity, temperature, surface_pressure, dewpoint, relative_humidity
    for nc_path in nc_paths.values():
        os.remove(nc_path)
    logger.info("Reduced the CFS members in %.1f s", time.time() - T1)

    # %% Validate the 6-hourly stack before interpolating
    T2 = time.time()
    stack_dask = mask_invalid_data(
        da.from_zarr(stack_6h).rechunk(
            (len(zarr_vars), lead_count, process_chunk, process_chunk)
        ),
        ignoreAxis=MEMBER_MASK_INDICES,
    )
    state_indices = [zarr_vars.index(v) for v in STATE_VARS]
    ensemble_indices = [zarr_vars.index(v) for v in ENSEMBLE_ONLY_VARS]
    # Every state variable has data at every lead and cell of member 01's range;
    # NaN here is a missing record or a value outside the valid range
    state_nan = da.isnan(stack_dask[state_indices]).any(axis=(1, 2, 3))
    # Ensemble statistics are complete inside the four-member range, NaN after it
    ensemble_nan = da.isnan(stack_dask[ensemble_indices, :ensemble_lead_count]).any(
        axis=(1, 2, 3)
    )
    ensemble_after = da.isfinite(
        stack_dask[ensemble_indices, ensemble_lead_count:]
    ).any(axis=(1, 2, 3))
    state_nan, ensemble_nan, ensemble_after = da.compute(
        state_nan, ensemble_nan, ensemble_after
    )
    for names, failed, problem in (
        (STATE_VARS, state_nan, "has missing or out-of-range values"),
        (ENSEMBLE_ONLY_VARS, ensemble_nan, "is incomplete in the four-member range"),
        (ENSEMBLE_ONLY_VARS, ensemble_after, "has values after the four-member range"),
    ):
        bad = [name for name, flag in zip(names, failed, strict=True) if flag]
        if bad:
            raise RuntimeError(f"CFS run {base_time}: {bad} {problem}")

    # %% Interpolate to hourly and write the final store
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    stacked_timesUnix = (forecast_valid_times - unix_epoch) / one_second
    # Hourly axis from the first lead (run + 6 h) to member 01's last lead
    hourly_timesUnix = np.arange(
        stacked_timesUnix[0], stacked_timesUnix[-1] + 1, SECONDS_PER_HOUR
    )
    if not np.all(np.diff(hourly_timesUnix) == SECONDS_PER_HOUR):
        raise RuntimeError("CFS hourly axis is not strictly hourly")

    hourly_stack = interp_time_take_blend(
        stack_dask,
        stacked_timesUnix=stacked_timesUnix,
        hourly_timesUnix=hourly_timesUnix,
        next_lead_vars=NEXT_LEAD_INDICES,
        dtype="float32",
        fill_value=np.nan,
    )
    cfs_store_name = "CFS.zarr"
    cfs_store_path = os.path.join(forecast_process_dir, cfs_store_name)
    ensemble_end_unix = int((ensemble_end - unix_epoch) / one_second)
    write_final_zarr(
        pad_to_chunk_size(hourly_stack, final_chunk),
        cfs_store_path,
        final_chunk,
        FINAL_SHARD_SIZES["CFS"],
        zarr_store_workers,
        zarr_vars,
        times=hourly_timesUnix,
        time_chunk=time_chunk,
        time_shard=time_shard,
        # Last valid time of the four-member mean and statistics, Unix seconds
        extra_attributes={"ensemble_end": ensemble_end_unix},
    )
    del stack_dask, hourly_stack
    logger.info("Interpolated and wrote %s in %.1f s", cfs_store_name, time.time() - T2)

    # %% Reopen the written store before publishing
    written = zarr.open_array(cfs_store_path, mode="r")
    if (
        written.shape[:2] != (len(zarr_vars), len(hourly_timesUnix))
        or list(written.attrs["variables"]) != list(zarr_vars)
        or len(written.attrs["time"]) != len(hourly_timesUnix)
        or written.attrs["ensemble_end"] != ensemble_end_unix
    ):
        raise RuntimeError(f"{cfs_store_path} does not reopen as written")

    # %% Publish the store, then the run time the API uses to detect it
    publish_ingest_run(
        [cfs_store_name],
        base_time,
        "CFS.time.pickle",
        process_dir=forecast_process_dir,
        remote=forecast_remote,
    )

    # %% Clean up
    shutil.rmtree(forecast_process_dir)
    logger.info("CFS run %s published in %.1f s", base_time, time.time() - T0)


if __name__ == "__main__":
    main()
