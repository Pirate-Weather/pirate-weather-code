"""Daily calculation and object construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from API.api_utils import _safe_int, select_daily_precip_type, zero_small_values
from API.constants.api_const import (
    PRECIP_IDX,
    PRECIP_PROB_NOISE_THRESHOLD,
    PRECIP_TYPE_DISPLAY,
    PRECIP_TYPES,
    ROUNDING_RULES,
    TEMPERATURE_UNITS_THRESH,
)
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.shared_const import MISSING_DATA
from API.legacy.daily import (
    apply_legacy_half_day_text,
    pick_day_icon_and_summary,
    select_hour_window,
)
from API.PirateDailyText import HourColumns, calculate_day_text


@dataclass
class DailySection:
    day_list: list
    day_list_si: list
    day_icon_list: list
    day_text_list: list
    day_night_list: list


def _get_night_window_end(
    day_array_6am_grib: np.ndarray, idx: int, start_time: int
) -> int:
    """Return the closing 6am boundary for a night window.

    Test fixtures may only provide `daily_days` boundaries rather than the
    next-day 6am marker required by the night summary window.
    """
    if idx + 1 < len(day_array_6am_grib):
        return int(day_array_6am_grib[idx + 1])
    return start_time + (12 * 3600)


def _circular_mean_by_day(directions, day_index, daily_days):
    """Mean direction in degrees (0 to 360) per day, from unit vectors.

    An arithmetic mean of 350 and 10 degrees would report 180: due south for
    winds on either side of north. Hours whose index is outside the forecast days
    are ignored, and a NaN hour makes its day NaN, as in the other daily means.
    bincount handles every day in one pass, which is about 6x cheaper per request
    than a call per day inside _aggregate_stats.
    """
    in_range = (day_index >= 0) & (day_index < daily_days)
    days = day_index[in_range].astype(int)
    radians = np.radians(directions[in_range])
    sin_sum = np.bincount(days, weights=np.sin(radians), minlength=daily_days)
    cos_sum = np.bincount(days, weights=np.cos(radians), minlength=daily_days)
    return np.mod(np.degrees(np.arctan2(sin_sum, cos_sum)), 360.0)


def _most_common_precip_type(hour_types, day_of_row, daily_days):
    """Each day's most common precipitation type, in one pass over the hours.

    A day takes its most common wet type even when dry hours outnumber it, ties
    going to the type seen first; only a day with no wet hours is dry. Hours
    without a type count as dry.
    """
    codes = np.where(np.isnan(hour_types), 0, hour_types).astype(np.intp)
    n_codes = int(codes.max()) + 1
    if n_codes == 1:
        return np.zeros(daily_days)

    counts = np.bincount(
        day_of_row * n_codes + codes, minlength=daily_days * n_codes
    ).reshape(daily_days, n_codes)
    # Later assignments win, so walking the hours backwards leaves each type's
    # first hour. These must be index arrays: numpy flips a negative-stride view.
    backwards = np.arange(codes.size - 1, -1, -1)
    first_hour = np.zeros((daily_days, n_codes), dtype=np.intp)
    first_hour[day_of_row[backwards], codes[backwards]] = backwards

    # Most hours wins; a difference of one hour outweighs any first-seen offset.
    score = counts * (codes.size + 1) - first_hour
    score[counts == 0] = np.iinfo(np.intp).min
    score[:, 0] = np.iinfo(np.intp).min
    return score.argmax(axis=1).astype(float)


def _aggregate_stats(
    InterPhour,
    index_array,
    daily_days,
    calc_mean=False,
    calc_sum=False,
    calc_max=False,
    calc_min=False,
    calc_argmax=False,
    calc_argmin=False,
    calc_precip=False,
):
    """
    Aggregate statistics for daily data.

    The hours are stacked into a (day, hour, variable) array and reduced along
    the hour axis, which replaced about ten numpy calls per day per window. The
    forecast hours ascend, so each day's hours are one run in ``index_array``.

    Args:
        InterPhour: Hourly interpolated data.
        index_array: Array of indices mapping hours to days.
        daily_days: Number of days to process.
        calc_mean: Whether to calculate the mean.
        calc_sum: Whether to calculate the sum.
        calc_max: Whether to calculate the maximum.
        calc_min: Whether to calculate the minimum.
        calc_argmax: Whether to calculate the time of the maximum.
        calc_argmin: Whether to calculate the time of the minimum.
        calc_precip: Whether to calculate precipitation type.

    Returns:
        Tuple of (day, variable) arrays, None where not requested:
        (mean, sum, max, min, argmax, argmin), then the (day,) precip type.
    """
    in_window = (index_array >= 0) & (index_array < daily_days)
    day_of_row = index_array[in_window]
    rows = InterPhour[in_window]

    hours_per_day = np.bincount(day_of_row, minlength=daily_days)
    if hours_per_day.min() == 0:
        raise ValueError(f"A forecast day has no hours: {hours_per_day.tolist()}")
    max_hours = int(hours_per_day.max())
    first_row = np.cumsum(hours_per_day) - hours_per_day

    # Windows differ in length around DST changes. Shorter ones are padded at the
    # end with -0.0, the one value that leaves any sum unchanged, bit for bit.
    is_padded = hours_per_day.min() != max_hours
    if is_padded:
        stacked = np.full((daily_days, max_hours, InterPhour.shape[1]), -0.0)
        stacked[day_of_row, np.arange(day_of_row.size) - first_row[day_of_row]] = rows
        padding = np.arange(max_hours) >= hours_per_day[:, np.newaxis]
    else:
        stacked = rows.reshape(daily_days, max_hours, InterPhour.shape[1])

    res_mean = res_sum = res_max = res_min = res_argmax = res_argmin = None

    # Reducing across rows adds one hour at a time, unlike np.sum along a single
    # column (pairwise), so the sums match the hourly order to the last bit.
    if calc_mean or calc_sum:
        res_sum = np.add.reduce(stacked, axis=1)
    if calc_mean:
        res_mean = res_sum / hours_per_day[:, np.newaxis]

    hour_times = rows[:, DATA_HOURLY["time"]]
    first_row = first_row[:, np.newaxis]

    # A padded hour must never win. NaN still propagates through max and min, and
    # argmax/argmin still stop at a day's first NaN.
    if calc_max or calc_argmax:
        if is_padded:
            stacked[padding] = -np.inf
        if calc_max:
            res_max = np.maximum.reduce(stacked, axis=1)
        if calc_argmax:
            res_argmax = hour_times[first_row + stacked.argmax(axis=1)]
    if calc_min or calc_argmin:
        if is_padded:
            stacked[padding] = np.inf
        if calc_min:
            res_min = np.minimum.reduce(stacked, axis=1)
        if calc_argmin:
            res_argmin = hour_times[first_row + stacked.argmin(axis=1)]

    res_precip = (
        _most_common_precip_type(rows[:, DATA_HOURLY["type"]], day_of_row, daily_days)
        if calc_precip
        else None
    )

    return (
        res_mean,
        res_sum,
        res_max,
        res_min,
        res_argmax,
        res_argmin,
        res_precip,
    )


def _member_precip_share(
    precip_member_masks, index_arrays, daily_days, first_forecast_row
):
    """Share of ensemble members with precipitation at any hour of each window.

    The maximum of the hourly probabilities understates a day's chance of rain
    when the members disagree on timing (hourly 10/10/10/0/10/40/10 % can be
    most members wet at some hour; measured +12 points on GEFS over 24 h), so the
    ingests store per-hour member bitmask words (shared_const) and the words are
    ORed over the window's hours and popcounted here.

    Per day, the first source in priority order whose words cover every hour of
    the window is used; failing that, the source covering the most hours (a
    share over part of the window is still a valid lower bound). Sources are
    never combined, as their member counts differ. Hours before
    ``first_forecast_row`` (today's elapsed hours) are left out, as the hourly
    probability zeroes them.

    ``index_arrays`` holds one day index per window (midnight, 6am, ...); all
    windows are reduced in one pass, since numpy call overhead, not the few
    hundred rows, sets the cost. Returns a (window, day) array, NaN where no
    source covers a day.
    """
    windows = len(index_arrays)
    rows = np.arange(index_arrays[0].size)
    # Each window's days get their own slots, so one reduction covers them all
    in_window = []
    day_of_row = []
    for window, index_array in enumerate(index_arrays):
        selected = (
            (index_array >= 0)
            & (index_array < daily_days)
            & (rows >= first_forecast_row)
        )
        in_window.append(selected)
        day_of_row.append(index_array[selected] + window * daily_days)
    in_window = np.concatenate(in_window)
    day_of_row = np.concatenate(day_of_row)
    slots = windows * daily_days

    # The hours ascend, so each slot's rows are one run; reduceat over the runs
    # (a slot with no rows would otherwise be given its neighbour's first row)
    hours_per_slot = np.bincount(day_of_row, minlength=slots)
    has_rows = hours_per_slot > 0
    run_starts = np.searchsorted(day_of_row, np.flatnonzero(has_rows))
    best_share = np.full(slots, np.nan)
    best_hours = np.zeros(slots)
    for _, words, member_count in precip_member_masks:
        words = np.tile(words, (windows, 1))[in_window]
        covered = ~np.isnan(words).any(axis=1)
        if not covered.any():
            continue
        # NaN (outside the model's range, or padding) must not reach the cast
        bits = np.where(covered[:, np.newaxis], words, 0).astype(np.uint32)
        slot_bits = np.bitwise_or.reduceat(bits, run_starts, axis=0)
        covered_hours = np.bincount(day_of_row, weights=covered, minlength=slots)
        share = np.full(slots, np.nan)
        share[has_rows] = np.bitwise_count(slot_bits).sum(axis=1) / member_count
        take = covered_hours > best_hours
        best_share[take] = share[take]
        best_hours[take] = covered_hours[take]
        if np.array_equal(best_hours, hours_per_slot):
            break  # every slot fully covered; later sources cannot win
    return best_share.reshape(windows, daily_days)


def _raise_prob_to_member_share(max_arr, prob_column, share):
    """Replace a window's maximum hourly probability by the member share where
    that is higher. The share goes through the same noise threshold as the
    hourly probability; a day no ensemble covers keeps the hourly maximum."""
    share = zero_small_values(share.copy(), threshold=PRECIP_PROB_NOISE_THRESHOLD)
    max_arr[:, prob_column] = np.fmax(max_arr[:, prob_column], share)


def _select_precip_type(window_sum, most_common_type, prepAccumUnit, logger, loc_tag):
    """A window's precipitation type from its accumulations, falling back to its
    most common hourly type."""
    try:
        return select_daily_precip_type(
            window_sum, DATA_DAY, most_common_type, PRECIP_IDX, prepAccumUnit
        )
    except Exception:
        logger.exception("select_daily_precip_type error %s", loc_tag)
        return most_common_type


def _conv_temp(arr, tempUnits):
    """
    Convert temperature from Celsius to Fahrenheit if needed.

    Args:
        arr: Temperature array.
        tempUnits: Temperature unit (0 for Fahrenheit, 1 for Celsius).

    Returns:
        Converted temperature array.
    """
    return arr * 9 / 5 + 32 if tempUnits == 0 else arr


def _build_display_data(
    InterPday,
    InterPdayHigh,
    InterPdayLow,
    InterPdayMin,
    InterPdayMax,
    InterPdaySum,
    tempUnits,
    windUnit,
    visUnits,
    prepIntensityUnit,
    prepAccumUnit,
    extraVars,
):
    """
    Build display data for the daily block.

    Args:
        InterPday: Daily mean data.
        InterPdayHigh: Daily high data.
        InterPdayLow: Daily low data.
        InterPdayMin: Daily minimum data.
        InterPdayMax: Daily maximum data.
        InterPdaySum: Daily sum data.
        tempUnits: Temperature unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        prepIntensityUnit: Precipitation intensity unit.
        prepAccumUnit: Precipitation accumulation unit.
        extraVars: Extra variables to include.

    Returns:
        Tuple containing processed display data arrays.
    """
    daily_display_mean = InterPday.copy()
    daily_display_mean[:, DATA_DAY["dew"]] = _conv_temp(
        InterPday[:, DATA_DAY["dew"]], tempUnits
    )
    daily_display_mean[:, DATA_DAY["pressure"]] = (
        InterPday[:, DATA_DAY["pressure"]] / 100
    )
    daily_display_mean[:, DATA_DAY["wind"]] = InterPday[:, DATA_DAY["wind"]] * windUnit
    daily_display_mean[:, DATA_DAY["gust"]] = InterPday[:, DATA_DAY["gust"]] * windUnit
    daily_display_mean[:, DATA_DAY["vis"]] = InterPday[:, DATA_DAY["vis"]] * visUnits
    daily_display_mean[:, DATA_DAY["intensity"]] = (
        InterPday[:, DATA_DAY["intensity"]] * prepIntensityUnit
    )
    daily_display_mean[:, DATA_DAY["rain_intensity"]] = (
        InterPday[:, DATA_DAY["rain_intensity"]] * prepIntensityUnit
    )
    daily_display_mean[:, DATA_DAY["snow_intensity"]] = (
        InterPday[:, DATA_DAY["snow_intensity"]] * prepIntensityUnit
    )
    daily_display_mean[:, DATA_DAY["ice_intensity"]] = (
        InterPday[:, DATA_DAY["ice_intensity"]] * prepIntensityUnit
    )

    daily_display_high = InterPdayHigh.copy()
    daily_display_high[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayHigh[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_high[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayHigh[:, DATA_DAY["apparent"]], tempUnits
    )

    daily_display_low = InterPdayLow.copy()
    daily_display_low[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayLow[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_low[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayLow[:, DATA_DAY["apparent"]], tempUnits
    )

    daily_display_min = InterPdayMin.copy()
    daily_display_min[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayMin[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_min[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayMin[:, DATA_DAY["apparent"]], tempUnits
    )
    daily_display_min[:, DATA_DAY["lifted_index"]] = InterPdayMin[
        :, DATA_DAY["lifted_index"]
    ]
    daily_display_min[:, DATA_DAY["vertical_velocity"]] = InterPdayMin[
        :, DATA_DAY["vertical_velocity"]
    ]

    daily_display_max = InterPdayMax.copy()
    daily_display_max[:, DATA_DAY["temp"]] = _conv_temp(
        InterPdayMax[:, DATA_DAY["temp"]], tempUnits
    )
    daily_display_max[:, DATA_DAY["apparent"]] = _conv_temp(
        InterPdayMax[:, DATA_DAY["apparent"]], tempUnits
    )
    daily_display_max[:, DATA_DAY["intensity"]] = (
        InterPdayMax[:, DATA_DAY["intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["rain_intensity"]] = (
        InterPdayMax[:, DATA_DAY["rain_intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["snow_intensity"]] = (
        InterPdayMax[:, DATA_DAY["snow_intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["ice_intensity"]] = (
        InterPdayMax[:, DATA_DAY["ice_intensity"]] * prepIntensityUnit
    )
    daily_display_max[:, DATA_DAY["convective_inhibition"]] = InterPdayMax[
        :, DATA_DAY["convective_inhibition"]
    ]
    daily_display_max[:, DATA_DAY["k_index"]] = InterPdayMax[:, DATA_DAY["k_index"]]

    daily_display_sum = InterPdaySum.copy()
    daily_display_sum[:, DATA_DAY["rain"]] = (
        InterPdaySum[:, DATA_DAY["rain"]] * prepAccumUnit
    )
    daily_display_sum[:, DATA_DAY["snow"]] = (
        InterPdaySum[:, DATA_DAY["snow"]] * prepAccumUnit
    )
    daily_display_sum[:, DATA_DAY["ice"]] = (
        InterPdaySum[:, DATA_DAY["ice"]] * prepAccumUnit
    )

    if "stationPressure" in extraVars:
        daily_display_mean[:, DATA_DAY["station_pressure"]] = (
            InterPday[:, DATA_DAY["station_pressure"]] / 100
        )

    return (
        daily_display_mean,
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
        daily_display_sum,
    )


def _build_half_display_data(
    half_mean,
    half_max,
    half_sum,
    tempUnits,
    windUnit,
    visUnits,
    prepIntensityUnit,
    prepAccumUnit,
    extraVars,
):
    """Display data for one half-day window (the 6am to 6pm or 6pm to 6am half)."""
    display_mean = half_mean.copy()
    display_mean[:, DATA_HOURLY["dew"]] = _conv_temp(
        half_mean[:, DATA_HOURLY["dew"]], tempUnits
    )
    display_mean[:, DATA_HOURLY["pressure"]] = (
        half_mean[:, DATA_HOURLY["pressure"]] / 100
    )
    display_mean[:, DATA_HOURLY["wind"]] = half_mean[:, DATA_HOURLY["wind"]] * windUnit
    display_mean[:, DATA_HOURLY["gust"]] = half_mean[:, DATA_HOURLY["gust"]] * windUnit
    display_mean[:, DATA_HOURLY["vis"]] = half_mean[:, DATA_HOURLY["vis"]] * visUnits

    display_max = half_max.copy()
    display_sum = half_sum.copy()
    for column in ("intensity", "rain", "snow", "ice"):
        display_mean[:, DATA_HOURLY[column]] = (
            half_mean[:, DATA_HOURLY[column]] * prepIntensityUnit
        )
        display_max[:, DATA_HOURLY[column]] = (
            half_max[:, DATA_HOURLY[column]] * prepIntensityUnit
        )
    for column in ("rain", "snow", "ice"):
        display_sum[:, DATA_HOURLY[column]] = (
            half_sum[:, DATA_HOURLY[column]] * prepAccumUnit
        )

    if "stationPressure" in extraVars:
        display_mean[:, DATA_HOURLY["station_pressure"]] = (
            half_mean[:, DATA_HOURLY["station_pressure"]] / 100
        )

    return display_mean, display_max, display_sum


def _round_columns_by_decimals(arr, rounding_map):
    """Round columns in place, one np.round call per distinct decimal count."""
    cols_by_dec = {}
    for idx_field, decimals in rounding_map.items():
        cols_by_dec.setdefault(decimals, []).append(idx_field)
    for decimals, cols in cols_by_dec.items():
        arr[:, cols] = np.round(arr[:, cols], decimals)


def _apply_rounding(
    daily_display_mean,
    daily_display_high,
    daily_display_low,
    daily_display_min,
    daily_display_max,
    daily_display_sum,
):
    """
    Apply rounding rules to the daily block's display data.

    Args:
        daily_display_mean: Daily mean display data.
        daily_display_high: Daily high display data.
        daily_display_low: Daily low display data.
        daily_display_min: Daily minimum display data.
        daily_display_max: Daily maximum display data.
        daily_display_sum: Daily sum display data.
    """
    daily_mean_rounding_map = {
        DATA_DAY["dew"]: ROUNDING_RULES.get("dewPoint", 2),
        DATA_DAY["pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_DAY["wind"]: ROUNDING_RULES.get("windSpeed", 2),
        DATA_DAY["gust"]: ROUNDING_RULES.get("windGust", 2),
        DATA_DAY["vis"]: ROUNDING_RULES.get("visibility", 2),
        DATA_DAY["intensity"]: ROUNDING_RULES.get("precipIntensity", 4),
        DATA_DAY["rain_intensity"]: ROUNDING_RULES.get("rainIntensity", 4),
        DATA_DAY["snow_intensity"]: ROUNDING_RULES.get("snowIntensity", 4),
        DATA_DAY["ice_intensity"]: ROUNDING_RULES.get("iceIntensity", 4),
        DATA_DAY["prob"]: ROUNDING_RULES.get("precipProbability", 2),
        DATA_DAY["humidity"]: ROUNDING_RULES.get("humidity", 2),
        DATA_DAY["cloud"]: ROUNDING_RULES.get("cloudCover", 2),
        DATA_DAY["uv"]: ROUNDING_RULES.get("uvIndex", 0),
        DATA_DAY["smoke"]: ROUNDING_RULES.get("smoke", 2),
        DATA_DAY["fire"]: ROUNDING_RULES.get("fireIndex", 2),
        DATA_DAY["solar"]: ROUNDING_RULES.get("solar", 2),
        DATA_DAY["station_pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_DAY["cape"]: ROUNDING_RULES.get("cape", 0),
        DATA_DAY["bearing"]: ROUNDING_RULES.get("windBearing", 0),
        DATA_DAY["moon_phase"]: ROUNDING_RULES.get("moonPhase", 2),
    }

    _round_columns_by_decimals(daily_display_mean, daily_mean_rounding_map)

    temp_dec = ROUNDING_RULES.get("temperature", 2)
    app_dec = ROUNDING_RULES.get("apparentTemperature", 2)
    for temp_arr in (
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
    ):
        _round_columns_by_decimals(
            temp_arr, {DATA_DAY["temp"]: temp_dec, DATA_DAY["apparent"]: app_dec}
        )
    daily_display_max[:, DATA_DAY["k_index"]] = np.round(
        daily_display_max[:, DATA_DAY["k_index"]], 0
    )
    daily_display_min[:, DATA_DAY["lifted_index"]] = np.round(
        daily_display_min[:, DATA_DAY["lifted_index"]], 0
    )
    daily_display_max[:, DATA_DAY["convective_inhibition"]] = np.round(
        daily_display_max[:, DATA_DAY["convective_inhibition"]], 0
    )
    daily_display_min[:, DATA_DAY["vertical_velocity"]] = np.round(
        daily_display_min[:, DATA_DAY["vertical_velocity"]], 2
    )

    max_cols_by_dec = {}
    for idx_field in (
        DATA_DAY["intensity"],
        DATA_DAY["rain_intensity"],
        DATA_DAY["snow_intensity"],
        DATA_DAY["ice_intensity"],
        DATA_DAY["uv"],
        DATA_DAY["smoke"],
        DATA_DAY["fire"],
        DATA_DAY["solar"],
        DATA_DAY["prob"],
    ):
        dec = daily_mean_rounding_map.get(idx_field, 2)
        if dec == 0:
            daily_display_max[:, idx_field] = np.round(
                daily_display_max[:, idx_field]
            ).astype(int)
        else:
            max_cols_by_dec.setdefault(dec, []).append(idx_field)
    for dec, cols in max_cols_by_dec.items():
        daily_display_max[:, cols] = np.round(daily_display_max[:, cols], dec)

    accum_dec = ROUNDING_RULES.get("precipAccumulation", 2)
    daily_accum_cols = [DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]]
    daily_display_sum[:, daily_accum_cols] = np.round(
        daily_display_sum[:, daily_accum_cols], accum_dec
    )


def _apply_half_rounding(display_mean, display_max, display_sum):
    """Apply rounding rules to one half-day window's display data."""
    half_rounding_map = {
        DATA_HOURLY["dew"]: ROUNDING_RULES.get("dewPoint", 2),
        DATA_HOURLY["pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_HOURLY["wind"]: ROUNDING_RULES.get("windSpeed", 2),
        DATA_HOURLY["gust"]: ROUNDING_RULES.get("windGust", 2),
        DATA_HOURLY["vis"]: ROUNDING_RULES.get("visibility", 2),
        DATA_HOURLY["intensity"]: ROUNDING_RULES.get("precipIntensity", 4),
        DATA_HOURLY["rain"]: ROUNDING_RULES.get("rainIntensity", 4),
        DATA_HOURLY["snow"]: ROUNDING_RULES.get("snowIntensity", 4),
        DATA_HOURLY["ice"]: ROUNDING_RULES.get("iceIntensity", 4),
        DATA_HOURLY["prob"]: ROUNDING_RULES.get("precipProbability", 2),
        DATA_HOURLY["humidity"]: ROUNDING_RULES.get("humidity", 2),
        DATA_HOURLY["cloud"]: ROUNDING_RULES.get("cloudCover", 2),
        DATA_HOURLY["uv"]: ROUNDING_RULES.get("uvIndex", 0),
        DATA_HOURLY["ozone"]: ROUNDING_RULES.get("ozone", 2),
        DATA_HOURLY["smoke"]: ROUNDING_RULES.get("smoke", 2),
        DATA_HOURLY["fire"]: ROUNDING_RULES.get("fireIndex", 2),
        DATA_HOURLY["solar"]: ROUNDING_RULES.get("solar", 2),
        DATA_HOURLY["station_pressure"]: ROUNDING_RULES.get("pressure", 2),
        DATA_HOURLY["cape"]: ROUNDING_RULES.get("cape", 0),
        DATA_HOURLY["bearing"]: ROUNDING_RULES.get("windBearing", 0),
        DATA_HOURLY["vertical_velocity"]: ROUNDING_RULES.get("verticalVelocity", 2),
        DATA_HOURLY["lifted_index"]: ROUNDING_RULES.get("liftedIndex", 0),
        DATA_HOURLY["convective_inhibition"]: ROUNDING_RULES.get(
            "convectiveInhibition", 0
        ),
        DATA_HOURLY["k_index"]: ROUNDING_RULES.get("kIndex", 0),
    }

    _round_columns_by_decimals(display_mean, half_rounding_map)
    _round_columns_by_decimals(display_max, half_rounding_map)

    half_accum_cols = [DATA_HOURLY["rain"], DATA_HOURLY["snow"], DATA_HOURLY["ice"]]
    display_sum[:, half_accum_cols] = np.round(
        display_sum[:, half_accum_cols], ROUNDING_RULES.get("precipAccumulation", 2)
    )


def _build_half_day_item(
    idx,
    time_val,
    icon,
    text,
    precip_type_val,
    temp_val,
    apparent_val,
    display_mean,
    display_max,
    display_sum,
    interp_mean,
    extraVars,
):
    """
    Build a single half-day item dictionary.

    Args:
        idx: Index of the item.
        time_val: Time value.
        icon: Icon string.
        text: Summary text.
        precip_type_val: Precipitation type.
        temp_val: Temperature value.
        apparent_val: Apparent temperature value.
        display_mean: Mean display data array.
        display_max: Maximum display data array.
        display_sum: Sum display data array.
        interp_mean: Interpolated mean data array.
        extraVars: Extra variables to include.

    Returns:
        Dictionary representing the half-day item.
    """
    liquid_accum = display_sum[idx, DATA_HOURLY["rain"]]
    snow_accum = display_sum[idx, DATA_HOURLY["snow"]]
    ice_accum = display_sum[idx, DATA_HOURLY["ice"]]
    precip_accum = liquid_accum + snow_accum + ice_accum

    wind_bearing_val = interp_mean[idx, DATA_HOURLY["bearing"]]
    wind_bearing = int(wind_bearing_val) if not np.isnan(wind_bearing_val) else 0
    cape_val = interp_mean[idx, DATA_HOURLY["cape"]]
    cape_int = int(cape_val) if not np.isnan(cape_val) else 0

    item = {
        "time": int(time_val),
        "summary": text,
        "icon": icon,
        "precipIntensity": display_mean[idx, DATA_HOURLY["intensity"]],
        "precipIntensityMax": display_max[idx, DATA_HOURLY["intensity"]],
        "rainIntensity": display_mean[idx, DATA_HOURLY["rain"]],
        "rainIntensityMax": display_max[idx, DATA_HOURLY["rain"]],
        "snowIntensity": display_mean[idx, DATA_HOURLY["snow"]],
        "snowIntensityMax": display_max[idx, DATA_HOURLY["snow"]],
        "iceIntensity": display_mean[idx, DATA_HOURLY["ice"]],
        "iceIntensityMax": display_max[idx, DATA_HOURLY["ice"]],
        "precipProbability": display_max[idx, DATA_HOURLY["prob"]],
        "precipAccumulation": precip_accum,
        "precipType": precip_type_val,
        "temperature": temp_val,
        "apparentTemperature": apparent_val,
        "dewPoint": display_mean[idx, DATA_HOURLY["dew"]],
        "humidity": display_mean[idx, DATA_HOURLY["humidity"]],
        "pressure": display_mean[idx, DATA_HOURLY["pressure"]],
        "windSpeed": display_mean[idx, DATA_HOURLY["wind"]],
        "windGust": display_mean[idx, DATA_HOURLY["gust"]],
        "windBearing": wind_bearing,
        "cloudCover": display_mean[idx, DATA_HOURLY["cloud"]],
        "uvIndex": display_mean[idx, DATA_HOURLY["uv"]],
        "visibility": display_mean[idx, DATA_HOURLY["vis"]],
        "ozone": display_mean[idx, DATA_HOURLY["ozone"]],
        "smoke": display_mean[idx, DATA_HOURLY["smoke"]],
        "liquidAccumulation": liquid_accum,
        "snowAccumulation": snow_accum,
        "iceAccumulation": ice_accum,
        "fireIndex": display_mean[idx, DATA_HOURLY["fire"]],
        "solar": display_mean[idx, DATA_HOURLY["solar"]],
        "cape": cape_int,
        "liftedIndex": _safe_int(display_mean[idx, DATA_HOURLY["lifted_index"]]),
        "verticalVelocity": display_mean[idx, DATA_HOURLY["vertical_velocity"]],
        "convectiveInhibition": _safe_int(
            display_mean[idx, DATA_HOURLY["convective_inhibition"]]
        ),
        "kIndex": _safe_int(display_mean[idx, DATA_HOURLY["k_index"]]),
    }

    if "stationPressure" in extraVars:
        item["stationPressure"] = display_mean[idx, DATA_HOURLY["station_pressure"]]

    return item


def build_daily_section(
    *,
    InterPhour: np.ndarray,
    hourlyDayIndex: np.ndarray,
    hourlyDay6amIndex: np.ndarray,
    hourlyHalfDayIndex: np.ndarray,
    hourlyHalfNightIndex: np.ndarray,
    hourlyHighIndex: np.ndarray,
    hourlyLowIndex: np.ndarray,
    daily_days: int,
    prepAccumUnit: float,
    prepIntensityUnit: float,
    windUnit: float,
    visUnits: float,
    tempUnits: float,
    extraVars,
    summaryText: bool,
    translation,
    is_all_night: bool,
    is_all_day: bool,
    tz_name,
    icon: str,
    unitSystem: str,
    version: int,
    timeMachine: bool,
    tmExtra: bool,
    day_array_grib: np.ndarray,
    day_array_6am_grib: np.ndarray,
    day_array_6pm_grib: np.ndarray,
    InterSday: np.ndarray,
    hourList_si: list,
    pTypeMap: np.ndarray,
    pTextMap: np.ndarray,
    logger,
    loc_tag: str,
    log_timing: Callable[[str], None] | None = None,
    aq_inputs=None,
    inc_airqualitydetails: int = 0,
    include_day_night: bool = True,
    hour_columns: HourColumns | None = None,
    current_hour_time: int | None = None,
    precip_member_masks=None,
) -> DailySection:
    """
    Build all daily- and half-day-level objects.

    This function aggregates hourly data into daily and half-day statistics,
    applies unit conversions and rounding, and constructs the final
    DailySection object containing the forecast data.

    Args:
        InterPhour: Hourly interpolated data.
        hourlyDayIndex: Indices for daily aggregation.
        hourlyDay6amIndex: Indices for 6am-to-6am aggregation (daily icon and precipitation type).
        hourlyHalfDayIndex: Indices for the 6am-to-6pm day half.
        hourlyHalfNightIndex: Indices for the 6pm-to-6am night half.
        hourlyHighIndex: Indices for daily high calculation.
        hourlyLowIndex: Indices for daily low calculation.
        daily_days: Number of days.
        prepAccumUnit: Precipitation accumulation unit.
        prepIntensityUnit: Precipitation intensity unit.
        windUnit: Wind speed unit.
        visUnits: Visibility unit.
        tempUnits: Temperature unit.
        extraVars: Extra variables.
        summaryText: Whether to generate summary text.
        translation: Translation function.
        is_all_night: Whether the forecast is for all night.
        is_all_day: Whether the forecast is for all day.
        tz_name: Timezone name.
        icon: Icon set.
        unitSystem: Unit system.
        version: API version.
        timeMachine: Whether this is a time machine request.
        tmExtra: Extra time machine parameters.
        day_array_grib: Daily GRIB array.
        day_array_6am_grib: Daily 6am GRIB array (daily summary and day half start).
        day_array_6pm_grib: Daily 6pm GRIB array (night half start).
        InterSday: Daily source data.
        hourList_si: Hourly list for SI units.
        pTypeMap: Precipitation type map.
        pTextMap: Precipitation text map.
        logger: Logger instance.
        loc_tag: Location tag.
        log_timing: Optional timing logger.
        include_day_night: Build the day/night (half-day) items; when False
            ``day_night_list`` is empty.
        current_hour_time: Unix time of the current hour, where today's summary
            starts. None (Time Machine) keeps the whole 6am to 6am window.
        hour_columns: The forecast's HourColumns (the same hours as
            ``hourList_si``), built once per request; built here from
            ``hourList_si`` when not given.
        precip_member_masks: The ensembles' per-hour member bitmask words in
            priority order (``prepare_data_inputs``), for the daily
            precipitation probability; None or empty keeps the hourly maximum.

    Returns:
        DailySection object containing the daily forecast.
    """
    if log_timing:
        log_timing("Daily start")

    (
        InterPday,
        InterPdaySum,
        InterPdayMax,
        InterPdayMin,
        InterPdayMaxTime,
        InterPdayMinTime,
        _,
    ) = _aggregate_stats(
        InterPhour,
        hourlyDayIndex,
        daily_days,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_min=True,
        calc_argmax=True,
        calc_argmin=True,
    )

    (
        InterPday6am,
        InterPdaySum6am,
        InterPdayMax6am,
        _,
        _,
        _,
        maxPchanceDay,
    ) = _aggregate_stats(
        InterPhour,
        hourlyDay6amIndex,
        daily_days,
        calc_mean=True,
        calc_sum=True,
        calc_max=True,
        calc_precip=True,
    )

    (
        _,
        _,
        InterPdayHigh,
        _,
        InterPdayHighTime,
        _,
        _,
    ) = _aggregate_stats(
        InterPhour,
        hourlyHighIndex,
        daily_days,
        calc_max=True,
        calc_argmax=True,
    )

    (
        _,
        _,
        _,
        InterPdayLow,
        _,
        InterPdayLowTime,
        _,
    ) = _aggregate_stats(
        InterPhour,
        hourlyLowIndex,
        daily_days,
        calc_min=True,
        calc_argmin=True,
    )

    # Bearings wrap at north, so _aggregate_stats' plain means don't apply to them
    hourly_bearing = InterPhour[:, DATA_HOURLY["bearing"]]
    InterPday[:, DATA_DAY["bearing"]] = _circular_mean_by_day(
        hourly_bearing, hourlyDayIndex, daily_days
    )
    InterPday6am[:, DATA_HOURLY["bearing"]] = _circular_mean_by_day(
        hourly_bearing, hourlyDay6amIndex, daily_days
    )

    # Daily probability of precipitation from the ensemble members, for the
    # midnight day (precipProbability) and the 6am-6am day (icon and summary);
    # today's elapsed hours (before the current hour) are left out
    first_forecast_row = (
        0
        if current_hour_time is None
        else int(np.searchsorted(InterPhour[:, DATA_HOURLY["time"]], current_hour_time))
    )
    if precip_member_masks:
        windows = [hourlyDayIndex, hourlyDay6amIndex]
        if include_day_night:
            windows += [hourlyHalfDayIndex, hourlyHalfNightIndex]
        member_share = _member_precip_share(
            precip_member_masks, windows, daily_days, first_forecast_row
        )
        _raise_prob_to_member_share(InterPdayMax, DATA_DAY["prob"], member_share[0])
        _raise_prob_to_member_share(InterPdayMax6am, DATA_DAY["prob"], member_share[1])

    maxPchanceDay = _select_precip_type(
        InterPdaySum, maxPchanceDay, prepAccumUnit, logger, loc_tag
    )

    day_night_list = []

    # The half-day windows only feed the day_night block, which most requests
    # don't return; together they cost about a quarter of the aggregation.
    if include_day_night:
        half_windows = []
        for half, half_index in enumerate((hourlyHalfDayIndex, hourlyHalfNightIndex)):
            half_mean, half_sum, half_max, _, _, _, half_type = _aggregate_stats(
                InterPhour,
                half_index,
                daily_days,
                calc_mean=True,
                calc_sum=True,
                calc_max=True,
                calc_precip=True,
            )
            half_mean[:, DATA_HOURLY["bearing"]] = _circular_mean_by_day(
                hourly_bearing, half_index, daily_days
            )
            if precip_member_masks:
                _raise_prob_to_member_share(
                    half_max, DATA_HOURLY["prob"], member_share[2 + half]
                )
            half_type = _select_precip_type(
                half_sum, half_type, prepAccumUnit, logger, loc_tag
            )
            display_mean, display_max, display_sum = _build_half_display_data(
                half_mean,
                half_max,
                half_sum,
                tempUnits,
                windUnit,
                visUnits,
                prepIntensityUnit,
                prepAccumUnit,
                extraVars,
            )
            _apply_half_rounding(display_mean, display_max, display_sum)
            half_type = half_type.astype(int)
            half_windows.append(
                (
                    half_mean,
                    half_sum,
                    half_max,
                    display_mean,
                    display_max,
                    display_sum,
                    pTypeMap[half_type],
                    pTextMap[half_type],
                )
            )
        (
            interp_half_day_mean,
            interp_half_day_sum,
            interp_half_day_max,
            half_day_display_mean,
            half_day_display_max,
            half_day_display_sum,
            precip_type_half_day,
            precip_text_half_day,
        ) = half_windows[0]
        (
            interp_half_night_mean,
            interp_half_night_sum,
            interp_half_night_max,
            half_night_display_mean,
            half_night_display_max,
            half_night_display_sum,
            precip_type_half_night,
            precip_text_half_night,
        ) = half_windows[1]

    dayList = []
    dayList_si = []
    dayIconList = []
    dayTextList = []

    maxPchanceDay = maxPchanceDay.astype(int)
    PTypeDay = pTypeMap[maxPchanceDay]
    PTextDay = pTextMap[maxPchanceDay]

    if log_timing:
        log_timing("Daily Loop start")

    (
        daily_display_mean,
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
        daily_display_sum,
    ) = _build_display_data(
        InterPday,
        InterPdayHigh,
        InterPdayLow,
        InterPdayMin,
        InterPdayMax,
        InterPdaySum,
        tempUnits,
        windUnit,
        visUnits,
        prepIntensityUnit,
        prepAccumUnit,
        extraVars,
    )

    _apply_rounding(
        daily_display_mean,
        daily_display_high,
        daily_display_low,
        daily_display_min,
        daily_display_max,
        daily_display_sum,
    )

    # The summary windows are slices of the forecast's hour columns. Without
    # summary text no window is summarised, so the columns aren't built and
    # the windows are list slices found by bisection on the sorted times.
    if hour_columns is None and summaryText:
        hour_columns = HourColumns.from_hours(hourList_si)
    if hour_columns is not None:
        window_source = hour_columns
        hour_times_si = None
    else:
        window_source = hourList_si
        hour_times_si = [hour["time"] for hour in hourList_si]

    for idx in range(daily_days):
        # The half-day text costs as much as the daily text; skip it when the
        # day_night block isn't returned.
        if include_day_night:
            day_icon, day_text = pick_day_icon_and_summary(
                max_arr=interp_half_day_max,
                mean_arr=interp_half_day_mean,
                sum_arr=interp_half_day_sum,
                precip_type_arr=precip_type_half_day,
                precip_text_arr=precip_text_half_day,
                idx=idx,
                is_night=is_all_night,
                mode="hourly",
                prep_accum_unit=prepAccumUnit,
                vis_units=visUnits,
                wind_unit=windUnit,
            )

            day_item = _build_half_day_item(
                idx,
                day_array_6am_grib[idx],
                day_icon,
                day_text,
                precip_type_half_day[idx],
                daily_display_high[idx, DATA_DAY["temp"]],
                daily_display_high[idx, DATA_DAY["apparent"]],
                half_day_display_mean,
                half_day_display_max,
                half_day_display_sum,
                interp_half_day_mean,
                extraVars,
            )

            if idx < 8:
                day_text, day_icon = apply_legacy_half_day_text(
                    summary_text=summaryText,
                    translation=translation,
                    hour_list_slice=select_hour_window(
                        window_source,
                        start_time=int(day_array_6am_grib[idx]),
                        end_time=int(day_array_6pm_grib[idx]),
                        hour_times=hour_times_si,
                    ),
                    is_day=not is_all_night,
                    tz_name=tz_name,
                    icon_set=icon,
                    unit_system=unitSystem,
                    fallback_text=day_item["summary"],
                    fallback_icon=day_item["icon"],
                    logger=logger,
                    loc_tag=loc_tag,
                    phase="DAY HALF DAY",
                )
                day_item["summary"] = day_text
                day_item["icon"] = day_icon

            day_night_list.append(day_item)

            day_icon, day_text = pick_day_icon_and_summary(
                max_arr=interp_half_night_max,
                mean_arr=interp_half_night_mean,
                sum_arr=interp_half_night_sum,
                precip_type_arr=precip_type_half_night,
                precip_text_arr=precip_text_half_night,
                idx=idx,
                is_night=not is_all_day,
                mode="hourly",
                prep_accum_unit=prepAccumUnit,
                vis_units=visUnits,
                wind_unit=windUnit,
            )

            day_item = _build_half_day_item(
                idx,
                day_array_6pm_grib[idx],
                day_icon,
                day_text,
                precip_type_half_night[idx],
                daily_display_low[idx, DATA_DAY["temp"]],
                daily_display_low[idx, DATA_DAY["apparent"]],
                half_night_display_mean,
                half_night_display_max,
                half_night_display_sum,
                interp_half_night_mean,
                extraVars,
            )

            if idx < 8:
                night_start_time = int(day_array_6pm_grib[idx])
                day_text, day_icon = apply_legacy_half_day_text(
                    summary_text=summaryText,
                    translation=translation,
                    hour_list_slice=select_hour_window(
                        window_source,
                        start_time=night_start_time,
                        end_time=_get_night_window_end(
                            day_array_6am_grib, idx, night_start_time
                        ),
                        hour_times=hour_times_si,
                    ),
                    is_day=is_all_day,
                    tz_name=tz_name,
                    icon_set=icon,
                    unit_system=unitSystem,
                    fallback_text=day_item["summary"],
                    fallback_icon=day_item["icon"],
                    logger=logger,
                    loc_tag=loc_tag,
                    phase="NIGHT HALF DAY",
                )
                day_item["summary"] = day_text
                day_item["icon"] = day_icon

            day_night_list.append(day_item)

        dayIcon, dayText = pick_day_icon_and_summary(
            max_arr=InterPdayMax6am,
            mean_arr=InterPday6am,
            sum_arr=InterPdaySum6am,
            precip_type_arr=PTypeDay,
            precip_text_arr=PTextDay,
            idx=idx,
            is_night=is_all_night,
            mode="daily",
            prep_accum_unit=prepAccumUnit,
            vis_units=visUnits,
            wind_unit=windUnit,
        )

        if dayIcon == PRECIP_TYPES["none"]:
            if tempUnits == 0:
                tempThresh = TEMPERATURE_UNITS_THRESH["f"]
            else:
                tempThresh = TEMPERATURE_UNITS_THRESH["c"]

            if InterPday[idx, DATA_DAY["temp"]] > tempThresh:
                dayIcon = PRECIP_TYPES["rain"]
                dayText = PRECIP_TYPE_DISPLAY["rain"]
            else:
                dayIcon = PRECIP_TYPES["snow"]
                dayText = PRECIP_TYPE_DISPLAY["snow"]

        temp_high = daily_display_high[idx, DATA_DAY["temp"]]
        temp_low = daily_display_low[idx, DATA_DAY["temp"]]
        temp_min = daily_display_min[idx, DATA_DAY["temp"]]
        temp_max = daily_display_max[idx, DATA_DAY["temp"]]
        apparent_high = daily_display_high[idx, DATA_DAY["apparent"]]
        apparent_low = daily_display_low[idx, DATA_DAY["apparent"]]
        apparent_min = daily_display_min[idx, DATA_DAY["apparent"]]
        apparent_max = daily_display_max[idx, DATA_DAY["apparent"]]
        dew_point = daily_display_mean[idx, DATA_DAY["dew"]]
        pressure_hpa = daily_display_mean[idx, DATA_DAY["pressure"]]
        wind_bearing_day = InterPday[idx, DATA_DAY["bearing"]]
        wind_bearing_day = (
            int(wind_bearing_day) if not np.isnan(wind_bearing_day) else 0
        )

        dayObject = {
            "time": int(day_array_grib[idx]),
            "summary": dayText,
            "icon": dayIcon,
            "dawnTime": int(InterSday[idx, DATA_DAY["dawn"]]),
            "sunriseTime": int(InterSday[idx, DATA_DAY["sunrise"]]),
            "sunsetTime": int(InterSday[idx, DATA_DAY["sunset"]]),
            "duskTime": int(InterSday[idx, DATA_DAY["dusk"]]),
            "moonPhase": InterSday[idx, DATA_DAY["moon_phase"]],
            "precipIntensity": daily_display_mean[idx, DATA_DAY["intensity"]],
            "precipIntensityMax": daily_display_max[idx, DATA_DAY["intensity"]],
            "precipIntensityMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["intensity"]]),
            "precipProbability": daily_display_max[idx, DATA_DAY["prob"]],
            "precipAccumulation": (
                daily_display_sum[idx, DATA_DAY["rain"]]
                + daily_display_sum[idx, DATA_DAY["snow"]]
                + daily_display_sum[idx, DATA_DAY["ice"]]
            ),
            "precipType": PTypeDay[idx],
            "rainIntensity": daily_display_mean[idx, DATA_DAY["rain_intensity"]],
            "rainIntensityMax": daily_display_max[idx, DATA_DAY["rain_intensity"]],
            "snowIntensity": daily_display_mean[idx, DATA_DAY["snow_intensity"]],
            "snowIntensityMax": daily_display_max[idx, DATA_DAY["snow_intensity"]],
            "iceIntensity": daily_display_mean[idx, DATA_DAY["ice_intensity"]],
            "iceIntensityMax": daily_display_max[idx, DATA_DAY["ice_intensity"]],
            "temperatureHigh": temp_high,
            "temperatureHighTime": int(InterPdayHighTime[idx, DATA_DAY["temp"]]),
            "temperatureLow": temp_low,
            "temperatureLowTime": int(InterPdayLowTime[idx, DATA_DAY["temp"]]),
            "apparentTemperatureHigh": apparent_high,
            "apparentTemperatureHighTime": int(
                InterPdayHighTime[idx, DATA_DAY["apparent"]]
            ),
            "apparentTemperatureLow": apparent_low,
            "apparentTemperatureLowTime": int(
                InterPdayLowTime[idx, DATA_DAY["apparent"]]
            ),
            "dewPoint": dew_point,
            "humidity": daily_display_mean[idx, DATA_DAY["humidity"]],
            "pressure": pressure_hpa,
            "windSpeed": daily_display_mean[idx, DATA_DAY["wind"]],
            "windGust": daily_display_mean[idx, DATA_DAY["gust"]],
            "windGustTime": int(InterPdayMaxTime[idx, DATA_DAY["gust"]]),
            "windBearing": wind_bearing_day,
            "cloudCover": daily_display_mean[idx, DATA_DAY["cloud"]],
            "uvIndex": daily_display_max[idx, DATA_DAY["uv"]],
            "uvIndexTime": int(InterPdayMaxTime[idx, DATA_DAY["uv"]]),
            "visibility": daily_display_mean[idx, DATA_DAY["vis"]],
            "temperatureMin": temp_min,
            "temperatureMinTime": int(InterPdayMinTime[idx, DATA_DAY["temp"]]),
            "temperatureMax": temp_max,
            "temperatureMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["temp"]]),
            "apparentTemperatureMin": apparent_min,
            "apparentTemperatureMinTime": int(
                InterPdayMinTime[idx, DATA_DAY["apparent"]]
            ),
            "apparentTemperatureMax": apparent_max,
            "apparentTemperatureMaxTime": int(
                InterPdayMaxTime[idx, DATA_DAY["apparent"]]
            ),
            "smokeMax": daily_display_max[idx, DATA_DAY["smoke"]],
            "smokeMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["smoke"]])
            if not np.isnan(InterPdayMax[idx, DATA_DAY["smoke"]])
            else MISSING_DATA,
            "liquidAccumulation": daily_display_sum[idx, DATA_DAY["rain"]],
            "snowAccumulation": daily_display_sum[idx, DATA_DAY["snow"]],
            "iceAccumulation": daily_display_sum[idx, DATA_DAY["ice"]],
            "fireIndexMax": daily_display_max[idx, DATA_DAY["fire"]],
            "fireIndexMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["fire"]])
            if not np.isnan(InterPdayMax[idx, DATA_DAY["fire"]])
            else MISSING_DATA,
            "solarMax": daily_display_max[idx, DATA_DAY["solar"]],
            "solarMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["solar"]]),
            "capeMax": InterPdayMax[idx, DATA_DAY["cape"]],
            "capeMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["cape"]]),
            "liftedIndexMin": _safe_int(
                daily_display_min[idx, DATA_DAY["lifted_index"]]
            ),
            "liftedIndexMinTime": int(InterPdayMinTime[idx, DATA_DAY["lifted_index"]]),
            "verticalVelocityMin": daily_display_min[
                idx, DATA_DAY["vertical_velocity"]
            ],
            "verticalVelocityMinTime": int(
                InterPdayMinTime[idx, DATA_DAY["vertical_velocity"]]
            ),
            "convectiveInhibitionMax": _safe_int(
                daily_display_max[idx, DATA_DAY["convective_inhibition"]]
            ),
            "convectiveInhibitionMaxTime": int(
                InterPdayMaxTime[idx, DATA_DAY["convective_inhibition"]]
            ),
            "kIndexMax": _safe_int(daily_display_max[idx, DATA_DAY["k_index"]]),
            "kIndexMaxTime": int(InterPdayMaxTime[idx, DATA_DAY["k_index"]]),
        }

        if version >= 2:
            aqi_col = DATA_HOURLY["aqi"]
            aqi_mean = (
                daily_display_mean[idx, aqi_col]
                if aqi_col < daily_display_mean.shape[1]
                else np.nan
            )
            aqi_max = (
                daily_display_max[idx, aqi_col]
                if aqi_col < daily_display_max.shape[1]
                else np.nan
            )
            aqi_min = (
                InterPdayMin[idx, aqi_col]
                if aqi_col < InterPdayMin.shape[1]
                else np.nan
            )
            dayObject["airQualityIndex"] = (
                round(float(aqi_mean)) if not np.isnan(aqi_mean) else np.nan
            )
            dayObject["airQualityIndexMax"] = (
                round(float(aqi_max)) if not np.isnan(aqi_max) else np.nan
            )
            aqi_max_time_raw = (
                InterPdayMaxTime[idx, aqi_col]
                if aqi_col < InterPdayMaxTime.shape[1]
                else np.nan
            )
            dayObject["airQualityIndexMaxTime"] = (
                int(aqi_max_time_raw)
                if not np.isnan(aqi_max_time_raw)
                else MISSING_DATA
            )
            dayObject["airQualityIndexMin"] = (
                round(float(aqi_min)) if not np.isnan(aqi_min) else np.nan
            )
            aqi_min_time_raw = (
                InterPdayMinTime[idx, aqi_col]
                if aqi_col < InterPdayMinTime.shape[1]
                else np.nan
            )
            dayObject["airQualityIndexMinTime"] = (
                int(aqi_min_time_raw)
                if not np.isnan(aqi_min_time_raw)
                else MISSING_DATA
            )

        if "stationPressure" in extraVars:
            dayObject["stationPressure"] = daily_display_mean[
                idx, DATA_DAY["station_pressure"]
            ]

        day_summary_text = None
        try:
            if idx < 8 and summaryText:
                # The daily summary describes 6am to 6am the next day. Today's
                # describes what is left of that: past hours have no precipitation
                # (zeroed in the hourly block), which would misdescribe the day.
                text_start = int(day_array_6am_grib[idx])
                partial_day = (
                    idx == 0
                    and current_hour_time is not None
                    and current_hour_time > text_start
                )
                if partial_day:
                    text_start = int(current_hour_time)
                dayIcon, dayText = calculate_day_text(
                    select_hour_window(
                        window_source,
                        start_time=text_start,
                        end_time=int(day_array_6am_grib[idx + 1]),
                        hour_times=hour_times_si,
                    ),
                    not is_all_night,
                    str(tz_name),
                    "day",
                    icon,
                    unitSystem,
                    partial_day=partial_day,
                )
                day_summary_text = dayText

                dayObject["summary"] = translation.translate(["sentence", dayText])
                dayObject["icon"] = dayIcon
        except Exception:
            logger.exception("DAILY TEXT GEN ERROR %s", loc_tag)

        dayList.append(dayObject)

        dayObject_si = {
            "time": int(day_array_grib[idx]),
            "icon": dayIcon,
            "precipType": PTypeDay[idx],
            "precipProbability": InterPdayMax[idx, DATA_DAY["prob"]],
            "precipIntensity": InterPday[idx, DATA_DAY["intensity"]],
            "snowAccumulation": InterPdaySum[idx, DATA_DAY["snow"]],
            "iceAccumulation": InterPdaySum[idx, DATA_DAY["ice"]],
            "liquidAccumulation": InterPdaySum[idx, DATA_DAY["rain"]],
            "rainIntensityMax": InterPdayMax[idx, DATA_DAY["rain_intensity"]],
            "snowIntensityMax": InterPdayMax[idx, DATA_DAY["snow_intensity"]],
            "iceIntensityMax": InterPdayMax[idx, DATA_DAY["ice_intensity"]],
            "temperatureHigh": InterPdayHigh[idx, DATA_DAY["temp"]],
            "temperatureLow": InterPdayLow[idx, DATA_DAY["temp"]],
            "apparentTemperatureHigh": InterPdayHigh[idx, DATA_DAY["apparent"]],
            "apparentTemperatureLow": InterPdayLow[idx, DATA_DAY["apparent"]],
            "dewPoint": InterPday[idx, DATA_DAY["dew"]],
            "humidity": InterPday[idx, DATA_DAY["humidity"]],
            "windSpeed": InterPday[idx, DATA_DAY["wind"]],
            "cloudCover": InterPday[idx, DATA_DAY["cloud"]],
            "visibility": InterPday[idx, DATA_DAY["vis"]],
            "liftedIndexMin": InterPdayMin[idx, DATA_DAY["lifted_index"]],
            "verticalVelocityMin": daily_display_min[
                idx, DATA_DAY["vertical_velocity"]
            ],
            "convectiveInhibitionMax": InterPdayMax[
                idx, DATA_DAY["convective_inhibition"]
            ],
            "kIndexMax": InterPdayMax[idx, DATA_DAY["k_index"]],
            # The untranslated summary, for the weekly summary's conditions
            "summaryText": day_summary_text,
        }
        dayList_si.append(dayObject_si)

        dayTextList.append(dayObject["summary"])
        dayIconList.append(dayIcon)

    return DailySection(
        day_list=dayList,
        day_list_si=dayList_si,
        day_icon_list=dayIconList,
        day_text_list=dayTextList,
        day_night_list=day_night_list,
    )
