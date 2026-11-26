"""Precipitation type selection for daily blocks and GEPS/REPS ensembles."""

import numpy as np
import pytest

from API.api_utils import map_ensemble_precip_rates_to_ptype, select_daily_precip_type
from API.constants.api_const import PRECIP_IDX
from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.model_const import GEPS
from API.constants.shared_const import ENSEMBLE_MEMBER_COUNTS, MISSING_DATA
from API.data_inputs import prepare_data_inputs
from API.hourly.block import build_hourly_block
from API.PirateTextHelper import calculate_precip_text, estimate_snow_height
from tests.helpers.hourly import hourly_block_kwargs


def test_daily_precip_type_follows_dominant_accumulation():
    """Each day's precipType must be the type with the most accumulation, and
    "mixed" only when rain, snow and ice all fall.

    Regression for #582: 13.6 mm rain + 3.6 mm snow was reported as "snow"
    because of threshold overrides; it must be "rain".
    """
    days = [
        # (rain, snow, ice) mm, expected type
        ((13.6, 3.6, 0.0), "rain"),  # #582
        ((3.0, 6.0, 0.0), "snow"),
        ((6.0, 6.0, 0.0), "rain"),  # tie resolves to rain
        ((1.0, 1.0, 1.0), "mixed"),
    ]
    day_sums = np.zeros((len(days), max(DATA_DAY.values()) + 1))
    for row, ((rain, snow, ice), _) in enumerate(days):
        day_sums[row, [DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]]] = (
            rain,
            snow,
            ice,
        )

    result = select_daily_precip_type(
        day_sums, DATA_DAY, np.zeros(len(days), dtype=int), PRECIP_IDX
    )

    assert result.tolist() == [PRECIP_IDX[expected] for _, expected in days]


def test_ensemble_rates_map_to_precip_type():
    """GEPS/REPS have no categorical type, only per-type rates. A single channel
    above the threshold must map to its type (ice pellets -> sleet, freezing
    rain -> ice), noise-level rates to "none", and all-NaN to MISSING_DATA so
    missing data is not reported as dry.

    Several strong channels are mixed, not any one of them. Weak hours with no
    channel over the threshold still take the dominant channel's type, so a light
    ensemble-mean snow is not left for the air-temperature fallback to call sleet.
    """
    nan = np.nan
    result = map_ensemble_precip_rates_to_ptype(
        snow=np.array([4.0, 0.0, 0.0, 0.0, 0.005, nan, 2.0, 0.08, 0.01]),
        ice=np.array([0.0, 3.0, 0.0, 0.0, 0.0, nan, 0.0, 0.02, 0.0]),
        freezing_rain=np.array([0.0, 0.0, 1.0, 0.0, 0.0, nan, 0.0, 0.0, 0.0]),
        rain=np.array([0.0, 0.0, 0.0, 2.0, 0.005, nan, 3.0, 0.0, 0.06]),
        threshold=0.1,
    )

    np.testing.assert_array_equal(
        result,
        [
            PRECIP_IDX["snow"],
            PRECIP_IDX["sleet"],
            PRECIP_IDX["ice"],
            PRECIP_IDX["rain"],
            PRECIP_IDX["none"],
            MISSING_DATA,
            PRECIP_IDX["mixed"],
            PRECIP_IDX["snow"],
            PRECIP_IDX["rain"],
        ],
    )


@pytest.mark.parametrize(
    ("precip_type", "other_accum", "icon_set", "expected"),
    [
        pytest.param(
            "ice", 0.0, "darksky", ("light-freezing-rain", "sleet"), id="ice-darksky"
        ),
        pytest.param(
            "ice",
            0.0,
            "pirate",
            ("light-freezing-rain", "light-freezing-rain"),
            id="ice-pirate",
        ),
        pytest.param(
            "sleet", 0.0, "pirate", ("light-sleet", "light-sleet"), id="sleet"
        ),
        pytest.param(
            "mixed", 1.0, "pirate", ("mixed-precipitation", "mixed"), id="mixed"
        ),
    ],
)
def test_ice_sleet_and_mixed_precip_text(precip_type, other_accum, icon_set, expected):
    """Freezing rain, sleet and mixed precipitation must keep distinct wording
    and icons; Dark Sky has no freezing-rain icon so it falls back to sleet.

    ``other_accum`` is the rain and snow accumulation (mm) next to 2 mm of ice.
    """
    assert (
        calculate_precip_text(
            precipType=precip_type,
            type="hourly",
            rainAccum=other_accum,
            snowAccum=other_accum,
            sleetAccum=2.0,
            pop=1.0,
            icon=icon_set,
            mode="both",
            eff_ice_intensity=0.5,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("hourly_type", "accumulation", "expected"),
    [
        pytest.param("sleet", (0.0, 0.0, 3.0), "sleet", id="ice-pellets-stay-sleet"),
        pytest.param("ice", (0.0, 0.0, 3.0), "ice", id="freezing-rain-stays-ice"),
        # Ice is the most accumulation, but the hours weren't sleet
        pytest.param("rain", (1.0, 0.0, 3.0), "ice", id="other-hours-default-to-ice"),
        # A little sleet doesn't outrank the rain that fell
        pytest.param("sleet", (5.0, 0.0, 1.0), "rain", id="dominant-rain-wins"),
    ],
)
def test_daily_sleet_is_kept_apart_from_freezing_rain(
    hourly_type, accumulation, expected
):
    """Sleet and freezing rain share one accumulation column, so only the hourly
    type tells them apart. It was overwritten before the check that keeps it, and
    every ice-pellet day was reported as freezing rain, a more hazardous forecast.
    """
    day_sums = np.zeros((1, max(DATA_DAY.values()) + 1))
    day_sums[0, [DATA_DAY["rain"], DATA_DAY["snow"], DATA_DAY["ice"]]] = accumulation

    result = select_daily_precip_type(
        day_sums, DATA_DAY, np.array([PRECIP_IDX[hourly_type]]), PRECIP_IDX
    )

    assert result.tolist() == [PRECIP_IDX[expected]]


def test_geps_hours_type_by_member_vote_and_split_mixed_hours():
    """Regression for #851. GEPS/REPS typed an hour by the largest type mean, so two
    members with heavy freezing rain outweighed ten with light snow (Ottawa got 13
    hours of freezing drizzle), and a mixed hour put all its precipitation in the
    ice columns. The member vote must decide the type, a mixed hour must split its
    accumulation and intensity between the types without losing any, its ice must
    count as sleet and freezing rain by the type means, and a store from before the
    votes (NaN) must keep the mean-based type.
    """
    import datetime

    from API.constants.api_const import PRECIP_TYPES
    from API.daily.builder import _frozen_accum_by_type
    from API.minutely.builder import _process_geps_ptype
    from API.PirateDailyText import (
        HourColumns,
        _aggregate_periods,
        _window_accumulation,
    )

    members = ENSEMBLE_MEMBER_COUNTS["GEPS"]
    num_hours = 5
    geps = np.full((num_hours, max(GEPS.values()) + 1), np.nan)
    # (freezing rain, ice pellets, rain, snow) means in mm/h, then votes in members
    hours = [
        # One member with 6 mm/h of freezing rain against ten with light snow
        ((6.0, 0.0, 0.0, 4.0), (1, 0, 0, 10)),
        # Rain and snow members nearly even: mixed although no mean reaches the
        # 0.5 mm/h the mean-based rule needs
        ((0.1, 0.1, 0.3, 0.2), (1, 0, 4, 3)),
        # A store from before the votes types by the largest mean
        ((0.3, 0.0, 0.0, 0.1), None),
        # Mixed rain and freezing rain without temperature or wind (set below)
        ((0.2, 0.2, 0.3, 0.0), (3, 0, 4, 0)),
        # Ice pellets and freezing rain tie on votes; pellets have the larger mean
        ((0.1, 0.4, 0.0, 0.0), (3, 3, 0, 0)),
    ]
    for row, (means, votes) in enumerate(hours):
        means = np.array(means) / (members if row == 0 else 1)
        geps[row, [GEPS[k] for k in ("freezing_rain", "ice", "rain", "snow")]] = means
        geps[row, GEPS["accum"]] = means.sum()
        geps[row, GEPS["prob"]] = 0.5
        if votes is not None:
            geps[
                row,
                [GEPS[f"{k}_votes"] for k in ("freezing_rain", "ice", "rain", "snow")],
            ] = np.array(votes) / members
    inputs = prepare_data_inputs(
        source_list=["geps"],
        nbm_merged=None,
        hrrr_merged=None,
        dwd_mosmix_merged=None,
        ecmwf_merged=None,
        gefs_merged=None,
        gfs_merged=None,
        era5_merged=None,
        extra_vars=[],
        num_hours=num_hours,
        lat=45.42,
        lon=-75.70,
        geps_merged=geps,
    )
    # GEPS has no temperature or wind; above -1 C, so rain stays rain
    temp_c = 0.5
    inputs["temperature_inputs"] = np.full((num_hours, 1), temp_c)
    inputs["wind_inputs"] = np.full((num_hours, 1), 2.0)
    inputs["temperature_inputs"][3] = np.nan
    inputs["wind_inputs"][3] = np.nan
    *_, InterPhour = build_hourly_block(
        **hourly_block_kwargs(num_hours, source_list=["geps"], **inputs)
    )
    hour_types = InterPhour[:, DATA_HOURLY["type"]].tolist()
    assert hour_types == [
        PRECIP_IDX["snow"],
        PRECIP_IDX["mixed"],
        PRECIP_IDX["ice"],
        PRECIP_IDX["mixed"],
        PRECIP_IDX["mixed"],
    ]

    # A mixed hour without a snow share has no snow. The snow density is NaN
    # without temperature or wind, and a zero share turned into NaN snow that
    # reached the hourly and daily totals.
    assert InterPhour[3, DATA_HOURLY["snow"]] == 0
    np.testing.assert_allclose(
        InterPhour[3, [DATA_HOURLY["rain"], DATA_HOURLY["ice"]]], [0.3, 0.4]
    )

    # Minutely takes the vote like the hourly type: a tie goes to the larger
    # mean (argmax picked freezing rain, the first column), and a clear vote
    # beats a larger mean (one heavy freezing-rain member against ten snow)
    InterTminute = np.zeros((num_hours, 5))
    _process_geps_ptype(geps, InterTminute)
    minute_types = np.argmax(InterTminute, axis=1)
    assert minute_types[0] == PRECIP_IDX["snow"]
    assert minute_types[4] == PRECIP_IDX["sleet"]

    # The mixed hour: 0.3 rain, 0.2 snow (liquid), 0.2 ice of its 0.7 mm
    mixed = InterPhour[1]
    np.testing.assert_allclose(
        [mixed[DATA_HOURLY["rain"]], mixed[DATA_HOURLY["ice"]]], [0.3, 0.2]
    )
    np.testing.assert_allclose(
        mixed[DATA_HOURLY["snow"]], estimate_snow_height(np.array([0.2]), temp_c, 2.0)
    )
    np.testing.assert_allclose(
        mixed[DATA_HOURLY["rain_intensity"]]
        + mixed[DATA_HOURLY["snow_intensity"]] / 10
        + mixed[DATA_HOURLY["ice_intensity"]],
        mixed[DATA_HOURLY["intensity"]],
    )
    np.testing.assert_allclose(mixed[DATA_HOURLY["intensity"]], 0.7)

    # Mixed hours' ice by the type means: 0.1 sleet + 0.1 freezing rain, then
    # 0.2 + 0.2 and 0.4 + 0.1; the third hour's 0.4 mm is freezing rain
    expected_sleet, expected_freezing_rain = 0.7, 0.8
    sleet, freezing_rain = _frozen_accum_by_type(
        InterPhour, np.zeros(num_hours, dtype=int), 1
    )
    np.testing.assert_allclose(
        [sleet[0], freezing_rain[0]], [expected_sleet, expected_freezing_rain]
    )

    # The summary text splits it the same way, both in the period text, which
    # put all of a mixed hour's ice in sleet, and in the expected amount
    start = int(datetime.datetime(2026, 11, 26, 12, tzinfo=datetime.UTC).timestamp())
    type_names = {index: PRECIP_TYPES[name] for name, index in PRECIP_IDX.items()}
    columns = HourColumns.from_interphour(
        InterPhour,
        np.arange(start, start + num_hours * 3600, 3600),
        [type_names[hour_type] for hour_type in hour_types],
    )
    window = columns.window(0, num_hours)
    period_stats = _aggregate_periods(window, datetime.UTC, "daily", 0.0)[1]
    np.testing.assert_allclose(
        [
            sum(period["sleet_accum"] for period in period_stats),
            sum(period["frzrain_accum"] for period in period_stats),
        ],
        [expected_sleet, expected_freezing_rain],
    )
    *_, window_freezing_rain, window_sleet = _window_accumulation(window)
    np.testing.assert_allclose(
        [window_sleet, window_freezing_rain], [expected_sleet, expected_freezing_rain]
    )


def _rain_and_frozen_day(rain_hours, rain_mm, frozen_type, frozen_hours, frozen_mm):
    """24 hourly rows (one day): rain hours, then hours of a frozen type, each
    putting down the given amount (mm), and the day's summed accumulation."""
    InterPhour = np.zeros((24, max(DATA_HOURLY.values()) + 1))
    rain_rows = slice(0, rain_hours)
    frozen_rows = slice(rain_hours, rain_hours + frozen_hours)
    InterPhour[rain_rows, DATA_HOURLY["type"]] = PRECIP_IDX["rain"]
    InterPhour[rain_rows, DATA_HOURLY["rain"]] = rain_mm
    InterPhour[frozen_rows, DATA_HOURLY["type"]] = PRECIP_IDX[frozen_type]
    InterPhour[frozen_rows, DATA_HOURLY["ice"]] = frozen_mm

    day_sum = np.zeros((1, max(DATA_DAY.values()) + 1))
    day_sum[0, DATA_DAY["rain"]] = InterPhour[:, DATA_HOURLY["rain"]].sum()
    day_sum[0, DATA_DAY["ice"]] = InterPhour[:, DATA_HOURLY["ice"]].sum()
    return InterPhour, day_sum


@pytest.mark.parametrize(
    ("rain_hours", "rain_mm", "frozen_type", "frozen_hours", "frozen_mm", "expected"),
    [
        # Mostly rain with over 1 mm of sleet: the significant-ice override
        # used to make this freezing rain
        pytest.param(14, 0.4, "sleet", 6, 0.3, "sleet", id="rain-dominant-sleet"),
        # Sleet out-accumulates rain, but rain hours are the most common type
        pytest.param(14, 0.1, "sleet", 6, 0.6, "sleet", id="sleet-dominant-rain-hours"),
        # The same days with freezing rain stay freezing rain
        pytest.param(14, 0.4, "ice", 6, 0.3, "ice", id="rain-dominant-freezing-rain"),
        pytest.param(14, 0.1, "ice", 6, 0.6, "ice", id="freezing-rain-dominant"),
        # Under 1 mm of ice doesn't take over a rainy day
        pytest.param(14, 0.4, "sleet", 2, 0.4, "rain", id="little-sleet-stays-rain"),
    ],
)
def test_daily_frozen_type_follows_hourly_types(
    rain_hours, rain_mm, frozen_type, frozen_hours, frozen_mm, expected
):
    """A day with rain and sleet hours was reported as freezing rain (and the
    weekly summary as freezing drizzle): the ice accumulation can't tell sleet
    from freezing rain, and the day's most common hourly type was rain. The
    frozen type must come from the hours that put down the ice.
    """
    import logging

    from API.daily.builder import _select_precip_type

    InterPhour, day_sum = _rain_and_frozen_day(
        rain_hours, rain_mm, frozen_type, frozen_hours, frozen_mm
    )

    result = _select_precip_type(
        day_sum,
        np.array([float(PRECIP_IDX["rain"])]),
        InterPhour,
        np.zeros(24, dtype=int),
        1,
        logging.getLogger(__name__),
        "test",
    )

    assert result.tolist() == [PRECIP_IDX[expected]]
