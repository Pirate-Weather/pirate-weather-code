"""Daily precipitation probability from the ensemble member bitmasks."""

from unittest.mock import MagicMock

import numpy as np

from API.constants.forecast_const import DATA_DAY, DATA_HOURLY
from API.constants.model_const import ECMWF, GEFS
from API.constants.shared_const import (
    ENSEMBLE_MEMBER_COUNTS,
    MEMBER_MASK_BITS_PER_WORD,
)
from API.daily.builder import _member_precip_share, build_daily_section
from tests.helpers.model_arrays import hourly_inputs

HOURS = 24
HOUR_SECONDS = 3600


def _pack(wet):
    """(member, HOURS) boolean -> (HOURS, words) float words, as the ingests
    store them (pack_member_mask; not imported, as the API test environment has
    no ingest dependencies). Bit m of word w is member 24 w + m."""
    member_count = wet.shape[0]
    words = []
    for first in range(0, member_count, MEMBER_MASK_BITS_PER_WORD):
        bits = wet[first : first + MEMBER_MASK_BITS_PER_WORD]
        weights = 2.0 ** np.arange(bits.shape[0])
        words.append(weights @ bits)
    return np.column_stack(words)


def _mask_words(member_count, wet_hours_by_member):
    """(HOURS, words) float mask words with each member wet at its listed hours."""
    wet = np.zeros((member_count, HOURS), dtype=bool)
    for member, hours in wet_hours_by_member.items():
        wet[member, hours] = True
    return _pack(wet)


def _gefs_merged(words):
    merged = np.full((HOURS, max(GEFS.values()) + 1), np.nan)
    merged[:, 0] = np.arange(HOURS) * HOUR_SECONDS
    merged[:, [GEFS["member_mask_0"], GEFS["member_mask_1"]]] = words
    return merged


def _build_one_day(InterPhour, **kwargs):
    day_index = np.zeros(HOURS, dtype=int)
    defaults = {
        "InterPhour": InterPhour,
        "hourlyDayIndex": day_index,
        "hourlyDay6amIndex": day_index,
        "hourlyHalfDayIndex": day_index,
        "hourlyHalfNightIndex": day_index,
        "hourlyHighIndex": day_index,
        "hourlyLowIndex": day_index,
        "daily_days": 1,
        "prepAccumUnit": 1.0,
        "prepIntensityUnit": 1.0,
        "windUnit": 1.0,
        "visUnits": 1.0,
        "tempUnits": 1.0,
        "extraVars": [],
        "summaryText": False,
        "translation": MagicMock(),
        "is_all_night": False,
        "is_all_day": False,
        "tz_name": "UTC",
        "icon": "default",
        "unitSystem": "si",
        "version": 2,
        "timeMachine": False,
        "tmExtra": False,
        "day_array_grib": np.array([0]),
        "day_array_6am_grib": np.array([6 * HOUR_SECONDS, 30 * HOUR_SECONDS]),
        "day_array_6pm_grib": np.array([18 * HOUR_SECONDS]),
        "InterSday": np.zeros((1, max(DATA_DAY.values()) + 1)),
        "hourList_si": [
            {"time": i * HOUR_SECONDS, "icon": "clear-day", "summary": "Clear"}
            for i in range(HOURS + 48)
        ],
        "pTypeMap": np.array(["none", "rain", "snow", "sleet", "ice"]),
        "pTextMap": np.array(["None", "Rain", "Snow", "Sleet", "Ice"]),
        "logger": MagicMock(),
        "loc_tag": "test_loc",
    }
    return build_daily_section(**(defaults | kwargs))


def test_daily_pop_counts_members_wet_at_any_hour():
    """The daily probability was the maximum of the hourly ones, which
    understates rain whose timing the members disagree on (discussion #773:
    hourly 10/10/10/0/10/40/10 % is not a 40 % day). Three members each wet at
    a different hour are a 3/30 day, not a 1/30 one; and today's elapsed hours
    must drop out, as they do from the hourly probability, or a morning shower
    keeps raising the afternoon's chance. The whole daily section runs so the
    field, the icon window and the day/night halves are what is checked.
    """
    members = ENSEMBLE_MEMBER_COUNTS["GEFS"]
    words = _mask_words(members, {0: [3], 1: [9], 2: [15]})
    inputs = hourly_inputs(
        lat=40.0, lon=-75.0, source_list=["gefs"], gefs=_gefs_merged(words)
    )
    assert [source for source, _, _ in inputs["precip_member_masks"]] == ["gefs"]

    InterPhour = np.zeros((HOURS, max(DATA_HOURLY.values()) + 1))
    InterPhour[:, DATA_HOURLY["time"]] = np.arange(HOURS) * HOUR_SECONDS
    InterPhour[[3, 9, 15], DATA_HOURLY["prob"]] = 1 / members

    whole_day = _build_one_day(
        InterPhour, precip_member_masks=inputs["precip_member_masks"]
    )
    assert whole_day.day_list[0]["precipProbability"] == 0.1
    assert whole_day.day_night_list[0]["precipProbability"] == 0.1

    # From 06:00 the 03:00 member no longer counts
    from_six = _build_one_day(
        InterPhour,
        precip_member_masks=inputs["precip_member_masks"],
        current_hour_time=6 * HOUR_SECONDS,
    )
    assert from_six.day_list[0]["precipProbability"] == 0.07

    # A store from before the masks leaves the (map-padded) columns NaN, and
    # Time Machine has no ensembles at all: the hourly maximum stands
    old_store = _gefs_merged(words)
    old_store[:, [GEFS["member_mask_0"], GEFS["member_mask_1"]]] = np.nan
    inputs = hourly_inputs(lat=40.0, lon=-75.0, source_list=["gefs"], gefs=old_store)
    no_masks = _build_one_day(
        InterPhour, precip_member_masks=inputs["precip_member_masks"]
    )
    assert no_masks.day_list[0]["precipProbability"] == 0.03
    no_masks = _build_one_day(InterPhour, precip_member_masks=[])
    assert no_masks.day_list[0]["precipProbability"] == 0.03


def test_daily_pop_uses_one_fully_covering_ensemble():
    """Ensembles have different member counts, so ORing one source's words
    into another's would count phantom members, and a source whose forecast
    ends inside the window (REPS at 72 h, a store's last day) would understate
    it. The first source in priority order that covers every hour wins; only
    when none does is the source covering the most hours used.
    """
    gefs_members = ENSEMBLE_MEMBER_COUNTS["GEFS"]
    ecmwf_members = ENSEMBLE_MEMBER_COUNTS["ECMWF_IFS"]
    gefs_words = _mask_words(gefs_members, {m: [m] for m in range(3)})
    # ECMWF: ten members wet at hour 0, then no data from noon on
    ecmwf_wet = (np.arange(ecmwf_members) < 10)[:, np.newaxis] & (
        np.arange(HOURS) == 0
    )[np.newaxis, :]
    ecmwf_words = _pack(ecmwf_wet)
    ecmwf_words[12:] = np.nan
    day_index = np.zeros(HOURS, dtype=int)
    masks = [
        ("ecmwf", ecmwf_words, ecmwf_members),
        ("gefs", gefs_words, gefs_members),
    ]

    share = _member_precip_share(masks, [day_index], 1, 0)[0]
    assert share.tolist() == [3 / gefs_members]

    # A half-day window ECMWF does cover goes to ECMWF, with its own count
    morning = np.where(np.arange(HOURS) < 12, 0, -1)
    share = _member_precip_share(masks, [morning], 1, 0)[0]
    assert share.tolist() == [10 / ecmwf_members]

    # No full coverage: the source covering the most hours, never a mix
    gefs_words[6:] = np.nan
    share = _member_precip_share(masks, [day_index], 1, 0)[0]
    assert share.tolist() == [10 / ecmwf_members]

    # No coverage at all is NaN, so the hourly maximum stands
    share = _member_precip_share(masks, [day_index], 1, first_forecast_row=HOURS)[0]
    assert np.isnan(share).all()


def test_ai_ensemble_masks_reach_the_daily_pop():
    """Under include=aimodels AIGEFS stands in for GEFS and AIFS for ECMWF, so
    their mask words must be copied into the borrowed columns; otherwise the
    daily probability silently drops back to the hourly maximum in AI mode.
    """
    from API.constants.model_const import AIGEFS, ECMWF_AIFS
    from API.forecast_sources import _merge_aifs_as_ecmwf, _merge_aigefs_as_gefs

    base = 0.0
    aigefs = np.full((HOURS, max(AIGEFS.values()) + 1), 1.0)
    aigefs[:, 0] = np.arange(HOURS) * HOUR_SECONDS
    aigefs[:, [AIGEFS["member_mask_0"], AIGEFS["member_mask_1"]]] = [5.0, 6.0]
    as_gefs = _merge_aigefs_as_gefs(aigefs, base, HOURS)
    assert as_gefs[0, [GEFS["member_mask_0"], GEFS["member_mask_1"]]].tolist() == [
        5.0,
        6.0,
    ]

    aifs = np.full((HOURS, max(ECMWF_AIFS.values()) + 1), 1.0)
    aifs[:, 0] = np.arange(HOURS) * HOUR_SECONDS
    mask_cols = [ECMWF_AIFS[f"member_mask_{w}"] for w in range(3)]
    aifs[:, mask_cols] = [7.0, 8.0, 9.0]
    as_ecmwf = _merge_aifs_as_ecmwf(aifs, base, HOURS)
    assert as_ecmwf[0, [ECMWF[f"member_mask_{w}"] for w in range(3)]].tolist() == [
        7.0,
        8.0,
        9.0,
    ]
