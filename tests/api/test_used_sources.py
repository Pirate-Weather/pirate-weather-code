"""flags.sources lists only the sources that supplied a returned value (#554).

Version 2 responses used to list every model covering the point; the hourly
selection now marks the source of each hour's value, and the final selection
keeps the sources whose hours (or block selections) reach a returned block.
"""

import datetime

import numpy as np
import pytest

from API.constants.model_const import ECMWF, GFS, HRRR, NBM
from API.forecast_sources import (
    USED_SOURCE_COLUMN,
    USED_SOURCE_KEYS,
    SourceMetadata,
    next_update_time,
    select_used_sources,
)
from API.hourly.block import build_hourly_block
from tests.helpers.hourly import hourly_block_kwargs
from tests.helpers.model_arrays import hourly_inputs, model_array

NEW_YORK = (40.7128, -74.0060)


def test_hourly_used_sources_follow_the_per_hour_selection():
    """The hourly mask must record the model each hour's value was taken from,
    not the models that were merely read: NBM and HRRR lead in New York, ECMWF
    only fills the hours they lack, and GFS (below ECMWF) supplies nothing. A
    mask that marked every available model would bring back the #554 list."""
    num_hours = 6
    nbm = model_array(NBM, num_hours, temp=10.0)
    hrrr = model_array(HRRR, num_hours, temp=11.0)
    nbm[4:, NBM["temp"]] = np.nan
    hrrr[4:, HRRR["temp"]] = np.nan
    inputs = hourly_inputs(
        lat=NEW_YORK[0],
        lon=NEW_YORK[1],
        source_list=["nbm", "hrrr_0-18", "hrrr_18-48", "ecmwf_ifs", "gfs"],
        nbm=nbm,
        hrrr=hrrr,
        ecmwf=model_array(ECMWF, num_hours, temp=12.0),
        gfs=model_array(GFS, num_hours, temp=13.0),
    )
    used_hours = np.zeros((num_hours, len(USED_SOURCE_KEYS)), dtype=bool)

    hour_list, *_ = build_hourly_block(
        **hourly_block_kwargs(num_hours, used_hours=used_hours, **inputs)
    )

    # The values themselves are unchanged by the bookkeeping
    assert [hour["temperature"] for hour in hour_list] == [10.0] * 4 + [12.0] * 2
    np.testing.assert_array_equal(
        used_hours[:, USED_SOURCE_COLUMN["nbm"]], [1] * 4 + [0] * 2
    )
    np.testing.assert_array_equal(used_hours[:, USED_SOURCE_COLUMN["hrrr"]], [0] * 6)
    np.testing.assert_array_equal(
        used_hours[:, USED_SOURCE_COLUMN["ecmwf_ifs"]], [0] * 4 + [1] * 2
    )
    assert not used_hours[:, USED_SOURCE_COLUMN["gfs"]].any()


NUM_HOURS = 240
BASE_HOUR = 10
SOURCE_LIST = ["ETOPO1", "hrrrsubh", "librewxr", "nbm", "ecmwf_ifs", "gfs", "aigfs"]
SOURCE_TIMES = {
    "hrrr_subh": "2026-11-09 12Z",
    "librewxr": "2026-11-09 12:10Z",
    "nbm": "2026-11-09 12Z",
    "ecmwf_ifs": "2026-11-09 00Z",
    "gfs": "2026-11-09 06Z",
    "aigfs": "2026-11-09 06Z",
}


def _used_hours():
    """NBM supplies the first 60 hours, ECMWF the rest; GFS nothing."""
    used_hours = np.zeros((NUM_HOURS, len(USED_SOURCE_KEYS)), dtype=bool)
    used_hours[:60, USED_SOURCE_COLUMN["nbm"]] = True
    used_hours[60:, USED_SOURCE_COLUMN["ecmwf_ifs"]] = True
    return used_hours


@pytest.mark.parametrize(
    ("blocks", "kwargs", "expected"),
    [
        pytest.param(
            {},
            {},
            ["ETOPO1", "hrrrsubh", "librewxr", "nbm", "ecmwf_ifs"],
            id="all-blocks",
        ),
        # The returned 48 hours come from NBM; ECMWF only feeds later hours
        pytest.param(
            {"ex_daily": True},
            {},
            ["ETOPO1", "hrrrsubh", "librewxr", "nbm"],
            id="exclude-daily",
        ),
        pytest.param(
            {"ex_daily": True},
            {"output_hours": 168},
            ["ETOPO1", "hrrrsubh", "librewxr", "nbm", "ecmwf_ifs"],
            id="exclude-daily-extend-hourly",
        ),
        # Currently's precipitation is the first minute, so the minutely
        # sources count without the minutely block
        pytest.param(
            {"ex_daily": True, "ex_hourly": True, "ex_minutely": True},
            {},
            ["ETOPO1", "hrrrsubh", "librewxr", "nbm"],
            id="currently-only",
        ),
        pytest.param(
            {"ex_daily": True, "ex_currently": True, "ex_minutely": True},
            {},
            ["ETOPO1", "nbm"],
            id="hourly-only",
        ),
        # include=aimodels: AIGFS supplied every value of the blended GFS array
        pytest.param(
            {},
            {
                "used": {"gfs": slice(60, None)},
                "ai_blends": {
                    "gfs": (
                        "aigfs",
                        np.ones(NUM_HOURS, dtype=bool),
                        np.zeros(NUM_HOURS, dtype=bool),
                    )
                },
            },
            ["ETOPO1", "hrrrsubh", "librewxr", "nbm", "ecmwf_ifs", "aigfs"],
            id="ai-blend",
        ),
    ],
)
def test_used_sources_follow_the_returned_blocks(blocks, kwargs, expected):
    """A source counts only through a block that is returned: the daily block
    aggregates every hour, the hourly block its returned slice (48 hours, 168
    with extend=hourly), and currently/minutely their own selections. Getting
    this wrong lists models a response doesn't contain, or drops ones it does.
    The available list (what version 1 still returns) must stay as it was."""
    used_hours = _used_hours()
    for key, rows in kwargs.get("used", {}).items():
        used_hours[rows, USED_SOURCE_COLUMN[key]] = True
    metadata = SourceMetadata(list(SOURCE_LIST), dict(SOURCE_TIMES), {})

    used = select_used_sources(
        metadata,
        used_hours,
        minutely_used={"hrrrsubh", "librewxr"},
        current_used={"nbm"},
        daily_used=set(),
        ai_blends=kwargs.get("ai_blends"),
        ex_currently=blocks.get("ex_currently", False),
        ex_minutely=blocks.get("ex_minutely", False),
        ex_hourly=blocks.get("ex_hourly", False),
        ex_daily=blocks.get("ex_daily", False),
        day_night=False,
        time_machine=False,
        base_time_offset=BASE_HOUR,
        output_hours=kwargs.get("output_hours", 48),
        rrfs_grid=None,
    )

    assert used == metadata.used_sources == expected
    # sourceTimes follows the same sources (HRRR SubH's time key differs)
    expected_times = {
        key: time
        for key, time in SOURCE_TIMES.items()
        if {"hrrr_subh": "hrrrsubh"}.get(key, key) in expected
    }
    assert metadata.used_source_times == expected_times
    assert metadata.source_list == SOURCE_LIST
    assert metadata.source_times == SOURCE_TIMES


def test_next_update_is_the_earliest_future_refresh_of_the_used_sources():
    """flags.nextUpdate (#668) tells clients when to poll again, so it must be
    the soonest new run among the sources in the response, and never a time
    already past: a late ingest rolls forward to its next scheduled publish
    instead of telling clients to poll in a loop. Sources with no run time
    (GEFS appended by the AI merge) or no schedule (ETOPO1, ERA5) are skipped,
    and the 15-minute RTMA-RU cadence must win where it is used."""

    def utc(hour, minute=0, day=9):
        return datetime.datetime(2026, 11, day, hour, minute, tzinfo=datetime.UTC)

    now = utc(12, 20).timestamp()
    metadata = SourceMetadata([], {}, {})
    metadata.add("ETOPO1")
    metadata.add("era5")
    metadata.add("gfs", run_time=utc(6))
    metadata.add("nbm", run_time=utc(11))
    # 00Z IFS from yesterday: its 12Z successor is due at 11:30, already past
    metadata.add("ecmwf_ifs", run_time=utc(0, day=8))
    metadata.add("gefs")
    metadata.add("rtma_ru", run_time=utc(12), time_fmt="%Y-%m-%d %H:%MZ")
    assert metadata.source_times["rtma_ru"] == "2026-11-09 12:00Z"

    # NBM 11Z: the 12Z run is published 1.25 h after it, at 13:15
    metadata.used_sources = ["ETOPO1", "era5", "gfs", "nbm", "ecmwf_ifs", "gefs"]
    assert next_update_time(metadata, now) == utc(13, 15).timestamp()

    # IFS alone: overdue since 11:30, so the next cycle's publish at 23:30
    metadata.used_sources = ["ecmwf_ifs"]
    assert next_update_time(metadata, now) == utc(23, 30).timestamp()

    # RTMA-RU 12:00: the 12:15 analysis is published 25 min later
    metadata.used_sources = ["ETOPO1", "gfs", "nbm", "rtma_ru"]
    assert next_update_time(metadata, now) == utc(12, 15).timestamp() + 0.42 * 3600

    # A dropped source takes its run time with it
    metadata.drop("rtma_ru")
    assert next_update_time(metadata, now) == utc(13, 15).timestamp()

    metadata.used_sources = ["ETOPO1", "era5", "gefs"]
    assert next_update_time(metadata, now) is None
