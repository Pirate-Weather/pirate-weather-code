# TODO

Known bugs found during the v3 test-suite review (2026-10-19). Each has been confirmed
by reading the code; none is fixed yet. When fixing one, add a single regression test
that would have caught it, then remove the entry.

## Bugs

### Local midnight is off by an hour on DST transition days
`API/request/preprocess.py:793`: `base_day = base_time.replace(hour=0, ...)` keeps the
pytz UTC offset in effect at the *request* time rather than at midnight (the line
above builds `base_hour` correctly with `pytz_tz.localize`). On a DST change day the
local day therefore starts an hour early or late: a New York time machine request for
2024-03-10 starts at 23:00 EST the previous evening, and 2024-11-03 starts at 01:00
EDT, and both return 24 hours instead of 23 / 25. `base_day` comes from the shared
request preprocessing, so forecast day boundaries on DST days are likely affected too.
Fix with `pytz_tz.localize(datetime(base_time.year, base_time.month, base_time.day))`.
Regression test ready: `tests/live/test_timemachine_dst_live.py` (currently xfail).

### `exclude=aigfs` / `exclude=aigefs` also exclude GFS / GEFS
`API/request/preprocess.py:445-446` uses substring checks (`"gfs" in exclude_params`),
so `exclude=aigfs` also sets `ex_gfs=1` and `exclude=aigefs` also sets `ex_gefs=1`.
Excluding an AI model silently drops its conventional counterpart. The model flags
should match whole comma-separated tokens.

### Minutely summary raises `UnboundLocalError` instead of falling back
`API/legacy/summary.py:30-38` (`build_minutely_summary`): if `calculate_minutely_text`
or `translation.translate` raises, the `except` block only logs, then
`return minute_translate, minute_icon` references names that were never assigned.
The intended "log and continue" fallback never happens; the request errors instead.

### `aqiunits=tw` is silently ignored
`API/request/preprocess.py:731` documents `tw → Taiwan AQI`, but `tw` is missing from
`_VALID_AQI_UNITS` and `_AQI_UNITS_MAP`, so the parameter falls back to the unit
system's default. Either add Taiwan AQI support or fix the comment.

### `useETOPO` environment variable does nothing
`API/responseLocal.py:108` reads `use_etopo`, but the README Docker example
(`README.md`, "Option 2") sets `useETOPO=FALSE`. ETOPO is loaded anyway.
Fix the README (or accept both names).

### `is_in_north_america` exclusion branch is dead; western Aleutians excluded
`API/utils/geo.py:295-303`: the outer box requires `lon >= -168`, so the inner
`lon_normalized < -170` "US Minor Outlying Islands" branch can never run. Locations
west of -168° (western Aleutians) get rest-of-world source priority. Decide the
intended boundary and simplify.

### One implausible value blanks every above-max value in `clipLog`
`API/api_utils.py:142-156`: if any value exceeds 125% of the max, *every* value
above the max becomes `MISSING_DATA`, including plausible ones that would otherwise
be clamped (one 1000 °C hour also wipes out 61 °C hours instead of clamping them to
60). The same pattern applies to the min side. Only the out-of-range values should
be replaced; the rest should still be clipped.

### Daily sleet type is never preserved
`API/api_utils.py:360-389` (`select_daily_precip_type`): the argmax step overwrites
`maxPchanceDay` with rain/snow/ice on every day with precipitation (unless all three
types are present) *before* the "preserve ice/sleet from the hourly data" check runs,
so that check can never see sleet. Ice-pellet days are always reported as freezing
rain ("ice").

### Minutely "-later" durations may be off by one
`API/PirateMinutelyText.py:46-70`: "stopping-then-starting-later" reports the end as
`precipEnd1 + 1` minutes but the gap as `precipStart2 - precipEnd1`, and
"starting-then-stopping-later" reports the duration as `precipEnd1 - precipStart1`
without the `+ 1`. E.g. rain in minutes 0-19 and again from 41 reads "ending in 20
min, returning 22 min later". Decide whether `precipEnd1` is the last wet minute or
the first dry one and make all three numbers consistent.
`tests/api/test_minutely_thunderstorm.py` deliberately doesn't pin the second number.

### `interpolate_temporal_gaps_efficiently` fills gaps longer than `max_gap_hours`
`API/ingest_utils.py` (~1400-1458): `interpolate_na(limit=max_gap_hours)` fills the
first `limit` NaNs of *any* gap, and the unconditional `ffill`/`bfill` that follows
fills the rest, so `[1, nan×5, 7]` with `max_gap_hours=3` becomes
`[1, 2, 3, 4, 4, 4, 7]` (partial ramp, flat line, jump). The docstring implies
xarray's `max_gap` semantics (leave long gaps alone). Used by the DWD ingest with
`max_gap_hours=3`; `fill_station_gaps` runs first and already blanks whole series
with long gaps, so the impact is limited to gaps it doesn't catch.

## Improvements

### Improve the Canada vs US model selection
Canadian-first source priority (HRDPS → REPS → GDPS → GEPS ahead of NBM/HRRR) is
chosen by `is_in_canada` (`API/utils/geo.py`), a bounding box (lat 41.7-83,
lon -141 to -52) rather than the border. Large parts of the northern US fall inside
it and get Canadian-first priority instead of NBM/HRRR, e.g. Seattle, Chicago,
Detroit, Boston and Minneapolis. Every US priority test uses New York (40.7°N),
just south of the box, so this is untested. Options:
- a real border check, such as a precomputed country mask on the model grid;
- choosing by which model domain covers the point best.

Keep it cheap on the read path: no per-request polygon tests against large
geometries. Update `docs/forecast-source-selection/README.md` and
`tests/api/test_regional_source_priority.py` with points on both sides of the border.
Related: the `is_in_north_america` boundary issue above.

## Docs

### Source-selection docs predate #734 and the AI ordering
`docs/forecast-source-selection/README.md` lists current UV as "GFS-compatible → ERA5"
(and "GFS-compatible → HRDPS → GDPS" in the element table), but since #734 HRDPS UV and
GDPS ozone lead GFS in the Canada box (pinned by
`tests/api/test_regional_source_priority.py`). It also only says the AI source leads
in AI mode; in North America the currently order then puts ECMWF IFS second, ahead of
RTMA-RU, which isn't documented.

## Needs verification

### AQI breakpoint tables for Vietnam and Malaysia
`API/constants/aqi_const.py`: `VNAQI_PM25_BP` / `VNAQI_PM10_BP` may not match VEA
Decision 1459, and `API_PM25_BP` / `API_CO_BP` (Malaysia) may not match the DOE
tables. Flagged from memory during the test review, not checked against the official
documents. `tests/api/test_aqi_calculations.py` only pins values where both readings
agree. Comment-only issues in the same file: the EPA O3/CO tables are labelled µg/m³
but are in ppb, and the module docstring says "three AQI systems" (there are eleven).

## Needs a decision

### GEPS/REPS ensemble precip type: sleet or mixed?
`API/api_utils.py:541` (`map_ensemble_precip_rates_to_ptype`): the docstring says
"if more than one channel exceeds the threshold we fall back to mixed precipitation",
but the code assigns `PRECIP_IDX["sleet"]` (`api_utils.py:606-607`). One of them is wrong.

### Minutely and currently disagree on thunderstorms
Minutely uses a raw `maxCAPE >= 2500` gate (`API/PirateMinutelyText.py:432`), while
currently/hourly use the multi-index score with CIN/PoP/dewpoint suppressors
(`API/PirateTextHelper.py`). Minutely can say "thunderstorm" while currently
suppresses it. Decide whether minutely should use the same score.
