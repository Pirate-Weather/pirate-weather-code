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

### Read-time interpolation of missing values
`zarr_read` (`API/io/store_reader.py`) linearly fills interior NaN runs along time
(`has_interior_nan_holes` / `_interp_row`, logged as "Interpolating missing data")
for every model except DWD MOSMIX and URMA. A value that is genuinely missing in the
source (e.g. HRDPS CAPE -999, which the ingest stores as NaN) is therefore filled
from its neighbours at read time instead of falling back to another model, which is
what the per-hour source priority is for. Decide whether to keep NaN for all models
(more consistent; changes API output wherever holes exist, e.g. HRRR smoke) or just
for the MSC models, whose stores now only carry real gaps.

Before deciding, check how often genuine NaN values show up in each model's source
data and published stores: count NaN per variable over each store's real grid
(the store's `shape` minus the padding to `FINAL_CHUNK_SIZES`), and
read the reader's "Interpolating missing data" warnings in production logs.

### `interp_time_take_blend`: blend in the data's dtype
`API/ingest_utils.py` (`interp_time_take_blend`): the blend weights `w` are float64,
so `(1 - w) * y0 + w * y1` promotes every float32 value to float64 (double the
temporary memory) and casts back afterwards. Casting `w` to the array's dtype made a
GDPS-sized block (21 variables, 136 -> 240 hours, 200 x 200 points) 2.3x faster
(1.00 s -> 0.43 s, single thread); that is already faster than a hand-written numpy
loop. Values change only by float32 rounding: after the store's 5-decimal rounding
about 8% of temperatures (max 3e-5 K) and 14% of pressures (max 0.016 Pa, the float32
resolution near 1e5 Pa) differ. Every ingest using the helper (10 models) then stops
being bit-identical to its previous output, so do it on its own, not alongside a
change that needs an old-vs-new bit comparison.

## v3 goals

From the v3 planning post (pirate-weather-code discussion #773). Goals already done
there (forecast/historic split, test refactor, ingest standardization) are not listed.

### ECMWF 0.1° open data
[pirateweather#697](https://github.com/Pirate-Weather/pirateweather/issues/697): ECMWF's
9 km feeds are on AWS since the 2026-10-20 06Z run. Blocked until AWS holds 10 days of
0.1° history (not before about 2026-10-30). Keep the ingest's short-term history at
48 h (`HISTORY_PERIODS["ECMWF"]`). Touches `ingest/ECMWF_Local_Ingest.py` (0.25° now)
and the API's ECMWF grid lookup; the store grows about 6x, so check chunking against
the read path.

### Validate wind direction in the regional models
As done for HRRR, which rotates grid-relative winds with
`earth_relative_wind_components` (`API/ingest_utils.py`). Check whether HRDPS
`WIND`/`WDIR` on its rotated lat/lon grid are already earth-relative.

### LibreWXR for maps, radar coverage and minutely precipitation
Plan in [pirateweather discussion #414](https://github.com/Pirate-Weather/pirateweather/discussions/414#discussioncomment-18934138):
use LibreWXR's global radar nowcast for minutely precipitation worldwide (likely
better than HRRR sub-hourly even inside its domain). LibreWXR gets three small hooks
that run user scripts (plus `zarr` as a dependency):
- on each finished radar composite: write one global minutely zarr, chunked along
  time, and upload it to S3 for the Pirate Weather minutely forecast;
- on each completed full ingest (`state.json` updated): sync everything to S3 for the
  render nodes;
- on each lightning update: sync it to S3 for the render nodes.

This decouples ingest from serving: render nodes can run in several regions, and the
last good data stays available if the ingest pipeline fails. The old `*_Maps.zarr`
stores were removed from the ingests in v3 (`75a2443`). Pirate Weather side: read the
minutely zarr in the minutely builder (`API/minutely/`) and decide its priority
against HRRR sub-hourly.

### Own ERA5 dataset
Only the needed variables, chunked for point reads, based on the Google ARCO pipeline
the API reads today (`ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, one whole-globe chunk per
hour and variable). Design so far: the 18 surface variables the API reads, 1940 to
present, `(variable, time, y, x)` with 120 h time blocks and 8 x 8 point tiles, one
shard per 120 h block; about 11-19 TB. Daily ERA5T appends (about 6 days behind) and a
monthly replacement with final ERA5 (3 months behind). Prototype one month first.
Replaces `init_ERA5` (`API/io/ZarrHelpers.py`) and `_load_era5_slice`
(`API/request/grid_indexing.py`), including the 20 GB disk cache.

### Move to OVHCloud
Including a Terraform setup script.

### Open the history stores in parallel
The history ingests open each history store and then each variable's array one
at a time, and from S3 every open is a sequential round trip. RTMA spent 33 s of
its 239 s merge opening 48 stores; GFS spends ~13-17 s per variable building the
merge graph over 48 stores (28 variables, ~7 min at ~4% CPU) before reading any
data. Open the stores (and their arrays) concurrently, or fetch each store's
array metadata once, in `open_history_store` / the merge loops.

### Read the alert grids (and maybe DWD) with tensorstore
The NWS and WMO alert grids are now sharded (10 x 10 chunks in 100 x 100 shards:
~276 and ~1,400 objects instead of 25,793 and 133,633), but they are
variable-length string arrays, which tensorstore cannot read, so they are the
only stores still read with zarr-python: two scalar reads per request, ~0.3-0.5
ms against ~1.3 ms for the other 16 reads together. Re-encode them for
tensorstore: an int32 grid of indices into a table of the distinct alert
strings, published next to the grid (or in its attributes if small enough).
Update `API/alerts.py` and the WMO read in `API/request/grid_indexing.py`, and
measure. WMO's gridding also peaks at ~6 GB on one core.

DWD MOSMIX is a 720 x 1440 grid that is NaN except at ~5,000 stations; a
station-indexed store (or keeping only the shards with stations) may suit it
better than a mostly-empty grid. Check its size and read time first.

### NBM: interpolate in 100 x 100 tiles
NBM's interpolate + final write is the fleet's memory peak (24.0 GB). Rechunking
the `interp_time_take_blend` input to 100 x 100 tiles cut the peak to 19.6 GB for
+24 s (369 vs 345 s) with a byte-identical store (same final 5 x 5 chunks, so
reads are unchanged). Measured end to end on the 2026-10-23 03Z run; the patch
used was a one-line rechunk before the blend.

### NBM: release memory before publishing
NBM holds ~10 GB of RSS through the whole 17 min upload of 416,807 objects. Free
the interpolated arrays (or publish from a fresh process) before uploading so the
upload doesn't pin memory another ingest on the host could use.

### NBM: latest-run search waits on missing files
The latest-run probe checks 36 GRIB files with Herbie; when the newest cycle is
only partly posted (25/36 missing), the misses took ~4 min before processing
began. Probe with a short timeout, or check the newest candidate first and stop
at the first complete run.

### URMA: 5 min idle before the final merge
In the cold-start run, after the last hourly store was published (3,981 s) nothing
happened for ~5 min (no CPU, no network) before "Starting final URMA merge from
283 hourly stores" (4,275 s), then 2.5 min opening the stores one at a time. Find
the wait; the store opens fall under "Open the history stores in parallel".

### Canadian models: parallel GRIB conversion
HRDPS, GDPS, REPS and GEPS download one variable at a time (~6 MB/s from the MSC
datamart), then convert on a single core: GDPS spends 12.6 min downloading and
11 min converting (writing ~31 GB of intermediates and reading ~87 GB back) of a
39 min run; HRDPS 4.5 + 4.5 min of 18. Run the downloads and conversions 3-4 at
a time, and avoid re-reading the intermediate files.

### AIGEFS: download takes ~5 h
AIGEFS only comes from NOMADS, which rate-limits: ~10.3 min per member, ~5.2 h for
30 members before the final write starts, using under 1 GB and ~1% CPU. Fetch
members in parallel within the rate limit, or switch source if AIGEFS appears on
a cloud mirror.

### Test an NFS share as the publish target
Same flow as object storage, with an NFS share (OVH offers up to 16,000 IOPS /
128 MiB/s) in place of the bucket: ingests publish to the share (`open_remote`
already treats a mounted path as a local remote) and each API server copies new
stores to its local disk, into a staging directory renamed into place so the API
never sees a partial store (`API/store_sync.py` does this from S3 today; with
sharded stores HRRR is 199 files, so the estimate below is now far lower). The
load was file count, not bytes (HRRR unsharded: 76k files, 318 MiB): a small-file write is ~3-5 NFS operations and a copy-down read ~2-3,
so every reader adds about as much load as the publish. Estimated peak, when NBM,
URMA and HRRR publish together (~1,000 files/s): ~4-5k IOPS to write plus ~2.5k
per reader, so ~12k with three readers. Copies must be parallel (rclone with high
--transfers, or obstore local-to-local); a serial cp/rsync pays a round trip per
file. Measure on a mounted share: publishing the real HRRR store and copying it
down to local disk (files/s, IOPS used, wall time), then size IOPS from those.

### Remaining storage clean-up
- Re-publish the static `ETOPO_DA_C` store as a zarr v3 directory on the remote
  (`API/store_sync.py` copies `ETOPO_DA_C.zarr/` once; the API opens it with
  tensorstore, zarr v3 or v2).
- Re-publish the static developer snapshot linked from `README.md` in the
  sharded, versioned layout.
- Expire old history stores on the remote (nothing deletes them).
- The DWD station and RAQDPS lat/lon pickles are loaded once at API start; the
  sync updates the files but a running worker does not reload them.
- Run the API with several uvicorn workers in production (`--workers`); each
  worker opens its own tensorstore context and heartbeat.

### Tune the shard and cache sizes from real requests
`FINAL_SHARD_SIZES` follow a ~150 MB-per-shard write-memory rule and
`TS_CACHE_BYTES` defaults to 256 MB per worker; both were sized from the HRRR
benchmarks. Replay a day of real request coordinates against the new reader to
pick the cache size (the shard indexes alone are ~35 MB), and check the write
peak of the largest shards (GFS, NBM, ECMWF) in the next end-to-end run.

### Confirm the KEEPBITS steps for the less-used variables
`KEEPBITS` (`API/ingest_utils.py`) bitrounds every stored variable to twice its
output step. The steps for VVEL, PM_FRP_column, BLH and the Storm_Distance maximum
were assumed, not checked against how the API uses them.

### Response code performance pass
Profile the request path for room to improve; read performance is the first priority
(AGENTS.md).

### Daily PoP from ensemble members
Daily PoP from hourly max/mean understates rain whose timing the models disagree on
(e.g. hourly 10/10/10/0/10/40/10% gives max 40%, mean 15%). Report instead the share
of ensemble members above the threshold anywhere in the period, for day, night and
24 h windows in local time. The ensemble ingests (GEFS, GEPS, REPS, ECMWF ensemble)
would need to provide windowed member counts, with windows that depend on the time zone.

### Future dates in time machine
[pirateweather#689](https://github.com/Pirate-Weather/pirateweather/issues/689): Dark
Sky allowed time-machine requests for future dates; Pirate Weather returns HTTP 400.

### Only list the sources used
[pirateweather#554](https://github.com/Pirate-Weather/pirateweather/issues/554):
`sources` lists every model covering the point, not just the ones used in the response.

### Next-update field
[pirateweather#668](https://github.com/Pirate-Weather/pirateweather/issues/668): report
when the forecast data will next refresh. The run-time pickles and the ingest
schedules already hold what is needed.

### Consistent clipping
[pirate-weather-code#760](https://github.com/Pirate-Weather/pirate-weather-code/issues/760):
some fields use `clipLog`, others `np.clip`. See also the `clipLog` bug above ("One
implausible value blanks every above-max value").

## Done

### Sharded stores, times in attributes, tensorstore reader, new sync (2026-10-23)
- **Stores:** every final store is a sharded zarr v3 array (`FINAL_SHARD_SIZES`
  in `API/ingest_utils.py`: 100 x 100 for the regional models, 48-99 for the
  long global ones), and the alert grids are sharded too. The time variable is
  gone from the array; its valid times are the `time` attribute, and the reader
  rebuilds column 0 (NaN where every variable is NaN, as before).
- **Publish:** each run uploads to `<MODEL>/<stamp>.zarr`, then moves
  `<MODEL>.current.json`; the version before is kept, older ones are deleted.
- **Read:** `API/io/store_reader.py` opens every store with tensorstore (one
  shared context, 256 MB cache, `recheck_cached_data="open"`), polls the
  pointers every 10 s, swaps in new versions while serving, and writes a
  heartbeat of the versions it holds. The TESTING stages read S3 through the
  same code.
- **Sync:** `API/store_sync.py` (Python + obstore) replaces `zip_sync.sh`. It
  copies new versions to local disk and deletes an old one only when no live
  heartbeat lists it.
- **Legacy:** the old zarr-python reader, `zip_sync.sh`, and the LMDB and
  container prototypes are in `legacy/io/`.

Expected effect, from `Notes/STORAGE_AND_READ_BENCHMARKS.md`. These were measured
on HRRR with the benchmark harness, not yet in production:
- **Requests per host:** 5-6x (4 workers, 16 reads per request): 1,250-1,530
  requests/s against 225-240 with zarr-python on directory stores.
- **Read time:** a lone 16-read request drops from 9.8 ms to ~1.3 ms, and p99
  at 16 concurrent requests from ~90 ms to ~23 ms. Cold-cache requests are
  ~2-3 ms against 12-25 ms.
- **File counts:** HRRR goes from 76,321 files to 199. Publishing and syncing a
  store take seconds instead of minutes (HRRR to OVH S3 took ~4 min unsharded),
  and the WMO alert upload drops from 331 s to seconds.
- **Not yet measured:** the ingest write peak with whole-shard dask tasks, and
  the new reader inside the full API response (the 16 reads were ~5-8 ms of
  each request before).

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
