# TODO

Open bugs, improvements and v3 goals, then a log of finished work with its measured
results. When fixing a bug, add a single regression test that would have caught it,
then move the entry to Done.

## Bugs

## Improvements

## Plan (2026-11-09)

Work through the open items in this order, small ones first:
1. Small: reload the DWD station and RAQDPS lat/lon pickles when the sync updates
   them; re-publish the README developer snapshot in the sharded layout; consistent
   clipping (#760).
2. Bigger: confirm the KEEPBITS steps; only list the sources used (#554); next-update
   field (#668).

## v3 goals

From the v3 planning post (pirate-weather-code discussion #773). Goals already done
there (forecast/historic split, test refactor, ingest standardization) are not listed.

### LibreWXR: build from the upstream release
The hooks and snapshot sync are upstream (JoshuaKimsey/LibreWXR #55 and #56, released
in v0.9.0), but `LIBREWXR_REF` in `build.yml` and `sanity.yml` still names the fork
branch `network-split`. Switch both to `v0.9.0`, then move the RBX `librewxr` image
pins (pirate-weather-ovh `values.local.yaml`) to the rebuilt images.

### ERA5 archive: remaining work
- Alert if Google's ERA5T end stops moving (the daily job exits 1 after 10 days; nothing
  watches for that yet).
- Open a week with one metadata read instead of two: `_resolve_era5_archive_week` reads
  the week's `zarr.json`, then `ts.open` reads it again (`API/io/ZarrHelpers.py`).

### Remaining storage clean-up
- Re-publish the static developer snapshot linked from `README.md` in the
  sharded, versioned layout.
- Expire old history stores on the remote (nothing deletes them).
- The DWD station and RAQDPS lat/lon pickles are loaded once at API start; the
  sync updates the files but a running worker does not reload them.

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

### Only list the sources used
[pirateweather#554](https://github.com/Pirate-Weather/pirateweather/issues/554):
`sources` lists every model covering the point, not just the ones used in the response.

### Next-update field
[pirateweather#668](https://github.com/Pirate-Weather/pirateweather/issues/668): report
when the forecast data will next refresh. The run-time pickles and the ingest
schedules already hold what is needed.

### Consistent clipping
[pirate-weather-code#760](https://github.com/Pirate-Weather/pirate-weather-code/issues/760):
some fields use `clipLog`, others `np.clip`.

## Done

### CFS ingest for months-out future requests (2026-11-10)
`ingest/CFS_Local_Ingest.py` publishes `CFS.zarr` for future-day requests past the
240 h GFS/GEFS stores. Source: the CFSv2 time-series files on AWS through Herbie
(`product="time_series"`), one object per variable per member holding the whole run at
6-hourly leads from 6 h, on the T126 Gaussian grid (384 x 190) except `prmsl` (1 degree,
regridded bilinearly). Eight variables (tmp2m, q2m, pressfc, prmsl, wnd10m, prate,
tcdcclm, dswsfc) x 4 members = 32 files, ~1.1 GB, capped by a Herbie `search` at
`CFS_FORECAST_DAYS = 184` (6 months; member 01 runs ~295 days).

- **00Z only, once a day:** its members 02-04 run ~111 days (45 at 06/12/18Z), and a
  months-out forecast gains nothing from 6-hourly refreshes.
- **Ensemble hand-over:** state variables are the 4-member mean while all four run and
  member 01 after (2026-11-10 00Z: 4 members to 2027-02-28 18Z, lead 2658 h; member 01
  to the cap, 4416 h). Dew point and RH come from the mean T, q and surface pressure.
  `Precipitation_Prob`, `APCP_StdDev` and the member-mask word exist only in the
  4-member range and are NaN after the store's `ensemble_end` attribute.
- **Leads from the data:** Herbie drops the last index message of a file when it is
  selected, so members 02-04 came back with 443 of 444 leads (wnd10m's u and v one lead
  apart); each member keeps the leads all its variables share.
- **Store:** hourly (4,411 steps), the one time-chunked final store
  (`FINAL_TIME_CHUNKS`, read with `zarr_read_window`); size and read times in
  `Notes/STORAGE_AND_READ_BENCHMARKS.md`.
- **Measured** (8-core VM, local publish): 186 s wall (download + wgrib2 decode 56 s,
  member reduction 17 s, validation + hourly interpolation + write 110 s), 3.9 GB peak
  RSS (7.9 GB while the 6-hourly stack was held in memory; it is now a local zarr),
  2.3 CPU cores on average, 7.4 GB peak scratch. Store 2.4 GB, 257 files, 15.7 GiB
  uncompressed. wgrib2's OpenMP threads do not speed up these JPEG 2000 files (9.0 s
  on 1 thread, 10.9 s on 8), so conversions run single-threaded side by side, and
  `-stats` shares the NetCDF pass. Spot-checked against `wgrib2 -lon` at 4 points,
  leads 120 h (4-member mean) and 3000 h (member 01): equal within bitround precision.
- **Rejected: TMAX/TMIN correction** of the interpolated temperature. 6-hourly
  sampling understates the daily high by 0.21 K on average (p90 0.58 K, p99 2.8 K,
  |lat| < 60), far below the forecast error at these ranges, so tmax/tmin are not
  downloaded.
- **API side (API v3.0.0a7):** a future day may end at the CFS store's last stored hour
  (~4,400 h, 183 days) instead of 218 h while CFS is loaded, not `exclude=cfs`d and its
  run is at most `CFS_MAX_RUN_AGE_HOURS` (72) old; the check uses the in-memory store
  times (`cfs_last_time` in `grid_indexing.py`, called from `prepare_initial_request`),
  and the 400 states the limit that applied. CFS is read (`zarr_read_window`, the day's
  25 rows plus an hour either side) only when `future_day` is set and the day's last hour
  is after the GFS store's last hour (or no GFS store is loaded); nearest Gaussian row,
  columns wrap at 360. It is last in every hourly, precipitation, probability, error and
  `currently` order, with its member mask word feeding the daily probability; it has no
  precipitation type, so its hours get "none" and the hourly temperature fallback types
  them. `cfs` is in `FORECAST_SOURCES`, `sources`, `sourceTimes` and `sourceIDX`.
- **Measured** (TESTING stage reading the production v32 stores on OVH RBX in-process,
  plus this store locally; 6 cities incl. 359.8 E and three southern ones; days 5, 9.6,
  12, 60, 150, 200): every day to 150 is 200 with 24 hourly rows; 200 days is a 400 at
  4,400 h; `exclude=cfs` at 60 days is a 400 at 218 h. Values match the store cell
  (60 days, New York, 4 hours: temperature, wind, cloud, probability, intensity equal).
  Daily probability comes in quarters inside the 4-member range and is -999 at 150 days
  (after `ensemble_end`); Moscow in January is typed snow. A straddle day takes GFS to
  its last hour, DWD MOSMIX where it runs longer, then CFS (New York 2026-11-20: 12.1 C
  MOSMIX to 5.6 C CFS across the switch). Handler time, in-process, 6 cities, median of
  180 warm requests, HEAD vs this change: normal request (`exclude=alerts`, as the alert
  grids read from S3 add ~190 ms) 10.0/10.1/10.0 vs 10.0/10.0/10.0 ms with no CFS read; a
  5-day future day 7.2 vs 7.1-7.4 ms with no CFS read; a 60-day day 6.7-7.1 ms with one
  CFS window read (local store).
- **Fixed on the way:** a straddle day where a store's last hour is local midnight
  (GDPS, London 2026-11-20) left one row, which crashed the minutely interpolation
  (`np.interp` on a 0-d time column); `_rows_with_time` now drops a model with fewer than
  two timed rows. Unreachable before, as such days were past 218 h.
- **Feels-like and `currently` precipitation on CFS hours:** CFS has no apparent
  temperature, so feels-like is computed from its temperature, humidity, wind and solar
  (last in `feels_like_inputs` and `_get_feels_like`), and CFS is the last minutely
  source (intensity, probability, error; typed from temperature like DWD MOSMIX), which
  `currently` copies. Before, they were -999 and 0. Measured as above (60 and 150 days,
  8 cities): hourly feels-like has no -999 left; Jakarta 60 days `currently` 0.81 mm/h,
  probability 1.0, error 0.78 (was 0/0/0); New York 150 days 2.09 mm/h with probability
  and error -999 (after `ensemble_end`). Normal and 5-day responses (6 cities) are
  identical to before apart from `processTime`; processing median of 180 requests, two
  rounds, 10.0/10.0 ms normal and 6.3/6.4 ms 5-day vs 10.1/9.9 and 6.6/6.3 before.
- **`currently` on the day GFS ends, and dry minutes after `ensemble_end`:** the day
  holding GFS's last hour reads CFS, but GFS, GEFS or ECMWF rows from earlier in it won
  the per-day minutely source choice and were NaN past their end, so `currently` came
  out -999 "Mixed Precipitation" (or a NaN zeroed to 0 "none" where the minute was
  untyped). When CFS is loaded, `build_minutely_block` now fills the minutes whose
  chosen intensity is NaN from CFS (typed from its temperature), and a NaN
  probability or error from CFS's. Past `ensemble_end` a dry CFS minute's probability
  and error stay -999 instead of being zeroed by the "none"-type rule, as in the hourly
  block. Measured with GFS ending 2026-11-20 06Z: New York 07Z and 08Z -999/-999/-999
  "mixed" to 0/0/0 "none" "Partly Cloudy"; Sydney 06-08Z 0/0/0 "none" to 0.81 mm/h,
  0.5, 1.04 rain (ECMWF ended 00Z); Seattle and Sydney at 150 days dry minutes
  probability and error 0 to -999. 60-day minutes, hourly and daily unchanged.
  Normal and 5-day responses (6 cities) identical to before apart from `processTime`;
  handler medians of 90 requests, two rounds: normal 198.5/199.6 ms vs 200.0/224.1
  (S3 grid reads), 5-day 7.8/7.4 vs 7.5/7.4, 60-day 7.3/7.3 vs 7.3/7.3.
- **Not done:** visibility, UV, ozone, CAPE are -999 on CFS-only hours (no CFS field);
  `sources` still lists models with no rows on the day (#554).
- **Not done (minutely on the day GFS ends):** the minutely block picks one source per
  element for the whole hour, not per minute like the hourly block, so a minute past
  ECMWF's end takes CFS even where GFS still runs (Sydney 03-06Z: `currently` 0.33-0.81
  mm/h CFS, hourly 0.08-0.11 GFS). An hour whose 61 minutes cross GFS's last hour is all
  "mixed" (any NaN type row forces it), including the covered first minute (New York
  06Z: 0 mm/h "Mixed Precipitation"). Both predate CFS; by the same code (not measured)
  a future day GFS covers but ECMWF ends in shows such minutes as 0 "none".

### Future-dated requests (2026-11-09)
[pirateweather#689](https://github.com/Pirate-Weather/pirateweather/issues/689): a time
more than an hour ahead returned HTTP 400; it now returns Dark Sky's Time Machine shape
for a future day: `currently` at the requested time, hourly from local midnight to the
next (23 or 25 hours on a DST change day), one daily point, no minutely or alerts, and
time-machine style hourly/daily summaries. It is still a forecast (`time_machine` stays
False, `future_day` is set): the forecast models, fields and sources are those of a normal
request, except RTMA-RU and LibreWXR, which describe the present. `currentDay*` totals run
from local midnight to the requested hour. Not version-gated. Times within an hour are
still read as now. The horizon is `FUTURE_FORECAST_MAX_HOURS` (218 h, `api_const.py`):
the requested local day must end within 240 h of the newest GFS/GEFS run allowing that run
to be ~22 h old; later days are a 400. Since 2026-11-10 a fresh CFS store extends it to the
store's last hour (previous entry). Run-age freshness checks in `calculate_grid_indexing`
are now measured from `min(requested time, now)`, so a future day keeps the current runs;
the sub-hourly HRRR/RRFS, RTMA-RU and LibreWXR checks stay on the requested time and drop
out. `_aligned_rows` gave a day after a store's end that store's last row at local midnight
(and the one-row REPS array crashed the minutely interpolation); such a day now gets no
rows from it. Checked in-process against the local GEFS/REPS stores (Toronto, +4 days):
24 hourly rows from local midnight, one daily point, no minutely or alerts; +10 days is a
400. `sources` still lists a model whose store ends before the requested day (#554).

### Moved to OVHCloud (2026-11)
The API, the ingests, LibreWXR and the daily ERA5 job run on OVH's RBX Kubernetes
cluster, set up with Terraform in Pirate-Weather/pirate-weather-ovh (cluster metrics
pushed to Grafana Cloud). Production runs the API with 4 uvicorn workers and time
machine with 2. Every API container runs the `pirate-alpine-zarr:dev` image until v3
is released, because v0.7.4 predates the versioned stores.

### LibreWXR hooks and snapshot sync upstream (2026-11-05)
LibreWXR's nowcast/state/lightning hooks and the snapshot sync that mirrors its cache
to object storage for render nodes are in upstream LibreWXR (#55, #56; v0.9.0). On RBX
the `pirate-librewxr` pipeline runs `ingest/LibreWXR_Local_Ingest.py` from its nowcast
hook and publishes snapshots to `s3://pirate-librewxr/librewxr`, which a render node
pulls.

### Daily PoP from ensemble members (2026-11-09)
Daily `precipProbability` was the maximum of the hourly probabilities over the local day,
which understates rain whose timing the members disagree on. On real GEFS members (31
members, 24 h window, global grid), the share of members wet at any hour exceeds the
hourly maximum by 11.8 points on average where any member is wet (p90 29); cells at
30-50 % average 62 %. The ensemble ingests (GEFS, AIGEFS, GEPS, REPS, ECMWF IFS, AIFS)
now append per-hour member bitmask words to their stores (`Precipitation_Member_Mask_<w>`,
24 members per float32 word, exact under `KEEPBITS`, columns appended so older stores
still read). Each hour takes the word of the lead whose accumulation interval contains it
(`next_lead_vars` in `interp_time_take_blend`; nearest neighbour would hand the first
hours of a day the interval that ended at midnight). `API/daily/builder.py` ORs the words
over each local window (midnight day, 6am-6am icon window, day/night halves; today from
the current hour) and popcounts, taking the first ensemble in precipitation priority that
covers the whole window. The result is never below the hourly maximum, which also covers
Time Machine, stores from before the masks and the ECMWF history period (the history
runs skip the masks). Member counts are `shared_const.ENSEMBLE_MEMBER_COUNTS`; the
ingests fail before publishing on a mismatch (REPS confirmed 21; GEPS 21, IFS 51 and
AIFS 51 still to be confirmed by a run).

Measured on this machine, cycle 2026-11-09 06Z, local publish: GEFS 741 s, 6.2 GiB peak
(6.0 before), store 1,271 MiB (a 7-variable copy of the same store is 1,107 MiB, matching
production); REPS 268 s, 1.6 GiB, store 320 MiB (297). A point read returns the same
single chunk, now 9/7 (GEFS) or 16/13 (ECMWF) of the bytes: a local tensorstore point
read of the GEFS store took 0.25 ms against 0.21 ms for the 7-variable copy (280 random
points, page cache warm). The daily reduction costs about 0.25 ms per request for all
four windows. ECMWF shards grow from 146 to 180 MB uncompressed (`FINAL_SHARD_SIZES`
comment); its run was not measured. Toronto's third forecast day went from 30 % to 50 %,
Seattle's from 80 % to 97 %. The ingest version moved to v32 so the ingests publish
the wider stores to a new prefix while the running API keeps reading v31; the API side
(daily builder, merge width fix, API version a6, reading v32) followed in a separate PR
once the v32 stores existed. `_merge_simple_source` used to fail on a store wider than its column map.

### Google ERA5 opened without dask (2026-11-09)
`init_ERA5` opened Google ERA5 with dask in 24 h chunks: a graph of 273 variables x 55,152
chunks in every process that loads ERA5 (historic workers and dev). On RBX a historic
worker sat at 1.8-2.0 GiB against 0.5 GiB for a prod worker. It now opens with
`chunks=None`, and `_load_era5_slice` reads the 18 point series and the precipitation
window in parallel on `ERA5_read_pool`. Measured locally against live Google ERA5 with
four 49 h reads (incl. lon 0 and a polar point): output byte-identical to the old code,
5.9-6.5 s per read instead of 8.4-9.3 s, process memory 985 MiB instead of 1,747 MiB.
Reads the own ERA5 archive covers are unchanged.

Google ERA5 is now also opened only on the first fallback (`open_google_era5`), not at
startup: RBX logged no fallback in 1,822 historic requests over 72 h, so a worker
normally never opens it. `init_ERA5` adds 16 MiB instead of opening the dataset; the
first fallback pays the ~1.3 s open once per worker.

### Every ingest under 16 GB (2026-11-07)
A processing tile that doesn't divide evenly into final shards stays in memory until the
tiles sharing its shards are written. `CHUNK_SIZES` now divides `FINAL_SHARD_SIZES` (NBM,
GEFS and ECMWF 198, GFS 96, GDPS 150), and a test pins it. DWD fills and writes one
variable at a time instead of holding all 12. ECMWF frees each lead's precipitation
statistics as it goes. NBM now publishes at about 1.5 GiB instead of ~10 GB with no change
to the publish itself (local publish; an OVH upload wasn't run).

Write threads no longer follow the node: the `os.cpu_count()` defaults are gone, RTMA and
URMA default to 8, and `configure_zarr_writes` fixes dask's thread count
(`dask_num_workers`, 4). Peaks below are VmHWM, with the final stores byte-identical
(sha256 of every file) to `fa87fe1` on the same cycle. The capped runs share cached
downloads, so their times compare with each other but not with a full ingest.

| Model | Old peak | New peak | 4 cores, 14 GiB cap | 2 cores, 7 GiB cap |
|---|---|---|---|---|
| DWD | 16.9 GiB | 6.7 GiB | 440 s | 421 s |
| GFS | 10.4 GiB | 3.3 GiB | 933 s | 1061 s |
| NBM | 8.9 GiB at 4 threads | 3.7 GiB | 552 s | 649 s |
| ECMWF | 9.1 GiB | 5.3 GiB | 2113 s | 2272 s |
| GDPS | 8.3 GiB | 5.1 GiB | not run | not run |
| GEFS | 6.4 GiB | 6.0 GiB | not run | not run |

AIFS (5.6 GiB), AIGEFS (2.1) and AIGFS (1.9) only changed thread defaults. CFS (added
2026-11-10): 3.9 GiB, 186 s on 8 cores, 7.4 GB scratch.

History merged at a different tile holds tiles in memory (GFS 14.9 GB), so GFS history
moved to `GFS_Hist_v5`, written at the new tile. The first run rebuilds all 48 stores
(8.9 GB): 1642 s and 4.7 GiB with a local publish, against 933 s warm, plus the upload,
which wasn't measured. The store built on rebuilt history is byte-identical too.
### Canadian ingests download and convert 4 fields at a time (2026-11-07)
HRDPS, GDPS, REPS and GEPS fetched one field at a time and ran one `cat | wgrib2 -netcdf`
over everything. `download_fields_concurrently` now fetches 4 fields at once
(`msc_download_fields`), splitting `herbie_download_threads` (20) between them so the
connection count to the MSC datamart does not grow, and `convert_fields_to_netcdf`
runs one wgrib2 per field, 4 at once (`msc_convert_workers`), then merges the files
lazily, read in processing tiles (dask, `compat="override"`: the default compat loads
everything, 26 GB on GDPS, and one dask chunk per field held whole fields, 12.2 GiB).
Final stores are byte-identical to the old code (sha256 of every file). Cycle
2026-11-06 12Z, 4 cores (`taskset`), `zarr_store_workers=2`, cold downloads:

| Model | Old: download / convert / total | New: download / convert / total | Peak RSS (VmHWM) | Scratch |
|-------|---------------------------------|---------------------------------|-------------------------|---------|
| HRDPS | 286 / 272 / 739 s | 151 / 108 / 466 s (335 s in an earlier run) | 3.0 -> 0.8 GiB | 14 GB |
| GDPS  | 870 / 744 / 2,313 s | 432 / 368 / 1,450 s (1,219-1,294 s in earlier runs) | 8.1 -> 2.1 GiB | 36-40 GB |
| REPS  | 169 / 176 / 406 s | 90 / 81 / 226 s | 1.5 -> 1.5 GiB | 9 GB |
| GEPS  | 207 / 212 / 463 s | 166 / 123 / 328 s | 2.2 -> 2.2 GiB | 8.5 GB |

New peaks were measured separately, on later cycles, with `ingest-memory-16gb`'s tiles
(GDPS 150). Stores match with and without reading in tiles.

Download time varies a lot run to run. With 4 fields at once MSC sometimes refused
connections (ConnectTimeout, several fields at the same second); the field's retry then
re-downloads all of its files because retries use `overwrite=True` even though files are
written atomically. Not seen in the old serial runs.

### Own ERA5 archive (2026-11-04 to 11-06)
Replaces Google's ARCO store (`ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, one whole-globe
chunk per hour and variable) for time machine reads; Google is now only the fallback.
`scripts/build_era5_point_test_store.py` builds the int16 layout measured below
(`ERA5_ARCHIVE*` in `API/constants/model_const.py`, `encode_era5_archive_field` /
`write_era5_archive_zarr` in `API/ingest_utils.py`, `decode_era5_archive` in
`API/io/ZarrHelpers.py`, contract test `tests/ingest/test_era5_archive.py`): 19
variables including direct solar (`fdir`) and cloud base height (`cbh`, NaN where there
is none). Real 2015-06-29 week (2026-11-04): 1125 MiB, 121 objects, 11.4 GB peak
memory (27 GB for the float32 builder), 15 s write, 2:22 wall; 16 spot cells x 19
variables decode within one quantization step of Google's values.
The API reads it first when `ERA5_ARCHIVE` is set (`s3://bucket/prefix` or a local
directory; `ERA5_ARCHIVE_S3_ENDPOINT` / `_REGION` for the BHS private endpoint):
`_load_era5_slice` -> `read_era5_archive_point`, which returns Google's layout and
units, so nothing downstream changed. A request touching any week the archive lacks
is read entirely from Google. Live, BHS private endpoint (2026-11-04): 15 ms warm,
~1 s for the first read in a process (connection setup, week metadata and shard
index), 37 ms to find a week missing; values within half a step of Google's.

Backfill done (2026-11-04 to 11-06): bucket `pirate-era5` (RBX, High Performance),
`ERA5_archive/<week>.zarr` for 4,478 weeks, 1940-12-30 .. 2026-10-19 (complete weeks
to Google's ERA5T end, 2026-10-29), 5.27 TB, ~1.1 GB per week in every decade. Built
with `scripts/backfill_era5_archive.py`: 2020-2026 on a b3-64 in RBX-A (4 at a time,
2.9 h), 1941-2019 on an r3-128 in GRA11 uploading to RBX (8 at a time, 5.1 h + 15.1 h,
~3.4 weeks/min, bound by Google reads at ~110-180 s per week). API config:
`ERA5_ARCHIVE=s3://pirate-era5`, `ERA5_ARCHIVE_S3_ENDPOINT=https://s3.rbx.io.cloud.ovh.net`,
`ERA5_ARCHIVE_S3_REGION=rbx`. Starts at 1941: ERA5's forecast fields (gusts,
precipitation, radiation, cloud base) are missing for 1940-01-01 00-06 UTC.
Source quirks met: Google stalled for minutes once (one week hung; hence the per-week
timeout), and ERA5 has spurious CAPE spikes over steep terrain (up to ~2e6 J/kg in the
Andes), now capped at 20,000 J/kg (101 weeks before 2000 and a few after).
Daily updates: `scripts/era5_archive_daily.py`, run by the RBX CronJob
`ingest-era5-daily` at 06:30 UTC (deployed 2026-11-06, pirate-weather-ovh #1). Final weeks are immutable
`<week>.zarr`; ERA5T weeks and the current partial week are versioned behind
`<week>.current.json` and republished when Google has more; weeks move to final ERA5
when Google's does. Readers re-resolve changing weeks every 5 minutes and never read a
store that changes under them. Its first run converts the 17 ERA5T weeks the backfill
wrote as immutable stores (2026-06-29 onward) to versions (dry run, 2026-11-06: 18
versioned builds, ~45 min). Tested live on a test prefix (2026-11-06): 3 weeks in 169
s, a second run did nothing, and a reader kept serving through a republish and moved
to the new version at its re-check.

### ERA5 archive encoding chosen for storage cost (2026-11-03)
The archive uses the int16 layout below. The one-week test store
(`scripts/build_era5_point_test_store.py`, 2015-06-29, 18 variables, 10 x 10 inner chunks of the whole week, 100 x 100 shards) is 1885 MiB, which
would make 1940 to present roughly 8 TB. It uses the stock `KEEPBITS` classes and Blosc
zstd 3, so there is room to shrink it.

Measured 2026-11-03 (June and January 2015 weeks; sizes from a 15 % block sample,
checked against full one-week builds; reads from an RBX-A VM against RBX buckets):
- **Store int16, not bitrounded float32.** Linear quantization in API units (0.1 K,
  0.1 hPa, 0.1 m/s, 1 % cloud, ozone 2 DU, UV index 0.05, solar 2 W/m2, CAPE 10 J/kg)
  and `round(100 * sqrt(mm))` for precipitation and rates (error <= 0.01 mm below
  1 mm, <= 1 % above). That is finer than today's bitrounding for temperature (0.05 K
  max error against 0.125 K), and it makes the K -> C conversion moot.
- **Bitshuffle, not byte shuffle** (Blosc zstd 3): small ints compress far better
  bit-plane by bit-plane. Bitround floats gain nothing from it.
- **Hour-to-hour delta for the smooth fields only** (gust, MSLP, surface pressure,
  temperature, dewpoint, u, v, ozone), stored as values in the array, so any zarr
  reader works; the reader takes a cumulative sum over the chunk's hours. Delta makes
  precipitation, cloud, radiation, CAPE and precipitation type larger. The built-in
  `numcodecs.delta` filter is no substitute: tensorstore cannot open a store that uses
  it, and it differences the flattened chunk (along longitude: 1090 MiB/week against
  920; with a transpose codec to difference along time: 1276).
- **Merge the four rain/snow rates into rain and snow** (the API only uses the sums),
  and **precompute the 3x3 precipitation probability** as a stored variable (+1.2 %):
  a point read then never needs a neighbouring chunk (with 10 x 10 chunks about a
  third of points straddle a chunk edge today).
- Result (v1: 17 vars, 10 x 10 x 168 h chunks, 100 x 100 shards): **953 MiB/week
  against 1885 (-49 %), about 4.3 TB for 1940-2025**; Jan week -52 %. Point read
  93 KiB against 184 KiB. Without delta (plain tensorstore-readable values) 1220 MiB.
  A ~2 % precision set (0.2 K, 0.2 m/s, 0.2 hPa, 2 % cloud, coarser precipitation)
  gives 805 MiB/week (3.8 TB).
- No help: dewpoint depression instead of dewpoint (-0.4 %), time-innermost chunk
  order (+15-35 %), separate per-variable chunks (+1 %, and 17 requests per point),
  lz4/lz4hc, pcodec (+7 %, not readable by tensorstore), zfp (2.5x larger at this
  chunk size). zstd 5 is -1.3 % at similar write speed; zstd 9 -3.5 % but ~30x slower.
- Chunk shape barely moves size once bitshuffled: 5 x 5 cells costs +4.6 % for 24 KiB
  reads, but the full-archive shard index grows from 0.75 GB to 3 GB (10 x 10 x 168 h)
  - keep 10 x 10 unless indexes live on local disk. Day-long time chunks: 2 requests
  per point and larger.
- RBX High Performance, 16 reads in flight: today's layout 294-299 points/s at
  ~40 ms p50; v1 460-493 points/s at 21 ms; 5 x 5 chunks 531-590 points/s. Without a
  cached index (index GET + chunk GET) a read roughly doubles. RBX Standard is seek
  bound (190-320 ms p50, p99 up to 1.7 s, ~60 points/s) whatever the layout, so the
  time-machine path needs High Performance storage or local disk.
- Untested idea, measured for size only: latitude bands with fewer longitudes toward
  the poles (constant ~28 km east-west spacing, like ERA5's native reduced Gaussian
  grid) store 64 % of the points in 75 % of the bytes (710 MiB/week); it resamples
  Google's 0.25 deg grid, so check the API's nearest-cell lookup and values first.
- Extra variables Google's store already has, cost per week in this encoding:
  `fg10` gust since last hour 83 MiB (replace `i10fg` at no net cost), `mx2t`/`mn2t`
  as offsets from `t2m` 49 MiB each, snowfall `sf` 14, snow depth `sd` ~1-2, direct
  solar `fdir` 59, low/mid/high cloud 90/79/66, boundary layer height 80, cloud base
  112. ERA5 has no visibility or sunshine duration; ERA5-Land and the EDA spread are
  CDS-only.

### DWD doesn't fill over long gaps (2026-11-04)
Long gaps are genuinely missing data, and now stay missing so another model fills those
hours. Before, `interpolate_temporal_gaps_efficiently` turned `[1, nan x 5, 7]` into
`[1, 2, 3, 4, 4, 4, 7]` with a 3 hour limit and held the end values to the ends of the
series, and `fill_station_gaps` zero-filled long precipitation gaps and carried the type
code across them. In the 2026-11-04 file, `ww` ends 70 hours early at 499 of 5,648 stations
and 235 hours early at 87, and those hours held the last code. RR1c's routine 6 hour gaps
are within the 6 hour station limit, so they are still filled.

### `useETOPO` docs and the stale `aqiunits=tw` entry (2026-11-04)
The README's Docker example and a compose comment set `useETOPO`, but the API reads
`use_etopo`, so ETOPO loaded anyway. Both now use the real name. The `aqiunits=tw` entry was
already out of date: the comment no longer mentions Taiwan and nothing implements it, so
`tw` falls back like any unknown value. Taiwan AQI support would be a feature, and needs
breakpoints checked against the official tables.

### SILAM runs after FMI posts each cycle (2026-11-04)
The 07:00 UTC run always missed that morning's file, leaving the API a day behind. Over the
last 21 cycles, the file timestamps show FMI posting 00Z near 12:55 UTC the next day (run +
37 h), twice near 17:50. It now runs at 14:00 and 19:00 UTC, in both compose files.

### AQI averages reach back before local midnight (2026-11-04)
The AQI series started at local midnight, so hour *h* of today averaged only *h* + 1 hours
(a `currently` AQI just after midnight was a 1-2 h average). `prepare_aq_inputs` now also
gives the 24 hours before midnight, for the averages only. No extra reads: the stores are
read whole and SILAM keeps 48 h of history. RAQDPS only reaches back to its latest run, so
a window crossing the run time uses SILAM before it, per hour.

### Response speedups across the request path (2026-11-06)
Handler median on local stores (9 locations, 15 rounds, warm): default 13.2 -> 10.0 ms (p95
18.0 -> 11.3), `version=2` 12.9 -> 10.7, `extend=hourly` 14.1 -> 12.6, day_night 14.8 -> 13.1,
`exclude=summary` 11.2 -> 9.3. Responses identical to the previous code on 240 real requests
read from OVH. The biggest parts: no AQI below version 2 (1.5 ms), current AQI reused from
the hourly AQI when clipping changed nothing (0.6 ms), and a distance bound on the 2-D grid
lookup (up to 5.7 ms outside the RAQDPS domain). In the TESTING stages almost all of a
request's ~200 ms is the synchronous alert reads, through zarr-python on S3.

### Daily aggregation stacked, half-day work skipped, TESTING reads S3 directly (2026-11-06)
`_aggregate_stats` stacks each window's hours into a (day, hour, variable) array instead of
looping over days, and the half-day aggregates, display values and rounding only run when
the day_night block is returned. Daily section on 45 real requests (9 locations x 5 query
types, stores read from OVH RBX): default 3.34 -> 2.28 ms, `exclude=summary` 1.91 -> 0.85 ms,
day_night 5.42 -> 4.87 ms. Output is identical, sums included to the last bit (24k fuzzed
windows). An hour with a NaN precip type now counts as dry; it used to end up as a NaN day
type, which fails when cast to an index.

The TESTING stages read the bucket named by `REMOTE_PATH` directly; the
`api.pirateweather.net/files` proxy path is gone (it had no v31 stores).

### Daily sleet is reported as sleet (2026-11-04)
`select_daily_precip_type` overwrote the day's hourly type with the dominant accumulation
before its "preserve sleet" check ran, so every ice-pellet day read as freezing rain.

### `clipLog` only blanks the implausible values (2026-11-04)
One value past the implausible limit (125 % of the maximum) blanked every value above the
maximum, including ones that should clamp; the minimum side did the same. A NaN anywhere
in the array also turned the whole check off, so 1000 C would clamp to 60 C. Each value is
now judged alone.

### Canada vs US source selection follows the border (2026-11-04)
`is_in_canada` was a lat/lon box holding much of the northern US, so Seattle, Chicago,
Detroit, Boston and Minneapolis took Canadian models first. It now uses the point's time
zone (`CANADA_TIMEZONES`), whose polygons follow the border: right where a nearest-city
lookup is wrong (Windsor, Northwest Angle, Alert) and already looked up on every request
(about 2 us; the extra TimezoneFinder was avoided by sharing one). Open water, in `Etc/`
zones, takes the nearest city's country (about 20 us, cached per point). The box still
skips both elsewhere. `is_in_north_america` now includes the western Aleutians, out to
Attu across the antimeridian, by their zones, and still excludes Chukotka and Midway.
### `pytz` replaced with `zoneinfo` (2026-11-04)
`tzdata` replaces the `pytz` pin, and alone loads all 444 zones. Two behaviours to remember:
two times in one `ZoneInfo` subtract by wall clock (so `baseTimeOffset` uses timestamps), and
adding days moves the wall clock. Across 10,617 cases at every 2024-2026 transition in 30
zones, the time grids and day buckets are unchanged except in two places, both now correct:
- **Lord Howe's 30 minute DST start:** the current row was an hour behind for one hour.
- **`offset` in negative-DST zones (Dublin, Casablanca):** 6 cases matched the true offset.

Still open, and the same as before: `offset` reads UTC as local time, so it can be an hour off
around a change (about 600 of those cases), and Lord Howe's current row can miss the current
time on its other transition days (63 cases).

### `interp_time_take_blend` blends in the data's dtype (2026-11-04)
float64 weights promoted every float32 value to float64 and cast back. Casting the weights
took a GDPS-sized block with random values from 1.99 s to 1.37 s on one thread (the earlier
measurement on a real-shaped block was 1.00 s to 0.43 s). Values change by float32 rounding
only (at most 3e-5 K and 0.008 Pa), so ingests using the helper stop being bit-identical to
their previous output.

### Summary and source decisions (2026-11-04)
- **Dry weeks mention notable conditions.** A condition on at least two days replaces
  "No precipitation", ranked as in the daily summary (whole window, then earliest start,
  then visibility ahead of wind) and read from the days' summaries, because every Dark
  Sky visibility icon is `fog`. Cairo: "Misty today through Friday, with highs ...". The
  week keeps one condition, and the icon follows it.
- **Today's summary covers the rest of the day.** `calculate_day_text(partial_day=True)`
  starts at the current hour, and a condition filling it lasts "for the rest of the
  day" (`for-rest-of-day`, new in pirateweather-translations 2.1.0). Today's icon comes
  from the same call, so it follows too, and today can now join a rainy run in the weekly
  text. Half-day summaries still cover their whole window.
- **GEPS and REPS use mixed** when several channels are strong. The accumulation split
  and the daily text only knew the four types, so these hours count as ice accumulation
  (as they did when labelled sleet) and read as mixed. The minutely columns have no mixed
  category, so they stay sleet there, and an "expected" sentence still says sleet.
- **ECMWF feeds hourly humidity**, in its usual place ahead of GFS. Only seen in the
  offline test so far: the local ECMWF store predates its humidity column.

### DST-day midnight and overlapping model exclusions (2026-11-04)
`base_day` kept the UTC offset in force at request time, so after the clocks changed
the local day started an hour off (New York 2024-03-10 at 23:00 the evening before). It
now localizes midnight itself, which matches `day_array_grib`. Time machine `days`
requests also return the day's real hours; against ERA5, New York returns 23 hours on
2024-03-10 and 25 on 2024-11-03, and an explicit `hours=24` stays 24. Also fixed:
- **Clocks changing at midnight:** `localize_day_start` takes the first midnight where it
  happens twice (Havana, which lost its first hour) and the moment of the jump where it
  never happens (Beirut).
- **The repeated hour:** `base_hour` keeps the current UTC offset, so the first 01:30 of
  a fall-back day no longer starts the forecast an hour ahead.
- **Single-day time machine dailies** counted the grid rows past the end of the day:
  next midnight on a normal day, so lows and gusts could land on the next day. This
  changes daily values on every such request (New York 2024-06-15: wind 3.33 -> 3.30,
  precipIntensity 0.1153 -> 0.1201), and leaves the hourly block unchanged.

`exclude=aigfs` / `aigefs` no longer drop GFS / GEFS. Only those two flags moved to
whole tokens: the others have no overlap, and source names such as `hrrr_0-18` still
exclude HRRR through the substring check.

### Wind direction checked across the models (2026-11-04)
Checked against real GRIBs (2026-11-04 12z/18z) and through the API per model. HRDPS
`WDIR` is earth-relative even though its GRIB flag says grid-relative (median 0.08 deg
from rotated U/V, 24 deg from raw U/V), so it must not be rotated. GDPS, NBM, HRRR and
RRFS were right. Three bugs fixed:
- **RTMA, RTMA-RU, URMA were effectively unrotated.** `earth_relative_wind_components`
  read Herbie's index x/y as metres, so every point used the convergence at 25 N, 95 W:
  median 7 deg off, up to 21 deg, with correct speeds. It now uses the closed-form
  Lambert rotation: a real RTMA-RU ingest's store is 0.04 deg (median) from wgrib2's
  `-new_grid_winds earth`. Published RTMA and URMA history stores keep the old winds
  until they age out or are rebuilt.
- **Daily `windBearing` was an arithmetic mean**, up to 177 deg off when winds sat either
  side of north. Now a circular mean of unit vectors (57 us per request). Against the
  same commit and stores across 16 locations, it is the only field that changed.
- **NBM and GDPS 3-hourly leads, and HRDPS/GDPS `currently`, blended direction through
  south** (340, 227, 114, 0). Both now take the shorter arc, which leaves directions that
  do not cross north unchanged but for float rounding (3e-5 deg). Not yet run through a
  full NBM or GDPS ingest; the fix reaches their stores at their next cycles.

### ECMWF IFS 0.1° open data, with humidity stored at ingest
[pirateweather#697](https://github.com/Pirate-Weather/pirateweather/issues/697). The IFS
ingest reads the 9 km feed through Herbie 2026.10.0 (`resolution="0p1"`); AIFS has no
0.1° feed and stays at 0.25°, so the API looks up the two grids separately and skips an
IFS store still on the old grid. 2 m relative humidity is now a store column. Measured
on the 2026-11-04 00Z run with 4 write workers and a cold history: 3,514 s, 8.9 GB peak,
store 17 GB in 706 files (3.0 GB before), 4 history stores 1 GB. Point read through the
API context 0.46 ms first touch against 0.33 ms at 0.25°, 0.04 ms cached for both. Stored
values match the source GRIB within bit-rounding. Stored humidity has the same 4-bit
rounding as GFS (4 % steps above 64 %), so current humidity can differ from the old
per-request value by up to about 2.8 points.
The 240 h ensemble file reaches ECMWF about 09:40 after a 00Z run, AWS 09:56 and Google
12:27, hence the 10:30/22:30 UTC schedule. Publishing keeps two versions, so `Prod` grows
by about 28 GB: check the disk before deploying. Deploy the API before the ingest: an
old API would read the new store at the wrong grid cell.

### LibreWXR radar nowcast for the minutely block (2026-11-02)
`ingest/LibreWXR_Local_Ingest.py`, run by LibreWXR's nowcast hook, composites LibreWXR's latest radar frame (T+0) and
its nine 10-minute nowcast steps onto one global 0.04 degree grid as composite
reflectivity (`REFC_composite`, dBZ; NaN outside radar coverage, 0 below
`REFC_THRESHOLD`) and publishes `LibreWXR.zarr` with `write_final_zarr` /
`publish_ingest_run`, local disk or S3 like every other ingest. The API reads it
like any model (`FORECAST_STORES`, `exclude=librewxr`) and the minutely builder
lets it override every model's intensity minute by minute wherever it is finite;
precipitation type still comes from the models (HRRR sub-hourly temperature for
untyped echoes, else rain). `currently` follows minute 0.
- LibreWXR trial on this host: `/home/ubuntu/LibreWXR` (v0.8.0), run from
  `/home/ubuntu/librewxr-run` with the cache on `/mnt/disk/data/librewxr/cache`,
  9 nowcast frames; pipeline ~7 GB RSS, one cycle 60 s, cache ~5 GB.
- Export: composite 11 s (one thread), peak 3.5 GB; the write dominated at 5-cell
  chunks (1.6 M chunks, 530 s) and takes 36 s at 25-cell chunks, still one sub-KB
  chunk per point read. Store ~60 MB per cycle, old versions pruned by `publish_store`.
- Only the radar extrapolation is exported, never LibreWXR's NWP blend; the
  blend weight is not stored. RRQPE's display offset is undone; a saturated pixel
  (95.5 dBZ) becomes NaN.

### No read-time interpolation of missing values (2026-11-02)
`zarr_read` no longer fills interior NaN runs along time (`has_interior_nan_holes` /
`_interp_row` are gone): a value the model did not give stays NaN and the merge takes
the next source for that hour, as URMA and DWD MOSMIX already did.
- **Scan first:** ~2000 points per current Prod store; only three had interior holes.
  - GDPS CAPE/CIN (48% of points): MSC bitmap-masks them where no parcel is
    buoyant (40-50% of the globe). They stay NaN; GFS fills those hours.
  - AIGFS APCP (100%): an ingest bug. wgrib2 writes the 24 h multiples as
    `0-1 day acc fcst`, which the `hour`-only match skipped, so APCP was missing
    at every 24th lead and the reader drew a line across it. Fixed in the match.
  - RTMA `tp`: one point, one 2-hour gap.
- No store had an hour missing for every variable.

### Summaries: lists of three or more days or periods
The weekly and next-24-hour summaries joined three or more days or periods with nested
`and`s ("Rain today and Friday and Sunday and next Wednesday"). They now use the
translations' `list` template (`_day_range_phrase` in `API/PirateWeeklyText.py`,
`_get_time_phrase` in `API/PirateDailyText.py`): "today, Friday, Sunday and next Wednesday".

### Weekly summary: steady temperatures
`calculate_temp_summary` always picked rising, falling, peaking or valleying, even for a
flat week (Honolulu highs 26/26/25/26/26/25/26/26 °C). When the week's highs span at most
`STEADY_TEMPERATURE_RANGE_C` it now returns `temperatures-steady` with the rounded mean.

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

### AIGEFS download: one request per file, adaptive NOMADS pacing (2026-10-23)
AIGEFS fetched each file through Herbie, which sent two HEAD probes before every
GET; at 5 s pacing that was ~13 s per file and ~5 h per run. It now GETs each
member file straight from its NOMADS URL, and NOMADS pacing starts at 1 s and
doubles (to 60 s) on a rate-limit redirect, 429, 503 or timeout, easing back
afterwards. One member took 41 s live (was ~9-10 min), so ~20 min for all 30.
The AWS EAGLE ensemble is not a mirror (control-member precipitation r = 0.78
against NOMADS), so NOMADS stays the only source.

### History stores opened concurrently (2026-10-24)
The history merges opened each store, each array and each store's time one
round trip at a time from S3. `open_history_arrays` now opens the stores
`history_open_concurrency` (32) at a time and fetches each store's array
metadata in one concurrent call, and the time reads run concurrently too
(`map_history_stores`). Measured on the e2e history stores: GFS 48 stores
6.4 s + 0.9 s (was ~13 min across opens, per-variable array opens and time
reads), RTMA 48 stores 2.2 s + 1.5 s (was ~100 s), URMA 283 stores 5.3 s +
2.0 s (was ~7 min before its merge started, plus its time reads). Reading the
history data itself (GFS ~9 GB at ~3 MB/s) is unchanged.

### History stores: one shard per variable, downloaded before the merge (2026-10-24)
The merges read history straight from S3, one ~60 KB chunk per request with
~12 in flight, at 3-4 MB/s (OVH answers each GET in 100-400 ms): GFS spent
~48 min reading 9 GB, URMA ~20 min reading ~17 GB. History stores now give each
variable a single shard (`shard_whole_variables`; GFS ~30 objects per store
instead of ~3,600) and `download_history_stores` copies them to local disk with
`history_download_concurrency` (64) requests in flight before the merge.
Measured end to end (sharded copies of the e2e history, warm runs):
- GFS 3,167 -> 1,027 s: 48 stores (2,928 objects, ~9 GB) in 77 s.
- URMA 1,415 -> 1,097 s: 283 stores (5,943 objects) in 152 s; merge 703 s.
- RTMA 291 -> 235 s: 48 stores (1,104 objects) in 28 s.
- ECMWF and SILAM unchanged (few stores).
Unsharded stores written before this still work (64 in flight got ~14 MB/s
on them) and are replaced as the window moves on. The local copy adds working
disk: ~9 GB for GFS and ~17 GB for URMA.

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
