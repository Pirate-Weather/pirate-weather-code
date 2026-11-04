# TODO

Known bugs found during the v3 test-suite review (2026-10-19). Each has been confirmed
by reading the code; none is fixed yet. When fixing one, add a single regression test
that would have caught it, then remove the entry.

## Bugs

## Improvements

## v3 goals

From the v3 planning post (pirate-weather-code discussion #773). Goals already done
there (forecast/historic split, test refactor, ingest standardization) are not listed.

### LibreWXR for maps, radar coverage and minutely precipitation
Plan in [pirateweather discussion #414](https://github.com/Pirate-Weather/pirateweather/discussions/414#discussioncomment-18934138):
use LibreWXR's global radar nowcast for minutely precipitation worldwide (likely
better than HRRR sub-hourly even inside its domain), and mirror its cache to S3 so
render nodes can run in several regions with the last good data if the pipeline fails.
- [x] Minutely store: `ingest/LibreWXR_Local_Ingest.py`, run by LibreWXR's nowcast
  hook once per nowcast, publishes `LibreWXR.zarr` (see Done, 2026-11-02).
- [ ] Upstream the hooks: branch `cycle-hooks` of the LibreWXR fork adds
  `LIBREWXR_HOOK_STATE_CMD` / `LIBREWXR_HOOK_NOWCAST_CMD` / `LIBREWXR_HOOK_LIGHTNING_CMD`
  (`src/librewxr/hooks.py`); submit it as a PR to JoshuaKimsey/LibreWXR.
- [x] Mirror the LibreWXR cache to object storage for render nodes: done inside
  LibreWXR as "snapshot sync" (fork branch `network-split`, on top of `cycle-hooks`,
  for LibreWXR issue #51): the pipeline publishes each `state.json` and the files it
  names to an fsspec URL (`LIBREWXR_SNAPSHOT_SYNC_URL`, S3 or `file://`), and
  `python -m librewxr.snapshot_pull` on a render host pulls them into its cache,
  rewrites the pipeline's absolute paths, writes `state.json` last and prunes the
  per-node caches. Measured on this host: ~1 GB raw per 10-minute cycle at full
  regions. Nothing is needed on the Pirate Weather side.
- [ ] Deploy: run LibreWXR's pipeline service from the `pirate-librewxr` image
  (`Docker/pirate-librewxr-dockerfile`: LibreWXR from the fork branch plus this
  repo's code; built and pushed to ECR Public by `build.yml`, build-checked by
  `sanity.yml`) with `LIBREWXR_HOOK_NOWCAST_CMD` and `LIBREWXR_SNAPSHOT_SYNC_URL`
  set. Switch `LIBREWXR_REF` in both workflows to an upstream release tag once
  the hooks and snapshot sync PRs are merged.

### Own ERA5 dataset
Only the needed variables, chunked for point reads, based on the Google ARCO pipeline
the API reads today (`ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, one whole-globe chunk per
hour and variable). Design so far: the 18 surface variables the API reads, 1940 to
present, `(variable, time, y, x)` with 120 h time blocks and 8 x 8 point tiles, one
shard per 120 h block; about 11-19 TB. Daily ERA5T appends (about 6 days behind) and a
monthly replacement with final ERA5 (3 months behind). Prototype one month first.
Replaces `init_ERA5` (`API/io/ZarrHelpers.py`) and `_load_era5_slice`
(`API/request/grid_indexing.py`), including the 20 GB disk cache.
`scripts/build_era5_point_test_store.py` now builds the int16 layout measured below
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
Still to do: daily ERA5T appends (recent weeks fall back to Google until then), and
open a week with one metadata read instead of two (KvStore read + `ts.open`).

Backfill estimate: ~4,500 weeks from 1940; each pulls 7.2 GiB from Google (20 source
variables x 168 hourly 2.2 MiB chunks), so ~32 TiB in total. The read took 35 s
(~190 MiB/s) on 2026-11-03 and 125 s on 2026-11-04 (Google or a busy test server; the
int16 encoding adds ~1 s per variable); write 15 s, upload ~6-10 s to RBX at the store's
1.1 GB. So ~1-2.5 min per week sequential, 3-8 days. With ~12 GB per worker, 6-8 weeks
in parallel fit a 128 GB VM; the download (~50 h at 190 MiB/s) is then the bound, about
2 days. Check the build VM's bandwidth cap and Google's sustained throughput first.

### ERA5: tune rounding and compression for storage cost
The one-week test store (`scripts/build_era5_point_test_store.py`, 2015-06-29, 18
variables, 10 x 10 inner chunks of the whole week, 100 x 100 shards) is 1885 MiB, which
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

### Move to OVHCloud
Including a Terraform setup script.

### DWD MOSMIX: cut the 16.4 GB memory peak
DWD is the only ingest that fits no 2-core instance up to 16 GB (one pod per job
in Kubernetes): it peaks at 16.4 GB against ~4.4 GB for the rest of the run
(2026-10-23 end-to-end run). The spike lasts a few seconds, while the gridded
stations are written to `_chunk.zarr`. `interpolate_dwd_to_grid_knearest_dask`
fills one dense NumPy array per variable (240 times x 720 x 1440 float32, ~1 GB
each) and wraps each in `da.from_array`, so all 12 are in memory together when
`to_zarr` runs, plus its rechunk copies. Fill and write one variable at a time
(or scatter the nearest-station values straight into a zarr per time chunk:
only cells within 50 km of a station are ever non-NaN) so the peak is about one
variable. Check the ~4.4 GB KML/DataFrame plateau too, then size the pod; the
target is under ~12 GB for a 2c/16 GB instance, or under 6 GB for 2c/8 GB.

### Set write threads explicitly for each job (Kubernetes)
Each write thread holds about one shard (up to ~150 MB, plus its bitrounded
copy and compression buffers), so peak memory scales with the thread count.
The defaults differ by ingest and some follow the machine:
- `os.cpu_count()`: AIGFS, AIGEFS, NBM. In a pod this is the node's core
  count, not the pod's CPU limit, so on a 16-core node NBM would write with 16
  threads and its memory would depend on where the pod lands.
- 64: RTMA, URMA. 12: GFS, ECMWF, SILAM. 4: ECMWF_AIFS. 2: the rest.
- Dask work outside `write_final_zarr` that is not wrapped in
  `dask.config.set(num_workers=...)` also defaults to one thread per node core.

Set `zarr_store_workers` (and `zarr_async_concurrency`) in every job's pod spec,
sized to its CPU and memory limits, and replace the `os.cpu_count()` defaults
with a fixed value so a missing setting cannot scale with the node. The
2026-10-23 peaks were measured on a 4-core VM with these defaults; re-measure
each job at its chosen thread count before fixing the pod's memory limit.

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
some fields use `clipLog`, others `np.clip`.

## Done

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

### Summaries: lists of three or more days or periods read awkwardly (2026-10-31)
Weekly (`API/PirateWeeklyText.py`, `calculate_summary_text`) and next 24 hour
(`API/PirateDailyText.py`, `_get_time_phrase`) summaries join separate days or
periods with nested `and`s, e.g. "Rain today and Friday and Sunday and next
Wednesday." The translations' `and` only uses a comma when its first argument
already contains one, and no template produces one, so there is no way to say
"today, Friday, Sunday and next Wednesday" today.
Needs a translations change first, e.g. a `list` template that takes the items
as one list and joins them the language's way, then use it for three or more
items here.

### Weekly summary: steady temperatures (2026-10-31)
`calculate_temp_summary` always picks rising, falling, peaking or valleying, even
for a flat week (Honolulu: highs of 26/26/25/26/26/25/26/26 °C reads as if the
temperature trends). Needs a translations key, e.g. `temperatures-steady`
("highs steady around $1"). Then use it when the week's highs span a small range
(e.g. 2 °C / 4 °F) and give the rounded mean.

New keys raise `ValueError` in translations that don't have them yet; the
`Translation.is_supported()` method can be used to switch over only once the
installed translations support a key.

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
