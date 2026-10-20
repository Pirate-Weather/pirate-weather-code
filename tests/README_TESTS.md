# Pirate Weather tests

This folder holds a deliberately small test suite. Each test exists to protect
behaviour that affects forecast output, a data-layout or API contract, a major
fallback path, or a real past bug. Every test function's docstring says what goes
wrong if the test is removed. The standard for adding tests is section 27 of
[`AGENTS.md`](../AGENTS.md) ("Keep Tests Minimal, Broad, and End-to-End").

| Folder | Tests | Network | Runs by default | Runs in CI |
|---|---|---|---|---|
| `tests/api` | 164 | no | yes | yes (`Tests` workflow, API requirements) |
| `tests/ingest` | 51 | no | yes | yes (`Ingest Test` workflow, ingest requirements) |
| `tests/live` | 25 | **yes** | no, opt-in with `PW_RUN_LIVE_TESTS=1` | no; manual `Live tests` workflow only |
| `tests/helpers` | (shared builders) | | | |

## Running the tests

```bash
source .venvs/ingest-test/bin/pirate-env.sh   # after scripts/setup_ingest_test_env
python -m pytest                               # tests/api + tests/ingest, ~8 s, offline
ruff check . && ruff format --check .          # CI pins ruff==0.16.10 (requirements-lint.txt)
```

`pytest.ini` sets `testpaths = tests/api tests/ingest`, so a bare `pytest` never
touches `tests/live`. CI installs the API and ingest requirements in separate
environments. A test in `tests/api` must not import anything that needs the ingest
dependencies (Herbie, cfgrib, ecCodes), and a test in `tests/ingest` must not need
the API ones (FastAPI, timezonefinder). See [`CONTRIBUTING.md`](../CONTRIBUTING.md)
for the per-suite install commands.

### Live tests

```bash
PW_RUN_LIVE_TESTS=1 python -m pytest tests/live
```

- **Agents and routine checks should not run these.** Run them when asked, before
  a release, or when a change touches live data access (S3 stores, ingest scripts,
  the production comparison).
- **Resources:**
  - about 10 GB of RAM: the shared `live_client` fixture loads every S3 store once per session;
  - 5 to 10 minutes;
  - several GB free in `TMPDIR` for the live ingest runs. If `/tmp` is a small
    tmpfs, point `TMPDIR` at a real disk, or the HRRR_6H ingest fails with "Disk
    quota exceeded".
- **Credentials:** taken from the environment, falling back to a repository-root
  `.env`, which is only read for live runs. Supported keys: `PW_API`, `AWS_KEY`,
  `AWS_SECRET`, `GRIBSTREAM_API_KEY` and `s3_bucket`. A test whose data or
  credentials are missing skips with a reason; run with `-rs` to see them.

| Variable | Needed by |
|---|---|
| `PW_API` | everything that uses `live_client` (S3 proxy and production API) |
| `GRIBSTREAM_API_KEY` | the four GribStream comparisons |
| `PW_URMA_DATA_DIR` (default `/mnt/nvme/data/ProdSync`) | `test_compare_urma_gribstream.py`: local `URMA_Hist.zarr.zip` + `GFS.zarr.zip` |
| `PW_URMA_OUTPUT_ROOT` (default `/mnt/nvme/data/Prod/URMA`) | `test_urma_output_live.py`: output of a local URMA ingest |
| `wgrib2_path` or `wgrib2` on `PATH` | `test_hrrr_6h_ingest.py` |

- **Flakiness:** live results depend on upstream data. If a model archive is
  replaced on S3 during a run, reads fail with S3 `PreconditionFailed` and a
  regional model can drop out of `flags.sources` for that run. Re-run before
  treating a live failure as a regression.

## What the tests cover

### `tests/api`: forecast response logic (offline)

These tests drive the real response-building functions with small synthetic
arrays laid out the way the merged Zarr reads produce them. None of them reads
real model data. The live suite is the only place the full request path runs
against real stores.

**Request handling**

| File | Protects |
|---|---|
| `test_request_parsing.py` | Relative and invalid request times; `City,Country[,time]` locations; AI-model mode dropping CMC models; `exclude=` flags (URMA independent of GFS, #746); Time Machine `days`/`hours` sizing and range validation. |
| `test_offline_geocode.py` | Country-name normalisation; unknown places returning 404 instead of a NaN location. |
| `test_filtering_queryparams.py` | The MCP `blocks` and `*_indices` parameters, which return 400 on bad input; version/`include` gating of the air-quality fields (an API-compatibility contract). |
| `test_mcp_public_url.py` | MCP server: public-origin rewriting behind the proxy, `tmextra` forwarding, `get_forecast` response shaping, City,Country forwarding, and no tool exposing a `version` argument (#641). |

**Reading model data**

| File | Protects |
|---|---|
| `test_grid_indexing.py` | Point lookup on non-trivial grids: CMC rotated-pole corner and out-of-domain rejection; RAQDPS spherical nearest cell near the antimeridian and poles; ERA5 slice timestamps, precip-type rounding and the 3x3 precip-probability neighbourhood with longitude wrap. |
| `test_zarr_reader_max_square.py` | The 3x3 max read for air quality shrinks at grid edges instead of wrapping (#682). |
| `test_era5_cache.py` | The ERA5 read-through disk cache reuses chunks and stays within its size limit (#696); `SKIP_ERA5` disables ERA5 entirely. |
| `test_zip_sync.py` | Every store the API opens is listed in `API/zip_sync.sh` (#730/#742 bug class), and the script parses. |

**Source selection and merging**

| File | Protects |
|---|---|
| `test_regional_source_priority.py` | Regional source order (Canada box / North America / rest of world, with and without `include=aimodels`) for currently and hourly, following `docs/forecast-source-selection/README.md`; Canadian UV and ozone ahead of GFS (#734); ECMWF ahead of DWD in North America; AIGEFS first for precipitation probability in AI mode. |
| `test_minutely_sources.py` | Minutely fallbacks: no sources gives a dry block; DWD without RR1c falls back to ECMWF; AI models override HRRR-SubH/NBM; Canada uses REPS probability with NBM amounts; AIGEFS precip type comes from temperature. |
| `test_data_inputs.py` | Short ERA5 series padded to `num_hours` (#624); Canadian model fields stacked in priority with unit handling; Kelvin to Celsius for every model ingested in Kelvin. |
| `test_dwd_mosmix_hourly.py` | DWD cloud % becomes a fraction, visibility stays in metres, and present-weather codes map to a precip type. |
| `test_urma_response.py` | URMA as the recent Time Machine source: alignment by timestamp, per-hour GFS fallback, station versus sea-level pressure, recomputed feels-like on URMA hours, gaps kept for GFS to fill, read only when recent and in-domain (#746). |
| `test_forecast_merge_alignment.py` | Forecast-only model stores (no short-term history) are placed at their valid hours, HRRR/NBM are kept when they start after local midnight, and GFS fills the earlier hours. |
| `test_aq_timemachine_sources.py` | RAQDPS/SILAM are listed and used only when they cover the requested hours (#751/#752); RAQDPS is forecast-only, so recent Time Machine AQ comes from SILAM, including at half-hour UTC offsets (#684). |

**Building the response blocks**

| File | Protects |
|---|---|
| `test_hourly_block_outputs.py` | ERA5 precipitation probability converted from 0-100 to 0-1 (#664); temperature clipping (a slight overshoot is clamped, an absurd value becomes missing). |
| `test_current_conditions.py` | Current air quality interpolated between hours (#689); Canadian UV/ozone in currently (#734). |
| `test_daily_air_quality.py` | Daily AQI summarises the hourly AQI column, not a pollutant column. |
| `test_daily_precip_type.py` | Daily precip type follows the dominant accumulation (#582), "mixed" only with all three types; GEPS/REPS rate-to-type mapping; ice, sleet and mixed text and icons. |
| `test_dst_time_indexing.py` | Daily buckets follow local midnight, so DST days are 23 or 25 hours; multi-day Time Machine requests use one bucket per day (#624). |
| `test_polar_daynight_logic.py` | Sunrise/sunset fallback and the all-day/all-night flags for polar day and night (Longyearbyen). |
| `test_aqi_calculations.py` | One exact official-table value for each of the 11 AQI systems (catches breakpoint changes such as #686); NowCast and rolling-mean averaging; RAQDPS/SILAM per-pollutant source choice; smoke boundary-layer floor; float32 SILAM times at half-hour offsets (#684). |

**Summary text** (keys are rendered by `pirateweather_translations`)

| File | Protects |
|---|---|
| `test_thunderstorm_summary.py` | When rain text becomes "thunderstorm" (CAPE, CIN cap, low PoP, "possible" in currently); daily, 24-hour and weekly storm wording; daily storm strength ignores CAPE from low-PoP hours. |
| `test_minutely_thunderstorm.py` | Minutely "thunderstorm" wording keeps the timing for every sentence shape; inclusive CAPE threshold; no storm without precipitation. |
| `test_snow_accumulation_range.py` | Exact daily snow amount phrase for each error regime (range, "less-than", exact) and unit system, including the #510 tolerance. |
| `test_text_helper.py` | Golden values for the new-snow density formula behind every snow depth. |
| `test_daynight_forecast.py` | Day/night summaries: clear, cloudy night, pirate "possible" icons, the `MAX_HOURS` limit (#385), and three-period slices (#624). |

### `tests/ingest`: ingest helpers and output contracts (offline)

The `API/*_Local_Ingest.py` scripts keep their run-time code in `main()` behind
`if __name__ == "__main__"`, so tests import their functions directly. Model
downloads are replaced with fakes; nothing here downloads data. No offline test
runs a whole ingest end to end; only the live RTMA-RU and HRRR_6H tests do that.

| File | Protects |
|---|---|
| `test_ingest_output_layout.py` | Each ingest's `zarr_vars` order matches the index map the API reads it with in `API/constants/model_const.py` (HRRR, HRRR_6H, SubH, RTMA-RU, URMA, DWD, SILAM, RAQDPS). This is the most important ingest/API contract. |
| `test_urma_zarr_encoding.py` | URMA's hourly store is written in the format the historic merge reads back (Zarr v3, float32, chunking, names). |
| `test_historic_archive_extract.py` | Historic archives are validated on extract; incomplete ones are deleted so the next run rebuilds them (#644); `s3://` temp dirs are rejected (#742). |
| `test_ingest_utils_download_retry.py` | Herbie downloads retry a bounded number of times, ignore stale files from earlier runs, pace NOMADS requests and reject NOMADS 302 rate-limit redirects (#772). |
| `test_ingest_grib_utils_retry.py` | FastHerbie is rebuilt between retries so late-published hours are seen. |
| `test_nbm_download.py` | NBM downloads fail loudly when FastHerbie swallows worker errors (302, missing or empty file), and recover on retry. |
| `test_aigefs_download.py` | A missing AIGEFS member file is retried a bounded number of times, then aborts the ingest. |
| `test_ingest_utils_validate_grib_stats.py` | The wgrib2 `-stats` gate parses scientific notation and honours per-variable exclusions. |
| `test_ingest_utils_deaccumulate.py` | GEPS deaccumulation with uneven 3 h/6 h steps; GDPS/HRDPS energy-to-flux with negatives clipped. |
| `test_ingest_utils_interp_time.py` | Step-to-hourly interpolation in column batches (GEPS/GDPS/HRDPS/REPS), with nearest-neighbour for categorical variables. |
| `test_dwd_ingest.py` | DWD station gap rules: long gaps are nulled so the API falls back to global models; precip and precip type are filled; gridded gap fill uses nearest for precip type (#463). |
| `test_raqdps_ingest.py` | ECCC Datamart URL format and the Herbie template. |
| `test_wmo_alerts_ingest.py` | WMO alert catalog feeds (RSS, MeteoAlarm Atom, Canadian Atom) fetch each current CAP once; the empty-alert schema; the `WMO_Alerts.zarr` grid string contract; first-language-only CAP parsing. |
| `test_wmo_alerts_geocode.py` | Canadian many-geocode alerts are not duplicated (#439); geocode-only CAPs give one row per area; MeteoAlarm codes (including aliases) resolve to bundled polygons. |

### `tests/live`: external data (opt-in)

| File | What it does |
|---|---|
| `test_forecast_live.py` | Full forecast requests against the production S3 stores: CONUS, Canada and rest-of-world are plausible and served by the expected regional models (`flags.sources`); `blocks`/indices filtering; excluding every model returns 400 (#570). |
| `test_timemachine_dst_live.py` | Time Machine on the two US DST days must return a 23- and a 25-hour local day. **Currently a strict `xfail`**: it is the regression test for the "local midnight is off by an hour on DST transition days" bug in [`TODO.md`](../TODO.md). Remove the marker when that bug is fixed. |
| `test_compare_production.py` | Diffs local responses against the production API at 10 locations plus two Time Machine requests. It **never fails on data differences**; it emits `DiffWarning`s for a person to review before a release. |
| `test_compare_urma_gribstream.py` | A URMA-backed Time Machine day must match GribStream's independent URMA analyses (temperature, dew point, gust, visibility, station pressure). Needs local archives. |
| `test_urma_output_live.py` | The newest analysis in a locally produced URMA Zarr matches GribStream at several points. |
| `test_rtma_ingest.py`, `test_hrrr_6h_ingest.py` | Run the real ingest script as a subprocess on the latest NOAA data, check the output Zarr, and compare it with GribStream when a key is set. |

### `tests/helpers`

| Module | Use |
|---|---|
| `model_arrays.py` | `model_array(MODEL, n, temp=...)` builds an all-NaN merged array with named columns; `hourly_inputs(...)` calls `prepare_data_inputs` with every other source set to `None`. |
| `hourly.py` | `hourly_block_kwargs(num_hours, **overrides)` gives all-NaN defaults for `build_hourly_block`. |
| `text.py` | `make_hour` / `make_hours` build the SI hour dicts the summary-text functions take. |
| `urma.py` | URMA grid lookup and the GribStream request helper shared by the live comparisons. |

## Important notes

- **Known bugs are in [`TODO.md`](../TODO.md), not in the tests.**
  - No test asserts a bug's current, wrong behaviour.
  - No offline test is left failing.
  - When you fix a bug, add one regression test that would have caught it, then
    remove the entry.
  - The live DST test is the one deliberate strict `xfail`.
- **Writing a new test:**
  - Call real production code.
  - Pin exact values where the output is deterministic.
  - Put repeated cases in one `parametrize` with readable ids.
  - Write a docstring that names the risk, not the function.
  - Don't add tests that copy production logic, test numpy or the standard
    library, restate constants, or grep production source.
- **Offline means offline.** The default suite passes with networking disabled
  (`sudo unshare -n ...`), and nothing in `tests/api` or `tests/ingest` may touch
  the network or the wall clock. Anything that needs external data goes in
  `tests/live`.
- **Importing ingest scripts:**
  - Importing an `API/*_Local_Ingest.py` module has no side effects, but its
    configuration (paths, `tmp_dir`, `wgrib2_path`, retry counts) is read from the
    environment at import time. Use `monkeypatch.setattr(module, "tmp_dir", ...)`
    rather than setting env vars after import.
  - GFS and the CMC scripts call `load_dotenv(override=True)` at import time, so
    importing them loads a repository-root `.env`.
- **One test still reads production source.** `test_urma_zarr_encoding.py`
  evaluates URMA's `to_zarr(...)` call from the script source, because that call
  is written inline in `main()`. Pulling it into a helper function would remove
  the need.
- **Warnings are not suppressed globally.** Expect a Herbie deprecation and a
  Zarr consolidated-metadata warning. A new numpy `RuntimeWarning`, such as an
  all-NaN slice, is worth reading.
- **Known gaps (not covered by any offline test):**
  - a deterministic end-to-end request served from a small checked-in Zarr;
  - processing of a real sample GRIB file through an ingest;
  - the `flags.nearest-station` value and its unit conversion;
  - summary keys actually rendering through `pirateweather_translations`;
  - daily day/night/high/low hour selection;
  - layout checks for ingests that still define `zarr_vars` inside `main()`
    (GFS, GEFS, NBM, ECMWF, AIFS, AIGFS, AIGEFS, GDPS, HRDPS).
