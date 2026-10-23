# Storage, Publishing and Read-Path Benchmarks (October 2026)

Measurements taken on the `v3` branch to choose the final-store layout, the
publish target and the API read path. Everything below was measured, not
estimated, unless marked otherwise.

## Summary

- **The read path is bound by zarr-python, not storage.** A warm point read
  through zarr-python costs ~250–450 µs, of which fetching the chunk is
  2–11 µs and Blosc decompression ~34 µs. The ~16 model reads in one API
  request serialize under the GIL, so a request costs their sum: 5–8 ms.
- **Reading the same files without zarr-python is 10–20× faster.**
  tensorstore, a Rust `zarrs` extension, or a minimal custom reader serve a
  16-read request in 0.34–0.6 ms warm and 2–3 ms cold, against 6.5–16 ms
  and 12–25 ms through zarr-python. All of them work best on sharded stores.
  Under 4-worker load, tensorstore on 100×100 shards served 1,250–1,530
  requests/s against ~225–240 for today's zarr-python directory-store setup.
- **Publishing is bound by file count.** HRRR is 76,321 files unsharded and
  199 with 100×100 shards. Small files publish at ~240 files/s to the NFS
  share and ~500 objects/s to OVH S3; large files move at 200+ MiB/s.
- **OVH S3 cannot serve requests directly**: 100–400 ms time to first byte on
  every GET.

**Test setup:** 4-core, 30 GB VM. The local ext4 disk measured NVMe-class
(0.12 ms per 4 KiB read at queue depth 1, ~61k IOPS at queue depth 32). Read
tests used the real HRRR final store (20 variables × 18 hours × 1060 × 1800,
5×5 chunks, Blosc zstd) with zarr 3.4.1 on Python 3.14. "Cold" means the
page, dentry and inode caches were dropped before the request.

## 1. End-to-end ingest (all models to OVH S3, production defaults)

| Model | Wall | Peak memory | Final store | Bound by |
|---|---|---|---|---|
| GFS | 7,625 s | 23.8 GB | 115,681 obj, 9.5 GB; history 48 stores, 8.8 GB | S3 history read (~73 min at ~2 MB/s) |
| URMA (cold start) | 5,664 s | 7.7 GB | 150,401 obj, 5.8 GB; history 283 stores, 17.6 GB | serial history build, S3 history read |
| GDPS | 2,319 s | 6.7 GB | 115,681 obj, 8.5 GB | ~6 MB/s download, single-core GRIB conversion |
| NBM | 2,038 s | 24.0 GB (19.6 GB with 100×100 interpolation tiles) | 416,807 obj, 4.6 GB | memory; 17 min upload |
| RTMA (cold start) | 1,355 s | 3.3 GB | 150,081 obj, 1.1 GB; history 48 stores, 3.0 GB | round trips, object count |
| HRDPS | 1,088 s | 4.6 GB | 131,065 obj, 1.4 GB | download, single-core conversion |
| ECMWF | 1,079 s | 14.1 GB | 115,681 obj, 3.0 GB | final-write memory |
| FMI_Silam | 1,030 s | 6.7 GB | 64,801 obj, 2.3 GB | download (12 GB pulled) |
| GEFS | 838 s | 8.9 GB | 115,681 obj, 1.2 GB | download |
| ECMWF_AIFS | 690 s | 7.7 GB | 115,681 obj, 2.0 GB | balanced; ~4 min upload |
| WMO_Alerts | 563 s | 5.9 GB | 133,633 obj, 3 MiB | object count (331 s upload) |
| DWD_Mosmix | 477 s | 16.6 GB | 41,482 obj, 32 MiB | single-core parsing, memory spike |
| REPS / GEPS | ~410 s | < 2 GB | 35k / 11k obj | download, conversion |
| AIGFS / HRRR / SubH / HRRR_6H | 259–321 s | 2–5 GB | 76k–116k obj | upload object count |
| RAQDPS / RTMA-RU / NWS_Alerts | 66–150 s | 2–5 GB | small | — |
| AIGEFS | stopped at 3 h test limit | 0.9 GB | — | NOMADS rate limit (~5.2 h for 30 members) |

- **Totals:** final stores 2.09M objects, 40.4 GiB (298 GiB uncompressed,
  7.2× on average); history stores 0.49M objects, 29.5 GiB.
- **VM sizing:** 32 GB RAM covers every model (NBM and GFS peak ~24 GB);
  4 cores suffice, as no model averaged more than ~1.5 cores.
- **Recurring bottlenecks** (all logged in `TODO.md`): upload object count,
  reading history from S3, and serial download plus single-core conversion
  for the Canadian models.

## 2. Storage layouts (HRRR)

| Layout | Files | Size | Build from the directory store |
|---|---|---|---|
| Directory store (current) | 76,321 | 596 MiB (730 MiB on disk) | — |
| Shards 50×50 / 100×100 / 200×200 | 793 / 199 / 55 | 598 MiB | ~18 s write |
| Single-file container prototype | 2 + `CURRENT` | 596 MiB + 2.2 MB index | 1.5 s from cache |
| LMDB | 1 | 731 MiB (+23%, page rounding) | 14.9 s |
| Zip, uncompressed | 1 | 603 MiB | 4.4 s |
| Icechunk, 100×100 shards | 205 | 598 MiB | 20 s; commit instant |

Compression on HRRR: per-variable bitround + Blosc zstd3 is 451 MiB against
596 MiB without bitround. lz4 was 594 MiB with no read gain through zarr.

## 3. Where a warm point read spends its time

| Step | Time |
|---|---|
| Container index lookup / `pread` | 0.3 / 1.9 µs |
| Directory store open + read + close | 10.5 µs |
| Blosc decompression (14.7 KiB out) | 34 µs |
| `asyncio.to_thread` hand-off (zarr `LocalStore` does one per chunk) | 56 µs |
| **zarr `arr[:, :, y, x]`**: directory / LMDB / container / 100×100 shards | **448 / 329 / 307 / 805 µs** |

## 4. Read path: 16-read requests on local disk

Medians in ms, one process, each reader using its natural concurrency
(zarr-python: `asyncio.gather` of `asyncio.to_thread`, as the API does;
tensorstore: 16 futures; Rust: one call with the GIL released).

| Reader | Store | Single read | **Warm request** | **Cold request** |
|---|---|---|---|---|
| zarr-python, default / fused pipeline | directory | 0.28–0.46 | 8.5 | 12.8 |
| zarr-python, fused | 100×100 shards | 0.54 | 16.1 | 15.8 |
| zarr-python, zarrs pipeline (best option) | 100×100 shards | 0.24–0.33 | 6.5–8.0 | 13.6–14.3 |
| zarr-python, fused | LMDB / container prototypes | 0.31 / 0.25 | 4.8 / 5.4 | — |
| zarr-python | Icechunk / obstore backend | 1.48 / 1.18 | 15.9 / 20.9 | 23.8 / 18.6 |
| **tensorstore** | directory | 0.09 | 0.52 | 6.2 |
| **tensorstore**, cache + `recheck_cached_data="open"` | 100×100 shards | 0.13–0.14 | **0.55–0.58** | **2.8** |
| **Rust `zarrs` extension**, parallel, shard-index cache | 100×100 / 200×200 shards | 0.06 | **0.34–0.41** | **2.5–2.9** |
| Rust `zarrs` extension | Icechunk, 100×100 shards | 0.19 | 2.1 | 4.7 |
| **Custom Python reader**, sequential | 100×100 shards | **0.035** | 0.55–0.59 | 3.1–3.3 |
| **Custom Python reader**, 16 threads | 100×100 shards | 0.035 | 0.76–1.07 | **1.9–2.0** |

- Shard size (50, 100 or 200) makes little difference to any reader.
- zarrs-python options (Rust thread count, direct I/O, file-handle cache)
  move results by about ±20%; direct I/O is worse warm.
- tensorstore only caches data when `recheck_cached_data` is not `True`; the
  default revalidates on every read. Use `"open"` and reopen per generation.
- Every reader matched zarr-python exactly on 50–305 test points.

## 5. Serving under load (4 worker processes, 16 reads per request)

Requests/s, with p99 latency in brackets. tensorstore reads are awaited as
futures inside each worker's event loop; zarr-python reads use
`asyncio.to_thread` as the API does today. The tensorstore cache is 256 MB
with `recheck_cached_data="open"`; "capped" also limits tensorstore's
`data_copy_concurrency` and `file_io_concurrency` to 2 per worker.

| In flight | Directory, zarr-python (today) | 100×100 shards, zarrs pipeline | tensorstore, no cache | **tensorstore, cache** | tensorstore, cache, capped |
|---|---|---|---|---|---|
| 1 | 97 (23.5 ms) | 173 (7.9 ms) | 600 (3.1 ms) | **804 (1.9 ms)** | 731 (2.0 ms) |
| 4 | 224 (40.4 ms) | 353 (19.7 ms) | 1,150 (7.1 ms) | 1,250 (6.9 ms) | **1,306 (5.9 ms)** |
| 8 | 225 (59.9 ms) | 333 (36.8 ms) | 1,109 (14.1 ms) | 1,210 (13.1 ms) | **1,414 (11.4 ms)** |
| 16 | 240 (89.9 ms) | 421 (50.8 ms) | **1,471 (25.0 ms)** | 1,434 (23.7 ms) | 1,403 (22.3 ms) |
| 32 | 235 (239 ms) | 320 (146 ms) | 1,257 (50.7 ms) | 1,523 (41.6 ms) | **1,529 (37.2 ms)** |
| 1, cold | 91 (23.2 ms) | 182 (8.2 ms) | 673 (2.3 ms) | 682 (2.4 ms) | **739 (2.3 ms)** |
| 16, cold | 236 (107 ms) | 321 (72.1 ms) | 1,152 (25.4 ms) | **1,320 (29.0 ms)** | 1,257 (23.0 ms) |

- **tensorstore serves 5–6× the requests of today's setup** (1,250–1,530 vs
  ~225–240 req/s) and 3.5–4.5× the best zarr-python option, with 4–8× lower
  tail latency. A lone request drops from 9.8 ms to 1.2–1.6 ms.
- CPU rises to ~97% with tensorstore (from ~86%): the cores are finally doing
  the work rather than waiting on zarr-python's single event loop.
- The cache helps most at low concurrency (1.59 → 1.23 ms median for a lone
  request); capping tensorstore's thread pools helps at 4–8 in flight. The
  differences between the three tensorstore variants are mostly within the
  ~10–20% run-to-run noise.
- Earlier zarr-python-only runs: throughput topped out around 300–420 req/s
  regardless of store (fused ~290–335, 100×100 shards under zarrs ~390–417).

## 6. Remote storage

| Target | Publish HRRR | Warm 16-read request | Cold 16-read request |
|---|---|---|---|
| Local disk | — (needs a sync) | 0.34–0.6 ms fast readers; 5–8 ms zarr-python | 2–3 ms fast readers |
| **NFS share, few-file stores** | **2.6–6 s (~200 MiB/s)** | 0.71 ms tensorstore; ~5.5 ms zarr-python (same as local) | 6.5–7.9 ms |
| NFS share, directory store | 312–343 s (~240 files/s; threads don't help) | 7.8–10.5 ms, p99 up to 32 ms | 13–19 ms |
| OVH S3 (High Performance) | ~2.5 min (~500 obj/s) | **944 ms** zarr-python | — |

- **S3:** single GET p50 242 ms; 16 concurrent GETs 376 ms. Almost all of it
  is server-side time to first byte (100–400 ms, HEAD requests included),
  with TCP connect ~1 ms and TLS ~10 ms from a VM 0.3 ms away. OVH's launch
  post claims time to first byte slightly better than AWS S3, so this looks
  abnormal and is worth raising with OVH.
- **NFS:** free when warm for few-file layouts; ~1.3 ms per uncached chunk.
  On a directory store every file open revalidates with the server, ~0.28 ms
  per chunk even when cached.

## 7. Memory

- Private memory per worker per store: directory, shards and LMDB < 1 MiB;
  container index 9.3 MiB (HRRR), ~51 MiB (NBM); zip 47 MiB (HRRR), ~250 MiB
  (NBM).
- LMDB's mapped pages count toward each worker's RSS, so monitoring
  overstates its real memory.
- Shard indexes for all stores total ~35 MB: the floor for a tensorstore
  cache. 64–256 MB per worker is a reasonable start; tune it by replaying real
  request coordinates.
- All compressed final stores (~41 GiB) fit in the page cache of a 64 GB
  host; uncompressed they would need ~300 GiB.

## 8. Lifecycle findings

- LMDB: closing a store while another thread is mid-read segfaults the
  process. Swap by dropping the reference instead.
- Container prototype: generation swaps under load were atomic (1–2 ms
  publish, 0 read errors); readers on the old generation finished cleanly.
- Directory stores with the current symlink sync: reads follow the symlink
  live but array metadata is cached from startup, and deleting the old run
  mid-request silently returns NaN.
- Every reader (zarr-python, tensorstore, zarrs) needs the same pattern:
  open each generation by its versioned path, swap the reference, and delete
  the old generation only after a short delay.

## 9. Recommendations

1. **Layout:** sharded zarr stores. 100×100 for the regional models, smaller
   for the long global stores, because a whole shard is held in memory while
   it is written (GFS at 100×100 is ~590 MB per shard).
2. **Read path:** tensorstore with one shared `Context` per worker, a
   64–256 MB cache, `recheck_cached_data="open"`, and a reopen per
   generation. A Rust `zarrs` extension is ~0.15 ms faster per request if a
   Rust build stage is acceptable.
3. **Publish:** few-file stores to the NFS share (seconds) or S3. Serve from
   local disk, or from NFS with enough RAM to cache the working set. Do not
   serve from OVH S3.
4. **Hosts:** API hosts with enough RAM for the page cache to hold all
   compressed stores (64 GB or more); ingest VMs with 32 GB.

## Not yet measured

- The Rust `zarrs` extension and the custom reader under the multi-process
  serving benchmark (tensorstore was).
- Cache sizes against real request coordinates.
- Uncached reads on the production NVMe hosts.
- The production maturity of zarrs-python (0.2.3 at the time of testing).

The benchmark scripts were ad hoc and are not part of the repository.
