"""Backfill the ERA5 archive: build and publish every complete week in a date range.

Runs ``scripts/build_era5_point_test_store.py`` once per 1940-aligned week, in its
own process (a week peaks at ~11.4 GB, measured 2026-11-04), ``--parallel`` weeks
at a time. A week whose root ``zarr.json`` is already published is skipped, so a
rerun resumes where a stopped one left off. A failed week is retried
``ATTEMPTS`` times; the others carry on, and the exit status is 1 if any week
still failed. Each week's local store is deleted once it is published.

Only complete weeks are built: the last one ends on or before ``--end`` (Google
ERA5 ends at ``valid_time_stop_era5t``, about 5 days behind real time).

Usage:
    python scripts/backfill_era5_archive.py --start 2020-01-01 --end 2026-10-29 \\
        --upload pirate-era5 --work /data/era5 [--parallel 4]
    # S3 settings from s3_endpoint, s3_region, s3_storage_class, AWS_KEY/AWS_SECRET
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from API.constants.model_const import (  # noqa: E402
    ERA5_ARCHIVE_STORE_DIR,
    ERA5_ARCHIVE_WEEK_EPOCH,
)
from API.ingest_utils import open_remote, remote_object_exists  # noqa: E402

logger = logging.getLogger("era5_backfill")

BUILDER = REPO_ROOT / "scripts" / "build_era5_point_test_store.py"
# Google and OVH both fail transiently (timeouts, 5xx); a week that fails three
# times is reported rather than retried forever.
ATTEMPTS = 3
RETRY_DELAY_S = 60


def week_starts(start: dt.date, end: dt.date) -> list[dt.date]:
    """Mondays of the 1940-aligned weeks from the one containing ``start`` to the
    last one that ends on or before ``end``."""
    epoch = dt.date.fromisoformat(ERA5_ARCHIVE_WEEK_EPOCH)
    first = epoch + dt.timedelta(days=7 * ((start - epoch).days // 7))
    weeks = []
    week = first
    while week + dt.timedelta(days=6) <= end:
        weeks.append(week)
        week += dt.timedelta(days=7)
    return weeks


def build_week(
    week: dt.date, work: Path, upload: str, log_dir: Path
) -> tuple[dt.date, bool, float]:
    """Build and publish one week; returns (week, succeeded, seconds)."""
    out = work / week.isoformat()
    log_path = log_dir / f"{week.isoformat()}.log"
    t0 = time.perf_counter()
    for attempt in range(1, ATTEMPTS + 1):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        with open(log_path, "a") as log:
            log.write(f"=== attempt {attempt} {dt.datetime.now(dt.UTC).isoformat()}\n")
            log.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--out",
                    str(out),
                    "--week-start",
                    week.isoformat(),
                    "--upload",
                    upload,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        shutil.rmtree(out, ignore_errors=True)
        if result.returncode == 0:
            return week, True, time.perf_counter() - t0
        logger.warning(
            "Week %s attempt %d failed (exit %d); see %s",
            week,
            attempt,
            result.returncode,
            log_path,
        )
        if attempt < ATTEMPTS:
            time.sleep(RETRY_DELAY_S)
    return week, False, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--start", required=True, help="first day to cover (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", required=True, help="last day of data to use (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--upload", required=True, help="bucket/prefix the weeks are published to"
    )
    parser.add_argument("--work", required=True, help="local scratch directory")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    weeks = week_starts(
        dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    )
    remote = open_remote(
        args.upload, "S3", os.getenv("AWS_KEY", ""), os.getenv("AWS_SECRET", "")
    )
    todo = [
        week
        for week in weeks
        if not remote_object_exists(
            remote, f"{ERA5_ARCHIVE_STORE_DIR}/{week.isoformat()}.zarr/zarr.json"
        )
    ]
    logger.info(
        "%d weeks %s .. %s; %d already published, %d to build, %d at a time",
        len(weeks),
        weeks[0],
        weeks[-1],
        len(weeks) - len(todo),
        len(todo),
        args.parallel,
    )
    work = Path(args.work)
    log_dir = work / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [
            pool.submit(build_week, week, work, args.upload, log_dir) for week in todo
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            week, ok, seconds = future.result()
            if not ok:
                failed.append(week)
            elapsed = time.perf_counter() - t0
            remaining = elapsed / done * (len(todo) - done)
            logger.info(
                "%s %s in %.0f s (%d/%d done, %d failed, ~%.1f h left)",
                week,
                "published" if ok else "FAILED",
                seconds,
                done,
                len(todo),
                len(failed),
                remaining / 3600,
            )
    logger.info(
        "Done in %.1f h; failed weeks: %s",
        (time.perf_counter() - t0) / 3600,
        [w.isoformat() for w in sorted(failed)] or "none",
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
