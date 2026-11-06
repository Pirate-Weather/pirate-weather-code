"""Daily ERA5 archive update: keep the archive in step with Google's ERA5 and ERA5T.

Runs once a day (the ``ingest-era5-daily`` CronJob). Google's ARCO ERA5 carries
two end dates: ``valid_time_stop`` for final ERA5 (about 3 months behind) and
``valid_time_stop_era5t`` for the preliminary ERA5T (about 5 days behind, moved
on daily). The archive mirrors that split:

* **immutable** ``ERA5_archive/<week>.zarr`` for complete weeks of final ERA5,
  written once and cached for good by readers;
* **versioned** ``ERA5_archive/<week>/<stamp>.zarr`` behind the pointer
  ``ERA5_archive/<week>.current.json`` for weeks that still change: ERA5T weeks
  and the current week, published partial with the hours Google has so far.

Each run:

1. reads Google's end dates (``.zattrs``, one small GET);
2. lists the archive (one delimited LIST), checks the immutable stores are
   complete, reads the pointers, and finds immutable stores holding ERA5T (written
   by a backfill that ran past final ERA5);
3. plans (``plan_updates``): an immutable build for every complete final week
   without a final immutable store (gaps, and ERA5T weeks final ERA5 now covers);
   a versioned build for every newer week whose pointer is missing or older than
   Google's data; and the clean-up below;
4. builds the weeks (``build_week``: own process, timeout, three attempts);
5. cleans up. A week promoted to an immutable store loses its pointer at once,
   but its versions are only deleted by the next run, so a reader still holding
   the old pointer (it re-resolves every 5 minutes) has a day to move on. An
   immutable store holding ERA5T is deleted once its versioned replacement is
   published, root ``zarr.json`` first so readers stop treating it as complete.

Readers never see a store change under them: changing weeks always go to a new
path and only then does the pointer move. The exit status is 1 if any week failed
or ERA5T has not advanced for ``--max-lag-days``, so the CronJob reports a failed
Job; the next run picks up whatever is left.

Usage:
    python scripts/era5_archive_daily.py --upload pirate-era5 --work /scratch/era5 \\
        [--first-week 1940-12-30] [--parallel 1]
    # S3 settings as for the builder: s3_endpoint, s3_region, s3_storage_class,
    # AWS_KEY/AWS_SECRET (or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import obstore  # noqa: E402
from backfill_era5_archive import build_week, week_starts  # noqa: E402
from build_era5_point_test_store import open_archive_remote  # noqa: E402

from API.constants.model_const import ERA5_ARCHIVE_STORE_DIR  # noqa: E402
from API.ingest_utils import remote_object_exists  # noqa: E402

logger = logging.getLogger("era5_daily")

GOOGLE_ERA5_ATTRS = (
    "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/.zattrs"
)
# The first archive week: ERA5's forecast fields (gusts, precipitation, radiation,
# cloud base) are missing for 1940-01-01 00-06 UTC, so the archive starts in 1941.
FIRST_WEEK = "1940-12-30"
# Immutable stores this recent are checked for ERA5T (written by a backfill that
# ran past final ERA5); older ones are final by construction.
ERA5T_CHECK_DAYS = 200
HEAD_CONCURRENCY = 32


@dataclass
class ArchiveState:
    """What the archive holds, by week start (ISO date)."""

    immutable: set[str] = field(default_factory=set)  # complete immutable stores
    immutable_era5t: set[str] = field(default_factory=set)  # ...of them holding ERA5T
    pointers: dict[str, str] = field(
        default_factory=dict
    )  # week -> last hour with data
    version_dirs: set[str] = field(default_factory=set)  # weeks with versions


@dataclass
class Plan:
    immutable: list[dt.date] = field(default_factory=list)
    versioned: list[dt.date] = field(default_factory=list)
    delete_pointers: list[str] = field(default_factory=list)  # now
    delete_versions: list[str] = field(default_factory=list)  # now (orphaned earlier)
    # After a successful build of the week:
    delete_pointer_after_immutable: list[str] = field(default_factory=list)
    delete_immutable_after_versioned: list[str] = field(default_factory=list)
    # ERA5T immutable stores whose versioned replacement is already published.
    delete_immutable: list[str] = field(default_factory=list)


def plan_updates(
    state: ArchiveState, first_week: dt.date, final_stop: dt.date, era5t_stop: dt.date
) -> Plan:
    """Decide what to build and delete. ``final_stop`` and ``era5t_stop`` are the
    last days of final ERA5 and of ERA5T at Google."""
    plan = Plan()
    for week in week_starts(first_week, era5t_stop + dt.timedelta(days=6)):
        if week > era5t_stop:
            break
        key = week.isoformat()
        week_end = week + dt.timedelta(days=6)
        has_pointer = key in state.pointers
        # The last hour Google has for this week; a version through it is current.
        wanted = f"{min(week_end, era5t_stop).isoformat()}T23"
        version_current = has_pointer and state.pointers[key] >= wanted
        if week_end <= final_stop and key in state.immutable_era5t:
            # Written as ERA5T and can't be overwritten in place (readers cache
            # immutable stores): replace it with a version, then delete it; the
            # next run builds the final immutable store.
            if version_current:
                plan.delete_immutable.append(key)
            else:
                plan.versioned.append(week)
                plan.delete_immutable_after_versioned.append(key)
        elif week_end <= final_stop:
            # A complete week of final ERA5 belongs in an immutable store.
            if key not in state.immutable:
                plan.immutable.append(week)
                if has_pointer:
                    plan.delete_pointer_after_immutable.append(key)
            elif has_pointer:
                plan.delete_pointers.append(key)  # promoted, clean-up left over
            elif key in state.version_dirs:
                plan.delete_versions.append(key)  # promoted on an earlier run
        else:
            # ERA5T, or the current week: versioned, rebuilt when Google has more.
            if not version_current:
                plan.versioned.append(week)
            if key in state.immutable:  # ERA5T in an immutable store
                if version_current:
                    plan.delete_immutable.append(key)
                else:
                    plan.delete_immutable_after_versioned.append(key)
    # Version directories with no pointer and no week above (a crash between
    # uploading a first version and writing its pointer) are unreferenced.
    planned = {w.isoformat() for w in plan.immutable + plan.versioned}
    for key in sorted(state.version_dirs):
        if (
            key not in state.pointers
            and key not in planned
            and key not in plan.delete_versions
        ):
            plan.delete_versions.append(key)
    return plan


def google_end_dates() -> tuple[dt.date, dt.date]:
    """Last days of final ERA5 and of ERA5T in Google's ARCO store."""
    import gcsfs

    attrs = json.loads(gcsfs.GCSFileSystem(token="anon").cat(GOOGLE_ERA5_ATTRS))
    return (
        dt.date.fromisoformat(attrs["valid_time_stop"]),
        dt.date.fromisoformat(attrs["valid_time_stop_era5t"]),
    )


def read_state(remote, final_stop: dt.date) -> ArchiveState:
    """List the archive and read the pointers (one LIST, a HEAD per immutable store,
    a GET per pointer and per recent immutable store)."""
    listing = obstore.list_with_delimiter(remote, prefix=f"{ERA5_ARCHIVE_STORE_DIR}/")
    state = ArchiveState()
    immutable_dirs = []
    for prefix in listing["common_prefixes"]:
        name = prefix.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            immutable_dirs.append(name.removesuffix(".zarr"))
        else:
            state.version_dirs.add(name)
    with ThreadPoolExecutor(HEAD_CONCURRENCY) as pool:
        complete = pool.map(
            lambda week: remote_object_exists(
                remote, f"{ERA5_ARCHIVE_STORE_DIR}/{week}.zarr/zarr.json"
            ),
            immutable_dirs,
        )
        state.immutable = {w for w, ok in zip(immutable_dirs, complete) if ok}
    recent = dt.date.fromordinal(final_stop.toordinal() - ERA5T_CHECK_DAYS).isoformat()
    for week in sorted(w for w in state.immutable if w >= recent):
        meta = json.loads(
            bytes(
                obstore.get(
                    remote, f"{ERA5_ARCHIVE_STORE_DIR}/{week}.zarr/zarr.json"
                ).bytes()
            )
        )
        if meta["attributes"].get("era5t"):
            state.immutable_era5t.add(week)
    for meta in listing["objects"]:
        name = meta["path"].rsplit("/", 1)[-1]
        if name.endswith(".current.json"):
            pointer = json.loads(bytes(obstore.get(remote, meta["path"]).bytes()))
            state.pointers[name.removesuffix(".current.json")] = pointer["run_time"]
    return state


def delete_prefix(remote, prefix: str, root_metadata_first: bool = False) -> int:
    """Delete every object under ``prefix``; returns the count."""
    keys = [m["path"] for batch in obstore.list(remote, prefix=prefix) for m in batch]
    if root_metadata_first and f"{prefix}zarr.json" in keys:
        obstore.delete(remote, f"{prefix}zarr.json")
        keys.remove(f"{prefix}zarr.json")
    for start in range(0, len(keys), 500):
        obstore.delete(remote, keys[start : start + 500])
    return len(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--upload", required=True, help="bucket/prefix (S3) or local directory"
    )
    parser.add_argument("--work", required=True, help="local scratch directory")
    parser.add_argument("--first-week", default=FIRST_WEEK)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-lag-days", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()

    final_stop, era5t_stop = google_end_dates()
    remote = open_archive_remote(args.upload)
    state = read_state(remote, final_stop)
    plan = plan_updates(
        state, dt.date.fromisoformat(args.first_week), final_stop, era5t_stop
    )
    logger.info(
        "Google: final ERA5 to %s, ERA5T to %s. Archive: %d immutable (%d ERA5T), "
        "%d pointers. Plan: %d immutable %s, %d versioned %s; delete %d pointers, "
        "%d version sets, %d ERA5T immutable",
        final_stop,
        era5t_stop,
        len(state.immutable),
        len(state.immutable_era5t),
        len(state.pointers),
        len(plan.immutable),
        [w.isoformat() for w in plan.immutable[:5]],
        len(plan.versioned),
        [w.isoformat() for w in plan.versioned[:5]],
        len(plan.delete_pointers) + len(plan.delete_pointer_after_immutable),
        len(plan.delete_versions),
        len(plan.delete_immutable_after_versioned) + len(plan.delete_immutable),
    )
    if args.dry_run:
        print(
            json.dumps(
                {k: [str(v) for v in vs] for k, vs in plan.__dict__.items()}, indent=1
            )
        )
        return

    # Versions orphaned by an earlier run: no reader can still be using them.
    for key in plan.delete_versions:
        removed = delete_prefix(remote, f"{ERA5_ARCHIVE_STORE_DIR}/{key}/")
        logger.info("Deleted %d objects of the old versions of %s", removed, key)
    for key in plan.delete_pointers:
        obstore.delete(remote, f"{ERA5_ARCHIVE_STORE_DIR}/{key}.current.json")
        logger.info("Deleted the pointer of %s (immutable store exists)", key)
    for key in plan.delete_immutable:
        removed = delete_prefix(
            remote, f"{ERA5_ARCHIVE_STORE_DIR}/{key}.zarr/", root_metadata_first=True
        )
        logger.info(
            "Deleted the ERA5T immutable store of %s (%d objects)", key, removed
        )

    work = Path(args.work)
    log_dir = work / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(w, "immutable") for w in plan.immutable] + [
        (w, "versioned") for w in plan.versioned
    ]
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = pool.map(
            lambda job: (
                job[1],
                *build_week(job[0], work, args.upload, log_dir, job[1]),
            ),
            jobs,
        )
        for publish, week, ok, seconds in results:
            key = week.isoformat()
            logger.info(
                "%s %s %s in %.0f s",
                key,
                publish,
                "published" if ok else "FAILED",
                seconds,
            )
            if not ok:
                failed.append(key)
                continue
            if publish == "immutable" and key in plan.delete_pointer_after_immutable:
                # Readers prefer the immutable store; versions go on the next run.
                obstore.delete(remote, f"{ERA5_ARCHIVE_STORE_DIR}/{key}.current.json")
                logger.info("Promoted %s to final ERA5; deleted its pointer", key)
            if publish == "versioned" and key in plan.delete_immutable_after_versioned:
                removed = delete_prefix(
                    remote,
                    f"{ERA5_ARCHIVE_STORE_DIR}/{key}.zarr/",
                    root_metadata_first=True,
                )
                logger.info(
                    "Replaced the ERA5T immutable store of %s (%d objects deleted)",
                    key,
                    removed,
                )

    lag = (dt.datetime.now(dt.UTC).date() - era5t_stop).days
    stale = lag > args.max_lag_days
    if stale:
        logger.error(
            "Google ERA5T ends %s, %d days ago: is Google's update stuck?",
            era5t_stop,
            lag,
        )
    logger.info(
        "Done in %.0f s; failed weeks: %s", time.perf_counter() - t0, failed or "none"
    )
    sys.exit(1 if failed or stale else 0)


if __name__ == "__main__":
    main()
