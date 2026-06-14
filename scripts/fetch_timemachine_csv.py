#!/usr/bin/env python3
"""Fetch local Pirate Weather timemachine responses for CSV points."""

# Example: start local API in background
# cd /home/ubuntu/pwcode_nov/pirate-weather-code && nohup "/home/ubuntu/pwcode_nov/pirate-weather-code/.venvs/ingest-test/bin/python3.14" -m uvicorn API.responseLocal:app --host 0.0.0.0 --port 8081 --workers 12 --env-file /home/ubuntu/pwcode_nov/pirate-weather-code/.env > uvicorn_local.log 2>&1 & echo $!
#
# Example: run timemachine CSV fetch in background
# cd /home/ubuntu/pwcode_nov/pirate-weather-code && nohup /home/ubuntu/pwcode_nov/pirate-weather-code/.venvs/ingest-test/bin/python3.14 scripts/fetch_timemachine_csv.py /mnt/efs/scripts/SiteCodes_2026_05_16.csv --api-base http://localhost:8081 --api-key abc123 --workers 12 --start-date 2024-01-01 --end-date 2026-05-25 --output-dir /mnt/nvme/data/tm_out/ > fetch_timemachine.log 2>&1 & echo $!

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import parse, request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from timezonefinder import TimezoneFinder

DEFAULT_START_DATE = dt.date(2024, 1, 1)
DEFAULT_END_DATE = dt.date(2026, 5, 25)
DEFAULT_API_BASE = "http://localhost:8081"
DEFAULT_API_KEY = "abc123"
DEFAULT_DAYS = 2
MAX_TIME_MACHINE_DAYS = 8
MAX_TIME_MACHINE_HOURS = MAX_TIME_MACHINE_DAYS * 24
DEFAULT_QUERY_PARAMS = {
    "version": "2",
    "units": "us",
    "timemachine": "1",
    "tmextra": "1",
}


@dataclass(frozen=True)
class Point:
    row_number: int
    label: str
    lat: float
    lon: float
    timezone_name: str


@dataclass(frozen=True)
class RequestJob:
    point_index: int
    point_count: int
    day_index: int
    date_count: int
    point: Point
    target_date: dt.date
    unix_time: int
    output_path: Path
    url: str
    attempt_number: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Loop through a CSV of latitude/longitude points, request Pirate "
            "Weather timemachine data for each day at 12:00 local time, and "
            "save JSON responses to disk."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV with lat/lon columns.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("timemachine_output"),
        help="Directory where JSON output and progress files are written.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="Base URL for the local Pirate Weather API.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key segment used in the request path.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=DEFAULT_START_DATE,
        help="First local date to request, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=DEFAULT_END_DATE,
        help="Last local date to request, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=(
            "Number of timemachine days to request per API call "
            f"(1-{MAX_TIME_MACHINE_DAYS})."
        ),
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help=(
            "Optional number of timemachine hours to request per API call "
            f"(1-{MAX_TIME_MACHINE_HOURS})."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between requests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Maximum number of concurrent request threads.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retry attempts for a missing or failed download.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Delay between retry attempts for the same output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fetch JSON files even when the output file already exists.",
    )
    return parser.parse_args()


def parse_date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def date_range(start_date: dt.date, end_date: dt.date) -> Iterable[dt.date]:
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def detect_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        match = lowered.get(candidate)
        if match:
            return match
    raise ValueError(
        f"Could not find any of the required CSV columns: {', '.join(candidates)}"
    )


def slugify(value: str) -> str:
    cleaned = []
    for character in value.strip():
        if character.isalnum() or character in {"-", "_"}:
            cleaned.append(character)
        elif character in {" ", "."}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "point"


def resolve_timezone(tf: TimezoneFinder, lat: float, lon: float) -> str:
    timezone_name = tf.timezone_at(lat=lat, lng=lon)
    if timezone_name:
        return timezone_name

    timezone_name = tf.closest_timezone_at(lat=lat, lng=lon)
    if timezone_name:
        return timezone_name

    return "UTC"


def load_points(csv_path: Path, tf: TimezoneFinder) -> list[Point]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file is missing a header row.")

        lat_column = detect_column(reader.fieldnames, ("lat", "latitude", "slatitude"))
        lon_column = detect_column(
            reader.fieldnames,
            ("lon", "lng", "longitude", "long", "slongitude", "slongitud"),
        )

        label_column = None
        for candidate in (
            "id",
            "locationid",
            "name",
            "label",
            "site",
            "station",
        ):
            try:
                label_column = detect_column(reader.fieldnames, (candidate,))
                break
            except ValueError:
                continue

        points: list[Point] = []
        for row_number, row in enumerate(reader, start=2):
            lat_value = row.get(lat_column, "").strip()
            lon_value = row.get(lon_column, "").strip()
            if not lat_value or not lon_value:
                raise ValueError(f"Row {row_number} is missing latitude or longitude.")

            lat = float(lat_value)
            lon = float(lon_value)
            raw_label = row.get(label_column, "") if label_column else ""
            label = slugify(raw_label) if raw_label else f"row_{row_number}"
            timezone_name = resolve_timezone(tf, lat, lon)
            points.append(
                Point(
                    row_number=row_number,
                    label=label,
                    lat=lat,
                    lon=lon,
                    timezone_name=timezone_name,
                )
            )

    if not points:
        raise ValueError("CSV file does not contain any data rows.")

    return points


def local_noon_unix(target_date: dt.date, timezone_name: str) -> int:
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("UTC")

    local_noon = dt.datetime.combine(
        target_date,
        dt.time(hour=12, minute=0, second=0),
        tzinfo=timezone,
    )
    return int(local_noon.timestamp())


def build_request_url(
    api_base: str,
    api_key: str,
    point: Point,
    unix_time: int,
    days: int,
    hours: int | None,
) -> str:
    path = (
        f"{api_base.rstrip('/')}/forecast/{parse.quote(api_key)}/"
        f"{point.lat},{point.lon},{unix_time}"
    )
    query_params = {**DEFAULT_QUERY_PARAMS, "days": str(days)}
    if hours is not None:
        query_params["hours"] = str(hours)
    query = parse.urlencode(query_params)
    return f"{path}?{query}"


def build_progress_payload(
    *,
    args: argparse.Namespace,
    started_at: dt.datetime,
    point: Point,
    point_index: int,
    point_count: int,
    target_date: dt.date,
    day_index: int,
    date_count: int,
    completed: int,
    failed: int,
    skipped: int,
    total_requests: int,
    output_path: Path,
) -> dict:
    return {
        "updated_at": dt.datetime.now(dt.UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "csv_path": str(args.csv_path.resolve()),
        "api_base": args.api_base,
        "point_index": point_index,
        "point_count": point_count,
        "date_index": day_index,
        "date_count": date_count,
        "current_point": {
            "label": point.label,
            "lat": point.lat,
            "lon": point.lon,
            "timezone": point.timezone_name,
        },
        "current_date": target_date.isoformat(),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "total_requests": total_requests,
        "last_output": str(output_path),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def fetch_json(url: str, timeout: float) -> tuple[int, dict]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=timeout) as response:
        status_code = response.getcode()
        body = response.read().decode("utf-8")
    return status_code, json.loads(body)


def create_request_jobs(
    *,
    args: argparse.Namespace,
    points: list[Point],
    dates: list[dt.date],
    total_requests: int,
    started_at: dt.datetime,
    progress_path: Path,
) -> list[RequestJob]:
    jobs: list[RequestJob] = []
    skipped = 0

    for point_index, point in enumerate(points, start=1):
        for day_index, target_date in enumerate(dates, start=1):
            unix_time = local_noon_unix(target_date, point.timezone_name)
            output_path = (
                point_output_dir(args.output_dir, point)
                / f"{target_date.isoformat()}.json"
            )
            progress_payload = build_progress_payload(
                args=args,
                started_at=started_at,
                point=point,
                point_index=point_index,
                point_count=len(points),
                target_date=target_date,
                day_index=day_index,
                date_count=len(dates),
                completed=0,
                failed=0,
                skipped=skipped,
                total_requests=total_requests,
                output_path=output_path,
            )

            if output_path.exists() and not args.overwrite:
                skipped += 1
                progress_payload["skipped"] = skipped
                progress_payload["status"] = "skipped-existing"
                write_progress(progress_path, progress_payload)
                continue

            jobs.append(
                RequestJob(
                    point_index=point_index,
                    point_count=len(points),
                    day_index=day_index,
                    date_count=len(dates),
                    point=point,
                    target_date=target_date,
                    unix_time=unix_time,
                    output_path=output_path,
                    url=build_request_url(
                        args.api_base,
                        args.api_key,
                        point,
                        unix_time,
                        args.days,
                        args.hours,
                    ),
                )
            )

    if skipped > 0 and skipped % 25 == 0:
        print_progress(started_at, 0, 0, skipped, total_requests)

    return jobs


def process_job(job: RequestJob, timeout: float) -> tuple[RequestJob, int, dict]:
    status_code, payload = fetch_json(job.url, timeout=timeout)
    return job, status_code, payload


def enqueue_retry(
    *,
    executor: ThreadPoolExecutor,
    pending_futures: dict,
    job: RequestJob,
    args: argparse.Namespace,
    error_message: str,
) -> bool:
    if job.attempt_number > args.max_retries:
        return False

    if args.retry_delay_seconds > 0:
        time.sleep(args.retry_delay_seconds)

    retry_job = RequestJob(
        point_index=job.point_index,
        point_count=job.point_count,
        day_index=job.day_index,
        date_count=job.date_count,
        point=job.point,
        target_date=job.target_date,
        unix_time=job.unix_time,
        output_path=job.output_path,
        url=job.url,
        attempt_number=job.attempt_number + 1,
    )
    pending_futures[executor.submit(process_job, retry_job, args.timeout)] = retry_job
    print(
        "RETRYING "
        f"point={job.point.label} date={job.target_date.isoformat()} "
        f"attempt={retry_job.attempt_number}/{args.max_retries + 1} error={error_message}",
        file=sys.stderr,
    )
    return True


def drain_completed_futures(
    *,
    executor: ThreadPoolExecutor,
    pending_futures: dict,
    timeout: float,
    failures_path: Path,
    progress_path: Path,
    args: argparse.Namespace,
    started_at: dt.datetime,
    total_requests: int,
    completed: int,
    failed: int,
    skipped: int,
) -> tuple[int, int]:
    done, _ = wait(set(pending_futures), return_when=FIRST_COMPLETED)
    for future in done:
        job = pending_futures.pop(future)
        progress_payload = build_progress_payload(
            args=args,
            started_at=started_at,
            point=job.point,
            point_index=job.point_index,
            point_count=job.point_count,
            target_date=job.target_date,
            day_index=job.day_index,
            date_count=job.date_count,
            completed=completed,
            failed=failed,
            skipped=skipped,
            total_requests=total_requests,
            output_path=job.output_path,
        )

        try:
            _, status_code, payload = future.result(timeout=timeout)
            if status_code != 200:
                raise RuntimeError(f"Unexpected HTTP status {status_code}")
            save_json(job.output_path, payload)
            if not job.output_path.exists():
                raise RuntimeError("Output file missing after save")
            completed += 1
            progress_payload["completed"] = completed
            progress_payload["status"] = "completed"
            progress_payload["attempt"] = job.attempt_number
            write_progress(progress_path, progress_payload)
        except Exception as exc:
            if enqueue_retry(
                executor=executor,
                pending_futures=pending_futures,
                job=job,
                args=args,
                error_message=str(exc),
            ):
                progress_payload["status"] = "retrying"
                progress_payload["attempt"] = job.attempt_number
                progress_payload["last_error"] = str(exc)
                write_progress(progress_path, progress_payload)
                continue

            failed += 1
            failure_record = {
                "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                "point": {
                    "label": job.point.label,
                    "lat": job.point.lat,
                    "lon": job.point.lon,
                    "timezone": job.point.timezone_name,
                },
                "date": job.target_date.isoformat(),
                "unix_time": job.unix_time,
                "url": job.url,
                "attempts": job.attempt_number,
                "error": str(exc),
            }
            append_jsonl(failures_path, failure_record)
            progress_payload["failed"] = failed
            progress_payload["status"] = "failed"
            progress_payload["attempt"] = job.attempt_number
            progress_payload["last_error"] = str(exc)
            write_progress(progress_path, progress_payload)
            print(
                "FAILED "
                f"point={job.point.label} date={job.target_date.isoformat()} error={exc}",
                file=sys.stderr,
            )

        if (completed + failed + skipped) % 25 == 0:
            print_progress(started_at, completed, failed, skipped, total_requests)

    return completed, failed


def point_output_dir(output_dir: Path, point: Point) -> Path:
    lat_text = f"{point.lat:.4f}".replace("-", "m").replace(".", "p")
    lon_text = f"{point.lon:.4f}".replace("-", "m").replace(".", "p")
    directory_name = f"{point.label}_{lat_text}_{lon_text}"
    return output_dir / directory_name


def write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.end_date < args.start_date:
        print("end-date must be on or after start-date", file=sys.stderr)
        return 2
    if args.days < 1 or args.days > MAX_TIME_MACHINE_DAYS:
        print(
            f"days must be between 1 and {MAX_TIME_MACHINE_DAYS}",
            file=sys.stderr,
        )
        return 2
    if args.hours is not None and (
        args.hours < 1 or args.hours > MAX_TIME_MACHINE_HOURS
    ):
        print(
            f"hours must be between 1 and {MAX_TIME_MACHINE_HOURS}",
            file=sys.stderr,
        )
        return 2
    if args.workers < 1:
        print("workers must be at least 1", file=sys.stderr)
        return 2
    if args.max_retries < 0:
        print("max-retries must be 0 or greater", file=sys.stderr)
        return 2
    if args.retry_delay_seconds < 0:
        print("retry-delay-seconds must be 0 or greater", file=sys.stderr)
        return 2

    tf = TimezoneFinder(in_memory=True)
    try:
        points = load_points(args.csv_path, tf)
    except Exception as exc:
        print(f"Failed to load CSV: {exc}", file=sys.stderr)
        return 2

    dates = list(date_range(args.start_date, args.end_date))
    total_requests = len(points) * len(dates)
    progress_path = args.output_dir / "progress.json"
    failures_path = args.output_dir / "failures.jsonl"
    started_at = dt.datetime.now(dt.UTC)
    jobs = create_request_jobs(
        args=args,
        points=points,
        dates=dates,
        total_requests=total_requests,
        started_at=started_at,
        progress_path=progress_path,
    )
    skipped = total_requests - len(jobs)
    completed = 0
    failed = 0

    print(
        "Starting fetch run "
        f"for {len(points)} points across {len(dates)} dates "
        f"({total_requests} requests, days={args.days}, "
        f"hours={args.hours if args.hours is not None else 'default'}, "
        f"workers={args.workers}, skipped={skipped})."
    )

    job_iter = iter(jobs)
    pending_futures = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(pending_futures) < args.workers:
            try:
                job = next(job_iter)
            except StopIteration:
                break
            pending_futures[executor.submit(process_job, job, args.timeout)] = job
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

        while pending_futures:
            completed, failed = drain_completed_futures(
                executor=executor,
                pending_futures=pending_futures,
                timeout=args.timeout,
                failures_path=failures_path,
                progress_path=progress_path,
                args=args,
                started_at=started_at,
                total_requests=total_requests,
                completed=completed,
                failed=failed,
                skipped=skipped,
            )

            while len(pending_futures) < args.workers:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                pending_futures[executor.submit(process_job, job, args.timeout)] = job
                if args.pause_seconds > 0:
                    time.sleep(args.pause_seconds)

    print_progress(started_at, completed, failed, skipped, total_requests, final=True)
    return 0 if failed == 0 else 1


def print_progress(
    started_at: dt.datetime,
    completed: int,
    failed: int,
    skipped: int,
    total_requests: int,
    *,
    final: bool = False,
) -> None:
    processed = completed + failed + skipped
    elapsed_seconds = max((dt.datetime.now(dt.UTC) - started_at).total_seconds(), 1e-6)
    rate = processed / elapsed_seconds
    remaining = max(total_requests - processed, 0)
    eta_seconds = int(remaining / rate) if rate > 0 else -1
    eta_text = format_eta(eta_seconds) if eta_seconds >= 0 else "unknown"
    prefix = "Finished" if final else "Progress"
    print(
        f"{prefix}: processed={processed}/{total_requests} completed={completed} "
        f"failed={failed} skipped={skipped} rate={rate:.2f}/s eta={eta_text}"
    )


def format_eta(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
