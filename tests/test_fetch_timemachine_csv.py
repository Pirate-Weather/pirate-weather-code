import argparse
import datetime as dt
from urllib import parse

from scripts.fetch_timemachine_csv import (
    Point,
    build_request_url,
    create_request_jobs,
    group_jobs_by_time_slot,
)


def test_build_request_url_includes_exclude() -> None:
    point = Point(
        row_number=2,
        label="test",
        lat=40.0,
        lon=-75.0,
        timezone_name="America/New_York",
    )

    url = build_request_url(
        api_base="http://localhost:8081",
        api_key="abc123",
        point=point,
        unix_time=1_700_000_000,
        days=2,
        hours=None,
        exclude="summary",
    )

    query = parse.parse_qs(parse.urlsplit(url).query)
    assert query["exclude"] == ["summary"]


def test_build_request_url_omits_exclude_by_default() -> None:
    point = Point(
        row_number=2,
        label="test",
        lat=40.0,
        lon=-75.0,
        timezone_name="America/New_York",
    )

    url = build_request_url(
        api_base="http://localhost:8081",
        api_key="abc123",
        point=point,
        unix_time=1_700_000_000,
        days=2,
        hours=None,
    )

    query = parse.parse_qs(parse.urlsplit(url).query)
    assert "exclude" not in query


def test_request_jobs_are_grouped_by_time_before_point(tmp_path) -> None:
    points = [
        Point(2, "first", 40.0, -75.0, "UTC"),
        Point(3, "second", 41.0, -76.0, "UTC"),
    ]
    dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]
    args = argparse.Namespace(
        request_frequency="daily",
        output_dir=tmp_path,
        overwrite=False,
        api_base="http://localhost:8081",
        api_key="abc123",
        days=2,
        hours=None,
        exclude=None,
        csv_path=tmp_path / "points.csv",
    )

    jobs = create_request_jobs(
        args=args,
        points=points,
        dates=dates,
        total_requests=4,
        started_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        progress_path=tmp_path / "progress.json",
    )

    assert [(job.target_date, job.point.label) for job in jobs] == [
        (dates[0], "first"),
        (dates[0], "second"),
        (dates[1], "first"),
        (dates[1], "second"),
    ]
    assert [
        [job.point.label for job in batch] for batch in group_jobs_by_time_slot(jobs)
    ] == [
        ["first", "second"],
        ["first", "second"],
    ]


def test_hourly_request_batches_finish_each_hour_for_all_points(tmp_path) -> None:
    points = [
        Point(2, "first", 40.0, -75.0, "UTC"),
        Point(3, "second", 41.0, -76.0, "UTC"),
    ]
    target_date = dt.date(2024, 1, 1)
    args = argparse.Namespace(
        request_frequency="hourly",
        output_dir=tmp_path,
        overwrite=False,
        api_base="http://localhost:8081",
        api_key="abc123",
        days=2,
        hours=None,
        exclude=None,
        csv_path=tmp_path / "points.csv",
    )

    jobs = create_request_jobs(
        args=args,
        points=points,
        dates=[target_date],
        total_requests=48,
        started_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        progress_path=tmp_path / "progress.json",
    )
    batches = list(group_jobs_by_time_slot(jobs))

    assert len(batches) == 24
    assert all(len(batch) == 2 for batch in batches)
    assert [
        (batch[0].target_hour, [job.point.label for job in batch])
        for batch in batches[:2]
    ] == [
        (0, ["first", "second"]),
        (1, ["first", "second"]),
    ]
