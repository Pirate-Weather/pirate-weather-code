"""RDAQA ingest: Datamart URLs, download outcomes and the hourly history merge."""

import json
import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import requests
import zarr

from API.ingest_utils import open_remote, publish_history_store, publish_ingest_run
from ingest import RDAQA_Local_Ingest as ingest

HOUR = datetime(2026, 11, 19, 15, tzinfo=UTC)


def test_build_rdaqa_url():
    """Protect the Datamart filename contract: prefixed analyses (Prelim_, FW_)
    use "RDAQA-" and the final one "RDAQA_", with no grib2/ folder, so a wrong
    separator or path makes every download of that analysis 404. FW is only
    published for PM.
    """
    base = "https://dd.weather.gc.ca/20261119/WXO-DD/model_rdaqa/10km/15/"
    assert ingest.build_rdaqa_url(HOUR, "PM2.5", "prelim") == (
        base + "20261119T15Z_MSC_RDAQA-Prelim_PM2.5_Sfc_RLatLon0.09_PT0H.grib2"
    )
    assert ingest.build_rdaqa_url(HOUR, "O3", "final") == (
        base + "20261119T15Z_MSC_RDAQA_O3_Sfc_RLatLon0.09_PT0H.grib2"
    )
    assert ingest.build_rdaqa_url(HOUR, "PM10", "fw") == (
        base + "20261119T15Z_MSC_RDAQA-FW_PM10_Sfc_RLatLon0.09_PT0H.grib2"
    )
    with pytest.raises(ValueError, match="'fw' analysis"):
        ingest.build_rdaqa_url(HOUR, "NO2", "fw")


class _Response:
    def __init__(self, status_code, body=b"GRIB"):
        self.status_code = status_code
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body


def test_only_a_404_marks_an_hour_not_posted(tmp_path, monkeypatch):
    """Regression: Herbie's existence check also reports "not found" when a
    Datamart request fails, so posted hours were stored as NaN (3 of 49 in a
    backfill). A dropped connection must be retried and the file downloaded;
    only a 404 may mean the hour is not posted.
    """
    monkeypatch.setattr(ingest, "tmp_dir", str(tmp_path))
    monkeypatch.setattr(ingest, "RDAQA_RETRY_SLEEP_SECONDS", 0)
    calls = []

    def flaky_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ConnectionError("SSL: UNEXPECTED_EOF_WHILE_READING")
        return _Response(200)

    monkeypatch.setattr(ingest.requests, "get", flaky_get)
    path = ingest.download_rdaqa_file(HOUR, "PM2.5")
    assert len(calls) == 2
    assert Path(path).read_bytes() == b"GRIB"

    monkeypatch.setattr(ingest.requests, "get", lambda url, **kw: _Response(404))
    assert ingest.download_rdaqa_file(HOUR, "NO2") is None

    monkeypatch.setattr(ingest.requests, "get", lambda url, **kw: _Response(503))
    with pytest.raises(RuntimeError, match="failed after"):
        ingest.download_rdaqa_file(HOUR, "O3")


def _archive(remote, tmp_path, valid_time, value):
    fields = {v: np.full((3, 4), value, dtype=np.float32) for v in ingest.zarr_vars}
    tmp_store = str(tmp_path / "tmp_hour.zarr")
    ingest.write_hour_store(valid_time, fields, tmp_store)
    publish_history_store(tmp_store, remote, ingest.hourly_store_name(valid_time))


def test_missing_hour_is_a_nan_row(tmp_path, monkeypatch):
    """The API reads RDAQA rows by position, so an hour the Datamart never
    posted must be a NaN row at its own place, not skipped (which would shift
    every later hour by one) and not a failed run. The newest hour is the run
    time, so it must be there.
    """
    monkeypatch.setattr(ingest, "his_period", 3)
    remote = open_remote(str(tmp_path / "Hist"), "Download")
    hours = ingest.window_hours(HOUR)
    for index, valid_time in enumerate(hours):
        if index != 1:
            _archive(remote, tmp_path, valid_time, float(index))

    stacked, times = ingest.merge_history(HOUR, remote, str(tmp_path / "local"))

    assert stacked.shape == (len(ingest.zarr_vars), 4, 3, 4)
    assert times.tolist() == [int(t.timestamp()) for t in hours]
    assert np.isnan(stacked[:, 1]).all()
    for index in (0, 2, 3):
        assert (stacked[:, index] == index).all()

    newest = HOUR + timedelta(hours=1)
    with pytest.raises(RuntimeError, match="Newest RDAQA hour"):
        ingest.merge_history(newest, remote, str(tmp_path / "local2"))


def test_late_hour_is_published_when_newest_hour_is_unchanged(tmp_path, monkeypatch):
    """Regression: a run whose newest complete hour was already published exited
    before retrying the missing hours, so an hour posted late (while the next
    one was still incomplete) stayed NaN until a newer hour came. The run must
    archive it and republish the window under the same run time.
    """
    monkeypatch.setattr(ingest, "his_period", 3)
    monkeypatch.setattr(ingest, "forecast_process_dir", str(tmp_path / "work"))
    monkeypatch.setattr(ingest, "tmp_dir", str(tmp_path / "work" / "Downloads"))
    monkeypatch.setattr(ingest, "forecast_path", str(tmp_path / "Prod"))
    monkeypatch.setattr(ingest, "historic_path", str(tmp_path / "Hist"))
    history = open_remote(str(tmp_path / "Hist"), "Download")
    forecast = open_remote(str(tmp_path / "Prod" / ingest.ingest_version), "Download")
    hours = ingest.window_hours(HOUR)
    late_hour = hours[1]
    for index, valid_time in enumerate(hours):
        if valid_time != late_hour:
            _archive(history, tmp_path, valid_time, float(index))
    # The last run published through HOUR, with late_hour missing
    (tmp_path / "work").mkdir()
    publish_ingest_run(
        [],
        HOUR,
        ingest.RDAQA_TIME_PICKLE,
        process_dir=str(tmp_path / "work"),
        remote=forecast,
    )

    # The Datamart: HOUR is still the newest complete hour, and late_hour is posted
    monkeypatch.setattr(ingest, "get_latest_rdaqa_hour", lambda: HOUR)
    late_fields = {v: np.full((3, 4), 7.0, dtype=np.float32) for v in ingest.zarr_vars}
    monkeypatch.setattr(
        ingest,
        "load_hours",
        lambda times: {t: late_fields for t in times if t == late_hour},
    )
    ingest.main()

    pointer = json.loads(
        (tmp_path / "Prod" / ingest.ingest_version / "RDAQA.current.json").read_text()
    )
    store = zarr.open_array(
        str(tmp_path / "Prod" / ingest.ingest_version / pointer["store"]), mode="r"
    )
    np.testing.assert_array_equal(store[:, 1, :3, :4], 7.0)
    assert store.attrs["time"] == [int(t.timestamp()) for t in hours]
    run_time = pickle.loads(
        (
            tmp_path / "Prod" / ingest.ingest_version / ingest.RDAQA_TIME_PICKLE
        ).read_bytes()
    )
    assert run_time == HOUR
