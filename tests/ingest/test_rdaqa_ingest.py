"""RDAQA ingest: Datamart URLs, download outcomes and the hourly history merge."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import requests

from API.ingest_utils import open_remote, publish_history_store
from API.rdaqa_herbie_template import build_rdaqa_url, rdaqa
from ingest import RDAQA_Local_Ingest as ingest

HOUR = datetime(2026, 11, 19, 15, tzinfo=UTC)


def test_build_rdaqa_url_and_template():
    """Protect the Datamart filename contract: prefixed analyses (Prelim_, FW_)
    use "RDAQA-" and the final one "RDAQA_", with no grib2/ folder, so a wrong
    separator or path makes every download of that analysis 404. FW is only
    published for PM, and the Herbie template must point at the same file.
    """
    base = "https://dd.weather.gc.ca/20261119/WXO-DD/model_rdaqa/10km/15/"
    assert build_rdaqa_url(HOUR, "PM2.5", "prelim") == (
        base + "20261119T15Z_MSC_RDAQA-Prelim_PM2.5_Sfc_RLatLon0.09_PT0H.grib2"
    )
    assert build_rdaqa_url(HOUR, "O3", "final") == (
        base + "20261119T15Z_MSC_RDAQA_O3_Sfc_RLatLon0.09_PT0H.grib2"
    )
    assert build_rdaqa_url(HOUR, "PM10", "fw") == (
        base + "20261119T15Z_MSC_RDAQA-FW_PM10_Sfc_RLatLon0.09_PT0H.grib2"
    )
    with pytest.raises(ValueError, match="'fw' analysis"):
        build_rdaqa_url(HOUR, "NO2", "fw")

    class DummyHerbie:
        date = HOUR
        product = None
        variable = "SO2"
        analysis = "prelim"

        @property
        def get_remoteFileName(self):
            return self.SOURCES["msc"].split("/")[-1]

    dummy = DummyHerbie()
    rdaqa.template(dummy)
    assert dummy.SOURCES == {"msc": build_rdaqa_url(HOUR, "SO2", "prelim")}
    assert dummy.LOCALFILE == dummy.SOURCES["msc"].split("/")[-1]


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
