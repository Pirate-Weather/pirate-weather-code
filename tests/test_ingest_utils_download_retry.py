import logging
from typing import ClassVar

import pytest

from API import ingest_utils


class _FakeHeadResponse:
    ok = True
    headers: ClassVar[dict[str, str]] = {"Content-Length": "11"}


class _FakeRef:
    def __init__(self, path, grib="https://example.test/file.grib2"):
        self._path = path
        self.grib = grib

    def get_localFilePath(self, _search):
        return self._path


class _FakeHerbie:
    def __init__(self, refs):
        self.file_exists = refs
        self.download_calls = 0

    def download(self, _search, verbose=False, overwrite=False):
        self.download_calls += 1
        return []


@pytest.mark.parametrize("retry_sleep_s", [1])
def test_download_retry_retries_until_local_files_exist(
    tmp_path, monkeypatch, retry_sleep_s
):
    f1 = tmp_path / "f1.grib2"
    f2 = tmp_path / "f2.grib2"

    refs = [_FakeRef(str(f1)), _FakeRef(str(f2))]
    herbie = _FakeHerbie(refs)

    def _fake_download(_search, verbose=False, overwrite=False):
        herbie.download_calls += 1
        if herbie.download_calls == 1:
            f1.write_text("ok", encoding="utf-8")
            if f2.exists():
                f2.unlink()
            return [str(f1)]
        else:
            f1.write_text("ok", encoding="utf-8")
            f2.write_text("ok", encoding="utf-8")
            return [str(f1), str(f2)]

    herbie.download = _fake_download
    monkeypatch.setattr(ingest_utils.time, "sleep", lambda _s: None)

    ingest_utils.download_herbie_with_retry(
        herbie_obj=herbie,
        search=":TMP:",
        expected_count=2,
        dataset_name="test",
        retries=3,
        retry_sleep_s=retry_sleep_s,
    )

    assert herbie.download_calls == 2


def test_configure_herbie_request_timeouts_patches_availability_heads(monkeypatch):
    from herbie.core import Herbie

    original_check_grib = Herbie._check_grib
    original_check_idx = Herbie._check_idx
    original_timeout = getattr(Herbie, "_pirate_weather_request_timeout_s", None)
    monkeypatch.delattr(Herbie, "_pirate_weather_request_timeout_s", raising=False)

    calls = []

    def _fake_head(url, *, timeout):
        calls.append((url, timeout))
        return _FakeHeadResponse()

    monkeypatch.setattr(ingest_utils.requests, "head", _fake_head)

    try:
        ingest_utils.configure_herbie_request_timeouts(request_timeout_s=7)
        assert Herbie._check_grib(object(), "https://example.test/file.grib2")
    finally:
        Herbie._check_grib = original_check_grib
        Herbie._check_idx = original_check_idx
        if original_timeout is None:
            if hasattr(Herbie, "_pirate_weather_request_timeout_s"):
                delattr(Herbie, "_pirate_weather_request_timeout_s")
        else:
            Herbie._pirate_weather_request_timeout_s = original_timeout

    assert calls == [("https://example.test/file.grib2", (10, 7))]


def test_download_retry_full_file_downloads_respect_timeout_and_verbosity(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger=ingest_utils.logger.name)
    grib_path = tmp_path / "f1.grib2"
    ref = _FakeRef(str(grib_path))
    herbie = _FakeHerbie([ref])

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("full-file downloads should not use Herbie.download")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            return [b"grib", b"data"]

    def _fake_get(url, *, stream, timeout):
        assert url == ref.grib
        assert stream is True
        assert timeout == (10, 120)
        return _FakeResponse()

    herbie.download = _unexpected_download
    monkeypatch.setattr(ingest_utils.requests, "get", _fake_get)

    ingest_utils.download_herbie_with_retry(
        herbie_obj=herbie,
        search=None,
        expected_count=1,
        dataset_name="test",
        retries=1,
        retry_sleep_s=1,
        verbose=False,
    )

    assert grib_path.read_bytes() == b"gribdata"
    assert herbie.download_calls == 0
    assert "Downloading 1 full Herbie files" not in caplog.text


def test_download_retry_raises_after_exhausting_attempts(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.grib2"
    f2 = tmp_path / "f2.grib2"

    refs = [_FakeRef(str(f1)), _FakeRef(str(f2))]
    herbie = _FakeHerbie(refs)

    def _fake_download(_search, verbose=False, overwrite=False):
        herbie.download_calls += 1
        f1.write_text("ok", encoding="utf-8")
        if f2.exists():
            f2.unlink()
        return [str(f1)]

    herbie.download = _fake_download
    monkeypatch.setattr(ingest_utils.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="downloaded test paths"):
        ingest_utils.download_herbie_with_retry(
            herbie_obj=herbie,
            search=":TMP:",
            expected_count=2,
            dataset_name="test",
            retries=2,
            retry_sleep_s=1,
        )

    assert herbie.download_calls == 2


def test_download_retry_uses_download_results_not_stale_file_exists(
    tmp_path, monkeypatch
):
    f1 = tmp_path / "f1.grib2"
    f2 = tmp_path / "f2.grib2"
    f1.write_text("old", encoding="utf-8")
    f2.write_text("old", encoding="utf-8")

    refs = [_FakeRef(str(f1)), _FakeRef(str(f2))]
    herbie = _FakeHerbie(refs)

    def _fake_download(_search, verbose=False, overwrite=False):
        herbie.download_calls += 1
        f1.write_text("new", encoding="utf-8")
        return [str(f1)]

    herbie.download = _fake_download
    monkeypatch.setattr(ingest_utils.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="downloaded test paths"):
        ingest_utils.download_herbie_with_retry(
            herbie_obj=herbie,
            search=":TMP:",
            expected_count=2,
            dataset_name="test",
            retries=2,
            retry_sleep_s=1,
        )

    assert herbie.download_calls == 2


def test_download_retry_recovers_from_503_with_limited_threads(tmp_path, monkeypatch):
    grib_path = tmp_path / "aifs.grib2"
    herbie = _FakeHerbie([_FakeRef(str(grib_path))])
    calls = []

    def _fake_download(_search, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ingest_utils.requests.HTTPError("503 Slow Down")
        grib_path.write_bytes(b"grib")
        return [str(grib_path)]

    herbie.download = _fake_download
    monkeypatch.setattr(ingest_utils.time, "sleep", lambda _seconds: None)

    ingest_utils.download_herbie_with_retry(
        herbie_obj=herbie,
        search=":TP:",
        expected_count=1,
        dataset_name="AIFS",
        retries=2,
        retry_sleep_s=1,
        max_threads=4,
        verbose=False,
    )

    assert [call["max_threads"] for call in calls] == [4, 4]
    assert [call["overwrite"] for call in calls] == [False, True]
    assert [call["verbose"] for call in calls] == [False, False]


def test_herbie_nomads_pacing_covers_probes_indexes_and_byte_ranges(monkeypatch):
    from herbie import core

    clock = [0.0]
    calls = []
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    def request(url, **kwargs):
        calls.append((url, clock[0], kwargs))
        response = ingest_utils.requests.Response()
        response.status_code = 200
        response.headers["Content-Length"] = "11"
        return response

    monkeypatch.setattr(ingest_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ingest_utils.time, "sleep", sleep)
    monkeypatch.setattr(ingest_utils.requests, "head", request)
    monkeypatch.setattr(ingest_utils.requests, "get", request)
    monkeypatch.setattr(core, "requests", ingest_utils.requests)
    monkeypatch.setattr(core.Herbie, "_check_grib", core.Herbie._check_grib)
    monkeypatch.setattr(core.Herbie, "_check_idx", core.Herbie._check_idx)
    monkeypatch.setattr(
        core.Herbie, "_pirate_weather_request_timeout_s", None, raising=False
    )
    ingest_utils.configure_herbie_request_timeouts(7, nomads_request_interval_s=5)
    client = core.requests
    # The shared download helper calls configuration again; pacing must survive it.
    ingest_utils.configure_herbie_request_timeouts(7)
    assert core.requests is client

    nomads_url = "https://nomads.ncep.noaa.gov/pub/data/nbm.grib2"
    assert core.Herbie._check_grib(object(), nomads_url)
    core.requests.get(nomads_url + ".idx")
    core.requests.get(nomads_url, headers={"Range": "bytes=0-99"})
    core.requests.get("https://noaa-nbm-grib2-pds.s3.amazonaws.com/nbm.grib2")

    assert sleeps == [5, 5]
    assert [start for _, start, _ in calls] == [0, 5, 10, 10]
    assert all(kwargs["allow_redirects"] is False for _, _, kwargs in calls[:3])
    assert "allow_redirects" not in calls[3][2]
    assert calls[1][2]["timeout"] == (10, 7)


def test_nomads_rate_limit_redirect_is_an_error(monkeypatch):
    response = ingest_utils.requests.Response()
    response.status_code = 302
    closed = []
    monkeypatch.setattr(response, "close", lambda: closed.append(True))

    def request(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        return response

    monkeypatch.setattr(ingest_utils.requests, "get", request)
    client = ingest_utils._HerbieRequests(interval_s=5, timeout_s=7)
    with pytest.raises(
        ingest_utils.requests.HTTPError, match="NOMADS returned HTTP 302"
    ):
        client.get("https://nomads.ncep.noaa.gov/pub/data/nbm.grib2")
    assert closed == [True]
