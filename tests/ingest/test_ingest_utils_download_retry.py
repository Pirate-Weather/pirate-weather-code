import pytest

from API import ingest_utils


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


@pytest.mark.parametrize(
    ("second_file_arrives", "retries"),
    [
        pytest.param(True, 3, id="recovers-on-retry"),
        pytest.param(False, 2, id="raises-after-last-attempt"),
    ],
)
def test_download_retry_is_bounded_and_requires_every_file(
    tmp_path, monkeypatch, second_file_arrives, retries
):
    """Partial downloads must be retried, but only a bounded number of times:
    an ingest that loops forever (or proceeds with a missing forecast hour)
    blocks or corrupts the published product.
    """
    f1 = tmp_path / "f1.grib2"
    f2 = tmp_path / "f2.grib2"
    herbie = _FakeHerbie([_FakeRef(str(f1)), _FakeRef(str(f2))])

    def _fake_download(_search, verbose=False, overwrite=False):
        herbie.download_calls += 1
        f1.write_text("ok", encoding="utf-8")
        if second_file_arrives and herbie.download_calls > 1:
            f2.write_text("ok", encoding="utf-8")
            return [str(f1), str(f2)]
        return [str(f1)]

    herbie.download = _fake_download
    monkeypatch.setattr(ingest_utils.time, "sleep", lambda _s: None)

    def run():
        ingest_utils.download_herbie_with_retry(
            herbie_obj=herbie,
            search=":TMP:",
            expected_count=2,
            dataset_name="test",
            retries=retries,
            retry_sleep_s=1,
        )

    if second_file_arrives:
        run()
    else:
        with pytest.raises(RuntimeError, match="downloaded test paths"):
            run()
    assert herbie.download_calls == 2


def test_download_retry_uses_download_results_not_stale_file_exists(
    tmp_path, monkeypatch
):
    """Regression: a stale GRIB file left over from a previous run must not
    count as downloaded. Completeness is judged from what this download returned,
    otherwise an old forecast hour would be silently mixed into a new run.
    """
    f1 = tmp_path / "f1.grib2"
    f2 = tmp_path / "f2.grib2"
    f1.write_text("old", encoding="utf-8")
    f2.write_text("old", encoding="utf-8")

    herbie = _FakeHerbie([_FakeRef(str(f1)), _FakeRef(str(f2))])

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


def test_herbie_nomads_pacing_covers_probes_indexes_and_byte_ranges(monkeypatch):
    """NOMADS rate-limits aggressive clients with 302 redirects (#772). Every
    NOMADS request Herbie makes (availability HEAD, .idx fetch, byte-range GET)
    must be paced and must not follow redirects, the pacing must survive the
    shared helper reconfiguring Herbie, and non-NOMADS sources (AWS) must not be
    slowed down.
    """
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


def test_full_file_nomads_download_uses_pacing_and_rejects_redirect(
    tmp_path, monkeypatch
):
    """Full-file downloads (search=None, e.g. AIGEFS) bypass Herbie.download, so
    they must still go through the paced NOMADS client (#772). A 302 rate-limit
    redirect must raise and leave neither the target file nor a temp file behind,
    otherwise the redirect's HTML body could be accepted as a GRIB file.
    """
    from herbie import core

    response = ingest_utils.requests.Response()
    response.status_code = 302
    closed = []
    monkeypatch.setattr(response, "close", lambda: closed.append(True))

    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        return response

    monkeypatch.setattr(ingest_utils.requests, "get", get)
    monkeypatch.setattr(core, "requests", ingest_utils._HerbieRequests(5, 120))
    path = tmp_path / "aigefs.grib2"
    with pytest.raises(
        ingest_utils.requests.HTTPError, match="NOMADS returned HTTP 302"
    ):
        ingest_utils.download_herbie_with_retry(
            herbie_obj=_FakeHerbie(
                [_FakeRef(str(path), grib="https://nomads.ncep.noaa.gov/aigefs.grib2")]
            ),
            expected_count=1,
            dataset_name="AIGEFS",
            retries=1,
            retry_sleep_s=1,
        )
    assert closed == [True]
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
