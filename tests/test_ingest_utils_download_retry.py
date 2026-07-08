import pytest

from API import ingest_utils


class _FakeRef:
    def __init__(self, path):
        self._path = path

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
