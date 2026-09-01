import pandas as pd

from API import ingest_grib_utils


class _FakeRef:
    def __init__(self, path):
        self.path = path

    def get_localFilePath(self, _search):
        return self.path


def test_download_subset_rebuilds_fast_herbie_between_retries(tmp_path, monkeypatch):
    f1 = tmp_path / "f1.grib2"
    f2 = tmp_path / "f2.grib2"

    class _FakeFastHerbie:
        calls = 0

        def __init__(self, *_args, **_kwargs):
            type(self).calls += 1
            if type(self).calls == 1:
                self.file_exists = [_FakeRef(str(f1))]
            else:
                self.file_exists = [_FakeRef(str(f1)), _FakeRef(str(f2))]

    def _fake_download_herbie_with_retry(
        *, herbie_obj, expected_count, **_kwargs
    ):
        if len(herbie_obj.file_exists) != expected_count:
            raise RuntimeError("missing availability refs")
        f1.write_text("ok", encoding="utf-8")
        f2.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(ingest_grib_utils, "FastHerbie", _FakeFastHerbie)
    monkeypatch.setattr(
        ingest_grib_utils,
        "download_herbie_with_retry",
        _fake_download_herbie_with_retry,
    )
    monkeypatch.setattr(ingest_grib_utils, "configure_herbie_request_timeouts", lambda: None)
    monkeypatch.setattr(ingest_grib_utils.time, "sleep", lambda _seconds: None)

    paths = ingest_grib_utils.download_and_validate_gfs_subset(
        model="gdps",
        product="15km",
        search=None,
        dataset_name="test",
        base_time=pd.Timestamp("2026-08-31 00:00"),
        wgrib2_exe="/does/not/matter",
        forecast_hours=[1, 2],
        expected_forecast_hours=[1, 2],
        skip_wgrib2_validation=True,
        herbie_save_dir=str(tmp_path),
        herbie_download_retries=2,
        herbie_retry_sleep_seconds=1,
    )

    assert _FakeFastHerbie.calls == 2
    assert paths == [str(f1), str(f2)]
