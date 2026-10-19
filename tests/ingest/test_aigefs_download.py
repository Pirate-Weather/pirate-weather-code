"""Exercise AIGEFS member downloads without starting a full ensemble ingest."""

from types import SimpleNamespace

import pandas as pd
import pytest

from API import AIGEFS_Local_Ingest, ingest_grib_utils, ingest_utils
from API.AIGEFS_Local_Ingest import download_aigefs_member


@pytest.mark.parametrize(
    "recover",
    [
        pytest.param(True, id="late-files-recover"),
        pytest.param(False, id="still-missing-aborts"),
    ],
)
def test_aigefs_member_download_retries_missing_files_then_aborts(
    tmp_path, monkeypatch, recover
):
    """NOMADS often publishes AIGEFS forecast hours late. A member missing a
    forecast-hour file must be retried a bounded number of times and then raise,
    so an incomplete ensemble member is never processed and published.
    """
    attempts = []
    hours = list(range(6, 241, 6))
    run_date = pd.Timestamp("2026-10-14 12:00")

    class Ref:
        def __init__(self, hour):
            self.path = tmp_path / f"f{hour:03d}.grib2"

        def get_localFilePath(self, search):
            return self.path

    def fast_herbie(dates, **kwargs):
        attempts.append(kwargs)
        assert dates == [run_date]
        assert kwargs["model"] == "aigefs"
        assert kwargs["member"] == "mem001"
        # The last forecast hour is unavailable until the second attempt (if ever).
        available_hours = hours if recover and len(attempts) > 1 else hours[:-1]
        return SimpleNamespace(file_exists=[Ref(hour) for hour in available_hours])

    def download(ref, **kwargs):
        ref.path.write_bytes(b"GRIB data")
        return str(ref.path)

    monkeypatch.setattr(AIGEFS_Local_Ingest, "tmp_dir", str(tmp_path))
    monkeypatch.setattr(AIGEFS_Local_Ingest, "wgrib2_path", "/unused/wgrib2 ")
    monkeypatch.setattr(AIGEFS_Local_Ingest, "AIGEFS_DOWNLOAD_RETRIES", 3)
    monkeypatch.setattr(ingest_grib_utils, "FastHerbie", fast_herbie)
    monkeypatch.setattr(
        ingest_grib_utils, "configure_herbie_request_timeouts", lambda: None
    )
    monkeypatch.setattr(ingest_utils, "_download_full_herbie_file", download)
    monkeypatch.setattr(ingest_grib_utils.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        ingest_grib_utils, "run_checked", lambda *args: SimpleNamespace(stdout="stats")
    )
    monkeypatch.setattr(
        ingest_grib_utils, "validate_grib_stats", lambda check, **kwargs: True
    )

    if recover:
        paths = download_aigefs_member(run_date, hours, 1, "AIGEFS member 1")
        assert paths == [str(tmp_path / f"f{hour:03d}.grib2") for hour in hours]
        assert len(attempts) == 2
    else:
        with pytest.raises(RuntimeError, match="downloaded AIGEFS member 1 paths"):
            download_aigefs_member(run_date, hours, 1, "AIGEFS member 1")
        assert len(attempts) == 3
