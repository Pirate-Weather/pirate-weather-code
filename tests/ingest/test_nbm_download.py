"""Test NBM's shared download checks without running the ingest script."""

from types import MethodType, SimpleNamespace

import pandas as pd
import pytest
import requests
from herbie import FastHerbie

from API import NBM_Local_Ingest, ingest_grib_utils
from API.NBM_Local_Ingest import _download_nbm_subset


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("302", id="nomads-302-swallowed"),
        pytest.param("missing", id="file-never-written"),
        pytest.param("empty", id="empty-file"),
        pytest.param("recover", id="recovers-on-retry"),
    ],
)
def test_nbm_download_rejects_silent_failures_and_retries(
    tmp_path, monkeypatch, failure
):
    """FastHerbie.download logs and swallows per-file worker errors (e.g. a NOMADS
    302 rate-limit redirect), so an NBM subset could "succeed" with a missing or
    empty GRIB file and be processed as if complete. The real FastHerbie.download
    is used here so its error swallowing is reproduced; NBM must raise after the
    bounded retries, and must succeed once the file arrives on a retry.
    """
    attempts = []
    base_time = pd.Timestamp("2026-10-14 12:00")
    hours = [1, 2]

    class Ref:
        def __init__(self, hour, attempt):
            self.hour = hour
            self.attempt = attempt
            self.path = tmp_path / f"f{hour}.grib2"

        def get_localFilePath(self, search):
            return self.path

        def download(self, search, **kwargs):
            if self.hour == hours[-1] and not (
                failure == "recover" and self.attempt > 1
            ):
                if failure in {"302", "recover"}:
                    raise requests.HTTPError("302 NOMADS rate limit")
                if failure == "missing":
                    return self.path
                self.path.write_bytes(b"")
                return self.path
            self.path.write_bytes(b"GRIB data")
            return self.path

    def fast_herbie(dates, **kwargs):
        attempts.append(kwargs)
        assert dates == [base_time]
        assert kwargs["model"] == "nbm"
        refs = [Ref(hour, len(attempts)) for hour in hours]
        obj = SimpleNamespace(file_exists=refs, tasks=len(refs))
        obj.download = MethodType(FastHerbie.download, obj)
        return obj

    monkeypatch.setattr(NBM_Local_Ingest, "tmp_dir", str(tmp_path))
    monkeypatch.setattr(NBM_Local_Ingest, "wgrib2_path", "/unused/wgrib2 ")
    monkeypatch.setattr(NBM_Local_Ingest, "herbie_download_retries", 2)
    monkeypatch.setattr(NBM_Local_Ingest, "herbie_max_threads", 1)
    monkeypatch.setattr(ingest_grib_utils, "FastHerbie", fast_herbie)
    monkeypatch.setattr(
        ingest_grib_utils, "configure_herbie_request_timeouts", lambda: None
    )
    monkeypatch.setattr(ingest_grib_utils.time, "sleep", lambda seconds: None)

    if failure == "recover":
        paths = _download_nbm_subset(hours, ":TMP:", "NBM test", base_time=base_time)
        assert paths == [str(tmp_path / f"f{hour}.grib2") for hour in hours]
    else:
        with pytest.raises(RuntimeError, match="downloaded NBM test"):
            _download_nbm_subset(hours, ":TMP:", "NBM test", base_time=base_time)
    assert len(attempts) == 2
