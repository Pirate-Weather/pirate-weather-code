"""Exercise AIGEFS member downloads without starting a full ensemble ingest."""

from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from API import ingest_grib_utils
from ingest import AIGEFS_Local_Ingest
from ingest.AIGEFS_Local_Ingest import download_aigefs_member


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
    so an incomplete ensemble member is never processed and published. Each
    file must be fetched straight from its NOMADS URL (one paced request, no
    Herbie probes).
    """
    hours = list(range(6, 241, 6))
    run_date = pd.Timestamp("2026-10-14 12:00")
    fetched = []
    attempt = {"n": 0}

    def download(url, out_file, **kwargs):
        if url.endswith("f006.grib2"):
            attempt["n"] += 1
        fetched.append(url)
        if url.endswith("f240.grib2") and not (recover and attempt["n"] > 1):
            raise requests.HTTPError(f"404 Client Error: Not Found for url: {url}")
        return str(out_file)

    monkeypatch.setattr(AIGEFS_Local_Ingest, "tmp_dir", str(tmp_path))
    monkeypatch.setattr(AIGEFS_Local_Ingest, "wgrib2_path", "/unused/wgrib2 ")
    monkeypatch.setattr(AIGEFS_Local_Ingest, "AIGEFS_DOWNLOAD_RETRIES", 3)
    monkeypatch.setattr(AIGEFS_Local_Ingest, "download_grib_file", download)
    monkeypatch.setattr(AIGEFS_Local_Ingest.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        ingest_grib_utils, "run_checked", lambda *args: SimpleNamespace(stdout="stats")
    )
    monkeypatch.setattr(
        ingest_grib_utils, "validate_grib_stats", lambda check, **kwargs: True
    )

    if recover:
        paths = download_aigefs_member(run_date, hours, 1, "AIGEFS member 1")
        assert len(paths) == len(hours) and paths[-1].endswith("f240.grib2")
        assert attempt["n"] == 2
    else:
        with pytest.raises(requests.HTTPError, match="f240"):
            download_aigefs_member(run_date, hours, 1, "AIGEFS member 1")
        assert attempt["n"] == 3
    assert fetched[0] == (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/"
        "aigefs.20261014/12/mem001/model/atmos/grib2/aigefs.t12z.sfc.f006.grib2"
    )
