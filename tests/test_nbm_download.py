"""Test NBM's shared download checks without running the ingest script."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pandas as pd
import pytest
import requests
from herbie import FastHerbie

from API import ingest_grib_utils

NBM_SCRIPT = Path(__file__).resolve().parents[1] / "API" / "NBM_Local_Ingest.py"


def _nbm_download(namespace):
    tree = ast.parse(NBM_SCRIPT.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_download_nbm_subset"
    )
    exec(  # noqa: S102 - Run only the download helper from the trusted local script.
        compile(ast.Module(body=[function], type_ignores=[]), str(NBM_SCRIPT), "exec"),
        namespace,
    )
    return namespace["_download_nbm_subset"]


@pytest.mark.parametrize("failure", ["302", "missing", "empty", "recover"])
@pytest.mark.parametrize("historic", [False, True])
def test_nbm_download_rejects_silent_failures_and_retries(
    tmp_path, monkeypatch, failure, historic
):
    attempts = []
    base_time = pd.Timestamp("2026-10-14 12:00")
    run_date = base_time - pd.Timedelta(hours=1) if historic else None
    hours = [1] if historic else [1, 2]

    class Ref:
        def __init__(self, hour, attempt):
            self.hour = hour
            self.attempt = attempt
            self.path = tmp_path / f"f{hour}.grib2"

        def get_localFilePath(self, search):
            return self.path

        def download(self, search, **kwargs):
            assert kwargs["overwrite"] == (self.attempt > 1)
            if self.hour == hours[-1] and not (
                failure == "recover" and self.attempt > 1
            ):
                if failure in {"302", "recover"}:
                    # A worker error is logged and swallowed by FastHerbie.download.
                    raise requests.HTTPError("302 NOMADS rate limit")
                if failure == "missing":
                    return self.path
                self.path.write_bytes(b"")
                return self.path
            self.path.write_bytes(b"GRIB data")
            return self.path

    def fast_herbie(dates, **kwargs):
        attempts.append(kwargs)
        assert dates == [run_date or base_time]
        assert kwargs["model"] == "nbm"
        assert kwargs["max_threads"] == 1
        refs = [Ref(hour, len(attempts)) for hour in hours]
        obj = SimpleNamespace(file_exists=refs, tasks=len(refs))
        # Use the actual concurrent downloader to reproduce its error swallowing.
        obj.download = MethodType(FastHerbie.download, obj)
        return obj

    monkeypatch.setattr(ingest_grib_utils, "FastHerbie", fast_herbie)
    monkeypatch.setattr(
        ingest_grib_utils, "configure_herbie_request_timeouts", lambda: None
    )
    monkeypatch.setattr(ingest_grib_utils.time, "sleep", lambda seconds: None)
    namespace = {
        "download_and_validate_gfs_subset": ingest_grib_utils.download_and_validate_gfs_subset,
        "base_time": base_time,
        "tmp_dir": str(tmp_path),
        "herbie_download_retries": 2,
        "herbie_retry_sleep_seconds": 20,
        "herbie_max_threads": 1,
        "wgrib2_path": "/unused/wgrib2 ",
    }
    download = _nbm_download(namespace)

    if failure == "recover":
        paths = download(hours, ":TMP:", "NBM test", run_date=run_date)
        assert paths == [str(tmp_path / f"f{hour}.grib2") for hour in hours]
    else:
        with pytest.raises(RuntimeError, match="downloaded NBM test"):
            download(hours, ":TMP:", "NBM test", run_date=run_date)
    assert len(attempts) == 2


def test_all_nbm_download_stages_use_shared_checks():
    tree = ast.parse(NBM_SCRIPT.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Attribute) and node.func.attr == "download"
        for node in calls
    )
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "_download_nbm_subset"
            for node in calls
        )
        == 6
    )


def test_nbm_aborts_when_grib_validation_fails():
    tree = ast.parse(NBM_SCRIPT.read_text(encoding="utf-8"))
    validations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.operand, ast.Call)
        and isinstance(node.test.operand.func, ast.Name)
        and node.test.operand.func.id == "validate_grib_stats"
    ]
    assert len(validations) == 4
    for validation in validations:
        with pytest.raises(RuntimeError, match="NBM GRIB validation failed"):
            exec(  # noqa: S102 - Exercise the production guard without starting ingest.
                compile(
                    ast.Module(body=[validation], type_ignores=[]),
                    str(NBM_SCRIPT),
                    "exec",
                ),
                {"grib_check": None, "validate_grib_stats": lambda check: False},
            )
