"""Exercise AIGEFS member failures without starting a full ensemble ingest."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from API import ingest_grib_utils, ingest_utils

AIGEFS_SCRIPT = Path(__file__).resolve().parents[2] / "API" / "AIGEFS_Local_Ingest.py"


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("historic", [False, True])
def test_aigefs_member_download_retries_missing_files_and_stops_on_failure(
    tmp_path, monkeypatch, recover, historic
):
    attempts = []
    sleeps = []
    hours = [6] if historic else list(range(6, 241, 6))
    run_date = pd.Timestamp("2026-10-14 12:00")

    class Ref:
        def __init__(self, hour):
            self.path = tmp_path / f"f{hour:03d}.grib2"

        def get_localFilePath(self, search):
            assert search is None
            return self.path

    def fast_herbie(dates, **kwargs):
        attempts.append(kwargs)
        assert dates == [run_date]
        assert kwargs["model"] == "aigefs"
        assert kwargs["member"] == "mem001"
        assert kwargs["priority"] == ["nomads"]
        assert kwargs["max_threads"] == 4
        available_hours = hours if recover and len(attempts) > 1 else hours[:-1]
        return SimpleNamespace(file_exists=[Ref(hour) for hour in available_hours])

    def download(ref, **kwargs):
        ref.path.write_bytes(b"GRIB data")
        return str(ref.path)

    monkeypatch.setattr(ingest_grib_utils, "FastHerbie", fast_herbie)
    monkeypatch.setattr(
        ingest_grib_utils, "configure_herbie_request_timeouts", lambda: None
    )
    monkeypatch.setattr(ingest_utils, "_download_full_herbie_file", download)
    monkeypatch.setattr(ingest_grib_utils.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        ingest_grib_utils, "run_checked", lambda *args: SimpleNamespace(stdout="stats")
    )
    monkeypatch.setattr(
        ingest_grib_utils, "validate_grib_stats", lambda check, **kwargs: True
    )

    tree = ast.parse(AIGEFS_SCRIPT.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "download_aigefs_member"
    )
    namespace = {
        "pd": pd,
        "download_and_validate_gfs_subset": ingest_grib_utils.download_and_validate_gfs_subset,
        "AIGEFS_SOURCE_PRIORITY": ["nomads"],
        "AIGEFS_MAX_THREADS": 4,
        "AIGEFS_DOWNLOAD_RETRIES": 3,
        "AIGEFS_RETRY_SLEEP_SECONDS": 20,
        "tmp_dir": str(tmp_path),
        "wgrib2_path": "/unused/wgrib2 ",
    }
    exec(  # noqa: S102 - Execute only a helper from the trusted local ingest script.
        compile(
            ast.Module(body=[function], type_ignores=[]), str(AIGEFS_SCRIPT), "exec"
        ),
        namespace,
    )
    member_download = namespace["download_aigefs_member"]
    if recover:
        paths = member_download(run_date, hours, 1, "AIGEFS member 1")
        assert paths == [str(tmp_path / f"f{hour:03d}.grib2") for hour in hours]
        assert len(attempts) == 2
        assert sleeps == [20]
    else:
        with pytest.raises(RuntimeError, match="downloaded AIGEFS member 1 paths"):
            member_download(run_date, hours, 1, "AIGEFS member 1")
        assert len(attempts) == 3
        assert sleeps == [20, 40]


def test_aigefs_member_failure_stops_before_time_series_creation():
    tree = ast.parse(AIGEFS_SCRIPT.read_text())
    member_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "mem"
    )
    calls = []

    def download(*args):
        calls.append(args)
        raise RuntimeError("AIGEFS member 1 is incomplete")

    with pytest.raises(RuntimeError, match="AIGEFS member 1 is incomplete"):
        exec(  # noqa: S102 - Exercise the production loop without network or GRIB processing.
            compile(
                ast.Module(body=[member_loop], type_ignores=[]),
                str(AIGEFS_SCRIPT),
                "exec",
            ),
            {
                "AIGEFS_MEMBER_COUNT": 30,
                "base_time": pd.Timestamp("2026-10-14 12:00"),
                "aigefs_range": range(6, 241, 6),
                "download_aigefs_member": download,
            },
        )
    assert len(calls) == 1


def test_aigefs_does_not_continue_with_incomplete_forecast_timesteps():
    tree = ast.parse(AIGEFS_SCRIPT.read_text())
    validation = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Subscript)
        and isinstance(node.test.left.value, ast.Attribute)
        and isinstance(node.test.left.value.value, ast.Name)
        and node.test.left.value.value.id == "xarray_wgrib"
    )
    with pytest.raises(RuntimeError, match="39 forecast timesteps; expected 40"):
        exec(  # noqa: S102 - Exercise only the production timestep guard.
            compile(
                ast.Module(body=[validation], type_ignores=[]),
                str(AIGEFS_SCRIPT),
                "exec",
            ),
            {
                "xarray_wgrib": SimpleNamespace(sizes={"time": 39}),
                "aigefs_range": range(6, 241, 6),
                "mem": 1,
            },
        )
