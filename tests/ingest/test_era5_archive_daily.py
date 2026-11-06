"""ERA5 archive daily job: what gets built, promoted and deleted."""

import datetime as dt
import importlib.util
import sys
from pathlib import Path

DAILY = Path(__file__).resolve().parents[2] / "scripts" / "era5_archive_daily.py"


def _daily_module():
    spec = importlib.util.spec_from_file_location("era5_archive_daily", DAILY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def test_era5_daily_plan_promotes_appends_and_never_overwrites_in_place():
    """Protect the daily job's contract with readers: complete final weeks become
    immutable stores, changing weeks (ERA5T, the current partial week) are only
    ever republished as new versions, an immutable store holding ERA5T is replaced
    rather than overwritten, and old versions outlive their pointer by a run.
    A wrong plan either overwrites a store a reader has cached or leaves preliminary
    or missing data in place for good.
    """
    daily = _daily_module()
    d = dt.date.fromisoformat
    # Final ERA5 to Sun 2026-02-15, ERA5T to Wed 2026-03-04.
    final_stop, era5t_stop = d("2026-02-15"), d("2026-03-04")
    state = daily.ArchiveState(
        immutable={
            "2026-01-05",
            "2026-01-19",
            "2026-01-26",
            "2026-02-09",
            "2026-02-23",
        },
        immutable_era5t={"2026-02-09", "2026-02-23"},
        pointers={
            "2026-01-19": "2026-01-25T23",  # promoted earlier, pointer left over
            "2026-02-02": "2026-02-08T23",  # ERA5T version; final ERA5 now covers it
            "2026-02-16": "2026-02-22T23",  # ERA5T, up to date
            "2026-02-23": "2026-03-01T23",  # ERA5T version published, old immutable left
            "2026-03-02": "2026-03-03T23",  # current week, Google has a day more
        },
        version_dirs={
            "2025-12-29",
            "2026-01-19",
            "2026-01-26",
            "2026-02-02",
            "2026-02-16",
            "2026-02-23",
            "2026-03-02",
        },
    )
    plan = daily.plan_updates(state, d("2026-01-05"), final_stop, era5t_stop)

    iso = lambda weeks: [w.isoformat() for w in weeks]  # noqa: E731
    # 2026-01-12 is a gap; 2026-02-02 is now final ERA5.
    assert iso(plan.immutable) == ["2026-01-12", "2026-02-02"]
    assert plan.delete_pointer_after_immutable == ["2026-02-02"]
    # 2026-02-09 holds ERA5T in an immutable store: replaced by a version first.
    # 2026-03-02 is the current week and Google has more hours.
    assert iso(plan.versioned) == ["2026-02-09", "2026-03-02"]
    assert plan.delete_immutable_after_versioned == ["2026-02-09"]
    assert plan.delete_immutable == ["2026-02-23"]
    assert plan.delete_pointers == ["2026-01-19"]
    # Versions of a week promoted on an earlier run, and unreferenced versions.
    # 2026-01-19's versions stay one more run: its pointer goes only now.
    assert sorted(plan.delete_versions) == ["2025-12-29", "2026-01-26"]
    # Nothing is built for up-to-date weeks, and no week is built twice.
    built = iso(plan.immutable + plan.versioned)
    assert "2026-01-05" not in built and "2026-02-16" not in built
    assert len(built) == len(set(built))
