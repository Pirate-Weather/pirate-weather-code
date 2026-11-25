"""API.store_sync resumes a store copy that a failed pass left unfinished."""

import asyncio
import json

import obstore
import obstore.store
import pytest

from API import store_sync


def _publish(root, version: str, files: dict[str, bytes]) -> None:
    for key, data in files.items():
        path = root / version / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    model = version.split("/")[0]
    (root / f"{model}.current.json").write_text(json.dumps({"store": version}))


def test_failed_pass_resumes_instead_of_restarting(tmp_path, monkeypatch):
    """Across the Atlantic, one GFS chunk can fail after obstore's own retries.
    The next pass must keep the chunks it already has, not copy the store again."""
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    local_root.mkdir()
    version = "GFS/20261125T112628Z.zarr"
    chunks = {f"c/0/0/{i}": bytes([i]) * 10 for i in range(20)}
    _publish(remote_root, version, {"zarr.json": b"{}", **chunks})
    remote = obstore.store.LocalStore(str(remote_root))

    real_get = obstore.get_async
    fetched: list[str] = []
    failing = {f"{version}/c/0/0/7"}

    async def flaky_get(store, key, *args, **kwargs):
        if key in failing:
            failing.discard(key)
            await asyncio.sleep(0.5)  # a slow GET: the other chunks finish first
            raise obstore.exceptions.GenericError("Error performing GET: timed out")
        fetched.append(key)
        return await real_get(store, key, *args, **kwargs)

    monkeypatch.setattr(store_sync.obstore, "get_async", flaky_get)

    with pytest.raises(obstore.exceptions.GenericError):
        store_sync.sync_store(remote, "GFS", str(local_root))
    assert not (local_root / version).exists()
    assert not (local_root / "GFS.current.json").exists()
    partial = local_root / f"{version}.partial"
    kept = {
        f"{version}/{path.relative_to(partial)}"
        for path in partial.rglob("*")
        if path.is_file() and path.suffix != ".tmp"
    }
    assert kept and f"{version}/c/0/0/7" not in kept

    fetched.clear()
    assert store_sync.sync_store(remote, "GFS", str(local_root))
    # Only what the failed pass lacked is fetched again.
    assert set(fetched).isdisjoint(kept)
    assert f"{version}/c/0/0/7" in fetched
    for key, data in chunks.items():
        assert (local_root / version / key).read_bytes() == data
    assert not list((local_root / version).rglob("*.tmp"))
    assert json.loads((local_root / "GFS.current.json").read_text()) == {
        "store": version
    }


def test_partial_of_a_superseded_version_is_deleted(tmp_path):
    """If the pointer moves on before a copy finishes, the old partial copy is
    never needed again and must not fill the disk."""
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    stale = local_root / "GFS" / "20261125T052628Z.zarr.partial"
    (stale / "c").mkdir(parents=True)
    (stale / "c" / "0").write_bytes(b"old")
    version = "GFS/20261125T112628Z.zarr"
    _publish(remote_root, version, {"zarr.json": b"{}", "c/0": b"new"})

    remote = obstore.store.LocalStore(str(remote_root))
    assert store_sync.sync_store(remote, "GFS", str(local_root))
    assert not stale.exists()
    assert (local_root / version / "c" / "0").read_bytes() == b"new"
