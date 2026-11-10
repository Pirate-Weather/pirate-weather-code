"""Support pickles are reloaded by the running store poll."""

import logging
import os
import pickle

import pytest

from API.io.store_reader import load_stores, local_store_root, refresh_stores


def _publish(directory, name: str, payload: bytes) -> None:
    # Same atomic replace as store_sync.sync_file
    tmp_path = directory / f"{name}.tmp"
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, directory / name)


@pytest.mark.asyncio
async def test_store_poll_reloads_changed_support_pickles(tmp_path):
    """A worker runs for days while the sync replaces the DWD MOSMIX station map
    and the RAQDPS grid. Loaded only at startup, a worker served a stale station
    map (wrong nearest-station flags) until restarted. The poll must pick up a
    replaced file, and a damaged replacement must keep the copy already held
    rather than drop the DWD flags or the RAQDPS source. Until a RAQDPS grid
    is published, the copy bundled with the API must be used.
    """
    root = local_store_root(str(tmp_path))
    logger = logging.getLogger("test")
    stores, _ = await load_stores(
        stage="PROD",
        save_dir=str(tmp_path),
        use_etopo=False,
        save_type="Download",
        s3_bucket="",
        logger=logger,
    )
    assert stores.DWD_MOSMIX_Stations is None
    assert stores.RAQDPS_LatLon is not None  # the bundled API/data copy

    first_map = {(10, 20): [{"id": "A", "lat": 50.0, "lon": 8.0}]}
    _publish(tmp_path, "DWD_MOSMIX_stations.pickle", pickle.dumps(first_map))
    _publish(tmp_path, "RAQDPS.lat_lon.pickle", pickle.dumps({"lat": [1.0]}))
    await refresh_stores(stores, root, "PROD", logger=logger)
    assert stores.DWD_MOSMIX_Stations == first_map
    assert stores.RAQDPS_LatLon == {"lat": [1.0]}

    second_map = {(10, 20): [{"id": "B", "lat": 51.0, "lon": 9.0}]}
    _publish(tmp_path, "DWD_MOSMIX_stations.pickle", pickle.dumps(second_map))
    await refresh_stores(stores, root, "PROD", logger=logger)
    assert stores.DWD_MOSMIX_Stations == second_map
    assert stores.RAQDPS_LatLon == {"lat": [1.0]}

    _publish(tmp_path, "DWD_MOSMIX_stations.pickle", b"not a pickle")
    await refresh_stores(stores, root, "PROD", logger=logger)
    assert stores.DWD_MOSMIX_Stations == second_map
