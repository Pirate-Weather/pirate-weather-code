"""End-to-end check of the LibreWXR sidecar export.

A synthetic LibreWXR cache (``state.json`` plus region memmaps, exactly what the
LibreWXR data pipeline writes) must come out as a ``LibreWXR.zarr`` in the final
store layout, with true dBZ inside radar coverage and NaN outside it. It protects
the ingest/API contract for the minutely block: a wrong grid orientation, pixel
decode or coverage test would silently put radar echoes in the wrong place or
replace real echoes with model data. Needs the ``librewxr`` package (the sidecar
runs beside LibreWXR), so it is skipped where that is not installed.
"""

import gzip
import json
import time

import numpy as np
import pytest

pytest.importorskip("librewxr")
import zarr

from API.constants.shared_const import (
    LIBREWXR_GRID_HEIGHT,
    LIBREWXR_GRID_RESOLUTION_DEG,
    LIBREWXR_GRID_WIDTH,
)
from API.request.grid_indexing import (
    LIBREWXR_LAT_START,
    LIBREWXR_LON_START,
    _nearest_regular_grid_index,
)
from ingest import LibreWXR_Local_Ingest as sidecar

# Guam: inside the GUCOMP radar range; the test cells are the radar site itself
GUAM_LAT, GUAM_LON = 13.45, 144.8
# London: far from every region, so no radar coverage
LONDON_LAT, LONDON_LON = 51.5, -0.1
# uint8 pixel = (dBZ + 32) * 2
PIXEL_40_DBZ = 144
PIXEL_2_DBZ = 68


def _write_memmap(path, shape, value):
    pixels = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    pixels[:] = value
    pixels.flush()


def _grid_cell(lat, lon):
    y = _nearest_regular_grid_index(
        lat, LIBREWXR_LAT_START, -LIBREWXR_GRID_RESOLUTION_DEG, LIBREWXR_GRID_HEIGHT
    )
    x = _nearest_regular_grid_index(
        lon, LIBREWXR_LON_START, LIBREWXR_GRID_RESOLUTION_DEG, LIBREWXR_GRID_WIDTH
    )
    return y, x


def test_export_publishes_radar_composite_for_api(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "radar").mkdir(parents=True)
    (cache / "nowcast").mkdir()
    shape = (1000, 1000)  # GUCOMP's grid

    # The nowcast must be recent or the sidecar refuses to publish it
    t0 = int(time.time()) // 600 * 600
    t1 = t0 + 600
    _write_memmap(cache / "radar" / f"{t0}_GUCOMP.dat", shape, PIXEL_40_DBZ)
    _write_memmap(cache / "nowcast" / f"frame_{t1}_GUCOMP.dat", shape, PIXEL_2_DBZ)
    state = {
        "version": 1,
        "written_at": t0,
        "stores": {
            "frame_store": {
                "max_frames": 12,
                "memmap_dir": str(cache / "radar"),
                "frames": [
                    {
                        "timestamp": t0,
                        "regions": {"GUCOMP": [f"{t0}_GUCOMP.dat", "|u1", list(shape)]},
                    }
                ],
                "frame_versions": {str(t0): 1},
            },
            "nowcast_store": {
                "memmap_dir": str(cache / "nowcast"),
                "frames": [
                    {
                        "timestamp": t1,
                        "blend_weight": 0.82,
                        "regions": {
                            "GUCOMP": [f"frame_{t1}_GUCOMP.dat", "|u1", list(shape)]
                        },
                    }
                ],
                "flows": {},
                "nwp_flow": None,
                "flow_version": 1,
            },
        },
    }
    (cache / "state.json").write_bytes(gzip.compress(json.dumps(state).encode()))

    monkeypatch.setattr(sidecar, "forecast_process_dir", str(tmp_path / "work"))
    remote = sidecar.open_remote(str(tmp_path / "prod"), "Download")
    lib = sidecar.load_librewxr()
    sidecar.prepare_coverage_masks(lib, cache)

    assert sidecar.export_nowcast(lib, cache, remote, zarr_store_workers=1)
    # The same nowcast is not published twice
    assert not sidecar.export_nowcast(lib, cache, remote, zarr_store_workers=1)

    pointer = json.loads((tmp_path / "prod" / "LibreWXR.current.json").read_text())
    store = zarr.open(str(tmp_path / "prod" / pointer["store"]), mode="r")
    assert store.attrs["time"] == [t0, t1]
    assert store.attrs["variables"] == ["REFC_composite"]
    assert store.shape == (1, 2, LIBREWXR_GRID_HEIGHT, LIBREWXR_GRID_WIDTH)

    y, x = _grid_cell(GUAM_LAT, GUAM_LON)
    assert store[0, 0, y, x] == pytest.approx(40.0)
    # Below REFC_THRESHOLD counts as no echo inside coverage
    assert store[0, 1, y, x] == 0.0
    y, x = _grid_cell(LONDON_LAT, LONDON_LON)
    assert np.isnan(store[0, :, y, x]).all()
