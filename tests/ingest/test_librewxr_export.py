"""End-to-end check of the LibreWXR sidecar export.

A synthetic LibreWXR cache (``state.json`` plus region memmaps, exactly what the
LibreWXR data pipeline writes) must come out as a ``LibreWXR.zarr`` in the final
store layout, with true dBZ inside radar coverage and NaN outside it. It protects
the ingest/API contract for the minutely block: a wrong grid orientation, pixel
decode or coverage test would silently put radar echoes in the wrong place or
replace real echoes with model data. The same export carries LibreWXR's satellite
lightning as the nearest-strike distance and bearing at T+0, which the API serves as
the current nearest storm: a wrong coverage or staleness rule would report "no storm"
where a satellite cannot see one. Needs the ``librewxr`` package (the sidecar runs
beside LibreWXR), so it is skipped where that is not installed.
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
# A GOES-19 strike at Miami; Fort Lauderdale is ~40 km north of it, and Denver is
# inside GOES-19 coverage but over 300 km from the strike
MIAMI_LAT, MIAMI_LON = 25.76, -80.19
FORT_LAUDERDALE_LAT, FORT_LAUDERDALE_LON = 26.12, -80.14
DENVER_LAT, DENVER_LON = 39.74, -104.99
# LibreWXR's lightning point dtype (librewxr.data.lightning_store)
LIGHTNING_POINT_DTYPE = np.dtype(
    [
        ("time_s", "int64"),
        ("lat", "float32"),
        ("lon", "float32"),
        ("energy", "float32"),
        ("satellite", "int8"),
    ]
)
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

    # Lightning: GOES-19 saw two flashes at Miami, and GOES-18 reported one of them
    # too (the satellites overlap there), so the count must be 2, not 3. Meteosat-12's
    # newest data is 3 h old, so its coverage (Europe) is dropped rather than read
    # as "no lightning"
    now = int(time.time())
    (cache / "lightning").mkdir()
    points = np.array(
        [
            (now - 120, MIAMI_LAT, MIAMI_LON, 1e-14, 19),
            (now - 90, MIAMI_LAT - 0.01, MIAMI_LON, 1e-14, 19),
            (now - 120, MIAMI_LAT, MIAMI_LON, 1e-14, 18),
        ],
        dtype=LIGHTNING_POINT_DTYPE,
    )
    meta = {
        "satellites": {
            "goes18": {"last_e": now - 60},
            "goes19": {"last_e": now - 60},
            "meteosat12": {"last_e": now - 3 * 3600},
        }
    }
    np.savez_compressed(
        cache / "lightning" / "current.npz",
        points=points,
        meta=np.array(json.dumps(meta)),
    )

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
    assert store.attrs["variables"] == [
        "REFC_composite",
        "Lightning_Distance",
        "Lightning_Direction",
        "Lightning_Count",
    ]
    assert store.shape == (4, 2, LIBREWXR_GRID_HEIGHT, LIBREWXR_GRID_WIDTH)

    y, x = _grid_cell(GUAM_LAT, GUAM_LON)
    assert store[0, 0, y, x] == pytest.approx(40.0)
    # Below REFC_THRESHOLD counts as no echo inside coverage
    assert store[0, 1, y, x] == 0.0
    y, x = _grid_cell(LONDON_LAT, LONDON_LON)
    assert np.isnan(store[:, :, y, x]).all()

    # Nearest strike: ~40 km south (a little west) of Fort Lauderdale, at T+0 only
    y, x = _grid_cell(FORT_LAUDERDALE_LAT, FORT_LAUDERDALE_LON)
    assert store[1, 0, y, x] == pytest.approx(40_000, rel=0.05)
    assert store[2, 0, y, x] == pytest.approx(187, abs=5)
    assert np.isnan(store[1:, 1, y, x]).all()
    # Beyond 15 km, so no strikes counted
    assert store[3, 0, y, x] == 0
    y, x = _grid_cell(MIAMI_LAT, MIAMI_LON)
    assert store[1, 0, y, x] < 5_000
    assert store[3, 0, y, x] == 2
    # Covered with no strike in range: lightning rules a nearer storm out
    y, x = _grid_cell(DENVER_LAT, DENVER_LON)
    assert np.isposinf(store[1, 0, y, x])
    assert store[3, 0, y, x] == 0


def test_stale_nowcast_is_not_exported_under_new_radar_frame(tmp_path):
    """LibreWXR also writes state.json mid-cycle, after a new radar frame lands
    but before its nowcast exists. Exporting then would publish the previous
    frame's extrapolation under the new run time and block the real nowcast as
    already published, so the snapshot reader must refuse it.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    t_old, t_new = 1_793_650_200, 1_793_650_800
    state = {
        "version": 1,
        "written_at": t_new,
        "stores": {
            "frame_store": {
                "max_frames": 12,
                "memmap_dir": str(cache / "radar"),
                "frames": [
                    {"timestamp": t_old, "regions": {}},
                    {"timestamp": t_new, "regions": {}},
                ],
                "frame_versions": {},
            },
            "nowcast_store": {
                "memmap_dir": str(cache / "nowcast"),
                # Steps from the old frame: the first one coincides with t_new
                "frames": [
                    {"timestamp": t_old + 600 * k, "blend_weight": 0.5, "regions": {}}
                    for k in range(1, 4)
                ],
                "flows": {},
                "nwp_flow": None,
                "flow_version": 1,
            },
        },
    }
    (cache / "state.json").write_bytes(gzip.compress(json.dumps(state).encode()))

    lib = sidecar.load_librewxr()
    with pytest.raises(LookupError, match="waiting for this cycle"):
        sidecar.read_nowcast_snapshot(lib, cache)
