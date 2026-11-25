"""NWS alert ingest: active-alert merge, duplicate handling and grid output."""

import datetime
import io
import json
import logging
import tarfile

import geopandas as gp
import numpy as np
import zarr
from shapely.geometry import box

from API import alerts
from ingest import NWS_Alerts_Local as nws

# Modelled on the 2026-10-22 Peachtree City SPS (issue 698): the original, a
# RESENT copy with the same issuance time, and a correction 26 minutes later
# that references the original. All three were active at once.
ORIGINAL_ID = "urn:oid:2.49.0.1.840.0.original.001.1"
RESENT_ID = "urn:oid:2.49.0.1.840.0.resent.001.1"
CORRECTION_ID = "urn:oid:2.49.0.1.840.0.correction.001.1"
EXPIRATION = "2026-10-22T13:15:00-04:00"
CORRECTION_EFFECTIVE = "2026-10-22T01:28:00-04:00"

# A 2x2 grid; only (34.1, -84.5) is inside the zone polygons below.
GRID_LONS = np.array([-84.5, -84.0])
GRID_LATS = np.array([34.1, 35.0])


def _active_alert(cap_id, sent, references):
    """One JSON-LD record from api.weather.gov/alerts/active."""
    return {
        "id": cap_id,
        "geometry": None,
        "headline": "Special Weather Statement issued October 22",
        "event": "Special Weather Statement",
        "description": "Relative humidities of 28 percent or less are expected.",
        "areaDesc": "Cherokee; Forsyth; Cobb",
        "severity": "Moderate",
        "sent": sent,
        "effective": sent,
        "expires": EXPIRATION,
        "references": [{"identifier": ref} for ref in references],
    }


def _empty_tar_gz():
    """Stand-in for current_all.tar.gz; main() reads the shapefile separately."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz"):
        pass
    return buffer.getvalue()


def test_resent_and_corrected_alert_written_once(tmp_path, monkeypatch):
    """Protect the ingest/API alert contract against issue 698: NWS can keep an
    alert, a RESENT copy and its correction active together (each its own CAP
    message), and one alert can be listed on overlapping zone polygons. The
    API must see exactly one alert in the cell, with the correction's time.
    """
    zone = box(-84.7, 34.0, -84.3, 34.3)
    shapefile_rows = gp.GeoDataFrame(
        {
            "CAP_ID": [ORIGINAL_ID, RESENT_ID, CORRECTION_ID, CORRECTION_ID],
            "URL": "https://api.weather.gov/alerts/example",
            "EXPIRATION": EXPIRATION,
            "PROD_TYPE": "Special Weather Statement",
            "WFO": "KFFC",
            "SIG": "",
        },
        geometry=[zone, zone, zone, box(-84.6, 34.05, -84.4, 34.2)],
        crs="EPSG:4326",
    )
    active_alerts = [
        _active_alert(ORIGINAL_ID, "2026-10-22T01:02:00-04:00", []),
        _active_alert(RESENT_ID, "2026-10-22T01:02:00-04:00", []),
        _active_alert(CORRECTION_ID, CORRECTION_EFFECTIVE, [ORIGINAL_ID]),
    ]

    class FakeWxAPI:
        def __init__(self, useragent_id):
            pass

        def active_alerts(self, return_format=None):
            return {"@graph": active_alerts}

    tar_bytes = _empty_tar_gz()

    def fake_get(url, *args, **kwargs):
        # Every shapefile row has an active record, so a .cap fallback fetch
        # here would mean a superseded message was being added back.
        assert url.endswith("current_all.tar.gz"), url
        return type("Response", (), {"content": tar_bytes})()

    monkeypatch.setattr(nws.requests, "get", fake_get)
    monkeypatch.setattr(nws.gp, "read_file", lambda path: shapefile_rows.copy())
    monkeypatch.setattr(nws.nwswx, "WxAPI", FakeWxAPI)

    process_dir = tmp_path / "process"
    forecast_dir = tmp_path / "forecast"
    monkeypatch.setattr(nws, "forecast_process_dir", str(process_dir))
    monkeypatch.setattr(nws, "tmpDIR", str(process_dir / "Downloads"))
    monkeypatch.setattr(nws, "forecast_path", str(forecast_dir))
    monkeypatch.setattr(nws, "historic_path", str(tmp_path / "history"))
    monkeypatch.setattr(nws, "saveType", "Download")
    # main() grids the CONUS at 0.025 degrees (~2.6M points). Shrink both the
    # ingest grid and the API's matching lookup axes to the same 2x2 grid.
    monkeypatch.setattr(
        nws,
        "np",
        type(
            "SmallGridNumpy",
            (),
            {
                "arange": staticmethod(
                    lambda start, stop, step: GRID_LONS if start < 0 else GRID_LATS
                ),
                "meshgrid": staticmethod(np.meshgrid),
            },
        ),
    )
    monkeypatch.setattr(alerts, "NWS_ALERT_LONS", GRID_LONS)
    monkeypatch.setattr(alerts, "NWS_ALERT_LATS", GRID_LATS)

    nws.main()

    published = forecast_dir / nws.ingestVersion
    version = json.loads((published / "NWS_Alerts.current.json").read_text())["store"]
    alert_grid = zarr.open_array(str(published / version), mode="r")
    cell_alerts = alerts._read_nws_alerts(
        lat=34.1,
        az_lon=-84.5,
        alerts_zarr=alert_grid,
        now_utc=datetime.datetime(2026, 10, 22, 12, 0, tzinfo=datetime.UTC),
        logger=logging.getLogger(__name__),
        loc_tag="test",
    )

    assert len(cell_alerts) == 1
    correction_time = datetime.datetime.fromisoformat(CORRECTION_EFFECTIVE)
    assert cell_alerts[0]["time"] == int(correction_time.timestamp())
    assert alert_grid[1, 1] == ""
