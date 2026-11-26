"""NWS alert ingest: active-alert merge, duplicate handling and grid output."""

import datetime
import io
import json
import logging
import tarfile

import geopandas as gp
import numpy as np
import pytest
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
ORIGINAL_SENT = "2026-10-22T01:02:00-04:00"
CORRECTION_EFFECTIVE = "2026-10-22T01:28:00-04:00"

# A 2x2 grid: ZONE covers only cell [0, 0] (34.1, -84.5), OTHER_ZONE only
# cell [1, 1] (35.0, -84.0).
GRID_LONS = np.array([-84.5, -84.0])
GRID_LATS = np.array([34.1, 35.0])
ZONE = box(-84.7, 34.0, -84.3, 34.3)
OTHER_ZONE = box(-84.2, 34.8, -83.8, 35.2)


def _active_alert(cap_id, sent, references, message_type="Alert"):
    """One JSON-LD record from api.weather.gov/alerts/active."""
    return {
        "id": cap_id,
        "geometry": None,
        "headline": "Special Weather Statement issued October 22",
        "event": "Special Weather Statement",
        "description": "Relative humidities of 28 percent or less are expected.",
        "areaDesc": "Cherokee; Forsyth; Cobb",
        "severity": "Moderate",
        "messageType": message_type,
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


@pytest.mark.parametrize(
    ("shapefile_polygons", "active_alerts", "cell_times", "other_cell_times"),
    [
        # Issue 698: original, RESENT copy and correction all active, and the
        # correction listed on overlapping polygons; one alert must remain.
        pytest.param(
            [
                (ORIGINAL_ID, ZONE),
                (RESENT_ID, ZONE),
                (CORRECTION_ID, ZONE),
                (CORRECTION_ID, box(-84.6, 34.05, -84.4, 34.2)),
            ],
            [
                _active_alert(ORIGINAL_ID, ORIGINAL_SENT, []),
                _active_alert(RESENT_ID, ORIGINAL_SENT, []),
                _active_alert(
                    CORRECTION_ID, CORRECTION_EFFECTIVE, [ORIGINAL_ID], "Update"
                ),
            ],
            [CORRECTION_EFFECTIVE],
            [],
            id="resent-and-corrected",
        ),
        # Race between the two downloads: the update is active but has no
        # shapefile polygons yet, so the original must keep covering its area.
        pytest.param(
            [(ORIGINAL_ID, ZONE)],
            [
                _active_alert(ORIGINAL_ID, ORIGINAL_SENT, []),
                _active_alert(
                    CORRECTION_ID, CORRECTION_EFFECTIVE, [ORIGINAL_ID], "Update"
                ),
            ],
            [ORIGINAL_SENT],
            [],
            id="update-without-polygons-yet",
        ),
        # A cancellation needs no polygons of its own to remove the coverage.
        pytest.param(
            [(ORIGINAL_ID, ZONE)],
            [
                _active_alert(ORIGINAL_ID, ORIGINAL_SENT, []),
                _active_alert(
                    CORRECTION_ID, CORRECTION_EFFECTIVE, [ORIGINAL_ID], "Cancel"
                ),
            ],
            [],
            [],
            id="cancelled",
        ),
        # A newer message with the same text on other polygons must not take
        # the older message's alert away from cells only the older one covers.
        pytest.param(
            [(ORIGINAL_ID, ZONE), (CORRECTION_ID, OTHER_ZONE)],
            [
                _active_alert(ORIGINAL_ID, ORIGINAL_SENT, []),
                _active_alert(CORRECTION_ID, CORRECTION_EFFECTIVE, []),
            ],
            [ORIGINAL_SENT],
            [CORRECTION_EFFECTIVE],
            id="same-text-other-polygons",
        ),
    ],
)
def test_nws_alert_duplicates_and_superseded_messages(
    shapefile_polygons,
    active_alerts,
    cell_times,
    other_cell_times,
    tmp_path,
    monkeypatch,
):
    """Protect the ingest/API alert contract against issue 698 and its fix:
    NWS can keep an alert, a RESENT copy and its correction active together
    (each its own CAP message), and one alert can be listed on overlapping zone
    polygons. The API must see each alert once per cell, with the newest
    message's time, without the duplicate handling removing real coverage.
    """
    cap_ids, polygons = zip(*shapefile_polygons, strict=True)
    shapefile_rows = gp.GeoDataFrame(
        {
            "CAP_ID": list(cap_ids),
            "URL": "https://api.weather.gov/alerts/example",
            "EXPIRATION": EXPIRATION,
            "PROD_TYPE": "Special Weather Statement",
            "WFO": "KFFC",
            "SIG": "",
        },
        geometry=list(polygons),
        crs="EPSG:4326",
    )

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

    def cell_alert_times(lat, lon):
        cell_alerts = alerts._read_nws_alerts(
            lat=lat,
            az_lon=lon,
            alerts_zarr=alert_grid,
            now_utc=datetime.datetime(2026, 10, 22, 12, 0, tzinfo=datetime.UTC),
            logger=logging.getLogger(__name__),
            loc_tag="test",
        )
        return [alert["time"] for alert in cell_alerts]

    def timestamps(times):
        return [int(datetime.datetime.fromisoformat(t).timestamp()) for t in times]

    assert cell_alert_times(34.1, -84.5) == timestamps(cell_times)
    assert cell_alert_times(35.0, -84.0) == timestamps(other_cell_times)
