"""WMO alert ingest: catalog feed discovery, CAP parsing and grid output."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from ingest import WMO_Alerts_Local as wmo

CAP_XML = """
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>alert-1</identifier>
  <info>
    <language>en</language>
    <event>Wind warning</event>
    <description>Strong winds expected.</description>
    <severity>Severe</severity>
    <urgency>Expected</urgency>
    <effective>2026-10-14T10:00:00+00:00</effective>
    <expires>2026-10-15T10:00:00+00:00</expires>
    <area>
      <areaDesc>Test region</areaDesc>
      <polygon>0,0 0,1 1,1 1,0 0,0</polygon>
    </area>
  </info>
</alert>
"""


def _mock_downloads(monkeypatch, feed_xml=None, current=True, extra_feeds=None):
    """Serve a one-agency WMO catalog whose feed(s) point at CAP_XML, offline."""
    calls = []
    feed_url = "https://agency.example/feed"
    cap_url = "https://agency.example/alert.xml"
    feeds = {feed_url: feed_xml, **(extra_feeds or {})}

    async def fetch_json(session, url, timeout):
        if url.endswith("sources.json"):
            return {
                "sources": [
                    {
                        "source": {
                            "sourceId": "test-agency",
                            "capAlertFeedStatus": "operating",
                            "capAlertFeed": ", ".join(feeds),
                        }
                    }
                ]
            }
        assert url.endswith("wmo_all.json")
        return {
            "items": [{"id": "alert-1", "capURL": "test-agency/alert.xml"}]
            if current
            else []
        }

    async def fetch_text(session, url, timeout):
        calls.append(url)
        assert url in {*feeds, cap_url}
        return feeds.get(url, CAP_XML)

    monkeypatch.setattr(wmo, "_fetch_json", fetch_json)
    monkeypatch.setattr(wmo, "_fetch_text", fetch_text)
    monkeypatch.setattr(wmo, "load_meteoalarm_geocodes", lambda: None)
    return calls


RSS_FEED = """<rss><channel><item>
  <link>https://agency.example/alert.xml</link><guid>alert-1</guid>
</item></channel></rss>"""

FEED_URL = "https://agency.example/feed"
SECOND_FEED_URL = "https://agency.example/second-feed"
CAP_URL = "https://agency.example/alert.xml"


@pytest.mark.parametrize(
    ("feed_xml", "extra_feeds", "expected_calls"),
    [
        pytest.param(RSS_FEED, None, [FEED_URL, CAP_URL], id="rss"),
        pytest.param(
            """<feed xmlns="http://www.w3.org/2005/Atom"
                    xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2">
              <entry>
                <id>https://agency.example/alert.xml?index_area=0</id>
                <cap:identifier>alert-1</cap:identifier>
                <link href="https://agency.example/map" type="text/html"/>
                <link href="https://agency.example/alert.xml" type="application/cap+xml"/>
              </entry>
              <entry>
                <id>https://agency.example/alert.xml?index_area=1</id>
                <cap:identifier>alert-1</cap:identifier>
                <link href="https://agency.example/alert.xml" type="application/cap+xml"/>
              </entry>
            </feed>""",
            None,
            [FEED_URL, CAP_URL],
            id="meteoalarm-atom",
        ),
        pytest.param(
            """<feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>tag:agency.example,2026-10-14:feed.atom/alert-1</id>
                <link href="https://agency.example/alert.xml" rel="alternate"/>
              </entry>
              <entry>
                <id>tag:agency.example,2026-10-14:feed.atom/alert-1</id>
                <link href="https://agency.example/alert.xml" rel="alternate"/>
              </entry>
              <entry>
                <id>tag:agency.example,2026-10-14:feed.atom/old-alert</id>
                <link href="https://agency.example/expired.xml" rel="alternate"/>
              </entry>
            </feed>""",
            None,
            [FEED_URL, CAP_URL],
            id="canadian-atom",
        ),
        pytest.param(
            RSS_FEED,
            {SECOND_FEED_URL: RSS_FEED},
            [FEED_URL, SECOND_FEED_URL, CAP_URL],
            id="two-feeds-same-cap",
        ),
        pytest.param(
            None,
            {SECOND_FEED_URL: RSS_FEED},
            [FEED_URL, SECOND_FEED_URL, CAP_URL],
            id="first-feed-down",
        ),
    ],
)
def test_ingest_uses_catalog_feed_and_fetches_current_cap_once(
    monkeypatch, feed_xml, extra_feeds, expected_calls
):
    """Agencies publish RSS, MeteoAlarm Atom (one entry per area) and Canadian
    Atom feeds, sometimes several per agency. Every format must resolve to the
    current CAP, fetch it exactly once (no duplicate alerts, no wasted
    requests), skip non-current CAPs, and survive one feed being unavailable.
    """
    calls = _mock_downloads(monkeypatch, feed_xml, extra_feeds=extra_feeds)

    result = asyncio.run(wmo.gather_cap_polygons_async())

    assert calls == expected_calls
    assert len(result) == 1
    row = result.iloc[0]
    assert row["source_id"] == "test-agency"
    assert row["event"] == "Wind warning"
    assert row["URL"] == CAP_URL
    assert result.geometry.iloc[0].area == 1
    assert result.crs.to_epsg() == 4326


def test_empty_ingest_keeps_alert_columns(monkeypatch):
    """With no current alerts the GeoDataFrame must still carry the full alert
    schema, otherwise main()'s spatial join and string build raise KeyError
    and no (blank) alert grid is published.
    """
    calls = _mock_downloads(monkeypatch, current=False)

    result = asyncio.run(wmo.gather_cap_polygons_async())

    assert calls == []
    assert result.empty
    assert set(result.columns) == {*wmo.ALERT_COLUMNS, "geometry"}


@pytest.mark.parametrize(
    ("current", "inside_cell"),
    [
        pytest.param(False, "", id="no-alerts-blank-grid"),
        pytest.param(
            True,
            "Wind warning}{Strong winds expected.}{Test region}"
            "{2026-10-14T10:00:00+00:00}{2026-10-15T10:00:00+00:00}{Severe}"
            "{https://agency.example/alert.xml}{}{",
            id="alert-written-inside-polygon",
        ),
    ],
)
def test_main_writes_alert_grid(tmp_path, monkeypatch, current, inside_cell):
    """Protect the WMO_Alerts.zarr contract the API reads: each grid cell holds
    the "}{"-delimited fields of every alert whose polygon contains it, and
    cells outside all alerts (or every cell when nothing is active) are "".
    """
    _mock_downloads(monkeypatch, RSS_FEED, current=current)
    process_dir = tmp_path / "process"
    forecast_dir = tmp_path / "forecast"
    monkeypatch.setattr(wmo, "forecast_process_dir", str(process_dir))
    monkeypatch.setattr(wmo, "forecast_path", str(forecast_dir))
    monkeypatch.setattr(wmo, "historic_path", str(tmp_path / "history"))
    monkeypatch.setattr(wmo, "save_type", "Download")
    # main() hard-codes a global 0.0625 degree grid (~13M points). Swap the
    # module's np for a 2x2 grid at lat/lon {0.5, 2.0} so only (0.5, 0.5)
    # falls inside the unit-square alert polygon.
    monkeypatch.setattr(
        wmo,
        "np",
        SimpleNamespace(
            arange=lambda start, stop, step: np.array([0.5, 2.0]),
            meshgrid=np.meshgrid,
        ),
    )

    wmo.main()

    result = zarr.open_array(
        str(forecast_dir / wmo.ingest_version / "WMO_Alerts.zarr"), mode="r"
    )
    # Rows are latitude, columns longitude.
    expected = np.array([[inside_cell, ""], ["", ""]], dtype=object)
    np.testing.assert_array_equal(result[:], expected)
    assert not process_dir.exists()


def test_cap_keeps_first_language_and_all_its_info_blocks():
    """Multilingual CAPs repeat each <info> per language; only the first
    language may be kept (else every alert appears 2-3 times), but all of its
    <info> blocks must be kept even when other languages are interleaved.
    """
    first_info = CAP_XML[CAP_XML.index("  <info>") : CAP_XML.index("</alert>")]
    other_language = first_info.replace(
        "<language>en</language>", "<language>fr</language>"
    )
    cap_xml = CAP_XML.replace(
        "</alert>", other_language + other_language + first_info + "</alert>"
    )

    result = wmo._extract_polygons_from_cap(cap_xml, "test-agency", "test.xml")

    assert [row[1] for row in result] == ["Wind warning", "Wind warning"]
