import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from API import WMO_Alerts_Local as wmo

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


@pytest.mark.parametrize(
    "feed_xml",
    [
        """<rss><channel><item>
          <link>https://agency.example/alert.xml</link><guid>alert-1</guid>
        </item></channel></rss>""",
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
    ],
    ids=["rss", "meteoalarm-atom", "canadian-atom"],
)
def test_ingest_uses_catalog_feed_and_fetches_current_cap_once(monkeypatch, feed_xml):
    calls = _mock_downloads(monkeypatch, feed_xml)

    result = asyncio.run(wmo.gather_cap_polygons_async())

    assert calls == ["https://agency.example/feed", "https://agency.example/alert.xml"]
    assert len(result) == 1
    assert result.iloc[0]["source_id"] == "test-agency"
    assert result.iloc[0]["event"] == "Wind warning"
    assert result.iloc[0]["URL"] == "https://agency.example/alert.xml"
    assert result.geometry.iloc[0].area == 1
    assert result.crs.to_epsg() == 4326


def test_empty_ingest_keeps_alert_columns(monkeypatch):
    calls = _mock_downloads(monkeypatch, current=False)

    result = asyncio.run(wmo.gather_cap_polygons_async())

    assert calls == []
    assert result.empty
    assert result["source_id"].nunique() == 0
    assert set(result.columns) == {*wmo.ALERT_COLUMNS, "geometry"}


@pytest.mark.parametrize("first_feed_available", [True, False])
def test_catalog_multiple_feeds_are_processed_without_duplicate_caps(
    monkeypatch, first_feed_available
):
    feed = """<rss><channel><item>
      <link>https://agency.example/alert.xml</link><guid>alert-1</guid>
    </item></channel></rss>"""
    calls = _mock_downloads(
        monkeypatch,
        feed if first_feed_available else None,
        extra_feeds={"https://agency.example/second-feed": feed},
    )

    result = asyncio.run(wmo.gather_cap_polygons_async())

    assert calls == [
        "https://agency.example/feed",
        "https://agency.example/second-feed",
        "https://agency.example/alert.xml",
    ]
    assert len(result) == 1


def test_main_writes_blank_grid_when_no_alerts_are_current(tmp_path, monkeypatch):
    _mock_downloads(monkeypatch, current=False)
    process_dir = tmp_path / "process"
    forecast_dir = tmp_path / "forecast"
    monkeypatch.setattr(wmo, "forecast_process_dir", str(process_dir))
    monkeypatch.setattr(wmo, "forecast_path", str(forecast_dir))
    monkeypatch.setattr(wmo, "historic_path", str(tmp_path / "history"))
    monkeypatch.setattr(wmo, "save_type", "Download")
    monkeypatch.setattr(
        wmo,
        "np",
        SimpleNamespace(
            arange=lambda start, stop, step: np.array([start, start + step]),
            meshgrid=np.meshgrid,
        ),
    )

    wmo.main()

    result = zarr.open_array(
        str(forecast_dir / wmo.ingest_version / "WMO_Alerts.zarr"), mode="r"
    )
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[:], np.full((2, 2), ""))
    assert not process_dir.exists()


def test_cap_keeps_first_language_and_all_its_info_blocks():
    first_info = CAP_XML[CAP_XML.index("  <info>") : CAP_XML.index("</alert>")]
    other_language = first_info.replace(
        "<language>en</language>", "<language>fr</language>"
    )
    cap_xml = CAP_XML.replace(
        "</alert>", other_language + other_language + first_info + "</alert>"
    )

    result = wmo._extract_polygons_from_cap(cap_xml, "test-agency", "test.xml")

    assert len(result) == 2
    assert all(row[1] == "Wind warning" for row in result)
