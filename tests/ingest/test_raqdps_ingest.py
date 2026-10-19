"""RAQDPS ingest Datamart URL and history-mapping tests."""

from datetime import datetime, timezone

import pytest

from API.raqdps_herbie_template import raqdps
from API.raqdps_utils import build_raqdps_url, history_run_for_valid_time


def test_build_raqdps_url():
    """Protect the ECCC Datamart path/filename contract: a wrong date, run-hour,
    zero-padded lead or variable segment makes every RAQDPS download 404.
    """
    run_time = datetime(2026, 7, 6, 12, tzinfo=timezone.utc)
    assert build_raqdps_url(run_time, "PM2.5", 72) == (
        "https://dd.weather.gc.ca/20260706/WXO-DD/model_raqdps/10km/grib2/"
        "12/072/20260706T12Z_MSC_RAQDPS_PM2.5_Sfc_RLatLon0.09_PT072H.grib2"
    )


@pytest.mark.parametrize(
    ("valid_hour", "expected_run_hour", "expected_lead"),
    [
        pytest.param(0, 0, 0, id="00z-analysis"),
        pytest.param(11, 0, 11, id="before-12z-uses-00z-run"),
        pytest.param(12, 12, 0, id="12z-analysis"),
        pytest.param(18, 12, 6, id="after-12z-uses-12z-run"),
    ],
)
def test_history_run_for_valid_time_prefers_smallest_0_to_12h_lead(
    valid_hour, expected_run_hour, expected_lead
):
    """History is stitched from the shortest-lead forecast for each valid hour;
    mapping to the wrong 00/12 UTC cycle would pull older, less accurate leads
    (or leads the Datamart does not keep) into the air-quality history.
    """
    run_time, forecast_hour = history_run_for_valid_time(
        datetime(2026, 7, 6, valid_hour, tzinfo=timezone.utc)
    )
    assert run_time == datetime(2026, 7, 6, expected_run_hour, tzinfo=timezone.utc)
    assert forecast_hour == expected_lead


def test_herbie_template_sets_raqdps_source():
    """Herbie downloads RAQDPS through the repo-local template; it must point
    the "msc" source at the Datamart file for the requested run/variable/lead
    and use that filename locally, otherwise Herbie fetches or caches the wrong file.
    """

    class DummyHerbie:
        date = datetime(2026, 7, 6, 0, tzinfo=timezone.utc)
        product = None
        variable = "O3"
        level = "Sfc"
        fxx = 12

        @property
        def get_remoteFileName(self):
            # Mirrors Herbie's real property: the basename of the source URL.
            return self.SOURCES["msc"].split("/")[-1]

    dummy = DummyHerbie()
    raqdps.template(dummy)

    assert dummy.product == "10km/grib2"
    assert dummy.SOURCES == {
        "msc": "https://dd.weather.gc.ca/20260706/WXO-DD/model_raqdps/10km/grib2/"
        "00/012/20260706T00Z_MSC_RAQDPS_O3_Sfc_RLatLon0.09_PT012H.grib2"
    }
    assert dummy.LOCALFILE == (
        "20260706T00Z_MSC_RAQDPS_O3_Sfc_RLatLon0.09_PT012H.grib2"
    )
