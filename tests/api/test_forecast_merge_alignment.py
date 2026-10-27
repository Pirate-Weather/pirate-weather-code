"""Forecast-only model stores must be merged by valid time, not by row index."""

import datetime
import logging

import numpy as np

from API.constants.model_const import GFS, HRRR, NBM
from API.data_inputs import prepare_data_inputs
from API.forecast_sources import SourceMetadata, merge_hourly_models

NUM_HOURS = 24
# Local midnight of the request (the first row of every merged array).
BASE = datetime.datetime(2026, 10, 19, 4, tzinfo=datetime.UTC).timestamp()
# A 12Z HRRR/NBM run: first lead (13Z) is 9 hours after local midnight.
FIRST_LEAD_OFFSET_HOURS = 9


def _store(columns: int, first_time: float, rows: int, temp_col: int, temp: float):
    data = np.full((rows, columns), np.nan)
    data[:, 0] = first_time + np.arange(rows) * 3600
    data[:, temp_col] = temp
    return data


def test_forecast_only_stores_land_on_their_valid_hours():
    """Most model stores no longer carry short-term history, so they start after
    local midnight. Their rows must land on their own valid hours (not be shifted
    to midnight), HRRR/NBM must not be dropped for starting late, and the hours
    before the first lead must stay missing so GFS (which keeps history) fills them.
    """
    first_lead = BASE + FIRST_LEAD_OFFSET_HOURS * 3600
    hrrr_columns = max(HRRR.values()) + 1
    data_hrrrh = _store(hrrr_columns, first_lead, 18, HRRR["temp"], 20.0)
    data_h2 = _store(hrrr_columns, first_lead + 17 * 3600, 31, HRRR["temp"], 21.0)
    data_nbm = _store(max(NBM.values()) + 1, first_lead, 36, NBM["temp"], 22.0)
    # GFS keeps history, so its store starts well before midnight.
    data_gfs = _store(max(GFS.values()) + 1, BASE - 12 * 3600, 60, GFS["temp"], 10.0)

    result = merge_hourly_models(
        metadata=SourceMetadata(["hrrr_0-18", "hrrr_18-48", "nbm", "gfs"], {}, {}),
        num_hours=NUM_HOURS,
        base_day_utc_grib=BASE,
        data_hrrrh=data_hrrrh,
        data_h2=data_h2,
        data_nbm=data_nbm,
        data_gfs=data_gfs,
        data_ecmwf=None,
        data_gefs=None,
        data_hrdps=None,
        data_gdps=None,
        data_geps=None,
        data_reps=None,
        data_dwd_mosmix=None,
        data_aigfs=None,
        data_aigefs=None,
        data_aifs=None,
        logger=logging.getLogger(__name__),
        loc_tag="test",
    )

    assert result.metadata.has_sources("hrrr_0-18", "hrrr_18-48", "nbm", "gfs")
    request_times = BASE + np.arange(NUM_HOURS) * 3600
    for merged in (result.hrrr, result.nbm):
        assert np.all(np.isnan(merged[:FIRST_LEAD_OFFSET_HOURS, 0]))
        np.testing.assert_array_equal(
            merged[FIRST_LEAD_OFFSET_HOURS:, 0],
            request_times[FIRST_LEAD_OFFSET_HOURS:],
        )
    np.testing.assert_array_equal(result.gfs[:, 0], request_times)

    inputs = prepare_data_inputs(
        source_list=result.metadata.source_list,
        nbm_merged=result.nbm,
        hrrr_merged=result.hrrr,
        dwd_mosmix_merged=None,
        ecmwf_merged=None,
        gefs_merged=None,
        gfs_merged=result.gfs,
        era5_merged=None,
        extra_vars=[],
        num_hours=NUM_HOURS,
        lat=40.7,
        lon=-74.0,
    )
    # Columns are in priority order (NBM, HRRR, GFS): the hourly block takes the
    # first finite value, so GFS supplies the hours before the NBM/HRRR run.
    temperature = inputs["temperature_inputs"]
    assert np.all(np.isnan(temperature[:FIRST_LEAD_OFFSET_HOURS, :2]))
    np.testing.assert_array_equal(temperature[FIRST_LEAD_OFFSET_HOURS:, 0], 22.0)
    np.testing.assert_array_equal(temperature[FIRST_LEAD_OFFSET_HOURS:, 1], 20.0)
    np.testing.assert_array_equal(temperature[:, 2], 10.0)
