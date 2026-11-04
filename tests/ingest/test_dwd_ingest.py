"""DWD MOSMIX ingest gap-filling tests."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from API.ingest_utils import interpolate_temporal_gaps_efficiently
from ingest.DWD_Mosmix_Local_Ingest import fill_station_gaps


def _station_frame(station_ids, n_times, **columns):
    times = pd.date_range("2026-01-01", periods=n_times, freq="1h")
    n_rows = len(station_ids) * n_times
    return pd.DataFrame(
        {
            "station_id": [sid for sid in station_ids for _ in range(n_times)],
            "time": list(times) * len(station_ids),
            "latitude": [50.0] * n_rows,
            "longitude": [10.0] * n_rows,
            "altitude": [100.0] * n_rows,
            **columns,
        }
    )


def test_long_station_gap_is_nulled_and_short_gaps_are_interpolated():
    """A station/variable with a gap longer than max_gap_hours must become all
    NaN so the API falls back to GFS/ECMWF instead of interpolating across a
    long DWD outage; short gaps must still be linearly interpolated.
    """
    nan = np.nan
    df = _station_frame(
        ["A", "B"],
        8,
        TMP_2maboveground=[1.0, 2.0, nan, nan, nan, nan, 7.0, 8.0]
        + [1.0, 2.0, nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        VIS_surface=[10.0, 11.0, nan, nan, 14.0, 15.0, 16.0, 17.0]
        + [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
    )

    result = fill_station_gaps(df, max_gap_hours=3)

    station_a = result[result["station_id"] == "A"]
    station_b = result[result["station_id"] == "B"]
    # Station A: 4-hour temperature gap > 3 -> whole series dropped.
    assert station_a["TMP_2maboveground"].isna().all()
    # Station A: 2-hour visibility gap is interpolated; other variables unaffected.
    np.testing.assert_allclose(
        station_a["VIS_surface"].to_numpy(), np.arange(10.0, 18.0)
    )
    np.testing.assert_allclose(
        station_b["TMP_2maboveground"].to_numpy(), np.arange(1.0, 9.0)
    )


def test_precip_and_ptype_fill_short_gaps_and_leave_long_gaps_missing():
    """A long gap is genuinely missing data. Precipitation was zero-filled across it,
    publishing "no rain" with no forecast behind it, and the type code was carried
    across it (ww ends 70 hours early at many stations). Those hours must stay
    missing so another model fills them, while the rest of the station's series stays
    usable and DWD's routine short gaps in RR1c are still filled.
    """
    nan = np.nan
    df = _station_frame(
        ["A"],
        12,
        APCP_surface=[0.0, nan, nan, 3.0, 2.0, nan, nan, nan, nan, 1.0, 1.0, 1.0],
        PTYPE_surface=[1.0, nan, nan, 2.0, 2.0, 2.0, 2.0, 2.0, nan, nan, nan, nan],
    )

    result = fill_station_gaps(df, max_gap_hours=3)

    np.testing.assert_allclose(
        result["APCP_surface"].to_numpy(),
        [0.0, 1.0, 2.0, 3.0, 2.0, nan, nan, nan, nan, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        result["PTYPE_surface"].to_numpy(),
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, nan, nan, nan, nan],
    )


def test_gridded_temporal_gap_fill_uses_nearest_for_ptype():
    """After gridding, DWD fills time gaps with interpolate_temporal_gaps_efficiently.
    Continuous fields are interpolated linearly and edges extended, but the
    categorical precipitation type must use nearest-neighbour so codes are never
    averaged into a different (invalid) type (#463).
    """
    nan = np.nan
    values = np.array([nan, 0.0, nan, nan, 3.0, nan]).reshape(6, 1, 1)

    def _var():
        return xr.DataArray(
            da.from_array(values, chunks=values.shape),
            dims=("time", "y", "x"),
            coords={"time": np.arange(6)},
        )

    ds = xr.Dataset({"TMP_2maboveground": _var(), "PTYPE_surface": _var()})

    result = interpolate_temporal_gaps_efficiently(
        ds, nearest_vars=["PTYPE_surface"], max_gap_hours=3, time_dim="time"
    ).compute()

    np.testing.assert_allclose(
        result["TMP_2maboveground"].values.ravel(), [0.0, 0.0, 1.0, 2.0, 3.0, 3.0]
    )
    np.testing.assert_allclose(
        result["PTYPE_surface"].values.ravel(), [0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
    )


@pytest.mark.parametrize(
    "times",
    [
        pytest.param(np.arange(9), id="numeric-time"),
        pytest.param(
            np.datetime64("2026-11-04T00") + np.arange(9) * np.timedelta64(1, "h"),
            id="datetime-time",
        ),
    ],
)
def test_gridded_gap_fill_leaves_long_gaps_missing(times):
    """Long gaps are genuinely missing. [1, nan x 5, 7] became [1, 2, 3, 4, 4, 4, 7]
    with a 3 hour limit: a partial ramp, a flat line and a jump. A gap past the limit
    must stay missing, in the series or at its ends, so another model can fill those
    hours, and a gap at the limit is still filled. The real store has datetimes, where
    the limit is a duration.
    """

    def _gridded(values):
        values = np.array(values).reshape(9, 1, 1)
        return xr.Dataset(
            {
                "TMP_2maboveground": xr.DataArray(
                    da.from_array(values, chunks=values.shape),
                    dims=("time", "y", "x"),
                    coords={"time": times},
                )
            }
        )

    def _filled(values):
        return (
            interpolate_temporal_gaps_efficiently(
                _gridded(values), max_gap_hours=3, time_dim="time"
            )
            .compute()["TMP_2maboveground"]
            .values.ravel()
        )

    nan = np.nan
    np.testing.assert_allclose(
        _filled([nan, 1.0, nan, nan, nan, nan, nan, 7.0, nan]),
        [1.0, 1.0, nan, nan, nan, nan, nan, 7.0, 7.0],
    )
    np.testing.assert_allclose(
        _filled([nan, 1.0, nan, nan, nan, 5.0, nan, 7.0, nan]),
        [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0],
    )
    # Long ends stay missing too, with short ones still extended
    np.testing.assert_allclose(
        _filled([nan, nan, nan, nan, 5.0, 5.0, 5.0, nan, nan]),
        [nan, nan, nan, nan, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    np.testing.assert_allclose(
        _filled([nan, 5.0, 5.0, 5.0, nan, nan, nan, nan, nan]),
        [5.0, 5.0, 5.0, 5.0, nan, nan, nan, nan, nan],
    )
