"""DWD MOSMIX ingest gap-filling tests."""

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from API.DWD_Mosmix_Local_Ingest import fill_station_gaps
from API.ingest_utils import interpolate_temporal_gaps_efficiently


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


def test_precip_and_ptype_are_filled_even_across_long_gaps():
    """DWD omits RR1c (APCP) for some hours: precipitation must be interpolated
    then zero-filled, and the categorical PTYPE code forward/back-filled, rather
    than being nulled like continuous variables (which would drop DWD precip
    for the station entirely).
    """
    nan = np.nan
    df = _station_frame(
        ["A"],
        6,
        APCP_surface=[0.0, nan, nan, nan, nan, 0.0],
        PTYPE_surface=[1.0, nan, nan, nan, nan, 2.0],
    )

    result = fill_station_gaps(df, max_gap_hours=3)

    np.testing.assert_allclose(result["APCP_surface"].to_numpy(), np.zeros(6))
    np.testing.assert_allclose(
        result["PTYPE_surface"].to_numpy(), [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
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
