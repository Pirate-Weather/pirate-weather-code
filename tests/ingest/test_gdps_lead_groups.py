import dask.array as da
import numpy as np

from ingest import GDPS_Local_Ingest as gdps


def test_gdps_lead_groups_interpolate_mixed_leads_without_gaps():
    """GDPS stacks fields published hourly, 3-hourly and 6-hourly on one lead
    axis, so fields such as CAPE are NaN at leads MSC never publishes. Blending
    across that axis published NaN; each field must instead be interpolated
    from its own leads, hold its first lead back to the earlier hours, keep
    precipitation type a published category, and stay in zarr_vars order.
    """
    hours = np.array(gdps.GDPS_FORECAST_HOURS)
    stacked_times = hours * 3600.0
    hourly_times = np.arange(hours[0], hours[-1] + 1) * 3600.0
    # Every field is 2 * lead hour + its index, NaN where it is not published
    stack = np.empty((len(gdps.zarr_vars), len(hours), 1, 1), dtype=np.float32)
    for v, var in enumerate(gdps.zarr_vars):
        stack[v, :, 0, 0] = 2 * hours + v
        published = gdps.GDPS_REDUCED_LEAD_VARS.get(var, gdps.GDPS_FORECAST_HOURS)
        stack[v, ~np.isin(hours, published)] = np.nan
    ptype = gdps.zarr_vars.index("PTYPE_surface")
    reduced = np.isin(hours, gdps.GDPS_REDUCED_FORECAST_HOURS)
    stack[ptype, :, 0, 0] = np.where(reduced, 3.0, np.nan)

    out = gdps.interp_lead_groups(
        da.from_array(stack, chunks=(1, -1, 1, 1)), stacked_times, hourly_times
    ).compute()[:, :, 0, 0]

    assert not np.isnan(out).any()
    target_hours = hourly_times / 3600
    for v, var in enumerate(gdps.zarr_vars):
        # Wind direction wraps at 360; test_gdps_wind_direction_blends_across_north
        if v == ptype or var == "WDIR_10maboveground":
            continue
        first = gdps.GDPS_REDUCED_LEAD_VARS.get(var, gdps.GDPS_FORECAST_HOURS)[0]
        np.testing.assert_allclose(out[v], 2 * np.maximum(target_hours, first) + v)
    np.testing.assert_array_equal(out[ptype], 3.0)


def test_gdps_wind_direction_blends_across_north():
    """GDPS and NBM store wind direction as degrees and fill 3-hourly leads with
    a time blend. A linear blend of a shift from 350 to 10 degrees publishes
    south-easterly hours for a northerly wind (seen in the API as 340, 227, 114,
    0), so direction must take the short way round while wind speed, in the same
    lead group, stays linear.
    """
    hours = np.array(gdps.GDPS_FORECAST_HOURS)
    three_hourly = np.flatnonzero(np.diff(hours) == 3)[0]
    stack = np.ones((len(gdps.zarr_vars), len(hours), 1, 1), dtype=np.float32)
    wdir = gdps.zarr_vars.index("WDIR_10maboveground")
    wind = gdps.zarr_vars.index("WIND_10maboveground")
    stack[wdir, :, 0, 0] = np.where(np.arange(len(hours)) <= three_hourly, 350.0, 10.0)
    stack[wind, :, 0, 0] = np.where(np.arange(len(hours)) <= three_hourly, 3.0, 6.0)

    start_hour = hours[three_hourly]
    hourly_times = np.arange(start_hour, start_hour + 4) * 3600.0
    out = gdps.interp_lead_groups(
        da.from_array(stack, chunks=(1, -1, 1, 1)), hours * 3600.0, hourly_times
    ).compute()[:, :, 0, 0]

    np.testing.assert_allclose(out[wdir], [350, 350 + 20 / 3, 10 / 3, 10], atol=1e-3)
    np.testing.assert_allclose(out[wind], [3, 4, 5, 6])
