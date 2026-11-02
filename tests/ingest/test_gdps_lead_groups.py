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
        if v == ptype:
            continue
        first = gdps.GDPS_REDUCED_LEAD_VARS.get(var, gdps.GDPS_FORECAST_HOURS)[0]
        np.testing.assert_allclose(out[v], 2 * np.maximum(target_hours, first) + v)
    np.testing.assert_array_equal(out[ptype], 3.0)


def test_gdps_bitmap_masked_cape_and_cin_become_zero_not_holes():
    """MSC publishes GDPS CAPE and CIN with a GRIB bitmap (undefined where no
    parcel is buoyant, about 45% of the globe), which arrives as NaN at
    published leads. Left as NaN, one masked 3-hourly lead blended into a
    3-12 hour NaN hole that the API filled linearly at read time on almost
    every request. The ingest must store 0 there and keep NaN for other fields.
    """
    hours = np.array(gdps.GDPS_FORECAST_HOURS)
    stacked_times = hours * 3600.0
    hourly_times = np.arange(hours[0], hours[-1] + 1) * 3600.0
    published = np.isin(hours, gdps.GDPS_REDUCED_FORECAST_HOURS)
    # CAPE and CIN: 100 at their published leads, except a bitmap-masked lead
    # (NaN) at 24 h; temperature as a control field with a genuine gap at 24 h
    masked_lead = int(np.flatnonzero(hours == 24)[0])
    stack = np.full((len(gdps.zarr_vars), len(hours), 1, 1), np.nan, np.float32)
    for var in ("CAPE_surface", "CIN_surface"):
        v = gdps.zarr_vars.index(var)
        stack[v, published, 0, 0] = 100.0
        stack[v, masked_lead, 0, 0] = np.nan
    temp = gdps.zarr_vars.index("TMP_2maboveground")
    stack[temp, :, 0, 0] = 280.0
    stack[temp, masked_lead, 0, 0] = np.nan

    filled = da.stack(
        [
            gdps.zero_bitmap_points(var, da.from_array(stack[v], chunks=(-1, 1, 1)))
            for v, var in enumerate(gdps.zarr_vars)
        ],
        axis=0,
    )
    out = gdps.interp_lead_groups(filled, stacked_times, hourly_times).compute()[
        :, :, 0, 0
    ]

    for var in ("CAPE_surface", "CIN_surface"):
        v = gdps.zarr_vars.index(var)
        assert not np.isnan(out[v]).any()
        # 0 at the masked lead, blending back to 100 at the neighbouring leads
        assert out[v][24 - hours[0]] == 0.0
        assert out[v][21 - hours[0]] == 100.0 and out[v][27 - hours[0]] == 100.0
    # A field without a bitmap keeps its gap for the API's source fallback
    assert np.isnan(out[temp][24 - hours[0]])
