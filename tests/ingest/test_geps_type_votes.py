"""GEPS/REPS precipitation-type member votes (#851)."""

import numpy as np
import pandas as pd
import xarray as xr

from API.constants.shared_const import ENSEMBLE_MEMBER_COUNTS
from ingest.GEPS_Local_Ingest import TYPE_VOTE_VARS, write_ensemble_stats


def test_wet_members_vote_for_their_dominant_type(tmp_path):
    """The API types GEPS and REPS hours by these shares, so a vote that counted
    dry members, members with no type rate, or the total instead of the member's
    dominant type would mistype every hour with no error. Ten light-snow members
    must outvote one heavy freezing-rain member (the means say the reverse), a
    rain/snow tie must go to the first type in TYPE_VOTE_VARS (rain), and the
    vote must use the deaccumulated rate of each lead, not the running total.
    """
    members = ENSEMBLE_MEMBER_COUNTS["GEPS"]
    forecast_hours = [3, 6]
    # Hourly rates (mm/h) per member and lead for one grid cell, by type
    rates = {name: np.zeros((members, len(forecast_hours))) for name in TYPE_VOTE_VARS}
    rates["ASNOW_Prob"][:10, 0] = 0.4  # ten light-snow members
    rates["AFRAIN_Prob"][10, 0] = 6.0  # one heavy freezing-rain member
    rates["ARAIN_Prob"][11, 0] = rates["ASNOW_Prob"][11, 0] = 0.3  # a tie
    rates["ASNOW_Prob"][12, 0] = 0.05  # under the wet threshold: no vote
    apcp_rates = sum(rates.values())
    apcp_rates[13, 0] = 1.0  # wet, but no type rate: no vote
    # Nothing falls in the second interval; its totals equal the first lead's

    def member_totals(rate):
        # GEPS files hold totals since the run start (3-hour intervals)
        return np.cumsum(rate * np.array([3.0, 0.0]), axis=1)

    fields = {"APCP_SFC": apcp_rates}
    fields.update({f"{name.split('_')[0]}_SFC": rates[name] for name in TYPE_VOTE_VARS})
    data_vars = {}
    for base_name, field_rates in fields.items():
        totals = member_totals(field_rates).astype("float32")
        for member in range(members):
            # wgrib2 -set_ext_name: VAR_SFC for the control, VAR_SFC.<n> after
            name = base_name if member == 0 else f"{base_name}.{member}"
            data_vars[name] = (
                ("time", "latitude", "longitude"),
                totals[member][:, np.newaxis, np.newaxis],
            )
    run = pd.Timestamp("2026-11-26 12:00")
    times = run + pd.to_timedelta(forecast_hours, unit="h")
    dataset = xr.Dataset(data_vars, coords={"time": times})

    write_ensemble_stats(
        dataset, forecast_hours, str(tmp_path / "stats.zarr"), 1, 1, "GEPS test run"
    )

    stats = xr.open_zarr(tmp_path / "stats.zarr", consolidated=False)
    shares = {name: stats[name].values[:, 0, 0] * members for name in TYPE_VOTE_VARS}
    np.testing.assert_allclose(shares["ASNOW_Prob"], [10, 0])
    np.testing.assert_allclose(shares["AFRAIN_Prob"], [1, 0])
    np.testing.assert_allclose(shares["ARAIN_Prob"], [1, 0])
    np.testing.assert_allclose(shares["AICEP_Prob"], [0, 0])
    # The means favour freezing rain, which the vote overrules
    assert stats["AFRAIN_Mean"].values[0, 0, 0] > stats["ASNOW_Mean"].values[0, 0, 0]
