"""CFS ensemble hand-over from four members to member 01."""

import numpy as np

from API.constants.shared_const import ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H
from ingest.CFS_Local_Ingest import MEMBER_MASK_VARS, reduce_cfs_members


def test_four_member_statistics_stop_where_members_02_to_04_end():
    """Members 02-04 of a 00Z cycle stop at ~111 days and member 01 runs to
    the 6-month cap. Before the shortest member ends the store must hold the
    four-member mean and statistics; after it, member 01 alone with NaN
    probability, spread and member mask. Averaging past the end (NaN or a
    shorter member's last lead broadcast) would corrupt months of forecast, and
    a 0/1 "probability" or zero spread from a single member would be published
    as if it came from an ensemble.
    """
    rng = np.random.default_rng(3)
    shape = (2, 3)
    member_01 = rng.random((10, *shape), dtype="float32")
    # Member 02 is one lead longer than 03-04 (Herbie's dropped final message)
    others = [rng.random((n, *shape), dtype="float32") for n in (7, 6, 6)]
    members = [member_01, *others]

    state = reduce_cfs_members(members)["mean"]
    expected_mean = np.mean([m[:6] for m in members], axis=0)
    np.testing.assert_allclose(state[:6], expected_mean, rtol=1e-6)
    np.testing.assert_array_equal(state[6:], member_01[6:])

    wet = ENSEMBLE_PRECIP_PROB_THRESHOLD_MM_H * 2
    rates = [np.zeros((n, *shape), dtype="float32") for n in (10, 7, 6, 6)]
    rates[0][:] = wet  # member 01 always wet
    rates[2][2] = wet  # member 03 wet at lead 2
    reduced = reduce_cfs_members(rates, precipitation=True)
    np.testing.assert_allclose(
        reduced["Precipitation_Prob"][:6, 0, 0], [0.25] * 2 + [0.5] + [0.25] * 3
    )
    np.testing.assert_allclose(
        reduced["APCP_StdDev"][:6], np.std([r[:6] for r in rates], axis=0), rtol=1e-6
    )
    # Bit 0 is member 01, bit 2 member 03
    np.testing.assert_array_equal(
        reduced[MEMBER_MASK_VARS[0]][:6, 0, 0], [1, 1, 5, 1, 1, 1]
    )
    for name in ("Precipitation_Prob", "APCP_StdDev", *MEMBER_MASK_VARS):
        assert np.isnan(reduced[name][6:]).all(), name
    np.testing.assert_allclose(reduced["mean"][6:], wet)
