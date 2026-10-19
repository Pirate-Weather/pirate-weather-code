from types import SimpleNamespace

import pytest

from API.ingest_utils import validate_grib_stats


def _stats_line(var, level, max_value):
    return (
        f"1:0:d=2026073000:{var}:{level}:81 hour fcst:min=0 max={max_value} avg=123\n"
    )


@pytest.mark.parametrize(
    ("stdout", "excluded", "expected"),
    [
        pytest.param(
            _stats_line("DSWRF", "surface", "3.576e+03"),
            None,
            True,
            id="scientific-notation-in-range",
        ),
        pytest.param(
            _stats_line("TMP", "2 m above ground", "3.232e+08"),
            None,
            False,
            id="scientific-notation-out-of-range",
        ),
        pytest.param(
            _stats_line("DSWRF", "surface", "323200000"),
            ["DSWRF"],
            True,
            id="excluded-variable-skipped",
        ),
        pytest.param(
            _stats_line("TMP", "2 m above ground", "323200000"),
            ["DSWRF"],
            False,
            id="other-variables-still-checked",
        ),
    ],
)
def test_validate_grib_stats(stdout, excluded, expected):
    """validate_grib_stats is the gate that stops corrupt GRIB data from being
    published. wgrib2 -stats prints large values in scientific notation
    (max=3.576e+03); misparsing them would either reject valid files or let
    out-of-range data through. Per-variable exclusions (used for fields with
    legitimately large values such as accumulated DSWRF) must skip only the
    named variable, not disable the check for the whole file.
    """
    grib_check = SimpleNamespace(stdout=stdout)

    assert validate_grib_stats(grib_check, excluded_variables=excluded) is expected
