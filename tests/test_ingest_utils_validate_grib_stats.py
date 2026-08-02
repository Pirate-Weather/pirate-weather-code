from types import SimpleNamespace

from API.ingest_utils import validate_grib_stats


def test_validate_grib_stats_accepts_signed_scientific_notation():
    grib_check = SimpleNamespace(
        stdout=(
            "1:0:d=2026073000:DSWRF:surface:81 hour fcst:min=0 max=3.576e+03 avg=123\n"
        )
    )

    assert validate_grib_stats(grib_check) is True


def test_validate_grib_stats_skips_excluded_variables():
    grib_check = SimpleNamespace(
        stdout=(
            "1:0:d=2026073000:DSWRF:surface:81 hour fcst:min=0 max=323200000 avg=123\n"
        )
    )

    assert validate_grib_stats(grib_check, excluded_variables=["DSWRF"]) is True


def test_validate_grib_stats_keeps_checking_non_excluded_variables():
    grib_check = SimpleNamespace(
        stdout=(
            "1:0:d=2026073000:TMP:2 m above ground:81 hour fcst:"
            "min=0 max=323200000 avg=123\n"
        )
    )

    assert validate_grib_stats(grib_check, excluded_variables=["DSWRF"]) is False
