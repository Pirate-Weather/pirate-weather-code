"""Source priority and fallbacks in the minutely block.

``build_minutely_block`` picks intensity, probability and precip type from
whichever models are present for a location. These tests pin the fallback paths
that decide which model a user's minutely forecast actually comes from.
"""

import numpy as np

from API.constants.forecast_const import DATA_MINUTELY
from API.constants.model_const import (
    DWD_MOSMIX,
    ECMWF,
    GEFS,
    GFS,
    HRRR_SUBH,
    NBM,
    REPS,
)
from API.minutely.builder import build_minutely_block

MINUTES = np.arange(0, 61 * 60, 60, dtype=float)

# Minutely snow intensity is reported as snow depth at a 10:1 ratio.
SNOW_RATIO = 10


def _source_array(model, fill=0.0, **columns):
    """Return a 61-minute model array with a time column and ``columns`` set."""
    data = np.full((len(MINUTES), max(model.values()) + 2), fill)
    data[:, 0] = MINUTES
    for name, value in columns.items():
        data[:, model[name]] = value
    return data


def _build(source_list, lat, lon, **inputs):
    """Run build_minutely_block with only the given model inputs."""
    kwargs = {
        "hrrr_subh_data": None,
        "hrrr_merged": None,
        "nbm_data": None,
        "dwd_mosmix_data": None,
        "gefs_data": None,
        "gfs_data": None,
        "ecmwf_data": None,
        "era5_data": None,
    }
    kwargs.update(inputs)
    return build_minutely_block(
        minute_array_grib=MINUTES,
        source_list=source_list,
        prep_intensity_unit=1.0,
        version=2,
        lat=lat,
        lon=lon,
        **kwargs,
    )


def _column(minutely, name):
    return minutely[:, DATA_MINUTELY[name]]


def test_no_sources_returns_dry_minutely_block():
    """A location with no minutely-capable model must still produce a full,
    dry minutely block rather than crashing the request.
    """
    minute_items = _build([], lat=40.7, lon=-74.0)[2]

    assert len(minute_items) == 61
    assert {item["precipIntensity"] for item in minute_items} == {0.0}
    assert {item["precipType"] for item in minute_items} == {"none"}


def test_dwd_without_rr1c_falls_back_to_ecmwf():
    """DWD MOSMIX stations without RR1c (precip) must not blank minutely
    precipitation; the ECMWF intensity is used instead, matching hourly.
    """
    dwd = _source_array(DWD_MOSMIX, fill=1.0, accum=np.nan, temp=300.0, ptype=0.0)
    ecmwf = _source_array(ECMWF, intensity=0.001)  # m/s -> 3.6 mm/h

    minutely = _build(
        ["dwd_mosmix", "ecmwf_ifs"],
        lat=-22.9,
        lon=-43.2,
        dwd_mosmix_data=dwd,
        ecmwf_data=ecmwf,
    )[0]

    np.testing.assert_allclose(_column(minutely, "intensity"), 3.6)


def test_ai_models_override_hrrr_subh_and_nbm():
    """With prioritize_ai_models, the AI ensemble must supply amount, chance,
    error and type even where HRRR sub-hourly and NBM (both saying rain) exist.
    """
    hrrr_subh = _source_array(HRRR_SUBH, temp=10.0, rain=1.0, refc=30.0)
    nbm = _source_array(NBM, accum=9.0, prob=10.0, rain=1.0)
    aigefs = _source_array(GEFS, accum=0.5, prob=0.8, error=0.3, snow=1.0)

    minutely = _build(
        ["hrrrsubh", "nbm", "gefs"],
        lat=49.2827,
        lon=-123.1207,
        hrrr_subh_data=hrrr_subh,
        nbm_data=nbm,
        gefs_data=aigefs,
        prioritize_ai_models=True,
    )[0]

    np.testing.assert_allclose(_column(minutely, "intensity"), 0.5)
    np.testing.assert_allclose(_column(minutely, "prob"), 0.8)
    np.testing.assert_allclose(_column(minutely, "error"), 0.3)
    np.testing.assert_allclose(_column(minutely, "snow_intensity"), 0.5 * SNOW_RATIO)
    np.testing.assert_allclose(_column(minutely, "rain_intensity"), 0.0)


def test_canada_uses_reps_probability_with_nbm_amounts():
    """In Canada the chance of precipitation comes from REPS while the amount
    still comes from NBM.
    """
    nbm = _source_array(NBM, prob=35.0, accum=1.25, rain=1.0)
    reps = _source_array(REPS, prob=0.8)

    minutely = _build(
        ["reps", "nbm"],
        lat=49.2827,
        lon=-123.1207,
        nbm_data=nbm,
        reps_data=reps,
    )[0]

    np.testing.assert_allclose(_column(minutely, "prob"), 0.8)
    np.testing.assert_allclose(_column(minutely, "intensity"), 1.25)
    np.testing.assert_allclose(_column(minutely, "rain_intensity"), 1.25)


def test_aigefs_precip_type_falls_back_to_temperature():
    """AIGEFS has no precip-type channels, so its type must come from the
    GFS temperature (-5 C -> snow) instead of defaulting to rain.
    """
    aigefs = _source_array(GEFS, fill=np.nan, accum=0.4, prob=0.8)
    gfs = _source_array(GFS, fill=np.nan, temp=-5.0)

    minutely = _build(
        ["gefs", "gfs"],
        lat=45.0,
        lon=-75.0,
        gefs_data=aigefs,
        gfs_data=gfs,
        prioritize_ai_models=True,
    )[0]

    np.testing.assert_allclose(_column(minutely, "snow_intensity"), 0.4 * SNOW_RATIO)
    np.testing.assert_allclose(_column(minutely, "rain_intensity"), 0.0)
