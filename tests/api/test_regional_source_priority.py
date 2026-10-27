"""Regional forecast-source priority (Canada box / North America / rest of world).

Expected orders come from docs/forecast-source-selection/README.md, the
documented contract for which model supplies each element, not from the
order constants in API/current/metrics.py or API/data_inputs.py.
"""

import numpy as np
import pytest

from API.constants.api_const import PRECIP_IDX
from API.constants.model_const import (
    DWD_MOSMIX,
    ECMWF,
    GDPS,
    GEFS,
    GFS,
    HRDPS,
    HRRR,
    NBM,
)
from API.current.metrics import _build_source_strategies
from tests.helpers.model_arrays import hourly_inputs, model_array

OTTAWA = (45.4215, -75.6972)
VANCOUVER = (49.2827, -123.1207)
NEW_YORK = (40.7128, -74.0060)
BERLIN = (52.5200, 13.4050)

# Sources the README lists for compatible "currently" surface elements.
CONVENTIONAL_CURRENT_SOURCES = (
    "rtma_ru",
    "hrrrsubh",
    "hrdps",
    "gdps",
    "nbm",
    "hrrr",
    "ecmwf_ifs",
    "gfs",
    "dwd_mosmix",
    "era5",
)
# include=aimodels excludes the CMC models before their grids are read.
AI_CURRENT_SOURCES = tuple(
    s for s in CONVENTIONAL_CURRENT_SOURCES if s not in {"hrdps", "gdps"}
)


@pytest.mark.parametrize(
    ("location", "ai", "expected"),
    [
        pytest.param(
            OTTAWA,
            False,
            [
                "rtma_ru",
                "hrrrsubh",
                "hrdps",
                "gdps",
                "nbm",
                "hrrr",
                "ecmwf_ifs",
                "gfs",
                "dwd_mosmix",
                "era5",
            ],
            id="canada",
        ),
        pytest.param(
            NEW_YORK,
            False,
            [
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "ecmwf_ifs",
                "gfs",
                "gdps",
                "dwd_mosmix",
                "era5",
                "hrdps",
            ],
            id="north-america",
        ),
        pytest.param(
            BERLIN,
            False,
            [
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "dwd_mosmix",
                "ecmwf_ifs",
                "gfs",
                "gdps",
                "era5",
                "hrdps",
            ],
            id="rest-of-world",
        ),
        # AI precedence overrides the Canada-box order: AIGFS (GFS schema) leads.
        # ECMWF IFS second is not spelled out for "currently" in the README; it
        # matches the documented AI stack AIGEFS -> ECMWF IFS -> NBM.
        pytest.param(
            VANCOUVER,
            True,
            [
                "gfs",
                "ecmwf_ifs",
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "dwd_mosmix",
                "era5",
            ],
            id="ai-north-america",
        ),
        # Outside North America, ECMWF AIFS (ECMWF IFS schema) leads.
        pytest.param(
            BERLIN,
            True,
            [
                "ecmwf_ifs",
                "rtma_ru",
                "hrrrsubh",
                "nbm",
                "hrrr",
                "dwd_mosmix",
                "gfs",
                "era5",
            ],
            id="ai-rest-of-world",
        ),
    ],
)
def test_currently_source_order_by_region(location, ai, expected):
    """A wrong regional order silently swaps which model supplies current
    conditions (e.g. GFS ahead of HRDPS in Canada, or DWD ahead of ECMWF in the US).
    """
    sources = AI_CURRENT_SOURCES if ai else CONVENTIONAL_CURRENT_SOURCES
    source_map = {name: name for name in sources}

    order = _build_source_strategies(source_map, *location, prioritize_ai_models=ai)

    assert order == expected


@pytest.mark.parametrize(
    ("location", "temps", "expected_columns"),
    [
        pytest.param(
            NEW_YORK,
            {
                "nbm": 20.0,
                "hrrr": 18.0,
                "dwd_mosmix": 12.0,
                "ecmwf": 14.0,
                "gfs": 10.0,
                "gdps": 16.0,
            },
            # NBM -> HRRR -> ECMWF IFS -> GFS -> GDPS -> DWD MOSMIX
            [20.0, 18.0, 14.0, 10.0, 16.0, 12.0],
            id="north-america",
        ),
        pytest.param(
            BERLIN,
            {"dwd_mosmix": 12.0, "ecmwf": 14.0, "gfs": 10.0, "gdps": 16.0},
            # DWD MOSMIX -> ECMWF IFS -> GFS -> GDPS
            [12.0, 14.0, 10.0, 16.0],
            id="rest-of-world",
        ),
    ],
)
def test_hourly_temperature_source_order_by_region(location, temps, expected_columns):
    """Hourly output takes the first non-missing column, so the column order
    decides which model's temperature is published in each region.
    """
    consts = {
        "nbm": NBM,
        "hrrr": HRRR,
        "dwd_mosmix": DWD_MOSMIX,
        "ecmwf": ECMWF,
        "gfs": GFS,
        "gdps": GDPS,
    }
    source_names = {"hrrr": ["hrrr_0-18", "hrrr_18-48"], "ecmwf": ["ecmwf_ifs"]}
    source_list = [s for name in temps for s in source_names.get(name, [name])]
    merged = {
        name: model_array(consts[name], 3, temp=value) for name, value in temps.items()
    }

    inputs = hourly_inputs(
        lat=location[0], lon=location[1], source_list=source_list, **merged
    )

    np.testing.assert_array_equal(inputs["temperature_inputs"][0], expected_columns)


def test_hourly_canada_prefers_canadian_uv_and_ozone_over_gfs():
    """Regression for #734: inside Canada, HRDPS UV and GDPS ozone must be used
    ahead of GFS rather than being shadowed by it.
    """
    gfs = model_array(GFS, 3, uv=9.0 / (18.9 * 0.025), ozone=300.0)
    hrdps = model_array(HRDPS, 3, uv=7.0)
    gdps = model_array(GDPS, 3, uv=6.0, ozone=350.0)

    inputs = hourly_inputs(
        lat=OTTAWA[0],
        lon=OTTAWA[1],
        source_list=["gfs", "hrdps", "gdps"],
        gfs=gfs,
        hrdps=hrdps,
        gdps=gdps,
    )

    np.testing.assert_array_equal(inputs["uv_inputs"][:, 0], [7.0] * 3)
    np.testing.assert_array_equal(inputs["ozone_inputs"][:, 0], [350.0] * 3)


def test_hourly_precip_intensity_north_america_prefers_ecmwf_over_dwd():
    """In North America ECMWF IFS precipitation must lead DWD MOSMIX, and ECMWF's
    accumulation (m/h) must arrive as mm/h with its precipitation type.
    """
    dwd = model_array(DWD_MOSMIX, 4, accum=1.0, ptype=1.0, temp=5.0)
    ecmwf = model_array(ECMWF, 4, accum_mean=0.002, ptype=1.0)

    inputs = hourly_inputs(
        lat=NEW_YORK[0],
        lon=NEW_YORK[1],
        source_list=["dwd_mosmix", "ecmwf_ifs"],
        dwd_mosmix=dwd,
        ecmwf=ecmwf,
    )

    assert inputs["prcipIntensity_inputs"][0, 0] == pytest.approx(2.0)
    assert inputs["prcipType_inputs"][0, 0] == PRECIP_IDX["rain"]
    assert inputs["prcipIntensity_inputs"][0, 1] == pytest.approx(1.0)


def test_hourly_ai_mode_precip_probability_prefers_gefs_in_north_america():
    """With include=aimodels, AIGEFS (carried in the GEFS schema) must lead
    precipitation probability ahead of NBM and ECMWF IFS in North America.
    """
    inputs = hourly_inputs(
        lat=NEW_YORK[0],
        lon=NEW_YORK[1],
        source_list=["nbm", "ecmwf_ifs", "gefs"],
        prioritize_ai_models=True,
        nbm=model_array(NBM, 4, prob=20.0),
        ecmwf=model_array(ECMWF, 4, prob=0.4),
        gefs=model_array(GEFS, 4, prob=0.8),
    )

    assert inputs["prcipProbability_inputs"][0, 0] == pytest.approx(0.8)
