"""Regional forecast-source priority (Canada / North America / rest of world).

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
    RRFS,
)
from API.current.metrics import _build_source_strategies
from API.utils.geo import is_in_canada, is_in_north_america
from tests.helpers.model_arrays import hourly_inputs, model_array

OTTAWA = (45.4215, -75.6972)
VANCOUVER = (49.2827, -123.1207)
SEATTLE = (47.6062, -122.3321)
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


@pytest.mark.parametrize("ai", [False, True], ids=["conventional", "ai-mode"])
def test_include_rrfs_leads_every_element_even_in_canada(ai):
    """include=rrfs promises RRFS ahead of every model. Inside the Canada
    and in AI mode other models normally lead, so a missing prepend in one of
    the order builders would quietly publish HRDPS or AI values to a user who
    asked for RRFS. Without RRFS data the orders are the existing ones (the
    tests above).
    """
    source_map = {name: name for name in (*AI_CURRENT_SOURCES, "rrfssubh", "rrfs")}
    order = _build_source_strategies(source_map, *OTTAWA, prioritize_ai_models=ai)
    assert order[:2] == ["rrfssubh", "rrfs"]

    merged = {
        "rrfs": model_array(RRFS, 3, temp=5.0, intensity=1 / 3600, accum=1.0),
        "hrdps": model_array(HRDPS, 3, temp=6.0, intensity=2 / 3600, accum=2.0),
        "nbm": model_array(NBM, 3, temp=7.0, intensity=3.0, accum=3.0),
        "gfs": model_array(GFS, 3, temp=8.0, intensity=4 / 3600, accum=4.0),
    }
    inputs = hourly_inputs(
        lat=OTTAWA[0],
        lon=OTTAWA[1],
        source_list=["rrfs_0-18", "hrdps", "nbm", "gfs"],
        prioritize_ai_models=ai,
        **merged,
    )

    # RRFS's values (precipitation intensity in mm/h) sit in the first column
    expected_first = {
        "temperature_inputs": 5.0,
        "prcipIntensity_inputs": 1.0,
        "accum_inputs": 1.0,
    }
    for element, value in expected_first.items():
        assert inputs[element][0, 0] == pytest.approx(value), element


@pytest.mark.parametrize(
    ("place", "lat", "lon", "in_canada"),
    [
        # Cities that sit inside the old lat/lon box
        ("Seattle", 47.61, -122.33, False),
        ("Chicago", 41.88, -87.63, False),
        ("Detroit", 42.33, -83.05, False),
        ("Boston", 42.36, -71.06, False),
        ("Minneapolis", 44.98, -93.27, False),
        ("Juneau", 58.30, -134.42, False),
        # Either side of the border, and enclaves only the border can separate
        ("Windsor", 42.31, -83.04, True),
        ("Sault Ste Marie, Ontario", 46.53, -84.35, True),
        ("Sault Ste Marie, Michigan", 46.49, -84.35, False),
        ("Point Roberts", 48.98, -123.05, False),
        ("Northwest Angle", 49.35, -95.05, False),
        # Canadian zones outside the main ones
        ("Vancouver", 49.28, -123.12, True),
        ("Creston", 49.09, -116.51, True),
        ("Atikokan", 48.75, -91.62, True),
        ("Blanc-Sablon", 51.43, -57.13, True),
        ("Alert", 82.50, -62.35, True),
        # Water
        ("Hudson Bay", 60.0, -86.0, True),
        ("Lake Michigan", 43.5, -87.0, False),
        ("Beaufort Sea", 72.0, -135.0, True),
        ("Off Cape Cod", 42.0, -68.5, False),
        # Also in the old box, not Canada
        ("Nuuk", 64.18, -51.72, False),
        ("Saint-Pierre", 46.78, -56.18, False),
        ("New York", 40.71, -74.01, False),
    ],
)
def test_canada_follows_the_border(place, lat, lon, in_canada):
    """Canadian-first priority came from a lat/lon box holding much of the northern
    US, so Seattle, Chicago, Detroit, Boston and Minneapolis took HRDPS ahead of
    NBM and HRRR. Every US test used New York, just south of the box, which hid it.
    """
    assert is_in_canada(lat, lon) is in_canada, place


def test_northern_us_city_uses_the_us_order():
    """End to end from the region check to the stack: Seattle's hourly temperature
    must lead with NBM, and Vancouver's, 190 km away, with HRDPS.
    """
    merged = {
        "nbm": model_array(NBM, 3, temp=1.0),
        "hrdps": model_array(HRDPS, 3, temp=2.0),
    }

    def first_temperature(location):
        inputs = hourly_inputs(
            lat=location[0], lon=location[1], source_list=["nbm", "hrdps"], **merged
        )
        return inputs["temperature_inputs"][0][0]

    assert first_temperature(SEATTLE) == 1.0
    assert first_temperature(VANCOUVER) == 2.0


@pytest.mark.parametrize(
    ("place", "lat", "lon", "expected"),
    [
        ("Dutch Harbor", 53.89, -166.54, True),
        ("Adak", 51.88, -176.66, True),
        ("Attu, across the antimeridian", 52.90, 172.90, True),
        ("St Lawrence Island", 63.40, -170.30, True),
        ("Chukotka", 66.0, -171.0, False),
        ("Midway", 28.20, -177.40, False),
        ("Wake Island", 19.28, 166.65, False),
        ("Hawaii", 21.31, -157.86, True),
    ],
)
def test_north_america_includes_the_aleutians(place, lat, lon, expected):
    """West of -168 the old check never ran its Alaska branch, as the box above it
    had already excluded those longitudes, so the western Aleutians got rest-of-world
    source priority, with DWD MOSMIX ahead of GFS.
    """
    assert is_in_north_america(lat, lon) is expected, place
