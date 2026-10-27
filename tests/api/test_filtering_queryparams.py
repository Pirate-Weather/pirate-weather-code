"""MCP response-filtering query parameters (blocks, *_indices) and the
version / include gating of air-quality fields.
"""

import pytest
from fastapi import HTTPException

from API.api_utils import remove_conditional_fields
from API.utils.filtering import apply_block_indices, apply_blocks_param, parse_indices


def test_parse_indices_accepts_ascending_csv():
    """MCP clients send whitespace-separated, possibly repeated indices."""
    assert parse_indices("0, 1,1, 6", "hourly_indices") == [0, 1, 1, 6]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param("0,a,2", "CSV list of non-negative integers", id="non-integer"),
        pytest.param("0,1.5,2", "CSV list of non-negative integers", id="float"),
        pytest.param("0,-1,2", "must contain non-negative integers", id="negative"),
        pytest.param("2,1,0", "must be in ascending order", id="out-of-order"),
    ],
)
def test_parse_indices_rejects_invalid_input(value, message):
    """Bad index lists must be a 400 naming the parameter, not a 500 or a
    silently reordered/negative (Python wrap-around) selection.
    """
    with pytest.raises(HTTPException) as exc:
        parse_indices(value, "hourly_indices")

    assert exc.value.status_code == 400
    assert exc.value.detail.startswith("hourly_indices")
    assert message in exc.value.detail


def test_blocks_param_translates_to_exclude_and_include():
    """blocks is an allowlist layered on exclude/include: every unrequested block
    is excluded, day_night maps to include=day_night_forecast, existing values
    survive, and unknown block names are a 400.
    """
    new_exclude, new_include = apply_blocks_param(
        "currently,day_night", "nbm", "airqualitydetails"
    )

    assert set(new_exclude.split(",")) == {
        "nbm",
        "minutely",
        "hourly",
        "daily",
        "alerts",
        "flags",
    }
    assert set(new_include.split(",")) == {"airqualitydetails", "day_night_forecast"}

    _, include_without_day_night = apply_blocks_param(
        "hourly", None, "day_night_forecast"
    )
    assert include_without_day_night is None

    with pytest.raises(HTTPException) as exc:
        apply_blocks_param("currently,bogus_block", None, None)
    assert exc.value.status_code == 400
    assert "bogus_block" in exc.value.detail


def test_block_indices_filter_and_reject_out_of_range():
    """Index filtering must keep exactly the requested rows and turn an index past
    the end of the block into a 400 instead of an IndexError.
    """
    response = {"daily": {"data": ["a", "b", "c", "d"]}}
    apply_block_indices(response, "daily", [0, 2])
    assert response["daily"]["data"] == ["a", "c"]

    with pytest.raises(HTTPException) as exc:
        apply_block_indices(response, "daily", [0, 10])
    assert exc.value.status_code == 400
    assert exc.value.detail == "daily_indices contains out-of-range values: [10]"


AQ_ITEM = {
    "temperature": 20.0,
    "airQualityIndex": 42,
    "airQualityIndexMin": 20,
    "airQualityIndexMinTime": 1700003600,
    "pm25": 10.0,
    "ozoneConcentration": 30.0,
    "smoke": 0.1,
}
AQI_FIELDS = {"airQualityIndex", "airQualityIndexMin", "airQualityIndexMinTime"}
AQ_DETAIL_FIELDS = {"pm25", "ozoneConcentration"}


@pytest.mark.parametrize(
    ("version", "details", "expected_fields"),
    [
        pytest.param(1, 0, {"temperature"}, id="v1"),
        pytest.param(2, 0, {"temperature", "smoke"} | AQI_FIELDS, id="v2"),
        pytest.param(
            2,
            1,
            {"temperature", "smoke"} | AQI_FIELDS | AQ_DETAIL_FIELDS,
            id="v2-details",
        ),
    ],
)
def test_air_quality_fields_gated_by_version_and_details(
    version, details, expected_fields
):
    """API compatibility: version<2 clients must never see AQI/smoke fields, and
    pollutant concentrations appear only with include=airqualitydetails.
    """
    item = dict(AQ_ITEM)

    remove_conditional_fields(
        item,
        version=version,
        time_machine=False,
        tm_extra=False,
        inc_airqualitydetails=details,
    )

    assert set(item) == expected_fields
