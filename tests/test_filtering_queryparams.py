"""Tests for MCP filtering query parameters (blocks, *_indices)."""

import pytest
from fastapi import HTTPException

from API.utils.filtering import (
    ALL_BLOCKS,
    apply_block_indices,
    apply_blocks_param,
    parse_indices,
)


class TestParseIndices:
    """Tests for parse_indices helper."""

    def test_valid_ascending_integers(self):
        assert parse_indices("0,1,2,6", "hourly_indices") == [0, 1, 2, 6]

    def test_single_index(self):
        assert parse_indices("3", "daily_indices") == [3]

    def test_zero_index(self):
        assert parse_indices("0", "hourly_indices") == [0]

    def test_whitespace_tolerant(self):
        assert parse_indices("0, 1, 2", "hourly_indices") == [0, 1, 2]

    def test_non_integer_raises(self):
        with pytest.raises(HTTPException) as exc:
            parse_indices("0,a,2", "hourly_indices")
        assert exc.value.status_code == 400
        assert "non-negative integers" in exc.value.detail

    def test_float_raises(self):
        with pytest.raises(HTTPException):
            parse_indices("0,1.5,2", "hourly_indices")

    def test_negative_raises(self):
        with pytest.raises(HTTPException) as exc:
            parse_indices("0,-1,2", "hourly_indices")
        assert exc.value.status_code == 400
        assert "non-negative" in exc.value.detail

    def test_out_of_order_raises(self):
        with pytest.raises(HTTPException) as exc:
            parse_indices("2,1,0", "daily_indices")
        assert exc.value.status_code == 400
        assert "ascending order" in exc.value.detail

    def test_duplicate_values_allowed(self):
        # Duplicates are "in order" per spec (same element selected twice)
        assert parse_indices("0,1,1,2", "daily_indices") == [0, 1, 1, 2]

    def test_empty_string_returns_empty(self):
        assert parse_indices("", "hourly_indices") == []


class TestAllBlocks:
    """Tests for ALL_BLOCKS constant."""

    def test_contains_all_expected_blocks(self):
        expected = {
            "currently",
            "minutely",
            "hourly",
            "daily",
            "day_night",
            "alerts",
            "flags",
        }
        assert ALL_BLOCKS == expected

    def test_is_frozenset(self):
        assert isinstance(ALL_BLOCKS, frozenset)


class TestApplyBlocksParam:
    """Tests for apply_blocks_param (the blocks-to-excludes conversion)."""

    def test_single_block_excludes_others(self):
        new_exclude, _ = apply_blocks_param("currently", None, None)
        excluded = set(new_exclude.split(","))
        assert "minutely" in excluded
        assert "hourly" in excluded
        assert "daily" in excluded
        assert "alerts" in excluded
        assert "flags" in excluded
        assert "currently" not in excluded

    def test_multiple_blocks_keeps_requested(self):
        new_exclude, _ = apply_blocks_param("currently,hourly", None, None)
        excluded = set(new_exclude.split(","))
        assert "currently" not in excluded
        assert "hourly" not in excluded
        assert "minutely" in excluded
        assert "daily" in excluded

    def test_all_blocks_excludes_nothing_standard(self):
        new_exclude, _ = apply_blocks_param(
            "currently,minutely,hourly,daily,day_night,alerts,flags", None, None
        )
        excluded = set(new_exclude.split(",")) if new_exclude else set()
        for block in ("currently", "minutely", "hourly", "daily", "alerts", "flags"):
            assert block not in excluded

    def test_day_night_added_to_include(self):
        _, new_include = apply_blocks_param("currently,day_night", None, None)
        assert new_include is not None
        assert "day_night_forecast" in new_include.split(",")

    def test_day_night_not_in_exclude(self):
        new_exclude, _ = apply_blocks_param("currently,day_night", None, None)
        excluded = set(new_exclude.split(",")) if new_exclude else set()
        assert "day_night" not in excluded

    def test_invalid_block_raises(self):
        with pytest.raises(HTTPException) as exc:
            apply_blocks_param("currently,bogus_block", None, None)
        assert exc.value.status_code == 400
        assert "bogus_block" in exc.value.detail

    def test_existing_exclude_is_preserved(self):
        new_exclude, _ = apply_blocks_param("hourly", "nbm", None)
        excluded = set(new_exclude.split(","))
        assert "nbm" in excluded
        assert "currently" in excluded

    def test_no_day_night_in_include_when_not_requested(self):
        _, new_include = apply_blocks_param("currently,hourly", None, None)
        if new_include:
            assert "day_night_forecast" not in new_include

    def test_existing_include_is_preserved_with_day_night(self):
        _, new_include = apply_blocks_param("day_night", None, "some_other_flag")
        assert new_include is not None
        parts = new_include.split(",")
        assert "day_night_forecast" in parts
        assert "some_other_flag" in parts


class TestApplyBlockIndices:
    """Tests for apply_block_indices helper."""

    def test_filters_data_by_indices(self):
        obj = {"daily": {"data": ["a", "b", "c", "d", "e"]}}
        apply_block_indices(obj, "daily", [0, 2, 4])
        assert obj["daily"]["data"] == ["a", "c", "e"]

    def test_single_index(self):
        obj = {"hourly": {"data": list(range(48))}}
        apply_block_indices(obj, "hourly", [5])
        assert obj["hourly"]["data"] == [5]

    def test_out_of_range_raises(self):
        obj = {"daily": {"data": ["a", "b", "c"]}}
        with pytest.raises(HTTPException) as exc:
            apply_block_indices(obj, "daily", [0, 10])
        assert exc.value.status_code == 400
        assert "out-of-range" in exc.value.detail
        assert "daily_indices" in exc.value.detail

    def test_missing_block_is_noop(self):
        obj = {"hourly": {"data": ["x"]}}
        # "daily" not in obj – should not raise
        apply_block_indices(obj, "daily", [0])
        assert "daily" not in obj

    def test_empty_data_with_empty_indices(self):
        obj = {"daily": {"data": []}}
        apply_block_indices(obj, "daily", [])
        assert obj["daily"]["data"] == []
