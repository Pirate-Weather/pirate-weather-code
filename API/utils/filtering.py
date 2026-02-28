"""Utilities for MCP filtering query parameters (blocks and *_indices)."""

from __future__ import annotations

from fastapi import HTTPException

#: All valid forecast block names accepted by the ``blocks`` query parameter.
ALL_BLOCKS: frozenset[str] = frozenset(
    {"currently", "minutely", "hourly", "daily", "day_night", "alerts", "flags"}
)


def parse_indices(indices_str: str, name: str) -> list[int]:
    """Parse and validate a CSV of non-negative integers for index filtering.

    Args:
        indices_str: Comma-separated string of non-negative integers,
            e.g. ``"0,1,2,6"``.
        name: Parameter name used in error messages.

    Returns:
        List of non-negative integers in the original (ascending) order.

    Raises:
        HTTPException(400): If the string is not valid CSV integers,
            contains negative values, or is not in ascending order.
    """
    try:
        parts = [s for s in (tok.strip() for tok in indices_str.split(",")) if s]
        indices = [int(p) for p in parts]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be a CSV list of non-negative integers",
        )
    if any(i < 0 for i in indices):
        raise HTTPException(
            status_code=400,
            detail=f"{name} must contain non-negative integers",
        )
    if indices != sorted(indices):
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be in ascending order",
        )
    return indices


def apply_blocks_param(
    blocks: str,
    exclude: str | None,
    include: str | None,
) -> tuple[str | None, str | None]:
    """Convert a ``blocks`` allowlist into exclude/include strings.

    The ``blocks`` parameter is a comma-separated list of block names to
    *include* in the response.  This function converts it into the existing
    ``exclude``/``include`` mechanism so the rest of the request pipeline
    does not need to be changed.

    ``day_night`` is opt-in (via ``include=day_night_forecast``) rather than
    opt-out, so it is handled separately from the regular exclude logic.

    Args:
        blocks: Comma-separated allowlist of block names (e.g. ``"currently,hourly"``).
        exclude: Existing ``exclude`` query-param value (may be ``None``).
        include: Existing ``include`` query-param value (may be ``None``).

    Returns:
        A ``(new_exclude, new_include)`` tuple with the merged values.

    Raises:
        HTTPException(400): If ``blocks`` contains an unrecognised block name.
    """
    requested = {b.strip() for b in blocks.split(",") if b.strip()}
    invalid = requested - ALL_BLOCKS
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid blocks: {', '.join(sorted(invalid))}",
        )

    # Exclude every standard block that was NOT requested.
    # day_night is opt-in so it is kept out of the exclude calculation.
    blocks_to_exclude = (ALL_BLOCKS - {"day_night"}) - requested
    existing_excludes = set(exclude.split(",") if exclude else [])
    all_excludes = existing_excludes | blocks_to_exclude
    new_exclude = ",".join(all_excludes) if all_excludes else None

    # day_night is added to include when explicitly requested.
    new_include = include
    if "day_night" in requested:
        existing_includes = set(include.split(",") if include else [])
        existing_includes.add("day_night_forecast")
        new_include = ",".join(existing_includes)

    return new_exclude, new_include


def apply_block_indices(
    return_obj: dict,
    block_name: str,
    indices: list[int],
) -> None:
    """Apply index filtering to a ``data`` list inside a response block.

    Filters ``return_obj[block_name]["data"]`` to the specified indices in-place.

    Args:
        return_obj: The top-level response dictionary (modified in-place).
        block_name: The key of the block to filter (e.g. ``"daily"``).
        indices: List of non-negative integer indices (ascending order).

    Raises:
        HTTPException(400): If any index is out of range for the data list.
    """
    if block_name not in return_obj:
        return
    data = return_obj[block_name].get("data", [])
    out_of_range = [i for i in indices if i >= len(data)]
    if out_of_range:
        raise HTTPException(
            status_code=400,
            detail=f"{block_name}_indices contains out-of-range values: {out_of_range}",
        )
    return_obj[block_name]["data"] = [data[i] for i in indices]
