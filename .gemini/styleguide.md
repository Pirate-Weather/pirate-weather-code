# Pirate Weather Python Style Guide

This document defines Python coding and review conventions for Pirate Weather.

For repository-wide engineering decisions—architecture, model-specific logic, read-performance priorities, testing philosophy, fallbacks, storage design, and scope discipline—follow [`../AGENTS.md`](../AGENTS.md).

If this file and `AGENTS.md` appear to conflict, **`AGENTS.md` takes precedence**.

---

## 1. Principles

Python code should be:

- **Readable:** Easy to understand in the context of the weather-data workflow.
- **Maintainable:** Straightforward to modify without introducing unnecessary architecture.
- **Consistent:** Mechanically consistent where tooling can enforce consistency.
- **Traceable:** Easy to connect to upstream model variables, units, and processing decisions.
- **Performance-aware:** Especially on API/data-read paths.

For Pirate Weather, **read performance is the primary optimization constraint**. Among implementations with comparable read performance, prefer the clearer and more maintainable solution.

---

## 2. Formatting

Pirate Weather uses Ruff for formatting and linting.

### Line Length

Maximum line length: **120 characters**.

Do not manually optimize formatting that Ruff will handle.

### Indentation

Use **4 spaces** per indentation level.

---

## 3. Imports

Group imports in this order:

1. standard library;
2. third-party packages;
3. local application/library imports.

Prefer absolute imports.

Let Ruff manage import ordering where configured.

Do not create review comments solely about import ordering that automated tooling can fix.

---

## 4. Naming

For new internal Python names:

- variables: `snake_case`;
- functions: `snake_case`;
- constants: `UPPER_SNAKE_CASE`;
- classes: `CapWords`;
- modules: `snake_case`.

Use descriptive names.

Avoid single-character names except for conventional indexes, dimensions, or equations where the meaning is obvious.

### Upstream Dataset Names

Preserve authoritative upstream weather/model identifiers when doing so improves provenance and debugging.

For example:

```python
APCP_surface
TMP_2maboveground
PTYPE_prob_GE_1_LT_2_prob_fcst_1_1_surface
```

do not need to be renamed merely to satisfy local naming conventions.

Use `snake_case` for the surrounding internal code.

---

## 5. Docstrings

Use triple double quotes for docstrings.

Add docstrings when they materially clarify:

- a reusable helper's purpose;
- non-obvious parameters or return values;
- units;
- exceptions;
- side effects;
- important domain assumptions.

Use Google-style structure for complex public/reusable functions where detailed parameter and return documentation is useful.

Do **not** add verbose docstrings to trivial helpers merely for stylistic completeness.

A concise docstring is preferable when the interface is simple.

---

## 6. Type Hints

Use standard Python type hints where they improve interfaces, readability, or correctness.

Prioritize type hints for:

- public/reusable functions;
- non-obvious return structures;
- configuration/state boundaries;
- functions where static typing can prevent realistic mistakes.

Do not add typing churn to unrelated legacy code.

Do not annotate every local variable merely to increase type-hint coverage.

---

## 7. Comments

Comments should explain **why**, especially when code reflects unusual behaviour in a weather product or serving path.

Useful comments explain:

- model quirks;
- accumulation timing;
- interpolation choices;
- grid assumptions;
- source precedence;
- fallbacks;
- unusual performance decisions;
- compatibility constraints.

Avoid comments that merely restate the next line of Python.

Prefer concise complete sentences when practical.

Remove obsolete commented-out code rather than preserving it indefinitely.

---

## 8. Logging

Use Python's built-in `logging` module for new production paths.

Use appropriate log levels:

- `DEBUG`: detailed diagnostic information;
- `INFO`: normal important lifecycle events;
- `WARNING`: degraded but recoverable behaviour;
- `ERROR`: failed operation or invalid product;
- `CRITICAL`: severe system-level failure.

Include useful context such as:

- model;
- model run/base time;
- source;
- path/store;
- elapsed time;
- retry count;
- failure reason.

Avoid adding high-volume logging to hot read paths if it materially affects serving performance.

Simple `print()` statements are acceptable in one-off utilities and temporary development diagnostics.

---

## 9. Error Handling

Catch specific exceptions.

Avoid:

```python
except:
    ...
```

and generally avoid broad `except Exception:` handlers unless the code is at a deliberate boundary where broad containment is justified.

Handle external-service failures gracefully **only when the resulting output remains valid**.

If a failed operation could produce scientifically invalid data, fail before publication rather than silently continuing.

Do not suppress programming errors behind generic fallback behaviour.

---

## 10. Subprocesses and Shell Commands

Important subprocess calls should use `check=True` or otherwise verify success.

Prefer argument lists:

```python
subprocess.run(
    [wgrib2_path, input_file, "-netcdf", output_file],
    check=True,
)
```

Avoid `shell=True` unless it is genuinely required.

Prefer Python filesystem APIs for operations such as file deletion, directory removal, and moves.

When external commands fail, preserve enough context to identify the model, run, input, and output involved.

---

## 11. Units

Use metric/SI-style units internally unless an established Pirate Weather convention requires otherwise.

Wind speed is represented internally in **metres per second**.

Make units explicit in variable names, constants, comments, or docstrings where ambiguity is possible.

Do not repeatedly hardcode conversion factors.

Prefer shared conversion utilities or named constants for reused conversions.

---

## 12. Weather Thresholds and Constants

Weather thresholds should be named constants.

Shared thresholds should live in the repository's shared constants location rather than being duplicated across modules.

Examples:

```python
LIGHT_PRECIP_MM_PER_HOUR
HEAVY_SNOW_MM_PER_HOUR
DEFAULT_VISIBILITY_M
HRRR_GRID_NX
```

A threshold or constant should communicate both its meaning and, where relevant, its units.

Do not replace authoritative upstream dataset values with generic constants when preserving the upstream identifier is more useful.

---

## 13. API Compatibility

Preserve backwards compatibility for existing API responses.

Do not silently change:

- response fields;
- units;
- meanings;
- defaults;
- source precedence;
- summary semantics.

New response items should normally be exposed through the existing API versioning or include mechanisms rather than being added unconditionally.

Any deliberate compatibility change should be explicit in the pull request.

---

## 14. Localization

Do not change forecast summary text, condition labels, or other user-facing strings without considering translation/localization behaviour.

Condition thresholds may be adjusted when justified, but changes to emitted text should account for the translation layer and API compatibility.

---

## 15. Performance-Sensitive Python

Do not add convenience abstractions to the hot read path without considering their cost.

Be cautious about:

- repeated xarray object construction;
- unnecessary `.compute()` calls;
- repeated metadata access;
- additional array copies;
- extra object-store requests;
- per-request configuration parsing;
- avoidable serialization/deserialization;
- reading more chunks than necessary.

If a Python change could affect request latency or throughput, benchmark it where practical.

Read-path performance takes priority over stylistic neatness.

---

## 16. Tests

Testing policy is defined primarily in `AGENTS.md`.

In summary:

- keep the test suite small;
- prefer broad end-to-end, integration, and regression tests;
- avoid trivial unit tests;
- do not add tests merely to raise coverage;
- every new test must document why it exists and what meaningful failure it protects against.

Example:

```python
def test_gfs_ingest_can_be_served_by_api(...):
    """Protect the ingest/API contract: GFS output must be directly readable
    by the forecast endpoint without an intermediate conversion step.
    """
    ...
```

Do not mirror every implementation detail with a corresponding test.

---

## 17. Review Noise

Do not comment on:

- import ordering handled by Ruff;
- formatting handled by Ruff;
- line wrapping within configured limits;
- minor naming differences that are consistent with nearby code;
- legacy style inconsistencies outside the scope of the current pull request;
- suggestions that would require a large refactor without a clear functional benefit;
- lack of trivial unit tests;
- missing type annotations in unrelated legacy code.

Focus reviews on:

- correctness;
- meteorological logic;
- units and timing;
- read performance;
- data layout;
- source selection and fallback behaviour;
- API compatibility;
- failure handling;
- meaningful maintainability concerns.

---

## 18. Tooling

Use the repository's configured tooling.

Primary tools:

- **Formatter:** Ruff
- **Linter:** Ruff

Run the repository-provided lint/format commands before submission.

Do not manually reformat unrelated files merely to make them look more consistent.

---

## 19. Example Style

Preferred:

```python
# NBM switches to 3-hour accumulation intervals after this point.
# Normalize to hourly values before merging onto the API time grid.
NBM_THREE_HOUR_START = 48

if forecast_hour >= NBM_THREE_HOUR_START:
    hourly_apcp = APCP_surface / 3
else:
    hourly_apcp = APCP_surface
```

Avoid:

```python
# Divide by 3.
x = x / 3
```

Also avoid introducing a generalized strategy/factory layer for a single model-specific rule unless the same abstraction solves a demonstrated recurring problem elsewhere.

---

## 20. Summary

Python style in Pirate Weather should support the repository's broader engineering philosophy:

> Keep code explicit, traceable to the underlying weather data, mechanically consistent, and fast on the read path.

Let Ruff handle formatting.

Use naming, comments, types, logging, constants, and docstrings to clarify real behaviour—not to add ceremony.

When a style preference conflicts with serving performance or clear meteorological logic, follow `AGENTS.md`.
