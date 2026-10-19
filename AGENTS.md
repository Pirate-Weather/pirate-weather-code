# Pirate Weather Agent Guide

This file defines the repository-wide engineering expectations for coding agents and contributors working on Pirate Weather.

It is the **authoritative guide for architecture, data-flow design, performance priorities, testing philosophy, failure handling, and scope discipline**.

For Python formatting and mechanical conventions, also follow [`.gemini/styleguide.md`](.gemini/styleguide.md). If the two documents appear to conflict, **this file takes precedence**.

The goal is not to reproduce every historical implementation detail. Preserve the strengths of the original Pirate Weather codebase—explicit meteorological logic, traceable data flow, practical operational behaviour, and fast reads—while avoiding historical technical debt.

---

## 1. Core Engineering Philosophy

Pirate Weather should remain a **practical, domain-first weather processing system**.

Prefer:

- explicit model- and dataset-specific logic;
- code organized around the real processing workflow;
- terminology that matches upstream weather products;
- visible operational behaviour;
- comments that explain unusual meteorological or numerical decisions;
- simple control flow;
- graceful fallbacks when they remain scientifically valid;
- modest abstractions with clear, demonstrated value;
- storage layouts and code paths optimized for serving performance.

Avoid adding architectural layers merely because they are common software-engineering patterns.

A good Pirate Weather implementation is usually **verbose about the weather problem and conservative about software architecture**.

---

## 2. Read Performance Is the First Optimization Priority

Pirate Weather is a high-volume data-serving system. **Read-path performance is the primary optimization constraint.**

Changes should not make forecast reads slower unless there is a compelling, documented reason.

When designing or reviewing a change, consider its effect on:

- number of files or stores opened per request;
- number of Zarr chunks read;
- chunk shape and alignment with API access patterns;
- number of remote/object-store requests;
- decompression cost;
- serialization/deserialization overhead;
- unnecessary array copies;
- repeated coordinate or metadata reads;
- per-request initialization;
- runtime transformations;
- validation on the hot path;
- network round trips;
- lock contention;
- synchronous I/O.

Prefer data layouts and code paths that let the API retrieve required data with as few reads, chunks, requests, and transformations as practical.

Do not add extra storage layers, metadata lookups, wrappers, indirection, or runtime transformations to the serving path if they measurably slow reads.

If a design makes ingest code cleaner but slows API reads, **the read path normally wins**.

Performance-sensitive changes should be benchmarked against the existing implementation where practical. Document meaningful before/after latency, throughput, chunk-read, or request-count differences when they could affect serving performance.

Among implementations with comparable read performance, prefer the clearer and more maintainable one.

---

## 3. Organize Code Around the Real Workflow

Top-level code should generally read in the same order that the work happens.

For an ingest script:

```text
determine model run
→ check whether it is new
→ define variables and forecast hours
→ download source data
→ transform GRIB fields
→ correct source-specific quirks
→ validate arrays
→ write Zarr
→ publish
→ clean up
```

For API code:

```text
parse request
→ determine location and time
→ determine applicable models
→ locate grid cells
→ read model data
→ merge/interpolate
→ calculate derived fields
→ build API response
```

Prefer visible, coherent processing sections over scattering one workflow across many files.

Do not split a linear processing sequence into many tiny functions simply to reduce function length.

Extract code when the extracted unit:

- has a meaningful independent purpose;
- is genuinely reused;
- materially improves testability or reliability;
- removes infrastructure duplication without hiding domain logic.

---

## 4. Keep Model-Specific Behaviour Visible

Weather models and source datasets are not interchangeable.

If HRRR, GFS, NBM, GEFS, RTMA, URMA, ECMWF, CMC, or another source requires different:

- spatial coverage;
- projection or grid lookup;
- forecast-hour handling;
- accumulation handling;
- interpolation;
- variable names;
- update cadence;
- historical handling;
- source precedence;
- fallback behaviour;

keep that difference explicit.

Preferred:

```python
if location_in_hrrr_domain:
    # HRRR-specific projection and grid lookup.
    ...

# GFS uses a global latitude/longitude grid.
...
```

Avoid hiding meaningful source behaviour behind generalized abstractions such as factories, providers, adapters, or strategy hierarchies unless they clearly reduce recurring complexity.

---

## 5. Extract Calculations, Not Architecture

Small helpers are encouraged when they represent a concrete operation.

Good examples:

```python
find_nearest(...)
lambert_grid_match(...)
round_time(...)
get_timezone_offset(...)
publish_product(...)
retry_download(...)
```

Be cautious about introducing:

- factories;
- registries;
- managers;
- provider hierarchies;
- service classes;
- adapters;
- strategy patterns;
- configuration frameworks;
- dependency-injection layers.

Do not create a class merely to group related functions.

Before introducing a new abstraction, ask:

1. Is this logic reused in multiple places?
2. Does the abstraction reduce actual complexity?
3. Does it preserve visibility of model-specific behaviour?
4. Will a maintainer be able to trace the data flow more easily afterward?
5. Does it solve an existing bug, maintenance problem, or testability issue?

If the answer is mostly no, keep the implementation simple.

---

## 6. Preserve Meteorological and Upstream Vocabulary

Use names that connect directly to the source dataset and processing step.

Examples:

```python
gfs_start_idx
hrrr_merged
nbm_fire_merged
APCP_surface
TMP_2maboveground
base_day_utc
hour_array_grib
```

Exact upstream identifiers such as `APCP_surface` should be preserved where they help trace values back to GRIB inventories or source documentation.

Use `snake_case` for new internal Python names, but do **not** rename authoritative upstream fields merely to satisfy naming consistency.

Prefer:

```python
APCP_surface
```

over:

```python
precipitation_data
```

when the former improves provenance and debugging.

---

## 7. Use Meaningful Intermediate Variables

Intermediate values are useful when they correspond to real processing steps.

Preferred:

```python
hours_to_subtract = base_day_utc.hour % 6
rounded_time = base_day_utc - datetime.timedelta(hours=hours_to_subtract)
forecast_times = pd.date_range(...)
```

Avoid compressing several meaningful transformations into one dense expression solely to reduce line count.

Weather-processing bugs are often timing, indexing, unit, or accumulation bugs. Intermediate values make those easier to inspect.

---

## 8. Comment the Reason, Not the Syntax

Comments should preserve knowledge that is not obvious from the code.

Useful comments explain:

- why an upstream product behaves unexpectedly;
- why a forecast range is split;
- why an accumulation is differenced;
- why interpolation is or is not needed;
- why a fallback exists;
- why a grid dimension or threshold is used;
- why a performance workaround exists;
- why a timestamp is rounded a certain way;
- why a source takes precedence.

Good:

```python
# NBM switches from hourly to 3-hour accumulation intervals here.
# Convert to hourly values before merging onto the API time grid.
```

Poor:

```python
# Divide precipitation by 3.
precip = precip / 3
```

Remove obsolete commented-out implementations. Git history is the archive.

---

## 9. Prefer Explicit Control Flow Over Cleverness

Use straightforward loops, conditionals, and transformations when they make the data flow easier to understand.

A little repetition is acceptable when it keeps materially different model behaviour visible.

Do not eliminate small amounts of duplication at the cost of making the processing logic harder to follow.

Do not create one-line helper functions that force the reader to jump around the repository without adding semantic value.

---

## 10. Distinguish Domain Repetition From Infrastructure Duplication

Model-specific repetition can be useful.

Infrastructure repetition usually is not.

Good candidates for shared helpers include:

- bounded download retries;
- checking whether a model run is new;
- common S3/local publishing;
- temporary-directory creation and cleanup;
- Zarr publication;
- subprocess execution;
- file timestamp checks;
- shared validation routines;
- unit conversions.

Keep meteorological transformations model-specific unless they are genuinely identical.

---

## 11. Make Operational Behaviour Visible

Schedules, paths, data movement, storage decisions, model dependencies, and external commands should be easy to locate.

A maintainer should be able to answer:

- When does this ingest run?
- Which model cycle does it use?
- Where are temporary files written?
- What gets published?
- What happens if the source is unavailable?
- Can two runs overlap?
- Which file or store does the API read?

without reconstructing a large abstraction hierarchy.

---

## 12. Treat Storage Layout as Part of the Algorithm

Storage design directly affects read performance and should be considered part of the implementation.

It is acceptable for code to make explicit decisions about:

- Zarr layout and chunking;
- compressor choice;
- Dask chunk sizes;
- NumPy vs. xarray operations;
- `.compute()` placement;
- concurrent reads;
- preloaded datasets;
- temporary files;
- local vs. object storage;
- rechunking;
- memory cleanup.

Prefer measured performance over theoretical elegance.

When an implementation is intentionally unusual for performance reasons, add a short comment explaining why.

Read performance takes precedence over ingest-time neatness, abstraction, or minor storage savings unless the tradeoff is explicitly justified.

---

## 13. Use Practical, Domain-Aware Fallbacks

External weather sources fail.

Where scientifically acceptable, Pirate Weather should degrade to a useful result rather than fail the whole request.

Examples:

- fall back from a regional model to a global model;
- retain basic alert information if enrichment fails;
- omit an optional source rather than fail the entire API response;
- use an older valid product when the newest product is incomplete and doing so is explicitly acceptable.

Fallback logic should be close to the operation that can fail and should explain why the fallback is valid.

Do **not** hide failures that could make the output scientifically invalid.

If a failed operation means the product cannot be trusted, fail before publication.

---

## 14. Preserve API Compatibility

Backwards compatibility is important.

Do not silently change:

- existing response fields;
- units;
- field meanings;
- model precedence;
- interpolation behaviour;
- summary semantics;
- default query behaviour.

New API response items should normally be gated through the existing API-version or include mechanisms rather than appearing unconditionally.

Any intentional compatibility change should be explicit in the pull request.

---

## 15. Preserve Unit Consistency

Use metric/SI-style units internally unless an established Pirate Weather internal convention requires otherwise. Wind speed is represented internally in metres per second.

Do not scatter raw conversion factors through the code.

Prefer shared conversion utilities or named constants where conversions are reused.

Always make the source and target units unambiguous.

---

## 16. Use Named Weather Thresholds

Weather thresholds should be named constants rather than unexplained literals.

Shared thresholds should live in the repository's shared constants location instead of being duplicated across modules.

Examples include:

- precipitation intensity thresholds;
- snow thresholds;
- visibility thresholds;
- wind thresholds;
- icon breakpoints;
- alert or summary thresholds.

A threshold should communicate both its meaning and units.

---

## 17. Consider Localization When Changing User-Facing Text

Do not casually change forecast summary text, condition labels, or other user-facing strings without considering localization and translation behaviour.

Changes to condition breakpoints may be appropriate, but changes to emitted text should account for translation support and API compatibility.

---

# Historical Patterns That Should NOT Be Preserved

The original codebase contains useful engineering instincts as well as accumulated technical debt. Matching the old style does **not** mean reproducing the following patterns.

---

## 18. Do Not Use Bare or Overly Broad Exceptions

Avoid:

```python
try:
    ...
except:
    ...
```

and generally avoid catching `Exception` unless there is a compelling boundary-level reason.

Catch expected exception types and preserve enough context to diagnose the failure.

Do not convert programming errors such as `KeyError`, `TypeError`, or `NameError` into misleading download or model failures.

---

## 19. Check External Command Failures

Important external commands must check their return status.

Prefer:

```python
subprocess.run(
    [wgrib2_path, input_file, "-netcdf", output_file],
    check=True,
)
```

over:

```python
subprocess.run(command, shell=True)
```

If a command fails, surface enough context to identify:

- model;
- model run;
- input;
- command;
- output.

Do not allow a failed conversion to become an unrelated downstream array error.

---

## 20. Avoid `shell=True` Unless It Is Required

Prefer argument lists and Python filesystem APIs.

Use:

```python
shutil.rmtree(path)
os.remove(path)
subprocess.run(["unzip", "-q", "-o", archive, "-d", destination], check=True)
```

rather than constructing shell command strings.

If `shell=True` is genuinely necessary, document why and ensure no untrusted value can enter the command.

---

## 21. Use Bounded Retry Loops

Do not copy and paste nested retry chains.

Use a finite, explicit retry loop with the relevant exception type and delay/backoff.

Example:

```python
for attempt in range(max_attempts):
    try:
        return download(...)
    except ExpectedDownloadError:
        if attempt == max_attempts - 1:
            raise
        time.sleep(retry_delay)
```

Retries should not silently convert permanent failures into infinite waits.

---

## 22. Reduce Unnecessary Global Mutable State

Avoid adding new module-level mutable globals for datasets, request state, or configuration.

For long-lived application resources, prefer explicit application/framework state.

Global constants are fine.

Long-lived read-only handles may be appropriate for performance reasons, but initialization and replacement behaviour must remain easy to understand.

---

## 23. Name Important Constants

Avoid unexplained values such as:

```python
if x > 1799:
    ...
num_hours = 193
```

Prefer named constants or values derived from source data:

```python
HRRR_GRID_NX = 1800
API_FORECAST_HOURS = 193
```

Numbers defined by an external model specification should be named and documented where useful.

---

## 24. Remove Dead and Debug Code

Do not leave large blocks of commented-out implementations in production files.

Remove:

- abandoned approaches;
- old test timestamps;
- commented debug statements;
- unused variables;
- temporary experiments;
- obsolete storage alternatives.

Keep explanatory comments.

Use Git history for previous implementations.

---

## 25. Prefer Structured Logging Over Production `print()`

New production code should normally use the standard `logging` module.

Logs should include useful context such as model, run time, source, elapsed time, and failure reason.

Simple `print()` calls remain acceptable for one-off utilities and temporary local diagnostics.

---

## 26. Validate Data at Boundaries

Before publishing a product, validate assumptions that would make output unusable if violated.

Depending on the dataset, consider checking:

- expected variables exist;
- expected dimensions are present;
- spatial dimensions are plausible;
- timestamps are monotonic;
- run timestamp matches the requested model cycle;
- expected forecast periods are present;
- arrays do not contain obviously invalid ranges;
- accumulation fields behave as expected;
- output stores can be reopened successfully.

Prefer validation at ingest or publication boundaries rather than adding avoidable work to the hot API read path.

Fail before publication if a product is incomplete or scientifically invalid.

---

## 27. Keep Tests Minimal, Broad, and End-to-End

Pirate Weather should not accumulate a large suite of low-value tests.

Prefer a **small number of broad tests that exercise meaningful end-to-end behaviour** over many narrow tests that mirror implementation details.

Good tests cover things such as:

- an ingest path from representative source input to a readable output product;
- an API request through model selection, data read, transformation, and response generation;
- a major fallback path;
- a historically fragile behaviour that could produce incorrect forecasts;
- a critical data-layout or compatibility contract;
- a regression for a real bug that is likely to recur.

Avoid adding tests for:

- trivial getters or wrappers;
- simple assignments;
- standard-library behaviour;
- obvious one-line calculations;
- implementation details that can change without affecting behaviour;
- every helper function simply because it exists;
- exhaustive unit coverage for code already exercised by a broader integration test.

Every new test must include a short comment or docstring explaining **why the test exists** and what meaningful failure it is intended to catch.

Example:

```python
def test_gfs_ingest_can_be_served_by_api(...):
    """Protect the ingest/API contract: a completed GFS ingest must be
    directly readable by the forecast endpoint without a conversion step.
    """
    ...
```

The rationale should describe the risk being protected against, not merely restate the test name.

Preferred hierarchy:

```text
a few end-to-end/regression tests
→ targeted integration tests where needed
→ narrow unit tests only when they protect non-obvious logic
```

Do not add tests solely to increase test count or coverage percentage.

A test should earn its place by protecting important behaviour.

---

## 28. Avoid Unrelated Refactors

Changes should normally be narrowly scoped.

When fixing a bug or adding a data source:

- avoid unrelated renaming;
- avoid reformatting untouched sections beyond automatic formatting;
- avoid replacing working architecture without a demonstrated need;
- preserve existing API behaviour unless the change intentionally modifies it;
- do not silently change model precedence, timing, units, interpolation, or fallback behaviour.

Broad architectural cleanup should be a separate, justified change.

---

# Agent Review Behaviour

Agents reviewing code should focus on functional, scientific, performance, reliability, and maintainability issues.

Do not create review noise about:

- import ordering handled by Ruff;
- formatting handled by Ruff;
- line wrapping within configured limits;
- minor naming differences that are consistent with nearby code;
- legacy style inconsistencies outside the scope of the current pull request;
- suggestions requiring a large refactor without a clear functional benefit.

Do flag:

- read-path regressions;
- incorrect model/source behaviour;
- unsafe fallbacks;
- unit or timing errors;
- hidden API compatibility changes;
- scientifically invalid assumptions;
- unbounded retries;
- swallowed exceptions;
- unchecked subprocess failures;
- unnecessary abstraction;
- trivial or unjustified tests.

---

# Before Submitting Code

Agents should verify:

- [ ] Read performance was treated as the first optimization priority.
- [ ] The change does not add unnecessary reads, chunks, remote requests, transformations, or hot-path overhead.
- [ ] Performance-sensitive changes were measured where practical.
- [ ] The code follows the actual processing workflow and is easy to trace.
- [ ] Model-specific behaviour remains visible.
- [ ] Any new abstraction solves a demonstrated problem.
- [ ] Upstream variable names and units remain traceable.
- [ ] Unusual meteorological, numerical, or performance decisions are explained.
- [ ] Expected external failures have explicit fallback or failure behaviour.
- [ ] Scientifically invalid output cannot be silently published.
- [ ] There are no bare or unjustifiably broad exception handlers.
- [ ] Important subprocess calls check for failure.
- [ ] No unnecessary `shell=True` commands were introduced.
- [ ] Retry loops are bounded.
- [ ] Important constants and thresholds are named.
- [ ] Shared unit conversions and thresholds are not unnecessarily duplicated.
- [ ] API compatibility and localization were considered where relevant.
- [ ] Dead/debug code was removed.
- [ ] New production paths use appropriate logging.
- [ ] Boundary validation is present where bad data could otherwise be published.
- [ ] Tests are limited to meaningful end-to-end, integration, or regression coverage.
- [ ] Every new test documents its rationale and the failure it protects against.
- [ ] No trivial tests were added solely for coverage or test count.
- [ ] Ruff/linting passes.
- [ ] The change does not include unrelated architectural cleanup.

---

# Summary

Pirate Weather should remain a **practical, explicit, performance-first weather processing system**.

The preferred style is:

> Explicit about models, data, grids, timing, fallbacks, storage, and operational behaviour; conservative about architectural abstraction.

**Optimize reads first. Keep domain logic visible. Keep tests few, broad, and justified by meaningful risk.**

When in doubt, optimize for the next maintainer being able to answer:

**"What happens to the weather data here, why does it happen, and does this make reads slower?"**
