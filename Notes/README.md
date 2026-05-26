# AI Model Integration Notes

This folder now documents how AI forecast models are wired into the response pipeline.

## Feature flag and request behavior

- AI model usage is gated behind `include=aimodels`.
- Model-level excludes are supported via `exclude=aigefs`, `exclude=aigfs`, and `exclude=aifs`.
- Exclude flags take priority over the include flag.

## Regional model selection

- In North America, `AIGEFS` and `AIGFS` are preferred when `include=aimodels` is set.
- Outside North America, `AIFS` is preferred when `include=aimodels` is set.

## Integration pattern used

1. **Load stores** in `API/io/zarr_reader.py` and thread through `ZarrSources`.
2. **Read model slices** in `API/request/grid_indexing.py` only when `include=aimodels` is enabled and excludes allow them.
3. **Normalize AI arrays** in `API/forecast_sources.py` to existing response-friendly shapes:
   - AIGFS -> GFS-like layout
   - AIGEFS -> GEFS-like layout
   - AIFS -> ECMWF-like layout
4. **Prioritize AI-backed data** in currently/minutely/hourly-daily input selection when `include=aimodels` is active.
5. **Precipitation type fallback** for AIGEFS minutely output uses temperature thresholds (snow/rain/sleet) when explicit precip-type channels are not available.

## Adding future models quickly

Follow the same path: request parsing -> zarr loading -> grid indexing -> merge normalization -> source-priority hooks -> targeted tests.
