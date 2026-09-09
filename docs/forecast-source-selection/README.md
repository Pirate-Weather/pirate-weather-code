# Forecast Source Selection

This document describes the forecast-source selection implemented by the API. It
is intended as a companion or replacement section for the public
`Forecast Element Sources` documentation.

The API does not select one model for an entire response. It selects the first
valid source for each forecast element and timestamp. A response can therefore
combine models: for example, HRDPS temperature, REPS precipitation probability,
and NBM precipitation intensity can all appear in the same Canadian forecast.

## Selection model

Source selection follows this process:

1. Read sources that have not been excluded, cover the requested location, and
   pass their freshness checks.
2. Interpolate their data to the API's minute or hourly time grid.
3. Build an ordered, element-specific list of compatible sources.
4. Select the first non-missing value at each timestamp.

The `flags.sources` field reports the models that were available to a request.
It is not a per-element record of the model that produced every output value.

Daily and day/night values aggregate the selected hourly values. They do not
perform a separate model-selection pass.

## Regions

The API uses three source-priority regions:

| Region | Definition |
| --- | --- |
| Canada box | Latitude 41.7 to 83.0 and longitude -141.0 to -52.0. This is a coverage box, not a political-border lookup. |
| North America outside the Canada box | North American coverage outside that box. |
| Global / standard | All other locations. |

Models without coverage at the requested point are omitted before priorities are
evaluated. NBM, HRRR, RTMA-RU, and HRRR SubH consequently do not become global
fallbacks merely because they appear in a generic ordering.

## Excluding sources and enabling AI models

The `exclude` parameter can remove either the entire CMC group or individual
CMC models:

```text
exclude=cmc
exclude=cmcmodels
exclude=hrdps,gdps,geps,reps
```

`include=aimodels` is an explicit priority mode. When it is present:

- CMC sources (HRDPS, GDPS, GEPS, and REPS) are excluded before their grids are
  read.
- AI output is preferred for every supported element, including `currently`,
  `minutely`, `hourly`, `daily`, and day/night output.
- In North America, AIGFS and AIGEFS are represented through the GFS and GEFS
  data schemas respectively.
- Outside North America, ECMWF AIFS is represented through the ECMWF IFS data
  schema.
- Conventional model values fill only missing or unsupported AI values.

This means AI precedence does not require each API field to have a separately
named AI source. The AI data is merged into its compatible conventional schema,
then selected first.

## Conventional hourly priority

For most hourly weather elements, source stacks begin with the following
regional order. The element's supported-source set filters this order; for
example, REPS and GEPS are skipped for temperature because they provide
ensemble-precipitation fields rather than temperature fields.

| Region | Priority |
| --- | --- |
| Canada box | HRDPS → REPS → GDPS → GEPS → NBM → HRRR → ECMWF IFS → GFS → GEFS → DWD MOSMIX → ERA5 |
| North America outside Canada | NBM → HRRR → ECMWF IFS → GFS → GEFS → GDPS → GEPS → DWD MOSMIX → ERA5 → HRDPS → REPS |
| Global / standard | NBM → HRRR → DWD MOSMIX → ECMWF IFS → GFS → GEFS → GDPS → GEPS → ERA5 → HRDPS → REPS |

When `include=aimodels` is enabled, the compatible AI model is placed ahead of
the conventional order. In North America, AIGEFS leads ensemble precipitation
fields and AIGFS leads deterministic GFS-compatible fields. Globally, ECMWF
AIFS leads ECMWF-compatible fields.

## Currently

Current conditions use a dedicated source selector. For most surface elements,
it applies the regional order below and skips sources that do not supply the
requested field.

| Region | Priority for compatible surface elements |
| --- | --- |
| Canada box | RTMA-RU → HRRR SubH → HRDPS → GDPS → NBM → HRRR → ECMWF IFS → GFS → DWD MOSMIX → ERA5 |
| North America outside Canada | RTMA-RU → HRRR SubH → NBM → HRRR → ECMWF IFS → GFS → GDPS → DWD MOSMIX → ERA5 → HRDPS |
| Global / standard | RTMA-RU → HRRR SubH → NBM → HRRR → DWD MOSMIX → ECMWF IFS → GFS → GDPS → ERA5 → HRDPS |

With AI models enabled, compatible AI data is selected before RTMA-RU, HRRR
SubH, NBM, and the other conventional sources.

Current precipitation fields (`precipIntensity`, `precipProbability`,
`precipIntensityError`, and `precipType`) are copied from the first minute of
the minutely block and therefore use the minutely rules below.

Some current elements have intentionally narrower source sets:

| Element | Source logic |
| --- | --- |
| UV index | GFS-compatible source → ERA5 |
| Ozone | GFS-compatible source → ERA5 |
| Nearest storm | GFS-compatible source |
| Smoke | HRRR → SILAM |
| Feels-like temperature | NBM → GFS-compatible source |

## Minutely

Minutely output contains precipitation intensity, probability, error, and type.
It is the most specialized selector in the API.

### AI mode

When `include=aimodels` is set, AI models are preferred for every minutely
element they support:

| Region | Intensity and type | Probability and error |
| --- | --- | --- |
| North America | AIGEFS → AIGFS → conventional fallback | AIGEFS → ECMWF IFS → NBM → conventional fallback |
| Global / standard | ECMWF AIFS → conventional fallback | ECMWF AIFS → conventional fallback |

This precedence is deliberate: AI mode overrides HRRR SubH and NBM for
minutely intensity and type as well as probability and error. CMC sources are
not eligible in AI mode.

### Conventional mode in the Canada box

| Element | Priority |
| --- | --- |
| Precipitation intensity | HRRR SubH → NBM → DWD MOSMIX → ECMWF IFS → GEFS → GFS → GDPS → GEPS → ERA5 |
| Precipitation type | HRRR SubH → NBM → ECMWF IFS → GFS → DWD MOSMIX → GEFS → GDPS → GEPS → ERA5 |
| Precipitation probability | REPS → GEPS → NBM → ECMWF IFS → GEFS → ERA5 |
| Precipitation intensity error | REPS → ECMWF IFS → GEFS → GEPS |

The HRRR SubH priority applies whenever its data is available and AI mode is not
enabled. NBM is the next conventional intensity/type source. REPS is the
primary Canadian probability and error source; HRDPS and GDPS do not provide a
precipitation-probability field.

### Conventional mode outside the Canada box

| Element | North America | Global / standard |
| --- | --- | --- |
| Precipitation intensity | HRRR SubH → NBM → DWD MOSMIX → ECMWF IFS → GEFS → GFS → GDPS → GEPS → ERA5 | HRRR SubH → NBM → DWD MOSMIX → ECMWF IFS → GEFS → GFS → GDPS → GEPS → ERA5 |
| Precipitation type | HRRR SubH → NBM → ECMWF IFS → GFS → DWD MOSMIX → GEFS → GDPS → GEPS → ERA5 | HRRR SubH → NBM → DWD MOSMIX → ECMWF IFS → GFS → GEFS → GDPS → GEPS → ERA5 |
| Precipitation probability | NBM → ECMWF IFS → GEFS → GEPS → ERA5 | NBM → ECMWF IFS → GEFS → GEPS → ERA5 |
| Precipitation intensity error | ECMWF IFS → GEFS → GEPS | ECMWF IFS → GEFS → GEPS |

GDPS and GEPS are deliberately late minutely fallbacks outside the Canada box,
after the GFS/GEFS family.

## Hourly, daily, and day/night precipitation

Hourly precipitation uses separate source stacks because intensity,
probability, type, accumulation, and error have different source coverage.
Daily and day/night outputs aggregate these hourly results.

### Canada box, conventional mode

| Element | Effective priority |
| --- | --- |
| Precipitation intensity | HRDPS → GDPS → NBM → HRRR → ECMWF IFS → GEFS → GFS → DWD MOSMIX → ERA5 |
| Precipitation probability | REPS → GEPS → NBM → ECMWF IFS → GEFS → ERA5 |
| Precipitation type | REPS → GEPS → HRDPS → GDPS → NBM → HRRR → ECMWF IFS → GEFS → GFS → DWD MOSMIX → ERA5 |
| Precipitation accumulation | HRDPS → REPS → GDPS → GEPS → NBM → HRRR → ECMWF IFS → GFS → GEFS → DWD MOSMIX → ERA5 |
| Precipitation error | ECMWF IFS → GEFS → GEPS → REPS |

### Outside the Canada box, conventional mode

For North America, NBM and HRRR remain ahead of ECMWF IFS, GFS/GEFS, GDPS/GEPS,
and global fallbacks. For global locations, DWD MOSMIX is considered before
ECMWF IFS when a valid station forecast is available. GDPS and GEPS are placed
after GFS/GEFS and before the lower-priority fallback sources they support.

## Element-coverage exceptions

The generic regional stack is not used when only a small set of models provides
an element:

| Element family | Available sources, in its dedicated order |
| --- | --- |
| UV index | GFS-compatible source → HRDPS → GDPS → ERA5 |
| Ozone | GFS-compatible source → GDPS → ERA5 |
| Visibility | Regional stack filtered to NBM, HRRR, DWD MOSMIX, GFS-compatible source, and ERA5 |
| Nearest storm | GFS-compatible source only |
| Feels-like temperature | NBM → GFS-compatible source |
| CAPE | NBM → HRRR → HRDPS → GDPS → GFS-compatible source → ERA5 |
| Smoke | HRRR, with SILAM used for air-quality detail where applicable |
| Air quality | RAQDPS → SILAM; carbon monoxide uses SILAM |

## Fallback behavior

Priority is not a guarantee that one source supplies every time step. If the
preferred source has a missing value, the API falls through to the next source
for that timestamp and element. This also applies to AI models: AI output wins
where it exists, while conventional output fills gaps or fields that the AI
model does not provide.

Time-machine requests use their historical-source path and do not enable the
real-time CMC, NBM, HRRR, or AI precedence described above.
