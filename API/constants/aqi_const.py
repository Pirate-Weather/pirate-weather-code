"""Air Quality Index (AQI) constants and calculation helpers.

Supports three AQI systems:
  - US EPA AQI  (unit_system="us")
  - Canadian AQHI (unit_system="ca")
  - EU CAQI     (unit_system="uk" or "si")

All input concentrations use the model-native units:
  - PM2.5, PM10: µg/m³
  - O3, NO2, SO2, CO: ppb

References:
    - EPA AQI Technical Assistance Document: https://www.airnow.gov/aqi/aqi-basics/
    - EPA AQI Breakpoints: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    - Health Canada AQHI: https://www.canada.ca/en/environment-climate-change/services/air-quality-health-index.html
    - EU CAQI: https://www.airqualitynow.eu/about_indices_definition.php
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# EPA averaging helpers
# NowCast for PM2.5 and PM10; 8-hour rolling mean for O3/CO.
# References:
#   - EPA NowCast: https://usepa.servicenowservices.com/airnow?id=kb_article_view&sys_id=bb8b65ef1b06bc10028420eae54bcb98
#   - EPA AQI Technical Assistance Document (Sept 2018)
# ---------------------------------------------------------------------------


def nowcast_pm(conc: np.ndarray) -> np.ndarray:
    """Apply EPA NowCast algorithm to an hourly PM2.5 or PM10 concentration array.

    NowCast is a weighted average over up to 12 previous hours.  The weight factor
    is derived from the ratio of minimum to maximum concentration in the window.

    Validity rule: at least 2 of the 3 most-recent hours must contain valid data;
    otherwise the output for that hour is NaN.

    Args:
        conc: 1-D array of hourly concentrations (µg/m³), chronological order
              (index 0 = earliest hour, last index = most-recent hour).
              NaN indicates missing/invalid data.

    Returns:
        1-D float64 array of the same length as *conc*.
    """
    n = len(conc)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        # Build window of up to 12 hours, newest first
        start = max(0, i - 11)
        window = conc[start : i + 1][::-1]  # newest = index 0

        # Validity check: at least 2 of the 3 most-recent hours must be valid
        recent_valid = np.sum(~np.isnan(window[:3]))
        if recent_valid < 2:
            continue

        valid_mask = ~np.isnan(window)
        if not np.any(valid_mask):
            continue

        valid_conc = window[valid_mask]
        max_c = float(np.max(valid_conc))
        min_c = float(np.min(valid_conc))

        # Weight factor: bounded at [0.5, 1.0]
        if max_c == 0.0:
            w = 1.0
        else:
            w = max(min_c / max_c, 0.5)

        total_weight = 0.0
        weighted_sum = 0.0
        for j, c in enumerate(window):
            if not math.isnan(c):
                wi = w**j
                total_weight += wi
                weighted_sum += wi * float(c)

        if total_weight > 0.0:
            out[i] = weighted_sum / total_weight

    return out


def rolling_mean(conc: np.ndarray, window: int) -> np.ndarray:
    """Compute a backward-looking rolling mean ignoring NaN values.

    At least one valid observation in the window is required; otherwise the
    output for that hour is NaN.

    Args:
        conc:   1-D array of hourly concentrations, chronological order.
        window: Number of hours to include (e.g., 8 for O3/CO, 24 for PM10).

    Returns:
        1-D float64 array of the same length as *conc*.
    """
    n = len(conc)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        start = max(0, i - window + 1)
        segment = conc[start : i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            out[i] = float(np.mean(valid))

    return out


# ---------------------------------------------------------------------------
# EPA AQI breakpoints (concentrations in µg/m³)
# The model stores O3, NO2, SO2, CO in ppb; convert before lookup.
# ---------------------------------------------------------------------------

# PM2.5 (Fine Particulate Matter, µg/m³)
# Breakpoints for 24-hour average PM2.5 concentrations
PM25_BP = [0, 9.0, 35.4, 55.4, 125.4, 225.4, 325.4]
PM25_AQI = [0, 50, 100, 150, 200, 300, 500]

# PM10 (Coarse Particulate Matter, µg/m³)
# Breakpoints for 24-hour average PM10 concentrations
PM10_BP = [0, 54, 154, 254, 354, 424, 504, 604]
PM10_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# O3 (Ozone, µg/m³) — EPA breakpoints converted to µg/m³ (1 ppm O3 ≈ 1996 µg/m³ @ 25°C)
# Breakpoints for 8-hour average ozone concentrations
O3_8H_BP = [0, 55, 71, 86, 106, 201]
O3_8H_AQI = [0, 50, 100, 150, 200, 300]
# Breakpoints for 1-hour average ozone concentrations
O3_1H_BP = [125, 165, 205, 405, 505, 605]
O3_1H_AQI = [100, 150, 200, 300, 400, 500]

# NO2 (Nitrogen Dioxide, ppb) — EPA breakpoints
# Breakpoints for 1-hour average NO2 concentrations
NO2_BP = [0, 53, 100, 360, 649, 1249, 1649, 2049]
NO2_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# SO2 (Sulfur Dioxide, ppb) — EPA breakpoints
# Breakpoints for 1-hour average SO2 concentrations
SO2_BP = [0, 35, 75, 185, 304, 604, 804, 1004]
SO2_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# CO (Carbon Monoxide, ppb) — EPA breakpoints
# Breakpoints for 8-hour average CO concentrations
CO_BP = [0, 4400, 9400, 12400, 15400, 30400, 40400, 50400]
CO_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# ---------------------------------------------------------------------------
# Unit conversion factors (ppb → µg/m³ at 25 °C, 1 atm)
# Used to bring model-native ppb values to µg/m³ before EPA breakpoint lookup.
# ---------------------------------------------------------------------------
PPB_O3_TO_UG_M3 = 1.996  # 1 ppb O3  ≈ 1.996 µg/m³
PPB_NO2_TO_UG_M3 = 1.88  # 1 ppb NO2 ≈ 1.88  µg/m³
PPB_SO2_TO_UG_M3 = 2.62  # 1 ppb SO2 ≈ 2.62  µg/m³
PPB_CO_TO_UG_M3 = 1.145  # 1 ppb CO  ≈ 1.145 µg/m³

# ---------------------------------------------------------------------------
# EU CAQI breakpoints (µg/m³ for all species)
# Reference: https://www.airqualitynow.eu/about_indices_definition.php
# Roadside index uses the same ranges as background for the species we expose.
# ---------------------------------------------------------------------------
CAQI_PM25_BP = [0, 15, 30, 55, 110]
CAQI_PM10_BP = [0, 25, 50, 90, 180]
CAQI_O3_BP = [0, 60, 120, 180, 240]  # µg/m³
CAQI_NO2_BP = [0, 50, 100, 200, 400]  # µg/m³
CAQI_INDEX = [0, 25, 50, 75, 100]  # CAQI 0–100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_interp(
    conc: float, bp_lo: float, bp_hi: float, idx_lo: int, idx_hi: int
) -> int:
    """Linearly interpolate AQI from concentration within a breakpoint range."""
    if bp_hi == bp_lo:
        return idx_lo
    ratio = (conc - bp_lo) / (bp_hi - bp_lo)
    return round(idx_lo + ratio * (idx_hi - idx_lo))


def _aqi_from_breakpoints(conc: float, bp: list, aqi_vals: list) -> float:
    """Return the sub-index AQI value for *conc* using the given breakpoint tables."""
    if math.isnan(conc) or conc < 0:
        return float("nan")
    for i in range(len(bp) - 1):
        if bp[i] <= conc <= bp[i + 1]:
            return float(
                _linear_interp(conc, bp[i], bp[i + 1], aqi_vals[i], aqi_vals[i + 1])
            )
    # Above top breakpoint → cap at maximum
    return float(aqi_vals[-1])


# ---------------------------------------------------------------------------
# EPA AQI
# ---------------------------------------------------------------------------


def _epa_sub_index(conc_ug: float, bp: list, aqi_vals: list) -> float:
    return _aqi_from_breakpoints(conc_ug, bp, aqi_vals)


def compute_epa_aqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_8h_ppb: float = float("nan"),
    o3_1h_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute US EPA AQI as the maximum sub-index across available pollutants.

    Pollutant concentrations should be in model-native units:
      pm25_ug, pm10_ug → µg/m³
      o3_8h_ppb, o3_1h_ppb, no2_ppb, so2_ppb, co_ppb → ppb
    """
    sub_indices = []

    o3_sub = float("nan")

    # 8-hour ozone
    if not math.isnan(o3_8h_ppb):
        o3_sub = _epa_sub_index(o3_8h_ppb, O3_8H_BP, O3_8H_AQI)

    # 1-hour ozone only applies when 8-hour AQI > 100
    if not math.isnan(o3_1h_ppb) and not math.isnan(o3_sub) and o3_sub > 100:
        o3_1h_sub = _epa_sub_index(o3_1h_ppb, O3_1H_BP, O3_1H_AQI)
        if not math.isnan(o3_1h_sub):
            o3_sub = max(o3_sub, o3_1h_sub)

    if not math.isnan(pm25_ug):
        sub_indices.append(_epa_sub_index(pm25_ug, PM25_BP, PM25_AQI))
    if not math.isnan(pm10_ug):
        sub_indices.append(_epa_sub_index(pm10_ug, PM10_BP, PM10_AQI))
    if not math.isnan(o3_sub):
        sub_indices.append(o3_sub)
    if not math.isnan(no2_ppb):
        sub_indices.append(_epa_sub_index(no2_ppb, NO2_BP, NO2_AQI))
    if not math.isnan(so2_ppb):
        sub_indices.append(_epa_sub_index(so2_ppb, SO2_BP, SO2_AQI))
    if not math.isnan(co_ppb):
        sub_indices.append(_epa_sub_index(co_ppb, CO_BP, CO_AQI))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Canadian AQHI
# Formula: AQHI = (10/10.4) * sum(e^(beta_i * C_i) - 1) * 100
# where beta coefficients are from Health Canada (2008).
# C_i are 3-hour rolling averages in µg/m³ for PM2.5 and ppb for NO2 and O3
# Reference: https://www.canada.ca/en/environment-climate-change/services/air-quality-health-index/
# ---------------------------------------------------------------------------
AQHI_BETA_O3 = 0.000537  # Coefficient expects ppb
AQHI_BETA_NO2 = 0.000871  # Coefficient expects ppb
AQHI_BETA_PM25 = 0.000487  # Coefficient expects µg/m³
AQHI_SCALE = 10.0 / 10.4
AQHI_MAX = 15.0


def compute_aqhi(
    pm25_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
) -> float:
    """Compute Canadian AQHI (1–10+ scale, capped at 15)."""
    # Convert gases from ppb to µg/m³
    o3 = o3_ppb if not math.isnan(o3_ppb) else float("nan")
    no2 = no2_ppb if not math.isnan(no2_ppb) else float("nan")
    pm25 = pm25_ug if not math.isnan(pm25_ug) else float("nan")

    total = 0.0
    count = 0
    if not math.isnan(o3):
        total += math.exp(AQHI_BETA_O3 * o3) - 1
        count += 1
    if not math.isnan(no2):
        total += math.exp(AQHI_BETA_NO2 * no2) - 1
        count += 1
    if not math.isnan(pm25):
        total += math.exp(AQHI_BETA_PM25 * pm25) - 1
        count += 1

    if count == 0:
        return float("nan")

    aqhi = AQHI_SCALE * total * 100.0
    # Scale to familiar 1–10 range (10+ is "very high risk")
    return round(min(aqhi, AQHI_MAX), 1)


# ---------------------------------------------------------------------------
# EU CAQI
# ---------------------------------------------------------------------------


def compute_caqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
) -> float:
    """Compute EU Common Air Quality Index (0–100+ scale).

    Returns the maximum sub-index across available pollutants.
    """
    o3_ug = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")

    sub_indices = []
    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, CAQI_PM25_BP, CAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, CAQI_PM10_BP, CAQI_INDEX))
    if not math.isnan(o3_ug):
        sub_indices.append(_aqi_from_breakpoints(o3_ug, CAQI_O3_BP, CAQI_INDEX))
    if not math.isnan(no2_ug):
        sub_indices.append(_aqi_from_breakpoints(no2_ug, CAQI_NO2_BP, CAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

# Maps unit_system → (system_name, aqi_scale_max)
AQI_SYSTEM_MAP = {
    "us": "EPA",
    "ca": "AQHI",
    "uk": "CAQI",
    "si": "CAQI",
}


def compute_aqi_for_unit_system(
    unit_system: str,
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute AQI using the system appropriate for *unit_system*.

    Returns a scalar float (nan when no pollutant data is available).
    """
    system = AQI_SYSTEM_MAP.get(unit_system, "EPA")
    if system == "EPA":
        return compute_epa_aqi(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb, co_ppb)
    elif system == "AQHI":
        return compute_aqhi(pm25_ug, o3_ppb, no2_ppb)
    else:  # CAQI
        return compute_caqi(pm25_ug, pm10_ug, o3_ppb, no2_ppb)


def compute_aqi_array(
    unit_system: str,
    pm25: np.ndarray | None,
    pm10: np.ndarray | None,
    o3: np.ndarray | None,
    no2: np.ndarray | None,
    so2: np.ndarray | None,
    co: np.ndarray | None,
) -> np.ndarray:
    """Vectorised AQI computation over hourly arrays.

    For the US EPA system the appropriate EPA averaging periods are applied
    before the breakpoint lookup:
      - PM2.5: NowCast (12-hour weighted average)
      - PM10:  24-hour rolling mean
      - O3:    8-hour rolling mean
      - CO:    8-hour rolling mean
      - NO2, SO2: 1-hour (no additional averaging)

    For AQHI (CA) and CAQI (UK/SI) the raw hourly concentrations are used
    since those indices are designed for hourly values.

    All input arrays should have the same length (num_hours); None entries are
    treated as fully-missing for that pollutant.

    Returns a float32 array of shape (num_hours,) with NaN where data is missing.
    """
    n = (
        max(len(arr) for arr in (pm25, pm10, o3, no2, so2, co) if arr is not None)
        if any(arr is not None for arr in (pm25, pm10, o3, no2, so2, co))
        else 0
    )
    if n == 0:
        return np.full(0, np.nan, dtype=np.float32)

    def _get(arr):
        if arr is None:
            return np.full(n, np.nan, dtype=np.float64)
        a = np.asarray(arr, dtype=np.float64)
        if len(a) < n:
            out = np.full(n, np.nan, dtype=np.float64)
            out[: len(a)] = a
            return out
        return a[:n]

    pm25_v = _get(pm25)
    pm10_v = _get(pm10)
    o3_v = _get(o3)
    no2_v = _get(no2)
    so2_v = _get(so2)
    co_v = _get(co)

    system = AQI_SYSTEM_MAP.get(unit_system, "EPA")

    if system == "EPA":
        # Apply EPA-mandated averaging periods before the breakpoint lookup
        pm25_calc = nowcast_pm(pm25_v)
        pm10_calc = nowcast_pm(pm10_v)
        o3_calc = rolling_mean(o3_v, window=8)
        co_calc = rolling_mean(co_v, window=8)
        # NO2 and SO2 use 1-hour (instantaneous) averages
        no2_calc = no2_v
        so2_calc = so2_v
        o3_1h_calc = o3_v
    elif system == "AQHI":
        # AQHI uses 3-hour rolling averages for PM2.5, O3, NO2
        pm25_calc = rolling_mean(pm25_v, window=3)
        o3_calc = rolling_mean(o3_v, window=3)
        no2_calc = rolling_mean(no2_v, window=3)
        # Not used in AQHI but we still pass them through for consistency
        pm10_calc = pm10_v
        so2_calc = so2_v
        co_calc = co_v
    else:
        # CAQI use raw hourly concentrations
        pm25_calc = pm25_v
        pm10_calc = pm10_v
        o3_calc = o3_v
        no2_calc = no2_v
        so2_calc = so2_v
        co_calc = co_v

    result = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if system == "EPA":
            result[i] = compute_epa_aqi(
                pm25_ug=float(pm25_calc[i]),
                pm10_ug=float(pm10_calc[i]),
                o3_8h_ppb=float(o3_calc[i]),
                o3_1h_ppb=float(o3_1h_calc[i]),
                no2_ppb=float(no2_calc[i]),
                so2_ppb=float(so2_calc[i]),
                co_ppb=float(co_calc[i]),
            )
        else:
            result[i] = compute_aqi_for_unit_system(
                unit_system,
                float(pm25_calc[i]),
                float(pm10_calc[i]),
                float(o3_calc[i]),
                float(no2_calc[i]),
                float(so2_calc[i]),
                float(co_calc[i]),
            )
    return result
