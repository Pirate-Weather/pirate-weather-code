"""Air Quality Index (AQI) constants and calculation helpers.

Supports three AQI systems:
  - US EPA AQI  (unit_system="us")
  - Canadian AQHI (unit_system="ca")
  - EU CAQI     (unit_system="si")
  - UK DAQI      (unit_system="uk")
  - Hong Kong AQHI (aqisystem="hk")
  - Ireland AQIH (aqisystem="ie")
  - Israel AQHI (aqisystem="il")
  - Indonesia ISPU (aqisystem="id")
  - China AQI (aqisystem="cn")
  - Malaysia API (aqisystem="my")
  - Taiwan AQI (aqisystem="tw")
  - Vietnam VN_AQI (aqisystem="vn")

All input concentrations use the model-native units:
  - PM2.5, PM10: µg/m³
  - O3, NO2, SO2, CO: ppb

References:
    - EPA AQI Technical Assistance Document: https://www.airnow.gov/aqi/aqi-basics/
    - EPA AQI Breakpoints: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    - Health Canada AQHI: https://www.canada.ca/en/environment-climate-change/services/air-quality-health-index.html
    - EU CAQI: https://www.airqualitynow.eu/about_indices_definition.php
    - UK DAQI: https://aqihub.info/indices/uk
    - Hong Kong AQHI: https://www.aqhi.gov.hk/en.html
    - Ireland AQIH: https://aqihub.info/indices/ireland
    - Israel AQI: https://aqihub.info/indices/israel
    - Indonesia ISPU: https://aqihub.info/indices/indonesia
    - China AQI: https://aqihub.info/indices/china
    - Malaysia API: https://aqihub.info/indices/malaysia
    - Taiwan AQI: https://aqihub.info/indices/taiwan
    - Vietnam VN_AQI: https://aqihub.info/indices/vietnam
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
O3_8H_BP = [0, 54, 70, 85, 105, 200]
O3_8H_AQI = [0, 50, 100, 150, 200, 300]
# Breakpoints for 1-hour average ozone concentrations
O3_1H_BP = [124, 164, 204, 404, 504, 604]
O3_1H_AQI = [100, 150, 200, 300, 400, 500]

# NO2 (Nitrogen Dioxide, ppb) — EPA breakpoints
# Breakpoints for 1-hour average NO2 concentrations
NO2_BP = [0, 53, 100, 360, 649, 1249, 1649, 2049]
NO2_AQI = [0, 50, 100, 150, 200, 300, 400, 500]

# SO2 (Sulfur Dioxide, ppb) — EPA breakpoints
# Breakpoints for 1-hour average SO2 concentrations
# EPA SO2 Breakpoints
SO2_BP = [0, 35, 75, 185, 304]
SO2_AQI = [0, 50, 100, 150, 200]

SO2_24H_BP = [305, 604, 804, 1004]
SO2_24H_AQI = [201, 300, 400, 500]

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
# Normalized EU EAQI breakpoints (µg/m³ for all species)
# Reference: https://airindex.eea.europa.eu/AQI/index.html
# ---------------------------------------------------------------------------
EAQI_PM25_BP = [0, 5, 15, 50, 90, 140]
EAQI_PM10_BP = [0, 15, 45, 120, 195, 270]
EAQI_O3_BP = [0, 60, 100, 120, 160, 180]  # µg/m³
EAQI_NO2_BP = [0, 10, 25, 60, 100, 150]  # µg/m³
EAQI_SO2_BP = [0, 20, 40, 125, 190, 275]  # µg/m³
EAQI_INDEX = [0, 20, 40, 60, 80, 100]  # EAQI 0–100

# ---------------------------------------------------------------------------
# UK DAQI breakpoints (µg/m³ for all species)
# Reference: https://aqihub.info/indices/uk
# ---------------------------------------------------------------------------
DAQI_PM25_BP = [0, 12, 24, 36, 42, 48, 54, 59, 65, 71]
DAQI_PM10_BP = [0, 17, 34, 51, 59, 67, 76, 84, 92, 101]
DAQI_O3_BP = [0, 34, 67, 101, 121, 141, 161, 188, 214, 241]  # µg/m³
DAQI_NO2_BP = [0, 68, 135, 201, 268, 335, 401, 468, 535, 601]  # µg/m³
DAQI_SO2_BP = [0, 89, 178, 267, 355, 444, 533, 711, 888, 1065]  # µg/m³
DAQI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # DAQI 1–10

# ---------------------------------------------------------------------------
# Ireland AQIH breakpoints (µg/m³ for all species)
# Reference: https://aqihub.info/indices/ireland
# ---------------------------------------------------------------------------
# The Irish AQIH uses the same breakpoints as the UK DAQI, but the SO2 breakpoints are slightly different. The AQIH SO2 breakpoints are:
AQIH_SO2_BP = [0, 30, 60, 90, 120, 150, 180, 237, 296, 355]  # µg/m³

# ---------------------------------------------------------------------------
# Israel AQI breakpoints (µg/m³ for all species except CO, which is in mg/m³)
# Reference: https://aqihub.info/indices/israel
# ---------------------------------------------------------------------------
IAQI_PM25_BP = [0, 18.5, 37, 84, 130, 165, 200]
IAQI_PM10_BP = [0, 65, 129, 215, 300, 355, 430]
IAQI_O3_BP = [0, 35, 70, 97, 117, 155, 188]  # µg/m³
IAQI_NO2_BP = [0, 53, 105, 160, 213, 260, 316]  # µg/m³
IAQI_SO2_BP = [0, 67, 133, 163, 191, 253, 303]  # µg/m³
IAQI_CO_BP = [0, 26, 51, 78, 104, 130, 156]  # mg/m³
IAQI_INDEX = [0, 50, 100, 200, 300, 400, 500]  # Israel 0–500

# ---------------------------------------------------------------------------
# Indonesia ISPU breakpoints (µg/m³ for all species)
# Reference: https://aqihub.info/indices/indonesia
# ---------------------------------------------------------------------------
ISPU_PM25_BP = [0, 15.5, 55.4, 150.4, 250.4, 500]
ISPU_PM10_BP = [0, 50, 150, 350, 420, 500]
ISPU_O3_BP = [0, 120, 235, 400, 800, 1000]  # µg/m³
ISPU_NO2_BP = [0, 80, 200, 1130, 2260, 3000]  # µg/m³
ISPU_SO2_BP = [0, 52, 180, 400, 800, 1200]  # µg/m³
ISPU_CO_BP = [0, 4000, 8000, 15000, 30000, 45000]  # µg/m³
ISPU_INDEX = [0, 50, 100, 200, 300, 500]  # ISPU 0–500

# ---------------------------------------------------------------------------
# China AQI breakpoints (µg/m³ for all species except CO, which is in mg/m³)
# Reference: https://aqihub.info/indices/china
# ---------------------------------------------------------------------------
# Master IAQI Scale (0 to 500)
CHINA_AQI_INDEX = [0, 50, 100, 150, 200, 300, 400, 500]
CHINA_PM25_24H_BP = [0.0, 35.0, 75.0, 115.0, 150.0, 250.0, 350.0, 500.0]
CHINA_PM10_24H_BP = [0.0, 50.0, 150.0, 250.0, 350.0, 420.0, 500.0, 600.0]
CHINA_NO2_1H_BP = [0.0, 100.0, 200.0, 700.0, 1200.0, 2340.0, 3090.0, 3840.0]
CHINA_NO2_24H_BP = [0.0, 40.0, 80.0, 180.0, 280.0, 565.0, 750.0, 940.0]
CHINA_O3_1H_BP = [0.0, 160.0, 200.0, 300.0, 400.0, 800.0, 1000.0, 1200.0]
CHINA_O3_8H_BP = [0.0, 100.0, 160.0, 215.0, 265.0, 800.0]
CHINA_SO2_1H_BP = [0.0, 150.0, 500.0, 650.0, 800.0]
CHINA_SO2_24H_BP = [0.0, 50.0, 150.0, 475.0, 800.0, 1600.0, 2100.0, 2620.0]
CHINA_CO_1H_BP = [0.0, 5.0, 10.0, 35.0, 60.0, 90.0, 120.0, 150.0]
CHINA_CO_24H_BP = [0.0, 2.0, 4.0, 14.0, 24.0, 36.0, 48.0, 60.0]

# ---------------------------------------------------------------------------
# Malaysia API breakpoints (µg/m³ for PM and ppb for gases)
# Reference: https://aqihub.info/indices/malaysia
# ---------------------------------------------------------------------------
API_PM25_BP = [0, 103, 153, 203, 253, 508]
API_PM10_BP = [0, 190, 240, 290, 340, 682]
API_O3_BP = [0, 80, 134, 187, 240, 482]
API_NO2_BP = [0, 106, 178, 249, 320, 642]
API_SO2_BP = [0, 134, 172, 210, 249, 500]
API_CO_BP = [0, 3000, 4000, 5000, 6000, 12000]
API_INDEX = [0, 50, 100, 200, 300, 500]  # API 0–500

# ---------------------------------------------------------------------------
# Taiwan AQI breakpoints (µg/m³ for PM and ppb for NO2 and SO2 and ppm for CO and O3)
# Reference: https://aqihub.info/indices/taiwan
# ---------------------------------------------------------------------------
TAQI_PM25_BP = [0, 15.4, 35.4, 54.4, 150.4, 250.4, 500.4]
TAQI_PM10_BP = [0, 50, 100, 254, 354, 424, 604]
TAQI_O3_1H_BP = [0, 0, 124, 164, 204, 404, 604]
TAQI_O3_8H_BP = [0, 54, 70, 85, 105, 200, 200]
TAQI_NO2_BP = [0, 30, 100, 360, 649, 1249, 2049]
TAQI_SO2_BP = [0, 20, 75, 185, 304, 604, 605]
TAQI_CO_BP = [0, 4400, 9400, 12400, 15400, 30400, 60400]
TAQI_INDEX = [0, 50, 100, 150, 200, 300, 500]  # Taiwan 0–500

# ---------------------------------------------------------------------------
# Vietnam VN_AQI breakpoints (µg/m³ for all species)
# Reference: https://aqihub.info/indices/vietnam
# ---------------------------------------------------------------------------
VNAQI_PM25_BP = [0, 25, 80, 150, 250, 350, 500]
VNAQI_PM10_BP = [0, 50, 250, 350, 430, 500, 600]
VNAQI_O3_1H_BP = [0, 160, 300, 400, 800, 1000, 1200]
VNAQI_O3_8H_BP = [0, 100, 170, 210, 400, 400, 400]
VNAQI_NO2_BP = [0, 100, 700, 1200, 2350, 3100, 3850]
VNAQI_SO2_BP = [0, 125, 550, 800, 1600, 2100, 2630]
VNAQI_CO_BP = [0, 10000, 45000, 60000, 90000, 120000, 150000]
VNAQI_INDEX = [0, 50, 100, 150, 200, 300, 500]  # Vietnam 0–500

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


def _index_from_breakpoints(conc: float, bp: list, aqi_vals: list) -> float:
    """Return the sub-index AQI value for *conc* using the given breakpoint tables."""
    if math.isnan(conc) or conc < 0:
        return float("nan")
    for i in range(len(bp) - 1, -1, -1):
        if conc >= bp[i]:
            return aqi_vals[i]
    # Below lowest breakpoint → return minimum AQI
    return aqi_vals[0]


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
    so2_24h_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute US EPA AQI as the maximum sub-index across available pollutants.

    Pollutant concentrations should be in model-native units:
      pm25_ug, pm10_ug → µg/m³
      o3_8h_ppb, o3_1h_ppb, no2_ppb, so2_ppb, so2_24h_ppb, co_ppb → ppb
    """
    sub_indices = []

    o3_sub = float("nan")

    # 8-hour ozone
    if not math.isnan(o3_8h_ppb):
        o3_sub = _epa_sub_index(o3_8h_ppb, O3_8H_BP, O3_8H_AQI)

    # 1-hour ozone only applies when 8-hour AQI > 100 and concentration is at least the minimum breakpoint
    if (
        not math.isnan(o3_1h_ppb)
        and not math.isnan(o3_sub)
        and o3_sub > 100
        and o3_1h_ppb >= O3_1H_BP[0]
    ):
        o3_1h_sub = _epa_sub_index(o3_1h_ppb, O3_1H_BP, O3_1H_AQI)
        if not math.isnan(o3_1h_sub):
            o3_sub = max(o3_sub, o3_1h_sub)

    # SO2 logic: Try 1-hour breakpoint first; if > 200 (or > 304 ppb), fall back to 24-hour average
    so2_sub = float("nan")
    if not math.isnan(so2_ppb):
        so2_sub = _epa_sub_index(so2_ppb, SO2_BP, SO2_AQI)

    if (math.isnan(so2_sub) or so2_sub > 200) and not math.isnan(so2_24h_ppb):
        so2_24h_sub = _epa_sub_index(so2_24h_ppb, SO2_24H_BP, SO2_24H_AQI)
        if not math.isnan(so2_24h_sub):
            so2_sub = so2_24h_sub

    if not math.isnan(pm25_ug):
        sub_indices.append(_epa_sub_index(pm25_ug, PM25_BP, PM25_AQI))
    if not math.isnan(pm10_ug):
        sub_indices.append(_epa_sub_index(pm10_ug, PM10_BP, PM10_AQI))
    if not math.isnan(o3_sub):
        sub_indices.append(o3_sub)
    if not math.isnan(no2_ppb):
        sub_indices.append(_epa_sub_index(no2_ppb, NO2_BP, NO2_AQI))
    if not math.isnan(so2_sub):
        sub_indices.append(so2_sub)
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
# Hong Kong AQHI
# Formula: AQHI = %AR = %ARNO2 + %ARSO2 + %ARO3 + %ARPM
# where beta coefficients are from the Hong Kong Environmental Protection Department.
# C_i are 3-hour rolling averages in µg/m³ for PM2.5, PM10, SO2, NO2 and O3
# Reference: https://www.aqhi.gov.hk/en.html
# ---------------------------------------------------------------------------
HK_AQHI_BETA_O3 = 0.0005116328  # Coefficient expects µg/m³
HK_AQHI_BETA_NO2 = 0.0004462559  # Coefficient expects µg/m³
HK_AQHI_BETA_PM25 = 0.0002180567  # Coefficient expects µg/m³
HK_AQHI_BETA_PM10 = 0.0002821751  # Coefficient expects µg/m³
HK_AQHI_BETA_SO2 = 0.0001393235  # Coefficient expects µg/m³
HK_AR_BP = [0, 1.89, 3.77, 5.65, 7.53, 9.42, 11.30, 12.92, 15.08, 17.23, 19.38]
HK_AQHI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def compute_hk_aqhi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
) -> float:
    """Compute Hong Kong AQHI (1–10+ scale, capped at 11)."""
    # Convert gases from ppb to µg/m³
    o3 = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2 = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2 = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")
    pm25 = pm25_ug if not math.isnan(pm25_ug) else float("nan")
    pm10 = pm10_ug if not math.isnan(pm10_ug) else float("nan")

    # Default PM Variables to NaN
    pm25_ar = float("nan")
    pm10_ar = float("nan")

    # Initialize all AR values to 0.0, so that if a pollutant is missing, it contributes 0 to the total AR
    pm_ar = 0.0
    o3_ar = 0.0
    no2_ar = 0.0
    so2_ar = 0.0

    count = 0
    if not math.isnan(o3):
        o3_ar = (math.exp(HK_AQHI_BETA_O3 * o3) - 1) * 100
        count += 1
    if not math.isnan(no2):
        no2_ar = (math.exp(HK_AQHI_BETA_NO2 * no2) - 1) * 100
        count += 1
    if not math.isnan(so2):
        so2_ar = (math.exp(HK_AQHI_BETA_SO2 * so2) - 1) * 100
        count += 1
    if not math.isnan(pm25):
        pm25_ar = (math.exp(HK_AQHI_BETA_PM25 * pm25) - 1) * 100
    if not math.isnan(pm10):
        pm10_ar = (math.exp(HK_AQHI_BETA_PM10 * pm10) - 1) * 100

    if not math.isnan(pm25) or not math.isnan(pm10):
        pm_ar += np.nanmax([pm25_ar, pm10_ar])
        count += 1
    if count == 0:
        return float("nan")

    ar_total = o3_ar + no2_ar + so2_ar + pm_ar
    # Scale to familiar 1–10 range (10+ is "very high risk")
    return _index_from_breakpoints(ar_total, HK_AR_BP, HK_AQHI)


# ---------------------------------------------------------------------------
# EU CAQI
# ---------------------------------------------------------------------------


def compute_caqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
) -> float:
    """Compute the European Air Quality Index (EAQI) on the legacy CAQI scale.

    Returns the maximum sub-index across available pollutants.
    """
    o3_ug = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2_ug = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")

    sub_indices = []
    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, EAQI_PM25_BP, EAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, EAQI_PM10_BP, EAQI_INDEX))
    if not math.isnan(o3_ug):
        sub_indices.append(_aqi_from_breakpoints(o3_ug, EAQI_O3_BP, EAQI_INDEX))
    if not math.isnan(no2_ug):
        sub_indices.append(_aqi_from_breakpoints(no2_ug, EAQI_NO2_BP, EAQI_INDEX))
    if not math.isnan(so2_ug):
        sub_indices.append(_aqi_from_breakpoints(so2_ug, EAQI_SO2_BP, EAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# UK DAQI
# ---------------------------------------------------------------------------


def compute_daqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
) -> float:
    """Compute DAQI AQI as the maximum sub-index across available pollutants.

    For the DAQI system the appropriate averaging periods are applied
    before the breakpoint lookup:
      - PM2.5, PM10: 24-hour rolling mean
      - O3: 8-hour rolling mean
      - NO2: 1-hour (no additional averaging)
      - SO2: 1-hour (no additional averaging)

    Pollutant concentrations should be in model-native units:
      pm25_ug, pm10_ug → µg/m³
      o3_ppb, no2_ppb, so2_ppb, co_ppb → ppb
    """

    o3_ug = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2_ug = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")

    sub_indices = []
    if not math.isnan(o3_ug):
        sub_indices.append(_index_from_breakpoints(o3_ug, DAQI_O3_BP, DAQI_INDEX))
    if not math.isnan(pm25_ug):
        sub_indices.append(_index_from_breakpoints(pm25_ug, DAQI_PM25_BP, DAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_index_from_breakpoints(pm10_ug, DAQI_PM10_BP, DAQI_INDEX))
    if not math.isnan(no2_ug):
        sub_indices.append(_index_from_breakpoints(no2_ug, DAQI_NO2_BP, DAQI_INDEX))
    if not math.isnan(so2_ug):
        sub_indices.append(_index_from_breakpoints(so2_ug, DAQI_SO2_BP, DAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Ireland AQIH
# ---------------------------------------------------------------------------


def compute_aqih(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
) -> float:
    """Compute Ireland AQIH as the maximum sub-index across available pollutants. AQIH uses the same breakpoints as the UK DAQI, but the SO2 breakpoints are slightly different.

    For the AQIH system the appropriate averaging periods are applied
    before the breakpoint lookup:
      - PM2.5, PM10: 24-hour rolling mean
      - O3: 8-hour rolling mean
      - NO2: 1-hour (no additional averaging)
      - SO2: 1-hour (no additional averaging)

    Pollutant concentrations should be in model-native units:
      pm25_ug, pm10_ug → µg/m³
      o3_ppb, no2_ppb, so2_ppb, co_ppb → ppb
    """

    o3_ug = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2_ug = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")

    sub_indices = []
    if not math.isnan(o3_ug):
        sub_indices.append(_index_from_breakpoints(o3_ug, DAQI_O3_BP, DAQI_INDEX))
    if not math.isnan(pm25_ug):
        sub_indices.append(_index_from_breakpoints(pm25_ug, DAQI_PM25_BP, DAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_index_from_breakpoints(pm10_ug, DAQI_PM10_BP, DAQI_INDEX))
    if not math.isnan(no2_ug):
        sub_indices.append(_index_from_breakpoints(no2_ug, DAQI_NO2_BP, DAQI_INDEX))
    if not math.isnan(so2_ug):
        sub_indices.append(_index_from_breakpoints(so2_ug, AQIH_SO2_BP, DAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Israel IAQI
# ---------------------------------------------------------------------------


def compute_iaqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute the Israeli Air Quality Index (IAQI).

    Returns the maximum sub-index across available pollutants.
    """

    # Convert gases from ppb to µg/m³
    o3_ug = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2_ug = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")
    co_mg = (
        (co_ppb * PPB_CO_TO_UG_M3) / 1000 if not math.isnan(co_ppb) else float("nan")
    )

    sub_indices = []
    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, IAQI_PM25_BP, IAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, IAQI_PM10_BP, IAQI_INDEX))
    if not math.isnan(o3_ug):
        sub_indices.append(_aqi_from_breakpoints(o3_ug, IAQI_O3_BP, IAQI_INDEX))
    if not math.isnan(no2_ug):
        sub_indices.append(_aqi_from_breakpoints(no2_ug, IAQI_NO2_BP, IAQI_INDEX))
    if not math.isnan(so2_ug):
        sub_indices.append(_aqi_from_breakpoints(so2_ug, IAQI_SO2_BP, IAQI_INDEX))
    if not math.isnan(co_mg):
        sub_indices.append(_aqi_from_breakpoints(co_mg, IAQI_CO_BP, IAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    # Israel AQI is defined as 100 to -400 (100 is best, 400 is worst), so we subtract the maximum sub-index from 100 to get the final AQI value.
    worst_sub_index = float(max(valid)) if valid else float("nan")

    return (
        float(100 - worst_sub_index)
        if not math.isnan(worst_sub_index)
        else float("nan")
    )


# ---------------------------------------------------------------------------
# Indonesia ISPU
# ---------------------------------------------------------------------------


def compute_ispu(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute the Indonesian Air Quality Index (ISPU).

    Returns the maximum sub-index across available pollutants.
    """

    # Convert gases from ppb to µg/m³
    o3_ug = o3_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_ppb) else float("nan")
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2_ug = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")
    co_ug = co_ppb * PPB_CO_TO_UG_M3 if not math.isnan(co_ppb) else float("nan")

    sub_indices = []
    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, ISPU_PM25_BP, ISPU_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, ISPU_PM10_BP, ISPU_INDEX))
    if not math.isnan(o3_ug):
        sub_indices.append(_aqi_from_breakpoints(o3_ug, ISPU_O3_BP, ISPU_INDEX))
    if not math.isnan(no2_ug):
        sub_indices.append(_aqi_from_breakpoints(no2_ug, ISPU_NO2_BP, ISPU_INDEX))
    if not math.isnan(so2_ug):
        sub_indices.append(_aqi_from_breakpoints(so2_ug, ISPU_SO2_BP, ISPU_INDEX))
    if not math.isnan(co_ug):
        sub_indices.append(_aqi_from_breakpoints(co_ug, ISPU_CO_BP, ISPU_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# China AQI
# ---------------------------------------------------------------------------


def compute_china_aqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_8h_ppb: float = float("nan"),
    o3_1h_ppb: float = float("nan"),
    no2_1h_ppb: float = float("nan"),
    no2_24h_ppb: float = float("nan"),
    so2_1h_ppb: float = float("nan"),
    so2_24h_ppb: float = float("nan"),
    co_1h_ppb: float = float("nan"),
    co_24h_ppb: float = float("nan"),
) -> float:
    """Compute China AQI as the maximum sub-index across available pollutants."""

    def safe_max(*vals):
        """Returns the maximum of valid numeric scores, or NaN if all are invalid."""
        valid = [v for v in vals if v is not None and not math.isnan(v)]
        return max(valid) if valid else float("nan")

    sub_indices = []

    # Convert gases from ppb to µg/m³
    o3_1h_ug = (
        o3_1h_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_1h_ppb) else float("nan")
    )
    o3_8h_ug = (
        o3_8h_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_8h_ppb) else float("nan")
    )
    no2_1h_ug = (
        no2_1h_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_1h_ppb) else float("nan")
    )
    no2_24h_ug = (
        no2_24h_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_24h_ppb) else float("nan")
    )
    so2_1h_ug = (
        so2_1h_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_1h_ppb) else float("nan")
    )
    so2_24h_ug = (
        so2_24h_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_24h_ppb) else float("nan")
    )
    co_1h_mg = (
        (co_1h_ppb * PPB_CO_TO_UG_M3) / 1000
        if not math.isnan(co_1h_ppb)
        else float("nan")
    )
    co_24h_mg = (
        (co_24h_ppb * PPB_CO_TO_UG_M3) / 1000
        if not math.isnan(co_24h_ppb)
        else float("nan")
    )

    o3_sub = float("nan")
    so2_sub = float("nan")
    no2_sub = float("nan")
    co_sub = float("nan")

    # --- 1. OZONE (O3) ---
    o3_1h_sub = float("nan")
    o3_8h_sub = float("nan")

    if not math.isnan(o3_1h_ug):
        o3_1h_sub = _aqi_from_breakpoints(o3_1h_ug, CHINA_O3_1H_BP, CHINA_AQI_INDEX)

    if not math.isnan(o3_8h_ug):
        # HJ 633: If O3 8-hr > 800, 8-hr calculation is invalidated in favor of 1-hr.
        # If 1-hr is unavailable, clamp at 800 ug/m3.
        if o3_8h_ug <= 800:
            o3_8h_sub = _aqi_from_breakpoints(o3_8h_ug, CHINA_O3_8H_BP, CHINA_AQI_INDEX)
        elif math.isnan(o3_1h_sub):
            o3_8h_sub = _aqi_from_breakpoints(800.0, CHINA_O3_8H_BP, CHINA_AQI_INDEX)

    o3_sub = safe_max(o3_1h_sub, o3_8h_sub)

    # --- 2. SULFUR DIOXIDE (SO2) ---
    so2_1h_sub = float("nan")
    so2_24h_sub = float("nan")

    if not math.isnan(so2_24h_ug):
        so2_24h_sub = _aqi_from_breakpoints(
            so2_24h_ug, CHINA_SO2_24H_BP, CHINA_AQI_INDEX
        )

    if not math.isnan(so2_1h_ug):
        # HJ 633: SO2 1-hr table only goes up to IAQI 200 (800 ug/m3).
        # For concentrations > 800, rely on 24-hr average. If 24-hr is missing, cap at 200.
        if so2_1h_ug <= 800:
            so2_1h_sub = _aqi_from_breakpoints(
                so2_1h_ug, CHINA_SO2_1H_BP, CHINA_AQI_INDEX
            )
        elif math.isnan(so2_24h_sub):
            so2_1h_sub = 200.0  # Cap at max available 1-hr IAQI

    so2_sub = safe_max(so2_1h_sub, so2_24h_sub)

    # --- 3. NITROGEN DIOXIDE (NO2) ---
    no2_1h_sub = float("nan")
    no2_24h_sub = float("nan")

    if not math.isnan(no2_1h_ug):
        no2_1h_sub = _aqi_from_breakpoints(no2_1h_ug, CHINA_NO2_1H_BP, CHINA_AQI_INDEX)

    if not math.isnan(no2_24h_ug):
        no2_24h_sub = _aqi_from_breakpoints(
            no2_24h_ug, CHINA_NO2_24H_BP, CHINA_AQI_INDEX
        )

    no2_sub = safe_max(no2_1h_sub, no2_24h_sub)

    # --- 4. CARBON MONOXIDE (CO) ---
    co_1h_sub = float("nan")
    co_24h_sub = float("nan")

    if not math.isnan(co_1h_mg):
        co_1h_sub = _aqi_from_breakpoints(co_1h_mg, CHINA_CO_1H_BP, CHINA_AQI_INDEX)

    if not math.isnan(co_24h_mg):
        co_24h_sub = _aqi_from_breakpoints(co_24h_mg, CHINA_CO_24H_BP, CHINA_AQI_INDEX)

    co_sub = safe_max(co_1h_sub, co_24h_sub)

    if not math.isnan(pm25_ug):
        sub_indices.append(
            _aqi_from_breakpoints(pm25_ug, CHINA_PM25_24H_BP, CHINA_AQI_INDEX)
        )
    if not math.isnan(pm10_ug):
        sub_indices.append(
            _aqi_from_breakpoints(pm10_ug, CHINA_PM10_24H_BP, CHINA_AQI_INDEX)
        )
    if not math.isnan(o3_sub):
        sub_indices.append(o3_sub)
    if not math.isnan(no2_sub):
        sub_indices.append(no2_sub)
    if not math.isnan(so2_sub):
        sub_indices.append(so2_sub)
    if not math.isnan(co_sub):
        sub_indices.append(co_sub)

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Malaysia API
# ---------------------------------------------------------------------------


def compute_api(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute the Malaysian Air Quality Index (API).

    Returns the maximum sub-index across available pollutants.
    """

    sub_indices = []
    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, API_PM25_BP, API_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, API_PM10_BP, API_INDEX))
    if not math.isnan(o3_ppb):
        sub_indices.append(_aqi_from_breakpoints(o3_ppb, API_O3_BP, API_INDEX))
    if not math.isnan(no2_ppb):
        sub_indices.append(_aqi_from_breakpoints(no2_ppb, API_NO2_BP, API_INDEX))
    if not math.isnan(so2_ppb):
        sub_indices.append(_aqi_from_breakpoints(so2_ppb, API_SO2_BP, API_INDEX))
    if not math.isnan(co_ppb):
        sub_indices.append(_aqi_from_breakpoints(co_ppb, API_CO_BP, API_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Taiwan AQI
# ---------------------------------------------------------------------------


def compute_taiwan_aqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_8h_ppb: float = float("nan"),
    o3_1h_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute Taiwan AQI as the maximum sub-index across available pollutants."""
    sub_indices = []

    # Initialize all sub-index values to NaN
    o3_sub = float("nan")
    sub_1h = float("nan")
    sub_8h = float("nan")

    # 1. Calculate 8-hour sub-index (valid up to AQI 300)
    if not math.isnan(o3_8h_ppb):
        # 8-hr table caps at 200 ppb (AQI 300)
        sub_8h = _aqi_from_breakpoints(o3_8h_ppb, TAQI_O3_8H_BP, TAQI_INDEX)

    # 2. Calculate 1-hour sub-index (only evaluated if >= 125 ppb / AQI >= 101)
    if not math.isnan(o3_1h_ppb) and o3_1h_ppb >= 125.0:
        sub_1h = _aqi_from_breakpoints(o3_1h_ppb, TAQI_O3_1H_BP, TAQI_INDEX)

    # 3. Handle threshold boundaries
    # If 8-hour AQI is >= 301, force fallback to 1-hour
    if sub_8h > 300:
        o3_sub = sub_1h if not math.isnan(sub_1h) else 300.0
    else:
        # Return max of available sub-indices
        o3_valid = [v for v in (sub_1h, sub_8h) if not math.isnan(v)]
        o3_sub = max(o3_valid) if o3_valid else float("nan")

    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, TAQI_PM25_BP, TAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, TAQI_PM10_BP, TAQI_INDEX))
    if not math.isnan(o3_sub):
        sub_indices.append(o3_sub)
    if not math.isnan(no2_ppb):
        sub_indices.append(_aqi_from_breakpoints(no2_ppb, TAQI_NO2_BP, TAQI_INDEX))
    if not math.isnan(so2_ppb):
        sub_indices.append(_aqi_from_breakpoints(so2_ppb, TAQI_SO2_BP, TAQI_INDEX))
    if not math.isnan(co_ppb):
        sub_indices.append(_aqi_from_breakpoints(co_ppb, TAQI_CO_BP, TAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Vietnam VN_AQI
# ---------------------------------------------------------------------------


def compute_vn_aqi(
    pm25_ug: float = float("nan"),
    pm10_ug: float = float("nan"),
    o3_8h_ppb: float = float("nan"),
    o3_1h_ppb: float = float("nan"),
    no2_ppb: float = float("nan"),
    so2_ppb: float = float("nan"),
    co_ppb: float = float("nan"),
) -> float:
    """Compute Vietnam VN_AQI as the maximum sub-index across available pollutants."""
    sub_indices = []

    # Convert gases from ppb to µg/m³
    o3_1h_ug = (
        o3_1h_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_1h_ppb) else float("nan")
    )
    o3_8h_ug = (
        o3_8h_ppb * PPB_O3_TO_UG_M3 if not math.isnan(o3_8h_ppb) else float("nan")
    )
    no2_ug = no2_ppb * PPB_NO2_TO_UG_M3 if not math.isnan(no2_ppb) else float("nan")
    so2_ug = so2_ppb * PPB_SO2_TO_UG_M3 if not math.isnan(so2_ppb) else float("nan")
    co_ug = co_ppb * PPB_CO_TO_UG_M3 if not math.isnan(co_ppb) else float("nan")

    # Initialize all sub-indices to NaN
    o3_sub = float("nan")
    sub_1h = float("nan")
    sub_8h = float("nan")

    # 1. Calculate 8-hour sub-index (valid up to AQI 200)
    if not math.isnan(o3_8h_ug):
        # 8-hr table caps at 400 µg/m³ (AQI 200)
        sub_8h = _aqi_from_breakpoints(o3_8h_ug, VNAQI_O3_8H_BP, VNAQI_INDEX)

    # 2. Calculate 1-hour sub-index
    if not math.isnan(o3_1h_ug):
        sub_1h = _aqi_from_breakpoints(o3_1h_ug, VNAQI_O3_1H_BP, VNAQI_INDEX)

    # 3. Handle threshold boundaries
    # If 8-hour AQI is >= 201, force fallback to 1-hour
    if sub_8h > 200:
        o3_sub = sub_1h if not math.isnan(sub_1h) else 200.0
    else:
        # Return max of available sub-indices
        o3_valid = [v for v in (sub_1h, sub_8h) if not math.isnan(v)]
        o3_sub = max(o3_valid) if o3_valid else float("nan")

    if not math.isnan(pm25_ug):
        sub_indices.append(_aqi_from_breakpoints(pm25_ug, VNAQI_PM25_BP, VNAQI_INDEX))
    if not math.isnan(pm10_ug):
        sub_indices.append(_aqi_from_breakpoints(pm10_ug, VNAQI_PM10_BP, VNAQI_INDEX))
    if not math.isnan(o3_sub):
        sub_indices.append(o3_sub)
    if not math.isnan(no2_ug):
        sub_indices.append(_aqi_from_breakpoints(no2_ug, VNAQI_NO2_BP, VNAQI_INDEX))
    if not math.isnan(so2_ug):
        sub_indices.append(_aqi_from_breakpoints(so2_ug, VNAQI_SO2_BP, VNAQI_INDEX))
    if not math.isnan(co_ug):
        sub_indices.append(_aqi_from_breakpoints(co_ug, VNAQI_CO_BP, VNAQI_INDEX))

    valid = [v for v in sub_indices if not math.isnan(v)]
    return float(max(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

# Maps unit_system → (system_name, aqi_scale_max)
AQI_SYSTEM_MAP = {
    "us": "EPA",
    "ca": "AQHI",
    "uk": "DAQI",
    "si": "CAQI",
    "hk": "HK_AQHI",
    "ie": "AQIH",
    "il": "IAQI",
    "id": "ISPU",
    "cn": "CHINA_AQI",
    "my": "API",
    "tw": "TAQI",
    "vn": "VN_AQI",
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
        return compute_epa_aqi(
            pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb, so2_ppb, co_ppb
        )
    elif system == "AQHI":
        return compute_aqhi(pm25_ug, o3_ppb, no2_ppb)
    elif system == "DAQI":
        return compute_daqi(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb)
    elif system == "HK_AQHI":
        return compute_hk_aqhi(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb)
    elif system == "AQIH":
        return compute_aqih(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb)
    elif system == "ISPU":
        return compute_ispu(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb, co_ppb)
    elif system == "API":
        return compute_api(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb, co_ppb)
    elif system == "IAQI":
        return compute_iaqi(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb, co_ppb)
    elif system == "CHINA_AQI":
        return compute_china_aqi(
            pm25_ug,
            pm10_ug,
            o3_ppb,
            o3_ppb,
            no2_ppb,
            no2_ppb,
            so2_ppb,
            so2_ppb,
            co_ppb,
            co_ppb,
        )
    elif system == "TAQI":
        return compute_taiwan_aqi(
            pm25_ug, pm10_ug, o3_ppb, o3_ppb, no2_ppb, so2_ppb, co_ppb
        )
    elif system == "VN_AQI":
        return compute_vn_aqi(
            pm25_ug, pm10_ug, o3_ppb, o3_ppb, no2_ppb, so2_ppb, co_ppb
        )
    else:  # EAQI
        return compute_caqi(pm25_ug, pm10_ug, o3_ppb, no2_ppb, so2_ppb)


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
      - PM2.5, PM10: NowCast (12-hour weighted average)
      - O3:    8-hour and 1-hour rolling mean
      - CO:    8-hour rolling mean
      - NO2, SO2: 1-hour (no additional averaging)

    For EAQI (SI) the raw hourly concentrations are used
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

    # Default to raw values (no averaging)
    pm25_calc = pm25_v
    pm10_calc = pm10_v
    o3_calc = o3_v
    no2_calc = no2_v
    so2_calc = so2_v
    co_calc = co_v

    # Override with system-specific averaging where applicable
    if system == "EPA":
        # Apply EPA-mandated averaging periods before the breakpoint lookup
        pm25_calc = nowcast_pm(pm25_v)
        pm10_calc = nowcast_pm(pm10_v)
        o3_calc = rolling_mean(o3_v, window=8)
        co_calc = rolling_mean(co_v, window=8)
        so2_24h_calc = rolling_mean(so2_v, window=24)
        o3_1h_calc = o3_v
    elif system == "AQHI":
        # AQHI uses 3-hour rolling averages for PM2.5, O3, NO2
        pm25_calc = rolling_mean(pm25_v, window=3)
        o3_calc = rolling_mean(o3_v, window=3)
        no2_calc = rolling_mean(no2_v, window=3)
    elif system == "DAQI" or system == "AQIH":
        # DAQI and AQIH use 24-hour rolling averages for PM2.5 and PM10, 8-hour for O3
        pm25_calc = rolling_mean(pm25_v, window=24)
        pm10_calc = rolling_mean(pm10_v, window=24)
        o3_calc = rolling_mean(o3_v, window=8)
    elif system == "HK_AQHI":
        # Hong Kong AQHI uses 3-hour rolling averages for PM2.5, PM10, O3, NO2, SO2
        pm25_calc = rolling_mean(pm25_v, window=3)
        pm10_calc = rolling_mean(pm10_v, window=3)
        o3_calc = rolling_mean(o3_v, window=3)
        no2_calc = rolling_mean(no2_v, window=3)
        so2_calc = rolling_mean(so2_v, window=3)
    elif system == "IAQI":
        # Israel IAQA uses 24-hour rolling averages for PM2.5 and PM10, 8-hour for O3 and 1-hour for NO2, SO2, CO
        pm25_calc = rolling_mean(pm25_v, window=24)
        pm10_calc = rolling_mean(pm10_v, window=24)
        o3_calc = rolling_mean(o3_v, window=8)
    elif system == "ISPU":
        # Indonesia ISPU uses 24-hour rolling averages
        # PM2.5 is rounded to 1 decimal place, PM10, O3, NO2, SO2, CO are rounded to whole numbers
        pm25_calc = np.round(rolling_mean(pm25_v, window=24), 1)
        pm10_calc = np.round(rolling_mean(pm10_v, window=24), 0)
        o3_calc = np.round(rolling_mean(o3_v, window=24), 0)
        co_calc = np.round(rolling_mean(co_v, window=24), 0)
        so2_calc = np.round(rolling_mean(so2_v, window=24), 0)
        no2_calc = np.round(rolling_mean(no2_v, window=24), 0)
    elif system == "CHINA_AQI":
        # China AQI uses 24-hour rolling averages for PM2.5, PM10, NO2, SO2, CO and 8-hour rolling average for O3
        pm25_calc = rolling_mean(pm25_v, window=24)
        pm10_calc = rolling_mean(pm10_v, window=24)
        o3_8h_calc = rolling_mean(o3_v, window=8)
        no2_24_calc = rolling_mean(no2_v, window=24)
        so2_24_calc = rolling_mean(so2_v, window=24)
        co_24_calc = rolling_mean(co_v, window=24)
    elif system == "API":
        # Malayia rounds everything to nearest whole number
        pm25_calc = np.round(pm25_v, 0)
        pm10_calc = np.round(pm10_v, 0)
        o3_calc = np.round(o3_v, 0)
        no2_calc = np.round(no2_v, 0)
        so2_calc = np.round(so2_v, 0)
        co_calc = np.round(co_v, 0)
    elif system == "TAQI":
        # Taiwan AQI uses 24-hour rolling averages for PM2.5, PM10, 8-hour for O3, and 1-hour for NO2, SO2, CO
        # For PM2.5, round to 1 decimal place; for PM10, O3, NO2, SO2, CO round to whole numbers
        pm25_calc = np.round(rolling_mean(pm25_v, window=24), 1)
        pm10_calc = np.round(rolling_mean(pm10_v, window=24), 0)
        o3_8h_calc = np.round(rolling_mean(o3_v, window=8), 0)
        o3_calc = np.round(rolling_mean(o3_v, window=1), 0)
        no2_calc = np.round(rolling_mean(no2_v, window=24), 0)
        so2_calc = np.round(rolling_mean(so2_v, window=24), 0)
        co_calc = np.round(rolling_mean(co_v, window=24), 0)
    elif system == "VN_AQI":
        # Vietnam AQI uses nowcast for PM2.5, PM10, 8-hour for O3, and 1-hour for NO2, SO2, CO
        # For all pollutants, round to whole numbers
        pm25_calc = np.round(nowcast_pm(pm25_v), 0)
        pm10_calc = np.round(nowcast_pm(pm10_v), 0)
        o3_8h_calc = np.round(rolling_mean(o3_v, window=8), 0)
        o3_calc = np.round(o3_v, 0)
        no2_calc = np.round(no2_v, 0)
        so2_calc = np.round(so2_v, 0)
        co_calc = np.round(co_v, 0)

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
                so2_24h_ppb=float(so2_24h_calc[i]),
                co_ppb=float(co_calc[i]),
            )
        elif system == "CHINA_AQI":
            result[i] = compute_china_aqi(
                pm25_ug=float(pm25_calc[i]),
                pm10_ug=float(pm10_calc[i]),
                o3_8h_ppb=float(o3_8h_calc[i]),
                o3_1h_ppb=float(o3_calc[i]),
                no2_24h_ppb=float(no2_24_calc[i]),
                so2_24h_ppb=float(so2_24_calc[i]),
                co_24h_ppb=float(co_24_calc[i]),
                no2_1h_ppb=float(no2_calc[i]),
                so2_1h_ppb=float(so2_calc[i]),
                co_1h_ppb=float(co_calc[i]),
            )
        elif system == "VN_AQI":
            result[i] = compute_vn_aqi(
                pm25_ug=float(pm25_calc[i]),
                pm10_ug=float(pm10_calc[i]),
                o3_8h_ppb=float(o3_8h_calc[i]),
                o3_1h_ppb=float(o3_calc[i]),
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
