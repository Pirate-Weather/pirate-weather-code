"""Tests for NOAA AQM (Air Quality Model) ingest script.

Tests cover:
- Script existence checks
- O3 unit conversion helpers (convert_o3_to_ug_m3)
- URL building (build_aqm_url)
- Run-time discovery (get_latest_aqm_run)
- AQI calculation with only PM2.5 and O3 (as provided by AQM)
"""

import ast
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from API.NOAA_AQM_Local_Ingest import (
    AQM_NOMADS_BASE,
    AQM_RUN_HOURS,
    O3_PPB_TO_UG_M3,
    O3_PPM_TO_UG_M3,
    build_aqm_url,
    convert_o3_to_ug_m3,
    get_latest_aqm_run,
)

from API.ingest_utils import calculate_aqi

AQM_SCRIPT = Path(__file__).resolve().parents[1] / "API" / "NOAA_AQM_Local_Ingest.py"


# ---------------------------------------------------------------------------
# Existence & syntax tests
# ---------------------------------------------------------------------------


class TestAQMScriptExists:
    """Basic script presence and syntax validation."""

    def test_script_exists(self):
        """Verify that NOAA_AQM_Local_Ingest.py has been created."""
        assert AQM_SCRIPT.exists(), f"Script not found: {AQM_SCRIPT}"

    def test_script_is_valid_python(self):
        """Verify the script is syntactically valid Python."""
        source = AQM_SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert tree is not None

    def test_script_imports_calculate_aqi(self):
        """Verify the script imports calculate_aqi from ingest_utils."""
        source = AQM_SCRIPT.read_text(encoding="utf-8")
        assert "calculate_aqi" in source

    def test_script_references_nomads_url(self):
        """Verify the script references the NOMADS base URL."""
        expected_base = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aqm/prod"
        assert AQM_NOMADS_BASE == expected_base

    def test_script_handles_pm25_and_o3(self):
        """Verify the script processes both PM2.5 and O3 variables."""
        source = AQM_SCRIPT.read_text(encoding="utf-8")
        assert "pm25" in source.lower()
        assert "o3" in source.lower()

    def test_script_has_main_guard(self):
        """Verify the script uses an if __name__ == '__main__' guard."""
        source = AQM_SCRIPT.read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in source or "__name__ == '__main__'" in source


# ---------------------------------------------------------------------------
# O3 unit conversion tests
# ---------------------------------------------------------------------------


class TestO3UnitConversion:
    """Tests for ozone concentration unit conversions."""

    def test_ppm_to_ug_m3_typical_value(self):
        """1 ppm O3 should convert to ~1962 µg/m³ at 25°C."""
        result = 1.0 * O3_PPM_TO_UG_M3
        np.testing.assert_allclose(result, 1962.0, rtol=0.01)

    def test_ppb_to_ug_m3_typical_value(self):
        """1 ppb O3 should convert to ~1.962 µg/m³ at 25°C."""
        result = 1.0 * O3_PPB_TO_UG_M3
        np.testing.assert_allclose(result, 1.962, rtol=0.01)

    def test_ppm_ppb_relationship(self):
        """1 ppm should equal 1000 ppb after conversion."""
        ppm_result = 1.0 * O3_PPM_TO_UG_M3
        ppb_result = 1000.0 * O3_PPB_TO_UG_M3
        np.testing.assert_allclose(ppm_result, ppb_result, rtol=1e-6)

    def test_zero_concentration_stays_zero(self):
        """Zero O3 should remain zero after conversion."""
        assert 0.0 * O3_PPM_TO_UG_M3 == 0.0
        assert 0.0 * O3_PPB_TO_UG_M3 == 0.0

    def test_typical_clean_air_o3(self):
        """Typical clean-air O3 (~40 ppb) should convert to ~79 µg/m³."""
        result = 40.0 * O3_PPB_TO_UG_M3  # 40 ppb
        np.testing.assert_allclose(result, 78.5, atol=2.0)

    def test_aqi_breakpoint_first_o3_threshold(self):
        """54 ppb O3 (AQI 50 boundary) should convert to ~106 µg/m³.

        The first AQI breakpoint in aqi_const.py is 108 µg/m³.
        0.054 ppm × 1962 µg/m³ per ppm ≈ 106 µg/m³, consistent with that value.
        """
        result = 54.0 * O3_PPB_TO_UG_M3  # 0.054 ppm = 54 ppb
        np.testing.assert_allclose(result, 106.0, atol=3.0)

    def test_convert_o3_ppm_units(self):
        """convert_o3_to_ug_m3 should multiply by O3_PPM_TO_UG_M3 for ppm input."""
        import xarray as xr

        da = xr.DataArray(np.array([1.0], dtype=np.float32))
        result = convert_o3_to_ug_m3(da, "ppm")
        np.testing.assert_allclose(result.values[0], O3_PPM_TO_UG_M3, rtol=1e-5)

    def test_convert_o3_ppb_units(self):
        """convert_o3_to_ug_m3 should multiply by O3_PPB_TO_UG_M3 for ppb input."""
        import xarray as xr

        da = xr.DataArray(np.array([1.0], dtype=np.float32))
        result = convert_o3_to_ug_m3(da, "ppb")
        np.testing.assert_allclose(result.values[0], O3_PPB_TO_UG_M3, rtol=1e-5)

    def test_convert_o3_actual_grib2_units_string(self):
        """convert_o3_to_ug_m3 handles the real GRIB2 unit string 'Ozone Concentration [ppb]'."""
        import xarray as xr

        da = xr.DataArray(np.array([40.0], dtype=np.float32))
        result = convert_o3_to_ug_m3(da, "Ozone Concentration [ppb]")
        np.testing.assert_allclose(result.values[0], 40.0 * O3_PPB_TO_UG_M3, rtol=1e-5)

    def test_convert_o3_ug_m3_units_no_change(self):
        """convert_o3_to_ug_m3 should not change values already in µg/m³."""
        import xarray as xr

        da = xr.DataArray(np.array([100.0], dtype=np.float32))
        result = convert_o3_to_ug_m3(da, "ug/m3")
        np.testing.assert_allclose(result.values[0], 100.0, rtol=1e-5)

    def test_convert_o3_unknown_units_converts_as_ppb(self):
        """convert_o3_to_ug_m3 should assume ppb and convert for unknown units."""
        import xarray as xr

        da = xr.DataArray(np.array([50.0], dtype=np.float32))
        result = convert_o3_to_ug_m3(da, "some_unknown_unit")
        np.testing.assert_allclose(result.values[0], 50.0 * O3_PPB_TO_UG_M3, rtol=1e-5)


# ---------------------------------------------------------------------------
# Run-time discovery and URL building tests
# ---------------------------------------------------------------------------


class TestAQMRunTimeDiscovery:
    """Tests for get_latest_aqm_run() and build_aqm_url() logic."""

    def test_get_latest_aqm_run_returns_datetime(self):
        """get_latest_aqm_run should always return a datetime."""
        result = get_latest_aqm_run()
        assert isinstance(result, datetime)

    def test_run_hour_is_06_or_12(self):
        """The run time returned should have hour 6 or 12."""
        result = get_latest_aqm_run()
        assert result.hour in AQM_RUN_HOURS

    def test_run_time_in_past(self):
        """The returned run time should not be in the future."""
        result = get_latest_aqm_run()
        now = datetime.now(timezone.utc)
        # run_time may be tz-naive (from replace()) – compare without tz
        run_naive = result.replace(tzinfo=None) if result.tzinfo else result
        now_naive = now.replace(tzinfo=None)
        assert run_naive <= now_naive

    def test_build_aqm_url_pm25(self):
        """build_aqm_url should produce the correct NOMADS URL for PM2.5."""
        run_time = datetime(2026, 1, 15, 12, 0, 0)
        url = build_aqm_url(run_time, "pm25", "148")
        assert "aqm.20260115" in url
        assert "t12z" in url
        assert "ave_1hr_pm25" in url
        assert "148" in url
        assert url.startswith(AQM_NOMADS_BASE)

    def test_build_aqm_url_o3(self):
        """build_aqm_url should produce the correct NOMADS URL for O3."""
        run_time = datetime(2026, 3, 1, 6, 0, 0)
        url = build_aqm_url(run_time, "o3", "227")
        assert "aqm.20260301" in url
        assert "t06z" in url
        assert "ave_1hr_o3" in url
        assert "227" in url

    def test_build_aqm_url_different_grids(self):
        """build_aqm_url should produce different URLs for different grid IDs."""
        run_time = datetime(2026, 1, 1, 12, 0, 0)
        url_148 = build_aqm_url(run_time, "pm25", "148")
        url_227 = build_aqm_url(run_time, "pm25", "227")
        assert url_148 != url_227
        assert "148" in url_148
        assert "227" in url_227


# ---------------------------------------------------------------------------
# AQI calculation with only PM2.5 and O3 (as NOAA AQM provides)
# ---------------------------------------------------------------------------


class TestAQMAQICalculation:
    """Tests for AQI calculation using only PM2.5 and O3 (NOAA AQM pollutants)."""

    def _nan_data(self, shape):
        return np.full(shape, np.nan, dtype=np.float32)

    def test_aqi_pm25_only(self):
        """AQI with only PM2.5 should reflect PM2.5 value correctly."""
        shape = (12, 3, 3)
        pm25 = np.full(shape, 35.4, dtype=np.float32)  # ~AQI 100 boundary
        aqi = calculate_aqi(
            pm25,
            self._nan_data(shape),  # PM10
            self._nan_data(shape),  # O3
            self._nan_data(shape),  # NO2
            self._nan_data(shape),  # SO2
            self._nan_data(shape),  # CO
            use_nowcast=False,
        )
        # At PM2.5 = 35.4 µg/m³ (24-h avg), AQI should be close to 100
        assert np.all(aqi[-1] >= 90)
        assert np.all(aqi[-1] <= 110)

    def test_aqi_o3_only(self):
        """AQI with only O3 should reflect O3 value correctly."""
        shape = (12, 3, 3)
        # O3 at ~135 µg/m³ → between breakpoints 108 (AQI 50) and 140 (AQI 100)
        o3 = np.full(shape, 135.0, dtype=np.float32)
        aqi = calculate_aqi(
            self._nan_data(shape),  # PM2.5
            self._nan_data(shape),  # PM10
            o3,
            self._nan_data(shape),  # NO2
            self._nan_data(shape),  # SO2
            self._nan_data(shape),  # CO
            use_nowcast=False,
        )
        # Should be in the moderate range (51–100)
        assert np.all(aqi[-1] > 50)
        assert np.all(aqi[-1] <= 100)

    def test_aqi_pm25_and_o3_dominated_by_pm25(self):
        """When PM2.5 AQI >> O3 AQI, result should be dominated by PM2.5."""
        shape = (12, 3, 3)
        pm25 = np.full(shape, 150.0, dtype=np.float32)  # Very high PM2.5 (~AQI 200)
        o3 = np.full(shape, 110.0, dtype=np.float32)  # Moderate O3 (~AQI 51)
        aqi = calculate_aqi(
            pm25,
            self._nan_data(shape),
            o3,
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            use_nowcast=False,
        )
        # AQI should be in the Unhealthy Sensitive range (151–200) due to PM2.5
        assert np.all(aqi[-1] >= 150)

    def test_aqi_pm25_and_o3_dominated_by_o3(self):
        """When O3 AQI >> PM2.5 AQI, result should be dominated by O3."""
        shape = (12, 3, 3)
        pm25 = np.full(shape, 5.0, dtype=np.float32)  # Very low PM2.5 (~AQI 21)
        # O3 at ~210 µg/m³ → AQI ~200
        o3 = np.full(shape, 210.0, dtype=np.float32)
        aqi = calculate_aqi(
            pm25,
            self._nan_data(shape),
            o3,
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            use_nowcast=False,
        )
        assert np.all(aqi[-1] >= 100)

    def test_aqi_clean_air(self):
        """Clean air (low PM2.5, low O3) should give low AQI."""
        shape = (12, 3, 3)
        pm25 = np.full(shape, 5.0, dtype=np.float32)  # Clean air
        o3 = np.full(shape, 40.0, dtype=np.float32)  # Clean air
        aqi = calculate_aqi(
            pm25,
            self._nan_data(shape),
            o3,
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            use_nowcast=False,
        )
        # AQI should be in Good range (0–50)
        assert np.all(aqi[-1] >= 0)
        assert np.all(aqi[-1] <= 50)

    def test_aqi_spatial_variation(self):
        """AQI should vary spatially when inputs vary spatially."""
        shape = (12, 5, 5)
        pm25 = np.zeros(shape, dtype=np.float32)
        for i in range(5):
            pm25[:, i, :] = float(i) * 15.0 + 5.0  # 5, 20, 35, 50, 65 µg/m³

        aqi = calculate_aqi(
            pm25,
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            use_nowcast=False,
        )
        # AQI should increase from row 0 to row 4
        assert aqi[-1, 0, 0] < aqi[-1, 4, 0]

    def test_aqi_all_nan_returns_nan(self):
        """AQI with all NaN inputs should return NaN (nanmax of all-NaN)."""
        shape = (12, 3, 3)
        aqi = calculate_aqi(
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            use_nowcast=False,
        )
        # np.nanmax of all-NaN slice returns nan
        assert np.all(np.isnan(aqi[-1]))

    def test_aqi_with_nowcast_pm25(self):
        """AQI with NowCast enabled for PM2.5 should work correctly."""
        shape = (12, 3, 3)
        pm25 = np.full(shape, 35.0, dtype=np.float32)
        aqi = calculate_aqi(
            pm25,
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            self._nan_data(shape),
            use_nowcast=True,
        )
        # NowCast with steady concentration should give same result as 24h avg
        assert np.all(aqi[-1] >= 0)
        assert np.all(~np.isnan(aqi[-1]))
