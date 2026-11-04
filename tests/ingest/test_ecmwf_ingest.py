"""ECMWF IFS 0.1 degree ingest transforms that can corrupt a forecast silently."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ingest import ECMWF_Local_Ingest as ingest


def _ensemble_file(step: float, totals_m: list[float]) -> xr.Dataset:
    """A single-lead ensemble file as cfgrib opens it: one 2 x 2 total per member."""
    return xr.Dataset(
        {
            "tp": (
                ("number", "latitude", "longitude"),
                np.array(totals_m, dtype=np.float32)[:, None, None]
                * np.ones((len(totals_m), 2, 2), dtype=np.float32),
            )
        },
        coords={
            "number": range(1, len(totals_m) + 1),
            "latitude": [90.0, 89.9],
            "longitude": [-180.0, -179.9],
            "step": step,
        },
    )


def test_ensemble_precip_rate_uses_each_interval_length(tmp_path, monkeypatch):
    """Leads step from 3-hourly to 6-hourly at 144 h and ``tp`` accumulates from
    the run start. Differencing against the wrong lead, or dividing by the wrong
    interval, doubles or halves precipitation and its probability past day 6.
    """
    files = {
        "subset_a__20261104000000-3h-enfo-ef-sfc.grib2": _ensemble_file(
            3, [0.003, 0.003]
        ),
        "subset_a__20261104000000-144h-enfo-ef-sfc.grib2": _ensemble_file(
            144, [0.003, 0.009]
        ),
        "subset_a__20261104000000-150h-enfo-ef-sfc.grib2": _ensemble_file(
            150, [0.003, 0.021]
        ),
    }
    monkeypatch.setattr(xr, "open_dataset", lambda path, **_: files[path])

    # Paths in the wrong order: the lead, not the list position, must decide
    stats = ingest.write_ifs_ensemble_precip_stats(
        list(reversed(files)), [3, 144, 150], str(tmp_path / "ens.zarr")
    )

    mm_per_h = stats["APCP_Mean"].isel(latitude=0, longitude=0).values * 1000
    # (3 mm / 3 h), (0 and 6 mm over 141 h), (0 and 12 mm over 6 h)
    np.testing.assert_allclose(mm_per_h, [1.0, 3 / 141, 1.0], rtol=1e-5)
    assert stats["Precipitation_Prob"].values[2, 0, 0] == pytest.approx(0.5)


def test_relative_humidity_is_percent_from_stored_kelvin():
    """Final-store temperatures are in Kelvin but the humidity formula takes
    Celsius, and the API reads this column as percent like GFS. Either slip would
    publish a plausible-looking but wrong humidity for the whole grid.
    """
    t2m = np.full((1, 2, 2), 293.15, dtype=np.float32)
    d2m = np.array([[[293.15, 283.15], [np.nan, 283.15]]], dtype=np.float32)
    columns = {name: t2m for name in ingest.PROCESS_VARS}
    columns["d2m"] = d2m
    stacked = da.from_array(
        np.stack([columns[name] for name in ingest.PROCESS_VARS]),
        chunks=(1, 1, 2, 2),
    )

    rh = ingest.append_relative_humidity(stacked).compute()[-1, 0]

    assert rh[0, 0] == pytest.approx(100.0)
    # Saturation vapour pressure at 10 C / 20 C
    assert rh[0, 1] == pytest.approx(52.5, abs=0.5)
    assert np.isnan(rh[1, 0])
