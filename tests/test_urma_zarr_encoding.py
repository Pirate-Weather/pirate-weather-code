"""Exercise URMA's hourly store write without running the GRIB download script."""

import ast
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import zarr

URMA_SCRIPT = Path(__file__).resolve().parents[1] / "API" / "URMA_Local_Ingest.py"


def test_urma_hourly_zarr_write_preserves_data_and_compression(tmp_path):
    tree = ast.parse(URMA_SCRIPT.read_text(encoding="utf-8"))
    vars_in = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "zarr_vars"
            for target in node.targets
        )
    )
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ds_out"
        and node.func.attr == "to_zarr"
    ]
    assert len(writes) == 1

    source_data = {}
    for index, var in enumerate(vars_in):
        data = np.arange(24, dtype=np.float64).reshape(1, 4, 6) + index
        if var == "time":
            data[:] = 1791979200
        else:
            data[0, 1, 1] = np.nan
        source_data[var] = data

    ds_out = xr.Dataset(
        {
            var: (("step", "y", "x"), da.from_array(data, chunks=(1, 2, 3)))
            for var, data in source_data.items()
        }
    )
    store_path = tmp_path / "URMA_Hist.zarr"
    store = zarr.storage.LocalStore(store_path)
    try:
        # Execute the production write itself; importing the script would start ingest.
        write = compile(ast.Expression(body=writes[0]), str(URMA_SCRIPT), "eval")
        eval(
            write,
            {"ds_out": ds_out, "zarr_store": store, "vars_in": vars_in, "zarr": zarr},
        )
    finally:
        store.close()

    group = zarr.open_group(store_path, mode="r")
    assert group.metadata.zarr_format == 3
    assert set(group.array_keys()) == set(vars_in)
    for var, data in source_data.items():
        array = group[var]
        assert array.dtype == np.dtype("float32")
        assert array.chunks == (1, 2, 3)
        assert len(array.compressors) == 1
        compressor = array.compressors[0]
        assert isinstance(compressor, zarr.codecs.BloscCodec)
        assert compressor.cname == "zstd"
        assert compressor.clevel == 3
        expected = data.astype("float32")
        np.testing.assert_array_equal(array[:], expected)
        # The historical merge reads each named component with Dask.
        np.testing.assert_array_equal(
            da.from_zarr(store_path, component=var).compute(), expected
        )
