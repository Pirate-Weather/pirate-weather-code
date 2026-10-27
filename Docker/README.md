# Docker Requirements Notes

## NumPy / Numba Compatibility

The ingest environment pins `numpy==2.5.3` with `numba==0.67.0`. Numba 0.67.0
is the first release in this stack with NumPy 2.5 support.

Implemented by: [numba/numba#10645](https://github.com/numba/numba/pull/10645)
Related pandas issue: [pandas-dev/pandas#66083](https://github.com/pandas-dev/pandas/issues/66083)

## API image: tensorstore and the store sync

The API reads the sharded final stores with `tensorstore` (pinned at the
benchmarked 0.1.85) and keeps `zarr` for the alert grids (variable-length
strings, which tensorstore cannot read) and ERA5. The same image runs
`python -m API.store_sync`, which needs `obstore`, to copy new store versions
from S3 to the API's local disk; see `API/store_sync.py` for its settings.
