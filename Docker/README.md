# Docker Requirements Notes

## NumPy / Numba Compatibility

The ingest environment pins `numpy==2.5.3` with `numba==0.67.0`. Numba 0.67.0
is the first release in this stack with NumPy 2.5 support.

Implemented by: [numba/numba#10645](https://github.com/numba/numba/pull/10645)
Related pandas issue: [pandas-dev/pandas#66083](https://github.com/pandas-dev/pandas/issues/66083)
