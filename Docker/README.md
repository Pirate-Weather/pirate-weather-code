# Docker Requirements Notes

## NumPy / Numba Compatibility

The ingest environment pins `numpy==2.3.5` because `numba==0.65.1` does not
support NumPy 2.5 yet. Once Numba support lands, update the ingest stack to
NumPy 2.5+ and the matching Numba release.

Tracking PR: [numba/numba#10645](https://github.com/numba/numba/pull/10645)
Related pandas issue: [pandas-dev/pandas#66083](https://github.com/pandas-dev/pandas/issues/66083)
