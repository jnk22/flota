# FLOTA

This branch is intended for generating test input and output files based on the
original [FLOTA](https://github.com/valentinhofmann/flota) implementation.

A subset of the generated files are used for regression tests in the
[main](https://github.com/jnk22/flota) branch.

## Available configurations

- `2_limited`: Based on small datasets (02), used for regression tests.
- `2_full`: Based on small datasets (02), including all texts.
- `3_full`: Based on full datasets (03), including all texts.

## Generate test files

```bash
./run_generate_test_data_2_limited.sh
./run_generate_test_data_2_full.sh
./run_generate_test_data_3_full.sh
```
