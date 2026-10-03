# C++ propagation kernel

## 1. Numerical scope

I implement the driftless propagation step in [funded_step.cpp](funded_step.cpp), with a readable Python/SciPy reference in [funded_cash.py](../src/qf_research/optimal_stopping/funded_cash.py). The kernel is one numerical component of the funded stopping model. It does not fit exercise policies, ingest financial observations or establish an empirical pricing model.

The [cross-language checks](../tests/test_funded_kernel.py) compare operator construction and propagated values, including limiting cases and invalid inputs. The standalone [C++ checks](test_funded_step.cpp) cover the native interface. Numerical agreement establishes parity within the tested cases, not a certified global error bound.

## 2. Build and verification

The build interface supports macOS and Linux with a C++17 compiler. From the repository root:

```sh
python scripts/build_funded_kernel.py --output data/private/cpp_build/local
python -m pytest tests/test_funded_kernel.py
```

Use a new output directory for each retained build. The build record includes the compiler, flags, source and library hashes, platform and compilation time. Fast-math transformations and floating-point contraction are disabled in the declared build.

## 3. Computational interpretation

The [benchmark entry point](../scripts/benchmark_funded_kernel.py) compares the kernels and records preparation, propagation and memory measurements on stipulated mathematical fixtures. Measurements are specific to the recorded machine and workload. A faster stepping kernel does not by itself imply a faster complete policy experiment. The [replication guide](../replication/README.md) explains the complete computational study and its separate CPU forecasting question.
