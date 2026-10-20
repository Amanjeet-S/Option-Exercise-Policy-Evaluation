# C++ pricing components

## 1. Native Bermudan tree

I implement the established Cox–Ross–Rubinstein recursion in [bermudan_tree.cpp](bermudan_tree.cpp). The C++17 executable prices calls and puts with constant rate, dividend yield and volatility on uniformly spaced positive exercise rights. A single vector supports backward induction with linear storage and quadratic work. Zero volatility uses deterministic discounted rewards on the same calendar.

The interface checks financial inputs, calendar divisibility, the risk-neutral probability and arithmetic range. It accepts at most 8,192 steps and rejects unresolved or subnormal intermediate values; the resource limit does not guarantee representable arithmetic for every parameter choice. Time zero is excluded from exercise. Returned prices are finite-tree values, with discretisation and rounding error.

The [pricing notebook](../notebooks/pricing_and_stopping.ipynb) builds the source temporarily and compares three same-calendar values with Python. The [verification driver](../scripts/verify_bermudan_tree.py) also uses independent decimal tree and deterministic references, European analytical checks, homogeneity, calendar ordering and invalid-input cases. From the repository root, with a C++17 compiler available:

```sh
python scripts/verify_bermudan_tree.py --output data/private/cpp_build/bermudan_check_01
```

The output directory must be new. The default compiler is `clang++`; `--compiler g++` selects an alternative. Compiler details, source identities, comparisons and failures are retained locally. No installation is attempted and no pricing-speed claim is made.

## 2. Funded propagation kernel

[funded_step.cpp](funded_step.cpp) implements the historical funded model's driftless propagation step, with [funded_cash.py](../src/qf_research/optimal_stopping/funded_cash.py) as its Python/SciPy reference. It is a separate component, rather than the tree used by the current pricing notebook.

The [cross-language checks](../tests/test_funded_kernel.py) compare operators and propagated values, including limiting cases and invalid inputs. The [standalone checks](test_funded_step.cpp) cover the native interface. Build and verify with:

```sh
python scripts/build_funded_kernel.py --output data/private/cpp_build/funded_check_01
python -m pytest tests/test_funded_kernel.py
```

Build records include compiler flags, source and library hashes, platform and compilation time. Fast-math and floating-point contraction are disabled. The [benchmark entry point](../scripts/benchmark_funded_kernel.py) measures this kernel on mathematical fixtures. Kernel parity and measured stepping speed do not establish global accuracy or complete experiment cost.
