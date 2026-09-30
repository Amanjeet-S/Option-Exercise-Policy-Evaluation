# Development and reproduction

For the paper's public material, start with the [reproduction instructions](reproducing_paper.md). The commands below cover development and the other projects as well.

## Installation

Clone the repository and run these commands from its root. Python 3.13 is the reference version. The existing numerical validation record was produced with Python 3.13.9 on macOS; continuous integration checks the supported environment separately.

```bash
git clone https://github.com/Amanjeet-S/Quantitative-Finance.git
cd Quantitative-Finance
python3 -m venv .venv
source .venv/bin/activate
python projects/01_optimal_stopping/replication/verify_sources.py
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/check_repository.py
python scripts/audit_germany_data.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Keep the checkout in a local folder. The [numerical reproduction guide](../projects/01_optimal_stopping/replication/README.md) describes the independent public example and its input requirements.

## Notebooks

```bash
python -m pip install -r requirements-notebooks.lock
python scripts/execute_notebooks.py
```

The runner executes every project notebook in the published checkout in a new kernel and checks for cell errors. Without `--write`, it leaves the saved notebooks unchanged. Use `--write` when intentionally updating their outputs, then inspect the diff. Use the same environment's interpreter in a notebook editor.

The published notebooks cover baseline numerical validation, conditional research replication and Germany data coverage. Their existence does not imply that the empirical projects are complete.

## Numerical benchmark

The commands are in the [numerical reproduction guide](../projects/01_optimal_stopping/replication/README.md); interpretations are in the linked research notes. Write a new output directory when preserving an earlier run. Timings and floating-point rounding can differ across machines.

## Continuous integration

GitHub Actions verifies the published implementation inventory, installs the frozen core and notebook dependencies, runs pytest, checks tracked Markdown file links and retained data hashes, reconstructs the CPU timing summaries, audits the saved Germany data, and executes the notebooks. These checks require no provider credentials or live data downloads. Actions are pinned to commit identifiers. Inventory verification establishes consistency with its listed hashes rather than authenticity or completeness of the release.

The optional [C++ propagation component](../projects/01_optimal_stopping/cpp/README.md) requires a C++17 compiler. Tests build it in a temporary directory; native tests explicitly skip if no compiler is installed. Continuous integration requires a compiler before running the suite. The retained performance study was run on macOS, and does not establish performance on the CI host. R work remains planned.
