# Development and reproduction

## Installation

Clone the repository and run these commands from its root. Python 3.13 is the reference version. The existing numerical validation record was produced with Python 3.13.9 on macOS; continuous integration checks the supported environment separately.

```bash
git clone https://github.com/Amanjeet-S/Quantitative-Finance.git
cd Quantitative-Finance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/check_repository.py
python scripts/audit_germany_data.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. For cloud-synchronised folders, place the virtual environment outside the checkout, as described in the [Project 1 overview](../projects/01_optimal_stopping/README.md).

## Notebooks

```bash
python -m pip install -r requirements-notebooks.lock
python scripts/execute_notebooks.py
```

The runner executes every tracked project notebook in a new kernel and checks for cell errors. Without `--write`, it leaves the saved notebooks unchanged. Use `--write` when intentionally updating their outputs, then inspect the diff. Use the same environment's interpreter in a notebook editor.

The current notebooks are numerical-validation and data-coverage entry points. Their existence does not imply that the empirical projects are complete.

## Numerical benchmark

The full command and interpretation are in the [Project 1 overview](../projects/01_optimal_stopping/README.md). Write a new output directory when preserving an earlier run. Timings and floating-point rounding can differ across machines.

## Continuous integration

GitHub Actions installs the frozen core and notebook dependencies, runs pytest, checks tracked Markdown file links and source hashes, audits the saved Germany data, and executes the notebooks. These checks require no provider credentials or live data downloads. Actions are pinned to commit identifiers.

C++ and R directories describe planned work. They do not contain completed implementations or require a compiler or R installation at this stage.
