# Optimal stopping and option pricing

**Stage 1 in progress: mathematical foundations and data feasibility.** This is an unfinished research project. The current code is a prototype for validating the methods that a later study of observed option markets will require.

The investigation asks how much value is lost through approximate exercise policies, how reliably that loss can be measured, and when greater computational accuracy matters financially. The intended study combines model calibration using observed data with numerical analysis, independent policy evaluation and economic interpretation.

## Read first

1. [Research design and stage completion criteria](research_design.md).
2. [Observed-data plan and source requirements](data_plan.md), [option-quote pilot request](option_quote_pilot_request.md) and [Cboe sample format audit](cboe_sample_audit.md).
3. [Model, derivations and numerical decisions](theory.md).
4. [Preliminary numerical validation record](results/baseline/findings.md).
5. [Sources and attribution](SOURCES.md).
6. [Executed numerical-validation notebook](notebooks/01_numerical_validation.ipynb).

The validation record uses analytically specified test inputs and simulated GBM paths. It checks mathematical and computational behaviour. It contains no observed option prices and supports no empirical conclusion about actual markets.

## Implemented so far

| Component | Purpose | Current limitation |
| --- | --- | --- |
| European prices and Greeks | Analytical reference and sensitivity checks | Constant coefficients and continuous dividend yield |
| Exact GBM simulation and Monte Carlo | Risk-neutral expectations and sampling diagnostics | One diffusion model; simulated paths are not market observations |
| CRR tree | Independent backward-induction reference | Grid-aligned exercise dates and lattice error |
| Finite-difference solver | PDE reference with exercise projection and smoothing | Uniform spot grid, finite domain and preliminary stress coverage |
| Least-squares exercise policy | Backward training and independent forward evaluation | Polynomial approximation; no certified optimality gap |
| Treasury data ingestion | Preserve official observations, source bytes and provenance | Par yields have not been converted to a pricing curve |

The package implements European and finite-calendar Bermudan contracts. Maturity is always an exercise date; time zero is excluded. A tree or PDE calculation rejects misaligned exercise dates. A listed American option will require explicit exercise refinement and its actual dividend and settlement conventions.

## Set up the Python environment

Run these commands from the repository root. The frozen environment has been checked with Python 3.13.9 on macOS. Other Python and operating-system combinations still need verification. A virtual environment outside the repository avoids cloud-sync interference with Python's import-path files.

```bash
python3 -m venv "$HOME/.venvs/qf-research"
source "$HOME/.venvs/qf-research/bin/activate"
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
python -m pytest -q
```

The dependency lock includes the test tools. It freezes versions for this validation environment; it is not a claim that these are the latest versions of every package.

### Optional notebook tools

The notebook environment supports the included numerical-validation notebook and future research notebooks. Install its frozen dependencies into the same activated environment:

```bash
python -m pip install -r requirements-notebooks.lock
```

The `notebooks` package extra also declares the direct notebook dependencies. The separate notebook lock includes their transitive dependencies and retains the core numerical versions. A notebook editor can use this environment's Python interpreter. Run `python scripts/execute_notebooks.py` from the repository root to verify all project notebooks in fresh kernels.

## Reproduce the numerical checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m qf_research.optimal_stopping.experiments \
  --config projects/01_optimal_stopping/configs/benchmark.json \
  --output projects/01_optimal_stopping/results/baseline
```

This command regenerates the specified results directory. Choose a new directory to retain a separate run. It writes grid comparisons, repeated Monte Carlo checks, independent policy evaluations, conditioning diagnostics, figures and a manifest with seeds, versions, hardware and source hashes. Runtime measurements are machine dependent.

Policies use separate training and evaluation random streams. Within each repetition, configurations share evaluation paths, so comparisons are dependent. Reported policy intervals quantify evaluation noise conditional on the fitted policy. They do not cover approximation error or training variability.

## Retrieve observed public rates

The saved [Treasury snapshot](../../data/public/treasury/2026-09-14/metadata.json) records the exact source and latest available observation. To create a fresh snapshot, supply a new output directory:

```bash
python -m qf_research.market_data.treasury \
  --as-of 2026-09-14 \
  --output data/public/treasury/new-review-snapshot
```

Set the cutoff deliberately for a later run. An existing snapshot directory is rejected to preserve saved evidence. The command retains missing values and never creates substitute observations. Filtering by historical date does not reconstruct a historical publication vintage.

The next data requirement is a recent option-quote pilot from a dedicated historical source. Capital IQ Pro is retained for underlying prices, dividends and corporate actions. A Cboe demonstration archive has passed basic structural checks, but its 2023 date and demonstration status make it unsuitable as the current empirical panel. [The data plan](data_plan.md) lists the remaining checks before calibration and historical evaluation can begin.

## Development standard

The remaining work includes a data audit, numerical stress tests, observed-contract modelling, calibration, credible policy-loss measurement, a verified C++ kernel and a full research paper. Each stage has a completion criterion. The number of implemented methods is not the measure of completion.

The [research log](research_log.md) records progress, revisions and unresolved questions. Established models and algorithms are credited in the source register; the project must earn its research conclusions through reproducible evidence.
