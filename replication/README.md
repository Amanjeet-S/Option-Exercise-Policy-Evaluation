# Replication of the conditional stopping study

## 1. Scope

I provide the numerical kernels, experiment entry points and evidence checks used to study exercise-policy loss, terminal adjustment and computational allocation. The published research summaries concern the retained conditional experiments. The short example below uses stipulated mathematical inputs and demonstrates the implementation. It does not reproduce those summaries or establish agreement with market prices.

I distinguish three operations. A summary audit checks the arithmetic and presentation of released findings. A numerical replay recomputes a retained experiment using its original inputs, random streams and source versions. A fresh experiment uses a new specification or new streams. CPU timings are new observations on the executing machine even when the financial calculation is an exact replay.

## 2. Installation and the public example

The [reader's reproduction instructions](../../../docs/reproducing_paper.md) provide a single command for the public paper checks, numerical fixture and two notebooks. That route uses published inputs only and does not read or export restricted raw financial data. The [CPU measurement supplement](../paper/cpu_timings.md) reconstructs Table 9 from released computational timings; its historical forecast values are retained inputs, not newly fitted forecasts.

For the numerical fixture alone, create an environment and install the package from the repository root. Python 3.11 or later is required; continuous integration checks Python 3.11 and 3.13.

```sh
python3 -m venv .venv
.venv/bin/python projects/01_optimal_stopping/replication/verify_sources.py
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python projects/01_optimal_stopping/replication/run.py \
  --output data/private/replication/fixture_run.json
```

The input [stipulated_fixture.json](configs/stipulated_fixture.json) specifies a put with unit maturity, one capped cash event, explicit settlement factors and a finite exercise calendar. The values are stipulated for a mathematical example. They are neither observations nor estimates from a financial instrument. The runner fits independent policies with 512 and 2,048 training paths, evaluates each pair on common independent samples of 256 and 1,024 paths, and repeats the comparison four times. A separate sample evaluates the conditional terminal adjustment.

The result contains paired policy-gain estimates, non-negative pathwise dual gaps, the signed training-variance moment estimate, conditional and realised terminal gains, and measured setup and pair costs. The common upper payoff makes the difference in paired mean gaps equal to the policy-gain estimate. The allocation calculation uses the measured costs and estimated variance components as a pilot diagnostic. It does not execute the selected counts or establish that the small example estimates an optimal allocation reliably. A negative component estimate remains visible and produces the selector's unresolved status.

The numerical checksum excludes timing, environment and provenance metadata. Exact equality is meaningful under the same numerical implementation and environment; different library versions or linear-algebra backends can change floating-point results. Timings and any allocation computed from them can change on repeated execution. The pair cost counts a full training and evaluation workload for each evaluation size. The actual runner shares its trained policies across sizes, so summing those pair charges would not measure its total process time. Terminal diagnostics are outside that cost comparison.

The [executed notebook](../notebooks/27_conditional_research_replication.ipynb) explains the public example alongside exact finite-law probability identities and the released-summary audit. The [source manifest](source_manifest.json) identifies the reviewed files, their byte counts and SHA-256 hashes. The [standalone verifier](verify_sources.py) checks every declared file with the Python standard library, before scientific dependencies are needed. It rejects malformed entries, duplicate or escaping paths, symbolic links and altered or missing files. This checks consistency with the supplied inventory; it does not authenticate that inventory, prove its completeness or verify a financial input.

The experiment runner checks the inventory and the origins of loaded project modules before numerical work and again before returning a result. Each result records the canonical input-specification digest, manifest digest and checked module origins separately from the numerical checksum. The module check prevents an unnoticed import from another installed copy of the project. These checks do not certify third-party binaries, hardware or the interpreter's in-memory state. Use a fresh kernel or interpreter for a reproducible run. An intentional source change requires its own reviewed inventory; the runner does not silently rewrite recorded hashes.

## 3. Inputs for a new authorised experiment

The [input schema](input.schema.json) describes the structural interface. The executable validator also checks chronological exercise order, increasing sample sizes, terminal coverage, cash-event phases, finite numerical parameters and compatibility of the terminal extension. The runner currently supports puts and requires the model's discounted ownership state, settlement factors and cash-event caps. It does not estimate these inputs or verify the funding construction behind them.

For a new study, copy the public specification into `data/private`, change its classification to `authorised_model`, and supply the financial model, contract, decision calendar and fixed simulation design from documented sources. The fixture classification is reserved for the exact published example. Preserve the source observations and their provenance separately, including the observation date, information available at the valuation cutoff, curve construction, settlement conventions and dividend treatment. Neither a licence assertion nor a successful structural check establishes empirical identification.

```sh
.venv/bin/python projects/01_optimal_stopping/replication/run.py \
  --config data/private/my_study/specification.json \
  --validate-only --output data/private/my_study/input_check.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python projects/01_optimal_stopping/replication/run.py \
  --config data/private/my_study/specification.json \
  --output data/private/my_study/fresh_experiment.json
```

The output path must be inside `data/private` and must not already exist. The command neither contacts a data service nor exports the input bundle. Any proposed publication of the results requires the repository's content review. A new specification represents a new conditional experiment and cannot be substituted for the paper's original inputs while retaining a claim of exact historical reproduction.

## 4. Implementation and historical replay

| Research operation | Principal implementation | Retained-evidence entry points |
| --- | --- | --- |
| Cash events, exercise and funding | `src/qf_research/optimal_stopping/funded_cash.py` | `scripts/validate_funded_cash.py`, `scripts/refine_funded_calendar.py` |
| Independent policy training and evaluation | `funded_policy.py`, `funded_dual.py` | `scripts/validate_funded_policy.py`, `scripts/validate_funded_dual.py` |
| Analytically centred upper construction | `value_martingale.py` | `scripts/study_value_martingale.py`, `scripts/audit_value_martingale.py` |
| Conditional terminal adjustment | `expiry_extension.py`, `terminal_repair.py`, `terminal_training.py` | `scripts/study_terminal_repair.py`, `scripts/study_terminal_training.py` |
| Training and evaluation uncertainty | `training_evaluation.py`, `allocation_workload.py` | `scripts/study_training_evaluation.py`, `scripts/reproduce_training_evaluation.py` |
| Discrete computational allocation | `training_allocation.py`, `allocation_sensitivity.py`, `allocation_variance_sensitivity.py` | `scripts/study_training_allocation.py`, `scripts/reproduce_training_allocation.py` |
| Variance-component uncertainty | `component_confidence.py`, `component_ratio.py` | `scripts/analyse_component_confidence.py`, `scripts/analyse_component_ratio.py` |
| Prospective allocation comparison | `prospective_allocation.py` | `scripts/study_prospective_allocation.py`, `scripts/verify_prospective_allocation.py`, `scripts/replay_prospective_allocation.py` |
| Complete-workload CPU forecasting | `allocation_workload.py` | `scripts/study_cpu_prediction.py`, `scripts/audit_cpu_prediction.py`, `scripts/report_cpu_prediction.py` |
| Residual enclosures and inference | `residual_envelope.py`, `sequential_inference.py` | `scripts/export_residual_certificate.py`, `scripts/certify_residual_envelope.py`, `scripts/audit_certified_residual.py` |
| C++ propagation comparison | `src/qf_research/computational/funded_kernel.py`, `../cpp/funded_step.cpp` | `scripts/build_funded_kernel.py`, `scripts/benchmark_funded_kernel.py` |

Module names without a directory in this table refer to `src/qf_research/optimal_stopping`. The source manifest includes dependency helpers, tests and the attributed historical session calendar in addition to these principal files. Earlier acquisition helpers are included where historical validation code depends on them. Their inclusion does not make acquisition necessary for the public mathematical example or establish permission to redistribute a response.

The historical study scripts retain their original dated registration paths. Exact reproduction requires the corresponding original private archive, including its input manifests, frozen policies, registrations, prior evidence and source versions. Access to Workspace alone does not supply that archive. No public substitute for a withheld curve, quote history or fitted model bundle is provided. Authorised users with an exact archive should preserve its relative paths under `data/private` and restore the source versions bound by its hashes before invoking an audit or replay. A hash mismatch is a failure to satisfy that reproduction contract; the check must not be disabled or repaired by changing the archived hashes.

For example, with the original prospective-allocation archive present, the following commands audit its retained records and replay three specified financial rows. The replay deliberately excludes elapsed and CPU times from equality comparisons.

```sh
.venv/bin/python scripts/verify_prospective_allocation.py \
  --registration data/private/prospective_allocation/2026-09-28/registration_v1 \
  --run data/private/prospective_allocation/2026-09-28/run_v1 \
  --output data/private/reproduction/prospective_audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/replay_prospective_allocation.py \
  --registration data/private/prospective_allocation/2026-09-28/registration_v1 \
  --run data/private/prospective_allocation/2026-09-28/run_v1 \
  --output data/private/reproduction/prospective_rows
```

These operations do not rerun all 3,978 primary allocation pairs or the 7,956 later CPU-validation pairs. The original study and audit entry points are retained for inspection and authorised reproduction. Their historical `publication_clearance` fields describe the recorded status when those experiments were written, rather than the status of the subsequently reviewed public summaries.

## 5. Optional numerical environments

The public example and most research calculations use NumPy and SciPy. Plotting uses Matplotlib. The C++ propagation comparison needs a C++17 compiler and is optional; the Python path remains available. The original C++ code is released under the repository's MIT licence.

The outward-rounded enclosure scripts require `python-flint==0.9.0`, the version used for the retained certification work. They are designed for a separate environment and exchange explicit JSON artifacts with the simulation environment. Installing that package is unnecessary for the public example. A certified enclosure only covers the quantities and mathematical assumptions encoded in its input and audit; it is not a certificate for market-data validity or the complete modelling chain.

From that separate arithmetic environment, the following commands check finite-innovation replay and the global residual enclosure on exact mathematical fixtures.

```sh
mkdir -p data/private
python -m unittest discover -s tests -p test_certify_gain_replay.py -v
python scripts/test_certified_residual.py
```

Historical acquisition helpers require a separately configured, entitled LSEG environment. Credentials and provider responses are excluded. Optional `threadpoolctl` diagnostics report loaded numerical thread pools when available; the original timing scripts record unavailability rather than silently inferring a pool size. The public runner records the four numerical thread environment variables and makes no claim to certify native-library threading.

## 6. Verification and interpretation

```sh
.venv/bin/python -m pytest tests/test_replication_runner.py \
  tests/test_funded_policy.py tests/test_value_martingale.py \
  tests/test_terminal_repair.py tests/test_training_allocation.py
.venv/bin/python projects/01_optimal_stopping/paper/verification/numerical_results.py
```

The first command checks the model implementation and example's input and output contracts. The second checks the released summaries without requiring restricted observations. Successful execution of either command does not establish market validation. In the final research scope, unavailable historical quote ordering and incomplete funding evidence delimit the empirical claims; they do not supply missing observations or become assumed inputs.
