# Licensed-input reproduction

## 1. Scope and terminology

I distinguish reconstruction of retained results, numerical replay and a new experiment. Reconstruction checks saved observations and recalculates their summaries. Numerical replay repeats the financial calculations with the original model inputs, random streams, registered counts and numerical implementation. A new experiment changes at least one of those ingredients. Repeating a processor measurement always produces a new timing observation, even when the financial calculation repeats exactly.

The [public reproduction route](../docs/reproducing_paper.md) needs no licensed archive. It executes mathematical examples, checks the released financial summaries and reconstructs the CPU comparison from published workload measurements. The instructions below concern the separate route for a reader with lawful access to the original inputs and retained evidence. A data subscription alone is insufficient: it does not supply the original acquisition vintage, model construction, frozen policies, registrations or source versions.

This guide specifies the numerical and evidential requirements of the existing implementation. It does not grant access to an archive or permission to redistribute its contents. No complete financial rerun is claimed merely because these instructions, or the public checks, are available. The [publication policy](../docs/publication_policy.md) continues to exclude restricted observations from the public repository.

Lawful possession must include permission to retain and use the inputs for the current research purpose. I do not supply or share the subscription archive, including with another licence holder. The university guidance requires deletion of underlying stored data when its project is completed, including relevant copies. If required source inputs are deleted, these commands remain a record of the original requirements but cannot perform an exact replay without independently and lawfully obtained matching inputs. A later acquisition or substituted vintage defines a new experiment. Historical hashes and publication-status fields remain unchanged; deletion is recorded separately. The retention review must distinguish restricted source copies from original model-only simulations and permitted findings.

The current repository places `paper/`, `replication/` and the implementation at its root. Historical source and configuration references retain their original archive namespace, including `projects/01_optimal_stopping/`. Exact financial replay runs the preserved original implementations in that original layout. The current archive checker uses `--archive-root` to inspect it; relocation does not replace its registered source bytes.

## 2. Required input roles

The following roles describe the inputs needed to identify the calculations. They are not a downloadable replacement for the retained archive.

| Role | Required content | Purpose |
| --- | --- | --- |
| Financial specification | The original case ordering; model, put contract and finite exercise calendar; cash-event timing and caps; ownership and settlement factors; calibrated volatility and premium conversion | Defines the conditional pricing problem in each case |
| Frozen policy record | Regression rules, basis degree, coefficients, scaling, fitting diagnostics, training size, model/calendar identity and the associated control record | Repeats assessment of the same instructions in Tables 1 and 2 |
| Numerical design | Approximation settings, regression rules, evaluation sizes, repetition counts, terminal extensions and inference conventions | Identifies the discretisation and estimators |
| Random-stream design | Bit generator, seed entropy, spawn keys, stream roles, sharing rules and exclusions of previously used streams | Repeats the draws and their intended independence structure |
| Registration and allocation | Original task lists, pilot and main registrations, frozen arm counts, execution schedules, budget definitions and contrast family | Preserves the design that preceded the relevant main outcomes |
| Retained outcomes and timings | Complete policy or pair records, setup costs, calibration observations, pilot results and original timing charges | Reconstructs table entries and compares replayed financial quantities with their retained values |
| Source and environment record | Required source bytes and hashes, configuration bytes, dependency versions, numerical backend, thread settings and run metadata | Identifies the implementation and the limits of numerical equality |
| Historical provenance | Input manifests and their referenced acquisition, convention, calibration and prior-verification records | Supports claims about how the financial specifications and registrations were obtained |

For Tables 3 to 8, the central simulation kernels use the supplied model, contract and calendar to fit fresh policies under a fixed design. They do not mathematically need every earlier fitted coefficient. The historical study wrappers nevertheless load those specifications through frozen-policy records and check a larger evidence chain. A smaller model bundle is therefore not a drop-in replacement for the original registrations.

The public [input schema](input.schema.json) describes a fresh single-model calculation. It supports policy fitting, martingale assessment, a terminal adjustment and repeated-training summaries. It does not encode the original case family, the historical frozen-policy record, all study-specific registrations or a complete replay of the nine paper tables. Changing its classification to `authorised_model` does not establish identity with the original study.

## 3. Table lineage

Table numbers refer to the [research paper](../paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf). The final public presentation is checked by [paper_results.py](../paper/verification/paper_results.py) against [numerical_results.json](../paper/numerical_results.json). That presentation check does not perform the earlier computations listed here.

| Paper table | Experiment and retained quantities | Reconstruction or verification |
| --- | --- | --- |
| 1. Expected gaps | [study_value_martingale.py](../scripts/study_value_martingale.py); `summary.json`, `variants[].methods`, for baseline, coarse and fine assessment of unchanged policies | [audit_value_martingale.py](../scripts/audit_value_martingale.py) checks coverage, deterministic value constructions and retained estimates |
| 2. Terminal adjustment | [study_terminal_repair.py](../scripts/study_terminal_repair.py); the 5,400-second scenario in `summary.json`, using `mean_gap_quote.original`, `mean_gap_quote.repaired` and `mean_integrated_gain` | [audit_terminal_repair.py](../scripts/audit_terminal_repair.py) checks the retained policy scenarios and accounting identities |
| 3. Repeated-training gains | [study_training_evaluation.py](../scripts/study_training_evaluation.py); primary fine-approximation estimates with the larger evaluation sample | [analyse_training_evaluation.py](../scripts/analyse_training_evaluation.py) builds `tables.json`; the evaluation share divides estimated evaluation variance by the squared across-pair standard error |
| 4. Allocation comparison | [study_training_allocation.py](../scripts/study_training_allocation.py), pilot and main stages; registered counts, pilot variance predictions and main estimates | [report_training_allocation.py](../scripts/report_training_allocation.py) derives the within-case ratios in `values.json` |
| 5. Allocation gains | The same independent main allocation records | The report derives each arm's mean gain and pointwise Student interval |
| 6. Allocation costs | Original pilot and main setup, training and evaluation measurements | [reproduce_training_allocation.py](../scripts/reproduce_training_allocation.py) reconstructs component charges; [build_training_allocation_paper_assets.py](../scripts/build_training_allocation_paper_assets.py) derives the table from the verified report and timing review |
| 7. Prospective allocation | [study_prospective_allocation.py](../scripts/study_prospective_allocation.py); all nine registered arms, their pair records and `comparisons.json` | [verify_prospective_allocation.py](../scripts/verify_prospective_allocation.py) reconstructs the arm statistics; [report_prospective_allocation.py](../scripts/report_prospective_allocation.py) prepares the report |
| 8. Registered contrasts | The same prospective arm records and registered contrast family | Mean and estimator-variance difference intervals use the recorded common critical value; ratios remain descriptive |
| 9. CPU forecasts | [study_cpu_prediction.py](../scripts/study_cpu_prediction.py); eighteen complete workloads and both calibration schemes | [audit_cpu_prediction.py](../scripts/audit_cpu_prediction.py) checks the original records; the public [CPU verifier](../paper/verification/cpu_timings.py) independently reconstructs the released timing ledger |

The paper also makes quantitative statements outside the nine tables. Their additional records belong to a full-paper reproduction target. The current document contains no external figures.

| Additional finding | Required retained evidence and computation |
| --- | --- |
| Terminal-target retraining intervals | `terminal_training/2026-09-20/run_v1/manifest.json`; [study_terminal_training.py](../scripts/study_terminal_training.py) and its retained summary. This is a separate experiment from the terminal adjustment in Table 2 |
| Negative pair gains and delete-one sensitivity | `training_evaluation/2026-09-26/analysis_v1/manifest.json` and `policy_paper/2026-09-27/inference_review_v1/manifest.json`; [review_training_inference.py](../scripts/review_training_inference.py) recomputes the deletion diagnostics and checks the retained paper inputs |
| Case definitions, exercise-right counts and original training coupling | Original calendar and fitted-policy tasks, configurations and manifests reached through the Table 1 evidence chain |
| Allocation pilot expense and component timing interpretation | Original pilot and main manifests plus `training_allocation/2026-09-27/timing_review_v1/manifest.json`; component selection must retain the observation supplying the median complete-pair cost |
| Rationale and uncertainty set for the prospective allocation | Original cost-sensitivity, variance-sensitivity and component-ratio records bound by the prospective registration, including `component_ratio/2026-09-28/run_v1/manifest.json` |
| Variance-influence concentration and additional budget summaries | The complete prospective arm records and CPU experiment records, already used by the table groups above |
| Unmet empirical admission criterion | `prospective_validation/2026-09-20/registration_v1/manifest.json` and `prospective_validation/2026-09-26/evaluation_v1/manifest.json`, with their referenced inputs; the historical entry points `scripts/evaluate_prospective_sample.py` and `scripts/audit_prospective_sample.py` implement the assessment and require their recorded source versions |

These archive paths are relative to `data/private`. A readiness assessment limited to the table groups must not be labelled a complete check of the paper. The exact finite-law examples in the appendices have a separate public route through [allocation_probability.py](../paper/verification/allocation_probability.py) and [conditional_drift.py](../paper/verification/conditional_drift.py). Optional numerical-certification and C++ performance studies in the repository are additional supporting work, rather than extra quantitative tables or figures in the current paper.

## 4. Preparation before numerical execution

Use a separate local checkout for the historical reproduction. Keep the current published checkout available for the public checks. Install the recorded numerical environment where it remains available, and document any unavoidable difference in interpreter, libraries or numerical backend. A successful import is insufficient evidence that floating-point results will agree exactly.

First identify the intended original run and its registration or configuration. Check the complete manifest closure before creating a new numerical output directory: each referenced file must exist at the expected relative path, its bytes must match the recorded hash, and the required source version must be available. Verify the run's case coverage, registered counts and completion status. An archive inventory is necessary but does not establish the financial validity of those inputs.

The [requirements record](licensed_requirements.json) identifies eleven original root manifests covering the table studies, the repeated-training registration and the additional quantitative claims described above. The registration requires an explicit entry because the completed training-evaluation run records its digest outside the path-to-hash maps. From the published checkout, run the standard-library [readiness checker](../scripts/check_licensed_reproduction.py) against a separately held, authorised archive checkout:

```sh
python scripts/check_licensed_reproduction.py \
  --archive-root "/absolute/path/to/authorised-archive-checkout" \
  --output data/private/reproduction/licensed_readiness
```

The destination must be new. The checker expands supported path-to-hash maps from the identified manifests, checks the referenced file bytes and writes `readiness.json` privately. It does not copy financial observations, run simulations or load historical native libraries. A missing or changed dependency gives an incomplete result and a non-zero exit status. Unsafe or contradictory metadata is rejected. The report distinguishes evidence readiness from numerical execution, environment compatibility and permission to use or transfer the archive. Opaque scalar digests and dependencies described only in prose are not inferred as file references.

The historical closure is not confined to `data/private`. It can refer to original configuration, report, notebook, source-review and build-product paths elsewhere in the checkout. Some of these files are not in the public release. Preserve their relative locations in the isolated reproduction checkout and establish lawful access before using them. Do not copy an entire working archive into a public repository. Do not overwrite current documents merely to satisfy an old source hash, and do not replace an archived hash with the digest of a different file.

In particular, the Table 1 runner reads `projects/01_optimal_stopping/configs/value_martingale_2026-09-20.json`; the Table 2 runner reads `projects/01_optimal_stopping/configs/terminal_repair_2026-09-20.json`. Both configurations refer to prior verified experiments. They are required original archive members, not values inferred from the released tables. Later registrations similarly bind their configuration, tasks, earlier evidence and source versions. Historical publication-status fields record the status at execution and must retain their original bytes.

After those checks, select fresh private destinations and retain the original archive unchanged. Each `--output` below must be unused. Run from the reproduction checkout's root with the intended Python interpreter. The single-worker allocation and timing studies require these numerical thread settings:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

The paths in the commands identify the original retained layout. They do not imply that those files are supplied publicly. Stop the archival route when a prerequisite is missing. An independently reacquired dataset can support a separately evidenced numerical reproduction if the dated model inputs, design and implementation are demonstrably identical. It does not acquire the identity or provenance of the original archive. A new data vintage or altered model specification instead defines a changed reproduction target or a new experiment.

## 5. Complete registered numerical designs

With the verified archive and matching implementation present, the following original study entry points execute the complete designs. They can require substantial time and memory. Resource guards and partial records must be respected: an interrupted run is not a completed experiment.

```sh
python scripts/study_value_martingale.py \
  --workers 1 --output data/private/reproduction/licensed/value_martingale
python scripts/study_terminal_repair.py \
  --workers 1 --output data/private/reproduction/licensed/terminal_repair
python scripts/study_terminal_training.py \
  --workers 1 --output data/private/reproduction/licensed/terminal_training
python scripts/study_training_evaluation.py run \
  --registration data/private/training_evaluation/2026-09-26/registration_v1 \
  --output data/private/reproduction/licensed/training_evaluation
python scripts/study_training_allocation.py main \
  --registration data/private/training_allocation/2026-09-27/main_registration_v1 \
  --output data/private/reproduction/licensed/training_allocation
python scripts/study_prospective_allocation.py run \
  --registration data/private/prospective_allocation/2026-09-28/registration_v1 \
  --output data/private/reproduction/licensed/prospective_allocation
python scripts/study_cpu_prediction.py run \
  --registration data/private/cpu_prediction/2026-09-28/registration_v1 \
  --output data/private/reproduction/licensed/cpu_prediction
```

The allocation main run above retains the original selected counts and pilot evidence. It does not rerun the pilot. The pilot's numerical workloads can be repeated separately using its original registration:

```sh
python scripts/study_training_allocation.py pilot \
  --registration data/private/training_allocation/2026-09-27/registration_v1 \
  --output data/private/reproduction/licensed/allocation_pilot
```

Reallocating with newly measured pilot costs can select different counts. That is a new allocation experiment, even if its financial inputs and random-seed root are unchanged. The original `register` and `allocate` entry points are available in the study source, but they must not be presented as recovering the original prespecification when given a new history.

These commands generate new run records; they do not automatically establish equality with the paper. Compare complete coverage and the intended numerical fields with the retained originals. Preserve the recorded seed roles and floating-point conventions. The study-specific `numerical_record` helpers exclude timings where provided, while the allocation audits apply their stated comparison rules. Equality checks should neither discard unfavourable rows nor silently relax a failed numerical comparison.

## 6. Reconstruction and representative replays

The following commands inspect the retained evidence and repeat selected financial calculations. They are useful checks of the archive and implementation, but are not substitutes for the complete runs above.

```sh
python scripts/reproduce_training_evaluation.py \
  --run data/private/training_evaluation/2026-09-26/run_v1 \
  --output data/private/reproduction/licensed/training_evaluation_checks
python scripts/reproduce_training_allocation.py \
  --study data/private/training_allocation/2026-09-27 \
  --output data/private/reproduction/licensed/allocation_checks
python scripts/replay_prospective_allocation.py \
  --registration data/private/prospective_allocation/2026-09-28/registration_v1 \
  --run data/private/prospective_allocation/2026-09-28/run_v1 \
  --output data/private/reproduction/licensed/prospective_checks
```

The training-evaluation command repeats four blocks: the first and last repetitions of the first and last cases. The allocation command reconstructs the retained inference and cost decomposition and repeats the pairs prescribed by its audit. The prospective command repeats three rows. Their records explicitly identify partial numerical coverage. The value-martingale and terminal-repair audits rebuild deterministic constructions and inspect retained estimates; they do not repeat all path simulations.

For example, the retained Table 9 record can be audited privately with:

```sh
python scripts/audit_cpu_prediction.py \
  --registration data/private/cpu_prediction/2026-09-28/registration_v1 \
  --run data/private/cpu_prediction/2026-09-28/run_v1 \
  --output data/private/reproduction/licensed/cpu_record_audit
```

Its public arithmetic reconstruction remains available without that archive:

```sh
python paper/verification/cpu_timings.py
```

## 7. Timing, provenance and reporting

Retained CPU measurements are observations of the original execution. Tables 4, 6, 7 and 9 contain timing quantities or ratios involving them. Reconstructing those entries uses the original measurements and accounting rules. A fresh numerical replay supplies new setup, training, evaluation and calibration costs. It cannot be required to reproduce the old times, affordability counts or timing ratios.

A model-input replay also has narrower scope than an acquisition replay. It can establish that supplied financial specifications lead to the retained numerical results without re-establishing the original quotes, funding construction, contractual interpretation or information available at valuation. Source PDFs and acquisition records belong to that larger provenance assessment. Their absence must be reported when it prevents the relevant historical claim, rather than filled with a substitute observation.

A completed reproduction record should identify the supplied archive and source versions, the calculations actually repeated, the retained quantities reconstructed, the comparison rules and any failures. Report numerical agreement separately from fresh timing measurements. Retain incomplete runs and discrepancies. A new authorised model study should use the [general input interface](README.md#3-inputs-for-a-new-authorised-experiment) and receive its own interpretation; it must not inherit the original paper's table labels as a claim of exact reproduction.

## 8. Comparing a completed replay

The standard-library [comparison tool](../scripts/compare_licensed_replay.py) checks the complete manifested record of a fresh study against its historical counterpart. Place both study directories beneath the published checkout's `data/private` directory, including any isolated archive checkout. The comparison destination must also be private and unused. For the value-martingale study:

```sh
python scripts/compare_licensed_replay.py \
  --study value_martingale \
  --archive-root data/private/authorised_archive \
  --fresh data/private/authorised_archive/data/private/reproduction/licensed/value_martingale \
  --output data/private/reproduction/value_martingale_comparison
```

The other study names are `terminal_repair`, `terminal_training`, `training_evaluation`, `allocation_pilot`, `training_allocation`, `prospective_allocation` and `cpu_prediction`. Each comparison verifies artifact hashes, coverage and retained values. Numerical equality is exact; small floating-point differences remain failures. Execution times, telemetry and quantities derived from fresh timings have explicit exclusions, recorded in compressed private ledgers. Original registered allocations and their historical cost inputs remain part of the comparison. A successful comparison of two directories does not, by itself, establish that either was freshly computed; the launch commands, process outcomes and source checks supply that separate evidence.

The original CPU study requires a narrower interpretation. Its records retain policy, path and approximation fingerprints, random-stream identities and workload counts, but omit the financial estimates themselves. Its `complete_financial_record_sha256` also incorporates measured durations. This digest cannot establish equality of unsaved estimates in a fresh run. The comparator records that limitation explicitly while checking the retained identities. Reconstruction of Table 9 continues to use the original timing observations.

The [historical replay record](historical_replay.md) distinguishes completed numerical execution from reconstruction of retained evidence and identifies the completed replay coverage and its comparison limits.

## 9. Final assessment of retained market observations

The [completed study](../README.md) retains the unsuccessful primary admission decision. The [observation-coverage supplement](../paper/data_admission.md) reports descriptive intersections of the previously specified quote and funding tolerances. It contains no newly calibrated price, market-performance estimate or provider observation.

A reader with lawful access to the original archive can reconstruct those aggregate counts from its adapter audit, configuration, cases and registration:

```sh
python scripts/summarise_market_admission.py \
  --archive-root /path/to/authorised/archive \
  --output data/private/reproduction/admission_summary
```

Run this command from the current repository root, using a new output path. It verifies the retained input relationships and reconstructs the mechanical predicates without network access. Its output remains under the current checkout's private data directory. No observation-level quotes, rates, strikes or instrument identifiers are exported. This reconstruction is separate from the public check of released counts and does not resolve absent documentary evidence concerning historical quotes or funding conventions. No further data acquisition is pending within the completed project.
