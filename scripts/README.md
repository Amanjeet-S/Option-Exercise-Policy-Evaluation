# Experiment and verification entry points

## 1. Current mathematical investigation

The [reproduction guide](../docs/reproducing_paper.md) gives the numerical study sequence and distinguishes fresh experiments from reconstruction of retained summaries. Current reader checks are:

| Command from the repository root | Purpose |
| --- | --- |
| `python scripts/reproduce_current_study.py --output data/private/reproduction/current_review` | Verify current sources, component calculations and a retained fresh-kernel notebook copy |
| `python paper/verification/revision_results.py` | Check current aggregate arithmetic and manuscript transcription |
| `python scripts/execute_release_notebooks.py --foundations` | Execute the current pricing notebook in a fresh kernel |
| `python scripts/verify_bermudan_tree.py --output data/private/cpp_build/reviewer_run` | Build the native tree and compare independent deterministic, tree and analytical references |

The native output directory must be new. Notebook execution needs the notebook dependency lock; native verification additionally needs a C++17 compiler. Without `--write`, notebook execution leaves saved files unchanged. The execution record identifies whether the optional native notebook comparison actually ran.

The larger study entry points retain their registered source and input requirements. Their public-model inputs can be generated without market-data subscriptions, but a summary check or the illustrative notebook does not reproduce those experiments. Later two-asset monetary and bounded-correction findings are outside the current paper and reader package.

## 2. Historical study

The [replication guide](../replication/README.md) maps the original funded model, study scripts and retained-evidence audits. In a separate `v1.0.0` checkout:

| Command from the historical repository root | Purpose |
| --- | --- |
| `python scripts/reproduce_paper.py --output data/private/reproduction/reviewer_run` | Run the historical public checks, mathematical fixture and two notebooks |
| `python replication/verify_sources.py` | Check the historical source inventory |
| `python scripts/check_repository.py` | Check historical Markdown targets and retained source identities |
| `python scripts/execute_release_notebooks.py` | Execute only the two historical notebooks |
| `python paper/verification/numerical_results.py` | Check historical Markdown summaries and selected arithmetic |
| `python paper/verification/paper_results.py` | Check historical LaTeX tables against those summaries |
| `python paper/verification/cpu_timings.py` | Reconstruct the historical CPU findings from released workload records |

Exact licensed financial replay requires the authorised archive specified in the [licensed-input guide](../replication/licensed_reproduction.md). Its readiness checker inspects that archive locally without copying observations or running financial workloads. The [development guide](../docs/development.md) states the separate status of current local verification, the prepared current workflow and the historical release.
