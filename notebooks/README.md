# Research notebooks

## 1. Reading and execution sequence

I use two notebooks to connect the numerical foundations to the completed conditional study. Each runs from a fresh kernel after the repository and notebook requirements have been installed. Neither needs subscription access.

1. [Baseline numerical validation](01_numerical_validation.ipynb) recomputes the continuous-yield analytical reference and inspects the retained tree and PDE refinement tables. It distinguishes a new analytical calculation from inspection of a saved experiment.
2. [Conditional research replication](27_conditional_research_replication.ipynb) executes funded exercise-policy fitting, analytically centred martingale assessment and a terminal adjustment on a stipulated mathematical configuration. It enumerates training and evaluation probability laws exactly, checks the allocation calculation, inspects the released financial summaries and reconstructs the CPU findings from eighteen published workload measurements.

The [public reproduction command](../../../docs/reproducing_paper.md) runs both notebooks alongside the paper checks and preserves executed copies. The [replication interface](../replication/README.md) documents the financial input contract and the executable calculation. Numerical methods remain in reusable Python modules; the notebook supplies the mathematical explanation and interpretation.

## 2. Reproduce the released notebooks

From the repository root, use the existing research environment or install the frozen requirements described in [development](../../../docs/development.md). Then run:

```bash
python scripts/execute_release_notebooks.py
```

The command selects these two notebooks explicitly and executes each in a fresh kernel. It does not overwrite saved outputs. Add `--write` to retain outputs after both notebooks succeed. The complete repository notebook command also executes the separate Germany data-coverage notebook in a published checkout.

The default financial configuration is a stipulated mathematical example. Its numerical outputs are not market observations, calibrated parameters or a reproduction of the paper's retained research tables. The released-summary cells check aggregate presentation and arithmetic, without accessing the restricted observations behind those summaries. Fresh processor timings are measured internally by the replication calculation but are not substituted for the published timing experiment.

## 3. Interpretation and scope

The notebooks support the study of expected exercise-policy loss, training uncertainty and computational allocation under explicit financial assumptions. The analytical limiting case and exact finite laws test stated identities. Their finite checks do not replace the general arguments in the [paper](../paper/main.tex).

The conditional empirical computations reported in the [numerical supplement](../paper/numerical_results.md) have a separate evidential status. Repeating those exact experiments requires their authorised inputs and retained design records. Independent historical market validation was not completed, and the incomplete funding construction was excluded. Its missing constituents are not filled with fabricated observations.

Earlier investigative notebooks are retained locally as research records. They are outside this deliberately selected reader route and are not dependencies of either released notebook. Successful execution alone does not determine publication eligibility; release decisions concern the reviewed source and output content.
