# Research notebooks

## 1. Current pricing computation

I connect the mathematical argument to executable pricing in [pricing and stopping](pricing_and_stopping.ipynb). The notebook constructs conditional expectation with exact rational arithmetic, checks a finite stopping identity, derives diffusion transitions, checks Greeks and compares tree and PDE refinements on the same exercise calendar. Independent training and evaluation expose regression rank, conditioning, sampling uncertainty and a European control that increases variance in the retained example. A native C++17 rollback supplies a separate implementation of the same tree when a compiler is available.

The example uses stipulated mathematical inputs and generated paths. It requires no provider account or restricted observations. It demonstrates the stated calculations; it does not reproduce the paper's fitted policies or certify a diffusion price. The [paper](../paper/main.tex) gives the general arguments and the larger experiments.

After installing the research and notebook requirements in the [reproduction guide](../docs/reproducing_paper.md), execute from a fresh kernel with:

```sh
python scripts/execute_release_notebooks.py --foundations
```

The command checks the computation without overwriting saved outputs. The [current reader command](../scripts/reproduce_current_study.py) additionally verifies the inventory and numerical fixtures and retains an executed notebook copy in a new private output directory. Add `--write` to retain a successful execution. The notebook checks the expected identities of its numerical sources. A changed source requires review and a new execution record. The optional [native verification](../scripts/verify_bermudan_tree.py) covers further deterministic and invalid-input cases.

## 2. Historical notebooks

[Baseline numerical validation](numerical_validation.ipynb) combines an analytical calculation with inspection of retained refinement tables. [Conditional research replication](conditional_research_replication.ipynb) executes a funded mathematical fixture, exact finite-law examples and reconstruction of released summaries and CPU findings. Neither notebook regenerates the original licensed financial experiments.

The historical selection remains explicit:

```sh
python scripts/execute_release_notebooks.py
```

This default runs only those two notebooks. The historical `v1.0.0` reproduction route and its source inventory remain separate from the current revision. Exact financial replay requires lawful access to the retained inputs and design records described in the [licensed-input guide](../replication/licensed_reproduction.md). No raw observations are exported by these notebook commands.
