# Assessing Bermudan Exercise Policies

Amanjeet Singh

20 October 2026

## 1. Current manuscript

The [final paper](Assessing_Bermudan_Exercise_Policies.pdf) is compiled from the self-contained [LaTeX source](main.tex). It studies policy shortfall, martingale slack, sampling coverage and verified conditional integration using numerical references and public multiasset mathematical benchmarks. Its figures and bibliography are embedded, so compilation requires no external section files. The foundations appendices derive conditional expectation, optional sampling, measure change, the pricing PDE, regression projection and finite-difference stability. The [pricing notebook](../notebooks/pricing_and_stopping.ipynb) makes the core calculations executable.

The completed monetary experiments distinguish the cost of reaching a financial tolerance from the cost of an individual numerical query. All eight computed European-assessor limits lie below one unit after the documented exposure correction; a simultaneous continuous-model guarantee additionally requires the stated sampling and ideal-observation enclosure premises. The [contribution review](contribution_review.md) records subsequent research and the limits of each comparison.

The paper is dated 20 October 2026. The PDF, source and reproduction package accompany this revision. [Aggregate results](revision_results.json) and [figure sources](figures) support the manuscript. Its six figures are embedded in the LaTeX source; external figure files are not required to build it.

## 2. Reproduction and publication

The [reproduction guide](../docs/reproducing_paper.md) gives one current reader-check command and separates its component calculations from full experiment regeneration and licensed historical replay. Later two-asset monetary and bounded-correction development findings remain outside this paper. Run the current aggregate and transcription check with:

```sh
python paper/verification/revision_results.py
```

Build the final PDF with `make -C paper`. `make -C paper verify` runs the combined historical and current paper checks. A separate python-flint environment supports outward numerical verification. The separate current inventory binds the reviewed reader sources. Source and PDF hashes are recorded in the [completion record](../completion.json).

Current numerical figures use stipulated mathematical models and generated paths. Historical LSEG-derived findings require attribution and review under the [publication policy](../docs/publication_policy.md); captions must accompany separately circulated figures. Raw observations and restricted input bundles remain private. The earlier study is retained in the [v1.0.0 archive](https://github.com/Amanjeet-S/Option-Exercise-Policy-Evaluation/tree/v1.0.0). Historical tables, numerical summaries and timing supplements refer to that archive. Their source acknowledgement is: 'Source: LSEG Workspace, accessed through the University of St Andrews subscription; my calculations.' This notice must accompany separate circulation of the archived paper. The same source review covers tables, JSON and notebook summaries together.
