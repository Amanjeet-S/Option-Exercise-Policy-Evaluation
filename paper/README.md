# Exercise-policy loss under computational and financial constraints

Amanjeet Singh

3 October 2026

## 1. Paper and supporting evidence

The [compiled paper](Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf) and [self-contained LaTeX source](main.tex) present the completed conditional computational study. Its nine tables concern loss assessment, terminal-policy adjustment, repeated-training precision and CPU prediction. The [project scope](../README.md#1-research-question-and-completion) retains independent market validity as a limitation.

The [numerical supplement](numerical_results.md) and [derived summaries](numerical_results.json) reproduce the paper's tables. The [CPU supplement](cpu_timings.md) contains the eighteen workload records used to reconstruct Table 9, including calibration costs and the budget overrun. The [literature comparison](contribution_review.md) distinguishes established theory from the project-specific contribution. The [observation assessment](data_admission.md) and [aggregate results](data_admission.json) preserve the unsuccessful original admission outcome. [references.bib](references.bib) retains citation metadata; the complete bibliography is also embedded in `main.tex`.

## 2. Build and analytical checks

From the repository root, a command-line build requires a TeX distribution with `latexmk` and pdfLaTeX:

```sh
make -C paper
make -C paper verify
```

Alternatively, compile `main.tex` directly in a single-document editor or upload it to Overleaf with pdfLaTeX. No external section or bibliography files are required. `make -C paper clean` removes auxiliary files while retaining the PDF.

The standard-library verification checks exact finite probability examples, conditional drift, repeated-training variance identities, table transcription and retained CPU arithmetic. These checks do not prove the general measure-theoretic statements, rerun financial experiments or establish statistical coverage.

## 3. Numerical reproduction and access

The [public reproduction instructions](../docs/reproducing_paper.md) execute the checks, stipulated numerical fixture and notebooks. The [replication guide](../replication/README.md) documents the kernels and historical experiment entry points. Exact financial replay requires the original licensed input and evidence archive; public reproduction reads no such archive and exports no raw financial data.

Derived findings are published under confirmed permission. Raw provider inputs remain private under the [publication policy](../docs/publication_policy.md), with original software and writing covered by the [licensing scope](../LICENSING.md).
