# Exercise-policy loss under computational and financial constraints

Amanjeet Singh

30 September 2026

## 1. Paper and scope

The [compiled paper](Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf) and [self-contained LaTeX source](main.tex) present the mathematical framework and completed conditional computational study. The paper contains nine result tables on loss assessment, terminal-policy adjustment, repeated-training precision and CPU prediction. Independent market validation remains unresolved, and the unsuccessful original observation-admission exercise is reported explicitly.

`main.tex` contains the full text and bibliography and compiles in a single-document editor. The `sections/` files preserve the preceding modular methods source; they are not inputs to the current document. [references.bib](references.bib) retains citation metadata. The compiled research paper is `Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf`.

The [numerical supplement](numerical_results.md) presents the same nine tables in Markdown. Their [derived summaries](numerical_results.json) are released research findings, not observation-level financial inputs. The [CPU measurement supplement](cpu_timings.md) releases the eighteen workload records needed to reconstruct Table 9, including calibration charges and the budget overrun. The [literature comparison](contribution_review.md) distinguishes the research contribution from established theory.

## 2. Build

The open `main.tex` compiles directly in the document editor. A command-line build needs a TeX distribution with `latexmk` and pdfLaTeX:

```sh
cd projects/01_optimal_stopping/paper
make
```

The source uses Computer Modern, 11-point text, A4 pages and 25 mm margins. Its embedded bibliography uses the existing author-year format and requires no external bibliography file during compilation. On Overleaf, upload `main.tex` and select pdfLaTeX. `make clean` removes auxiliary files while retaining the PDF.

## 3. Analytical and table checks

```sh
cd projects/01_optimal_stopping/paper
make verify
```

These standard-library checks enumerate finite probability laws using exact rational arithmetic, verify the conditional-drift example and repeated-training variance identities, and check the nine Markdown and LaTeX tables against the released summaries. The CPU checker separately reconstructs forecasts, errors and expenditure from the workload ledger. The table renderer fails if a marked table differs from its source summary. It changes only those blocks when explicitly invoked with `--write`.

The finite examples are stipulated mathematical constructions, not market observations. They check implementations of the arguments without proving the general measure-theoretic statements or establishing theoretical novelty. The summary checks verify presentation and selected arithmetic identities; they do not rerun the financial experiments or validate statistical coverage.

## 4. Numerical replication and access

The [reproduction instructions](../../../docs/reproducing_paper.md) provide a single command for the public checks, fixture and notebooks. The [replication guide](../replication/README.md) identifies the numerical kernels, input schema and original experiment entry points. The [research notebook](../notebooks/27_conditional_research_replication.ipynb) explains these computations alongside the released findings. Exact replay of the retained financial experiments requires their original licensed input archive. A fresh authorised specification defines a new conditional experiment. The public reproduction command does not read that archive or export raw financial data.

Derived findings are published under the confirmed permission. Raw provider exports and observation-level subscription inputs remain private under the [publication policy](../../../docs/publication_policy.md). The [licensing scope](../../../LICENSING.md) distinguishes original software and writing from third-party data.
