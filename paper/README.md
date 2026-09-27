# Exercise-policy loss under computational and financial constraints

In this methods-only paper for my optimal-stopping project, I develop the funded cash-event model, policy-loss decomposition, approximate-value martingales, expiry-calendar comparison, terminal adjustment, conditional inference and computational allocation design. Appendix A uses predictable compensation to quantify the effect of conditional drift in an approximate penalty.

[Read the compiled paper](methods.pdf) or start with [the LaTeX source](main.tex). `main.tex` is self-contained: it includes the complete text and bibliography and can be compiled directly in a single-document editor. The files in `sections/` retain the preceding modular source for reference; edit `main.tex` for the current paper. [references.bib](references.bib) retains the citation metadata.

The separate [numerical supplement](numerical_results.md) reports reviewed findings on bound tightness, terminal adjustments, training gains and computational allocation. It includes six tables, a figure and [derived summaries](numerical_results.json). It preserves the failed CPU-budget predictions and the absence of an eligible prospective sample. It does not replace the methods paper or alter its LaTeX source.

## 1. Build

The open `main.tex` document compiles directly with the built-in editor. For a local command-line build, use a TeX distribution with `latexmk` and pdfLaTeX. The source uses Computer Modern, 11-point text, A4 pages with 25 mm margins, and standard packages including `amsmath`, `amsthm`, `natbib` and `hyperref`. The embedded bibliography preserves the existing author-year reference format and requires no external bibliography file at compilation.

```bash
cd projects/01_optimal_stopping/paper
make
```

Alternatively:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=methods main.tex
```

The output is `methods.pdf`. `make clean` removes auxiliary files while retaining the PDF. On Overleaf, uploading `main.tex` alone is sufficient; select it as the main document and use pdfLaTeX.

## 2. Scope

The methods paper contains no raw or cleaned subscription observations, fitted market inputs, empirical summaries, result tables or data-derived figures. Its scope remains the mathematical framework and research design. Permission to publish derived research findings has been confirmed, and the separate numerical supplement releases selected findings under the [publication policy](../../../docs/publication_policy.md). Restricted observations, exact financial inputs, reconstructable supplements and the complete numerical manuscript remain local.

The supplement's summary check uses only Python's standard library. Run `python3 projects/01_optimal_stopping/paper/verification/numerical_results.py` from the repository root. It checks table presentation, case coverage and selected rounded arithmetic identities. It does not rerun the experiments or constitute independent verification against the private evidence. The public baseline code is not a complete implementation of the supplement's experiments.

I apply established methods to the stated stopping problem and cite their published sources. I make no claim that the duality result, regression method, lognormal option integral or elementary stopping comparisons are new general theorems.

## 3. Analytical verification

The [literature comparison](contribution_review.md) distinguishes the research question from its established mathematical foundations. The appendix includes a stipulated finite example, not a market observation. Its derivation states why the correction is needed, how it follows from compensation and optional sampling, and what it permits in a policy comparison. No claim of theoretical priority is made.

The finite-tree checks use Python's standard library and exact rational arithmetic:

```bash
python3 verification/conditional_drift.py
python3 -m unittest discover -s verification -v
```

The first command checks every admissible policy on a deterministic grid of finite trees and reports the analytical counterexample. The second also checks invalid stopping rules, probability inputs and the single-right boundary case. These are checks of the analytical implementation, not empirical evidence.
