# Exercise-policy loss under computational and financial constraints

In this methods-only paper for my optimal-stopping project, I develop the funded cash-event model, policy-loss decomposition, approximate-value martingales, expiry-calendar comparison, terminal adjustment, conditional inference and computational allocation design. Appendix A uses predictable compensation to quantify the effect of conditional drift in an approximate penalty.

[Read the compiled paper](methods.pdf) or start with [the LaTeX source](main.tex). The text is divided into the files in `sections/`; published methodological references are in [references.bib](references.bib).

## 1. Build

Use a TeX distribution with `latexmk`, pdfLaTeX and BibTeX. The source uses Computer Modern, 11-point text, A4 pages with 25 mm margins, and standard packages including `amsmath`, `amsthm`, `natbib` and `hyperref`. References use the `apalike` BibTeX style.

```bash
cd projects/01_optimal_stopping/paper
make
```

Alternatively:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=methods main.tex
```

The output is `methods.pdf`. `make clean` removes auxiliary files while retaining the PDF. On Overleaf, upload this directory, select `main.tex` as the main document and use pdfLaTeX.

## 2. Scope

This version contains no raw or cleaned subscription observations, fitted market inputs, empirical summaries, result tables or data-derived figures. I retain the full numerical manuscript locally while publication of its derived findings is clarified. Removing empirical findings changes the scope: this document presents the mathematical framework and research design, not the complete empirical paper.

I apply established methods to the stated stopping problem and cite their published sources. I make no claim that the duality result, regression method, lognormal option integral or elementary stopping comparisons are new general theorems.

## 3. Analytical verification

The [literature comparison](contribution_review.md) distinguishes the research question from its established mathematical foundations. The appendix includes a stipulated finite example, not a market observation. Its derivation states why the correction is needed, how it follows from compensation and optional sampling, and what it permits in a policy comparison. No claim of theoretical priority is made.

The finite-tree checks use Python's standard library and exact rational arithmetic:

```bash
python3 verification/conditional_drift.py
python3 -m unittest discover -s verification -v
```

The first command checks every admissible policy on a deterministic grid of finite trees and reports the analytical counterexample. The second also checks invalid stopping rules, probability inputs and the single-right boundary case. These are checks of the analytical implementation, not empirical evidence.
