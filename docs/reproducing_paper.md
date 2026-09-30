# Reproducing the Option Exercise-Policy Evaluation study

## 1. What can be checked publicly

The [paper](../projects/01_optimal_stopping/paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf) reports a conditional computational investigation. I provide one command for the public checks, numerical example and explanatory notebooks. It requires no provider account, licensed input archive or paid data. It does not read, copy or export restricted raw financial data.

| Material | Public reproduction | Interpretation |
| --- | --- | --- |
| Finite probability examples | Recompute exact rational identities and counterexamples | Checks the examples, not a proof of every general theorem |
| Numerical algorithms | Execute the stipulated funded-put fixture and baseline notebook | New model-conditional calculations using explicitly mathematical inputs |
| Tables 1 to 8 | Check released summaries, selected arithmetic and agreement with the LaTeX tables | Does not regenerate the original financial simulations |
| Table 9 and CPU accounting | Reconstruct forecasts, errors, affordability and expenditure from eighteen released workload records | Uses retained timings and historical forecast inputs; does not rerun the original workloads or pilot |
| Original financial experiments | Implementation and input requirements are documented | Exact replay requires the original private licensed input and evidence archive |

The public fixture is not a replacement dataset for the financial tables. Its stipulated inputs and simulated outputs are labelled accordingly. Independent market validation remains unresolved.

## 2. Install once

Use a local checkout and Python 3.11 or 3.13, the versions checked in continuous integration. From a terminal:

```sh
git clone https://github.com/Amanjeet-S/Quantitative-Finance.git
cd Quantitative-Finance
python3 -m venv .venv
source .venv/bin/activate
python projects/01_optimal_stopping/replication/verify_sources.py
python -m pip install -r requirements.lock -r requirements-notebooks.lock
python -m pip install --no-deps -e .
```

On Windows, use `py -3.13 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. Subsequent `python` commands are the same. Installation downloads the declared software dependencies. The reproduction command itself requires no data download or network service.

The source check uses only the Python standard library. A missing or altered listed file is an error. Re-clone the intended revision rather than changing recorded hashes to suppress a mismatch. The inventory checks consistency of its declared files; it does not independently authenticate the repository or establish mathematical correctness.

## 3. Run the public reproduction

From the repository root, in the activated environment:

```sh
python scripts/reproduce_paper.py --output data/private/reproduction/reviewer_run
```

The output directory must be new. The command verifies the published inventory, runs the analytical and result-table checks, executes the mathematical fixture and runs the two published Option Exercise-Policy Evaluation notebooks in fresh kernels. It preserves logs, executed notebook copies and a machine-readable verification record under the chosen directory. Existing paper files and saved notebook outputs are not overwritten. Local outputs are ignored by Git.

Allow the command to finish before interpreting its record. A non-zero exit status indicates failure; successful earlier stages do not imply that later stages passed. Use a different output-directory name for a second run. Exact CPU times are not expected to agree across machines, and library or numerical-backend differences can affect floating-point rounding. The fixture records source and input hashes separately from its numerical checksum.

The command accepts no financial-input configuration and reads no private research archive. Its numerical outputs come from the public stipulated fixture. Publishing any additional authorised experiment remains a separate content review under the [publication policy](publication_policy.md).

## 4. Lightweight checks and the PDF

Readers who only need the analytical examples and released table checks can run the following without scientific packages, a C++ compiler or LaTeX:

```sh
python projects/01_optimal_stopping/replication/verify_sources.py
python -m unittest discover -s projects/01_optimal_stopping/paper/verification -v
python projects/01_optimal_stopping/paper/verification/numerical_results.py
python projects/01_optimal_stopping/paper/verification/paper_results.py
python projects/01_optimal_stopping/paper/verification/cpu_timings.py
```

The PDF is already included. To rebuild it, compile the self-contained [main.tex](../projects/01_optimal_stopping/paper/main.tex) using pdfLaTeX, or use `make` in its directory with a TeX distribution and `latexmk`. The [paper guide](../projects/01_optimal_stopping/paper/README.md) describes this optional build. Neither PDF compilation nor the separate python-flint certification environment is required for the default public reproduction.

## 5. Further verification

The [development guide](development.md) covers the complete repository test suite, optional C++ checks, public-source audits and Germany data notebook. The [detailed replication guide](../projects/01_optimal_stopping/replication/README.md) maps the research questions to the original experiment entry points and specifies the evidence required for an authorised financial replay. Those historical scripts do not distribute their original inputs.
