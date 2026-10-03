# Experiment and verification entry points

## 1. Conditional stopping study

The [replication guide](../replication/README.md) maps the financial and computational questions to the numerical modules, original study scripts and retained-evidence audits. The [public example](../replication/run.py) executes the principal kernels without provider credentials or private observations. Historical scripts retain their original input and source-hash requirements; their presence does not make the required archive public.

## 2. Repository checks

| Command from the repository root | Purpose |
| --- | --- |
| `python scripts/reproduce_paper.py --output data/private/reproduction/reviewer_run` | Run the public paper checks, numerical fixture and two notebooks; preserve a verification record |
| `python replication/verify_sources.py` | Check listed file sizes and hashes against the published inventory |
| `python scripts/check_licensed_reproduction.py --archive-root /path/to/authorised-archive --output data/private/reproduction/licensed_readiness` | Check the required historical evidence locally without copying observations or running financial workloads |
| `python scripts/check_repository.py` | Check tracked Markdown targets and retained public-source hashes |
| `python scripts/execute_notebooks.py` | Execute all project notebooks in fresh kernels in the published checkout |
| `python scripts/execute_release_notebooks.py` | Execute the two reader notebooks for the stopping study |
| `python paper/verification/numerical_results.py` | Check released Markdown summaries and selected arithmetic |
| `python paper/verification/paper_results.py` | Check the LaTeX result tables against those summaries |
| `python paper/verification/cpu_timings.py` | Reconstruct Table 9 and expenditure from eighteen released computational workloads |

Notebook execution needs the notebook dependency lock. Without `--write`, execution leaves saved notebook files unchanged. The [development guide](../docs/development.md) describes installation and continuous integration. A summary check, a mathematical fixture and a financial replay answer distinct verification questions.
