# Option Exercise-Policy Evaluation

## 1. Research question

I study how accurately, and at what computational cost, an approximate Bermudan exercise policy can be assessed. The investigation separates policy shortfall from martingale upper-bound slack, numerical integration error and sampling uncertainty. I compare numerical pricing references, direct European-value assessors, polynomial approximations and analytically centred hinge functions on published one- and multiasset mathematical benchmarks.

The current study was completed on 20 October 2026. This repository contains the final paper and reviewed reproduction package. The historical study was completed on 30 September; substantive revision began on 4 October. Planning began in June, according to my retrospective chronology; the earlier dated files are unavailable. The [development guide](docs/development.md#4-revision-history-and-releases) explains the retained history.

## 2. Findings and scope

In the five-asset monetary experiment, all eight European-assessor upper limits lie below one unit at the final look, following the documented sixteen-row exposure correction. Neither it nor the zero assessor produces a limit at most half a unit. A simultaneous continuous-model policy guarantee additionally requires the ideal-sampling and observation-enclosure premises; the numerical certificates here concern supplied binary states. The [current manuscript](paper/main.tex) retains that distinction and the complete computational charges.

The later two-asset monetary and bounded-correction studies are exploratory extensions outside this paper and its current reproduction package. I retain their full local records, including adverse cost rankings, for separate development. The [contribution review](paper/contribution_review.md) defines the paper's scope.

The historical study tightened estimated dual gaps without changing exercise decisions. Its CPU forecasts failed to transfer reliably, and its independent market-admission screen yielded no performance estimate. I retain those outcomes in the historical supplement. The mathematical benchmarks require no paid dataset and do not establish market profitability.

## 3. Paper and reproduction

The [final paper](paper/Assessing_Bermudan_Exercise_Policies.pdf) and its self-contained [LaTeX source](paper/main.tex) report the completed study. The [paper guide](paper/README.md) identifies the separate historical paper and its evidence. The [completion record](completion.json) retains the current source and PDF hashes.

The [pricing notebook](notebooks/pricing_and_stopping.ipynb) connects exact conditional expectation and stopping calculations to diffusion pricing, Greeks, tree/PDE refinement, native C++ rollback and independently evaluated regression policies. The paper gives the corresponding measure-theoretic and numerical derivations in its appendices.

The [reproduction guide](docs/reproducing_paper.md) separates the current reader checks from full experiment regeneration. After installing the research and notebook requirements, run:

```sh
python scripts/reproduce_current_study.py --output data/private/reproduction/current_review
```

The command verifies the current inventory, checks aggregate arithmetic and manuscript transcription, runs selected numerical tests and executes a separate notebook copy. It preserves a record in a new private directory. Add `--with-native --compiler g++` for the independent C++ verification. It does not regenerate every experiment or use provider observations. The historical `v1.0.0` route remains separately documented.

The implementation is in [src/qf_research](src/qf_research), with [experiment entry points](scripts/README.md), [tests](tests) and [notebooks](notebooks). Optional C++ and python-flint components provide propagation and outward arithmetic checks. The historical source manifest belongs to `v1.0.0`; [the current inventory](replication/current_source_manifest.json) binds the selected revision files separately. Each experiment retains its own registered source identities.

The [licensed-input guide](replication/licensed_reproduction.md) specifies the original archive required for exact historical replay. The [replay record](replication/historical_replay.md) covers eight completed designs: seven reproduce financial outputs exactly; the CPU replay matches saved workload identities. Original acquisition and calibration were not rerun. Public summary verification has a narrower scope.

## 4. Sources and licensing

Original software is licensed under MIT and research writing under CC BY 4.0, subject to the [licensing scope](LICENSING.md). Established methods are attributed in the paper and [source register](SOURCES.md).

Raw LSEG observations and restricted transformed inputs remain private. University guidance permits only insubstantial, non-systematic academic derived distribution without independent commercial value, with source attribution; underlying observations require deletion at completion of their authorised use. The [publication policy](docs/publication_policy.md) applies to figures, tables and downloadable summaries together.
