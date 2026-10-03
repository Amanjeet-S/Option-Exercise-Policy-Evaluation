# Option Exercise-Policy Evaluation

## 1. Research question and completion

I study three questions in a finite-calendar put model with deterministic funding and capped discrete cash events: how to distinguish exercise-policy loss from upper-bound slack, how to allocate computation between independent training repetitions and evaluation paths, and whether measured numerical costs predict expenditure for complete workloads. Conditional expectations, martingale bounds and independently evaluated regression policies connect the financial problem to these computational questions.

The study was completed on 30 September 2026. It comprises the mathematical argument, conditional numerical experiments, implementation, paper and reproduction package, including an unsuccessful independent market-validation attempt. The original prospective screen admits none of fifteen observations. I retain that criterion and outcome. No further acquisition or purchased dataset is pending within the completed scope; independent market validity remains a limitation.

## 2. Findings

Analytically centred value functions reduce estimated dual gaps by 91.42 to 95.23 per cent relative to the slope construction while leaving exercise policies unchanged. This identifies tighter assessment, rather than increased policy reward. A common upper payoff separately identifies the expected gain from a feasible terminal adjustment; individual finite-sample retraining outcomes can still be unfavourable.

Repeating fixed training procedures distinguishes variation between fitted policies from evaluation noise. In the prespecified computational follow-up, selected estimator-variance estimates are 16.7 to 34.1 per cent of their reference estimates. The registered variance-difference intervals support the conditional precision comparison under asymptotic calibration, whose small-sample accuracy is limited by influential observations.

Historical CPU forecasts fail to transfer. Immediate local calibration reduces mean absolute relative forecast error from 29.065 to 2.841 per cent in the measured session, but adds cost and retains a budget overrun. These findings do not establish transferable processor performance, reliable future budget compliance, equal-realised-cost superiority or market profitability.

## 3. Paper and reproduction

The principal document is the [research paper (PDF)](paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf), with [self-contained LaTeX source](paper/main.tex). The [paper guide](paper/README.md) links the nine-table numerical supplement, computational timing ledger, literature comparison and final observation assessment.

The [reproduction instructions](docs/reproducing_paper.md) provide one command for public analytical checks, released-summary arithmetic and stipulated numerical examples. The [baseline notebook](notebooks/numerical_validation.ipynb) checks pricing kernels; the [research notebook](notebooks/conditional_research_replication.ipynb) explains the funded model, exact finite-law probability calculations and summary audit. Python implements simulation, regression, inference and experiment orchestration. A checked C++ propagation kernel and a separate python-flint environment provide optional computational and outward-rounded arithmetic checks.

The [replication guide](replication/README.md) documents the implementation and new authorised experiments. Exact historical financial replay requires the original input, design, source and evidence archive specified in the [licensed-input guide](replication/licensed_reproduction.md). The [historical replay record](replication/historical_replay.md) covers eight completed registered designs: seven reproduce retained financial outputs exactly; the CPU replay matches saved workload identities, with omitted financial estimates unavailable for comparison. Original acquisition and calibration were not rerun.

## 4. Evidence and licensing

I publish reviewed derived findings, original code and writing. Raw provider exports and observation-level subscription inputs remain private. Public summary checks verify presentation and selected arithmetic; stipulated examples do not reconstruct the financial tables. The [observation assessment](paper/data_admission.md) records why wider timing tolerances still do not establish market validation.

The [publication policy](docs/publication_policy.md) and [licensing scope](LICENSING.md) distinguish released original software under MIT, research writing under CC BY 4.0 and excluded third-party inputs. Established methods are attributed in the paper and [literature comparison](paper/contribution_review.md). The contribution is the controlled project-specific argument and evidence, without a claim of a new general pricing theorem.
