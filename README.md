# Option Exercise-Policy Evaluation

## 1. Research question

I study how approximate exercise policies lose value, how tightly that loss can be measured, and how computation should be allocated to estimate a training procedure's performance. The financial problem is a finite-calendar put with deterministic funding and capped discrete cash events. Conditional expectations and martingale bounds connect the stopping problem to independently evaluated regression policies. A separate cost study tests whether earlier processor measurements predict expenditure for complete numerical workloads.

The [paper](paper/README.md) contains the model, derivations, nine tables of conditional findings and the limits of their interpretation. The conditional computational investigation is complete. Independent market validation remains unresolved: the original prospective observation screen admits no primary observations. I retain that outcome without replacing its admission rule after inspecting the sample.

## 2. Findings and interpretation

Analytically centred value functions reduce estimated dual gaps by 91.42 to 95.23 per cent relative to the slope construction while leaving the exercise policies unchanged. This measures tighter assessment rather than increased policy reward. A common upper payoff separately identifies the expected gain from a feasible terminal adjustment. Individual finite-sample retraining outcomes can still be unfavourable.

Repeating fixed training procedures distinguishes variation between fitted policies from evaluation noise within each pair. In a prespecified computational follow-up, the selected estimator-variance estimates are 16.7 to 34.1 per cent of their reference estimates. The registered variance-difference intervals support the conditional precision comparison under their asymptotic calibration; influential observations limit confidence in its small-sample accuracy.

Historical CPU forecasts fail to transfer. Immediate local calibration reduces mean absolute relative forecast error from 29.065 to 2.841 per cent in the measured session, but incurs additional cost and retains a budget overrun. The study does not establish reliable future expenditure, equal-realised-cost superiority, market profitability or generalisation beyond the retained specifications.

## 3. Reading and replication

The [main paper (PDF)](paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf) is the principal research document, with a [self-contained LaTeX source](paper/main.tex). The [numerical supplement](paper/numerical_results.md) provides the same nine tables in Markdown, with a [machine-readable summary](paper/numerical_results.json). The [research design](research_design.md) identifies the completed questions and the unmet empirical criterion.

The [reproduction instructions](../../docs/reproducing_paper.md) provide a single command for the public checks after installation. The [numerical-validation notebook](notebooks/01_numerical_validation.ipynb) introduces the baseline pricing checks. The [conditional-research notebook](notebooks/27_conditional_research_replication.ipynb) executes the funded-model example, exact finite-law probability checks and released-summary audit. It reconstructs Table 9 from the eighteen workloads in the [CPU measurement supplement](paper/cpu_timings.md). Both notebooks distinguish stipulated mathematical examples from financial evidence.

The [replication guide](replication/README.md) describes installation, the public executable example and the input schema for a new authorised experiment. The [licensed-input guide](replication/licensed_reproduction.md) traces each table and additional quantitative findings to the original experiment entry points and specifies the archive requirements for exact replay. Python implements the model, simulation, regression, inference and experiment orchestration. The C++ propagation kernel has a Python/SciPy reference and cross-language checks. Optional outward-rounded verification uses a separate python-flint environment.

## 4. Evidence and publication

I publish derived findings and reviewed original code and writing. Raw provider exports and observation-level subscription inputs remain private. The public example uses explicitly stipulated mathematical quantities and does not reproduce the financial tables. Exact financial replay requires the original licensed input and evidence archive; the public summary checker verifies transcription and selected arithmetic, not a rerun of the experiments.

The [publication policy](../../docs/publication_policy.md) and [licensing scope](../../LICENSING.md) govern the release. Established mathematical methods are attributed in the paper and [literature comparison](paper/contribution_review.md). The contribution is the controlled argument and evidence for the stated computational questions, without a claim of a new general pricing theorem.

The [market-validation data plan](data_plan.md) restricts further acquisition to free sources and existing authorised access without additional payment. The [Cboe demonstration audit](cboe_sample_audit.md) provides an executable format check, while independent market validation remains unresolved.
