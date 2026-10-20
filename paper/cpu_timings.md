# Complete-workload CPU forecasts: measurement supplement

Amanjeet Singh

30 September 2026

## 1. Measurement and scope

I release the computational measurements underlying the CPU comparison in Table 9 of the [historical paper](https://github.com/Amanjeet-S/Option-Exercise-Policy-Evaluation/blob/v1.0.0/paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf). The [timing ledger](cpu_timings.json) contains eighteen complete workloads measured on 28 September 2026. Each workload has a fixed number of independent training pairs, with 5,000 and 20,000 training paths per pair, polynomial degree four and a prescribed evaluation count. The three case labels identify the same conditional specifications as the [numerical supplement](numerical_results.md). They do not identify independent market samples.

The two passes execute the nine workloads in opposite orders within one operating session. Counts were fixed before validation. There are 7,956 validation pairs, two sets of sixteen calibration pairs for each workload, and four warm-up pairs. The ledger contains computational durations and work counts only. Financial inputs, subscription observations, fitted policies and simulated payoffs are excluded.

The durations retain exact rational representations of the recorded binary floating-point timers. Summing those representations reproduces the retained arithmetic; it does not increase the physical accuracy of a timer. Decimal rounding is confined to the displayed tables. The fields pass_id and position are zero-based in JSON; the tables display them starting at one.

## 2. Forecast definitions

For workload $i$, let $R_i$ be its fixed number of training pairs. Let $s_i^v$ denote validation setup CPU and $t_i^v$ the sum of CPU charged to both policy fits and their common-path evaluation across all $R_i$ pairs. Actual numerical cost is $C_i=s_i^v+t_i^v$. For calibration scheme $k$, setup costs $s_i^k$ and the sum $t_i^k$ over sixteen complete calibration pairs give

$$\widehat C_i^k=s_i^k+R_i\frac{t_i^k}{16},\qquad K_i^k=s_i^k+t_i^k.$$

The forecast predicts a fresh validation workload. The separate charge $K_i^k$ is the cost paid to obtain that forecast. Pass-start calibration is completed before any validation in its pass. Immediate calibration precedes the corresponding workload. The historical forecast is retained from the earlier allocation plan. Its scalar value is included in the ledger; this supplement does not reconstruct the earlier pilot from which it arose.

I compute the equally weighted mean absolute relative error as

$$\mathrm{MARE}_k=\frac{100}{18}\sum_{i=1}^{18}\frac{|C_i-\widehat C_i^k|}{\widehat C_i^k}.$$

The denominator is forecast cost. The tolerance requires $0.9\leq C_i/\widehat C_i^k\leq1.1$. The upper allowance is $1.1\widehat C_i^k$. Neither rule is a statistical prediction interval. All comparisons below use exact ledger values before display rounding.

<!-- cpu:prediction -->
| Forecast | Within ten per cent | Within upper allowance | Mean absolute relative error (per cent) | Actual/forecast range |
| --- | --- | --- | --- | --- |
| Historical | 1/18 | 18/18 | 29.065 | [0.5154, 1.0782] |
| Pass-start calibration | 16/18 | 16/18 | 3.669 | [0.9237, 1.2081] |
| Immediate calibration | 17/18 | 17/18 | 2.841 | [0.9576, 1.2261] |
<!-- endcpu:prediction -->

A workload with unchanged budget $B_i$ is predicted affordable when $\widehat C_i^k\leq B_i$. A false affordability declaration occurs when $\widehat C_i^k\leq B_i<C_i$. Replacing the point forecast with its upper allowance gives the corresponding allowance-based rule.

<!-- cpu:affordability -->
| Forecast | Point forecast affordable | False point declarations | Upper allowance affordable | False allowance declarations |
| --- | --- | --- | --- | --- |
| Historical | 18/18 | 1/18 | 0/18 | 0/18 |
| Pass-start calibration | 18/18 | 1/18 | 18/18 | 1/18 |
| Immediate calibration | 18/18 | 1/18 | 17/18 | 1/18 |
<!-- endcpu:affordability -->

The historical allowances contain all observed costs but admit no workload at the original budgets. The immediate allowances admit seventeen workloads and make one false affordability declaration. Thus smaller average error, satisfaction of an upper allowance and practical affordability answer different questions.

## 3. Complete workload record

The ledger retains every workload. The table identifies the overrun without concealing the other execution outcomes. CPU, budget and budget excess are measured in seconds; evaluation paths are per independently trained pair.

<!-- cpu:workloads -->
| Pass | Position | Case | Evaluation paths | Training pairs | Actual CPU | Budget | Actual/immediate forecast | Budget excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 6 | 128 | 674 | 37.251 | 54.298 | 0.9782 | 0.000 |
| 1 | 2 | 5 | 4096 | 91 | 30.190 | 57.886 | 1.0161 | 0.000 |
| 1 | 3 | 10 | 256 | 399 | 28.242 | 50.636 | 1.0106 | 0.000 |
| 1 | 4 | 10 | 128 | 438 | 27.736 | 50.636 | 0.9997 | 0.000 |
| 1 | 5 | 5 | 128 | 776 | 43.337 | 57.886 | 0.9844 | 0.000 |
| 1 | 6 | 6 | 4096 | 141 | 46.905 | 54.298 | 1.0204 | 0.000 |
| 1 | 7 | 10 | 4096 | 140 | 54.487 | 50.636 | 1.2261 | 3.852 |
| 1 | 8 | 6 | 256 | 632 | 47.737 | 54.298 | 0.9623 | 0.000 |
| 1 | 9 | 5 | 256 | 687 | 43.954 | 57.886 | 0.9576 | 0.000 |
| 2 | 1 | 5 | 256 | 687 | 44.519 | 57.886 | 1.0345 | 0.000 |
| 2 | 2 | 6 | 256 | 632 | 39.981 | 54.298 | 1.0046 | 0.000 |
| 2 | 3 | 10 | 4096 | 140 | 44.025 | 50.636 | 1.0098 | 0.000 |
| 2 | 4 | 6 | 4096 | 141 | 46.406 | 54.298 | 1.0180 | 0.000 |
| 2 | 5 | 5 | 128 | 776 | 43.723 | 57.886 | 1.0192 | 0.000 |
| 2 | 6 | 10 | 128 | 438 | 27.617 | 50.636 | 0.9841 | 0.000 |
| 2 | 7 | 10 | 256 | 399 | 28.016 | 50.636 | 1.0013 | 0.000 |
| 2 | 8 | 5 | 4096 | 91 | 29.682 | 57.886 | 0.9999 | 0.000 |
| 2 | 9 | 6 | 128 | 674 | 36.466 | 54.298 | 0.9831 | 0.000 |
<!-- endcpu:workloads -->

In the first pass, case 10 with 4,096 evaluation paths consumes 54.487 CPU seconds against a 50.636-second budget. Its excess is 3.852 seconds when calculated before rounding. The immediate prediction understates its cost, producing a ratio of 1.2261. The matching workload in the second pass lies below budget. This within-session difference does not identify a causal execution-order effect or establish the probability of a future overrun.

## 4. Computational expenditure

The numerical charge adds validation, both calibration schemes and warm-up. Whole-process CPU additionally includes the measured recording and control overhead. The process clock excludes interpreter startup, pre-launch verification and post-run manifest hashing. Elapsed time uses a different clock and is not added to CPU time.

<!-- cpu:costs -->
| Component | Seconds |
| --- | --- |
| Validation numerical CPU | 700.274 |
| Pass-start calibration CPU | 44.623 |
| Immediate calibration CPU | 44.323 |
| Warm-up CPU | 0.332 |
| Total numerical CPU | 789.552 |
| Other measured process CPU | 22.106 |
| Whole-process CPU | 811.658 |
| Whole-process elapsed time | 842.795 |
<!-- endcpu:costs -->

I also compare each original workload budget with $C_i$, $C_i+K_i^{\mathrm{pass}}$, $C_i+K_i^{\mathrm{immediate}}$ and $C_i+K_i^{\mathrm{pass}}+K_i^{\mathrm{immediate}}$. These calculations charge calibration to its associated workload; the shared warm-up and other process overhead remain separate in the whole-run accounting.

<!-- cpu:charged -->
| Charge compared with the unchanged workload budget | Affordable workloads |
| --- | --- |
| Validation only | 17/18 |
| Validation and pass-start calibration | 17/18 |
| Validation and immediate calibration | 17/18 |
| Validation and both calibration schemes | 14/18 |
<!-- endcpu:charged -->

Charging both calibration schemes reproduces the experiment as performed. It differs from deploying either scheme alone. The two single-scheme calculations each leave seventeen affordable workloads, whereas charging both leaves fourteen. These are reconstructed accounting comparisons, not additional measured executions with one calibration scheme removed.

## 5. Verification and interpretation

From the repository root, run:

```sh
python3 paper/verification/cpu_timings.py
```

The standard-library [verifier](verification/cpu_timings.py) reconstructs forecasts from setup costs and complete-pair totals, recomputes the prediction errors, classifications and expenditure, and checks agreement with the existing [released summaries](numerical_results.json) at their published precision. It also checks these five table renderings. Its [tests](verification/test_cpu_timings.py) reject changes in the timing arithmetic, omitted workloads and unexpected fields. No simulation, data download or optional numerical package is needed.

The measurements originate in the repository's [fixed-workload study](../scripts/study_cpu_prediction.py), its [retained-evidence audit](../scripts/audit_cpu_prediction.py) and [report construction](../scripts/report_cpu_prediction.py). This supplement exposes their computational aggregation for independent inspection; it introduces no new estimator or research experiment. Private export records preserve the selected source files and their hashes. This public ledger cannot itself verify the chronology of withheld records or regenerate the original workload timings.

A repeat on another machine produces new timing observations. The present arithmetic check establishes what the retained measurements imply, without proving stable future cost, probabilistic budget protection or financial validity. The [replication guide](../replication/README.md) separately explains execution of the mathematical example and the requirements for a financial replay. The [publication policy](../docs/publication_policy.md) continues to exclude restricted financial observations.
