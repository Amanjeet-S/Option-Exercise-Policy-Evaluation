# Observation admission and timing sensitivity

## 1. Question and retained design

I examine whether the observations already retained for the original market-validation attempt can support a pricing comparison without additional data acquisition. The registered sample contains fifteen contract-date groups over five sessions, 21–25 September 2026. Each group joins a call, a put and a common underlying snapshot. The primary rule requires quote ages and option–underlying timing differences within one second, together with a uniquely resolved quote state. The five-, fifteen- and sixty-second alternatives were retained as descriptive sensitivities. I do not substitute them for the primary rule after inspecting the outcomes.

This supplement reconstructs counts from the original adapter audit, configuration and case records. The records were already inspected, and the diagnostic thresholds were previously specified. This is a descriptive reconstruction, not a fresh confirmation sample or a new theoretical contribution. It does not acquire observations, refit volatility or calculate a new option-price prediction. The [machine-readable results](data_admission.json) contain aggregate counts and source hashes. The underlying observations and detailed review records remain private. The screen is implemented in [prospective_inputs.py](../../../src/qf_research/market_data/prospective_inputs.py), with quote-state and timing checks in [quote_review.py](../../../src/qf_research/market_data/quote_review.py).

## 2. Descriptive mechanical coverage

The timing allowance changes which groups pass the mechanical quote checks. Funding is assessed separately at its original primary allowance, and a successful dated construction must be present. The joint column also requires no recorded reference discrepancy.

| Quote timing allowance, seconds | Groups passing quote checks | Groups also passing funding, construction and reference checks | Distinct sessions represented in the joint count |
| ---: | ---: | ---: | ---: |
| 1, primary | 0 | 0 | 0 |
| 5, descriptive | 7 | 3 | 1 |
| 15, descriptive | 9 | 3 | 1 |
| 60, descriptive | 9 | 3 | 1 |

All fifteen calls and fifteen puts pass their individual price screens and have resolved quote states. Underlying prices pass in twelve groups, and the underlying state is uniquely resolved in nine. Those group counts contain repeated information: the same underlying snapshot is shared across the three strikes on each date. There are only five distinct daily underlying snapshots, of which four pass the price screen and three have resolved states. Increasing the timing allowance cannot resolve a conflicting or ambiguous latest state.

The mechanical primary funding screen uses a 300-second allowance. The separately specified funding sensitivities use 60 and 1,800 seconds. Their intersections with the quote checks are as follows.

| Quote allowance, seconds | Joint count at 60-second funding allowance | Joint count at 300-second funding allowance | Joint count at 1,800-second funding allowance |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 5 | 0 | 3 | 3 |
| 15 | 0 | 3 | 3 |
| 60 | 0 | 3 | 3 |

All denominators remain fifteen groups. Increasing the quote allowance from five to fifteen seconds adds two quote-eligible groups, but no jointly eligible group. Increasing the funding allowance from 300 to 1,800 seconds adds none. These intersections describe the retained measurement conditions; they do not identify a causal explanation for the coverage pattern.

The mechanical primary funding screen and dated construction succeed on one session only. All three joint candidates under the descriptive timing alternatives belong to 21 September. They provide one date of coverage, rather than three independent market sessions. Wider quote allowances therefore do not remove the funding bottleneck or restore the registered five-session design.

## 3. Interpretation

The primary outcome remains zero admitted observations. The relaxed counts describe sensitivity of mechanical admission to a declared clock tolerance; they do not measure predictive accuracy, pricing performance or market profitability. A non-empty relaxed subset is not an independent validation result. I report no confidence intervals or significance tests for these dependent observations from five sessions.

Successful curve construction and an empty reference-issue list have narrower meanings than verified financial inputs. The retained construction does not certify its historical conventions. The required evidence concerning archive completeness, quote synchronisation and equal-time ordering is also unresolved. I do not infer that evidence from non-crossed prices, a working data connection or a successful numerical calculation. The available sample supports the admission diagnostic reported here, but does not establish the market-pricing validation originally sought.

The free Cboe demonstration supports a separate [format audit](../cboe_sample_audit.md). Its sample status and missing accompanying financial evidence prevent it from replacing this registered sample. No provider demonstration is reclassified as an independent validation set.

## 4. Reproduction and completion

The [summary script](../../../scripts/summarise_market_admission.py) verifies the retained inputs and constructs these counts without exporting quotes, strikes, instrument identifiers, rates or fitted quantities. A reader with lawful access to the original archive can reproduce the summary from the repository root:

```sh
python scripts/summarise_market_admission.py \
  --archive-root /path/to/authorised/archive \
  --output data/private/reproduction/admission_summary
```

Use a new output directory for each run. The result is written to `summary.json` within that directory. The command requires the original private input bundle and registration; it does not download data or request a subscription. Its output remains private until reviewed. The public result file permits inspection of the released aggregate findings, but cannot establish the validity of withheld observations by itself.

This diagnostic completes the declared assessment of the retained observation records. The [completed project](../completion.md) consists of the conditional computational paper, numerical implementation, verification, reproduction package and this unsuccessful empirical-admission assessment. No data purchase or further acquisition is pending. A separately specified empirical extension would require additional usable evidence, without changing the outcome of the original screen.
