# Observation admission and available data

## 1. Retained design

I assess whether retained observations support the original market-pricing comparison. The registered sample contains fifteen contract-date groups over five sessions, 21–25 September 2026, each joining a call, a put and a common underlying snapshot. The primary screen requires quote ages and option–underlying timing differences within one second, with uniquely resolved quote states. Five-, fifteen- and sixty-second tolerances were specified as descriptive sensitivities and do not replace the primary rule.

Source for this September admission analysis: LSEG Workspace, accessed through the University of St Andrews subscription; my calculations. I identify the observation period separately from the acquisition vintage. The Cboe demonstration discussed below is a different source and retains its own permissions.

The [aggregate results](data_admission.json) reconstruct previously inspected adapter, configuration and case records. They are not a fresh confirmation sample. The screen uses [prospective_inputs.py](../src/qf_research/market_data/prospective_inputs.py) and [quote_review.py](../src/qf_research/market_data/quote_review.py). Observation-level records remain private.

## 2. Mechanical coverage

The joint count requires quote checks, mechanical funding checks at their original 300-second tolerance, successful dated construction and no recorded reference discrepancy.

| Quote tolerance, seconds | Groups passing quote checks | Joint groups | Joint sessions |
| ---: | ---: | ---: | ---: |
| 1, primary | 0 | 0 | 0 |
| 5, descriptive | 7 | 3 | 1 |
| 15, descriptive | 9 | 3 | 1 |
| 60, descriptive | 9 | 3 | 1 |

All fifteen calls and fifteen puts pass individual price screens and have resolved states. Underlying prices pass in twelve groups and underlying states resolve in nine. These counts reuse one underlying snapshot across three strikes per date: only five distinct daily snapshots exist, four price-valid and three resolved. Wider clock tolerances cannot resolve conflicting or ambiguous latest states.

The separately specified funding tolerances give the following joint counts, each out of fifteen groups.

| Quote tolerance, seconds | Funding 60 seconds | Funding 300 seconds | Funding 1,800 seconds |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 |
| 5 | 0 | 3 | 3 |
| 15 | 0 | 3 | 3 |
| 60 | 0 | 3 | 3 |

Increasing quote tolerance from five to fifteen seconds adds two quote-eligible groups but no joint group. Increasing funding tolerance from 300 to 1,800 seconds adds none. All three joint candidates occur on 21 September and represent one session, not three independent dates.

## 3. Financial interpretation

The primary sample remains empty. Relaxed mechanical coverage does not estimate pricing accuracy, policy performance or profitability. I report no significance tests for these dependent observations. The retained curve construction does not verify its historical conventions; archive completeness, quote synchronisation and equal-time ordering also remain unresolved. Consequently, these observations support an admission diagnostic but no independent market-validation estimate.

A separate empirical extension would require documented contract identity, strike, expiry, exercise style, deliverable, multiplier and settlement; matched option and underlying quotes with timestamp semantics; dividend amounts, ex-dates, payment dates and information dates; and dated discount factors for every settlement obligation. A complete documented provider curve could replace the inaccessible construction. Treasury par yields and realised overnight fixings do not by themselves supply that curve. New observation rules and comparisons would need to be fixed before outcomes guide selection. Free sources or existing authorised access would be required, without a data purchase or entitlement upgrade.

## 4. Cboe demonstration audit

I separately audited Cboe's [free EOD demonstration](https://datashop.cboe.com/download/sample/217), described on its [product page](https://datashop.cboe.com/option-eod-summary) and [version 1.1 specification](https://datashop.cboe.com/documents/Option_EOD_Summary_Layout.pdf). The archive retrieved on 30 September matched the retained 14 September archive. Its two CSV variants describe the same 25 August 2023 demonstration and are audited separately.

Each variant has 32,672 rows, 34 columns and 7,630 SPY rows, with no malformed row widths or additional duplicates under the contract identity key. There are 3,815 SPY call-put pairs with non-crossed option quotes and a common positive underlying bid and ask, including 3,073 pairs with positive bids for both options. These are structural diagnostics across all sample expiries, not a validation set.

The specification assigns US Eastern time to the `1545` fields, with 12:45 on early-close days. It identifies open interest as start-of-day information and warns that zero values can encode missing observations or calculations. The deprecated `delivery_code` does not establish contract deliverables. Snapshot labels do not identify last-update times. Contract, dividend and funding evidence remain insufficient for the original comparison. The [auditor](../scripts/audit_cboe_snapshot.py) preserves exact inputs and never combines members automatically:

```sh
python scripts/audit_cboe_snapshot.py \
  --input data/private/cboe/option_eod_sample.zip \
  --member UnderlyingOptionsEODCalcs_2023-08-25_no_cgi_subscription.csv \
  --underlying SPY \
  --output data/private/cboe/reproduction/format_audit.json
```

## 5. Reproduction and closure

Public aggregate checks require no private observations:

```sh
python scripts/summarise_market_admission.py --check-summary paper/data_admission.json
```

A reader lawfully holding the original archive can reconstruct the counts using a new private output directory:

```sh
python scripts/summarise_market_admission.py \
  --archive-root /path/to/authorised/archive \
  --output data/private/reproduction/admission_summary
```

The script validates retained evidence and writes `summary.json`, without exporting quotes, strikes, identifiers, rates or fitted quantities. Free download does not establish redistribution permission. Raw provider inputs remain private under the [publication policy](../docs/publication_policy.md) and [licensing scope](../LICENSING.md). This assessment closes the empirical attempt within the [completed conditional study](../README.md#1-research-question-and-completion); no further acquisition is pending.
