# Historical numerical replay

## 1. Scope

I distinguish a fresh calculation from reconstruction of a saved result. The [machine-readable record](historical_replay.json) identifies the completed replays and the remaining studies. Detailed financial records remain private. The public record contains verification metadata and coverage counts, without provider observations, fitted coefficients or simulated financial paths.

## 2. Completed calculations

I repeated the complete value-martingale study from the preserved financial specifications and frozen policies. It covers thirteen model cases, 208 policies and 624 assessments across the baseline, coarse and fine constructions. Every retained numerical and structural field agrees exactly with the original run after explicitly excluding measured execution times and runtime metadata. The study summaries underlying Table 1 also agree. This comparison uses no numerical tolerance.

The value-martingale replay uses Python 3.13.9, NumPy 2.3.5 and SciPy 1.16.3, matching the versions in its original launch record. Its single-worker execution differs from the original two-worker schedule. The financial specifications, policies, evaluation sizes and random streams remain unchanged.

I also repeated the complete terminal-adjustment study. Across thirteen model cases and three calendar settings, all 624 combinations of policies and calendars agree exactly with the original numerical records. The complete summary agrees, including the extension reported in Table 2. The comparison covers 286,042 scalar fields and requires exact equality after excluding measured durations, execution timestamps and separately verified artifact hashes. The replay uses one worker. The original terminal-adjustment launch record does not identify interpreter or library versions, so it does not support a separate claim of environment-version identity.

Source and input verification covers 23,472 files across 89 manifests. These byte checks establish consistency with the preserved archive; they do not establish the financial validity of its inputs.

## 3. Reconstructed ancillary findings

I recomputed the original market-admission decisions from the retained acquisition snapshots. All fifteen intended observations remain excluded by the primary screen. The adapter files agree byte for byte, and the retained evaluation records and date summaries agree. This exercise obtains no new observations and establishes no market validation.

The delete-one sensitivity results and their tabular output also agree byte for byte. The checks recover the reported terminal-target interval counts and independently reconstruct the polynomial moments and influence-concentration summaries of all nine prospective allocation arms. These are checks of saved outcomes, rather than fresh executions of their financial simulations. The unavailable original market-audit entry point was not executed; its absence is not replaced by a claim of equivalent coverage.

## 4. Remaining numerical execution

| Study | Full historical replay |
| --- | --- |
| Value-martingale assessment | Complete; exact agreement of retained numerical outputs |
| Terminal adjustment | Complete; exact agreement of retained numerical outputs |
| Terminal-target retraining | Pending |
| Repeated-training evaluation | Pending |
| Allocation pilot | Pending |
| Main allocation | Pending |
| Prospective allocation | Pending |
| CPU workloads | Pending |

The [licensed-input guide](licensed_reproduction.md) supplies the original study commands and the comparison procedure. The CPU records support comparison of retained workload identities, with the documented limitation that financial estimates were not saved independently of their timing-contaminated digest. Fresh processor observations cannot reproduce historical durations by specification. Original acquisition, calibration and earlier pricing references remain supplied evidence rather than newly executed computations.
