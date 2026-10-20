# Reproducing the Option Exercise-Policy Evaluation study

## 1. What can be checked publicly

The [final paper](../paper/Assessing_Bermudan_Exercise_Policies.pdf), its [source](../paper/main.tex) and [aggregate results](../paper/revision_results.json) concern public mathematical benchmarks, repeated-sampling coverage and verified conditional integration. The [historical PDF](https://github.com/Amanjeet-S/Option-Exercise-Policy-Evaluation/blob/v1.0.0/paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf) is available in the `v1.0.0` archive. Sections 2 and 3 describe the historical release in a separate `v1.0.0` checkout. Sections 4 to 6 distinguish its lightweight checks from the current PDF build and optional arithmetic environment. Sections 7 to 15 describe the current public-model experiments. The historical source manifest belongs to `v1.0.0`; the separate current inventory covers the reader package below. Each new study instead records its own scoped source and evidence identities.

For the current reader package, install the research and notebook requirements in the current checkout's environment and run:

```sh
python -m pip install -r requirements.lock -r requirements-notebooks.lock
python -m pip install --no-deps -e .
python scripts/reproduce_current_study.py --output data/private/reproduction/current_review
```

The output directory must be new. The command checks the separate `replication/current_source_manifest.json`, verifies current aggregate arithmetic and manuscript transcription, runs selected numerical and execution tests, and executes a retained notebook copy in a fresh kernel. `--with-native --compiler g++` adds the independent C++ verification. A failed step leaves an incomplete record and its logs. Source and notebook files are not overwritten. The command reads no provider observations and makes no data download. It checks the stated components; it does not regenerate every coverage repetition, fitted policy or conditional-integration ledger in the paper. Sections 7 to 15 describe those larger model experiments.

The later two-asset monetary, Gaussian-compression and bounded-correction investigations remain exploratory extensions outside this paper and its reader package. Their full records stay local; a preparation script alone is not a complete assessment reproduction.

The historical [paper](https://github.com/Amanjeet-S/Option-Exercise-Policy-Evaluation/blob/v1.0.0/paper/Exercise_Policy_Loss_under_Computational_and_Financial_Constraints.pdf) reports the earlier conditional computational investigation. Its separate reproduction command checks the historical summaries, numerical example and two explanatory notebooks. It requires no provider account, licensed input archive or paid data. It does not read, copy or export restricted raw financial data.

| Material | Public reproduction | Interpretation |
| --- | --- | --- |
| Finite probability examples | Recompute exact rational identities and counterexamples | Checks the examples, not a proof of every general theorem |
| Numerical algorithms | Execute the stipulated funded-put fixture and baseline notebook | New model-conditional calculations using explicitly mathematical inputs |
| Tables 1 to 8 | Check released summaries, selected arithmetic and agreement with the historical LaTeX tables | Does not regenerate the original financial simulations |
| Table 9 and CPU accounting | Reconstruct forecasts, errors, affordability and expenditure from eighteen released workload records | Uses retained timings and historical forecast inputs; does not rerun the original workloads or pilot |
| Observation coverage supplement | Recalculate totals and timing intersections from released daily counts | Checks aggregate arithmetic only; reconstruction from private observation records requires the original authorised archive |
| Original financial experiments | Implementation and input requirements are documented in the [licensed-input guide](../replication/licensed_reproduction.md) | Exact replay requires lawful access to the original input, design, source and evidence archive; a provider subscription alone is insufficient |

The public fixture is not a replacement dataset for the financial tables. Its stipulated inputs and simulated outputs are labelled accordingly. The separate [historical replay record](../replication/historical_replay.md) documents all eight completed registered replays. Seven reproduce retained financial outputs exactly; the CPU replay matches saved workload identities, with financial estimates unavailable for comparison. Original acquisition and calibration were not rerun. The independent market-validation attempt is closed without a performance estimate. The [completion record](../README.md) distinguishes that historical completion from the current revision; no paid acquisition is required for the new mathematical benchmarks.

## 2. Install the historical release

Use Python 3.11 or 3.13, the versions checked in continuous integration. Create a separate historical checkout so that its inventory and paper source remain consistent:

```sh
git clone --branch v1.0.0 --single-branch https://github.com/Amanjeet-S/Option-Exercise-Policy-Evaluation.git Option-Exercise-Policy-Evaluation-historical
cd Option-Exercise-Policy-Evaluation-historical
python3 -m venv .venv
source .venv/bin/activate
python replication/verify_sources.py
python -m pip install -r requirements.lock -r requirements-notebooks.lock
python -m pip install --no-deps -e .
```

On Windows, use `py -3.13 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. Subsequent `python` commands are the same. Installation downloads the declared software dependencies. The reproduction command itself requires no data download or network service.

The source check uses only the Python standard library. A missing or altered listed file is an error. Re-clone the intended revision rather than changing recorded hashes to suppress a mismatch. The inventory checks consistency of its declared files; it does not independently authenticate the repository or establish mathematical correctness.

## 3. Run the historical public reproduction

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
python replication/verify_sources.py
python -m unittest discover -s paper/verification -v
python paper/verification/numerical_results.py
python paper/verification/paper_results.py
python paper/verification/cpu_timings.py
python scripts/summarise_market_admission.py --check-summary paper/data_admission.json
```

The [observation-coverage supplement](../paper/data_admission.md) preserves the original empty primary sample and reports the previously specified timing sensitivities. Its public checker uses aggregate counts only. It does not open the private archive, verify the underlying observations or establish market validity.

Return to the current checkout before building its PDF. The [current PDF](../paper/Assessing_Bermudan_Exercise_Policies.pdf) is compiled from the self-contained [main.tex](../paper/main.tex). Run `make -C paper` with a TeX distribution and `latexmk`, or compile the source in the document editor. The current tree includes only the final research-paper PDF; the earlier paper remains in the historical tag. The [completion record](../completion.json) retains the delivered PDF and source hashes; PDF byte identity can change with compiler versions and build dates. PDF compilation is optional for the reader checks, and its output is outside the source inventory. The separate python-flint environment is also optional for the default reader command.

## 5. Interval arithmetic

The outward-rounded arithmetic checks use a separate environment with `python-flint==0.9.0`. The commands below verify the current inventory; in the historical `v1.0.0` checkout, omit `--manifest replication/current_source_manifest.json` to verify its own retained inventory. From the repository root, create and activate the arithmetic environment:

```sh
python3 -m venv .venv-certification
source .venv-certification/bin/activate
python -m pip install -r requirements-certification.lock
python replication/verify_sources.py --manifest replication/current_source_manifest.json
mkdir -p data/private
python -m unittest discover -s tests -p test_certify_gain_replay.py -v
python scripts/test_certified_residual.py
```

On Windows, create the environment with `py -3.13 -m venv .venv-certification`, activate with `.venv-certification\Scripts\Activate.ps1` and use `New-Item -ItemType Directory -Force data/private` in place of `mkdir -p`. No installation of the simulation package is needed in this environment.

The first suite contains thirty finite-innovation replay checks; the second contains twenty-two residual-enclosure checks. They cover exact interpretation of binary inputs, outward rounding, cash-event and exercise boundaries, conditional expectations, global tails, ambiguous decisions and rejection of altered evidence. All inputs are stipulated mathematical fixtures. These tests check the certification implementation without certifying the paper's withheld financial inputs, general statistical coverage or market validity. A separate continuous-integration job runs both suites with the pinned arithmetic dependency.

## 6. Further verification

The [development guide](development.md) covers the full test suite and optional C++ checks. The [detailed replication guide](../replication/README.md) maps the numerical implementation. The [licensed-input guide](../replication/licensed_reproduction.md) traces the paper's tables and additional quantitative findings to their original designs, specifies the required archive and distinguishes complete runs from representative replays. Those historical scripts do not distribute their original inputs.

## 7. Revised public-model experiments

For a short calculation connecting the mathematics to the implementation, use the [pricing and stopping notebook](../notebooks/pricing_and_stopping.ipynb). Install `requirements.lock` and `requirements-notebooks.lock` in the current checkout's environment, then run:

```sh
python scripts/execute_release_notebooks.py --foundations
```

This executes only the current notebook in a fresh kernel. It checks exact finite identities, analytical sensitivities, same-calendar numerical refinements and independent policy evaluation. If a C++17 compiler is available, it also builds the native tree in a temporary directory and compares its values with Python. The calculation uses no provider observations. Source hashes and simulation counts are retained in notebook metadata; `--write` saves outputs only after successful execution. This command does not execute the historical reproduction inventory or the larger experiments below.

Return to the current revised checkout before running the following studies. Create its research environment locally, rather than importing the project installed from the historical checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

For the later certified studies, create a separate `.venv-certification` in the current checkout and install `requirements-certification.lock`, following Section 5. Verify this checkout with `python replication/verify_sources.py --manifest replication/current_source_manifest.json`. Each experiment retains its own registered source and evidence identities; the current inventory does not replace them.

The new studies use published mathematical model parameters and independent simulation streams. They neither read provider observations nor reproduce market quotations. Output directories must be new. In the research environment, run:

```sh
python scripts/study_multivariate_matrix.py --output data/private/revision_reproduction/matrix
python scripts/validate_gap_coverage.py --output data/private/revision_reproduction/coverage
python scripts/validate_clipped_gap_coverage.py --original data/private/revision_reproduction/coverage --output data/private/revision_reproduction/clipped_coverage
python scripts/certify_multivariate_gap.py prepare --outer-paths 32 --output data/private/revision_reproduction/gap_inputs.json
```

The matrix retains prices, matched expiry gains and a fitted exercise-margin surface. The coverage diagnostic has an explicitly modified two-right calendar. The clipped comparison uses a separately seeded pilot, frozen caps and analytical tail allowances. Its concentration guarantees concern ideal iid sampling; floating-point computation and quadrature targets are distinct numerical questions. Counts and means reproduce on the recorded stream layout, while timings and roundoff can differ.

In the separate certification environment, run:

```sh
python scripts/refine_multivariate_gap.py --input data/private/revision_reproduction/gap_inputs.json --target 1/100 --output data/private/revision_reproduction/gap_comparison.json
python -m unittest discover -s tests -p test_certify_max_hinge.py -v
python -m unittest discover -s tests -p test_certify_multivariate_gap.py -v
python -m unittest discover -s tests -p test_refine_multivariate_gap.py -v
```

The enclosure endpoints are exact rational records. They cover ideal conditional functions and real gain algebra at supplied states; they do not certify the simulator law or sampling coverage. The comparison reports both achieved widths and every charged integral. Different achieved widths must not be described as equal accuracy. The larger refinement study retains a distinct frozen design and reports a cold uniform baseline without unnecessary coarse initialisation.

## 8. Monetary decisions and a certified reference

The two-right decision experiment fixes five exercise policies and three monetary tolerances. In the certification environment, create the independent population reference and the separately executable analytical bounds:

```sh
python scripts/certify_policy_loss_reference.py --output data/private/revision_reproduction/decision_reference.json
python scripts/certify_policy_loss_reference.py --analytic-only --output data/private/revision_reproduction/decision_analytic.json
python -m unittest discover -s tests -p test_certify_policy_loss_reference.py -v
```

Return to the research environment and run:

```sh
python scripts/study_policy_certification.py --reference data/private/revision_reproduction/decision_reference.json --analytic-reference data/private/revision_reproduction/decision_analytic.json --output data/private/revision_reproduction/decision_study
python -m pytest tests/test_policy_certification.py -q
```

The experiment uses 256 repetitions, an independent frozen 8,192-path pilot where needed, and five nested evaluation looks. Its probability allocation covers the five policies, three simulation assessors and five looks jointly. The three tolerances reuse each upper limit. The first successful look determines the charged assessment cost; all later looks are retained for validation. Finite-threshold terminal-call bounds can certify without generating paths.

The output separates direct interval-certified shortfalls, root-free bounds, Monte Carlo certification frequencies and false acceptance anywhere in a repetition. A failed Monte Carlo assessment is unresolved, whereas a certified lower shortfall bound above the tolerance rejects it. Zero observed false acceptance does not establish coverage of every upper limit. Exact rational reference endpoints control deterministic decisions; simulation and confidence arithmetic remain floating-point calculations.

Standalone phase costs include the observed assessment prefix and one retained frozen setup measurement. Deterministic comparisons report complete executable family work, conservatively charged to each standalone decision. The overall validation-study duration also includes expected-gap reference integration and per-arm evidence writing. Timings can change across runs and machines. All inputs are public mathematical parameters, and these commands neither acquire nor reveal provider observations. The historical inventory and current reader inventory remain separate; each experiment preserves its registered sources.

## 9. Independently fitted refinement and moment benchmarks

The replicated design uses eight independently fitted five-asset policies, with root seeds 2026100410 to 2026100417. Each fit uses 50,000 policy-training paths, 50,000 separate value-fitting paths and 256 independent assessment paths. The maximum required path-gap width is 0.01. The prior design fixes these quantities, the source hashes and balanced execution orders before training. An independently fitted policy and value-function pair is the replication unit; the 2,048 assessment paths do not constitute 2,048 independently fitted models.

In the research environment, prepare a new directory:

```sh
python scripts/study_replicated_gap_refinement.py prepare --directory data/private/revision_reproduction/replicated_refinement
```

In the certification environment, assess cold uniform and gain-directed refinement:

```sh
python scripts/study_replicated_gap_refinement.py assess --directory data/private/revision_reproduction/replicated_refinement
```

Return to the research environment to fit complete, unsorted degree-two and degree-three polynomial continuation functions on the matched value-fitting streams:

```sh
python scripts/study_replicated_gap_refinement.py polynomial --directory data/private/revision_reproduction/replicated_refinement
```

Finally, in the certification environment, enclose both the complete polynomial martingales and hybrids with the exact discounted terminal max-call payoff:

```sh
python scripts/study_replicated_gap_refinement.py certify-polynomial --directory data/private/revision_reproduction/replicated_refinement
python -m unittest discover -s tests -p test_certify_polynomial_dual.py -v
```

The manuscript aggregate can then be regenerated from the complete retained ledgers in the research environment. The following writes a new derived summary in the reproduction directory without changing the paper:

```python
import json
from pathlib import Path
import sys
sys.path.insert(0, "paper/verification")
from revision_results import build_replicated_refinement

folder = Path("data/private/revision_reproduction/replicated_refinement")
def read(path):
    return json.loads(path.read_text())
artifacts = {
    i: {key: read(folder / f"fit_{i:02d}" / (name + ".json"))
        for key, name in (("refinement", "result"),
                          ("polynomial", "polynomial"),
                          ("verified_polynomial", "verified_polynomial"))}
    for i in range(8)
}
summary = build_replicated_refinement(
    read(folder / "summary.json"), fit_artifacts=artifacts,
    design=read(folder / "design.json"))
with (folder / "public_summary.json").open("x") as output:
    output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
```

To regenerate the standalone figure from the reviewed manuscript aggregate without rewriting earlier figures, run:

```sh
python scripts/plot_research_revision.py --verified-cost-quality-only --revision-results paper/revision_results.json --output data/private/revision_reproduction/verified_cost_quality.pdf
```

Each verified arm owns its moment-factor cache and is charged for its construction, arithmetic and any terminal integrations. Common policy and outer-path preparation enter the standalone polynomial budgets; hinge regression is excluded. Hybrid setup costs are reconstructed from measured interior regression components rather than a separately rerun hybrid fit. Pricing costs exclude process start-up and file serialisation; the private completion records separately retain complete validation-workflow costs. These are observations of the recorded implementation and machine, not transferable CPU forecasts.

Completed assessments can be checked and resumed without rerunning their fitted models, after verification of the design, sources, inputs, completion hashes and derived statistics. Interrupted or failed attempts are preserved and cannot be silently restarted. Use a new directory for an independent reproduction.

The verified intervals cover conditional functions and stopped-gap arithmetic at the retained binary states. They do not enclose the simulator law or certify statistical coverage. Paired gap differences compare the relative expected slack of assessment martingales under the maintained model, keeping each exercise policy fixed. Per-fit sampling diagnostics and variation across the eight fitted models are reported separately. These commands use published mathematical parameters and simulated paths, and reveal no subscription observations. The frozen historical release inventory remains separate from the current inventory.

## 10. Independent evaluation of retained fits

The larger-sample follow-up starts from the completed eight-fit study in section 9. It retains the original fitted policies, hinge functions and degree-three polynomial coefficients. Each fit receives 1,024 new outer paths from child stream `[3]`; the original training, value-fitting and outer streams remain `[0]`, `[1]` and `[2]`. These are independent samples rather than nested prefixes. The degree-three exact-terminal hybrid is selected from the preceding comparison before the new sample is drawn.

In the research environment, prepare a new version and independently verify the complete new path streams and stopping indices:

```sh
python scripts/study_frozen_gap_extension.py prepare --source-study data/private/revision_reproduction/replicated_refinement --directory data/private/revision_reproduction/frozen_extension
python scripts/study_frozen_gap_extension.py verify-streams --source-study data/private/revision_reproduction/replicated_refinement --directory data/private/revision_reproduction/frozen_extension
```

In the certification environment, execute the three assessments sequentially:

```sh
python scripts/study_frozen_gap_extension.py assess --source-study data/private/revision_reproduction/replicated_refinement --directory data/private/revision_reproduction/frozen_extension
```

The prior record fixes the source-study hashes, stream, count, precision, maximum width and arm order before sampling. All six arm permutations followed by two rotations give each arm two or three occurrences at each position. Every retained fit must complete. A failed version preserves its failure and any completed arm ledgers; it cannot be silently restarted. Completed fits may be checked and resumed after their full ledgers and completion hashes have been verified. Preparation always requires a fresh directory.

To review every path ledger and export a compact summary, run this in the research environment after the preceding commands. The original public summary is the one generated from the same reproduced study in section 9, including that run's measured setup costs.

```python
import json
from pathlib import Path
import sys

sys.path.insert(0, "paper/verification")
from revision_results import build_frozen_extension

prior = Path("data/private/revision_reproduction/replicated_refinement")
folder = Path("data/private/revision_reproduction/frozen_extension")
artifacts = {}
for i in range(8):
    artifacts[i] = {
        key: json.loads((base / f"fit_{i:02d}" / name).read_text())
        for key, base, name in (
            ("result", folder, "result.json"),
            ("input", folder, "inputs.json"),
            ("polynomial_view", folder, "polynomial_view.json"),
            ("original_input", prior, "inputs.json"),
            ("original_polynomial", prior, "polynomial.json"),
        )
    }
summary = build_frozen_extension(
    json.loads((folder / "summary.json").read_text()),
    fit_artifacts=artifacts,
    design=json.loads((folder / "design.json").read_text()),
    original_replicated_refinement=json.loads((prior / "public_summary.json").read_text()),
)
with (folder / "public_summary.json").open("x") as output:
    output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
```

Conditional sampling standard errors hold the eight retained fits and earlier comparator selection fixed. The standard error of their equally weighted mean is the square root of the sum of squared within-fit standard errors, divided by eight. Matched hybrid-minus-hinge differences use the variance of the differences, rather than adding the separate arm variances. Dispersion between fit means is reported separately. Neither diagnostic is a certified confidence interval; exact arithmetic intervals enclose sample means, not population expectations.

Fresh expenditure includes the new outer simulation, frozen-policy evaluation and complete assessment. The reconstructed standalone budget adds retained fitting components, replaces the original outer preparation and excludes all earlier assessments. It therefore does not claim to have rerun fitting. Setup reconstruction and newly incurred expenditure are distinct quantities. All three arms meet the same required maximum width of 0.01, while their achieved widths may differ. These commands generate public-model simulations and disclose no subscription observations.

## 11. Common-width allocation controls

This comparison starts from the completed studies in sections 9 and 10. It reuses all eight retained 1,024-path inputs, exercise policies and hinge functions. No path simulation or fitting is repeated. The cold-common control initialises every conditional bracket at width 0.01; the coarse-common control initialises at width one. Both freeze their initial predictable centres and test the actual signed gap width. If that width exceeds 0.01, a complete global round quarters one common conditional target and queries every retained bracket exceeding it. Already compliant brackets are retained. There is no candidate-span, exercise-date or dominance pruning. A successful actual gap check ends refinement, even if a conservative sufficient-width bound remains loose. At most sixteen refinement rounds follow initialisation.

In the certification environment, register a new directory before executing any allocation assessment:

```sh
python scripts/study_uniform_gap_allocation.py register --source-study data/private/revision_reproduction/replicated_refinement --extension-study data/private/revision_reproduction/frozen_extension --directory data/private/revision_reproduction/uniform_allocation
python scripts/study_uniform_gap_allocation.py assess --source-study data/private/revision_reproduction/replicated_refinement --extension-study data/private/revision_reproduction/frozen_extension --directory data/private/revision_reproduction/uniform_allocation
```

Registration verifies and binds the original study, the completed larger-sample study and their source hashes. All six arm permutations followed by two rotations give each arm two or three occurrences at each execution position. The adaptive arm is rerun and must reproduce its retained numerical certificates, work counts, frozen fingerprints and refinement histories exactly; only the explicitly listed timer fields are excluded. All three new path intervals must intersect each other and the retained cold-uniform interval. Coarse-common and adaptive initial centres must match. Completed common-width traces are checked by reconstructing the rewards, exact realised hinge values, fixed centres, shared-increment cancellation and signed gap after each round.

A failed or interrupted attempt is retained and cannot be silently restarted. When storage permits, failure records preserve completed arms, completed paths, query traces and elapsed expenditure. If an oracle raises before returning its internal work count, that count is identified as unavailable. Completed fits can be verified and resumed without re-execution. Every fit is required; no failed path or seed is replaced. The 256 MiB storage preflight is a minimum threshold, rather than reserved capacity or a bound on total output. I recommend several gigabytes of free storage before starting, together with sufficient memory for decoded ledgers and their verification. A storage failure requires preservation of the original attempt and an explicitly recorded continuation in a fresh directory. Lossless compression may reduce storage, provided exact bytes and recorded identities remain unchanged.

After assessment, review the complete retained ledgers in the research environment. The preceding public summary must come from the same reproduced larger-sample study and its measured preparation components. The mapping below decodes each fitting unit on demand, rather than retaining all eight complete ledgers simultaneously. It does not bound memory used by an active ledger, its exact arithmetic or replay comparisons:

```python
from collections.abc import Mapping
import json
from pathlib import Path
import sys

sys.path.insert(0, "paper/verification")
from revision_results import build_uniform_allocation

prior = Path("data/private/revision_reproduction/frozen_extension")
folder = Path("data/private/revision_reproduction/uniform_allocation")


def read(path):
    with path.open() as stream:
        return json.load(stream)


class FitArtifacts(Mapping):
    def __len__(self):
        return 8

    def __iter__(self):
        return iter(range(8))

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < 8:
            raise KeyError(index)
        fit = f"fit_{index:02d}"
        return {
            "result": read(folder / fit / "result.json"),
            "input": read(prior / fit / "inputs.json"),
            "retained_result": read(prior / fit / "result.json"),
        }


summary = build_uniform_allocation(
    read(folder / "summary.json"),
    fit_artifacts=FitArtifacts(),
    design=read(folder / "design.json"),
    frozen_extension=read(prior / "public_summary.json"),
)
with (folder / "public_summary.json").open("x") as output:
    output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
```

Fresh expenditure is assessment CPU only, since the input paths already exist. Reconstructed standalone budgets add the retained 1,024-path preparation charge once. Each arm pays for its own initialisation, policy checks, conditional queries, successful oracle retries and result construction. The common-width implementations retain fuller query traces than adaptive, so complete CPU costs compare these implementations, including their differing ledger construction. No estimated bookkeeping cost is subtracted. Conditional-integral and integrand-evaluation counts provide separate work measures. Whole-controller verification and file I/O are reported separately in completion records.

These controls test whether selective allocation improves on stopping-aware common tolerances at the same required maximum signed gap width. They do not establish globally optimal computational allocation. The retained states use public mathematical model inputs only; this route acquires no provider observations and generates no new independent sampling evidence.

## 12. A dominated initial exercise right

For the retained five-asset benchmark at initial state 100 and strike 100, the initial reward is exactly zero and every later reward is non-negative. The strict fitted policy never exercises at date zero. Restricting the dual maximum to dates 1 to 9 preserves the optimal stopping value, while cancelling the first conditional increment from every retained candidate-policy difference. This is a separately identified evaluation functional; the previous comparisons keep their full candidate calendar.

After section 11, reconstruct the restriction in the research environment without new simulation or integration:

```sh
python scripts/certify_restricted_gain.py audit --directory data/private/revision_reproduction/uniform_allocation --extension-study data/private/revision_reproduction/frozen_extension --output data/private/revision_reproduction/restricted_initial_right.json
python -m unittest discover -s tests -p test_certify_restricted_gain.py -v
```

The audit checks recorded completions, registered sources, input identities and complete coarse-common query chains before forming exact rational gain intervals. Conditional row one is never read by the restricted gain calculation. The existing row is used only to reconstruct the full-calendar comparator. Candidate zero is strictly below the retained lower maximum on all original 8,192 paths, and every restricted interval equals its original full interval. A changed candidate set can increase interval width in other examples; endpoint dominance alone does not preserve an earlier width target.

The audit supplies reconstruction evidence, with no new fitting, evaluation sample, conditional oracle or CPU comparison. It neither estimates a population improvement nor provides a clipping-tail allowance. Fresh financial decisions require a separately justified tail bound and an independent evaluation stream after the assessor and cap have been frozen.


## 13. European-value tail feasibility

I provide `scripts/certify_european_gain_tail.py` for the population clipping tail of the restricted European-value evaluator under homogeneous independent GBM. The function `build_european_restricted_tail_bound` accepts exact rational or binary-float parameters, the full calendar starting at zero, and the positive and negative cap allocations. Its default route obtains the required at-the-money maximum-call enclosures from the existing verified integrator. Supplying enclosures or an injected integrator makes their validity an explicit input premise. The returned exact rational endpoints are authoritative.

The component runs in the separate certification environment. Its boundary checks can be reproduced from the repository root with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:tests .venv-certification/bin/python -m unittest test_certify_european_gain_tail
```

The tail bound applies to this evaluator and stopping dates from the first positive right to expiry. It does not certify a fitted hinge gap, generate evaluation paths or establish a monetary policy decision. The ideal-model probability assumptions remain separate from floating-point simulation. I retain the cap-50 feasibility attempts and planning calculations locally; they do not extend the released source inventory or the completed study registrations.


## 14. Direct European-value gain refinement

I compare common and selective refinement of the European-value martingale on the first 32 retained paths from each of the eight fits in sections 9 to 11. The study generates no paths and changes no fitted policy. Its gain uses discounted European terminal-payoff values directly, with exact cancellation of the terminal level and selected candidate. Both arms share exact-zero-reward screening, interval dominance, their conditional-pricing routine, local cache scope and query-record fields. They start with price-width target one and stop at actual gain width 0.01. Common refinement permits sixteen global quartering rounds; selective refinement permits sixteen quartering depths and actual queries per date, with the same minimum request width. Every raw query has at most eight recorded attempts.

In the certification environment, register the comparison against the completed retained studies before assessment:

```sh
python scripts/study_european_gain_allocation.py register --root . --extension-study data/private/revision_reproduction/frozen_extension --control-study data/private/revision_reproduction/uniform_allocation --allocator-module scripts/refine_european_gain.py --tail-module scripts/certify_european_gain_tail.py --additional-source tests/test_refine_european_gain.py --additional-source tests/test_study_european_gain_allocation.py --additional-source tests/test_certify_european_gain_tail.py --directory data/private/revision_reproduction/european_allocation
```

Run the same command with `assess` in place of `register`, preserving every argument. The driver fixes the prefix, balances arm order across fits and binds the original fifteen scientific sources, eight retained inputs, actual imported helper and current numerical sources. Registration also records the Python and python-flint environment. A changed source, input, runtime or design is rejected. Existing attempts cannot be overwritten or silently restarted. Each completed fitted-unit ledger is compressed losslessly and bound by both compressed and uncompressed hashes. A 128 MiB storage preflight is a minimum threshold, not reserved space; retain sufficient additional capacity for the outputs.

The driver reconstructs every query intersection, refinement schedule, complete-round gain and final work count. A separately recomputed reference uses the same pricing oracle at higher precision and target width $10^{-8}$; it checks numerical consistency and is not an independent optimal stopping reference. Assessment CPU includes the API, validation and record construction for each arm. Input reading, source checks, reference work and compression are reported separately. All attempts are charged, and unavailable internal work stays identified as unavailable. These timings concern incremental assessment of retained states, without a new fitting or simulation charge.

Boundary and study-integrity checks can be run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:tests .venv-certification/bin/python -m unittest test_refine_european_gain test_certify_european_gain_tail test_study_european_gain_allocation
```

The retained comparison establishes numerical performance on 256 selected paths. It supplies neither a new independent evaluation sample nor a monetary confidence decision. The European tail component in section 13 belongs to this evaluator; it cannot be transferred to the fitted hinge gains.

## 15. Multivariate monetary decisions

The monetary experiment assesses the eight retained coefficient sets from section 10 on separately generated model paths. It uses the published homogeneous five-asset specification, two assessors, fixed tolerances of 0.5 and 1 price unit, and cumulative looks at 4,096 and 16,384 paths per fit. No market observations or provider account are required. The complete controlled reproduction requires the preceding retained-policy and control records; it cannot be reconstructed from aggregate means alone.

The registration command reads the preceding control source inventory at `data/private/revision/2026-10-05/uniform_allocation_recovery_v1/study/design.json`. For a fresh reproduction, retain the newly reproduced section-11 control design at this location before registration. Preserve any existing historical inventory instead of overwriting it. The numerical source hashes must agree. This dated path identifies a dependency of the current implementation; it does not assign a historical timestamp to the reproduction.

Use the separate certification environment for registration and assessment, and the research environment for path preparation. Set the same thread environment throughout. Replace the executable paths below with the corresponding environments on the reproducing machine.

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
.venv-certification/bin/python scripts/study_financial_decision.py register --root . --retained-study data/private/revision_reproduction/frozen_extension --output data/private/revision_reproduction/financial_decision/study --research-python .venv/bin/python
.venv/bin/python scripts/prepare_financial_decision.py --root . --design data/private/revision_reproduction/financial_decision/study/design.json --retained-study data/private/revision_reproduction/frozen_extension --output data/private/revision_reproduction/financial_decision/preparation
.venv-certification/bin/python scripts/study_financial_decision.py assess --root . --study data/private/revision_reproduction/financial_decision/study --preparation data/private/revision_reproduction/financial_decision/preparation --output data/private/revision_reproduction/financial_decision/assessment
```

Registration records the source and policy-input hashes, both environments and the unconsumed date-generator states before sampling. New streams have keys `(4,j)` for positive dates, with fixed batches of 256 paths. Each exercise decision evaluates the frozen coefficients in an own-state order using products and `math.fsum`. This preserves the coefficients, strict inequalities and tie convention while removing any dependence of rounding on other surviving paths in a matrix product. Non-finite polynomial arithmetic continues; invalid simulated states fail the retained attempt. No fitting stream is sampled or policy refitted.

The private bridge contains binary states, stopping indices, complete generator checkpoints and prefix hash chains. Successful assessments retain complete compressed numerical ledgers. Source changes, altered binary paths, incomplete attempts and unsupported numerical outcomes are rejected. Existing output directories cannot be overwritten. The storage guard requires 256 MiB free during assessment, with additional space required for all retained outputs; it is a threshold rather than reserved capacity.

Exact rational clipped statistics retain the uncertainty of every gain interval. The upper sampling construction uses the midpoint/radius variance enclosure, each assessor's own analytical clipping tail, and failure probability `1/640` for each of 32 fixed upper events. The first acceptable look determines a tolerance's required prefix. All unresolved cases retain the full budget. The guarantee concerns the conditional ideal model and valid observation enclosures; interval pricing alone does not certify floating-point Gaussian simulation or bound unrounded ideal states.

Required-prefix costs include each assessor's own numerical work and evidence handling, fresh preparation and verification charges, and retained training once. They are reconstructed costs because the full sample was prepared before assessment. Actual newly executed workflow costs are reported separately. The retained observations and raw binary inputs remain private.

Run these component and execution fixtures after the production assessment. The preparation fixtures reuse the first sixteen vectors of the fit-0 date streams. Executing them before a study exposes that prefix and requires a documented eligibility amendment; it cannot be treated as wholly fresh conditional evaluation. The checks themselves are bounded verification fixtures rather than additional research observations:

```sh
.venv-certification/bin/python -m unittest discover -s tests -p test_certify_financial_decision.py -v
.venv-certification/bin/python -m unittest discover -s tests -p test_study_financial_decision.py -v
.venv/bin/python -m unittest discover -s tests -p test_prepare_financial_decision.py -v
```

The retained 5 October experiment has a separately dated exposure amendment. It preserves all 131,072 generated paths and 262,144 numerical assessments, but excludes the first sixteen fit-0 rows from sampling inference. The eligible final total is 131,056 paths per assessor. Counts for fit 0 are 4,080 and 16,368, while the other fits retain 4,096 and 16,384. The probability allocation, caps and tolerances remain unchanged. The full correction cost remains separately recorded and is conservatively added to every selected original prefix charge.

`scripts/correct_financial_decision_inference.py` reproduces that particular amended construction from its original private study, observation ledgers and pinned amendment. Its hash guards intentionally identify the retained experiment rather than accepting a different design as the same historical correction. A new reproduction following the production-before-fixtures order above is a new study and does not inherit this exposure. Preserve its own registration, stream evidence and realised costs. All private ledgers and raw paths remain local; the paper includes aggregate results only.
