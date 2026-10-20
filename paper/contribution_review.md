# Research question and contribution

## 1. Financial objective

I study whether a fitted Bermudan exercise policy can be assessed at a monetary tolerance, and at what computational cost. The expected dual gap contains both policy shortfall and martingale slack. Sampling uncertainty and conditional integration error add separate allowances. The paper compares these quantities rather than treating a smaller reported gap as evidence of better exercise decisions.

The contribution is a reproducible comparative evaluation: a directly integrable diagnostic tests confidence coverage, multidimensional benchmarks compare verified numerical assessments, and frozen-policy calculations test whether computed upper limits cross declared monetary tolerances. The [paper](main.tex) derives the mathematical objects and interprets their financial roles. Standard derivations and lengthy proofs appear in its appendices.

## 2. Findings within the paper

The two-right diagnostic has zero theoretical dual slack. At 2,048 paths, 5,000 repetitions give 96.50 per cent coverage for nominal 97.5 per cent Student upper limits. Clipping with an independently selected cap and analytical tail allowance gives 99.98 per cent observed coverage, with smaller mean excess than the Cantelli comparator. The experiment tests an inference problem without claiming a new concentration theorem.

Across eight five-asset fits, gain-directed integration meets the 0.01 realised gain-width requirement using a median 75.7 per cent of the matched common-tolerance implementation's reconstructed standalone CPU budget. Cheaper exact-moment alternatives retain larger observed gaps. The measured ratio compares complete implementations, including their differing evidence work; it does not isolate an abstract allocation rule's speed.

A separate evaluation uses 16,384 costed paths per fit. All eight computed European-assessor limits lie below one unit; half-unit targets remain unresolved. The sixteen exposed fit-0 rows are excluded from inference while their costs remain charged. A simultaneous continuous-model guarantee additionally requires the ideal-sampling and observation-enclosure premises. The numerical certificates here concern supplied binary states. The complete monetary-cost advantage of gain-directed refinement is not established.

## 3. Attribution and reproducible scope

The policy and duality foundations are Longstaff and Schwartz (2001), Rogers (2002), Haugh and Kogan (2004), and Andersen and Broadie (2004). The paper attributes the conditional-moment, numerical and concentration results it uses. My derivations explain their application and assumptions without claiming theoretical priority. The bibliography and [source register](../SOURCES.md) give the references.

The [current reproduction package](../docs/reproducing_paper.md) verifies selected source bytes, aggregate arithmetic and manuscript transcription, runs numerical fixtures and executes the explanatory notebook. Full experiment regeneration is documented separately and produces new model simulations and timings. Neither route establishes independent market validity. The historical licensed study and its empty primary market-admission sample remain separate.

## 4. Exploratory extensions

The later two-asset monetary, Gaussian-compression and bounded-correction investigations are outside this paper's result set and current reader package. Their complete records remain local, including unsuccessful cost comparisons and unresolved targets. I do not use their outcomes as completed paper findings or provide preparation commands as a substitute for full assessment reproduction. They require separate source-bound reproduction before inclusion in a further study.
