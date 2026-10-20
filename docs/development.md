# Development

## 1. Current revision checks

The [reproduction guide](reproducing_paper.md) separates the historical release from the current mathematical investigation. After installing the research and notebook requirements in the current checkout, run:

```sh
python scripts/reproduce_current_study.py --output data/private/reproduction/current_review
```

The output directory must be new. The command checks the current source inventory, aggregate arithmetic and manuscript transcription, selected numerical tests and a separate fresh-kernel notebook copy. It records every step and failure. Add `--with-native --compiler g++` for the optional native verification. These checks do not regenerate the full research experiments. The [C++ guide](../cpp/README.md) and reproduction guide describe the component and larger study routes.

Changed numerical routines require appropriate analytical benchmarks or independent comparisons. The notebook runner uses its invoking interpreter and the checkout's source directory. It leaves saved notebooks unchanged unless `--write` is supplied. Each successful execution records its actual kernel version, completion time and native comparison outcome. Authorship, mathematical inputs, random-stream design and expected numerical source identities remain distinct from execution metadata.

## 2. Historical release and optional environments

The `v1.0.0` source inventory, `scripts/reproduce_paper.py`, the default two-notebook selection and `scripts/check_repository.py` belong to the historical release route. Use the separate checkout described in the reproduction guide. The current sources deliberately differ from that inventory; updating its hashes would not verify the revision.

C++ checks require a C++17 compiler. The [interval-arithmetic instructions](reproducing_paper.md#5-interval-arithmetic) use a separate pinned python-flint environment. Provider acquisition helpers require an entitled local environment; credentials and returned observations remain private.

## 3. Continuous integration

The workflow selects the separate current inventory and reader command on Python 3.11 and 3.13, with a separate outward-arithmetic job. Its commands use public mathematical fixtures and require no market-data account. GitHub Actions records the outcome for each published commit; local verification establishes only the recorded machine's result. The historical workflow remains in `v1.0.0`, and its inventory and experiment registrations are unchanged.

## 4. Revision history and releases

I preserve the recorded author and commit dates of recovered project revisions. I have divided the largest bundled release into functional groups for review. Their shared date identifies the original release; the grouping does not establish when each underlying task was performed. I retain the original checkpoints in a private archive.

I use the annotated tag `v1.0.0` to identify the completed, reviewed study snapshot. The completed public-model revision is identified by `v2.0.0`; `v2.0.1` applies title capitalisation without changing its calculations. Subsequent changes require their own commits and verification before another release.

## 5. Current inventory maintenance

`replication/current_source_manifest.json` binds a literal selection of current reader sources, tests, documentation and software requirements. It is separate from the historical inventory and excludes private inputs, provider observations and compiled outputs. Its hashes establish consistency, rather than mathematical correctness or publication permission. After reviewing a deliberate source change, inspect the selection with `python scripts/build_current_inventory.py` and use `--write` to record it. Do not refresh hashes merely to suppress an unexplained mismatch. Registered experiment sources retain their original identities even when a later implementation is developed separately.
