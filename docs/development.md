# Development

## 1. Environment and checks

The [public reproduction guide](reproducing_paper.md) provides installation and a single command for the paper checks, mathematical example and notebooks. For development, use that environment and run:

```sh
python -m pytest -q
python scripts/check_repository.py
python scripts/execute_release_notebooks.py
```

Changed numerical routines require appropriate analytical benchmarks or independent comparisons. Execute changed notebooks in fresh kernels. The notebook command uses its invoking Python interpreter and the checkout's source directory; a separately registered default kernel is not used. It leaves saved files unchanged unless `--write` is supplied. Review outputs before publication.

## 2. Optional environments

C++ propagation tests require a C++17 compiler and compare the native kernel with Python/SciPy. The [interval-arithmetic instructions](reproducing_paper.md#5-interval-arithmetic) use a separate pinned python-flint environment. Provider acquisition helpers require an entitled, separately configured local environment; credentials and returned observations remain private.

## 3. Continuous integration

GitHub checks Python 3.11 and 3.13, the reviewed source inventory, numerical tests, documentation links, released summaries, public reproduction and fresh notebook execution. A separate job checks outward-rounded arithmetic. These checks require no market-data account. Timings can change across machines; numerical and empirical conclusions retain the limits stated in the paper.

## 4. Revision history and releases

I preserve the recorded author and commit dates of recovered project revisions. I have divided the largest bundled release into functional groups for review. Their shared date identifies the original release; the grouping does not establish when each underlying task was performed. I retain the original checkpoints in a private archive.

I use the annotated tag `v1.0.0` to identify the completed, reviewed study snapshot. Subsequent changes require their own commits and verification before another release.
