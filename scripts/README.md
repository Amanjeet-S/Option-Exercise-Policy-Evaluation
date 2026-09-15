# Repository utilities

Run these commands from the repository root:

- `python scripts/check_repository.py`: check tracked Markdown file links and retained source hashes.
- `python scripts/audit_germany_data.py`: report source coverage and common months without estimating effects.
- `python scripts/execute_notebooks.py`: execute project notebooks in fresh kernels; add `--write` only to update their saved outputs.

The first two utilities use the Python standard library. Notebook execution requires `requirements-notebooks.lock`. Existing numerical experiments and Treasury acquisition are implemented in the source package and documented in Project 1.
