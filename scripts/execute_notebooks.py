"""Execute each project notebook with a fresh kernel; save only with --write."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Replace saved notebook outputs after successful execution")
    args = parser.parse_args()
    paths = sorted((ROOT / "projects").glob("*/notebooks/*.ipynb"))
    if not paths:
        raise SystemExit("No project notebooks found")
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        if args.write:
            nbformat.write(notebook, path)
        print(f"Executed {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
