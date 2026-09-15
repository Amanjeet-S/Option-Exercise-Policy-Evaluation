"""Check local Markdown targets and retained source hashes without network access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    failures = []
    links_checked = 0
    for name in filter(None, names):
        path = ROOT / name
        if path.suffix != ".md":
            continue
        text = path.read_text()
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            links_checked += 1
            if not destination.is_relative_to(ROOT) or not destination.exists():
                failures.append(f"{name}: missing or external local target {target}")

    treasury = ROOT / "data/public/treasury/2026-09-14"
    metadata = json.loads((treasury / "metadata.json").read_text())
    if hashlib.sha256((treasury / "source.xml").read_bytes()).hexdigest() != metadata["raw_sha256"]:
        failures.append("Treasury source checksum differs from the saved record")
    snapshot = ROOT / "data/public/germany_monetary_policy/2026-09-15"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for entry in manifest["sources"]:
        if hashlib.sha256((snapshot / entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Germany source checksum differs: {entry['file']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {links_checked} local Markdown targets and six retained source hashes.")


if __name__ == "__main__":
    main()
