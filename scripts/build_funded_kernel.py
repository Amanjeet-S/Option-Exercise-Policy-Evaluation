"""Build the optional original C++ propagation kernel and retain compiler provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cpp/funded_step.cpp"


def build(output, compiler=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new build directory")
    compiler = compiler or os.environ.get("CXX", "c++")
    resolved = shutil.which(compiler)
    if not resolved:
        raise FileNotFoundError(f"C++ compiler unavailable: {compiler}")
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("This build entry point is verified for macOS/Linux conventions only")
    flags = ["-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off",
             "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-fPIC"]
    flags.append("-dynamiclib" if platform.system() == "Darwin" else "-shared")
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    version = subprocess.check_output([resolved, "--version"], text=True)
    output.mkdir(parents=True)
    library = output / ("libqf_funded_step" + suffix)
    started = time.perf_counter()
    subprocess.run([resolved, *flags, str(SOURCE), "-o", str(library)], check=True, capture_output=True, text=True)
    record = {"built_at_utc": datetime.now(timezone.utc).isoformat(),
              "compiler": resolved, "compiler_version": version, "flags": flags,
              "source": str(SOURCE.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "library": library.name, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "compile_seconds": time.perf_counter()-started,
              "platform": platform.platform(), "machine": platform.machine()}
    (output/"build.json").write_text(json.dumps(record, indent=2)+"\n")
    return library


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler")
    args = parser.parse_args()
    print(build(args.output, args.compiler))
