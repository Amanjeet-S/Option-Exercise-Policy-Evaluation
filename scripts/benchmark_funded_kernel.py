"""Declared mathematical accuracy/cost experiment, without vendor inputs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy

from build_funded_kernel import build
from qf_research.computational.funded_kernel import NativeKernel, advance_scipy, prepare, price_prepared
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    FundedCashModel, DividendCap, FundedExercise, finite_difference_price, european_reference,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/01_optimal_stopping/configs/cpp_propagation_validation.json"
METHODS = ("original", "prepared_scipy", "prepared_cpp")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


def inputs(cfg, grid):
    f = cfg["fixture"]
    model = FundedCashModel(f["quoted_spot"], f["volatility"], f["initial_share_factor"],
                            f["premium_factor"], tuple(DividendCap(*e) for e in f["cash_events"]))
    option = Option(f["strike"], f["maturity"], f["kind"])
    n = f["exercise_count"]
    rights = tuple(FundedExercise(k*option.maturity/n, 1-.03*k*option.maturity/n) for k in range(1,n+1))
    return model, option, grid["space_steps"], grid["time_steps"], f["domain_maximum"], rights


def timed(fn):
    started = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns()-started)/1e9


def memory_worker(cfg, index, method, library):
    args = inputs(cfg, cfg["grids"][index])
    if method == "original": result = finite_difference_price(*args)
    else:
        backend = advance_scipy if method == "prepared_scipy" else NativeKernel(library)
        result = price_prepared(prepare(*args), backend)
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mib = raw/(1024**2) if platform.system() == "Darwin" else raw/1024
    return {"grid_index": index, "method": method, "peak_process_rss_mib": mib,
            "premium_quote": result.premium_quote}


def convergence(cfg, native):
    rows = []
    for kind in ("put", "call"):
        for cash in ((), (DividendCap(.3, 6.),)):
            model = FundedCashModel(100.37, .25, .999, .998, cash)
            option = Option(103.11, 1., kind)
            terminal = FundedExercise(1., .97)
            ref = european_reference(model, option, terminal)
            for g in cfg["grids"]:
                p = prepare(model, option, g["space_steps"], g["time_steps"], 450., (terminal,))
                old = finite_difference_price(model, option, g["space_steps"], g["time_steps"], 450., (terminal,))
                cpp = price_prepared(p, native)
                py = price_prepared(p, advance_scipy)
                rows.append({"kind": kind, "cash_events": len(cash), **g,
                             "reference": asdict(ref), "original": asdict(old),
                             "prepared_scipy": asdict(py), "prepared_cpp": asdict(cpp),
                             "cpp_minus_reference": cpp.premium_quote-ref.premium_quote,
                             "original_minus_reference": old.premium_quote-ref.premium_quote})
    return rows


def run(output, build_dir):
    output = output.resolve()
    if output.exists(): raise FileExistsError("Use a fresh result directory")
    cfg = json.loads(CONFIG.read_text())
    if any(os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")):
        raise ValueError("Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before starting Python")
    sources = [Path(__file__), ROOT/"scripts/build_funded_kernel.py",
               ROOT/"projects/01_optimal_stopping/cpp/funded_step.cpp",
               ROOT/"src/qf_research/computational/funded_kernel.py",
               ROOT/"tests/test_funded_kernel.py", CONFIG,
               *sorted((ROOT/"src/qf_research/optimal_stopping").glob("*.py"))]
    hashes = {str(p.resolve().relative_to(ROOT)):sha(p) for p in sources}
    library = build(build_dir)
    native = NativeKernel(library)
    output.mkdir(parents=True)
    save(output/"configuration.json", cfg)
    save(output/"build.json", json.loads((library.parent/"build.json").read_text()))
    rows = []
    for index,g in enumerate(cfg["grids"]):
        args = inputs(cfg,g); prepared = prepare(*args)
        calls = {"original": lambda:finite_difference_price(*args),
                 "prepared_scipy": lambda:price_prepared(prepared,advance_scipy),
                 "prepared_cpp": lambda:price_prepared(prepared,native)}
        for method in METHODS:
            for _ in range(cfg["warmups_per_method"]):calls[method]()
        durations = {method:[] for method in METHODS}; results={}
        order = []
        for repeat in range(cfg["timing_repetitions"]):
            rotated = METHODS[repeat%3:]+METHODS[:repeat%3]; order.append(rotated)
            for method in rotated:
                result,seconds = timed(calls[method]);durations[method].append(seconds)
                results[method] = asdict(result)
        preparation=[]; full={m:[] for m in METHODS[1:]}
        for repeat in range(cfg["timing_repetitions"]):
            _,seconds = timed(lambda:prepare(*args));preparation.append(seconds)
            backends=[("prepared_scipy",advance_scipy),("prepared_cpp",native)]
            if repeat%2:backends.reverse()
            for method,backend in backends:
                _,seconds = timed(lambda:price_prepared(prepare(*args),backend))
                full[method].append(seconds)
        reference=results["original"]["premium_quote"]
        tolerance=cfg["maximum_parity_error"]["absolute_premium"]+cfg["maximum_parity_error"]["relative_premium"]*abs(reference)
        errors={m:abs(results[m]["premium_quote"]-reference) for m in METHODS[1:]}
        if any(e>tolerance for e in errors.values()):raise AssertionError("Cross-language price tolerance failed")
        rows.append({**g,"results":results,"absolute_premium_differences":errors,
                     "parity_tolerance":tolerance,"seconds":durations,"method_order":order,
                     "preparation_seconds":preparation,"prepare_and_value_seconds":full,
                     "time_intervals":prepared.time_intervals,"event_free_blocks":len(prepared.blocks),
                     "implicit_substeps":sum(len(b.widths) for b in prepared.blocks)})
        print(f"Completed timing grid {index+1}/{len(cfg['grids'])}",flush=True)
    save(output/"timings.json",rows)
    checks=convergence(cfg,native)
    for row in checks:
        baseline=row['original']['premium_quote']
        tolerance=cfg['maximum_parity_error']['absolute_premium']+cfg['maximum_parity_error']['relative_premium']*abs(baseline)
        if any(abs(row[m]['premium_quote']-baseline)>tolerance for m in METHODS[1:]):
            raise AssertionError('Convergence-case backend parity failed')
    save(output/"accuracy.json",checks)
    memory=[]
    for index in range(len(cfg['grids'])):
        for method in METHODS:
            payload=subprocess.check_output([sys.executable,str(Path(__file__).resolve()),'--memory-worker',
                    str(index),'--method',method,'--library',str(library)],text=True,cwd=ROOT)
            row=json.loads(payload)
            baseline=rows[index]['results']['original']['premium_quote']
            if abs(row['premium_quote']-baseline)>rows[index]['parity_tolerance']:
                raise AssertionError('Fresh-process memory worker price differs')
            memory.append(row)
    save(output/"memory.json",memory)
    profile=cProfile.Profile();profile.runcall(finite_difference_price,*inputs(cfg,cfg['grids'][1]))
    stats=pstats.Stats(profile)
    profile_rows=[{'file':Path(key[0]).name,'line':key[1],'function':key[2],
                   'calls':v[1],'self_seconds':v[2],'cumulative_seconds':v[3]}
                  for key,v in stats.stats.items()]
    save(output/"profile.json",sorted(profile_rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:20])
    summary=[]
    for row in rows:
        med={m:statistics.median(row['seconds'][m]) for m in METHODS}
        end={m:statistics.median(row['prepare_and_value_seconds'][m]) for m in METHODS[1:]}
        summary.append({'space_steps':row['space_steps'],'time_steps':row['time_steps'],
                        'median_seconds':med,'median_prepare_and_value_seconds':end,
                        'original_over_cpp_including_preparation':med['original']/end['prepared_cpp'],
                        'prepared_scipy_over_cpp':med['prepared_scipy']/med['prepared_cpp'],
                        'maximum_backend_price_difference':max(row['absolute_premium_differences'].values())})
    save(output/"summary.json",summary)
    for name,digest in hashes.items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed during benchmark')
    save(output/"manifest.json",{'completed_at_utc':datetime.now(timezone.utc).isoformat(),
         'purpose':cfg['purpose'],'source_sha256':hashes,
         'artifact_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
         'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
         'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
         'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
         'peak_rss_scope':'Fresh child process including common imports and method preparation; not incremental allocation',
         'empirical_interpretation':False})
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--build-dir',type=Path)
    parser.add_argument('--memory-worker',type=int)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--library',type=Path)
    args=parser.parse_args()
    if args.memory_worker is not None:
        print(json.dumps(memory_worker(json.loads(CONFIG.read_text()),args.memory_worker,args.method,args.library)))
    elif args.output and args.build_dir:run(args.output,args.build_dir)
    else:parser.error('Provide --output and --build-dir')
