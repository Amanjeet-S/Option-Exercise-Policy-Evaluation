"""Independent deterministic checks of the native CRR component.

Author: Amanjeet Singh. This driver compiles the native source and compares it
with the existing NumPy tree, analytical European values and decimal references.
Every result and failure is retained. These checks make no performance claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

FLAGS = ('-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off')
PARITY_ABSOLUTE = 2e-10
PARITY_RELATIVE = 2e-11
NATIVE_RECORDS = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_root():
    for root in Path(__file__).resolve().parents:
        if (root/'src/qf_research/optimal_stopping/tree.py').is_file():
            return root
    raise RuntimeError('The existing Python reference source is unavailable')


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')


def inputs(spot=100., strike=100., maturity=1., rate=.05, dividend=.02,
           volatility=.2, kind='put', steps=256, rights=4):
    return [spot, strike, maturity, rate, dividend, volatility, kind, steps, rights]


def native(executable, values):
    command = [str(executable), *(str(x) for x in values)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=20)
    record = dict(arguments=command[1:], returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
    NATIVE_RECORDS.append(record)
    try:
        result = json.loads(run.stdout)
    except (ValueError, TypeError) as error:
        raise AssertionError('Native output is not one JSON result: '+repr(record)) from error
    record['result'] = result
    if result.get('author') != 'Amanjeet Singh':
        raise AssertionError('Missing or unexpected author metadata')
    return record


def successful(executable, values):
    record = native(executable, values)
    if record['returncode'] != 0 or record['stderr'] or record['result'].get('ok') is not True:
        raise AssertionError('Expected successful native pricing: '+repr(record))
    result = record['result']
    if result.get('time_zero_exercise') is not False or result.get('steps') != values[7] or result.get('exercise_rights') != values[8]:
        raise AssertionError('Native calendar metadata differ')
    if not math.isfinite(result['value']) or result['value'] < 0:
        raise AssertionError('Native price must be finite and nonnegative')
    if values[5] == 0:
        if result.get('zero_volatility') is not True or result.get('probability') is not None:
            raise AssertionError('Zero-volatility metadata differ')
    elif result.get('zero_volatility') is not False or not 0 <= result['probability'] <= 1:
        raise AssertionError('Stochastic probability metadata differ')
    return record


def close(actual, expected, absolute=PARITY_ABSOLUTE, relative=PARITY_RELATIVE):
    allowance = absolute+relative*abs(expected)
    if not math.isfinite(expected) or abs(actual-expected) > allowance:
        raise AssertionError(f'Value difference {actual-expected} exceeds {allowance}')
    return dict(reference=expected, absolute_difference=abs(actual-expected), allowance=allowance)


def deterministic_reference(values, endpoint=None):
    spot, strike, maturity, rate, dividend, volatility, kind, steps, rights = values
    with localcontext() as context:
        context.prec = 80
        s, k, t, r, q = [Decimal.from_float(float(x)) for x in values[:5]]
        if endpoint is None:
            drift = r-q
        else:
            h = t/Decimal(steps)
            drift = Decimal(endpoint)*Decimal.from_float(float(volatility))/h.sqrt()
        rewards = []
        for right in range(1, rights+1):
            time = t*Decimal(right)/Decimal(rights)
            stock = s*(drift*time).exp()
            intrinsic = max(stock-k if kind == 'call' else k-stock, Decimal(0))
            rewards.append((-r*time).exp()*intrinsic)
        return float(max(rewards))


def exhaustive_reference(values):
    """Enumerate a small non-recombining tree using 80-digit decimal arithmetic."""
    spot, strike, maturity, rate, dividend, volatility, kind, steps, rights = values
    if steps > 8 or volatility <= 0:
        raise ValueError('Exhaustive checks require at most eight stochastic steps')
    with localcontext() as context:
        context.prec = 80
        s, k, t, r, q, sigma = [Decimal.from_float(float(x)) for x in values[:6]]
        h = t/Decimal(steps)
        u = (sigma*h.sqrt()).exp()
        d = Decimal(1)/u
        p = ((r-q)*h).exp()-d
        p /= u-d
        discount = (-r*h).exp()
        stride = steps//rights
        def payoff(stock):
            return max(stock-k if kind == 'call' else k-stock, Decimal(0))
        def descend(level, stock):
            if level == steps:
                return payoff(stock)
            continuation = discount*((Decimal(1)-p)*descend(level+1,stock*d)
                                     +p*descend(level+1,stock*u))
            return max(payoff(stock),continuation) if level > 0 and level%stride == 0 else continuation
        return float(descend(0,s))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', default='clang++')
    parser.add_argument('--source', type=Path, default=project_root()/'cpp/bermudan_tree.cpp')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        write_json(output/'verification.json', {'status':'compiler unavailable','compiler':args.compiler})
        raise RuntimeError('A C++17 compiler is unavailable; no installation was attempted')
    source = args.source.resolve()
    root = project_root()
    sys.path.insert(0, str(root/'src'))
    import numpy as np
    import scipy
    from qf_research.optimal_stopping.analytical import black_scholes
    from qf_research.optimal_stopping.contracts import Market, Option
    from qf_research.optimal_stopping.tree import binomial_price
    source_files = [source, Path(__file__).resolve(), root/'src/qf_research/optimal_stopping/tree.py',
                    root/'src/qf_research/optimal_stopping/contracts.py', root/'src/qf_research/optimal_stopping/analytical.py']
    frozen_sources = {str(path):digest(path) for path in source_files}
    version = subprocess.run([compiler,'--version'],capture_output=True,text=True,timeout=20)
    executable = output/'bermudan_tree'
    command = [compiler,*FLAGS,str(source),'-o',str(executable)]
    build = subprocess.run(command,capture_output=True,text=True,timeout=60)
    (output/'compiler_stdout.txt').write_text(build.stdout)
    (output/'compiler_stderr.txt').write_text(build.stderr)
    record = dict(author='Amanjeet Singh', created_utc=datetime.now(timezone.utc).isoformat(),
        status='in progress', compiler=compiler, compiler_version=version.stdout,
        compiler_version_returncode=version.returncode, flags=list(FLAGS), command=command,
        compilation_returncode=build.returncode, platform=platform.platform(), python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, source_sha256=frozen_sources, native_executions=NATIVE_RECORDS,
        numeric_policy=dict(parity_absolute=PARITY_ABSOLUTE,parity_relative=PARITY_RELATIVE,
            european_absolute=.003,zero_volatility_decimal_precision=80,resource_cap_steps=8192,
            result_is_interval_certificate=False), cases=[], failures=[])
    write_json(output/'verification.json',record)
    if build.returncode != 0 or version.returncode != 0:
        record['status']='compilation failed';write_json(output/'verification.json',record)
        return 1
    record['executable_sha256']=digest(executable)

    def check(name, function):
        case={'name':name}
        try:
            case.update(function());case['passed']=True
        except Exception as error:
            case.update(passed=False, exception_type=type(error).__name__, exception=str(error))
            record['failures'].append(name)
        record['cases'].append(case)
        write_json(output/'verification.json',record)

    def parity(values):
        result=successful(executable,values)
        s,k,t,r,q,sigma,kind,n,m=values
        market,option=Market(s,r,sigma,q),Option(k,t,kind)
        dates=np.array([t*i/m for i in range(1,m+1)])
        python_price=binomial_price(market,option,n,dates).value
        result['comparison']=close(result['result']['value'],python_price)
        return result

    variants=[(.05,.02,.2,1.),(-.03,.04,.25,2.),(.01,0.,.1,.25),(.02,.20,.4,1.),(.05,-.03,.3,1.)]
    for kind in ('call','put'):
        for index,(r,q,sigma,t) in enumerate(variants):
            for rights in (1,4,16):
                values=inputs(rate=r,dividend=q,volatility=sigma,maturity=t,kind=kind,rights=rights)
                check(f'python parity {kind} variant {index} rights {rights}',lambda values=values:parity(values))
        for spot in (50.,150.):
            values=inputs(spot=spot,kind=kind,steps=96,rights=12)
            check(f'python parity {kind} spot {spot}',lambda values=values:parity(values))

    for kind in ('call','put'):
        for rights in (1,2,3,6):
            for rate,dividend in ((.05,.02),(-.03,.04),(.02,.20)):
                values=inputs(rate=rate,dividend=dividend,kind=kind,steps=6,rights=rights)
                def exhaustive(values=values):
                    result=successful(executable,values)
                    result['decimal_tree_comparison']=close(result['result']['value'],exhaustive_reference(values))
                    return result
                check(f'Exhaustive decimal tree {kind} rights {rights} rate {rate}',exhaustive)

    for steps in (360,720,1440):
        values=inputs(maturity=3.,rate=.05,dividend=.10,volatility=.20,kind='call',steps=steps,rights=9)
        check(f'Tutorial contract parity steps {steps}',lambda values=values:parity(values))

    for kind in ('call','put'):
        values=inputs(kind=kind,steps=1024,rights=1)
        def european(values=values):
            result=parity(values)
            s,k,t,r,q,sigma,kind,n,m=values
            result['analytical_comparison']=close(result['result']['value'],black_scholes(Market(s,r,sigma,q),Option(k,t,kind)),absolute=.003,relative=0.)
            return result
        check('European analytical '+kind,european)

    for kind in ('call','put'):
        for r,q in ((.05,.02),(-.03,.04),(.02,.20)):
            for rights in (1,4):
                values=inputs(rate=r,dividend=q,volatility=0.,kind=kind,rights=rights)
                def zero(values=values):
                    result=parity(values)
                    result['decimal_comparison']=close(result['result']['value'],deterministic_reference(values))
                    return result
                check(f'zero volatility {kind} {r} {q} rights {rights}',zero)

    values=inputs(volatility=0.,kind='call',steps=8192,rights=1)
    check('Accepted exact resource-cap boundary',lambda:parity(values))

    for endpoint in (-1,1):
        for kind in ('call','put'):
            values=inputs(rate=.4*endpoint,dividend=0.,kind=kind,steps=4,rights=4)
            def endpoint_check(values=values,endpoint=endpoint):
                result=parity(values)
                expected_probability=0. if endpoint==-1 else 1.
                if result['result']['probability']!=expected_probability:raise AssertionError('Endpoint probability is not exact')
                result['endpoint_comparison']=close(result['result']['value'],deterministic_reference(values,endpoint))
                return result
            check(f'probability endpoint {endpoint} {kind}',endpoint_check)

    def no_initial_right():
        values=inputs(spot=150.,rate=.02,dividend=.2,volatility=0.,kind='call',rights=1)
        result=successful(executable,values)
        if not result['result']['value']<50.:raise AssertionError('Time-zero intrinsic reward was included')
        result['decimal_comparison']=close(result['result']['value'],deterministic_reference(values))
        return result
    check('No time-zero exercise',no_initial_right)

    def nested_calendars():
        results=[successful(executable,inputs(rights=m)) for m in (1,2,4,8,16)]
        prices=[row['result']['value'] for row in results]
        if any(left>right+PARITY_ABSOLUTE for left,right in zip(prices,prices[1:])):
            raise AssertionError('Nested-calendar values decreased')
        return dict(results=results)
    check('Nested-calendar ordering',nested_calendars)

    def put_call_parity():
        values=inputs(rights=1)
        call=successful(executable,inputs(kind='call',rights=1))
        put=successful(executable,values)
        expected=values[0]*math.exp(-values[4]*values[2])-values[1]*math.exp(-values[3]*values[2])
        return dict(call=call,put=put,comparison=close(call['result']['value']-put['result']['value'],expected))
    check('European put-call parity',put_call_parity)

    def homogeneous_scaling():
        base=successful(executable,inputs(kind='call'))
        scaled=successful(executable,inputs(spot=37.5,strike=37.5,kind='call'))
        return dict(base=base,scaled=scaled,comparison=close(scaled['result']['value'],.375*base['result']['value']))
    check('Spot and strike homogeneity',homogeneous_scaling)


    invalid=[]
    for index,name,value in [(0,'zero spot',0.),(1,'zero strike',0.),(2,'zero maturity',0.),
        (0,'negative spot',-1.),(1,'negative strike',-1.),(2,'negative maturity',-1.),
        (3,'nonfinite rate','nan'),(4,'nonfinite yield','inf'),(5,'negative volatility',-.2),
        (5,'nonfinite volatility','nan'),(6,'unsupported payoff','digital'),
        (7,'zero steps',0),(8,'zero rights',0),(7,'step resource cap',8193),(8,'right resource cap',8193),
        (7,'noninteger steps','2.5'),(8,'negative rights','-1'),(0,'malformed floating input','100x'),
        (0,'whitespace input',' 100'),(0,'overflow floating input','1e999'),(0,'underflow floating input','1e-999')]:
        values=inputs();values[index]=value;invalid.append((name,values))
    invalid += [('large-tree call subnormal continuation',inputs(kind='call',steps=2048,rights=1)),
        ('large-tree put subnormal continuation',inputs(kind='put',steps=2048,rights=1)),
        ('misaligned calendar',inputs(steps=10,rights=3)),('rights exceed steps',inputs(steps=1,rights=2)),
        ('invalid probability',inputs(rate=1.,dividend=0.,volatility=.01,steps=1,rights=1)),
        ('unresolved small volatility',inputs(volatility=1e-30)),
        ('overflow stock nodes',inputs(spot=1e308,volatility=.5,steps=128)),
        ('underflow stock nodes',inputs(spot=1e-300,volatility=2.,steps=128)),
        ('overflow carry subtraction',inputs(rate=1e308,dividend=-1e308)),
        ('zero-volatility exponential overflow',inputs(rate=1000.,dividend=0.,volatility=0.,rights=1)),
        ('zero-volatility discount underflow',inputs(rate=1000.,dividend=1000.,volatility=0.,rights=1)),
        ('huge integer token',inputs(steps='999999999999999999999999999999999999')),
        ('missing arguments',[]),('extra arguments',inputs()+['extra'])]
    for name,values in invalid:
        def rejected(values=values):
            result=native(executable,values)
            if result['returncode']!=2 or result['result'].get('ok') is not False or not isinstance(result['result'].get('error'),str):
                raise AssertionError('Invalid or unsupported input was not rejected: '+repr(result))
            return result
        check('Reject '+name,rejected)

    for path in source_files:
        if digest(path)!=record['source_sha256'][str(path)]:record['failures'].append('source changed: '+str(path))
    record.update(status='passed' if not record['failures'] else 'failed',
        checks=len(record['cases']), passed_checks=sum(row['passed'] for row in record['cases']),
        driver_process_cpu_seconds=time.process_time()-started_cpu, wall_seconds=time.perf_counter()-started_wall,
        cost_scope='Verification workflow only; no pricing-speed or complete-research cost claim')
    write_json(output/'verification.json',record)
    print(json.dumps(dict(status=record['status'],checks=record['checks'],passed=record['passed_checks'],failures=record['failures'])))
    return 0 if not record['failures'] else 1


if __name__=='__main__':
    raise SystemExit(main())
