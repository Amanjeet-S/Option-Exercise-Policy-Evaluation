"""Fixed-look monetary decisions for eight frozen five-asset policies.

I compare the zero martingale with the direct European-value martingale on
one fresh private sample per policy. Successful endpoint records and native ledgers are streamed
losslessly; complete native evidence is retained for a failed assessment.
A stopped experiment is preserved and cannot be silently restarted.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

FITS=8
LOOKS=[4096,16384]
ASSESSORS=('zero','european_common')
TOLERANCES=(Fraction(1,2),Fraction(1))
DELTA=Fraction(1,640)
CAPS={'zero':Fraction(150),'european_common':Fraction(50)}
TARGET=Fraction(1,100)
INITIAL=Fraction(1)
PRECISION=96
MINIMUM_FREE_BYTES=256*1024*1024
SOURCE_FILES=(
 'scripts/study_financial_decision.py','tests/test_study_financial_decision.py',
 'scripts/prepare_financial_decision.py','tests/test_prepare_financial_decision.py',
 'scripts/certify_financial_decision.py','tests/test_certify_financial_decision.py',
 'scripts/refine_european_gain.py','tests/test_refine_european_gain.py',
 'scripts/certify_european_gain_tail.py','tests/test_certify_european_gain_tail.py',
 'scripts/study_european_gain_allocation.py','tests/test_study_european_gain_allocation.py')


def now():return datetime.now(timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hashed(value):return hashlib.sha256(canonical(value)).hexdigest()
def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):result.update(block)
    return result.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as destination:destination.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def clocks():return time.process_time(),time.perf_counter()
def elapsed(start):return dict(cpu_seconds=time.process_time()-start[0],wall_seconds=time.perf_counter()-start[1])
def zero_cost():return dict(cpu_seconds=0.,wall_seconds=0.)
def valid_cost(cost):
    if set(cost)!={'cpu_seconds','wall_seconds'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in cost.values()):raise ValueError('Finite non-negative CPU and wall costs are required')
    return cost

def summed_cost(*costs):
    answer=zero_cost()
    for cost in costs:
        valid_cost(cost)
        for key in answer:answer[key]+=cost[key]
    return answer

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise RuntimeError('Numerical source cannot be loaded')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def private_path(root,path,*,fresh=False):
    root,path=Path(root).resolve(),Path(path).resolve()
    if not path.is_relative_to((root/'data/private').resolve()) or (fresh and path.exists()):raise ValueError('A private path is required; a preceding attempt cannot be overwritten')
    return path

def guard_snapshot(root,retained,design):
    actual=dict(source_sha256={name:digest(root/name) for name in design['source_sha256']},
      retained_input_sha256={name:digest(retained/name) for name in design['retained_input_sha256']})
    for key in actual:
        if actual[key]!=design[key]:raise ValueError('A sealed source or retained policy input changed')
    return actual

def policy_identity(record):
    return hashed(dict(model=record['model'],frozen_policy=record['frozen_policy'],
        policy_degree=record['design']['policy_degree'],training_paths=record['design']['training_paths'],
        root_seed=record['design']['root_seed']))

def settings(preparation):
    return dict(schema_version=1,author='Amanjeet Singh',preparation=preparation,
        looks=LOOKS,assessors=list(ASSESSORS),tolerance_units=['1/2','1'],
        tolerance_scope='Fixed engineering tolerances of 0.5 and 1 per cent of strike; no commercial tolerance standard is asserted.',
        tail_caps={'zero':'150','european_common':'50'},
        european_tail_split={'positive':'20','negative':'30'},
        family=dict(alpha='1/20',delta='1/640',fits=8,ideal_gains=2,looks=2,events=32),
        european_numerics=dict(arm='coarse_common',target_width='1/100',initial_value_width='1',
          max_rounds=16,precision_bits=96,max_oracle_attempts=8,cutoff_multiplier=2),
        tail_numerics=dict(precision_bits=96,european_cutoff_multiplier=8,
          absolute_tolerance='1/10000000000',relative_tolerance='1/10000000000'),
        assessor_order_by_fit=[list(ASSESSORS if i%2==0 else ASSESSORS[::-1]) for i in range(8)],
        stopping_rule_scope='Canonical pathwise evaluation of the retained sorted full-state degree-two coefficients. Each own-state basis and fixed-order math.fsum decision is independent of every other path, with strict exercise inequalities, continuation on ties or non-finite polynomial arithmetic, and terminal exercise. No fitting is performed. This preserves coefficients and the threshold convention, rather than asserting universal bitwise equivalence with the historical variable-row BLAS implementation.',
        inference_scope='Conditional ideal-model guarantee given fresh iid paths, valid gain enclosures and deterministic population tail allowances. The floating-point Gaussian simulation law is not numerically certified.',
        stopping_scope='Both registered looks execute. First acceptance identifies the required prefix for each tolerance; an unresolved case consumes the full declared maximum.',
        cost_scope='Actual new preparation and assessment workflows are measured separately. Required-prefix standalone charges are reconstructed from measured prefix preparation, the assessor own setup and assessment costs, shared guards, and retained policy-training CPU once. Earlier assessments and value fitting are excluded.',
        compressed_observation_scope='Every successful gain interval, work count and full native ledger is streamed losslessly in the private compressed record. Failed native ledgers and partial work are retained separately. No public path or model-input export is produced.')

def validate_design(design,preparer):
    expected=settings(preparer.PRODUCTION_PREPARATION)
    for key,value in expected.items():
        if design.get(key)!=value:raise ValueError('A fixed monetary-decision setting changed: '+key)
    preparer.validate_design(design)
    if design.get('recorded_before_sampling') is not True:raise ValueError('The design must precede every fresh evaluation draw')
    return True

def register(args):
    root=Path(args.root).resolve();retained=private_path(root,args.retained_study)
    destination=private_path(root,args.output,fresh=True)
    start=clocks()
    preparer=load_module(root/'scripts/prepare_financial_decision.py','financial_preparation_registration')
    original=read(root/'data/private/revision/2026-10-05/uniform_allocation_recovery_v1/study/design.json')['source_sha256']
    for name,expected in original.items():
        if digest(root/name)!=expected:raise ValueError('An original immutable numerical source changed')
    names=sorted(set(original)|set(SOURCE_FILES))
    sources={name:digest(root/name) for name in names}
    if digest(Path(__file__))!=sources['scripts/study_financial_decision.py']:raise ValueError('The executed registration driver is not the sealed installed source')
    inputs={f'fit_{i:02d}/inputs.json':digest(retained/f'fit_{i:02d}/inputs.json') for i in range(FITS)}
    # A read-only runtime query imports numerical libraries but draws no paths.
    query="import importlib.util,json; s=importlib.util.spec_from_file_location('p',"+repr(str(root/'scripts/prepare_financial_decision.py'))+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(dict(runtime=m.runtime_identity(),initial_date_streams={str(i):m.construct_streams(2026100410+i,list(range(1,10)))[1] for i in range(8)}),sort_keys=True,allow_nan=False))"
    process=subprocess.run([str(args.research_python),'-B','-c',query],check=True,text=True,capture_output=True,cwd=root)
    runtime_query=json.loads(process.stdout)
    preparation_runtime=runtime_query["runtime"]
    validator=load_module(root/'scripts/study_european_gain_allocation.py','financial_native_validation_registration')
    design=settings(preparer.PRODUCTION_PREPARATION)
    design.update(root=str(root),retained_study=str(retained),recorded_at_utc=now(),recorded_before_sampling=True,
      source_sha256=sources,retained_input_sha256=inputs,preparation_runtime=preparation_runtime,
      initial_date_streams=runtime_query["initial_date_streams"],
      certification_runtime=validator.runtime_identity(),policy_implementation=preparer.POLICY_IMPLEMENTATION,fits=[])
    for i in range(FITS):
        record=read(retained/f'fit_{i:02d}/inputs.json')
        phase=record['original_preparation_phases']['policy_training']
        valid_cost(phase)
        design['fits'].append(dict(index=i,root_seed=2026100410+i,
          frozen_model_policy_sha256=policy_identity(record),retained_policy_training_cost=phase))
    if args.design_note is not None:
        design['design_note_sha256']=digest(args.design_note)
    if args.proof_note is not None:
        design['proof_note_sha256']=digest(args.proof_note)
    validate_design(design,preparer)
    guard_snapshot(root,retained,design)
    destination.mkdir(parents=True)
    design['registration_parent_prewrite_cost']=elapsed(start)
    design['registration_timing_scope']='Parent-process preparation of the source and design record. The separate read-only runtime-query process and its startup/import CPU are not included. This is not an assessment cost.'
    write(destination/'design.json',design)
    print(json.dumps(dict(registered=True,design_sha256=digest(destination/'design.json'),paths_generated=0,policies_fitted=0)),flush=True)


def verify_bridge(folder,design,fit,preparer):
    start=clocks();completion=read(folder/'completion.json');header=read(folder/'input_manifest.json')
    if completion.get('completed') is not True:raise ValueError('The fresh binary bridge has not completed')
    binding=completion['binding']
    if header['policy_implementation']!=preparer.POLICY_IMPLEMENTATION or completion['policy_implementation']!=preparer.POLICY_IMPLEMENTATION or design['policy_implementation']!=preparer.POLICY_IMPLEMENTATION:raise ValueError('The canonical pathwise policy implementation changed')
    expected=dict(design_sha256=design['_sha256'],source_manifest_sha256=hashed(design['source_sha256']),
      input_sha256=design['retained_input_sha256'][f'fit_{fit["index"]:02d}/inputs.json'],
      fit_index=fit['index'],root_seed=fit['root_seed'],frozen_model_policy_sha256=fit['frozen_model_policy_sha256'],
      runtime_sha256=hashed(design['preparation_runtime']))
    if binding!=expected or header['binding']!=binding or header['runtime']!=design['preparation_runtime'] or header['preparation']!=design['preparation']:raise ValueError('The bridge runtime, model-policy or source binding changed')
    for filename,key in [('input_manifest.json','input_manifest_sha256'),('block_metadata.jsonl','block_metadata_sha256'),('states.bin','state_sha256'),('stops.bin','stopping_sha256')]:
        if digest(folder/filename)!=completion[key]:raise ValueError('A prepared bridge artifact changed: '+filename)
    n=design['preparation']['total_paths_per_fit'];batch=design['preparation']['batch_paths'];size=10*5*8
    if completion['paths']!=n or (folder/'states.bin').stat().st_size!=n*size or (folder/'stops.bin').stat().st_size!=n:raise ValueError('The bridge contains an undeclared path count')
    rows=[json.loads(line) for line in (folder/'block_metadata.jsonl').read_text().splitlines()]
    if len(rows)!=n//batch or preparer.verify_chain(rows,header['chain_genesis_sha256'])!=completion['last_block_sha256']:raise ValueError('The bridge ordered hash chain changed')
    for i,row in enumerate(rows):
        if row['path_start']!=i*batch or row['path_stop']!=(i+1)*batch or row['paths']!=batch or row['source_manifest_sha256']!=binding['source_manifest_sha256'] or row['design_sha256']!=binding['design_sha256'] or row['frozen_model_policy_sha256']!=binding['frozen_model_policy_sha256'] or row['input_sha256']!=binding['input_sha256']:raise ValueError('A bridge block changed its registered index or identity')
        if i and rows[i-1]['after_generator_states']!=row['before_generator_states']:raise ValueError('A date-stream checkpoint changed between blocks')
        for phase in ('before','after'):
            if {k:hashed(v) for k,v in row[phase+'_generator_states'].items()}!=row[phase+'_generator_sha256']:raise ValueError('An actual generator state differs from its digest')
    if header['evaluation_streams']!=design['initial_date_streams'][str(fit['index'])] or completion['initial_generator_states']!=header['evaluation_streams']:raise ValueError('The bridge evaluation streams differ from their prior unconsumed registration')
    if {k:hashed(v) for k,v in completion['final_generator_states'].items()}!=completion['final_generator_sha256']:raise ValueError('A complete final generator state changed')
    if rows[0]['before_generator_states']!={k:v['initial_state'] for k,v in header['evaluation_streams'].items()} or rows[-1]['after_generator_states']!=completion['final_generator_states']:raise ValueError('The initial or final date-stream state changed')
    for count in LOOKS:
        prefix=read(folder/f'prepared_prefix_{count}.json')
        if prefix!=next(v for v in completion['prefix_preparation_records'] if v['paths']==count) or prefix['last_block_sha256']!=rows[count//batch-1]['block_sha256']:raise ValueError('A retained prefix changed')
        if prefix['paths']!=count or prefix['completed_blocks']!=count//batch or prefix['state_prefix_sha256']!=rows[count//batch-1]['cumulative_state_sha256'] or prefix['stopping_prefix_sha256']!=rows[count//batch-1]['cumulative_stopping_sha256']:raise ValueError('The fixed-look count or binary prefix digest changed')
        valid_cost(prefix['complete_preparation_prefix_cost'])
    return header,rows,completion,elapsed(start)


def binary_batches(folder,rows):
    layout=struct.Struct('<'+'d'*50)
    with (folder/'states.bin').open('rb') as states,(folder/'stops.bin').open('rb') as stops:
        sh=hashlib.sha256();th=hashlib.sha256()
        for row in rows:
            raw=states.read(row['state_bytes']);chosen=stops.read(row['stopping_bytes'])
            sh.update(raw);th.update(chosen)
            if hashlib.sha256(raw).hexdigest()!=row['state_sha256'] or hashlib.sha256(chosen).hexdigest()!=row['stopping_sha256'] or sh.hexdigest()!=row['cumulative_state_sha256'] or th.hexdigest()!=row['cumulative_stopping_sha256']:raise ValueError('A binary block changed from its sealed generator checkpoint')
            observations=[]
            for i,values in enumerate(layout.iter_unpack(raw)):
                state=[list(values[j:j+5]) for j in range(0,50,5)]
                if len(state)!=10 or any(not math.isfinite(x) or x<=0 for v in state for x in v) or state[0]!=[100.]*5 or not 1<=chosen[i]<10:raise ValueError('A bridge path has an invalid state or stopping date')
                observations.append((row['path_start']+i,state,int(chosen[i])))
            if len(observations)!=row['paths']:raise ValueError('An incomplete binary path block cannot be evaluated')
            yield row,observations
        if states.read(1) or stops.read(1):raise ValueError('Undeclared trailing paths cannot enter a fixed-look study')


TIMING_FIELDS={'complete_path_cost','complete_query_cost','complete_attempt_cost','cost'}

def numerical_record(value):
    if isinstance(value,list):return [numerical_record(v) for v in value]
    if isinstance(value,dict):return {k:numerical_record(v) for k,v in value.items() if k not in TIMING_FIELDS}
    return value


def validate_zero_native(native,state,stop,helper):
    if native.get('completed') is not True or native.get('error') is not None or native.get('stopping_index')!=stop or native.get('rights')!=len(state):raise ValueError('The zero-martingale record changed its outcome or stopping date')
    rewards=[helper.decode_interval(v) for v in native['reward_certificates']]
    candidates={str(j):((Fraction(0),Fraction(0)) if j==stop else (rewards[j][0]-rewards[stop][1],rewards[j][1]-rewards[stop][0])) for j in range(1,len(state))}
    if {str(j):helper.decode_interval(v) for j,v in native['candidate_certificates'].items()}!=candidates:raise ValueError('The zero-martingale candidate reconstruction differs')
    expected=(max(Fraction(0),max(p[0] for p in candidates.values())),max(Fraction(0),max(p[1] for p in candidates.values())))
    if helper.decode_interval(native['gain_certificate'])!=expected:raise ValueError('The zero-martingale gain reconstruction differs')
    expected_state=hashlib.sha256(json.dumps([[[str(Fraction(x).numerator),str(Fraction(x).denominator)] for x in vector] for vector in state],separators=(',',':')).encode()).hexdigest()
    if native['state_sha256']!=expected_state:raise ValueError('The zero-martingale assessor changed its supplied path')
    return expected


def assess_one(folder,output,rows,fit,design,assessor,tail,tail_cost,setup_cost,training,helper,allocator,validator,model,shared_guard_cost):
    if output.exists():raise ValueError('An assessor attempt cannot be overwritten')
    output.mkdir(parents=True)
    accumulator=helper.IntervalClippedAccumulator(CAPS[assessor]);work={k:0 for k in validator.WORK_KEYS};retries=0
    count=0;look_records=[];raw_hash=hashlib.sha256();native=None;phase='start';index=None;maximum=Fraction(0)
    raw_lower=Fraction(0);raw_upper=Fraction(0)
    phase_costs={key:zero_cost() for key in ('gain_evaluation','native_validation','statistics_and_record_construction','evidence_compression','fixed_look_upper_limit')}
    start=clocks()
    try:
        with (output/'observations.jsonl.gz').open('xb') as destination:
            with gzip.GzipFile(filename='',mode='wb',fileobj=destination,mtime=0) as archive:
                for row,observations in binary_batches(folder,rows):
                    if shutil.disk_usage(output).free<MINIMUM_FREE_BYTES:raise OSError('Private evidence reserve is insufficient; preserve this attempt')
                    for index,state,stop in observations:
                        phase='gain_evaluation';native=None;phase_start=clocks()
                        if assessor=='zero':
                            native=helper.evaluate_zero_gain(state,model['exercise_times'],stop,strike=model['strike'],rate=model['rate'],precision=PRECISION)
                            lower,upper=helper.decode_interval(native['gain_certificate'])
                            path_work={k:0 for k in work};path_retries=0
                            path_cost=dict(cpu_seconds=native['cost']['assessment_cpu_seconds'],wall_seconds=native['cost']['assessment_wall_seconds'])
                            phase_costs['gain_evaluation']=summed_cost(phase_costs['gain_evaluation'],elapsed(phase_start))
                            phase_start=clocks();validate_zero_native(native,state,stop,helper)
                            phase_costs['native_validation']=summed_cost(phase_costs['native_validation'],elapsed(phase_start))
                        else:
                            native=allocator.evaluate_gain(state,model['exercise_times'],stop,
                              strike=model['strike'],rate=model['rate'],dividend=model['dividends'][0],volatility=model['volatilities'][0],
                              arm='coarse_common',target_width=TARGET,initial_value_width=INITIAL,max_rounds=16,
                              precision=PRECISION,repository=Path(design['root']),max_oracle_attempts=8,cutoff_multiplier=2)
                            phase_costs['gain_evaluation']=summed_cost(phase_costs['gain_evaluation'],elapsed(phase_start))
                            phase='native_validation';phase_start=clocks()
                            validator.validate_native(native,TARGET,arm='coarse_common',initial=INITIAL,precision=PRECISION,calendar_length=10,stopping_index=stop,states=state)
                            lower,upper=helper.decode_interval(native['gain_certificate'])
                            path_work=native['total_work'];path_retries=validator.retry_count(native)
                            path_cost=native['complete_path_cost']
                            phase_costs['native_validation']=summed_cost(phase_costs['native_validation'],elapsed(phase_start))
                        if upper-lower>TARGET:raise ArithmeticError('The completed gain interval exceeds its fixed numerical tolerance')
                        phase='statistics_and_compact_evidence';phase_start=clocks()
                        raw_lower+=lower;raw_upper+=upper
                        accumulator.add_interval(lower,upper);maximum=max(maximum,upper-lower)
                        for key in work:work[key]+=path_work[key]
                        retries+=path_retries;count+=1
                        record=dict(path_index=index,stopping_index=stop,gain_certificate=helper.interval_record(lower,upper),
                          native_ledger_sha256=hashed(native),numerical_ledger_sha256=hashed(numerical_record(native)),total_work=path_work,internal_retries=path_retries,
                          path_cpu_and_wall_cost=valid_cost(path_cost),assessment=native)
                        encoded=canonical(record)+b'\n'
                        phase_costs['statistics_and_record_construction']=summed_cost(phase_costs['statistics_and_record_construction'],elapsed(phase_start))
                        phase_start=clocks();archive.write(encoded);raw_hash.update(encoded)
                        phase_costs['evidence_compression']=summed_cost(phase_costs['evidence_compression'],elapsed(phase_start))
                        native=None
                    if count in LOOKS:
                        phase='fixed_look_upper_limit';phase_start=clocks()
                        limit=accumulator.upper_limit(delta=DELTA,tail_upper=helper.decode_interval(tail['clipping_tail_allowance'])[1],precision=PRECISION)
                        authoritative=helper.decode_rational(limit['upper_limit'])
                        acceptance={str(t):authoritative<=t for t in TOLERANCES}
                        phase_costs['fixed_look_upper_limit']=summed_cost(phase_costs['fixed_look_upper_limit'],elapsed(phase_start))
                        phase_start=clocks();archive.flush()
                        phase_costs['evidence_compression']=summed_cost(phase_costs['evidence_compression'],elapsed(phase_start))
                        assess_cost=elapsed(start)
                        preparation=read(folder/f'prepared_prefix_{count}.json')['complete_preparation_prefix_cost']
                        standalone=summed_cost(training,preparation,shared_guard_cost,setup_cost,tail_cost,assess_cost)
                        look=dict(paths=count,upper_limit=limit,acceptance=acceptance,
                          clipped_gain_mean_certificate=helper.interval_record(accumulator.sum_lower/count,accumulator.sum_upper/count),
                          raw_gain_mean_certificate=helper.interval_record(raw_lower/count,raw_upper/count),
                          exclusive_phase_costs={key:value.copy() for key,value in phase_costs.items()},
                          cumulative_work=work.copy(),internal_retries=retries,maximum_gain_width=helper.rational_record(maximum),
                          actual_assessment_prefix_cost=assess_cost,required_prefix_preparation_charge=preparation,
                          reconstructed_standalone_prefix_cost=standalone,
                          observation_prefix_sha256=raw_hash.hexdigest(),
                          cost_scope='Reconstructed required-prefix charge. All fresh preparation was executed before assessment, so this is not the chronological cost spent when the decision became available. The measured full preparation-family global overhead, shared full bridge verification, own module setup and own population-tail construction are charged once; retained policy training is historical, with earlier assessments and value fitting excluded.')
                        look_records.append(look);write(output/f'look_{count}.json',look)
                        print(f'Fit {fit["index"]}, {assessor}, {count} paths: upper limit {float(authoritative):.6f}',flush=True)
        complete=elapsed(start)
        decisions={}
        for tolerance in TOLERANCES:
            eligible=[r for r in look_records if r['acceptance'][str(tolerance)]]
            chosen=eligible[0] if eligible else look_records[-1]
            decisions[str(tolerance)]=dict(accepted=bool(eligible),required_paths=chosen['paths'],
              upper_limit=chosen['upper_limit']['upper_limit'],required_prefix_cost=chosen['reconstructed_standalone_prefix_cost'],
              full_budget_charged_if_unresolved=not bool(eligible))
        result=dict(schema_version=1,author='Amanjeet Singh',completed=True,fit_index=fit['index'],assessor=assessor,
          design_sha256=design['_sha256'],paths=count,look_records=look_records,decisions=decisions,
          total_work=work,internal_retries=retries,exclusive_phase_costs=phase_costs,actual_complete_assessment_cost=complete,
          complete_unallocated_bookkeeping_cost={key:complete[key]-sum(v[key] for v in phase_costs.values()) for key in complete},
          tail_certificate_sha256=hashed(tail),own_tail_construction_cost=tail_cost,own_module_setup_cost=setup_cost,
          shared_full_bridge_verification_charge=shared_guard_cost,retained_training_charge=training,
          compressed_observation_sha256=digest(output/'observations.jsonl.gz'),uncompressed_observation_sha256=raw_hash.hexdigest(),
          compressed_observation_bytes=(output/'observations.jsonl.gz').stat().st_size,
          numerical_gain_construction_completed=True,ideal_iid_sampling_premise_verified_by_computer=False,
          policy_improved=False,market_data_used=False)
        write(output/'completion.json',result);return result
    except Exception as error:
        carried=getattr(error,'record',None) or getattr(error,'partial_record',None)
        if index is not None:write(output/'failed_path_input.json',dict(path_index=index,states=state,stopping_index=stop))
        if native is not None:write(output/'failed_native_ledger.json',native)
        if carried is not None:write(output/'exception_carried_native_ledger.json',carried)
        failure=dict(schema_version=1,author='Amanjeet Singh',completed=False,fit_index=fit['index'],assessor=assessor,
          design_sha256=design['_sha256'],failed_path_index=index,failed_phase=phase,completed_paths=count,numerically_completed_paths=count,
          evidence_scope='Completed paths count numerical results; a compression failure may leave fewer independently readable records. Raw binary inputs, partial sufficient statistics and failed native evidence remain retained.',
          exact_partial_statistics=accumulator.snapshot(),known_successful_work=work,known_successful_internal_retries=retries,
          failed_native_work=None if carried is None else carried.get('total_work'),
          failed_native_work_scope='Failure ledger is separate from successful counts; unavailable internal work remains explicitly unavailable.',
          actual_failure_cost=elapsed(start),exception_type=type(error).__name__,message=str(error),
          continuation='Preserve the attempt and partial evidence. No substitution, restart, count reduction or tolerance relaxation is permitted.')
        write(output/'failure.json',failure);raise


def assess(args):
    root=Path(args.root).resolve();study=private_path(root,args.study)
    prep=private_path(root,args.preparation);output=private_path(root,args.output,fresh=True)
    whole=clocks();design_bytes=(study/'design.json').read_bytes();design=json.loads(design_bytes)
    preparation_start=clocks();preparer=load_module(root/'scripts/prepare_financial_decision.py','financial_preparation_assessment')
    validate_design(design,preparer);retained=private_path(root,design['retained_study'])
    before=guard_snapshot(root,retained,design)
    if digest(Path(__file__))!=design['source_sha256']['scripts/study_financial_decision.py']:raise ValueError('The executed assessment driver is not the sealed installed source')
    validator=load_module(root/'scripts/study_european_gain_allocation.py','financial_native_validation_assessment')
    if design['certification_runtime']!=validator.runtime_identity():raise ValueError('The certification runtime changed')
    design['_sha256']=hashlib.sha256(design_bytes).hexdigest()
    prep_completion=read(prep/'completion.json')
    if prep_completion.get('completed') is not True or prep_completion['design_sha256']!=design['_sha256'] or prep_completion['completed_fits']!=8:raise ValueError('The complete registered preparation is required')
    shared_source_validation=elapsed(preparation_start)
    preparation_global_overhead={key:prep_completion['complete_preparation_workflow_cost'][key]-sum(f['complete_preparation_cost'][key] for f in prep_completion['fits']) for key in zero_cost()}
    valid_cost(preparation_global_overhead)
    output.mkdir(parents=True)
    write(output/'initial_attempt.json',dict(schema_version=1,author='Amanjeet Singh',recorded_at_utc=now(),design_sha256=design['_sha256'],source_snapshot_before=before,stage='module_and_population_tail_setup'))
    tail_setup_start=clocks();setup_phase='helper_module_loading'
    try:
        start=clocks();helper=load_module(root/'scripts/certify_financial_decision.py','financial_decision_helper')
        helper_setup=elapsed(start)
        setup_phase='european_module_loading'
        start=clocks();allocator=load_module(root/'scripts/refine_european_gain.py','financial_decision_european')
        european_setup=elapsed(start)
        model=read(retained/'fit_00/inputs.json')['model']
        setup_phase='zero_population_tail'
        start=clocks();zero_tail=helper.build_zero_gain_tail_bound(model['spots'],model['exercise_times'],
          strike=model['strike'],rate=model['rate'],dividend=model['dividends'][0],volatility=model['volatilities'][0],cap=150,precision=PRECISION,repository=root)
        zero_tail_cost=elapsed(start)
        write(output/'zero_tail_setup.json',dict(certificate=zero_tail,complete_setup_cost=zero_tail_cost))
        setup_phase='european_population_tail'
        start=clocks();eu_tail=allocator.tail.build_european_restricted_tail_bound(model['spots'],model['exercise_times'],
          strike=model['strike'],rate=model['rate'],dividend=model['dividends'][0],volatility=model['volatilities'][0],b_A=20,b_D=30,
          precision=PRECISION,repository=root,cutoff_multiplier=8,absolute_tolerance=Fraction(1,10**10),relative_tolerance=Fraction(1,10**10))
        european_tail_cost=elapsed(start)
        tails={'zero':zero_tail,'european_common':eu_tail}
        tail_costs={'zero':zero_tail_cost,'european_common':european_tail_cost}
        setup_costs={'zero':helper_setup,'european_common':summed_cost(helper_setup,european_setup)}
    except Exception as error:
        carried=getattr(error,'record',None)
        if carried is not None:write(output/'tail_exception_evidence.json',carried)
        write(output/'failure.json',dict(schema_version=1,author='Amanjeet Singh',completed=False,design_sha256=design['_sha256'],failed_phase=setup_phase,actual_setup_failure_cost=elapsed(tail_setup_start),exception_type=type(error).__name__,message=str(error),unavailable_internal_work_explicit=True,continuation='Preserve this failed setup and any carried evidence. Do not overwrite or silently restart it.'))
        raise
    for assessor,tail in tails.items():write(output/(assessor+'_tail.json'),tail)
    write(output/'attempt.json',dict(schema_version=1,author='Amanjeet Singh',recorded_at_utc=now(),design_sha256=design['_sha256'],
      source_snapshot_before=before,preparation_completion_sha256=digest(prep/'completion.json'),
      shared_source_and_runtime_validation_cost=shared_source_validation,
      complete_preparation_global_overhead_charge=preparation_global_overhead,own_module_setup_costs=setup_costs,own_tail_construction_costs=tail_costs))
    results=[];fit_index=None
    try:
        for fit in design['fits']:
            fit_index=fit['index'];folder=prep/f'fit_{fit_index:02d}'
            header,rows,bridge,bridge_cost=verify_bridge(folder,design,fit,preparer)
            if bridge!=prep_completion['fits'][fit_index] or prep_completion['runtime']!=design['preparation_runtime'] or prep_completion['source_snapshot_before']!=before or prep_completion['source_snapshot_after']!=before:raise ValueError('A fit bridge differs from the sealed preparation workflow')
            if model!=header['model']:raise ValueError('A fit changed its complete declared model')
            fit_results={};fit_output=output/f'fit_{fit_index:02d}';fit_output.mkdir()
            for assessor in design['assessor_order_by_fit'][fit_index]:
                fit_results[assessor]=assess_one(folder,fit_output/assessor,rows,fit,design,assessor,tails[assessor],
                  tail_costs[assessor],setup_costs[assessor],fit['retained_policy_training_cost'],helper,allocator,validator,model,
                  summed_cost(shared_source_validation,bridge_cost,preparation_global_overhead))
            post_start=clocks()
            current_bridge=read(folder/'completion.json')
            if current_bridge!=bridge:raise ValueError('The bridge completion changed after assessment')
            for filename,key in [('input_manifest.json','input_manifest_sha256'),('block_metadata.jsonl','block_metadata_sha256'),('states.bin','state_sha256'),('stops.bin','stopping_sha256')]:
                if digest(folder/filename)!=bridge[key]:raise ValueError('A prepared bridge changed during assessment')
            post_bridge_cost=elapsed(post_start)
            results.append(dict(fit_index=fit_index,assessors=fit_results,bridge_verification_cost=bridge_cost,post_assessment_bridge_verification_cost=post_bridge_cost))
            guard_snapshot(root,retained,design)
        if (study/'design.json').read_bytes()!=design_bytes:raise ValueError('The fixed design changed during assessment')
        after=guard_snapshot(root,retained,design)
        summary=dict(schema_version=1,author='Amanjeet Singh',completed=True,design_sha256=design['_sha256'],
          recorded_at_utc=now(),fits=results,completed_fits=8,paths_per_assessor=8*16384,
          fixed_upper_events=32,family_failure_probability='1/20',source_snapshot_before=before,source_snapshot_after=after,
          actual_assessment_workflow_cost=elapsed(whole),actual_preparation_workflow_cost=prep_completion['complete_preparation_workflow_cost'],
          execution_scope='Fresh sampling and frozen-policy evaluation plus outward gain, tail and fixed-look sampling constructions. Original training costs are retained historical observations. Exact numerical endpoints do not certify the floating-point simulator law or market profitability.',
          substantive_original_research_contribution_established=False)
        write(output/'summary.json',summary)
        print(json.dumps(dict(completed=True,completed_fits=8,summary_sha256=digest(output/'summary.json'))),flush=True)
        return summary
    except Exception as error:
        write(output/'failure.json',dict(schema_version=1,author='Amanjeet Singh',completed=False,design_sha256=design['_sha256'],
          failed_fit_index=fit_index,completed_fits=len(results),actual_failure_workflow_cost=elapsed(whole),
          exception_type=type(error).__name__,message=str(error),continuation='Preserve this attempt. Do not silently continue or replace failed observations.'))
        raise


def main():
    parser=ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    registration=commands.add_parser('register');registration.add_argument('--root',type=Path,required=True)
    registration.add_argument('--retained-study',type=Path,required=True);registration.add_argument('--output',type=Path,required=True)
    registration.add_argument('--research-python',type=Path,required=True);registration.add_argument('--design-note',type=Path);registration.add_argument('--proof-note',type=Path)
    execution=commands.add_parser('assess');execution.add_argument('--root',type=Path,required=True);execution.add_argument('--study',type=Path,required=True)
    execution.add_argument('--preparation',type=Path,required=True);execution.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();(register if args.command=='register' else assess)(args)

if __name__=='__main__':main()
