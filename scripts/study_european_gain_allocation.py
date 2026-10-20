"""Compare allocation rules for the same European-value stopped gain.

I reuse a fixed prefix of each retained public-model evaluation stream. No
simulation or fitting is performed. Both rules start at the same requested
value width and use the same numerical oracle and ledger format. A separately
recomputed tighter reference uses that same oracle, not an independent
algorithm or an optimal stopping value. All results remain private.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time

FITS=8
PATHS_PER_FIT=32
ARMS=('coarse_common','gain_adaptive')
TARGET=Fraction(1,100)
INITIAL_WIDTH=Fraction(1)
MAX_ROUNDS=16
MAX_ORACLE_ATTEMPTS=8
PRECISION=96
REFERENCE_TARGET=Fraction(1,100000000)
REFERENCE_PRECISION=128
WORK_KEYS=('conditional_integral_calls','successful_integral_certificates',
           'analytic_evaluations','integrand_evaluations_known',
           'integrand_evaluations_unavailable_attempts','oracle_attempts')


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime_identity(*,require_flint=True):
    try:flint_version=importlib.metadata.version('python-flint')
    except importlib.metadata.PackageNotFoundError:
        if require_flint:raise RuntimeError('The certification runtime must identify its python-flint installation')
        flint_version=None
    return dict(python_version=sys.version,python_executable=sys.executable,
        platform=platform.platform(),machine=platform.machine(),python_flint_version=flint_version)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path):
    with path.open() as f:return json.load(f)


def write_json(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def write_gzip(path,value):
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    with path.open('xb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=destination,mtime=0) as archive:
            archive.write(encoded)
    return dict(compressed_sha256=digest(path),uncompressed_json_sha256=hashlib.sha256(encoded).hexdigest(),
                compressed_bytes=path.stat().st_size,uncompressed_bytes=len(encoded))


def rational(value):
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),approximation=float(value))


def pair_record(pair):
    return dict(lower=rational(pair[0]),upper=rational(pair[1]),width=rational(pair[1]-pair[0]))


def decode(value):
    if type(value) is not dict or int(value['denominator'])<=0:raise ValueError('Invalid rational endpoint')
    return Fraction(int(value['numerator']),int(value['denominator']))


def decode_pair(value):
    lo,hi=decode(value['lower']),decode(value['upper'])
    if lo>hi or decode(value['width'])!=hi-lo:raise ValueError('Invalid interval certificate')
    return lo,hi


def valid_cost(value):
    for key in ('cpu_seconds','wall_seconds'):
        if isinstance(value[key],bool) or not isinstance(value[key],(int,float)) or not math.isfinite(value[key]) or value[key]<0:
            raise ValueError('A complete finite nonnegative cost is required')
    return value


def measured(start):
    return dict(cpu_seconds=time.process_time()-start[0],wall_seconds=time.perf_counter()-start[1])


def clock_pair():
    return time.process_time(),time.perf_counter()


def add_cost(total,cost):
    for k in total:total[k]+=cost[k]


def aggregate_work(rows):
    answer={key:0 for key in WORK_KEYS}
    for row in rows:
        work=row['total_work']
        if set(work)!=set(WORK_KEYS):raise ValueError('Unexpected native work inventory')
        for key in WORK_KEYS:
            if type(work[key]) is not int or work[key]<0:raise ValueError('Invalid native work counter')
            answer[key]+=work[key]
    return answer


def retry_count(row):
    queries=list(row['initial_queries'])
    for round_record in row['refinement_rounds']:
        queries.extend(round_record['queries'])
    if any(not isinstance(query['attempts'],list) for query in queries):
        raise ValueError('Every query must retain its complete raw attempt list')
    return sum(max(0,len(query['attempts'])-1) for query in queries)


def dated_intervals(values,calendar_length):
    if isinstance(values,dict):
        answer={int(key):decode_pair(row) for key,row in values.items()}
    elif isinstance(values,list) and len(values)==calendar_length:
        answer={j:decode_pair(row) for j,row in enumerate(values) if j>0}
    elif isinstance(values,list) and len(values)==calendar_length-1:
        answer={j:decode_pair(row) for j,row in enumerate(values,1)}
    else:raise ValueError('Unexpected dated-certificate layout')
    if set(answer)!=(set(range(1,calendar_length))):
        raise ValueError('Every positive exercise date needs a final certificate')
    return answer


def reconstruct_gain(native,calendar_length,stopping_index):
    rewards=dated_intervals(native['reward_certificates'],calendar_length)
    values=dated_intervals(native['value_certificates'],calendar_length)
    final=calendar_length-1
    premiums={j:(rewards[j][0]-values[j][1],rewards[j][1]-values[j][0]) for j in rewards}
    # The same terminal payoff defines both reward and European value.
    if rewards[final]!=values[final]:
        raise ValueError('Terminal reward and value certificates must be exactly identical')
    premiums[final]=(Fraction(0),Fraction(0))
    candidates=[(Fraction(0),Fraction(0)) if j==stopping_index else
                (premiums[j][0]-premiums[stopping_index][1],
                 premiums[j][1]-premiums[stopping_index][0]) for j in range(1,calendar_length)]
    return max(x[0] for x in candidates),max(x[1] for x in candidates)


def level_boxes(rewards,values,final):
    if rewards[final]!=values[final]:raise ValueError('Terminal reward and value certificates must be exactly identical')
    return {j:(Fraction(0),Fraction(0)) if j==final else
        (rewards[j][0]-values[j][1],rewards[j][1]-values[j][0]) for j in rewards}


def gain_from_boxes(levels,stopping_index):
    selected=levels[stopping_index]
    return (max([Fraction(0)]+[x[0]-selected[1] for j,x in levels.items() if j!=stopping_index]),
        max([Fraction(0)]+[x[1]-selected[0] for j,x in levels.items() if j!=stopping_index]))


def active_dates(levels,stopping_index,final):
    candidates=[j for j in levels if j!=stopping_index]
    if not candidates:return []
    witness=max(candidates,key=lambda j:(levels[j][0],j==final,-j))
    return [j for j in candidates if not (levels[j][1]<=levels[stopping_index][0] or
        (j!=witness and levels[j][1]<=levels[witness][0]))]


def validate_query(query,requested,previous,*,phase,date):
    if query['date']!=date or type(query['date']) is not int or query['phase']!=phase:
        raise ValueError('A numerical query changed its declared date or phase')
    if decode_pair(query['requested_width_certificate'])!=(requested,requested):
        raise ValueError('A numerical query changed its prescribed common or dated request')
    actual_previous=None if query['previous_value_certificate'] is None else decode_pair(query['previous_value_certificate'])
    if actual_previous!=previous:raise ValueError('The query preceding enclosure changed')
    attempts=query['attempts']
    if not isinstance(attempts,list) or not 1<=len(attempts)<=MAX_ORACLE_ATTEMPTS:
        raise ValueError('A query exceeded its registered raw attempt budget')
    returned=decode_pair(query['value_certificate'])
    if returned[0]<0 or returned[1]-returned[0]>requested:
        raise ValueError('A returned value enclosure did not attain its prescribed width')
    retained=returned if previous is None else (max(returned[0],previous[0]),min(returned[1],previous[1]))
    if retained[1]<retained[0] or decode_pair(query['retained_value_certificate'])!=retained:
        raise ValueError('The retained value does not equal the complete intersection')
    total={key:0 for key in WORK_KEYS}
    for index,attempt in enumerate(attempts,1):
        if type(attempt['attempt_index']) is not int or attempt['attempt_index']!=index:
            raise ValueError('Raw attempt indices are incomplete or unordered')
        work=aggregate_work([dict(total_work=attempt['work'])])
        if work['oracle_attempts']!=1:raise ValueError('Every raw attempt must be counted once')
        for key in WORK_KEYS:total[key]+=work[key]
        valid_cost(attempt['complete_attempt_cost'])
        if type(attempt['returned_finite_certificate']) is not bool or type(attempt['requested_width_attained']) is not bool:
            raise ValueError('Raw attempt outcomes must be explicit booleans')
    last=attempts[-1]
    if last['returned_finite_certificate'] is not True or last['requested_width_attained'] is not True or decode_pair(last['value_certificate'])!=returned:
        raise ValueError('A completed query contradicts its final raw attempt')
    if query['work']!=total or query['work_complete'] is not (total['integrand_evaluations_unavailable_attempts']==0):
        raise ValueError('Query work does not preserve the complete raw attempt ledger')
    valid_cost(query['complete_query_cost'])
    return retained,total


def validate_trace(native,*,arm,target,initial,calendar_length,stopping_index):
    final=calendar_length-1
    dates=native['initially_queried_nonterminal_dates']
    if not isinstance(dates,list) or dates!=sorted(set(dates)) or any(type(j) is not int or not 1<=j<final for j in dates):
        raise ValueError('Initially queried nonterminal dates must be complete, unique and ordered')
    n=len(dates);budget=MAX_ROUNDS*n
    if native['maximum_refinements_per_date']!=MAX_ROUNDS or type(native['maximum_refinements_per_date']) is not int:
        raise ValueError('The per-date refinement cap changed')
    if native['additional_query_budget']!=budget or type(native['additional_query_budget']) is not int:
        raise ValueError('The additional-query budget changed')
    decisions=budget if arm=='gain_adaptive' else MAX_ROUNDS
    if native['maximum_refinement_decisions']!=decisions or type(native['maximum_refinement_decisions']) is not int:
        raise ValueError('The registered decision cap changed')
    if decode_pair(native['request_width_floor_certificate'])!=(initial/4**MAX_ROUNDS,initial/4**MAX_ROUNDS):
        raise ValueError('The per-date request floor changed')
    rounds=native['refinement_rounds'];initial_queries=native['initial_queries']
    if not isinstance(rounds,list) or len(rounds)>decisions or [row['date'] for row in initial_queries]!=dates:
        raise ValueError('Initial queries or native refinement decisions exceed the declared design')
    rewards=dated_intervals(native['reward_certificates'],calendar_length)
    final_values=dated_intervals(native['value_certificates'],calendar_length)
    if rewards[final]!=final_values[final]:raise ValueError('Terminal reward and value certificates must be exactly identical')
    values={final:rewards[final]};work={key:0 for key in WORK_KEYS};counts={j:0 for j in dates};depths={j:0 for j in dates}
    for excluded in native['excluded_dates']:
        j=excluded['date']
        if type(j) is not int or not 1<=j<final or j==stopping_index or j in values or j in dates:
            raise ValueError('An excluded date conflicts with an initial or selected date')
        if rewards[j]!=(Fraction(0),Fraction(0)) or excluded['reason']!='nonselected_zero_reward_and_nonnegative_European_value':
            raise ValueError('A structural exclusion is unsupported')
        values[j]=decode_pair(excluded['structural_value_certificate'])
        if values[j][0]!=0:raise ValueError('A screened premium requires its nonnegative structural lower endpoint')
    for j,query in zip(dates,initial_queries):
        values[j],effort=validate_query(query,initial,None,phase='initial',date=j)
        for key in work:work[key]+=effort[key]
    if set(values)!=set(rewards):raise ValueError('The retained initial enclosures omit an exercise date')
    for index,round_record in enumerate(rounds,1):
        common_keys={'round_index','selected_dates','dominance_certificates','queries','gain_certificate_before','gain_certificate_after','attained_width_reduction'}
        variant_keys={'optimistic_scores','skipped_already_attained_quarter_requests','requested_quarter_depth'} if arm=='gain_adaptive' else {'common_requested_width_certificate'}
        if set(round_record)!=common_keys|variant_keys:
            raise ValueError('The native round schema does not match its registered allocation arm')
        before=gain_from_boxes(level_boxes(rewards,values,final),stopping_index)
        if before[1]-before[0]<=target:raise ValueError('Refinement continued after the actual gain target was attained')
        if type(round_record['round_index']) is not int or round_record['round_index']!=index or decode_pair(round_record['gain_certificate_before'])!=before:
            raise ValueError('The native round index or preceding gain enclosure changed')
        queries=round_record['queries'];selected=round_record['selected_dates']
        active=active_dates(level_boxes(rewards,values,final),stopping_index,final)
        if arm=='gain_adaptive':
            if len(queries)!=1 or len(selected)!=1 or selected!=[queries[0]['date']]:
                raise ValueError('An adaptive decision must retain exactly one actual query')
            j=selected[0]
            if j not in dates or not (j==stopping_index or j in active) or counts[j]>=MAX_ROUNDS:
                raise ValueError('An adaptive query violates shared eligibility or its per-date cap')
            depth=depths[j]+1;skipped=0
            while values[j][1]-values[j][0]<=initial/4**depth and depth<MAX_ROUNDS:depth+=1;skipped+=1
            if depth>MAX_ROUNDS or round_record['requested_quarter_depth']!=depth or round_record['skipped_already_attained_quarter_requests']!=skipped:
                raise ValueError('The adaptive dated quarter-depth changed')
            requested=initial/4**depth
            if values[j][1]-values[j][0]<=requested:raise ValueError('An adaptive query refines an already compliant bracket')
        else:
            requested=initial/4**index
            if decode_pair(round_record['common_requested_width_certificate'])!=(requested,requested):
                raise ValueError('The common global quarter-width changed')
            expected=[j for j in dates if (j==stopping_index or j in active) and values[j][1]-values[j][0]>requested]
            if selected!=expected or [query['date'] for query in queries]!=expected:
                raise ValueError('The common round omitted or added a surviving eligible value')
            depth=index
        for query in queries:
            j=query['date'];counts[j]+=1;depths[j]=depth
            if counts[j]>MAX_ROUNDS:raise ValueError('A date exceeded its actual refinement-query cap')
            values[j],effort=validate_query(query,requested,values[j],phase='refinement',date=j)
            for key in work:work[key]+=effort[key]
        after=gain_from_boxes(level_boxes(rewards,values,final),stopping_index)
        if decode_pair(round_record['gain_certificate_after'])!=after or decode(round_record['attained_width_reduction'])!=before[1]-before[0]-(after[1]-after[0]):
            raise ValueError('The complete-round gain reconstruction changed')
    if native['additional_queries']!=sum(counts.values()) or type(native['additional_queries']) is not int or sum(counts.values())>budget:
        raise ValueError('The complete additional-query count changed or exceeded its cap')
    count_record=native['refinement_counts_by_date'];depth_record=native['requested_quarter_depths_by_date']
    if any(type(value) is not int for value in list(count_record.values())+list(depth_record.values())) or count_record!={str(j):counts[j] for j in dates} or depth_record!={str(j):depths[j] for j in dates}:
        raise ValueError('Per-date refinement or quarter-depth counts differ from the retained trace')
    if values!=final_values or native['total_work']!=work:
        raise ValueError('Final premium enclosures or work differ from the complete query trace')
    levels=level_boxes(rewards,values,final)
    if dated_intervals(native['F_certificates'],calendar_length)!=levels:
        raise ValueError('Final exercise-premium levels differ from the shared terminal identity')
    return gain_from_boxes(levels,stopping_index)


def validate_native(native,target,*,arm='coarse_common',initial=INITIAL_WIDTH,precision=PRECISION,calendar_length=None,stopping_index=None,states=None):
    if native['completed'] is not True or native['error'] is not None:raise ValueError('The native assessment did not complete')
    if native['arm']!=arm or native['precision_bits']!=precision or type(native['precision_bits']) is not int:
        raise ValueError('The requested arm or arithmetic precision changed')
    if decode_pair(native['target_width_certificate'])!=(target,target) or decode_pair(native['initial_value_width_certificate'])!=(initial,initial):
        raise ValueError('The requested gain or initial premium target changed')
    if native['maximum_oracle_attempts_per_requested_value']!=MAX_ORACLE_ATTEMPTS or type(native['maximum_oracle_attempts_per_requested_value']) is not int:
        raise ValueError('The raw oracle-attempt cap changed')
    if native['shared_structural_screening_and_dominance_pruning'] is not True or native['policy_improvement_claim'] is not False or native['paths_generated'] is not False or native['policy_fitting_executed'] is not False:
        raise ValueError('The registered structural rules or retained-sample scope changed')
    calendar_length=native['rights'] if calendar_length is None else calendar_length
    stopping_index=native['stopping_index'] if stopping_index is None else stopping_index
    if type(native['rights']) is not int or native['rights']!=calendar_length or type(native['stopping_index']) is not int or native['stopping_index']!=stopping_index:
        raise ValueError('The calendar or selected exercise date changed')
    if not 1<=stopping_index<calendar_length:raise ValueError('A positive selected exercise date is required')
    if states is not None:
        exact_states=[[[str(Fraction(x).numerator),str(Fraction(x).denominator)] for x in state] for state in states]
        state_hash=hashlib.sha256(json.dumps(exact_states,separators=(',',':')).encode()).hexdigest()
        if native['state_sha256']!=state_hash or native['dimension']!=len(states[0]):raise ValueError('The assessment changed its retained path')
    gain=decode_pair(native['gain_certificate'])
    if gain[0]<0 or gain[1]-gain[0]>target:raise ValueError('The requested gain target was not attained')
    valid_cost(native['complete_path_cost'])
    aggregate_work([native])
    retry_count(native)
    if validate_trace(native,arm=arm,target=target,initial=initial,calendar_length=calendar_length,stopping_index=stopping_index)!=gain:
        raise ValueError('The native gain differs from direct full-trace reconstruction')
    return gain


def arm_summary(observations,complete_cost):
    intervals=[decode_pair(row['assessment']['gain_certificate']) for row in observations]
    n=len(intervals)
    if n!=PATHS_PER_FIT:raise ValueError('A complete registered path prefix is required')
    means=sum(x[0] for x in intervals)/n,sum(x[1] for x in intervals)/n
    mids=[float(sum(x)/2) for x in intervals]
    return dict(paths=n,mean_gain_certificate=pair_record(means),maximum_gain_width=rational(max(x[1]-x[0] for x in intervals)),
        descriptive_midpoint_mean=statistics.mean(mids),descriptive_midpoint_sample_standard_deviation=statistics.stdev(mids),
        total_work=aggregate_work([x['assessment'] for x in observations]),
        retry_attempts=sum(retry_count(x['assessment']) for x in observations),
        complete_assessment_cost=complete_cost,observations=observations)


def model_context(record,index):
    model=record['model'];times=model['exercise_times'];d=len(model['spots'])
    if record['design']['root_seed']!=2026100410+index or record['design']['outer_paths']!=1024:
        raise ValueError('Unexpected retained fit identity or evaluation count')
    if len(record['states'])!=1024 or len(record['stopping_indices'])!=1024 or d!=5 or len(times)!=10:
        raise ValueError('The complete retained public benchmark input is required')
    if times[0]!=0 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('A strictly increasing calendar starting at zero is required')
    if model['rate']<0 or model['strike']<=0 or model['dividends'][0]<0 or model['volatilities'][0]<=0:
        raise ValueError('Unsupported parameter domain')
    if any(x!=model['spots'][0] for x in model['spots']) or any(x!=model['dividends'][0] for x in model['dividends']) or any(x!=model['volatilities'][0] for x in model['volatilities']):
        raise ValueError('Homogeneous benchmark dynamics are required')
    if any(x!=float(i==j) for i,row in enumerate(model['correlation']) for j,x in enumerate(row)):
        raise ValueError('The product-CDF oracle requires independent coordinates')
    if max(model['spots'])>model['strike'] or any(type(tau) is not int or not 1<=tau<len(times) for tau in record['stopping_indices']):
        raise ValueError('Zero initial payoff and retained positive exercise dates are required')
    return dict(fit_index=index,root_seed=2026100410+index,retained_paths=1024,
        selected_path_indices=list(range(PATHS_PER_FIT)),
        arm_order=list(ARMS if index%2==0 else reversed(ARMS)),
        frozen_model_policy_sha256=canonical_hash(dict(model=model,policy=record['frozen_policy'])))


def snapshot(args):
    control=read_json(args.control_study/'design.json');declared=control['source_sha256']
    if len(declared)!=15:raise ValueError('Fifteen registered scientific sources are required')
    source={key:digest(args.root/key) for key in declared}
    if source!=declared:raise ValueError('A registered scientific source changed')
    inputs={f'fit_{i:02d}/inputs.json':digest(args.extension_study/f'fit_{i:02d}/inputs.json') for i in range(FITS)}
    expected=control['source_evidence']['frozen_extension_files']
    if any(expected[key]!=value for key,value in inputs.items()):raise ValueError('Retained input identities changed')
    extension_design=digest(args.extension_study/'design.json')
    if extension_design!=control['source_evidence']['frozen_extension_design_sha256']:
        raise ValueError('The retained extension design changed')
    actual_tail=args.allocator_module.with_name('certify_european_gain_tail.py').resolve()
    if args.tail_module.resolve()!=actual_tail:
        raise ValueError('The registered helper must be the allocator actual sibling dependency')
    paths=[Path(__file__).resolve(),args.allocator_module.resolve(),actual_tail]+[x.resolve() for x in args.additional_source]
    if len(set(paths))!=len(paths):raise ValueError('Duplicate additional source path')
    dependencies={str(path):digest(path) for path in paths}
    return dict(registered_source_sha256=source,retained_input_sha256=inputs,
        control_design_sha256=digest(args.control_study/'design.json'),extension_design_sha256=extension_design,
        actual_allocator_sibling_tail_helper=str(actual_tail),new_numerical_sources_sha256=dependencies)


def design_settings():
    return dict(fitting_units=FITS,paths_per_fit=PATHS_PER_FIT,total_retained_paths=FITS*PATHS_PER_FIT,
        path_selection='Stored indices 0 through 31 from each preceding 1024-path evaluation; no new draws.',
        arms=list(ARMS),target_width='1/100',initial_value_width='1',maximum_rounds=MAX_ROUNDS,
        precision_bits=PRECISION,common_width_factor='1/4',
        maximum_oracle_attempts_per_requested_value=MAX_ORACLE_ATTEMPTS,
        common_global_round_cap=MAX_ROUNDS,adaptive_actual_query_cap_per_date=MAX_ROUNDS,
        adaptive_additional_query_cap='16 times the initially queried nonterminal date count',
        request_width_floor='initial value width / 4**16',
        reference=dict(arm='reference_common',target_width='1/100000000',initial_value_width='1/100000000',precision_bits=REFERENCE_PRECISION,
            maximum_oracle_attempts_per_requested_value=MAX_ORACLE_ATTEMPTS,maximum_global_rounds=MAX_ROUNDS,
            scope='Separately recomputed accuracy audit using the same verified oracle; not an independent algorithm, optimal value or new sample.'),
        execution_order='Whole-arm execution per fit: common first for even fit indices, adaptive first for odd indices. Four fits in each order; reference follows both arms.',
        cache_scope='Fresh local oracle cache for every path and arm; identical native ledger detail for both compared arms.',
        shared_structural_rules='Both arms screen nonselected exact-zero rewards with finite structural European-value brackets, and apply the same safe interval level-dominance pruning with deterministic retained witnesses, including ties, before each refinement decision.',
        common_refinement='Quarter one global requested value width; query every surviving eligible value and the selected nonterminal value when unresolved.',
        adaptive_refinement='Rank only among the same surviving eligible values and the selected unresolved nonterminal value.',
        numerical_object='G=max_{j>=1}(H_j-P_j)-(H_tau-P_tau), where P_j=exp(-rT)E[(max_i S_T^i-K)^+|S_j].',
        inference_scope='Selected retained numerical allocation comparison. No fresh financial certificate, confidence-coverage or scalability claim.',
        timing_scope='Incremental complete assessment on retained states, including all API queries, retries, validation, caches and result construction. Source hashing, input reading, accuracy audit and JSON/gzip I/O are separate workflow costs. No new policy-training or full standalone pricing cost is claimed.',
        outcome_scope='No advantage is an admissible outcome. Arms cannot be dropped, tolerances relaxed or failed paths replaced.')


def register(args):
    if args.directory.exists():raise FileExistsError('Use a new immutable study directory')
    start=clock_pair();evidence=snapshot(args);runtime=runtime_identity();cases=[]
    for i in range(FITS):cases.append(model_context(read_json(args.extension_study/f'fit_{i:02d}/inputs.json'),i))
    if snapshot(args)!=evidence:raise ValueError('Sources or inputs changed during registration')
    design=dict(schema_version=1,author='Amanjeet Singh',recorded_at_utc=now(),recorded_before_assessment=True,
        root=str(args.root.resolve()),extension_study=str(args.extension_study.resolve()),control_study=str(args.control_study.resolve()),
        allocator_module=str(args.allocator_module.resolve()),tail_module=str(args.tail_module.resolve()),
        additional_sources=[str(x.resolve()) for x in args.additional_source],evidence=evidence,
        runtime_context=runtime,settings=design_settings(),fits=cases,registration_prewrite_cost=measured(start))
    args.directory.mkdir(parents=True)
    write_json(args.directory/'design.json',design)
    write_json(args.directory/'registration_completion.json',dict(design_sha256=digest(args.directory/'design.json'),evidence=evidence,runtime_context=runtime))
    return design


def registration(args):
    design=read_json(args.directory/'design.json');completion=read_json(args.directory/'registration_completion.json')
    if design['schema_version']!=1 or design['author']!='Amanjeet Singh' or design['recorded_before_assessment'] is not True:
        raise ValueError('Invalid registration context')
    if design['settings']!=design_settings() or completion['design_sha256']!=digest(args.directory/'design.json'):
        raise ValueError('Registered settings or design changed')
    if design['runtime_context']!=runtime_identity() or completion['runtime_context']!=design['runtime_context']:
        raise ValueError('The registered numerical runtime changed')
    for key,path in (('root',args.root),('extension_study',args.extension_study),('control_study',args.control_study),('allocator_module',args.allocator_module),('tail_module',args.tail_module)):
        if design[key]!=str(path.resolve()):raise ValueError('Registered study context changed')
    if design['additional_sources']!=[str(x.resolve()) for x in args.additional_source]:raise ValueError('Additional source inventory changed')
    current=snapshot(args)
    if design['evidence']!=current or completion['evidence']!=current:raise ValueError('Registered source or input hashes changed')
    return design


def load_allocator(path,expected_tail):
    spec=importlib.util.spec_from_file_location('registered_european_allocator',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if not callable(getattr(module,'evaluate_gain',None)):raise ValueError('The reviewed allocation API is unavailable')
    actual_tail=path.with_name('certify_european_gain_tail.py').resolve()
    if expected_tail.resolve()!=actual_tail or Path(module.TAIL_HELPER_PATH).resolve()!=actual_tail or Path(module.tail.__file__).resolve()!=actual_tail:
        raise ValueError('The loaded allocator does not use the registered sibling tail helper')
    return module


def evaluate_arm(module,record,arm,*,target,initial,precision,completed,repository):
    start=clock_pair();model=record['model']
    try:
        for path in range(PATHS_PER_FIT):
            native=module.evaluate_gain(record['states'][path],model['exercise_times'],record['stopping_indices'][path],
                strike=model['strike'],rate=model['rate'],dividend=model['dividends'][0],volatility=model['volatilities'][0],
                arm=arm,target_width=target,initial_value_width=initial,max_rounds=MAX_ROUNDS,
                max_oracle_attempts=MAX_ORACLE_ATTEMPTS,precision=precision,repository=repository)
            # Retain a returned ledger before any subsequent validation fails.
            completed.append(dict(path_index=path,stopping_index=record['stopping_indices'][path],assessment=native))
            validate_native(native,target,arm=arm,initial=initial,precision=precision,
                calendar_length=len(model['exercise_times']),stopping_index=record['stopping_indices'][path],states=record['states'][path])
        summary=arm_summary(completed,dict(cpu_seconds=0.,wall_seconds=0.))
        summary['complete_assessment_cost']=measured(start)
        return summary
    except Exception as error:
        retained=getattr(error,'record',getattr(error,'partial_result',None))
        error.study_failure=dict(arm=arm,path_index=locals().get('path'),exception_type=type(error).__name__,message=str(error),
            complete_failed_arm_cost=measured(start),completed_or_returned_paths=completed,failed_path_partial_record=retained,
            internal_work_scope='Returned and exception-carried native ledgers are preserved. Work unavailable inside an exception remains explicitly unresolved.')
        raise


def consistency_audit(record,arms,reference):
    for path in range(PATHS_PER_FIT):
        intervals=[]
        for row in (arms['coarse_common']['observations'][path],arms['gain_adaptive']['observations'][path],reference['observations'][path]):
            native=row['assessment'];direct=reconstruct_gain(native,len(record['model']['exercise_times']),row['stopping_index'])
            if direct!=decode_pair(native['gain_certificate']):raise ValueError('Direct premium reconstruction differs from the retained gain')
            intervals.append(direct)
        if max(x[0] for x in intervals)>min(x[1] for x in intervals):raise ValueError('Allocation arms and tighter reference do not intersect')
    return dict(paths=PATHS_PER_FIT,all_direct_gain_reconstructions_exact=True,all_three_intervals_intersect=True,
                reference_scope='Same verified expectation oracle at tighter accuracy; no independent optimal-value reference.')


def distribution(values):
    if len(values)!=FITS or any(not math.isfinite(x) for x in values):raise ValueError('Eight finite fitted-unit summaries are required')
    return dict(fitting_units=FITS,minimum=min(values),median=statistics.median(values),maximum=max(values))


def summary(results,design_sha,evidence):
    if [x['fit_index'] for x in results]!=list(range(FITS)):raise ValueError('All eight registered fits must complete')
    ratios={key:[] for key in ('complete_assessment_cpu','conditional_integral_calls','integrand_evaluations_known','oracle_attempts','retry_attempts')}
    compact=[]
    for result in results:
        arms=result['arms'];common=arms['coarse_common'];adaptive=arms['gain_adaptive']
        for key in ratios:
            if key=='complete_assessment_cpu':a=adaptive['complete_assessment_cost']['cpu_seconds'];b=common['complete_assessment_cost']['cpu_seconds']
            elif key=='retry_attempts':a=adaptive[key];b=common[key]
            else:a=adaptive['total_work'][key];b=common['total_work'][key]
            ratios[key].append(None if b==0 else a/b)
        item={k:v for k,v in result.items() if k not in ('arms','reference')}
        item['arms']={name:{k:v for k,v in arm.items() if k!='observations'} for name,arm in arms.items()}
        item['reference']={k:v for k,v in result['reference'].items() if k!='observations'}
        compact.append(item)
    output_ratios={key:distribution(values) if all(x is not None for x in values) else dict(per_fit_ratios=values,median=None,scope='Zero denominator prevents a full fitted-unit ratio summary.') for key,values in ratios.items()}
    return dict(schema_version=1,author='Amanjeet Singh',design_sha256=design_sha,evidence=evidence,completed_fits=FITS,
        retained_paths_per_fit=PATHS_PER_FIT,total_retained_paths=FITS*PATHS_PER_FIT,cases=compact,
        adaptive_over_common_ratios=output_ratios,
        work_scope='API invocation counts include failed attempts. Known integrand evaluations are partial whenever unavailable-attempt counts are nonzero; they are not total work in that case.',
        scope=design_settings()['inference_scope'])


def assess(args):
    if (args.directory/'assessment_attempt.json').exists():
        raise FileExistsError('An assessment attempt already exists. Do not silently restart; review retained evidence and use a controlled new attempt.')
    overall=clock_pair();design=registration(args);design_sha=digest(args.directory/'design.json');evidence=design['evidence'];phases={}
    phases['initial_source_and_registration_checks']=measured(overall)
    write_json(args.directory/'assessment_attempt.json',dict(started_at_utc=now(),design_sha256=design_sha,evidence=evidence,complete_registered_fits=list(range(FITS))))
    results=[];failure=None
    try:
        module_start=clock_pair();module=load_allocator(args.allocator_module,args.tail_module);phases['allocator_import']=measured(module_start)
        for index in range(FITS):
            if __import__('shutil').disk_usage(args.directory).free<128*(1<<20):raise OSError('Insufficient storage for an immutable fit ledger; preserve completed work.')
            fit=args.directory/f'fit_{index:02d}';fit.mkdir()
            write_json(fit/'assessment_attempt.json',dict(fit_index=index,design_sha256=design_sha,evidence=evidence))
            fit_start=clock_pair();read_start=clock_pair();record=read_json(args.extension_study/f'fit_{index:02d}/inputs.json')
            if model_context(record,index)!=design['fits'][index]:raise ValueError('Retained fit context changed')
            fit_phases=dict(input_reading_and_context_validation=measured(read_start));arms={};partial={};reference=None
            for arm in design['fits'][index]['arm_order']:
                partial[arm]=[]
                arms[arm]=evaluate_arm(module,record,arm,target=TARGET,initial=INITIAL_WIDTH,precision=PRECISION,completed=partial[arm],repository=args.root)
            partial['reference_common']=[]
            reference=evaluate_arm(module,record,'reference_common',target=REFERENCE_TARGET,initial=REFERENCE_TARGET,precision=REFERENCE_PRECISION,completed=partial['reference_common'],repository=args.root)
            check_start=clock_pair();checked=consistency_audit(record,arms,reference);fit_phases['direct_gain_and_overlap_audit']=measured(check_start)
            result=dict(schema_version=1,author='Amanjeet Singh',fit_index=index,root_seed=2026100410+index,arm_order=design['fits'][index]['arm_order'],
                design_sha256=design_sha,retained_input_sha256=evidence['retained_input_sha256'][f'fit_{index:02d}/inputs.json'],
                frozen_model_policy_sha256=design['fits'][index]['frozen_model_policy_sha256'],evidence=evidence,
                selected_path_indices=list(range(PATHS_PER_FIT)),arms=arms,reference=reference,consistency_audit=checked,separate_workflow_phases=fit_phases)
            io_start=clock_pair();artifact=write_gzip(fit/'result.json.gz',result);io_cost=measured(io_start)
            completion=dict(fit_index=index,design_sha256=design_sha,evidence=evidence,artifact=artifact,
                compressed_result_serialisation_and_io_cost=io_cost,complete_fit_precompletion_cost=measured(fit_start))
            write_json(fit/'assessment_completion.json',completion)
            # Keep only compact summaries after each complete fit.
            compact=dict(result);compact['arms']={k:{key:value for key,value in arm.items() if key!='observations'} for k,arm in arms.items()};compact['reference']={key:value for key,value in reference.items() if key!='observations'}
            compact['artifact']=artifact;compact['compressed_result_serialisation_and_io_cost']=io_cost;results.append(compact)
            del record,result,arms,partial,reference
        final_start=clock_pair()
        if registration(args)['evidence']!=evidence:raise ValueError('Scientific sources or retained inputs changed during assessment')
        phases['final_source_and_registration_checks']=measured(final_start)
        output=summary(results,design_sha,evidence)
        write_json(args.directory/'summary.json',output)
        write_json(args.directory/'assessment_completion.json',dict(completed_at_utc=now(),design_sha256=design_sha,evidence=evidence,
            summary_sha256=digest(args.directory/'summary.json'),completed_fit_result_compressed_sha256={f'fit_{i:02d}':row['artifact']['compressed_sha256'] for i,row in enumerate(results)},
            complete_assessment_precompletion_workflow_cost=measured(overall),separate_workflow_phases=phases,
            timing_scope='All completed assessment calls, reference audits, imports, source hashing, input reading and JSON/gzip I/O since assess() began. Completion-marker serialisation is excluded. Timed arm assessments are incremental on retained states; no new training cost is claimed.'))
        return output
    except Exception as error:
        failed=getattr(error,'study_failure',None)
        failure=dict(failed_at_utc=now(),fit_index=locals().get('index'),exception_type=type(error).__name__,message=str(error),design_sha256=design_sha,evidence=evidence,
            completed_fit_summaries=results,completed_current_arms=locals().get('arms',{}),partial_current_paths=locals().get('partial',{}),failed_arm=failed,
            complete_assessment_failed_workflow_cost=measured(overall),no_restart_or_seed_replacement=True)
        write_json(args.directory/'assessment_failure.json',failure)
        raise


def main():
    parser=ArgumentParser()
    parser.add_argument('mode',choices=('register','assess'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--extension-study',type=Path,required=True)
    parser.add_argument('--control-study',type=Path,required=True)
    parser.add_argument('--allocator-module',type=Path,required=True)
    parser.add_argument('--tail-module',type=Path,required=True)
    parser.add_argument('--additional-source',type=Path,action='append',default=[])
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    output=register(args) if args.mode=='register' else assess(args)
    print(json.dumps(dict(mode=args.mode,directory=str(args.directory),completed_fits=output.get('completed_fits'),adaptive_over_common_ratios=output.get('adaptive_over_common_ratios'))))


if __name__=='__main__':main()
