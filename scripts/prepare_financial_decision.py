"""Prepare a fresh bounded private sample for retained max-call exercise policies.

I generate fixed-size batches from separate declared PCG64 date streams. The
binary bridge preserves the supplied IEEE-754 states and stopping decisions
for a separate interval-arithmetic assessment process. No policy or value
function is fitted. The generation schedule is new and is not a prefix of the
preceding date-major simulator. Successful numerical assessment traces are
not created by this preparation command.
"""
from argparse import ArgumentParser
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

FIT_COUNT = 8
ROOT_SEEDS = tuple(range(2026100410, 2026100418))
PRODUCTION_PREPARATION = dict(new_parent_spawn_key=[4], date_indices=list(range(1,10)),
    generator='PCG64', batch_paths=256, total_paths_per_fit=16384, exercise_dates=10,
    dimension=5, state_dtype='<f8', stopping_dtype='u1', state_shape=[16384,10,5],
    binary_storage='private')
PRODUCTION_LOOKS = [4096, 16384]
POLICY_IMPLEMENTATION = dict(name='pathwise_sorted_full_state_polynomial_fsum',
    coordinate_order='Each own current state sorted in descending asset order',
    normalisation='state/strike-1 in binary64 arithmetic',
    monomial_order='Immutable monomial_indices total-degree combinations with replacement',
    term_arithmetic='Own-state deterministic math.prod; binary64 coefficient products; math.fsum of those products',
    reward_discount='NumPy scalar exp(-rate*time) in binary64 arithmetic',
    exercise_rule='Strict immediate reward greater than max(continuation,0); ties continue',
    arithmetic_fallback='Continue at a nonterminal date if own-state polynomial arithmetic overflows or is non-finite; exercise every survivor at expiry',
    cross_path_dependence=False,coefficient_refitting=False,
    implementation_scope='Frozen mathematical coefficients with a declared pathwise numerical implementation, not a claim of bitwise identity to preceding alive-batch BLAS decisions')
PHASES = ('stream_initialisation', 'simulation', 'frozen_policy_evaluation',
          'state_and_stopping_serialisation', 'block_validation', 'binary_file_io',
          'block_metadata_io')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def clocks():
    return time.process_time(), time.perf_counter()


def elapsed(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.perf_counter()-start[1])


def valid_cost(cost):
    if set(cost)!={'cpu_seconds','wall_seconds'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in cost.values()):
        raise ValueError('A finite non-negative exclusive CPU and wall cost is required')
    return cost


def zero_cost():
    return dict(cpu_seconds=0.,wall_seconds=0.)


def add_cost(target, value):
    valid_cost(value)
    for key in target:
        target[key]+=value[key]


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as destination:
        json.dump(value,destination,indent=2,sort_keys=True,allow_nan=False)
        destination.write('\n')


def runtime_identity():
    import numpy as np
    import scipy
    configuration=io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    return dict(python_executable=sys.executable,python_version=sys.version,
        numpy_version=np.__version__,scipy_version=scipy.__version__,
        platform=platform.platform(),machine=platform.machine(),byteorder=sys.byteorder,
        numpy_configuration=configuration.getvalue(),
        thread_environment={key:os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')})


def expected_model():
    return dict(spots=[100.]*5,volatilities=[.2]*5,dividends=[.1]*5,
        correlation=[[float(i==j) for j in range(5)] for i in range(5)],
        rate=.05,strike=100.,exercise_times=[3.*i/9 for i in range(10)])


def model_policy_identity(record):
    return canonical_hash(dict(model=record['model'],frozen_policy=record['frozen_policy'],
        policy_degree=record['design']['policy_degree'],training_paths=record['design']['training_paths'],
        root_seed=record['design']['root_seed']))


def validate_retained_input(record,fit):
    import numpy as np
    if type(fit['index']) is not int or not 0<=fit['index']<FIT_COUNT or fit['root_seed']!=ROOT_SEEDS[fit['index']]:
        raise ValueError('The retained fitting seed and index must agree')
    if record['model']!=expected_model():
        raise ValueError('The new decision sample must retain the complete published five-asset model')
    design=record['design']
    if (design['root_seed']!=fit['root_seed'] or design['policy_degree']!=2 or
        design['training_paths']!=50000 or design['value_paths']!=50000 or
        design['spawn_keys'] not in ([[0],[1],[2]],[[0],[1],[3]])):
        raise ValueError('The retained fitting counts, degree or independent stream roles changed')
    if len(record['frozen_policy'])!=10:
        raise ValueError('Every retained exercise date needs a policy rule')
    for row in record['frozen_policy']:
        if not isinstance(row,list) or (row and (len(row)!=21 or not np.all(np.isfinite(row)))):
            raise ValueError('The retained sorted full-state degree-two rule is incomplete')
    if record['frozen_policy'][-1]:
        raise ValueError('The final exercise right must exercise every surviving path')
    return model_policy_identity(record)


def canonical_state(generator):
    # PCG64 state uses ordinary integers and strings. JSON round-trip separates
    # each snapshot from any mutable runtime state returned by the generator.
    return json.loads(json.dumps(generator.bit_generator.state,sort_keys=True,allow_nan=False))


def legacy_stream_identity(generator_state):
    """Match the retained simulator's identity convention, including spacing."""
    return hashlib.sha256(json.dumps(generator_state,sort_keys=True).encode()).hexdigest()


def construct_streams(root_seed,date_indices):
    import numpy as np
    if type(root_seed) is not int or not isinstance(date_indices,list) or date_indices!=list(range(1,len(date_indices)+1)):
        raise ValueError('A declared root and all positive date indices are required')
    streams={}
    records={}
    for date in date_indices:
        sequence=np.random.SeedSequence(root_seed,spawn_key=(4,date),pool_size=4)
        generator=np.random.Generator(np.random.PCG64(sequence))
        state=canonical_state(generator)
        streams[date]=generator
        records[str(date)]=dict(entropy=sequence.entropy,spawn_key=list(sequence.spawn_key),
            pool_size=sequence.pool_size,bit_generator='PCG64',initial_state=state,
            initial_state_sha256=canonical_hash(state),legacy_spacing_state_sha256=legacy_stream_identity(state))
    prior={}
    for role in range(4):
        generator=np.random.Generator(np.random.PCG64(np.random.SeedSequence(root_seed,spawn_key=(role,),pool_size=4)))
        state=canonical_state(generator)
        prior[str(role)]=dict(spawn_key=[role],initial_state=state,
            initial_state_sha256=canonical_hash(state),legacy_spacing_state_sha256=legacy_stream_identity(state),sampled=False)
    new_ids=[row['initial_state_sha256'] for row in records.values()]
    prior_ids=[row['initial_state_sha256'] for row in prior.values()]
    if len(set(new_ids))!=len(new_ids) or set(new_ids)&set(prior_ids):
        raise ValueError('An evaluation date stream repeats a declared generator identity')
    return streams,records,prior


def policy_from_record(record,training_stream_id):
    from qf_research.optimal_stopping.multivariate_benchmark import MaxCallModel,MaxCallPolicy,MaxCallRule
    model=MaxCallModel(**record['model'])
    rules=tuple(MaxCallRule(tuple(coefficients),0,0,None) for coefficients in record['frozen_policy'])
    policy=MaxCallPolicy(model,record['design']['policy_degree'],rules,training_stream_id,record['design']['training_paths'])
    return model,policy


def pathwise_stopping(policy,states,*,return_diagnostics=False):
    """Apply fixed coefficients using only one path's current state at a time.

    Own-state products and fsum replace alive-batch BLAS arithmetic. Numerical
    decisions need not be bitwise identical to the preceding implementation;
    the frozen coefficient array, basis conventions and tie rule are retained.
    No decision uses another path's state or survival status. Non-finite or
    overflowing polynomial arithmetic continues at that date. This supplies
    an explicit measurable continuation fallback, rather than an incomplete
    stopping instruction on the model's unbounded state space. Actual raw
    simulation states remain subject to separate finite-positive checks.
    """
    import numpy as np
    from qf_research.optimal_stopping.multivariate_benchmark import monomial_indices
    model=policy.model
    states=np.asarray(states,dtype=float)
    if (states.ndim!=3 or states.shape[1:]!=(len(model.exercise_times),model.dimension) or
        len(states)<1 or not np.all(np.isfinite(states)) or np.any(states<=0)):
        raise ValueError('Complete finite positive own-state paths are required')
    powers=monomial_indices(model.dimension,policy.degree)
    discounts=tuple(float(np.exp(-model.rate*instant)) for instant in model.exercise_times)
    stopping=np.full(len(states),len(model.exercise_times)-1,dtype=int)
    fallbacks=[];fallback_counts={str(j):0 for j in range(len(model.exercise_times)-1)}
    for path_index,path in enumerate(states):
        for date,rule in enumerate(policy.rules[:-1]):
            if not rule.coefficients:
                continue
            current=tuple(map(float,path[date]))
            try:
                coordinates=tuple(value/model.strike-1. for value in sorted(current,reverse=True))
                features=(1.,)+tuple(math.prod(coordinates[index] for index in term) for term in powers[1:])
                if any(not math.isfinite(feature) for feature in features):
                    raise ArithmeticError('Own-state monomial arithmetic is non-finite')
                continuation=math.fsum(coefficient*feature for coefficient,feature in zip(rule.coefficients,features))
                immediate=discounts[date]*max(max(current)-model.strike,0.)
                if not math.isfinite(continuation) or not math.isfinite(immediate):
                    raise ArithmeticError('Own-state continuation or immediate arithmetic is non-finite')
            except (OverflowError,ValueError,ArithmeticError) as error:
                fallbacks.append(dict(path_index=path_index,date=date,exception_type=type(error).__name__,reason=str(error)))
                fallback_counts[str(date)]+=1
                continue
            if immediate>max(continuation,0.):
                stopping[path_index]=date
                break
    stopping.setflags(write=False)
    diagnostics=dict(arithmetic_fallback_occurrences=len(fallbacks),arithmetic_fallback_counts_by_date=fallback_counts,
        arithmetic_fallbacks=fallbacks,own_state_policy_evaluation=True)
    return (stopping,diagnostics) if return_diagnostics else stopping


def generate_batch(model,policy,streams,count,*,stream_binding):
    """Generate one frozen-shape batch using separate complete date streams."""
    import numpy as np
    from qf_research.optimal_stopping.multivariate_benchmark import (
        MaxCallPaths,correlation_factor)
    if type(count) is not int or count<1 or set(streams)!=set(range(1,len(model.exercise_times))):
        raise ValueError('A positive complete batch and every positive date stream are required')
    before={str(j):canonical_state(streams[j]) for j in sorted(streams)}
    start=clocks()
    factor=correlation_factor(model.correlation)
    volatility=np.asarray(model.volatilities)
    drift=model.rate-np.asarray(model.dividends)-.5*volatility*volatility
    states=np.empty((count,len(model.exercise_times),model.dimension),dtype=np.float64)
    states[:,0]=model.spots
    for date,duration in enumerate(np.diff(model.exercise_times),1):
        noise=streams[date].standard_normal((count,model.dimension)) @ factor.T
        states[:,date]=states[:,date-1]*np.exp(drift*duration+noise*(volatility*math.sqrt(duration)))
    simulation=elapsed(start)
    after={str(j):canonical_state(streams[j]) for j in sorted(streams)}
    start=clocks()
    paths=MaxCallPaths(model,states,stream_binding)
    stopping,policy_diagnostics=pathwise_stopping(policy,paths.states,return_diagnostics=True)
    policy_cost=elapsed(start)
    start=clocks()
    state_bytes=np.asarray(paths.states,dtype='<f8',order='C').tobytes(order='C')
    stop_bytes=np.asarray(stopping,dtype=np.uint8).tobytes(order='C')
    serialisation=elapsed(start)
    return dict(state_bytes=state_bytes,stop_bytes=stop_bytes,before_states=before,after_states=after,
        count=count,policy_diagnostics=policy_diagnostics,phase_costs=dict(simulation=simulation,frozen_policy_evaluation=policy_cost,
            state_and_stopping_serialisation=serialisation))


def validate_batch(block,model,count):
    """Independently read the binary bridge and reject malformed path decisions."""
    import numpy as np
    rights=len(model.exercise_times)
    if (block['count']!=count or len(block['state_bytes'])!=count*rights*model.dimension*8 or
        len(block['stop_bytes'])!=count):
        raise ValueError('The bridge byte counts do not match the declared complete paths')
    states=np.frombuffer(block['state_bytes'],dtype='<f8').reshape(count,rights,model.dimension)
    stops=np.frombuffer(block['stop_bytes'],dtype=np.uint8)
    if (not np.all(np.isfinite(states)) or np.any(states<=0) or
        not np.array_equal(states[:,0],np.broadcast_to(model.spots,states[:,0].shape)) or
        np.any(stops<1) or np.any(stops>=rights)):
        raise ValueError('The bridge requires complete positive paths and positive stopping dates')
    if set(block['before_states'])!=set(map(str,range(1,rights))) or set(block['after_states'])!=set(block['before_states']):
        raise ValueError('The bridge omits a declared date generator state')
    return True


def chain_digest(previous, payload):
    if not isinstance(previous,str) or len(previous)!=64:
        raise ValueError('A complete preceding chain digest is required')
    return hashlib.sha256(bytes.fromhex(previous)+canonical_bytes(payload)).hexdigest()


def verify_chain(rows,genesis):
    previous=genesis
    for index,row in enumerate(rows):
        payload={key:value for key,value in row.items() if key!='block_sha256'}
        if row['block_index']!=index or row['previous_block_sha256']!=previous or row['block_sha256']!=chain_digest(previous,payload):
            raise ValueError('The complete ordered block metadata chain changed')
        previous=row['block_sha256']
    return previous


def prepare_fit(record,fit,folder,preparation,looks,*,design_sha256,sources_sha256,input_sha256,runtime):
    """Retain complete private binary blocks or an explicit partial failure.

    The public entry point fixes the production counts. Smaller complete
    counts are accepted here solely for bounded verification fixtures.
    """
    start=clocks()
    total,batch=preparation['total_paths_per_fit'],preparation['batch_paths']
    if (type(total) is not int or type(batch) is not int or batch<1 or total<batch or total%batch or
        not isinstance(looks,list) or looks!=sorted(set(looks)) or looks[-1]!=total or
        any(type(n) is not int or n<batch or n%batch for n in looks)):
        raise ValueError('Complete declared batches and fixed prefix looks are required')
    if folder.exists():
        raise ValueError('A fresh fit directory is required; preserve every preceding attempt')
    folder.mkdir(parents=True)
    phase='stream_initialisation';batch_index=None;partial_block=None
    phases={key:zero_cost() for key in PHASES}
    completed=[];prefixes=[];streams={};failed_block_start_states=None;states_digest=hashlib.sha256();stops_digest=hashlib.sha256()
    try:
        before=clocks()
        streams,initial,prior=construct_streams(fit['root_seed'],preparation['date_indices'])
        model,policy=policy_from_record(record,prior['0']['legacy_spacing_state_sha256'])
        add_cost(phases['stream_initialisation'],elapsed(before))
        model_policy_sha=model_policy_identity(record)
        binding=dict(design_sha256=design_sha256,source_manifest_sha256=canonical_hash(sources_sha256),
            input_sha256=input_sha256,fit_index=fit['index'],root_seed=fit['root_seed'],
            frozen_model_policy_sha256=model_policy_sha,runtime_sha256=canonical_hash(runtime))
        genesis=canonical_hash(binding)
        previous=genesis
        header=dict(schema_version=1,author='Amanjeet Singh',recorded_at_utc=now(),binding=binding,
            model=asdict(model),policy_degree=policy.degree,training_paths=policy.training_paths,
            training_stream_id=policy.training_stream_id,preparation=preparation,looks=looks,
            policy_implementation=POLICY_IMPLEMENTATION,mathematical_policy_coefficients_preserved=True,
            evaluation_streams=initial,prior_streams_reconstructed_without_sampling=prior,
            runtime=runtime,chain_genesis_sha256=genesis,
            generation_scope='New separate positive-date streams in fixed batches. No earlier date-major stream is replayed; no fitting stream is sampled. Frozen coefficients use the declared own-state implementation rather than preceding alive-batch BLAS arithmetic.',
            sampling_scope='Floating NumPy normal generation and exponential transitions with a declared deterministic own-state stopping map; the ideal iid sampling law is not computer-certified.')
        write_new(folder/'input_manifest.json',header)
        states_path=folder/'states.bin';stops_path=folder/'stops.bin'
        metadata_path=folder/'block_metadata.jsonl'
        with states_path.open('xb') as state_file,stops_path.open('xb') as stop_file,metadata_path.open('x',encoding='utf-8') as metadata_file:
            for batch_index,offset in enumerate(range(0,total,batch)):
                phase='batch_storage_guard'
                if shutil.disk_usage(folder).free < 128*(1<<20):
                    raise OSError('The private bridge requires at least 128 MiB free before a new batch')
                failed_block_start_states={str(date):canonical_state(generator) for date,generator in streams.items()}
                phase='generation_and_frozen_policy_evaluation'
                block_binding=canonical_hash(dict(binding=binding,path_start=offset,paths=batch))
                partial_block=generate_batch(model,policy,streams,batch,stream_binding=block_binding)
                for key,value in partial_block['phase_costs'].items():
                    add_cost(phases[key],value)
                phase='block_validation';before=clocks()
                validate_batch(partial_block,model,batch)
                add_cost(phases['block_validation'],elapsed(before))
                state_bytes,stop_bytes=partial_block['state_bytes'],partial_block['stop_bytes']
                phase='binary_file_io';before=clocks()
                state_file.write(state_bytes);stop_file.write(stop_bytes)
                state_file.flush();stop_file.flush()
                states_digest.update(state_bytes);stops_digest.update(stop_bytes)
                add_cost(phases['binary_file_io'],elapsed(before))
                payload=dict(schema_version=1,block_index=batch_index,fit_index=fit['index'],
                    path_start=offset,path_stop=offset+batch,paths=batch,state_byte_offset=offset*len(model.exercise_times)*model.dimension*8,
                    stopping_byte_offset=offset,state_bytes=len(state_bytes),stopping_bytes=len(stop_bytes),
                    state_sha256=hashlib.sha256(state_bytes).hexdigest(),stopping_sha256=hashlib.sha256(stop_bytes).hexdigest(),
                    cumulative_state_sha256=states_digest.hexdigest(),cumulative_stopping_sha256=stops_digest.hexdigest(),
                    before_generator_states=partial_block['before_states'],after_generator_states=partial_block['after_states'],
                    before_generator_sha256={key:canonical_hash(value) for key,value in partial_block['before_states'].items()},
                    after_generator_sha256={key:canonical_hash(value) for key,value in partial_block['after_states'].items()},
                    requested_normal_shape_per_date=[batch,model.dimension],source_manifest_sha256=binding['source_manifest_sha256'],
                    frozen_model_policy_sha256=model_policy_sha,design_sha256=design_sha256,input_sha256=input_sha256,
                    policy_diagnostics=partial_block['policy_diagnostics'],
                    exclusive_phase_costs=partial_block['phase_costs'],previous_block_sha256=previous)
                payload['block_sha256']=chain_digest(previous,payload)
                phase='block_metadata_io';before=clocks()
                metadata_file.write(canonical_bytes(payload).decode()+'\n');metadata_file.flush()
                add_cost(phases['block_metadata_io'],elapsed(before))
                completed.append(payload);previous=payload['block_sha256'];partial_block=None
                if offset+batch in looks:
                    prefix=dict(paths=offset+batch,completed_blocks=len(completed),last_block_sha256=previous,
                        state_prefix_sha256=states_digest.hexdigest(),stopping_prefix_sha256=stops_digest.hexdigest(),
                        complete_preparation_prefix_cost=elapsed(start),
                        cost_scope='Includes input-manifest writing and every preceding generated block, validation, binary and metadata writing. This look record and final completion serialisation are excluded from their own clocks.')
                    prefixes.append(prefix)
                    write_new(folder/f'prepared_prefix_{offset+batch}.json',prefix)
        phase='completed_bridge_validation';before=clocks()
        if (states_path.stat().st_size!=total*len(model.exercise_times)*model.dimension*8 or stops_path.stat().st_size!=total or
            hash_file(states_path)!=states_digest.hexdigest() or hash_file(stops_path)!=stops_digest.hexdigest()):
            raise ValueError('The complete bridge file bytes differ from their generated digest')
        if verify_chain(completed,genesis)!=previous:
            raise ValueError('The final block chain differs')
        for left,right in zip(completed,completed[1:]):
            if left['after_generator_states']!=right['before_generator_states']:
                raise ValueError('Consecutive date generators have an altered checkpoint')
        add_cost(phases['block_validation'],elapsed(before))
        phase='completion_record_construction'
        final_states={str(date):canonical_state(streams[date]) for date in sorted(streams)}
        complete_cost=elapsed(start)
        remainder={key:complete_cost[key]-sum(row[key] for row in phases.values()) for key in complete_cost}
        if any(value < -1e-9 for value in remainder.values()):
            raise ArithmeticError('Exclusive preparation phase clocks overlap')
        phases['preparation_bookkeeping']={key:max(0.,value) for key,value in remainder.items()}
        answer=dict(schema_version=1,author='Amanjeet Singh',completed=True,fit_index=fit['index'],root_seed=fit['root_seed'],
            binding=binding,paths=total,completed_blocks=len(completed),state_bytes=states_path.stat().st_size,
            stopping_bytes=stops_path.stat().st_size,state_sha256=states_digest.hexdigest(),stopping_sha256=stops_digest.hexdigest(),
            input_manifest_sha256=hash_file(folder/'input_manifest.json'),block_metadata_sha256=hash_file(metadata_path),
            chain_genesis_sha256=genesis,last_block_sha256=previous,initial_generator_states=initial,
            final_generator_states=final_states,final_generator_sha256={key:canonical_hash(value) for key,value in final_states.items()},
            prefix_preparation_records=prefixes,exclusive_phase_costs=phases,complete_preparation_cost=complete_cost,
            policy_implementation=POLICY_IMPLEMENTATION,
            arithmetic_fallback_occurrences=sum(row['policy_diagnostics']['arithmetic_fallback_occurrences'] for row in completed),
            arithmetic_fallback_counts_by_date={str(j):sum(row['policy_diagnostics']['arithmetic_fallback_counts_by_date'][str(j)] for row in completed) for j in range(len(model.exercise_times)-1)},
            cost_scope='Fresh fixed-policy preparation, binary bridge validation and evidence I/O, excluding process startup, imports, retained source reading and this completion-record serialisation. Retained policy training is a separate historical charge.',
            policy_fitting_executed=False,value_fitting_executed=False,earlier_streams_sampled=False,
            market_data_used=False,ideal_simulation_law_certified=False)
        write_new(folder/'completion.json',answer)
        return answer
    except Exception as error:
        failure=dict(schema_version=1,author='Amanjeet Singh',completed=False,fit_index=fit['index'],root_seed=fit['root_seed'],
            design_sha256=design_sha256,input_sha256=input_sha256,failed_phase=phase,failed_block_index=batch_index,
            completed_blocks=len(completed),completed_paths=len(completed)*batch,
            last_completed_block_sha256=None if not completed else completed[-1]['block_sha256'],
            failed_block_start_generator_states=failed_block_start_states,
            current_generator_states={str(date):canonical_state(generator) for date,generator in streams.items()},
            exclusive_known_phase_costs=phases,complete_failure_cost=elapsed(start),
            exception_type=type(error).__name__,message=str(error),
            continuation='Preserve this attempt. No failed path or partial block may be replaced or silently restarted.')
        if partial_block is not None:
            failure['failed_block_before_states']=partial_block['before_states'];failure['failed_block_after_states']=partial_block['after_states']
            (folder/'failed_block_states.bin').write_bytes(partial_block['state_bytes'])
            (folder/'failed_block_stops.bin').write_bytes(partial_block['stop_bytes'])
            failure['failed_block_state_sha256']=hashlib.sha256(partial_block['state_bytes']).hexdigest()
            failure['failed_block_stopping_sha256']=hashlib.sha256(partial_block['stop_bytes']).hexdigest()
        write_new(folder/'preparation_failure.json',failure)
        raise


def validate_design(design):
    if design.get('schema_version')!=1 or design.get('author')!='Amanjeet Singh':
        raise ValueError('The preparation requires the sealed decision-design schema and author')
    if design.get('preparation')!=PRODUCTION_PREPARATION or design.get('looks')!=PRODUCTION_LOOKS:
        raise ValueError('The production batch, maximum count, layout, streams or fixed looks changed')
    fits=design.get('fits')
    if not isinstance(fits,list) or len(fits)!=FIT_COUNT:
        raise ValueError('The complete retained eight-policy family is required')
    for index,fit in enumerate(fits):
        if fit.get('index')!=index or type(fit.get('index')) is not int or fit.get('root_seed')!=ROOT_SEEDS[index]:
            raise ValueError('The frozen ordered fitting roots changed')
    sources=design.get('source_sha256');inputs=design.get('retained_input_sha256')
    if not isinstance(sources,dict) or not sources or not isinstance(inputs,dict) or set(inputs)!={f'fit_{i:02d}/inputs.json' for i in range(FIT_COUNT)}:
        raise ValueError('Every numerical source and retained input needs a prior identity')
    for mapping in (sources,inputs):
        for name,value in mapping.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(value,str) or len(value)!=64:
                raise ValueError('Source and input identities require safe relative names and SHA-256 values')
            bytes.fromhex(value)
    return True


def snapshot(root,retained,design):
    return dict(source_sha256={name:hash_file(root/name) for name in design['source_sha256']},
        retained_input_sha256={name:hash_file(retained/name) for name in design['retained_input_sha256']})


def compare_snapshot(actual,design):
    if actual['source_sha256']!=design['source_sha256'] or actual['retained_input_sha256']!=design['retained_input_sha256']:
        raise ValueError('The preparation numerical sources or retained policy inputs changed')


def prepare_directory(root,retained,directory,design_path):
    root,retained,directory,design_path=map(lambda p:Path(p).resolve(),(root,retained,directory,design_path))
    private=(root/'data/private').resolve()
    if not directory.is_relative_to(private) or not retained.is_relative_to(private) or directory.exists():
        raise ValueError('A fresh private output and verified private retained input are required')
    source_directory=str(root/'src')
    if source_directory not in sys.path:
        sys.path.insert(0,source_directory)
    whole=clocks()
    from qf_research.optimal_stopping import multivariate_benchmark as executed_model,contracts as executed_contracts
    if (Path(executed_model.__file__).resolve()!=root/'src/qf_research/optimal_stopping/multivariate_benchmark.py' or
        Path(executed_contracts.__file__).resolve()!=root/'src/qf_research/optimal_stopping/contracts.py'):
        raise ValueError('The executed model or contracts helper differs from the declared local source')
    design_bytes=design_path.read_bytes();design=json.loads(design_bytes);validate_design(design)
    design_sha=hashlib.sha256(design_bytes).hexdigest()
    # Sources include this installed file before registration. A draft path
    # must never be mistaken for the registered installed source.
    if 'scripts/prepare_financial_decision.py' not in design['source_sha256'] or hash_file(Path(__file__))!=design['source_sha256']['scripts/prepare_financial_decision.py']:
        raise ValueError('The executed preparation source is not the sealed source')
    before=snapshot(root,retained,design);compare_snapshot(before,design)
    source_validation_cost=elapsed(whole)
    runtime=runtime_identity()
    if design.get('preparation_runtime') is not None and design['preparation_runtime']!=runtime:
        raise ValueError('The executed preparation runtime differs from the prior frozen identity')
    records=[]
    for fit in design['fits']:
        with (retained/f'fit_{fit["index"]:02d}/inputs.json').open() as source:
            record=json.load(source)
        identity=validate_retained_input(record,fit)
        if fit.get('frozen_model_policy_sha256') is not None and fit['frozen_model_policy_sha256']!=identity:
            raise ValueError('A retained policy identity differs from the prior design')
        records.append(record)
    directory.mkdir(parents=True)
    write_new(directory/'preparation_attempt.json',dict(schema_version=1,author='Amanjeet Singh',recorded_at_utc=now(),
        design_path=str(design_path),design_sha256=design_sha,retained_source=str(retained),source_snapshot=before,runtime=runtime,
        source_validation_cost=source_validation_cost,planned_fits=FIT_COUNT))
    completed=[];failed_fit=None
    try:
        for fit,record in zip(design['fits'],records):
            failed_fit=fit['index']
            row=prepare_fit(record,fit,directory/f'fit_{fit["index"]:02d}',design['preparation'],design['looks'],
                design_sha256=design_sha,sources_sha256=design['source_sha256'],
                input_sha256=design['retained_input_sha256'][f'fit_{fit["index"]:02d}/inputs.json'],runtime=runtime)
            completed.append(row)
            print(f'Prepared fresh decision input for fit {fit["index"]}',flush=True)
        after=snapshot(root,retained,design);compare_snapshot(after,design)
        if design_path.read_bytes()!=design_bytes:
            raise ValueError('The sealed decision design changed during preparation')
        result=dict(schema_version=1,author='Amanjeet Singh',completed=True,design_sha256=design_sha,
            completed_fits=len(completed),source_snapshot_before=before,source_snapshot_after=after,
            runtime=runtime,fits=completed,complete_preparation_workflow_cost=elapsed(whole),
            timing_scope='Executed source/input validation, preparation, retained-policy reconstruction and all private evidence I/O. Process startup, imports and this final completion-record serialisation are excluded.',
            paths_generated=FIT_COUNT*design['preparation']['total_paths_per_fit'],policy_fitting_executed=False,
            value_fitting_executed=False,earlier_streams_sampled=False,provider_data_used=False)
        write_new(directory/'completion.json',result)
        return result
    except Exception as error:
        write_new(directory/'preparation_failure.json',dict(schema_version=1,author='Amanjeet Singh',completed=False,
            design_sha256=design_sha,failed_fit_index=failed_fit,completed_fits=len(completed),
            complete_preparation_workflow_cost_through_failure=elapsed(whole),exception_type=type(error).__name__,message=str(error),
            continuation='Preserve this fresh version. Do not restart a failed fit or change the registered sample.'))
        raise


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--retained-study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare_directory(args.root,args.retained_study,args.output,args.design)


if __name__=='__main__':
    main()
