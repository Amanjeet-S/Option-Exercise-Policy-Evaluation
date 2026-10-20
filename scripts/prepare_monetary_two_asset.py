"""Preregister, freeze training rules and prepare independent private samples.

I retain binary64 normal coordinates, stocks and diagnostic stopping dates in
bounded batches. These files are numerical inputs, not monetary confidence
claims. Exact-model assessment must propagate the normal-coordinate law and
resolve exercise-boundary cells rather than adopting diagnostic decisions.
"""
from argparse import ArgumentParser
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY/'src'))
from qf_research.optimal_stopping import monetary_two_asset as policy_module

DEFAULT_PHASE = REPOSITORY/'data/private/revision/2026-10-08/monetary_two_asset_v1'
ROOT_SEED = 2026100801
BATCH_PATHS = 256
TRAINING_PATHS = 32768
SAMPLE_PATHS = 16384
LOOKS = [4096, 16384]
ROLES = ('training', 'evaluation', 'control')
SOURCES = ('scripts/prepare_monetary_two_asset.py',
           'src/qf_research/optimal_stopping/monetary_two_asset.py',
           'src/qf_research/optimal_stopping/multivariate_benchmark.py',
           'src/qf_research/optimal_stopping/contracts.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write('\n')


def state_record(generator):
    return json.loads(json.dumps(generator.bit_generator.state, sort_keys=True))


def construct_streams(model_id, role):
    import numpy as np
    if model_id not in policy_module.MODEL_IDS or role not in ROLES:
        raise ValueError('The declared model and independent role are required')
    streams, records = {}, {}
    for date in range(1, 10):
        key = (policy_module.MODEL_IDS.index(model_id), ROLES.index(role), date)
        sequence = np.random.SeedSequence(ROOT_SEED, spawn_key=key, pool_size=4)
        generator = np.random.Generator(np.random.PCG64DXSM(sequence))
        state = state_record(generator)
        streams[date] = generator
        records[str(date)] = dict(entropy=ROOT_SEED, spawn_key=list(key), pool_size=4,
            generator='PCG64DXSM', initial_state=state,
            initial_state_sha256=policy_module.digest(state), sampled=False)
    return streams, records


def runtime_record():
    import numpy as np
    import scipy
    if np.__version__ != '2.3.5':
        raise RuntimeError('The registered harness requires NumPy 2.3.5')
    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    return dict(python_executable=sys.executable, python_version=sys.version,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        platform=platform.platform(), byteorder=sys.byteorder,
        numpy_configuration=configuration.getvalue(),
        thread_environment={key:os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')})


def protocol():
    return dict(schema_version=1, author='Amanjeet Singh', root_seed=ROOT_SEED,
        models={name:policy_module.exact_model(name) for name in policy_module.MODEL_IDS},
        fit_parameters=policy_module.FIT_PARAMETERS, fits_per_model=1,
        training_paths=TRAINING_PATHS, evaluation_paths=SAMPLE_PATHS,
        control_paths=SAMPLE_PATHS, control_role='Reserved; sampled only by explicit control preparation',
        batch_paths=BATCH_PATHS, looks=LOOKS, generator='PCG64DXSM',
        spawn_keys='(model_index,role_index,date); declared ordered models and roles; date 1,...,9',
        model_order=list(policy_module.MODEL_IDS), role_order=list(ROLES),
        generation=dict(normal_dtype='<f8', normal_shape_per_batch=[BATCH_PATHS,9,2],
            state_dtype='<f8', state_shape_per_batch=[BATCH_PATHS,10,2],
            assessment_uniform_prefix_dtype='<u8', uniform_prefix_bits=128,
            assessment_uniform_prefix_shape_per_batch=[BATCH_PATHS,9,2,2],
            assessment_raw_word_order='Last axis [low_word,high_word]; k=low_word+(high_word<<64); one adjacent pair per path/date/asset',
            training_normal_sampler='NumPy standard_normal; training coefficients are conditioned on after freezing',
            assessment_normal_sampler='Two random_raw uint64 words form one uint128 prefix per asset/date; midpoint inverse CDF is diagnostic only',
            assessment_normal_cells='Z in inverse-Phi([k/2^128,(k+1)/2^128]); assessor must enclose the entire uniform cell and retain unknown clipped [0,B] for k=0 or k=2^128-1 without replacement',
            dt_hex=float(Fraction(1,3)).hex(),
            expression='Each asset uses its own unsorted normal. S_j=S_(j-1)*exp((r-q-sigma^2/2)*float(1/3)+sigma*sqrt(float(1/3))*Z_j)',
            no_artificial_stock_clipping=True,
            law_scope='Training standard_normal and diagnostic binary64 GBM are approximate. Assessment uniform prefix cells permit outward coupling under the ideal iid bit-source premise; the computer PRNG law is not certified here'),
        coefficient_frozen_before_evaluation=True, all_model_fits_before_any_evaluation=True,
        assessor_martingale_fitted=False, bootstrap_variability_assessed=False,
        market_data_used=False, public_or_paper_files_updated=False,
        storage_bytes_per_sample=dict(assessment_uniform_prefixes=288, normals=144, states=160, diagnostic_stopping=1),
        training_path_storage_bytes_per_model=TRAINING_PATHS*(144+160),
        assessment_storage_bytes_per_model_per_role=SAMPLE_PATHS*(288+144+160+1),
        generation_rule='Preserve a failed attempt; never discard, replace or restart a failed sample silently')


def register(folder):
    folder = Path(folder)
    if folder.exists():
        if (not folder.is_dir() or folder.is_symlink()
                or any(item.name!='tools' or not item.is_dir() or item.is_symlink() for item in folder.iterdir())
                or any(item.is_symlink() or (item.is_file() and item.suffix not in ('.py','.md'))
                       for item in folder.rglob('*'))):
            raise ValueError('Registration accepts only a fresh folder or existing tools-only source files')
    else:
        folder.mkdir(parents=True)
    records = {model:{role:construct_streams(model,role)[1] for role in ROLES}
               for model in policy_module.MODEL_IDS}
    ids = [row['initial_state_sha256'] for roles in records.values()
           for dates in roles.values() for row in dates.values()]
    if len(set(ids)) != 54:
        raise ValueError('The preregistered independent stream identities repeat')
    design = dict(protocol(), recorded_at_utc=now(), streams=records, runtime=runtime_record(),
        source_sha256={name:hash_file(REPOSITORY/name) for name in SOURCES})
    write_new(folder/'design.json',design)
    return dict(registered=True, design_sha256=hash_file(folder/'design.json'), folder=str(folder), draws=0)


def checked_design(folder):
    folder = Path(folder)
    with (folder/'design.json').open() as source:
        design = json.load(source)
    expected = protocol()
    if any(design.get(key) != value for key,value in expected.items()):
        raise ValueError('The sealed production protocol changed')
    if design['source_sha256'] != {name:hash_file(REPOSITORY/name) for name in SOURCES}:
        raise ValueError('A registered numerical source changed')
    if design['runtime'] != runtime_record():
        raise ValueError('The registered numerical runtime or thread configuration changed')
    for model in policy_module.MODEL_IDS:
        for role in ROLES:
            if design['streams'][model][role] != construct_streams(model,role)[1]:
                raise ValueError('A preregistered stream changed')
    return design


def checked_assessment_design(folder, assessment_design=None):
    """Verify the separate complete root-owned monetary protocol before draws."""
    folder=Path(folder)
    path=folder/'assessment_design.json' if assessment_design is None else Path(assessment_design)
    if not path.is_file():
        raise ValueError('Seal the complete monetary assessment design before evaluation or control draws')
    record=json.loads(path.read_text())
    if (record.get('schema_version')!=1 or record.get('author')!='Amanjeet Singh'
            or record.get('complete') is not True or not isinstance(record.get('protocol'),dict)
            or not record['protocol'] or record.get('preparation_design_sha256')!=hash_file(folder/'design.json')):
        raise ValueError('The complete assessment protocol or preparation binding changed')
    policies={model:hash_file(folder/model/'training'/'freeze.json') for model in policy_module.MODEL_IDS}
    if record.get('frozen_policy_sha256')!=policies:
        raise ValueError('The separate assessment design does not seal both frozen policies')
    sources=record.get('source_sha256')
    if not isinstance(sources,dict) or not sources:
        raise ValueError('The complete monetary assessment needs its numerical source seal')
    for name,identity in sources.items():
        relative=Path(name)
        source=(REPOSITORY/relative).resolve()
        if (relative.is_absolute() or '..' in relative.parts or not source.is_relative_to(REPOSITORY.resolve())
                or not source.is_file() or not isinstance(identity,str) or len(identity)!=64
                or hash_file(source)!=identity):
            raise ValueError('A sealed monetary assessment numerical source changed')
    return dict(path=str(path.resolve()),sha256=hash_file(path))


def midpoint_normals(prefixes):
    """Use a representable lower-tail probability, including extreme prefixes."""
    import numpy as np
    from scipy.special import ndtri
    prefixes=np.asarray(prefixes,dtype=np.uint64)
    if prefixes.ndim < 1 or prefixes.shape[-1]!=2:
        raise ValueError('Each 128-bit prefix requires adjacent low and high words')
    values=np.empty(prefixes.shape[:-1],dtype=np.float64)
    for index in np.ndindex(values.shape):
        word=int(prefixes[index+(0,)])+(int(prefixes[index+(1,)]) << 64)
        lower=word < (1 << 127)
        tail=(word+.5)/(1 << 128) if lower else ((1 << 128)-1-word+.5)/(1 << 128)
        normal=float(ndtri(tail))
        values[index]=normal if lower else -normal
    return values


def generate_batch(model_id, streams, count, *, role='training'):
    import numpy as np
    if type(count) is not int or count < 1 or set(streams) != set(range(1,10)):
        raise ValueError('The complete positive-date stream family is required')
    if role not in ROLES:
        raise ValueError('The complete preregistered sampler role is required')
    model = policy_module.floating_model(model_id)
    before = {str(j):state_record(streams[j]) for j in range(1,10)}
    normals = np.empty((count,9,2),dtype=np.float64)
    states = np.empty((count,10,2),dtype=np.float64)
    prefixes = None if role=='training' else np.empty((count,9,2,2),dtype=np.uint64)
    states[:,0] = (100.,100.)
    volatility = np.asarray(model.volatilities)
    drift = model.rate-np.asarray(model.dividends)-.5*volatility*volatility
    dt = float(Fraction(1,3))
    for date in range(1,10):
        if role=='training':
            normals[:,date-1] = streams[date].standard_normal((count,2))
        else:
            prefixes[:,date-1] = streams[date].bit_generator.random_raw((count,2,2))
            normals[:,date-1] = midpoint_normals(prefixes[:,date-1])
        states[:,date] = states[:,date-1]*np.exp(drift*dt+normals[:,date-1]*(volatility*dt**.5))
    if not np.all(np.isfinite(normals)):
        raise ArithmeticError('Generated normal arithmetic failed; preserve the entire attempt')
    policy_module.validate_paths(states)
    return dict(normals=normals, states=states, uniform_prefixes=prefixes, before=before,
                after={str(j):state_record(streams[j]) for j in range(1,10)})


def prepare_role(folder, model_id, role, *, count=None, batch=None, fixture=False, assessment_design=None):
    """Counts override only bounded fixtures; the command line fixes production."""
    import numpy as np
    folder = Path(folder)
    design = checked_design(folder)
    if model_id not in policy_module.MODEL_IDS or role not in ROLES:
        raise ValueError('The declared model and role are required')
    if role != 'training':
        if any(not (folder/model/'training'/'freeze.json').is_file() for model in policy_module.MODEL_IDS):
            raise ValueError('Freeze every model before drawing any independent assessment sample')
    requested = TRAINING_PATHS if role == 'training' else SAMPLE_PATHS
    count = requested if count is None else count
    batch = BATCH_PATHS if batch is None else batch
    if (type(count) is not int or type(batch) is not int or batch < 1 or count < batch or count%batch):
        raise ValueError('A fixed complete number of bounded batches is required')
    if type(fixture) is not bool or (not fixture and (count!=requested or batch!=BATCH_PATHS)):
        raise ValueError('Production counts are sealed; bounded overrides require explicit fixture mode')
    assessment_identity=None if role=='training' or fixture else checked_assessment_design(folder,assessment_design)
    target = folder/model_id/role
    if target.exists():
        raise ValueError('Preserve prior preparation; a fresh role folder is required')
    target.mkdir(parents=True)
    streams, initial = construct_streams(model_id,role)
    binding = dict(design_sha256=hash_file(folder/'design.json'), model_id=model_id, role=role,
        stream_sha256=policy_module.digest(initial), paths=count, batch_paths=batch, bounded_fixture=fixture)
    policy = None
    if role != 'training':
        freeze_path = folder/model_id/'training'/'freeze.json'
        policy = policy_module.policy_from_record(json.loads(freeze_path.read_text()),
            (freeze_path.parent/'coefficients.bin').read_bytes())
        if not fixture:
            for model in policy_module.MODEL_IDS:
                trained=json.loads((folder/model/'training'/'completion.json').read_text())
                expected_stream=policy_module.digest(construct_streams(model,'training')[1])
                if (not trained.get('completed') or trained['binding'].get('bounded_fixture')
                        or trained['binding']['paths']!=TRAINING_PATHS
                        or trained['binding']['batch_paths']!=BATCH_PATHS
                        or trained['binding']['stream_sha256']!=expected_stream):
                    raise ValueError('Every production model needs its declared complete single training fit')
            if policy.training_paths!=TRAINING_PATHS or policy.training_stream_sha256!=expected_stream_for_model(model_id):
                raise ValueError('The frozen policy training count or stream changed')
        binding['frozen_policy_sha256'] = hash_file(freeze_path)
        binding['assessment_design']=assessment_identity
    write_new(target/'input.json',dict(binding=binding, initial_streams=initial, recorded_at_utc=now(),
        diagnostic_stopping_scope='Only a float screening map, never the ideal-model stopping oracle'))
    cpu,wall = time.process_time(),time.perf_counter()
    phase='initialisation';completed=0;metadata=[]
    normal_hash,state_hash,stop_hash,prefix_hash = hashlib.sha256(),hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
    try:
        with (target/'normals.bin').open('xb') as normal_file, (target/'states.bin').open('xb') as state_file:
            stop_file = (target/'diagnostic_stops.bin').open('xb') if policy is not None else None
            prefix_file = (target/'uniform_prefixes.bin').open('xb') if policy is not None else None
            try:
                for offset in range(0,count,batch):
                    phase='normal_and_stock_generation'
                    block=generate_batch(model_id,streams,batch,role=role)
                    normal_bytes=np.asarray(block['normals'],dtype='<f8',order='C').tobytes()
                    state_bytes=np.asarray(block['states'],dtype='<f8',order='C').tobytes()
                    prefix_bytes=b'' if block['uniform_prefixes'] is None else np.asarray(block['uniform_prefixes'],dtype='<u8',order='C').tobytes()
                    stop_bytes=b''
                    if policy is not None:
                        phase='diagnostic_frozen_policy_application'
                        stop_bytes=policy_module.diagnostic_stopping(policy,block['states']).tobytes()
                    phase='binary_storage'
                    normal_file.write(normal_bytes);state_file.write(state_bytes)
                    if stop_file is not None:stop_file.write(stop_bytes)
                    if prefix_file is not None:prefix_file.write(prefix_bytes)
                    normal_hash.update(normal_bytes);state_hash.update(state_bytes);stop_hash.update(stop_bytes)
                    prefix_hash.update(prefix_bytes)
                    metadata.append(dict(path_start=offset,path_stop=offset+batch,before=block['before'],after=block['after'],
                        normal_sha256=hashlib.sha256(normal_bytes).hexdigest(),state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                        uniform_prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest() if policy else None,
                        diagnostic_stop_sha256=hashlib.sha256(stop_bytes).hexdigest() if policy else None))
                    completed+=batch
            finally:
                if stop_file is not None:stop_file.close()
                if prefix_file is not None:prefix_file.close()
        phase='completed_binary_validation'
        for name,width,value in [('normals.bin',144,normal_hash),('states.bin',160,state_hash)]:
            if (target/name).stat().st_size != count*width or hash_file(target/name) != value.hexdigest():
                raise ValueError('The completed binary bridge differs from its declared bytes')
        if policy is not None and ((target/'diagnostic_stops.bin').stat().st_size != count or hash_file(target/'diagnostic_stops.bin') != stop_hash.hexdigest()):
            raise ValueError('The diagnostic stopping bytes changed')
        if policy is not None and ((target/'uniform_prefixes.bin').stat().st_size != count*288 or hash_file(target/'uniform_prefixes.bin') != prefix_hash.hexdigest()):
            raise ValueError('The exact uniform-prefix bytes changed')
        write_new(target/'blocks.json',metadata)
        if role == 'training':
            phase='single_training_fit'
            states=np.memmap(target/'states.bin',dtype='<f8',mode='r',shape=(count,10,2))
            policy=policy_module.fit_policy(states,model_id,binding['stream_sha256'])
            (target/'coefficients.bin').write_bytes(policy.coefficient_bytes())
            write_new(target/'freeze.json',policy.record())
        summary=dict(binding=binding,completed=True,completed_paths=completed,
            normal_sha256=normal_hash.hexdigest(),state_sha256=state_hash.hexdigest(),
            diagnostic_stopping_sha256=stop_hash.hexdigest() if role!='training' else None,
            uniform_prefix_sha256=prefix_hash.hexdigest() if role!='training' else None,
            final_streams={str(j):state_record(streams[j]) for j in range(1,10)},
            cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
            cost_scope='Preparation and single training fit when applicable; imports, preregistration and completion writing excluded',
            market_data_used=False,ideal_law_certified=False,monetary_inference_executed=False,
            policy_fits_executed=1 if role=='training' else 0)
        write_new(target/'completion.json',summary)
        return summary
    except Exception as error:
        write_new(target/'failure.json',dict(binding=binding,completed=False,completed_paths=completed,
            failed_phase=phase,exception_type=type(error).__name__,message=str(error),
            current_streams={str(j):state_record(streams[j]) for j in range(1,10)},
            preserve_all_bytes=True,no_failed_observation_replacement=True))
        raise


def expected_stream_for_model(model_id):
    return policy_module.digest(construct_streams(model_id,'training')[1])


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','train','prepare'))
    parser.add_argument('--phase',type=Path,default=DEFAULT_PHASE)
    parser.add_argument('--model',choices=policy_module.MODEL_IDS)
    parser.add_argument('--role',choices=('evaluation','control'),default='evaluation')
    parser.add_argument('--assessment-design',type=Path,
                        help='Separate complete source-sealed assessment protocol; default phase/assessment_design.json')
    args=parser.parse_args()
    if args.action=='register':
        if args.model:parser.error('Registration records both models before any draw')
        answer=register(args.phase)
    else:
        if not args.model:parser.error('The declared model is required')
        answer=prepare_role(args.phase,args.model,'training' if args.action=='train' else args.role,
                            assessment_design=args.assessment_design)
    print(json.dumps(answer,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
