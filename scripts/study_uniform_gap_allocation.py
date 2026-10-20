"""Compare common-width and selective refinement on retained five-asset paths.

Both common-width assessors test the actual stopped-gap width after complete
rounds. Each round uses one tolerance for every conditional bracket exceeding
it, with no exercise-date ranking or candidate-span pruning. The cold-common
arm starts at 0.01; the coarse-common arm starts at one. Predictable initial
centres remain fixed. Existing adaptive assessment is rerun without alteration.
This comparison changes numerical allocation, not policies, value functions,
paths or the ideal martingale. It introduces no new sampling or priority claim.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from itertools import permutations
import json
import math
from pathlib import Path
import statistics
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import study_frozen_gap_extension as extension
from certify_multivariate_gap import (ball, check_inputs, check_policy, conditional_value,
    ctx, endpoints, exact, interval_record, realised_value, signed_gap)
from refine_multivariate_gap import drift_intervals, frozen_hash, intersect, work_sum

ARMS = ('cold_common','coarse_common','adaptive')
TARGET = Fraction(1,100)
INITIAL_WIDTHS = dict(cold_common=TARGET,coarse_common=Fraction(1))
MAXIMUM_ROUNDS = 16
ADDITIONAL_SOURCES = ('scripts/study_uniform_gap_allocation.py','tests/test_uniform_gap_allocation.py')
EXTENSION_FILES = ('design.json','prepared.json','preparation_completion.json','summary.json',
                   'assessment_completion.json','stream_verification.json','stream_verification_completion.json')
EXTENSION_FIT_FILES = ('inputs.json','polynomial_view.json','assessment_attempt.json','result.json','assessment_completion.json')
MINIMUM_FREE_BYTES = 256*1024**2
TIMER_FIELDS = ('assessment_cpu_seconds','assessment_wall_seconds','input_and_policy_validation_cpu_seconds',
    'kernel_recorded_assessment_cpu_seconds','kernel_recorded_assessment_wall_seconds','certificate_phase_cpu_seconds',
    'cpu_seconds_including_shared_coarse','refinement_cpu_seconds','coarse_cpu_seconds','reward_cpu_seconds')


class UniformRefinementError(ArithmeticError):
    """Retain completed rounds, queries and known work when refinement fails."""
    def __init__(self,message,details):
        super().__init__(message)
        self.details=details


class UniformAssessmentError(ArithmeticError):
    def __init__(self,message,details):
        super().__init__(message)
        self.details=details


clocks,elapsed,digest,write_new = extension.clocks,extension.elapsed,extension.digest,extension.write_new


def check_storage():
    if shutil.disk_usage(ROOT).free<MINIMUM_FREE_BYTES:
        raise OSError('At least 256 MiB of free local space is required before a new allocation attempt; retain all existing evidence')


def source_hashes():
    sources=extension.source_hashes()
    sources.update({name:digest((ROOT/name).read_bytes()) for name in ADDITIONAL_SOURCES})
    return sources


def without_timers(value):
    """Remove only enumerated timing fields for exact deterministic replay."""
    if isinstance(value,dict):
        return {key:without_timers(row) for key,row in value.items() if key not in TIMER_FIELDS}
    if isinstance(value,list):
        return [without_timers(row) for row in value]
    return value


def returned_enclosure_record(lower,upper):
    if lower<=upper:
        return interval_record(lower,upper)
    return dict(invalid_endpoint_order=True,returned_lower=str(lower),returned_upper=str(upper))


def uniform_refine(rewards,penalty,centres,brackets,stopping,oracle,*,initial_width,
                   target=TARGET,maximum_rounds=MAXIMUM_ROUNDS):
    """Uniform global tolerances, with an injectable verified conditional oracle."""
    target,theta=exact(target),exact(initial_width)
    if target<=0 or theta<=0 or type(maximum_rounds) is not int or maximum_rounds<0:
        raise ValueError('Positive widths and a non-negative integer round limit are required')
    penalty,centres,current=tuple(penalty),tuple(centres),tuple(brackets)
    if len(current)!=len(penalty) or len(current)!=len(centres) or len(current)<2:
        raise ValueError('A complete matching finite calendar is required')
    if any(lo>hi or hi-lo>theta for lo,hi in current):
        raise ValueError('All initial brackets must comply with the common initial width')
    fixed=frozen_hash(centres,penalty)
    history=[]
    for round_index in range(maximum_rounds+1):
        certificate=signed_gap(rewards,penalty,drift_intervals(current,centres),stopping)
        if certificate[1]-certificate[0]<=target:
            return current,certificate,history
        if round_index==maximum_rounds or not any(hi>lo for lo,hi in current[1:]):
            raise UniformRefinementError('The actual gap width remains unresolved under the common schedule',
                dict(rounds=history,current_brackets=[interval_record(lo,hi) for lo,hi in current],
                     current_gap_certificate=interval_record(certificate[0],certificate[1]),
                     frozen_centres_and_penalty_sha256=fixed,internal_work_of_failed_oracle_known=True))
        theta/=4
        eligible=[j for j in range(1,len(current)) if current[j][1]-current[j][0]>theta]
        row=dict(round_index=round_index+1,common_requested_width_certificate=interval_record(theta,theta),
            eligible_increments=eligible,queries=[],completed=False)
        history.append(row)
        for j in eligible:
            start=clocks(); returned_work=None
            try:
                lo,hi,work=oracle(j,theta)
                returned_work=work
                if hi<lo or hi-lo>theta:
                    raise ArithmeticError('The returned conditional enclosure is invalid or exceeds its common requested width')
                tighter=intersect(current[j],(lo,hi))
                if tighter[1]-tighter[0]>theta:
                    raise ArithmeticError('A conditional bracket did not attain the common requested width')
                row['queries'].append(dict(increment=j,requested_width_certificate=interval_record(theta,theta),
                    previous_certificate=interval_record(*current[j]),returned_certificate=interval_record(lo,hi),
                    retained_certificate=interval_record(*tighter),work=work,complete_query_cost=elapsed(start)))
                current=(*current[:j],tighter,*current[j+1:])
            except Exception as error:
                row['failed_increment']=j
                row['failed_query_cost']=elapsed(start)
                row['failed_query_exception_type']=type(error).__name__
                row['failed_query_message']=str(error)
                row['failed_query_returned_work']=returned_work
                row['failed_query_returned_certificate']=(returned_enclosure_record(lo,hi) if returned_work is not None else None)
                raise UniformRefinementError(str(error),dict(rounds=history,
                    current_brackets=[interval_record(lo,hi) for lo,hi in current],
                    frozen_centres_and_penalty_sha256=fixed,internal_work_of_failed_oracle_known=returned_work is not None)) from error
        if any(hi-lo>theta for lo,hi in current):
            raise UniformRefinementError('A completed common round contains a non-compliant bracket',dict(rounds=history))
        if frozen_hash(centres,penalty)!=fixed:
            raise UniformRefinementError('A common round changed its frozen centre or penalty',dict(rounds=history))
        row['completed']=True
        final=signed_gap(rewards,penalty,drift_intervals(current,centres),stopping)
        row['gap_certificate_after_round']=interval_record(final[0],final[1])
        row['all_retained_conditional_widths_comply']=True
    raise AssertionError('Unreachable common-refinement state')


def uniform_assess(record,arm):
    """Own initialisation, policy checks, verified queries and result construction."""
    if ctx is None: raise RuntimeError('The separate python-flint environment is required')
    if arm not in INITIAL_WIDTHS: raise ValueError('A declared common-width arm is required')
    start=clocks(); validation_start=clocks()
    model,paths,stops,values,times=check_inputs(record)
    check_policy(record)
    validation_cost=elapsed(validation_start)
    rows=[]
    for index,(states,stop) in enumerate(zip(paths,stops)):
        path_start=clocks(); initial_queries=[]; phase='reward_construction'
        try:
            with ctx.workprec(96):
                rewards=[endpoints((-ball(model['rate'])*ball(instant)).exp()*
                    ball(max(max(map(exact,state))-100,Fraction(0)))) for state,instant in zip(states,times)]
            def oracle(j,width):
                return conditional_value(states[j-1],values[j],record['knots'],model,times[j]-times[j-1],width,precision=96)
            brackets=[(Fraction(0),Fraction(0))]; centres=[Fraction(0)]; penalty=[Fraction(0)]
            phase='initial_conditional_brackets'
            for j in range(1,len(times)):
                query_start=clocks(); returned_work=None
                try:
                    lo,hi,work=oracle(j,INITIAL_WIDTHS[arm])
                    returned_work=work
                except Exception as error:
                    raise UniformRefinementError(str(error),dict(failed_initial_increment=j,
                        failed_initial_requested_width_certificate=interval_record(INITIAL_WIDTHS[arm],INITIAL_WIDTHS[arm]),
                        failed_query_cost=elapsed(query_start),failed_query_returned_work=returned_work,
                        internal_work_of_failed_oracle_known=returned_work is not None)) from error
                if hi<lo or hi-lo>INITIAL_WIDTHS[arm]:
                    raise UniformRefinementError('An initial bracket exceeds its common width',
                        dict(failed_initial_increment=j,failed_query_cost=elapsed(query_start),
                             failed_query_returned_work=work,failed_query_returned_certificate=returned_enclosure_record(lo,hi),
                             internal_work_of_failed_oracle_known=True))
                centre=(lo+hi)/2
                initial_queries.append(dict(increment=j,
                    requested_width_certificate=interval_record(INITIAL_WIDTHS[arm],INITIAL_WIDTHS[arm]),
                    returned_certificate=interval_record(lo,hi),work=work,complete_query_cost=elapsed(query_start)))
                brackets.append((lo,hi)); centres.append(centre)
                penalty.append(penalty[-1]+realised_value(states[j],values[j],record['knots'])-centre)
            brackets,centres,penalty=tuple(brackets),tuple(centres),tuple(penalty)
            phase='common_refinement'
            refined,certificate,history=uniform_refine(rewards,penalty,centres,brackets,stop,oracle,
                initial_width=INITIAL_WIDTHS[arm])
            initial=work_sum([row['work'] for row in initial_queries])
            extra=work_sum([query['work'] for step in history for query in step['queries']])
            total={key:initial[key]+extra[key] for key in initial}
            lo,hi,alo,ahi=certificate
            rows.append(dict(path_index=index,stopping_index=stop,
                signed_gap_certificate=interval_record(lo,hi),absolute_gap_certificate=interval_record(alo,ahi),
                signed_width=float(hi-lo),reward_certificates=[interval_record(*reward) for reward in rewards],
                frozen_centres_and_penalty_sha256=frozen_hash(centres,penalty),
                initial_common_width_certificate=interval_record(INITIAL_WIDTHS[arm],INITIAL_WIDTHS[arm]),
                reward_difference_width_allowance_certificate=interval_record(
                    max(rewards[j][1]-rewards[j][0]+rewards[stop][1]-rewards[stop][0]
                        for j in range(len(times)) if j!=stop),
                    max(rewards[j][1]-rewards[j][0]+rewards[stop][1]-rewards[stop][0]
                        for j in range(len(times)) if j!=stop)),
                maximum_stopping_span_length=max(stop,len(times)-1-stop),
                initial_queries=initial_queries,global_rounds=history,coarse_work=initial,
                refinement_work=extra,total_work=total,complete_path_cost=elapsed(path_start)))
        except Exception as error:
            raise UniformAssessmentError(str(error),dict(arm=arm,failed_path_index=index,failed_phase=phase,
                completed_path_observations=rows,completed_initial_queries_on_failed_path=initial_queries,
                refinement_failure_details=getattr(error,'details',None),
                failed_path_cost=elapsed(path_start),complete_assessment_cost_through_failure=elapsed(start),
                internal_work_of_failed_oracle_known=isinstance(error,UniformRefinementError) and
                    error.details.get('internal_work_of_failed_oracle_known',False))) from error
    intervals=[extension.decoded(row['signed_gap_certificate']) for row in rows]
    aggregate=extension.aggregate_intervals(intervals)
    result=dict(arm=arm,outer_paths=len(rows),observations=rows,all_targets_met=True,
        all_signed_inside_absolute=True,mean_signed_gap_certificate=aggregate['mean_certificate'],
        mean_signed_width=float(extension.decoded(aggregate['mean_certificate'])[1]-extension.decoded(aggregate['mean_certificate'])[0]),
        maximum_signed_width=aggregate['maximum_observation_width'],
        approximate_midpoint_mean=aggregate['approximate_midpoint_mean'],
        approximate_midpoint_mean_standard_error=aggregate['approximate_midpoint_mean_standard_error'],
        midpoint_mean_numerical_error_bound=aggregate['midpoint_mean_numerical_error_bound'],
        conditional_integrals=sum(row['total_work']['conditional_integrals'] for row in rows),
        integrand_evaluations=sum(row['total_work']['integrand_evaluations'] for row in rows),
        initialisation_integrals=sum(row['coarse_work']['conditional_integrals'] for row in rows),
        global_refinement_integrals=sum(row['refinement_work']['conditional_integrals'] for row in rows),
        global_refinement_rounds=sum(len(row['global_rounds']) for row in rows),
        input_and_policy_validation_cpu_seconds=validation_cost['cpu_seconds'],
        initial_common_width= str(INITIAL_WIDTHS[arm]),
        allocation_scope='One common target per round; every retained conditional width above it is queried. No candidate-span, date-ranking or dominance pruning.',
        assessment_timing_scope='Own input and policy checks, own initialisation, every returned oracle retry, complete global rounds and result construction; imports and artifact I/O excluded.')
    cost=elapsed(start)
    result.update(assessment_cpu_seconds=cost['cpu_seconds'],assessment_wall_seconds=cost['wall_seconds'],
        certificate_phase_cpu_seconds=cost['cpu_seconds']-validation_cost['cpu_seconds'])
    return result


def fit_designs():
    orders=list(permutations(ARMS))+[ARMS,('coarse_common','adaptive','cold_common')]
    return [dict(index=i,root_seed=seed,directory=f'fit_{i:02d}',arm_order=list(orders[i]))
            for i,seed in enumerate(extension.original.SEEDS)]


def verify_sources(source,prior_extension):
    source,prior_extension=extension.private_directory(source),extension.private_directory(prior_extension)
    names=(*EXTENSION_FILES,*(f'fit_{i:02d}/{name}' for i in range(8) for name in EXTENSION_FIT_FILES))
    if any(not (prior_extension/name).is_file() for name in names) or list(prior_extension.rglob('*failure.json')):
        raise ValueError('Every frozen extension artifact must already be complete and failure-free')
    design,manifest,_,records,views,old_sources,old_design_sha=extension.registration(source,prior_extension)
    # Complete branches only: all attempts/results/completions were checked
    # before calling this controller. No simulation or assessment can begin.
    extension.assessment(source,prior_extension)
    stream_source=(prior_extension/'stream_verification.json').read_bytes()
    stream=json.loads(stream_source)
    done=json.loads((prior_extension/'stream_verification_completion.json').read_bytes())
    checks=[dict(fit_index=fit['index'],root_seed=fit['root_seed'],input_sha256=row['input_sha256'],
        complete_states_reproduced=True,complete_stopping_indices_reproduced=True,spawn_key=[3])
        for fit,row in zip(design['fits'],manifest['fits'])]
    if (stream.get('design_sha256')!=old_design_sha or stream.get('source_sha256')!=old_sources
        or stream.get('checks')!=checks or stream.get('training_and_value_streams_reconstructed') is not False
        or done.get('design_sha256')!=old_design_sha or done.get('source_sha256')!=old_sources
        or done.get('verification_sha256')!=digest(stream_source)):
        raise ValueError('The complete retained outer-stream verification changed')
    extension.original.validate_cost(done['complete_validation_workflow_cost'])
    prior=dict(original_study=design['source_study'],frozen_extension_directory=str(prior_extension),
        frozen_extension_design_sha256=old_design_sha,frozen_extension_source_sha256=old_sources,
        frozen_extension_files={name:digest((prior_extension/name).read_bytes()) for name in names})
    retained_results=[json.loads((prior_extension/f'fit_{i:02d}/result.json').read_bytes()) for i in range(8)]
    return prior,records,views,retained_results


def design_record(prior,sources):
    return dict(schema_version=1,author='Amanjeet Singh',recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        recorded_before_new_assessment=True,source_evidence=prior,source_sha256=sources,
        fits=fit_designs(),fitting_units=8,outer_paths_per_fit=1024,precision_bits=96,target='1/100',
        initial_common_widths={key:str(value) for key,value in INITIAL_WIDTHS.items()},
        common_tolerance_factor='1/4',maximum_global_rounds=MAXIMUM_ROUNDS,
        new_simulation_or_fitting=False,
        allocation_rule='After a completed round, test the actual signed gap width. If unresolved, quarter one common width and query every conditional bracket exceeding it. Already compliant brackets are retained. No date-ranking, candidate-span or dominance pruning.',
        centres='Each common-width arm fixes its own initial centres. Coarse-common initialisation matches adaptive initialisation exactly. All arms assess the same ideal compensated hinge martingale.',
        arm_order_scope='All six permutations followed by two rotations; every arm occupies every position two or three times. No exact position-balance claim.',
        exact_replay_scope=dict(adaptive_numerical_replay=True,excluded_timing_fields=list(TIMER_FIELDS)),
        implementation_timing_scope='Full kernel costs include each implementation\'s own result construction and retained ledgers. Common-width implementations retain fuller query traces than adaptive; no estimated bookkeeping subtraction or abstract allocation-rule runtime is reported.',
        budget_scope='Fresh assessment expenditure only, since the paths already exist. Reconstructed standalone budgets add the retained 1,024-path input preparation charge once. No fitting or simulation is newly executed.',
        sampling_scope='The same eight retained fits and 8,192 retained paths are reused. This is a numerical allocation and timing comparison, not an independent evaluation sample or statistical coverage experiment.',
        priority_claim=False,public_preregistration_claim=False)


def register(source,prior_extension,directory):
    directory=extension.private_directory(directory)
    if directory.exists(): raise ValueError('Use a fresh allocation-study version')
    check_storage()
    start=clocks()
    prior,_,_,_=verify_sources(source,prior_extension)
    sources=source_hashes(); design=design_record(prior,sources)
    directory.mkdir(parents=True)
    write_new(directory/'design.json',design)
    write_new(directory/'registration_completion.json',dict(author='Amanjeet Singh',
        design_sha256=digest((directory/'design.json').read_bytes()),source_sha256=sources,
        complete_registration_validation_workflow_cost=elapsed(start)))
    print('Recorded the common-width allocation design before numerical assessment.',flush=True)
    return design


def registration(source,prior_extension,directory):
    directory=extension.private_directory(directory)
    prior,records,views,retained=verify_sources(source,prior_extension)
    sources=source_hashes(); source_design=(directory/'design.json').read_bytes()
    design=json.loads(source_design); design_sha=digest(source_design)
    expected=design_record(prior,sources)
    for key,value in expected.items():
        if key!='recorded_at_utc' and design.get(key)!=value:
            raise ValueError('The common-width allocation registration differs: '+key)
    if not isinstance(design.get('recorded_at_utc'),str): raise ValueError('Retain the actual recording time')
    done=json.loads((directory/'registration_completion.json').read_bytes())
    if done.get('design_sha256')!=design_sha or done.get('source_sha256')!=sources:
        raise ValueError('The completed registration changed')
    extension.original.validate_cost(done['complete_registration_validation_workflow_cost'])
    return design,records,views,retained,sources,design_sha


def discounted_rewards(record,path_index):
    """Reconstruct the same 96-bit reward enclosures, without integration."""
    model=record['model']; states=record['states'][path_index]
    output=[]
    for state,instant in zip(states,model['exercise_times']):
        hinge=max(max(map(exact,state))-100,Fraction(0))
        if hinge==0:
            output.append((Fraction(0),Fraction(0)))
        else:
            if ctx is None:
                raise RuntimeError('Non-zero reward reconstruction requires the separate Flint environment')
            with ctx.workprec(96):
                output.append(endpoints((-ball(model['rate'])*ball(instant)).exp()*ball(hinge)))
    return output


def validate_trace(observation,record,arm):
    theta=INITIAL_WIDTHS[arm]
    if extension.decoded(observation['initial_common_width_certificate'])!=(theta,theta):
        raise ValueError('A uniform arm changes its declared initial width')
    path_index=observation['path_index']; stop=observation['stopping_index']
    rewards=discounted_rewards(record,path_index)
    if observation['reward_certificates']!=[interval_record(*reward) for reward in rewards]:
        raise ValueError('The recorded rewards differ from their model reconstruction')
    initial=observation['initial_queries']; rights=len(record['model']['exercise_times'])
    if [query['increment'] for query in initial]!=list(range(1,rights)):
        raise ValueError('Initialisation must contain every conditional date in order')
    brackets=[(Fraction(0),Fraction(0))]; centres=[Fraction(0)]; penalty=[Fraction(0)]
    for query in initial:
        if extension.decoded(query['requested_width_certificate'])!=(theta,theta):
            raise ValueError('Initial dates use unequal requested widths')
        bracket=extension.decoded(query['returned_certificate'],nonnegative=False)
        if bracket[1]-bracket[0]>theta: raise ValueError('An initial bracket exceeds its declared width')
        brackets.append(bracket); centre=sum(bracket)/2; centres.append(centre)
        j=query['increment']
        value=realised_value(record['states'][path_index][j],record['coefficients'][j],record['knots'])
        penalty.append(penalty[-1]+value-centre)
    if observation['frozen_centres_and_penalty_sha256']!=frozen_hash(centres,penalty):
        raise ValueError('An initial centre or realised penalty changed')
    rho=max(rewards[j][1]-rewards[j][0]+rewards[stop][1]-rewards[stop][0]
            for j in range(rights) if j!=stop)
    if (extension.decoded(observation['reward_difference_width_allowance_certificate'])!=(rho,rho)
        or observation['maximum_stopping_span_length']!=max(stop,rights-1-stop)):
        raise ValueError('A sufficient-width diagnostic changed')
    certificate=signed_gap(rewards,penalty,drift_intervals(brackets,centres),stop)
    if len(observation['global_rounds'])>MAXIMUM_ROUNDS:
        raise ValueError('A uniform arm exceeds its registered round limit')
    for index,step in enumerate(observation['global_rounds']):
        if certificate[1]-certificate[0]<=TARGET:
            raise ValueError('A uniform arm performs an extra round after actual gap success')
        theta/=4
        eligible=[j for j in range(1,len(brackets)) if brackets[j][1]-brackets[j][0]>theta]
        if (step['round_index']!=index+1 or step['eligible_increments']!=eligible or step['completed'] is not True
            or extension.decoded(step['common_requested_width_certificate'])!=(theta,theta)
            or [query['increment'] for query in step['queries']]!=eligible
            or step['all_retained_conditional_widths_comply'] is not True):
            raise ValueError('A common round omits an eligible date or changes its schedule')
        for query in step['queries']:
            j=query['increment']
            if (extension.decoded(query['requested_width_certificate'])!=(theta,theta)
                or extension.decoded(query['previous_certificate'],nonnegative=False)!=brackets[j]):
                raise ValueError('A common query changes its requested width or preceding bracket')
            returned=extension.decoded(query['returned_certificate'],nonnegative=False)
            if returned[1]-returned[0]>theta:
                raise ValueError('A returned conditional bracket exceeds its requested width')
            retained=extension.decoded(query['retained_certificate'],nonnegative=False)
            if retained!=intersect(brackets[j],returned): raise ValueError('A retained bracket is not the exact intersection')
            brackets[j]=retained
        if any(hi-lo>theta for lo,hi in brackets): raise ValueError('A completed round is not globally compliant')
        certificate=signed_gap(rewards,penalty,drift_intervals(brackets,centres),stop)
        if step['gap_certificate_after_round']!=interval_record(certificate[0],certificate[1]):
            raise ValueError('A completed round changes its reconstructed signed gap')
    if (certificate[1]-certificate[0]>TARGET
        or observation['signed_gap_certificate']!=interval_record(certificate[0],certificate[1])
        or observation['absolute_gap_certificate']!=interval_record(certificate[2],certificate[3])):
        raise ValueError('The final signed/absolute certificate differs from the reconstructed path algebra')
    all_queries=initial+[query for step in observation['global_rounds'] for query in step['queries']]
    for query in all_queries:
        extension.original.validate_cost(query['complete_query_cost'])
        if any(type(query['work'][field]) is not int or query['work'][field]<0 for field in ('integrals','integrand_evaluations')):
            raise ValueError('A query reports invalid numerical work')
    if (observation['coarse_work']!=work_sum([query['work'] for query in initial])
        or observation['refinement_work']!=work_sum([query['work'] for query in all_queries[len(initial):]])
        or observation['total_work']!=work_sum([query['work'] for query in all_queries])):
        raise ValueError('Uniform work counts omit initialisation or global rounds')
    extension.original.validate_cost(observation['complete_path_cost'])


def context(fit,prior,design_sha,sources):
    prefix=fit['directory']+'/'
    return dict(fit_index=fit['index'],root_seed=fit['root_seed'],arm_order=fit['arm_order'],
        design_sha256=design_sha,source_sha256=sources,
        retained_input_sha256=prior['frozen_extension_files'][prefix+'inputs.json'],
        retained_result_sha256=prior['frozen_extension_files'][prefix+'result.json'])


def validate_result(result,record,retained,identities):
    if any(result.get(key)!=value for key,value in identities.items()): raise ValueError('A fit context changed')
    if (result.get('outer_paths')!=len(record['states']) or list(result['arms'])!=identities['arm_order']
        or result.get('all_path_certificates_intersect') is not True
        or result.get('adaptive_exact_numerical_replay') is not True
        or result.get('coarse_centres_match_adaptive') is not True
        or result.get('retained_preparation_cost')!=record['preparation_cost']):
        raise ValueError('A fit changes its sample, arms or verification outcomes')
    if without_timers(result['arms']['adaptive'])!=without_timers(retained['arms']['adaptive']):
        raise ValueError('Adaptive numerical certificates, work, frozen hashes or histories failed exact replay')
    by_arm={}
    for name,row in result['arms'].items():
        observations=row['observations']
        if len(observations)!=len(record['states']) or row['all_targets_met'] is not True:
            raise ValueError('An arm omits paths or did not meet the target')
        intervals=[]
        for index,observation in enumerate(observations):
            if (type(observation['path_index']) is not int or observation['path_index']!=index
                or type(observation['stopping_index']) is not int or observation['stopping_index']!=record['stopping_indices'][index]):
                raise ValueError('An observation changed path or stopping identity')
            lo,hi=extension.decoded(observation['signed_gap_certificate'])
            alo,ahi=extension.decoded(observation['absolute_gap_certificate'])
            if hi-lo>TARGET or not alo<=lo<=hi<=ahi: raise ValueError('A path certificate is invalid or misses its target')
            if name in INITIAL_WIDTHS: validate_trace(observation,record,name)
            intervals.append((lo,hi))
        aggregate=extension.aggregate_intervals(intervals)
        if row['mean_signed_gap_certificate']!=aggregate['mean_certificate']: raise ValueError('An exact mean changed')
        for field,expected in (('maximum_signed_width',aggregate['maximum_observation_width']),
            ('mean_signed_width',float(extension.decoded(aggregate['mean_certificate'])[1]-extension.decoded(aggregate['mean_certificate'])[0])),
            ('approximate_midpoint_mean',aggregate['approximate_midpoint_mean']),
            ('approximate_midpoint_mean_standard_error',aggregate['approximate_midpoint_mean_standard_error']),
            ('midpoint_mean_numerical_error_bound',aggregate['midpoint_mean_numerical_error_bound'])):
            extension.original.equal(row[field],expected,'A path-derived common-width statistic changed: '+field)
        for field in ('conditional_integrals','integrand_evaluations'):
            if row[field]!=sum(o['total_work'][field] for o in observations): raise ValueError('An arm omits work')
        if name in INITIAL_WIDTHS:
            if row.get('arm')!=name or row.get('initial_common_width')!=str(INITIAL_WIDTHS[name]):
                raise ValueError('A uniform result changes its arm or initial width')
            for field,expected in (('initialisation_integrals',sum(o['coarse_work']['conditional_integrals'] for o in observations)),
                ('global_refinement_integrals',sum(o['refinement_work']['conditional_integrals'] for o in observations)),
                ('global_refinement_rounds',sum(len(o['global_rounds']) for o in observations))):
                if row[field]!=expected: raise ValueError('A uniform phase work count changed')
        extension.original.validate_cost(dict(cpu_seconds=row['assessment_cpu_seconds'],wall_seconds=row['assessment_wall_seconds']))
        if any(not math.isfinite(row[field]) or row[field]<0 for field in ('input_and_policy_validation_cpu_seconds','certificate_phase_cpu_seconds')):
            raise ValueError('An assessment phase timer is invalid')
        extension.original.equal(row['certificate_phase_cpu_seconds'],row['assessment_cpu_seconds']-row['input_and_policy_validation_cpu_seconds'],
                                 'Validation was omitted or duplicated in an assessment timer')
        extension.original.equal(result['assessment_only_cpu_seconds_by_arm'][name],row['assessment_cpu_seconds'],
                                 'A fresh assessment budget changed')
        extension.original.equal(result['standalone_pricing_cpu_seconds_by_arm'][name],
            record['preparation_cost']['cpu_seconds']+row['assessment_cpu_seconds'],'A retained preparation charge changed')
        by_arm[name]=intervals
    for index in range(len(record['states'])):
        intervals=[by_arm[name][index] for name in ARMS]+[
            extension.decoded(retained['arms']['cold_uniform']['observations'][index]['signed_gap_certificate'])]
        if max(lo for lo,hi in intervals)>min(hi for lo,hi in intervals):
            raise ValueError('Certificates for the same ideal path gap do not intersect')
        coarse=result['arms']['coarse_common']['observations'][index]['frozen_centres_and_penalty_sha256']
        adaptive=result['arms']['adaptive']['observations'][index]['frozen_centres_and_penalty_sha256']
        if coarse!=adaptive: raise ValueError('Coarse-common and adaptive initial centres do not match')
    return True


def assessment_fit(directory,fit,record,view,retained,prior,design_sha,sources):
    folder=directory/fit['directory']; identities=context(fit,prior,design_sha,sources)
    names=('assessment_attempt','result','assessment_completion','assessment_failure')
    paths={name:folder/(name+'.json') for name in names}
    if paths['result'].exists() or paths['assessment_completion'].exists():
        if not all(paths[name].is_file() for name in names[:-1]) or paths['assessment_failure'].exists():
            raise ValueError('Preserve an incomplete or failed allocation fit')
        result_source=paths['result'].read_bytes()
        attempt=json.loads(paths['assessment_attempt'].read_bytes()); done=json.loads(paths['assessment_completion'].read_bytes())
        if (any(attempt.get(key)!=value or done.get(key)!=value for key,value in identities.items())
            or done.get('result_sha256')!=digest(result_source)):
            raise ValueError('A completed allocation assessment changed')
        extension.original.validate_cost(done['complete_assessment_validation_workflow_cost'])
        result=json.loads(result_source); validate_result(result,record,retained,identities); return result
    if paths['assessment_attempt'].exists() or paths['assessment_failure'].exists():
        raise ValueError('Do not restart an interrupted or failed allocation attempt')
    check_storage()
    if not folder.exists(): folder.mkdir()
    start=clocks(); arms={}; phase='attempt_recording'
    write_new(paths['assessment_attempt'],dict(**identities,started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        for name in fit['arm_order']:
            phase=name
            arms[name]=(extension.arm_result(record,view,'adaptive') if name=='adaptive' else uniform_assess(record,name))
        phase='exact_pairwise_verification'
        result=dict(schema_version=1,author='Amanjeet Singh',**identities,outer_paths=len(record['states']),
            arms=arms,adaptive_exact_numerical_replay=True,all_path_certificates_intersect=True,
            coarse_centres_match_adaptive=True,retained_preparation_cost=record['preparation_cost'],
            assessment_only_cpu_seconds_by_arm={name:row['assessment_cpu_seconds'] for name,row in arms.items()},
            standalone_pricing_cpu_seconds_by_arm={name:record['preparation_cost']['cpu_seconds']+row['assessment_cpu_seconds'] for name,row in arms.items()})
        validate_result(result,record,retained,identities)
        phase='result_writing'; write_new(paths['result'],result)
        write_new(paths['assessment_completion'],dict(**identities,result_sha256=digest(paths['result'].read_bytes()),
            complete_assessment_validation_workflow_cost=elapsed(start),
            timing_scope='All arm executions, exact numerical replay, interval/trace checks and artifact I/O; completion-marker serialisation excluded.'))
    except Exception as error:
        write_new(paths['assessment_failure'],dict(**identities,failed_phase=phase,
            exception_type=type(error).__name__,message=str(error),completed_arm_results=arms,
            failed_arm_details=getattr(error,'details',None),complete_workflow_cost_through_failure=elapsed(start),
            continuation='Preserve all completed work and unresolved paths. No omitted fit, replacement seed or silent restart.'))
        raise
    print(f'Completed common-width allocation fit {fit["index"]}',flush=True)
    return result


def summarise(results,design_sha,sources):
    if len(results)!=8: raise ValueError('All eight retained fits are required')
    cases=deepcopy(results)
    for case in cases:
        for row in case['arms'].values(): row.pop('observations')
    arm_summaries={}
    for name in ARMS:
        rows=[result['arms'][name] for result in results]
        interval=extension.aggregate_intervals([extension.decoded(row['mean_signed_gap_certificate']) for row in rows])
        arm_summaries[name]=dict(mean_gap_certificate=interval['mean_certificate'],
            arithmetic_mean_gap=interval['approximate_midpoint_mean'],
            conditional_integrals=sum(row['conditional_integrals'] for row in rows),
            integrand_evaluations=sum(row['integrand_evaluations'] for row in rows),
            maximum_signed_width=max(row['maximum_signed_width'] for row in rows),
            assessment_cpu_seconds=dict(minimum=min(row['assessment_cpu_seconds'] for row in rows),
                arithmetic_mean=statistics.mean(row['assessment_cpu_seconds'] for row in rows),
                median=statistics.median(row['assessment_cpu_seconds'] for row in rows),
                maximum=max(row['assessment_cpu_seconds'] for row in rows)))
    ratios={}
    for comparator in ('cold_common','coarse_common'):
        ratios[comparator]={label:extension.ratio_summary([
            (result['arms']['adaptive']['assessment_cpu_seconds']/result['arms'][comparator]['assessment_cpu_seconds'])
            if field is None else result[field]['adaptive']/result[field][comparator] for result in results])
            for label,field in (('assessment_only_cpu',None),('standalone_pricing_cpu','standalone_pricing_cpu_seconds_by_arm'))}
        ratios[comparator]['conditional_integrals']=extension.ratio_summary([
            result['arms']['adaptive']['conditional_integrals']/result['arms'][comparator]['conditional_integrals'] for result in results])
    return dict(schema_version=1,author='Amanjeet Singh',design_sha256=design_sha,source_sha256=sources,
        completed_fits=8,retained_fits=8,outer_paths_per_fit=1024,total_retained_paths=8192,cases=cases,
        arm_summaries=arm_summaries,adaptive_over_common_ratios=ratios,
        adaptive_exact_numerical_replay=True,all_path_certificates_intersect=True,
        interpretation='A matched numerical allocation comparison on previously retained states. No new independent sample is generated. Each arm meets the same required actual gap-width target; achieved widths differ. CPU summaries describe complete implementations including their differing result-ledger construction, on this machine and recorded execution order. No bookkeeping cost is subtracted and no abstract allocation-rule speed is identified.')


def assessment(source,prior_extension,directory):
    directory=extension.private_directory(directory); start=clocks()
    design,records,views,retained,sources,design_sha=registration(source,prior_extension,directory)
    results=[assessment_fit(directory,fit,record,view,old,design['source_evidence'],design_sha,sources)
        for fit,record,view,old in zip(design['fits'],records,views,retained)]
    summary=summarise(results,design_sha,sources)
    path,done_path=directory/'summary.json',directory/'assessment_completion.json'
    if path.exists() or done_path.exists():
        if not path.is_file() or not done_path.is_file(): raise ValueError('Preserve an incomplete allocation summary')
        saved=path.read_bytes(); done=json.loads(done_path.read_bytes())
        if (json.loads(saved)!=summary or done.get('summary_sha256')!=digest(saved)
            or done.get('design_sha256')!=design_sha or done.get('source_sha256')!=sources or done.get('completed_fits')!=8):
            raise ValueError('The allocation summary changed')
        extension.original.validate_cost(done['complete_assessment_validation_workflow_cost'])
        return summary
    write_new(path,summary)
    write_new(done_path,dict(author='Amanjeet Singh',design_sha256=design_sha,source_sha256=sources,
        completed_fits=8,summary_sha256=digest(path.read_bytes()),complete_assessment_validation_workflow_cost=elapsed(start)))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','assess'))
    parser.add_argument('--source-study',required=True,type=Path)
    parser.add_argument('--extension-study',required=True,type=Path)
    parser.add_argument('--directory',required=True,type=Path)
    args=parser.parse_args()
    {'register':register,'assess':assessment}[args.mode](args.source_study,args.extension_study,args.directory)


if __name__=='__main__': main()
