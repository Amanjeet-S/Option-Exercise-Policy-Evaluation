"""Enclose and refine one restricted European-value dual gain.

I compare common requested premium widths with a gain-dependent allocation on
the same conditional max-call oracle. The numerical object is the maximum of
H_j-P_j minus the selected level. Its selected candidate and terminal level
are exactly zero. Bounds are sharp for the supplied rectangular level boxes,
not for the joint financial model. Allocation scores are optimistic numerical
heuristics; completion always requires the attained gain width.

All caches and query traces belong to one call. The optional oracle supports
small arithmetic checks; the default uses verified max-hinge integration or
analytic one-asset/deterministic values. No simulation or policy fitting is
performed. All parameters and retained states mean exact binary or rational
values. Sampling inference is a separate operation.
"""
from collections.abc import Mapping, Sequence
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter, process_time

TAIL_HELPER_PATH=Path(__file__).with_name('certify_european_gain_tail.py').resolve()
_spec=importlib.util.spec_from_file_location('european_gain_tail_helpers',TAIL_HELPER_PATH)
tail=importlib.util.module_from_spec(_spec)
_previous_bytecode=sys.dont_write_bytecode
try:
    sys.dont_write_bytecode=True
    _spec.loader.exec_module(tail)
finally:
    sys.dont_write_bytecode=_previous_bytecode
DEFAULT_REPOSITORY=Path(__file__).resolve().parents[1]
WORK_KEYS=('conditional_integral_calls','successful_integral_certificates','analytic_evaluations',
           'integrand_evaluations_known','integrand_evaluations_unavailable_attempts','oracle_attempts')


class OracleCertificationFailure(ArithmeticError):
    def __init__(self,message,record):
        super().__init__(message)
        self.record=record


class GainAssessmentFailure(ArithmeticError):
    def __init__(self,message,record):
        super().__init__(message)
        self.record=record
        self.partial_result=record


def _zero_work():
    return {key:0 for key in WORK_KEYS}


def _add_work(total,row):
    if set(row)!=set(WORK_KEYS) or any(type(value) is not int or value<0 for value in row.values()):
        raise ValueError('Every oracle work counter must be an explicit non-negative integer')
    for key in WORK_KEYS:
        total[key]+=row[key]


def _validate_oracle_payload(payload,max_attempts):
    if not isinstance(payload,Mapping) or not isinstance(payload.get('attempts'),list) or not 1<=len(payload['attempts'])<=max_attempts:
        raise ValueError('An oracle query must retain every attempt within its declared limit')
    summed=_zero_work()
    for index,attempt in enumerate(payload['attempts'],1):
        if attempt.get('attempt_index')!=index or type(attempt.get('attempt_index')) is not int:
            raise ValueError('Oracle attempt indices must be complete and ordered')
        _add_work(summed,attempt['work'])
        if attempt['work']['oracle_attempts']!=1:
            raise ValueError('Each raw oracle attempt must be counted once')
        if type(attempt.get('returned_finite_certificate')) is not bool or type(attempt.get('requested_width_attained')) is not bool:
            raise ValueError('Oracle attempt outcomes must be explicit booleans')
        cost=attempt['complete_attempt_cost']
        if set(cost)!={'cpu_seconds','wall_seconds'} or any(tail.exact(value)<0 for value in cost.values()):
            raise ValueError('Both complete attempt clocks must be finite and non-negative')
    if summed!=payload['work']:
        raise ValueError('Complete query work differs from the sum of all raw attempts')
    cost=payload['complete_query_cost']
    if set(cost)!={'cpu_seconds','wall_seconds'} or any(tail.exact(value)<0 for value in cost.values()):
        raise ValueError('Both complete query clocks must be finite and non-negative')
    if payload['work_complete'] is not (summed['integrand_evaluations_unavailable_attempts']==0):
        raise ValueError('The query work-completeness flag contradicts its missing counts')
    final=payload['attempts'][-1]
    if final['returned_finite_certificate'] is not True or final['requested_width_attained'] is not True:
        raise ValueError('A successful payload contradicts its final oracle outcome')
    if _interval(final['value_certificate'])!=_interval(payload['value_certificate']):
        raise ValueError('A successful payload changes its final attained certificate')
    for clock in ('cpu_seconds','wall_seconds'):
        attempted=sum(tail.exact(row['complete_attempt_cost'][clock]) for row in payload['attempts'])
        complete=tail.exact(payload['complete_query_cost'][clock])
        tolerance=F(1,10**9)+attempted/F(10**8)
        if complete+tolerance<attempted:
            raise ValueError('A complete query clock excludes retained attempt time')


def _interval(record):
    if not isinstance(record,Mapping) or set(record)!={'lower','upper','width'}:
        raise ValueError('An exact interval requires lower, upper and width records')
    lower,upper=tail._decode_rational(record['lower']),tail._decode_rational(record['upper'])
    if upper<lower or tail._decode_rational(record['width'])!=upper-lower:
        raise ValueError('An exact interval has invalid order or width')
    return lower,upper


def _record(pair):
    return tail.interval_record(*pair)


def _width(pair):
    return pair[1]-pair[0]


def _gain(levels,stopping):
    """Cartesian-sharp gain endpoints with the selected level shared once."""
    J=len(levels)-1
    others=[levels[j] for j in range(1,J+1) if j!=stopping]
    lower=max([F(0)]+[pair[0] for pair in others])
    upper=max([F(0)]+[pair[1] for pair in others])
    if stopping==J:
        return lower,upper
    chosen=levels[stopping]
    return max(F(0),lower-chosen[1]),max(F(0),upper-chosen[0])


def _level_boxes(rewards,values):
    J=len(rewards)-1
    result=[None]
    for j in range(1,J):
        result.append((rewards[j][0]-values[j][1],rewards[j][1]-values[j][0]))
    result.append((F(0),F(0)))
    return result


def _dominance(levels,stopping):
    """Retain a deterministic witness; equality never deletes every tied date."""
    J=len(levels)-1
    candidates=[j for j in range(1,J+1) if j!=stopping]
    if not candidates:
        return [],[]
    witness=max(candidates,key=lambda j:(levels[j][0],j==J,-j))
    active=[];excluded=[]
    for j in candidates:
        if levels[j][1]<=levels[stopping][0]:
            excluded.append(dict(date=j,witness_date=stopping,reason='selected_zero_candidate',
                candidate_upper=tail.rational_record(levels[j][1]),witness_lower=tail.rational_record(levels[stopping][0])))
        elif j!=witness and levels[j][1]<=levels[witness][0]:
            excluded.append(dict(date=j,witness_date=witness,reason='nonselected_level_dominance',
                candidate_upper=tail.rational_record(levels[j][1]),witness_lower=tail.rational_record(levels[witness][0])))
        else:
            active.append(j)
    return active,excluded


def _point_score(date,rewards,values,levels,stopping):
    """Largest possible width reduction after one price becomes a point.

    Reward uncertainty remains. Piecewise-linear breakpoints and both premium
    endpoints are enumerated exactly. A realised query need not attain this
    optimistic score.
    """
    lo,hi=values[date]
    points={lo,hi}
    others=[levels[j] for j in range(1,len(levels)) if j not in (date,stopping)]
    other_lo=max([F(0)]+[pair[0] for pair in others])
    other_hi=max([F(0)]+[pair[1] for pair in others])
    Hlo,Hhi=rewards[date]
    if date==stopping:
        points.update((Hhi-other_lo,Hlo-other_hi))
    else:
        selected=levels[stopping] if stopping<len(levels)-1 else (F(0),F(0))
        points.update((Hlo-other_lo,Hhi-other_hi,Hhi-selected[0],Hlo-selected[1]))
    candidates=[]
    for point in points:
        if lo<=point<=hi:
            changed=list(values);changed[date]=(point,point)
            candidate=_gain(_level_boxes(rewards,changed),stopping)
            candidates.append(_width(candidate))
    return max(F(0),_width(_gain(levels,stopping))-min(candidates))


def _default_oracle_factory(repository,*,max_attempts,cutoff_multiplier):
    certifier=None
    def oracle(state,horizon,requested_width,*,terminal_time,strike,rate,dividend,volatility,precision):
        nonlocal certifier
        start_cpu,start_wall=process_time(),perf_counter()
        attempts=[];total=_zero_work();final=None
        numerical=len(state)>1 and volatility>0
        cutoff=max(strike*cutoff_multiplier,max(state)*cutoff_multiplier)
        for attempt_index in range(max_attempts):
            bits=precision+32*attempt_index
            request=requested_width/F(16*4**attempt_index)
            work=_zero_work();work['oracle_attempts']=1
            before_cpu,before_wall=process_time(),perf_counter()
            attempt=dict(attempt_index=attempt_index+1,precision_bits=bits,
                cutoff=tail.rational_record(cutoff),absolute_tolerance=tail.rational_record(request),
                relative_tolerance=tail.rational_record(request/max(strike,max(state))),work=work)
            try:
                with tail.ctx.workprec(bits):
                    discount=tail.endpoints((-tail.ball(rate)*tail.ball(terminal_time)).exp())
                    if numerical:
                        if certifier is None:
                            certifier=tail._load_certifier(repository)
                        work['conditional_integral_calls']=1
                        raw=certifier(state,strike=strike,rate=rate,dividend=dividend,volatility=volatility,
                            step=horizon,cutoff=cutoff,precision=bits,absolute_tolerance=request,
                            relative_tolerance=request/max(strike,max(state)),correlation=0)
                        bounds=_interval(raw['expectation'])
                        work['successful_integral_certificates']=1
                        work['integrand_evaluations_known']=raw['work']['integrand_evaluations']
                        attempt['raw_expectation_certificate']=raw['expectation']
                        attempt['finite_integral']=raw['finite_integral']
                        attempt['tail_allowance']=raw['tail_allowance']
                        attempt['nonfinite_integrand_evaluations']=raw['work']['nonfinite_integrand_evaluations']
                    else:
                        work['analytic_evaluations']=1
                        if len(state)==1:
                            bounds=tail.lognormal_call_moment(state[0],horizon,strike,rate=rate,
                                dividend=dividend,volatility=volatility,precision=bits)
                        else:
                            value=tail.ball(max(state))*(tail.ball(rate-dividend)*tail.ball(horizon)).exp()-tail.ball(strike)
                            bounds=tail._positive_part_endpoints(value)
                        attempt['raw_expectation_certificate']=_record(bounds)
                        attempt['nonfinite_integrand_evaluations']=0
                    final=(max(F(0),discount[0]*bounds[0]),discount[1]*bounds[1])
                    attempt['value_certificate']=_record(final)
                    attempt['requested_width_attained']=_width(final)<=requested_width
                    attempt['returned_finite_certificate']=True
            except Exception as error:
                attempt['returned_finite_certificate']=False
                attempt['requested_width_attained']=False
                attempt['error_type']=type(error).__name__
                attempt['error']=str(error)
                if numerical and work['conditional_integral_calls']:
                    work['integrand_evaluations_unavailable_attempts']=1
                    attempt['integrand_evaluations']=None
            attempt['complete_attempt_cost']=dict(cpu_seconds=process_time()-before_cpu,wall_seconds=perf_counter()-before_wall)
            attempts.append(attempt);_add_work(total,work)
            if attempt['requested_width_attained']:
                return dict(value_certificate=_record(final),attempts=attempts,work=total,
                    complete_query_cost=dict(cpu_seconds=process_time()-start_cpu,wall_seconds=perf_counter()-start_wall),
                    work_complete=total['integrand_evaluations_unavailable_attempts']==0)
            cutoff*=2
        record=dict(attempts=attempts,work=total,
            complete_query_cost=dict(cpu_seconds=process_time()-start_cpu,wall_seconds=perf_counter()-start_wall),
            work_complete=total['integrand_evaluations_unavailable_attempts']==0,
            value_certificate=None if final is None else _record(final))
        raise OracleCertificationFailure('The European value oracle exhausted its declared attempts without the requested width',record)
    return oracle


def evaluate_gain(states,exercise_times,stopping_index,*,strike=100,rate=.05,dividend=.10,volatility=.20,
                  arm='gain_adaptive',target_width=F(1,100),initial_value_width=1,max_rounds=16,
                  precision=96,oracle=None,repository=DEFAULT_REPOSITORY,correlation=0,
                  max_oracle_attempts=8,cutoff_multiplier=2):
    """Return one complete exact-endpoint ledger, or a preserved partial failure.

    ``max_rounds`` permits that many refinements per queried nonterminal date;
    the additional-query budget is max_rounds times the initial query count.
    Both arms stop quartering at initial_value_width/4**max_rounds.
    Common allocation refines each eligible date to a quartered common width.
    Adaptive allocation quarters one date's request, skipping requests already
    met, then uses the same oracle and exact intersection checks.

    Reward/value/F arrays have J+1 entries; value and F at date0 are unused
    null entries. Nonselected zero rewards are screened identically in every
    arm using P>=0 and a finite first-moment upper bound. A selected nonterminal
    zero reward is still evaluated. No first-date stopping policy is accepted.
    """
    cpu_start,wall_start=process_time(),perf_counter()
    if tail.flint is None:
        raise RuntimeError('The separate python-flint environment is required')
    if arm not in ('coarse_common','gain_adaptive','reference_common'):
        raise ValueError('Unknown direct-European allocation arm')
    if type(precision) is not int or precision<32 or type(max_rounds) is not int or max_rounds<1:
        raise ValueError('Positive refinement limits and at least32 precision bits are required')
    if type(max_oracle_attempts) is not int or max_oracle_attempts<1:
        raise ValueError('The oracle attempt limit must be a positive integer')
    strike,rate,dividend,volatility,target_width,initial_value_width,correlation,cutoff_multiplier=(
        tail.exact(value) for value in (strike,rate,dividend,volatility,target_width,initial_value_width,correlation,cutoff_multiplier))
    if strike<=0 or rate<0 or dividend<0 or volatility<0 or target_width<=0 or initial_value_width<=0 or cutoff_multiplier<=1:
        raise ValueError('The supported homogeneous GBM parameters and widths must satisfy their non-negative domain')
    if correlation!=0:
        raise ValueError('Only homogeneous independent GBM is supported')
    if not isinstance(exercise_times,Sequence) or isinstance(exercise_times,(str,bytes)):
        raise ValueError('A complete indexed exercise calendar is required')
    times=tuple(tail.exact(value) for value in exercise_times)
    J=len(times)-1
    if J<1 or times[0]!=0 or any(right<=left for left,right in zip(times,times[1:])):
        raise ValueError('Exercise dates must increase strictly from zero to expiry')
    if type(stopping_index) is not int or not 1<=stopping_index<=J:
        raise ValueError('The selected stopping date must lie in1,...,J')
    if not isinstance(states,Sequence) or isinstance(states,(str,bytes)) or len(states)!=J+1:
        raise ValueError('Every retained date requires a complete state vector')
    vectors=[]
    for state in states:
        if not isinstance(state,Sequence) or isinstance(state,(str,bytes)):
            raise ValueError('Each state must be a non-empty indexed vector')
        vector=tuple(tail.exact(value) for value in state)
        if not vector or any(value<=0 for value in vector):
            raise ValueError('All retained coordinates must be finite and strictly positive')
        vectors.append(vector)
    if any(len(vector)!=len(vectors[0]) for vector in vectors):
        raise ValueError('All dates require the same asset dimension')
    if max(vectors[0])>strike:
        raise ValueError('Removal of the initial exercise right requires an exactly zero initial payoff')
    total=_zero_work();queries=[];rounds=[];excluded=[]
    rewards=[];values=[None]*(J+1)
    with tail.ctx.workprec(precision):
        for time,state in zip(times,vectors):
            payoff=max(max(state)-strike,F(0))
            rewards.append((F(0),F(0)) if payoff==0 else tail.endpoints(tail.ball(payoff)*(-tail.ball(rate)*tail.ball(time)).exp()))
        for date in range(1,J):
            if rewards[date]==(F(0),F(0)) and date!=stopping_index:
                first=tail.ball(sum(vectors[date]))*(tail.ball(rate-dividend)*tail.ball(times[-1]-times[date])).exp()*(-tail.ball(rate)*tail.ball(times[-1])).exp()
                upper=tail.endpoints(first)[1]
                values[date]=(F(0),upper)
                excluded.append(dict(date=date,reason='nonselected_zero_reward_and_nonnegative_European_value',
                    structural_value_certificate=_record(values[date])))
    values[J]=rewards[J]
    dates=[date for date in range(1,J) if values[date] is None]
    counts={date:0 for date in dates};requests={date:initial_value_width for date in dates}
    depths={date:0 for date in dates}
    request_floor=initial_value_width/F(4**max_rounds)
    actual_oracle=oracle or _default_oracle_factory(repository,max_attempts=max_oracle_attempts,cutoff_multiplier=cutoff_multiplier)
    state_digest=hashlib.sha256(json.dumps([[[str(x.numerator),str(x.denominator)] for x in state] for state in vectors],separators=(',',':')).encode()).hexdigest()

    def result(completed,error=None):
        levels=None if any(values[j] is None for j in range(1,J)) else _level_boxes(rewards,values)
        gain=None if levels is None else _gain(levels,stopping_index)
        active,dominance=([],[]) if levels is None else _dominance(levels,stopping_index)
        payload=dict(schema_version=1,arm=arm,stopping_index=stopping_index,rights=J+1,dimension=len(vectors[0]),
            state_sha256=state_digest,target_width_certificate=_record((target_width,target_width)),
            initial_value_width_certificate=_record((initial_value_width,initial_value_width)),
            reward_certificates=[_record(value) for value in rewards],
            value_certificates=[None if value is None else _record(value) for value in values],
            F_certificates=None if levels is None else [None if value is None else _record(value) for value in levels],
            gain_certificate=None if gain is None else _record(gain),completed=completed,error=error,
            initial_queries=[row for row in queries if row['phase']=='initial'],refinement_rounds=rounds,
            excluded_dates=excluded,active_nonselected_dates=active,dominance_certificates=dominance,
            refinement_counts_by_date={str(date):counts[date] for date in dates},
            requested_quarter_depths_by_date={str(date):depths[date] for date in dates},
            initially_queried_nonterminal_dates=dates,request_width_floor_certificate=_record((request_floor,request_floor)),
            maximum_refinements_per_date=max_rounds,additional_query_budget=max_rounds*len(dates),
            maximum_oracle_attempts_per_requested_value=max_oracle_attempts,precision_bits=precision,
            maximum_refinement_decisions=max_rounds if arm!='gain_adaptive' else max_rounds*len(dates),
            additional_queries=sum(counts.values()),total_work=total,
            work_complete=total['integrand_evaluations_unavailable_attempts']==0,
            shared_structural_screening_and_dominance_pruning=True,
            numerical_scope='Exact-endpoint Cartesian level-box propagation, not a sharp joint-model or sampling-coverage statement.',
            unknown_outer_oracle_invocations=sum(row.get('primitive_attempts_unavailable',False) for row in queries),
            total_work_scope='Known instrumented raw attempts only. An unresolved outer oracle invocation has an unavailable primitive attempt count; it is not counted as one known raw primitive attempt.',
            policy_improvement_claim=False,paths_generated=False,policy_fitting_executed=False)
        payload['complete_path_cost']=dict(cpu_seconds=process_time()-cpu_start,wall_seconds=perf_counter()-wall_start)
        return payload

    def query(date,requested,phase,round_record=None):
        entry=dict(date=date,phase=phase,remaining_horizon=tail.rational_record(times[-1]-times[date]),
            requested_width_certificate=_record((requested,requested)),
            previous_value_certificate=None if values[date] is None else _record(values[date]))
        queries.append(entry)
        if round_record is not None:
            round_record['queries'].append(entry)
        work_added=False
        query_cpu,query_wall=process_time(),perf_counter()
        try:
            payload=actual_oracle(vectors[date],times[-1]-times[date],requested,terminal_time=times[-1],
                strike=strike,rate=rate,dividend=dividend,volatility=volatility,precision=precision)
            entry.update(payload)
            _validate_oracle_payload(payload,max_oracle_attempts)
            _add_work(total,payload['work']);work_added=True
            bounds=_interval(payload['value_certificate'])
            if bounds[0]<0:
                raise ValueError('A European premium enclosure cannot have a negative lower endpoint')
            if _width(bounds)>requested:
                raise ValueError('A finite returned premium does not attain its requested width')
            if values[date] is not None:
                bounds=(max(bounds[0],values[date][0]),min(bounds[1],values[date][1]))
                if bounds[1]<bounds[0]:
                    raise ValueError('Returned European value enclosures have an empty retained intersection')
            values[date]=bounds
            entry['retained_value_certificate']=_record(bounds)
            return entry
        except OracleCertificationFailure as error:
            entry.update(error.record);_add_work(total,error.record['work'])
            entry['error']=str(error)
            raise GainAssessmentFailure(str(error),result(False,str(error))) from error
        except Exception as error:
            if not work_added:
                unavailable=_zero_work();unavailable['integrand_evaluations_unavailable_attempts']=1
                _add_work(total,unavailable)
                entry['unavailable_query_work']=unavailable
                entry['primitive_attempts_unavailable']=True
                entry['primitive_attempt_count']=None
                entry['work_complete']=False
                if 'attempts' not in entry:
                    entry.update(attempts=[dict(attempt_index=None,attempt_kind='unresolved_outer_oracle_invocation',primitive_attempt_count=None,returned_finite_certificate=False,requested_width_attained=False,
                        error_type=type(error).__name__,error=str(error),integrand_evaluations=None,work=unavailable,
                        complete_attempt_cost=dict(cpu_seconds=process_time()-query_cpu,wall_seconds=perf_counter()-query_wall))],
                        work=unavailable)
                entry['complete_query_cost']=dict(cpu_seconds=process_time()-query_cpu,wall_seconds=perf_counter()-query_wall)
            entry['error']=str(error)
            raise GainAssessmentFailure(str(error),result(False,str(error))) from error

    for date in dates:
        query(date,initial_value_width,'initial')
    common=initial_value_width
    while True:
        levels=_level_boxes(rewards,values);current=_gain(levels,stopping_index)
        if _width(current)<=target_width:
            return result(True)
        active,dominance=_dominance(levels,stopping_index)
        if arm in ('coarse_common','reference_common'):
            if len(rounds)>=max_rounds:
                raise GainAssessmentFailure('Common allocation exhausted its declared global rounds',result(False,'refinement_budget_exhausted'))
            common/=4
            selected=[date for date in dates if (date==stopping_index or date in active) and _width(values[date])>common]
            if any(counts[date]>=max_rounds for date in selected):
                raise GainAssessmentFailure('A common date exhausted its refinement budget',result(False,'refinement_budget_exhausted'))
            step=dict(round_index=len(rounds)+1,common_requested_width_certificate=_record((common,common)),
                selected_dates=selected,dominance_certificates=dominance,queries=[])
            rounds.append(step)
            for date in selected:
                counts[date]+=1;requests[date]=common;depths[date]=len(rounds)
                query(date,common,'refinement',step)
            # A no-query common round still advances the prescribed width.
        else:
            eligible=[date for date in dates if (date==stopping_index or date in active)
                and _width(values[date])>request_floor and counts[date]<max_rounds and depths[date]<max_rounds]
            if not eligible:
                raise GainAssessmentFailure('No eligible premium query can continue within the declared budget',result(False,'no_eligible_premium'))
            scores={date:_point_score(date,rewards,values,levels,stopping_index) for date in eligible}
            selected=max(eligible,key=lambda date:(scores[date],_width(values[date]),-date))
            depth=depths[selected]+1
            requested=initial_value_width/F(4**depth);skipped=0
            while _width(values[selected])<=requested and depth<max_rounds:
                depth+=1;requested=initial_value_width/F(4**depth);skipped+=1
            requests[selected]=requested;depths[selected]=depth;counts[selected]+=1
            step=dict(round_index=len(rounds)+1,selected_dates=[selected],
                optimistic_scores={str(date):tail.rational_record(score) for date,score in scores.items()},
                skipped_already_attained_quarter_requests=skipped,requested_quarter_depth=depth,
                dominance_certificates=dominance,queries=[])
            rounds.append(step)
            query(selected,requested,'refinement',step)
        step['gain_certificate_before']=_record(current)
        final=_gain(_level_boxes(rewards,values),stopping_index)
        step['gain_certificate_after']=_record(final)
        step['attained_width_reduction']=tail.rational_record(_width(current)-_width(final))
