"""Consistent curve replacement with nominal cash and contract clocks fixed."""

from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime
import json
import math
from zoneinfo import ZoneInfo

from .funded_inputs import DatedDividend, build_dated_problem


def rebuild_funding(raw, source, calendar, curve, *, allow_observation_date=False):
    observation = datetime.fromisoformat(source['cutoff'])
    if observation.tzinfo is None or observation.utcoffset() is None:
        raise ValueError('An explicitly zoned observation is required')
    day = observation.astimezone(ZoneInfo('America/New_York')).date()
    if day.isoformat() != source['observation_date']:
        raise ValueError('Observation date and cutoff differ')
    if curve.anchor > day or curve.anchor == day and not allow_observation_date:
        raise ValueError('An observation-date curve requires explicit retrospective opt-in')
    if (raw['observation_id'] != source['observation_id'] or
        raw['option']['strike'] != source['strike'] or raw['option']['kind'] != source['kind'] or
        raw['model']['quoted_spot'] != source['quoted_spot']):
        raise ValueError('Source and problem identities differ')
    forecast = raw['forecast']
    if datetime.fromisoformat(forecast['as_of']) != observation or forecast['currency'] != 'USD':
        raise ValueError('Forecast cutoff or currency differs')
    publication = datetime.fromisoformat(forecast['latest_input_publication_bound'])
    if publication.tzinfo is None or publication >= observation:
        raise ValueError('Forecast inputs must predate the observation')
    problem = build_dated_problem(calendar, curve, observation, date.fromisoformat(source['expiry']),
        source['quoted_spot'], source['strike'], raw['model']['volatility'], source['kind'],
        (DatedDividend(date.fromisoformat(forecast['ex_date']),date.fromisoformat(forecast['pay_date']),
                       float(forecast['regular_dividend_estimate']),
                       'unchanged dated nominal forecast; retrospective funding sensitivity'),),
        premium_payment_date=date.fromisoformat(raw['premium_payment_date']))
    if problem.provisional_delivery:
        raise ValueError('Provisional delivery is outside this comparison')
    if asdict(problem.option) != raw['option']:
        raise ValueError('Curve replacement changed the contract or diffusion clock')
    old_rights = [(r['time'],r['phase']) for r in raw['decisions']]
    if [(r.time,r.phase) for r in problem.decisions] != old_rights:
        raise ValueError('Curve replacement changed the exercise calendar')
    if [e.ex_time for e in problem.model.dividends] != [e['ex_time'] for e in raw['model']['dividends']]:
        raise ValueError('Curve replacement changed the cash-event clock')
    result = deepcopy(raw)
    result['model'] = json.loads(json.dumps(asdict(problem.model)))
    result['decisions'] = [asdict(d) for d in problem.decisions]
    result['curve_date'] = curve.anchor.isoformat()
    return result


def lower_corner_components(*, call_premium, ownership_premium, terminal_strike_premium,
                            expected_cash_premium, put_premium, cap_allowance, numerical_allowance):
    """Account for a numerical lower bound in premium units, not causal effects."""
    values = (call_premium,ownership_premium,terminal_strike_premium,expected_cash_premium,
              put_premium,cap_allowance,numerical_allowance)
    if any(isinstance(x,bool) or not math.isfinite(x) for x in values):
        raise ValueError('Finite numerical components required')
    if min(ownership_premium,terminal_strike_premium) <= 0 or min(expected_cash_premium,cap_allowance,numerical_allowance) < 0:
        raise ValueError('Invalid funding or allowance components')
    european_put = call_premium - ownership_premium + terminal_strike_premium + expected_cash_premium
    early_premium = put_premium-european_put
    untruncated = put_premium-cap_allowance-numerical_allowance
    # The max(0, ...) adjustment is explicit, so the identity also covers zero bounds.
    components = {'call_inversion':call_premium,'initial_ownership':-ownership_premium,
                  'terminal_strike_funding':terminal_strike_premium,
                  'expected_capped_payment':expected_cash_premium,
                  'numerical_put_exercise_premium':early_premium,
                  'cap_allowance':-cap_allowance,'numerical_allowance':-numerical_allowance,
                  'zero_floor':max(0.,-untruncated)}
    return {'components':components,'european_put_premium':european_put,
            'conditional_lower_bound':max(0.,untruncated)}


def component_changes(baseline, alternative):
    keys = set(baseline['components'])
    if keys != set(alternative['components']):
        raise ValueError('Decompositions have different components')
    changes = {k:alternative['components'][k]-baseline['components'][k] for k in sorted(keys)}
    total = alternative['conditional_lower_bound']-baseline['conditional_lower_bound']
    residual = math.fsum(changes.values())-total
    scale = max(1., *(abs(x) for d in (baseline,alternative) for x in d['components'].values()))
    if abs(residual) > 5e-12*scale:
        raise ValueError('Lower-bound decomposition does not close')
    return {'changes':changes,'lower_bound_change':total,'identity_residual':residual}
