"""Matched pricing and policy comparisons for supplied call-calibrated models.

Dates and observed quotes are supplied by the checked adapter. Simulated paths
are model draws, never market observations. No policy or repetition is selected.
"""
from dataclasses import asdict, replace
import numpy as np

from .contracts import Option
from .funded_cash import FundedCashModel, FundedExercise, european_reference
from .funded_policy import simulate_paths, prefix_paths
from .value_martingale import build_values, observation_rights, value_martingale
from .expiry_extension import extended_calendar, restrict_paths, price_terminal_extension
from .terminal_repair import evaluate_terminal_repair
from .terminal_training import fit_terminal_policy
from .policy_transfer import family_upper_limit
from .simulation import estimate_samples
from ..computational.funded_kernel import prepare, advance_scipy
from ..market_data.calendar_robustness import mesh_changes
from ..market_data.prospective_inputs import interval_residual

YEAR = 365 * 86400


def restore_case(case):
    if case['status'] != 'call_calibrated' or case['put_used_in_calibration']:
        raise ValueError('A call-only calibrated case is required')
    raw = dict(case['model'])
    if raw['dividends'] or raw['premium_factor'] != 1 or raw['initial_share_factor'] != 1:
        raise ValueError('Require the no-cash, premium-normalised specification')
    raw['dividends'] = ()
    return FundedCashModel(**raw), Option(**case['option']), tuple(FundedExercise(**r) for r in case['decisions'])


def stream(config, case, role, repetition):
    if config['bit_generator'] != 'PCG64' or role not in (0, 1) or role == 1 and repetition != 0:
        raise ValueError('Random stream role differs from the declared allocation')
    indices = (case['date_index'], case['strike_index'], role, repetition)
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
        raise ValueError('Non-negative integer stream indices required')
    seed = np.random.SeedSequence(config['root_seed'], spawn_key=indices)
    return np.random.Generator(np.random.PCG64(seed)), dict(entropy=seed.entropy, spawn_key=list(indices))


def price_case(case, config, advance=advance_scipy):
    """Retain five meshes, signed residuals and endpoint changes without refitting."""
    model, option, rights = restore_case(case)
    threshold = max(model.initial_state, *(option.strike*r.settlement_factor for r in rights))
    prepared = {name: prepare(model, option, g['space_steps'], g['time_steps'],
                             threshold*g['domain_multiple'], rights)
                for name, g in config['pde_meshes'].items()}
    put = case['observed']['put']
    allowance = config['pde_numerical_allowance_quote']
    rows = []
    for seconds in config['extensions_seconds']:
        extension = seconds/YEAR
        prices = {name: price_terminal_extension(p, extension, advance).premium_quote
                  for name, p in prepared.items()}
        changes = mesh_changes(prices)
        screens = dict(space=all(abs(changes[k]) <= allowance for k in ('space_at_coarse_time', 'space_at_fine_time')),
            time=all(abs(changes[k]) <= allowance for k in ('time_at_coarse_space', 'time_at_fine_space')),
            domain=abs(changes['domain']) <= allowance)
        # Independent Python execution is retained even when native propagation
        # was supplied. Matching implementations is not a PDE error certificate.
        python_value = price_terminal_extension(prepared['fine'], extension).premium_quote
        tolerance = config['python_native_tolerance']
        difference = python_value - prices['fine']
        screens['implementation_agreement'] = abs(difference) <= tolerance['absolute'] + tolerance['relative']*abs(python_value)
        call = replace(option, kind='call', maturity=option.maturity+extension)
        terminal = replace(rights[-1], time=call.maturity)
        call_value = european_reference(model, call, terminal).premium_quote
        rows.append(dict(extension_seconds=seconds, prices=prices, mesh_changes=changes, numerical_screens=screens,
            python_fine_quote=python_value, implementation_difference_quote=difference,
            numerical_allowance_quote=allowance, allowance_is_certified_error_bound=False,
            put_signed_interval_residual=interval_residual(prices['fine'], put['BID'], put['ASK']),
            put_inside_interval=put['BID'] <= prices['fine'] <= put['ASK'],
            call_quote=call_value, call_midpoint_residual=call_value-case['call_midpoint'],
            volatility_recalibrated=False))
    return rows


def dual_record(result, spread, config):
    limit = family_upper_limit(result.gap_bound, len(config['extensions_seconds']), config['family_error_probability'])
    return dict(policy_value=asdict(result.policy_value), upper_value=asdict(result.upper_value),
        gap_bound=asdict(result.gap_bound), gap_mean_over_spread=result.gap_bound.value/spread,
        pointwise_upper_below_spread=result.gap_bound.upper < spread,
        family_upper_quote=limit, family_upper_below_spread=limit < spread)


def evaluate_case(case, config):
    """Train first, then share a fresh evaluation stream across all comparisons."""
    model, option, rights = restore_case(case)
    if config['confidence'] != .95:
        raise ValueError('The existing estimator uses 95% Student intervals')
    seconds_list = config['extensions_seconds']
    if seconds_list != [0, 900, 5400]:
        raise ValueError('Require the three prescribed endpoints')
    spread = case['observed']['put']['ASK'] - case['observed']['put']['BID']
    if spread <= 0:
        raise ValueError('A positive observed put spread is required')
    training, fitted = [], []
    for repetition in range(config['training_repetitions']):
        rng, provenance = stream(config, case, 0, repetition)
        paths = simulate_paths(model, option, rights, max(config['training_paths']), rng)
        provenance.update(paths=len(paths.states), stream_id=paths.stream_id)
        for n in config['training_paths']:
            subset = prefix_paths(paths, n)
            for degree in config['degrees']:
                policies = {seconds: fit_terminal_policy(option, subset, degree, seconds/YEAR,
                    **config['regression_safeguards']) for seconds in seconds_list}
                fitted.append(policies)
                training.append(dict(repetition=repetition, training_paths=n, degree=degree, stream=provenance,
                    policies={str(seconds): dict(policy=asdict(p.policy), fingerprint=p.fingerprint)
                              for seconds, p in policies.items()}))
    calendars = {0: (option, rights)}
    for j, seconds in enumerate(seconds_list[1:], 1):
        calendars[seconds] = extended_calendar(model, option, rights, tuple(s/YEAR for s in seconds_list[1:j+1]))
    values = {(seconds, name): build_values(model, *calendars[seconds], **settings)
              for seconds in seconds_list for name, settings in config['value_approximations'].items()}
    longest = values[(max(seconds_list), 'fine')]
    rng, evaluation_stream = stream(config, case, 1, 0)
    observations = simulate_paths(model, calendars[max(seconds_list)][0], observation_rights(longest), config['evaluation_paths'], rng)
    evaluation_stream.update(paths=len(observations.states), stream_id=observations.stream_id)
    if evaluation_stream['stream_id'] in {r['stream']['stream_id'] for r in training}:
        raise ValueError('Evaluation stream overlaps a training stream')
    endpoints = []
    for seconds in seconds_list:
        samples = {name: value_martingale(values[(seconds, name)], restrict_paths(
            observations, calendars[seconds][0], observation_rights(values[(seconds, name)])))
                   for name in config['value_approximations']}
        records, paired = [], []
        degree_two = {}
        for record, policies in zip(training, fitted):
            comparisons, gaps = {}, {}
            for name, sample in samples.items():
                baseline = evaluate_terminal_repair(policies[0].policy, sample, seconds/YEAR)
                retrained = evaluate_terminal_repair(policies[seconds].policy, sample, seconds/YEAR) if seconds else baseline
                if not np.array_equal(baseline.repaired.upper_payoffs, retrained.repaired.upper_payoffs):
                    raise ArithmeticError('Policy comparison changed the common upper payoff')
                gain = baseline.repaired.gap_payoffs - retrained.repaired.gap_payoffs
                comparisons[name] = dict(
                    terminal_only=dual_record(baseline.repaired, spread, config),
                    retrained=dual_record(retrained.repaired, spread, config),
                    retraining_gain=asdict(estimate_samples(gain)),
                    terminal_adjustment_gain=asdict(estimate_samples(baseline.integrated_gain)),
                    changed_stops=int(np.count_nonzero(baseline.stopping_indices != retrained.stopping_indices)))
                gaps[name] = baseline.repaired.gap_payoffs
                if name == 'fine' and seconds == 0 and record['training_paths'] == max(config['training_paths']):
                    rep = record['repetition']
                    if record['degree'] == 2:
                        degree_two[rep] = baseline.repaired.adjusted_policy_payoffs.copy()
                    elif record['degree'] == 4:
                        paired.append(dict(repetition=rep, degree_four_minus_two=asdict(estimate_samples(
                            baseline.repaired.adjusted_policy_payoffs - degree_two.pop(rep)))))
            records.append(dict(repetition=record['repetition'], training_paths=record['training_paths'], degree=record['degree'],
                comparisons=comparisons, paired_coarse_minus_fine_gap=asdict(estimate_samples(gaps['coarse']-gaps['fine']))))
        endpoints.append(dict(extension_seconds=seconds, policies=records, larger_training_degree_comparison=paired,
            approximations={name: dict(fingerprint=values[(seconds, name)].fingerprint)
                            for name in config['value_approximations']}))
    families = []
    for i, record in enumerate(training):
        limits = {name: {method: max(e['policies'][i]['comparisons'][name][method]['family_upper_quote'] for e in endpoints)
                         for method in ('terminal_only', 'retrained')} for name in config['value_approximations']}
        families.append(dict(repetition=record['repetition'], training_paths=record['training_paths'], degree=record['degree'],
                             maximum_endpoint_upper_quote=limits))
    return dict(training=training, endpoints=endpoints, endpoint_families=families,
        evaluation_stream=evaluation_stream, spread=spread, value_unit=case['value_unit'],
        simulation_is_market_data=False, policy_selection=False)
