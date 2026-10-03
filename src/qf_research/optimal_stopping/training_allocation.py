"""Allocate independent training pairs using a measured discrete cost curve.

The statistical objective is A/R + B/(R*N), conditional on a fixed model and
training procedures. Costs and variance components come from a separate pilot.
This selector neither fits a timing curve nor extrapolates to unmeasured sizes.
Its budget is a computational prediction, not an elapsed-time guarantee.
"""

import math
from numbers import Real


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f'{name} must be a finite real number') from error
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return value


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _product(count, cost):
    try:
        return count*cost
    except OverflowError:
        return math.inf


def choose_allocation(A, B, setup_cpu, pair_costs, reference_repetitions=64,
                      reference_paths=4096, min_repetitions=64,
                      max_repetitions=1024):
    """Select fixed integer counts at the supplied, measured evaluation sizes.

    ``pair_costs[N]`` is the complete cost of independently training both
    procedures and jointly evaluating them on N shared paths, excluding the
    one-off ``setup_cpu``. No monotonicity is imposed on measured costs.

    The budget is setup_cpu + reference_repetitions*pair_costs[reference_paths].
    A 32-ULP allowance prevents floating-point division from losing a boundary
    repetition; both variable and total costs must satisfy their allowances.
    Unused budget is the nonnegative remaining variable budget, with any tiny
    permitted excess reported separately. ``capped`` means another repetition
    would be affordable beyond max_repetitions.

    Nonpositive finite A or B is retained as unresolved pilot evidence. In that
    case the reference allocation is returned without a predicted variance.
    Otherwise the smallest predicted variance wins, with ascending N resolving
    exact ties. All returned values are JSON-native, and invalid or
    unrepresentable inputs raise ValueError.
    """
    A, B = _finite(A, 'A'), _finite(B, 'B')
    setup = _finite(setup_cpu, 'setup_cpu')
    if setup < 0:
        raise ValueError('setup_cpu must be nonnegative')
    for name, count in (('reference_repetitions', reference_repetitions),
                        ('reference_paths', reference_paths),
                        ('min_repetitions', min_repetitions),
                        ('max_repetitions', max_repetitions)):
        _positive_integer(count, name)
    if not min_repetitions <= reference_repetitions <= max_repetitions:
        raise ValueError('Reference repetitions must lie within the repetition limits')
    if not isinstance(pair_costs, dict) or not pair_costs:
        raise ValueError('Require a nonempty dictionary of measured pair costs')
    costs = {}
    for paths, value in pair_costs.items():
        _positive_integer(paths, 'Evaluation paths')
        cost = _finite(value, 'Pair cost')
        if cost <= 0:
            raise ValueError('Pair costs must be strictly positive')
        costs[paths] = cost
    if reference_paths not in costs:
        raise ValueError('Reference evaluation size has no measured pair cost')

    usable = _product(reference_repetitions, costs[reference_paths])
    budget = setup+usable
    if not math.isfinite(budget) or not math.isfinite(usable) or usable <= 0:
        raise ValueError('The reference budget is not representable as a finite float')
    variable_tolerance = 32*math.ulp(usable)
    total_tolerance = 32*math.ulp(budget)
    resolved = A > 0 and B > 0

    def affordable(repetitions, cost):
        variable = _product(repetitions, cost)
        total = setup+variable
        return (math.isfinite(variable) and math.isfinite(total)
                and variable-usable <= variable_tolerance
                and total-budget <= total_tolerance)

    def record(paths, repetitions, capped):
        variable = _product(repetitions, costs[paths])
        total = setup+variable
        feasible = repetitions >= min_repetitions and affordable(repetitions, costs[paths])
        variance = A/repetitions+B/repetitions/paths if resolved and feasible else None
        if variance is not None and not math.isfinite(variance):
            raise ValueError('The predicted variance is not representable as a finite float')
        return dict(evaluation_paths=paths, repetitions=repetitions,
                    pair_cpu_seconds=costs[paths], total_cpu_seconds=total,
                    unused_cpu_seconds=max(0., usable-variable),
                    budget_excess_cpu_seconds=max(0., total-budget),
                    predicted_variance=variance, feasible=feasible, capped=capped)

    candidates = []
    for paths in sorted(costs):
        # Binary search also handles a tiny measured cost without an overflowing
        # budget/cost quotient. One extra count determines whether the cap binds.
        low, high = 0, max_repetitions+1
        while low < high:
            middle = (low+high+1)//2
            if affordable(middle, costs[paths]):
                low = middle
            else:
                high = middle-1
        candidates.append(record(paths, min(low, max_repetitions), low > max_repetitions))

    reference = record(reference_paths, reference_repetitions, False)
    selected = (min((row for row in candidates if row['feasible']),
                    key=lambda row: row['predicted_variance'])
                if resolved else reference)
    return dict(status='selected' if resolved else 'unresolved_variance',
                raw_variance_components=dict(training=A, evaluation=B),
                budget=dict(cpu_seconds=budget, setup_cpu_seconds=setup,
                            usable_cpu_seconds=usable,
                            tolerance_cpu_seconds=total_tolerance,
                            variable_tolerance_cpu_seconds=variable_tolerance),
                reference=reference, selected=selected, candidates=candidates)


def continuous_optimum(A, B, training_cpu, evaluation_cpu_per_path, budget, setup):
    """Return the elementary continuous optimum for an assumed affine cost.

    This non-novel calculus diagnostic is separate from the measured-cost
    selector. It assumes positive A, B and both marginal costs, with a positive
    budget after nonnegative setup. It imposes no integer, minimum or maximum
    counts and does not estimate an empirical timing relationship.
    """
    values = {name: _finite(value, name) for name, value in
              (('A', A), ('B', B), ('training_cpu', training_cpu),
               ('evaluation_cpu_per_path', evaluation_cpu_per_path),
               ('budget', budget), ('setup', setup))}
    A, B = values['A'], values['B']
    training, evaluation = values['training_cpu'], values['evaluation_cpu_per_path']
    available = values['budget']-values['setup']
    if (min(A, B, training, evaluation) <= 0 or values['setup'] < 0
            or not math.isfinite(available) or available <= 0):
        raise ValueError('Require positive variances and costs, and budget above nonnegative setup')
    try:
        paths = math.exp((math.log(B)+math.log(training)-math.log(A)-math.log(evaluation))/2)
        repetitions = available/(training+evaluation*paths)
        variance = A/repetitions+B/repetitions/paths
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError('The continuous optimum is not representable as finite floats') from error
    if not all(math.isfinite(x) and x > 0 for x in (paths, repetitions, variance)):
        raise ValueError('The continuous optimum is not representable as finite positive floats')
    return dict(evaluation_paths=paths, repetitions=repetitions, minimum_variance=variance)
